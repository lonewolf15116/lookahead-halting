"""Phase 3 fresh-data inventory and audit (docs/PHASE3_PLAN.md section 9). CPU only; reads datasets and manifests, nothing else.

    python scripts/phase3_inventory.py [--tokenizer]

INVENTORY ONLY. It does not read model outputs, correctness labels of any model, hidden states, thresholds, probes or
confirmation predictions, trains nothing, and writes only
  results/phase3/inventory.json            canonical JSON: pools, exclusion counts, revisions, fingerprints, unresolved list
  results/phase3/inventory_clean_ids.csv   identities (not outcomes) of every audit-clean row; sha256 is in the JSON
  docs/PHASE3_INVENTORY.md                 generated audit summary
It refuses to overwrite existing outputs and never writes under results/gate1, results/gate2 or data/manifests.

What "used" means (everything earlier stages touched, rebuilt rather than assumed):
  * Phase 1 rebuild of all six benchmarks with lm_eval's few-shot sampler instrumented: every scored question (Gate 1, and
    Gate 2 train/validation, are all subsets of these) and every few-shot exemplar drawn for them.
  * The Gate 2 confirmation set rebuilt with the confirmation builder: its questions and their exemplars.
  The rebuilds are verified against the committed manifests (Phase 1 doc ids = data/manifests/splits.csv per benchmark, and with
  --tokenizer also the question-set hashes; confirmation ids sha256 = data/manifests/confirmation_ids.sha256). If a
  verification fails, or a dataset revision cannot be established or differs from the one used before, the affected pools are
  listed as UNRESOLVED and are never counted as clean. Exit code 0 = everything resolved, 3 = report written with unresolved pools.

Candidate pools: for the six in-distribution benchmarks every labelled-or-not split the dataset offers (so the reader sees that
the used splits are empty after exclusion); for PIQA, WinoGrande and Social IQa (new benchmarks, kept separate) likewise.
A pool needs >= 300 clean rows to be eligible. That is an eligibility filter, not evidence of adequate precision; the report
adds an illustrative precision table per pool.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from lookahead import confirm as C
from lookahead import phase3_inventory as inv
from lookahead.config import REPO_ROOT, load_config
from lookahead.questions import BENCHMARKS, load_lm_eval_questions, question_set_hash

NEW_BENCHMARKS = ("piqa", "winogrande", "social_iqa")
NEW_HF_DATASETS = {"piqa": "baber/piqa", "winogrande": "allenai/winogrande", "social_iqa": "lighteval/siqa"}
SPLIT_NAMES = ("train", "validation", "test")
OUT_JSON, OUT_IDS, OUT_MD = "results/phase3/inventory.json", "results/phase3/inventory_clean_ids.csv", "docs/PHASE3_INVENTORY.md"
INPUT_FILES = ("data/manifests/splits.csv", "data/manifests/questions.json", "data/manifests/confirmation.json",
               "data/manifests/confirmation_ids.csv", "data/manifests/confirmation_ids.sha256",
               "results/gate2/confirmation_audit.json", "docs/PHASE3_PLAN.md")
UNRESOLVED_RISKS = [
    "Pretraining and SFT contamination of Ouro-1.4B is unknowable from the data: its training mixture is not itemised, so any "
    "benchmark's train or validation split may have been seen by the model. The inventory removes overlap with what OUR "
    "pipeline used; it cannot remove overlap with what the model saw. A finding that depends on a pool being unseen by "
    "the model is not supported by this audit.",
    "Text matching is exact after lower-casing and whitespace normalisation. Paraphrases and near-duplicates (for example "
    "ARC items that differ in one word) are not detected.",
    "Native ids are only unique within a dataset revision; HellaSwag ids repeat inside train, so an id match can exclude "
    "more rows than were used (conservative; `excluded_by_id_only` shows how many).",
    "A pool that is clean today is clean for one dataset revision. The revisions recorded here must be pinned when the fresh "
    "allocation is frozen.",
    "Few-shot exemplars for the future fresh tests are not drawn yet; the sampler draws them at build time. For new "
    "benchmarks the report states the few-shot source split and the overlap of each candidate pool with it.",
]


class StubLM:
    """Exemplar selection and text identities do not depend on tokenisation, so the inventory does not need the model's
    tokenizer. Contexts share one encoding, which is all lm_eval's pair check needs."""

    def _encode_pair(self, context, continuation):
        return [0], [0]


# --------------------------------------------------------------------------- identities for the new benchmarks

def inv_text_hash(benchmark, doc, task):
    """Template-independent content hash. The three new benchmarks have their own raw fields; the rest use confirm.text_hash."""
    parts = None
    if benchmark == "piqa" and "goal" in doc:
        parts = [doc["goal"], doc.get("sol1", ""), doc.get("sol2", "")]
    elif benchmark == "winogrande" and "sentence" in doc:
        parts = [doc["sentence"], doc.get("option1", ""), doc.get("option2", "")]
    elif benchmark == "social_iqa" and "question" in doc and "answerA" in doc:
        parts = [doc.get("context", ""), doc["question"], doc["answerA"], doc.get("answerB", ""), doc.get("answerC", "")]
    if parts is None:
        return C.text_hash(benchmark, doc, task)
    return hashlib.sha256(C.UNIT.join(C._norm(p) for p in parts).encode("utf-8")).hexdigest()


def inv_native_id(benchmark, doc, split):
    """Dataset-native id (None for the new benchmarks, which ship none: identity is then the content hash)."""
    return None if benchmark in NEW_BENCHMARKS else C.native_id(benchmark, doc, split)


def is_labeled(task, doc) -> bool:
    """Does the DATASET ship a gold answer for this row? (a property of the data, not of any model)"""
    try:
        gold = task.doc_to_target(doc)
        choices = task.doc_to_choice(doc)
        if isinstance(gold, str):
            return gold in choices
        return isinstance(gold, int) and 0 <= gold < len(choices)
    except Exception:
        return False


# --------------------------------------------------------------------------- used set

def read_split_ids(root: Path) -> dict:
    out = {}
    with (root / "data/manifests/splits.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            out.setdefault(r["benchmark"], set()).add(r["doc_id"])
    return out


def build_used_set(root, benchmarks, lm, tm, specs, split_plan, take, salt, with_hashes):
    """Rebuild Phase 1 and the confirmation set and return (UsedSet, checks, global_failures)."""
    used, checks, failures = inv.UsedSet(), {"phase1": {}, "confirmation": {}}, []
    frozen = json.loads((root / "data/manifests/questions.json").read_text(encoding="utf-8"))["benchmarks"]
    split_ids = read_split_ids(root)
    recorders = {}
    for b in benchmarks:
        with C.record_exemplars() as rec:
            qs = load_lm_eval_questions(b, lm, task_manager=tm, spec=specs[b], on_task=rec.begin_task, on_doc=rec.on_doc)
        recorders[b] = rec
        ids = {q.doc_id for q in qs}
        chk = {"n_questions": len(qs), "manifest_n_questions": frozen[b]["n_questions"],
               "doc_ids_equal_splits_csv": ids == split_ids.get(b, set()), "exemplar_live_failures": len(rec.live_failures)}
        chk["n_questions_match"] = chk["n_questions"] == chk["manifest_n_questions"]
        if with_hashes:
            chk["question_set_hash_matches"] = question_set_hash(qs) == frozen[b]["question_set_sha256"]
        else:
            chk["question_set_hash_matches"] = "not_checked (stub tokenizer)"
        checks["phase1"][b] = chk
        if not (chk["doc_ids_equal_splits_csv"] and chk["n_questions_match"] and chk["question_set_hash_matches"] is not False
                and chk["exemplar_live_failures"] == 0):
            failures.append(f"phase1 rebuild of {b} does not match the committed manifest")
        for r in rec.by_doc_id.values():
            used.add("P1_EVAL", b, r.native_id, r.text_hash)
            for d in r.superset + r.inserted:  # superset is conservative: it includes the dropped extra draw
                used.add("P1_EXEMPLAR", b, d.native_id, d.text_hash)
    exclusion = C.Exclusion.from_recorders(recorders)
    builds = {}
    for b in benchmarks:
        builds[b] = C.build_confirmation_questions(b, lm, take[b], exclusion, salt=salt, task_manager=tm, spec=specs[b],
                                                   split_plan=split_plan)
        bd = builds[b]
        for c in bd.selected:
            used.add("CONF_EVAL", b, c.native_id, c.text_hash)
        for r in bd.recorder.by_doc_id.values():
            for d in r.superset + r.inserted:
                used.add("CONF_EXEMPLAR", b, d.native_id, d.text_hash)
    ids_sha = C.sha256_bytes(C.ids_csv_text(builds).encode("utf-8"))
    pinned = (root / "data/manifests/confirmation_ids.sha256").read_text(encoding="utf-8").split()[0]
    checks["confirmation"] = {"rebuilt_ids_sha256": ids_sha, "manifest_ids_sha256": pinned, "matches": ids_sha == pinned,
                              "n_questions": {b: len(bd.questions) for b, bd in builds.items()}}
    if ids_sha != pinned:
        failures.append("confirmation rebuild does not reproduce data/manifests/confirmation_ids.sha256")
    return used, checks, failures


# --------------------------------------------------------------------------- pools

def enumerate_pools(tm, benchmark):
    """{source_split: [Rec]} for every split the dataset offers, plus few-shot source info. Raises on load problems."""
    loaded = tm.load([benchmark])["tasks"]
    pools, tasks_info = {}, {}
    for task_name in sorted(loaded):
        task = loaded[task_name]
        avail = [s for s in SPLIT_NAMES if s in task.dataset]
        fs = task.config.fewshot_split
        if fs is None:
            fs = "train" if task.has_training_docs() else ("validation" if task.has_validation_docs() else "test")
        tasks_info[task_name] = {"splits": avail, "fewshot_source_split": fs, "fewshot_split_configured": task.config.fewshot_split}
        for split in avail:
            C.switch_eval_split(task, split)
            for i, doc in enumerate(task.eval_docs):
                pools.setdefault(split, []).append(inv.Rec(
                    benchmark, split, task_name, i, inv_native_id(benchmark, doc, split), inv_text_hash(benchmark, doc, task),
                    is_labeled(task, doc)))
    return pools, tasks_info


def fewshot_overlap(pools, tasks_info):
    """Per source split: rows whose text also occurs in the few-shot source split (report only)."""
    src = {i["fewshot_source_split"] for i in tasks_info.values()}
    out = {}
    for split, recs in pools.items():
        if len(src) != 1:
            out[split] = {"status": "mixed few-shot sources", "sources": sorted(src)}
            continue
        (s,) = src
        if s == split:
            out[split] = {"status": "same_split: exemplars would be drawn from the pool itself", "source": s}
        else:
            texts = {r.text_hash for r in pools.get(s, [])}
            out[split] = {"status": "disjoint_split", "source": s, "rows_with_text_in_source": sum(r.text_hash in texts for r in recs)}
    return out


# --------------------------------------------------------------------------- driver

def file_sha(p: Path) -> str:
    return C.sha256_path(p)


def main(argv=None, *, task_manager=None, lm=None, root=None, specs=None, split_plan=None, take=None, benchmarks=None,
         new_benchmarks=None, hf_datasets=None, fetch_revision=None, salt=None, min_clean=inv.MIN_CLEAN) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", action="store_true", help="use the real Ouro tokenizer so Phase 1 question-set hashes are also verified")
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)
    root = Path(root or args.root or REPO_ROOT)
    benchmarks = list(benchmarks or BENCHMARKS)
    new_benchmarks = list(NEW_BENCHMARKS if new_benchmarks is None else new_benchmarks)
    specs, split_plan = specs or BENCHMARKS, split_plan or C.SPLIT_PLAN
    take = take or {b: C.DEFAULT_TAKE.get(b) for b in benchmarks}
    hf = dict(C.HF_DATASETS if hf_datasets is None else hf_datasets)
    hf.update(NEW_HF_DATASETS if hf_datasets is None else {})
    fetch_revision = fetch_revision or C.dataset_revision
    salt = salt or C.SALT
    outs = [root / OUT_JSON, root / OUT_IDS, root / OUT_MD]
    if any(p.exists() for p in outs):
        print("refusing: an inventory output already exists:", [str(p) for p in outs if p.exists()], file=sys.stderr)
        return 2
    for rel in INPUT_FILES:
        if not (root / rel).exists():
            print(f"missing input {rel}", file=sys.stderr)
            return 2

    from lm_eval.tasks import TaskManager

    tm = task_manager or TaskManager()
    with_hashes = bool(args.tokenizer and lm is None)
    if lm is None:
        if args.tokenizer:
            from transformers import AutoTokenizer

            from lookahead.lm_eval_ouro import LoopedOuroLM

            lm = LoopedOuroLM(None, AutoTokenizer.from_pretrained(str(REPO_ROOT / load_config("model")["local_dir"])))
        else:
            lm = StubLM()
    unresolved, global_failure = [], []

    print("rebuilding Phase 1 and confirmation used-sets ...", flush=True)
    used, used_checks, global_failure = build_used_set(root, benchmarks, lm, tm, specs, split_plan, take, salt, with_hashes)
    for msg in global_failure:
        unresolved.append({"pool": "*", "reason": msg})
    print("used:", used.counts, "fingerprint", used.fingerprint()[:12], flush=True)

    # dataset revisions
    manifest = json.loads((root / "data/manifests/confirmation.json").read_text(encoding="utf-8"))["benchmarks"]
    revisions = {}
    for b in benchmarks + new_benchmarks:
        repo = hf.get(b)
        rev = fetch_revision(repo) if repo else "unavailable (no dataset path)"
        entry = {"hf_dataset": repo, "hf_revision": rev}
        if b in manifest:
            entry["revision_used_before"] = manifest[b]["hf_revision"]
            entry["same_as_before"] = rev == manifest[b]["hf_revision"]
            if not entry["same_as_before"]:
                unresolved.append({"pool": f"{b}/*", "reason": f"dataset revision {rev} differs from the one used before "
                                                               f"({manifest[b]['hf_revision']}) or cannot be established"})
        elif str(rev).startswith("unavailable"):
            unresolved.append({"pool": f"{b}/*", "reason": f"dataset revision cannot be established: {rev}"})
        revisions[b] = entry

    # candidate pools
    pools_out, clean_rows, groups, fewshot, tasks_meta = [], [], {}, {}, {}
    earlier_texts: set[str] = set()
    for group, names in (("in_distribution", benchmarks), ("new_benchmark", new_benchmarks)):
        for b in names:
            try:
                pools, tinfo = enumerate_pools(tm, b)
            except Exception as e:  # provenance or loading problem: unresolved, never assumed clean
                unresolved.append({"pool": f"{b}/*", "reason": f"pool could not be loaded: {type(e).__name__}: {str(e)[:200]}"})
                groups[f"{b}/*"] = group
                print(f"{b}: UNRESOLVED ({type(e).__name__})", flush=True)
                continue
            tasks_meta[b] = {t: i for t, i in sorted(tinfo.items())} if len(tinfo) <= 5 else {
                "n_tasks": len(tinfo), "fewshot_source_splits": sorted({i["fewshot_source_split"] for i in tinfo.values()})}
            if group == "new_benchmark":
                fewshot[b] = fewshot_overlap(pools, tinfo)
            for split in sorted(pools, key=SPLIT_NAMES.index):
                res = inv.audit_pool(pools[split], used, benchmark=b, source_split=split, earlier_pool_texts=earlier_texts,
                                     min_clean=min_clean)
                groups[f"{b}/{split}"] = group
                pools_out.append(res)
                earlier_texts |= {r.text_hash for r in res["_clean"]}
                clean_rows += [(b, split, r) for r in res["_clean"]]
                print(f"{b}/{split}: raw {res['raw']} clean {res['clean']} eligible {res['eligible']}", flush=True)

    ids_text = "\n".join(["benchmark,source_split,task,row,native_id,text_hash"] + [
        f"{b},{s},{r.task},{r.row},{r.native_id or ''},{r.text_hash}" for b, s, r in sorted(
            clean_rows, key=lambda x: (x[0], x[1], x[2].task, x[2].row))]) + "\n"
    code_sha = {p: file_sha(root / p) for p in ("src/lookahead/phase3_inventory.py", "scripts/phase3_inventory.py")
                if (root / p).exists()}
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except Exception:
            return "unavailable"

    report = inv.build_report(
        pools=pools_out, groups=groups,
        provenance={"revisions": revisions, "lm_eval_version": ver("lm_eval"), "datasets_version": ver("datasets"),
                    "tokenizer_mode": "real Ouro tokenizer (Phase 1 question-set hashes verified)" if with_hashes
                    else "stub (exemplar selection and text identities do not depend on tokenisation; token hashes not checked)",
                    "tasks": tasks_meta, "fewshot_overlap_new_benchmarks": fewshot, "code_sha256": code_sha,
                    "confirmation_salt": salt, "split_plan": split_plan, "take": take},
        inputs={"files_sha256": {rel: file_sha(root / rel) for rel in INPUT_FILES}, "verification": used_checks,
                "clean_ids_csv_sha256": C.sha256_bytes(ids_text.encode("utf-8")), "n_clean_ids": len(clean_rows)},
        used_summary={"counts_added": used.counts, "n_distinct_ids": len(used.ids), "n_distinct_texts": len(used.texts),
                      "fingerprint": used.fingerprint(),
                      "kinds": {"P1_EVAL": "every Phase 1 scored question (covers Gate 1 and Gate 2 train/validation)",
                                "CONF_EVAL": "Gate 2 confirmation question", "P1_EXEMPLAR": "few-shot exemplar of a Phase 1 question",
                                "CONF_EXEMPLAR": "few-shot exemplar of a confirmation question"}},
        unresolved=unresolved)
    report["unresolved_contamination_risks"] = UNRESOLVED_RISKS
    for p in outs:
        p.parent.mkdir(parents=True, exist_ok=True)
    outs[0].write_text(inv.canonical_json(report), encoding="utf-8", newline="\n")
    outs[1].write_text(ids_text, encoding="utf-8", newline="\n")
    outs[2].write_text(render_summary(report), encoding="utf-8", newline="\n")
    print("written", *map(str, outs), sep="\n  ")
    print("report sha256", C.sha256_bytes(outs[0].read_bytes()))
    return 3 if unresolved else 0


def render_summary(rep: dict) -> str:
    L = ["# Phase 3 fresh-data inventory (generated; inventory only)", "",
         f"Audit rule `{rep['audit_rule_version']}`; eligibility filter: at least {rep['min_clean']} clean questions. The filter is "
         "not evidence of statistical adequacy: see the precision column and decide per pool before spending it.", "",
         "No model outputs, correctness labels of any model, hidden states, thresholds or confirmation predictions were read. "
         "Counts below are rows; `primary` exclusions add up to `raw - clean`.", ""]
    for group, title in (("in_distribution", "In-distribution candidates"), ("new_benchmark", "New-benchmark (OOD) candidates")):
        L += [f"## {title}", "", "| pool | raw | used (P1/CONF eval) | used as exemplar | unlabeled | duplicates | clean | status | "
              "paired half-width at 10% discordance |", "|---|---|---|---|---|---|---|---|---|"]
        for key, s in rep["summary"][group].items():
            q = rep["pools"].get(key)
            if q is None:
                L.append(f"| {key} | n/a | | | | | n/a | **{s['status']}** | |")
                continue
            e = q["excluded_primary"]
            hw = q["precision"]["paired_difference_halfwidth"].get("0.1")
            L.append(f"| {key} | {q['raw']} | {e['P1_EVAL'] + e['CONF_EVAL']} | {e['P1_EXEMPLAR'] + e['CONF_EXEMPLAR']} | "
                     f"{e['UNLABELED']} | {e['INTERNAL_DUPLICATE'] + e['CROSS_POOL_DUPLICATE']} | {q['clean']} | **{s['status']}** | "
                     f"{'' if hw is None else '±' + str(hw)} |")
        L.append("")
    L += ["## Unresolved pools", ""]
    L += [f"- `{u['pool']}`: {u['reason']}" for u in rep["unresolved"]] or ["- none"]
    L += ["", "## Unresolved contamination risks", ""] + [f"- {r}" for r in rep["unresolved_contamination_risks"]]
    L += ["", "## Verification of the rebuilt used-set", "",
          f"- confirmation ids reproduce the committed sha256: {rep['inputs']['verification']['confirmation'].get('matches')}"]
    for b, c in rep["inputs"]["verification"]["phase1"].items():
        L.append(f"- Phase 1 {b}: doc ids equal splits.csv: {c['doc_ids_equal_splits_csv']}, n matches manifest: "
                 f"{c['n_questions_match']}, token-hash check: {c['question_set_hash_matches']}")
    L += ["", f"Report: `{OUT_JSON}`. Clean-row identities: `{OUT_IDS}` (sha256 in the report). "
          "The fresh allocation (which clean rows are dev, which are the one-time test) is NOT decided here.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
