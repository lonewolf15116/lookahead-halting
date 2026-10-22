"""Phase 3 (A1 frozen): build the DEV+ and sealed TEST questions from the frozen allocation, and check them before extraction.

Questions are built exactly as Phase 1 and the Gate 2 confirmation set were (lm_eval prompt format, Phase 1 shots and few-shot
seed), with one addition required by A1: the few-shot pool of every task is stripped of every allocated row (TEST and DEV+), so
no TEST or DEV+ question is ever shown as an exemplar, to any question. Exemplars therefore come only from rows the pipeline had
already used. Every built doc must reproduce its allocation text hash.

TEST questions live under paths containing `phase3_test`. `assert_not_sealed` refuses such paths (and the Gate 2 `confirm`
paths); every development or selection loader must call it. Nothing here reads a model output.
"""
import json
from pathlib import Path

from lookahead import confirm as C
from lookahead.questions import BENCHMARKS, load_lm_eval_questions, question_set_hash

# A1 frozen artifacts (docs/PHASE3_AMENDMENT_A1.md, tag phase3-a1)
PINNED = {"allocation_json": "1327e1cca5360f9b80de7759360b6d06519e8c881aad984e988a723417ea9a1b",
          "test_csv": "298873a90b26617685fa872e6f77825784829a8bb677f075575cff4c32cefeff",
          "dev_csv": "c1ed6e3851c6ee2c3deec178a67970f11133272b529bc4fdb8a04b2344d097d1"}
# A1.4: OOD 0-shot. Metric = the one lm_eval's task reports first (PIQA acc_norm is its standard headline; Social IQa has acc only)
OOD_SPECS = {"piqa": {"shots": 0, "metric": "acc_norm"}, "social_iqa": {"shots": 0, "metric": "acc"}}
ROLE_SPLIT = {"dev": ("dev",), "test": ("test", "test_ood")}
QDIR = {"dev": "questions_phase3_dev", "test": "questions_phase3_test"}
XDIR = {"dev": "extracted_phase3_dev", "test": "extracted_phase3_test"}
MANIFEST = "data/manifests/phase3_questions.json"
IDS = {"dev": "data/manifests/phase3_ids_dev.csv", "test": "data/manifests/phase3_ids_test.csv"}
SEALED_MARKERS = ("phase3_test", "confirm")


def assert_not_sealed(path) -> None:
    p = str(path).replace("\\", "/").lower()
    for m in SEALED_MARKERS:
        if m in p:
            raise PermissionError(f"sealed data must not be opened by development or selection code: {path}")


def _sha_ok(path: Path, expected: str) -> bool:
    raw = Path(path).read_bytes()
    return expected in (C.sha256_bytes(raw), C.sha256_bytes(raw.replace(b"\r\n", b"\n")))


def read_allocation(root: Path, pinned: dict = PINNED) -> dict:
    """Frozen allocation -> {role: {benchmark: (source_split, [row dicts])}}. SystemExit on any hash mismatch."""
    root = Path(root)
    a = root / "results/phase3/allocation.json"
    t, d = root / "results/phase3/allocation_test_ids.csv", root / "results/phase3/allocation_dev_ids.csv"
    if not _sha_ok(a, pinned["allocation_json"]):
        raise SystemExit("allocation.json is not the frozen A1 file")
    rep = json.loads(a.read_text(encoding="utf-8"))
    for path, key, rkey in ((t, "test_csv", "test_ids_sha256"), (d, "dev_csv", "dev_ids_sha256")):
        if not _sha_ok(path, pinned[key]) or not _sha_ok(path, rep[rkey]):
            raise SystemExit(f"{path.name} is not the frozen A1 list")
    out = {"dev": {}, "test": {}}
    for path in (t, d):
        lines = path.read_text(encoding="utf-8").splitlines()
        head = lines[0].split(",")
        for ln in lines[1:]:
            r = dict(zip(head, ln.split(",")))
            role = "dev" if r["role"] == "dev" else "test"
            r["row"] = int(r["row"])
            split, rows = out[role].setdefault(r["benchmark"], (r["source_split"], []))
            if split != r["source_split"]:
                raise SystemExit(f"{r['benchmark']}: more than one source split in role {role}")
            rows.append(r)
    return out


def build_set(benchmark, lm, rows, split, tm, spec, excluded_texts: set, text_hash_fn, role):
    """Build the questions of `rows` (one benchmark, one source split). Returns (questions, report).

    Raises on any identity problem (a built doc whose text hash differs from its allocation row, duplicate ids). Exemplar
    problems are counted in the report; the caller refuses to write anything if any count is non-zero.
    """
    by_task: dict = {}
    for r in rows:
        by_task.setdefault(r["task"], []).append(r)
    index_of = {t: sorted(rs, key=lambda r: r["row"]) for t, rs in by_task.items()}
    report = {"benchmark": benchmark, "role": role, "split": split, "n_rows": len(rows), "exemplar_draws": 0,
              "exemplar_is_allocated_row": 0, "self_exemplar": 0, "pool_removed": 0, "pool_size_after": 0}
    built = []
    with C.record_exemplars() as rec:
        def on_task(task):
            C.switch_eval_split(task, split)
            s = getattr(task, "sampler", None)
            if s is not None:
                docs = list(s.fewshot_docs())
                keep = [d for d in docs if text_hash_fn(benchmark, d, task) not in excluded_texts]
                report["pool_removed"] += len(docs) - len(keep)
                report["pool_size_after"] += len(keep)
                s.df, s._loaded = keep, True
            rec.begin_task(task)

        def on_doc(q, task, doc):
            r = index_of[q.subtask][int(q.doc_id.rsplit("/", 1)[1])]  # lm_eval renumbers built docs 0..k-1 in order
            th = text_hash_fn(benchmark, doc, task)
            if th != r["text_hash"]:
                raise RuntimeError(f"{benchmark} {q.subtask}[{r['row']}]: text hash differs from the frozen allocation")
            rec.on_doc(q, task, doc)
            qr = rec.by_doc_id[q.doc_id]
            report["exemplar_draws"] += len(qr.inserted)
            report["exemplar_is_allocated_row"] += sum(d.text_hash in excluded_texts for d in qr.superset + qr.inserted)
            report["self_exemplar"] += sum(d.text_hash == th for d in qr.inserted)
            new = f"p3:{q.subtask}/{split}@r{r['row']}"
            rec.rekey(q.doc_id, new)
            q.doc_id = new
            built.append(q)

        load_lm_eval_questions(benchmark, lm, task_manager=tm, spec=spec, on_task=on_task, on_doc=on_doc,
                               samples_for=lambda name: [r["row"] for r in index_of.get(name, [])])
        report["live_exemplar_failures"] = len(rec.live_failures)
    ids = [q.doc_id for q in built]
    if len(built) != len(rows) or len(set(ids)) != len(ids) or any("," in i or "\n" in i for i in ids):
        raise RuntimeError(f"{benchmark}/{role}: built {len(built)} questions for {len(rows)} rows, or ids not unique/safe")
    order = {(r["task"], r["row"]): k for k, r in enumerate(rows)}
    built.sort(key=lambda q: order[(q.subtask, int(q.doc_id.rsplit("@r", 1)[1]))])
    report["question_set_sha256"] = question_set_hash(built)
    return built, report


def ids_csv_text(sets: dict, role: str) -> str:
    """sets: {(role, benchmark): (split_label, [questions])} -> one role's load_splits-compatible CSV
    (split = dev, or test | test_ood)."""
    rows = sorted((b, q.subtask, q.doc_id, lab) for (r, b), (lab, qs) in sets.items() if r == role for q in qs)
    return "\n".join(["benchmark,stratum,doc_id,split"] + [",".join(r) for r in rows]) + "\n"


def check_phase3_inputs(root: Path, benches, limit, role, allocation_sha: str = PINNED["allocation_json"]):
    """Frozen-input check for `extract_phase1.py --set phase3_dev|phase3_test` (same contract as check_confirmation_inputs).
    Returns (manifest, ids_sha, splits, questions). SystemExit on any mismatch."""
    from lookahead.extract import load_questions, load_splits

    root = Path(root)
    man = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if not _sha_ok(root / IDS[role], man["ids_sha256"][role]):
        raise SystemExit(f"{Path(IDS[role]).name} does not match its manifest")
    if man.get("allocation_json_sha256") != allocation_sha:
        raise SystemExit("phase3 manifest was not built from the frozen A1 allocation")
    splits = load_splits(root / IDS[role], allowed=ROLE_SPLIT[role])
    questions, bman = {}, {}
    for b in benches:
        entry = man["sets"].get(f"{role}/{b}")
        if entry is None:
            raise SystemExit(f"{b}: no {role} set in the phase3 manifest")
        qs = load_questions(root / "data" / QDIR[role] / f"{b}.jsonl.gz")
        if question_set_hash(qs) != entry["question_set_sha256"]:
            raise SystemExit(f"{b}/{role}: questions do not match the manifest hash; re-run scripts/prepare_phase3.py")
        if sum(1 for (bb, _) in splits if bb == b) != len(qs):
            raise SystemExit(f"{b}/{role}: question count differs from the id list")
        questions[b] = qs[:limit] if limit else qs
        bman[b] = {k: entry.get(k) for k in ("hf_dataset", "hf_revision", "question_set_sha256", "shots", "metric")}
    return {"benchmarks": bman}, man["ids_sha256"][role], splits, questions


def default_specs() -> dict:
    return {**BENCHMARKS, **OOD_SPECS}
