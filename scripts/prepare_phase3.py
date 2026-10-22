"""Phase 3 (A1 frozen): build the DEV+ and sealed TEST questions. CPU only; needs the Ouro tokenizer and the datasets.

    python scripts/prepare_phase3.py

Reads only the frozen A1 allocation (hash-checked), the datasets (revisions must equal the inventory's) and the tokenizer. No
model output exists yet and none is read. Writes, refusing to overwrite:
  data/questions_phase3_dev/<benchmark>.jsonl.gz    DEV+ questions (gitignored; rebuilt deterministically)
  data/questions_phase3_test/<benchmark>.jsonl.gz   TEST questions (gitignored; sealed path)
  data/manifests/phase3_questions.json             per set: n, question-set sha256, shots, metric, revision, exemplar checks
  data/manifests/phase3_ids_dev.csv, phase3_ids_test.csv   benchmark,stratum,doc_id,split (dev; test | test_ood)
Always writes results/phase3/build_report.json. If any check fails (an exemplar that is an allocated row or not a previously
used row, a self-exemplar, an exemplar missing from its context, a revision change, a used-set verification failure), only
the report is written and the exit code is 2. The used-set is rebuilt and verified exactly as in the inventory.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

from lookahead import confirm as C
from lookahead import phase3_build as P
from lookahead.config import REPO_ROOT, load_config

_spec = importlib.util.spec_from_file_location("phase3_inventory_script", REPO_ROOT / "scripts" / "phase3_inventory.py")
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)

OOD = ("piqa", "social_iqa")


def main(argv=None, *, task_manager=None, lm=None, root=None, pinned=None, specs=None, fetch_revision=None, ood=OOD,
         used_texts=None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    root = Path(root or REPO_ROOT)
    pinned = pinned or P.PINNED
    specs = specs or P.default_specs()
    fetch_revision = fetch_revision or C.dataset_revision
    outs = [root / P.MANIFEST, root / P.IDS["dev"], root / P.IDS["test"], root / "data" / P.QDIR["dev"], root / "data" / P.QDIR["test"]]
    if any(p.exists() for p in outs):
        print("refusing: phase3 questions or manifests already exist:", [str(p) for p in outs if p.exists()], file=sys.stderr)
        return 2
    alloc = P.read_allocation(root, pinned)
    excluded = {r["text_hash"] for role in alloc.values() for _, rows in role.values() for r in rows}

    inv = json.loads((root / "results/phase3/inventory.json").read_text(encoding="utf-8"))
    revisions = inv["provenance"]["revisions"]
    benches = sorted({b for role in alloc.values() for b in role})
    report = {"allocation_json_sha256": pinned["allocation_json"], "n_allocated_rows": len(excluded), "sets": {},
              "revisions": {}, "failures": []}
    for b in benches:
        expected = revisions.get(b, {})
        now = fetch_revision(expected.get("hf_dataset")) if expected.get("hf_dataset") else "unavailable (no dataset path)"
        report["revisions"][b] = {"hf_dataset": expected.get("hf_dataset"), "inventory": expected.get("hf_revision"), "now": now}
        if now != expected.get("hf_revision"):
            report["failures"].append(f"{b}: dataset revision {now} differs from the inventory's {expected.get('hf_revision')}")

    if lm is None:
        from transformers import AutoTokenizer

        from lookahead.lm_eval_ouro import LoopedOuroLM

        lm = LoopedOuroLM(None, AutoTokenizer.from_pretrained(str(REPO_ROOT / load_config("model")["local_dir"])))
    from lm_eval.tasks import TaskManager

    tm = task_manager or TaskManager()
    if used_texts is None and not report["failures"]:
        from lookahead.questions import BENCHMARKS

        print("rebuilding the verified used-set (permitted exemplar pool) ...", flush=True)
        used, checks, fails = S.build_used_set(root, list(BENCHMARKS), lm, tm, BENCHMARKS, C.SPLIT_PLAN,
                                               {b: C.DEFAULT_TAKE.get(b) for b in BENCHMARKS}, C.SALT, True)
        report["used_set"] = {"verification": checks, "fingerprint": used.fingerprint(), "n_texts": len(used.texts)}
        report["failures"] += [f"used-set: {f}" for f in fails]
        used_texts = set(used.texts)
    sets = {}
    if not report["failures"]:
        for role in ("dev", "test"):
            for b, (split, rows) in sorted(alloc[role].items()):
                qs, rep = P.build_set(b, lm, rows, split, tm, specs[b], excluded, S.inv_text_hash, role, used_texts)
                label = "test_ood" if (role == "test" and b in ood) else role
                rep.update({"split_label": label, "shots": specs[b]["shots"], "metric": specs[b]["metric"],
                            "hf_dataset": report["revisions"][b]["hf_dataset"], "hf_revision": report["revisions"][b]["now"]})
                report["sets"][f"{role}/{b}"] = rep
                sets[(role, b)] = (label, qs)
                bad = (rep["exemplar_is_allocated_row"] + rep["exemplar_not_previously_used"] + rep["self_exemplar"]
                       + rep["live_exemplar_failures"])
                if bad:
                    report["failures"].append(f"{role}/{b}: {bad} exemplar violations")
                print(f"{role}/{b}: {len(qs)} questions, {rep['exemplar_draws']} exemplars, pool -{rep['pool_removed']}, "
                      f"violations {bad}", flush=True)
    rdir = root / "results/phase3"
    rdir.mkdir(parents=True, exist_ok=True)
    report["passed"] = not report["failures"]
    (rdir / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    if not report["passed"]:
        print("BUILD CHECKS FAILED:", report["failures"])
        return 2
    for (role, b), (_, qs) in sets.items():
        d = root / "data" / P.QDIR[role]
        d.mkdir(parents=True, exist_ok=True)
        C.write_questions(d / f"{b}.jsonl.gz", qs)
    ids_sha = {}
    for role in ("dev", "test"):
        ids = P.ids_csv_text(sets, role)
        (root / P.IDS[role]).write_text(ids, encoding="utf-8", newline="\n")
        ids_sha[role] = C.sha256_bytes(ids.encode("utf-8"))
    manifest = {"amendment": "A1", "allocation_json_sha256": pinned["allocation_json"],
                "ids_sha256": ids_sha, "lm_eval_version": C.lm_eval_version(),
                "build_report_sha256": C.sha256_path(rdir / "build_report.json"),
                "sets": {k: {kk: v[kk] for kk in ("n_rows", "question_set_sha256", "shots", "metric", "hf_dataset", "hf_revision",
                                                   "split", "split_label", "exemplar_draws", "pool_removed",
                                                   "exemplar_selection_sha256")}
                         for k, v in report["sets"].items()}}
    (root / P.MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"BUILD PASSED. {sum(len(qs) for _, qs in sets.values())} questions; ids sha256 dev {ids_sha['dev'][:12]} test {ids_sha['test'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
