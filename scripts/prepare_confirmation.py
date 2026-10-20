"""Gate 2 confirmation set: build the questions and run the contamination audit (docs/GATE2_PLAN.md sections 3, 3a). CPU only.

    python scripts/prepare_confirmation.py                                   # real run: hellaswag and commonsense_qa 1000 each
    python scripts/prepare_confirmation.py --take hellaswag=500 --limit 50   # quick test, writes *_limit50 files only

Always writes results/gate2/confirmation_audit.json. Only if every check passes (zero overlap, zero verbatim failures, the
Phase 1 rebuild reproducing data/manifests/questions.json) does it write
  data/questions_confirm/<benchmark>.jsonl.gz   (gitignored; rebuilt deterministically)
  data/manifests/confirmation.json              per benchmark: n, question-set sha256, source split, dataset revision, shots
  data/manifests/confirmation_ids.csv (+ .sha256)
Exit code 0 = audit passed, 2 = audit failed (nothing but the audit file written). With --limit N the Phase 1 rebuild covers
only the first N docs per benchmark, the manifest comparison is skipped, and every output name carries _limitN.
"""
import argparse
import json
import sys
from pathlib import Path

from lookahead.config import REPO_ROOT, load_config
from lookahead.confirm import (DEFAULT_TAKE, EXPECTED_POOL, HF_DATASETS, SALT, SPLIT_PLAN, Exclusion, audit,
                               build_confirmation_questions, dataset_revision, ids_csv_text, lm_eval_version,
                               record_exemplars, sha256_bytes, task_yaml_hashes, write_questions)
from lookahead.questions import BENCHMARKS, load_lm_eval_questions, question_set_hash


def parse_take(items, benchmarks):
    take = {b: DEFAULT_TAKE.get(b) for b in benchmarks}
    for it in items or []:
        b, _, n = it.partition("=")
        if b not in take or not n.isdigit():
            raise SystemExit(f"bad --take {it!r}: expected benchmark=N with benchmark in {list(take)}")
        take[b] = int(n)
    return take


def main(argv=None, *, task_manager=None, lm=None, specs=None, split_plan=None, fetch_revision=None, expected_pool=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", nargs="*", default=[], help="benchmark=N (default: hellaswag=1000 commonsense_qa=1000, others all)")
    ap.add_argument("--limit", type=int, default=None, help="quick test: first N Phase 1 docs, at most N per benchmark, *_limitN outputs")
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS))
    ap.add_argument("--root", default=None, help="read/write data/ and results/ under this folder (default: the repo)")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else REPO_ROOT
    specs = specs or BENCHMARKS
    split_plan = split_plan or SPLIT_PLAN
    fetch_revision = fetch_revision or dataset_revision
    expected_pool = EXPECTED_POOL if expected_pool is None else expected_pool
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    take = parse_take(args.take, args.benchmarks)
    if args.limit:
        take = {b: min(n, args.limit) if n is not None else args.limit for b, n in take.items()}

    if lm is None:
        from transformers import AutoTokenizer

        from lookahead.lm_eval_ouro import LoopedOuroLM

        lm = LoopedOuroLM(None, AutoTokenizer.from_pretrained(str(REPO_ROOT / load_config("model")["local_dir"])))
    mdir, qdir, rdir = root / "data" / "manifests", root / "data" / f"questions_confirm{suffix}", root / "results" / "gate2"

    # 1. Phase 1 rebuilt with the sampler instrumented
    frozen = json.loads((mdir / "questions.json").read_text(encoding="utf-8"))["benchmarks"]
    manifest_hashes = {b: None if args.limit else frozen[b]["question_set_sha256"] for b in args.benchmarks}
    phase1, recorders, stored = {}, {}, {}
    for b in args.benchmarks:
        with record_exemplars() as rec:
            qs = load_lm_eval_questions(b, lm, limit=args.limit, task_manager=task_manager, spec=specs[b],
                                        on_task=rec.begin_task, on_doc=rec.on_doc)
        phase1[b], recorders[b] = (qs, rec), rec
        sp = root / "data" / "questions" / f"{b}.jsonl.gz"
        stored[b] = _stored_contexts(sp) if sp.exists() else None
        state = "not compared (--limit)" if args.limit else "matches" if manifest_hashes[b] == question_set_hash(qs) else "DIFFERS"
        print(f"{b:15s} phase 1 rebuilt: {len(qs)} q, {sum(len(r.inserted) for r in rec.by_doc_id.values())} exemplar "
              f"draws, hash {state}", flush=True)
    exclusion = Exclusion.from_recorders(recorders)

    # 2. confirmation questions
    builds = {}
    for b in args.benchmarks:
        builds[b] = build_confirmation_questions(b, lm, take[b], exclusion, salt=SALT, task_manager=task_manager,
                                                 spec=specs[b], split_plan=split_plan)
        st = builds[b].stats
        print(f"{b:15s} confirmation {st['split']}: pool {st['pool_size']}, excluded id {st['n_excluded_id']} / text "
              f"{st['n_excluded_text']}, eligible {st['n_eligible']}, taken {st['n_taken']}", flush=True)

    # 3. audit (always written)
    phase1_ids = _phase1_doc_ids(mdir / "splits.csv")
    from lm_eval.tasks import TaskManager

    tm = task_manager or TaskManager()
    fingerprints = {"task_yaml_sha256": task_yaml_hashes(tm, [t for bd in builds.values() for t in bd.task_names]),
                    "datasets": {}}
    for b in args.benchmarks:
        repo = HF_DATASETS.get(b) or _dataset_path(tm, b)
        fingerprints["datasets"][b] = {"hf_dataset": repo, "hf_revision": fetch_revision(repo), "source_split": builds[b].split}
    report = audit(phase1, builds, exclusion=exclusion, manifest_hashes=manifest_hashes, stored_contexts=stored,
                   phase1_doc_ids=phase1_ids, salt=SALT, take=take, fingerprints=fingerprints, expected_pool=expected_pool,
                   limit=args.limit)
    rdir.mkdir(parents=True, exist_ok=True)
    audit_path = rdir / f"confirmation_audit{suffix}.json"
    audit_text = json.dumps(report, indent=2, default=str)
    audit_path.write_text(audit_text, encoding="utf-8", newline="\n")
    if not report["passed"]:
        print(f"AUDIT FAILED: {', '.join(report['failed_checks'])}. See {audit_path}. No questions or manifest written.")
        return 2

    # 4. only now the questions and manifests
    qdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)
    for b, bd in builds.items():
        write_questions(qdir / f"{b}.jsonl.gz", bd.questions)
    ids_text = ids_csv_text(builds)
    ids_sha = sha256_bytes(ids_text.encode("utf-8"))
    (mdir / f"confirmation_ids{suffix}.csv").write_text(ids_text, encoding="utf-8", newline="\n")
    (mdir / f"confirmation_ids{suffix}.sha256").write_text(ids_sha + "\n", encoding="utf-8")
    manifest = {"salt": SALT, "limit": args.limit, "lm_eval_version": lm_eval_version(), "ids_sha256": ids_sha,
                "audit_sha256": sha256_bytes(audit_text.replace("\r\n", "\n").encode("utf-8")), "benchmarks": {
                    b: {"n_questions": len(bd.questions), "question_set_sha256": question_set_hash(bd.questions),
                        "source_split": bd.split, **fingerprints["datasets"][b], "shots": specs[b]["shots"],
                        "metric": specs[b]["metric"], "n_take": take[b], "pool_size": bd.stats["pool_size"]}
                    for b, bd in builds.items()}}
    (mdir / f"confirmation{suffix}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"AUDIT PASSED. {sum(len(bd.questions) for bd in builds.values())} confirmation questions; ids sha256 {ids_sha[:12]}")
    return 0


def _stored_contexts(path):
    from lookahead.extract import load_questions

    return {q.doc_id: q.context for q in load_questions(path)}


def _phase1_doc_ids(path):
    if not Path(path).exists():
        return set()
    from lookahead.extract import load_splits

    return {d for (_, d) in load_splits(path)}


def _dataset_path(tm, benchmark):
    try:
        return next(iter(tm.load([benchmark])["tasks"].values())).config.dataset_path
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


if __name__ == "__main__":
    sys.exit(main())
