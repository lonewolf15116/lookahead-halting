"""Phase 1, steps 1-2 (CPU only): build every question, count tokens, write the split manifest.

    python scripts/prepare_phase1.py                 # all six benchmarks + both synthetic tasks
    python scripts/prepare_phase1.py --limit 20      # quick check of the pipeline

Writes
  data/questions/<benchmark>.jsonl.gz   tokenised questions (gitignored; rebuilt deterministically)
  data/manifests/questions.json         per benchmark: count, question-set sha256, dataset revision
  data/manifests/splits.csv (+ .sha256) question-disjoint 60/20/20 splits, stratified
  results/phase1/token_counts.json/.csv token statistics and the runtime estimate (plan section 10)
"""
import argparse
import csv
import gzip
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from lookahead.config import REPO_ROOT, load_config, results_dir
from lookahead.questions import (BENCHMARKS, SYNTHETIC, load_lm_eval_questions, load_synthetic_questions,
                                 question_set_hash)
from lookahead.splits import assign_splits

HF_DATASETS = {"arc_challenge": "allenai/ai2_arc", "arc_easy": "allenai/ai2_arc", "hellaswag": "Rowan/hellaswag",
               "mmlu": "cais/mmlu", "commonsense_qa": "tau/commonsense_qa", "openbookqa": "allenai/openbookqa"}
MS_PER_TOKEN_LOOP = 0.1533  # RTX 5050, batch 1, bf16 (gpu_profile.json: 77.5 ms @ 512, 234.4 ms @ 1536); the pilot replaces it
LOOPS, MAX_LENGTH = 4, 4096


def dataset_revision(repo):
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(repo).sha
    except Exception as e:  # recorded, not fatal: the question-set hash is the real identity check
        return f"unavailable ({type(e).__name__})"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--benchmarks", nargs="*", default=list(BENCHMARKS) + list(SYNTHETIC))
    ap.add_argument("--root", default=None, help="write data/ and results/ under this folder (default: the repo)")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from lookahead.lm_eval_ouro import LoopedOuroLM

    tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / load_config("model")["local_dir"]))
    lm = LoopedOuroLM(None, tok)
    root = Path(args.root) if args.root else REPO_ROOT
    qdir, mdir = root / "data" / "questions", root / "data" / "manifests"
    rdir = root / "results" / "phase1" if args.root else results_dir("phase1")
    rdir.mkdir(parents=True, exist_ok=True)
    qdir.mkdir(parents=True, exist_ok=True)
    mdir.mkdir(parents=True, exist_ok=True)

    manifest, stats, split_rows = {}, [], []
    for b in args.benchmarks:
        t0 = time.time()
        qs = load_synthetic_questions(b, tok) if b in SYNTHETIC else load_lm_eval_questions(b, lm, limit=args.limit)
        if args.limit and b in SYNTHETIC:
            qs = qs[: args.limit]
        with gzip.open(qdir / f"{b}.jsonl.gz", "wt", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q.__dict__) + "\n")
        manifest[b] = {
            "n_questions": len(qs), "question_set_sha256": question_set_hash(qs),
            "hf_dataset": HF_DATASETS.get(b, "generated (data/synthetic)"),
            "hf_revision": dataset_revision(HF_DATASETS[b]) if b in HF_DATASETS else None,
            "shots": BENCHMARKS[b]["shots"] if b in BENCHMARKS else 5,
            "metric": qs[0].metric, "limit": args.limit,
        }
        ctx = [len(q.context_enc) for q in qs]
        packed = [q.packed_len for q in qs]
        per_opt = [q.per_option_tokens for q in qs]
        too_long = sum(len(q.context_enc) + max(len(c) for c in q.cont_encs) > MAX_LENGTH + 1 for q in qs)
        row = {
            "benchmark": b, "questions": len(qs),
            "options": dict(sorted(Counter(len(q.cont_encs) for q in qs).items())),
            "ctx_mean": round(statistics.mean(ctx), 1), "ctx_p50": pct(ctx, 50), "ctx_p95": pct(ctx, 95), "ctx_max": max(ctx),
            "packed_mean": round(statistics.mean(packed), 1), "packed_p95": pct(packed, 95), "packed_max": max(packed),
            "packed_tokens_total": sum(packed), "per_option_tokens_total": sum(per_opt),
            "est_packed_h": round(LOOPS * MS_PER_TOKEN_LOOP * sum(packed) / 3.6e6, 2),
            "est_per_option_h": round(LOOPS * MS_PER_TOKEN_LOOP * sum(per_opt) / 3.6e6, 2),
            "over_max_length": too_long,
        }
        stats.append(row)
        split_rows += [(q.benchmark, q.subtask, q.doc_id) for q in qs]
        print(f"{b:15s} {len(qs):6d} q  ctx mean {row['ctx_mean']:7.1f}  packed mean {row['packed_mean']:7.1f}  "
              f"est {row['est_packed_h']:.2f} h packed / {row['est_per_option_h']:.2f} h per-option  "
              f"({time.time() - t0:.0f} s)", flush=True)

    splits = assign_splits(split_rows)
    lines = ["benchmark,stratum,doc_id,split"] + [f"{b},{s},{d},{splits[(b, d)]}" for b, s, d in sorted(split_rows)]
    text = "\n".join(lines) + "\n"
    suffix = "" if args.limit is None else f"_limit{args.limit}"
    (mdir / f"splits{suffix}.csv").write_text(text, encoding="utf-8", newline="\n")
    split_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (mdir / f"splits{suffix}.sha256").write_text(split_sha + "\n", encoding="utf-8")
    counts = {b: dict(Counter(splits[(bb, d)] for bb, _, d in split_rows if bb == b)) for b in args.benchmarks}
    (mdir / f"questions{suffix}.json").write_text(json.dumps({"benchmarks": manifest, "splits_sha256": split_sha,
                                                              "split_counts": counts}, indent=2), encoding="utf-8")

    natural = [r for r in stats if r["benchmark"] in BENCHMARKS]
    total = {k: round(sum(r[k] for r in stats), 2) for k in ("questions", "est_packed_h", "est_per_option_h")}
    (rdir / f"token_counts{suffix}.json").write_text(json.dumps({"ms_per_token_loop": MS_PER_TOKEN_LOOP, "rows": stats,
                                                                 "total": total}, indent=2), encoding="utf-8")
    with (rdir / f"token_counts{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in stats[0] if k != "options"])
        w.writeheader()
        w.writerows({k: v for k, v in r.items() if k != "options"} for r in stats)
    print(f"\nTotal: {total['questions']} questions ({sum(r['questions'] for r in natural)} natural). Estimated extraction "
          f"{total['est_packed_h']} h packed, {total['est_per_option_h']} h per-option. Splits sha256 {split_sha[:12]}")
    if any(r["over_max_length"] for r in stats):
        print("WARNING: some questions exceed the 4096-token limit Gate 0 used; see over_max_length")
    return 0


if __name__ == "__main__":
    sys.exit(main())
