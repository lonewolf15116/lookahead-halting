"""Gate 0 accuracy check: ARC-Challenge, 25-shot, Ouro-1.4B at loops 1-4, through lm-evaluation-harness.

The Ouro paper (Table 10) reports ARC-C at each loop using this framework. Resumable: every scored request is cached
in results/gate0/arc_cache.jsonl, and one pass over the data gives all four loops.

    python scripts/reproduce_ouro.py --limit 25     # smoke test of the pipeline
    python scripts/reproduce_ouro.py                # full test set: 1172 questions, 4688 requests

Writes results/gate0/ouro_reproduction.csv (and per-question samples). The 2-point accuracy criterion is judged only on
the full test set: a subset is too noisy to say anything about it.
"""
import argparse
import csv
import json
import sys
import time

import lm_eval

from lookahead.config import results_dir
from lookahead.lm_eval_ouro import LoopedOuroLM
from lookahead.ouro_adapter import load_ouro

PAPER = {1: 37.63, 2: 54.86, 3: 59.47, 4: 60.92}  # Ouro paper Table 10, ARC-C 25-shot (acc vs acc_norm not stated)
TASK, SHOTS, TOLERANCE = "arc_challenge", 25, 2.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="score only the first N questions (smoke test)")
    args = parser.parse_args()

    out_dir = results_dir("gate0")
    model, tok = load_ouro()
    lm = LoopedOuroLM(model, tok, cache_path=out_dir / "arc_cache.jsonl")
    print(f"{len(lm.cache)} requests already cached", flush=True)

    rows = []
    started = time.time()
    for loop in range(1, lm.runner.total_loops + 1):
        lm.exit_loop = loop
        result = lm_eval.simple_evaluate(
            model=lm, tasks=[TASK], num_fewshot=SHOTS, limit=args.limit, batch_size=1, log_samples=True
        )
        r = result["results"][TASK]
        n = result["n-samples"][TASK]["effective"]
        row = {
            "loop": loop, "n": n,
            "acc": 100 * r["acc,none"], "acc_stderr": 100 * r["acc_stderr,none"],
            "acc_norm": 100 * r["acc_norm,none"], "acc_norm_stderr": 100 * r["acc_norm_stderr,none"],
            "paper": PAPER[loop],
        }
        row["diff_acc"] = row["acc"] - PAPER[loop]
        row["diff_acc_norm"] = row["acc_norm"] - PAPER[loop]
        rows.append(row)
        with (out_dir / f"arc_samples_loop{loop}.jsonl").open("w", encoding="utf-8") as f:
            for s in result["samples"][TASK]:
                f.write(json.dumps({"doc_id": s["doc_id"], "target": s["target"], "resps": s["resps"],
                                    "acc": s["acc"], "acc_norm": s["acc_norm"]}) + "\n")
        print(f"loop {loop}: acc {row['acc']:.2f}  acc_norm {row['acc_norm']:.2f}  (paper {PAPER[loop]})  n={n}", flush=True)

    csv_name = "ouro_reproduction.csv" if args.limit is None else f"ouro_reproduction_limit{args.limit}.csv"
    with (out_dir / csv_name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{lm.newly_scored} requests scored this run in {(time.time() - started) / 60:.1f} min. Wrote {out_dir / csv_name}")
    if args.limit is not None:
        print("Smoke test only: no Gate 0 verdict (a subset is too noisy for the 2-point check).")
        return 0
    verdict = {m: all(abs(r[f"diff_{m}"]) <= TOLERANCE for r in rows) for m in ("acc", "acc_norm")}
    print(f"Within {TOLERANCE} points of the paper at every loop: {verdict}")
    (out_dir / "gate0_accuracy.json").write_text(json.dumps({"within_tolerance": verdict, "rows": rows}, indent=2), encoding="utf-8")
    return 0 if any(verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
