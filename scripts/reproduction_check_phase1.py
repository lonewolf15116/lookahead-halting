"""Phase 1 sanity checks fixed in PHASE1_PLAN.md before extraction (not gates, no new thresholds):

1. Free reproduction check (section 1): per-loop accuracy against Ouro Table 10 for the benchmarks the paper reports,
   with the Gate 0 rule (within 2 points; acc and acc_norm both reported). A miss means investigate before Gate 1.
2. Learnability rule (section 2): a synthetic task or level whose loop-4 accuracy is below the level's majority-class
   rate + 10 points is reported as not solved and dropped from Phases 2-3.

Uses all questions (all splits), as the reproduction check is defined on the full benchmark. Reads only the stored
option log-likelihoods; needs pandas + pyarrow, not torch.

    python scripts/reproduction_check_phase1.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = {  # Ouro-1.4B, Table 10, loops 1-4
    "arc_challenge": [37.63, 54.86, 59.47, 60.92], "arc_easy": [63.85, 80.30, 83.33, 83.96],
    "hellaswag": [55.24, 71.15, 74.07, 74.29], "mmlu": [41.21, 60.43, 66.71, 67.45],
    "commonsense_qa": [44.64, 67.98, 74.37, 75.43],
}
ALL = ["arc_challenge", "arc_easy", "hellaswag", "mmlu", "commonsense_qa", "openbookqa", "modarith", "parity"]
TOL = 2.0


def load(b):
    parts = sorted((ROOT / "data" / "extracted" / b).glob("part-*.parquet"))
    done = {p.name.replace(".done.json", "") for p in (ROOT / "data" / "extracted" / b).glob("*.done.json")}
    parts = [p for p in parts if p.name.replace(".parquet", "") in done]
    return pd.concat([pd.read_parquet(p, columns=["doc_id", "stratum", "split", "target", "loop", "status", "opt_ll",
                                                  "opt_lln", "correct"]) for p in parts], ignore_index=True)


def accuracies(df):
    acc = (df.opt_ll.map(lambda x: int(np.argmax(x))) == df.target)
    acc_norm = (df.opt_lln.map(lambda x: int(np.argmax(x))) == df.target)
    return acc, acc_norm


def main() -> int:
    man = json.loads((ROOT / "data" / "synthetic" / "manifest.json").read_text(encoding="utf-8"))
    report = {"reproduction": {}, "learnability": {}}
    for b in ALL:
        df = load(b)
        df["acc"], df["acc_norm"] = accuracies(df)
        df.loc[df.status != "ok", ["acc", "acc_norm"]] = False  # failures count as wrong (section 7)
        per_loop = df.groupby("loop")[["acc", "acc_norm", "correct"]].mean().mul(100).round(2)
        row = {"n": int((df.loop == 1).sum()), "acc": per_loop.acc.tolist(), "acc_norm": per_loop.acc_norm.tolist(),
               "primary": per_loop.correct.tolist()}
        if b == "mmlu":
            sub = df.groupby(["stratum", "loop"]).acc.mean().unstack().mean().mul(100).round(2)
            row["acc_subject_macro"] = sub.tolist()
        if b in PAPER:
            row["paper"] = PAPER[b]
            for m in ("acc", "acc_norm"):
                row[f"diff_{m}"] = [round(x - p, 2) for x, p in zip(row[m], PAPER[b])]
                row[f"within_{TOL}_{m}"] = all(abs(d) <= TOL for d in row[f"diff_{m}"])
            row["within_tolerance_either"] = row[f"within_{TOL}_acc"] or row[f"within_{TOL}_acc_norm"]
        report["reproduction"][b] = row
        if b in ("modarith", "parity"):
            lvl = df[df.loop == 4].groupby("stratum").correct.mean().mul(100).round(2)
            out = {}
            for stratum, a in lvl.items():
                level = stratum.split("-")[1]
                maj = 100 * man["benchmarks"][b]["majority_class_rate"][level]
                out[stratum] = {"loop4_acc": float(a), "majority_class": round(maj, 1), "bar": round(maj + 10, 1),
                                "solved": bool(a >= maj + 10)}
            report["learnability"][b] = out
        flag = "" if b not in PAPER else ("  within 2 pts: " + ("yes" if row["within_tolerance_either"] else "NO") +
                                          f" (acc {row[f'within_{TOL}_acc']}, acc_norm {row[f'within_{TOL}_acc_norm']})")
        print(f"{b:15s} n={row['n']:6d}  acc {row['acc']}  acc_norm {row['acc_norm']}" +
              (f"  paper {PAPER[b]}" if b in PAPER else "") + flag)
    if "acc_subject_macro" in report["reproduction"]["mmlu"]:
        print(f"mmlu subject-macro acc {report['reproduction']['mmlu']['acc_subject_macro']}")
    for b, lv in report["learnability"].items():
        for s, v in lv.items():
            print(f"{s:14s} loop-4 acc {v['loop4_acc']:6.2f}  bar {v['bar']:5.1f}  {'solved' if v['solved'] else 'NOT SOLVED'}")
    (ROOT / "results" / "phase1" / "reproduction_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
