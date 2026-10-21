"""Gate 2 amendment v3.1: freeze the benchmark-specific fixed-depth baseline B. TRAIN and VALIDATION only; CPU only.

    python scripts/gate2_freeze_baseline.py [--root .]

B is a task-aware control: it uses benchmark identity, which a deployable halting policy would not have.
It is derived two ways that must agree (they are two constructions of one policy, not independent evidence):
  M. exhaustive search: all 4^6 depth vectors; max macro validation saving subject to macro loss <= 1 pp (+TOL);
     ties -> smaller loss -> lexicographically smaller vector (the Gate 1 rule with a deterministic last tie-break).
  P. benchmark prior: score every question by its benchmark's TRAIN mean of S_t (no features at all), apply the Gate 1
     operating-point rule on validation, and read each benchmark's exit loop (must be a single loop per benchmark).
Also frozen: pooled fixed depth 3 (secondary control) and the per-benchmark 1 pp-cap vector (sensitivity only).
Writes results/gate2/baseline_B_frozen.json. Refuses to overwrite. Does not touch results/gate2/frozen_selection.json.
Reads nothing from the confirmation set (gate2.load_probe_data refuses any path containing 'confirm').
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from lookahead import gate1 as g
from lookahead import gate2 as g2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate2_control_audit import fixed_depth  # noqa: E402  (same function the audit used)

AMENDMENT = "v3.1"


def prior_route(train, val, benchmarks):
    S = train.targets["S"]
    scores = np.zeros((val.N, g2.PROBE_LOOPS))
    rates = {}
    for b in benchmarks:
        a, e = train.slices[b]
        rows = a + np.flatnonzero(train.ok[a:e])
        r = S[rows].mean(axis=0)
        rates[b] = [float(x) for x in r]
        va, ve = val.slices[b]
        scores[va:ve] = r
    data = g2.probe_policy_data(val, scores)
    op = g2.select_operating_point(data, benchmarks)
    if not op["qualified"]:
        raise SystemExit("benchmark-prior policy has no qualifying operating point")
    thr = op["threshold"]
    depths, dist = [], {}
    for b in benchmarks:
        e, _ = g.run_policy(data[b], "native_gate", thr)
        ks, cnt = np.unique(e, return_counts=True)
        dist[b] = {int(k): float(c / len(e)) for k, c in zip(ks, cnt)}
        depths.append(int(ks[np.argmax(cnt)]))
    return {"train_S_rates_per_loop": rates, "threshold": thr, "exit_distribution": dist, "depths": tuple(depths),
            "macro_saving": op["selected"]["macro_saving"], "macro_loss": op["selected"]["macro_loss"],
            "per_benchmark_saving": op["selected"]["per_benchmark_saving"],
            "per_benchmark_loss": op["selected"]["per_benchmark_loss"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    out_path = root / "results/gate2/baseline_B_frozen.json"
    if out_path.exists():
        print(f"refusing: {out_path} exists", file=sys.stderr)
        return 2
    frozen = json.loads((root / "results/gate2/frozen_selection.json").read_text(encoding="utf-8"))
    benchmarks = frozen["benchmarks"]
    prov = {}
    train = g2.load_probe_data(root, "train", benchmarks, provenance=prov)
    val = g2.load_probe_data(root, "validation", benchmarks, g_clip=train.g_clip, provenance=prov)

    fd = fixed_depth(val, benchmarks)
    m = fd["rule_M_macro_constraint"]
    p = prior_route(train, val, benchmarks)
    agree = tuple(m["depths"]) == tuple(p["depths"])
    same_perf = abs(m["macro_saving"] - p["macro_saving"]) < 1e-12 and abs(m["macro_loss"] - p["macro_loss"]) < 1e-12
    print("exhaustive:", m["depths"], round(m["macro_saving"], 6), round(m["macro_loss"], 6))
    print("prior     :", p["depths"], round(p["macro_saving"], 6), round(p["macro_loss"], 6))
    if not (agree and same_perf):
        print("STOP: the two constructions disagree; nothing written", file=sys.stderr)
        return 3
    if tuple(m["depths"]) != (2, 2, 2, 3, 3, 2):
        print("STOP: vector differs from the one the reviewer approved (2,2,2,3,3,2)", file=sys.stderr)
        return 3

    # near-ties documented so the selection is reproducible and its margin is visible
    rows = []
    import itertools
    per = {b: {k: g.summarize(val.arrays, [b], "fixed", k)["per_benchmark"][b] for k in (1, 2, 3, 4)} for b in benchmarks}
    for combo in itertools.product((1, 2, 3, 4), repeat=len(benchmarks)):
        rows.append({"depths": combo,
                     "macro_saving": float(np.mean([per[b][k]["saving"] for b, k in zip(benchmarks, combo)])),
                     "macro_loss": float(np.mean([per[b][k]["loss"] for b, k in zip(benchmarks, combo)]))})
    qual = sorted([r for r in rows if r["macro_loss"] <= g.LOSS_LIMIT + g.TOL],
                  key=lambda r: (-r["macro_saving"], r["macro_loss"], r["depths"]))

    obj = {
        "amendment": AMENDMENT,
        "label": "frozen on train+validation only; confirmation set not opened",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_order": benchmarks,
        "task_aware": "B uses benchmark identity; it is a task-aware control, not a deployable policy",
        "B": {"depths": list(m["depths"]), "validation_macro_saving": m["macro_saving"],
              "validation_macro_loss": m["macro_loss"],
              "per_benchmark_validation": {b: {"depth": d, "saving": per[b][d]["saving"], "loss": per[b][d]["loss"]}
                                           for b, d in zip(benchmarks, m["depths"])},
              "selection_rule": "max macro validation saving over all 4^6 depth vectors subject to macro loss <= "
                                f"{g.LOSS_LIMIT} pp (+{g.TOL}); ties -> smaller loss -> lexicographically smaller vector",
              "n_qualified_vectors": len(qual), "top10_qualified": qual[:10]},
        "B_via_benchmark_prior": {"description": "feature-free probe: benchmark TRAIN mean of S_t per loop, Gate 1 "
                                  "operating-point rule on validation; same policy as B by construction, not "
                                  "independent evidence", **{k: (list(v) if isinstance(v, tuple) else v)
                                                              for k, v in p.items()}},
        "constructions_agree": True,
        "pooled_fixed_3": {"depth": 3, "macro_saving": fd["pooled_fixed"]["3"]["macro_saving"],
                           "macro_loss": fd["pooled_fixed"]["3"]["macro_loss"],
                           "note": "pooled fixed depth 2 breaks the budget (macro loss "
                                   f"{fd['pooled_fixed']['2']['macro_loss']:.2f} pp); depth 3 is the shallowest compliant"},
        "sensitivity_per_benchmark_cap": {"depths": list(fd["rule_B_per_benchmark_constraint"]["depths"]),
                                          "macro_saving": fd["rule_B_per_benchmark_constraint"]["macro_saving"],
                                          "macro_loss": fd["rule_B_per_benchmark_constraint"]["macro_loss"],
                                          "rule": "each benchmark takes its shallowest depth with its own loss <= 1 pp "
                                                  "(sensitivity only; not a Gate 2 condition)"},
        "fixed_depth_table_validation": fd["per_benchmark"],
        "frozen_selection_sha256": g.sha256_file(root / "results/gate2/frozen_selection.json"),
        "input_hashes": prov, "code": g.code_provenance(),
    }
    g.write_json(out_path, obj)
    print("written", out_path, "sha256", g.sha256_file(out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
