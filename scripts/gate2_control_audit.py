"""Gate 2 shuffled-control audit and per-benchmark fixed-depth analysis. TRAIN and VALIDATION only; CPU only.

    python scripts/gate2_control_audit.py [--root .]

Reads nothing from the confirmation set (gate2.load_probe_data refuses any path containing 'confirm'), and does not
modify results/gate2/frozen_selection.json. Writes results/gate2/control_audit.json.

Sections
 A. Shuffle verification: the labels used for control (c) are re-derived and compared to the true labels.
 B. Frozen P/Q reproduced from the saved model files: threshold, saving, loss must match the frozen file.
 C. Control (c) refitted with the frozen code: must reproduce the validation_report numbers.
 D. Benchmark-prior probe (no features at all: per-benchmark, per-loop train mean of S) through the same selection rule.
 E. Per-benchmark stopping behaviour of P, Q, control (c), D at their selected thresholds (validation).
 F. Fixed-depth analyses on validation: pooled fixed-k, and per-benchmark depth vectors under two candidate rules.
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from lookahead import gate1 as g
from lookahead import gate2 as g2

LOOPS = 4


def load_probe(root: Path, entry: dict):
    arrs = dict(np.load(root / entry["model_file"], allow_pickle=False))
    scal = dict(np.load(root / entry["standardisation_file"], allow_pickle=False))
    return g2.Probe.from_arrays(arrs, scal)


def stopping(val, scores, thr):
    """Per-benchmark exit-loop distribution, saving and loss at threshold thr (validation)."""
    data = g2.probe_policy_data(val, scores)
    out = {}
    for b in val.benchmarks:
        d = data[b]
        e, c = g.run_policy(d, "native_gate", thr)
        acc4 = 100.0 * d["correct"][:, 3].mean()
        out[b] = {"n": int(d["n"]), "mean_exit": float(e.mean()),
                  "frac_exit": {str(k): float((e == k).mean()) for k in range(1, 5)},
                  "saving": float(1 - e.mean() / LOOPS), "loss": float(acc4 - 100.0 * c.mean())}
    out["_macro"] = {"saving": float(np.mean([out[b]["saving"] for b in val.benchmarks])),
                     "loss": float(np.mean([out[b]["loss"] for b in val.benchmarks]))}
    return out


def op_summary(val, scores):
    op = g2.select_operating_point(g2.probe_policy_data(val, scores), val.benchmarks)
    s = op["selected"]
    return op, {"qualified": op["qualified"], "threshold": op["threshold"],
                "saving": None if s is None else s["macro_saving"], "loss": None if s is None else s["macro_loss"]}


def fixed_depth(val, benchmarks):
    per = {b: {k: g.summarize(val.arrays, [b], "fixed", k)["per_benchmark"][b] for k in (1, 2, 3, 4)} for b in benchmarks}
    table = {b: {str(k): {"saving": per[b][k]["saving"], "loss": per[b][k]["loss"], "acc": per[b][k]["acc"]}
                 for k in (1, 2, 3, 4)} for b in benchmarks}
    pooled = {}
    for k in (1, 2, 3, 4):
        s = g.summarize(val.arrays, list(benchmarks), "fixed", k)
        pooled[str(k)] = {"macro_saving": s["macro"]["saving"], "macro_loss": s["macro"]["loss"],
                          "pooled_saving": s["pooled"]["saving"], "pooled_loss": s["pooled"]["loss"]}
    # rule M: depth vector maximising macro saving subject to macro loss <= LOSS_LIMIT (+TOL); tie -> smaller loss -> lexicographic
    rows = []
    for combo in itertools.product((1, 2, 3, 4), repeat=len(benchmarks)):
        sav = float(np.mean([per[b][k]["saving"] for b, k in zip(benchmarks, combo)]))
        los = float(np.mean([per[b][k]["loss"] for b, k in zip(benchmarks, combo)]))
        rows.append({"depths": combo, "macro_saving": sav, "macro_loss": los})
    q = [r for r in rows if r["macro_loss"] <= g.LOSS_LIMIT + g.TOL]
    best = max(r["macro_saving"] for r in q)
    c = [r for r in q if r["macro_saving"] >= best - g.TOL]
    lo = min(r["macro_loss"] for r in c)
    c = sorted([r for r in c if r["macro_loss"] <= lo + g.TOL], key=lambda r: r["depths"])
    rule_m = c[0]
    # rule B: each benchmark takes its smallest depth whose own loss <= LOSS_LIMIT (+TOL)
    depths_b = []
    for b in benchmarks:
        ok = [k for k in (1, 2, 3, 4) if per[b][k]["loss"] <= g.LOSS_LIMIT + g.TOL]
        depths_b.append(min(ok))
    rule_b = {"depths": tuple(depths_b),
              "macro_saving": float(np.mean([per[b][k]["saving"] for b, k in zip(benchmarks, depths_b)])),
              "macro_loss": float(np.mean([per[b][k]["loss"] for b, k in zip(benchmarks, depths_b)]))}
    return {"per_benchmark": table, "pooled_fixed": pooled, "rule_M_macro_constraint": rule_m,
            "rule_B_per_benchmark_constraint": rule_b, "n_combinations": len(rows), "n_qualified_M": len(q),
            "benchmark_order": list(benchmarks)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root).resolve()
    frozen = json.loads((root / "results/gate2/frozen_selection.json").read_text(encoding="utf-8"))
    report = json.loads((root / "results/gate2/validation_report.json").read_text(encoding="utf-8"))
    benchmarks = frozen["benchmarks"]
    train = g2.load_probe_data(root, "train", benchmarks)
    val = g2.load_probe_data(root, "validation", benchmarks, g_clip=train.g_clip)
    print(f"train {train.N} / validation {val.N}", flush=True)
    out = {"label": "gate2-control-audit (train and validation only)", "benchmarks": benchmarks}

    # A. shuffle verification
    y = train.targets["S"]
    ys = g2.shuffle_labels_within(train, y)
    A = {"seed": g2.SEED, "unchanged_outside_ok": bool(np.array_equal(ys[~train.ok], y[~train.ok])), "per_benchmark_loop": {}}
    for b in benchmarks:
        a, e = train.slices[b]
        rows = a + np.flatnonzero(train.ok[a:e])
        for t in range(g2.PROBE_LOOPS):
            o, s = y[rows, t], ys[rows, t]
            A["per_benchmark_loop"][f"{b}/loop{t + 1}"] = {
                "n": int(len(rows)), "rate_true": float(o.mean()), "rate_shuffled": float(s.mean()),
                "multiset_equal": bool(np.array_equal(np.sort(o), np.sort(s))),
                "frac_positions_changed": float((o != s).mean()),
                "corr_true_vs_shuffled": None if o.std() == 0 or s.std() == 0 else float(np.corrcoef(o, s)[0, 1])}
    A["note"] = ("labels are permuted within (benchmark, loop): each benchmark keeps its own base rate of S, so a probe "
                 "that can read benchmark identity from h still learns the benchmark prior")
    out["A_shuffle"] = A

    # B. reproduce frozen P and Q
    P = load_probe(root, frozen["P"])
    Q = load_probe(root, frozen["Q"])
    p_scores, q_scores = P.predict(val), Q.predict(val)
    p_op, p_sum = op_summary(val, p_scores)
    q_op, q_sum = op_summary(val, q_scores)
    out["B_frozen_reproduced"] = {
        "P": {**p_sum, "frozen_threshold": frozen["P"]["threshold"], "frozen_saving": frozen["P"]["validation_saving"],
              "frozen_loss": frozen["P"]["validation_loss"],
              "threshold_matches": bool(abs(p_sum["threshold"] - frozen["P"]["threshold"]) < 1e-12)},
        "Q": {**q_sum, "frozen_threshold": frozen["Q"]["threshold"], "frozen_saving": frozen["Q"]["validation_saving"],
              "frozen_loss": frozen["Q"]["validation_loss"],
              "threshold_matches": bool(abs(q_sum["threshold"] - frozen["Q"]["threshold"]) < 1e-12)}}
    print("B", json.dumps(out["B_frozen_reproduced"]), flush=True)

    # C. refit control (c)
    shuf = g2.fit_probe(train, 1, "h", "S", labels=ys)
    c_scores = shuf.predict(val)
    c_op, c_sum = op_summary(val, c_scores)
    ref = [r for r in report["control_records"] if r["name"].startswith("control-c")][0]
    out["C_control_refit"] = {**c_sum, "report_threshold": ref["threshold"], "report_saving": ref["saving"],
                              "report_loss": ref["loss"],
                              "reproduces_report": bool(abs(c_sum["saving"] - ref["saving"]) < 1e-9)}
    print("C", json.dumps(out["C_control_refit"]), flush=True)

    # D. benchmark-prior probe: per-benchmark, per-loop train mean of S; no features
    prior = np.zeros((val.N, g2.PROBE_LOOPS))
    prior_rates = {}
    for b in benchmarks:
        ta, te = train.slices[b]
        rows = ta + np.flatnonzero(train.ok[ta:te])
        rates = y[rows].mean(axis=0)
        prior_rates[b] = [float(r) for r in rates]
        va, ve = val.slices[b]
        prior[va:ve] = rates
    d_op, d_sum = op_summary(val, prior)
    out["D_benchmark_prior"] = {**d_sum, "train_S_rates_per_loop": prior_rates}
    print("D", json.dumps(d_sum), flush=True)

    # E. stopping behaviour
    out["E_stopping"] = {}
    for name, sc, summ in (("P", p_scores, p_sum), ("Q", q_scores, q_sum), ("control_c", c_scores, c_sum),
                           ("benchmark_prior", prior, d_sum)):
        if summ["threshold"] is not None:
            out["E_stopping"][name] = {"threshold": summ["threshold"], **stopping(val, sc, summ["threshold"])}
    for nme, s in out["E_stopping"].items():
        print(f"E {nme} thr={s['threshold']:.4f} macro saving {s['_macro']['saving']:.3f} loss {s['_macro']['loss']:.3f}")
        for b in benchmarks:
            print(f"    {b:15s} exit-dist {[round(s[b]['frac_exit'][str(k)], 2) for k in range(1, 5)]} "
                  f"saving {s[b]['saving']:.3f} loss {s[b]['loss']:.2f}")

    # G. within-benchmark AUROC for S (validation): does any scorer rank questions inside a benchmark?
    G = {}
    Sv = val.targets["S"]
    for name, sc in (("P", p_scores), ("Q", q_scores), ("control_c", c_scores), ("benchmark_prior", prior)):
        G[name] = {}
        for b in benchmarks:
            a, e = val.slices[b]
            rows = a + np.flatnonzero(val.ok[a:e])
            G[name][b] = [g2._auroc(Sv[rows, t], sc[rows, t]) for t in range(g2.PROBE_LOOPS)]
        print(f"G {name} within-benchmark AUROC (loops 1-3):")
        for b in benchmarks:
            print("    %-15s %s" % (b, [None if v is None else round(v, 3) for v in G[name][b]]))
    out["G_within_benchmark_auroc_S"] = G

    # F. fixed-depth analyses
    out["F_fixed_depth"] = fixed_depth(val, benchmarks)
    fd = out["F_fixed_depth"]
    print("F rule M:", fd["rule_M_macro_constraint"])
    print("F rule B:", fd["rule_B_per_benchmark_constraint"])
    print("F pooled fixed:", json.dumps(fd["pooled_fixed"]))

    path = root / "results/gate2/control_audit.json"
    g.write_json(path, out)
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
