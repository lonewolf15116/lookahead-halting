"""C1 freeze (docs/C1_PROTOCOL.md): everything chosen for the confirmatory study, from DEVELOPMENT data only.

Reuses the reviewed code unchanged: `e1` (the probe, the combined risk-controlled selection `select_risk`, the bootstrap
draws) and `phase3_g3a` (data, baseline statistics, evaluation). This module never reads TEST and has no TEST function:
the guarded one-time TEST script is a later step (protocol C1.7 step 3).

Frozen here (C1.2):
- pi1-x: per-loop L2 logistic on Gate 2's 9 cheap features, target S_t, Gate 2 frozen C = 1.0, refit on all development
  data; its selection statistic is the 5-fold cross-fitted out-of-fold score over all development data (lh-fold-v1).
- native-pb (Ouro's cdf, no fitted part) and B_pb (fixed depth, reference).
- Per primary benchmark (tau_b, m_b) for each, by `e1.select_risk` with the SAME bootstrap draws
  (rng default_rng([0, 99, benchmark index])); point-rule selections beside them (secondary 1).
- Every A2 baseline under the same rule (secondary 2).
- Task-agnostic pooled (tau, m) for the OOD benchmarks (secondary 4): one shared pair, qualifying iff EVERY development
  benchmark meets the combined rule, maximising the six-benchmark macro saving; same tie-breaks.
- Expected precision of the primary paired difference at the TEST sizes (C1.5), from development discordance.
"""
import math

import numpy as np

from lookahead import e1 as E
from lookahead import gate1 as g1
from lookahead import gate2 as g2
from lookahead import phase3_g3a as G
from lookahead import phase3_readouts as R

BOOT_FOLD_ID = 99                     # C1.2: rng default_rng([0, 99, benchmark index])
TEST_N = {"openbookqa": 3000, "commonsense_qa": 1000, "hellaswag": 750}   # A1 allocation (sealed TEST sizes)
DELTA = 0.02                          # C1.3 / D-b
N_BOOT_TEST = 10_000                  # C1.5 / D-d (used by the later TEST script; recorded here)
TEST_SEED = 2026
PRIMARY = G.PRIMARY


def pi1x_statistic(dev: G.DevData):
    """(selection statistic [N,3] = 5-fold OOF over all development data, final models fitted on all ok rows)."""
    S = E.targets(dev)["S"]
    oof = E.crossfit_e1(dev, "x", S)
    models = E.fit_e1(dev, np.flatnonzero(dev.ok), "x", S)
    return oof, models


def select_risk_pooled(d: G.DevData, S: np.ndarray, boot: dict, fixed_only: bool = False) -> dict:
    """Task-agnostic (A1.2, C1.6 item 4): ONE (tau, m) for all benchmarks. Qualifies iff, for EVERY benchmark of d, the point
    loss <= 1 pp and the bootstrap 95% upper bound <= 1 pp (same draws as select_risk). Maximises the macro saving over all
    benchmarks of d; frozen tie-breaks. tau = inf, m = 4 always qualifies."""
    allm = np.ones(d.N, dtype=bool)
    taus = [math.inf] if fixed_only else R.grid([S], [d.ok])
    benches = d.benchmarks
    rows = {b: E._rows(d, b, allm) for b in benches}
    cands = []
    for tau in taus:
        for m in (1, 2, 3, 4):
            per = {}
            for b in benches:
                r = rows[b]
                sav, loss = R.outcome(d.correct[r], d.ok[r], R.exit_loops(S[r], d.ok[r], tau, m))
                per[b] = (float(sav), float(loss))
            cands.append({"tau": tau, "m": m, "per": per,
                          "saving": float(np.mean([v[0] for v in per.values()])),
                          "loss": float(np.mean([v[1] for v in per.values()]))})
    cands.sort(key=lambda c: -c["saving"])

    def ub(c, b):
        r = rows[b]
        ok, corr = d.ok[r], d.correct[r]
        e = R.exit_loops(S[r], ok, c["tau"], c["m"])
        l = 100.0 * (np.where(ok, corr[:, 3], 0) - np.where(ok, corr[np.arange(len(r)), e - 1], 0))
        nz = np.flatnonzero(l)
        if not len(nz):
            return 0.0
        return float(np.quantile(boot[b][:, nz] @ l[nz] / len(r), E.RISK_QUANTILE))

    best, qual = None, []
    for c in cands:
        if best is not None and c["saving"] < best - g1.TOL:
            break
        if any(c["per"][b][1] > E.LIMIT for b in benches):
            continue
        ubs = {}
        for b in benches:
            ubs[b] = ub(c, b)
            if ubs[b] > E.LIMIT:
                break
        else:
            qual.append({**c, "upper_bounds": ubs})
            if best is None:
                best = c["saving"]
    assert qual, "no pooled candidate qualifies (fixed depth 4 must)"
    lo = min(c["loss"] for c in qual)
    tie = [c for c in qual if c["loss"] <= lo + g1.TOL]
    tie.sort(key=lambda c: (0 if math.isinf(c["tau"]) else 1, -c["tau"] if math.isfinite(c["tau"]) else 0, c["m"]))
    s = tie[0]
    return {"tau": s["tau"], "m": int(s["m"]), "macro_saving": s["saving"], "macro_loss": s["loss"],
            "per_benchmark": {b: {"saving": v[0], "loss": v[1]} for b, v in s["per"].items()},
            "upper_bounds": s["upper_bounds"]}


def _sel_json(sel: dict) -> dict:
    return {b: {k: v for k, v in s.items()} for b, s in sel.items()}


def expected_precision(dev: G.DevData, S_pi, S_nat, sel_pi, sel_nat) -> dict:
    """C1.5: expected half-width of the 95% interval of the primary paired difference at the TEST sizes, from the
    per-question development differences under the frozen selections (equal-weight macro, independent benchmarks)."""
    var = {}
    for b in PRIMARY:
        r = E._rows(dev, b, np.ones(dev.N, bool))
        e_pi = R.exit_loops(S_pi[r], dev.ok[r], sel_pi[b]["tau"], sel_pi[b]["m"])
        e_na = R.exit_loops(S_nat[r], dev.ok[r], sel_nat[b]["tau"], sel_nat[b]["m"])
        u = (e_na - e_pi) / 4.0
        var[b] = float(u.var(ddof=1))
    se = math.sqrt(sum(var[b] / TEST_N[b] for b in PRIMARY)) / len(PRIMARY)
    return {"per_question_variance_dev": var, "se_macro": se, "expected_half_width_95": 1.96 * se,
            "test_n": TEST_N, "note": "planning estimate from development data under the frozen selections"}


def freeze(dev: G.DevData) -> dict:
    """Everything frozen for C1 from development data (no file is written here). Returns (frozen dict, pi1-x models)."""
    if (dev.source == "p1_validation").any() or (dev.source == "test").any():
        raise PermissionError("C1 selection uses development data only (D-a)")
    allm = np.ones(dev.N, dtype=bool)
    boot = E.make_boot(dev, allm, BOOT_FOLD_ID)
    S_pi, models = pi1x_statistic(dev)
    stats = {n: G.baseline_stat(dev, n) for n in G.BASELINES}
    S_all = {**stats, "pi1-x": S_pi}
    risk = {n: E.select_risk(dev, S, allm, n == "B_pb", boot) for n, S in S_all.items()}
    point = {n: G.select_all(dev, S, allm, fixed_only=(n == "B_pb")) for n, S in S_all.items()}
    pooled = {n: select_risk_pooled(dev, S_all[n], boot, fixed_only=(n == "B_pb")) for n in ("pi1-x", "native-pb", "B_pb")}
    macro = lambda sel: G.macro({b: {"saving": sel[b]["saving"]} for b in PRIMARY}, PRIMARY)  # noqa: E731
    frozen = {
        "protocol": "docs/C1_PROTOCOL.md", "delta": DELTA, "n_boot_test": N_BOOT_TEST, "test_seed": TEST_SEED,
        "selection_rule": "combined: point loss <= 1 pp AND bootstrap 95% upper bound <= 1 pp (e1.select_risk)",
        "bootstrap_selection": {"n": E.N_BOOT, "rng": f"default_rng([0, {BOOT_FOLD_ID}, benchmark index in sorted list])",
                                "fingerprint": E.boot_fingerprint(boot)},
        "pi1x": {"features": list(g2.CHEAP_NAMES), "target": "S", "C": E.FEAT_C["x"],
                 "selection_statistic": "5-fold out-of-fold over all development data (lh-fold-v1)",
                 "model_sha256": G.model_fingerprint(models)},
        "primary": {"pi1-x": _sel_json({b: risk["pi1-x"][b] for b in PRIMARY}),
                    "native-pb": _sel_json({b: risk["native-pb"][b] for b in PRIMARY})},
        "reference_B_pb": _sel_json({b: risk["B_pb"][b] for b in PRIMARY}),
        "secondary_risk_all_baselines": {n: _sel_json({b: risk[n][b] for b in PRIMARY}) for n in G.BASELINES},
        "secondary_point_rule": {n: _sel_json({b: point[n][b] for b in PRIMARY}) for n in ("pi1-x", "native-pb", "B_pb")},
        "ood_pooled": pooled,
        "selection_macro_saving_dev": {n: macro(risk[n]) for n in S_all},
        "expected_precision": expected_precision(dev, S_pi, stats["native-pb"], risk["pi1-x"], risk["native-pb"]),
        "n_dev": {b: int((dev.bench == b).sum()) for b in dev.benchmarks},
    }
    return frozen, models
