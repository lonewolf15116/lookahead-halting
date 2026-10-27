"""Exploratory study X1 (docs/X1_FLIP_PROTOCOL.md rev 2): correctness flips at matched saving. DEVELOPMENT DATA ONLY, [E].

Does pi1-x change the correctness of fewer questions than native-pb at the same gross saving? Held-out statistics come from
`e1.build_methods` (unchanged), per outer fold. Thresholds are matched to a saving level WITHOUT correctness
(`match_saving` receives no correctness array); only then are gains/losses read, interpolated with the same weight w.
This module never opens a file; real data is only loaded by `phase3_g3a.load_g3a` in the entry script.
"""
import json

import numpy as np

from lookahead import e1 as E
from lookahead import gate2 as g2
from lookahead import phase3_g3a as G
from lookahead import phase3_readouts as R

PROTOCOL = "docs/X1_FLIP_PROTOCOL.md rev 2"
STATUS = "exploratory"
LEVELS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
CORE = (0.15, 0.20, 0.25, 0.30)
CAPS = {"primary": {"openbookqa": 4, "commonsense_qa": 4, "hellaswag": 3},
        "sensitivity_cap4": {"openbookqa": 4, "commonsense_qa": 4, "hellaswag": 4}}
METHODS = ("native-pb", "pi1-x", "r-kl", "pi1-x-nocdf")
CONTRASTS = {"pi1-x_vs_native-pb": ("pi1-x", "native-pb"), "r-kl_vs_native-pb": ("r-kl", "native-pb"),
             "pi1-x-nocdf_vs_native-pb": ("pi1-x-nocdf", "native-pb")}
PRIMARY_CONTRAST = "pi1-x_vs_native-pb"
MEASURES = ("phi", "kappa", "net", "g", "l")
N_BOOT = 2000
SEED = [31, 1]
TOL = 1e-12
Z95 = 1.645


def _lv(s: float) -> str:
    return f"{s:.2f}"


# ------------------------------------------------------------------------------------------- label-free matching
def saving_curve(S_rows: np.ndarray, ok_rows: np.ndarray, cap: int):
    """(taus ascending, savings): candidates are the sorted distinct finite values of S_rows[:, :cap-1] plus +inf;
    saving = mean(1 - e/4) with e from `exit_loops`. Uses no correctness."""
    v = S_rows[:, :cap - 1]
    taus = np.append(np.unique(v[np.isfinite(v)]), np.inf)
    sav = np.array([np.mean(1.0 - R.exit_loops(S_rows, ok_rows, t, cap) / 4.0) for t in taus])
    return taus, sav


def pick_bracket(taus: np.ndarray, sav: np.ndarray, s: float):
    """Adjacent candidates bracketing s: {tau1, tau2, saving1, saving2, w} with s = (1-w)*saving1 + w*saving2, saving1 <= s
    <= saving2. A candidate with saving == s (1e-12; the largest such tau on ties) gives tau1 = tau2 and w = 0. None if s is
    not attainable. Saving is non-increasing in tau, so the candidates above s form a prefix and those below a suffix."""
    eq = np.flatnonzero(np.abs(sav - s) <= TOL)
    if len(eq):
        j = eq[-1]
        return {"tau1": float(taus[j]), "tau2": float(taus[j]), "saving1": float(sav[j]), "saving2": float(sav[j]), "w": 0.0}
    below, above = np.flatnonzero(sav < s), np.flatnonzero(sav > s)
    if not len(below) or not len(above):
        return None
    j1, j2 = below[0], above[-1]            # adjacent in tau: smallest tau under s, largest tau over s
    return {"tau1": float(taus[j1]), "tau2": float(taus[j2]), "saving1": float(sav[j1]), "saving2": float(sav[j2]),
            "w": float((s - sav[j1]) / (sav[j2] - sav[j1]))}


def match_saving(S_rows, ok_rows, cap, s):
    """Label-free threshold matching at saving level s (see `saving_curve`, `pick_bracket`). No correctness argument."""
    return pick_bracket(*saving_curve(S_rows, ok_rows, cap), s)


# ------------------------------------------------------------------------------------------------ flips and rates
def flip_vectors(S_rows, ok_rows, correct_rows, tau, cap):
    """Per-question (gain, loss) 0/1 vectors of the policy (tau, cap): gain = correct at exit and wrong at loop 4; loss =
    wrong at exit and correct at loop 4. Failed rows exit at loop 4 and count as wrong (no gain, no loss)."""
    e = R.exit_loops(S_rows, ok_rows, tau, cap)
    c_exit = np.where(ok_rows, correct_rows[np.arange(len(e)), e - 1], 0)
    c4 = np.where(ok_rows, correct_rows[:, 3], 0)
    return ((c_exit == 1) & (c4 == 0)).astype(float), ((c_exit == 0) & (c4 == 1)).astype(float)


def interpolated_flips(S_rows, ok_rows, correct_rows, cap, m):
    """Per-question gain/loss vectors at a matched level: (1-w) * (tau1 policy) + w * (tau2 policy)."""
    g1, l1 = flip_vectors(S_rows, ok_rows, correct_rows, m["tau1"], cap)
    g2_, l2 = flip_vectors(S_rows, ok_rows, correct_rows, m["tau2"], cap)
    return (1 - m["w"]) * g1 + m["w"] * g2_, (1 - m["w"]) * l1 + m["w"] * l2


def rates(g, l) -> dict:
    """Measures from pooled gain/loss rates (fractions): phi = g + l, kappa = 2 min(g, l), net = (l - g) in pp."""
    g, l = np.asarray(g, dtype=float), np.asarray(l, dtype=float)
    return {"g": g, "l": l, "phi": g + l, "kappa": 2 * np.minimum(g, l), "net": 100.0 * (l - g)}


def implied_margin(g, l, n_b: int):
    """Descriptive screen margin 1.645 * sqrt((phi - (l-g)^2) / n_b), as a fraction (benchmark's full development count)."""
    g, l = np.asarray(g, dtype=float), np.asarray(l, dtype=float)
    return Z95 * np.sqrt(np.maximum(g + l - (l - g) ** 2, 0.0) / n_b)


# ------------------------------------------------------------------------------------------------------ bootstrap
def draw_counts(sizes: dict, n_boot: int = N_BOOT) -> dict:
    """One multinomial count vector (size n, equal probabilities) per stratum in each replicate. `sizes`: {(benchmark,
    fold): n}; strata processed in sorted order within a replicate; rng = default_rng([31, 1]). Returns {stratum: int32
    [n_boot, n]}; the same counts serve every method, threshold and measure."""
    rng = np.random.default_rng(SEED)
    out = {k: np.empty((n_boot, n), dtype=np.int32) for k, n in sizes.items()}
    for r in range(n_boot):
        for k in sorted(sizes):
            out[k][r] = rng.multinomial(sizes[k], np.full(sizes[k], 1.0 / sizes[k]))
    return out


def _ci(a: np.ndarray) -> dict:
    """a[0] = full-sample value, a[1:] = bootstrap replicates; two-sided 95% percentile interval."""
    lo, hi = np.quantile(a[1:], [0.025, 0.975])
    return {"point": float(a[0]), "lo": float(lo), "hi": float(hi)}


class Pool:
    """Per-stratum interpolated flip vectors for every (cap set, method, level), pooled over folds with per-question
    weights. Column layout: [g..., l...] over `cols`. `acc[b]` has row 0 = full sample and rows 1.. = replicates, holding the
    pooled g and l of benchmark b (per-question pooling: sum over folds of counts . vector / N_b)."""

    def __init__(self, cols, vecs, n_strata, counts, matched_ok):
        self.cols, self.ci, self.ok = cols, {c: i for i, c in enumerate(cols)}, matched_ok
        C = len(cols)
        self.N, self.acc, self.fold_pt = {}, {}, {}
        for (b, k), n in n_strata.items():
            V = np.zeros((n, 2 * C))
            for c, (gv, lv) in vecs.get((b, k), {}).items():
                V[:, self.ci[c]], V[:, C + self.ci[c]] = gv, lv
            self.N[b] = self.N.get(b, 0) + n
            self.fold_pt[(b, k)] = V.mean(axis=0)
            full = np.vstack([V.sum(axis=0), counts[(b, k)].astype(float) @ V])
            self.acc[b] = self.acc.get(b, 0) + full
        for b in self.acc:
            self.acc[b] = self.acc[b] / self.N[b]

    def gl(self, b, cs, m, s, fold=None):
        """(g, l) arrays [B+1] pooled over folds, or scalars for one fold's full-sample rates."""
        i, C = self.ci[(cs, m, s)], len(self.cols)
        src = self.acc[b] if fold is None else self.fold_pt[(b, fold)]
        return src[..., i], src[..., C + i]


def contrast(pool: Pool, cs: str, m1: str, m2: str, benches, folds=range(g2.N_FOLDS)) -> dict:
    """Per core/grid level: included benchmarks, macro measures per method, deltas (m1 - m2) with intervals, per-fold dphi."""
    out = {}
    for s in LEVELS:
        inc = [b for b in benches if pool.ok[(cs, m1, b, s)] and pool.ok[(cs, m2, b, s)]]
        if not inc:
            out[_lv(s)] = None
            continue
        macro, per_b = {}, {}
        for m in (m1, m2):
            rb = {b: rates(*pool.gl(b, cs, m, s)) for b in inc}
            macro[m] = {k: np.mean([rb[b][k] for b in inc], axis=0) for k in MEASURES}
            per_b[m] = {b: {**{k: float(rb[b][k][0]) for k in MEASURES},
                            "implied_margin": float(implied_margin(rb[b]["g"][0], rb[b]["l"][0], pool.N[b])),
                            "implied_margin_pp": float(100 * implied_margin(rb[b]["g"][0], rb[b]["l"][0], pool.N[b]))}
                        for b in inc}
        pf = [float(np.mean([rates(*pool.gl(b, cs, m1, s, k))["phi"] - rates(*pool.gl(b, cs, m2, s, k))["phi"]
                             for b in inc])) for k in folds]
        out[_lv(s)] = {"included": inc, "macro": {m: {k: _ci(v) for k, v in macro[m].items()} for m in macro},
                       "per_benchmark_point": per_b,
                       "delta": {k: _ci(macro[m1][k] - macro[m2][k]) for k in MEASURES}, "per_fold_dphi": pf}
    return out


# ------------------------------------------------------------------------------------------------------- readings
def reading(results: dict) -> dict:
    """Protocol 8.1. `results`: a contrast's {level string: entry | None} as made by `contrast`. Rows evaluated in order
    (first match wins); F, K, L on the core levels. A core level with no included benchmark fails every condition."""
    F, K, L, nonneg = [], [], [], 0
    for s in CORE:
        e = results.get(_lv(s))
        if not e:
            continue
        d = e["delta"]
        f = d["phi"]["point"] < 0 and d["phi"]["hi"] < 0 and sum(v < 0 for v in e["per_fold_dphi"]) >= 4
        k = d["kappa"]["point"] < 0 and d["kappa"]["hi"] < 0
        l = d["l"]["point"] < 0 and d["l"]["hi"] < 0 and abs(d["l"]["point"]) > abs(d["g"]["point"]) and not k
        nonneg += d["phi"]["point"] >= 0
        for flag, acc in ((f, F), (k, K), (l, L)):
            if flag:
                acc.append(_lv(s))
    FK = [s for s in F if s in K]
    losses = len([s for s in F if s in L]) >= 3
    if len(FK) >= 3:
        name = "FLIP-SUPPORTED"
    elif len(F) >= 3:
        name = "FLIP REDUCTION, CANCELLING NOT DEMONSTRATED"
    elif not F and nonneg >= 3:
        name = "NOT SUPPORTED"
    else:
        name = "MIXED"
    return {"reading": name, "F_levels": F, "K_levels": K, "L_levels": L, "FK_levels": FK,
            "losses_sentence_allowed": bool(name.startswith("FLIP REDUCTION") and losses)}


# ----------------------------------------------------------------------------------------------------------- the run
def _jf(x: float):
    return "inf" if np.isinf(x) else float(x)


def _mrec(x):
    return None if x is None else {**x, "tau1": _jf(x["tau1"]), "tau2": _jf(x["tau2"])}


def run_x1(dev: G.DevData, n_boot: int = N_BOOT) -> dict:
    """All of X1 on in-memory development data; returns a JSON-serialisable dict (no file is written here)."""
    if (dev.source == "p1_validation").any() or (dev.source == "test").any():
        raise PermissionError("development data must not contain validation or sealed rows")
    missing = [b for b in G.PRIMARY if b not in dev.benchmarks]
    if missing:
        raise ValueError(f"primary benchmarks missing from the development data: {missing}")
    prim = sorted(G.PRIMARY)
    base = {n: G.baseline_stat(dev, n) for n in G.BASELINES}
    stats, fps = {}, []
    for k in range(g2.N_FOLDS):
        ms, fp = E.build_methods(dev, dev.fold != k, dev.fold == k, base)
        stats[k] = {m: ms[m] for m in METHODS}
        fps.append({n: v for n, v in fp.items() if v})
    rows = {(b, k): np.flatnonzero((dev.bench == b) & (dev.fold == k)) for b in prim for k in range(g2.N_FOLDS)}
    sizes = {key: len(r) for key, r in rows.items()}
    if min(sizes.values()) == 0:
        raise ValueError("an outer fold has no held-out questions for a primary benchmark")
    cols = [(cs, m, s) for cs in CAPS for m in METHODS for s in LEVELS]
    match, vecs, attain = {}, {key: {} for key in rows}, {}
    for cs, caps in CAPS.items():
        for m in METHODS:
            for (b, k), r in rows.items():
                curve = saving_curve(stats[k][m][r], dev.ok[r], caps[b])
                for s in LEVELS:
                    mt = match[(cs, m, b, s, k)] = pick_bracket(*curve, s)
                    if mt:
                        vecs[(b, k)][(cs, m, s)] = interpolated_flips(stats[k][m][r], dev.ok[r], dev.correct[r], caps[b], mt)
            for b in prim:
                for s in LEVELS:
                    attain[(cs, m, b, s)] = all(match[(cs, m, b, s, k)] for k in range(g2.N_FOLDS))
    pool = Pool(cols, vecs, sizes, draw_counts(sizes, n_boot), attain)
    rep = {"protocol": PROTOCOL, "status": STATUS, "n_boot": n_boot, "bootstrap_seed": SEED, "levels": list(LEVELS),
           "core_levels": list(CORE), "caps": CAPS, "primary_contrast": PRIMARY_CONTRAST,
           "n_dev": {b: int((dev.bench == b).sum()) for b in prim},
           "fold_sizes": {b: [sizes[(b, k)] for k in range(g2.N_FOLDS)] for b in prim},
           "contrasts": {}, "included_benchmarks": {}}
    for cs in CAPS:
        rep["contrasts"][cs], rep["included_benchmarks"][cs] = {}, {}
        for name, (m1, m2) in CONTRASTS.items():
            lev = contrast(pool, cs, m1, m2, prim)
            rd = reading(lev)
            gov = cs == "primary" and name == PRIMARY_CONTRAST
            rd["governing"] = gov
            if not gov:
                rd["note"] = "descriptive only; does not govern the X1 reading"
            rep["contrasts"][cs][name] = {
                "methods": [m1, m2], "levels": lev, "reading": rd,
                "matching": {m: {b: {_lv(s): {str(k): _mrec(match[(cs, m, b, s, k)]) for k in range(g2.N_FOLDS)}
                                         for s in LEVELS} for b in prim}
                             for m in (m1, m2)}}
            rep["included_benchmarks"][cs][name] = {l: (e["included"] if e else []) for l, e in lev.items()}
    rep["reading"] = rep["contrasts"]["primary"][PRIMARY_CONTRAST]["reading"]
    rep["six_benchmark_macro"] = None
    rep["six_benchmark_macro_note"] = ("omitted: the protocol defines caps and bootstrap strata only for the three primary "
                                       "benchmarks, so a six-benchmark macro would need undefined caps for the other three")
    rep["fold_fingerprints"] = fps
    return json.loads(json.dumps(rep, default=str))
