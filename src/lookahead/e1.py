"""Exploratory study E1 (docs/EXPLORATORY_E1_NATIVE_GATE.md, rev 1): why does Ouro's native gate beat the post-hoc probes?

EXPLORATORY DIAGNOSTICS ON DEVELOPMENT DATA ONLY. No confirmatory claim, no policy is selected for any test.
It reuses the G3a code (`phase3_g3a`, unchanged) and the FROZEN selection code (`phase3_readouts`): the same folds
(lh-fold-v1), the same per-benchmark rule (tau_b, m_b), the same 51-point grid, gross saving, primary macro with the
six-benchmark macro beside it. Only the factor under study changes between methods.

Data boundaries. Real data is only ever loaded by `phase3_g3a.load_g3a` (its sealed-path guard, id allow-list and deny-list
stay in force); this module never opens a file. `run_e1` refuses development data that contains validation or sealed rows.
Validation (A7) is descriptive: nothing is selected on it, and the A7 selection function does not receive it.

Nested protocol (as G3a rev 2). In outer fold k, the statistics of the SELECTION rows (fold != k) come from INNER
cross-fitting that uses only selection rows; the held-out rows' statistics come from models fitted on all selection rows.
Everything chosen for selection (models, rank maps, bootstrap draws, (tau, m)) is a function of selection rows only.

No future information (rev 1). A score at loop t is a function of loop-t features of the same question only (cummax: loops
<= t; rank: loop-t map). Labels (targets) may use later loops: they are labels, not inputs.
"""
import hashlib
import json
import math

import numpy as np

from lookahead import gate1 as g1
from lookahead import gate2 as g2
from lookahead import phase3_g3a as G
from lookahead import phase3_readouts as R
from lookahead.phase3_g3a import BASELINES, FROZEN_C, PRIMARY

PLAN = "docs/EXPLORATORY_E1_NATIVE_GATE.md rev 1"
STATUS = "exploratory; no confirmatory claim"
LOOPS = 4
CDF_COL = 3                                   # the native gate's cumulative exit probability inside Gate 2's cheap features
assert g2.CHEAP_NAMES[CDF_COL] == "cdf", "cheap-feature layout changed: column 3 is no longer the native cdf"
N_BOOT = 2000                                  # A6 bootstrap resamples (plan A6; as in Gate 2)
RISK_QUANTILE = 0.95                           # one-sided 95% upper bound
LIMIT = g1.LOSS_LIMIT + g1.TOL                 # 1 pp (+ the frozen tolerance)
FEAT_C = {"x": FROZEN_C["x"], "cdf": FROZEN_C["x"], "x-nocdf": FROZEN_C["x"], "hx": FROZEN_C["hx"]}   # frozen Gate 2 C
LADDER = ("native-pb", "pi1-cdf", "pi1-x-nocdf", "pi1-x", "pi1-hx")          # A1
# probe variants: name -> (feature set, target, transform). Every one goes through the SAME nested protocol.
PROBES = {"pi1-cdf": ("cdf", "S", None), "pi1-x-nocdf": ("x-nocdf", "S", None), "pi1-x": ("x", "S", None),
          "pi1-hx": ("hx", "S", None), "pi1-x@c_t": ("x", "c_t", None), "pi1-x@safe": ("x", "safe", None),
          "pi1-x+cummax": ("x", "S", "cummax"), "pi1-x+rank": ("x", "S", "rank")}
A2_METHODS = ("pi1-x", "pi1-x@c_t", "pi1-x@safe")
A3_METHODS = ("pi1-x", "pi1-x+cummax", "pi1-x+rank")
EXTRA = ("combo-max",)                         # A5: elementwise max(native cdf, pi1-x)
A7_METHODS = ("native-pb",) + LADDER[1:] + A2_METHODS[1:] + A3_METHODS[1:]
ALL_METHODS = BASELINES + tuple(PROBES) + EXTRA


# ------------------------------------------------------------------------------------------------ features, targets
def design(d: G.DevData, rows: np.ndarray, t: int, feat: str) -> np.ndarray:
    """Loop-t design matrix of the given rows. Only loop-t arrays are read (no future information)."""
    x = d.x[rows, t]
    if feat == "x":
        return x
    if feat == "hx":
        return np.concatenate([d.h[rows, t], x], axis=1)
    if feat == "cdf":
        return x[:, CDF_COL:CDF_COL + 1]
    if feat == "x-nocdf":
        return np.delete(x, CDF_COL, axis=1)
    raise ValueError(feat)


def targets(d: G.DevData) -> dict:
    """[N, 3] labels for loops 1..3. S: answer never changes again; c_t: correct now; safe: stopping now loses nothing
    against loop 4 (c_t >= c_4). Labels may use later loops."""
    c = d.correct[:, :3]
    return {"S": d.stability(), "c_t": c.astype(float), "safe": (c >= d.correct[:, [3]]).astype(float)}


def fit_e1(d: G.DevData, rows: np.ndarray, feat: str, target: np.ndarray) -> list:
    """Per-loop (scaler, model) on `rows`; exactly phase3_g3a.fit_loopwise with E1's feature sets."""
    out = []
    for t in range(3):
        X = design(d, rows, t, feat)
        sc = g2.fit_scaler(X)
        m, _ = g2.fit_logistic(g2.apply_scaler(X, sc), target[rows, t], FEAT_C[feat])
        out.append((sc, m))
    return out


def predict_e1(models: list, d: G.DevData, rows: np.ndarray, feat: str) -> np.ndarray:
    out = np.full((len(rows), 3), np.nan)
    for t, (sc, m) in enumerate(models):
        if len(rows):
            out[:, t] = m.predict(g2.apply_scaler(design(d, rows, t, feat), sc))
    return out


def crossfit_e1(d: G.DevData, feat: str, target: np.ndarray, within: np.ndarray | None = None) -> np.ndarray:
    """[N, 3] out-of-fold scores of the ok rows inside `within` (default all): one model per lh-fold-v1 label present there,
    fitted on the ok rows of `within` with the other labels. Rows outside `within` stay NaN and are never used."""
    within = np.ones(d.N, dtype=bool) if within is None else within
    out = np.full((d.N, 3), np.nan)
    for k in np.unique(d.fold[within]):
        tr = np.flatnonzero(within & d.ok & (d.fold != k))
        te = np.flatnonzero(within & d.ok & (d.fold == k))
        if len(te):
            out[te] = predict_e1(fit_e1(d, tr, feat, target), d, te, feat)
    return out


def _stat_sha(S: np.ndarray, mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.nan_to_num(S[mask], nan=-9e9)).tobytes()).hexdigest()


def _fitted_full(d, feat, target, sel_mask, held_mask):
    """Nested G3a rev-2 pattern. Selection rows: INNER cross-fitted out-of-fold scores from selection rows only. Held rows:
    scores from models fitted on all ok selection rows. Returns (S, fingerprints, models)."""
    if (sel_mask & held_mask).any():
        raise ValueError("selection and held-out rows overlap")
    S = crossfit_e1(d, feat, target, within=sel_mask)
    models = fit_e1(d, np.flatnonzero(sel_mask & d.ok), feat, target)
    hr = np.flatnonzero(held_mask & d.ok)
    S[hr] = predict_e1(models, d, hr, feat)
    return S, {"outer_model_sha256": G.model_fingerprint(models), "selection_stat_sha256": _stat_sha(S, sel_mask)}, models


def fitted_stat(d: G.DevData, feat: str, target: np.ndarray, sel_mask: np.ndarray, held_mask: np.ndarray):
    """(S [N,3], fingerprint dict). Rows outside sel_mask are never used for fitting; failed rows and rows in neither mask
    are NaN. The fingerprints (outer models, selection-row scores) serve the isolation tests."""
    S, fp, _ = _fitted_full(d, feat, target, sel_mask, held_mask)
    return S, fp


# ---------------------------------------------------------------------------------------------------- transforms (A3)
def cummax(S: np.ndarray) -> np.ndarray:
    """s'_t = max over u <= t of s_u, ignoring NaN (all NaN so far -> NaN). Monotone in t; reads loops <= t only."""
    return np.fmax.accumulate(S, axis=1)


def fit_rank_maps(S: np.ndarray, rows: np.ndarray) -> list:
    """Per loop, the sorted finite scores of `rows` at that loop (an empirical CDF). Fit on SELECTION rows only."""
    return [np.sort(S[rows, t][np.isfinite(S[rows, t])]) for t in range(S.shape[1])]


def apply_rank(S: np.ndarray, maps: list) -> np.ndarray:
    """s'_t = empirical CDF of loop t's map evaluated at S[:, t] (in [0, 1]); NaN stays NaN. Uses loop t only."""
    out = np.full(S.shape, np.nan)
    for t, mp in enumerate(maps):
        fin = np.isfinite(S[:, t])
        if len(mp) and fin.any():
            out[fin, t] = np.searchsorted(mp, S[fin, t], side="right") / len(mp)
    return out


def maps_sha(maps: list) -> str:
    h = hashlib.sha256()
    for m in maps:
        h.update(np.ascontiguousarray(m, dtype=np.float64).tobytes())
        h.update(b"|")
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------- methods for one fold
def build_methods(d: G.DevData, sel_mask: np.ndarray, held_mask: np.ndarray, base: dict):
    """All E1 statistics for one outer fold: (methods {name: S [N,3]}, fingerprints {name: dict}). `base` holds the
    row-wise baseline statistics (B_pb, native-pb, readouts), which have no fitted component."""
    tg = targets(d)
    methods = {n: base[n] for n in BASELINES}
    fps, cache = {}, {}
    for name, (feat, tname, tr) in PROBES.items():
        if (feat, tname) not in cache:
            cache[(feat, tname)] = _fitted_full(d, feat, tg[tname], sel_mask, held_mask)[:2]
        S, fp = cache[(feat, tname)]
        fp = dict(fp)
        if tr == "cummax":
            S = cummax(S)
        elif tr == "rank":                       # maps from SELECTION rows' inner-OOF scores, applied unchanged to held rows
            maps = fit_rank_maps(S, np.flatnonzero(sel_mask & d.ok))
            S, fp["rank_maps_sha256"] = apply_rank(S, maps), maps_sha(maps)
        if tr:
            fp["selection_stat_sha256"] = _stat_sha(S, sel_mask)
        methods[name], fps[name] = S, fp
    combo = np.fmax(base["native-pb"], methods["pi1-x"])
    methods["combo-max"] = combo
    fps["combo-max"] = {**fps["pi1-x"], "selection_stat_sha256": _stat_sha(combo, sel_mask)}
    return methods, fps


# -------------------------------------------------------------------------------------- risk-controlled selection (A6)
def make_boot(d: G.DevData, sel_mask: np.ndarray, fold_index: int) -> dict:
    """Bootstrap resampling counts, drawn ONCE per (outer fold, benchmark) and reused by every method and candidate:
    {benchmark: int32 [N_BOOT, n_b]}, rng = default_rng([0, fold_index, index of the benchmark in the sorted list]).
    Depends only on the number of selection questions of the benchmark."""
    out = {}
    for bi, b in enumerate(sorted(d.benchmarks)):
        n = int(((d.bench == b) & sel_mask).sum())
        if n == 0:
            raise ValueError(f"{b}: no selection rows")
        rng = np.random.default_rng([0, fold_index, bi])
        out[b] = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.int32)
    return out


def boot_fingerprint(boot: dict) -> str:
    h = hashlib.sha256()
    for b in sorted(boot):
        h.update(b.encode())
        h.update(np.ascontiguousarray(boot[b]).tobytes())
    return h.hexdigest()


def _rows(d, b, mask):
    return np.flatnonzero((d.bench == b) & mask)


def select_risk(d: G.DevData, S: np.ndarray, sel_mask: np.ndarray, fixed_only: bool, boot: dict) -> dict:
    """Risk-controlled per-benchmark (tau, m): the ONE function used for every method (native-pb, B_pb, the readouts and
    every probe), with no per-method special case except fixed_only for B_pb.

    Same candidates as G.select_all (51-point grid pooled over the selection rows, or [inf]; m in 1..4). Per question,
    l_i = 100 (c_i4 - c_i,policy) (failed questions: both 0). A candidate QUALIFIES iff the one-sided 95% bootstrap upper
    bound of mean(l) is <= 1 pp (and its point loss is <= 1 pp, which the bound implies except in degenerate skewed cases;
    stated so that risk-rule saving <= point-rule saving holds by construction). Among qualifiers the frozen tie-break of
    R.select applies: max saving (1e-12) -> smaller point loss -> tau = inf -> larger tau -> smaller m.
    Efficiency: l is sparse, so bootstrap means are counts[:, nz] @ l[nz] / n_b, and candidates are tried in order of
    decreasing saving until the first qualifier's tie block is exhausted."""
    rows = np.flatnonzero(sel_mask)
    taus = [math.inf] if fixed_only else R.grid([S[rows]], [d.ok[rows]])
    out = {}
    for b in d.benchmarks:
        r = _rows(d, b, sel_mask)
        counts = boot[b]
        if counts.shape != (N_BOOT, len(r)):
            raise ValueError(f"{b}: bootstrap counts {counts.shape} do not match {len(r)} selection questions")
        ok, corr, Sb = d.ok[r], d.correct[r], S[r]
        c4 = np.where(ok, corr[:, LOOPS - 1], 0)
        cand = []
        for tau in taus:
            for m in (1, 2, 3, 4):
                sav, loss = R.outcome(corr, ok, R.exit_loops(Sb, ok, tau, m))
                cand.append({"tau": tau, "m": m, "saving": sav, "loss": loss})
        cand.sort(key=lambda c: -c["saving"])          # stable: order within ties is irrelevant (tie-break applied below)

        def upper(c):
            e = R.exit_loops(Sb, ok, c["tau"], c["m"])
            l = 100.0 * (c4 - np.where(ok, corr[np.arange(len(r)), e - 1], 0))
            nz = np.flatnonzero(l)
            if not len(nz):
                return 0.0
            return float(np.quantile(counts[:, nz] @ l[nz] / len(r), RISK_QUANTILE))

        best, qual = None, []
        for c in cand:
            if best is not None and c["saving"] < best - g1.TOL:
                break                                   # the tie block of the best qualifier is exhausted
            if c["loss"] > LIMIT:
                continue                                # cannot qualify; skip the bootstrap
            u = upper(c)
            if u <= LIMIT:
                qual.append({**c, "upper_bound": u})
                if best is None:
                    best = c["saving"]
        assert qual, f"{b}: no qualifying candidate (fixed depth 4 has l = 0 and must qualify)"
        lo = min(c["loss"] for c in qual)
        tie = [c for c in qual if c["loss"] <= lo + g1.TOL]
        tie.sort(key=lambda c: (0 if math.isinf(c["tau"]) else 1, -c["tau"] if math.isfinite(c["tau"]) else 0, c["m"]))
        s = tie[0]
        out[b] = {"tau": s["tau"], "m": int(s["m"]), "saving": float(s["saving"]), "loss": float(s["loss"]),
                  "upper_bound": float(s["upper_bound"])}
    return out


# --------------------------------------------------------------------------------------------------- one outer fold
def held_exits(d: G.DevData, S: np.ndarray, sel: dict, held: np.ndarray) -> dict:
    """Per-question exit loop and correctness of the selected policies on the held-out rows (failed: exit 4, wrong)."""
    idx, e, c = [], [], []
    for b in sel:
        r = _rows(d, b, held)
        eb = R.exit_loops(S[r], d.ok[r], sel[b]["tau"], sel[b]["m"])
        idx.append(r)
        e.append(eb)
        c.append(np.where(d.ok[r], d.correct[r, eb - 1], 0))
    return {"idx": np.concatenate(idx), "e": np.concatenate(e), "c": np.concatenate(c)}


def _rule_record(d, ev, sel):
    viol = [ev[b]["loss"] > LIMIT for b in PRIMARY]
    rec = {"selected": {b: {k: (s[k] if k == "tau" else float(s[k]) if k != "m" else int(s[k]))
                            for k in ("tau", "m", "saving", "loss", "upper_bound") if k in s} for b, s in sel.items()},
           "heldout": ev,
           "selection_saving": G.macro({b: {"saving": sel[b]["saving"]} for b in PRIMARY}, PRIMARY),
           "heldout_saving": G.macro(ev, PRIMARY), "heldout_saving_six": G.macro(ev, d.benchmarks),
           "heldout_loss_primary": {b: ev[b]["loss"] for b in PRIMARY},
           "n_violations_primary": int(sum(viol)), "worst_loss_primary": float(max(ev[b]["loss"] for b in PRIMARY)),
           "worst_loss_six": float(max(ev[b]["loss"] for b in d.benchmarks))}
    return rec


def outer_fold_e1(d: G.DevData, k: int, base: dict):
    """One outer fold, fully isolated from fold k for everything chosen on the selection rows. Every method gets BOTH
    selections (point rule = frozen G.select_all; risk rule = select_risk) and is evaluated on the same held-out
    questions. Returns (record per method, held-out exits per method for the paired analyses)."""
    sel_mask, held = d.fold != k, d.fold == k
    methods, fps = build_methods(d, sel_mask, held, base)
    boot = make_boot(d, sel_mask, k)
    rec, pairs, ids = {}, {}, {}
    for name, S in methods.items():
        fixed = name == "B_pb"
        rec[name] = {"fingerprints": fps.get(name)}
        for rule in ("point", "risk"):
            sel = G.select_all(d, S, sel_mask, fixed_only=fixed) if rule == "point" \
                else select_risk(d, S, sel_mask, fixed, boot)
            ev = G.evaluate(d, S, sel, held)
            rec[name][rule] = _rule_record(d, ev, sel)
            ids[(name, rule)] = json.dumps({b: v["ids_sha256"] for b, v in ev.items()}, sort_keys=True)
            if rule == "point":
                pairs[name] = held_exits(d, S, sel, held)
    if len(set(ids.values())) != 1:      # paired: every method/selection evaluated on exactly the same held-out QUESTIONS
        raise AssertionError(f"fold {k}: methods evaluated on different held-out questions")
    for name, p in pairs.items():
        if not np.array_equal(p["idx"], pairs["B_pb"]["idx"]):
            raise AssertionError(f"fold {k}: {name} paired exits are on different held-out questions")
    return rec, pairs


# --------------------------------------------------------------------------------------------------------- summaries
def _f(x):
    return None if x is None else float(x)


def auc(y, s):
    """AUROC over the finite scores; None if one class or nothing scorable."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    m = np.isfinite(s)
    if not m.any():
        return None
    return _f(g2._auroc(y[m], s[m]))


def ece(y, p, bins: int = 10):
    """Expected calibration error, equal-width bins on [0, 1]; None if empty."""
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    m = np.isfinite(p)
    y, p = y[m], p[m]
    if not len(y):
        return None
    b = np.minimum((np.clip(p, 0, 1) * bins).astype(int), bins - 1)
    return float(sum((b == i).sum() / len(y) * abs(y[b == i].mean() - p[b == i].mean()) for i in range(bins) if (b == i).any()))


def utility(per_fold: list, name: str, rule: str, ref: str = "native-pb") -> dict:
    """Held-out utility of one method over the outer folds (macro over PRIMARY, six beside it) and vs `ref`, same rule."""
    g = lambda n, key: [f[n][rule][key] for f in per_fold]   # noqa: E731
    sav = g(name, "heldout_saving")
    out = {"heldout_saving_primary": float(np.mean(sav)), "per_fold_primary": [float(v) for v in sav],
           "heldout_saving_six": float(np.mean(g(name, "heldout_saving_six"))),
           "selection_saving_primary": float(np.mean(g(name, "selection_saving"))),
           "n_violations_primary": int(sum(g(name, "n_violations_primary"))),
           "worst_loss_primary": float(max(g(name, "worst_loss_primary"))),
           "improvement_vs_native_pb": float(np.mean(np.array(sav) - np.array(g(ref, "heldout_saving")))),
           "improvement_vs_native_pb_per_fold": [float(v) for v in np.array(sav) - np.array(g(ref, "heldout_saving"))],
           "improvement_vs_native_pb_six": float(np.mean(np.array(g(name, "heldout_saving_six"))
                                                         - np.array(g(ref, "heldout_saving_six"))))}
    return out


def pool_pairs(pairs_per_fold: list) -> dict:
    out = {}
    for name in pairs_per_fold[0]:
        out[name] = {key: np.concatenate([p[name][key] for p in pairs_per_fold]) for key in ("idx", "e", "c")}
    ref = out["B_pb"]["idx"]
    assert all(np.array_equal(v["idx"], ref) for v in out.values()), "pooled held-out questions differ between methods"
    return out


def paired_vs_native(d: G.DevData, pooled: dict, names, boot_counts: dict) -> dict:
    """A1 paired question-level comparison against native-pb on the pooled held-out rows."""
    nat = pooled["native-pb"]
    bench = d.bench[nat["idx"]]
    out = {}
    for name in names:
        m = pooled[name]
        rec = {}
        for label, sel in (("primary", np.isin(bench, PRIMARY)), ("six", np.ones(len(bench), bool))):
            diff = m["e"][sel] - nat["e"][sel]
            rec[label] = {"exit_loop_difference_hist": {str(v): int((diff == v).sum()) for v in range(-3, 4)},
                          "correctness_2x2": {f"native{a}_method{c}": int(((nat["c"][sel] == a) & (m["c"][sel] == c)).sum())
                                              for a in (0, 1) for c in (0, 1)},
                          "n": int(sel.sum())}
        # paired bootstrap of the primary-macro saving difference (method - native), stratified by benchmark
        u_by_b = {b: (nat["e"][bench == b] - m["e"][bench == b]) / float(LOOPS) for b in PRIMARY}
        point = float(np.mean([u.mean() for u in u_by_b.values()]))
        boots = np.mean([boot_counts[b] @ u_by_b[b] / len(u_by_b[b]) for b in PRIMARY], axis=0)
        rec["saving_difference_primary"] = {"point": point, "ci95": [float(np.quantile(boots, 0.025)),
                                                                        float(np.quantile(boots, 0.975))],
                                            "n_boot": N_BOOT}
        out[name] = rec
    return out


def paired_bootstrap_counts(d: G.DevData, pooled: dict) -> dict:
    """Resampling counts per primary benchmark, drawn once (rng default_rng(0), PRIMARY order) and shared by all methods."""
    bench = d.bench[pooled["native-pb"]["idx"]]
    rng, out = np.random.default_rng(0), {}
    for b in PRIMARY:
        n = int((bench == b).sum())
        out[b] = rng.multinomial(n, np.full(n, 1.0 / n), size=N_BOOT).astype(np.int32)
    return out


def a5_disagreement(d: G.DevData, pooled: dict, per_fold_pairs: list) -> dict:
    nat, px = pooled["native-pb"], pooled["pi1-x"]
    bench = d.bench[nat["idx"]]

    def block(sel):
        if not sel.any():
            return None
        en, ex, cn, cx = nat["e"][sel], px["e"][sel], nat["c"][sel], px["c"][sel]
        tab = np.zeros((LOOPS, LOOPS), dtype=int)
        np.add.at(tab, (en - 1, ex - 1), 1)
        dis = en != ex
        rec = {"n": int(sel.sum()), "crosstab_native_rows_pi1x_cols": tab.tolist(),
               "pi1x_earlier": int((ex < en).sum()), "same": int((ex == en).sum()), "pi1x_later": int((ex > en).sum())}
        for label, m in (("disagreeing", dis), ("pi1x_earlier", ex < en), ("pi1x_later", ex > en)):
            rec[label] = None if not m.any() else {
                "n": int(m.sum()), "mean_loops_saved_by_pi1x_vs_native": float((en[m] - ex[m]).mean()),
                "accuracy_diff_pp_pi1x_minus_native": float(100.0 * (cx[m].mean() - cn[m].mean()))}
        return rec

    out = {"primary_pooled": block(np.isin(bench, PRIMARY)), "six_pooled": block(np.ones(len(bench), bool)),
           "per_benchmark": {b: block(bench == b) for b in d.benchmarks}}
    out["per_fold_primary"] = []
    for p in per_fold_pairs:
        bf = d.bench[p["native-pb"]["idx"]]
        s = np.isin(bf, PRIMARY)
        en, ex = p["native-pb"]["e"][s], p["pi1-x"]["e"][s]
        out["per_fold_primary"].append({"pi1x_earlier": int((ex < en).sum()), "same": int((ex == en).sum()),
                                        "pi1x_later": int((ex > en).sum())})
    return out


def oof_scores(d: G.DevData) -> dict:
    """Plain 5-fold out-of-fold scores over ALL development rows (A2a, A4). Descriptive; nothing is selected on them."""
    tg = targets(d)
    return {("x", "S"): crossfit_e1(d, "x", tg["S"]), ("x", "c_t"): crossfit_e1(d, "x", tg["c_t"]),
            ("x", "safe"): crossfit_e1(d, "x", tg["safe"]), ("hx", "S"): crossfit_e1(d, "hx", tg["S"])}


def _cell_auroc(d, score, label):
    out = {}
    for b in d.benchmarks:
        r = np.flatnonzero((d.bench == b) & d.ok)
        out[b] = {f"loop{t + 1}": auc(label[r, t], score[r, t]) for t in range(3)}
    return out


def _mean_cells(cells, benches=PRIMARY):
    v = [x for b in benches for x in cells[b].values() if x is not None]
    return float(np.mean(v)) if v else None


def a2(d, oof, per_fold):
    tg = targets(d)
    pred_q = {}
    for tname in ("S", "c_t", "safe"):
        cells = _cell_auroc(d, oof[("x", tname)], tg[tname])          # each model's score vs ITS OWN target
        pred_q[tname] = {"auroc_by_benchmark_loop": cells, "mean_auroc_primary": _mean_cells(cells),
                         "mean_auroc_six": _mean_cells(cells, d.benchmarks)}
    names = {"S": "pi1-x", "c_t": "pi1-x@c_t", "safe": "pi1-x@safe"}
    return {"prediction_quality": pred_q,
            "stopping_utility": {t: {rule: utility(per_fold, n, rule) for rule in ("point",)} for t, n in names.items()}}


def a4(d, oof):
    tg = targets(d)
    scores = {"native-cdf": G.baseline_stat(d, "native-pb"), "pi1-x": oof[("x", "S")], "pi1-hx": oof[("hx", "S")]}
    out = {"auroc": {}, "mean_auroc_primary": {}, "ece_S": {}}
    for sn, sc in scores.items():
        out["auroc"][sn], out["mean_auroc_primary"][sn] = {}, {}
        for ln, lab in tg.items():
            cells = _cell_auroc(d, sc, lab)
            out["auroc"][sn][ln] = cells
            out["mean_auroc_primary"][sn][ln] = _mean_cells(cells)
    for sn in ("pi1-x", "pi1-hx"):
        sc, cells = scores[sn], {}
        for b in d.benchmarks:
            r = np.flatnonzero((d.bench == b) & d.ok)
            cells[b] = {f"loop{t + 1}": ece(tg["S"][r, t], sc[r, t]) for t in range(3)}
        r = np.flatnonzero(d.ok)
        cells["all_benchmarks"] = {f"loop{t + 1}": ece(tg["S"][r, t], sc[r, t]) for t in range(3)}
        out["ece_S"][sn] = cells
    return out


# --------------------------------------------------------------------------------------- A7: exposed validation
def validation_pack(dev: G.DevData) -> dict:
    """Everything chosen for the exposed check, from DEVELOPMENT data only (this function has no validation argument):
    statistics by plain 5-fold cross-fitting over all development data, rank maps fitted on those scores, (tau_b, m_b) by
    the frozen point rule, and the models that will score validation. Descriptive only."""
    allm, none = np.ones(dev.N, bool), np.zeros(dev.N, bool)
    tg, cache, S_dev, fitted = targets(dev), {}, {"native-pb": G.baseline_stat(dev, "native-pb")}, {}
    for name in A7_METHODS[1:]:
        feat, tname, tr = PROBES[name]
        if (feat, tname) not in cache:
            S, _, models = _fitted_full(dev, feat, tg[tname], allm, none)
            cache[(feat, tname)] = (S, models)
        S, models = cache[(feat, tname)]
        maps = None
        if tr == "cummax":
            S = cummax(S)
        elif tr == "rank":
            maps = fit_rank_maps(S, np.flatnonzero(dev.ok))
            S = apply_rank(S, maps)
        S_dev[name], fitted[name] = S, (feat, tname, tr, models, maps)
    sels = {n: G.select_all(dev, S, allm) for n, S in S_dev.items()}
    return {"sel": sels, "fitted": fitted,
            "selection_saving": {n: G.macro({b: {"saving": s[b]["saving"]} for b in PRIMARY}, PRIMARY) for n, s in sels.items()}}


def validation_apply(pack: dict, val: G.DevData) -> dict:
    allv, cache, res = np.ones(val.N, bool), {}, {}
    for name in A7_METHODS:
        if name == "native-pb":
            S = G.baseline_stat(val, "native-pb")
        else:
            feat, tname, tr, models, maps = pack["fitted"][name]
            if (feat, tname) not in cache:
                S0 = np.full((val.N, 3), np.nan)
                rows = np.flatnonzero(val.ok)
                S0[rows] = predict_e1(models, val, rows, feat)
                cache[(feat, tname)] = S0
            S = cache[(feat, tname)]
            S = cummax(S) if tr == "cummax" else apply_rank(S, maps) if tr == "rank" else S
        sel = {b: s for b, s in pack["sel"][name].items() if b in val.benchmarks}
        ev = G.evaluate(val, S, sel, allv)
        res[name] = {"selection_saving_dev": pack["selection_saving"][name], "validation_saving_primary": G.macro(ev, PRIMARY),
                     "validation_saving_six": G.macro(ev, val.benchmarks), "validation": ev,
                     "selected": {b: {"tau": s["tau"], "m": int(s["m"])} for b, s in sel.items()}}
    for name, r in res.items():
        r["improvement_vs_native_pb"] = r["validation_saving_primary"] - res["native-pb"]["validation_saving_primary"]
        r["improvement_vs_native_pb_six"] = r["validation_saving_six"] - res["native-pb"]["validation_saving_six"]
    return {"descriptive_only": True, "methods": res}


# ---------------------------------------------------------------------------------------------------------- the run
def run_e1(dev: G.DevData, val: G.DevData | None = None) -> dict:
    """All of E1 on in-memory data; returns the JSON-serialisable report dict (no file is written here)."""
    if (dev.source == "p1_validation").any() or (dev.source == "test").any():
        raise PermissionError("development data must not contain validation or sealed rows")
    if val is not None and not (val.source == "p1_validation").all():
        raise PermissionError("the exposed set must consist of Phase 1 validation rows only")
    missing = [b for b in PRIMARY if b not in dev.benchmarks]
    if missing:
        raise ValueError(f"primary benchmarks missing from the development data: {missing}")
    base = {n: G.baseline_stat(dev, n) for n in BASELINES}
    folds = [outer_fold_e1(dev, k, base) for k in range(g2.N_FOLDS)]
    per_fold, pairs_pf = [f[0] for f in folds], [f[1] for f in folds]
    pooled = pool_pairs(pairs_pf)
    boot_counts = paired_bootstrap_counts(dev, pooled)
    oof = oof_scores(dev)

    rep = {"plan": PLAN, "status": STATUS,
           "n": {"dev": {b: int((dev.bench == b).sum()) for b in dev.benchmarks},
                 "val": None if val is None else {b: int((val.bench == b).sum()) for b in val.benchmarks},
                 "dev_by_source": {s: int((dev.source == s).sum()) for s in np.unique(dev.source)},
                 "dev_failed": int((~dev.ok).sum()), "risk_bootstrap_resamples": N_BOOT},
           "saving": "gross (rho = 0); primary macro over " + ",".join(PRIMARY) + ", six-benchmark macro beside it"}
    rep["A1"] = {"ladder": {n: utility(per_fold, n, "point") for n in LADDER},
                 "paired_vs_native_pb": paired_vs_native(dev, pooled, LADDER[1:], boot_counts),
                 "fold_selections_point": [{n: f[n]["point"]["selected"] for n in LADDER} for f in per_fold]}
    rep["A2"] = a2(dev, oof, per_fold)
    rep["A3"] = {n: utility(per_fold, n, "point") for n in A3_METHODS}
    rep["A4"] = a4(dev, oof)
    rep["A5"] = {"disagreement_native_vs_pi1x": a5_disagreement(dev, pooled, pairs_pf),
                 "combo_max": utility(per_fold, "combo-max", "point"),
                 "native_pb": utility(per_fold, "native-pb", "point"), "pi1_x": utility(per_fold, "pi1-x", "point")}
    rep["A6"] = {rule: {n: {**utility(per_fold, n, rule),
                            "heldout_loss_primary_per_fold": [f[n][rule]["heldout_loss_primary"] for f in per_fold],
                            "selection_time_upper_bounds_primary": (
                                [{b: f[n][rule]["selected"][b].get("upper_bound") for b in PRIMARY} for f in per_fold]
                                if rule == "risk" else None)}
                        for n in ALL_METHODS} for rule in ("point", "risk")}
    rep["A6"]["note"] = ("improvement_vs_native_pb is computed within each rule: the risk-rule entries compare against "
                         "risk-rule native-pb")
    rep["A7"] = None if val is None else validation_apply(validation_pack(dev), val)
    rep["fold_fingerprints"] = [{n: f[n]["fingerprints"] for n in f if f[n]["fingerprints"]} for f in per_fold]
    return json.loads(json.dumps(rep, default=str))
