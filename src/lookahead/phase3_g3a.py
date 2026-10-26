"""Phase 3 G3a feasibility diagnostics D1-D4 (plan v1.1 section 7, amended by A1.7 and A2.4). Development data only.

Data (A1.7): Phase 1 TRAIN (six benchmarks, for fitting) + DEV+ (three primary benchmarks), folds by `lh-fold-v1`.
Phase 1 VALIDATION is loaded only for the exposed, reported-only D2 estimate. TEST is never read: every opened path goes
through phase3_build.assert_not_sealed, DEV+ doc ids must be in the frozen DEV+ allow-list and disjoint from the frozen
TEST id list, and the extraction directory for TEST is never named here.

Everything outcome-related is a pure function of an in-memory `DevData`, so it is tested on synthetic data.
Selection, exit rule, grid and the D2 comparator come from the FROZEN module `phase3_readouts` (A2), unchanged.
Savings at G3a are gross (controller overhead rho = 0): rho_b is measured only at G3b (plan 5a); see OPEN_POINTS.
"""
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from lookahead import gate1 as g1
from lookahead import gate2 as g2
from lookahead import phase3_build as P
from lookahead import phase3_readouts as R

PRIMARY = ("openbookqa", "commonsense_qa", "hellaswag")            # A1.2: macro quantities and stop rules
LOOPS = 4
D1_MIN = 0.05            # plan 7 / A1.7: stop if ceiling(ii) - B_pb < 0.05 (primary macro)
D2_MIN = 0.02            # plan 7 / A2.4: stop if nested improvement over the A2 comparator < 0.02 (primary macro)
D3_MIN_TRANSITIONS = 30  # plan 7 / A1.7: cells with >= 30 transitions
FROZEN_C = {"x": 1.0, "hx": 0.001}   # Gate 2 frozen family-1 C (Q = x, P = hx), identical for loops 1-3; checked at run time
BASELINES = ("B_pb", "native-pb") + R.READOUTS                      # A2.4 comparator pool
CANDIDATES = ("pi1-x", "pi1-hx")                                    # plan 7 D2 (frozen)
DEVPLUS_DIR = "extracted_phase3_dev"
P1_DIR = "extracted"

# Points the frozen documents leave open; resolved here as stated and listed in the report for reviewer confirmation
# BEFORE any real run. Changing any of them after a real run is not allowed.
OPEN_POINTS = {
    "OP1_net_saving_at_G3a": "A2.4 says 'macro net saving', but rho_b is measured only at G3b (plan 5a). G3a uses gross "
                             "saving (rho = 0) for every policy and baseline.",
    "OP2_D2_candidate_rule": "Plan 7 names two screened candidates (pi1-x, pi1-hx) but one stop rule. Implemented: STOP iff "
                             "BOTH candidates' nested improvement over the A2 comparator is < 0.02 (continue if either "
                             "reaches 0.02). Both values and the improvement over B_pb alone are reported.",
    "OP3_grid_pool": "The 51-point grid (plan 5, A2.3) pools the statistic over the ok questions of ALL benchmarks in the "
                     "selection rows (six benchmarks; Phase 1 train + DEV+), loops where defined; in D2 only the selection "
                     "folds' rows are pooled.",
    "OP4_D1_nonbinding_baseline": "A2.4's non-binding Delta_oracle-readout uses the best baseline chosen on ALL "
                                  "development data (no held-out fold exists for D1).",
    "OP5_D3_model": "D3 classifies WC (1) vs CW (0) directly on the transitions of each loop (rows with label WC or CW), "
                    "per-loop L2-logistic on h_t or on x_t, C chosen by inner CV over the Gate 2 C grid on the training "
                    "folds, out of fold over the 5 lh-fold-v1 folds, fitted on all six benchmarks; AUROC within "
                    "(benchmark, loop).",
    "OP6_D3_cells": "D3 cells are (kind, benchmark, loop) with kind in {adjacent t->t+1, to-final t->4}, over the three "
                    "primary benchmarks; a cell counts if it has >= 30 WC+CW transitions and both classes; hx is kept iff "
                    "h > x in at least half of the counted cells; with no counted cell the result is 'not kept'.",
    "OP7_D4_model": "D4 uses per-loop L2-logistic with the frozen Gate 2 C (x: 1.0, hx: 0.001), out of fold, for c_t and "
                    "c_4; Delta_t is scored by P(c_4) - P(c_t) (Spearman). Informative only.",
    "OP8_oracle_failed": "Failed questions (none expected) exit at loop 4 and are wrong in both oracles, as under every "
                         "policy (Gate 1 rule).",
}


# ------------------------------------------------------------------------------------------------------------ data
@dataclass
class DevData:
    bench: np.ndarray        # [N] str
    doc_id: np.ndarray       # [N] str
    source: np.ndarray       # [N] 'p1_train' | 'devplus' | 'p1_validation'
    ok: np.ndarray           # [N] bool
    correct: np.ndarray      # [N, 4] int (0 for failed)
    pred: np.ndarray         # [N, 4] int
    cdf: np.ndarray          # [N, 4] float (-inf for failed)
    T: np.ndarray            # [N, 3] str transition t -> t+1
    Tfin: np.ndarray         # [N, 3] str transition t -> 4
    x: np.ndarray            # [N, 3, 9] float32 Gate 2 cheap features
    h: np.ndarray            # [N, 3, d] float32 hidden state at the last context token, loops 1-3
    z: list                  # N arrays [4, K]: option scores whose softmax is p (opt_lln for acc_norm, opt_ll for acc)
    fold: np.ndarray = field(default=None)
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.fold is None:
            self.fold = np.array([g2.fold_of(d) for d in self.doc_id])
        n = len(self.doc_id)
        for name in ("bench", "source", "ok", "correct", "pred", "cdf", "T", "Tfin", "x", "h", "fold"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"DevData.{name} has {len(getattr(self, name))} rows, expected {n}")
        if len(self.z) != n:
            raise ValueError("DevData.z length mismatch")
        if len(set(self.doc_id.tolist())) != n:
            dup = [d for d in set(self.doc_id.tolist()) if (self.doc_id == d).sum() > 1][:3]
            raise ValueError(f"duplicate questions: {dup}")
        if not np.isin(self.correct, (0, 1)).all():
            raise ValueError("correct must be 0/1")

    @property
    def N(self):
        return len(self.doc_id)

    @property
    def benchmarks(self):
        return list(dict.fromkeys(self.bench.tolist()))

    def stability(self) -> np.ndarray:
        """Gate 2 target S_t (loops 1-3): the loop-t prediction equals every later prediction."""
        return np.stack([(self.pred[:, [t]] == self.pred[:, t + 1:]).all(axis=1) for t in range(3)], axis=1).astype(float)


def concat(parts: list) -> DevData:
    def cat(name):
        return np.concatenate([getattr(p, name) for p in parts])

    prov = {}
    for p in parts:
        prov.update(p.provenance)
    return DevData(bench=cat("bench"), doc_id=cat("doc_id"), source=cat("source"), ok=cat("ok"), correct=cat("correct"),
                   pred=cat("pred"), cdf=cat("cdf"), T=cat("T"), Tfin=cat("Tfin"), x=cat("x"), h=cat("h"),
                   z=[z for p in parts for z in p.z], fold=cat("fold"), provenance=prov)


def subset(d: DevData, mask: np.ndarray) -> DevData:
    idx = np.flatnonzero(mask)
    return DevData(bench=d.bench[idx], doc_id=d.doc_id[idx], source=d.source[idx], ok=d.ok[idx], correct=d.correct[idx],
                   pred=d.pred[idx], cdf=d.cdf[idx], T=d.T[idx], Tfin=d.Tfin[idx], x=d.x[idx], h=d.h[idx],
                   z=[d.z[i] for i in idx], fold=d.fold[idx], provenance=d.provenance)


# ------------------------------------------------------------------------------------------------------- loading
def read_id_list(path: Path) -> set:
    """doc ids of a frozen phase3 id list (benchmark,stratum,doc_id,split)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines[0] != "benchmark,stratum,doc_id,split":
        raise ValueError(f"{path}: unexpected header")
    return {ln.split(",")[2] for ln in lines[1:] if ln}


def _guard(p) -> None:
    P.assert_not_sealed(p)
    g2._assert_allowed_path(p)


def load_source(root, dirname: str, split: str, source: str, benchmarks, allow_ids: set | None = None,
                deny_ids: set = frozenset(), expected_n: dict | None = None) -> DevData:
    """Read one extraction directory (checksummed parts, hidden states aligned as in Gate 2) for the rows of `split`."""
    import pyarrow.parquet as pq

    root = Path(root)
    base = root / "data" / dirname
    _guard(base)
    parts, prov = [], {}
    for bench in benchmarks:
        bdir = base / bench
        _guard(bdir)
        done_paths = sorted(bdir.glob("part-*.done.json"))
        if not done_paths:
            raise FileNotFoundError(f"no completed parts for {bench} under {bdir}")
        frames, hiddens, zs = [], [], []
        for dp in done_paths:
            _guard(dp)
            df, h, _ = g2._read_part_with_hidden(root, bench, dp, split, prov)
            if not len(df):
                continue
            pq_path = Path(str(dp)[: -len(".done.json")] + ".parquet")
            oc = pq.read_table(pq_path, columns=["doc_id", "loop", "split", "opt_ll", "opt_lln", "metric"],
                               filters=[("split", "==", split)]).to_pandas()
            oc = oc[oc["split"] == split].reset_index(drop=True)
            if len(oc) != len(df) or not (oc["doc_id"].to_numpy() == df["doc_id"].to_numpy()).all() \
                    or not (oc["loop"].to_numpy() == df["loop"].to_numpy()).all():
                raise ValueError(f"{pq_path}: option-score rows do not align with the record rows")
            frames.append(df.assign(_opt_ll=oc["opt_ll"].to_list(), _opt_lln=oc["opt_lln"].to_list(), _metric=oc["metric"]))
            hiddens.append(h)
        if not frames:
            raise ValueError(f"{bench}: no {split} rows under {bdir}")
        import pandas as pd

        df = pd.concat(frames, ignore_index=True)
        hid = np.concatenate(hiddens)
        qids = df["doc_id"].to_numpy().reshape(-1, LOOPS)[:, 0]
        if len(set(qids)) != len(qids):
            raise ValueError(f"{bench}: a doc_id appears twice")
        if allow_ids is not None and not set(qids) <= allow_ids:
            raise PermissionError(f"{bench}: {len(set(qids) - allow_ids)} questions are not in the DEV+ allow-list")
        if set(qids) & set(deny_ids):
            raise PermissionError(f"{bench}: TEST question ids present; refusing")
        if expected_n is not None and len(qids) != expected_n[bench]:
            raise ValueError(f"{bench}: {len(qids)} questions, manifest says {expected_n[bench]}")
        order = np.argsort(qids, kind="stable")
        df = df.iloc[(order[:, None] * LOOPS + np.arange(LOOPS)).ravel()].reset_index(drop=True)
        hid = hid[order]
        a = g1.build_arrays(df, split)
        n = a["n"]

        def col(name):
            return df[name].to_numpy().astype(float).reshape(n, LOOPS)

        vent = np.nan_to_num(col("vent")[:, :3], nan=0.0)
        n_opt = np.nan_to_num(df["n_options"].to_numpy().astype(float).reshape(n, LOOPS)[:, 0], nan=0.0)
        n_tok = np.nan_to_num(df["n_tokens"].to_numpy().astype(float).reshape(n, LOOPS)[:, 0], nan=1.0)
        ok = a["ok"]
        h32 = hid[:, :3].astype(np.float32)
        h32[~ok] = 0.0
        ll, lln, met = df["_opt_ll"].to_list(), df["_opt_lln"].to_list(), df["_metric"].to_numpy()
        z = []
        for i in range(n):
            if not ok[i]:
                z.append(np.full((LOOPS, 2), np.nan))
                continue
            src = lln if met[i * LOOPS] == "acc_norm" else ll
            z.append(np.array([list(src[i * LOOPS + t]) for t in range(LOOPS)], dtype=np.float64))
        parts.append(DevData(bench=np.array([bench] * n), doc_id=a["doc_id"].astype(str), source=np.array([source] * n),
                             ok=ok, correct=a["correct"], pred=a["pred"], cdf=a["cdf"], T=a["T"], Tfin=a["Tfin"],
                             x=g2.cheap_features(a, vent, n_opt, n_tok), h=h32, z=z))
    d = concat(parts)
    d.provenance = prov
    return d


def load_g3a(root, p1_benchmarks, with_validation: bool = True):
    """(dev, validation | None). dev = Phase 1 train (all `p1_benchmarks`) + DEV+ (primary three). TEST is refused."""
    root = Path(root)
    man = json.loads((root / P.MANIFEST).read_text(encoding="utf-8"))
    if not P._sha_ok(root / P.IDS["dev"], man["ids_sha256"]["dev"]) or not P._sha_ok(root / P.IDS["test"], man["ids_sha256"]["test"]):
        raise SystemExit("phase3 id lists do not match the manifest")
    allow, deny = read_id_list(root / P.IDS["dev"]), read_id_list(root / P.IDS["test"])
    if allow & deny:
        raise SystemExit("DEV+ and TEST id lists overlap")
    expected = {b: man["sets"][f"dev/{b}"]["n_rows"] for b in PRIMARY}
    p1 = load_source(root, P1_DIR, "train", "p1_train", p1_benchmarks, deny_ids=deny)
    dv = load_source(root, DEVPLUS_DIR, "dev", "devplus", PRIMARY, allow_ids=allow, deny_ids=deny, expected_n=expected)
    dev = concat([p1, dv])
    val = load_source(root, P1_DIR, "validation", "p1_validation", p1_benchmarks, deny_ids=deny) if with_validation else None
    return dev, val


# ---------------------------------------------------------------------------------------------------- statistics
def baseline_stat(d: DevData, name: str) -> np.ndarray:
    """[N, 3] statistic, halt when s_t >= tau. B_pb has no statistic (it only uses tau = +inf)."""
    if name == "B_pb":
        return np.zeros((d.N, 3))
    if name == "native-pb":
        return np.where(d.ok[:, None], d.cdf[:, :3], np.nan)
    hidden = name in ("r-hl2", "r-hcos")
    out = np.full((d.N, 3), np.nan)
    for i in np.flatnonzero(d.ok):
        out[i] = R.question_stats(name, d.z[i], d.h[i] if hidden else None)
    return out


def _design(d: DevData, rows: np.ndarray, t: int, variant: str) -> np.ndarray:
    if variant == "x":
        return d.x[rows, t]
    if variant == "h":
        return d.h[rows, t]
    if variant == "hx":
        return np.concatenate([d.h[rows, t], d.x[rows, t]], axis=1)
    raise ValueError(variant)


def _fit_predict(Xtr, ytr, Xte, C):
    sc = g2.fit_scaler(Xtr)
    m, _ = g2.fit_logistic(g2.apply_scaler(Xtr, sc), ytr, C)
    return m.predict(g2.apply_scaler(Xte, sc))


def crossfit(d: DevData, variant: str, target: np.ndarray, C: float, fit_mask: np.ndarray | None = None) -> np.ndarray:
    """[N, 3] out-of-fold P(target_t = 1): for fold k, per loop, fitted on the ok rows (within fit_mask) of the other folds."""
    fit_mask = d.ok if fit_mask is None else (fit_mask & d.ok)
    out = np.full((d.N, 3), np.nan)
    for k in range(g2.N_FOLDS):
        tr, te = np.flatnonzero(fit_mask & (d.fold != k)), np.flatnonzero(d.ok & (d.fold == k))
        if not len(te):
            continue
        for t in range(3):
            out[te, t] = _fit_predict(_design(d, tr, t, variant), target[tr, t], _design(d, te, t, variant), C)
    return out


def fit_all_predict(d_fit: DevData, d_new: DevData, variant: str, target: np.ndarray, C: float) -> np.ndarray:
    tr, te = np.flatnonzero(d_fit.ok), np.flatnonzero(d_new.ok)
    out = np.full((d_new.N, 3), np.nan)
    for t in range(3):
        out[te, t] = _fit_predict(_design(d_fit, tr, t, variant), target[tr, t], _design(d_new, te, t, variant), C)
    return out


# ------------------------------------------------------------------------------------------- selection helpers
def _rows(d: DevData, b: str, mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero((d.bench == b) & mask)


def select_all(d: DevData, S: np.ndarray, sel_mask: np.ndarray, fixed_only: bool = False) -> dict:
    """Per-benchmark (tau, m) under the frozen plan-5 rule on the rows in sel_mask; grid pooled over all benchmarks'
    selection rows (OP3)."""
    if fixed_only:
        taus = [math.inf]
    else:
        rows = np.flatnonzero(sel_mask)
        taus = R.grid([S[rows]], [d.ok[rows]])
    out = {}
    for b in d.benchmarks:
        r = _rows(d, b, sel_mask)
        if not len(r):
            raise ValueError(f"{b}: no selection rows")
        out[b] = R.select(S[r], d.ok[r], d.correct[r], taus)
    return out


def evaluate(d: DevData, S: np.ndarray, sel: dict, mask: np.ndarray) -> dict:
    """Per-benchmark (saving, loss) of the selected (tau, m) on the rows in mask."""
    out = {}
    for b, s in sel.items():
        r = _rows(d, b, mask)
        if not len(r):
            raise ValueError(f"{b}: no evaluation rows")
        sav, loss = R.outcome(d.correct[r], d.ok[r], R.exit_loops(S[r], d.ok[r], s["tau"], s["m"]))
        out[b] = {"saving": float(sav), "loss": float(loss), "n": int(len(r))}
    return out


def macro(per: dict, benches, key="saving") -> float:
    return float(np.mean([per[b][key] for b in benches]))


# ------------------------------------------------------------------------------------------------------ stop rules
def d1_stop(headroom_ii: float) -> bool:
    """Plan 7 / A1.7: stop if ceiling (ii) - B_pb < 0.05 (primary macro). Exactly 0.05 continues."""
    return bool(headroom_ii < D1_MIN)


def d2_stop(improvements) -> bool:
    """Plan 7 / A2.4 with OP2: stop iff every frozen candidate's nested improvement over the comparator is < 0.02."""
    imps = list(improvements)
    if not imps or any(not math.isfinite(v) for v in imps):
        raise ValueError("D2 needs a finite improvement for every candidate")
    return bool(all(v < D2_MIN for v in imps))


def keep_hx(cells: list) -> bool:
    """Plan 7 D3 / A1.7 with OP6."""
    counted = [c for c in cells if c["counted"]]
    return bool(counted) and sum(c["h_better"] for c in counted) >= len(counted) / 2.0


# ------------------------------------------------------------------------------------------------------------ D1
def accuracy_budget_oracle(correct: np.ndarray, ok: np.ndarray, doc_id: np.ndarray) -> np.ndarray:
    """Exact per-benchmark solution of plan 7 D1 (ii): exit loops e_i minimising sum e_i subject to
    sum c_{i,e_i} >= sum c_{i,4} - 0.01 n. May keep a wrong early answer within the budget (by design)."""
    n = len(correct)
    corr = (correct == 1) & ok[:, None]
    e = np.ones(n, dtype=np.int64)
    e[~ok] = LOOPS                                         # OP8: failed questions run 4 loops, wrong
    k, K = int(corr[:, 0].sum()), int(corr[:, LOOPS - 1].sum())
    need = K - 0.01 * n - k
    r = max(0, math.ceil(need - g1.TOL))
    if r:
        cand = np.flatnonzero(ok & ~corr[:, 0] & corr.any(axis=1))
        eps = corr[cand].argmax(axis=1) + 1                # earliest correct loop
        order = sorted(range(len(cand)), key=lambda j: (eps[j] - 1, doc_id[cand[j]]))
        if len(order) < r:
            raise AssertionError("accuracy-budget oracle infeasible (cannot happen: loop-4-correct questions exist)")
        for j in order[:r]:
            e[cand[j]] = eps[j]
    acc = corr[np.arange(n), e - 1].sum()
    assert acc >= K - 0.01 * n - g1.TOL, "oracle violates its budget"
    return e


def stable_final_oracle(pred: np.ndarray, ok: np.ndarray) -> np.ndarray:
    eq = pred == pred[:, [LOOPS - 1]]
    suffix = np.logical_and.accumulate(eq[:, ::-1], axis=1)[:, ::-1]
    return np.where(ok, suffix.argmax(axis=1) + 1, LOOPS).astype(np.int64)


def d1(d: DevData, stats: dict) -> dict:
    allrows = np.ones(d.N, dtype=bool)
    bpb = select_all(d, stats["B_pb"], allrows, fixed_only=True)
    per = {}
    for b in d.benchmarks:
        r = _rows(d, b, allrows)
        ii = accuracy_budget_oracle(d.correct[r], d.ok[r], d.doc_id[r])
        i = stable_final_oracle(d.pred[r], d.ok[r])
        per[b] = {"oracle_ii": dict(zip(("saving", "loss"), map(float, R.outcome(d.correct[r], d.ok[r], ii)))),
                  "oracle_i": dict(zip(("saving", "loss"), map(float, R.outcome(d.correct[r], d.ok[r], i)))),
                  "B_pb": {"m": int(bpb[b]["m"]), "saving": float(bpb[b]["saving"]), "loss": float(bpb[b]["loss"])},
                  "n": int(len(r))}
    out = {"per_benchmark": per}
    for label, benches in (("primary", PRIMARY), ("six", d.benchmarks)):
        o2 = float(np.mean([per[b]["oracle_ii"]["saving"] for b in benches]))
        o1 = float(np.mean([per[b]["oracle_i"]["saving"] for b in benches]))
        bp = float(np.mean([per[b]["B_pb"]["saving"] for b in benches]))
        out[label] = {"ceiling_ii": o2, "ceiling_i": o1, "B_pb": bp, "headroom_ii": o2 - bp, "headroom_i": o1 - bp}
    # non-binding (A2.4): oracle (ii) minus the best baseline chosen on all development data (OP4)
    best = {}
    for name in BASELINES:
        sel = select_all(d, stats[name], allrows, fixed_only=(name == "B_pb"))
        best[name] = macro({b: {"saving": sel[b]["saving"]} for b in PRIMARY}, PRIMARY)
    top = sorted(best, key=lambda k: (-best[k], k))[0]
    out["nonbinding_oracle_minus_best_baseline"] = {"best_baseline": top, "best_baseline_saving": best[top],
                                                    "delta": out["primary"]["ceiling_ii"] - best[top],
                                                    "all_baselines": best}
    out["stop"] = d1_stop(out["primary"]["headroom_ii"])
    return out


# ------------------------------------------------------------------------------------------------------------ D2
def d2(d: DevData, stats: dict, cand_stats: dict) -> dict:
    """Nested: for each outer fold k, every method's (tau_b, m_b) selected on the other folds, evaluated on fold k."""
    methods = {**{b: stats[b] for b in BASELINES}, **cand_stats}
    folds, per_fold = [], []
    for k in range(g2.N_FOLDS):
        sel_mask, held = d.fold != k, d.fold == k
        rec = {}
        for name, S in methods.items():
            sel = select_all(d, S, sel_mask, fixed_only=(name == "B_pb"))
            ev = evaluate(d, S, sel, held)
            rec[name] = {"selected": {b: {"tau": s["tau"], "m": int(s["m"]), "saving": float(s["saving"]),
                                          "loss": float(s["loss"])} for b, s in sel.items()},
                         "heldout": ev,
                         "eligible": all(sel[b]["loss"] <= g1.LOSS_LIMIT + g1.TOL for b in PRIMARY),
                         "selection_saving": macro({b: {"saving": sel[b]["saving"]} for b in PRIMARY}, PRIMARY),
                         "heldout_saving": macro(ev, PRIMARY),
                         "heldout_saving_six": macro(ev, d.benchmarks)}
        per_fold.append(rec)
        folds.append({b: {kk: rec[b][kk] for kk in ("eligible", "selection_saving", "heldout_saving")} for b in BASELINES})
    comp = R.nested_comparator(folds)
    out = {"comparator_per_fold": comp["chosen"], "comparator_heldout_saving": comp["heldout_saving"], "candidates": {}}
    for c in cand_stats:
        diffs = [per_fold[k][c]["heldout_saving"] - per_fold[k][comp["chosen"][k]]["heldout_saving"] for k in range(g2.N_FOLDS)]
        vs_bpb = [per_fold[k][c]["heldout_saving"] - per_fold[k]["B_pb"]["heldout_saving"] for k in range(g2.N_FOLDS)]
        best_held = [max(per_fold[k][b]["heldout_saving"] for b in BASELINES) for k in range(g2.N_FOLDS)]
        out["candidates"][c] = {
            "improvement_vs_comparator": float(np.mean(diffs)), "per_fold": [float(x) for x in diffs],
            "improvement_vs_B_pb": float(np.mean(vs_bpb)),
            "improvement_vs_best_on_heldout_conservative": float(np.mean(
                [per_fold[k][c]["heldout_saving"] - best_held[k] for k in range(g2.N_FOLDS)])),
            "heldout_saving": float(np.mean([per_fold[k][c]["heldout_saving"] for k in range(g2.N_FOLDS)])),
            "heldout_max_benchmark_loss_per_fold": [max(per_fold[k][c]["heldout"][b]["loss"] for b in PRIMARY)
                                                    for k in range(g2.N_FOLDS)],
            "improvement_vs_comparator_six": float(np.mean(
                [per_fold[k][c]["heldout_saving_six"] - per_fold[k][comp["chosen"][k]]["heldout_saving_six"]
                 for k in range(g2.N_FOLDS)]))}
    out["stop"] = d2_stop([v["improvement_vs_comparator"] for v in out["candidates"].values()])   # OP2
    out["per_fold_detail"] = per_fold
    return out


def d2_validation_exposed(dev: DevData, val: DevData, stats_dev: dict, stats_val: dict, cand_dev: dict, cand_val: dict) -> dict:
    """Reported only (plan 7): methods selected on all development data, evaluated on the exposed Phase 1 validation."""
    allrows = np.ones(dev.N, dtype=bool)
    vrows = np.ones(val.N, dtype=bool)
    res = {}
    for name in BASELINES + tuple(cand_dev):
        Sd = cand_dev[name] if name in cand_dev else stats_dev[name]
        Sv = cand_val[name] if name in cand_val else stats_val[name]
        sel = select_all(dev, Sd, allrows, fixed_only=(name == "B_pb"))
        sel = {b: s for b, s in sel.items() if b in val.benchmarks}
        ev = evaluate(val, Sv, sel, vrows)
        res[name] = {"selection_saving": macro({b: {"saving": sel[b]["saving"]} for b in PRIMARY}, PRIMARY),
                     "validation_saving": macro(ev, PRIMARY), "validation": ev}
    comp = sorted(BASELINES, key=lambda b: (-res[b]["selection_saving"], b))[0]
    return {"comparator": comp, "methods": res,
            "improvement_vs_comparator": {c: res[c]["validation_saving"] - res[comp]["validation_saving"] for c in cand_dev}}


# ------------------------------------------------------------------------------------------------------------ D3
def _oof_binary(d: DevData, rows_t: list, y_t: list, variant: str) -> list:
    """Out-of-fold P(y=1) for the given rows of each loop; C by inner CV over the Gate 2 grid on the training folds."""
    out = []
    for t in range(3):
        rows, y = rows_t[t], y_t[t]
        p = np.full(len(rows), np.nan)
        for k in range(g2.N_FOLDS):
            tr, te = d.fold[rows] != k, d.fold[rows] == k
            if not te.any() or len(np.unique(y[tr])) < 2:
                continue
            Xtr = _design(d, rows[tr], t, variant)
            sc = g2.fit_scaler(Xtr)
            Xs = g2.apply_scaler(Xtr, sc)
            C = g2.pick_param(g2.cv_scores(Xs, y[tr], d.fold[rows[tr]], g2.C_GRID, "clf"), "smaller")
            m, _ = g2.fit_logistic(Xs, y[tr], C)
            p[te] = m.predict(g2.apply_scaler(_design(d, rows[te], t, variant), sc))
        out.append(p)
    return out


def d3(d: DevData) -> dict:
    cells, out = [], {"cells": []}
    for kind, lab in (("adjacent", d.T), ("to_final", d.Tfin)):
        rows_t = [np.flatnonzero(d.ok & np.isin(lab[:, t], ("WC", "CW"))) for t in range(3)]
        y_t = [(lab[r, t] == "WC").astype(float) for t, r in enumerate(rows_t)]
        preds = {v: _oof_binary(d, rows_t, y_t, v) for v in ("h", "x")}
        for t in range(3):
            for b in d.benchmarks:
                sel = d.bench[rows_t[t]] == b
                y = y_t[t][sel]
                n_wc, n_cw = int((y == 1).sum()), int((y == 0).sum())
                au = {v: (g2._auroc(y, preds[v][t][sel]) if np.isfinite(preds[v][t][sel]).all() else None) for v in ("h", "x")}
                cell = {"kind": kind, "benchmark": b, "loop": t + 1, "n_wc": n_wc, "n_cw": n_cw,
                        "auroc_h": au["h"], "auroc_x": au["x"],
                        "counted": b in PRIMARY and n_wc + n_cw >= D3_MIN_TRANSITIONS and n_wc > 0 and n_cw > 0
                        and au["h"] is not None and au["x"] is not None}
                cell["h_better"] = bool(cell["counted"] and au["h"] > au["x"])
                out["cells"].append(cell)
    counted = [c for c in out["cells"] if c["counted"]]
    nb = sum(c["h_better"] for c in counted)
    out.update({"n_counted": len(counted), "n_h_better": nb, "keep_hx": keep_hx(out["cells"])})
    six = [c for c in out["cells"] if c["n_wc"] + c["n_cw"] >= D3_MIN_TRANSITIONS and c["n_wc"] and c["n_cw"]
           and c["auroc_h"] is not None and c["auroc_x"] is not None]
    out["six_benchmark_info"] = {"n_counted": len(six), "n_h_better": sum(c["auroc_h"] > c["auroc_x"] for c in six)}
    return out


# ------------------------------------------------------------------------------------------------------------ D4
def d4(d: DevData) -> dict:
    c = d.correct.astype(float)
    targets = {"c_t": c[:, :3], "c_4": np.repeat(c[:, [3]], 3, axis=1)}
    pr = {(v, k): crossfit(d, v, y, FROZEN_C[v]) for v in ("x", "hx") for k, y in targets.items()}
    out = {}
    for v in ("x", "hx"):
        for b in d.benchmarks:
            for t in range(3):
                r = np.flatnonzero((d.bench == b) & d.ok)
                delta = c[r, 3] - c[r, t]
                score = pr[(v, "c_4")][r, t] - pr[(v, "c_t")][r, t]
                out[f"{v}/{b}/loop{t + 1}"] = {
                    "auroc_c_t": g2._auroc(c[r, t], pr[(v, "c_t")][r, t]),
                    "auroc_c_4": g2._auroc(c[r, 3], pr[(v, "c_4")][r, t]),
                    "spearman_delta": g2._spearman(delta, score), "n": int(len(r))}
    return out


# ------------------------------------------------------------------------------------------------------- the run
def candidate_stats_crossfit(d: DevData) -> dict:
    S = d.stability()
    return {f"pi1-{v}": crossfit(d, v, S, FROZEN_C[v]) for v in ("x", "hx")}


def run_g3a(dev: DevData, val: DevData | None = None) -> dict:
    """All of G3a on in-memory data. Returns the report dict (no file is written here)."""
    if (dev.source == "p1_validation").any() or (dev.source == "test").any():
        raise PermissionError("development data must not contain validation or TEST rows")
    missing = [b for b in PRIMARY if b not in dev.benchmarks]
    if missing:
        raise ValueError(f"primary benchmarks missing from the development data: {missing}")
    stats = {name: baseline_stat(dev, name) for name in BASELINES}
    cands = candidate_stats_crossfit(dev)
    rep = {"open_points": OPEN_POINTS, "n": {b: int((dev.bench == b).sum()) for b in dev.benchmarks},
           "n_by_source": {s: int((dev.source == s).sum()) for s in np.unique(dev.source)},
           "failed": int((~dev.ok).sum()), "thresholds": {"D1": D1_MIN, "D2": D2_MIN, "D3_min_transitions": D3_MIN_TRANSITIONS},
           "D1": d1(dev, stats), "D2": d2(dev, stats, cands), "D3": d3(dev), "D4": d4(dev)}
    if val is not None:
        sv = {name: baseline_stat(val, name) for name in BASELINES}
        Sd = dev.stability()
        cv = {f"pi1-{v}": fit_all_predict(dev, val, v, Sd, FROZEN_C[v]) for v in ("x", "hx")}
        rep["D2_validation_exposed"] = d2_validation_exposed(dev, val, stats, sv, cands, cv)
    stop = [k for k in ("D1", "D2") if rep[k]["stop"]]
    rep["decision"] = ("STOP (" + ", ".join(stop) + ")") if stop else "CONTINUE (subject to reviewer GO/STOP)"
    rep["keep_hx_variants"] = rep["D3"]["keep_hx"]
    return rep


def report_digest(rep: dict) -> str:
    return hashlib.sha256(json.dumps(rep, sort_keys=True, default=str).encode()).hexdigest()
