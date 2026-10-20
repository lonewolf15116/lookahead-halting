"""Gate 1 headroom analysis: loading, policies, threshold selection, headroom, bootstrap, transition tables.

No torch here: only numpy, pandas, pyarrow and the standard library.

Test blindness: ``load_split`` filters every parquet part to the requested split immediately after reading it, before
anything is computed or concatenated, and keeps only the columns Gate 1 needs. Gate 1 never loads modarith or parity.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LOOPS = 4
GATE1_BENCHMARKS = ("arc_challenge", "arc_easy", "hellaswag", "mmlu", "commonsense_qa", "openbookqa")
SENSITIVITY_BENCHMARKS = tuple(b for b in GATE1_BENCHMARKS if b != "openbookqa")
EXCLUDED_BENCHMARKS = ("modarith", "parity")
COLUMNS = ["benchmark", "doc_id", "split", "status", "loop", "pred", "correct", "conf", "ent", "cdf", "D", "ms", "T",
           "T_final"]
SPLITS = ("train", "validation", "test")
TRANSITIONS = ("CC", "CW", "WC", "WW")

LOSS_LIMIT = 1.0          # percentage points
TOL = 1e-12
N_GRID = 50
N_BOOT = 2000
SEED = 0
S_OR_MIN = 0.25
R_MAX = 0.75

CHEAP_BASELINES = ("fixed-1", "fixed-2", "fixed-3", "confidence", "entropy", "native_gate", "deployable_kl")
THRESHOLDED = ("confidence", "entropy", "native_gate", "deployable_kl")
STAT_KEY = {"confidence": "conf", "entropy": "ent", "native_gate": "cdf", "deployable_kl": "D",
            "retrospective_kl": "D"}
# which end of the grid is the most conservative (rule fires least often)
CONSERVATIVE = {"confidence": "largest", "native_gate": "largest", "entropy": "smallest", "deployable_kl": "smallest"}

SELECTION_RULES = (
    "Grid: 50 thresholds = np.quantile(values, np.linspace(0, 1, 50)), values = the statistic pooled over all validation "
    "questions of the benchmark set (ok questions only) and loops 1..3 (deployable_kl: D over its 3 columns); exact "
    "repeated thresholds are removed keeping order. Qualified point: macro validation loss <= 1.0 percentage point "
    "(+1e-12). Selection among qualified points: maximize macro validation saving; ties (1e-12) broken by smaller macro "
    "validation loss; remaining ties broken by the most conservative threshold (confidence, native_gate: largest; "
    "entropy, deployable_kl: smallest). A baseline with no qualified point is not qualified. fixed-k has one point each. "
    "Macro = unweighted mean over benchmarks. Thresholded policies halt at the earliest loop t in {1,2,3} whose rule "
    "fires, else run to loop 4. deployable_kl pays for loop t+1 and answers with loop t+1.")


# ----------------------------------------------------------------------------------------------------------------- io
def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def code_provenance() -> dict:
    here = Path(__file__).resolve()
    repo = here.parents[2]
    files = {"src/lookahead/gate1.py": here, "scripts/gate1_validation.py": repo / "scripts" / "gate1_validation.py",
             "scripts/gate1_test.py": repo / "scripts" / "gate1_test.py"}
    return {k: (sha256_file(v) if v.exists() else None) for k, v in files.items()}


def to_jsonable(o):
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2), encoding="utf-8")


def _read_part(root: Path, bench: str, done_path: Path, split: str, provenance: dict | None) -> pd.DataFrame:
    stem = str(done_path)[: -len(".done.json")]
    pq_path = Path(stem + ".parquet")
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if "parquet_sha256" not in done:
        raise ValueError(f"{done_path}: no parquet_sha256")
    if not pq_path.exists():
        raise FileNotFoundError(f"{pq_path} is missing although {done_path.name} exists")
    sha = sha256_file(pq_path)
    if sha != done["parquet_sha256"]:
        raise ValueError(f"checksum mismatch for {pq_path}: file {sha} != done.json {done['parquet_sha256']}")
    if provenance is not None:
        provenance[pq_path.relative_to(root).as_posix()] = sha
    # Only rows of the requested split are materialized, and only the needed columns.
    table = pq.read_table(pq_path, columns=COLUMNS, filters=[("split", "==", split)])
    df = table.to_pandas()
    df = df[df["split"] == split]  # immediate filter, belt and braces
    if len(df) and not (df["benchmark"] == bench).all():
        raise ValueError(f"{pq_path}: rows of another benchmark")
    return df.reset_index(drop=True)


def build_arrays(df: pd.DataFrame, split: str) -> dict:
    """Per-question arrays (sorted by doc_id) from the rows of one benchmark and one split."""
    if len(df) == 0:
        raise ValueError("no rows")
    if (df["split"] != split).any():
        raise AssertionError("rows of another split present")
    if df.duplicated(["doc_id", "loop"]).any():
        raise ValueError("a doc_id appears twice (duplicate doc_id/loop rows)")
    counts = df.groupby("doc_id").size()
    if (counts > LOOPS).any():
        raise ValueError(f"a doc_id appears twice: {list(counts[counts > LOOPS].index[:3])}")
    if (counts != LOOPS).any():
        raise ValueError(f"questions without exactly {LOOPS} loop rows: {list(counts[counts != LOOPS].index[:3])}")
    df = df.sort_values(["doc_id", "loop"], kind="stable").reset_index(drop=True)
    n = len(df) // LOOPS
    loops = df["loop"].to_numpy().reshape(n, LOOPS)
    if not (loops == np.arange(1, LOOPS + 1)).all():
        raise ValueError("loop values are not exactly 1..4 per question")
    ids = df["doc_id"].to_numpy().reshape(n, LOOPS)
    if not (ids == ids[:, [0]]).all():
        raise ValueError("doc_id rows are not contiguous")

    def col(name, dtype=float):
        return df[name].to_numpy().astype(dtype).reshape(n, LOOPS)

    ok = df["status"].to_numpy().reshape(n, LOOPS)[:, 0] == "ok"
    correct = pd.to_numeric(df["correct"]).fillna(0).to_numpy().astype(np.int64).reshape(n, LOOPS)
    if not np.isin(correct, (0, 1)).all():
        raise ValueError("correct must be 0/1")
    pred = pd.to_numeric(df["pred"]).fillna(-1).to_numpy().astype(np.int64).reshape(n, LOOPS)
    conf, ent, cdf = col("conf"), col("ent"), col("cdf")
    D = col("D")[:, :3]
    ms = col("ms")
    T = df["T"].to_numpy().astype(str).reshape(n, LOOPS)[:, :3]
    Tfin = df["T_final"].to_numpy().astype(str).reshape(n, LOOPS)[:, :3]
    for name, a in (("conf", conf[ok]), ("ent", ent[ok]), ("cdf", cdf[ok]), ("D", D[ok])):
        if not np.isfinite(a).all():
            raise ValueError(f"non-finite {name} in an ok question")
    for name, a in (("T", T[ok]), ("T_final", Tfin[ok])):
        if not np.isin(a, TRANSITIONS).all():
            raise ValueError(f"bad transition label in {name}")
    # protocol failure rule: wrong at every loop, 4 loops under every policy, no statistic can fire
    bad = ~ok
    correct[bad] = 0
    conf[bad] = -np.inf
    ent[bad] = np.inf
    cdf[bad] = -np.inf
    D[bad] = np.inf
    T = T.copy()
    Tfin = Tfin.copy()
    T[bad] = "WW"
    Tfin[bad] = "WW"
    return {"n": n, "doc_id": ids[:, 0], "split": df["split"].to_numpy().reshape(n, LOOPS)[:, 0], "ok": ok,
            "pred": pred, "correct": correct, "conf": conf, "ent": ent, "cdf": cdf, "D": D, "ms": ms, "T": T,
            "Tfin": Tfin}


def load_split(root, benchmarks, split: str, provenance: dict | None = None) -> dict:
    """benchmark -> per-question arrays for rows of `split` only. `provenance` (if given) receives path -> sha256."""
    if split not in SPLITS:
        raise ValueError(f"bad split {split!r}")
    root = Path(root)
    out = {}
    for bench in benchmarks:
        if bench in EXCLUDED_BENCHMARKS or bench not in GATE1_BENCHMARKS:
            raise ValueError(f"{bench!r} is not a Gate 1 benchmark")
        done_paths = sorted((root / "data" / "extracted" / bench).glob("part-*.done.json"))
        if not done_paths:
            raise FileNotFoundError(f"no completed parts for {bench} under {root / 'data' / 'extracted'}")
        frames = [_read_part(root, bench, p, split, provenance) for p in done_paths]
        df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else None
        if df is None:
            raise ValueError(f"{bench}: no {split} rows")
        arrays = build_arrays(df, split)
        assert (arrays["split"] == split).all()
        out[bench] = arrays
    return out


# ------------------------------------------------------------------------------------------------------- policies
def _first_fire(F: np.ndarray, ok: np.ndarray):
    F = F & ok[:, None]
    return F.any(axis=1), F.argmax(axis=1) + 1


def run_policy(d: dict, kind: str, param=None):
    """Return (e, c): exit loop (== executed loops == loop of the answer) and answer correctness, both int arrays [n].

    Failed questions always execute 4 loops and are wrong."""
    n, ok, idx = d["n"], d["ok"], np.arange(d["n"])
    if kind == "fixed":
        if param not in (1, 2, 3, 4):
            raise ValueError("fixed-k needs k in 1..4")
        e = np.full(n, int(param))
    elif kind == "confidence":
        hit, first = _first_fire(d["conf"][:, :3] >= param, ok)
        e = np.where(hit, first, 4)
    elif kind == "entropy":
        hit, first = _first_fire(d["ent"][:, :3] <= param, ok)
        e = np.where(hit, first, 4)
    elif kind == "native_gate":
        hit, first = _first_fire(d["cdf"][:, :3] >= param, ok)
        e = np.where(hit, first, 4)
    elif kind == "deployable_kl":  # D[:, t-1] <= tau after running loop t+1: answer loop t+1, pay t+1
        hit, first = _first_fire(d["D"] <= param, ok)
        e = np.where(hit, first + 1, 4)
    elif kind == "retrospective_kl":  # analysis only: answer loop t, pay t
        hit, first = _first_fire(d["D"] <= param, ok)
        e = np.where(hit, first, 4)
    elif kind == "stable_oracle":
        pred = d["pred"]
        eq = pred == pred[:, [3]]
        suffix = np.logical_and.accumulate(eq[:, ::-1], axis=1)[:, ::-1]
        e = np.where(ok, suffix.argmax(axis=1) + 1, 4)
        assert np.array_equal(pred[idx, e - 1], pred[:, 3]), "stable_final_oracle must answer with the loop-4 prediction"
    elif kind == "earliest_correct":
        corr = d["correct"] == 1
        e = np.where(corr.any(axis=1), corr.argmax(axis=1) + 1, 4)
    else:
        raise ValueError(f"unknown policy {kind!r}")
    e = np.where(ok, e, 4).astype(np.int64)
    c = d["correct"][idx, e - 1].astype(np.int64)
    return e, c


def wall_clock(d: dict, e: np.ndarray) -> np.ndarray:
    """Sum of ms over loops 1..e per question; NaN counts as 0."""
    cum = np.cumsum(np.nan_to_num(d["ms"], nan=0.0), axis=1)
    return cum[np.arange(d["n"]), e - 1]


def policy_of(name: str, frozen_entry: dict | None = None):
    if name.startswith("fixed-"):
        return "fixed", int(name.split("-")[1])
    thr = None if frozen_entry is None else frozen_entry.get("threshold")
    return name, thr


def summarize(data: dict, benchmarks, kind: str, param=None) -> dict:
    per, tot_c, tot_e, tot_c4, tot_ms, tot_n = {}, 0, 0, 0, 0.0, 0
    for b in benchmarks:
        d = data[b]
        e, c = run_policy(d, kind, param)
        acc = 100.0 * c.mean()
        acc4 = 100.0 * d["correct"][:, 3].mean()
        ms = wall_clock(d, e)
        per[b] = {"acc": acc, "saving": 1.0 - e.mean() / LOOPS, "loss": acc4 - acc, "mean_ms": float(ms.mean())}
        tot_c += int(c.sum())
        tot_e += int(e.sum())
        tot_c4 += int(d["correct"][:, 3].sum())
        tot_ms += float(ms.sum())
        tot_n += d["n"]
    macro = {k: float(np.mean([per[b][k] for b in benchmarks])) for k in ("acc", "saving", "loss", "mean_ms")}
    pooled_acc = 100.0 * tot_c / tot_n
    pooled = {"acc": pooled_acc, "saving": 1.0 - tot_e / (LOOPS * tot_n), "loss": 100.0 * tot_c4 / tot_n - pooled_acc,
              "mean_ms": tot_ms / tot_n}
    return {"per_benchmark": per, "macro": macro, "pooled": pooled}


def row_of(summary: dict, threshold=None) -> dict:
    return {"threshold": threshold, "macro_saving": summary["macro"]["saving"], "macro_loss": summary["macro"]["loss"],
            "macro_acc": summary["macro"]["acc"], "macro_mean_ms": summary["macro"]["mean_ms"],
            "per_benchmark_saving": {b: v["saving"] for b, v in summary["per_benchmark"].items()},
            "per_benchmark_loss": {b: v["loss"] for b, v in summary["per_benchmark"].items()},
            "per_benchmark_acc": {b: v["acc"] for b, v in summary["per_benchmark"].items()},
            "pooled_saving": summary["pooled"]["saving"], "pooled_loss": summary["pooled"]["loss"],
            "pooled_acc": summary["pooled"]["acc"]}


# ------------------------------------------------------------------------------------------------------- selection
def pooled_values(data: dict, benchmarks, stat: str) -> np.ndarray:
    parts = []
    for b in benchmarks:
        d = data[b]
        v = d["D"][d["ok"]] if stat == "D" else d[stat][d["ok"]][:, :3]
        parts.append(v.ravel())
    vals = np.concatenate(parts)
    if vals.size == 0:
        raise ValueError(f"no ok questions to build the {stat} grid")
    return vals


def grid_thresholds(values: np.ndarray) -> list:
    out, seen = [], set()
    for t in np.quantile(values, np.linspace(0, 1, N_GRID)):
        t = float(t)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def select_point(rows: list, kind: str):
    """Frozen selection rule over candidate rows (dicts with threshold, macro_saving, macro_loss). None if none qualify."""
    q = [r for r in rows if r["macro_loss"] <= LOSS_LIMIT + TOL]
    if not q:
        return None
    best = max(r["macro_saving"] for r in q)
    c = [r for r in q if r["macro_saving"] >= best - TOL]
    lo = min(r["macro_loss"] for r in c)
    c = [r for r in c if r["macro_loss"] <= lo + TOL]
    if kind.startswith("fixed") or len(c) == 1:
        return c[0]
    if CONSERVATIVE[kind] == "largest":
        return max(c, key=lambda r: r["threshold"])
    return min(c, key=lambda r: r["threshold"])


def run_selection(data: dict, benchmarks) -> dict:
    benchmarks = list(benchmarks)
    baselines = {}
    for k in (1, 2, 3):
        name = f"fixed-{k}"
        row = row_of(summarize(data, benchmarks, "fixed", k))
        sel = select_point([row], "fixed")
        baselines[name] = {"name": name, "kind": "fixed", "k": k, "qualified": sel is not None, "threshold": None,
                           "selected": sel, "grid_thresholds": [], "grid": [row]}
    grids = {}
    for kind in THRESHOLDED:
        thr = grid_thresholds(pooled_values(data, benchmarks, STAT_KEY[kind]))
        grids[kind] = thr
        rows = [row_of(summarize(data, benchmarks, kind, t), t) for t in thr]
        sel = select_point(rows, kind)
        baselines[kind] = {"name": kind, "kind": kind, "qualified": sel is not None,
                           "threshold": None if sel is None else sel["threshold"], "selected": sel,
                           "grid_thresholds": thr, "grid": rows}
    reference = {
        "fixed-4": row_of(summarize(data, benchmarks, "fixed", 4)),
        "stable_final_oracle": row_of(summarize(data, benchmarks, "stable_oracle")),
        "earliest_correct_oracle": row_of(summarize(data, benchmarks, "earliest_correct")),
        "retrospective_kl_grid": [row_of(summarize(data, benchmarks, "retrospective_kl", t), t)
                                  for t in grids["deployable_kl"]],
    }
    return {"benchmarks": benchmarks, "baselines": baselines, "reference": reference, "rules": SELECTION_RULES}


def freeze(selection: dict) -> dict:
    """The part of a selection that is written to frozen_selection.json."""
    out = {}
    for name in CHEAP_BASELINES:
        b = selection["baselines"][name]
        entry = {"kind": b["kind"], "qualified": bool(b["qualified"]), "threshold": b["threshold"],
                 "grid_thresholds": list(b["grid_thresholds"])}
        if b["kind"] == "fixed":
            entry["k"] = b["k"]
        out[name] = entry
    return {"benchmarks": list(selection["benchmarks"]), "baselines": out}


# --------------------------------------------------------------------------------------------------- headroom
def ratio(num: float, den: float) -> float:
    if den > 0:
        return float(num / den)
    return float("inf") if num > 0 else 0.0


def headroom_R(s_or: float, savings: dict):
    """savings: name -> S_b for qualified cheap baselines. Returns (R, argmax name or None)."""
    if not savings:
        return 0.0, None
    best = max(savings, key=lambda k: ratio(savings[k], s_or))
    return ratio(savings[best], s_or), best


def decide(s_or: float, R: float) -> str:
    return "PASS" if (s_or >= S_OR_MIN and R < R_MAX) else "FAIL"


def _check_frozen(frozen: dict) -> None:
    if set(frozen) != set(CHEAP_BASELINES):
        raise ValueError(f"frozen baselines must be exactly {CHEAP_BASELINES}")


def evaluate_frozen(data: dict, benchmarks, frozen: dict) -> dict:
    """Headroom of the frozen baselines on whatever split `data` holds. `frozen`: name -> {qualified, threshold, ...}."""
    benchmarks = list(benchmarks)
    _check_frozen(frozen)
    stable = summarize(data, benchmarks, "stable_oracle")
    s_or, s_or_pooled = stable["macro"]["saving"], stable["pooled"]["saving"]
    base, sav, sav_p = {}, {}, {}
    for name in CHEAP_BASELINES:
        f = frozen[name]
        if not f["qualified"]:
            base[name] = {"qualified": False, "threshold": None, "summary": None}
            continue
        kind, param = policy_of(name, f)
        if param is None:
            raise ValueError(f"{name} is qualified but has no threshold")
        s = summarize(data, benchmarks, kind, param)
        base[name] = {"qualified": True, "threshold": f["threshold"], "summary": s, "saving": s["macro"]["saving"],
                      "loss": s["macro"]["loss"], "acc": s["macro"]["acc"], "mean_ms": s["macro"]["mean_ms"],
                      "pooled_saving": s["pooled"]["saving"], "pooled_loss": s["pooled"]["loss"],
                      "constraint_violation": bool(s["macro"]["loss"] > LOSS_LIMIT + TOL)}
        sav[name] = s["macro"]["saving"]
        sav_p[name] = s["pooled"]["saving"]
    R, arg = headroom_R(s_or, sav)
    R_pooled, arg_pooled = headroom_R(s_or_pooled, sav_p)
    per_b_R, per_b_arg, per_b_sor = {}, {}, {}
    for b in benchmarks:
        sb = stable["per_benchmark"][b]["saving"]
        per_b_sor[b] = sb
        r, a = headroom_R(sb, {n: base[n]["summary"]["per_benchmark"][b]["saving"] for n in sav})
        per_b_R[b], per_b_arg[b] = r, a
    verdict = decide(s_or, R)
    return {"benchmarks": benchmarks, "S_or": s_or, "S_or_pooled": s_or_pooled, "S_or_per_benchmark": per_b_sor,
            "stable_final_oracle": stable, "earliest_correct_oracle": summarize(data, benchmarks, "earliest_correct"),
            "fixed-4": summarize(data, benchmarks, "fixed", 4), "baselines": base, "R": R, "R_argmax": arg,
            "R_pooled": R_pooled, "R_pooled_argmax": arg_pooled, "R_per_benchmark": per_b_R,
            "R_argmax_per_benchmark": per_b_arg, "verdict": verdict}


def r_unc(data: dict, benchmarks, frozen: dict, s_or: float) -> dict:
    """Test-only diagnostic: best saving per baseline over the frozen grid among points with loss <= 1 on `data`."""
    benchmarks = list(benchmarks)
    _check_frozen(frozen)
    best = {}
    for name in CHEAP_BASELINES:
        f = frozen[name]
        if name.startswith("fixed-"):
            cands = [(None, summarize(data, benchmarks, "fixed", int(name[6:])))]
        else:
            cands = [(t, summarize(data, benchmarks, name, t)) for t in f["grid_thresholds"]]
        ok = [(t, s) for t, s in cands if s["macro"]["loss"] <= LOSS_LIMIT + TOL]
        if not ok:
            best[name] = None
            continue
        t, s = max(ok, key=lambda x: x[1]["macro"]["saving"])
        best[name] = {"threshold": t, "saving": s["macro"]["saving"], "loss": s["macro"]["loss"]}
    sv = {n: v["saving"] for n, v in best.items() if v is not None}
    R, arg = headroom_R(s_or, sv)
    return {"per_baseline": best, "R_unc": R, "argmax": arg}


# --------------------------------------------------------------------------------------------------- bootstrap
def ci95(x) -> list:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def is_marginal(s_or_ci, r_ci) -> bool:
    return bool(s_or_ci[0] <= S_OR_MIN <= s_or_ci[1] or r_ci[0] <= R_MAX <= r_ci[1])


def bootstrap_headroom(data: dict, benchmarks, frozen: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired, benchmark-stratified bootstrap of S_or, qualified S_b / loss, and R at the frozen thresholds."""
    benchmarks = list(benchmarks)
    _check_frozen(frozen)
    names = [n for n in CHEAP_BASELINES if frozen[n]["qualified"]]
    cols = [("fixed", 4), ("stable_oracle", None)] + [policy_of(n, frozen[n]) for n in names]
    C, E = [], []
    for b in benchmarks:
        cs, es = [], []
        for kind, param in cols:
            e, c = run_policy(data[b], kind, param)
            cs.append(c)
            es.append(e)
        C.append(np.stack(cs, axis=1).astype(float))
        E.append(np.stack(es, axis=1).astype(float))
    P = len(cols)
    rng = np.random.default_rng(seed)
    S_or = np.empty(n_boot)
    R = np.zeros(n_boot)
    S_b = np.empty((n_boot, len(names)))
    L_b = np.empty((n_boot, len(names)))
    for r in range(n_boot):
        acc = np.zeros(P)
        sav = np.zeros(P)
        for bi in range(len(benchmarks)):
            n = C[bi].shape[0]
            idx = rng.integers(0, n, n)
            acc += 100.0 * C[bi][idx].mean(axis=0)
            sav += 1.0 - E[bi][idx].mean(axis=0) / LOOPS
        acc /= len(benchmarks)
        sav /= len(benchmarks)
        S_or[r] = sav[1]
        S_b[r] = sav[2:]
        L_b[r] = acc[0] - acc[2:]
        if names:
            den = sav[1]
            if den > 0:
                R[r] = (sav[2:] / den).max()
            else:
                R[r] = np.inf if (sav[2:] > 0).any() else 0.0
    sor_ci, r_ci = ci95(S_or), ci95(R)
    return {"n_boot": n_boot, "seed": seed, "S_or_ci": sor_ci, "R_ci": r_ci, "marginal": is_marginal(sor_ci, r_ci),
            "baselines": {n: {"saving_ci": ci95(S_b[:, j]), "loss_ci": ci95(L_b[:, j])} for j, n in enumerate(names)}}


TRANSITION_TABLES = (("1->2", "T", 0), ("2->3", "T", 1), ("3->4", "T", 2),
                     ("1->4", "Tfin", 0), ("2->4", "Tfin", 1), ("3->4 (T_final)", "Tfin", 2))


def transition_tables(data: dict, benchmarks, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    benchmarks = list(benchmarks)
    out = {}
    for title, key, j in TRANSITION_TABLES:
        per, net_vec = {}, []
        for b in benchmarks:
            t = data[b][key][:, j]
            cnt = {k: int((t == k).sum()) for k in TRANSITIONS}
            n = data[b]["n"]
            net = cnt["WC"] - cnt["CW"]
            per[b] = {**cnt, "n": n, "net": net, "net_pts": 100.0 * net / n}
            net_vec.append(((t == "WC").astype(float) - (t == "CW").astype(float)))
        rng = np.random.default_rng(seed)
        boot = np.empty((n_boot, len(benchmarks)))
        for r in range(n_boot):
            for bi, v in enumerate(net_vec):
                idx = rng.integers(0, len(v), len(v))
                boot[r, bi] = 100.0 * v[idx].mean()
        for bi, b in enumerate(benchmarks):
            per[b]["net_pts_ci"] = ci95(boot[:, bi])
        out[title] = {"per_benchmark": per, "macro_net_pts": float(np.mean([per[b]["net_pts"] for b in benchmarks])),
                      "macro_net_pts_ci": ci95(boot.mean(axis=1))}
    return out


def analyze(data: dict, benchmarks, frozen: dict, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    ev = evaluate_frozen(data, benchmarks, frozen)
    bs = bootstrap_headroom(data, benchmarks, frozen, n_boot, seed)
    ev["bootstrap"] = bs
    ev["marginal"] = bs["marginal"]
    ev["label"] = ev["verdict"] + (" (MARGINAL)" if bs["marginal"] else "")
    return ev


# ---------------------------------------------------------------------------------------------------- markdown
def f(x, nd=3) -> str:
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def md_selection(sel: dict) -> str:
    rows = []
    for name in CHEAP_BASELINES:
        b = sel["baselines"][name]
        s = b["selected"]
        rows.append([name, "yes" if b["qualified"] else "no", f(b["threshold"], 6),
                     f(s and s["macro_saving"]), f(s and s["macro_loss"]), f(s and s["macro_acc"], 2)])
    return md_table(["baseline", "qualified", "threshold", "macro saving", "macro loss (pp)", "macro acc (%)"], rows)


def md_evaluation(ev: dict, title: str) -> str:
    out = [f"### {title}\n"]
    rows = []
    for name in CHEAP_BASELINES:
        b = ev["baselines"][name]
        if not b["qualified"]:
            rows.append([name, "no", "-", "-", "-", "-", "-", "-"])
            continue
        ci = ev["bootstrap"]["baselines"][name]
        rows.append([name, "yes", f(b["threshold"], 6), f(b["saving"]), f"[{f(ci['saving_ci'][0])}, {f(ci['saving_ci'][1])}]",
                     f(b["loss"]), f"[{f(ci['loss_ci'][0])}, {f(ci['loss_ci'][1])}]",
                     "VIOLATES loss<=1" if b["constraint_violation"] else "ok"])
    out.append(md_table(["baseline", "qualified", "threshold", "S_b", "S_b 95% CI", "macro loss (pp)", "loss 95% CI",
                         "constraint"], rows))
    bs = ev["bootstrap"]
    out.append(f"- S_or (macro saving of stable_final_oracle): {f(ev['S_or'])}  95% CI [{f(bs['S_or_ci'][0])}, {f(bs['S_or_ci'][1])}]"
               f"  (question-weighted {f(ev['S_or_pooled'])})")
    out.append(f"- R = max S_b / S_or: {f(ev['R'])}  95% CI [{f(bs['R_ci'][0])}, {f(bs['R_ci'][1])}]  attained by: {ev['R_argmax']}"
               f"  (question-weighted R {f(ev['R_pooled'])}, {ev['R_pooled_argmax']})")
    out.append(f"- Decision (PASS iff S_or >= 0.25 and R < 0.75): **{ev['label']}**\n")
    rows = [[b, f(ev["S_or_per_benchmark"][b]), f(ev["R_per_benchmark"][b]), ev["R_argmax_per_benchmark"][b]]
            for b in ev["benchmarks"]]
    out.append(md_table(["benchmark", "S_or", "R", "argmax"], rows))
    return "\n".join(out) + "\n"


def md_reference(ev: dict, sel: dict | None = None) -> str:
    rows = []
    items = [("fixed-4", ev["fixed-4"]), ("stable_final_oracle", ev["stable_final_oracle"]),
             ("earliest_correct_oracle (reference only)", ev["earliest_correct_oracle"])]
    for name, s in items:
        rows.append([name, f(s["macro"]["saving"]), f(s["macro"]["loss"]), f(s["macro"]["acc"], 2),
                     f(s["macro"]["mean_ms"], 1), f(s["pooled"]["saving"]), f(s["pooled"]["loss"])])
    for name in CHEAP_BASELINES:
        b = ev["baselines"][name]
        if b["qualified"]:
            s = b["summary"]
            rows.append([name, f(s["macro"]["saving"]), f(s["macro"]["loss"]), f(s["macro"]["acc"], 2),
                         f(s["macro"]["mean_ms"], 1), f(s["pooled"]["saving"]), f(s["pooled"]["loss"])])
    out = md_table(["policy", "macro saving", "macro loss (pp)", "macro acc (%)", "mean ms/question",
                    "pooled saving", "pooled loss (pp)"], rows)
    if sel is not None:
        rr = [[f(r["threshold"], 6), f(r["macro_saving"]), f(r["macro_loss"]), f(r["macro_acc"], 2)]
              for r in sel["reference"]["retrospective_kl_grid"]]
        out += "\nRetrospective-KL grid (analysis only; answers with loop t, never a baseline):\n\n"
        out += md_table(["threshold", "macro saving", "macro loss (pp)", "macro acc (%)"], rr)
    return out


def md_transitions(tr: dict) -> str:
    out = []
    for title, t in tr.items():
        rows = []
        for b, v in t["per_benchmark"].items():
            rows.append([b, v["CC"], v["CW"], v["WC"], v["WW"], v["n"], v["net"], f(v["net_pts"], 2),
                         f"[{f(v['net_pts_ci'][0], 2)}, {f(v['net_pts_ci'][1], 2)}]"])
        rows.append(["macro", "", "", "", "", "", "", f(t["macro_net_pts"], 2),
                     f"[{f(t['macro_net_pts_ci'][0], 2)}, {f(t['macro_net_pts_ci'][1], 2)}]"])
        out.append(f"#### Transition {title}\n\n" + md_table(
            ["benchmark", "CC", "CW", "WC", "WW", "n", "net (WC-CW)", "net_pts", "net_pts 95% CI"], rows))
    return "\n".join(out)
