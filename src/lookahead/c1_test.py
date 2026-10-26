"""C1 one-time TEST evaluation (docs/C1_PROTOCOL.md C1.5-C1.7). Applies the FROZEN C1 selections once to the sealed TEST.

Pure logic here (testable on synthetic data); the guarded entry point is scripts/c1_test_run.py, which:
  1. verifies every frozen hash (protocol, frozen.json, model file, the C1 code recorded at freeze) BEFORE anything else;
  2. refuses if results/c1/TEST_OPENED.json exists (no reruns), then creates it with exclusive-create BEFORE reading any
     TEST outcome;
  3. reads data/extracted_phase3_test once, through the reviewed G3a loader (`load_test`), restricted to the frozen TEST id
     list, with the manifest's counts;
  4. computes the verdict and every secondary analysis and writes results/c1/test_report.json (refusing to overwrite).
Nothing here selects or tunes anything: every (tau, m), model and rho comes from results/c1/frozen.json.
"""
import contextlib
import math

import numpy as np

from lookahead import c1 as C1
from lookahead import e1 as E
from lookahead import gate1 as g1
from lookahead import gate2 as g2
from lookahead import phase3_build as P
from lookahead import phase3_g3a as G
from lookahead import phase3_readouts as R

PRIMARY = G.PRIMARY
OOD = ("piqa", "social_iqa")
LIMIT = g1.LOSS_LIMIT + g1.TOL
VERDICTS = ("FAIL (budget)", "PASS", "FAIL (no meaningful advantage)", "INCONCLUSIVE")


# ---------------------------------------------------------------------------------------------- frozen artifacts
def load_models(npz) -> list:
    """pi1-x per-loop (scaler, LinearModel) from the frozen model file."""
    out = []
    for t in range(1, 4):
        sc = {"mean": np.asarray(npz[f"loop{t}_mean"], np.float64), "sd": np.asarray(npz[f"loop{t}_sd"], np.float64)}
        out.append((sc, g2.LinearModel(npz[f"loop{t}_coef"], float(npz[f"loop{t}_intercept"]))))
    return out


def pi1x_test_stat(d: G.DevData, models: list) -> np.ndarray:
    S = np.full((d.N, 3), np.nan)
    rows = np.flatnonzero(d.ok)
    S[rows] = E.predict_e1(models, d, rows, "x")
    return S


def _sel(frozen_sel: dict) -> dict:
    """Frozen (tau, m) per benchmark; tau may have been serialised as the string 'inf'."""
    out = {}
    for b, s in frozen_sel.items():
        tau = s["tau"]
        tau = math.inf if (isinstance(tau, str) and tau.lower() in ("inf", "infinity")) else float(tau)
        out[b] = {"tau": tau, "m": int(s["m"])}
    return out


# ---------------------------------------------------------------------------------------------- per-question outcome
def apply_policy(d: G.DevData, S: np.ndarray, sel: dict, benches) -> dict:
    """{benchmark: {e, c, c4, halted_by_score, m}} under the frozen (tau, m); failed questions exit at 4 and are wrong."""
    out = {}
    for b in benches:
        r = np.flatnonzero(d.bench == b)
        if not len(r):
            raise ValueError(f"{b}: no TEST rows")
        tau, m = sel[b]["tau"], sel[b]["m"]
        e = R.exit_loops(S[r], d.ok[r], tau, m)
        c = np.where(d.ok[r], d.correct[r, e - 1], 0)
        c4 = np.where(d.ok[r], d.correct[r, 3], 0)
        out[b] = {"e": e, "c": c, "c4": c4, "halted_by_score": (e < m) & d.ok[r], "m": m, "ids": d.doc_id[r]}
    return out


def summary(pol: dict, rho: dict | None = None) -> dict:
    out = {}
    for b, p in pol.items():
        rec = {"n": int(len(p["e"])), "saving": float(1 - p["e"].mean() / 4.0),
               "loss_pp": float(100.0 * (p["c4"].mean() - p["c"].mean())),
               "acc_pp": float(100.0 * p["c"].mean()), "acc4_pp": float(100.0 * p["c4"].mean()),
               "exit_hist": {str(k): int((p["e"] == k).sum()) for k in range(1, 5)}}
        if rho is not None:
            rec["net_saving"] = C1.net_saving(p["e"], p["halted_by_score"], p["m"], rho[b]["rho_score"], rho[b]["rho_rest"])
        out[b] = rec
    return out


# ---------------------------------------------------------------------------------------------- bootstrap
def boot_indices(n_by_bench: dict, n_boot: int, seed: int) -> dict:
    """Paired, benchmark-stratified resampling of questions: {benchmark: int32 [n_boot, n_b] indices}, one rng, benchmarks
    in sorted order. Shared by every quantity of the report."""
    rng = np.random.default_rng(seed)
    return {b: rng.integers(0, n_by_bench[b], size=(n_boot, n_by_bench[b]), dtype=np.int32) for b in sorted(n_by_bench)}


def _boot_mean(x: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return x[idx].mean(axis=1)


def paired_macro(a: dict, b: dict, benches, idx: dict, key: str = "gross", rho_a=None, rho_b=None) -> dict:
    """Δ = macro_b-equal-weight (saving_a - saving_b) on the same questions; bootstrap distribution with shared indices.
    key 'gross' uses 1 - e/4; 'net' uses the per-question net cost (C1.3)."""
    def per_q(p, rho, bench):
        if key == "gross" or rho is None:
            return 1.0 - p["e"] / 4.0
        hb = p["halted_by_score"]
        n_eval = np.where(hb, p["e"], p["m"] - 1)
        n_score = np.where(hb, p["e"] - 1, p["m"] - 1)
        return 1.0 - (p["e"] + rho[bench]["rho_score"] * n_score + rho[bench]["rho_rest"] * n_eval) / 4.0

    diffs, point = [], []
    for bench in benches:
        if not np.array_equal(a[bench]["ids"], b[bench]["ids"]):
            raise AssertionError(f"{bench}: the two policies are not evaluated on the same questions")
        u = per_q(a[bench], rho_a, bench) - per_q(b[bench], rho_b, bench)
        point.append(u.mean())
        diffs.append(_boot_mean(u, idx[bench]))
    dist = np.mean(diffs, axis=0)
    return {"delta": float(np.mean(point)), "L": float(np.quantile(dist, 0.025)), "U": float(np.quantile(dist, 0.975)),
            "dist": dist}


def loss_upper_bound(p: dict, idx: np.ndarray) -> float:
    """One-sided 95% bootstrap upper bound of the loss in pp (demonstrated compliance, A1.5), shared indices."""
    l = 100.0 * (p["c4"] - p["c"]).astype(float)
    return float(np.quantile(_boot_mean(l, idx), 0.95))


def verdict(pi_summary: dict, delta: float, L: float, U: float, delta_min: float = C1.DELTA) -> str:
    """C1.5 exhaustive table; the first matching row applies."""
    if any(pi_summary[b]["loss_pp"] > LIMIT for b in PRIMARY):
        return VERDICTS[0]
    if delta >= delta_min and L > 0:
        return VERDICTS[1]
    if U < delta_min:
        return VERDICTS[2]
    return VERDICTS[3]


# ---------------------------------------------------------------------------------------------- the evaluation
def evaluate_c1(frozen: dict, models: list, test: G.DevData, n_boot: int | None = None) -> dict:
    """The whole C1 TEST evaluation on in-memory TEST data (primary + OOD rows). Returns the report dict."""
    n_boot = n_boot or frozen["n_boot_test"]
    rho = frozen["rho"] if isinstance(frozen["rho"], dict) else None
    S = {"pi1-x": pi1x_test_stat(test, models), **{n: G.baseline_stat(test, n) for n in G.BASELINES}}
    risk = {"pi1-x": _sel(frozen["primary"]["pi1-x"]), "native-pb": _sel(frozen["primary"]["native-pb"]),
            **{n: _sel(v) for n, v in frozen["secondary_risk_all_baselines"].items()}}
    pol = {n: apply_policy(test, S[n], risk[n], PRIMARY) for n in risk}
    n_b = {b: len(pol["pi1-x"][b]["e"]) for b in PRIMARY}
    idx = boot_indices(n_b, n_boot, frozen["test_seed"])

    prim = paired_macro(pol["pi1-x"], pol["native-pb"], PRIMARY, idx)
    sums = {n: summary(p, rho if n == "pi1-x" else None) for n, p in pol.items()}
    v = verdict(sums["pi1-x"], prim["delta"], prim["L"], prim["U"])
    ub = {n: {b: loss_upper_bound(pol[n][b], idx[b]) for b in PRIMARY} for n in ("pi1-x", "native-pb", "B_pb")}

    rep = {"protocol": frozen["protocol"], "verdict": v, "delta_min": C1.DELTA, "n_boot": n_boot, "seed": frozen["test_seed"],
           "primary": {"delta_gross": prim["delta"], "ci95": [prim["L"], prim["U"]],
                       "L_above_delta_min": bool(prim["L"] > C1.DELTA),
                       "pass_meaning": "L > 0 and observed delta >= 0.02; NOT a 95% claim that the effect is >= 0.02"},
           "budget_table": {n: {b: {"loss_pp": sums[n][b]["loss_pp"], "upper_bound_95_pp": ub[n][b],
                                    "point_over_1pp": sums[n][b]["loss_pp"] > LIMIT,
                                    "upper_bound_over_1pp": ub[n][b] > LIMIT} for b in PRIMARY}
                            for n in ("pi1-x", "native-pb", "B_pb")},
           "per_benchmark": {n: sums[n] for n in ("pi1-x", "native-pb", "B_pb")},
           "macro_gross_saving": {n: float(np.mean([sums[n][b]["saving"] for b in PRIMARY])) for n in sums},
           "n": n_b}
    # secondary 3: net saving (estimated deployment cost under the defined boundary; not measured latency)
    if rho is not None:
        net = paired_macro(pol["pi1-x"], pol["native-pb"], PRIMARY, idx, key="net", rho_a=rho)
        rep["secondary_net"] = {"delta_net": net["delta"], "ci95": [net["L"], net["U"]],
                                "note": "estimated deployment cost under the C1.3 accounting boundary"}
    # secondary 1: point-rule selections
    pt = {n: apply_policy(test, S[n], _sel(frozen["secondary_point_rule"][n]), PRIMARY) for n in ("pi1-x", "native-pb")}
    ptd = paired_macro(pt["pi1-x"], pt["native-pb"], PRIMARY, idx)
    rep["secondary_point_rule"] = {"delta": ptd["delta"], "ci95": [ptd["L"], ptd["U"]],
                                   "per_benchmark": {n: summary(p) for n, p in pt.items()}}
    # secondary 2: every baseline under the same risk rule, and the strongest re-selected in every resample (A2.5)
    others = [n for n in G.BASELINES if n in pol]
    vs = {}
    for n in others:
        d = paired_macro(pol["pi1-x"], pol[n], PRIMARY, idx)
        vs[n] = {"delta": d["delta"], "ci95": [d["L"], d["U"]], "dist": d["dist"]}
    macro_pts = {n: float(np.mean([sums[n][b]["saving"] for b in PRIMARY])) for n in others}
    strongest = max(others, key=lambda n: (macro_pts[n], -others.index(n)))
    resel = np.min(np.stack([vs[n]["dist"] for n in others]), axis=0)           # pi1-x minus the best baseline per resample
    winners = np.stack([-vs[n]["dist"] for n in others]).argmax(axis=0)
    rep["secondary_all_baselines"] = {
        "per_baseline": {n: {"delta": vs[n]["delta"], "ci95": vs[n]["ci95"]} for n in others},
        "label": ("secondary; the strongest baseline is a TEST-selected comparator (chosen using TEST outcomes) and never "
                  "affects the primary verdict"),
        "strongest_on_test_point": strongest, "delta_vs_strongest_point": vs[strongest]["delta"],
        "reselecting_bootstrap_ci95": [float(np.quantile(resel, 0.025)), float(np.quantile(resel, 0.975))],
        "selection_frequency": {others[i]: int((winners == i).sum()) for i in range(len(others))}}
    # secondary 4: OOD, each benchmark on its own, with the task-agnostic pooled (tau, m)
    ood = {}
    present = [b for b in OOD if b in test.benchmarks]
    for b in present:
        r = {}
        for n in ("pi1-x", "native-pb", "B_pb"):
            s = frozen["ood_pooled"][n]
            tau = math.inf if str(s["tau"]).lower() in ("inf", "infinity") else float(s["tau"])
            r[n] = apply_policy(test, S[n] if n != "B_pb" else np.zeros((test.N, 3)), {b: {"tau": tau, "m": int(s["m"])}}, [b])[b]
        ib = boot_indices({b: len(r["pi1-x"]["e"])}, n_boot, frozen["test_seed"] + 1 + OOD.index(b))
        d = paired_macro({b: r["pi1-x"]}, {b: r["native-pb"]}, [b], ib)
        ood[b] = {"summary": {n: summary({b: p})[b] for n, p in r.items()},
                  "upper_bound_95_pp": {n: loss_upper_bound(p, ib[b]) for n, p in r.items()},
                  "delta_pi1x_vs_native": d["delta"], "ci95": [d["L"], d["U"]]}
    rep["secondary_ood"] = ood
    # secondary 5: violation table for every policy under the risk rule
    rep["secondary_violations"] = {n: {b: sums[n][b]["loss_pp"] for b in PRIMARY} for n in sums}
    return rep


# ---------------------------------------------------------------------------------------------- TEST loading
TEST_DIR = "extracted_phase3_test"


def _within(p, base) -> bool:
    from pathlib import Path

    try:
        Path(p).resolve().relative_to(Path(base).resolve())
        return True
    except ValueError:
        return False


@contextlib.contextmanager
def unsealed(root):
    """The ONE place the sealed TEST extraction may be opened. Inside this context the reviewed G3a loader's guard allows a
    path ONLY if, after resolving symlinks and '..', it lies inside <root>/data/extracted_phase3_test; every other path,
    including Gate 2 confirmation data, other extraction directories and traversal out of the TEST directory, is refused.
    The original guard is restored on exit, also after an exception. Used only by scripts/c1_test_run.py after the marker
    exists."""
    from pathlib import Path

    base = Path(root) / "data" / TEST_DIR
    orig = G._guard

    def guard(p):
        s = str(p).replace("\\", "/").lower()
        if "confirm" in s:
            raise PermissionError(f"refusing Gate 2 confirmation path: {p}")
        if not _within(p, base):
            raise PermissionError(f"outside the approved TEST extraction {base}: {p}")
        g2._assert_allowed_path(p)

    G._guard = guard
    try:
        yield
    finally:
        G._guard = orig


def metadata_preflight(root) -> dict:
    """Metadata-only checks of the TEST extraction BEFORE the marker (reviewer, TEST script review). Reads only the manifest,
    the frozen id lists, the committed TEST extraction check report and each part's done.json (doc ids, statuses, counts,
    file checksums; each benchmark's id set must EQUAL its frozen TEST list); computes sha256 of the raw part files WITHOUT parsing them. No prediction, label, score or other TEST
    measurement is read. Raises SystemExit on any problem."""
    import json
    from pathlib import Path

    root = Path(root)
    man = json.loads((root / P.MANIFEST).read_text(encoding="utf-8"))
    if not P._sha_ok(root / P.IDS["test"], man["ids_sha256"]["test"]) or not P._sha_ok(root / P.IDS["dev"], man["ids_sha256"]["dev"]):
        raise SystemExit("phase3 id lists do not match the manifest")
    chk = json.loads((root / "results/phase3/extraction_check_phase3_test.json").read_text(encoding="utf-8"))
    if not chk.get("passed"):
        raise SystemExit("the committed TEST extraction check did not pass")
    expected = {}
    lines = (root / P.IDS["test"]).read_text(encoding="utf-8").splitlines()
    if lines[0] != "benchmark,stratum,doc_id,split":
        raise SystemExit("unexpected TEST id-list header")
    for ln in lines[1:]:
        if ln:
            b, _, doc, _ = ln.split(",")
            expected.setdefault(b, set()).add(doc)
    base = root / "data" / TEST_DIR
    out = {}
    for b in PRIMARY + OOD:
        bdir = base / b
        dones = sorted(bdir.glob("part-*.done.json")) if bdir.is_dir() else []
        if not dones:
            raise SystemExit(f"{b}: no completed TEST parts under {bdir}")
        ids, n = [], 0
        for dp in dones:
            d = json.loads(dp.read_text(encoding="utf-8"))
            stem = str(dp)[: -len(".done.json")]
            for suffix, key in ((".parquet", "parquet_sha256"), (".hidden.npy", "hidden_sha256")):
                f = Path(stem + suffix)
                if not f.exists() or g1.sha256_file(f) != d[key]:
                    raise SystemExit(f"{f.name}: missing or checksum mismatch")
            ids += d["doc_ids"]
            n += d["n"]
            if any(st != "ok" for st in d["statuses"]):
                raise SystemExit(f"{dp.name}: a TEST question did not extract")
        expn = man["sets"][f"test/{b}"]["n_rows"]
        if n != expn or len(ids) != expn or len(set(ids)) != len(ids):
            raise SystemExit(f"{b}: {n} questions (manifest {expn}) or duplicate ids")
        if set(ids) != expected.get(b, set()):                         # EXACT per-benchmark id set (reviewer)
            raise SystemExit(f"{b}: the TEST question ids differ from the frozen TEST list for this benchmark")
        out[b] = {"n": n, "parts": len(dones)}
    return out


def load_test(root) -> G.DevData:
    """Primary (split 'test') and OOD (split 'test_ood') TEST rows through G.load_source, restricted to the frozen TEST id
    list (hash-checked against the manifest), with the manifest's counts. DEV+ ids are refused. Call inside
    `unsealed(root)`."""
    import json
    from pathlib import Path

    root = Path(root)
    man = json.loads((root / P.MANIFEST).read_text(encoding="utf-8"))
    if not P._sha_ok(root / P.IDS["test"], man["ids_sha256"]["test"]) or not P._sha_ok(root / P.IDS["dev"], man["ids_sha256"]["dev"]):
        raise SystemExit("phase3 id lists do not match the manifest")
    allow, deny = G.read_id_list(root / P.IDS["test"]), G.read_id_list(root / P.IDS["dev"])
    exp = {b: man["sets"][f"test/{b}"]["n_rows"] for b in PRIMARY + OOD}
    a = G.load_source(root, TEST_DIR, "test", "test", PRIMARY, allow_ids=allow, deny_ids=deny,
                      expected_n={b: exp[b] for b in PRIMARY})
    o = G.load_source(root, TEST_DIR, "test_ood", "test_ood", OOD, allow_ids=allow, deny_ids=deny,
                      expected_n={b: exp[b] for b in OOD})
    return G.concat([a, o])
