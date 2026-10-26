"""Exploratory study E1: synthetic data only. Nested isolation, no future information, pairing, identical risk control,
feature sets, transforms, refusals, and an end-to-end synthetic run."""
import hashlib
import json
import os
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase3_g3a import _perturb_fold, synth  # noqa: E402

from lookahead import e1 as E  # noqa: E402
from lookahead import gate2 as g2  # noqa: E402
from lookahead import phase3_g3a as G  # noqa: E402
from lookahead import phase3_readouts as R  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _base(d):
    return {n: G.baseline_stat(d, n) for n in G.BASELINES}


@pytest.fixture(scope="module")
def dev():
    return synth(n=90, seed=31)


# ------------------------------------------------------------------------------------------ 1. nested isolation
@pytest.mark.parametrize("k", [0, 3])
def test_selection_side_is_untouched_by_the_held_out_fold(dev, k):
    e = _perturb_fold(dev, k)
    sel = dev.fold != k
    ma, fa = E.build_methods(dev, sel, dev.fold == k, _base(dev))
    mb, fb = E.build_methods(e, sel, e.fold == k, _base(e))
    assert fa == fb and set(fa) >= {"pi1-x", "pi1-cdf", "pi1-x@c_t", "pi1-x@safe", "pi1-x+rank", "pi1-x+cummax", "combo-max"}
    assert "rank_maps_sha256" in fa["pi1-x+rank"]
    for n in ma:                                          # selection-row scores identical (NaN-aware)
        assert np.array_equal(ma[n][sel], mb[n][sel], equal_nan=True), n
    boot_a, boot_b = E.make_boot(dev, sel, k), E.make_boot(e, sel, k)
    assert E.boot_fingerprint(boot_a) == E.boot_fingerprint(boot_b)
    for n in ma:
        fixed = n == "B_pb"
        pa, pb = G.select_all(dev, ma[n], sel, fixed), G.select_all(e, mb[n], sel, fixed)
        ra, rb = E.select_risk(dev, ma[n], sel, fixed, boot_a), E.select_risk(e, mb[n], sel, fixed, boot_b)
        assert pa == pb and ra == rb, n                   # (tau, m, saving, loss, selection-time bound) all unchanged
    # non-vacuous: held-out evaluation does see the change
    held_a, held_b = dev.fold == k, e.fold == k
    ev = lambda d, S, n: G.evaluate(d, S, G.select_all(d, S, sel, n == "B_pb"), held_a)  # noqa: E731
    assert any(ev(dev, ma[n], n) != ev(e, mb[n], n) for n in ("pi1-x", "pi1-x+rank", "native-pb", "pi1-x@c_t", "combo-max"))
    assert not np.array_equal(ma["pi1-x"][held_a], mb["pi1-x"][held_b], equal_nan=True)


def test_fitted_stat_never_uses_rows_outside_the_selection_mask(dev):
    sel = dev.fold != 1
    S, fp = E.fitted_stat(dev, "x", E.targets(dev)["S"], sel, dev.fold == 1)
    assert np.isnan(S[~sel & (dev.fold != 1)]).all()
    e = _perturb_fold(dev, 1)
    S2, fp2 = E.fitted_stat(e, "x", E.targets(e)["S"], sel, e.fold == 1)
    assert fp == fp2 and np.array_equal(S[sel], S2[sel], equal_nan=True)
    with pytest.raises(ValueError):
        E.fitted_stat(dev, "x", E.targets(dev)["S"], sel, sel)


# ------------------------------------------------------------------------------------ 2. future-loop invariance
def test_cummax_and_rank_ignore_later_loops():
    rng = np.random.default_rng(0)
    S = rng.random((50, 3))
    S[::7, 1] = np.nan
    maps = E.fit_rank_maps(S, np.arange(50))
    for t in range(2):
        S2 = S.copy()
        S2[:, t + 1:] = rng.random((50, 3 - t - 1)) * 9
        assert np.array_equal(E.cummax(S)[:, :t + 1], E.cummax(S2)[:, :t + 1], equal_nan=True)
        assert np.array_equal(E.apply_rank(S, maps)[:, :t + 1], E.apply_rank(S2, maps)[:, :t + 1], equal_nan=True)
    # a rank map of loop t is fitted from loop-t scores only
    S3 = S.copy()
    S3[:, 2] = 77
    assert all(np.array_equal(a, b) for a, b in zip(E.fit_rank_maps(S, np.arange(50))[:2], E.fit_rank_maps(S3, np.arange(50))[:2]))


@pytest.mark.parametrize("feat,t", [("x", 0), ("x", 1), ("hx", 0), ("cdf", 1), ("x-nocdf", 0)])
def test_fitted_score_at_loop_t_ignores_features_of_later_loops(dev, feat, t):
    tgt = E.targets(dev)["S"]
    sel, held = dev.fold != 2, dev.fold == 2
    a, _ = E.fitted_stat(dev, feat, tgt, sel, held)
    for rows in (np.flatnonzero(held), np.arange(dev.N)):          # held rows only, then every row
        e = G.subset(dev, np.ones(dev.N, bool))
        rng = np.random.default_rng(5)
        e.x, e.h = e.x.copy(), e.h.copy()
        e.x[rows, t + 1:] = rng.normal(size=e.x[rows, t + 1:].shape) * 9
        e.h[rows, t + 1:] = rng.normal(size=e.h[rows, t + 1:].shape) * 9
        b, _ = E.fitted_stat(e, feat, tgt, sel, held)
        assert np.array_equal(a[:, :t + 1], b[:, :t + 1], equal_nan=True)


# ----------------------------------------------------------------------------------------------- 3. pairing
def test_pairing_assertion_holds_and_fires_on_different_question_sets(monkeypatch):
    d = synth(n=60, seed=32)
    base = _base(d)
    rec, pairs = E.outer_fold_e1(d, 0, base)
    ref = pairs["B_pb"]["idx"]
    assert all(np.array_equal(p["idx"], ref) for p in pairs.values())
    ids = {json.dumps({b: v["ids_sha256"] for b, v in r[rule]["heldout"].items()}, sort_keys=True)
           for r in rec.values() for rule in ("point", "risk")}
    assert len(ids) == 1
    real, calls = G.evaluate, {"n": 0}

    def skewed(dd, S, sel, mask):
        calls["n"] += 1
        if calls["n"] == 3:
            mask = mask.copy()
            r0 = np.flatnonzero(mask)[0]
            r1 = np.flatnonzero(~mask & (dd.bench == dd.bench[r0]))[0]
            mask[r0], mask[r1] = False, True
        return real(dd, S, sel, mask)

    monkeypatch.setattr(G, "evaluate", skewed)
    with pytest.raises(AssertionError, match="different held-out questions"):
        E.outer_fold_e1(d, 0, base)


# ------------------------------------------------------------------------------------------ 4. identical risk control
def test_select_risk_is_one_function_for_every_method_with_shared_bootstrap(monkeypatch):
    d = synth(n=60, seed=33)
    base = _base(d)
    sel0 = d.fold != 0
    methods, _ = E.build_methods(d, sel0, d.fold == 0, base)
    real, seen = E.select_risk, []

    def spy(dd, S, sel_mask, fixed_only, boot):
        seen.append((hashlib.sha256(np.nan_to_num(S, nan=-9e9).tobytes()).hexdigest(), fixed_only, id(boot),
                     E.boot_fingerprint(boot)))
        return real(dd, S, sel_mask, fixed_only, boot)

    monkeypatch.setattr(E, "select_risk", spy)
    E.outer_fold_e1(d, 0, base)
    exp = [hashlib.sha256(np.nan_to_num(S, nan=-9e9).tobytes()).hexdigest() for S in methods.values()]
    assert [s[0] for s in seen] == exp and len(seen) == len(E.ALL_METHODS)       # exactly once per method, same code path
    assert [s[1] for s in seen] == [n == "B_pb" for n in methods]                  # fixed_only only for B_pb
    assert len({s[2] for s in seen}) == 1 and {s[3] for s in seen} == {E.boot_fingerprint(E.make_boot(d, sel0, 0))}
    assert E.boot_fingerprint(E.make_boot(d, sel0, 0)) != E.boot_fingerprint(E.make_boot(d, sel0, 1))
    b = E.make_boot(d, sel0, 0)
    assert all(v.dtype == np.int32 and v.shape[0] == 2000 and (v.sum(axis=1) == v.shape[1]).all() for v in b.values())


def _one_bench(correct1_wrong):
    """One benchmark, 200 questions, correct everywhere except `correct1_wrong` wrong answers at loop 1; S fires at loop 1."""
    d = synth(benches=("openbookqa",), n=200, seed=34)
    d.correct = np.ones_like(d.correct)
    d.correct[:correct1_wrong, 0] = 0
    S = np.zeros((d.N, 3))
    S[:, 0] = 1.0
    return d, S


def test_fixed_depth_4_always_qualifies_and_a_variable_candidate_can_fail_the_risk_rule():
    d, S = _one_bench(2)                       # point loss exactly 1.0 pp at loop-1 exit, but only 2 bad questions of 200
    allr = np.ones(d.N, bool)
    boot = E.make_boot(d, allr, 0)
    point = G.select_all(d, S, allr)["openbookqa"]
    risk = E.select_risk(d, S, allr, False, boot)["openbookqa"]
    assert point["m"] == 1 and point["loss"] <= 1.0 + 1e-9 and point["saving"] == pytest.approx(0.75)
    assert risk["saving"] < point["saving"] and risk["upper_bound"] <= 1.0 + 1e-12 and risk["saving"] == pytest.approx(0.5)
    # fixed depth 4 (loss identically 0) is always available, even if every earlier exit is awful
    d2, S2 = _one_bench(100)
    r2 = E.select_risk(d2, S2, allr, False, boot)["openbookqa"]
    assert r2["saving"] == pytest.approx(0.5) and r2["upper_bound"] == 0.0     # only the fixed depth-2 policy survives
    d3 = synth(benches=("openbookqa",), n=200, seed=35)
    d3.correct[:, :3] = 0
    d3.correct[:, 3] = 1
    r3 = E.select_risk(d3, np.random.default_rng(0).random((d3.N, 3)), allr, False, boot)["openbookqa"]
    assert (r3["tau"], r3["m"], r3["upper_bound"]) == (math.inf, 4, 0.0)
    # with many bad questions the bound is tight enough that no early exit survives; and zero bad -> point rule's choice
    d4, S4 = _one_bench(0)
    assert E.select_risk(d4, S4, allr, False, boot)["openbookqa"]["saving"] == pytest.approx(0.75)


def test_risk_rule_never_saves_more_than_the_point_rule_and_matches_the_frozen_tie_break():
    d = synth(n=90, seed=36)
    base = _base(d)
    for k in (0, 1):
        sel = d.fold != k
        boot = E.make_boot(d, sel, k)
        methods, _ = E.build_methods(d, sel, d.fold == k, base)
        for n, S in methods.items():
            fixed = n == "B_pb"
            p, r = G.select_all(d, S, sel, fixed), E.select_risk(d, S, sel, fixed, boot)
            for b in d.benchmarks:
                assert r[b]["saving"] <= p[b]["saving"] + 1e-12, (n, b)
                assert r[b]["upper_bound"] <= 1.0 + 1e-12 and r[b]["loss"] <= 1.0 + 1e-9
    # brute-force check of the efficient search against an exhaustive scan on one method
    S, b = methods["pi1-x"], d.benchmarks[0]
    rows = np.flatnonzero(sel)
    r = np.flatnonzero((d.bench == b) & sel)
    best = None
    for tau in R.grid([S[rows]], [d.ok[rows]]):
        for m in (1, 2, 3, 4):
            e = R.exit_loops(S[r], d.ok[r], tau, m)
            sav, loss = R.outcome(d.correct[r], d.ok[r], e)
            l = 100.0 * (d.correct[r, 3] - d.correct[r, e - 1])
            ub = np.quantile(boot[b] @ l / len(r), 0.95)
            if ub <= 1.0 + 1e-12 and loss <= 1.0 + 1e-12:
                key = (-round(sav, 9), loss, 0 if math.isinf(tau) else 1, -tau if math.isfinite(tau) else 0, m)
                best = key if best is None or key < best else best
    got = E.select_risk(d, S, sel, False, boot)[b]
    assert got["saving"] == pytest.approx(-best[0], abs=1e-8)


# ----------------------------------------------------------------------------------------------- 5. feature sets
def test_feature_sets():
    assert g2.CHEAP_NAMES[3] == "cdf"
    d = synth(benches=("openbookqa",), n=20, seed=37)
    rows = np.arange(10)
    for t in range(3):
        assert np.array_equal(E.design(d, rows, t, "cdf"), d.x[rows, t, 3:4])
        nc = E.design(d, rows, t, "x-nocdf")
        assert nc.shape == (10, 8) and np.array_equal(nc, np.delete(d.x[rows, t], 3, axis=1))
        assert np.array_equal(E.design(d, rows, t, "x"), d.x[rows, t])
        assert np.array_equal(E.design(d, rows, t, "hx"), np.concatenate([d.h[rows, t], d.x[rows, t]], axis=1))
    assert E.FEAT_C == {"x": 1.0, "cdf": 1.0, "x-nocdf": 1.0, "hx": 0.001}


def test_targets():
    d = synth(benches=("openbookqa",), n=30, seed=38)
    t = E.targets(d)
    assert all(v.shape == (30, 3) for v in t.values())
    assert np.array_equal(t["c_t"], d.correct[:, :3]) and np.array_equal(t["S"], d.stability())
    assert np.array_equal(t["safe"], (d.correct[:, :3] >= d.correct[:, [3]]).astype(float))


# ------------------------------------------------------------------------------------------ 6. transforms by hand
def test_cummax_and_rank_hand_examples():
    S = np.array([[0.2, 0.5, 0.1], [np.nan, 0.3, np.nan], [np.nan, np.nan, np.nan], [0.9, np.nan, 0.4]])
    cm = E.cummax(S)
    exp = np.array([[0.2, 0.5, 0.5], [np.nan, 0.3, 0.3], [np.nan, np.nan, np.nan], [0.9, 0.9, 0.9]])
    assert np.array_equal(cm, exp, equal_nan=True)
    assert (np.diff(np.nan_to_num(cm, nan=-1), axis=1) >= 0).all()
    maps = E.fit_rank_maps(S, np.arange(4))
    assert [m.tolist() for m in maps] == [[0.2, 0.9], [0.3, 0.5], [0.1, 0.4]]
    rk = E.apply_rank(S, maps)
    assert np.array_equal(rk, np.array([[0.5, 1.0, 0.5], [np.nan, 0.5, np.nan], [np.nan] * 3, [1.0, np.nan, 1.0]]), equal_nan=True)
    held = np.array([[0.0, 9.9, 0.25]])
    assert np.array_equal(E.apply_rank(held, maps), [[0.0, 1.0, 0.5]])      # maps from selection rows applied unchanged
    r = np.random.default_rng(1).normal(size=(40, 3))
    ra = E.apply_rank(r, E.fit_rank_maps(r, np.arange(40)))
    assert ((ra >= 0) & (ra <= 1)).all()
    assert np.array_equal(np.argsort(r[:, 1], kind="stable"), np.argsort(ra[:, 1], kind="stable"))   # within-loop order kept


# ---------------------------------------------------------------------------------------- 7. end-to-end synthetic
def _dev_val():
    dev = G.concat([synth(n=90, seed=10), synth(benches=G.PRIMARY, n=40, seed=11, source="devplus", prefix="p3:")])
    return dev, synth(n=50, seed=12, source="p1_validation", prefix="v/")


def test_run_e1_end_to_end_is_complete_serialisable_and_deterministic():
    dev, val = _dev_val()
    rep = E.run_e1(dev, val)
    assert {"A1", "A2", "A3", "A4", "A5", "A6", "A7", "plan", "status", "n"} <= set(rep)
    assert rep["plan"] == "docs/EXPLORATORY_E1_NATIVE_GATE.md rev 1" and rep["status"] == "exploratory; no confirmatory claim"
    assert set(rep["A1"]["ladder"]) == set(E.LADDER) and set(rep["A1"]["paired_vs_native_pb"]) == set(E.LADDER[1:])
    pv = rep["A1"]["paired_vs_native_pb"]["pi1-x"]
    assert set(pv["primary"]["exit_loop_difference_hist"]) == {str(v) for v in range(-3, 4)}
    assert sum(pv["primary"]["correctness_2x2"].values()) == pv["primary"]["n"]
    lo, hi = pv["saving_difference_primary"]["ci95"]
    assert lo <= pv["saving_difference_primary"]["point"] <= hi
    assert rep["A1"]["ladder"]["native-pb"]["improvement_vs_native_pb"] == 0.0
    assert set(rep["A2"]["prediction_quality"]) == {"S", "c_t", "safe"} and set(rep["A2"]["stopping_utility"]) == {"S", "c_t", "safe"}
    assert set(rep["A3"]) == set(E.A3_METHODS)
    assert set(rep["A6"]["point"]) == set(E.ALL_METHODS) == set(rep["A6"]["risk"])
    assert all(v["selection_time_upper_bounds_primary"] is not None for v in rep["A6"]["risk"].values())
    assert rep["A6"]["risk"]["native-pb"]["improvement_vs_native_pb"] == 0.0
    assert set(rep["A7"]["methods"]) == set(E.A7_METHODS) and rep["A7"]["descriptive_only"]
    assert rep["A5"]["disagreement_native_vs_pi1x"]["primary_pooled"]["n"] == sum(rep["n"]["dev"][b] for b in G.PRIMARY)
    json.dumps(rep, default=str)
    assert json.dumps(E.run_e1(dev, val), sort_keys=True) == json.dumps(rep, sort_keys=True)     # deterministic


def test_a7_selection_never_reads_validation():
    dev, val = _dev_val()
    pack = E.validation_pack(dev)                   # takes no validation argument
    v2 = _perturb_fold(_perturb_fold(val, 0), 1, seed=5)
    a, b = E.validation_apply(pack, val), E.validation_apply(pack, v2)
    for n in a["methods"]:
        assert a["methods"][n]["selected"] == b["methods"][n]["selected"]
        assert a["methods"][n]["selection_saving_dev"] == b["methods"][n]["selection_saving_dev"]


def test_entry_script_synthetic_dry_run_writes_and_refuses_to_overwrite(tmp_path):
    out = tmp_path / "e1.json"
    # keep the inherited environment (Windows needs SYSTEMROOT etc. for its socket layer); only PYTHONPATH is set
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    cmd = [sys.executable, str(ROOT / "scripts" / "e1_run.py"), "--synthetic", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rep = json.loads(out.read_text())
    assert rep["provenance"]["inputs"] == {"synthetic": True} and "src/lookahead/e1.py" in rep["provenance"]["code"]
    before = out.read_bytes()
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r2.returncode == 2 and "refusing" in r2.stderr and out.read_bytes() == before


# ----------------------------------------------------------------------------------------------------- 8. refusals
def test_validation_or_sealed_rows_in_development_data_are_refused():
    base = synth(n=30, seed=39)
    for src in ("p1_validation", "test"):
        bad = G.concat([base, synth(benches=("openbookqa",), n=5, seed=40, source=src, prefix=f"{src}/")])
        with pytest.raises(PermissionError):
            E.run_e1(bad)
    with pytest.raises(PermissionError):
        E.run_e1(base, synth(n=10, seed=41, source="devplus", prefix="z/"))
    with pytest.raises(ValueError, match="commonsense_qa"):
        E.run_e1(synth(benches=("arc_easy", "openbookqa", "hellaswag"), n=40))


def test_sources_never_name_sealed_or_confirmation_paths():
    for p in (ROOT / "src/lookahead/e1.py", ROOT / "scripts/e1_run.py", ROOT / "scripts/run_e1_windows.ps1"):
        txt = p.read_text(encoding="utf-8")
        assert "extracted_phase3_test" not in txt and "phase3_test" not in txt and "extracted_confirm" not in txt, p
        assert "data/confirm" not in txt and "data\\confirm" not in txt, p
    assert "open(" not in (ROOT / "src/lookahead/e1.py").read_text(encoding="utf-8")   # e1 never opens a file itself
