"""Exploratory study X1 (docs/X1_FLIP_PROTOCOL.md rev 2): synthetic data only. Label-free matching, flips by hand, pooling,
the paired bootstrap, the four readings, the end-to-end run and the script's refusals."""
import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase3_g3a import synth  # noqa: E402

from lookahead import phase3_g3a as G  # noqa: E402
from lookahead import phase3_readouts as R  # noqa: E402
from lookahead import x1_flips as X  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "x1_flip_check.py"


@pytest.fixture(scope="module")
def dev():
    return G.concat([synth(n=90, seed=10), synth(benches=G.PRIMARY, n=40, seed=11, source="devplus", prefix="p3:")])


# ------------------------------------------------------------------------------------------------------- matching
def test_match_saving_signature_and_no_access_to_correctness(dev):
    assert list(inspect.signature(X.match_saving).parameters) == ["S_rows", "ok_rows", "cap", "s"]
    assert list(inspect.signature(X.saving_curve).parameters) == ["S_rows", "ok_rows", "cap"]
    S = G.baseline_stat(dev, "native-pb")
    r = np.flatnonzero((dev.bench == "openbookqa") & (dev.fold == 0))
    before = [X.match_saving(S[r], dev.ok[r], 4, s) for s in X.LEVELS]
    e = G.subset(dev, np.ones(dev.N, bool))
    e.correct = e.correct[np.random.default_rng(0).permutation(dev.N)]       # correctness permuted: matching must not change
    assert not np.array_equal(e.correct, dev.correct)
    assert [X.match_saving(S[r], e.ok[r], 4, s) for s in X.LEVELS] == before
    assert any(b is not None for b in before)


def test_interpolated_saving_equals_s(dev):
    ok_fail = dev.ok.copy()
    ok_fail[::13] = False                                                    # failed rows exit at 4
    n_checked = 0
    for name in ("native-pb", "r-kl"):
        S = G.baseline_stat(dev, name)
        for b, cap in X.CAPS["primary"].items():
            r = np.flatnonzero((dev.bench == b) & (dev.fold == 1))
            for ok in (dev.ok[r], ok_fail[r]):
                for s in X.LEVELS:
                    m = X.match_saving(S[r], ok, cap, s)
                    if m is None:
                        continue
                    sv = lambda t: float(np.mean(1 - R.exit_loops(S[r], ok, t, cap) / 4))  # noqa: E731
                    assert (1 - m["w"]) * sv(m["tau1"]) + m["w"] * sv(m["tau2"]) == pytest.approx(s, abs=1e-12)
                    assert (1 - m["w"]) * m["saving1"] + m["w"] * m["saving2"] == pytest.approx(s, abs=1e-12)
                    assert m["saving1"] <= s + 1e-12 and m["saving2"] >= s - 1e-12 and 0 <= m["w"] <= 1
                    n_checked += 1
    assert n_checked > 20


def test_exact_hit_gives_w_zero_and_unattainable_gives_none():
    S = np.array([[0.9, 0, 0], [0.5, 0, 0], [0.1, 0, 0], [0.1, 0, 0]])
    ok = np.ones(4, bool)
    # cap 2 -> candidates 0.1, 0.5, 0.9, inf; savings 0.75 (all fire at 1), 0.625, 0.5625, 0.5
    taus, sav = X.saving_curve(S, ok, 2)
    assert taus.tolist() == [0.1, 0.5, 0.9, np.inf] and sav == pytest.approx([0.75, 0.625, 0.5625, 0.5])
    hit = X.match_saving(S, ok, 2, 0.625)
    assert hit["w"] == 0 and hit["tau1"] == hit["tau2"] == 0.5 and hit["saving1"] == hit["saving2"] == 0.625
    mid = X.match_saving(S, ok, 2, 0.6)
    assert (mid["tau1"], mid["tau2"]) == (0.9, 0.5) and mid["w"] == pytest.approx(0.6)
    assert X.match_saving(S, ok, 2, 0.8) is None and X.match_saving(S, ok, 2, 0.4) is None


# ------------------------------------------------------------------------------------------------------ flips
def test_gains_losses_phi_kappa_net_by_hand():
    S = np.array([[.9, 0, 0], [.9, 0, 0], [.1, .9, 0], [.1, .1, .9], [.1, .1, .1], [.9, 0, 0]])
    ok = np.array([1, 1, 1, 1, 1, 0], bool)
    correct = np.array([[1, 1, 1, 0], [0, 1, 1, 1], [0, 1, 0, 0], [1, 1, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0]])
    g, l = X.flip_vectors(S, ok, correct, 0.5, 4)           # exits 1,1,2,3,4,4(failed)
    assert g.tolist() == [1, 0, 1, 0, 0, 0] and l.tolist() == [0, 1, 0, 1, 0, 0]
    r = X.rates(g.mean(), l.mean())
    assert (r["g"], r["l"], r["phi"], r["kappa"], r["net"]) == pytest.approx((1 / 3, 1 / 3, 2 / 3, 2 / 3, 0.0))
    g, l = X.flip_vectors(S, ok, correct, np.inf, 3)        # fixed depth 3 (failed rows still exit at 4)
    assert g.tolist() == [1, 0, 0, 0, 0, 0] and l.tolist() == [0, 0, 0, 1, 1, 0]
    r = X.rates(g.mean(), l.mean())
    assert (r["phi"], r["kappa"], r["net"]) == pytest.approx((0.5, 1 / 3, 100 / 6))
    assert X.implied_margin(1 / 6, 1 / 3, 100) == pytest.approx(1.645 * np.sqrt((0.5 - (1 / 6) ** 2) / 100))
    gi, li = X.interpolated_flips(S, ok, correct, 4, {"tau1": 0.5, "tau2": np.inf, "w": 0.25})   # fixed depth 4 flips nothing
    assert gi.tolist() == pytest.approx([.75, 0, .75, 0, 0, 0]) and li.tolist() == pytest.approx([0, .75, 0, .75, 0, 0])


def _pool(rng, folds=range(5), benches=("b1", "b2")):
    sizes = {(b, k): 7 + k + 3 * i for i, b in enumerate(benches) for k in folds}
    cols = [("primary", m, s) for m in ("a", "b") for s in X.LEVELS]
    vecs = {key: {c: (rng.integers(0, 2, n) * 0.5, rng.integers(0, 2, n) * 1.0) for c in cols} for key, n in sizes.items()}
    ok = {(cs, m, b, s): True for (cs, m, s) in cols for b in benches}
    return X.Pool(cols, vecs, sizes, X.draw_counts(sizes, 40), ok), sizes, vecs


def test_pooled_rates_equal_per_question_rates_over_the_union_of_folds():
    pool, sizes, vecs = _pool(np.random.default_rng(1))
    c = ("primary", "a", 0.2)
    for b in ("b1", "b2"):
        gu = np.concatenate([vecs[(b, k)][c][0] for k in range(5)])
        lu = np.concatenate([vecs[(b, k)][c][1] for k in range(5)])
        g, l = pool.gl(b, *c)
        assert (g[0], l[0]) == pytest.approx((gu.mean(), lu.mean()), abs=1e-12)
        # phi/kappa/net are derived from the pooled g, l (not averaged over folds)
        r = X.rates(g[0], l[0])
        assert r["kappa"] == pytest.approx(2 * min(gu.mean(), lu.mean())) and r["net"] == pytest.approx(100 * (lu.mean() - gu.mean()))
    e = X.contrast(pool, "primary", "a", "b", ["b1", "b2"])["0.20"]
    pa, pb = [np.mean([X.rates(*pool.gl(b, "primary", m, 0.2))["phi"][0] for b in ("b1", "b2")]) for m in ("a", "b")]
    assert e["delta"]["phi"]["point"] == pytest.approx(pa - pb, abs=1e-12)


# ------------------------------------------------------------------------------------------------------ bootstrap
def test_bootstrap_is_reproducible_and_draws_one_vector_per_stratum_in_sorted_order(monkeypatch):
    sizes = {("hellaswag", 1): 5, ("commonsense_qa", 0): 4, ("commonsense_qa", 1): 6, ("hellaswag", 0): 3}
    a, b = X.draw_counts(sizes, 30), X.draw_counts(sizes, 30)
    assert all(np.array_equal(a[k], b[k]) for k in sizes)
    assert all((a[k].sum(axis=1) == n).all() and a[k].shape == (30, n) for k, n in sizes.items())
    calls, real = [], np.random.default_rng

    class Spy:
        def __init__(self, g):
            self.g = g

        def multinomial(self, n, p, *a, **k):
            calls.append((n, len(p)))
            return self.g.multinomial(n, p, *a, **k)

    seeds = []
    monkeypatch.setattr(X.np.random, "default_rng", lambda s: (seeds.append(s), Spy(real(s)))[1])
    c = X.draw_counts(sizes, 2)
    assert seeds == [[31, 1]] and calls == [(4, 4), (6, 6), (3, 3), (5, 5)] * 2     # sorted (benchmark, fold) order, per replicate
    assert all(np.array_equal(c[k], a[k][:2]) for k in sizes)                         # prefix of the same stream
    assert X.N_BOOT == 2000 and X.SEED == [31, 1]


def test_the_same_count_vector_is_used_for_both_methods():
    pool, sizes, _ = _pool(np.random.default_rng(2))
    same = X.contrast(pool, "primary", "a", "a", ["b1", "b2"])
    for e in same.values():
        assert all(v == {"point": 0.0, "lo": 0.0, "hi": 0.0} for v in e["delta"].values()) and e["per_fold_dphi"] == [0.0] * 5
    # contrast replicate = difference of the two methods' replicates under identical counts (independently recomputed)
    counts = X.draw_counts(sizes, 40)
    pool2 = X.Pool(pool.cols, {k: {} for k in sizes}, sizes, counts, pool.ok)
    assert pool2.acc["b1"].shape == (41, 2 * len(pool.cols))
    e = X.contrast(pool, "primary", "a", "b", ["b1", "b2"])["0.25"]
    ia, ib = pool.ci[("primary", "a", 0.25)], pool.ci[("primary", "b", 0.25)]
    C = len(pool.cols)
    boot = np.mean([(pool.acc[b][1:, ia] + pool.acc[b][1:, C + ia]) - (pool.acc[b][1:, ib] + pool.acc[b][1:, C + ib])
                    for b in ("b1", "b2")], axis=0)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    assert (e["delta"]["phi"]["lo"], e["delta"]["phi"]["hi"]) == pytest.approx((lo, hi), abs=1e-12)


# -------------------------------------------------------------------------------------------------------- readings
def _entry(phi=(-0.01, -0.005), kappa=(-0.01, -0.005), l=(-0.02, -0.01), g=(0.005, 0.01), folds=(-1,) * 5):
    ci = lambda t: {"point": t[0], "lo": t[0] - 1, "hi": t[1]}  # noqa: E731
    return {"delta": {"phi": ci(phi), "kappa": ci(kappa), "l": ci(l), "g": ci(g), "net": ci((0, 0))}, "per_fold_dphi": list(folds)}


def _res(entries):
    return {f"{s:.2f}": e for s, e in zip(X.CORE, entries)}


def test_reading_returns_each_of_the_four_readings():
    sup = X.reading(_res([_entry()] * 4))
    assert sup["reading"] == "FLIP-SUPPORTED" and sup["FK_levels"] == ["0.15", "0.20", "0.25", "0.30"]
    sup3 = X.reading(_res([_entry()] * 3 + [_entry(phi=(0.01, 0.02), kappa=(0.01, 0.02))]))
    assert sup3["reading"] == "FLIP-SUPPORTED"
    # F holds, K fails (kappa upper bound >= 0); L holds (losses explain it) -> row 2 with the losses sentence
    nk = _entry(kappa=(-0.01, 0.01))
    r2 = X.reading(_res([nk] * 4))
    assert r2["reading"] == "FLIP REDUCTION, CANCELLING NOT DEMONSTRATED" and r2["losses_sentence_allowed"]
    assert r2["L_levels"] == ["0.15", "0.20", "0.25", "0.30"]
    # same but losses do not dominate (|dl| <= |dg|) -> row 2 without the stronger sentence
    r2b = X.reading(_res([_entry(kappa=(-0.01, 0.01), l=(-0.01, -0.005), g=(0.02, 0.03))] * 4))
    assert r2b["reading"].startswith("FLIP REDUCTION") and not r2b["losses_sentence_allowed"]
    # L needs >= 3 of the F levels
    mix = [nk, nk, _entry(kappa=(-0.01, 0.01), l=(0.0, 0.01)), _entry(kappa=(-0.01, 0.01), l=(0.0, 0.01))]
    assert not X.reading(_res(mix))["losses_sentence_allowed"]
    bad = _entry(phi=(0.01, 0.02), kappa=(0.0, 0.01))
    ns = X.reading(_res([bad, bad, bad, _entry(phi=(-0.01, 0.005))]))
    assert ns["reading"] == "NOT SUPPORTED" and not ns["losses_sentence_allowed"]
    assert X.reading(_res([bad, bad, _entry(phi=(-0.01, 0.005)), _entry(phi=(-0.01, 0.005))]))["reading"] == "MIXED"
    assert X.reading(_res([_entry()] * 2 + [bad, bad]))["reading"] == "MIXED"       # F at only 2 levels, so neither row 2 nor 3
    assert X.reading(_res([_entry()] + [bad] * 3))["reading"] == "MIXED"           # F at one level blocks row 3
    # fold criterion: negative in only 3 of 5 folds -> F fails
    assert X.reading(_res([_entry(folds=(-1, -1, -1, 1, 1))] * 4))["reading"] == "MIXED"
    assert X.reading(_res([_entry(folds=(-1, -1, -1, -1, 1))] * 4))["reading"] == "FLIP-SUPPORTED"
    # a core level without included benchmarks fails every condition; non-core levels are ignored
    assert X.reading({**_res([_entry()] * 3 + [None]), "0.10": _entry()})["reading"] == "FLIP-SUPPORTED"
    assert X.reading(_res([_entry()] * 2 + [None, None]))["reading"] == "MIXED"
    assert X.reading({})["reading"] == "MIXED"


def test_reading_rows_are_mutually_exclusive():
    rng = np.random.default_rng(3)
    names = set()
    for _ in range(300):
        es = [_entry(phi=tuple(rng.choice([-0.02, -0.01, 0.01], 1)) + (rng.choice([-0.005, 0.005]),),
                     kappa=(-0.01, rng.choice([-0.005, 0.005])), l=(rng.choice([-0.02, 0.0]), rng.choice([-0.01, 0.01])),
                     folds=tuple(rng.choice([-1, 1], 5))) for _ in range(4)]
        r = X.reading(_res(es))
        names.add(r["reading"])
        F, FK, nonneg = set(r["F_levels"]), set(r["FK_levels"]), sum(e["delta"]["phi"]["point"] >= 0 for e in es)
        expect = ("FLIP-SUPPORTED" if len(FK) >= 3 else "FLIP REDUCTION, CANCELLING NOT DEMONSTRATED" if len(F) >= 3
                  else "NOT SUPPORTED" if not F and nonneg >= 3 else "MIXED")
        assert r["reading"] == expect
    assert len(names) >= 3


# -------------------------------------------------------------------------------------------------------- end to end
def test_run_x1_end_to_end_is_serialisable_and_deterministic(dev):
    rep = X.run_x1(dev, n_boot=60)
    json.dumps(rep, allow_nan=False)
    assert {"protocol", "status", "fold_sizes", "contrasts", "included_benchmarks", "reading", "six_benchmark_macro_note"} <= set(rep)
    assert rep["protocol"] == "docs/X1_FLIP_PROTOCOL.md rev 2" and rep["status"] == "exploratory"
    assert set(rep["contrasts"]) == {"primary", "sensitivity_cap4"} and set(rep["contrasts"]["primary"]) == set(X.CONTRASTS)
    assert all(sum(v) == rep["n_dev"][b] for b, v in rep["fold_sizes"].items()) and set(rep["fold_sizes"]) == set(G.PRIMARY)
    prim = rep["contrasts"]["primary"][X.PRIMARY_CONTRAST]
    assert rep["reading"] == prim["reading"] and prim["reading"]["governing"]
    assert not rep["contrasts"]["sensitivity_cap4"][X.PRIMARY_CONTRAST]["reading"]["governing"]
    mt = prim["matching"]["pi1-x"]["openbookqa"]["0.20"]["0"]
    assert set(mt) == {"tau1", "tau2", "saving1", "saving2", "w"}
    assert rep["included_benchmarks"]["primary"][X.PRIMARY_CONTRAST]["0.10"] != ["hellaswag"]
    assert "hellaswag" not in rep["included_benchmarks"]["primary"][X.PRIMARY_CONTRAST]["0.15"]     # cap 3 cannot reach 0.15
    e = prim["levels"]["0.25"]
    assert set(e["delta"]) == set(X.MEASURES) and len(e["per_fold_dphi"]) == 5 and e["delta"]["phi"]["lo"] <= e["delta"]["phi"]["hi"]
    assert json.dumps(X.run_x1(dev, n_boot=60), sort_keys=True) == json.dumps(rep, sort_keys=True)
    with pytest.raises(PermissionError):
        X.run_x1(G.concat([dev, synth(benches=("openbookqa",), n=5, seed=40, source="p1_validation", prefix="v/")]), n_boot=5)
    with pytest.raises(PermissionError):
        X.run_x1(G.concat([dev, synth(benches=("openbookqa",), n=5, seed=41, source="test", prefix="t/")]), n_boot=5)


def _script():
    spec = importlib.util.spec_from_file_location("x1_flip_check", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_synthetic_run_writes_exclusively_and_refuses_overwrite_and_test_paths(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = tmp_path / "x1.json"
    cmd = [sys.executable, str(SCRIPT), "--synthetic", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    rep = json.loads(out.read_text())
    code = rep["provenance"]["code"]
    assert rep["provenance"]["inputs"] == {"synthetic": True} and rep["n_boot"] == 2000 and "git_commit" in code
    assert {"protocol_sha256", "src/lookahead/x1_flips.py", "scripts/x1_flip_check.py", "src/lookahead/e1.py",
            "src/lookahead/phase3_g3a.py", "src/lookahead/phase3_readouts.py"} <= set(code) and len(rep["fold_fingerprints"]) == 5
    before = out.read_bytes()
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    assert r2.returncode == 2 and "refusing" in r2.stderr and out.read_bytes() == before
    bad = tmp_path / "data" / "extracted_phase3_test" / "x1.json"
    r3 = subprocess.run([sys.executable, str(SCRIPT), "--synthetic", str(bad)], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r3.returncode == 2 and "extracted_phase3_test" in r3.stderr and not bad.exists()


def test_test_paths_are_refused():
    mod = _script()
    for p in ("data/extracted_phase3_test/a.jsonl", "C:\\x\\extracted_phase3_test\\a", Path("results") / "extracted_phase3_test"):
        with pytest.raises(SystemExit) as ei:
            mod.refuse_sealed("results/ok.json", p)
        assert ei.value.code == 2
    mod.refuse_sealed("results/exploratory/x1_flips.json", "data/extracted_phase3_dev/a.jsonl")
    assert mod.REPORT == "results/exploratory/x1_flips.json"
    assert "extracted_phase3_test" not in (ROOT / "src/lookahead/x1_flips.py").read_text(encoding="utf-8")
    assert "open(" not in (ROOT / "src/lookahead/x1_flips.py").read_text(encoding="utf-8")


def test_provenance_values_with_a_sealed_path_are_refused():
    """The real-data path checks the serialised provenance (keys and values), so a sealed path in a VALUE is caught."""
    import json as _json
    mod = _script()
    prov = {"phase3_dev": "data/extracted_phase3_test/x.jsonl"}           # key is harmless, value is sealed
    with pytest.raises(SystemExit):
        mod.refuse_sealed(_json.dumps(prov))
    mod.refuse_sealed(_json.dumps({"phase3_dev": "data/extracted_phase3_dev/x.jsonl"}))
    assert "refuse_sealed(json.dumps(dev.provenance" in (ROOT / "scripts/x1_flip_check.py").read_text(encoding="utf-8")
