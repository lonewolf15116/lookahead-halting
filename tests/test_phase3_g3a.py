"""Phase 3 G3a (D1-D4): synthetic data only. Oracles (checked against brute force), stop-rule boundaries, the A2 comparator,
TEST refusal, duplicates and missing data, a loader fixture on disk, and an end-to-end synthetic dry run."""
import itertools
import json
import math

import numpy as np
import pytest

from lookahead import gate1 as g1
from lookahead import gate2 as g2
from lookahead import phase3_g3a as G
from lookahead import phase3_readouts as R

SIX = g1.GATE1_BENCHMARKS


def _lab(a, b):
    return np.where(a == 1, "C", "W").astype(object) + np.where(b == 1, "C", "W").astype(object)


def synth(benches=SIX, n=60, d=8, seed=0, source="p1_train", prefix="", fail=0):
    """Plausible fake development data: accuracy rises with loops, predictions consistent with option scores."""
    rng = np.random.default_rng(seed)
    parts = []
    for bi, b in enumerate(benches):
        K = 4
        target = rng.integers(0, K, n)
        skill = rng.normal(size=n)
        pc = 1 / (1 + np.exp(-(skill[:, None] + np.array([-1.0, 0.0, 0.4, 0.5]))))
        correct = (rng.random((n, 4)) < pc).astype(int)
        pred = np.where(correct == 1, target[:, None], (target[:, None] + 1 + rng.integers(0, K - 1, (n, 4))) % K)
        z = []
        for i in range(n):
            zi = rng.normal(size=(4, K))
            zi[np.arange(4), pred[i]] += 3.0
            z.append(zi)
        cdf = np.cumsum(rng.random((n, 4)), axis=1)
        cdf = cdf / cdf[:, [3]]
        T = _lab(correct[:, :3], correct[:, 1:]).astype(str)
        Tfin = _lab(correct[:, :3], np.repeat(correct[:, [3]], 3, axis=1)).astype(str)
        x = rng.normal(size=(n, 3, 9)).astype(np.float32)
        x[:, :, 0] += correct[:, :3]
        h = rng.normal(size=(n, 3, d)).astype(np.float32)
        h[:, :, 0] += 2 * (pred[:, :3] == pred[:, [3]])
        ok = np.ones(n, dtype=bool)
        if fail:
            ok[:fail] = False
            correct[:fail] = 0
            cdf[:fail] = -np.inf
        parts.append(G.DevData(bench=np.array([b] * n), doc_id=np.array([f"{prefix}{b}/{i}" for i in range(n)]),
                               source=np.array([source] * n), ok=ok, correct=correct, pred=pred, cdf=cdf, T=T, Tfin=Tfin,
                               x=x, h=h, z=z))
    return G.concat(parts)


# ---------------------------------------------------------------------------------------------------- oracles

def _brute_budget(correct):
    n = len(correct)
    K = correct[:, 3].sum()
    best = None
    for e in itertools.product(range(1, 5), repeat=n):
        e = np.array(e)
        if correct[np.arange(n), e - 1].sum() >= K - 0.01 * n - 1e-12:
            if best is None or e.sum() < best:
                best = e.sum()
    return best


@pytest.mark.parametrize("seed", range(25))
def test_accuracy_budget_oracle_is_exactly_optimal_against_brute_force(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 7))
    correct = (rng.random((n, 4)) < rng.random()).astype(int)
    e = G.accuracy_budget_oracle(correct, np.ones(n, bool), np.array([f"q{i}" for i in range(n)]))
    assert e.sum() == _brute_budget(correct)
    assert correct[np.arange(n), e - 1].sum() >= correct[:, 3].sum() - 0.01 * n - 1e-12


def test_accuracy_budget_oracle_keeps_wrong_early_answers_within_budget_and_breaks_ties_by_doc_id():
    # 200 questions: 100 correct everywhere; 2 wrong at loop 1 but correct from loop 2 (ids b, a); rest wrong everywhere
    correct = np.zeros((200, 4), int)
    correct[:100] = 1
    correct[100] = [0, 1, 1, 1]
    correct[101] = [0, 1, 1, 1]
    ids = np.array([f"z{i:03d}" for i in range(200)])
    ids[100], ids[101] = "b", "a"
    e = G.accuracy_budget_oracle(correct, np.ones(200, bool), ids)
    # K = 102, budget allows 2 fewer: k = 100 already >= 102 - 2 -> everyone exits at loop 1 (keeps 2 wrong answers)
    assert (e == 1).all()
    correct[102] = [0, 0, 1, 1]  # K = 103, need 101: buy the cheapest one -> cost 1 (loop 2), tie a < b by doc id
    e = G.accuracy_budget_oracle(correct, np.ones(200, bool), ids)
    assert e[101] == 2 and e[100] == 1 and e[102] == 1 and e.sum() == 201


def test_failed_questions_exit_at_loop_4_in_both_oracles():
    correct = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])
    ok = np.array([True, False])
    assert G.accuracy_budget_oracle(correct, ok, np.array(["a", "b"])).tolist() == [1, 4]
    assert G.stable_final_oracle(np.array([[2, 2, 2, 2], [1, 1, 1, 1]]), ok).tolist() == [1, 4]


def test_stable_final_oracle_matches_gate1():
    d = synth(benches=("openbookqa",), n=80, seed=3)
    arrays = {"n": d.N, "ok": d.ok, "pred": d.pred, "correct": d.correct}
    e_g1, _ = g1.run_policy(arrays, "stable_oracle")
    assert np.array_equal(G.stable_final_oracle(d.pred, d.ok), e_g1)


# ------------------------------------------------------------------------------------------------- stop rules

def test_d1_stop_boundary():
    assert G.d1_stop(0.0499999) and not G.d1_stop(0.05) and not G.d1_stop(0.2) and G.D1_MIN == 0.05


def test_d2_stop_boundary_and_either_candidate_continues():
    assert G.D2_MIN == 0.02
    assert G.d2_stop([0.0199, 0.0199]) and not G.d2_stop([0.02, -0.5]) and not G.d2_stop([-0.1, 0.03])
    with pytest.raises(ValueError):
        G.d2_stop([])
    with pytest.raises(ValueError):
        G.d2_stop([0.01, float("nan")])


def test_d3_keep_rule_boundary():
    cell = lambda counted, better: {"counted": counted, "h_better": better}  # noqa: E731
    assert G.keep_hx([cell(True, True), cell(True, False)])            # exactly half
    assert not G.keep_hx([cell(True, True), cell(True, False), cell(True, False)])
    assert not G.keep_hx([cell(False, True)])                          # nothing counted -> not kept (OP6)
    assert G.D3_MIN_TRANSITIONS == 30


# --------------------------------------------------------------------------------------------- comparator, D1, D2

def test_d2_comparator_is_chosen_on_selection_folds_and_improvement_is_the_mean_paired_difference():
    d = synth(n=120, seed=1)
    stats = {name: G.baseline_stat(d, name) for name in G.BASELINES}
    cands = G.candidate_stats_crossfit(d)
    out = G.d2(d, stats, cands)
    per = out["per_fold_detail"]
    for k, chosen in enumerate(out["comparator_per_fold"]):
        sel = {b: per[k][b]["selection_saving"] for b in G.BASELINES if per[k][b]["eligible"]}
        assert chosen == sorted(sel, key=lambda b: (-sel[b], b))[0]       # selection folds only, ties by name
    for c, v in out["candidates"].items():
        manual = np.mean([per[k][c]["heldout_saving"] - per[k][out["comparator_per_fold"][k]]["heldout_saving"]
                          for k in range(5)])
        assert v["improvement_vs_comparator"] == pytest.approx(manual)
        assert v["improvement_vs_best_on_heldout_conservative"] <= v["improvement_vs_comparator"] + 1e-12
    assert out["stop"] == G.d2_stop([v["improvement_vs_comparator"] for v in out["candidates"].values()])


def test_every_selected_point_respects_the_budget_on_selection_data_and_B_pb_is_fixed_depth():
    d = synth(n=100, seed=2)
    allrows = np.ones(d.N, bool)
    for name in ("B_pb", "r-conf", "native-pb", "r-hl2"):
        sel = G.select_all(d, G.baseline_stat(d, name), allrows, fixed_only=(name == "B_pb"))
        for b, s in sel.items():
            assert s["loss"] <= g1.LOSS_LIMIT + g1.TOL
            if name == "B_pb":
                assert math.isinf(s["tau"])


def test_d1_report_and_nonbinding_quantity():
    d = synth(n=150, seed=4)
    stats = {name: G.baseline_stat(d, name) for name in G.BASELINES}
    out = G.d1(d, stats)
    p = out["primary"]
    assert p["headroom_ii"] == pytest.approx(p["ceiling_ii"] - p["B_pb"])
    assert p["ceiling_ii"] >= p["B_pb"] - 1e-12      # the oracle can always do at least as well as fixed depth
    for b in G.PRIMARY:
        assert out["per_benchmark"][b]["oracle_ii"]["loss"] <= 1.0 + 1e-9
    nb = out["nonbinding_oracle_minus_best_baseline"]
    assert nb["best_baseline"] in G.BASELINES and nb["delta"] == pytest.approx(p["ceiling_ii"] - nb["best_baseline_saving"])
    assert out["stop"] == G.d1_stop(p["headroom_ii"])


def test_readout_statistics_are_the_frozen_ones():
    d = synth(benches=("openbookqa",), n=10, seed=5)
    S = G.baseline_stat(d, "r-kl")
    for i in range(d.N):
        assert np.allclose(S[i], R.question_stats("r-kl", d.z[i]), equal_nan=True)
    assert np.isnan(S[:, 0]).all()                       # undefined at loop 1


# ------------------------------------------------------------------------------------------- refusals / integrity

def test_duplicate_questions_are_refused():
    a = synth(benches=("openbookqa",), n=5, seed=6)
    with pytest.raises(ValueError, match="duplicate"):
        G.concat([a, a])


def test_missing_primary_benchmark_is_refused():
    d = synth(benches=("arc_easy", "openbookqa", "hellaswag"), n=40)
    with pytest.raises(ValueError, match="commonsense_qa"):
        G.run_g3a(d)


def test_validation_or_test_rows_in_development_data_are_refused():
    d = G.concat([synth(n=30, seed=7), synth(benches=("openbookqa",), n=5, seed=8, source="p1_validation", prefix="v/")])
    with pytest.raises(PermissionError):
        G.run_g3a(d)


def test_failed_questions_are_never_halted_early():
    d = synth(n=60, seed=9, fail=3)
    S = G.baseline_stat(d, "r-conf")
    assert np.isnan(S[~d.ok]).all()
    e = R.exit_loops(S, d.ok, -1e9, 2)                   # threshold that fires everywhere it is defined
    assert (e[~d.ok] == 4).all()


# ------------------------------------------------------------------------------------------- loader fixture (disk)

def _write_fixture(root, dirname, bench, split, ids, d=4, metric="acc_norm"):
    import pandas as pd

    bdir = root / "data" / dirname / bench
    bdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for q in ids:
        for t in range(1, 5):
            rows.append({"benchmark": bench, "doc_id": q, "split": split, "status": "ok", "loop": t, "pred": 0,
                         "correct": 1, "conf": 0.9, "ent": 0.3, "cdf": t / 4, "D": 0.01 if t < 4 else float("nan"),
                         "ms": 1.0, "T": "CC" if t < 4 else "", "T_final": "CC" if t < 4 else "", "vent": 1.0,
                         "G": 0.0, "n_options": 2, "n_tokens": 10, "stratum": bench, "opt_ll": [-1.0, -2.0],
                         "opt_lln": [-0.5, -1.0], "metric": metric})
    stem = bdir / "part-00000"
    pd.DataFrame(rows).to_parquet(str(stem) + ".parquet", index=False)
    np.save(str(stem) + ".hidden.npy", np.ones((len(ids), 4, d), np.float16))
    done = {"parquet_sha256": g1.sha256_file(str(stem) + ".parquet"), "hidden_sha256": g1.sha256_file(str(stem) + ".hidden.npy"),
            "doc_ids": list(ids), "statuses": ["ok"] * len(ids), "n": len(ids)}
    (bdir / "part-00000.done.json").write_text(json.dumps(done), encoding="utf-8")


def test_loader_reads_a_fixture_and_picks_the_metric_scores(tmp_path):
    _write_fixture(tmp_path, "extracted_phase3_dev", "openbookqa", "dev", ["p3:a", "p3:b"])
    d = G.load_source(tmp_path, "extracted_phase3_dev", "dev", "devplus", ["openbookqa"], allow_ids={"p3:a", "p3:b"},
                      deny_ids={"p3:zz"}, expected_n={"openbookqa": 2})
    assert d.N == 2 and d.h.shape == (2, 3, 4) and np.allclose(d.z[0][0], [-0.5, -1.0])   # acc_norm -> opt_lln


def test_loader_refuses_test_ids_ids_outside_the_allow_list_and_count_mismatch(tmp_path):
    _write_fixture(tmp_path, "extracted_phase3_dev", "openbookqa", "dev", ["p3:a", "p3:t"])
    with pytest.raises(PermissionError, match="TEST"):
        G.load_source(tmp_path, "extracted_phase3_dev", "dev", "devplus", ["openbookqa"], allow_ids={"p3:a", "p3:t"},
                      deny_ids={"p3:t"})
    with pytest.raises(PermissionError, match="allow-list"):
        G.load_source(tmp_path, "extracted_phase3_dev", "dev", "devplus", ["openbookqa"], allow_ids={"p3:a"})
    with pytest.raises(ValueError, match="manifest"):
        G.load_source(tmp_path, "extracted_phase3_dev", "dev", "devplus", ["openbookqa"], allow_ids={"p3:a", "p3:t"},
                      expected_n={"openbookqa": 3})


def test_loader_refuses_sealed_directories_and_missing_parts(tmp_path):
    _write_fixture(tmp_path, "extracted_phase3_test", "openbookqa", "test", ["p3:x"])
    with pytest.raises(PermissionError):
        G.load_source(tmp_path, "extracted_phase3_test", "test", "x", ["openbookqa"])
    with pytest.raises(PermissionError):
        G.load_source(tmp_path, "extracted_confirm", "train", "x", ["openbookqa"])
    (tmp_path / "data" / "extracted_phase3_dev" / "hellaswag").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        G.load_source(tmp_path, "extracted_phase3_dev", "dev", "devplus", ["hellaswag"])


def test_loader_detects_a_tampered_part(tmp_path):
    _write_fixture(tmp_path, "extracted_phase3_dev", "openbookqa", "dev", ["p3:a"])
    np.save(str(tmp_path / "data/extracted_phase3_dev/openbookqa/part-00000.hidden.npy"), np.zeros((1, 4, 4), np.float16))
    with pytest.raises(ValueError, match="checksum"):
        G.load_source(tmp_path, "extracted_phase3_dev", "dev", "devplus", ["openbookqa"])


def test_module_never_names_the_sealed_extraction():
    src = (G.__file__ and open(G.__file__, encoding="utf-8").read())
    assert "extracted_phase3_test" not in src and "questions_phase3_test" not in src


# --------------------------------------------------------------------------------------------- synthetic dry run

def test_end_to_end_synthetic_dry_run():
    dev = G.concat([synth(n=90, seed=10), synth(benches=G.PRIMARY, n=40, seed=11, source="devplus", prefix="p3:")])
    val = synth(n=50, seed=12, source="p1_validation", prefix="v/")
    rep = G.run_g3a(dev, val)
    assert set(rep) >= {"D1", "D2", "D3", "D4", "D2_validation_exposed", "decision", "open_points", "keep_hx_variants"}
    assert rep["decision"].startswith(("STOP", "CONTINUE"))
    assert rep["n_by_source"] == {"devplus": 120, "p1_train": 540}
    assert len(rep["D2"]["comparator_per_fold"]) == 5
    assert set(rep["D2"]["candidates"]) == set(G.CANDIDATES)
    assert all(c["kind"] in ("adjacent", "to_final") for c in rep["D3"]["cells"])
    json.dumps(rep, default=str)   # serialisable
    assert G.run_g3a(dev, val)["D2"]["candidates"] == rep["D2"]["candidates"]   # deterministic
