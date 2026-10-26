"""Phase 3 A2 readout baselines: signs, indexing, loop-1 behaviour, grid, selection, nested comparator (synthetic only)."""
import math

import numpy as np
import pytest

from lookahead import phase3_readouts as R
from lookahead.extract import kl_logp, option_record
from lookahead.questions import Question


def q(metric="acc_norm", target=0, choice_text=("a", "bbbb", "cc")):
    return Question("b", "b", "b/0", "ctx", [" x"] * len(choice_text), list(choice_text), target, metric)


LL = [[-3.0, -2.0, -1.0], [-1.0, -2.5, -2.0], [-0.5, -3.0, -2.0], [-0.4, -3.2, -2.2]]  # option log-likelihoods, loops 1-4


def stored(metric):
    """What the extraction stores per loop for these option log-likelihoods (the real code path)."""
    return [option_record(q(metric), ll) for ll in LL]


@pytest.mark.parametrize("metric", ["acc_norm", "acc"])
def test_signs_and_indexing_match_the_stored_records(metric):
    recs = stored(metric)
    z = np.array([r["opt_lln"] if metric == "acc_norm" else r["opt_ll"] for r in recs])
    for t in range(3):  # loops 1..3
        assert R.question_stats("r-conf", z)[t] == pytest.approx(recs[t]["conf"])  # stored conf = max p_t
        assert R.question_stats("r-ent", z)[t] == pytest.approx(-recs[t]["ent"])  # stored ent = H(p_t) >= 0
        s = np.sort(z[t])[::-1]
        assert R.question_stats("r-margin", z)[t] == pytest.approx(s[0] - s[1])  # logit (score) gap, not probability gap
    kl = R.question_stats("r-kl", z)
    assert math.isnan(kl[0])  # loop 1: undefined
    for t in (1, 2):
        # paper's direction KL(p_t || p_{t-1}); the stored `D` of loop t-1 is the other direction KL(p_{t-1} || p_t)
        assert kl[t] == pytest.approx(-kl_logp(recs[t]["logp"], recs[t - 1]["logp"]))
        stored_D_prev = kl_logp(recs[t - 1]["logp"], recs[t]["logp"])
        assert kl[t] != pytest.approx(-stored_D_prev)  # guards against silently using the stored direction
    dl = R.question_stats("r-dlogit", z)
    assert math.isnan(dl[0]) and dl[2] == pytest.approx(-np.linalg.norm(z[2] - z[1]))


def test_acc_and_acc_norm_use_their_own_scores():
    za = np.array([r["opt_ll"] for r in stored("acc")])
    zn = np.array([r["opt_lln"] for r in stored("acc_norm")])
    assert not np.allclose(R.question_stats("r-margin", za), R.question_stats("r-margin", zn))


def test_hidden_readouts_follow_ouro_and_are_undefined_at_loop_1():
    h = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 2.1]], dtype=np.float16)
    z = np.zeros((4, 3))
    l2, cs = R.question_stats("r-hl2", z, h), R.question_stats("r-hcos", z, h)
    assert math.isnan(l2[0]) and math.isnan(cs[0])
    assert l2[1] == pytest.approx(-math.sqrt(2)) and l2[2] == pytest.approx(-1.0)  # -||h_t - h_{t-1}||
    assert cs[1] == pytest.approx(0.0) and cs[2] == pytest.approx(1.0)  # cos(h_t, h_{t-1})
    zero = np.zeros((4, 2))
    assert math.isnan(R.question_stats("r-hcos", z, zero)[1])  # zero vector: undefined, never fires


def test_unknown_readout_is_rejected():
    with pytest.raises(ValueError):
        R.question_stats("r-nope", np.zeros((4, 2)))


# ---------------------------------------------------------------- the frozen exit rule

def test_exit_rule_loop1_cap_and_failed_questions():
    S = np.array([[np.nan, 0.9, 0.1],   # convergence-style: undefined at loop 1
                  [0.95, 0.1, 0.1],     # fires at loop 1
                  [0.1, 0.1, 0.1],      # never fires
                  [0.95, 0.95, 0.95]])  # failed question
    ok = np.array([True, True, True, False])
    assert R.exit_loops(S, ok, 0.5, 4).tolist() == [2, 1, 4, 4]   # undefined loop 1 -> continue; failed -> 4
    assert R.exit_loops(S, ok, 0.5, 1).tolist() == [1, 1, 1, 4]   # cap 1: stop at loop 1 whatever the score
    assert R.exit_loops(S, ok, 0.5, 2).tolist() == [2, 1, 2, 4]   # loop 2 is the cap: stop there regardless
    assert R.exit_loops(S, ok, 0.95, 3).tolist() == [3, 1, 3, 4]  # threshold applies at loops 1-2 only below cap 3
    assert R.exit_loops(S, ok, math.inf, 3).tolist() == [3, 3, 3, 4]  # tau = inf is fixed depth
    with pytest.raises(ValueError):
        R.exit_loops(S, ok, 0.5, 5)


def test_outcome_counts_failed_questions_wrong():
    correct = np.array([[1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1]])
    ok = np.array([True, True, False])
    sav, loss = R.outcome(correct, ok, np.array([1, 1, 4]))
    assert sav == pytest.approx(1 - 6 / 12) and loss == pytest.approx(100 * (2 / 3 - 1 / 3))


# ---------------------------------------------------------------- grid and selection

def test_grid_pools_ok_defined_values_dedups_and_appends_inf():
    S = np.array([[np.nan, 0.2, 0.2], [0.5, 0.5, np.nan]])
    g = R.grid([S], [np.array([True, True])])
    assert math.isinf(g[-1]) and all(math.isfinite(x) for x in g[:-1])
    assert len(set(g)) == len(g) and min(g[:-1]) == 0.2 and max(g[:-1]) == 0.5
    with pytest.raises(ValueError):
        R.grid([np.full((2, 3), np.nan)], [np.array([True, True])])


def test_selection_prefers_fixed_depth_on_ties_and_respects_the_budget():
    n = 200
    ok = np.ones(n, dtype=bool)
    correct = np.ones((n, 4), dtype=int)  # always correct: every policy has zero loss
    S = np.full((n, 3), 0.5)
    sel = R.select(S, ok, correct, [0.4, math.inf])
    # tau 0.4 fires at loop 1 = fixed depth 1 in effect (saving 0.75); fixed depth 1 ties and is preferred
    assert sel["m"] == 1 and math.isinf(sel["tau"]) and sel["saving"] == pytest.approx(0.75)
    correct2 = correct.copy()
    correct2[:10, :3] = 0  # 5% of questions are only right at loop 4
    S2 = np.full((n, 3), 0.5)
    S2[:10] = 0.0  # ...and the readout flags them as not converged
    sel2 = R.select(S2, ok, correct2, [0.4, math.inf])
    assert sel2["loss"] <= 1.0 and sel2["tau"] == 0.4 and sel2["saving"] > 0.7  # adaptive beats every fixed depth here


def test_nested_comparator_chooses_on_selection_folds_not_on_held_out():
    folds = [{"B_pb": {"eligible": True, "selection_saving": 0.40, "heldout_saving": 0.35},
              "r-hl2": {"eligible": True, "selection_saving": 0.45, "heldout_saving": 0.30},
              "r-conf": {"eligible": False, "selection_saving": 0.60, "heldout_saving": 0.60}},
             {"B_pb": {"eligible": True, "selection_saving": 0.42, "heldout_saving": 0.41},
              "r-hl2": {"eligible": True, "selection_saving": 0.41, "heldout_saving": 0.50}}]
    out = R.nested_comparator(folds)
    assert out["chosen"] == ["r-hl2", "B_pb"]  # ineligible r-conf ignored; held-out values never drive the choice
    assert out["heldout_saving"] == pytest.approx((0.30 + 0.41) / 2)
    with pytest.raises(ValueError):
        R.nested_comparator([{"x": {"eligible": False, "selection_saving": 1, "heldout_saving": 1}}])


def test_readout_list_is_closed_and_matches_the_amendment():
    assert R.READOUTS == ("r-conf", "r-margin", "r-ent", "r-kl", "r-dlogit", "r-hl2", "r-hcos")
    assert R.CONVERGENCE == {"r-kl", "r-dlogit", "r-hl2", "r-hcos"}
