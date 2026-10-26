"""C1 freeze (development only): selections under the identical combined rule, pooled OOD selection checked by brute force,
expected precision, rho measurement on a tiny real-Ouro model (CPU), refusals and the synthetic script path."""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from lookahead import c1 as C1
from lookahead import e1 as E
from lookahead import gate1 as g1
from lookahead import phase3_g3a as G
from lookahead import phase3_readouts as R
from lookahead.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_phase3_g3a import synth  # noqa: E402


@pytest.fixture(scope="module")
def dev():
    return G.concat([synth(n=90, seed=10), synth(benches=G.PRIMARY, n=40, seed=11, source="devplus", prefix="p3:")])


@pytest.fixture(scope="module")
def frozen(dev):
    return C1.freeze(dev)


def test_primary_selections_use_the_identical_combined_rule_and_shared_draws(dev, frozen):
    fz, models = frozen
    allm = np.ones(dev.N, bool)
    boot = E.make_boot(dev, allm, C1.BOOT_FOLD_ID)
    assert fz["bootstrap_selection"]["fingerprint"] == E.boot_fingerprint(boot)
    S_pi, _ = C1.pi1x_statistic(dev)
    for name, S in (("pi1-x", S_pi), ("native-pb", G.baseline_stat(dev, "native-pb"))):
        exp = E.select_risk(dev, S, allm, False, boot)
        for b in G.PRIMARY:
            got = fz["primary"][name][b]
            assert got == exp[b]
            assert got["loss"] <= E.LIMIT and got["upper_bound"] <= E.LIMIT
    assert all(fz["reference_B_pb"][b]["tau"] == math.inf for b in G.PRIMARY)
    assert fz["delta"] == 0.02 and fz["n_boot_test"] == 10_000 and fz["test_seed"] == 2026


def test_pi1x_is_the_g3a_candidate_spec(frozen):
    fz, models = frozen
    assert fz["pi1x"]["features"] == ["conf", "ent", "vent", "cdf", "n_options", "log_n_tokens", "loop1", "loop2", "loop3"]
    assert fz["pi1x"]["C"] == 1.0 and fz["pi1x"]["target"] == "S" and len(models) == 3


def _brute_pooled(d, S, boot):
    allm = np.ones(d.N, bool)
    taus = R.grid([S], [d.ok])
    best = []
    for tau in taus:
        for m in (1, 2, 3, 4):
            per, ok_all = {}, True
            for b in d.benchmarks:
                r = E._rows(d, b, allm)
                e = R.exit_loops(S[r], d.ok[r], tau, m)
                sav, loss = R.outcome(d.correct[r], d.ok[r], e)
                l = 100.0 * (np.where(d.ok[r], d.correct[r, 3], 0) - np.where(d.ok[r], d.correct[r][np.arange(len(r)), e - 1], 0))
                ub = float(np.quantile(boot[b] @ l / len(r), E.RISK_QUANTILE))
                ok_all &= loss <= E.LIMIT and ub <= E.LIMIT
                per[b] = (sav, loss)
            if ok_all:
                best.append((float(np.mean([v[0] for v in per.values()])), float(np.mean([v[1] for v in per.values()])), tau, m))
    top = max(x[0] for x in best)
    c = [x for x in best if x[0] >= top - g1.TOL]
    lo = min(x[1] for x in c)
    c = [x for x in c if x[1] <= lo + g1.TOL]
    c.sort(key=lambda x: (0 if math.isinf(x[2]) else 1, -x[2] if math.isfinite(x[2]) else 0, x[3]))
    return c[0]


@pytest.mark.parametrize("name", ["native-pb", "r-conf"])
def test_pooled_ood_selection_matches_brute_force(dev, name):
    boot = E.make_boot(dev, np.ones(dev.N, bool), C1.BOOT_FOLD_ID)
    S = G.baseline_stat(dev, name)
    got = C1.select_risk_pooled(dev, S, boot)
    sav, loss, tau, m = _brute_pooled(dev, S, boot)
    assert (got["tau"], got["m"]) == (tau, m) and got["macro_saving"] == pytest.approx(sav)
    assert all(v <= E.LIMIT for v in got["upper_bounds"].values())


def test_pooled_fixed_depth_4_always_qualifies(dev):
    boot = E.make_boot(dev, np.ones(dev.N, bool), C1.BOOT_FOLD_ID)
    got = C1.select_risk_pooled(dev, np.zeros((dev.N, 3)), boot, fixed_only=True)
    assert got["tau"] == math.inf and got["m"] in (1, 2, 3, 4) and got["macro_loss"] <= E.LIMIT


def test_expected_precision_is_finite_and_shrinks_with_n(frozen):
    ep = frozen[0]["expected_precision"]
    assert 0 < ep["expected_half_width_95"] < 1 and ep["test_n"] == C1.TEST_N


def test_freeze_is_deterministic_and_refuses_validation_rows(dev, frozen):
    a = json.dumps(frozen[0], sort_keys=True, default=str)
    assert json.dumps(C1.freeze(dev)[0], sort_keys=True, default=str) == a
    bad = G.concat([dev, synth(benches=("openbookqa",), n=5, seed=3, source="p1_validation", prefix="v/")])
    with pytest.raises(PermissionError):
        C1.freeze(bad)


def test_module_has_no_test_access():
    src = open(C1.__file__, encoding="utf-8").read()
    assert "extracted_phase3_test" not in src and "questions_phase3_test" not in src and "load_source" not in src


def test_rho_measurement_runs_on_a_tiny_real_ouro_model():
    import torch

    from lookahead.config import load_config
    from lookahead.questions import Question

    local = REPO_ROOT / load_config("model")["local_dir"]
    if not (local / "modeling_ouro.py").exists():
        pytest.skip("Ouro modeling code not present")
    from transformers import AutoConfig, AutoModelForCausalLM

    from lookahead.c1_rho import measure_rho

    cfg = AutoConfig.from_pretrained(str(local), trust_remote_code=True)
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=4, head_dim=16, vocab_size=500, max_window_layers=2).items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval()
    rng = np.random.default_rng(0)
    qs = [Question("openbookqa", "openbookqa", f"q{i}", "c", [" a", " b"], ["aa", "bbb"], 0, "acc_norm",
                   context_enc=list(rng.integers(1, 400, 8)), cont_encs=[[5, 6], [7, 8, 9]]) for i in range(12)]
    models = [({"mean": np.zeros(9), "sd": np.ones(9)}, g1_lin()) for _ in range(3)]
    out = measure_rho(model, {"openbookqa": qs}, models, n_per_bench=6)
    v = out["openbookqa"]
    assert v["n_questions"] == 6 and v["loop_ms"] > 0
    assert all(math.isfinite(v[k]) and v[k] > 0 for k in ("rho_score", "rho_rest"))
    assert v["rho_total"] == pytest.approx(v["rho_score"] + v["rho_rest"])


def g1_lin():
    from lookahead.gate2 import LinearModel

    return LinearModel(np.ones(9) * 0.1, 0.0)


def test_freeze_script_synthetic_path_writes_and_refuses_to_overwrite(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")}
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "c1_freeze.py"), "--synthetic", str(tmp_path / "c1")]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert r.returncode == 0, r.stderr
    fz = json.loads((tmp_path / "c1" / "frozen.json").read_text(encoding="utf-8"))
    assert fz["pi1x"]["model_file_sha256"] == g1.sha256_file(tmp_path / "c1" / "pi1x_models.npz")
    r2 = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO_ROOT)
    assert r2.returncode == 2 and "refusing" in r2.stderr


# ------------------------------------------------------------------------------ reviewer C1 implementation review fixes

def _reviewer_hypothetical(qual):
    """The flawed logic the reviewer described: minimum loss over ALL qualifiers, before restricting to top saving."""
    lo = min(c["loss"] for c in qual)
    tie = [c for c in qual if c["loss"] <= lo + g1.TOL]
    tie.sort(key=lambda c: (0 if math.isinf(c["tau"]) else 1, -c["tau"] if math.isfinite(c["tau"]) else 0, c["m"]))
    return tie[0]


def test_tie_break_restricts_to_maximum_saving_before_loss():
    qual = [{"tau": 0.5, "m": 3, "saving": 0.40, "loss": 0.9},     # the optimum
            {"tau": 0.9, "m": 4, "saving": 0.10, "loss": 0.0},     # safest, lower saving
            {"tau": math.inf, "m": 4, "saving": 0.0, "loss": 0.0}]
    assert C1.tie_break(qual)["saving"] == 0.40
    assert _reviewer_hypothetical(qual)["saving"] != 0.40          # the regression this test guards against
    tied = [{"tau": 0.5, "m": 3, "saving": 0.40, "loss": 0.9}, {"tau": 0.7, "m": 3, "saving": 0.40, "loss": 0.2},
            {"tau": math.inf, "m": 2, "saving": 0.40, "loss": 0.2}]
    assert C1.tie_break(tied)["tau"] == math.inf                    # among top saving: smaller loss, then fixed depth
    with pytest.raises(ValueError):
        C1.tie_break([])


def test_pooled_selection_uses_the_explicit_tie_break(monkeypatch, dev):
    calls = []
    real = C1.tie_break
    monkeypatch.setattr(C1, "tie_break", lambda q: calls.append(q) or real(q))
    boot = E.make_boot(dev, np.ones(dev.N, bool), C1.BOOT_FOLD_ID)
    C1.select_risk_pooled(dev, G.baseline_stat(dev, "native-pb"), boot)
    assert len(calls) == 1


def test_net_saving_charges_option_scoring_except_at_the_halting_loop():
    # q1 halts by score at loop 2: evaluations at 1, 2; scoring paid at 1 only (loop 2's scoring answers). q2 hits cap 3:
    # evaluations at 1, 2; scoring paid at both; the answer's scoring at loop 3 is shared with every controller.
    got = C1.net_saving([2, 3], [True, False], 3, rho_score=0.1, rho_rest=0.01)
    cost = (2 + 0.1 * 1 + 0.01 * 2) + (3 + 0.1 * 2 + 0.01 * 2)
    assert got == pytest.approx(1 - cost / 2 / 4)
    assert C1.net_saving([2, 3], [True, False], 3, 0.0, 0.0) == pytest.approx(1 - 2.5 / 4)


def test_probe_features_match_extraction_exactly():
    """Option lengths and features: c1_rho._features == extract.option_record (acc_norm divides by len(choice_text))."""
    from lookahead.c1_rho import _features
    from lookahead.extract import option_record
    from lookahead.questions import Question

    for metric in ("acc_norm", "acc"):
        q = Question("b", "b", "b/0", "ctx", [" x", " yy", " z"], ["a", "bbbb", "cc"], 1, metric,
                     context_enc=[1, 2, 3], cont_encs=[[4], [5, 6, 7], [8, 9]])   # token lengths differ from char lengths
        ll = np.array([-3.0, -5.5, -2.5])
        rec = option_record(q, ll.tolist())
        x = _features(ll, np.array([float(len(c)) for c in q.choice_text]), metric, 1.25, 0.3, 3, q.packed_len, 1)
        assert x[0] == pytest.approx(rec["conf"]) and x[1] == pytest.approx(rec["ent"])
        assert x[2] == np.float32(1.25) and x[3] == np.float32(0.3) and x[4] == 3
        assert x[5] == pytest.approx(math.log(q.packed_len)) and list(x[6:]) == [0, 1, 0]


def test_vent_reuses_the_scoring_rows_last_context_distribution_as_extraction_does():
    import torch

    from lookahead.config import load_config

    local = REPO_ROOT / load_config("model")["local_dir"]
    if not (local / "modeling_ouro.py").exists():
        pytest.skip("Ouro modeling code not present")
    from transformers import AutoConfig, AutoModelForCausalLM

    from lookahead.packed import PackedScorer, build_packed

    cfg = AutoConfig.from_pretrained(str(local), trust_remote_code=True)
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=4, head_dim=16, vocab_size=500, max_window_layers=2).items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval()
    packed = build_packed([5, 17, 99, 3], [[7, 8], [42, 43, 44]])
    assert packed.score_pos[0] == packed.n_context - 1
    sc = PackedScorer(model)
    pos = torch.tensor(packed.score_pos)
    with torch.no_grad():
        for out in sc.loops(packed, n_loops=1):
            pass
        state = sc.start(packed)
        sc.runner.step(state)
        logp = torch.log_softmax(sc.runner.readout(state.hidden, pos[None])[0], dim=-1)
    assert torch.allclose(logp[0], out.vocab_logp)   # the same distribution extraction stores for vent
