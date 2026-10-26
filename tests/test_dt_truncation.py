"""D-t truncated-execution check (docs/DT_TRUNCATION_PLAN.md): CPU, tiny random Ouro, stored data written by the real extraction
writer and read back by the real G3a loader, plus synthetic DevData for the model-free adapter checks. No real data, no GPU."""
import hashlib
import importlib.util
import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("pyarrow")
pytest.importorskip("pandas")

from lookahead import c1_test as T  # noqa: E402
from lookahead import dt_truncation as D  # noqa: E402
from lookahead import gate2 as g2  # noqa: E402
from lookahead import phase3_g3a as G  # noqa: E402
from lookahead import phase3_readouts as R  # noqa: E402
from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.extract import run_benchmark  # noqa: E402
from lookahead.packed import PackedScorer  # noqa: E402
from lookahead.questions import Question  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "tests"))
from test_phase3_g3a import synth  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
pytestmark = pytest.mark.skipif(not (LOCAL / "modeling_ouro.py").exists(), reason="run scripts/inspect_ouro.py first")
VOCAB, N_Q = 200, 8
METRIC = {"openbookqa": "acc_norm", "commonsense_qa": "acc", "hellaswag": "acc_norm"}

_spec = importlib.util.spec_from_file_location("dt_truncation_check", REPO_ROOT / "scripts" / "dt_truncation_check.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)


# ------------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=4, head_dim=16, vocab_size=VOCAB, total_ut_steps=4, layer_types=["full_attention"] * 2,
        max_position_embeddings=256)
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        model.model.early_exit_gate.weight.normal_(0, 1.0)
    return model


def make_questions(bench, n, seed):
    gen = torch.Generator().manual_seed(seed)
    qs = []
    for k in range(n):
        ctx = torch.randint(0, VOCAB, (int(torch.randint(5, 14, (1,), generator=gen)),), generator=gen).tolist()
        lens = [int(x) for x in torch.randint(1, 5, (3 + k % 2,), generator=gen)]
        conts = [torch.randint(0, VOCAB, (m,), generator=gen).tolist() for m in lens]
        qs.append(Question(benchmark=bench, subtask="s", doc_id=f"p3:{bench}/dev@r{k}", context="c", continuations=["x"] * len(lens),
                           choice_text=["a" * (j + 1 + k % 3) for j in range(len(lens))], target=k % len(lens), metric=METRIC[bench],
                           context_enc=ctx, cont_encs=conts))
    return qs


@pytest.fixture(scope="module")
def env(tiny, tmp_path_factory):
    """Mini repo with a DEV+ extraction written by the real writer, loaded by the real loader; fake-but-valid frozen policies."""
    root = tmp_path_factory.mktemp("dtroot")
    scorer = PackedScorer(tiny)
    questions = {b: make_questions(b, N_Q, seed=i) for i, b in enumerate(G.PRIMARY)}
    for b, qs in questions.items():
        splits = {(b, q.doc_id): ("s", "dev") for q in qs}
        done = run_benchmark(scorer, b, qs, splits, root / "data" / G.DEVPLUS_DIR, chunk=5, hidden_size=64, log_every=0)
        assert done == (N_Q, True)
    dev = G.load_source(root, G.DEVPLUS_DIR, "dev", "devplus", G.PRIMARY)
    assert dev.ok.all() and dev.N == 3 * N_Q
    rng = np.random.default_rng(3)
    models = [({"mean": g2.fit_scaler(dev.x[:, t])["mean"], "sd": g2.fit_scaler(dev.x[:, t])["sd"]},
               g2.LinearModel(rng.normal(size=9) * 0.6, 0.1)) for t in range(3)]
    S = {p: D.policy_stat(p, dev, models) for p in D.POLICIES}
    frozen = {"primary": {}, "secondary_risk_all_baselines": {}}
    for p, (sect, key) in D.FROZEN_SELECTION.items():
        sel = {}
        for bi, b in enumerate(G.PRIMARY):
            col = np.nanmedian(S[p][dev.bench == b][:, 1])
            sel[b] = {"tau": "inf" if (p == "r-kl" and b == "hellaswag") else float(col), "m": (4, 3, 3)[bi]}
        frozen[sect][key] = sel
    return {"root": root, "scorer": scorer, "questions": questions, "dev": dev, "models": models, "frozen": frozen, "tiny": tiny}


@pytest.fixture(scope="module")
def report(env):
    return D.run_check(env["root"], env["dev"], env["questions"], env["models"], env["frozen"], env["scorer"], n_per_bench=6)


def full_trace(env, q, bench):
    tr, _ = D.run_full(env["scorer"], q, bench, q.doc_id, lambda: None, __import__("time").perf_counter)
    return tr


# ------------------------------------------------------------------------------------------------ selection
def test_selection_is_the_lowest_salted_sha256_among_ok_devplus_only():
    d = synth(benches=G.PRIMARY, n=40, seed=1, source="devplus", prefix="p3:", fail=5)
    p1 = synth(benches=G.PRIMARY, n=10, seed=2, source="p1_train")
    both = G.concat([p1, d])
    ids = D.select_ids(both, n=7)
    for b in G.PRIMARY:
        ok = [i for i, o, bb, s in zip(d.doc_id, d.ok, d.bench, d.source) if bb == b and o]
        want = sorted(ok, key=lambda i: hashlib.sha256(("lh-dt-v1:" + i).encode()).hexdigest())[:7]
        assert ids[b] == want and len(set(ids[b])) == 7
        assert all(i.startswith("p3:") for i in ids[b])                  # never a Phase 1 row
        assert not set(ids[b]) & set(d.doc_id[~d.ok].tolist())           # never a failed one
    assert D.select_ids(both, n=7) == ids                                # fixed
    with pytest.raises(ValueError):
        D.select_ids(d, n=40)                                            # only 35 ok per benchmark
    with pytest.raises(PermissionError):
        D.select_ids(synth(benches=G.PRIMARY, n=10, source="test"), n=2)


def test_questions_must_match_the_stored_traces(env):
    dev, qs = env["dev"], env["questions"]
    ids = D.select_ids(dev, n=4)
    sub = G.subset(dev, np.isin(dev.doc_id, [i for v in ids.values() for i in v]))
    m = D.match_questions(sub, ids, qs)
    assert len(m) == 12
    with pytest.raises(AssertionError, match="expected exactly 1"):                       # id without a question
        D.match_questions(sub, ids, {b: v[1:] for b, v in qs.items()})
    wrong = {b: [q for q in v] for b, v in qs.items()}
    q0 = wrong["openbookqa"][0]
    wrong["openbookqa"][0] = Question(**{**q0.__dict__, "cont_encs": q0.cont_encs + [[1, 2]]})
    sel0 = {"openbookqa": [q0.doc_id]}
    sub0 = G.subset(dev, np.isin(dev.doc_id, [q0.doc_id]))
    with pytest.raises(AssertionError, match="does not match the stored trace"):          # different question than extracted
        D.match_questions(sub0, sel0, wrong)


# ---------------------------------------------------------------------------- per-loop values == the extraction's
def test_fresh_per_loop_values_equal_extract_question_and_the_stored_trace(env):
    from lookahead.extract import extract_question

    q = env["questions"]["commonsense_qa"][2]
    rows, hid, status = extract_question(env["scorer"], q)
    tr = full_trace(env, q, "commonsense_qa")
    assert status == "ok" and not tr.failed
    for r, row in zip(tr.rows, rows):
        for k in ("opt_ll", "opt_lln", "pred", "correct", "conf", "ent", "vent", "gate", "cdf"):
            assert r[k] == row[k], k
    raw = D.load_stored_trace(env["root"], "commonsense_qa", [q.doc_id])[q.doc_id]
    assert max(D.fresh_vs_stored(tr, raw).values()) == 0.0               # CPU float32: identical to the stored values
    # and the single-question DevData is the stored row, element for element
    k = int(np.flatnonzero(env["dev"].doc_id == q.doc_id)[0])
    d4, st = tr.devdata(4), env["dev"]
    assert np.array_equal(d4.x, st.x[k:k + 1]) and np.array_equal(d4.cdf, st.cdf[k:k + 1])
    assert np.array_equal(d4.z[0], st.z[k]) and np.array_equal(d4.h, st.h[k:k + 1])


# ------------------------------------------------------------------------- statistic with NaN future == stored full
@pytest.mark.parametrize("policy", D.POLICIES)
def test_nan_future_statistic_equals_stored_full_statistic_at_every_loop(env, policy):
    dev, models = env["dev"], env["models"]
    for k in range(dev.N):
        single = D.policy_stat(policy, G.subset(dev, np.arange(dev.N) == k), models)[0]
        for t in (1, 2, 3, 4):
            kk = min(t, 3)
            masked = D.policy_stat(policy, D.stored_prefix(dev, k, t), models)[0]
            assert D.bits_equal(masked[:kk], single[:kk], policy), (k, t)
            assert np.isnan(masked[t:]).all()                                  # the future really is undefined


@pytest.mark.parametrize("policy", D.POLICIES)
def test_online_view_of_fresh_loops_equals_stored_full_statistic(env, policy):
    dev, models = env["dev"], env["models"]
    for b in G.PRIMARY:
        q = env["questions"][b][1]
        tr = full_trace(env, q, b)
        k = int(np.flatnonzero(dev.doc_id == q.doc_id)[0])
        single = D.policy_stat(policy, G.subset(dev, np.arange(dev.N) == k), models)[0]
        fresh_full = D.policy_stat(policy, tr.devdata(4), models)[0]
        for t in (1, 2, 3, 4):
            online = D.policy_stat(policy, tr.devdata(t), models)[0]
            kk = min(t, 3)
            assert D.bits_equal(online[:kk], single[:kk], policy)
            assert D.bits_equal(online[:kk], fresh_full[:kk], policy)


def test_p2_comparison_rejects_nans_and_one_ulp_differences():
    a = np.array([0.25, 0.5, 0.75])
    assert D.bits_equal(a, a.copy(), "pi1-x")
    assert not D.bits_equal(a, np.nextafter(a, 1.0), "pi1-x")
    b = a.copy()
    b[1] = np.nan
    assert not D.bits_equal(a, b, "pi1-x") and not D.bits_equal(b, b, "pi1-x")     # a NaN is a failure
    r = np.array([np.nan, -0.1, -0.2])
    assert D.bits_equal(r, r.copy(), "r-kl")                                       # r-kl is undefined at loop 1 by definition
    assert not D.bits_equal(r, np.array([np.nan, -0.1, np.nan]), "r-kl")
    assert not D.bits_equal(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3]), "pi1-x")


# --------------------------------------------- the online decision equals C1's apply_policy (model-free, synthetic)
def test_prefix_decision_reproduces_c1_apply_policy_on_synthetic_full_arrays():
    fz = json.loads((REPO_ROOT / "results/c1/frozen.json").read_text(encoding="utf-8"))
    with np.load(REPO_ROOT / "results/c1/pi1x_models.npz") as npz:
        models = T.load_models(npz)
    sel = D.frozen_selections(fz)
    d = synth(benches=G.PRIMARY, n=80, seed=5, source="devplus", prefix="p3:", fail=4)
    for p in D.POLICIES:
        S = D.policy_stat(p, d, models)
        pol = T.apply_policy(d, S, sel[p], G.PRIMARY)
        for b in G.PRIMARY:
            rows = np.flatnonzero(d.bench == b)
            e_online = np.array([D.exit_by_prefixes(p, d, int(i), models, sel[p][b]) for i in rows])
            assert np.array_equal(e_online, pol[b]["e"]), (p, b)
            assert np.array_equal(np.where(d.ok[rows], d.correct[rows, e_online - 1], 0), pol[b]["c"])
        assert len({m["m"] for m in pol.values()}) >= 1


def test_frozen_selections_read_the_frozen_file_and_pins_match_the_repo():
    fz = json.loads((REPO_ROOT / "results/c1/frozen.json").read_text(encoding="utf-8"))
    sel = D.frozen_selections(fz)
    assert set(sel) == set(D.POLICIES) and all(set(v) == set(G.PRIMARY) for v in sel.values())
    assert sel["r-kl"]["openbookqa"]["m"] == fz["secondary_risk_all_baselines"]["r-kl"]["openbookqa"]["m"]
    assert sel["pi1-x"]["hellaswag"]["tau"] == fz["primary"]["pi1-x"]["hellaswag"]["tau"]
    D.check_pinned(REPO_ROOT)                                                       # the pinned hashes are the repo's files


# ------------------------------------------------------------------------------------- real truncation
class Counting(PackedScorer):
    def __init__(self, model):
        super().__init__(model)
        self.yielded = self.started = self.closed = 0

    def loops(self, packed, n_loops=None):
        self.started += 1
        try:
            for out in super().loops(packed, n_loops):
                self.yielded += 1
                yield out
        finally:
            self.closed += 1


@pytest.mark.parametrize("tau,m,expect", [(-1.0, 4, 1), (float("inf"), 2, 2), (float("inf"), 3, 3), (float("inf"), 4, 4),
                                           (-1.0, 1, 1)])
def test_truncated_run_really_stops_and_closes_the_generator(env, tau, m, expect):
    sc = Counting(env["tiny"])
    q = env["questions"]["hellaswag"][0]
    out = D.run_truncated(sc, q, "hellaswag", q.doc_id, "native-pb", {"tau": tau, "m": m}, env["models"], lambda: None,
                          __import__("time").perf_counter)
    assert out["e"] == expect and out["loops_run"] == expect
    assert sc.yielded == expect and sc.started == 1 and sc.closed == 1               # no loop beyond the exit was computed


def test_truncated_run_matches_the_stored_correctness_at_the_exit_loop(env):
    dev = env["dev"]
    for p in D.POLICIES:
        for b in G.PRIMARY:
            sel_b = D.frozen_selections(env["frozen"])[p][b]
            for q in env["questions"][b][:4]:
                out = D.run_truncated(env["scorer"], q, b, q.doc_id, p, sel_b, env["models"], lambda: None,
                                      __import__("time").perf_counter)
                k = int(np.flatnonzero(dev.doc_id == q.doc_id)[0])
                assert out["correct"] == dev.correct[k, out["e"] - 1] and out["pred"] == dev.pred[k, out["e"] - 1]


# --------------------------------------------------------------------------- the whole check, end to end
def test_report_p1_p2_p3_and_structure(env, report):
    v = D.verdict(report)
    assert v == {"P1_pass": True, "P2_pass": True, "P2_code_level_causality_pass": True}, report["policies"]
    for p in D.POLICIES:
        for b in (*G.PRIMARY, "all"):
            r = report["policies"][p][b]
            assert r["n"] == (6 if b != "all" else 18)
            assert r["exit_agreement_pct"] == 100.0 and r["correctness_agreement_pct"] == 100.0 and r["prediction_agreement_pct"] == 100.0 and r["disagreements"] == []
            assert r["P2"]["checks"] >= r["n"] and r["P2"]["fail_vs_stored"] == 0
            assert r["stat_max_abs_diff_batch_vs_single"] < 1e-12            # batch shape may move the last bit (BLAS), nothing more
            assert r["P2"]["fail_vs_stored_batch"] == 0 or p == "pi1-x"
            assert r["loops_run_equals_exit"] == r["n"]                               # truncated runs executed exactly e loops
            assert abs(r["simulated_gross_saving"] - r["simulated_gross_saving_retrospective"]) < 1e-12
            assert r["total_full_ms"] > 0 and np.isfinite(r["measured_time_saving"])
    assert all(max(report["p3"][b].values()) == 0.0 for b in report["p3"])           # CPU float32: fresh == stored
    ids = [i for b in G.PRIMARY for i in report["selected_ids"][b]]
    assert len(ids) == 18 and report["salt"] == "lh-dt-v1:"
    assert any(report["policies"][p]["all"]["exit_hist_truncated"][k] > 0 for p in D.POLICIES for k in ("1", "2", "3"))   # real early exits
    json.dumps(report, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def test_a_disagreement_is_reported_in_full_not_hidden(env):
    dev = env["dev"]
    bad = G.concat([dev])
    k = int(np.flatnonzero(bad.doc_id == D.select_ids(dev, 6)["hellaswag"][0])[0])
    bad.cdf = bad.cdf.copy()
    bad.cdf[k] = bad.cdf[k][::-1]                       # a corrupted stored trace -> the stored decision differs from the fresh run
    bad.x = bad.x.copy()
    bad.x[k, :, 3] = bad.cdf[k, :3]
    rep = D.run_check(env["root"], bad, env["questions"], env["models"], env["frozen"], env["scorer"], n_per_bench=6)
    assert not D.verdict(rep)["P2_pass"]                                              # NaN-future vs corrupted stored arrays fails
    assert rep["policies"]["native-pb"]["all"]["P2"]["fail_vs_stored"] > 0
    assert rep["policies"]["native-pb"]["all"]["P2"]["pass_code_level_causality"]     # the code itself is still causal


# ------------------------------------------------------------------------------------------------ guards
def test_refuses_any_path_under_the_sealed_test_extraction(tmp_path):
    sealed = tmp_path / "data" / "extracted_phase3_test"
    sealed.mkdir(parents=True)
    for p in (sealed, sealed / "openbookqa" / "part-00000.parquet", tmp_path / "data" / "x" / ".." / "extracted_phase3_test" / "a",
              "C:\\Users\\x\\Projects\\lookahead-halting\\data\\extracted_phase3_test\\hellaswag", tmp_path / "data" / "confirm"):
        with pytest.raises(PermissionError):
            D.refuse_sealed(tmp_path, p)
    D.refuse_sealed(tmp_path, tmp_path / "data" / "extracted_phase3_dev", tmp_path / D.REPORT)       # allowed
    with pytest.raises(PermissionError):
        D.load_stored_trace(tmp_path, "openbookqa", ["x"], dirname="extracted_phase3_test")


def _mini_root(tmp_path, env, frozen=None):
    (tmp_path / "results" / "c1").mkdir(parents=True)
    fz = tmp_path / "results/c1/frozen.json"
    fz.write_text(json.dumps(frozen or env["frozen"]), encoding="utf-8")
    arrs = {}
    for t, (sc, m) in enumerate(env["models"]):
        arrs[f"loop{t + 1}_mean"], arrs[f"loop{t + 1}_sd"] = sc["mean"], sc["sd"]
        arrs[f"loop{t + 1}_coef"], arrs[f"loop{t + 1}_intercept"] = m.coef, np.array(m.intercept)
    with open(tmp_path / "results/c1/pi1x_models.npz", "wb") as f:
        np.savez(f, **arrs)
    return {"results/c1/frozen.json": hashlib.sha256(fz.read_bytes()).hexdigest(),
            "results/c1/pi1x_models.npz": hashlib.sha256((tmp_path / "results/c1/pi1x_models.npz").read_bytes()).hexdigest()}


def _stubs(env, called):
    def loader(root):
        called.append("loader")
        return env["dev"], env["questions"], {"stub": True}

    def scorer_factory(device):
        called.append("scorer")
        return env["scorer"], env["tiny"]

    return loader, scorer_factory


def test_hash_mismatch_is_refused_before_anything_is_read(tmp_path, env):
    pinned = _mini_root(tmp_path, env)
    called = []
    loader, sf = _stubs(env, called)
    bad = dict(pinned, **{"results/c1/frozen.json": "0" * 64})
    assert RUN.main(["--device", "cpu", "--n-per-bench", "6"], root=tmp_path, loader=loader, scorer_factory=sf, pinned=bad) == 2
    assert called == [] and not (tmp_path / D.REPORT).exists()
    with pytest.raises(SystemExit):
        D.check_pinned(tmp_path, bad)
    (tmp_path / "results/c1/pi1x_models.npz").write_bytes(b"tampered")                 # a changed model file is refused too
    assert RUN.main(["--device", "cpu"], root=tmp_path, loader=loader, scorer_factory=sf, pinned=pinned) == 2
    assert called == []


def test_main_refuses_a_report_path_under_the_sealed_test_extraction(tmp_path, env):
    pinned = _mini_root(tmp_path, env)
    called = []
    loader, sf = _stubs(env, called)
    rel = "data/extracted_phase3_test/dt_truncation_report.json"
    assert RUN.main(["--device", "cpu"], root=tmp_path, loader=loader, scorer_factory=sf, pinned=pinned, report_rel=rel) == 2
    assert called == [] and not (tmp_path / rel).exists()


def test_main_writes_the_report_once_and_never_overwrites(tmp_path, env, capsys):
    import shutil

    shutil.copytree(env["root"] / "data", tmp_path / "data")                              # the stored DEV+ extraction
    pinned = _mini_root(tmp_path, env)
    called = []
    loader, sf = _stubs(env, called)
    args = ["--device", "cpu", "--n-per-bench", "6"]
    assert RUN.main(args, root=tmp_path, loader=loader, scorer_factory=sf, pinned=pinned) == 0
    path = tmp_path / D.REPORT
    rep = json.loads(path.read_text(encoding="utf-8"))
    assert rep["verdict"] == {"P1_pass": True, "P2_pass": True, "P2_code_level_causality_pass": True}
    assert rep["plan_conformant_n"] is False and rep["provenance"]["pinned"] == pinned      # n=6 is recorded as a deviation
    assert called == ["loader", "scorer"]
    before = path.read_bytes()
    called.clear()
    assert RUN.main(args, root=tmp_path, loader=loader, scorer_factory=sf, pinned=pinned) == 2     # exists -> refuse
    assert path.read_bytes() == before and called == []
    with pytest.raises(FileExistsError):                                                    # and the open mode itself is exclusive
        open(path, "x")
    assert "P-1" in capsys.readouterr().out
