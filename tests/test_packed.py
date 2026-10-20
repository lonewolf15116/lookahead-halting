"""Packed scoring must equal per-option scoring (plan section 4, tests P1-P4), on a tiny float32 Ouro."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.packed import PackedScorer, build_packed, reference_mask, score_per_option  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
pytestmark = pytest.mark.skipif(not (LOCAL / "modeling_ouro.py").exists(), reason="run scripts/inspect_ouro.py first")
LOOPS, VOCAB = 3, 200
IMPLS = ["sdpa", "eager"]


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, vocab_size=VOCAB, total_ut_steps=LOOPS,
        layer_types=["full_attention"] * 3, max_position_embeddings=256,
    )
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()
    with torch.no_grad():
        model.model.early_exit_gate.weight.normal_(0, 1.0)
    return model


def use_impl(model, impl):
    model.config._attn_implementation = impl


def rand_tokens(gen, n):
    return torch.randint(0, VOCAB, (n,), generator=gen).tolist()


def question(seed, n_ctx=13, lens=(3, 1, 5, 2)):
    gen = torch.Generator().manual_seed(seed)
    return rand_tokens(gen, n_ctx), [rand_tokens(gen, m) for m in lens]


def packed_run(model, ctx, conts):
    outs = list(PackedScorer(model).loops(build_packed(ctx, conts)))
    return [o.option_ll for o in outs], [o.option_greedy for o in outs], [o.hidden for o in outs], outs


# P3: layout, positions, no padding
def test_layout_positions_and_segments():
    p = build_packed([10, 11, 12], [[20, 21, 22], [30], [40, 41]])
    assert p.input_ids == [10, 11, 12, 20, 21, 40]
    assert p.position_ids == [0, 1, 2, 3, 4, 3]
    assert p.segment == [0, 0, 0, 1, 1, 3]
    # option 0 tokens predicted at: last ctx, its 1st token, its 2nd token; option 1 at last ctx; option 2 at last ctx, its 1st
    assert p.score_pos == [2, 3, 4, 2, 2, 5]
    assert p.score_tgt == [20, 21, 22, 30, 40, 41]
    assert p.score_opt == [0, 0, 0, 1, 2, 2]
    assert len(p.input_ids) == len(p.position_ids) == len(p.segment)  # one row, nothing padded


def test_letter_options_add_no_tokens():
    p = build_packed([1, 2, 3, 4], [[5], [6], [7], [8]])
    assert p.input_ids == [1, 2, 3, 4] and p.score_pos == [3, 3, 3, 3]


@pytest.mark.parametrize("impl", IMPLS)
def test_mask_equals_reference(tiny, impl):
    use_impl(tiny, impl)
    ctx, conts = question(1)
    p = build_packed(ctx, conts)
    state = PackedScorer(tiny).start(p)
    m = state.masks["full_attention"]
    assert m is not None, "mask was skipped: the custom pattern would be lost"
    allowed = m[0, 0] if m.dtype == torch.bool else (m[0, 0] == 0)
    assert torch.equal(allowed.cpu(), reference_mask(p.segment))
    assert state.position_ids[0].tolist() == p.position_ids


# P1 + P4: equal to per-option scoring, every loop
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("seed,lens", [(2, (3, 1, 5, 2)), (3, (1, 1, 1, 1, 1)), (4, (7, 6)), (5, (2, 9, 1))])
def test_packed_equals_per_option(tiny, impl, seed, lens):
    use_impl(tiny, impl)
    ctx, conts = question(seed, lens=lens)
    ref_ll, ref_greedy, ref_h = score_per_option(tiny, ctx, conts)
    ll, greedy, h, _ = packed_run(tiny, ctx, conts)
    for t in range(LOOPS):
        assert max(abs(a - b) for a, b in zip(ll[t], ref_ll[t])) <= 1e-4, f"loop {t + 1}"
        assert greedy[t] == ref_greedy[t]
        assert (h[t] - ref_h[t]).abs().max() <= 1e-5


@pytest.mark.parametrize("impl", IMPLS)
def test_shared_hidden_equals_context_only_pass(tiny, impl):
    from lookahead.ouro_adapter import OuroLoopRunner

    use_impl(tiny, impl)
    ctx, conts = question(6)
    runner = OuroLoopRunner(tiny)
    state = runner.start(torch.tensor([ctx]))
    _, _, h, outs = packed_run(tiny, ctx, conts)
    for t in range(LOOPS):
        runner.step(state)
        assert (h[t] - state.hidden[0, -1]).abs().max() <= 1e-5
        assert abs(outs[t].gate_logit - float(runner.gate_logit(state.hidden, torch.tensor([len(ctx) - 1])))) <= 1e-5
        full = torch.log_softmax(runner.readout(state.hidden, torch.tensor([len(ctx) - 1]))[0], -1)
        assert (outs[t].vocab_logp - full).abs().max() <= 1e-4


# P2: changing one option leaves the others untouched
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("new_len", [5, 2, 8])  # same length as before, shorter, longer
def test_branch_isolation(tiny, impl, new_len):
    use_impl(tiny, impl)
    ctx, conts = question(7, lens=(3, 5, 4))
    ll_a, _, h_a, _ = packed_run(tiny, ctx, conts)
    gen = torch.Generator().manual_seed(99)
    changed = [conts[0], rand_tokens(gen, new_len), conts[2]]
    ll_b, _, h_b, _ = packed_run(tiny, ctx, changed)
    for t in range(LOOPS):
        assert abs(ll_a[t][0] - ll_b[t][0]) <= 1e-5 and abs(ll_a[t][2] - ll_b[t][2]) <= 1e-5
        assert (h_a[t] - h_b[t]).abs().max() <= 1e-5
