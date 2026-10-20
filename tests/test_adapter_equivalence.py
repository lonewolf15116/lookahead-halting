"""The loop-by-loop runner must reproduce Ouro's own forward (tiny random model built from the real Ouro code)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.ouro_adapter import NativeGate, OuroLoopRunner, exit_distribution  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
needs_code = pytest.mark.skipif(not (LOCAL / "modeling_ouro.py").exists(), reason="run scripts/inspect_ouro.py first")

LOOPS = 3


def stock_pdf(gate_list):
    """Ouro's exit distribution, transcribed from OuroForCausalLM.forward."""
    remaining = torch.ones_like(gate_list[0].squeeze(-1))
    pdf = []
    for i, gate in enumerate(gate_list):
        lam = torch.sigmoid(gate.squeeze(-1))
        if i < len(gate_list) - 1:
            p = lam * remaining
            remaining = remaining * (1.0 - lam)
        else:
            p = remaining
        pdf.append(p)
    return torch.stack(pdf, dim=2)


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, vocab_size=200, total_ut_steps=LOOPS,
        layer_types=["full_attention"] * 3, max_position_embeddings=256,
    )
    model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()
    with torch.no_grad():  # a freshly initialised gate sits near zero; spread it so exit probabilities differ
        model.model.early_exit_gate.weight.normal_(0, 1.0)
    return model


def test_exit_distribution_known_values():
    pdf = exit_distribution(torch.zeros(1, 4))  # lambda = 0.5 at every loop
    assert pdf.squeeze().tolist() == pytest.approx([0.5, 0.25, 0.125, 0.125])


def test_exit_distribution_is_a_distribution():
    pdf = exit_distribution(torch.randn(6, 4))
    assert (pdf >= 0).all()
    assert torch.allclose(pdf.sum(-1), torch.ones(6), atol=1e-6)


def test_streaming_gate_matches_batch_form():
    gates = torch.randn(7, 4)
    cdf = exit_distribution(gates).cumsum(-1)
    native = NativeGate(total_loops=4, batch=7)
    for t in range(4):
        assert torch.allclose(native.update(gates[:, t]), cdf[:, t], atol=1e-6)


@needs_code
def test_loop_states_and_gates_match_stock(tiny):
    ids = torch.randint(0, 200, (2, 11))
    with torch.no_grad():
        _, hidden_list, gate_list = tiny.model(input_ids=ids, use_cache=False)
    runner = OuroLoopRunner(tiny)
    state = runner.start(ids)
    for k in range(LOOPS):
        runner.step(state)
        assert torch.allclose(state.hidden, hidden_list[k], atol=1e-6), f"hidden state differs at loop {k + 1}"
        assert torch.allclose(runner.core.early_exit_gate(state.hidden), gate_list[k], atol=1e-6)


@needs_code
def test_exit_distribution_matches_stock_formula(tiny):
    ids = torch.randint(0, 200, (2, 11))
    with torch.no_grad():
        _, _, gate_list = tiny.model(input_ids=ids, use_cache=False)
    ours = exit_distribution(torch.cat(gate_list, dim=-1))
    assert torch.allclose(ours, stock_pdf(gate_list), atol=1e-6)


@needs_code
def test_logits_match_stock_exit_at_step(tiny):
    ids = torch.randint(0, 200, (2, 11))
    runner = OuroLoopRunner(tiny)
    logits, _ = runner.all_loops(ids, torch.full((2,), ids.shape[1] - 1))
    for k in range(LOOPS):
        with torch.no_grad():
            stock = tiny(input_ids=ids, use_cache=False, exit_at_step=k).logits[:, -1].float()
        assert torch.allclose(logits[k], stock, atol=1e-5), f"logits differ at loop {k + 1}"


@needs_code
def test_right_padding_does_not_change_the_answer_position(tiny):
    a = torch.randint(0, 200, (1, 9))
    b = torch.randint(0, 200, (1, 5))
    padded = torch.zeros(2, 9, dtype=torch.long)
    padded[0] = a[0]
    padded[1, :5] = b[0]
    mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4])
    runner = OuroLoopRunner(tiny)
    batched, _ = runner.all_loops(padded, torch.tensor([8, 4]), attention_mask=mask)
    alone_a, _ = runner.all_loops(a, torch.tensor([8]))
    alone_b, _ = runner.all_loops(b, torch.tensor([4]))
    assert torch.allclose(batched[:, 0], alone_a[:, 0], atol=1e-5)
    assert torch.allclose(batched[:, 1], alone_b[:, 0], atol=1e-5)
