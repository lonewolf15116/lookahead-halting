"""The short-context diagnostic must report exact agreement in float32 on a tiny model (so any gap it later finds on
the laptop is numerical, not a bug in the diagnostic)."""
import importlib.util

import pytest

torch = pytest.importorskip("torch")
from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.questions import load_synthetic_questions  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
pytestmark = pytest.mark.skipif(not (LOCAL / "tokenizer.json").exists(), reason="run scripts/inspect_ouro.py first")


@pytest.fixture(scope="module")
def tiny_real_vocab():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, total_ut_steps=4,
        layer_types=["full_attention"] * 2, max_position_embeddings=2048,
    )
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()


def test_diagnostic_exact_in_float32(tiny_real_vocab, tok):
    spec = importlib.util.spec_from_file_location("diag", REPO_ROOT / "scripts" / "diagnose_short_context.py")
    diag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(diag)
    rows = diag.compare(tiny_real_vocab, load_synthetic_questions("parity", tok)[:3])
    assert len(rows) == 12
    s = diag.summarize(rows)
    for key in ("packed_vs_ctx", "ctxmask_vs_ctx", "packed_vs_ctxmask"):
        assert s[key]["maxabs_max"] <= 1e-5, (key, s[key])
