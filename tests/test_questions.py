"""Questions built for packed scoring must be the ones lm_eval scores, and packed scores must give lm_eval's metrics."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("lm_eval")

from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.lm_eval_ouro import LoopedOuroLM  # noqa: E402
from lookahead.packed import PackedScorer, build_packed  # noqa: E402
from lookahead.questions import load_lm_eval_questions, load_synthetic_questions, predictions  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
pytestmark = pytest.mark.skipif(not (LOCAL / "tokenizer.json").exists(), reason="run scripts/inspect_ouro.py first")
LOOPS = 2


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, total_ut_steps=LOOPS,
        layer_types=["full_attention"] * 2, max_position_embeddings=1024,
    )
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()


def test_questions_and_packed_metrics_match_lm_eval(tiny, tok, fake_task):
    import hashlib

    import lm_eval

    spec = {"shots": 3, "metric": "acc_norm"}
    qs = load_lm_eval_questions("fake_arc", LoopedOuroLM(None, tok), task_manager=fake_task, spec=spec)
    assert len(qs) == 8 and all(len(q.cont_encs) == 4 for q in qs)

    lm = LoopedOuroLM(tiny, tok)
    scorer = PackedScorer(tiny)
    for loop in range(1, LOOPS + 1):
        lm.exit_loop = loop
        res = lm_eval.simple_evaluate(model=lm, tasks=["fake_arc"], num_fewshot=3, task_manager=fake_task,
                                      batch_size=1, log_samples=True)
        samples = {s["doc_id"]: s for s in res["samples"]["fake_arc"]}
        for q in qs:
            # same prompts as lm_eval: every (context, continuation) key is in the harness run's cache
            for c in q.continuations:
                assert hashlib.sha1(f"{q.context}\x00{c}".encode("utf-8")).hexdigest() in lm.cache
            out = list(scorer.loops(build_packed(q.context_enc, q.cont_encs), n_loops=loop))[-1]
            s = samples[int(q.doc_id.split("/")[1])]
            ref = [r[0][0] for r in s["resps"]]
            assert max(abs(a - b) for a, b in zip(out.option_ll, ref)) <= 1e-4
            p = predictions(q, out.option_ll)
            assert s["target"] == q.target
            assert float(p["pred_acc"] == q.target) == s["acc"]
            assert float(p["pred_acc_norm"] == q.target) == s["acc_norm"]


def test_synthetic_questions_tokenise_as_single_token_options(tok):
    for task in ("modarith", "parity"):
        qs = load_synthetic_questions(task, tok)
        assert len(qs) == 2000
        assert all(len(c) == 1 for q in qs for c in q.cont_encs)
        assert all(q.packed_len == len(q.context_enc) for q in qs)  # letter-style: one pass over the context only
