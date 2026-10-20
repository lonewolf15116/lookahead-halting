"""LoopedOuroLM must score like Ouro's own forward, cache correctly, and run end to end inside lm-evaluation-harness."""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("lm_eval")

from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.lm_eval_ouro import LoopedOuroLM  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
pytestmark = pytest.mark.skipif(not (LOCAL / "tokenizer.json").exists(), reason="run scripts/inspect_ouro.py first")
LOOPS = 3


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(LOCAL))


@pytest.fixture(scope="module")
def tiny():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, total_ut_steps=LOOPS,
        layer_types=["full_attention"] * 2, max_position_embeddings=1024,  # real vocab size, so real token ids fit
    )
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()


def test_scores_match_stock_forward(tiny, tok):
    lm = LoopedOuroLM(tiny, tok)
    pairs = [("Question: What is two plus two?\nAnswer:", " four"),
             ("Question: Which animal is a mammal?\nAnswer:", " a whale that swims")]
    for context, continuation in pairs:
        context_enc, cont_enc = lm._encode_pair(context, continuation)
        lls, greedy = lm.score_all_loops(context_enc, cont_enc)
        ids = torch.tensor([context_enc + cont_enc])
        targets = torch.tensor(cont_enc)
        for k in range(LOOPS):
            with torch.no_grad():
                logits = tiny(input_ids=ids[:, :-1], use_cache=False, exit_at_step=k).logits[0].float()
            rows = torch.log_softmax(logits, -1)[-len(cont_enc):]
            assert lls[k] == pytest.approx(rows.gather(1, targets[:, None]).sum().item(), abs=1e-3)
            assert greedy[k] == bool((rows.argmax(-1) == targets).all())


def test_cache_is_reused_persisted_and_exit_loop_selects(tmp_path, tiny, tok):
    cache = tmp_path / "cache.jsonl"
    lm = LoopedOuroLM(tiny, tok, cache_path=cache)
    context, continuation = "Question: What is 1+1?\nAnswer:", " 2"
    req = [((context, continuation), *lm._encode_pair(context, continuation))]
    lm.exit_loop = 1
    first = lm._loglikelihood_tokens(req)
    assert lm.newly_scored == 1
    assert lm._loglikelihood_tokens(req) == first and lm.newly_scored == 1
    lm.exit_loop = 2
    assert lm._loglikelihood_tokens(req) != first  # a different loop gives different scores; still no new work
    assert lm.newly_scored == 1
    reloaded = LoopedOuroLM(tiny, tok, cache_path=cache)
    reloaded.exit_loop = 1
    assert reloaded._loglikelihood_tokens(req) == first and reloaded.newly_scored == 0


def test_runs_inside_lm_eval_and_all_loops_come_from_one_pass(tmp_path, tiny, tok):
    import lm_eval
    from lm_eval.tasks import TaskManager

    def doc(i):
        return {"id": f"q{i}", "question": f"What is {i} plus {i}?", "answerKey": "A",
                "choices": {"text": [str(2 * i), str(2 * i + 1), str(2 * i + 2), str(2 * i + 3)], "label": list("ABCD")}}

    train, test = tmp_path / "train.json", tmp_path / "test.json"
    train.write_text("\n".join(json.dumps(doc(i)) for i in range(1, 13)), encoding="utf-8")
    test.write_text("\n".join(json.dumps(doc(i)) for i in range(20, 26)), encoding="utf-8")
    (tmp_path / "fake_arc.yaml").write_text(f"""task: fake_arc
dataset_path: json
dataset_kwargs:
  data_files:
    train: "{train.as_posix()}"
    test: "{test.as_posix()}"
output_type: multiple_choice
training_split: train
test_split: test
doc_to_text: "Question: {{{{question}}}}\\nAnswer:"
doc_to_target: "{{{{choices.label.index(answerKey)}}}}"
doc_to_choice: "{{{{choices.text}}}}"
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
  - metric: acc_norm
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
""", encoding="utf-8")
    tm = TaskManager(include_path=str(tmp_path))
    lm = LoopedOuroLM(tiny, tok)
    scored_after_first = None
    for loop in range(1, LOOPS + 1):
        lm.exit_loop = loop
        res = lm_eval.simple_evaluate(model=lm, tasks=["fake_arc"], num_fewshot=3, task_manager=tm,
                                      batch_size=1, log_samples=True)
        assert res["n-samples"]["fake_arc"]["effective"] == 6
        # every key scripts/reproduce_ouro.py reads from the result
        metrics = res["results"]["fake_arc"]
        for key in ("acc,none", "acc_stderr,none", "acc_norm,none", "acc_norm_stderr,none"):
            assert key in metrics, key
        assert {"doc_id", "target", "resps", "acc", "acc_norm"} <= set(res["samples"]["fake_arc"][0])
        if scored_after_first is None:
            scored_after_first = lm.newly_scored
    assert scored_after_first == 6 * 4  # 6 questions x 4 choices, scored once
    assert lm.newly_scored == scored_after_first  # loops 2 and 3 cost nothing
