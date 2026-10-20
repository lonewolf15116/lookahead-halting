"""Run the Phase 1 scripts end to end on CPU with a tiny Ouro and a fake ARC-style task (no download, no GPU)."""
import importlib.util
import json
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lm_eval")

from lookahead.config import REPO_ROOT, load_config  # noqa: E402
from lookahead.lm_eval_ouro import LoopedOuroLM  # noqa: E402

LOCAL = REPO_ROOT / load_config("model")["local_dir"]
pytestmark = pytest.mark.skipif(not (LOCAL / "tokenizer.json").exists(), reason="run scripts/inspect_ouro.py first")

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tiny4():
    from transformers import AutoConfig, AutoModelForCausalLM

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(
        str(LOCAL), trust_remote_code=True, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, head_dim=16, total_ut_steps=4,
        layer_types=["full_attention"] * 2, max_position_embeddings=1024,
    )
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True, torch_dtype=torch.float32).eval()


def test_pilot_runs_and_passes_against_per_option_cache(tmp_path, monkeypatch, tiny4, tok, fake_task):
    import lm_eval

    from lookahead import questions

    (tmp_path / "results" / "gate0").mkdir(parents=True)
    cache = tmp_path / "results" / "gate0" / "arc_cache.jsonl"
    lm = LoopedOuroLM(tiny4, tok, cache_path=cache)  # per-option reference, like Gate 0
    lm_eval.simple_evaluate(model=lm, tasks=["fake_arc"], num_fewshot=3, task_manager=fake_task, batch_size=1)

    pilot = load_script("pilot_packed")
    real_loader = questions.load_lm_eval_questions
    monkeypatch.setattr(pilot, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(pilot, "results_dir", lambda sub: (tmp_path / "results" / sub).mkdir(parents=True, exist_ok=True)
                        or tmp_path / "results" / sub)
    monkeypatch.setattr(pilot, "load_ouro", lambda: (tiny4, tok))
    monkeypatch.setattr(pilot, "load_lm_eval_questions", lambda b, lm, limit=None: real_loader(
        "fake_arc", lm, task_manager=fake_task, spec={"shots": 3, "metric": "acc_norm"}))
    monkeypatch.setattr(sys, "argv", ["pilot_packed.py"])
    assert pilot.main() == 0
    res = json.loads((tmp_path / "results" / "phase1" / "pilot_packed.json").read_text())
    assert res["missing_from_gate0_cache"] == 0 and res["passed"]
    for row in res["per_loop"]:
        assert row["ll_absdiff"]["max"] <= 1e-4 and row["top_identical_acc_norm"] == 1.0
        assert row["hidden_maxabs"]["max"] <= 1e-5
    assert all(t["flip_agreement"] == 1.0 for t in res["targets"])


def test_prepare_runs_on_synthetic(tmp_path, monkeypatch):
    prep = load_script("prepare_phase1")
    monkeypatch.setattr(sys, "argv", ["prepare_phase1.py", "--limit", "40", "--benchmarks", "modarith", "parity",
                                      "--root", str(tmp_path)])
    assert prep.main() == 0
    m = json.loads((tmp_path / "data" / "manifests" / "questions_limit40.json").read_text())
    assert m["benchmarks"]["parity"]["n_questions"] == 40
    assert sum(m["split_counts"]["parity"].values()) == 40
    tc = json.loads((tmp_path / "results" / "phase1" / "token_counts_limit40.json").read_text())
    assert {r["benchmark"] for r in tc["rows"]} == {"modarith", "parity"}
