"""Shared fixtures: the real Ouro tokenizer and a small local ARC-style lm_eval task (no download)."""
import json

import pytest

from lookahead.config import REPO_ROOT, load_config

LOCAL = REPO_ROOT / load_config("model")["local_dir"]


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(LOCAL))


@pytest.fixture()
def fake_task(tmp_path):
    from lm_eval.tasks import TaskManager

    def doc(i):  # options of different character lengths, so acc and acc_norm can disagree
        return {"question": f"What is {i} plus {i}?", "answerKey": "B",
                "choices": {"text": [str(2 * i), f"the number {2 * i + 1}", "x" * (i % 5 + 1), "none of these"],
                            "label": list("ABCD")}}

    train, test = tmp_path / "train.json", tmp_path / "test.json"
    train.write_text("\n".join(json.dumps(doc(i)) for i in range(1, 13)), encoding="utf-8")
    test.write_text("\n".join(json.dumps(doc(i)) for i in range(20, 28)), encoding="utf-8")
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
  - metric: acc_norm
metadata:
  version: 1.0
""", encoding="utf-8")
    return TaskManager(include_path=str(tmp_path))


