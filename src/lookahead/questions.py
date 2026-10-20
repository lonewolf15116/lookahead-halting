"""Turn benchmarks into questions: one shared context plus a list of answer continuations.

lm_eval tasks are built exactly the way lm_eval.simple_evaluate builds them (same few-shot override and seed), and each
(context, continuation) pair is tokenised with the harness's own `_encode_pair`, so a question's tokens are the ones
Gate 0 scored. Synthetic items are tokenised directly (context, then context + choice), as their generator checked.
"""
import hashlib
import json
import random
from dataclasses import dataclass, field

import numpy as np

# Plan section 1. Shots match Ouro Table 10 where the paper reports the benchmark.
BENCHMARKS = {
    "arc_challenge": {"shots": 25, "metric": "acc_norm"},
    "arc_easy": {"shots": 8, "metric": "acc_norm"},
    "hellaswag": {"shots": 10, "metric": "acc_norm"},
    "mmlu": {"shots": 5, "metric": "acc"},
    "commonsense_qa": {"shots": 10, "metric": "acc"},
    "openbookqa": {"shots": 0, "metric": "acc_norm"},
}
SYNTHETIC = ("modarith", "parity")
FEWSHOT_SEED = 1234  # lm_eval default (DEFAULT_OTHER_SEED), as in Gate 0


@dataclass
class Question:
    benchmark: str  # "arc_challenge", ..., "modarith", "parity"
    subtask: str  # lm_eval task name (MMLU subject task) or synthetic level, used for stratification
    doc_id: str
    context: str
    continuations: list[str]
    choice_text: list[str]  # what acc_norm divides by (lm_eval: len(doc_to_choice(doc)[k]))
    target: int
    metric: str  # "acc_norm" or "acc": which score's argmax is the prediction
    context_enc: list[int] = field(default_factory=list)
    cont_encs: list[list[int]] = field(default_factory=list)

    @property
    def packed_len(self) -> int:
        return len(self.context_enc) + sum(len(c) - 1 for c in self.cont_encs)

    @property
    def per_option_tokens(self) -> int:
        return sum(len(self.context_enc) + len(c) - 1 for c in self.cont_encs)


def question_set_hash(qs) -> str:
    """Identity of a benchmark's question set: tokens, targets and choice text, in order."""
    h = hashlib.sha256()
    for q in qs:
        h.update(json.dumps([q.doc_id, q.context_enc, q.cont_encs, q.target, q.choice_text], separators=(",", ":")).encode())
    return h.hexdigest()


def predictions(q: Question, option_ll: list[float]) -> dict:
    """acc / acc_norm predictions exactly as lm_eval computes them (np.argmax: first index wins ties)."""
    ll = np.asarray(option_ll, dtype=float)
    lens = np.asarray([float(len(c)) for c in q.choice_text])
    pred, pred_norm = int(np.argmax(ll)), int(np.argmax(ll / lens))
    return {"pred_acc": pred, "pred_acc_norm": pred_norm, "pred": pred_norm if q.metric == "acc_norm" else pred}


def primary_scores(q: Question, option_ll: list[float]) -> np.ndarray:
    """The scores whose softmax is p_t: length-normalised ll for acc_norm tasks, raw ll for acc tasks."""
    ll = np.asarray(option_ll, dtype=float)
    if q.metric == "acc_norm":
        return ll / np.asarray([float(len(c)) for c in q.choice_text])
    return ll


def load_lm_eval_questions(benchmark: str, lm, limit=None, task_manager=None, spec=None, *, on_task=None,
                           on_doc=None, samples_for=None) -> list[Question]:
    """Build one benchmark's questions with lm_eval, mirroring simple_evaluate's task setup.

    spec defaults to BENCHMARKS[benchmark]; task_manager defaults to lm_eval's (tests pass one with a local task).

    Optional hooks (all off by default, in which case behaviour and output are exactly as before):
      on_task(task)               after the shots and few-shot seed are set, before any request is built
      samples_for(task_name)      list of eval-doc indices to build, or None for every doc; an empty list skips the task.
                                  lm_eval renumbers the built docs 0..k-1 in dataset order, so the doc_id suffix is then
                                  the position within the sorted selection, not the dataset index
      on_doc(question, task, doc) after each question is built and tokenised (it may edit the question)
    """
    import torch
    from lm_eval.tasks import TaskManager

    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(1234)
    spec = spec or BENCHMARKS[benchmark]
    loaded = (task_manager or TaskManager()).load([benchmark])["tasks"]
    out = []
    for task_name in sorted(loaded):
        task = loaded[task_name]
        if task.get_config("num_fewshot") != 0:  # simple_evaluate never overrides a task whose config pins 0
            task.set_config(key="num_fewshot", value=spec["shots"])
        task.set_fewshot_seed(seed=FEWSHOT_SEED)
        if on_task is not None:
            on_task(task)
        samples = None
        if samples_for is not None:
            samples = samples_for(task_name)
            if samples is not None and len(samples) == 0:
                continue  # an empty list would make lm_eval build every doc
        task.build_all_requests(limit=limit, samples=None if samples is None else sorted(set(samples)), rank=0, world_size=1, cache_requests=False, rewrite_requests_cache=False,
                                system_instruction=None, apply_chat_template=False, fewshot_as_multiturn=False,
                                chat_template=None, tokenizer_name="")
        by_doc = {}
        for inst in task.instances:
            by_doc.setdefault(inst.doc_id, []).append(inst)
        for doc_id in sorted(by_doc):
            insts = sorted(by_doc[doc_id], key=lambda i: i.idx)
            doc = insts[0].doc
            contexts = {i.args[0] for i in insts}
            if len(contexts) != 1:
                raise ValueError(f"{task_name}/{doc_id}: options do not share one context")
            choices = task.doc_to_choice(doc)
            gold = task.doc_to_target(doc)
            if isinstance(gold, str):
                gold = choices.index(gold) if gold in choices else -100
            if not isinstance(gold, int) or not 0 <= gold < len(choices) or len(insts) != len(choices):
                raise ValueError(f"{task_name}/{doc_id}: bad target or option count")
            q = Question(benchmark, task_name, f"{task_name}/{doc_id}", insts[0].args[0], [i.args[1] for i in insts],
                         list(choices), gold, spec["metric"])
            encode_lm_eval(q, lm)
            if on_doc is not None:
                on_doc(q, task, doc)
            out.append(q)
    return out


def encode_lm_eval(q: Question, lm) -> None:
    pairs = [lm._encode_pair(q.context, c) for c in q.continuations]
    ctx = {tuple(p[0]) for p in pairs}
    if len(ctx) != 1:  # trailing-space handling could in principle give options different context tokens
        raise ValueError(f"{q.doc_id}: options tokenise the context differently")
    q.context_enc = list(pairs[0][0])
    q.cont_encs = [list(p[1]) for p in pairs]


def load_synthetic_questions(task: str, tokenizer) -> list[Question]:
    from lookahead.synthetic import generate

    out = []
    for it in generate(task):
        ctx_enc = tokenizer.encode(it.context)
        conts = []
        for c in it.choices:
            whole = tokenizer.encode(it.context + c)
            if whole[: len(ctx_enc)] != ctx_enc or len(whole) - len(ctx_enc) != 1:
                raise ValueError(f"{it.doc_id}: token boundary broken for {c!r}")
            conts.append(whole[len(ctx_enc):])
        out.append(Question(task, f"{task}-{it.level}", it.doc_id, it.context, list(it.choices),
                            [c.strip() for c in it.choices], it.target, "acc", ctx_enc, conts))
    return out
