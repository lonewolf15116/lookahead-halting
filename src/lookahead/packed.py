"""Score every answer option of a question in one forward pass per loop.

Per-option scoring (lm-evaluation-harness, Gate 0) runs context + option k as its own sequence. Packed scoring runs

    [ context tokens | option 1 inputs | option 2 inputs | ... ]

once. Every token attends causally, and an option token may attend only to the context and to earlier tokens of its own
option, so each option sees exactly the sequence it would see alone. An option's RoPE positions restart at len(context),
so they also match the positions it would have alone.

As in the per-option scorer, the last token of each option is never fed in (nothing is predicted from it). Its first
token is predicted from the last context token, so single-token options (letter answers) add no tokens at all.
"""
from dataclasses import dataclass

import torch

from lookahead.ouro_adapter import OuroLoopRunner


@dataclass
class PackedInput:
    input_ids: list[int]
    position_ids: list[int]
    segment: list[int]  # 0 = context, k + 1 = option k
    score_pos: list[int]  # packed index whose logits predict each continuation token
    score_tgt: list[int]  # the continuation token predicted there
    score_opt: list[int]  # which option it belongs to
    n_context: int
    n_options: int


def build_packed(context_enc: list[int], cont_encs: list[list[int]]) -> PackedInput:
    if not context_enc:
        raise ValueError("empty context")
    if any(len(c) == 0 for c in cont_encs):
        raise ValueError("empty continuation")
    n = len(context_enc)
    ids, pos, seg = list(context_enc), list(range(n)), [0] * n
    score_pos, score_tgt, score_opt = [], [], []
    for k, cont in enumerate(cont_encs):
        prev = n - 1  # the last context token predicts the first continuation token
        for j, tok in enumerate(cont):
            score_pos.append(prev)
            score_tgt.append(tok)
            score_opt.append(k)
            if j < len(cont) - 1:  # feed every continuation token except the last
                ids.append(tok)
                pos.append(n + j)
                seg.append(k + 1)
                prev = len(ids) - 1
    return PackedInput(ids, pos, seg, score_pos, score_tgt, score_opt, n, len(cont_encs))


def segment_mask_function(segment: torch.Tensor):
    """and_mask_function for transformers' mask builder: kv is visible if it is context or in the query's own option."""

    def fn(batch_idx, head_idx, q_idx, kv_idx):
        return (segment[kv_idx] == 0) | (segment[kv_idx] == segment[q_idx])

    return fn


def reference_mask(segment: list[int]) -> torch.Tensor:
    """The boolean [L, L] attention pattern packed scoring must produce (True = may attend)."""
    s = torch.tensor(segment)
    causal = torch.tril(torch.ones(len(s), len(s), dtype=torch.bool))
    return causal & ((s[None, :] == 0) | (s[None, :] == s[:, None]))


@dataclass
class LoopOutput:
    loop: int  # 1-based
    option_ll: list[float]  # summed log-likelihood per option (float32 log_softmax)
    option_greedy: list[bool]
    hidden: torch.Tensor  # [d] at the last context token (shared by all options), model dtype
    gate_logit: float  # Ouro exit gate at the last context token
    vocab_logp: torch.Tensor  # [V] float32 log-probabilities at the last context token


class PackedScorer:
    def __init__(self, model):
        self.runner = OuroLoopRunner(model)
        self.device = next(model.parameters()).device

    def start(self, packed: PackedInput):
        dev = self.device
        input_ids = torch.tensor([packed.input_ids], device=dev)
        position_ids = torch.tensor([packed.position_ids], device=dev)
        segment = torch.tensor(packed.segment, device=dev)
        return self.runner.start(input_ids, position_ids=position_ids, and_mask_function=segment_mask_function(segment))

    @torch.no_grad()
    def loops(self, packed: PackedInput, n_loops: int | None = None):
        """Yield a LoopOutput after each loop."""
        dev = self.device
        state = self.start(packed)
        pos = torch.tensor(packed.score_pos, device=dev)
        tgt = torch.tensor(packed.score_tgt, device=dev)
        opt = torch.tensor(packed.score_opt, device=dev)
        last = torch.tensor([packed.n_context - 1], device=dev)
        for _ in range(n_loops or self.runner.total_loops):
            self.runner.step(state)
            logp = torch.log_softmax(self.runner.readout(state.hidden, pos[None])[0], dim=-1)  # [P, V] float32
            tok_ll = logp.gather(1, tgt[:, None]).squeeze(1)
            tok_greedy = logp.argmax(-1) == tgt
            ll = torch.zeros(packed.n_options, device=dev).index_add_(0, opt, tok_ll)
            greedy = torch.ones(packed.n_options, dtype=torch.bool, device=dev)
            greedy.scatter_reduce_(0, opt, tok_greedy, reduce="amin")
            first = logp[0] if packed.score_pos[0] == packed.n_context - 1 else None
            vocab_logp = first if first is not None else torch.log_softmax(self.runner.readout(state.hidden, last)[0], -1)
            yield LoopOutput(
                loop=state.loop,
                option_ll=ll.tolist(),
                option_greedy=greedy.tolist(),
                hidden=state.hidden[0, packed.n_context - 1].clone(),
                gate_logit=float(self.runner.gate_logit(state.hidden, last)[0]),
                vocab_logp=vocab_logp,
            )


@torch.no_grad()
def score_per_option(model, context_enc, cont_encs, n_loops=None):
    """Reference: each option as its own sequence (the Gate 0 / lm_eval path). Returns [loops][options] ll and greedy,
    plus the hidden state at the last context token per loop (from option 0's sequence)."""
    runner = OuroLoopRunner(model)
    dev = next(model.parameters()).device
    loops = n_loops or runner.total_loops
    lls = [[0.0] * len(cont_encs) for _ in range(loops)]
    greedy = [[False] * len(cont_encs) for _ in range(loops)]
    hidden = []
    for k, cont in enumerate(cont_encs):
        inp = (context_enc + cont)[:-1]
        positions = torch.arange(len(inp) - len(cont), len(inp), device=dev)[None]
        targets = torch.tensor(cont, device=dev)
        state = runner.start(torch.tensor([inp], device=dev))
        for t in range(loops):
            runner.step(state)
            logp = torch.log_softmax(runner.readout(state.hidden, positions)[0], dim=-1)
            lls[t][k] = float(logp.gather(1, targets[:, None]).sum())
            greedy[t][k] = bool((logp.argmax(-1) == targets).all())
            if k == 0:
                hidden.append(state.hidden[0, len(context_enc) - 1].clone())
    return lls, greedy, hidden
