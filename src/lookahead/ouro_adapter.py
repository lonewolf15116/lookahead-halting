"""Run Ouro one loop at a time, so a halting policy can genuinely skip the loops it does not run.

Ouro's own forward (modeling_ouro.py) always executes all `total_ut_steps` loops and only then picks
which loop's output to read, so it cannot save compute. This adapter repeats the same computation
loop by loop and exposes the state after each loop, the exit gate, and the readout logits.

The computation it mirrors (transformers 4.54.x, revision pinned in configs/model.yaml):

    h = embed(input_ids)
    repeat total_ut_steps times:   h = final_norm(layers(h))     # the normed output feeds the next loop
    exit gate at loop t:           early_exit_gate(h_t)          # Linear(hidden, 1)

Equivalence with the stock forward is checked by tests/test_adapter_equivalence.py (tiny model, real Ouro
code) and scripts/verify_adapter.py (real weights).
"""
from dataclasses import dataclass

import torch
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

from lookahead.config import REPO_ROOT, load_config


def exit_distribution(gate_logits: torch.Tensor) -> torch.Tensor:
    """Ouro's exit distribution from gate logits of shape [..., T], one logit per loop.

    lambda_t = sigmoid(g_t); p_t = lambda_t * prod_{j<t}(1 - lambda_j); the last loop takes what is left.
    """
    lam = torch.sigmoid(gate_logits.float())
    survive = torch.cumprod(1.0 - lam, dim=-1)  # prod_{j<=t}(1 - lambda_j)
    remaining_before = torch.cat([torch.ones_like(survive[..., :1]), survive[..., :-1]], dim=-1)
    pdf = lam * remaining_before
    return torch.cat([pdf[..., :-1], remaining_before[..., -1:]], dim=-1)


class NativeGate:
    """Streaming form of exit_distribution, for a policy that decides loop by loop.

    update() takes one loop's gate logit per row and returns the cumulative probability that Ouro's own
    gate would have exited at or before that loop (1.0 at the last loop).
    """

    def __init__(self, total_loops: int, batch: int, device=None):
        self.total_loops = total_loops
        self.loops_seen = 0
        self.survive = torch.ones(batch, device=device)

    def update(self, gate_logit: torch.Tensor) -> torch.Tensor:
        self.loops_seen += 1
        if self.loops_seen >= self.total_loops:
            self.survive = torch.zeros_like(self.survive)
        else:
            self.survive = self.survive * (1.0 - torch.sigmoid(gate_logit.float()))
        return 1.0 - self.survive


@dataclass
class LoopState:
    hidden: torch.Tensor  # [B, L, d]: token embeddings at loop 0, else the output of the last completed loop
    loop: int  # loops completed so far
    position_ids: torch.Tensor
    cache_position: torch.Tensor
    position_embeddings: tuple
    masks: dict


class OuroLoopRunner:
    """Loop-by-loop execution of an OuroForCausalLM. No KV cache: each call sees the whole sequence."""

    def __init__(self, model):
        self.model = model
        self.core = model.model
        self.config = model.config
        self.total_loops = self.core.total_ut_steps

    @torch.no_grad()
    def start(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
              position_ids: torch.Tensor | None = None, and_mask_function=None) -> LoopState:
        """Embed the input and build masks and rotary embeddings.

        position_ids (optional) are the RoPE positions; by default 0..L-1. and_mask_function (optional) is combined with
        the causal mask by transformers (used by packed scoring). Mask construction always sees plain 0..L-1 positions
        and an explicit all-ones attention mask when none is given, so transformers never reinterprets non-monotonic
        RoPE positions as separate packed documents.
        """
        embeds = self.core.embed_tokens(input_ids)
        cache_position = torch.arange(embeds.shape[1], device=embeds.device)
        mask_positions = cache_position.unsqueeze(0)
        if position_ids is None:
            position_ids = mask_positions
        mask_kwargs = {
            "config": self.config,
            "input_embeds": embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": mask_positions,
        }
        if and_mask_function is not None:
            if attention_mask is None:
                mask_kwargs["attention_mask"] = torch.ones(input_ids.shape, dtype=torch.long, device=embeds.device)
            mask_kwargs["and_mask_function"] = and_mask_function
        masks = {"full_attention": create_causal_mask(**mask_kwargs)}
        if self.core.has_sliding_layers:
            if and_mask_function is not None:
                raise NotImplementedError("packed scoring assumes full attention only (true for Ouro-1.4B)")
            masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        return LoopState(
            hidden=embeds,
            loop=0,
            position_ids=position_ids,
            cache_position=cache_position,
            position_embeddings=self.core.rotary_emb(embeds, position_ids),
            masks=masks,
        )

    @torch.no_grad()
    def step(self, state: LoopState) -> LoopState:
        """Run one more loop in place and return the state."""
        hidden = state.hidden
        for layer in self.core.layers[: self.config.num_hidden_layers]:
            hidden = layer(
                hidden,
                attention_mask=state.masks[layer.attention_type],
                position_ids=state.position_ids,
                past_key_value=None,
                use_cache=False,
                cache_position=state.cache_position,
                position_embeddings=state.position_embeddings,
                current_ut=state.loop,
            )
        state.hidden = self.core.norm(hidden)
        state.loop += 1
        return state

    @torch.no_grad()
    def readout(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """lm_head logits (float32) at the given positions: [B] -> [B, V], or [B, K] -> [B, K, V]."""
        single = positions.dim() == 1
        pos = (positions[:, None] if single else positions).to(hidden.device)
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        logits = self.model.lm_head(hidden[rows, pos]).float()
        return logits[:, 0] if single else logits

    @torch.no_grad()
    def gate_logit(self, hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Ouro's exit-gate logit at one position per row: [B]."""
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return self.core.early_exit_gate(hidden[rows, positions.to(hidden.device)]).squeeze(-1)

    @torch.no_grad()
    def all_loops(self, input_ids, positions, attention_mask=None):
        """Run every loop. Returns answer-position logits [T, B, V] and gate logits [T, B] (float32)."""
        state = self.start(input_ids, attention_mask)
        logits, gates = [], []
        for _ in range(self.total_loops):
            self.step(state)
            logits.append(self.readout(state.hidden, positions))
            gates.append(self.gate_logit(state.hidden, positions).float())
        return torch.stack(logits), torch.stack(gates)


def load_ouro(device: str = "cuda"):
    """Load the pinned local Ouro copy and its tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config("model")
    local = str(REPO_ROOT / cfg["local_dir"])
    tok = AutoTokenizer.from_pretrained(local, trust_remote_code=cfg["trust_remote_code"])
    model = AutoModelForCausalLM.from_pretrained(
        local, trust_remote_code=cfg["trust_remote_code"], torch_dtype=getattr(torch, cfg["dtype"])
    )
    return model.to(device).eval(), tok
