"""Phase 3 D-a2: a genuine float64 reference for Ouro's forward pass (diagnostic use only; never used by extraction).

Ouro's remote modeling code forces float32 in three places regardless of the model dtype:
  OuroRMSNorm.forward            hidden_states.to(torch.float32) ... .to(input_dtype)
  rotary embedding forward       inv_freq (a float32 buffer) and cos/sin computed under "Force float32", then cast to x.dtype
  eager_attention_forward        softmax(..., dtype=torch.float32)
A model merely moved to float64 would therefore still round to float32 there, and would not be a float64 reference.

`patch_float64` replaces exactly those three computations with the SAME formulas evaluated in float64, and only when the
input is float64 (float32/bf16 behaviour is untouched). The rotary inverse frequencies are recomputed in float64 from the
same config with the same formula as transformers' default RoPE initialiser; `rotary_check` confirms they equal the stored
float32 buffer to float32 precision. `DtypeAudit` records every module output that is not float64 during a float64 pass.
"""
import contextlib
import sys

import torch

F64 = torch.float64


def rmsnorm_forward64(orig):
    def forward(self, hidden_states):
        if hidden_states.dtype != F64:
            return orig(self, hidden_states)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        return self.weight * (hidden_states * torch.rsqrt(variance + self.variance_epsilon))

    forward.__lh_f64__ = True
    return forward


def inv_freq64(config, device=None) -> torch.Tensor:
    """transformers' default RoPE inverse frequencies, evaluated in float64 (same formula, same config fields)."""
    base = config.rope_theta
    partial = getattr(config, "partial_rotary_factor", 1.0) or 1.0
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial)
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).to(F64) / dim))


def rotary_forward64(orig):
    @torch.no_grad()
    def forward(self, x, position_ids):
        if x.dtype != F64:
            return orig(self, x, position_ids)
        if getattr(self, "rope_type", "default") != "default":
            raise NotImplementedError(f"float64 reference supports only default RoPE, not {self.rope_type!r}")
        inv = inv_freq64(self.config, x.device)
        freqs = (inv[None, :, None].expand(position_ids.shape[0], -1, 1) @ position_ids[:, None, :].to(F64)).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling

    forward.__lh_f64__ = True
    return forward


def eager_attention64(orig, repeat_kv):
    def fn(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.dtype != F64:
            return orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        k, v = repeat_kv(key, module.num_key_value_groups), repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask[:, :, :, : k.shape[-2]]
        w = torch.nn.functional.softmax(w, dim=-1, dtype=F64)
        w = torch.nn.functional.dropout(w, p=dropout, training=module.training)
        return torch.matmul(w, v).transpose(1, 2).contiguous(), w

    fn.__lh_f64__ = True
    return fn


def patch_float64(model) -> dict:
    """Patch the model's classes in place (idempotent). Returns what was patched."""
    done = {"rmsnorm": [], "rotary": [], "eager_attention": False}
    for m in model.modules():
        cls = type(m)
        name = cls.__name__
        if name.endswith("RMSNorm") and not getattr(cls.forward, "__lh_f64__", False):
            cls.forward = rmsnorm_forward64(cls.forward)
            done["rmsnorm"].append(name)
        elif name.endswith("RotaryEmbedding") and not getattr(cls.forward, "__lh_f64__", False):
            cls.forward = rotary_forward64(cls.forward)
            done["rotary"].append(name)
    mod = sys.modules[type(model.model).__module__]
    if hasattr(mod, "eager_attention_forward") and not getattr(mod.eager_attention_forward, "__lh_f64__", False):
        mod.eager_attention_forward = eager_attention64(mod.eager_attention_forward, mod.repeat_kv)
        done["eager_attention"] = True
    return done


@torch.no_grad()
def rmsnorm_check(norm) -> float:
    """Max relative error of the (patched) RMSNorm on a float64 input against the formula in float64."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 7, norm.weight.shape[0], generator=g, dtype=F64).to(norm.weight.device) * 10
    ref = norm.weight.to(F64) * x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)
    out = norm(x)
    if out.dtype != F64:
        return float("inf")
    return float(((out - ref).abs() / ref.abs().clamp_min(1e-300)).max())


def rotary_check(rot) -> dict:
    """The float64 inverse frequencies must equal the module's stored buffer to float32 precision (same function)."""
    inv = inv_freq64(rot.config, rot.inv_freq.device)
    stored = rot.original_inv_freq if hasattr(rot, "original_inv_freq") else rot.inv_freq
    rel = float(((stored.to(F64) - inv).abs() / inv.abs()).max())
    return {"rope_type": getattr(rot, "rope_type", "default"), "attention_scaling": float(rot.attention_scaling),
            "inv_freq_rel_diff_vs_stored": rel, "ok": rel <= 1e-6 and getattr(rot, "rope_type", "default") == "default"}


class DtypeAudit:
    """Forward hooks that record every module whose floating output is not float64."""

    def __init__(self, model):
        self.bad: dict = {}
        self.n_calls = 0
        self.handles = [m.register_forward_hook(self._hook) for m in model.modules()]

    def _hook(self, module, _inputs, output):
        self.n_calls += 1
        for t in (output if isinstance(output, (tuple, list)) else (output,)):
            if torch.is_tensor(t) and t.is_floating_point() and t.dtype != F64:
                self.bad[type(module).__name__] = str(t.dtype)

    def close(self):
        for h in self.handles:
            h.remove()


class Float64Streaming:
    """Option 2 (memory-bounded float64): weights stay in bf16 at rest; every module that owns parameters has them cast to
    float64 just before its forward and restored just after. bf16 -> float64 -> bf16 is exact, so the arithmetic is the
    same as a model held entirely in float64, while at most one module's weights are float64 at a time. Combined with
    `patch_float64` (no forced-float32 steps) and `DtypeAudit` (no non-float64 module output). `close()` removes the hooks
    and checks that every parameter is back in its original dtype with bit-identical values."""

    def __init__(self, model, audit=None):
        self.model = model
        self.audit = audit  # an OpAudit, if any: weight casts are bookkeeping, not part of the computation
        self.owners = [m for m in model.modules() if any(True for _ in m.parameters(recurse=False))]
        covered = {id(p) for m in self.owners for p in m.parameters(recurse=False)}
        if covered != {id(p) for p in model.parameters()}:
            raise RuntimeError("some parameters are not owned by a hooked module")
        self.dtypes = {id(p): p.dtype for p in model.parameters()}
        self.digest = self._digest()
        self.n_casts = 0
        self.handles = []
        for m in self.owners:
            self.handles.append(m.register_forward_pre_hook(self._up))
            self.handles.append(m.register_forward_hook(self._down))

    def _digest(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for name, p in self.model.named_parameters():
            h.update(name.encode() + str(p.dtype).encode() + p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def _ctx(self):
        return self.audit.paused() if self.audit is not None else contextlib.nullcontext()

    def _up(self, module, _inputs):
        with self._ctx():
            for p in module.parameters(recurse=False):
                p.data = p.data.to(F64)
        self.n_casts += 1

    def _down(self, module, _inputs, _output):
        with self._ctx():
            for p in module.parameters(recurse=False):
                p.data = p.data.to(self.dtypes[id(p)])

    def close(self) -> bool:
        for h in self.handles:
            h.remove()
        return all(p.dtype == self.dtypes[id(p)] for p in self.model.parameters()) and self._digest() == self.digest


# ---------------------------------------------------------------- reviewer concern 2: internal downcasts (op-level audit)

from contextlib import contextmanager  # noqa: E402

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402


class OpAudit(TorchDispatchMode):
    """Records every ATen operation (below modules, inside them, inside SDPA's math path) whose floating output is not
    float64, while inside an `active()` region and not `paused()`. Unlike DtypeAudit it sees internal steps: a module that
    casts to float32 and back shows up as `_to_copy -> torch.float32`. The D-a2 script makes exactly the hidden-state
    computation active (OuroLoopRunner.start and .step: embedding, masks, RoPE, every layer, final norm); the vocabulary
    readout and exit gate, which the comparison does not use, stay outside, and the streamed weight casts are paused."""

    def __init__(self):
        super().__init__()
        self.bad: dict = {}
        self.n_ops = 0
        self._paused = 0
        self._active = 0

    @contextmanager
    def active(self):
        self._active += 1
        try:
            yield
        finally:
            self._active -= 1

    @contextmanager
    def paused(self):
        self._paused += 1
        try:
            yield
        finally:
            self._paused -= 1

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        if self._active and not self._paused:
            self.n_ops += 1
            for t in (out if isinstance(out, (tuple, list)) else (out,)):
                if torch.is_tensor(t) and t.is_floating_point() and t.dtype != F64:
                    key = f"{func} -> {t.dtype}"
                    self.bad[key] = self.bad.get(key, 0) + 1
        return out


@contextmanager
def active_methods(audit: OpAudit, cls, names):
    """Run the named methods of `cls` inside the op audit's active region (originals restored afterwards)."""
    orig = {n: getattr(cls, n) for n in names}

    def wrap(f):
        def g(*a, **k):
            with audit.active():
                return f(*a, **k)
        return g

    try:
        for n, f in orig.items():
            setattr(cls, n, wrap(f))
        yield
    finally:
        for n, f in orig.items():
            setattr(cls, n, f)


# ---------------------------------------------------------------- reviewer concern 3: real-model, real-size check

@contextmanager
def _hooks_detached(module):
    saved = [(m, dict(m._forward_pre_hooks), dict(m._forward_hooks)) for m in module.modules()]
    try:
        for m, _, _ in saved:
            m._forward_pre_hooks.clear()
            m._forward_hooks.clear()
        yield
    finally:
        for m, pre, post in saved:
            m._forward_pre_hooks.update(pre)
            m._forward_hooks.update(post)


def resident_float64_copy(module):
    """A hook-free deep copy of `module` held entirely in float64 (for comparison with the streamed original)."""
    import copy

    with _hooks_detached(module):
        res = copy.deepcopy(module)
    return res.to(F64)


@torch.no_grad()
def real_layer_equivalence(runner, state, layer_indices=(0,)) -> dict:
    """On the REAL model with REAL dimensions and kernels: each listed decoder layer and the final norm, streamed, must be
    bit-identical to a resident float64 copy of the same module on the same float64 input. `state` is a float64 LoopState."""
    out = {}
    h = state.hidden
    for i in layer_indices:
        layer = runner.core.layers[i]
        kw = dict(attention_mask=state.masks[layer.attention_type], position_ids=state.position_ids, past_key_value=None,
                  use_cache=False, cache_position=state.cache_position, position_embeddings=state.position_embeddings,
                  current_ut=state.loop)
        a = layer(h, **kw)
        b = resident_float64_copy(layer)(h, **kw)
        out[f"layer{i}"] = {"identical": bool(torch.equal(a, b)), "maxabs": float((a - b).abs().max()), "dtype": str(a.dtype)}
        h = a
    n_s, n_r = runner.core.norm(h), resident_float64_copy(runner.core.norm)(h)
    out["final_norm"] = {"identical": bool(torch.equal(n_s, n_r)), "maxabs": float((n_s - n_r).abs().max()), "dtype": str(n_s.dtype)}
    out["ok"] = all(v["identical"] and v["dtype"] == str(F64) for v in out.values() if isinstance(v, dict))
    return out
