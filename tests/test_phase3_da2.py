"""Phase 3 D-a2: the float64 reference patches are genuine float64 and leave other dtypes untouched; selection, per-loop
summary and interpretation rules; the committed D-a report is pinned (no model, no data download)."""
import importlib.util
import math
import types

import pytest
import torch

from lookahead import f64_reference as F
from lookahead import phase3_build as P
from lookahead.config import REPO_ROOT

_spec = importlib.util.spec_from_file_location("da2", REPO_ROOT / "scripts" / "diagnose_short_context_ood_a2.py")
A2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A2)


# ---------------------------------------------------------------- stand-ins with Ouro's exact float32-forcing code

class FakeRMSNorm(torch.nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(0.5, 1.5, n))
        self.variance_epsilon = eps

    def forward(self, hidden_states):  # verbatim from modeling_ouro.OuroRMSNorm
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FakeRotaryEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(rope_theta=1_000_000.0, head_dim=128, hidden_size=2048, num_attention_heads=16)
        self.rope_type, self.attention_scaling = "default", 1.0
        inv = 1.0 / (1_000_000.0 ** (torch.arange(0, 128, 2, dtype=torch.int64).float() / 128))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):  # Ouro's logic: float32 inv_freq and float32 cos/sin, cast to x.dtype
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        freqs = (inv @ position_ids[:, None, :].float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(x.dtype), (emb.sin() * self.attention_scaling).to(x.dtype)


def test_rmsnorm_patch_is_float64_and_untouched_otherwise():
    n = FakeRMSNorm(64).double()
    x = torch.randn(2, 5, 64, dtype=torch.float64) * 30
    before = n(x)  # float32 internally
    x32 = x.float()
    n32 = FakeRMSNorm(64)
    out32_before = n32(x32)
    FakeRMSNorm.forward = F.rmsnorm_forward64(FakeRMSNorm.forward)
    assert F.rmsnorm_check(n) <= 1e-14
    assert (n(x) - before).abs().max() > 1e-10  # the unpatched float64 run really was rounded to float32
    assert torch.equal(n32(x32), out32_before)  # float32 behaviour unchanged


def test_rotary_patch_is_float64_same_function_and_untouched_otherwise():
    r = FakeRotaryEmbedding()
    pos = torch.arange(0, 3000)[None]
    x64, x32 = torch.zeros(1, dtype=torch.float64), torch.zeros(1)
    c_before, _ = r(x64, pos)
    c32_before, s32_before = r(x32, pos)
    FakeRotaryEmbedding.forward = F.rotary_forward64(FakeRotaryEmbedding.forward)
    c, s = r(x64, pos)
    inv = 1.0 / (1e6 ** (torch.arange(0, 128, 2, dtype=torch.float64) / 128))
    f = pos[0, :, None].double() * inv[None, :]
    assert torch.allclose(c[0, :, :64], f.cos(), atol=1e-12, rtol=0) and torch.allclose(s[0, :, 64:], f.sin(), atol=1e-12, rtol=0)
    assert (c - c_before).abs().max() > 1e-8  # float32 positions*frequencies at large positions really were rounded
    assert F.rotary_check(r)["ok"]
    assert all(torch.equal(a, b) for a, b in zip(r(x32, pos), (c32_before, s32_before)))


def test_eager_attention_patch_softmax_in_float64():
    def repeat_kv(t, n):
        return t

    def orig(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        w = torch.nn.functional.softmax(torch.matmul(query, key.transpose(2, 3)) * scaling, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(w, value).transpose(1, 2), w

    mod = types.SimpleNamespace(num_key_value_groups=1, training=False)
    q, k, v = (torch.randn(1, 2, 4, 8, dtype=torch.float64) * 5 for _ in range(3))
    fn = F.eager_attention64(orig, repeat_kv)
    out, w = fn(mod, q, k, v, None, 0.3)
    ref = torch.softmax(torch.matmul(q, k.transpose(2, 3)) * 0.3, dim=-1)
    assert w.dtype == torch.float64 and torch.allclose(w, ref, atol=1e-15, rtol=0)
    assert torch.equal(fn(mod, q.float(), k.float(), v.float(), None, 0.3)[0], orig(mod, q.float(), k.float(), v.float(), None, 0.3)[0])


def test_dtype_audit_flags_non_float64_outputs():
    class Down(torch.nn.Module):
        def forward(self, x):
            return x.float()

    m = torch.nn.Sequential(torch.nn.Identity(), Down())
    a = F.DtypeAudit(m)
    m(torch.zeros(2, dtype=torch.float64))
    a.close()
    assert a.bad == {"Down": "torch.float32", "Sequential": "torch.float32"}


# ---------------------------------------------------------------- rules

def _rows(vals):
    return [{"doc_id": d, "loop": t, "packed_vs_ctx_maxabs": v, "packed_vs_ctxmask_maxabs": v, "packed_vs_ctx_rel_l2": v / 10}
            for d, vs in vals.items() for t, v in enumerate(vs, 1)]


def test_worst_k_by_numerical_difference_with_id_ties():
    rows = _rows({"a": [0, 1e-5, 0, 0], "b": [0, 0, 0, 2e-4], "c": [1e-5, 0, 0, 0], "d": [0, 0, 0, 0]})
    assert A2.worst_k(rows, 3) == ["b", "a", "c"]


def test_per_loop_summary():
    pl = A2.per_loop(_rows({"a": [0, 1e-6, 1e-5, 1e-4], "b": [0, 0, 0, 0]}))
    assert pl["1"]["share_exact"] == 1.0 and pl["4"]["maxabs_max"] == 1e-4 and pl["4"]["share_exact"] == 0.5


def test_interpretation_rule_is_the_frozen_one_and_requires_complete_finite_rows():
    ids = [f"q{i}" for i in range(10)]
    ok = _rows({d: [1e-12] * 4 for d in ids})
    assert A2.interpret(ok, True, True, ids) == ("numerical", 1e-12)
    worse = _rows({d: ([1e-12] * 3 + [2e-9] if d == "q3" else [1e-12] * 4) for d in ids})
    assert A2.interpret(worse, True, True, ids)[0] == "differs"
    assert A2.interpret(ok, False, True, ids)[0] == "unresolved" and A2.interpret(ok, True, False, ids)[0] == "unresolved"
    # safeguard 2: missing loop, duplicate row, extra question, non-finite value, wrong number of selected ids
    assert A2.interpret([r for r in ok if not (r["doc_id"] == "q5" and r["loop"] == 4)], True, True, ids)[0] == "unresolved"
    assert A2.interpret(ok + ok[:1], True, True, ids)[0] == "unresolved"
    assert A2.interpret(ok + _rows({"zz": [0.0] * 4}), True, True, ids)[0] == "unresolved"
    for bad_val in (float("nan"), float("inf"), None):
        rows = [dict(r) for r in ok]
        rows[7]["packed_vs_ctxmask_maxabs"] = bad_val
        assert A2.interpret(rows, True, True, ids)[0] == "unresolved"
    assert A2.interpret(ok, True, True, ids[:9])[0] == "unresolved"
    assert A2.interpret(_rows({d: [1e-12] * 4 for d in ids[:9]}), True, True, ids)[0] == "unresolved"
    assert math.isnan(A2.interpret([], True, True, ids)[1])
    assert A2.F64_TOL == 1e-9 and A2.K_WORST == 10


def test_committed_da_report_is_pinned_and_never_the_output():
    assert P._sha_ok(REPO_ROOT / A2.DA_REPORT, A2.DA_REPORT_SHA256)
    assert A2.REPORT != A2.DA_REPORT
    P.assert_not_sealed(A2.REPORT)
    src = (REPO_ROOT / "scripts" / "diagnose_short_context_ood_a2.py").read_text(encoding="utf-8")
    for word in ("correct", "pred", "accuracy", "score_per_option", "opt_ll"):
        assert f'"{word}"' not in src and f"'{word}'" not in src


# ---------------------------------------------------------------- end to end on the real Ouro classes (tiny random weights)

def test_patches_apply_to_the_real_ouro_modules_and_the_float64_pass_is_pure_float64():
    from lookahead.config import load_config

    local = REPO_ROOT / load_config("model")["local_dir"]
    if not (local / "modeling_ouro.py").exists():
        pytest.skip("Ouro modeling code not present")
    from transformers import AutoConfig, AutoModelForCausalLM

    from lookahead.ouro_adapter import OuroLoopRunner
    from lookahead.packed import PackedScorer, build_packed

    cfg = AutoConfig.from_pretrained(str(local), trust_remote_code=True)
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=4, head_dim=16, vocab_size=500, max_window_layers=2).items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval().to(torch.float64)
    done = F.patch_float64(m)
    assert done["rmsnorm"] and done["rotary"] and done["eager_attention"] is not None
    assert F.rmsnorm_check(m.model.norm) <= 1e-14 and F.rotary_check(m.model.rotary_emb)["ok"]
    audit = F.DtypeAudit(m)
    ctx, conts = [5, 17, 99, 3, 250, 11], [[7, 8, 9], [42, 43], [300, 301, 302, 303]]
    packed = [o.hidden for o in PackedScorer(m).loops(build_packed(ctx, conts))]
    r = OuroLoopRunner(m)
    s = r.start(torch.tensor([ctx]))
    for t in range(4):
        r.step(s)
        assert packed[t].dtype == torch.float64
        assert float((packed[t] - s.hidden[0, -1]).abs().max()) <= 1e-12  # packed == context-only in float64
    audit.close()
    assert audit.n_calls > 0 and audit.bad == {}


# ---------------------------------------------------------------- option 2: layer-streamed float64 (reviewer: develop, show equivalence)

def _tiny_ouro():
    from lookahead.config import load_config

    local = REPO_ROOT / load_config("model")["local_dir"]
    if not (local / "modeling_ouro.py").exists():
        pytest.skip("Ouro modeling code not present")
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(str(local), trust_remote_code=True)
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=4, head_dim=16, vocab_size=500, max_window_layers=2).items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval().to(torch.bfloat16)


def _all_states(m, device):
    from lookahead.ouro_adapter import OuroLoopRunner
    from lookahead.packed import PackedScorer, build_packed

    ctx, conts = [5, 17, 99, 3, 250, 11], [[7, 8, 9], [42, 43], [300, 301, 302, 303]]
    out = [o.hidden.clone() for o in PackedScorer(m).loops(build_packed(ctx, conts))]
    r = OuroLoopRunner(m)
    s = r.start(torch.tensor([ctx], device=device))
    for _ in range(4):
        r.step(s)
        out.append(s.hidden[0, -1].clone())
    return out


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                                         reason="no GPU"))])
def test_streamed_float64_is_bit_identical_to_a_full_float64_model_and_restores_weights(device):
    import copy

    base = _tiny_ouro()
    full = copy.deepcopy(base).to(device).to(torch.float64)
    F.patch_float64(full)
    ref = _all_states(full, device)
    st = copy.deepcopy(base).to(device)
    stream = F.Float64Streaming(st)
    F.patch_float64(st)
    audit = F.DtypeAudit(st)
    got = _all_states(st, device)
    audit.close()
    assert all(g.dtype == torch.float64 for g in got)
    assert all(torch.equal(a, b) for a, b in zip(ref, got))  # same arithmetic, bit for bit
    assert audit.bad == {} and stream.n_casts > 0
    assert stream.close()  # every weight back to bf16 with identical bits
    assert all(p.dtype == torch.bfloat16 for p in st.parameters())



# ---------------------------------------------------------------- reviewer implementation review: internal downcasts, real-size check

class _Float32InsideRMSNorm(torch.nn.Module):  # Ouro's unpatched RMSNorm: float64 in, float64 out, float32 inside
    def __init__(self, n):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(n, dtype=torch.float64))
        self.variance_epsilon = 1e-6

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def test_op_audit_catches_an_internal_downcast_that_the_module_audit_misses():
    n = _Float32InsideRMSNorm(8)
    x = torch.randn(2, 8, dtype=torch.float64)
    mod_audit = F.DtypeAudit(n)
    with F.OpAudit() as ops, ops.active():
        y = n(x)
    mod_audit.close()
    assert y.dtype == torch.float64 and mod_audit.bad == {}  # output-level audit is blind to it
    assert any("float32" in k for k in ops.bad)  # op-level audit sees the float32 steps inside
    with F.OpAudit() as ops2:  # outside an active region nothing is recorded
        n(x)
    assert ops2.bad == {} and ops2.n_ops == 0


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                                         reason="no GPU"))])
def test_hidden_state_path_has_no_non_float64_op_when_patched_and_streamed(device):
    from lookahead.ouro_adapter import OuroLoopRunner

    m = _tiny_ouro().to(device)  # (sensitivity to internal downcasts is shown by the RMSNorm test above)
    with F.OpAudit() as ops:
        stream = F.Float64Streaming(m, ops)
        F.patch_float64(m)
        with F.active_methods(ops, OuroLoopRunner, ("start", "step")):
            states = _all_states(m, device)
    assert stream.close()
    assert ops.n_ops > 0 and ops.bad == {}, ops.bad
    assert all(s.dtype == torch.float64 for s in states)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(),
                                                                                         reason="no GPU"))])
def test_real_layer_equivalence_check_passes_for_streaming_and_detects_a_difference(device):
    from lookahead.ouro_adapter import OuroLoopRunner

    m = _tiny_ouro().to(device)
    stream = F.Float64Streaming(m)
    F.patch_float64(m)
    r = OuroLoopRunner(m)
    st = r.start(torch.tensor([[5, 17, 99, 3, 250, 11]], device=device))
    res = F.real_layer_equivalence(r, st, (0, 1))
    assert res["ok"] and res["layer0"]["identical"] and res["final_norm"]["identical"]
    stream.close()

    class Broken(F.Float64Streaming):  # a faulty streamer: float64 weights that are not the stored values
        def _up(self, module, _inputs):
            for p in module.parameters(recurse=False):
                p.data = p.data.to(torch.float64) * (1 + 1e-12)

    m2 = _tiny_ouro().to(device)
    Broken(m2)
    r2 = OuroLoopRunner(m2)
    st2 = r2.start(torch.tensor([[5, 17, 99, 3, 250, 11]], device=device))
    assert not F.real_layer_equivalence(r2, st2, (0,))["ok"]


def test_streamed_weights_are_restored_even_when_a_module_raises():
    class Boom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4).to(torch.bfloat16)
            self.fail = False

        def forward(self, x):
            y = self.lin(x)
            if self.fail:
                raise RuntimeError("deliberate failure inside forward")
            return y

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = Boom()
            self.scale = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

        def forward(self, x):
            return self.inner(x) * self.scale

    m = Outer()
    s = F.Float64Streaming(m)
    x = torch.randn(2, 4, dtype=torch.float64)
    assert m(x).dtype == torch.float64
    m.inner.fail = True
    with pytest.raises(RuntimeError, match="deliberate"):
        m(x)
    # the always-call hooks already restored both the failing module and its parent
    assert all(p.dtype == torch.bfloat16 for p in m.parameters())
    assert s.close() and s.restored_at_close == 0


def test_close_restores_parameters_left_converted_and_still_verifies_bits():
    m = torch.nn.Linear(4, 4).to(torch.bfloat16)
    s = F.Float64Streaming(m)
    m.weight.data = m.weight.data.to(torch.float64)  # simulate a restoration that never happened
    assert s.close() and s.restored_at_close == 1 and m.weight.dtype == torch.bfloat16
