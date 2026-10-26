"""C1.3 net-saving overhead: rho_b = (pi1-x's incremental cost per evaluation) / (cost of one loop), measured on the GPU.

Representative implementation, same code path as extraction (packed scoring, batch size 1). Per loop, on DEV+ questions:
  loop cost  = OuroLoopRunner.step on the packed sequence (one recurrent loop).
  probe cost = everything pi1-x needs at that loop and the native gate does not:
               option scoring (lm_head at the scored positions, log-softmax, gather and sum per option -> conf, ent under the
               benchmark's metric), the full-vocabulary log-softmax at the last context token (-> vent), assembling the 9
               features, standardising them and the per-loop logistic.
The native gate's cdf comes from Ouro's own exit head, which the model runs anyway: rho = 0 (plan 5a), so its cost is in
neither term. Timed with torch.cuda.synchronize around each part; the first WARMUP questions are excluded.
"""
import math

import numpy as np

WARMUP = 5


def _features(ll, choice_lens, metric, vent, cdf, n_options, n_tokens, t):
    lln = ll / choice_lens
    s = lln if metric == "acc_norm" else ll
    s = s - s.max()
    logp = s - math.log(float(np.exp(s).sum()))
    p = np.exp(logp)
    x = np.zeros(9, np.float32)
    x[:6] = (p.max(), -(p * logp).sum(), vent, cdf, n_options, math.log(max(n_tokens, 1.0)))
    x[6 + t] = 1.0
    return x


def measure_rho(model, questions: dict, models: list, n_per_bench: int = 200) -> dict:
    """questions: {benchmark: [Question]} (DEV+). models: pi1-x's per-loop (scaler, LinearModel), loops 1-3.
    Returns {benchmark: {rho, loop_ms, probe_ms, n_questions}}."""
    import time

    import torch

    from lookahead.ouro_adapter import NativeGate
    from lookahead.packed import PackedScorer, build_packed

    dev = next(model.parameters()).device
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    scorer = PackedScorer(model)
    runner = scorer.runner
    out = {}
    with torch.no_grad():
        for bench, qs in questions.items():
            loop_t, probe_t = [], []
            for qi, q in enumerate(qs[: n_per_bench + WARMUP]):
                packed = build_packed(q.context_enc, q.cont_encs)
                state = scorer.start(packed)
                pos = torch.tensor(packed.score_pos, device=dev)
                tgt = torch.tensor(packed.score_tgt, device=dev)
                opt = torch.tensor(packed.score_opt, device=dev)
                last = torch.tensor([packed.n_context - 1], device=dev)
                lens = np.array([float(len(c)) for c in q.choice_text])
                gate = NativeGate(runner.total_loops, 1)
                for t in range(3):                       # the controller evaluates at loops 1-3
                    sync()
                    a = time.perf_counter()
                    runner.step(state)
                    sync()
                    b = time.perf_counter()
                    cdf = float(gate.update(runner.gate_logit(state.hidden, last).float().cpu())[0])   # free (rho = 0)
                    sync()
                    c = time.perf_counter()
                    logp = torch.log_softmax(runner.readout(state.hidden, pos[None])[0], dim=-1)
                    ll = torch.zeros(packed.n_options, device=dev).index_add_(0, opt, logp.gather(1, tgt[:, None]).squeeze(1))
                    vl = torch.log_softmax(runner.readout(state.hidden, last)[0], -1)
                    vent = float(-(vl.exp() * vl).sum())
                    x = _features(ll.double().cpu().numpy(), lens, q.metric, vent, cdf, len(q.cont_encs), q.packed_len, t)
                    sc, m = models[t]
                    m.predict(((x[None] - sc["mean"]) / sc["sd"]).astype(np.float32))
                    sync()
                    d = time.perf_counter()
                    if qi >= WARMUP:
                        loop_t.append(b - a)
                        probe_t.append(d - c)
            lm, pm = float(np.mean(loop_t)), float(np.mean(probe_t))
            out[bench] = {"rho": pm / lm, "loop_ms": 1e3 * lm, "probe_ms": 1e3 * pm, "n_questions": len(loop_t) // 3,
                          "device": str(dev)}
    return out
