"""C1.3 net-saving overhead for pi1-x, measured on the GPU, split at the deployment boundary (reviewer, C1 implementation
review). Same code path as extraction (PackedScorer.loops, batch size 1), on DEV+ questions; per loop t = 1..3:

  loop      OuroLoopRunner.step on the packed sequence: the recurrent loop itself.
  shared    NOT timed into either rho: Ouro's exit head -> cdf (native-pb's whole controller, rho = 0; pi1-x reuses it).
  score     option scoring, exactly as extraction: lm_head at the scored positions, log-softmax, gather, index_add per
            option. Its row 0 is ALSO the vocabulary log-distribution at the last context token (score_pos[0] is always
            the last context position), so vent needs no further readout (extraction does the same: vocab_logp = logp[0]).
            At the loop where a controller HALTS, this scoring is the answer's scoring, which every controller needs; it is
            incremental only at the other evaluated loops (c1.net_saving).
  rest      vent (entropy of logp[0]), host transfer of the option log-likelihoods, the 9 features exactly as
            extraction's option_record (acc_norm divides by len(choice_text)), standardisation and the per-loop logistic.
            Incremental at every evaluation.
rho_score = score / loop, rho_rest = rest / loop. The first WARMUP questions are excluded.
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
    Returns {benchmark: {rho_score, rho_rest, rho_total, loop_ms, score_ms, rest_ms, n_questions, device}}."""
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
            loop_t, score_t, rest_t = [], [], []
            for qi, q in enumerate(qs[: n_per_bench + WARMUP]):
                packed = build_packed(q.context_enc, q.cont_encs)
                if packed.score_pos[0] != packed.n_context - 1:
                    raise AssertionError("score row 0 must be the last context position (vent reuses it)")
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
                    logp = torch.log_softmax(runner.readout(state.hidden, pos[None])[0], dim=-1)          # score
                    ll = torch.zeros(packed.n_options, device=dev).index_add_(0, opt, logp.gather(1, tgt[:, None]).squeeze(1))
                    sync()
                    d = time.perf_counter()
                    vent = float(-(logp[0].exp() * logp[0]).sum())                                       # rest
                    x = _features(ll.double().cpu().numpy(), lens, q.metric, vent, cdf, len(q.cont_encs), q.packed_len, t)
                    sc, m = models[t]
                    m.predict(((x[None] - sc["mean"]) / sc["sd"]).astype(np.float32))
                    sync()
                    f = time.perf_counter()
                    if qi >= WARMUP:
                        loop_t.append(b - a)
                        score_t.append(d - c)
                        rest_t.append(f - d)
            lm, sm, rm = float(np.mean(loop_t)), float(np.mean(score_t)), float(np.mean(rest_t))
            out[bench] = {"rho_score": sm / lm, "rho_rest": rm / lm, "rho_total": (sm + rm) / lm, "loop_ms": 1e3 * lm,
                          "score_ms": 1e3 * sm, "rest_ms": 1e3 * rm, "n_questions": len(loop_t) // 3, "device": str(dev)}
    return out
