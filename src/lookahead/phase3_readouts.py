"""Phase 3 amendment A2: published post-hoc readouts as task-aware baselines (pure logic; no data access).

Statistics follow Popescu et al. 2026 (arXiv 2607.20519, Appendix B.6) and Ouro section 5.4.1, operationalised on the records
our extraction stores, per question and loop t = 1..4:
  p_t  option distribution (softmax of the option scores; `p`),  z_t  option scores (`opt_lln` for acc_norm tasks, `opt_ll`
  for acc tasks, i.e. the scores whose softmax is p_t),  h_t  hidden state at the last context token (float16 copy).
Every statistic is oriented so that HALT WHEN s_t >= tau:
  r-conf    max_k p_t(k)                                   loops 1-3
  r-margin  z_t(1st) - z_t(2nd)                            loops 1-3   (logit gap, as in the paper, on option scores)
  r-ent     -H(p_t)                                        loops 1-3
  r-kl      -KL(p_t || p_{t-1})                            loops 2-3   (paper's direction: current vs previous)
  r-dlogit  -||z_t - z_{t-1}||_2                           loops 2-3
  r-hl2     -||h_t - h_{t-1}||_2                           loops 2-3   (Ouro 5.4.1; paper's "hidden L2")
  r-hcos    cos(h_t, h_{t-1})                              loops 2-3   (paper names it but does not define it: ours)
Exit rule (A2, frozen with the reviewer's wording): at loop t < m_b, halt if s_t is defined and s_t >= tau_b; an undefined
statistic (loop 1 for the convergence readouts) never fires, so the policy continues; at the cap m_b it halts regardless.
Failed questions run 4 loops and count wrong (as in Gates 1-2).
"""
import math

import numpy as np

from lookahead.gate1 import LOSS_LIMIT, LOOPS, TOL, grid_thresholds

READOUTS = ("r-conf", "r-margin", "r-ent", "r-kl", "r-dlogit", "r-hl2", "r-hcos")
CONVERGENCE = {"r-kl", "r-dlogit", "r-hl2", "r-hcos"}  # undefined at loop 1


def _logsoftmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    return z - (m + math.log(np.exp(z - m).sum()))


def question_stats(name: str, z: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
    """s for loops 1..3 of one question. z: [4, K] option scores; h: [4, d] hidden states (hidden readouts only)."""
    out = np.full(LOOPS - 1, np.nan)
    z = np.asarray(z, dtype=np.float64)
    lp = np.stack([_logsoftmax(z[t]) for t in range(LOOPS)])
    p = np.exp(lp)
    for t in range(LOOPS - 1):  # index t = loop t+1
        if name == "r-conf":
            out[t] = p[t].max()
        elif name == "r-margin":
            s = np.sort(z[t])[::-1]
            out[t] = s[0] - s[1]
        elif name == "r-ent":
            out[t] = float((p[t] * lp[t]).sum())  # = -H(p_t)
        elif t == 0:
            continue  # convergence readouts: undefined at loop 1
        elif name == "r-kl":
            out[t] = -float((p[t] * (lp[t] - lp[t - 1])).sum())
        elif name == "r-dlogit":
            out[t] = -float(np.linalg.norm(z[t] - z[t - 1]))
        elif name in ("r-hl2", "r-hcos"):
            a, b = np.asarray(h[t], dtype=np.float64), np.asarray(h[t - 1], dtype=np.float64)
            if name == "r-hl2":
                out[t] = -float(np.linalg.norm(a - b))
            else:
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                out[t] = float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan
        else:
            raise ValueError(f"unknown readout {name!r}")
    return out


def stat_matrix(name: str, scores: list, hidden=None) -> np.ndarray:
    """[n, 3] statistic for loops 1..3. scores: per-question [4, K] arrays (K may differ between questions)."""
    return np.stack([question_stats(name, scores[i], None if hidden is None else hidden[i]) for i in range(len(scores))])


def exit_loops(S: np.ndarray, ok: np.ndarray, tau: float, cap: int) -> np.ndarray:
    """Exit loop per question under the A2 rule. S: [n, 3]; tau = +inf gives fixed depth `cap`."""
    if cap not in (1, 2, 3, 4):
        raise ValueError("cap must be in 1..4")
    n = S.shape[0]
    e = np.full(n, cap, dtype=np.int64)
    if math.isfinite(tau):
        fire = np.zeros((n, LOOPS - 1), dtype=bool)
        fire[:, : cap - 1] = np.nan_to_num(S[:, : cap - 1], nan=-np.inf) >= tau  # undefined never fires
        hit = fire.any(axis=1)
        e = np.where(hit, fire.argmax(axis=1) + 1, cap)
    return np.where(ok, e, LOOPS).astype(np.int64)


def outcome(correct: np.ndarray, ok: np.ndarray, e: np.ndarray):
    """(saving, loss in pp) for one benchmark; correct: [n, 4] 0/1; failed questions count wrong."""
    c = np.where(ok, correct[np.arange(len(e)), e - 1], 0)
    acc4 = 100.0 * np.where(ok, correct[:, LOOPS - 1], 0).mean()
    return 1.0 - e.mean() / LOOPS, acc4 - 100.0 * c.mean()


def grid(S_parts: list, ok_parts: list) -> list:
    """51-point grid: 50 quantiles of the statistic pooled over ok questions and defined loops (repeats removed), + inf."""
    vals = np.concatenate([S[ok].ravel() for S, ok in zip(S_parts, ok_parts)])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("no defined values for the grid")
    return grid_thresholds(vals) + [math.inf]


def select(S: np.ndarray, ok: np.ndarray, correct: np.ndarray, taus: list) -> dict:
    """Plan section 5 rule for one benchmark: among (tau, m) with loss <= 1 pp, max saving; ties (1e-12) -> smaller loss ->
    fixed depth (tau = inf) preferred -> larger tau -> smaller m. Fixed depth 4 always qualifies (loss 0)."""
    rows = []
    for tau in taus:
        for m in (1, 2, 3, 4):
            sav, loss = outcome(correct, ok, exit_loops(S, ok, tau, m))
            rows.append({"tau": tau, "m": m, "saving": sav, "loss": loss})
    q = [r for r in rows if r["loss"] <= LOSS_LIMIT + TOL]
    best = max(r["saving"] for r in q)
    c = [r for r in q if r["saving"] >= best - TOL]
    lo = min(r["loss"] for r in c)
    c = [r for r in c if r["loss"] <= lo + TOL]
    c.sort(key=lambda r: (0 if math.isinf(r["tau"]) else 1, -r["tau"] if math.isfinite(r["tau"]) else 0, r["m"]))
    return c[0]


def nested_comparator(folds: list) -> dict:
    """D2 comparator under A2 option (a), without optimism: in each outer fold the strongest ELIGIBLE baseline is chosen on
    the SELECTION folds (eligible = qualified under the per-benchmark rule there; highest selection macro net saving, ties ->
    name order), and only its HELD-OUT saving is used.

    folds: list of {baseline: {"eligible": bool, "selection_saving": float, "heldout_saving": float}}.
    Returns the chosen baseline per fold and the mean held-out saving of the chosen comparators."""
    chosen = []
    for f in folds:
        el = sorted((b for b, v in f.items() if v["eligible"]), key=lambda b: (-f[b]["selection_saving"], b))
        if not el:
            raise ValueError("no eligible baseline in a fold (B_pb always qualifies, so this is a bug)")
        chosen.append(el[0])
    return {"chosen": chosen, "heldout_saving": float(np.mean([f[b]["heldout_saving"] for f, b in zip(folds, chosen)]))}
