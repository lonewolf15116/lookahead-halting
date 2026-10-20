# Lookahead Halting — Protocol v1.1

Frozen 2026-10-20, before any extraction. Changes after this commit go in DECISIONS.md with a reason.

## Goal and claim

Does forecasting how the next loop will move tell you something about the value of another loop that existing halting signals miss? Two hypotheses are tested separately, because the first can hold while the second fails.

- **H1, forecastability:** the loop-t hidden state predicts next-loop change (KL, answer flip) better than confidence, entropy, Ouro's gate, and matched-capacity probes do.
- **H2, halting utility:** a policy using that forecast improves accuracy vs. executed loops and wall-clock latency over deployable baselines, with probe cost counted.
- **Three targets, studied together from Phase 1:** next-loop change D_t = KL(p_t || p_t+1), answer flip F_t, and loss gain G_t = l_t - l_t+1. Whether change tracks gain is a central result, not a Phase 4 extra.
- **Model:** frozen, public Ouro-1.4B (4 trained loops); no pretraining.
- **Outcomes, all reportable:** (A) Pareto improvement -> workshop paper; (B) predictable but not useful -> diagnostic paper on why; (C) not predictable -> negative-result report.

## Roadmap

Six phases, roughly 6-8 weeks part-time (estimate).

| Phase | Work | Gate after it |
| --- | --- | --- |
| 0 Setup | Load Ouro-1.4B; check per-loop accuracy against the paper | Gate 0, Reproduces: per-loop accuracy reproduced; peak VRAM and loop runtime recorded |
| 1 Extract | Cache per-loop states; compute KL, flips and loss gain | Gate 1, Headroom: stable-correct oracle saves real loops beyond cheap baselines |
| 2 Forecast | Train linear and MLP probes on next-loop KL and flips | Gate 2, Signal: lookahead adds signal beyond confidence and matched probes |
| 3 Halt | True early-exit runtime; Pareto and latency vs. baselines | Gate 3, Verdict: Pareto gain with probe cost counted; else diagnostic paper |
| 4 Stress | Loops 5-8, OOD tasks, loop-1 commit, error taxonomy | none |
| 5 Write-up | Workshop paper or arXiv preprint | none |

Gate 1 is the cheapest kill point: if Ouro's loops leave little to save, stop before building any probe.

## Halting unit and cost accounting

- For multiple-choice scoring the whole prompt passes through each loop together, so halting is **per question**, not per token. Early exit then saves real prefill compute and avoids per-token KV-cache problems.
- Phase 3 needs a true loop-by-loop runtime that skips unused loops. Latency from a run that executes all loops and picks one afterwards does not count.
- Cost = executed loops + probe time, measured in wall-clock as well as loop count.

## Oracles vs. deployable policies

| Reference | Deployable? | Use |
| --- | --- | --- |
| Earliest-correct oracle | No, uses labels | Optimistic upper bound |
| Earliest-stable-correct oracle | No, uses labels | Headroom for Gate 1: correct at t and every later loop up to 4 |
| Retrospective measured-KL | No, sees p_t+1 for free | Analysis only; never shown as an efficiency baseline |
| Deployable measured-KL | Yes | Pays for loop t+1 to decide, so it exits at t+1 at the earliest |

## Baselines (matched feature access and tuning budget)

| Baseline | Role |
| --- | --- |
| Fixed 1, 2, 3, 4 loops | Non-adaptive reference |
| Confidence and entropy thresholds | Cheapest strong baselines |
| Ouro native ponder gate | Model's own learned gate |
| Deployable measured-KL | Cost-accounted convergence rule |
| Linear/MLP probe predicting current correctness | Controls for learned-gate capacity |
| Linear/MLP probe predicting loss gain G_t | Separates change from gain |
| Small model on confidence, entropy, gate value only | Tests whether h_t adds anything |
| **Linear/MLP probe predicting D_t and F_t** | **Proposed method** |

## Targets and storage

- KL is computed two ways: over the answer options (the decision-relevant distribution) and over the full vocabulary at the answer position, both with log_softmax.
- Store per question per loop: hidden state, option logits, the scalar targets, confidence, entropy, gate value, runtime. Full-vocabulary logits are reduced to targets on the fly, not stored.
- Keep raw per-question, per-loop outputs so every aggregate can be rebuilt.

## Splits and statistics

- Split by question, 60/20/20 train/validation/test, before any probe rows exist; seeds recorded in a manifest.
- Probes, feature standardisation and thresholds are fit on train/validation only. Test is touched once per gate.
- Primary metric: accuracy vs. executed loops at matched average compute, per benchmark and pooled. Latency reported alongside.
- Flip prediction: prevalence, average precision, recall at low false-negative rates, calibration. AUROC alone is not enough.
- Uncertainty: paired question-level bootstrap 95% intervals. Report the full threshold grid, not the best point.
- Shared probe across loops vs. one probe per loop, both reported.

## Gate thresholds (confirmed 2026-10-20, frozen)

| Gate | Pass if |
| --- | --- |
| 0 | Per-loop accuracy within 2 points of Ouro's reported figure where one exists; extraction leaves predictions unchanged; peak VRAM and per-loop runtime recorded |
| 1 | Stable-correct oracle saves at least 25% of loops vs. fixed 4, and the best cheap baseline captures under 75% of that saving at no more than 1 point accuracy loss |
| 2 | On validation, lookahead flip AP beats the best matched cheap predictor by at least 0.05, bootstrap interval above zero |
| 3 | On test, at a budget saving at least 15% of loops, lookahead accuracy beats the best deployable baseline, paired interval above zero, probe latency included |

Failure policy: OOMs, timeouts and unparseable answers are logged and counted as errors, never silently dropped. Loops 5-8 stay exploratory.

## References

- [Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates (2607.20519)](https://arxiv.org/abs/2607.20519)
- [Ouro: Scaling Latent Reasoning via Looped Language Models (2510.25741)](https://arxiv.org/abs/2510.25741)
- [Ouro-1.4B weights](https://huggingface.co/ByteDance/Ouro-1.4B)
- [Huginn (2502.05171)](https://arxiv.org/abs/2502.05171)
- [CALM (2207.07061)](https://arxiv.org/abs/2207.07061)
- [Mixture-of-Recursions (2507.10524)](https://arxiv.org/abs/2507.10524)
- [Fixed-Point Reasoners (2606.18206)](https://www.alphaxiv.org/audio/2606.18206v1)
