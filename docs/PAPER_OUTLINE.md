# Paper outline (draft 1) and reference verification

Working title: **When Does Adaptive Halting Win? Selection-Rule Sensitivity and Distribution-Shift Failures in Looped Language Models**

Status: this is an outline only, with no prose drafted. Results come from `results/phase3/g3a_report.json` (49ee7ace…) and `results/c1/test_report.json`
(02ba4b26…). The Phase 3 TEST set is spent; no further confirmatory claims can be drawn from it.

## Evidence labels (used in every section, including the abstract)

- **[C]**: confirmatory. Pre-registered primary endpoint, decided on sealed TEST.
- **[S]**: pre-specified secondary. Listed in the frozen protocol before TEST was opened.
- **[P]**: post hoc emphasis or interpretation. Chosen or framed after seeing TEST.
- **[E]**: exploratory. Development data only (E1, D-a/D-a2 diagnostics).

| RQ | Question | Evidence |
|---|---|---|
| RQ1 | Does the accuracy-risk selection rule change which halting method saves more compute? | [C] C1 PASS under the combined risk rule. G3a STOP under the point-only rule (development/validation). [S] the point-only rule replicated on TEST: −0.022. |
| RQ2 | Does an added trained probe beat (a) Ouro's trained native gate and (b) a training-free Geiping-style KL exit? | (a) [C]. (b) [S] the comparison was pre-specified; making KL exit the headline comparator is [P]. |
| RQ3 | Do development-selected policies keep their accuracy budget on unfamiliar benchmarks? | [S] PIQA and SIQA only. This is not a general robustness assessment. |

## 1. Introduction

- Looped or recurrent-depth LMs (Ouro, Huginn) make compute depth a runtime choice. The question is when to stop looping.
- Comparisons of halting methods usually fix one operating rule. We show that the rule decides the ranking.
- Contributions (each labelled):
  1. A pre-registered comparison under two selection rules on a frozen Ouro-1.4B, with opposite outcomes: [C] and [S].
  2. The added trained probe shows no demonstrated advantage over a published training-free KL exit: [S], with [P] emphasis.
  3. One out-of-distribution budget failure (SIQA, 3.26 pp loss): [S].
  4. A guarded single-use TEST protocol (hash-pinned code bundle, exclusive marker), offered as a methodological contribution.
- State the non-claims explicitly: no new halting signal is claimed superior to prior art; there is no measured latency; there is one model.

## 2. Related work (verified, see the table below)

- **Learned halting for recurrent computation.** ACT (Graves 2016) uses a cumulative halting unit and a ponder cost. PonderNet (Banino et al. 2021)
  uses a probabilistic halting distribution and a geometric prior. Ouro's native gate (Zhu et al. 2025) follows the same family: a per-loop exit
  probability, the CDF, and a Q-exit threshold q.
- **Training-free convergence exits in recurrent depth.** Geiping et al. (NeurIPS 2025) section 6.1 stops when the KL between successive steps is
  below 5×10⁻⁴, per generated token. Our r-kl applies the same signal per question, with the threshold selected on development data under the risk rule.
- **Stability or patience exits over layers.** PABEE (Zhou et al. 2020) uses prediction agreement across consecutive layer classifiers. It is the
  ancestor of our stability target S_t, but it works over the layers of a fixed-depth encoder, not over loops.
- **Token-wise adaptive depth in recurrent LMs.** AdaPonderLM (Song et al. 2026, preprint) uses per-iteration gates trained with a ponder penalty.
  It is the closest setting to ours, but it is trained from scratch or by continued pretraining, not retrofitted.
- **Early exit of reasoning chains.** SpecExit (Yang et al., ICML 2026) and NEAT (Liu et al., Findings ACL 2026) truncate chain-of-thought text.
  This is a different unit (generated reasoning length, not latent depth). Cite them as adjacent work and do not compare numbers.
- **Risk-controlled selection.** A references gap remains to fill: conformal or risk control (e.g. Learn-then-Test, conformal risk control) and
  calibrated early exit. These still need finding and verifying.

## 3. Setting: model, signals, selection rules

- Ouro-1.4B frozen in bf16, with 4 loops. Multiple-choice scoring with packed options. A decision is made per question, at the last context token.
- **Signals:**
  - native gate (CDF threshold);
  - π1-x, a per-loop L2-logistic probe on 9 cheap features (cdf, conf, ent, vent, n_options, log_n_tokens, loop one-hot) trained on the stability target;
  - the r-* family of untrained rules: r-kl, r-ent, r-conf, r-margin, r-dlogit, r-hcos, r-hl2;
  - B_pb, the best fixed depth.
- **Selection:** per benchmark, a pair (τ_b, m_b) on a 51-point pooled grid plus ∞. Two rules:
  - **point:** loss ≤ 1 pp (the G3a rule);
  - **combined risk:** point loss ≤ 1 pp AND the bootstrap 95% upper bound ≤ 1 pp (the E1 `select_risk` rule, frozen for C1).
- **Cost:**
  - gross saving = 1 − e/4;
  - net saving adds ρ_score and ρ_rest, measured once on the GPU at freeze time. It is an estimated deployment cost, not latency.

## 4. Design and pre-registration

- The Phase 3 data split (manifest b2df1930…), development and validation, and the sealed TEST (8,517 questions; primary sets OBQA 3000,
  CSQA 1000, HellaSwag 750; out of distribution PIQA 1837, SIQA 1930).
- The sequence: G3a (plan v1.1) → STOP → E1 [E] → C1 protocol frozen (4f38b8b7…, tag `c1-protocol`) → freeze on development data → guarded TEST run.
- The C1 decision rule, quoted verbatim, including that "PASS is not a 95% claim that effect ≥ 0.02".
- Integrity: pinned hashes, the 24-file code bundle hash c0902af3…, the exclusive marker, the metadata-only preflight, and the refused second run.
- Deviations: D-a PIQA float32 (numerical, D-a2), and the console transcript lost to an overwrite.

## 5. Results: RQ1 and the native-gate comparison

| Rule | π1-x − native-pb (gross macro saving) | Evidence |
|---|---|---|
| point-only (G3a, validation) | −0.024 | G3a STOP |
| point-only (TEST) | −0.022 [−0.026, −0.018] | [S] |
| combined risk (TEST) | **+0.053 [+0.049, +0.057]** | [C] PASS |
| combined risk, net (TEST) | +0.038 [+0.033, +0.042] | [S], estimated |

- **Figure 1:** saving against loss for both methods under both rules (the "rank reversal" figure).
- **Table:** exit histograms per benchmark.
- **Mechanism to discuss:** this is [P] interpretation, to be checked with [E] development analyses. Under the point rule the native gate can
  take more risk on some benchmarks (TEST point loss 1.57 pp on OBQA). Under the upper-bound constraint the native CDF is less finely graded
  near the exit boundary.
- The TEST upper bounds exceed 1 pp, and this goes in the main text, not a footnote:
  - π1-x: CSQA 1.10, HellaSwag 1.60;
  - native-pb: HellaSwag 1.33;
  - B_pb: HellaSwag 1.60.

## 6. Results: RQ2 and RQ3

- **RQ2** [S], with [P] emphasis:

  | comparison | Δ | 95% CI |
  |---|---|---|
  | π1-x vs r-kl | +0.0012 | [−0.0041, +0.0065] |

  The reselecting bootstrap picked r-kl in 10,000 of 10,000 resamples. There is no demonstrated advantage, and no equivalence is claimed.
  The near-tie was also seen on development data (0.246 vs 0.243). All other r-* rules trail by 0.060–0.090.
- **RQ3** [S]:

  | benchmark | method | loss (pp) | upper bound | saving |
  |---|---|---|---|---|
  | SIQA | π1-x | 3.26 | 4.72 | 0.481 |
  | SIQA | native-pb | −0.05 | | |
  | PIQA | π1-x | 0.71 | 1.52 | |
  | PIQA | native-pb | 0.11 | | |

  π1-x halts at loop 2 for 1,785 of 1,930 SIQA questions. No generalisation is made beyond these two benchmarks.

## 7. Discussion, limitations, threats to validity

- Rankings that depend on the selection rule mean a halting paper should report results under more than one operating rule, with upper bounds.
- **Limitations:**
  - one model and one size;
  - multiple-choice scoring, not generation;
  - 4 loops;
  - estimated cost rather than latency;
  - HellaSwag n = 750 limits upper-bound precision;
  - the per-benchmark selection uses benchmark identity.
- **Threats:** r-kl being TEST-selected [P]; the D-a PIQA numerical deviation; the TEST set is now spent.

## 8. Conclusion

- One paragraph that restates the [C], [S] and [P] split.
- Future work, each item needing new held-out data: a pre-registered KL-exit vs native-gate test, an out-of-distribution guard, and Ouro-2.6B.

---

## Reference verification (Sonnet sub-agent, primary sources opened; 2026-10-27)

| Ref | Verified citation | Unit that halts / signal / training | Status |
|---|---|---|---|
| ACT | Graves, *Adaptive Computation Time for Recurrent Neural Networks*, arXiv:1603.08983 (2016), preprint, no venue | RNN steps per input; cumulative sigmoid halting unit to 1−ε; trained with ponder cost τ | VERIFIED |
| PonderNet | Banino, Balaguer, Blundell, *PonderNet: Learning to Ponder*, ICML 2021 Workshop (AutoML); arXiv:2107.05407 | recurrent steps per example; conditional λ_n with a geometric-prior KL; trained; sampled halting at evaluation | VERIFIED (cite as a workshop paper) |
| PABEE | Zhou, Xu, Ge, McAuley, Xu, Wei, *BERT Loses Patience: Fast and Robust Inference with Early Exit*, NeurIPS 2020, pp. 18330–18341; arXiv:2006.04152 | layers per sequence; patience t on agreement between consecutive layer predictions; internal classifiers trained | VERIFIED |
| Geiping | Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein, *Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach*, NeurIPS 2025; arXiv:2502.05171 | recurrent steps per token; KL between successive steps < 5×10⁻⁴ (section 6.1); zero-shot | VERIFIED (section 6.1 passage) |
| SpecExit | Yang, Bai, Liu, Yu, Fan, Dang, Zhang, Liu, Zhu, Chen, *SpecExit: Accelerating Large Reasoning Model via Speculative Exit*, ICML 2026, PMLR 306:144187–144210; arXiv:2509.24248 | reasoning chain at step-split tokens; trained heads in the multi-token-prediction draft model predict confidence, progress and remaining length; hand-set thresholds | VERIFIED (details from the arXiv version) |
| NEAT | Liu, Liu, Yang, Wang, Zhang, Feng, Zhang, Wang, *NEAT: Neuron-Based Early Exit for Large Reasoning Models*, Findings of ACL 2026, pp. 24616–24627; doi:10.18653/v1/2026.findings-acl.1231 | reasoning chain per decoding step; activation of termination-promoting FFN neurons; training-free with one calibration | VERIFIED |
| AdaPonderLM | Song, Li, Wang, Zeng, Song, Wang, Xu, He, Lin, *AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth*, arXiv:2603.01914 (2026), preprint | per token, per recurrent iteration; per-iteration MLP gates with a monotone mask and KV reuse; trained with a ponder penalty | VERIFIED (cite as a preprint) |
| Ouro | Zhu et al. (36 authors), *Scaling Latent Reasoning via Looped Language Models*, arXiv:2510.25741 (2025), preprint | per loop: λ_t = σ(Linear(h_t)), CDF, Q-exit at CDF ≥ q; Stage I with an entropy-regularised exit distribution, Stage II gate training on loss improvement | VERIFIED. Whether the exit is per token or per sequence is not stated in the text read; check before claiming |

Corrections to the third-party list:
- PonderNet is a workshop paper, not ICML main.
- AdaPonderLM and Ouro have no venue.
- Ouro is "Zhu et al.", not "ByteDance Seed".
- Strip the `utm_source` parameters.

Before final text, check these against the PDFs, because the formula-level numbers came through a summarising fetch: ε = 0.01 (ACT); t = 6 (PABEE); 5×10⁻⁴ (Geiping); τ = 10⁻⁴ (AdaPonderLM); k = 300 and thresholds 0.6 / 0.2 / 0.4 (NEAT); the SpecExit thresholds; Ouro's Stage II constants.
Still to find: references on risk-controlled selection (Learn-then-Test, conformal risk control) and calibrated early exit.
