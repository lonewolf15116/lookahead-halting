# Paper outline (draft 1) and reference verification

Working title: **When Does Adaptive Halting Win? Selection-Rule Sensitivity and Distribution-Shift Failures in Looped Language Models**

Status: outline v2 (reviewer revisions applied), with no prose drafted. Results come from `results/phase3/g3a_report.json` (49ee7ace…) and `results/c1/test_report.json`
(02ba4b26…). The Phase 3 TEST set is spent; no further confirmatory claims can be drawn from it.

## Evidence labels (used in every section, including the abstract)

- **[C]**: confirmatory. Pre-registered primary endpoint, decided on sealed TEST.
- **[S]**: pre-specified secondary. Listed in the frozen protocol before TEST was opened.
- **[P]**: post hoc emphasis or interpretation. Chosen or framed after seeing TEST.
- **[E]**: exploratory. Development data only (E1, D-a/D-a2 diagnostics).

| RQ | Question | Evidence |
|---|---|---|
| RQ1 | Does the accuracy-risk selection rule change which halting method is preferred? Each rule selects its own operating points (τ_b, m_b) per method, so the comparison is between rule-dependent selections, not the same frozen policies reversing. | [C] C1 PASS under the bootstrap-screened rule. G3a STOP under the point-only rule (development/validation). [S] the point-only rule replicated on TEST: −0.022. |
| RQ2 | Does an added trained probe beat (a) Ouro's trained native gate and (b) a training-free Geiping-style KL exit? | (a) [C]. (b) [S] the comparison was pre-specified; making KL exit the headline comparator is [P]. |
| RQ3 | Do development-selected policies keep their accuracy budget on unfamiliar benchmarks? | [S] PIQA and SIQA only. This is not a general robustness assessment. |

## 1. Introduction

- Looped or recurrent-depth LMs (Ouro, Huginn) make compute depth a runtime choice. The question is when to stop looping.
- Comparisons of halting methods usually fix one operating rule. We show that the rule decides the ranking.
- Contributions (each labelled):
  1. A pre-registered comparison under two selection rules on a frozen Ouro-1.4B. The method preferred by the comparison reverses with the rule: [C] and [S]. Each rule selects different operating points, so it is the selection-rule-dependent comparison that reverses, not the performance of fixed policies.
  2. The added trained probe shows no demonstrated advantage over a published training-free KL exit: [S], with [P] emphasis. This is non-superiority, not equivalence; no equivalence margin was pre-specified or tested.
  3. One out-of-distribution budget failure (SIQA, 3.26 pp loss): [S].
  (Not a contribution: the guarded single-use TEST procedure, with a hash-pinned code bundle and an exclusive marker, is described in section 4 as part of the methodological rigor, not claimed as novel.)
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
- **Risk-controlled early exit is prior work, and it goes in the core comparison.** Fast yet Safe (Jazbec et al., NeurIPS 2024) already selects
  early-exit thresholds with formal risk control (a UCB bound, with probability 1−δ, under i.i.d. calibration data, on a performance gap relative to the full model).
  Learn then Test (Angelopoulos et al., AoAS 2025) frames threshold selection as multiple testing with FWER control and finite-sample p-values.
  Conformal Risk Control (Angelopoulos et al., ICLR 2024) controls expected monotone loss. The paper must not imply that risk-constrained exit
  selection is new.
- **Our rule is a heuristic screen, not risk control.** C1's development-set rule (point loss ≤ 1 pp AND bootstrap 95% upper bound ≤ 1 pp) differs in four ways:
  - the bootstrap is asymptotic, not a finite-sample bound;
  - there is no multiplicity control across the 51-point grid (plus ∞);
  - selection and bound use the same data (winner's curse);
  - there is no (α, δ) statement.

  Call it "bootstrap-screened selection" throughout, never "risk-controlled". C1 inherits none of LTT's, CRC's or Fast-yet-Safe's guarantees. This
  also gives a [P] explanation of the TEST upper bounds exceeding 1 pp, to be stated as interpretation only.
- **Calibration is not the same as early-exit quality.** Kubaty et al. (ICML 2026) show that calibration alone does not characterise early-exit
  performance. This is relevant to why a better-ranked signal (π1-x AUROC, E1) does not automatically mean more savings.

## 3. Setting: model, signals, selection rules

- Ouro-1.4B frozen in bf16, with 4 loops. Multiple-choice scoring with packed options. A decision is made per question, at the last context token.
- **Ouro's exit unit, resolved from the released code** (`modeling_ouro.py`, sha256 c5c68fbb…). The released implementation runs all
  `total_ut_steps` loops, applies `early_exit_gate` (Linear(hidden, 1)) at every token position, and with `exit_threshold` q chooses, for each token
  position, the first loop whose cumulative exit probability is ≥ q. It then reads out that loop's hidden state. So the released Q-exit is per token
  and selects a readout without skipping compute. The paper's text (section 3.1) states the goal "per token and per example" but does not fix the
  inference unit. Our use is an adaptation: the gate's CDF at a single position (the last context token) gives one decision per question, and loops
  after the exit are counted as saved. State both points plainly in section 3.
- **Signals:**
  - native gate (CDF threshold);
  - π1-x, a per-loop L2-logistic probe on 9 cheap features (cdf, conf, ent, vent, n_options, log_n_tokens, loop one-hot) trained on the stability target;
  - the r-* family of untrained rules: r-kl, r-ent, r-conf, r-margin, r-dlogit, r-hcos, r-hl2;
  - B_pb, the best fixed depth.
- **Selection:** per benchmark, a pair (τ_b, m_b) on a 51-point pooled grid plus ∞. Two rules:
  - **point:** loss ≤ 1 pp (the G3a rule);
  - **bootstrap-screened** (the protocol calls it the "combined risk rule"): point loss ≤ 1 pp AND the bootstrap 95% upper bound on development data ≤ 1 pp
    (the E1 `select_risk` rule, frozen for C1). It is an empirical screen with no formal risk-control guarantee (see section 2).
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
| bootstrap-screened (TEST) | **+0.053 [+0.049, +0.057]** | [C] PASS |
| bootstrap-screened, net (TEST) | +0.038 [+0.033, +0.042] | [S], estimated |

- **Figure 1:** saving against loss for both methods under both rules, marking the operating point each rule selects (the "rule-dependent preference" figure).
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

  The reselecting bootstrap picked r-kl in 10,000 of 10,000 resamples. There is no demonstrated advantage. This is not evidence of equivalence: no equivalence margin was pre-specified, and the CI only shows that differences larger than about ±0.65 pp of saving are unlikely.
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
| Ouro | Zhu et al. (36 authors), *Scaling Latent Reasoning via Looped Language Models*, arXiv:2510.25741 (2025), preprint | per loop: λ_t = σ(Linear(h_t)), CDF, Q-exit at CDF ≥ q; Stage I with an entropy-regularised exit distribution, Stage II gate training on loss improvement (k = 50, γ = 0.005, section 3.4) | VERIFIED. Exit unit: per token position in the released code (all loops computed, readout selected); the paper text does not fix it |

Corrections to the third-party list:
- PonderNet is a workshop paper, not ICML main.
- AdaPonderLM and Ouro have no venue.
- Ouro is "Zhu et al.", not "ByteDance Seed".
- Strip the `utm_source` parameters.

Additional references (verified 2026-10-27, Sonnet sub-agent):

| Ref | Verified citation | Relevance | Status |
|---|---|---|---|
| Fast yet Safe | Jazbec, Timans, Hadži Veljković, Sakmann, Zhang, Naesseth, Nalisnick, *Fast yet Safe: Early-Exiting with Risk Control*, NeurIPS 2024; arXiv:2405.20915 | Thresholds for early-exit networks selected by risk control: UCB at 1−δ (δ = 0.1 in the experiments); performance-gap and consistency risks; i.i.d. calibration data; distribution shift left to future work. Core comparison. | VERIFIED |
| Learn then Test | Angelopoulos, Bates, Candès, Jordan, Lei, *Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control*, Annals of Applied Statistics 19(2):1641–1662 (2025), doi:10.1214/24-AOAS1998; arXiv:2110.01052 | Threshold selection as multiple testing; Hoeffding–Bentkus p-values; Bonferroni or fixed-sequence FWER control; P(R ≤ α) ≥ 1−δ; no monotonicity needed | VERIFIED (journal details from the Project Euclid PDF URL and title; the page itself was blocked) |
| Conformal Risk Control | Angelopoulos, Bates, Fisch, Lei, Schuster, *Conformal Risk Control*, ICLR 2024; arXiv:2208.02814 | E[loss] ≤ α for monotone bounded losses under exchangeability; extensions to covariate shift and total-variation shift | VERIFIED |
| Kubaty et al. | Kubaty, Szatkowski, Choczyński, Nalisnick, Wójcik, *Rethinking Calibration for Early-Exit Neural Networks*, ICML 2026, PMLR 306:61372–61389 | Calibration is not sufficient for early-exit quality; proposes early-exit failure prediction | VERIFIED (summary of the PMLR page) |

Formula-level constants, checked against the primary text (quotes as returned by the fetch tool):
- ACT ε = 0.01 (section 2)
- PABEE patience t = 6 (sections 4.3 and 4.5)
- Geiping KL < 5×10⁻⁴ (section 6.1)
- AdaPonderLM τ = 10⁻⁴ and λ = 0.1 (sections 3.1 and 4.1)
- NEAT k = 300, τ_sim = 0.6, τ_mag = 0.2, τ_sup = 0.4 (appendix A.5, Table 5)
- SpecExit*: confidence > 0.8, progress > 0.3, remaining length < 200 (appendix A.1)
- Ouro Stage II: k = 50, γ = 0.005 (section 3.4)

The fetch tool processes the text through a model, so do a final verbatim check from the PDFs at the camera-ready stage.
