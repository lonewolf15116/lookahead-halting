# Gate 2 plan (v2, for reviewer sign-off; nothing has been run)

Gate 1 is frozen (tag `gate1-final`: validation freeze f5fb931, test evaluation a6590fa). No probe has been trained. This plan, including the pass rule in section 7, must be signed off and committed before any probe is fit.

## 1. Question
Can a probe that reads the loop-t hidden state h_t make stopping decisions that save more loops than the native gate and the other cheap baselines at the same accuracy loss, on questions none of the Gate 1 analysis has touched, with the probe's own cost paid?

Two claims are kept separate:
- **Learned halting works**: a learned policy beats the baselines.
- **Lookahead from h_t works**: the gain comes from the hidden state, not from re-weighting cheap signals (confidence, entropy, gate) that are already available.

## 2. Data roles
| role | source | use |
|---|---|---|
| train | Phase 1 `train` split (17,615 natural questions) | fit probe parameters only |
| validation | Phase 1 `validation` split (5,870) | choose probe family and hyperparameters, define threshold grids, pick the operating point |
| Gate 1 test | Phase 1 `test` split (5,868) | now seen; reported as an exploratory development check only, never used for any choice |
| confirmation | new extraction, section 3 | the one-time Gate 2 verdict |

## 3. Confirmation set (new extraction)
Gate 1 test outcomes are visible, so a new policy is confirmed on questions that have never been scored. Verified against the lm_eval 0.4.9 task configs, these labelled questions are unused:

| benchmark | Phase 1 scored | unused labelled pool | exemplar pool | confirmation take |
|---|---|---|---|---|
| ARC-Challenge | test | validation (299) | train | all 299 |
| ARC-Easy | test | validation (570) | train | all 570 |
| MMLU | test | validation (1,531) | dev, first 5 per subject | all 1,531 |
| OpenBookQA | test | validation (500) | 0-shot | all 500 |
| HellaSwag | validation | train (39,905) | train | 1,000 sampled from train items never used as an exemplar in Phase 1 |
| CommonsenseQA | validation | train (9,741) | train | 1,000 sampled the same way |

About 4,900 questions; estimated about 1 h on the laptop using the same writer and verifier. Sizes are more balanced than Gate 1 test (OpenBookQA 500 instead of 100), which matters because the verdict is an equal-weight macro: on Gate 1 test, 100 OpenBookQA questions alone made one question worth 0.17 pp of macro loss.
- HellaSwag and CommonsenseQA items are drawn from the exemplar pool. To keep them clean, the Phase 1 few-shot sampling is replayed (seed 1234) to list every exemplar used. Only never-used train items are eligible, and each item's own few-shot context excludes itself (lm_eval's sampler already does this).
- The validation/train source is a mild distribution shift from Phase 1. It is a generalisation check, stated as such.
- The sample ids are drawn by sha256 order (salt `lh-confirm-v1`), fixed and committed before extraction.
- The confirmation outcomes are read once, by a guarded script (same pattern as `gate1_test.py`).

## 4. Probe targets
For each question and loop t in {1,2,3}:
- **Primary policy target**: S_t = 1[answer at t equals the answer at every later loop through 4], meaning stopping at t keeps the final answer. Stopping at the first t with predicted S_t above a threshold is the learned version of the Gate 1 oracle.
- **Original lookahead targets, kept and reported**: log next-loop KL D_t (regression), answer flip F_t (classification) and loss gain G_t (regression, clipped at the 1st and 99th percentiles of train).
- **Change vs improvement** (reviewer's distinction): flips split into WC and CW, with the probe's discrimination of each reported separately. A probe that predicts "will change" but not "will improve" is reported as such.
- C_t (answer correct at t) is a diagnostic only.

## 5. Probe families (closed list; one is chosen on validation and is the only one confirmed)
1. L2-logistic on h_t, one model per loop.
2. L2-logistic on h_t with a shared model and a loop indicator.
3. MLP, one hidden layer of 256, one per loop, with early stopping on a 10% slice of train.

Inputs: h_t (2048-d, standardised on train). Regularisation is chosen by 5-fold CV on train, with folds grouped by question.

Controls, trained the same way:
- (a) **cheap-features probe**: confidence, entropy, vocabulary entropy, native gate cdf, loop, number of options, log context length.
- (b) h_t plus the cheap features.
- (c) h_t with labels shuffled within benchmark and loop.

The family, and whether (b) replaces h_t-only, is chosen by the validation criterion in section 6, then frozen.

Overhead: the probe's multiply-adds and measured ms per question go into the comparison. A linear probe is about 2k multiply-adds per loop, against about 1.4 GFLOP per token per loop for Ouro, so it is negligible. It is still measured, not assumed. Savings are reported net of probe cost in loop-equivalents.

## 6. Policies and selection
- **Policies**: learned probe, cheap-features probe, fixed-3, native gate, confidence, entropy, and deployable measured-KL (pays for the extra loop). Retrospective KL and the oracles are reference only.
- **Grids**: thresholds are the 50 quantiles of each policy's pooled validation statistic over loops 1-3, as in Gate 1. Fixed-k is a single point.
- **Saving at budget ℓ, S_π(ℓ)**: the largest macro saving among a policy's grid points whose macro loss is ≤ ℓ (0 if none qualify). It is computed the same way for every policy, so no policy gets a selection advantage. The paired bootstrap recomputes it inside each resample, so the selection effect is in the CI.
- **Probe choice**: the family and control variant with the largest validation S(1 pp). Ties go to the simpler model (1 < 2 < 3).
- **Operating point (secondary, deployment realism)**: each policy's threshold is picked on validation by the exact Gate 1 rule (max saving with macro validation loss ≤ 1 pp) and applied unchanged on confirmation. Confirmation loss and saving are reported, and violations are flagged as in Gate 1.
- **Change from v1**: the draft's 90th-percentile risk margin is dropped. On Gate 1 test the macro loss CIs were about ±1.3 pp wide, so requiring the upper bound to be ≤ 1 pp would have disqualified nearly every policy. Budget curves plus the honest operating-point report answer the generalisation concern more directly.

## 7. Gate 2 decision rule (on confirmation, read once)
Let B(ℓ) = max over baselines (fixed-3, native gate, confidence, entropy, deployable KL) of S_b(ℓ), and let P, Q be the chosen probe and the cheap-features probe. Δ_base = S_P(1 pp) − B(1 pp), Δ_ctrl = S_P(1 pp) − S_Q(1 pp), net of probe cost.

- **PASS** (lookahead from h_t works): Δ_base ≥ 0.03 with paired-bootstrap 95% CI lower bound > 0, AND Δ_ctrl CI lower bound > 0.
- **PARTIAL** (learned halting works, but h_t adds nothing measurable): Δ_base passes but Δ_ctrl does not.
- **FAIL**: Δ_base fails. Reported as a negative result; no further tuning on these questions.

Rationale for 0.03: the native gate leaves about 0.21 of loop budget to the oracle on Gate 1 test, and 0.03 is about 15% of that gap. Smaller gains are within what threshold choice alone moves.

Statistics: 2,000 paired resamples, stratified by benchmark, seed 0. Macro over six benchmarks with equal weight. OpenBookQA stays in the primary verdict, and its exclusion is reported as a sensitivity analysis only.

Secondary, reported but not in the verdict:
- curves of S_π(ℓ) for ℓ in {0.5, 1, 2} pp
- the section 6 operating points
- per-benchmark results
- question-weighted results
- leave-one-benchmark-out probes (trained on five benchmarks, tested on the sixth, on confirmation)
- predictive quality of D, F, G, and WC vs CW
- the Gate 1 test development check

## 8. What is not allowed
- No confirmation outcome is read before the selections in sections 5-6 are committed.
- No new probe families, features or thresholds after confirmation.
- Gate 1 test results are never used to choose anything.
- If a check fails, stop, keep the artifacts, and investigate (standing rule).

## 9. Order of work
1. Reviewer signs off this plan, then it is committed.
2. Confirmation-set preparation: replay the exemplars, draw the ids, commit the manifest (CPU, laptop). Then extract plus verify (GPU, about 1 h). The outcomes are written but not analysed.
3. Probe data loader, training, controls and validation selection (laptop; minutes per model). Frozen and committed.
4. One-time guarded confirmation run, then `docs/GATE2_REPORT.md`.
