# Gate 2 plan (v3, frozen on reviewer sign-off; nothing has been run)

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

### 3a. Contamination audit (must pass before extraction; output committed)
`scripts/prepare_confirmation.py` writes `results/gate2/confirmation_audit.json` and refuses to write the manifest unless every check passes with zero overlap.
1. **Actual exemplars, not a presumed seed.** Phase 1 questions are rebuilt through the same pipeline with lm_eval's sampler instrumented to record every exemplar doc it returns. The rebuild must reproduce each benchmark's frozen `question_set_sha256`; otherwise the audit stops. Every recorded exemplar's text must also be found verbatim in the stored context of the question it was recorded for (`data/questions/*.jsonl.gz`), so the list is checked against what the model actually saw.
2. **No identity overlap.** Confirmation items share no dataset-native id (ARC/OBQA/CSQA `id`, HellaSwag `ind` plus split) with any Phase 1 question (all three splits) or any recorded exemplar.
3. **No overlap under another identifier.** A normalised text hash of question plus choices (lower-case, whitespace collapsed) is compared against all Phase 1 questions and all exemplars, across benchmarks. Any collision excludes the confirmation item and is listed.
4. **No leakage into probe data.** The audit asserts that no confirmation `doc_id` or text hash appears in the Phase 1 extraction records that the probes train or validate on.
5. **Fingerprints.** Recorded: lm_eval version, sha256 of each task YAML used, HF dataset name and revision sha for each source split, the sampling salt, and the sha256 of the final manifest.
6. **Disclosure.** The HellaSwag/CSQA items come from the train split, and ARC/MMLU/OBQA from validation. This is a distribution shift from Phase 1, so the confirmation set is a fresh, independent sample, but not a draw from the original evaluation distribution. Stated in the report.

## 4. Probe targets
For each question and loop t in {1,2,3}:
- **Primary policy target**: S_t = 1[answer at t equals the answer at every later loop through 4], meaning stopping at t keeps the final answer. Stopping at the first t with predicted S_t above a threshold is the learned version of the Gate 1 oracle.
- **Original lookahead targets, kept and reported**: log next-loop KL D_t (regression), answer flip F_t (classification) and loss gain G_t (regression, clipped at the 1st and 99th percentiles of train).
- **Change vs improvement** (reviewer's distinction): flips split into WC and CW, with the probe's discrimination of each reported separately. A probe that predicts "will change" but not "will improve" is reported as such.
- C_t (answer correct at t) is a diagnostic only.

## 5. Probe families and frozen training settings
Closed list. One configuration is chosen on validation (section 6), and only that one is confirmed.
1. L2-logistic on h_t, one model per loop t in {1,2,3}.
2. L2-logistic on h_t, one shared model with a one-hot loop indicator.
3. MLP 2048 -> 256 -> 1 (ReLU, dropout 0.1), one per loop.

Inputs: h_t in float32, standardised with train mean and sd. Controls, trained with identical settings:
- (a) cheap-features probe: confidence, entropy, vocabulary entropy, native gate cdf, loop indicator, number of options, log context length
- (b) h_t plus the cheap features
- (c) h_t with labels shuffled within (benchmark, loop), shuffle seed 0

Frozen before any fit:
- Seeds: global seed 0 (numpy, torch, sklearn).
- CV folds: 5 folds by `int(sha256("lh-fold-v1|"+doc_id),16) % 5`, so a question's loops never span folds.
- Logistic: sklearn `LogisticRegression(solver="lbfgs", max_iter=5000)`, C in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1}, chosen by mean CV log-loss; ties go to the smaller C. Refit on all of train.
- Regression targets (log D_t, clipped G_t): ridge with alpha in {1e-2, 1e-1, 1, 10, 100, 1e3, 1e4}, chosen by mean CV MSE; ties go to the larger alpha.
- MLP: AdamW, lr 1e-3, batch 512, float32, weight decay in {1e-4, 1e-3, 1e-2}, at most 50 epochs.
  - Early stopping: patience 5 on BCE over a train-only 10% slice (`sha256("lh-es-v1|"+doc_id) % 10 == 0`); the best epoch's weights are kept.
  - Weight decay is chosen by early-stopping-slice BCE, with ties going to the larger decay. Refit is not done: the early-stopping model is final.
- Nothing is tuned on validation except the closed choice in section 6.

Overhead, reported in two separate forms:
- Measured: probe inference wall-clock on the laptop (RTX 5050), median over 1,000 repeats, on the same device and dtype path as deployment, in ms per question per loop, next to the measured mean ms of a loop.
- Estimated: probe multiply-adds converted to loop-equivalents. Net saving = loop saving − probe loop-equivalents.

## 6. Policies and selection (validation only)
- **Policies**: learned probe P, cheap-features probe Q, fixed-3, native gate, confidence, entropy, and deployable measured-KL (pays for loop t+1). Retrospective KL and the oracles are reference only.
- **Grids**: each policy's thresholds are the 50 quantiles of its pooled validation statistic over loops 1-3, as in Gate 1. Fixed-k is a single point.
- **Operating point**: exactly the Gate 1 rule. Take the max-saving grid point whose macro validation loss is ≤ 1 pp; ties go to the smaller loss, then the most conservative threshold. A policy with no qualifying point is not qualified.
- **Probe choice**: among the 3 families × {h_t only, (b) h_t plus cheap features}, take the one with the largest validation operating-point saving. Ties go to the smaller validation loss, then the simpler model (1 < 2 < 3, h_t only < (b)). Q is fit and selected the same way, with family fixed to 1 for its 7 features.
- **Freeze**: every threshold, the chosen probe's weights, and their sha256 are written to `results/gate2/frozen_selection.json` and committed before any confirmation outcome is read.

## 7. Gate 2 decision rule (confirmatory; frozen thresholds applied once to confirmation)
Each frozen policy is applied unchanged to the confirmation set. Its macro loss and net macro saving are reported with paired-bootstrap 95% CIs (2,000 resamples, stratified by benchmark, seed 0; six benchmarks, equal weight).

Comparator: B = the largest confirmation saving among all validation-qualified baselines, including any baseline that exceeds 1 pp loss on confirmation. This is conservative toward the probe: a baseline buying saving with extra accuracy loss still sets the bar. The comparison against loss-compliant baselines only is reported as secondary.

Three conditions:
- **L (budget)**: P's confirmation macro loss point estimate ≤ 1.00 pp.
- **E (effect)**: Δ_base = S_P − S_B ≥ 0.03, and the paired-bootstrap 95% CI lower bound of Δ_base > 0. B is re-determined inside each resample.
- **H (hidden state)**: the paired-bootstrap 95% CI lower bound of Δ_ctrl = S_P − S_Q > 0.

Outcomes, exhaustive:
| L | E | H | verdict |
|---|---|---|---|
| yes | yes | yes | **PASS**: lookahead from h_t gives a deployable improvement |
| yes | yes | no | **PARTIAL**: learned halting improves on the baselines; a hidden-state benefit beyond cheap features is *not demonstrated* (not "shown to be absent": failing to reject a difference is not evidence of equivalence) |
| no | any | any | **FAIL** (probe exceeds the accuracy budget) |
| yes | no | any | **FAIL**, which includes 0 < Δ_base < 0.03 and Δ_base ≥ 0.03 with a CI lower bound ≤ 0; the observed Δ and CI are reported as a secondary result |

The 0.03 margin is a project-specific practical-significance criterion, not a statistical convention. It is about 15% of the 0.207 native-gate-to-oracle gap observed on Gate 1 test. OpenBookQA stays in the primary verdict; its exclusion is reported as a sensitivity analysis only.

### 7a. Secondary and retrospective analyses (never change the verdict)
- **Accuracy-budget frontier (retrospective diagnostic)**: S_π(ℓ) is the best confirmation saving among grid points with confirmation macro loss ≤ ℓ, for ℓ in {0.5, 1, 2} pp, with reselection inside each bootstrap resample. It uses confirmation labels to pick thresholds, so it is not deployable and is labelled as such.
- Comparison against loss-compliant baselines only; per-benchmark and question-weighted results; sensitivity without OpenBookQA.
- Leave-one-benchmark-out probes (fit on five benchmarks' train, frozen thresholds from the five benchmarks' validation, applied to the sixth on confirmation).
- Predictive quality on confirmation: AUROC for S_t, F_t, WC vs CW; Spearman for log D_t and G_t; each against the cheap-features model.
- Gate 1 test as an exploratory development check, run after the confirmation verdict is written.

## 8. What is not allowed
- No confirmation outcome is read before the selections in sections 5-6 are committed.
- No new probe families, features or thresholds after confirmation.
- Gate 1 test results are never used to choose anything.
- If a check fails, stop, keep the artifacts, and investigate (standing rule).

## 9. Order of work
1. Plan v3 committed (reviewer conditional approval, amendments A/B and the engineering freeze applied).
2. Confirmation-set preparation and contamination audit (section 3a; CPU, laptop). The manifest and audit are committed. Then extract plus verify (GPU, about 1 h). The outcomes are written but not analysed.
3. Probe data loader, training, controls and validation selection (laptop; minutes per model). Frozen and committed.
4. One-time guarded confirmation run, then `docs/GATE2_REPORT.md`.

## Change log
- v1 draft (88ce902). v2 (e7ca404): fresh confirmation set and the loss-budget comparison.
- v3: on reviewer review, (A) primary verdict uses validation-frozen operating points, and the confirmation-optimised frontier is retrospective only; (B) the PASS/PARTIAL/FAIL table is made exhaustive, with "not demonstrated" wording; contamination audit specified (3a); training seeds, grids, early-stopping split and tie-breaks frozen (section 5); measured vs estimated overhead separated.
