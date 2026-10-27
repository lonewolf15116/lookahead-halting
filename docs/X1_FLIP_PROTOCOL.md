# X1: Correctness flips at matched saving

- Protocol **rev 2** (pre-run amendment of rev 1, frozen at commit 54942ad). NOT RUN.
- Rev 2 changes only the readings (section 5) and the bootstrap definition. Section 8 records both; where rev 1 and rev 2 differ, rev 2 governs.
- Status: exploratory [E], development data only. Approved in principle by the reviewer (Step 3, 2026-10-27).
- This protocol must be frozen, by commit hash, before any X1 number is computed.
- It changes no frozen artifact, no C1 result and no [C] claim. TEST is not read.

## 1. Question

Step 3 separated a property of the screen from a hypothesis:
- **Property (mathematics).** The per-question loss difference d = c(loop 4) − c(policy) takes the values −1, 0 and +1. The bootstrap upper bound therefore widens with the number of questions whose correctness changes in either direction. Unlike the point-only rule, the screened rule penalises gains and losses that cancel.
- **Hypothesis (to be examined).** At equal saving, π1-x changes the correctness of fewer questions than native-pb, and specifically has fewer cancelling flips. If so, the flip penalty would explain part of π1-x's advantage under the screened rule.

X1 asks: on held-out development data, at matched gross saving, does π1-x produce fewer correctness flips than native-pb, and is any difference in the cancelling part rather than only in net loss?

Looking only at the already-selected operating points would not answer this, because those points were chosen by the rule under study. X1 compares whole saving curves on held-out folds.

## 2. Data and statistics

- **Data.**
  - The E1 development data and outer folds (`lh-fold-v1`, 5 folds), loaded only through `phase3_g3a.load_g3a` with its sealed-path guard.
  - Primary benchmarks only for the primary analysis: OpenBookQA, CommonsenseQA and HellaSwag.
  - No validation rows. No TEST rows: the script refuses any path under `data/extracted_phase3_test`.
- **Statistics, per outer fold k, exactly as E1 computes the held-out statistics (`e1.build_methods`).**
  - native-pb: the native CDF, which has no fitted part.
  - π1-x: held-out scores from models fitted on the selection rows (fold ≠ k) only.
  - Secondary: r-kl (no fitted part) and `pi1-x-nocdf`, the E1 probe without the CDF feature.
- **Depth cap, primary.** The cap that both methods share in the frozen C1 screened selections: m = 4 for OpenBookQA and CommonsenseQA, m = 3 for HellaSwag. Sensitivity: m = 4 for all three.

## 3. Matching on saving (label-free)

Gross saving, 1 − e/4, depends only on exit loops, and exit loops depend only on the statistic, τ and the cap. Matching on saving therefore uses no correctness information.

For each fold k, benchmark b, method and level s:
1. Take as candidate thresholds the sorted distinct finite values of the method's statistic on the held-out rows of b (loops before the cap), plus +∞.
2. Compute the held-out saving of each candidate with `phase3_readouts.exit_loops`.
3. Find the two adjacent candidates whose savings bracket s.
4. Record both thresholds, their savings and the interpolation weight w, where s = (1−w)·saving₁ + w·saving₂.

These steps are implemented in a function that receives only (statistic, ok, cap). It has no access to correctness, and a test asserts this.

Only after τ and w are fixed are correctness outcomes read. Rates are interpolated with the same w.

**Levels.**
- Grid: s ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40}.
- Core levels, used for the readings: {0.15, 0.20, 0.25, 0.30}. This range contains the screened selections on TEST (0.136 to 0.282).
- A level counts for a benchmark only if both methods can reach it in all 5 folds under the cap. With m = 3, HellaSwag cannot go below 0.25, so its core levels are {0.25, 0.30}.
- The benchmarks included at each level are reported. The macro at a level is the mean over the included benchmarks.

## 4. Measures

For each fold, benchmark, method and level, with n = the held-out questions of b in fold k:
- gain rate g = #{correct at exit, wrong at loop 4} / n;
- loss rate l = #{wrong at exit, correct at loop 4} / n;
- flip rate φ = g + l;
- net loss = l − g, in pp;
- cancelling component κ = 2·min(g, l), the part of φ that leaves net loss unchanged;
- implied screen margin (descriptive) = 1.645 · √((φ − (l−g)²) / n_b), using the benchmark's full development count n_b. This links φ to the screened rule's margin.

**Pooling.** Rates are pooled over the 5 folds, weighted by n, so that each development question counts once as held-out. Per-fold values are also reported.

## 5. Contrasts, uncertainty and readings

- **Primary contrast.** Δφ(s) = φ(π1-x) − φ(native-pb), macro over the included benchmarks, pooled over folds, at each core level.
- **Decomposition.**
  - Δκ(s), the difference in the cancelling component;
  - Δnet(s), the difference in net loss;
  - Δg(s) and Δl(s), with gains and losses reported separately, as the reviewer required.
- **Uncertainty.**
  - A percentile bootstrap over held-out questions, resampled within (benchmark, fold) strata.
  - B = 2,000, `default_rng([31, 1])`.
  - Thresholds and interpolation weights are held fixed at their full-sample values, so the intervals are conditional on the matching.
  - Two-sided 95% intervals. No multiplicity adjustment; the readings below are pre-declared instead.

**Pre-declared readings (rev 1, SUPERSEDED by section 8.1; kept for the record).**

| Reading | Condition | What the paper may say |
|---|---|---|
| **FLIP-SUPPORTED** | At ≥ 3 core levels: Δφ < 0 with 95% upper bound < 0, **and** Δφ < 0 in ≥ 4 of 5 folds, **and** Δκ < 0 with 95% upper bound < 0 | "On development data, at matched saving, π1-x produced fewer correctness flips than native-pb, including fewer cancelling flips. This is consistent with the screen's flip penalty contributing to the rule dependence [E]." It is not called a causal explanation. |
| **NET-LOSS ONLY** | The Δφ condition holds, but the Δκ condition fails at ≥ 2 of those levels | "π1-x's fewer flips at matched saving were mainly fewer losses, a difference in net loss rather than in cancelling flips; the variance mechanism is not supported as distinct from net loss [E]." The paper notes that this touches the matched-saving trade-off, which X2 would study properly. |
| **NOT SUPPORTED** | At ≥ 3 core levels, Δφ ≥ 0 or its 95% upper bound ≥ 0 | "We examined the flip hypothesis on development data and did not find support for it [E]." §7.3 keeps the property and states that the hypothesis was not supported. |
| **MIXED** | Anything else | Reported as inconclusive, with the per-level and per-fold values. No mechanism statement. |

The readings are about the hypothesis only. None of them changes or strengthens the [C] result.

**Secondary analyses (reported, not used for the readings):**
- r-kl against native-pb, and `pi1-x-nocdf` against native-pb, with the same procedure;
- the m = 4 cap for all benchmarks;
- the six-benchmark macro over the levels each benchmark can reach;
- the implied margins.

## 6. What X1 cannot show

- It cannot show that flips caused the TEST reversal. It uses development data, and TEST is spent.
- It does not test the matched-loss frontier (X2). X1 matches saving, not loss.
  - **Disclosure:** gains and losses are required separately, so net loss at matched saving is computed. It is reported only as part of the decomposition, labelled [E], and is not a frontier claim.
  - **D1, needs the reviewer's decision before running:** confirm that reporting Δnet in this way is acceptable while X2 is deferred.
- It does not establish anything for loops, models or rules beyond those tested.

## 7. Execution and records

- **Code.** A new module `src/lookahead/x1_flips.py` and a script `scripts/x1_flip_check.py`, reusing the E1 functions without changing them.
- **Tests before the run,** on synthetic data:
  - the matching function cannot access correctness, and its signature is checked;
  - the interpolated savings equal s;
  - pooled rates equal per-question rates;
  - the bootstrap is reproducible;
  - the script refuses TEST paths and refuses to overwrite its output.
- **Run.** On the laptop, where the development data are, through a PowerShell runner.
- **Output.** `results/exploratory/x1_flips.json`, created exclusively.
- **Recorded with the output:** the git commit, this protocol's sha256, the sha256 of the code files and the fold fingerprints, to be checked against E1's.
- **Deviations.** Any change after the freeze is a new revision with its own commit, logged in DECISIONS before any rerun. Results are never discarded.
- **Order.** First the reviewer's GO on this protocol and on D1, then implementation and tests, then the run. Reporting follows the pre-declared reading.

## 8. Rev 2: pre-run amendment (reviewer conditional GO, 2026-10-27)

Made before any X1 number was computed. D1 is approved: gains, losses and net loss at matched saving are reported only as parts of the flip decomposition, labelled [E]. X1 is never described as a matched-loss frontier or as a test of which signal is better at equal accuracy.

### 8.1 Readings (replace the rev 1 table)

These are evaluated on the 4 core levels {0.15, 0.20, 0.25, 0.30}, with the macro at each level taken over the benchmarks included there. Per core level ℓ:
- **F(ℓ), flip reduction:** pooled macro Δφ(ℓ) < 0 with 95% upper bound < 0, **and** Δφ(ℓ) < 0 in ≥ 4 of the 5 folds.
- **K(ℓ), cancelling-flip reduction:** pooled macro Δκ(ℓ) < 0 with 95% upper bound < 0.
- **L(ℓ), losses explain the reduction:** Δl(ℓ) < 0 with 95% upper bound < 0, **and** |Δl(ℓ)| > |Δg(ℓ)| (point estimates), **and** K(ℓ) does not hold.

The rows are evaluated in order, and the first that matches applies, so the readings are mutually exclusive.

| Order | Reading | Condition | What the paper may say |
|---|---|---|---|
| 1 | **FLIP-SUPPORTED** | F(ℓ) and K(ℓ) both hold at ≥ 3 core levels | "On development data, at matched saving, π1-x produced fewer correctness flips than native-pb, including fewer cancelling flips. This is consistent with the screen's flip penalty contributing to the rule dependence [E]." Not called a causal explanation. |
| 2 | **FLIP REDUCTION, CANCELLING NOT DEMONSTRATED** | F(ℓ) holds at ≥ 3 core levels (row 1 failed) | "At matched saving, π1-x produced fewer correctness flips than native-pb, but a reduction in cancelling flips was not demonstrated [E]." The stronger sentence "the reduction was mainly in losses, a difference in net loss" is added **only if** L(ℓ) holds at ≥ 3 of the levels where F(ℓ) holds. If added, the paper notes that this touches the matched-saving trade-off, which X1 does not test as a frontier. |
| 3 | **NOT SUPPORTED** | F(ℓ) holds at **no** core level, **and** the pooled macro point estimate Δφ(ℓ) ≥ 0 at ≥ 3 of the 4 core levels | "We examined the flip hypothesis on development data and did not find support for it [E]." §7.3 keeps the property and states that the hypothesis was not supported. |
| 4 | **MIXED** | Anything else | Inconclusive. Per-level and per-fold values are reported, with no mechanism statement. |

**Pre-registration commitment.** These rules are final for X1. They are not revised after results are seen. Any later change would be a new, labelled post hoc analysis, and the rev 2 reading stays the reported one.

### 8.2 Bootstrap: exact resampling unit (replaces the rev 1 uncertainty paragraph)

- **Unit.** One held-out development question.
- **Strata.** (benchmark, outer fold), for the primary benchmarks.
- **Draws.**
  - In replicate r = 1..2,000, each stratum gets one multinomial count vector over its held-out questions, of size n_stratum with equal probabilities.
  - Generator: `default_rng([31, 1])`, with strata processed in sorted (benchmark, fold) order.
- **Pairing.** The **same** count vector is applied to every method (π1-x, native-pb, and the secondary methods), to **both** bracketing thresholds of every level, and to every measure (g, l, φ, κ, net). Every contrast is therefore paired at the question level.
- **Fixed quantities.** Thresholds, bracketing pairs and interpolation weights are fixed at their full-sample values and are not re-matched within replicates. Fitted probe scores are the E1 held-out scores and are not refitted.
- **Scope of the intervals.** They are conditional, percentile intervals. They reflect resampling of held-out questions only. They do not account for uncertainty in the threshold matching, in probe fitting or in the fold assignment.
- **Fold criterion.** "Δφ < 0 in ≥ 4 of 5 folds" uses full-sample per-fold point estimates, with each fold's rates computed over its own held-out questions.

### 8.3 Unchanged safeguards

- No TEST access.
- No change to any C1 artifact.
- Output created exclusively.
- The matching function receives no correctness data.
- No revision of outcome criteria after results.
- X2 remains deferred. X1 can inform whether X2 is worth doing, but cannot establish a matched-accuracy advantage.
