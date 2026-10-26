# Confirmatory study C1: learned probe vs Ouro's native gate under identical risk-controlled selection (FINAL TEXT for review; not yet frozen)

**Status:** final protocol text incorporating the reviewer's conditional approval (D-a to D-e decided, PASS wording clarified, two freeze checks added). It is frozen and tagged `c1-protocol` only after the reviewer approves it together with the development-only freeze implementation. Nothing has been run, and TEST stays sealed.
- **Separation.** C1 is a new study. Phase 3 stays closed at G3a/D2, and its results and thresholds are not revised.
- **Motivation.** The exploratory E1 (`docs/E1_REPORT.md`, accepted by the reviewer as exploratory) found that under the
  same combined risk rule, π1-x saved more than native-pb on development data: +0.053, positive in all 5 folds. Under the
  point-estimate rule it saved less (−0.024). C1 asks whether that holds on data never used for any design decision.
- **One shot.** C1's single confirmatory test uses the sealed Phase 3 TEST (in-distribution: OpenBookQA 3,000,
  CommonsenseQA 1,000, HellaSwag 750). It is opened once, by a guarded script, after this protocol is frozen and tagged.

## C1.1 Question and hypothesis
**H-C1 (primary).** Both policies are fitted and selected on development data only, with the same combined risk-controlled
rule. On the TEST in-distribution set, does π1-x save more loops than Ouro's native gate (native-pb), by at least the minimum
meaningful improvement δ, while meeting the accuracy budget on every primary benchmark?

## C1.2 What is frozen before TEST is opened (all hashed and tagged)
- **Selection data.** Phase 1 train + DEV+ (20,250 questions), exactly as in G3a and E1. The exposed Phase 1 validation is
  not used for any choice (**D-a: development only**).
- **π1-x.**
  - Per-loop L2 logistic on Gate 2's 9 cheap features, target S_t (stability), Gate 2 frozen C = 1.0.
  - Statistics for selection: 5-fold cross-fitted out-of-fold scores over all development data (`lh-fold-v1`).
  - Final models: refit on all development data and saved, with the scaler.
- **native-pb.** The statistic is Ouro's cumulative exit probability (cdf). It has no fitted part.
- **Selection rule, applied identically to both** (E1's `select_risk`, unchanged):
  - per primary benchmark, among (τ_b, m_b) on the 51-point grid × m ∈ {1..4}, a candidate qualifies iff its point loss is
    ≤ 1 pp **and** the bootstrap 95% upper bound of its loss is ≤ 1 pp;
  - 2,000 resamples of selection questions, with fixed rng `default_rng([0, 99, benchmark index])`, the same draws for both
    policies;
  - then the frozen tie-breaks.
  - This is the *combined* rule, not an upper-bound-only rule, and **not a formal guarantee of risk control**: E1 still saw
    3 of 15 held-out violations for both policies.
- **Reference policy.** B_pb, fixed depth per benchmark, under the same rule. It is reported, and it is not the comparator.
- **Artifacts.** Models, scalers, (τ_b, m_b) for each policy and benchmark, the selection-time upper bounds, the code
  hashes, the bootstrap seeds and this file.

## C1.3 Endpoints
- **Primary.** Δ = S_π1-x − S_native, the paired difference in **gross** loop saving, macro-averaged with equal weight over
  the three primary benchmarks on TEST.
- **Minimum meaningful improvement δ = 0.02** (2 percentage points of loops; the same size as Phase 3's D2 screen).
  (**D-b: δ = 0.02**.)
  - The E1 development estimate (+0.053) is **not** used to set δ.
- **Net saving.** Reported beside the primary endpoint, with ρ_b measured on development data before freezing, following
  plan section 5a.
  - π1-x needs the vocabulary entropy (vent) at every loop, a full-vocabulary softmax at one position, so its ρ > 0.
  - The native gate's cdf is produced by the model anyway (ρ = 0).
  - **D-c: gross saving is primary; net saving is a mandatory secondary.**
  - **ρ includes the probe's whole incremental cost**, measured on the GPU before freezing with a representative
    implementation. Per loop, that is:
    - the option scoring needed for conf and ent;
    - the full-vocabulary log-softmax needed for vent;
    - assembling the features, scaling them, and running the per-loop logistic.
    It is divided by the measured time of one loop. Batch size 1, same GPU and code path as extraction, about 200 DEV+
    questions per primary benchmark.
  - **If the net advantage disappears**, a gross-saving PASS is still reported under the frozen rule. The practical
    conclusion must then say that once its overhead is counted, the probe's advantage is not realized.

## C1.4 Accuracy protection
- **Budget, per primary benchmark.** loss_b = acc_b(loop 4) − acc_b(policy) ≤ 1.00 pp, point estimate on TEST, with the
  Gate 1 tolerance of 1e-12.
- **π1-x must meet the budget on all three benchmarks.** If it does not, the result is **FAIL (budget)**, whatever Δ is. A
  saving advantage alone never counts as success.
- **If native-pb breaks the budget on TEST, it still sets the bar**, as in Phase 3 plan 5a. That is reported, and it does not
  change the verdict.
- **The point-estimate budget is an operational criterion declared in advance. It is not proof of population-level
  compliance.**
- **Demonstrated compliance**, the one-sided 95% upper bound of each benchmark's loss, is reported **prominently, in the
  verdict table itself**, for both policies. Every bound above 1 pp is flagged. Per A1.5 it is out of reach for
  CommonsenseQA (n = 1,000) and HellaSwag (n = 750), even for a policy that loses nothing, and this is stated before the
  test.

## C1.5 Statistical decision (fixed before TEST)
- **Interval.** Paired bootstrap over TEST questions, resampled within each benchmark, so each benchmark keeps its own n and
  the macro stays equal-weight. B = 10,000, seed `default_rng(2026)` (**D-d**). Two-sided 95% percentile interval [L, U] for Δ.
- **Verdict (exhaustive; the first matching row applies):**

| condition | verdict |
|---|---|
| π1-x loss > 1 pp on any primary benchmark | **FAIL (budget)** |
| Δ ≥ δ **and** L > 0 | **PASS**: a statistically supported positive improvement (L > 0) with an observed effect of at least δ = 0.02 (point estimate) |
| U < δ | **FAIL (no meaningful advantage)**: improvements of size δ or more are ruled out |
| otherwise | **INCONCLUSIVE**: Δ and [L, U] reported, no claim either way |

- **Expected precision, stated before the test.** E1's paired interval for π1-x vs native was 0.007 wide on about 9,700
  primary development questions. Scaled to TEST's 4,750 primary questions, the expected width is about 0.010
  (half-width ≈ 0.005). This will be recomputed at freeze from cross-fitted development discordance, as in A1.5. With
  δ = 0.02 an INCONCLUSIVE result is therefore unlikely unless the true Δ lies near δ.
- **What PASS does and does not claim (reviewer clarification).** PASS establishes, at 95% confidence, that the
  improvement is **positive**, and that the **observed** effect is at least 0.02. It does **not** establish with 95% confidence
  that the true improvement is at least 0.02; that would need L > 0.02, which is not the criterion. Reports must keep the two
  apart. Whether L > 0.02 also holds is reported as a secondary fact.
- **One primary test.** Everything else is secondary and never changes the verdict.

## C1.6 Secondary analyses (reported; never change the verdict)
1. **The same comparison under the point-estimate rule.** The E1 development direction was −0.024, so this shows whether the
   rule dependence replicates.
2. **π1-x against every A2 baseline** (B_pb, native-pb, the seven readouts) under the same risk rule, including the strongest
   of them. In E1 only r-kl was close to native-pb (+0.003).
   **D-e: native-pb is the sole primary comparator.** The strongest baseline under the risk rule, re-selected per resample as
   in A2.5, is reported here as secondary.
3. **Net saving** with the measured ρ.
4. **OOD (PIQA, Social IQa), reported separately, never pooled.**
   - Unseen benchmarks get the task-agnostic pooled (τ, m) selected under the same combined rule over all six development
     benchmarks (A1.2), for both policies.
   - The point-estimate verdict-style table is per benchmark.
5. **Held-out violation table** for every policy, with the worst loss per benchmark.

## C1.7 Procedure
1. **Reviewer decisions** on D-a to D-e: done (conditional approval, 2026-10-26). This file plus the freeze implementation go to the reviewer, then are frozen and tagged `c1-protocol`.
2. **Freeze script** (development only):
   - fits π1-x;
   - runs the selection for both policies and B_pb, and the pooled OOD selection;
   - measures ρ on the GPU;
   - recomputes the expected precision;
   - writes `results/c1/frozen.json` and the model files with their sha256.
   - The reviewer signs off the frozen artifacts.
3. **Guarded TEST script**, in the Gate 2 confirmation pattern:
   - it refuses to run unless every frozen hash matches and `results/c1/TEST_OPENED.json` does not exist;
   - it writes that marker before reading any outcome;
   - it reads only `data/extracted_phase3_test`;
   - it applies the frozen policies once and computes the verdict table.
   - It cannot be re-run.
4. **`docs/C1_REPORT.md`** reports the verdict and every secondary analysis, whatever they show.

## C1.8 Novelty, stated in advance
- **Risk-controlled threshold selection is not new.** Learn-then-Test, conformal risk control and CALM's calibrated early
  exit all use it.
- **C1 does not claim a new selection method.** Its contribution, if it passes, is empirical and specific: on Ouro-1.4B, a
  cheap post-hoc probe beats the model's jointly trained exit gate once both are held to the same risk-controlled selection,
  whereas under point-estimate selection the gate wins.
- **Claims are limited** to Ouro-1.4B, these benchmarks, gross (and net) loop saving, and this selection rule.

## C1.9 What C1 does not do
- It adds no candidate or feature, and does not tune π1-x on anything new: π1-x is exactly G3a's frozen candidate.
- It does not use the OOD train splits, or the TEST for anything but the single run.
- It does not revisit Phase 3.

## Reviewer decisions (C1 protocol review, 2026-10-26): conditional approval
- **D-a.** Development data only.
- **D-b.** δ = 0.02.
- **D-c.** Gross saving primary; measured net saving a mandatory secondary.
- **D-d.** 10,000 paired, benchmark-stratified resamples, seed 2026.
- **D-e.** native-pb is the sole primary comparator; fixed depth and the other baselines are secondary.
- **Clarification applied.** PASS = L > 0 and observed Δ ≥ 0.02. It is not a 95% claim that the effect is at least 0.02
  (C1.5).
- **Freeze checks applied.** Compliance uncertainty is reported prominently (C1.4). ρ is measured with the full feature
  extraction and inference cost, and if the net advantage disappears that limitation goes into the conclusion (C1.3).
