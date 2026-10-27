# X1 result: correctness flips at matched saving (exploratory, development data only)

## Run

- **When:** 2026-10-27, on the laptop (CPU), taking 1.2 minutes.
- **Code:** commit `5461ca7`. The protocol is rev 2 (`docs/X1_FLIP_PROTOCOL.md`, sha256 `b327cedc…`), which matches the frozen amendment `ef19f0a`.
- **Output:** `results/exploratory/x1_flips.json`, sha256 `457617c7c8bd7c18…`.
- **Tests:** 34 passed before the run.

**Integrity checks** (done in the cloud on the staged output):
- The protocol sha256 recorded in the output equals the frozen rev 2.
- The recorded sha256 of all five code files equals the committed files.
- All 95 fold-fingerprint fields (outer models and selection statistics) are identical to E1's, so X1 used exactly E1's held-out statistics.
- The output records `git_dirty_code: true`. On the laptop, `git status -- src scripts docs` shows one untracked file, `docs/GATE2_VALIDATION_REPORT_DRYRUN.md`, an old dry-run report. No code file differs.
- No TEST data were read.

**Data:** the development questions of the three primary benchmarks, each held out once across E1's 5 folds:
- OpenBookQA 2,252;
- CommonsenseQA 1,168;
- HellaSwag 6,273.

## Governing reading (protocol 8.1, primary contrast π1-x against native-pb, primary caps)

**FLIP-SUPPORTED.**
- F(ℓ) and K(ℓ) hold at all 4 core levels (0.15, 0.20, 0.25 and 0.30).
- Δφ < 0 in 5 of 5 folds at every core level.
- L(ℓ) holds at no level, so the "mainly losses" sentence does not apply.

Pre-declared wording: "On development data, at matched saving, π1-x produced fewer correctness flips than native-pb, including fewer cancelling flips. This is consistent with the screen's flip penalty contributing to the rule dependence [E]." It is not a causal explanation.

| Saving level | Benchmarks | Δφ (95% CI) | Δκ (95% CI) | Δg | Δl | Δnet, pp (95% CI) |
|---|---|---|---|---|---|---|
| 0.15 | CommonsenseQA, OpenBookQA | −0.0133 [−0.0179, −0.0088] | −0.0130 [−0.0170, −0.0065] | −0.0061 | −0.0072 | −0.11 [−0.56, +0.36] |
| 0.20 | CommonsenseQA, OpenBookQA | −0.0202 [−0.0262, −0.0143] | −0.0183 [−0.0255, −0.0113] | −0.0092 | −0.0110 | −0.19 [−0.78, +0.40] |
| 0.25 | all three | −0.0174 [−0.0225, −0.0126] | −0.0182 [−0.0235, −0.0119] | −0.0091 | −0.0083 | +0.08 [−0.40, +0.57] |
| 0.30 | all three | −0.0163 [−0.0216, −0.0113] | −0.0197 [−0.0256, −0.0124] | −0.0098 | −0.0064 | +0.34 [−0.17, +0.86] |

Notes on the table:
- Δ is π1-x − native-pb. The Δg and Δl intervals all exclude 0 at the core levels.
- Absolute macro flip rates: φ is 0.008, 0.013, 0.023 and 0.034 for π1-x, against 0.021, 0.034, 0.040 and 0.051 for native-pb. The reduction is large in relative terms and small in absolute terms (1 to 2 percentage points of questions).

## Facts that qualify the reading. They do not change it; a reviewer needs them.

1. **The reduction is in both gains and losses.**
   - Δg and Δl are both negative, with intervals excluding 0 at every core level.
   - The macro difference in net loss is not distinguishable from 0 at any core level.
   - So π1-x changes fewer answers in both directions at matched saving, without a demonstrated difference in net loss. This is the pattern the flip mechanism predicts. It is not a better accuracy trade-off.
2. **It is concentrated on CommonsenseQA.**
   - Per-benchmark point values (no per-benchmark intervals): φ for π1-x against native-pb is 0.001 / 0.015 at 0.15, 0.002 / 0.029 at 0.20, 0.003 / 0.043 at 0.25 and 0.011 / 0.048 at 0.30.
   - OpenBookQA shows a smaller reduction: 0.015 / 0.027, 0.025 / 0.038, 0.039 / 0.052, 0.058 / 0.066.
   - At 0.25 on HellaSwag, both methods equal fixed depth 3 (cap 3), so the difference is 0 by construction.
   - CommonsenseQA is also where most of the TEST macro advantage arose (§5.2).
3. **On OpenBookQA, π1-x's net loss at matched saving is higher** (point values, descriptive): 1.07 against 0.49 pp at 0.25, and 1.59 against 0.47 pp at 0.30.
   - Above the core range, the macro Δnet at 0.35 is +0.53 pp [+0.02, +1.04].
   - Under D1, these are reported only as parts of the decomposition. They are not a frontier claim, and they reinforce that X1 shows no matched-accuracy advantage for π1-x.
4. **Secondary: r-kl against native-pb gives NOT SUPPORTED** (descriptive; does not govern).
   - r-kl produces *more* flips than native-pb at matched saving: Δφ is between +0.007 and +0.013 at the core levels, with every 95% interval entirely above 0, under both cap sets.
   - Yet r-kl matched π1-x under the screened rule on TEST (+0.00125, Table 6).
   - So the flip pattern does not explain r-kl's performance under the screened rule. The mechanism, if real, is specific to π1-x and is not a general property of stability statistics.
   - This limits any statement that the flip penalty explains the rule dependence as a whole.
5. **Secondary: the probe without the CDF feature against native-pb gives FLIP-SUPPORTED at all core levels.** The flip reduction does not depend on π1-x containing the native gate's signal.
6. **Sensitivity (cap 4 for all three benchmarks):** π1-x against native-pb is FLIP-SUPPORTED at all core levels, so the result does not depend on the cap choice.

## What this permits in the paper (proposal for the reviewer; nothing is applied)

§7.3, "Correctness flips and the bound": replace the sentence "A hypothesis, not tested: …" with:

> "We examined this on development data in a pre-registered exploratory analysis (X1). At matched gross saving on held-out folds, π1-x changed the correctness of fewer questions than native-pb, in both directions, at every pre-declared saving level (for example 1.3 against 3.4 flips per 100 questions at saving 0.20), with no demonstrated difference in net loss [E]. This is consistent with the screen's flip penalty contributing to π1-x's advantage under the screened rule. It is not a causal demonstration. The reduction was concentrated on CommonsenseQA, and it does not explain r-kl: the untrained KL readout produced more flips than native-pb at matched saving, yet matched π1-x under the screened rule on TEST [E]. X1 compares methods at matched saving, not matched loss, and does not show a better accuracy trade-off."

The paper may also need a single disclosure sentence in §4.4 or §7.5, saying that X1 was pre-registered after the Step 3 audit, on development data only.

## Bearing on X2

X1 points to a difference in flips at no demonstrated difference in net loss. That suggests a matched-loss frontier (X2) would not show a clear π1-x advantage. OpenBookQA even points the other way at the higher saving levels. This is a reason X2 may be of limited value, not a result about it. X2 remains deferred.
