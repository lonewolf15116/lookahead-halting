# Step 3: Methodology and statistics audit

- Date: 2026-10-27.
- Auditor: Claude, Opus 5.5.
- Manuscript: `docs/PAPER_DRAFT.md` at `8bc0d36`.
- Status: PROPOSED. Nothing in the manuscript has been changed.

**Question set by the reviewer.** Does the selection-rule comparison stay scientifically informative once accuracy-budget compliance and unmatched operating points are taken into account?

**Sources.** Only existing records were used:
- the TEST values already printed in the manuscript (Tables 4 and 5);
- `results/c1/frozen.json` (development selections under both rules);
- `results/exploratory/e1_report.json` (E1, nested held-out development folds).

No TEST data were read, and no new analysis was run. Every number below is either copied from those files or derived from them by arithmetic. The script output is reproduced in section 6.

**Evidence tags used here:**
- **[derived]**: arithmetic on reported values; no interval unless stated.
- **[E]**: exploratory, development data.
- **[S]**: pre-specified secondary, TEST.

---

## 1. Short answer

The comparison is informative **about selection procedures**. It is **not informative about which signal is better at matched accuracy**. The two sides of the reversal have different standing.

- **The point-only side is mostly a boundary effect** [derived, E].
  - Its TEST preference for native-pb rests on one benchmark: π1-x − native-pb is −0.098 on OpenBookQA, −0.002 on CommonsenseQA and +0.035 on HellaSwag, giving −0.022 macro.
  - On OpenBookQA, native-pb's point-only selection is the most extreme of all of them. Its threshold is τ = 1.9×10⁻⁵ on a CDF, so the statistic barely filters. Its loss was 0.98 pp at selection and 1.57 pp on TEST, the largest overshoot.
  - The non-compliance was predictable before TEST. In E1's held-out folds, point-only selections broke 1 pp in 6 of 15 benchmark-by-fold cells (native-pb) and 7 of 15 (π1-x), against 3 of 15 each under the screened rule.
  - The point-only "win" therefore tells us what that rule does, which is to select near-boundary points that overshoot. It does not tell us that the native gate is better at aggressive operating points.
- **The screened-rule side ([C]) is a valid comparison of rule outputs, at unmatched loss** [derived].
  - On TEST, π1-x's selected points used more of the budget: macro loss 0.39 pp against 0.08 pp for native-pb.
  - Part of the +0.053 saving advantage is therefore budget spent, not a better trade-off. Whether π1-x saves more at equal loss is not known.
  - Only one of the 12 TEST operating points is dominated outright: on OpenBookQA under the screened rule, π1-x has both more saving and less loss than native-pb (0.157 at 0.17 pp, against 0.136 at 0.20 pp).
  - No native-pb point dominates any π1-x point.
- **The rule dependence itself is systematic on development data, not a one-off of TEST** [E].
  - In E1, the within-rule difference π1-x − native-pb rose from the point-only rule to the screened rule in all 5 held-out folds, by +0.051 to +0.127.
  - TEST reproduced the direction under both rules.
- **A candidate mechanism makes the reversal partly substantive, but it is untested** [P].
  - The screened rule penalises not only net accuracy loss but also how many questions change correctness. The bootstrap bound widens with that count.
  - π1-x is trained on a stability label, so it should be aligned with that penalty.
  - In-sample development data are consistent with this. TEST is mixed. Details are in section 3.

**Verdict.** The selection-rule finding is **SUPPORTED as a statement about selection procedures** and **UNSUPPORTED as a statement about relative signal quality at matched accuracy**. The manuscript currently keeps the second reading out of the claims, but it does not give readers the facts that show why the first reading is the right one. Changes M1 to M6 in section 4 fix that.

---

## 2. Findings in detail

### 2.1 The point-only reversal is concentrated on one benchmark [derived]

| Benchmark | Point-only: π1-x − native-pb | Screened: π1-x − native-pb |
|---|---|---|
| OpenBookQA | −0.098 | +0.021 |
| CommonsenseQA | −0.002 | +0.121 |
| HellaSwag | +0.035 | +0.017 |
| Macro | −0.022 | +0.053 |

These are differences of the TEST savings in Table 5. No per-benchmark intervals exist.

Each rule's macro winner is driven by one benchmark:
- the point-only rule by OpenBookQA, where the winner's TEST loss was 1.57 pp;
- the screened rule by CommonsenseQA, where the winner's TEST upper bound was 1.10 pp.

The manuscript now states the second fact (C6) but not the first.

### 2.2 The point-only selections sit at the boundary, and overshoot was foreseeable [E, derived]

- **At selection, on development data, in-sample:** point-only losses were between 0.84 and 0.98 pp for both methods, right at the 1 pp limit. Screened selections were between 0.17 and 0.49 pp.
- **From development to TEST:**
  - point-only losses rose by +0.29, +0.16 and +0.13 pp for π1-x, and by +0.59, +0.26 and −0.21 pp for native-pb;
  - screened losses mostly fell: −0.14, +0.17 and −0.09 pp for π1-x, and −0.24, −0.27 and −0.25 pp for native-pb.

  This is what selection on the same data predicts for points chosen at a hard boundary.
- **E1 held-out folds:**

| Rule | Method | Cells over 1 pp (of 15) | Worst cell loss |
|---|---|---|---|
| Point-only | native-pb | 6 | 2.12 pp |
| Point-only | π1-x | 7 | 3.98 pp |
| Screened | native-pb | 3 | 1.87 pp |
| Screened | π1-x | 3 | 1.75 pp |

- **native-pb's point-only threshold on OpenBookQA is τ = 1.9×10⁻⁵, at depth cap 4.** A threshold that low makes the policy close to unconditional early exiting. That is consistent with the [P] "resolution" hypothesis in §5.4 and §7.3, and with the E1 remark that some native point-only thresholds were extremely small.

### 2.3 The [C] comparison is at unmatched loss [derived]

Macro TEST point loss from the Table 5 entries:

| Rule | π1-x | native-pb |
|---|---|---|
| Point-only | 1.10 pp | 1.15 pp |
| Screened | 0.39 pp | 0.08 pp |

Under the screened rule, π1-x spends 0.31 pp more of the macro budget than native-pb. native-pb's screened selections ended well inside the budget on TEST: −0.10 pp on CommonsenseQA and 0.13 pp on HellaSwag.

The [C] claim is about the outputs of the pre-registered rule, and that is how it was pre-registered. A reader who sees only the saving difference will read it as a better trade-off. The manuscript gives both losses in Table 4 but never puts the macro losses next to the macro saving difference.

### 2.4 The rule dependence is systematic on development data [E]

E1, held-out folds, π1-x − native-pb within each rule:
- point-only: −0.025, −0.044, −0.014, −0.021, −0.015;
- screened: +0.060, +0.083, +0.037, +0.043, +0.044;
- screened minus point-only, per fold: +0.085, +0.127, +0.051, +0.063, +0.059, all positive.

This is the development-side evidence that the rule dependence is not a single TEST draw. It is held-out within development data, but it is exploratory, and it is the analysis that motivated C1. It is therefore corroboration, not independent confirmation.

No interval exists for the interaction on TEST, and none can be computed now without re-reading TEST, which is spent.

### 2.5 Candidate mechanism: the screened rule penalises correctness flips [P; partial support]

The per-question loss difference is d = c(loop 4) − c(policy), which takes the values −1, 0 or +1. The bootstrap upper bound exceeds the point loss by roughly 1.645 × sd(d)/√n. sd(d) grows with the number of questions whose correctness changes in either direction, whether the change is a gain or a loss.

So, unlike the point-only rule, the screened rule penalises flips that cancel out. A policy whose early exits rarely change the answer passes the screen at higher saving. π1-x is trained on a stability label ("the answer never changes after this loop"), and r-kl is a stability statistic, so both should be aligned with this penalty.

Evidence:
- **Development data, in-sample, at the screened selections.** The margin (upper bound − point loss) was smaller for π1-x than for native-pb on all three benchmarks, while π1-x also saved more:
  - OpenBookQA: margin 0.44 against 0.53 pp, at saving 0.154 against 0.142;
  - CommonsenseQA: 0.51 against 0.69 pp, at 0.292 against 0.165;
  - HellaSwag: 0.37 against 0.38 pp, at 0.292 against 0.273.

  This partly restates why the rule chose these points, so it is not independent evidence.
- **TEST (Table 4 margins), mixed:**
  - OpenBookQA: 0.36 against 0.47 pp, in favour of the mechanism;
  - CommonsenseQA: 0.50 against 0.40 pp, against it, at very different savings (0.282 against 0.161), so this pair is confounded;
  - HellaSwag: 1.20 against 1.20 pp.

**Reading.** The mechanism is plausible and consistent with development data. It is not established. If it holds, the reversal is partly substantive: the two rules reward different properties (net loss against net loss plus flip variance), and a stability-trained signal is the one the screened rule should favour. A direct test needs flip counts on development data (X1 below).

### 2.6 Nesting objection: partial answer in existing data [E]

π1-x includes the native CDF among its features. E1 also evaluated a probe without that feature (`pi1-x-nocdf`). Under the screened rule it still exceeded native-pb in all 5 held-out folds:
- per fold: +0.029, +0.046, +0.024, +0.001, +0.007;
- primary macro: +0.021.

Across all six development benchmarks it did not (−0.010). It is exploratory and was never evaluated on TEST. It partly answers "the probe wins only because it contains the gate", and it should be reported as such, with the six-benchmark caveat.

---

## 3. Statistical method: other checks

| Item | Verdict | Note |
|---|---|---|
| Primary test (paired, benchmark-stratified percentile bootstrap, B = 10,000, one test) | SUPPORTED | Matches the protocol, and the PASS logic is applied as written. |
| Budget gate on the TEST point loss only | SUPPORTED as pre-registered | A pass can rest on a noisy point loss. The upper bounds are reported next to it, which is the right mitigation. |
| Interval interpretation | SUPPORTED after C5 | Now conditional on the frozen policies. |
| Rule dependence as a claim | NEEDS QUALIFICATION | There is no inferential test of the interaction on TEST. The development evidence (E1, 5/5 folds) is not cited. M4 fixes this. |
| Point-only arm as evidence about signals | UNSUPPORTED as a signal-quality reading | The manuscript does not make that reading, but it does not show readers the boundary facts (2.1–2.2). M1 and M3 fix this. |
| [C] saving advantage as a trade-off advantage | UNSUPPORTED (not claimed) | The macro loss difference should be shown next to it (2.3). M2 fixes this. |
| Same-data selection bias | Documented in §3.4 | The development-to-TEST drift (2.2) quantifies it and could be cited. |
| Secondary comparisons unadjusted | SUPPORTED after C7 | |

---

## 4. Proposed manuscript changes (text only; need approval)

| ID | Location | Change |
|---|---|---|
| M1 | §5.4, after "Table 5 shows what changes" | Decompose the point-only reversal: π1-x − native-pb is −0.098 on OpenBookQA, −0.002 on CommonsenseQA and +0.035 on HellaSwag [derived]. State that native-pb's OpenBookQA selection used τ = 1.9×10⁻⁵ and had the largest TEST overshoot (0.98 pp at selection, 1.57 pp on TEST). |
| M2 | §5.1 after the PASS paragraph, and §7.1 item 1 | Put the macro TEST losses next to the saving difference: 0.39 pp for π1-x against 0.08 pp for native-pb. State that the comparison is at unmatched loss and that native-pb's screened selections left more of the budget unused. |
| M3 | §5.4 or §3.4 | State that the non-compliance of point-only selections was visible before TEST: E1 held-out cells over 1 pp were 6 and 7 of 15 under the point-only rule, against 3 and 3 under the screened rule [E]. |
| M4 | §5.4 and §7.1 item 2 | Cite the development-side consistency: the screened-minus-point-only difference was positive in all 5 E1 held-out folds (+0.051 to +0.127) [E]. Note that E1 motivated C1, so this is corroboration, and that no TEST interval exists for the interaction. |
| M5 | §7.3, "Variance of the bound" | Restate as the flip-count mechanism (2.5), with the in-sample development margins and the mixed TEST margins. Keep it labelled [P]. |
| M6 | §6.1 (RQ2(a)) and the new §7.5 nesting bullet | Add the E1 result for the probe without the CDF feature: +0.021 primary macro, positive in 5/5 folds, −0.010 over six benchmarks [E]. |

None of these changes a number in Tables 3–7, a label of a [C] result, or a frozen record. All new values are copied from `frozen.json` or `e1_report.json`, or derived from Tables 4 and 5. The derivations are in section 6.

## 5. Optional development-only analyses (each needs its own GO; no TEST)

- **X1. Flip counts (recommended first).**
  - What: for both methods and each primary benchmark, on E1's held-out folds, count questions gained and lost along the threshold grid.
  - Asks: does π1-x reach a given saving with fewer correctness flips than native-pb?
  - Why: it tests the mechanism in 2.5 directly. It is cheap, because the stored development traces suffice.
  - If confirmed, the paper can say why the rules disagree, not only that they do.
- **X2. Matched-loss frontier (the analysis deferred in Step 2).**
  - What: cross-fitted held-out saving at fixed loss levels (for example 0.25, 0.5, 0.75 and 1.0 pp) for both methods.
  - Asks: whether π1-x saves more at equal loss on development data.
  - Note: it must be cross-fitted, because in-sample frontiers repeat the selection bias in 2.2.

Recommendation: X1 before X2. Both would be reported as [E], development data only. Neither can change the [C] result.

## 6. Derivation record

The output of the script run on 2026-10-27 over `frozen.json`, `e1_report.json` and the Table 4 and 5 values is kept in the session transcript. Key values:
- **TEST per-benchmark differences, π1-x − native-pb:**
  - point-only: OpenBookQA −0.098, CommonsenseQA −0.002, HellaSwag +0.035, macro −0.0217;
  - screened: OpenBookQA +0.021, CommonsenseQA +0.121, HellaSwag +0.017, macro 0.053.
- **TEST macro loss:** point-only π1-x 1.10, native-pb 1.147; screened π1-x 0.39, native-pb 0.077.
- **Pareto dominance:** only OpenBookQA screened, π1-x over native-pb.
- **Development, screened rule (loss, upper bound, saving):**
  - π1-x: OpenBookQA 0.311 / 0.755 / 0.154; CommonsenseQA 0.428 / 0.942 / 0.292; HellaSwag 0.494 / 0.861 / 0.292.
  - native-pb: OpenBookQA 0.444 / 0.977 / 0.142; CommonsenseQA 0.171 / 0.856 / 0.165; HellaSwag 0.383 / 0.766 / 0.273.
- **Development, point-only rule (loss, saving, depth cap m, threshold τ):**
  - π1-x: OpenBookQA 0.844 / 0.303 / 3 / 0.899; CommonsenseQA 0.942 / 0.363 / 3 / 0.911; HellaSwag 0.941 / 0.363 / 3 / 0.857.
  - native-pb: OpenBookQA 0.977 / 0.403 / 4 / 1.9e-05; CommonsenseQA 0.942 / 0.365 / 3 / 0.0685; HellaSwag 0.877 / 0.338 / 3 / 0.107.
- **E1 per fold:**
  - point-only: −0.025, −0.044, −0.014, −0.021, −0.015;
  - screened: +0.060, +0.083, +0.037, +0.043, +0.044;
  - probe without the CDF feature, screened: +0.029, +0.046, +0.024, +0.001, +0.007.
