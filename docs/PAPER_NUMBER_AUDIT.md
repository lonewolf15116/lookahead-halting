# Paper number audit: docs/PAPER_DRAFT.md

Independent audit of every numeric value in Sections 1-8 against `results/c1/test_report.json`, `results/c1/frozen.json`, `results/phase3/g3a_report.json`, `results/exploratory/e1_report.json`, `data/manifests/phase3_questions.json`, `docs/C1_PROTOCOL.md` and the docs the draft cites. Script: `/tmp/claude-0/-home-claude/56fe7660-1ff4-58a8-b3e6-43d4a3efd394/scratchpad/audit/audit.py` (recomputes from JSON; the earlier provenance table was used only as a hint). Rounding rule: OK means equal to round-half-up of the recomputed value; ROUNDING means a tie or float-representation case where another convention gives a value that differs by 1 unit in the last digit.

## 1. Summary counts

| Status | Count |
|---|---|
| Checked (items in checklist, all statuses) | 338 |
| OK | 320 |
| ROUNDING | 3 |
| MISMATCH | 0 |
| UNVERIFIABLE | 6 |
| LITERATURE (not checkable here) | 9 groups |

Result: **no numerical MISMATCH found.** Every table cell, CI bound, derived difference, count, percentage, hash prefix and frozen threshold in the draft matches its source. The only non-OK recomputed items are three occurrences of one tie (below). The direction and attribution statements all hold (section 3).

## 2. MISMATCH and UNVERIFIABLE items

### MISMATCH

None.

### ROUNDING (informational)

| Draft line | Draft text | Exact value | Source key | Note |
|---|---|---|---|---|
| 38-39, 372, 378, 380 area, Table 6 | `+0.0012` (π1-x − r-kl) | 0.00125 exactly (0.23927778 − 0.23802778); stored float 0.0012499999999999987 | `secondary_all_baselines.per_baseline.r-kl.delta` | Round-half-up gives +0.0013; the draft shows +0.0012 (float/half-even). Within 1 unit of last digit. Suggest either keep (matches stored float and the JSON) or write +0.0013 / "+0.00125". The CI [−0.0041, +0.0065] and "0.65 pp / 0.41 pp" are unaffected. |
| 297 | realised half-width `0.0042` "matched the 0.0043 expected" | 0.004195 realised vs 0.004315 expected | `primary.ci95`; `frozen.expected_precision.expected_half_width_95` | Both roundings correct; they differ by one unit in the last digit, so "matched" is loose but fair. |

### UNVERIFIABLE (cannot be confirmed from the allowed sources)

| Draft line | Draft text | Why |
|---|---|---|
| 346-348 | Figure 1 marker positions (image, not data) | The PNG/PDF exists in figures/ but plotted positions were not read; the caption values (+0.053, −0.022 and CIs) were verified; the four operating points per panel equal Tables 4/5 entries by construction. |
| 176 | "up to 50 distinct quantile values" / grid construction | Grid size 51 is stated in C1_PROTOCOL.md; the quantile construction and "distinct" are in code only (src/lookahead/phase3_readouts.py), not examined here. |
| 179 | "all nine point losses within 1 pp" cross-checked; but "bound is the 95th percentile of loss over 2,000 resamples, fixed seed per benchmark, same draws" | Verified only through frozen.json bootstrap_selection (n=2000, rng default_rng([0,99,idx])) and C1 protocol text; implementation not read. |
| 239-241, 274 | Reviewer statements ("conditional approval", "confirmed by the reviewer", "accepted a documented PIQA deviation") | Process claims; DECISIONS.md was not exhaustively cross-read. |
| 241 | Git tag c1-protocol; commit identity | No git allowed; test_report marker records commit 196a882 but the tag was not checked. |
| 207 | Retrospective-exactness code reading (loops depend only on earlier loops) | Claim about source code, not numeric; not re-derived. |

### LITERATURE (quoted from other papers; not checkable here)

| Draft line | Item |
|---|---|
| 90, 382 | 5×10⁻⁴ KL threshold (Geiping et al.) |
| 113 | 7.35 pp loss at ~0.55 of loops (vllm-rlt issue #56) |
| 82 | 51 quantiles between 0.5th and 99.5th percentile (Popescu appendix B.5) |
| 75 | Popescu: released Ouro-1.4B and 2.6B, six benchmarks |
| 68-69 | Ouro MMLU analysis: hidden-state threshold within "one to two points" of gate |
| 111 | Olszowiec: "most of eight looped models" |
| 62, 101, 440 | ACT halts at cumulative probability 1−ε; Jazbec 1−δ UCB level |
| 545-564 | Reference-list items: arXiv ids, volumes, pages, years (e.g. 2208.02814, 19(2):1641–1662, PMLR 306:61372–61389, 306:144187–144210, pp. 24616–24627, 2607.20519, 2610.18127, accessed 2026-10-27, opened 2026-09-24) |
| 68, 80-82, 382 | Popescu section/appendix refs (B.5, B.6), Zhu2025 §5.4.1, Geiping2025 §6.1 |

Note: the figure "76-89%" mentioned in the audit brief does not occur anywhere in the current draft (grep for "76" and "89%" finds nothing).

## 3. Direction and attribution statements checked

All of the following are CORRECT against the data (no wrong direction or attribution found):

- L34, L411: "π1-x saved more of the four-loop budget than Ouro's gate" under bootstrap-screened rule on TEST: Δ = +0.0529, CI [+0.0488, +0.0571]. Correct.
- L36, L414: under the point-only rule on TEST the preference reverses (−0.0220, CI [−0.0258, −0.0182]); native-pb preferred. Correct.
- L37, L416, L525: "point-only test operating points exceeded the 1 pp budget on most benchmarks": 5 of 6 policy-benchmark cells exceed (π1-x 3 of 3: 1.13/1.10/1.07; native-pb 2 of 3: 1.57/1.20; HellaSwag native 0.67 does not). "native-pb on two of three, π1-x on all three" is exact.
- L38-40, L378: r-kl non-superiority: CI [−0.0041, +0.0065] contains 0; correct to call it non-superiority, not equivalence. r-kl is the TEST-strongest baseline, selected 10,000/10,000 resamples.
- L41-42, L393-399: Social IQa — π1-x loss 3.26 pp, UB 4.72 pp; native-pb stays within budget on both OOD sets (PIQA 0.11 pp, SIQA −0.05 pp). Correct. On PIQA π1-x point loss 0.71 pp is within budget but UB 1.52 pp exceeds: stated correctly.
- L43, L321, L472: "four of nine upper bounds exceed 1 pp, every point loss within": exceed = π1-x CSQA 1.10, π1-x HS 1.60, native HS 1.33, B_pb HS 1.60 (4); max point loss 0.60 pp. Correct. L323-325 attribution of the four to the right methods/benchmarks is correct; OBQA π1-x 0.53 is below.
- L317: "π1-x saved more than native-pb" on each of the three benchmarks (0.021, 0.121, 0.017): positive on all three; "dominated by CommonsenseQA" correct (0.1205 of 0.1586 summed).
- L317: OBQA "both adaptive policies split between loops 3 and 4" (π1-x 1,377/1,374; native 1,579/1,395): correct. HellaSwag "most exit at loop 3" (662/750, 714/750): correct. CSQA loop-2 exits 349 vs 1: correct.
- L342: "native-pb retreats further": drops 0.173 (native) vs 0.098 (π1-x). Correct. "largest single change is native-pb on OpenBookQA, 0.403 at 1.57 pp to 0.136 at 0.20 pp": drop 0.267, largest of six. Correct.
- L344, L447: AUROC of π1-x score for stability label 0.6889 vs native CDF 0.6025 (E1, primary mean, out-of-fold): π1-x higher, correct; it is E1 development data [E]. Native point-only thresholds are tiny (OBQA 1.3e-5 to 3.8e-5 across E1 folds; 1.9e-5 frozen): correct.
- L352: overhead 0.031 (OBQA) vs 0.007 (CSQA) and 0.007 (HS); "OBQA where loops are shortest" correct (20.7 ms vs 79.9/148.8; ρ_total 0.050 vs 0.014/0.015). "Overhead reduces advantage but does not remove it": 0.0529 → 0.0376, CI lower bound 0.033 > 0. Correct.
- L376: "all readouts other than r-kl trail π1-x by 0.060 to 0.090" (r-ent 0.0603 min, r-hl2 0.0901 max; native-pb is the primary comparator, not a "readout", and sits at 0.053 outside that range, consistent with the sentence). "Largest baseline point loss 0.93 pp for r-dlogit on HellaSwag": correct.
- L380: dev in-sample 0.246/0.243/0.193 are `frozen.selection_macro_saving_dev` (in-sample, selection data); E1 nested 0.258/0.207/0.204 are `e1.A6.risk` held-out. Attribution correct; "TEST matches in-sample rather than E1": TEST π1-x 0.239 vs r-kl 0.238 (near-tie) — correct.
- L240, L289, L299, L413: G3a −0.0236 is nested held-out on development data (primary three-benchmark macro, `improvement_vs_comparator`); the six-benchmark value (−0.0093) is not what is quoted. −0.029 is from the exposed Phase 1 validation (`D2_validation_exposed`). Per-fold range −0.044 to −0.014 correct. Attribution correct.
- L297: π1-x within point budget on all three primary benchmarks (0.17/0.60/0.40) and the PASS verdict — correct. Lower bound 0.049 > δ=0.02 — correct.
- L183, L240: E1 "3 of 15 held-out cells above 1 pp for both" — correct (n_violations_primary 3 and 3); E1 +0.053 positive in all 5 folds — correct (0.037 to 0.083).
- L327: "bounds of the frozen candidates were between 0.75 and 0.98 pp for the two adaptive methods" — frozen.primary UBs range 0.755 to 0.977. Correct. "HellaSwag exceeds 1 pp for every method including fixed depth" — correct (1.60, 1.33, 1.60).
- L386: OOD pooled τ π1-x 0.961 and native 0.386, both m=4, dev macro 0.159 each — correct (0.1588 and 0.1591).

Observations that are not errors but worth a reader's attention:

- L289/L413 give −0.024 for G3a while L295/L299 give −0.0236; consistent (rounding of the same value).
- L376 sentence "All the readouts other than r-kl trail π1-x by between 0.060 and 0.090" lists seven items only if native-pb is counted as a readout; the six untrained readouts besides r-kl are r-conf, r-margin, r-ent, r-dlogit, r-hcos, r-hl2 (0.060 to 0.090). Fine as written.
- Table 6 "Δ" column equals macro(π1-x) − macro(baseline) to rounding for every row, and the two are consistent with the CI centres.

## 4. Full checklist

Columns: item, draft line(s), source, recomputed value (in the unit/precision of the draft plus extra digits), status, note. Exit histograms and sums are checked for every Table 4 and Table 7 row.

| # | Item | Draft line(s) | Source | Recomputed | Status | Note |
|---|---|---|---|---|---|---|
| 1 | Δ π1x−native | 34,292,411 | test_report primary.delta_gross | 0.0528889 | OK |  [shown 0.053] |
| 2 | CI lo | 34,292,365,412 | primary.ci95[0] | 0.0487500 | OK |  [shown 0.049] |
| 3 | CI hi | 34,292,365,412 | primary.ci95[1] | 0.0571389 | OK |  [shown 0.057] |
| 4 | point-only Δ TEST | 36,291,299,348,414,525 | secondary_point_rule.delta | -0.0220000 | OK |  [shown -0.022] |
| 5 | point-only CI lo | 36,291,348,414 | secondary_point_rule.ci95[0] | -0.0258056 | OK |  [shown -0.026] |
| 6 | point-only CI hi | 36,291,348,414 | secondary_point_rule.ci95[1] | -0.0182215 | OK |  [shown -0.018] |
| 7 | Δ vs r-kl | 38,372,378,.. | per_baseline.r-kl.delta | 0.00125000 | ROUNDING | exact value 0.00125 (tie): half-up would be 0.0013 [shown 0.0012] |
| 8 | r-kl CI lo | 38,372,378 | per_baseline.r-kl.ci95[0] | -0.00405625 | OK |  [shown -0.0041] |
| 9 | r-kl CI hi | 38,372,378 | per_baseline.r-kl.ci95[1] | 0.00650000 | OK |  [shown 0.0065] |
| 10 | SIQA π1-x loss pp | 41,393,399 | secondary_ood.social_iqa.summary.pi1-x.loss_pp | 3.264249 | OK |  [shown 3.26] |
| 11 | SIQA π1-x UB pp | 41,393,399 | secondary_ood.social_iqa.upper_bound_95_pp.pi1-x | 4.715026 | OK |  [shown 4.72] |
| 12 | # upper bounds >1pp (of 9) | 43,321,472 | budget_table | 4.0000 | OK |  [shown 4] |
| 13 | # point losses >1pp (of 9) | 43,321,472 | budget_table | 0.0000 | OK |  [shown 0] |
| 14 | 8,517 TEST total | 31,219,221,231 | test_report.n_by_set sum | 8517.0000 | OK |  [shown 8517] |
| 15 | dev sum 20,250 | 219,225 | g3a n + n_by_source | 20250.0000 | OK |  [shown 20250] |
| 16 | P1 train 17,615 | 219,225 | g3a n_by_source.p1_train | 17615.0000 | OK |  [shown 17615] |
| 17 | DEV+ 2,635 | 219,225 | g3a n_by_source.devplus | 2635.0000 | OK |  [shown 2635] |
| 18 | 17615+2635 | 225 | arith | 20250.0000 | OK |  [shown 20250] |
| 19 | Phase1 val 5,870 | 219 | e1 n.val sum | 5870.0000 | OK |  [shown 5870] |
| 20 | TEST n openbookqa | 221,229,230 | test_report.test_metadata / manifest n_rows | 3000.0000 | OK |  [shown 3000] |
| 21 | manifest n openbookqa | 221 | manifest sets.test/openbookqa.n_rows | 3000.0000 | OK |  [shown 3000] |
| 22 | TEST n commonsense_qa | 221,229,230 | test_report.test_metadata / manifest n_rows | 1000.0000 | OK |  [shown 1000] |
| 23 | manifest n commonsense_qa | 221 | manifest sets.test/commonsense_qa.n_rows | 1000.0000 | OK |  [shown 1000] |
| 24 | TEST n hellaswag | 221,229,230 | test_report.test_metadata / manifest n_rows | 750.0000 | OK |  [shown 750] |
| 25 | manifest n hellaswag | 221 | manifest sets.test/hellaswag.n_rows | 750.0000 | OK |  [shown 750] |
| 26 | TEST n piqa | 221,229,230 | test_report.test_metadata / manifest n_rows | 1837.0000 | OK |  [shown 1837] |
| 27 | manifest n piqa | 221 | manifest sets.test/piqa.n_rows | 1837.0000 | OK |  [shown 1837] |
| 28 | TEST n social_iqa | 221,229,230 | test_report.test_metadata / manifest n_rows | 1930.0000 | OK |  [shown 1930] |
| 29 | manifest n social_iqa | 221 | manifest sets.test/social_iqa.n_rows | 1930.0000 | OK |  [shown 1930] |
| 30 | primary total 4,750 | 221,Table2,.. | 3000+1000+750 | 4750.0000 | OK |  [shown 4750] |
| 31 | ρ_score openbookqa | 197-199 T1 | frozen.rho.openbookqa.rho_score | 0.0356560 | OK |  [shown 0.036] |
| 32 | ρ_rest openbookqa | 197-199 T1 | frozen.rho.rho_rest | 0.0139259 | OK |  [shown 0.014] |
| 33 | ρ_total openbookqa | 197-199 T1 | frozen.rho.rho_total | 0.0495819 | OK |  [shown 0.050] |
| 34 | loop ms openbookqa | 197-199 T1 | frozen.rho.loop_ms | 20.72552 | OK |  [shown 20.7] |
| 35 | n_questions rho openbookqa | 193 | frozen.rho.n_questions | 200.0000 | OK |  [shown 200] |
| 36 | ρ_score commonsense_qa | 197-199 T1 | frozen.rho.commonsense_qa.rho_score | 0.0096941 | OK |  [shown 0.010] |
| 37 | ρ_rest commonsense_qa | 197-199 T1 | frozen.rho.rho_rest | 0.0042821 | OK |  [shown 0.004] |
| 38 | ρ_total commonsense_qa | 197-199 T1 | frozen.rho.rho_total | 0.0139761 | OK |  [shown 0.014] |
| 39 | loop ms commonsense_qa | 197-199 T1 | frozen.rho.loop_ms | 79.87238 | OK |  [shown 79.9] |
| 40 | n_questions rho commonsense_qa | 193 | frozen.rho.n_questions | 200.0000 | OK |  [shown 200] |
| 41 | ρ_score hellaswag | 197-199 T1 | frozen.rho.hellaswag.rho_score | 0.0125404 | OK |  [shown 0.013] |
| 42 | ρ_rest hellaswag | 197-199 T1 | frozen.rho.rho_rest | 0.0027394 | OK |  [shown 0.003] |
| 43 | ρ_total hellaswag | 197-199 T1 | frozen.rho.rho_total | 0.0152798 | OK |  [shown 0.015] |
| 44 | loop ms hellaswag | 197-199 T1 | frozen.rho.loop_ms | 148.80459 | OK |  [shown 148.8] |
| 45 | n_questions rho hellaswag | 193 | frozen.rho.n_questions | 200.0000 | OK |  [shown 200] |
| 46 | G3a nested Δ | 289,299,413 (−0.024 / −0.0236) | g3a D2.candidates.pi1-x.improvement_vs_comparator | -0.0236231 | OK |  [shown -0.024] |
| 47 | G3a nested Δ 4dp | 295,299 | same | -0.02362307 | OK |  [shown -0.0236] |
| 48 | G3a fold min | 289 | g3a D2.candidates.pi1-x.per_fold min | -0.0438370 | OK |  [shown -0.044] |
| 49 | G3a fold max | 289 | max | -0.0135058 | OK | -0.01351 -> -0.014 half-up OK [shown -0.014] |
| 50 | G3a exposed val Δ | 290 | g3a D2_validation_exposed.improvement_vs_comparator.pi1-x | -0.0288993 | OK |  [shown -0.029] |
| 51 | E1 point Δ | 240,.. | e1 A6.point.pi1-x.improvement_vs_native_pb | -0.0236231 | OK |  [shown -0.024] |
| 52 | E1 risk Δ | 240 | e1 A6.risk.pi1-x.improvement_vs_native_pb | 0.0533204 | OK |  [shown 0.053] |
| 53 | E1 risk folds all >0 | 240 | A6.risk.pi1-x per_fold | True | OK |  |
| 54 | net Δ | 293,352,412,523 | secondary_net.delta_net | 0.0375777 | OK |  [shown 0.038] |
| 55 | net CI lo | 293,352 | secondary_net.ci95[0] | 0.0333751 | OK |  [shown 0.033] |
| 56 | net CI hi | 293,352 | ci95[1] | 0.0418967 | OK |  [shown 0.042] |
| 57 | G3a required 0.02 | 239,299 | g3a thresholds.D2 | 0.020000 | OK |  [shown 0.02] |
| 58 | π1-x loss OBQA | 297,Table4 | budget_table.pi1-x.openbookqa.loss_pp | 0.166667 | OK |  [shown 0.17] |
| 59 | π1-x loss CSQA | 297,Table4 | budget_table.pi1-x.commonsense_qa.loss_pp | 0.600000 | OK |  [shown 0.60] |
| 60 | π1-x loss HS | 297,Table4 | budget_table.pi1-x.hellaswag.loss_pp | 0.400000 | OK |  [shown 0.40] |
| 61 | π1-x macro | 297,333,374 | macro_gross_saving.pi1-x | 0.2392778 | OK |  [shown 0.239] |
| 62 | native macro | 297,334,365 | macro_gross_saving.native-pb | 0.1863889 | OK |  [shown 0.186] |
| 63 | L>δ (0.049>0.02) | 297 | ci95[0]>0.02 | True | OK |  |
| 64 | realised half-width | 297 | (ci95[1]-ci95[0])/2 | 0.00419444 | OK |  [shown 0.0042] |
| 65 | expected half-width | 270,297 | frozen.expected_precision.expected_half_width_95 | 0.00431477 | OK |  [shown 0.0043] |
| 66 | T4 OBQA pi1-x saving | 305-313 | per_benchmark.pi1-x.openbookqa.saving | 0.1567500 | OK |  [shown 0.157] |
| 67 | T4 OBQA pi1-x loss | 305-313 | loss_pp | 0.166667 | OK |  [shown 0.17] |
| 68 | T4 OBQA pi1-x UB | 305-313 | budget_table.pi1-x.openbookqa.upper_bound_95_pp | 0.533333 | OK |  [shown 0.53] |
| 69 | T4 OBQA pi1-x exits | 305-313 | per_benchmark.pi1-x.openbookqa.exit_hist | 6/243/1377/1374 | OK | draft [6, 243, 1377, 1374] |
| 70 | T4 OBQA pi1-x exits sum=n | 305-313 | exit_hist sum | 3000 | OK |  |
| 71 | T4 OBQA native-pb saving | 305-313 | per_benchmark.native-pb.openbookqa.saving | 0.1359167 | OK |  [shown 0.136] |
| 72 | T4 OBQA native-pb loss | 305-313 | loss_pp | 0.200000 | OK |  [shown 0.20] |
| 73 | T4 OBQA native-pb UB | 305-313 | budget_table.native-pb.openbookqa.upper_bound_95_pp | 0.666667 | OK |  [shown 0.67] |
| 74 | T4 OBQA native-pb exits | 305-313 | per_benchmark.native-pb.openbookqa.exit_hist | 0/26/1579/1395 | OK | draft [0, 26, 1579, 1395] |
| 75 | T4 OBQA native-pb exits sum=n | 305-313 | exit_hist sum | 3000 | OK |  |
| 76 | T4 OBQA B_pb saving | 305-313 | per_benchmark.B_pb.openbookqa.saving | 0.0000000 | OK |  [shown 0.000] |
| 77 | T4 OBQA B_pb loss | 305-313 | loss_pp | 0.000000 | OK |  [shown 0.00] |
| 78 | T4 OBQA B_pb UB | 305-313 | budget_table.B_pb.openbookqa.upper_bound_95_pp | 0.000000 | OK |  [shown 0.00] |
| 79 | T4 OBQA B_pb exits | 305-313 | per_benchmark.B_pb.openbookqa.exit_hist | 0/0/0/3000 | OK | draft [0, 0, 0, 3000] |
| 80 | T4 OBQA B_pb exits sum=n | 305-313 | exit_hist sum | 3000 | OK |  |
| 81 | T4 CSQA pi1-x saving | 305-313 | per_benchmark.pi1-x.commonsense_qa.saving | 0.2817500 | OK |  [shown 0.282] |
| 82 | T4 CSQA pi1-x loss | 305-313 | loss_pp | 0.600000 | OK |  [shown 0.60] |
| 83 | T4 CSQA pi1-x UB | 305-313 | budget_table.pi1-x.commonsense_qa.upper_bound_95_pp | 1.100000 | OK |  [shown 1.10] |
| 84 | T4 CSQA pi1-x exits | 305-313 | per_benchmark.pi1-x.commonsense_qa.exit_hist | 8/349/405/238 | OK | draft [8, 349, 405, 238] |
| 85 | T4 CSQA pi1-x exits sum=n | 305-313 | exit_hist sum | 1000 | OK |  |
| 86 | T4 CSQA native-pb saving | 305-313 | per_benchmark.native-pb.commonsense_qa.saving | 0.1612500 | OK |  [shown 0.161] |
| 87 | T4 CSQA native-pb loss | 305-313 | loss_pp | -0.100000 | OK |  [shown -0.10] |
| 88 | T4 CSQA native-pb UB | 305-313 | budget_table.native-pb.commonsense_qa.upper_bound_95_pp | 0.300000 | OK |  [shown 0.30] |
| 89 | T4 CSQA native-pb exits | 305-313 | per_benchmark.native-pb.commonsense_qa.exit_hist | 0/1/643/356 | OK | draft [0, 1, 643, 356] |
| 90 | T4 CSQA native-pb exits sum=n | 305-313 | exit_hist sum | 1000 | OK |  |
| 91 | T4 CSQA B_pb saving | 305-313 | per_benchmark.B_pb.commonsense_qa.saving | 0.0000000 | OK |  [shown 0.000] |
| 92 | T4 CSQA B_pb loss | 305-313 | loss_pp | 0.000000 | OK |  [shown 0.00] |
| 93 | T4 CSQA B_pb UB | 305-313 | budget_table.B_pb.commonsense_qa.upper_bound_95_pp | 0.000000 | OK |  [shown 0.00] |
| 94 | T4 CSQA B_pb exits | 305-313 | per_benchmark.B_pb.commonsense_qa.exit_hist | 0/0/0/1000 | OK | draft [0, 0, 0, 1000] |
| 95 | T4 CSQA B_pb exits sum=n | 305-313 | exit_hist sum | 1000 | OK |  |
| 96 | T4 HS pi1-x saving | 305-313 | per_benchmark.pi1-x.hellaswag.saving | 0.2793333 | OK |  [shown 0.279] |
| 97 | T4 HS pi1-x loss | 305-313 | loss_pp | 0.400000 | OK |  [shown 0.40] |
| 98 | T4 HS pi1-x UB | 305-313 | budget_table.pi1-x.hellaswag.upper_bound_95_pp | 1.600000 | OK |  [shown 1.60] |
| 99 | T4 HS pi1-x exits | 305-313 | per_benchmark.pi1-x.hellaswag.exit_hist | 0/88/662/0 | OK | draft [0, 88, 662, 0] |
| 100 | T4 HS pi1-x exits sum=n | 305-313 | exit_hist sum | 750 | OK |  |
| 101 | T4 HS native-pb saving | 305-313 | per_benchmark.native-pb.hellaswag.saving | 0.2620000 | OK |  [shown 0.262] |
| 102 | T4 HS native-pb loss | 305-313 | loss_pp | 0.133333 | OK |  [shown 0.13] |
| 103 | T4 HS native-pb UB | 305-313 | budget_table.native-pb.hellaswag.upper_bound_95_pp | 1.333333 | OK |  [shown 1.33] |
| 104 | T4 HS native-pb exits | 305-313 | per_benchmark.native-pb.hellaswag.exit_hist | 0/36/714/0 | OK | draft [0, 36, 714, 0] |
| 105 | T4 HS native-pb exits sum=n | 305-313 | exit_hist sum | 750 | OK |  |
| 106 | T4 HS B_pb saving | 305-313 | per_benchmark.B_pb.hellaswag.saving | 0.2500000 | OK |  [shown 0.250] |
| 107 | T4 HS B_pb loss | 305-313 | loss_pp | 0.533333 | OK |  [shown 0.53] |
| 108 | T4 HS B_pb UB | 305-313 | budget_table.B_pb.hellaswag.upper_bound_95_pp | 1.600000 | OK |  [shown 1.60] |
| 109 | T4 HS B_pb exits | 305-313 | per_benchmark.B_pb.hellaswag.exit_hist | 0/0/750/0 | OK | draft [0, 0, 750, 0] |
| 110 | T4 HS B_pb exits sum=n | 305-313 | exit_hist sum | 750 | OK |  |
| 111 | T4 m_b OBQA pi1-x | 305-313 | frozen.primary/reference_B_pb m | 4 | OK |  |
| 112 | T4 m_b OBQA native-pb | 305-313 | frozen.primary/reference_B_pb m | 4 | OK |  |
| 113 | T4 m_b OBQA B_pb | 305-313 | frozen.primary/reference_B_pb m | 4 | OK |  |
| 114 | T4 m_b CSQA pi1-x | 305-313 | frozen.primary/reference_B_pb m | 4 | OK |  |
| 115 | T4 m_b CSQA native-pb | 305-313 | frozen.primary/reference_B_pb m | 4 | OK |  |
| 116 | T4 m_b CSQA B_pb | 305-313 | frozen.primary/reference_B_pb m | 4 | OK |  |
| 117 | T4 m_b HS pi1-x | 305-313 | frozen.primary/reference_B_pb m | 3 | OK |  |
| 118 | T4 m_b HS native-pb | 305-313 | frozen.primary/reference_B_pb m | 3 | OK |  |
| 119 | T4 m_b HS B_pb | 305-313 | frozen.primary/reference_B_pb m | 3 | OK |  |
| 120 | Δ saving OBQA | 317 | π1-x−native per_benchmark saving | 0.0208333 | OK |  [shown 0.021] |
| 121 | Δ saving CSQA | 317 | π1-x−native per_benchmark saving | 0.1205000 | OK |  [shown 0.121] |
| 122 | Δ saving HS | 317 | π1-x−native per_benchmark saving | 0.0173333 | OK |  [shown 0.017] |
| 123 | Δ from 3dp table entries CSQA | 317 | 0.282-0.161 | 0.1210000 | OK |  [shown 0.121] |
| 124 | mean of three diffs = macro Δ | 317 |  | 0.0528889 | OK |  [shown 0.053] |
| 125 | HS exits π1 662 of 750 | 317 |  | 662.0000 | OK |  [shown 662] |
| 126 | selection-time UB min (pp) | 327 | frozen.primary upper_bound min (pp) | 0.754885 | OK |  [shown 0.75] |
| 127 | selection-time UB max (pp) | 327 | frozen.primary upper_bound max (pp) | 0.976909 | OK |  [shown 0.98] |
| 128 | T5 point pi1-x OBQA saving | 331-332 | secondary_point_rule.per_benchmark.pi1-x.openbookqa.saving | 0.3052500 | OK |  [shown 0.305] |
| 129 | T5 point pi1-x OBQA loss | 331-332 | loss_pp | 1.133333 | OK |  [shown 1.13] |
| 130 | T5 point pi1-x CSQA saving | 331-332 | secondary_point_rule.per_benchmark.pi1-x.commonsense_qa.saving | 0.3565000 | OK |  [shown 0.357] |
| 131 | T5 point pi1-x CSQA loss | 331-332 | loss_pp | 1.100000 | OK |  [shown 1.10] |
| 132 | T5 point pi1-x HS saving | 331-332 | secondary_point_rule.per_benchmark.pi1-x.hellaswag.saving | 0.3506667 | OK |  [shown 0.351] |
| 133 | T5 point pi1-x HS loss | 331-332 | loss_pp | 1.066667 | OK |  [shown 1.07] |
| 134 | T5 point native-pb OBQA saving | 331-332 | secondary_point_rule.per_benchmark.native-pb.openbookqa.saving | 0.4031667 | OK |  [shown 0.403] |
| 135 | T5 point native-pb OBQA loss | 331-332 | loss_pp | 1.566667 | OK |  [shown 1.57] |
| 136 | T5 point native-pb CSQA saving | 331-332 | secondary_point_rule.per_benchmark.native-pb.commonsense_qa.saving | 0.3592500 | OK |  [shown 0.359] |
| 137 | T5 point native-pb CSQA loss | 331-332 | loss_pp | 1.200000 | OK |  [shown 1.20] |
| 138 | T5 point native-pb HS saving | 331-332 | secondary_point_rule.per_benchmark.native-pb.hellaswag.saving | 0.3160000 | OK |  [shown 0.316] |
| 139 | T5 point native-pb HS loss | 331-332 | loss_pp | 0.666667 | OK |  [shown 0.67] |
| 140 | T5 macro point π1-x | 331,342,459 | mean of per-bench | 0.3374722 | OK |  [shown 0.337] |
| 141 | T5 macro point native | 332,342,459 | mean | 0.3594722 | OK |  [shown 0.359] |
| 142 | T5 macro risk π1-x | 333 | macro_gross_saving | 0.2392778 | OK |  [shown 0.239] |
| 143 | T5 macro risk native | 334 | macro_gross_saving | 0.1863889 | OK |  [shown 0.186] |
| 144 | drop native | 342,460 | 0.35947-0.18639 | 0.1730833 | OK |  [shown 0.173] |
| 145 | drop π1-x | 342,460 |  | 0.0981944 | OK |  [shown 0.098] |
| 146 | drop native from 3dp | 342 | 0.359-0.186 | 0.1730000 | OK |  [shown 0.173] |
| 147 | drop π1 from 3dp | 342 | 0.337-0.239 | 0.0980000 | OK |  [shown 0.098] |
| 148 | point Δ from macro | 299,Table5 | 0.337-0.359 | -0.0220000 | OK |  [shown -0.022] |
| 149 | #point cells >1pp π1-x | 338 |  | 3.0000 | OK |  [shown 3] |
| 150 | #point cells >1pp native | 338 |  | 2.0000 | OK |  [shown 2] |
| 151 | π1-x point loss min | 338 |  | 1.066667 | OK |  [shown 1.07] |
| 152 | π1-x point loss max | 338 |  | 1.133333 | OK |  [shown 1.13] |
| 153 | AUROC π1-x S | 344,447 | e1 A4.mean_auroc_primary.pi1-x.S | 0.6889154 | OK |  [shown 0.689] |
| 154 | AUROC native-cdf S | 344,447 | e1 A4.mean_auroc_primary.native-cdf.S | 0.6025184 | OK |  [shown 0.603] |
| 155 | net saving OBQA | 352 | per_benchmark.pi1-x.openbookqa.net_saving | 0.1254486 | OK |  [shown 0.125] |
| 156 | gross saving OBQA | 352 | saving | 0.1567500 | OK |  [shown 0.157] |
| 157 | net saving CSQA | 352 | per_benchmark.pi1-x.commonsense_qa.net_saving | 0.2743899 | OK |  [shown 0.274] |
| 158 | gross saving CSQA | 352 | saving | 0.2817500 | OK |  [shown 0.282] |
| 159 | net saving HS | 352 | per_benchmark.pi1-x.hellaswag.net_saving | 0.2720613 | OK |  [shown 0.272] |
| 160 | gross saving HS | 352 | saving | 0.2793333 | OK |  [shown 0.279] |
| 161 | overhead OBQA | 352 | saving-net_saving | 0.0313014 | OK |  [shown 0.031] |
| 162 | overhead CSQA | 352 | saving-net_saving | 0.0073601 | OK |  [shown 0.007] |
| 163 | overhead HS | 352 | saving-net_saving | 0.0072720 | OK |  [shown 0.007] |
| 164 | net Δ recompute (macro net π1 − native gross) | 293,352 |  | 0.0375777 | OK | point-estimate cross-check vs stored delta_net [shown 0.038] |
| 165 | T6 B_pb macro | 363-372 | macro_gross_saving.B_pb | 0.0833333 | OK |  [shown 0.083] |
| 166 | T6 B_pb Δ | 363-372 | per_baseline.B_pb.delta | 0.1559444 | OK |  [shown 0.156] |
| 167 | T6 B_pb CI lo | 363-372 | ci95[0] | 0.1511667 | OK |  [shown 0.151] |
| 168 | T6 B_pb CI hi | 363-372 | ci95[1] | 0.1607222 | OK |  [shown 0.161] |
| 169 | T6 B_pb Δ = π1x macro − baseline macro | 363-372 |  | 0.1559444 | OK | consistency [shown 0.156] |
| 170 | T6 native-pb macro | 363-372 | macro_gross_saving.native-pb | 0.1863889 | OK |  [shown 0.186] |
| 171 | T6 native-pb Δ | 363-372 | per_baseline.native-pb.delta | 0.0528889 | OK |  [shown 0.053] |
| 172 | T6 native-pb CI lo | 363-372 | ci95[0] | 0.0487500 | OK |  [shown 0.049] |
| 173 | T6 native-pb CI hi | 363-372 | ci95[1] | 0.0571389 | OK |  [shown 0.057] |
| 174 | T6 native-pb Δ = π1x macro − baseline macro | 363-372 |  | 0.0528889 | OK | consistency [shown 0.053] |
| 175 | T6 r-conf macro | 363-372 | macro_gross_saving.r-conf | 0.1714444 | OK |  [shown 0.171] |
| 176 | T6 r-conf Δ | 363-372 | per_baseline.r-conf.delta | 0.0678333 | OK |  [shown 0.068] |
| 177 | T6 r-conf CI lo | 363-372 | ci95[0] | 0.0625833 | OK |  [shown 0.063] |
| 178 | T6 r-conf CI hi | 363-372 | ci95[1] | 0.0730556 | OK |  [shown 0.073] |
| 179 | T6 r-conf Δ = π1x macro − baseline macro | 363-372 |  | 0.0678333 | OK | consistency [shown 0.068] |
| 180 | T6 r-margin macro | 363-372 | macro_gross_saving.r-margin | 0.1699444 | OK |  [shown 0.170] |
| 181 | T6 r-margin Δ | 363-372 | per_baseline.r-margin.delta | 0.0693333 | OK |  [shown 0.069] |
| 182 | T6 r-margin CI lo | 363-372 | ci95[0] | 0.0641667 | OK |  [shown 0.064] |
| 183 | T6 r-margin CI hi | 363-372 | ci95[1] | 0.0746111 | OK |  [shown 0.075] |
| 184 | T6 r-margin Δ = π1x macro − baseline macro | 363-372 |  | 0.0693333 | OK | consistency [shown 0.069] |
| 185 | T6 r-ent macro | 363-372 | macro_gross_saving.r-ent | 0.1789444 | OK |  [shown 0.179] |
| 186 | T6 r-ent Δ | 363-372 | per_baseline.r-ent.delta | 0.0603333 | OK |  [shown 0.060] |
| 187 | T6 r-ent CI lo | 363-372 | ci95[0] | 0.0552500 | OK |  [shown 0.055] |
| 188 | T6 r-ent CI hi | 363-372 | ci95[1] | 0.0654167 | OK |  [shown 0.065] |
| 189 | T6 r-ent Δ = π1x macro − baseline macro | 363-372 |  | 0.0603333 | OK | consistency [shown 0.060] |
| 190 | T6 r-dlogit macro | 363-372 | macro_gross_saving.r-dlogit | 0.1623333 | OK |  [shown 0.162] |
| 191 | T6 r-dlogit Δ | 363-372 | per_baseline.r-dlogit.delta | 0.0769444 | OK |  [shown 0.077] |
| 192 | T6 r-dlogit CI lo | 363-372 | ci95[0] | 0.0705278 | OK |  [shown 0.071] |
| 193 | T6 r-dlogit CI hi | 363-372 | ci95[1] | 0.0830556 | OK |  [shown 0.083] |
| 194 | T6 r-dlogit Δ = π1x macro − baseline macro | 363-372 |  | 0.0769444 | OK | consistency [shown 0.077] |
| 195 | T6 r-hcos macro | 363-372 | macro_gross_saving.r-hcos | 0.1740278 | OK |  [shown 0.174] |
| 196 | T6 r-hcos Δ | 363-372 | per_baseline.r-hcos.delta | 0.0652500 | OK |  [shown 0.065] |
| 197 | T6 r-hcos CI lo | 363-372 | ci95[0] | 0.0588882 | OK |  [shown 0.059] |
| 198 | T6 r-hcos CI hi | 363-372 | ci95[1] | 0.0715833 | OK |  [shown 0.072] |
| 199 | T6 r-hcos Δ = π1x macro − baseline macro | 363-372 |  | 0.0652500 | OK | consistency [shown 0.065] |
| 200 | T6 r-hl2 macro | 363-372 | macro_gross_saving.r-hl2 | 0.1491389 | OK |  [shown 0.149] |
| 201 | T6 r-hl2 Δ | 363-372 | per_baseline.r-hl2.delta | 0.0901389 | OK |  [shown 0.090] |
| 202 | T6 r-hl2 CI lo | 363-372 | ci95[0] | 0.0837215 | OK |  [shown 0.084] |
| 203 | T6 r-hl2 CI hi | 363-372 | ci95[1] | 0.0963896 | OK |  [shown 0.096] |
| 204 | T6 r-hl2 Δ = π1x macro − baseline macro | 363-372 |  | 0.0901389 | OK | consistency [shown 0.090] |
| 205 | T6 r-kl macro | 363-372 | macro_gross_saving.r-kl | 0.2380278 | OK |  [shown 0.238] |
| 206 | T6 r-kl Δ | 363-372 | per_baseline.r-kl.delta | 0.00125000 | ROUNDING | tie [shown 0.0012] |
| 207 | T6 r-kl CI lo | 363-372 | ci95[0] | -0.00405625 | OK |  [shown -0.0041] |
| 208 | T6 r-kl CI hi | 363-372 | ci95[1] | 0.00650000 | OK |  [shown 0.0065] |
| 209 | T6 r-kl Δ = π1x macro − baseline macro | 363-372 |  | 0.00125000 | ROUNDING | consistency [shown 0.0012] |
| 210 | π1-x macro in T6 caption | 374 |  | 0.2392778 | OK |  [shown 0.239] |
| 211 | readouts excl r-kl: min Δ (but native-pb not a readout) | 376 |  | 0.0603333 | OK |  [shown 0.060] |
| 212 | readouts excl r-kl: max Δ | 376 |  | 0.0901389 | OK |  [shown 0.090] |
| 213 | B_pb Δ | 376 |  | 0.1559444 | OK |  [shown 0.156] |
| 214 | largest baseline loss pp | 376 | secondary_violations max | 0.933333 | OK | r-dlogit hellaswag [shown 0.93] |
| 215 | largest-loss baseline identity | 376 | secondary_violations | r-dlogit on hellaswag | OK |  |
| 216 | all baseline losses <=1pp | 376 |  | True | OK |  |
| 217 | r-kl reselected freq | 378 | selection_frequency.r-kl | 10000.0000 | OK |  [shown 10000] |
| 218 | reselecting CI == r-kl CI | 378 |  | True | OK |  |
| 219 | 0.65 pp (CI hi) | 378 |  | 0.650000 | OK | 0.0065 -> 0.65 pp [shown 0.65] |
| 220 | 0.41 pp (−CI lo) | 378 |  | 0.405625 | OK |  [shown 0.41] |
| 221 | r-kl − native macro | 380 |  | 0.0516389 | OK |  [shown 0.052] |
| 222 | r-kl macro | 380 |  | 0.2380278 | OK |  [shown 0.238] |
| 223 | dev macro π1-x | 380 | frozen.selection_macro_saving_dev | 0.2459640 | OK |  [shown 0.246] |
| 224 | dev macro r-kl | 380 |  | 0.2432643 | OK |  [shown 0.243] |
| 225 | dev macro native | 380 |  | 0.1932611 | OK |  [shown 0.193] |
| 226 | E1 nested π1-x | 380 | e1 A6.risk.pi1-x.heldout_saving_primary | 0.2576517 | OK |  [shown 0.258] |
| 227 | E1 nested r-kl | 380 |  | 0.2070164 | OK |  [shown 0.207] |
| 228 | E1 nested native | 380 |  | 0.2043314 | OK |  [shown 0.204] |
| 229 | E1 viol native | 183,Prot | e1 A6.risk.native-pb.n_violations_primary | 3.0000 | OK |  [shown 3] |
| 230 | E1 viol π1-x | 183 |  | 3.0000 | OK |  [shown 3] |
| 231 | r-kl KL thr OBQA | 382 | -tau | 0.00349503 | OK | 3.5e-3 [shown 0.0035] |
| 232 | r-kl cap OBQA | 382 | m | 3.0000 | OK |  [shown 3] |
| 233 | r-kl KL thr CSQA | 382 | -tau | 0.0374730 | OK | 3.7e-2 [shown 0.037] |
| 234 | r-kl cap CSQA | 382 | m | 4.0000 | OK |  [shown 4] |
| 235 | r-kl KL thr HS | 382 | -tau (×1e4) | 2.70631 | OK | 2.7e-4 -> shown as 2.7 ×1e-4 [shown 2.7] |
| 236 | r-kl cap HS | 382 | m | 3.0000 | OK |  [shown 3] |
| 237 | OOD π1-x τ | 386 | frozen.ood_pooled.pi1-x.tau | 0.9606691 | OK |  [shown 0.961] |
| 238 | OOD native τ | 386 |  | 0.3863837 | OK |  [shown 0.386] |
| 239 | OOD m π1-x | 386 |  | 4.0000 | OK |  [shown 4] |
| 240 | OOD m native | 386 |  | 4.0000 | OK |  [shown 4] |
| 241 | OOD dev macro π1-x | 386 | macro_saving | 0.1588239 | OK |  [shown 0.159] |
| 242 | OOD dev macro native | 386 |  | 0.1590735 | OK |  [shown 0.159] |
| 243 | T7 PIQA pi1-x saving | 390-395 | secondary_ood.piqa.summary.pi1-x.saving | 0.3082471 | OK |  [shown 0.308] |
| 244 | T7 PIQA pi1-x loss | 390-395 | loss_pp | 0.707676 | OK |  [shown 0.71] |
| 245 | T7 PIQA pi1-x UB | 390-395 | upper_bound_95_pp.pi1-x | 1.524224 | OK |  [shown 1.52] |
| 246 | T7 PIQA pi1-x exits | 390-395 | exit_hist | 0/433/1399/5 | OK |  |
| 247 | T7 PIQA pi1-x exits sum=n | 390-395 |  | 1837 | OK |  |
| 248 | T7 PIQA native-pb saving | 390-395 | secondary_ood.piqa.summary.native-pb.saving | 0.1256124 | OK |  [shown 0.126] |
| 249 | T7 PIQA native-pb loss | 390-395 | loss_pp | 0.108873 | OK |  [shown 0.11] |
| 250 | T7 PIQA native-pb UB | 390-395 | upper_bound_95_pp.native-pb | 0.598802 | OK |  [shown 0.60] |
| 251 | T7 PIQA native-pb exits | 390-395 | exit_hist | 0/0/923/914 | OK |  |
| 252 | T7 PIQA native-pb exits sum=n | 390-395 |  | 1837 | OK |  |
| 253 | T7 PIQA B_pb saving | 390-395 | secondary_ood.piqa.summary.B_pb.saving | 0.0000000 | OK |  [shown 0.000] |
| 254 | T7 PIQA B_pb loss | 390-395 | loss_pp | 0.000000 | OK |  [shown 0.00] |
| 255 | T7 PIQA B_pb UB | 390-395 | upper_bound_95_pp.B_pb | 0.000000 | OK |  [shown 0.00] |
| 256 | T7 PIQA B_pb exits | 390-395 | exit_hist | 0/0/0/1837 | OK |  |
| 257 | T7 PIQA B_pb exits sum=n | 390-395 |  | 1837 | OK |  |
| 258 | T7 SIQA pi1-x saving | 390-395 | secondary_ood.social_iqa.summary.pi1-x.saving | 0.4809585 | OK |  [shown 0.481] |
| 259 | T7 SIQA pi1-x loss | 390-395 | loss_pp | 3.264249 | OK |  [shown 3.26] |
| 260 | T7 SIQA pi1-x UB | 390-395 | upper_bound_95_pp.pi1-x | 4.715026 | OK |  [shown 4.72] |
| 261 | T7 SIQA pi1-x exits | 390-395 | exit_hist | 0/1785/143/2 | OK |  |
| 262 | T7 SIQA pi1-x exits sum=n | 390-395 |  | 1930 | OK |  |
| 263 | T7 SIQA native-pb saving | 390-395 | secondary_ood.social_iqa.summary.native-pb.saving | 0.0277202 | OK |  [shown 0.028] |
| 264 | T7 SIQA native-pb loss | 390-395 | loss_pp | -0.051813 | OK |  [shown -0.05] |
| 265 | T7 SIQA native-pb UB | 390-395 | upper_bound_95_pp.native-pb | 0.155440 | OK |  [shown 0.16] |
| 266 | T7 SIQA native-pb exits | 390-395 | exit_hist | 0/0/214/1716 | OK |  |
| 267 | T7 SIQA native-pb exits sum=n | 390-395 |  | 1930 | OK |  |
| 268 | T7 SIQA B_pb saving | 390-395 | secondary_ood.social_iqa.summary.B_pb.saving | 0.0000000 | OK |  [shown 0.000] |
| 269 | T7 SIQA B_pb loss | 390-395 | loss_pp | 0.000000 | OK |  [shown 0.00] |
| 270 | T7 SIQA B_pb UB | 390-395 | upper_bound_95_pp.B_pb | 0.000000 | OK |  [shown 0.00] |
| 271 | T7 SIQA B_pb exits | 390-395 | exit_hist | 0/0/0/1930 | OK |  |
| 272 | T7 SIQA B_pb exits sum=n | 390-395 |  | 1930 | OK |  |
| 273 | PIQA Δ π1x−native | 399 | secondary_ood.piqa.delta_pi1x_vs_native | 0.1826347 | OK |  [shown 0.183] |
| 274 | PIQA Δ CI lo | 399 |  | 0.1743332 | OK |  [shown 0.174] |
| 275 | PIQA Δ CI hi | 399 |  | 0.1910724 | OK |  [shown 0.191] |
| 276 | SIQA Δ | 399 |  | 0.4532383 | OK |  [shown 0.453] |
| 277 | SIQA Δ CI lo | 399 |  | 0.4485751 | OK |  [shown 0.449] |
| 278 | SIQA Δ CI hi | 399 |  | 0.4577720 | OK |  [shown 0.458] |
| 279 | SIQA acc4 % | 399 | summary.pi1-x.acc4_pp | 43.316062 | OK |  [shown 43.32] |
| 280 | SIQA acc π1-x % | 399 | acc_pp | 40.051813 | OK |  [shown 40.05] |
| 281 | SIQA loop-2 halts 1,785 | 399 |  | 1785.0000 | OK |  [shown 1785] |
| 282 | SIQA halted frac (most) | 399 | 1785/1930 | 0.9248705 | OK | 92.5% [shown 0.925] |
| 283 | SIQA native sav 0.028 / PIQA 0.126 in prose | 399 |  | 0.0277202 | OK |  [shown 0.028] |
| 284 |  | 399 |  | 0.1256124 | OK |  [shown 0.126] |
| 285 | all 9 point losses <=1pp | 43,321 |  | True | OK |  |
| 286 | max TEST point loss (primary, 9 cells) | 43,321 |  | 0.600000 | OK |  [shown 0.60] |
| 287 | A1.5 CSQA d=.05 | 270 | docs/PHASE3_AMENDMENT_A1.md A1.5 table | 1.16 | OK |  |
| 288 | A1.5 CSQA d=.10 | 270 | docs/PHASE3_AMENDMENT_A1.md A1.5 table | 1.65 | OK |  |
| 289 | A1.5 HS d=.05 | 270 | docs/PHASE3_AMENDMENT_A1.md A1.5 table | 1.34 | OK |  |
| 290 | A1.5 HS d=.10 | 270 | docs/PHASE3_AMENDMENT_A1.md A1.5 table | 1.90 | OK |  |
| 291 | D-a PIQA float32 max 1.17e-4 | 274 | PHASE3_EXTRACTION.md:231 | 1.17e-4 | OK |  |
| 292 | D-a SIQA float32 max 3.1e-5 | 274 | PHASE3_EXTRACTION.md:232 | 3.1e-5 | OK |  |
| 293 | D-a2 float64 max 1.67e-13 | 274,509 | PHASE3_EXTRACTION.md:404 | 1.67e-13 | OK |  |
| 294 | D-a2 criterion 1e-9 | 274 | PHASE3_EXTRACTION.md:407 | 1e-9 | OK |  |
| 295 | D-a tolerance 1e-4 | 274 | DECISIONS.md:198 | 1e-4 | OK |  |
| 296 | D-a rows 100/100 | 274 | PHASE3_EXTRACTION.md:183,226 | 100 | OK |  |
| 297 | D-a2 10 PIQA questions | 274 | PHASE3_EXTRACTION.md:399-400 | 10 | OK |  |
| 298 | C1 protocol: 3 of 15 viol | 183 | C1_PROTOCOL.md:31, E1_REPORT.md:25 | 3 of 15 | OK |  |
| 299 | Gate-1 tolerance 1e-12 | 178,253 | C1_PROTOCOL.md:69 | 1e-12 | OK |  |
| 300 | B=10,000; seed 2026 | 255 | C1_PROTOCOL.md:83; test_report n_boot, seed | 10000/2026 | OK |  |
| 301 | n_boot | 255 | test_report.n_boot | 10000.0000 | OK |  [shown 10000] |
| 302 | seed | 255 | test_report.seed | 2026.0000 | OK |  [shown 2026] |
| 303 | delta_min | 251 | test_report.delta_min | 0.020000 | OK |  [shown 0.02] |
| 304 | bootstrap resamples selection 2000 | 179 | frozen.bootstrap_selection.n | 2000.0000 | OK |  [shown 2000] |
| 305 | pi1x C | 148 | frozen.pi1x.C | 1.00000 | OK |  [shown 1.0] |
| 306 | # features | 148,156 | len(frozen.pi1x.features) | 9.0000 | OK |  [shown 9] |
| 307 | # candidates 51*4 | 176 |  | 204.0000 | OK |  [shown 204] |
| 308 | test code bundle files | 245 | provenance.marker.test_code_bundle_files | 24.0000 | OK |  [shown 24] |
| 309 | manifest sha b2df1930 | 221,245 | test_report.provenance.marker | b2df1930 | OK |  |
| 310 | protocol sha 4f38b8b7 | 241,249 | test_report.provenance.marker | 4f38b8b7 | OK |  |
| 311 | frozen sha 65728e16 | 242 | test_report.provenance.marker | 65728e16 | OK |  |
| 312 | model file sha 72f15508 | 242 | test_report.provenance.marker | 72f15508 | OK |  |
| 313 | bundle sha c0902af3 | 245 | test_report.provenance.marker | c0902af3 | OK |  |
| 314 | metric/shots test/openbookqa | 126 | manifest sets.test/openbookqa | acc_norm, 0-shot | OK |  |
| 315 | metric/shots test/commonsense_qa | 126 | manifest sets.test/commonsense_qa | acc, 10-shot | OK |  |
| 316 | metric/shots test/hellaswag | 126 | manifest sets.test/hellaswag | acc_norm, 10-shot | OK |  |
| 317 | metric/shots test/piqa | 126 | manifest sets.test/piqa | acc_norm, 0-shot | OK |  |
| 318 | metric/shots test/social_iqa | 126 | manifest sets.test/social_iqa | acc, 0-shot | OK |  |
| 319 | native point-only τ OBQA tiny (E1 folds 1.3e-5..3.8e-5; frozen 1.9e-5) | 344 | e1 A1.fold_selections_point; frozen.secondary_point_rule | 1.9e-05 | OK |  |
| 320 | Table 2 arithmetic 3000+1000+750+1837+1930 | 231 |  | 8517 | OK |  |
| 321 | verdict PASS | 292,297 | test_report.verdict | PASS | OK |  |
| 322 | modeling_ouro.py sha256 prefix c5c68fbb | 134 | sha256sum of hf_cache/ouro-1.4b/modeling_ouro.py | c5c68fbb368c... | OK | file present on disk, hash prefix matches |
| 323 | pi1x selection statistic = 5-fold OOF; refit on all dev | 158 | frozen.pi1x.selection_statistic | 5-fold out-of-fold over all development data | OK |  |
