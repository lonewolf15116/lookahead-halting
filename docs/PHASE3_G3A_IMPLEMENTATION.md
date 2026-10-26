# Phase 3 G3a (D1-D4): implementation and frozen-spec crosswalk (for reviewer sign-off; NOT RUN on real data)

Authorized by the reviewer's "Phase 3 extraction — Independent reviewer assessment" (2026-10-26). That authorization covers
writing D1-D4 and testing them on synthetic fixtures only. Running on real development data needs a separate sign-off of
this implementation. Training, model selection and TEST stay on HOLD.

Files: `src/lookahead/phase3_g3a.py` (logic and loader), `scripts/phase3_g3a.py` (guarded entry point),
`tests/test_phase3_g3a.py` (53 tests), `scripts/run_phase3_g3a_windows.ps1` (laptop runner; to be used only after
sign-off).

## Crosswalk: frozen specification → code → test
| frozen item (source) | implementation | test |
|---|---|---|
| G3a data = Phase 1 train + DEV+; validation exposed, reported only; TEST never read (A1.7) | `load_g3a`. Phase 1 `train` rows of all six benchmarks; DEV+ from `data/extracted_phase3_dev`; Phase 1 `validation` loaded separately, for the reported-only estimate only. `run_g3a` refuses development data containing validation or TEST rows | `test_validation_or_test_rows_in_development_data_are_refused` |
| TEST refused by allow-list (A1.7) | Every opened path goes through `assert_not_sealed` plus Gate 2's `confirm` guard. DEV+ ids must be in the frozen DEV+ list (hash-checked against the manifest). Any id on the frozen TEST list is refused. The module never names a sealed directory | `test_loader_refuses_test_ids_…`, `test_loader_refuses_sealed_directories_…`, `test_module_never_names_the_sealed_extraction` |
| Integrity: checksummed parts, aligned hidden states, 4 loops, no duplicates, manifest counts | The Gate 2 reader (`_read_part_with_hidden`: sha256, order and status alignment) and the Gate 1 `build_arrays` (4 loops, 0/1 labels). Duplicate ids are refused. DEV+ counts must equal the manifest. Option scores are row-aligned with the records | `test_loader_detects_a_tampered_part`, `test_duplicate_questions_are_refused`, count mismatch, `test_loader_reads_a_fixture_…` |
| Folds `lh-fold-v1` (plan 5, A1.3) | `gate2.fold_of`, the same hash for Phase 1 and DEV+ ids | (Gate 2 tests) |
| Macro and stop rules over the three primary benchmarks; six-benchmark versions informative only (A1.2) | `PRIMARY` drives every macro used by a rule. The `*_six` fields are reported only | dry run |
| Policy class, exit rule, 51-point grid, per-benchmark selection and tie-breaks, τ = +∞ nests fixed depth (plan 5, A2.2, A2.3) | **Frozen** `phase3_readouts`: `grid`, `select`, `exit_loops`, `outcome`, `question_stats`, unchanged (sha256 1b3a88a3… is checked before running) | `test_every_selected_point_respects_the_budget_…`, `test_readout_statistics_are_the_frozen_ones`, the A2 tests |
| B_pb = τ = +∞ subset, same rule (plan 5) | `select_all(..., fixed_only=True)` | same |
| Failed questions: 4 loops, wrong, never fire (Gate 1; A2.2) | Statistics are NaN for failed questions; `exit_loops` forces 4 | `test_failed_questions_are_never_halted_early`, `test_failed_questions_exit_at_loop_4_in_both_oracles` |
| **D1** (ii) accuracy-budget oracle, the exact algorithm, may keep wrong early answers, ties by doc_id; (i) stable-final for reference (plan 7) | `accuracy_budget_oracle`, `stable_final_oracle` | **Brute-force optimality on 25 random instances**; the worked example including the tie-break; equality with Gate 1's `stable_oracle` |
| D1 stop: ceiling (ii) − B_pb < 0.05, B_pb on all development data (plan 7, A1.7) | `d1`, `d1_stop`; exactly 0.05 continues | `test_d1_stop_boundary`, `test_d1_report_…` |
| D1 non-binding Δ_oracle-readout (A2.4) | `d1()["nonbinding_oracle_minus_best_baseline"]` | `test_d1_report_and_nonbinding_quantity` |
| **D2** candidates π1-x and π1-hx: family-1 per-loop L2 logistic on the stability target S_t, with Gate 2's **frozen** C (x 1.0, hx 0.001; checked against `results/gate2/frozen_selection.json`). (τ_b, m_b) are chosen on four folds' out-of-fold statistics and evaluated on the fifth, rotated (plan 6, plan 7). **Fully nested (rev 2):** inside each outer fold k, the selection-stage statistics come from an inner cross-fit over the four selection folds only, and fold k's statistics from models fitted on all four | `outer_fold` (with `crossfit(within=…)`), `d2`. Fitting uses all six benchmarks; selection is per benchmark (A1.2) | `test_outer_fold_selection_is_untouched_by_anything_in_the_held_out_fold` (k = 0, 3), `test_the_old_whole_data_crossfit_would_have_leaked`, `test_d2_methods_are_paired_on_identical_held_out_rows` |
| D2 comparator: baselines B_pb, native-pb and the seven readouts, each selected on the selection folds. Eligible if qualified there; strongest by selection-fold macro saving, ties by name; only its held-out saving is used; improvement = mean over folds of the paired difference. Also reported: vs B_pb, and the conservative "best on held-out" (A2.4) | `d2` uses the **frozen** `nested_comparator` | `test_d2_comparator_is_chosen_on_selection_folds_…` |
| D2 stop < 0.02 (plan 7, A2.4) | `d2_stop` (OP2 below) | `test_d2_stop_boundary_…` |
| D2 validation as an exposed second estimate (plan 7) | `validation_selection` takes no validation argument: models, (τ_b, m_b) and comparator are all chosen on development data. `d2_validation_exposed` then evaluates once on Phase 1 validation. Reported only | `test_validation_estimate_selection_never_reads_validation` |
| **D3** WC vs CW, adjacent and to-final, h_t vs x, within benchmark and loop, out of fold; keep hx iff h > x in ≥ half of the cells with ≥ 30 transitions (three primary benchmarks) (plan 7, A1.7) | `d3`, `keep_hx`. Decision method `indirect`: Gate 2's estimand P(WC) − P(CW), with Gate 2's recorded C (checked against `validation_report.json`). `direct` is reported as information. Every cell's status is reported (OP5, OP6) | `test_d3_keep_rule_boundary`, `test_d3_reports_both_methods_and_cell_status` |
| **D4** predictability of c_t, c_4 and Δ_t, informative only (plan 7) | `d4` (OP7) | dry run |
| Determinism; no overwrite; frozen inputs pinned; provenance | Fixed seeds, as in Gate 2. The script refuses to overwrite and checks plan v1.1, A1, the readout module, the DEV+ check report (passed) and the Gate 2 frozen C. The report records code hashes, the git commit and every data-file sha256 | `test_end_to_end_synthetic_dry_run` (repeat run identical); script `--synthetic` refusal checked |

## Open points: reviewer resolutions (G3a implementation review, 2026-10-26), locked before any real run
| point | resolution |
|---|---|
| OP1 gross vs net | **Accepted provisionally.** G3a uses gross saving (ρ = 0). D2 is printed and reported as a "gross-saving screen", not as evidence of net improvement |
| OP2 two D2 candidates | **Accepted provisionally, locked.** STOP iff both π1-x and π1-hx have improvement < 0.02, so either may pass |
| OP3 grid pool | **Accepted.** It matches the frozen `phase3_readouts.grid`, which pools over the supplied ok rows and the defined loops |
| OP4 D1 baseline | **Accepted** as descriptive and non-binding |
| OP5 D3 model | **Accepted (rev 2 review), with a disclosed qualification:** the to-final probes reuse the adjacent-transition C. That is an extrapolation fixed in advance, not a value Gate 2 validated for to-final prediction. Original proposal: The reviewer noted that the direct classifier changes the estimand. Now proposed: the decision uses the **indirect** Gate 2 estimand, P(WC) − P(CW) from two per-loop probes fitted on all ok rows with Gate 2's recorded per-loop C. To-final probes reuse the same C. The direct classifier is reported beside it as information |
| OP6 D3 cells | **Accepted provisionally.** Every cell's status (counted, too_few, empty or unscorable) and the counts over the primary cells are reported |
| OP7 D4 regularization | **Accepted** (informative only) |
| OP8 failed questions | **Accepted** |

## Revision 2: the reviewer's required fixes
1. **D2 outer-fold isolation (critical).** Rev 1 cross-fitted the candidate statistics once over all five folds. A
   selection-fold statistic could therefore come from a model trained on the outer held-out fold. Now `outer_fold(k)` does
   three things:
   - it inner-cross-fits the candidates over the four selection folds only;
   - it fits the outer models on all four selection folds, and only to score fold k;
   - it selects every method's (τ_b, m_b), grid and comparator on selection rows only.
   This is the strict reading of the frozen wording ("chosen on four folds' out-of-fold statistics"). Rev 1 followed the
   loose reading.
2. **Contamination test.** It replaces every label, feature, hidden state and option score in held-out fold k with random
   values. Every selected (τ_b, m_b), selection saving, eligibility flag, comparator, outer-model sha256 and
   selection-statistic sha256 must stay identical. It is checked for k = 0 and k = 3, and also checks that the held-out
   evaluation itself does change. A second test shows that the rev 1 whole-data cross-fit would have failed it.
3. **Paired comparisons.** `d2` asserts that every method is evaluated on identical held-out rows in every fold, and a test
   checks this.
4. **OP1-OP8** are documented above and in the code (`OPEN_POINTS`, written into the report).
5. **Validation estimate.** It is checked separately for leakage: selection is a function that does not receive validation
   data, and a test perturbs validation and finds every selection unchanged.

## Synthetic validation (no real outcome evaluated)
- 53 G3a tests pass, plus the 13 frozen readout tests.
- `python scripts/phase3_g3a.py --synthetic OUT.json` runs the whole script path on synthetic data and writes a report of
  the real format. The numbers it prints are synthetic and mean nothing.
- The frozen-input check passes against the committed files.

## Expected cost on the laptop
CPU only:
- The readout statistics: 7 × ~26k questions.
- D2, fully nested: per outer fold, 4 inner fits plus 1 outer fit, × 3 loops × 2 variants, so about 75 hx fits.
- D3 indirect: 2 kinds × 2 targets × 2 variants × 5 folds × 3 loops (60 h fits); direct: small subsets with inner CV.
- D4: 30 hx fits at C = 0.001.

Estimated 40-120 minutes, and about 1-2 GB of RAM.

## Revision 3: the reviewer's Revision 2 assessment (conditional approval)
1. **Exposed-validation comparator, made consistent with D2.** The frozen A2.4 comparator is "eligible (qualified under
   the per-benchmark rule on the selection data), then highest selection macro saving, ties by name".
   - `validation_selection` now computes eligibility on the development data and chooses the comparator with the same
     frozen `nested_comparator`. The comparator choice never reads a validation value.
   - Because τ = +∞ nests fixed depth 4 (A2 freeze note), every baseline is eligible at selection in practice. So the result
     equals rev 2's, but the rule is now literally the same.
   - Test: `test_validation_comparator_uses_the_same_eligibility_rule_as_d2`.
2. **Pairing by question identity.** Each held-out evaluation records the sha256 of its sorted question ids, per
   benchmark, and `d2` asserts that the fingerprints are identical across all methods in every fold.
   - Tests: the fingerprints equal fold k's actual questions, and the assertion fires when one method is evaluated on a
     different set of the same size.
3. **OP5 accepted**, with the extrapolation disclosed in the code and here.
4. **Tests:** 53 G3a tests plus 13 readout tests pass.
