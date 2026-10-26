# Phase 3 G3a (D1-D4): implementation and frozen-spec crosswalk (for reviewer sign-off; NOT RUN on real data)

Authorized by the reviewer's "Phase 3 extraction — Independent reviewer assessment" (2026-10-26). That authorization covers
writing D1-D4 and testing them on synthetic fixtures only. Running on real development data needs a separate sign-off of
this implementation. Training, model selection and TEST stay on HOLD.

Files: `src/lookahead/phase3_g3a.py` (logic and loader), `scripts/phase3_g3a.py` (guarded entry point),
`tests/test_phase3_g3a.py` (45 tests), `scripts/run_phase3_g3a_windows.ps1` (laptop runner; to be used only after
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
| **D2** candidates π1-x and π1-hx: family-1 per-loop L2 logistic on the stability target S_t, with Gate 2's **frozen** C (x 1.0, hx 0.001; checked against `results/gate2/frozen_selection.json`). 5-fold cross-fitted, then (τ_b, m_b) chosen on four folds' out-of-fold statistics and evaluated on the fifth, rotated (plan 6, plan 7) | `candidate_stats_crossfit`, `crossfit`, `d2`. Fitting uses all six benchmarks; selection is per benchmark (A1.2) | dry run |
| D2 comparator: baselines B_pb, native-pb and the seven readouts, each selected on the selection folds. Eligible if qualified there; strongest by selection-fold macro saving, ties by name; only its held-out saving is used; improvement = mean over folds of the paired difference. Also reported: vs B_pb, and the conservative "best on held-out" (A2.4) | `d2` uses the **frozen** `nested_comparator` | `test_d2_comparator_is_chosen_on_selection_folds_…` |
| D2 stop < 0.02 (plan 7, A2.4) | `d2_stop` (OP2 below) | `test_d2_stop_boundary_…` |
| D2 validation as an exposed second estimate (plan 7) | `d2_validation_exposed`: everything selected and fitted on all development data, evaluated on Phase 1 validation. Reported only | dry run |
| **D3** WC vs CW, adjacent and to-final, h_t vs x, within benchmark and loop, out of fold; keep hx iff h > x in ≥ half of the cells with ≥ 30 transitions (three primary benchmarks) (plan 7, A1.7) | `d3`, `keep_hx` (OP5, OP6) | `test_d3_keep_rule_boundary` |
| **D4** predictability of c_t, c_4 and Δ_t, informative only (plan 7) | `d4` (OP7) | dry run |
| Determinism; no overwrite; frozen inputs pinned; provenance | Fixed seeds, as in Gate 2. The script refuses to overwrite and checks plan v1.1, A1, the readout module, the DEV+ check report (passed) and the Gate 2 frozen C. The report records code hashes, the git commit and every data-file sha256 | `test_end_to_end_synthetic_dry_run` (repeat run identical); script `--synthetic` refusal checked |

## Open points: implementation choices the frozen documents do not settle (OPEN_POINTS in the code)
Each is stated here and written into the report. **The reviewer should confirm or change each one before any real run.** None
can change after it.
1. **OP1, net vs gross saving at G3a.** A2.4 says "macro net saving", but ρ_b is measured only at G3b (plan 5a). So G3a uses
   gross saving (ρ = 0) for every policy and baseline.
2. **OP2, D2 with two candidates.** Stop iff **both** π1-x and π1-hx improve by less than 0.02 on the A2 comparator. Phase 3
   continues if either reaches 0.02. Taking the better of two is mildly optimistic; the alternative would be to pre-name one
   candidate.
3. **OP3, grid pool.** The quantiles pool the statistic over the ok questions of all six benchmarks in the selection rows.
   In D2 that means the selection folds only.
4. **OP4, D1's non-binding baseline.** It is chosen on all development data, because D1 has no held-out fold.
5. **OP5, D3 model.** A direct WC (1) vs CW (0) classifier on each loop's changing transitions.
   - Per-loop L2 logistic on h_t or on x_t.
   - C is chosen by inner CV over the Gate 2 C grid on the training folds.
   - Out of fold over `lh-fold-v1`, fitted on all six benchmarks; AUROC within (benchmark, loop).
   - This replaces Gate 2's indirect "P(WC) − P(CW)" from two separate probes.
6. **OP6, D3 cells.** A cell is (kind, benchmark, loop), with kind ∈ {adjacent, to-final}, over the three primary
   benchmarks.
   - A cell counts if it has ≥ 30 WC+CW transitions and both classes.
   - With no counted cell, the hx variants are **not** kept.
7. **OP7, D4 model.** It uses the frozen Gate 2 C (x 1.0, hx 0.001) instead of inner CV, to bound the compute. Informative
   only.
8. **OP8, failed questions in the oracles.** They exit at loop 4 and count wrong, as under every policy. None are expected:
   every DEV+ and Phase 1 question has status ok.

## Synthetic validation (no real outcome evaluated)
- 45 tests pass.
- `python scripts/phase3_g3a.py --synthetic OUT.json` runs the whole script path on synthetic data and writes a report of
  the real format. The numbers it prints are synthetic and mean nothing.
- The frozen-input check passes against the committed files.

## Expected cost on the laptop
CPU only:
- The readout statistics: 7 × ~26k questions.
- D2 cross-fitting: 5 folds × 3 loops × 2 variants.
- D3: small subsets with inner CV.
- D4: 30 hx fits at C = 0.001.

Estimated 20-60 minutes, and about 1 GB of RAM for the hidden states.
