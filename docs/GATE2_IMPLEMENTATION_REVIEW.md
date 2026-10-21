# Gate 2 implementation review package

For the reviewer's independent implementation review (items named in the control-audit decision: the 0.853 threshold,
training-split isolation, model selection, macro-accuracy calculation, confirmation guard). Nothing here opens the
confirmation set. Frozen selection (commit e65eba1) is unchanged.

## 1. Files to review
| file | lines | sha256 |
|---|---|---|
| `src/lookahead/gate2.py` (probe data, fitting, policies, freeze) | 815 | `5df76cf15b8d62bf02aeadc06c94ef43d4c8d075ea0447eb1aa7f32f20df5e35` |
| `scripts/gate2_select.py` (validation-only selection driver) | 204 | `66a1d5fd7bd4a22ecec1455a3ad6b226b89078f2b4fc62d6f2a287eee8f48ce2` |
| `tests/test_gate2.py` (28 synthetic tests) | 550 | `77c39a0d18094c92a08a199187fd2a2c20e1037f341f4b56f132a2e84e4c7cca` |
| `results/gate2/frozen_selection.json` | | `aa4f49e6a686a44c3136a30b36ce4516468e9d91913f2ddbc7aace31160eed96` |
| `results/gate2/validation_report.json` | | `adc3baf0db6e19eb17cd0aece02001eaf672ef6f6c22c7e91301cdab28222e21` |
| `results/gate2/baseline_B_frozen.json` (amendment v3.1) | | `19fcf29583dc78a582d306fbafb82a0128ac6fb557cd36c8f1b14e1bb762eaaf` |
| `src/lookahead/gate1.py` (grid, selection rule, policies, macro metrics; frozen since Gate 1) | | see repo |
Added after the freeze, for the confirmation step (review these too): `src/lookahead/gate2_eval.py`, `scripts/gate2_confirm.py`,
`tests/test_gate2_eval.py`, `scripts/gate2_control_audit.py`, `scripts/gate2_freeze_baseline.py`.

## 2. Provenance fact checked
The frozen file reports `git_dirty_code: true` (the probe code was uncommitted when Sunny ran it) and
`code_sha256 6ccaccf41d8b...`. That hash is `gate2.code_identity()` over every `src/lookahead/*.py` plus
`scripts/gate2_select.py`. Recomputing it from the tree of commit e65eba1 (`git archive e65eba1`) gives the same
`6ccaccf41d8b24c73bbd98a10dad34ece35e10732f6d5462dd67cca2ef07fc95`: the committed code is byte-identical to the code that ran.
The remaining deviation is procedural (it ran before sign-off), recorded in `docs/DECISIONS.md`.

## 3. The five items
**(a) The 0.853 threshold** (P: 0.8529863981323346).
`gate2.select_operating_point` (gate2.py:592) builds the pooled validation score vector over ok questions and loops 1-3,
takes `gate1.grid_thresholds` (gate1.py:298: 50 quantiles, exact repeats removed), summarises each grid point with
`gate1.summarize(..., "native_gate", t)` and applies `gate1.select_point` (gate1.py:308): qualified = macro loss ≤ 1.0 + 1e-12 pp,
then max macro saving, ties to smaller loss, then the largest threshold. The threshold is a grid point of validation scores.
Independent reproduction: `scripts/gate2_control_audit.py` (section B) reloads the saved P and Q model files, rescales and
re-selects, and returns exactly the frozen threshold, saving and loss for both (`results/gate2/control_audit.json`).
Test: `test_operating_point_is_gate1s_rule`, `test_frozen_probe_reproduces_its_validation_point`.

**(b) Training split isolation.**
`PROBE_SPLITS = ("train","validation")` (gate2.py:32); `load_probe_data` raises for any other split (including `test` and
`confirmation`); parquet rows are filtered by the `split` column when read; `fit_probe` raises unless `ps.split == "train"`
(gate2.py:526); `_assert_allowed_path` (gate2.py:74) refuses any path containing "confirm"; `gate2_select.assert_inputs_allowed`
(gate2_select.py:26) repeats that for the driver. `g_clip` is computed on train and applied to validation. Hyperparameters come
from 5-fold CV inside train (folds by sha256 of the doc id) and an early-stopping slice of train; validation is used only for the
threshold and the configuration choice. Failed questions are excluded from fitting. The frozen file contains no confirmation paths.
Tests: `test_rejects_other_splits`, `test_loaded_data_is_split_pure_and_hidden_aligned`, `test_g_clip_from_train_applied_to_validation`,
`test_failed_questions_are_not_used_for_fitting`, `test_mlp_weight_decay_tie_goes_to_larger_and_es_slice_is_train_only`,
`test_dry_run_end_to_end_reads_only_train_and_validation`, `test_forbidden_paths_and_benchmarks_are_rejected`.

**(c) Model selection.**
`gate2.choose_config` (gate2.py:611): among configurations qualified on validation, largest saving; ties (1e-12) to smaller
loss, then simpler (family 1<2<3, h<hx). All six configurations (families 1-3 × {h, hx}) are in `validation_report.json`;
family 1 / hx wins at 0.388 (families 2 and 3: 0.378-0.380; family 1 / h: 0.353). Q is family 1 on the cheap features only.
Test: `test_config_choice_tie_breaks`.

**(d) Macro accuracy and saving.**
`gate1.run_policy` (gate1.py:201) gives each question's exit loop and correctness (failed questions run 4 loops and count
wrong); `gate1.summarize` (gate1.py:254): per benchmark acc = 100·mean(correct at exit), saving = 1 − mean(exit)/4,
loss = acc(loop 4) − acc; macro = unweighted mean over the six benchmarks. A second, independently written computation
(`gate2_eval.outcomes` / `point_estimates`, which builds the correctness and exit matrices itself and takes the macro
over a [policy × benchmark] array) reproduces every frozen validation number to 1e-9 in the rehearsal (section 5).

**(e) Confirmation guard.**
Selection code can never open confirmation data (b above). The only relaxation is `gate2_eval.allow_confirmation_paths()`,
a context manager used by `scripts/gate2_confirm.py` after all pre-flight checks. The script requires
`--confirm-one-time-gate2-evaluation`, the sha256 prefixes (≥ 12 hex) of the two frozen files, and a reviewer-approval
reference; refuses if `results/gate2/CONFIRMATION_STARTED.json` or `CONFIRMATION_EVALUATED.json` exists (no override flag exists); refuses
with any modified tracked file (no override flag exists); requires the shuffled control; checks the B vector equals
(2,2,2,3,3,2) and that `baseline_B_frozen.json` was made against this `frozen_selection.json`; verifies the model files
against their frozen sha256; writes STARTED before any confirmation outcome is opened and EVALUATED last. It computes
no threshold, depth or probe weight: all come from the frozen files.
Tests (`tests/test_gate2_eval.py`): refusal without a mode or with both, wrong/missing sha, missing approval reference,
existing markers, a changed B vector, a baseline file made against another frozen file, and that the path guard is open only
inside the context manager. The sha-refusal test also checks that no STARTED marker is written.

## 4. Test evidence
Full suite on the Linux workspace: **140 passed** (`pytest -q`, 6 min 29 s), including 28 in `tests/test_gate2.py`
and 19 in `tests/test_gate2_eval.py`. All Gate 2 tests use synthetic data only; none reads real or confirmation data.

## 5. Rehearsal on validation (no confirmation data)
`python scripts/gate2_confirm.py --rehearsal-on-validation` runs the confirmation code path on validation and fails (exit 3)
unless it reproduces: P 0.3880/0.9441, Q 0.4029/0.5331, the frozen baselines, B 0.4167/0.7775, the B_cap vector, and
`gate2.load_probe_data` array-for-array (hidden states, cheap features, targets, ok flags, doc ids).
Result: the full rehearsal on the laptop (shuffled control included, at the committed code 53fd6e1) printed
"every frozen validation number (P, Q, baselines, B, B_cap) and the loader are reproduced". The shuffled control is not part
of the script's automatic check; it was compared by hand: rehearsal 0.406015202046128 / 0.38953268172622896 against
`validation_report.json` 0.406015202046128 / 0.38953268172622896 (identical). Output: `results/gate2_rehearsal/rehearsal_report.json`,
`docs/GATE2_REHEARSAL.md`. The rehearsal's "verdict" line is computed on validation with frozen thresholds and is NOT a Gate 2
result (on validation: L true, E false, H false). Paired validation deltas for orientation only: P-B -0.029 [-0.036, -0.021],
Q-B -0.014 [-0.021, -0.007], control-B -0.011 [-0.014, -0.007].

## 6. Known limitations to weigh
- Procedural: training and selection ran before this review (recorded deviation; code identical to the committed e65eba1).
- Tests are synthetic; the real-data checks are the reproduction of frozen numbers from saved models, the audit, and the rehearsal.
- The shuffled control's model is not stored; the confirm script refits it on train (deterministic, reproduced the validation
  report exactly) and uses the validation threshold 0.8222508742366401.
- `confirm` must not appear in any path read by selection code, but this is string-based; the sealed directories are also
  gitignored and outside the selection loaders' directory.
- Not implemented in the confirmation script (retrospective analyses of plan 7a, to be added after the verdict is recorded):
  accuracy-budget frontier and leave-one-benchmark-out probes.

## 7. Changes after the reviewer's third note (conditional acceptance)
- `--allow-rerun` and `--allow-dirty` are **removed**; argparse rejects them (tests `test_override_flags_do_not_exist`). A repeat
  run or a run on a modified tree is impossible without a new commit and a new reviewer decision.
- The real run requires the shuffled control (`--skip-control` is refused in the real run).
- **Pre-flight before the confirmation set is opened** (`validation_preflight`): reloads train and validation, refits the
  shuffled control deterministically on **train only**, and requires P, Q and the control to reproduce their frozen/reported
  validation saving and loss (1e-9) at the frozen thresholds (P 0.8529863981323346, Q 0.7907418731925548, control
  0.8222508742366401). Exit code 3 and no STARTED marker if any differs. The rehearsal runs the same pre-flight.
- Source bundle for source-level review: `gate2_source_review.zip` (the four requested files, the two frozen JSON files, gate1.py
  and the tests, with a sha256 manifest).
