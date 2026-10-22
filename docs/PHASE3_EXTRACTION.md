# Phase 3 extraction workflow (A1, tag phase3-a1): implementation for review

Authorized by the reviewer's Final Allocation Review: implement and test the DEV+ extraction and the sealed TEST extraction,
with TEST outcomes inaccessible to development and policy selection. D1-D4 and training stay on HOLD until the reviewer has
verified these safeguards. Nothing here has been run on real data.

## Steps
1. **Build** (CPU, `scripts/prepare_phase3.py`, module `src/lookahead/phase3_build.py`).
   - It reads only the frozen A1 allocation. The hashes of `allocation.json` and the TEST and DEV+ CSVs are pinned to the A1
     values, so any other file is refused. It requires each dataset revision to equal the inventory's.
   - Questions are built exactly as in Phase 1 and the Gate 2 confirmation set: the lm_eval prompt format, Phase 1 shot counts
     and the few-shot seed 1234.
   - Every built doc must reproduce its allocation text hash. Doc ids take the form `p3:<task>/<split>@r<row>`.
   - It writes `data/questions_phase3_dev/`, `data/questions_phase3_test/` (gitignored), `data/manifests/phase3_questions.json`,
     `phase3_ids_dev.csv` and `phase3_ids_test.csv`. It always writes `results/phase3/build_report.json`, and if any check fails it
     writes only that report.
2. **Extract** (GPU) with the existing extractor: `extract_phase1.py --set phase3_dev | phase3_test` writes to
   `data/extracted_phase3_dev/` and `data/extracted_phase3_test/`. Before the model is loaded, the frozen-input check
   `check_phase3_inputs` verifies three things: the manifest was built from the pinned A1 allocation, the per-role id-list
   hash, and every question-set hash.
3. **Verify** with the existing verifier: `verify_extraction.py --set phase3_dev | phase3_test` writes
   `results/phase3/extraction_check_phase3_<role>.json`. It prints only PASS/FAIL, question counts and record-status counts,
   exactly as for the Gate 2 confirmation set.

## Safeguards
- **Sealed paths.** Every TEST path contains `phase3_test`. `assert_not_sealed()` refuses any path containing `phase3_test`
  (or the Gate 2 `confirm`), and every Phase 3 development or selection loader (G3a, G3b) must call it before opening a file.
  The tests check that it refuses `phase3_test` and `confirm` paths, including Windows paths and upper case, and that it
  allows the DEV+ and Phase 1 paths.
- **Separate id lists per role.** TEST ids are never in the file the DEV+ extraction or development code reads.
- **No TEST or DEV+ row is ever an exemplar.** lm_eval draws HellaSwag and CommonsenseQA few-shot examples from the train
  split, which is where the allocated rows live. So the build strips **every allocated row** (TEST and DEV+) from each task's
  few-shot pool before sampling, and exemplars come only from rows the pipeline had already used.
  - The recorder counts exemplars that are allocated rows, self-exemplars and exemplars missing from their context. Any
    non-zero count fails the build.
  - A test also checks the built prompts independently: no allocated question's text appears in any other question's
    context.
- **Nothing reads outcomes.** The build and the input checks never read a model output. The extraction computes outcomes for
  TEST, as it did for the Gate 2 confirmation set, but nothing in this workflow reports or analyses them.

## Decisions taken in the implementation (for the reviewer to confirm)
1. **Few-shot pool stripping** (above). Phase 1 drew exemplars from the full train split, so exemplar *identities* differ
   from Phase 1 for HellaSwag and CommonsenseQA. The prompt *format* and shot counts are unchanged. OpenBookQA is 0-shot and
   is unaffected.
2. **OOD metrics.** PIQA uses `acc_norm` (the headline metric of lm_eval's PIQA task). Social IQa uses `acc` (the only metric
   lm_eval's Social IQa task defines). A1.4 fixed 0-shot but did not name the metric. The metric decides which option counts
   as the prediction, so it is fixed now, before any outcome exists.
3. **Two stages.** `run_phase3_extraction_windows.ps1 -Stage build` runs the tests, the build, and 40-question smoke
   extractions with checks for both roles. `-Stage full` (about 80 GPU-minutes) runs only after the reviewer has seen the
   build report.

## Tests
- `tests/test_phase3_build.py`: 16 tests.
  - The guard refuses sealed paths and allows development paths.
  - The pinned hashes equal A1, and the frozen allocation reads back with the A1 counts.
  - End-to-end on local fake tasks: pool stripping (20 allocated rows removed, 0 violations) and the independent context
    check; per-role id lists; the extraction-side check accepts each role and refuses a manifest from another allocation.
  - The build is byte-deterministic, and refuses to overwrite.
  - A tampered question file is refused, a revision change writes the report only, a wrong allocation hash or a text-hash
    mismatch is refused, and the extractor routes both Phase 3 sets through the pinned check.
- `tests/test_phase1_scripts.py` and `tests/test_extract.py` still pass with the extractor and verifier changes.
