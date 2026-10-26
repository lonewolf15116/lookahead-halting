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

## Reviewer decision (Stage build GO, 2026-10-23) and how its two conditions are enforced
Approved: few-shot pool stripping, PIQA `acc_norm`, Social IQa `acc`, two stages, and a 40-question smoke extraction on the
condition that it does not expose TEST correctness or predictions. On HOLD: full extraction, D1-D4 and training.

1. **The smoke test stays sealed.**
   - TEST smoke records are written only to `data/extracted_phase3_test_smoke/`. That is a sealed path, which
     `assert_not_sealed` refuses.
   - The full extraction writes to `data/extracted_phase3_test/` and recomputes every question. Smoke records are never
     merged into it or reused, so they cannot influence development.
   - The verifier prints only PASS/FAIL, question counts and record-status counts. Its report holds integrity checks only:
     schema, finiteness, hashes, resume state, and the recomputation of the stored prediction fields. It holds no accuracy.
2. **Exemplars come only from permitted, previously used rows; this was tightened after the review.**
   - The few-shot pool is now restricted to rows whose text is in the verified used-set. The used-set is rebuilt and checked
     against the manifests, exactly as in the inventory. Allocated rows are also removed.
   - On real data this removes, for example, the 498 HellaSwag train rows that were never used. Those rows are neither
     allocated nor used: they were excluded from the clean pool only for sharing a source or context.
   - The build report gives, per set, the following, and any non-zero violation count fails the build:
     - `exemplar_is_allocated_row` (must be 0) and `exemplar_not_previously_used` (must be 0);
     - `pool_removed_allocated` and `pool_removed_not_used`;
     - `exemplar_selection_sha256`, a fingerprint of the exact exemplars of every question, so a rebuild can confirm
       determinism.
   - Tests: the permitted pool removes 20 allocated rows and 5 never-used rows, with 0 violations. The independent
     prompt check also confirms that no never-used row's text appears in any context.

## Stage-build review (2026-10-26): numerical-equivalence clarification
The reviewer accepted the allocation, the manifest, the exemplar exclusions and the TEST report sealing, and put the full
extraction on HOLD pending three questions. The answers below come from the verifier code (`scripts/verify_extraction.py`),
the Phase 1 diagnosis (`scripts/diagnose_short_context.py`, `results/phase1/diagnose_short_context.json`) and the
integrity fields of the two smoke reports. I read no outcome, prediction or correctness value to write them.

### 1. What `vs_context_only_reported` measures, and its thresholds
- It re-runs a sampled question as a **context-only** pass, without the option tokens, and compares the state, gate, cdf
  and vocabulary statistics at the last context token with the stored values. The stored values come from the **packed**
  pass, which runs the context and every option together under a custom mask.
- **It has no acceptance threshold. That is deliberate:** it is reported only. On 2026-10-20 the reviewer accepted this as
  a labelled post-diagnostic amendment (DECISIONS, 2026-10-20). The reason: in bf16 there is no canonical value to compare
  against, because both passes are bf16 approximations of the same float32 computation.
- **Two checks are gated:**
  - `rerun_same_input`: the same packed input is run again. Gate, cdf and the vocabulary statistics must agree within 1e-4,
    and the state within 1e-3. In both smoke reports the gate, cdf and state agree exactly (0) and the vocabulary
    statistics within 1.1e-6. The extraction is deterministic.
  - `float32_equivalence_recorded`: in float32 the packed and context-only passes must agree within 1e-4. The recorded
    maximum is 2.1e-5. This is what shows that the mask and positions are correct.

### 2. Why OpenBookQA, PIQA and Social IQa differ while HellaSwag and CommonsenseQA do not
- **Cause: prompt length.** These three benchmarks are 0-shot with very short prompts. OpenBookQA prompts are 3-66 tokens,
  and PIQA and Social IQa prompts are similarly short, a single goal or a single context and question. HellaSwag and
  CommonsenseQA are 10-shot, with prompts of about 800 tokens.
- **Why that matters.** The packed pass appends the option tokens, which changes the sequence length the GPU matmul kernels
  see. At short lengths that changes which bf16 kernel is chosen. At long lengths the relative change is small and the same
  kernels are used.
- **The Phase 1 diagnosis showed this pattern:**
  - In bf16, packed and context-only passes were exactly equal on modarith (85 tokens), ARC-E and CSQA. They differed only
    on OpenBookQA: maximum 0.375, relative L2 up to 0.078, and exactly equal on 64% of questions.
  - In float32 they agreed to 2.1e-5.
  - A context-only pass through the packed pass's own mask code path was identical to the plain context-only pass, so the
    mask path is not the cause.
- **The smoke values sit inside that envelope.** The state maximum is at most 0.25 and the relative L2 at most 0.041.
  The largest state coordinates are about 29, where the bf16 spacing is 0.125, so 0.25 is about two bf16 rounding steps.
  The gate-logit differences are at most 0.44, giving cdf differences of at most 0.024.
- **Small vocabulary-statistic differences on CSQA and HellaSwag**, with the state and gate exactly equal (vent at most
  0.005), come from the same mechanism in the output-head matmul, whose row count also differs between the passes.
- **Gap, stated plainly.** The float32 check is read from the Phase 1 diagnostic file, which covers OpenBookQA, modarith,
  ARC-E and CSQA. **It does not include PIQA or Social IQa**, so for those two benchmarks the kernel explanation is inferred
  from prompt length and not yet tested. Proposal D-a below closes this.

### 3. The `opt_ll` differences: can they change rankings or the A2 readouts, and is the tolerance adequate?
- **The field name is misleading.** `rerun_same_input.opt_ll` is not a same-input re-run. It compares the stored packed
  option log-likelihoods with `score_per_option`, a separate forward pass per option with different shapes. Its gate,
  `option_ll_vs_per_option_le_1nat`, was set in Phase 1 as a **gross-bug detector**: a wrong mask or position would
  produce errors of many nats.
- **Recorded sizes:**
  - Pilot P5: maximum 0.45 nats, p99 0.15-0.22.
  - Full Phase 1: maximum 0.70 nats (HellaSwag).
  - These smoke runs: maximum 0.92 nats, also HellaSwag (DEV).
- **HellaSwag is the largest because** its continuations are long and per-token bf16 noise adds up. HellaSwag uses
  `acc_norm`, which divides by the continuation's character length, so the effect on the ranking score is much smaller than
  the raw sum suggests.
- **Can it change rankings? Yes, for near-tied options.** The 1-nat gate does not bound this, so it is **not** a fidelity
  bound. The only measured agreement is on ARC-C (pilot P6/P8): same top answer on 99.57-99.91% of questions per loop, and
  flips agreeing on 99.49-99.74%. Agreement has not been measured on the Phase 3 benchmarks.
- **Effect inside Phase 3: none on internal validity.**
  - All seven A2 readouts, B_pb, native-pb and the π candidates are computed from the same stored values, so no comparison
    is advantaged.
  - The verdict is defined on the stored packed computation, the same one used in Phase 1 and Gate 2.
  - The stored values are deterministic, as the same-input re-run shows.
- **Effect on external validity, a limitation to report.**
  - A deployment that scores options per option, or computes the state from a context-only pass, could halt differently on
    near-ties.
  - For r-hl2, r-hcos and the hx probes on the short-prompt benchmarks, path noise of up to 4% relative L2 may be comparable
    to the change between loops at late loops. Neither quantity has been measured.
- **Recommendation: change no tolerance.** That would be a change made after seeing results, and the gate does its job of
  catching bugs. Instead, measure decision-level agreement as a reported diagnostic.
- **An unmet commitment.** On 2026-10-20 the reviewer said Phase 3 would include a sensitivity analysis of halting
  decisions near the thresholds, and that commitment is not in PHASE3_PLAN. Proposal D-b fulfils it.

### Proposals (need reviewer approval; neither changes A1, A2, any tolerance or any threshold)
- **D-a, before the full extraction (about 5 GPU-minutes):** extend `diagnose_short_context.py` to PIQA and Social IQa.
  - It uses rows from their **train** splits. These are in neither the TEST nor the DEV+ allocation, so no sealed question
    is touched. The prompt format and 0-shot setting are the same.
  - It runs the same bf16 and float32 packed-vs-context-only comparison.
  - Pass condition, fixed now: float32 maximum ≤ 1e-4, the existing FP32_TOL.
- **D-b, after the full extraction and before D1-D4, DEV+ only, reported and not gated:**
  - For every DEV+ question, compute the per-option scores and a context-only state.
  - Report the share of (question, loop) pairs where the top answer differs.
  - For each A2 readout and each B_pb and native-pb setting, report the share of halt decisions on the threshold grid that
    differ between the stored path and the alternative path.
  - TEST is not used.

## Reviewer decision (Numerical Fidelity Review, 2026-10-26) and its implementation
CONDITIONAL GO. The verifier tolerances stay as they are, with their limits documented above. D-a: GO. D-b: GO for writing
the protocol only, not for running it. Full extraction stays on HOLD until both D-a float32 checks pass and their provenance
has been checked. D1-D4 and training stay on HOLD until the full extraction has been verified. The reviewer's qualifications,
recorded as given:
- A float32 match does not show that bf16 differences are harmless for halting decisions.
- The stored packed scores make the comparison fair within this scoring implementation. They do not establish numerical
  invariance across implementations.
- A 0.92-nat difference is not intrinsically minor: it can decide a close comparison between options.

### D-a (implemented: `scripts/diagnose_short_context_ood.py`, runner `scripts/run_phase3_da_windows.ps1`)
- **Rows.** 100 per benchmark, from the PIQA and Social IQa **train** splits. They are taken from the frozen inventory's
  clean list, whose hash is checked against `inventory.json` (pinned 7ec501ae…). Every row on that list has never been used:
  it was not a Phase 1 or Gate 2 evaluation row, not an exemplar, and not a duplicate.
  - Every A1 allocation row, DEV+ and TEST, is removed by text hash. The A1 OOD rows are validation rows in any case.
  - Rows are chosen by sha256 rank with the salt `lh-phase3-da-v1`. The selection is fingerprinted, and the report lists
    the row ids.
  - Eligible rows: PIQA 16,107, Social IQa 33,381. Selection fingerprints: PIQA a27f3f74d95e810b…, Social IQa
    ab6d70b28b9ce639…
  - If fewer than 100 rows are eligible, the script stops rather than changing the sampling rule.
- **Build.** Exactly the Phase 3 build: lm_eval format, 0-shot, the frozen OOD metric, and the text hash checked against
  the inventory. Zero exemplar draws are required.
  - The dataset revisions must equal the inventory's.
- **Comparison.** The existing `diagnose_short_context.py` comparison: the state at the last context token, packed pass
  against context-only pass (plain and explicit-mask), all 4 loops. It runs in bf16 (reported) and float32 (gated).
- **Pass rule.** A separate verdict per benchmark. It passes iff the float32 maximum |diff| is ≤ 1e-4 on both packed
  comparisons. This is FP32_TOL from the verifier, unchanged, and a test asserts the two are equal.
- **Report.** `results/phase3/diagnose_short_context_ood.json`. The script refuses to overwrite it. It records the row ids,
  selection fingerprints, sample counts, question-set hashes, the model and its revision, the dtypes, the device, the torch
  and lm_eval versions, and the code identity (code sha256, git commit, script hashes).
- **What it does not compute.** No option score, prediction, label or correctness value is computed or written.
- **Tests.** `tests/test_phase3_da.py`, 8 tests:
  - Rows are unused train rows and outside every allocation.
  - The selection is deterministic, of size 100, and fingerprinted.
  - Too few rows stop the script.
  - The pass rule uses the unchanged tolerance.
  - The frozen inputs are hash-checked.
  - The report path is not sealed, and the script names no outcome field.

### D-b (protocol only; not run)
- **Purpose.** A non-binding sensitivity analysis on development data only. It fulfils the 2026-10-20 commitment.
- **Sequence (reviewer):**
  1. Full extraction and verification.
  2. D1-D4 under the frozen protocol, unchanged.
  3. Only if D1-D4 allow Phase 3 to continue, D-b on DEV+ and Phase 1 development data.
- **What it reports.** For each evaluated controller (B_pb, native-pb, the seven A2 readouts and the π candidates), with
  attention to the selected thresholds:
  - the share of questions whose top option changes, and the share whose halt decision changes, when the stored packed
    values are replaced by per-option scores and context-only states.
- **What it never touches.** TEST. It changes no decision rule, threshold or verdict. Its results are reported as a
  limitation on numerical invariance across implementations.

### D-a result (2026-10-26): Social IQa PASS, PIQA FAIL. Full extraction stays on HOLD
Run at commit dd000ae (code clean), torch 2.11.0+cu128, RTX 5050, Ouro 574fa66c, with the dataset revisions equal to the
inventory's. Report: `results/phase3/diagnose_short_context_ood.json`, kept as written. All 100 rows were built per
benchmark, with 0 exemplars.

| | context tokens | float32 packed vs ctx: max (median), rel L2, exact | verdict at 1e-4 | bf16 max, rel L2 | explicit-mask ctx vs ctx |
|---|---|---|---|---|---|
| PIQA | 6-25 | **1.17e-4** (6.4e-6), 1.7e-5, 7% | **FAIL** | 1.0, 0.13 | 0 (both dtypes) |
| Social IQa | 22-46 | 3.1e-5 (0), 7.0e-6, 87% | PASS | 1.31, 0.18 | 0 (both dtypes) |
| *Phase 1 OBQA (ref.)* | 3-66 | 2.1e-5, 3.2e-6, 44% | pass | 0.375, 0.078 | 0 |

- The tolerance is not changed. The PIQA failure stands as recorded.
- **What the numbers show.** These are facts, not yet an explanation:
  - The explicit-mask context-only pass again equals the plain one exactly, so the mask code path is still not the cause.
  - A mask or position bug that let option tokens leak into the context would normally produce differences of order 0.1-1
    in float32, not 1e-4. But nothing yet shows that a float32 difference of this size is only rounding.
  - The largest state value on PIQA is about 15 (on OBQA, about 29), so a larger scale does not explain the excess.
  - PIQA has the shortest contexts of any benchmark checked.
- **A second point for D-b and the OOD interpretation.** The bf16 path differences on PIQA and Social IQa (relative L2
  0.13 and 0.18) are larger than on OBQA (0.078). Hidden-state readouts and probes on the OOD benchmarks will therefore
  carry more numerical noise between computation paths.
- **Proposed investigation D-a2** (it needs reviewer approval; its criteria are fixed here, before any run):
  1. **Record the configuration.** Record the TF32 and float32-matmul-precision flags and the attention implementation. The
     code never sets TF32, and PyTorch's default leaves it off for matmuls, but nothing records this.
  2. **Locate the difference.** On the same 100 PIQA rows, with Social IQa as control, record float32 packed-vs-context
     differences per loop and per question. A difference that is already present at loop 1 and stays flat points to a
     structural difference. One that starts near 1e-6 and grows through the loops points to rounding being amplified.
  3. **The decisive test: float64 on the CPU** for the 10 PIQA questions with the largest float32 difference. Questions are
     ranked by that difference, which is a numerical quantity, not an outcome.
     - If float64 packed-vs-context max |diff| ≤ 1e-9: the two passes compute the same function, and the float32 excess is
       rounding. The reviewer then decides how the PIQA D-a FAIL is handled. The 1e-4 tolerance does not change either way.
     - If it is > 1e-9: there is a real computational difference. Stop and treat it as a possible bug. The full extraction
       stays on HOLD.
     - The script checks that at least 16 GB of RAM is free before loading the model in float64 (about 11 GB of weights),
       and stops if not.
  4. **Output.** `results/phase3/diagnose_short_context_ood_a2.json`. The D-a report is never overwritten. No scores,
     labels or outcomes are computed.
