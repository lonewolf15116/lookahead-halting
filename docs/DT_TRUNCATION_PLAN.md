# D-t: truncated-execution check of the retrospective evaluation (plan, rev 2 with implementation, NOT RUN)

Purpose: the reviewer's full-paper review (2026-10-27) asked us to validate the retrospective evaluation against real truncated execution, on
a small independent diagnostic set, without reopening TEST. Every C1 result was computed by applying frozen policies to stored full-depth
traces (paper Section 3.6). This check runs the same frozen policies with real early stopping and compares the outcomes with the stored traces.

Status: diagnostic, labelled [E] in the paper. It changes no frozen artifact and no C1 result.

## Data

- Development data only. The diagnostic set is drawn from the DEV+ extraction of the three primary benchmarks (OpenBookQA, CommonsenseQA,
  HellaSwag): **200 questions per benchmark, 600 in total**. Selection is by sha256 of `"lh-dt-v1:" + doc_id`, taking the 200 lowest per
  benchmark among questions with status `ok`. The selection is fixed before any run, and the selected ids are written to the report.
- These questions were part of the development data used to fit and select the frozen policies. That does not matter here, because the check
  concerns computational agreement, not generalisation. TEST is not touched: the run refuses any path under `data/extracted_phase3_test`.

## Policies

Frozen, exactly as in `results/c1/frozen.json` (`65728e16…`) and `results/c1/pi1x_models.npz` (`72f15508…`). Hashes are checked before the run.
- π1-x with the per-benchmark primary $(\tau_b, m_b)$.
- native-pb with the per-benchmark primary $(\tau_b, m_b)$.
- r-kl with its frozen per-benchmark $(\tau_b, m_b)$.

## Procedure, per question

1. **Truncated run.** Iterate `PackedScorer.loops()` one loop at a time. After loop t, compute the policy's statistic using only loops 1..t:
   - the arrays for later loops are filled with NaN;
   - the statistic is computed by the same code used for C1 (`E.predict_e1` / the readout and CDF functions);
   - the exit is decided by `R.exit_loops`.

   If the policy exits at t, stop: the generator is closed and no further loop is computed. Record the exit loop, the predicted option and
   correctness at that loop, and the wall time, with CUDA-synchronised timing, from start to exit.
2. **Full run.** Run all 4 loops for the same question and record the wall time.
3. **Stored comparison.** Apply the same policy retrospectively to the stored DEV+ arrays for the same question, through the C1 code path
   (`c1_test.apply_policy` on the loaded development data).

## Pre-declared outcomes (fixed before the run)

- **Primary agreement (P-1).** For each policy and benchmark, report the share of questions where the truncated exit loop equals the
  retrospective exit loop, and where truncated correctness equals retrospective correctness.
  - Expected: 100% for both.
  - Any disagreement is reported in full (ids, loops, statistic values), investigated, and disclosed in the paper.
  - No tolerance is set, and nothing is re-run to make disagreements go away.
- **Causality check (P-2).** At every loop t before the exit, the statistic computed with NaN for loops after t must equal the statistic from
  the stored full arrays at loop t, bit for bit. This is the code-level claim of Section 3.6, tested directly. A NaN or a difference is a failure.
- **Fresh vs stored values (P-3, descriptive).** Report the maximum absolute difference between freshly computed and stored per-loop values:
  gate logit, option log-likelihoods and conf/ent/vent. bf16 kernels may be nondeterministic across runs, so P-3 is descriptive. Any decision
  that changes is caught by P-1.
- **Measured vs simulated saving (E, descriptive only).** Per policy and benchmark:
  - simulated gross saving = 1 − mean(e)/4;
  - measured time saving = 1 − Σ truncated time / Σ full time.

  Report both. This is not a latency claim:
  - batch size 1;
  - Python loop overhead included;
  - the policy statistic is computed on the CPU after the loop;
  - laptop RTX 5050.

  It shows only whether loop-count savings appear in wall time on this implementation.

## What the paper will say

- If P-1 and P-2 pass, Section 3.6 changes from "rests on inspection, not experiment" to "confirmed on 600 development questions by truncated
  execution (D-t)". The measured time saving is reported as an implementation-specific observation [E], not a latency result.
- If either fails, the failure is reported. The retrospective results are then qualified, and the reviewer decides how to proceed.

## Run

`scripts/dt_truncation_check.py` and `scripts/run_dt_truncation_windows.ps1` (laptop GPU, about 600 × 2 question runs, expected a few minutes).
Output: `results/diagnostics/dt_truncation_report.json`, created exclusively (the run refuses to overwrite). Tests:
`tests/test_dt_truncation.py`, which run on a tiny random model and synthetic stored data.

## Rev 2: implementation details (code in `src/lookahead/dt_truncation.py`, `scripts/dt_truncation_check.py`; 24 tests)

Changes from rev 1:
- **Single-question vs batch evaluation.** C1 evaluates π1-x over a batch. Evaluating one question at a time differs in the last bit for some
  rows (85 of 600 synthetic rows, from BLAS accumulation order; native-pb and r-kl show 0). How this is handled:
  - P-2's headline compares the online statistic with the stored full row evaluated one question at a time.
  - The batch-vs-single difference is reported separately.
  - P-1 compares against the real C1 batch `apply_policy`, so a decision flip from last-bit differences would show up in P-1.
- **P-2 is reported three ways:**
  1. online vs stored full row (the headline);
  2. online vs the fresh full 4-loop row;
  3. stored row with the future set to NaN vs the stored full row, which is a model-free check of causality at the code level.
- **Runs.** Each question gets 1 full run plus 1 truncated run per policy, so 600 × 4 runs. Two untimed warm-up runs come first, and the order of
  full and truncated runs alternates per question. Both kinds of run do the same per-loop bookkeeping. The truncated run also computes the policy
  statistic on the CPU, so the measured time saving is conservative.
- **Non-finite loops.** A non-finite loop makes the question run all loops and count as wrong, as in extraction.
- **r-kl selection.** r-kl's frozen (τ, m) is taken from `frozen.json` → `secondary_risk_all_baselines["r-kl"]`.
- **C1 TEST bundle hash.** Adding `dt_truncation.py` to `src/lookahead` changes it. TEST is spent and the TEST script is never run again (its
  marker refuses), so this has no consequence. It is recorded here and in DECISIONS.
