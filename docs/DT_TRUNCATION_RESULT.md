# D-t result: truncated execution vs retrospective evaluation (development data, 600 questions)

Run: 2026-10-27 on the laptop.
- Code and data: commit 3452d0b, under the frozen plan rev 3. Report `results/diagnostics/dt_truncation_report.json`, sha256 `1601a7e7…`.
- Model: Ouro-1.4B revision 574fa66c…, bf16, SDPA attention.
- Hardware and software: RTX 5050 Laptop GPU (capability 12.0), CUDA 12.8, cuDNN 9.19, torch 2.11.0+cu128, transformers 4.54.1.
- Numerics settings: TF32 matmul off, TF32 cuDNN on, deterministic algorithms off.

Test data: none touched. All 600 questions are plan-conformant (200 per benchmark), and no loop was non-finite.

## Pre-declared outcomes

**P-1 (exit loop, prediction, correctness): PASS.** The truncated run and the retrospective run agreed on all 1,800 policy-question
pairs, in every benchmark and for every policy. That is 100% agreement on the exit loop, the predicted option and correctness, so the
accuracy difference is 0.00 pp and the saving difference is 0.000 for all three policies. Every truncated run executed exactly as many
loops as its exit loop: 600 of 600 for each policy.

**P-2 (bit-for-bit statistic): FAIL as pre-declared.** It fails on HellaSwag only, and only for the two policies that read option scores:

| check | native-pb | π1-x | r-kl |
|---|---|---|---|
| (a) model-free: stored arrays with the future masked vs stored full arrays | 0 / 1,949 | 0 / 1,825 | 0 / 1,820 |
| (b) online truncated run vs stored trace | 0 / 1,949 | **210 / 1,825** (all HellaSwag) | **298 / 1,820** (all HellaSwag) |
| (c) online truncated run vs fresh full run in the same session | 0 / 1,949 | **131 / 1,825** (all HellaSwag) | **239 / 1,820** (all HellaSwag) |

- The largest difference in the statistic between fresh and stored values is 1.6×10⁻⁷ for π1-x and 1.3×10⁻⁸ for r-kl.
- **P-3** (fresh vs stored per-loop values, descriptive):
  - Option log-likelihoods: up to 9.2×10⁻⁵, on HellaSwag only.
  - conf and ent: up to 5.8×10⁻⁸ and 2.4×10⁻⁸, also HellaSwag only.
  - Gate logit, CDF and vocabulary entropy (all read at the last context token): exactly 0 on every benchmark.
  - OpenBookQA and CommonsenseQA: exactly 0 for every quantity.

## Reading (interpretation, not yet investigated)

- **Causality of the evaluation code is supported.** Check (a) is the model-free test of the claim in Section 3.6: the statistic at loop t does not change when
  later loops are removed. It found no difference in 5,594 checks. The verdict line printed "code-level causality: FAIL" because the implementation
  defined that flag as (a) AND (c), and (c) compares two separate GPU computations. That flag's name overstates what failed. We report
  the flag as printed and do not redefine it after the run.
- **The failures are differences between separate GPU computations of the same inputs.** Check (c) fails even between two runs in the same session.
  The differences are confined to option-token log-likelihoods on HellaSwag, the benchmark with the longest packed sequences (10-shot contexts,
  four long continuations), and they never reach the quantities read at the last context token. The most likely cause is run-to-run
  floating-point nondeterminism in the GPU kernels: deterministic algorithms were off, and SDPA and cuBLAS may reduce in a different order. **This is a
  hypothesis that has not been tested.**
- **No decision changed.** The differences (at most 1.6×10⁻⁷ in the statistic) did not change any exit loop, prediction or correctness on these 600 questions.

## What this supports for the paper (proposed wording for Section 3.6, pending the reviewer)

"Exit loops, predicted options and correctness under truncated execution agreed with the retrospective evaluation for all 600 development questions
and all three policies (1,800 of 1,800). A model-free check confirmed that no policy statistic at loop t depends on later loops. The statistics were not
bit-for-bit reproducible across separate GPU runs on HellaSwag (differences up to 1.6×10⁻⁷), which we attribute, without having isolated the cause, to
floating-point nondeterminism. These differences changed no decision on this subset. This validation covers the tested development subset only. It
does not cover the sealed test set or runtime."

The measured time savings were close to the simulated savings per benchmark: for example, π1-x 0.289 measured against 0.297 simulated on
CommonsenseQA. These are descriptive, batch-size-1 measurements on one laptop and are not latency results.

## Proposed follow-up: D-t2, a determinism check (needs reviewer GO; development data only)

This would test the nondeterminism hypothesis directly, before the wording above is finalised:

1. Take 20 HellaSwag questions from the D-t set: the first 20 by the D-t hash order.
2. Run each 5 times at full depth in one session with the default settings, and compare option log-likelihoods bit for bit across repeats.
3. Repeat with `torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

Pre-declared reading:
- If repeats differ under the defaults and agree under deterministic settings, the cause is GPU nondeterminism.
- If repeats agree under the defaults, nondeterminism is ruled out, and the D-t differences need a different explanation, which would then be
  investigated before any wording is final.

## D-t2 result (2026-10-27; reviewer GO; frozen procedure, commit cc3236c)

- **Default mode:** deterministic algorithms off and `CUBLAS_WORKSPACE_CONFIG` unset. 16 of 20 HellaSwag questions showed bit differences across 5
  full-depth repeats: up to 6.1×10⁻⁵ in option log-likelihoods, 3.8×10⁻⁸ in conf and 1.5×10⁻⁸ in ent. vent, gate and CDF were identical.
- **Deterministic mode:** `torch.use_deterministic_algorithms(True)` with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before CUDA initialised. Enabled
  without error, and 0 of 20 questions differed.
- **Pre-declared reading:** "default differs, deterministic agrees: evidence supports GPU nondeterminism as a cause".
- **Scope:** this establishes reproducibility only within the deterministic configuration that was tested. D-t's P-2 remains FAIL as recorded.
