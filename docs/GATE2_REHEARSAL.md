# Gate 2 rehearsal (validation)

**REHEARSAL on VALIDATION data. This is not the Gate 2 verdict.** It checks the pipeline against the frozen validation numbers.

Generated 2026-10-21T10:41:53Z. Frozen selection sha256 `aa4f49e6a686a44c3136a30b36ce4516468e9d91913f2ddbc7aace31160eed96`; baseline B file sha256 `b9033ae4463beea84f99572802589df32929467c530c32212b76fa3ea1381cd9`. Nothing was reselected or retuned on this set.

## 1. Original Gate 2 verdict (plan v3 section 7, unchanged)

**FAIL**  (L budget: True, E effect: False, H hidden state: False)

- P: macro saving 0.388, macro loss 0.94 pp (budget 1.00 pp, point estimate).
- Δ_base = S_P − S_best-baseline = 0.029 (comparator native_gate, saving 0.359); 95% CI [0.021, 0.037]; needs ≥ 0.03 and CI lower bound > 0.
- Δ_ctrl = S_P − S_Q = -0.015; 95% CI [-0.024, -0.005]; needs CI lower bound > 0.

Baselines on this set (validation-qualified, frozen thresholds):

| baseline | macro saving | macro loss (pp) |
|---|---|---|
| fixed-3 | 0.250 | -0.09 |
| confidence | 0.102 | 0.70 |
| entropy | 0.112 | 0.96 |
| native_gate | 0.359 | 0.61 |
| deployable_kl | 0.325 | 0.88 |

## 2. All policies, macro (equal weight over benchmarks)

| policy | role | saving [95% CI] | loss pp [95% CI] | pooled saving | pooled loss |
|---|---|---|---|---|---|
| fixed-4 | reference | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.000 | 0.00 |
| P | candidate | 0.388 [0.380, 0.396] | 0.94 [-0.43, 2.25] | 0.354 | 1.72 |
| Q | control_Q | 0.403 [0.396, 0.409] | 0.53 [-0.91, 1.96] | 0.386 | 1.43 |
| fixed-3 | baseline | 0.250 [0.250, 0.250] | -0.09 [-1.06, 0.85] | 0.250 | 0.73 |
| confidence | baseline | 0.102 [0.096, 0.109] | 0.70 [0.24, 1.18] | 0.129 | 0.58 |
| entropy | baseline | 0.112 [0.106, 0.119] | 0.96 [0.46, 1.50] | 0.144 | 0.97 |
| native_gate | baseline | 0.359 [0.353, 0.366] | 0.61 [-0.66, 1.86] | 0.367 | 1.41 |
| deployable_kl | baseline | 0.325 [0.318, 0.331] | 0.88 [-0.55, 2.34] | 0.271 | 1.43 |
| B | amendment_B | 0.417 [0.417, 0.417] | 0.78 [-0.84, 2.41] | 0.370 | 1.89 |
| B_cap | sensitivity | 0.250 [0.250, 0.250] | -0.82 [-2.33, 0.58] | 0.135 | 0.03 |
| shuffled_control | control_c | 0.406 [0.402, 0.409] | 0.39 [-1.22, 1.91] | 0.367 | 1.87 |

## 3. Amendment v3.1 (separate from the verdict)

B uses benchmark identity: a task-aware control, not a deployable policy. B was chosen on validation among 1,024 vectors, so its validation numbers were optimistic.

| comparison | Δ saving | 95% CI |
|---|---|---|
| P_minus_B | -0.029 | [-0.036, -0.021] |
| Q_minus_B | -0.014 | [-0.021, -0.007] |
| P_minus_fixed3 | 0.138 | [0.130, 0.146] |
| Q_minus_fixed3 | 0.153 | [0.146, 0.159] |
| shuffled_control_minus_B | -0.011 | [-0.014, -0.007] |

B meets the macro budget on this set: True. The original verdict is reported separately above. B outperforming P or Q does not alter it.

### Per-benchmark saving / loss (pp), point estimates

| policy | arc_challenge | arc_easy | hellaswag | mmlu | commonsense_qa | openbookqa |
|---|---|---|---|---|---|---|
| fixed-4 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 |
| P | 0.40 / 0.4 | 0.43 / 1.9 | 0.43 / 3.2 | 0.28 / 0.9 | 0.37 / 1.2 | 0.41 / -2.0 |
| Q | 0.40 / 1.3 | 0.41 / 1.9 | 0.43 / 2.0 | 0.34 / 1.2 | 0.40 / 0.8 | 0.43 / -4.0 |
| fixed-3 | 0.25 / -0.9 | 0.25 / -1.1 | 0.25 / 0.6 | 0.25 / 1.4 | 0.25 / 0.4 | 0.25 / -1.0 |
| confidence | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.24 / 0.9 | 0.36 / 3.3 | 0.01 / 0.0 |
| entropy | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.27 / 1.7 | 0.38 / 4.1 | 0.02 / 0.0 |
| native_gate | 0.36 / -1.3 | 0.33 / -0.2 | 0.43 / 2.4 | 0.33 / 1.3 | 0.41 / 0.4 | 0.29 / 1.0 |
| deployable_kl | 0.44 / 2.1 | 0.43 / 1.7 | 0.50 / 3.1 | 0.08 / 0.4 | 0.07 / 0.0 | 0.43 / -2.0 |
| B | 0.50 / 1.7 | 0.50 / 2.1 | 0.50 / 3.1 | 0.25 / 1.4 | 0.25 / 0.4 | 0.50 / -4.0 |
| B_cap | 0.25 / -0.9 | 0.25 / -1.1 | 0.25 / 0.6 | 0.00 / 0.0 | 0.25 / 0.4 | 0.50 / -4.0 |
| shuffled_control | 0.49 / 1.3 | 0.49 / 2.1 | 0.50 / 3.2 | 0.25 / 1.4 | 0.26 / 0.4 | 0.46 / -6.0 |

### Sensitivity: per-benchmark loss cap (1.0 pp; not a Gate 2 condition)

| policy | benchmarks over the cap |
|---|---|
| P | arc_easy, hellaswag, commonsense_qa |
| Q | arc_challenge, arc_easy, hellaswag, mmlu |
| fixed-3 | mmlu |
| confidence | commonsense_qa |
| entropy | mmlu, commonsense_qa |
| native_gate | hellaswag, mmlu |
| deployable_kl | arc_challenge, arc_easy, hellaswag |
| B | arc_challenge, arc_easy, hellaswag, mmlu |
| B_cap | none |
| shuffled_control | arc_challenge, arc_easy, hellaswag, mmlu |

B_cap vector on this set: saving 0.250, loss -0.82 pp.

## 4. Secondary analyses (never change the verdict)

- P_minus_best_loss_compliant_baseline: Δ = 0.029, 95% CI [0.021, 0.037]
- Q_minus_best_loss_compliant_baseline: Δ = 0.044, 95% CI [0.034, 0.052]
- Q_minus_best_baseline_all_qualified: Δ = 0.044, 95% CI [0.034, 0.052]
- without OpenBookQA: P loss 1.53 pp; P − best baseline 0.011 [0.004, 0.017]; P − Q -0.013 [-0.021, -0.006]
- Not run here (retrospective, to be added only after the verdict is recorded): the accuracy-budget frontier and leave-one-benchmark-out probes (plan 7a).

AUROC for S_t (loops 1-3), pooled: P: 0.725, 0.782, 0.818; Q: 0.690, 0.785, 0.870; shuffled_control: 0.594, 0.613, 0.603
