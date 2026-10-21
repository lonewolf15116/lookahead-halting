# Gate 2 report (confirmation, frozen selection)

Gate 2 confirmation: frozen policies applied once to the never-scored confirmation set.

Generated 2026-10-21T15:28:04Z. Frozen selection sha256 `aa4f49e6a686a44c3136a30b36ce4516468e9d91913f2ddbc7aace31160eed96`; baseline B file sha256 `19fcf29583dc78a582d306fbafb82a0128ac6fb557cd36c8f1b14e1bb762eaaf`. Nothing was reselected or retuned on this set.

## 1. Original Gate 2 verdict (plan v3 section 7, unchanged)

**FAIL**  (L budget: False, E effect: False, H hidden state: False)

- P: macro saving 0.378, macro loss 2.02 pp (budget 1.00 pp, point estimate).
- Δ_base = S_P − S_best-baseline = 0.028 (comparator native_gate, saving 0.350); 95% CI [0.023, 0.033]; needs ≥ 0.03 and CI lower bound > 0.
- Δ_ctrl = S_P − S_Q = -0.022; 95% CI [-0.028, -0.015]; needs CI lower bound > 0.

Baselines on this set (validation-qualified, frozen thresholds):

| baseline | macro saving | macro loss (pp) |
|---|---|---|
| fixed-3 | 0.250 | 0.32 |
| confidence | 0.094 | 0.49 |
| entropy | 0.103 | 0.72 |
| native_gate | 0.350 | 1.16 |
| deployable_kl | 0.328 | 1.51 |

## 2. All policies, macro (equal weight over benchmarks)

| policy | role | saving [95% CI] | loss pp [95% CI] | pooled saving | pooled loss |
|---|---|---|---|---|---|
| fixed-4 | reference | 0.000 [0.000, 0.000] | 0.00 [0.00, 0.00] | 0.000 | 0.00 |
| P | candidate | 0.378 [0.373, 0.383] | 2.02 [1.19, 2.93] | 0.360 | 1.81 |
| Q | control_Q | 0.399 [0.395, 0.404] | 1.27 [0.41, 2.11] | 0.390 | 1.03 |
| fixed-3 | baseline | 0.250 [0.250, 0.250] | 0.32 [-0.34, 1.01] | 0.250 | 0.41 |
| confidence | baseline | 0.094 [0.090, 0.097] | 0.49 [0.28, 0.72] | 0.139 | 0.70 |
| entropy | baseline | 0.103 [0.099, 0.107] | 0.72 [0.47, 1.00] | 0.151 | 1.07 |
| native_gate | baseline | 0.350 [0.346, 0.354] | 1.16 [0.33, 2.01] | 0.359 | 1.13 |
| deployable_kl | baseline | 0.328 [0.324, 0.332] | 1.51 [0.58, 2.46] | 0.264 | 0.99 |
| B | amendment_B | 0.417 [0.417, 0.417] | 2.20 [1.18, 3.24] | 0.372 | 1.60 |
| B_cap | sensitivity | 0.250 [0.250, 0.250] | 0.61 [-0.12, 1.39] | 0.199 | 0.53 |
| shuffled_control | control_c | 0.407 [0.405, 0.409] | 2.18 [1.17, 3.18] | 0.367 | 1.60 |

## 3. Amendment v3.1 (separate from the verdict)

B uses benchmark identity: a task-aware control, not a deployable policy. B was chosen on validation among 1,024 vectors, so its validation numbers were optimistic.

| comparison | Δ saving | 95% CI |
|---|---|---|
| P_minus_B | -0.039 | [-0.044, -0.033] |
| Q_minus_B | -0.017 | [-0.022, -0.012] |
| P_minus_fixed3 | 0.128 | [0.123, 0.133] |
| Q_minus_fixed3 | 0.149 | [0.145, 0.154] |
| shuffled_control_minus_B | -0.009 | [-0.011, -0.008] |

B meets the macro budget on this set: False. The original verdict is reported separately above. B outperforming P or Q does not alter it.

### Per-benchmark saving / loss (pp), point estimates

| policy | arc_challenge | arc_easy | hellaswag | mmlu | commonsense_qa | openbookqa |
|---|---|---|---|---|---|---|
| fixed-4 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 |
| P | 0.39 / 1.7 | 0.43 / 3.0 | 0.44 / 1.9 | 0.27 / 1.1 | 0.36 / 1.7 | 0.39 / 2.8 |
| Q | 0.40 / 2.0 | 0.39 / 2.5 | 0.43 / 0.0 | 0.34 / 0.5 | 0.39 / 1.9 | 0.44 / 0.8 |
| fixed-3 | 0.25 / -0.7 | 0.25 / 0.5 | 0.25 / -0.3 | 0.25 / 0.3 | 0.25 / 1.5 | 0.25 / 0.6 |
| confidence | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.23 / 0.9 | 0.33 / 2.0 | 0.00 / 0.0 |
| entropy | 0.00 / 0.0 | 0.00 / 0.0 | 0.00 / 0.0 | 0.25 / 1.7 | 0.35 / 2.6 | 0.01 / 0.0 |
| native_gate | 0.33 / 2.0 | 0.33 / 0.9 | 0.42 / 1.0 | 0.32 / 0.7 | 0.41 / 2.4 | 0.28 / 0.0 |
| deployable_kl | 0.46 / 2.3 | 0.42 / 3.5 | 0.50 / 1.2 | 0.08 / 0.0 | 0.07 / -0.2 | 0.44 / 2.2 |
| B | 0.50 / 3.7 | 0.50 / 3.9 | 0.50 / 1.3 | 0.25 / 0.3 | 0.25 / 1.5 | 0.50 / 2.6 |
| B_cap | 0.25 / -0.7 | 0.25 / 0.5 | 0.25 / -0.3 | 0.00 / 0.0 | 0.25 / 1.5 | 0.50 / 2.6 |
| shuffled_control | 0.49 / 3.3 | 0.49 / 3.9 | 0.50 / 1.3 | 0.25 / 0.3 | 0.25 / 1.5 | 0.46 / 2.8 |

### Sensitivity: per-benchmark loss cap (1.0 pp; not a Gate 2 condition)

| policy | benchmarks over the cap |
|---|---|
| P | arc_challenge, arc_easy, hellaswag, mmlu, commonsense_qa, openbookqa |
| Q | arc_challenge, arc_easy, commonsense_qa |
| fixed-3 | commonsense_qa |
| confidence | commonsense_qa |
| entropy | mmlu, commonsense_qa |
| native_gate | arc_challenge, commonsense_qa |
| deployable_kl | arc_challenge, arc_easy, hellaswag, openbookqa |
| B | arc_challenge, arc_easy, hellaswag, commonsense_qa, openbookqa |
| B_cap | commonsense_qa, openbookqa |
| shuffled_control | arc_challenge, arc_easy, hellaswag, commonsense_qa, openbookqa |

B_cap vector on this set: saving 0.250, loss 0.61 pp.

## 4. Secondary analyses (never change the verdict)

- P_minus_best_loss_compliant_baseline: Δ = 0.128, 95% CI [0.123, 0.133]
- Q_minus_best_loss_compliant_baseline: Δ = 0.149, 95% CI [0.145, 0.154]
- Q_minus_best_baseline_all_qualified: Δ = 0.050, 95% CI [0.044, 0.055]
- without OpenBookQA: P loss 1.86 pp; P − best baseline 0.011 [0.006, 0.017]; P − Q -0.017 [-0.023, -0.010]
- Not run here (retrospective, to be added only after the verdict is recorded): the accuracy-budget frontier and leave-one-benchmark-out probes (plan 7a).

AUROC for S_t (loops 1-3), pooled: P: 0.714, 0.775, 0.794; Q: 0.687, 0.758, 0.854; shuffled_control: 0.595, 0.610, 0.605
