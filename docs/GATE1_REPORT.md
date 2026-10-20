# Gate 1 report (test split, frozen selection)

Generated 2026-10-20T22:18:19.021615+00:00. Thresholds frozen on validation (frozen_selection.json sha256 `4f3bc80fd62f8ccf70886a71adc015a5bbe71a066cd681d1ea873ba63d7c169f`); nothing was reselected on test.

**S_or is the final-answer-preserving headroom, not the maximum possible accuracy-preserving saving.**

## Gate 1 verdict (primary): **PASS**

### Primary (six benchmarks)

| baseline | qualified | threshold | S_b | S_b 95% CI | macro loss (pp) | loss 95% CI | constraint |
|---|---|---|---|---|---|---|---|
| fixed-1 | no | - | - | - | - | - | - |
| fixed-2 | no | - | - | - | - | - | - |
| fixed-3 | yes | - | 0.250 | [0.250, 0.250] | 0.981 | [-0.070, 2.038] | ok |
| confidence | yes | 0.698865 | 0.097 | [0.091, 0.104] | 1.009 | [0.543, 1.533] | VIOLATES loss<=1 |
| entropy | yes | 0.926218 | 0.110 | [0.103, 0.116] | 1.220 | [0.617, 1.870] | VIOLATES loss<=1 |
| native_gate | yes | 0.035145 | 0.357 | [0.350, 0.364] | 1.623 | [0.357, 2.895] | VIOLATES loss<=1 |
| deployable_kl | yes | 0.017275 | 0.328 | [0.321, 0.334] | 1.956 | [0.565, 3.435] | VIOLATES loss<=1 |

- S_or (macro saving of stable_final_oracle): 0.564  95% CI [0.552, 0.575]  (question-weighted 0.555)
- R = max S_b / S_or: 0.633  95% CI [0.617, 0.648]  attained by: native_gate  (question-weighted R 0.661, native_gate)
- Decision (PASS iff S_or >= 0.25 and R < 0.75): **PASS**

| benchmark | S_or | R | argmax |
|---|---|---|---|
| arc_challenge | 0.574 | 0.804 | deployable_kl |
| arc_easy | 0.617 | 0.675 | deployable_kl |
| hellaswag | 0.637 | 0.779 | deployable_kl |
| mmlu | 0.487 | 0.668 | native_gate |
| commonsense_qa | 0.526 | 0.799 | native_gate |
| openbookqa | 0.542 | 0.797 | deployable_kl |


**Constraint violation (test macro loss > 1 pp):** confidence, entropy, native_gate, deployable_kl

- R_unc (diagnostic, best test-feasible grid point per baseline): 0.553 (native_gate)

| policy | macro saving | macro loss (pp) | macro acc (%) | mean ms/question | pooled saving | pooled loss (pp) |
|---|---|---|---|---|---|---|
| fixed-4 | 0.000 | 0.000 | 65.30 | 389.3 | 0.000 | 0.000 |
| stable_final_oracle | 0.564 | 0.000 | 65.30 | 171.8 | 0.555 | 0.000 |
| earliest_correct_oracle (reference only) | 0.456 | -8.472 | 73.78 | 210.0 | 0.489 | -9.100 |
| fixed-3 | 0.250 | 0.981 | 64.32 | 293.3 | 0.250 | 0.784 |
| confidence | 0.097 | 1.009 | 64.29 | 353.5 | 0.125 | 0.665 |
| entropy | 0.110 | 1.220 | 64.08 | 349.4 | 0.139 | 1.091 |
| native_gate | 0.357 | 1.623 | 63.68 | 246.9 | 0.367 | 1.142 |
| deployable_kl | 0.328 | 1.956 | 63.35 | 261.7 | 0.269 | 1.704 |

### Transition tables (test)

#### Transition 1->2

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 78 | 11 | 40 | 106 | 235 | 29 | 12.34 | [6.38, 18.30] |
| arc_easy | 263 | 16 | 119 | 77 | 475 | 103 | 21.68 | [17.05, 26.11] |
| hellaswag | 1070 | 64 | 382 | 493 | 2009 | 318 | 15.83 | [13.84, 17.82] |
| mmlu | 889 | 236 | 762 | 918 | 2805 | 526 | 18.75 | [16.68, 20.86] |
| commonsense_qa | 88 | 16 | 73 | 67 | 244 | 57 | 23.36 | [16.39, 29.93] |
| openbookqa | 26 | 5 | 10 | 59 | 100 | 5 | 5.00 | [-2.00, 12.00] |
| macro |  |  |  |  |  |  | 16.16 | [14.07, 18.26] |

#### Transition 2->3

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 112 | 6 | 17 | 100 | 235 | 11 | 4.68 | [0.85, 8.94] |
| arc_easy | 372 | 10 | 30 | 63 | 475 | 20 | 4.21 | [1.68, 6.74] |
| hellaswag | 1406 | 46 | 93 | 464 | 2009 | 47 | 2.34 | [1.19, 3.43] |
| mmlu | 1496 | 155 | 356 | 798 | 2805 | 201 | 7.17 | [5.60, 8.81] |
| commonsense_qa | 154 | 7 | 26 | 57 | 244 | 19 | 7.79 | [3.28, 12.30] |
| openbookqa | 29 | 7 | 3 | 61 | 100 | -4 | -4.00 | [-10.00, 2.00] |
| macro |  |  |  |  |  |  | 3.70 | [2.18, 5.29] |

#### Transition 3->4

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 125 | 4 | 8 | 98 | 235 | 4 | 1.70 | [-0.85, 4.68] |
| arc_easy | 394 | 8 | 7 | 66 | 475 | -1 | -0.21 | [-1.89, 1.47] |
| hellaswag | 1480 | 19 | 21 | 489 | 2009 | 2 | 0.10 | [-0.50, 0.70] |
| mmlu | 1758 | 94 | 129 | 824 | 2805 | 35 | 1.25 | [0.21, 2.28] |
| commonsense_qa | 175 | 5 | 10 | 54 | 244 | 5 | 2.05 | [-0.82, 5.33] |
| openbookqa | 30 | 2 | 3 | 65 | 100 | 1 | 1.00 | [-3.00, 6.00] |
| macro |  |  |  |  |  |  | 0.98 | [-0.07, 2.04] |

#### Transition 1->4

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 78 | 11 | 55 | 91 | 235 | 44 | 18.72 | [12.34, 25.11] |
| arc_easy | 266 | 13 | 135 | 61 | 475 | 122 | 25.68 | [21.05, 30.32] |
| hellaswag | 1062 | 72 | 439 | 436 | 2009 | 367 | 18.27 | [16.18, 20.26] |
| mmlu | 900 | 225 | 987 | 693 | 2805 | 762 | 27.17 | [24.99, 29.45] |
| commonsense_qa | 90 | 14 | 95 | 45 | 244 | 81 | 33.20 | [26.23, 40.57] |
| openbookqa | 22 | 9 | 11 | 58 | 100 | 2 | 2.00 | [-7.00, 11.00] |
| macro |  |  |  |  |  |  | 20.84 | [18.56, 23.16] |

#### Transition 2->4

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 109 | 9 | 24 | 93 | 235 | 15 | 6.38 | [1.70, 11.07] |
| arc_easy | 369 | 13 | 32 | 61 | 475 | 19 | 4.00 | [1.26, 6.74] |
| hellaswag | 1399 | 53 | 102 | 455 | 2009 | 49 | 2.44 | [1.29, 3.58] |
| mmlu | 1468 | 183 | 419 | 735 | 2805 | 236 | 8.41 | [6.67, 10.23] |
| commonsense_qa | 151 | 10 | 34 | 49 | 244 | 24 | 9.84 | [4.92, 15.16] |
| openbookqa | 28 | 8 | 5 | 59 | 100 | -3 | -3.00 | [-10.00, 4.00] |
| macro |  |  |  |  |  |  | 4.68 | [2.96, 6.48] |

#### Transition 3->4 (T_final)

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 125 | 4 | 8 | 98 | 235 | 4 | 1.70 | [-0.85, 4.68] |
| arc_easy | 394 | 8 | 7 | 66 | 475 | -1 | -0.21 | [-1.89, 1.47] |
| hellaswag | 1480 | 19 | 21 | 489 | 2009 | 2 | 0.10 | [-0.50, 0.70] |
| mmlu | 1758 | 94 | 129 | 824 | 2805 | 35 | 1.25 | [0.21, 2.28] |
| commonsense_qa | 175 | 5 | 10 | 54 | 244 | 5 | 2.05 | [-0.82, 5.33] |
| openbookqa | 30 | 2 | 3 | 65 | 100 | 1 | 1.00 | [-3.00, 6.00] |
| macro |  |  |  |  |  |  | 0.98 | [-0.07, 2.04] |

## Sensitivity verdict (no openbookqa): **PASS**

### Sensitivity (five benchmarks)

| baseline | qualified | threshold | S_b | S_b 95% CI | macro loss (pp) | loss 95% CI | constraint |
|---|---|---|---|---|---|---|---|
| fixed-1 | no | - | - | - | - | - | - |
| fixed-2 | no | - | - | - | - | - | - |
| fixed-3 | yes | - | 0.250 | [0.250, 0.250] | 0.978 | [0.062, 1.892] | ok |
| confidence | yes | 0.705309 | 0.114 | [0.106, 0.121] | 1.197 | [0.627, 1.824] | VIOLATES loss<=1 |
| entropy | yes | 0.834934 | 0.114 | [0.106, 0.121] | 1.108 | [0.506, 1.761] | VIOLATES loss<=1 |
| native_gate | yes | 0.035712 | 0.371 | [0.365, 0.376] | 1.938 | [0.765, 3.086] | VIOLATES loss<=1 |
| deployable_kl | yes | 0.007709 | 0.265 | [0.259, 0.270] | 2.192 | [1.191, 3.199] | VIOLATES loss<=1 |

- S_or (macro saving of stable_final_oracle): 0.568  95% CI [0.559, 0.578]  (question-weighted 0.555)
- R = max S_b / S_or: 0.653  95% CI [0.640, 0.666]  attained by: native_gate  (question-weighted R 0.662, native_gate)
- Decision (PASS iff S_or >= 0.25 and R < 0.75): **PASS**

| benchmark | S_or | R | argmax |
|---|---|---|---|
| arc_challenge | 0.574 | 0.717 | deployable_kl |
| arc_easy | 0.617 | 0.578 | deployable_kl |
| hellaswag | 0.637 | 0.752 | deployable_kl |
| mmlu | 0.487 | 0.665 | native_gate |
| commonsense_qa | 0.526 | 0.795 | native_gate |


**Constraint violation (test macro loss > 1 pp):** confidence, entropy, native_gate, deployable_kl

- R_unc (diagnostic): 0.490 (native_gate)

| policy | macro saving | macro loss (pp) | macro acc (%) | mean ms/question | pooled saving | pooled loss (pp) |
|---|---|---|---|---|---|---|
| fixed-4 | 0.000 | 0.000 | 71.76 | 449.2 | 0.000 | 0.000 |
| stable_final_oracle | 0.568 | 0.000 | 71.76 | 197.5 | 0.555 | 0.000 |
| earliest_correct_oracle (reference only) | 0.489 | -7.566 | 79.33 | 239.0 | 0.492 | -9.033 |
| fixed-3 | 0.250 | 0.978 | 70.79 | 338.3 | 0.250 | 0.780 |
| confidence | 0.114 | 1.197 | 70.57 | 407.3 | 0.124 | 0.641 |
| entropy | 0.114 | 1.108 | 70.66 | 407.5 | 0.123 | 0.693 |
| native_gate | 0.371 | 1.938 | 69.83 | 283.9 | 0.367 | 1.144 |
| deployable_kl | 0.265 | 2.192 | 69.57 | 320.6 | 0.233 | 1.283 |
