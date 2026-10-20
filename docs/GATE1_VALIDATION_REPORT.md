# Gate 1 validation report (validation (not the Gate 1 verdict))

Generated 2026-10-20T22:00:21.887399+00:00. All numbers below are from the **validation** split only. No test-split outcome has been read: test rows were filtered out of every parquet part at read time.

**S_or is the final-answer-preserving headroom** (earliest loop after which the answer never changes before loop 4), **not the maximum possible accuracy-preserving saving**: a policy could stop at an earlier answer that differs from the final one and still match fixed-4 accuracy. The earliest-correct oracle (reference row below) is a different, label-using reference: it raises accuracy but runs never-correct questions to loop 4, so its saving can be lower than S_or.

Macro = unweighted mean over benchmarks; question-weighted numbers are secondary (in the JSON report).

## Selection rules

Grid: 50 thresholds = np.quantile(values, np.linspace(0, 1, 50)), values = the statistic pooled over all validation questions of the benchmark set (ok questions only) and loops 1..3 (deployable_kl: D over its 3 columns); exact repeated thresholds are removed keeping order. Qualified point: macro validation loss <= 1.0 percentage point (+1e-12). Selection among qualified points: maximize macro validation saving; ties (1e-12) broken by smaller macro validation loss; remaining ties broken by the most conservative threshold (confidence, native_gate: largest; entropy, deployable_kl: smallest). A baseline with no qualified point is not qualified. fixed-k has one point each. Macro = unweighted mean over benchmarks. Thresholded policies halt at the earliest loop t in {1,2,3} whose rule fires, else run to loop 4. deployable_kl pays for loop t+1 and answers with loop t+1.

## Primary (arc_challenge, arc_easy, hellaswag, mmlu, commonsense_qa, openbookqa)

### Qualification and frozen thresholds (validation)

| baseline | qualified | threshold | macro saving | macro loss (pp) | macro acc (%) |
|---|---|---|---|---|---|
| fixed-1 | no | - | - | - | - |
| fixed-2 | no | - | - | - | - |
| fixed-3 | yes | - | 0.250 | -0.091 | 66.47 |
| confidence | yes | 0.698865 | 0.102 | 0.701 | 65.68 |
| entropy | yes | 0.926218 | 0.112 | 0.962 | 65.42 |
| native_gate | yes | 0.035145 | 0.359 | 0.607 | 65.78 |
| deployable_kl | yes | 0.017275 | 0.325 | 0.883 | 65.50 |

### Headroom on validation, primary - validation (not the Gate 1 verdict)

| baseline | qualified | threshold | S_b | S_b 95% CI | macro loss (pp) | loss 95% CI | constraint |
|---|---|---|---|---|---|---|---|
| fixed-1 | no | - | - | - | - | - | - |
| fixed-2 | no | - | - | - | - | - | - |
| fixed-3 | yes | - | 0.250 | [0.250, 0.250] | -0.091 | [-1.058, 0.850] | ok |
| confidence | yes | 0.698865 | 0.102 | [0.096, 0.109] | 0.701 | [0.237, 1.182] | ok |
| entropy | yes | 0.926218 | 0.112 | [0.106, 0.119] | 0.962 | [0.457, 1.502] | ok |
| native_gate | yes | 0.035145 | 0.359 | [0.353, 0.366] | 0.607 | [-0.662, 1.864] | ok |
| deployable_kl | yes | 0.017275 | 0.325 | [0.318, 0.331] | 0.883 | [-0.545, 2.339] | ok |

- S_or (macro saving of stable_final_oracle): 0.571  95% CI [0.560, 0.581]  (question-weighted 0.557)
- R = max S_b / S_or: 0.630  95% CI [0.614, 0.646]  attained by: native_gate  (question-weighted R 0.659, native_gate)
- Decision (PASS iff S_or >= 0.25 and R < 0.75): **PASS**

| benchmark | S_or | R | argmax |
|---|---|---|---|
| arc_challenge | 0.561 | 0.783 | deployable_kl |
| arc_easy | 0.614 | 0.699 | deployable_kl |
| hellaswag | 0.624 | 0.796 | deployable_kl |
| mmlu | 0.501 | 0.653 | native_gate |
| commonsense_qa | 0.534 | 0.777 | native_gate |
| openbookqa | 0.590 | 0.733 | deployable_kl |


### Reference rows, wall-clock (mean ms/question) and per-policy numbers

| policy | macro saving | macro loss (pp) | macro acc (%) | mean ms/question | pooled saving | pooled loss (pp) |
|---|---|---|---|---|---|---|
| fixed-4 | 0.000 | 0.000 | 66.38 | 388.1 | 0.000 | 0.000 |
| stable_final_oracle | 0.571 | 0.000 | 66.38 | 172.1 | 0.557 | 0.000 |
| earliest_correct_oracle (reference only) | 0.469 | -9.638 | 76.02 | 207.1 | 0.490 | -9.199 |
| fixed-3 | 0.250 | -0.091 | 66.47 | 292.5 | 0.250 | 0.733 |
| confidence | 0.102 | 0.701 | 65.68 | 352.1 | 0.129 | 0.579 |
| entropy | 0.112 | 0.962 | 65.42 | 348.5 | 0.144 | 0.971 |
| native_gate | 0.359 | 0.607 | 65.78 | 245.5 | 0.367 | 1.414 |
| deployable_kl | 0.325 | 0.883 | 65.50 | 262.8 | 0.271 | 1.431 |

Retrospective-KL grid (analysis only; answers with loop t, never a baseline):

| threshold | macro saving | macro loss (pp) | macro acc (%) |
|---|---|---|---|
| 0.000000 | 0.000 | 0.000 | 66.38 |
| 0.000011 | 0.013 | 0.027 | 66.36 |
| 0.000021 | 0.026 | 0.027 | 66.36 |
| 0.000033 | 0.039 | 0.027 | 66.36 |
| 0.000047 | 0.054 | -0.011 | 66.39 |
| 0.000064 | 0.069 | -0.021 | 66.40 |
| 0.000089 | 0.087 | 0.014 | 66.37 |
| 0.000122 | 0.102 | -0.033 | 66.42 |
| 0.000163 | 0.119 | 0.028 | 66.35 |
| 0.000218 | 0.138 | 0.097 | 66.29 |
| 0.000281 | 0.159 | 0.013 | 66.37 |
| 0.000364 | 0.180 | -0.111 | 66.49 |
| 0.000452 | 0.198 | 0.054 | 66.33 |
| 0.000551 | 0.217 | 0.237 | 66.15 |
| 0.000682 | 0.236 | 0.610 | 65.77 |
| 0.000836 | 0.252 | 0.781 | 65.60 |
| 0.001022 | 0.274 | 1.284 | 65.10 |
| 0.001236 | 0.290 | 1.884 | 64.50 |
| 0.001501 | 0.314 | 2.063 | 64.32 |
| 0.001788 | 0.333 | 2.578 | 63.80 |
| 0.002134 | 0.352 | 2.477 | 63.91 |
| 0.002553 | 0.372 | 2.809 | 63.57 |
| 0.003100 | 0.393 | 3.331 | 63.05 |
| 0.003796 | 0.414 | 3.842 | 62.54 |
| 0.004805 | 0.431 | 4.470 | 61.91 |
| 0.005916 | 0.455 | 5.469 | 60.91 |
| 0.007455 | 0.473 | 5.496 | 60.89 |
| 0.009089 | 0.492 | 5.913 | 60.47 |
| 0.011175 | 0.509 | 6.389 | 59.99 |
| 0.013845 | 0.526 | 6.768 | 59.61 |
| 0.017275 | 0.545 | 7.686 | 58.70 |
| 0.021884 | 0.562 | 8.406 | 57.98 |
| 0.026871 | 0.577 | 9.124 | 57.26 |
| 0.032516 | 0.589 | 9.455 | 56.93 |
| 0.039470 | 0.600 | 9.717 | 56.67 |
| 0.047479 | 0.611 | 10.280 | 56.10 |
| 0.058484 | 0.622 | 10.679 | 55.70 |
| 0.072225 | 0.634 | 11.122 | 55.26 |
| 0.088573 | 0.644 | 11.722 | 54.66 |
| 0.106789 | 0.653 | 11.797 | 54.59 |
| 0.129955 | 0.662 | 12.384 | 54.00 |
| 0.159833 | 0.673 | 13.177 | 53.21 |
| 0.196284 | 0.682 | 14.064 | 52.32 |
| 0.249547 | 0.693 | 14.493 | 51.89 |
| 0.314893 | 0.703 | 15.391 | 50.99 |
| 0.406159 | 0.711 | 16.056 | 50.33 |
| 0.541533 | 0.720 | 16.979 | 49.40 |
| 0.763894 | 0.729 | 17.700 | 48.68 |
| 1.252969 | 0.740 | 19.137 | 47.25 |
| 5.333366 | 0.750 | 21.444 | 44.94 |

### Transition tables (validation)

#### Transition 1->2

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 81 | 8 | 51 | 94 | 234 | 43 | 18.38 | [12.39, 24.36] |
| arc_easy | 249 | 13 | 136 | 77 | 475 | 123 | 25.89 | [21.89, 30.32] |
| hellaswag | 1035 | 57 | 389 | 527 | 2008 | 332 | 16.53 | [14.64, 18.43] |
| mmlu | 968 | 201 | 751 | 889 | 2809 | 550 | 19.58 | [17.44, 21.61] |
| commonsense_qa | 92 | 14 | 69 | 69 | 244 | 55 | 22.54 | [15.57, 29.10] |
| openbookqa | 29 | 8 | 15 | 48 | 100 | 7 | 7.00 | [-2.00, 17.00] |
| macro |  |  |  |  |  |  | 18.32 | [15.97, 20.65] |

#### Transition 2->3

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 123 | 9 | 15 | 87 | 234 | 6 | 2.56 | [-1.71, 6.84] |
| arc_easy | 374 | 11 | 26 | 64 | 475 | 15 | 3.16 | [0.84, 5.68] |
| hellaswag | 1362 | 62 | 112 | 472 | 2008 | 50 | 2.49 | [1.25, 3.83] |
| mmlu | 1559 | 160 | 325 | 765 | 2809 | 165 | 5.87 | [4.38, 7.37] |
| commonsense_qa | 151 | 10 | 30 | 53 | 244 | 20 | 8.20 | [3.28, 13.52] |
| openbookqa | 36 | 8 | 5 | 51 | 100 | -3 | -3.00 | [-10.00, 4.00] |
| macro |  |  |  |  |  |  | 3.21 | [1.51, 4.94] |

#### Transition 3->4

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 130 | 8 | 6 | 90 | 234 | -2 | -0.85 | [-3.85, 2.14] |
| arc_easy | 391 | 9 | 4 | 71 | 475 | -5 | -1.05 | [-2.53, 0.42] |
| hellaswag | 1458 | 16 | 28 | 506 | 2008 | 12 | 0.60 | [-0.05, 1.25] |
| mmlu | 1792 | 92 | 130 | 795 | 2809 | 38 | 1.35 | [0.36, 2.35] |
| commonsense_qa | 176 | 5 | 6 | 57 | 244 | 1 | 0.41 | [-2.46, 3.28] |
| openbookqa | 39 | 2 | 1 | 58 | 100 | -1 | -1.00 | [-5.00, 2.00] |
| macro |  |  |  |  |  |  | -0.09 | [-1.06, 0.85] |

#### Transition 1->4

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 75 | 14 | 61 | 84 | 234 | 47 | 20.09 | [13.25, 26.92] |
| arc_easy | 245 | 17 | 150 | 63 | 475 | 133 | 28.00 | [23.58, 32.42] |
| hellaswag | 1013 | 79 | 473 | 443 | 2008 | 394 | 19.62 | [17.53, 21.71] |
| mmlu | 955 | 214 | 967 | 673 | 2809 | 753 | 26.81 | [24.53, 29.01] |
| commonsense_qa | 94 | 12 | 88 | 50 | 244 | 76 | 31.15 | [23.77, 38.11] |
| openbookqa | 28 | 9 | 12 | 51 | 100 | 3 | 3.00 | [-5.00, 12.00] |
| macro |  |  |  |  |  |  | 21.44 | [18.93, 23.76] |

#### Transition 2->4

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 121 | 11 | 15 | 87 | 234 | 4 | 1.71 | [-2.56, 6.41] |
| arc_easy | 370 | 15 | 25 | 65 | 475 | 10 | 2.11 | [-0.42, 4.63] |
| hellaswag | 1360 | 64 | 126 | 458 | 2008 | 62 | 3.09 | [1.84, 4.43] |
| mmlu | 1547 | 172 | 375 | 715 | 2809 | 203 | 7.23 | [5.59, 8.83] |
| commonsense_qa | 149 | 12 | 33 | 50 | 244 | 21 | 8.61 | [3.28, 14.34] |
| openbookqa | 34 | 10 | 6 | 50 | 100 | -4 | -4.00 | [-12.00, 4.00] |
| macro |  |  |  |  |  |  | 3.12 | [1.29, 4.97] |

#### Transition 3->4 (T_final)

| benchmark | CC | CW | WC | WW | n | net (WC-CW) | net_pts | net_pts 95% CI |
|---|---|---|---|---|---|---|---|---|
| arc_challenge | 130 | 8 | 6 | 90 | 234 | -2 | -0.85 | [-3.85, 2.14] |
| arc_easy | 391 | 9 | 4 | 71 | 475 | -5 | -1.05 | [-2.53, 0.42] |
| hellaswag | 1458 | 16 | 28 | 506 | 2008 | 12 | 0.60 | [-0.05, 1.25] |
| mmlu | 1792 | 92 | 130 | 795 | 2809 | 38 | 1.35 | [0.36, 2.35] |
| commonsense_qa | 176 | 5 | 6 | 57 | 244 | 1 | 0.41 | [-2.46, 3.28] |
| openbookqa | 39 | 2 | 1 | 58 | 100 | -1 | -1.00 | [-5.00, 2.00] |
| macro |  |  |  |  |  |  | -0.09 | [-1.06, 0.85] |

## Sensitivity (no openbookqa: arc_challenge, arc_easy, hellaswag, mmlu, commonsense_qa)

### Qualification and frozen thresholds (validation)

| baseline | qualified | threshold | macro saving | macro loss (pp) | macro acc (%) |
|---|---|---|---|---|---|
| fixed-1 | no | - | - | - | - |
| fixed-2 | no | - | - | - | - |
| fixed-3 | yes | - | 0.250 | 0.091 | 71.57 |
| confidence | yes | 0.705309 | 0.119 | 0.827 | 70.83 |
| entropy | yes | 0.834934 | 0.115 | 0.631 | 71.03 |
| native_gate | yes | 0.035712 | 0.372 | 0.573 | 71.09 |
| deployable_kl | yes | 0.007709 | 0.261 | 0.925 | 70.73 |

### Headroom on validation, sensitivity - validation (not the Gate 1 verdict)

| baseline | qualified | threshold | S_b | S_b 95% CI | macro loss (pp) | loss 95% CI | constraint |
|---|---|---|---|---|---|---|---|
| fixed-1 | no | - | - | - | - | - | - |
| fixed-2 | no | - | - | - | - | - | - |
| fixed-3 | yes | - | 0.250 | [0.250, 0.250] | 0.091 | [-0.842, 0.988] | ok |
| confidence | yes | 0.705309 | 0.119 | [0.112, 0.127] | 0.827 | [0.306, 1.415] | ok |
| entropy | yes | 0.834934 | 0.115 | [0.108, 0.122] | 0.631 | [0.214, 1.062] | ok |
| native_gate | yes | 0.035712 | 0.372 | [0.366, 0.377] | 0.573 | [-0.510, 1.641] | ok |
| deployable_kl | yes | 0.007709 | 0.261 | [0.256, 0.267] | 0.925 | [0.045, 1.821] | ok |

- S_or (macro saving of stable_final_oracle): 0.567  95% CI [0.557, 0.576]  (question-weighted 0.557)
- R = max S_b / S_or: 0.656  95% CI [0.644, 0.669]  attained by: native_gate  (question-weighted R 0.660, native_gate)
- Decision (PASS iff S_or >= 0.25 and R < 0.75): **PASS**

| benchmark | S_or | R | argmax |
|---|---|---|---|
| arc_challenge | 0.561 | 0.697 | deployable_kl |
| arc_easy | 0.614 | 0.574 | deployable_kl |
| hellaswag | 0.624 | 0.765 | deployable_kl |
| mmlu | 0.501 | 0.652 | native_gate |
| commonsense_qa | 0.534 | 0.774 | native_gate |


### Reference rows and wall-clock

| policy | macro saving | macro loss (pp) | macro acc (%) | mean ms/question | pooled saving | pooled loss (pp) |
|---|---|---|---|---|---|---|
| fixed-4 | 0.000 | 0.000 | 71.66 | 447.6 | 0.000 | 0.000 |
| stable_final_oracle | 0.567 | 0.000 | 71.66 | 198.5 | 0.557 | 0.000 |
| earliest_correct_oracle (reference only) | 0.491 | -8.365 | 80.02 | 236.6 | 0.493 | -9.081 |
| fixed-3 | 0.250 | 0.091 | 71.57 | 337.1 | 0.250 | 0.763 |
| confidence | 0.119 | 0.827 | 70.83 | 405.0 | 0.128 | 0.555 |
| entropy | 0.115 | 0.631 | 71.03 | 406.8 | 0.126 | 0.624 |
| native_gate | 0.372 | 0.573 | 71.09 | 282.1 | 0.367 | 1.369 |
| deployable_kl | 0.261 | 0.925 | 70.73 | 321.7 | 0.234 | 1.023 |

Retrospective-KL grid (analysis only; answers with loop t, never a baseline):

| threshold | macro saving | macro loss (pp) | macro acc (%) |
|---|---|---|---|
| 0.000000 | 0.000 | 0.000 | 71.66 |
| 0.000011 | 0.015 | 0.032 | 71.63 |
| 0.000021 | 0.029 | 0.032 | 71.63 |
| 0.000033 | 0.045 | 0.032 | 71.63 |
| 0.000046 | 0.061 | -0.013 | 71.67 |
| 0.000063 | 0.077 | -0.026 | 71.68 |
| 0.000087 | 0.098 | 0.016 | 71.64 |
| 0.000119 | 0.114 | -0.039 | 71.70 |
| 0.000159 | 0.133 | 0.034 | 71.62 |
| 0.000214 | 0.151 | 0.106 | 71.55 |
| 0.000277 | 0.171 | -0.036 | 71.69 |
| 0.000359 | 0.190 | -0.112 | 71.77 |
| 0.000444 | 0.208 | 0.065 | 71.59 |
| 0.000546 | 0.225 | 0.179 | 71.48 |
| 0.000672 | 0.242 | 0.732 | 70.93 |
| 0.000828 | 0.258 | 0.937 | 70.72 |
| 0.001018 | 0.275 | 1.541 | 70.12 |
| 0.001233 | 0.292 | 2.261 | 69.40 |
| 0.001498 | 0.311 | 2.476 | 69.18 |
| 0.001790 | 0.328 | 2.900 | 68.76 |
| 0.002149 | 0.345 | 3.130 | 68.53 |
| 0.002590 | 0.365 | 3.211 | 68.45 |
| 0.003154 | 0.384 | 4.017 | 67.64 |
| 0.003924 | 0.401 | 4.621 | 67.04 |
| 0.004954 | 0.417 | 5.195 | 66.46 |
| 0.006132 | 0.436 | 5.748 | 65.91 |
| 0.007709 | 0.454 | 6.397 | 65.26 |
| 0.009537 | 0.471 | 7.003 | 64.66 |
| 0.011644 | 0.485 | 7.561 | 64.10 |
| 0.014488 | 0.503 | 8.505 | 63.15 |
| 0.018140 | 0.521 | 9.247 | 62.41 |
| 0.022840 | 0.537 | 10.197 | 61.46 |
| 0.028345 | 0.552 | 10.869 | 60.79 |
| 0.033969 | 0.566 | 11.192 | 60.47 |
| 0.040797 | 0.577 | 11.881 | 59.78 |
| 0.049361 | 0.589 | 12.469 | 59.19 |
| 0.060804 | 0.602 | 12.996 | 58.66 |
| 0.074888 | 0.614 | 13.224 | 58.44 |
| 0.091937 | 0.626 | 13.448 | 58.21 |
| 0.110450 | 0.636 | 13.698 | 57.96 |
| 0.134774 | 0.647 | 14.399 | 57.26 |
| 0.164260 | 0.659 | 15.347 | 56.31 |
| 0.201644 | 0.670 | 16.209 | 55.45 |
| 0.255652 | 0.682 | 16.827 | 54.83 |
| 0.321277 | 0.694 | 17.987 | 53.67 |
| 0.416738 | 0.704 | 18.799 | 52.86 |
| 0.550267 | 0.715 | 19.796 | 51.86 |
| 0.773226 | 0.726 | 20.565 | 51.09 |
| 1.267918 | 0.738 | 22.386 | 49.27 |
| 5.333366 | 0.750 | 25.132 | 46.53 |
