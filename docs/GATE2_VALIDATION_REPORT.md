# Gate 2 validation report (validation (not the Gate 2 verdict))

Generated 2026-10-21T08:30:15Z. Plan v3. Train and validation splits only: the Gate 1 test split and the confirmation set were not opened. Nothing here is a Gate 2 verdict.

Macro = unweighted mean over benchmarks. Operating point: Gate 1's rule (max saving with macro validation loss <= 1 pp).

## Failed questions

Questions with status != ok are excluded from probe fitting and kept in policy evaluation exactly as in Gate 1 (wrong at every loop, 4 loops, no statistic fires).

| benchmark | train n | train failed | validation n | validation failed |
|---|---|---|---|---|
| arc_challenge | 703 | 0 | 234 | 0 |
| arc_easy | 1426 | 0 | 475 | 0 |
| hellaswag | 6025 | 0 | 2008 | 0 |
| mmlu | 8428 | 0 | 2809 | 0 |
| commonsense_qa | 733 | 0 | 244 | 0 |
| openbookqa | 300 | 0 | 100 | 0 |

## Probe configurations (validation)

| config | qualified | threshold | val macro saving | val macro loss (pp) | val macro acc (%) |  |
|---|---|---|---|---|---|---|
| family1-h | yes | 0.871083 | 0.353 | 0.772 | 65.61 |  |
| family1-hx | yes | 0.852986 | 0.388 | 0.944 | 65.44 | **chosen** |
| family2-h | yes | 0.845676 | 0.378 | 0.932 | 65.45 |  |
| family2-hx | yes | 0.851611 | 0.380 | 0.774 | 65.61 |  |
| family3-h | yes | 0.882689 | 0.380 | 0.795 | 65.59 |  |
| family3-hx | yes | 0.883145 | 0.380 | 0.795 | 65.59 |  |

## P (chosen), Q (cheap features) and baselines at their validation operating points

| policy | qualified | threshold | macro saving | macro loss (pp) | macro acc (%) |
|---|---|---|---|---|---|
| P family1-hx | yes | 0.852986 | 0.388 | 0.944 | 65.44 |
| Q Q-family1-x | yes | 0.790742 | 0.403 | 0.533 | 65.85 |
| fixed-1 | no | - | - | - | - |
| fixed-2 | no | - | - | - | - |
| fixed-3 | yes | - | 0.250 | -0.091 | 66.47 |
| confidence | yes | 0.698865 | 0.102 | 0.701 | 65.68 |
| entropy | yes | 0.926218 | 0.112 | 0.962 | 65.42 |
| native_gate | yes | 0.035145 | 0.359 | 0.607 | 65.78 |
| deployable_kl | yes | 0.017275 | 0.325 | 0.883 | 65.50 |

## Controls

| policy | qualified | threshold | macro saving | macro loss (pp) | macro acc (%) |
|---|---|---|---|---|---|
| control-c-family1-h-shuffled | yes | 0.822251 | 0.406 | 0.390 | 65.99 |

## Hyper-parameters chosen on train

- family1-h: {'0': {'C': 0.0003, 'cv_logloss': 0.6035160955869693, 'converged': True, 'n': 17615}, '1': {'C': 0.0003, 'cv_logloss': 0.41878011175148505, 'converged': True, 'n': 17615}, '2': {'C': 0.001, 'cv_logloss': 0.20622005996127468, 'converged': True, 'n': 17615}}
- family1-hx: {'0': {'C': 0.001, 'cv_logloss': 0.5987056665601311, 'converged': True, 'n': 17615}, '1': {'C': 0.001, 'cv_logloss': 0.41482847034547693, 'converged': True, 'n': 17615}, '2': {'C': 0.001, 'cv_logloss': 0.2033242097223515, 'converged': True, 'n': 17615}}
- family2-h: {'0': {'C': 0.0003, 'cv_logloss': 0.4102576650574285, 'converged': True, 'n': 52845}}
- family2-hx: {'0': {'C': 0.001, 'cv_logloss': 0.4032057936431478, 'converged': True, 'n': 52845}}
- family3-h: {'0': {'weight_decay': 0.0001, 'best_epoch': 2, 'es_bce': 0.5967499017715454, 'n_fit': 15807, 'n_es': 1808}, '1': {'weight_decay': 0.0001, 'best_epoch': 1, 'es_bce': 0.4367048740386963, 'n_fit': 15807, 'n_es': 1808}, '2': {'weight_decay': 0.01, 'best_epoch': 1, 'es_bce': 0.21945251524448395, 'n_fit': 15807, 'n_es': 1808}}
- family3-hx: {'0': {'weight_decay': 0.001, 'best_epoch': 2, 'es_bce': 0.5932345986366272, 'n_fit': 15807, 'n_es': 1808}, '1': {'weight_decay': 0.001, 'best_epoch': 1, 'es_bce': 0.435748815536499, 'n_fit': 15807, 'n_es': 1808}, '2': {'weight_decay': 0.001, 'best_epoch': 1, 'es_bce': 0.21839387714862823, 'n_fit': 15807, 'n_es': 1808}}
- Q-family1-x: {'0': {'C': 1.0, 'cv_logloss': 0.6336170792196338, 'converged': True, 'n': 17615}, '1': {'C': 1.0, 'cv_logloss': 0.4254206791372064, 'converged': True, 'n': 17615}, '2': {'C': 1.0, 'cv_logloss': 0.18817444648108186, 'converged': True, 'n': 17615}}
- control-c-family1-h-shuffled: {'0': {'C': 0.0001, 'cv_logloss': 0.6631706869327322, 'converged': True, 'n': 17615}, '1': {'C': 0.0001, 'cv_logloss': 0.4953140805051291, 'converged': True, 'n': 17615}, '2': {'C': 0.0001, 'cv_logloss': 0.24110228285489135, 'converged': True, 'n': 17615}}

## Secondary predictive quality on validation (family 1; no selection uses these)

| target/input | metric | pooled | loop 1 | loop 2 | loop 3 |
|---|---|---|---|---|---|
| S/P:family1-hx | auroc | 0.841 | 0.725 | 0.782 | 0.818 |
| S/Q:family1-x | auroc | 0.839 | 0.690 | 0.785 | 0.870 |
| S/control-c:family1-h-shuffled | auroc | 0.768 | 0.594 | 0.613 | 0.603 |
| F/h | auroc | 0.804 | 0.676 | 0.759 | 0.809 |
| C/h | auroc | 0.741 | 0.667 | 0.734 | 0.752 |
| WC/h | auroc | 0.787 | 0.629 | 0.732 | 0.770 |
| CW/h | auroc | 0.737 | 0.703 | 0.694 | 0.777 |
| logD/h | spearman | 0.925 | 0.876 | 0.872 | 0.889 |
| G/h | spearman | 0.342 | 0.411 | 0.299 | 0.067 |
| F/x | auroc | 0.817 | 0.677 | 0.784 | 0.870 |
| C/x | auroc | 0.714 | 0.633 | 0.696 | 0.714 |
| WC/x | auroc | 0.796 | 0.625 | 0.753 | 0.850 |
| CW/x | auroc | 0.744 | 0.652 | 0.721 | 0.841 |
| logD/x | spearman | 0.865 | 0.760 | 0.777 | 0.844 |
| G/x | spearman | 0.327 | 0.327 | 0.294 | 0.049 |

WC vs CW among WC/CW transitions (positive = WC):

| input | pairs | WC | CW | AUROC P(WC)-P(CW) | AUROC F-probe score |
|---|---|---|---|---|---|
| WCvsCW/h | 2792 | 2099 | 693 | 0.666 | 0.545 |
| WCvsCW/x | 2792 | 2099 | 693 | 0.615 | 0.548 |


## Probe cost (multiply-adds per question per loop, chosen P)

{1: 2057, 2: 2057, 3: 2057}


## Selection rules

Probe policy: s_t = predicted P(S_t) from the probe at loop t (t in 1..3); halt at the first t with s_t >= threshold, else loop 4. Failed questions have s_t = -inf (run 4 loops, wrong), as in Gate 1. Grid, operating-point rule and tie-breaks are exactly Gate 1's (gate1.grid_thresholds / gate1.select_point with kind native_gate: ties -> smaller loss, then the largest threshold). Probe choice: among 3 families x {h, hx}, largest validation operating-point saving; ties -> smaller validation loss -> simpler (family 1<2<3, h<hx); unqualified configs cannot be chosen.
