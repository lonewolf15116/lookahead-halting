# Phase 3 plan (v1.1, revised on reviewer review; for freeze. Nothing has been run)

Working title: **decision-aware halting beyond task-aware depth.**
Question: *can question-specific information improve computation allocation beyond what task identity alone achieves?*

Status: v1 specification. Reviewer decision on the draft: CONDITIONAL GO, five required changes (applied here; change log at
the end). Step 0 is on HOLD until this specification is frozen **and** the Gate 2 confirmation verdict is committed. No Gate 2
confirmation outcome is used to design anything here, and the Gate 2 confirmation set is never a Phase 3 test set.

## 1. What Gate 2 taught (validation, frozen selection; not the Gate 2 verdict)
| policy | macro saving | macro loss (pp) |
|---|---|---|
| P, hidden state + cheap features, one pooled threshold | 0.388 | 0.94 |
| Q, cheap features, one pooled threshold | 0.403 | 0.53 |
| label-shuffled control | 0.406 | 0.39 |
| B (Gate 2 v3.1), fixed depth per benchmark (2,2,2,3,3,2), macro rule | **0.417** | 0.78 |
| stable-final oracle (final-answer preserving) | 0.571 | 0.00 |

1. The Gate 2 policies used one pooled threshold and so could not express a per-benchmark depth policy.
2. Question-level signal exists but is uneven: within-benchmark AUROC for S_t up to 0.90 (MMLU) and 0.94 (CSQA) for Q at
   loop 3; 0.49-0.73 on ARC and HellaSwag (control audit, section G).
3. Possible hidden-state niche: separating wrong→correct (WC) from correct→wrong (CW) transitions, AUROC 0.666 (h_t) vs
   0.615 (cheap features), pooled over benchmarks; not yet checked within benchmark.
4. Stability is a proxy for value. The loss-gain target G was weakly predicted (Spearman 0.34 pooled, about 0.05-0.07 at loop 3).
5. The macro loss budget hid per-benchmark damage (OpenBookQA gains offset HellaSwag losses of 2-3 pp).

## 2. Hypotheses
- **H3 (primary, task-aware track):** a frozen per-question halting policy that nests B achieves ≥ 0.03 more **net**
  macro loop saving (controller overhead included, section 5a) than the **strongest eligible task-aware baseline** on a fresh
  in-distribution test set, with every benchmark's loss ≤ 1 pp (point estimate) and a paired-bootstrap 95% CI lower
  bound of the improvement > 0.
- **H3-h (secondary):** the hidden-state version of the selected policy beats its cheap-features-only twin (CI > 0).
- **H3-agn (secondary, task-agnostic track):** without benchmark identity, the best task-agnostic policy beats the best
  task-agnostic baseline by ≥ 0.03 (CI > 0) under the same per-benchmark loss rule.
- **H3-ood (separate):** transfer to unseen benchmarks without retuning (section 8), with its own criteria.

## 3. Data roles
| set | role | exposure |
|---|---|---|
| Phase 1 **train** (17,615 questions, six benchmarks) | fitting **and all selection** via 5-fold cross-fitting (frozen fold hash `lh-fold-v1`) | Gate 2 fitted probes on it |
| Phase 1 **validation** (5,870) | development check only; reported, never used to select | **exposed**: Gate 2 selected thresholds and P/Q on it |
| Phase 1 test, Gate 2 confirmation | not used | opened |
| **Fresh in-distribution set** (section 9) | one-time test, primary | none, after audit |
| **Out-of-distribution set** (PIQA, WinoGrande; Social IQa reserve) | one-time transfer test, separate | none, after audit |

## 4. Accuracy rule and uncertainty (pre-registered)
- **Primary:** for every benchmark b, loss_b = acc_b(loop 4) − acc_b(policy) ≤ 1.00 pp on the selection data (point
  estimate, + the Gate 1 tolerance 1e-12). On the test set, the same point-estimate rule decides "budget met".
- **Uncertainty:** every per-benchmark loss is reported with its paired-bootstrap 95% CI and the upper bound. A point
  estimate ≤ 1 pp is not a guarantee that the population loss is ≤ 1 pp, and no such claim is made. Whether any upper
  bound exceeds 1 pp is reported in the verdict table.
- **Secondary (continuity with Gates 1-2):** macro ≤ 1 pp; question-weighted pooled ≤ 1 pp.

## 5. Policy class with an explicit fallback to fixed depth (correction C)
Every task-aware policy has, per benchmark b, two parameters: a score threshold τ_b and a depth cap m_b ∈ {1,2,3,4}:
> halt at the first loop t < m_b with s_t ≥ τ_b; otherwise halt at m_b.

τ_b = +∞ is always in the grid, and with it the policy **is exactly fixed depth m_b**, whatever the scores. So every
task-aware policy contains every per-benchmark fixed-depth vector, including B_pb, by construction and not through extreme
learned thresholds.
- **Grid for τ_b:** the 50 quantiles of the policy's pooled out-of-fold train statistic over ok questions and loops 1-3
  (exact repeats removed, as in Gate 1), plus +∞. So 51 × 4 = 204 candidates per benchmark per policy.
- **B_pb (primary baseline):** the τ_b = +∞ subset of the same grid, chosen by the same rule on the same data. It is
  selected under the per-benchmark constraint on cross-fitted train, not taken from the earlier validation observation
  (3,3,3,4,3,2).
- **Selection rule, per benchmark independently** (the constraint is per benchmark, so the problem separates): among
  candidates with loss_b ≤ 1 pp on the selection data, max saving_b; ties (1e-12) → smaller loss_b → fixed depth
  (τ = +∞) preferred → larger τ_b → smaller m_b. If no adaptive candidate beats the best fixed depth, the policy *is*
  B_pb for that benchmark (fallback). Candidates with no qualifying point fall back to m_b = 4.
- **Selection data:** cross-fitted train. Each fold's statistic comes from models fitted on the other four folds. τ_b and
  m_b are chosen on the pooled out-of-fold statistics; final models are refit on all train with the frozen
  hyperparameters, and the out-of-fold-selected (τ_b, m_b) are applied unchanged.
- **Task-agnostic track:** the same class with one (τ, m) shared by all benchmarks and no benchmark-identity feature.
  The rule is: max macro saving subject to every benchmark's loss ≤ 1 pp, then the same tie-breaks. Its baselines are pooled
  fixed depth and native gate, chosen the same way.

## 5a. Net saving (controller overhead) and the comparator
- **Net saving per question** = 1 − (e + ρ_b · n_eval) / 4, with e the executed loops, n_eval the number of loops at which
  the controller computed a score (halting at loop t < m_b by score: n_eval = t; reaching the cap m_b: n_eval = m_b − 1),
  and ρ_b = measured controller milliseconds per evaluation divided by measured mean milliseconds per loop for benchmark b
  (batch size 1, same GPU, same code path as extraction). Policies that need no extra computation (fixed depth, Ouro's native
  gate CDF, which the model already produces) have ρ = 0. ρ_b is measured once on development data at G3b and frozen.
  Gross saving (ρ = 0) is reported beside it. The verdict uses net saving.
- **Strongest eligible task-aware baseline:** among the frozen task-aware baselines that were qualified at selection
  (B_pb and native-pb), the one with the highest net test saving, re-determined inside every bootstrap resample. As in
  Gate 2, a baseline that exceeds 1 pp on a test benchmark still sets the bar. The comparison against baselines that
  meet the cap on the test set is reported as secondary.

## 6. Candidate list (closed; no additions after freeze)
Features: x = Gate 2 cheap features (9); hx = h_t (2048) + x. Models: family 1 only (per-loop L2-logistic or ridge,
Gate 2 section 5 settings: C grid, alpha grid, 5-fold CV inside each training fold). Families 2 and 3 are dropped
(no validation gain in Gate 2).
| id | statistic s_t | features | note |
|---|---|---|---|
| π1-x / π1-hx | P̂(S_t), stability (Gate 2 target) | x / hx | task-aware stability |
| π3-x / π3-hx | −V_t with V_t = P̂(c_4=1 \| z_t) − P̂(c_t=1 \| z_t), the expected gain from committing to loop 4 | x / hx | final-loop gain; halt when the gain is small |
| π4-x / π4-hx | advantage of stopping, A_t = P̂(c_t=1 \| z_t) − Ĉ_t(z_t) | x / hx | sequential value-to-go (below) |
| native-pb | Ouro's native gate CDF | none | task-aware baseline |
| fixed-pb = B_pb | none (τ = +∞) | none | primary baseline |
| shuffled-pb | π1-hx with labels shuffled within (benchmark, loop) | hx | control: should reduce to ≈ B_pb |

**π4, sequential controller (one specification, no variants).** Finite horizon of 4 loops, cost one loop per step,
fitted value iteration with λ = 0 for the value function. Then a per-benchmark threshold on the stopping advantage plays the
role of the cost:
- Stopping value at loop t: p_t(z_t) = P̂(c_t = 1 | z_t), a cross-fitted logistic model per loop.
- Continuation value, by backward induction on train (realised next states):
  - t = 3: continuing means answering at loop 4, so the target is Y_3 = c_4;
  - t = 2: Y_2 = max(p_3(z_3), Ĉ_3(z_3));
  - t = 1: Y_1 = max(p_2(z_2), Ĉ_2(z_2)).
  Ĉ_t(z_t) is a ridge regression of Y_t on z_t. All p and Ĉ are cross-fitted: targets and statistics for fold k come
  from models that did not see fold k.
- s_t = A_t = P̂(c_t=1 \| z_t) − Ĉ_t(z_t); halt when A_t ≥ τ_b (τ_b ≤ 0 means "stop if within |τ_b| of the value of going
  on"). The per-benchmark τ_b grid is as in section 5.

**Distinction kept explicit.** π3's V_t = P̂(c_4=1 | z_t) − P̂(c_t=1 | z_t) estimates the benefit of *proceeding to loop 4*, not of taking *one more loop and reconsidering*; a controller that can stop at t+1 or t+2 is worth more than V_t says, and π4 is the policy that models that. π3 is kept as the simpler final-loop-gain policy, as the reviewer asked. Its sign is flipped so that "halt when s_t ≥ τ" means "halt
when the expected gain from continuing is small".

**Final policy choice** (task-aware track): among the six candidates (π1, π3, π4 × x, hx), the largest cross-fitted
macro saving under the primary rule; ties → smaller max per-benchmark
loss → simpler (π1 < π3 < π4; x < hx). **H3-h** compares the chosen policy with its twin on the other feature set
(the chosen one and its twin are both frozen).

## 7. Step 0 / G3a: feasibility (train and validation only; no new policy is selected for testing)
| id | diagnostic | stop rule |
|---|---|---|
| D1 | Oracle headroom over B_pb (B_pb chosen on cross-fitted train). Two retrospective ceilings, both on known outcomes, **not deployable policies**: (i) *final-answer-preserving oracle* (Gate 1's stable-final, for continuity); (ii) *accuracy-budget oracle* (correction A): per benchmark, choose each question's exit e_i ∈ {1..4} to minimise Σ e_i subject to Σ c_{i,e_i} ≥ Σ c_{i,4} − 0.01 n_b. **The oracle may keep an incorrect early answer**, as long as the benchmark stays within its 1 pp budget. Exact solution, per benchmark: let k_b = Σ c_{i,1} (correct at loop 1) and K_b = Σ c_{i,4} (correct at loop 4). Start with every question exiting at loop 1. If k_b ≥ K_b − 0.01·n_b the budget already holds. Otherwise "buy" r_b = ⌈K_b − 0.01·n_b − k_b⌉ extra correct answers: a question wrong at loop 1 and correct at some loop has earliest correct loop ε_i and gains +1 correct for ε_i − 1 extra loops. These costs are independent across questions, so buying the r_b cheapest (ties by doc_id) is optimal. Questions correct at loop 1 and questions never correct exit at loop 1. A question that is correct at an early loop but wrong at loop 4 counts as correct, so the oracle can profit from the model unlearning an answer; that is why this ceiling is loose and is only a necessary condition. | stop if ceiling (ii) − B_pb < 0.05 macro saving |
| D2 | Reachable headroom, screening only (correction B). π1-x and π1-hx with Gate 2's frozen hyperparameters, refit by 5-fold cross-fitting on train; (τ_b, m_b) chosen on four folds' out-of-fold statistics and evaluated on the fifth, rotated (nested). B_pb selected the same nested way. Validation is reported as a second, **exposed** estimate (Gate 2 used it for selection). | stop if the nested-train improvement over B_pb < 0.02 |
| D3 | Within-benchmark AUROC for WC vs CW (adjacent loops and to-final), h_t vs x, per benchmark and loop, out of fold. | H3-h and the hx variants are kept only if h > x within at least half of the (benchmark, loop) cells with ≥ 30 transitions |
| D4 | Within-benchmark predictability of c_t, c_4 and Δ_t = c_4 − c_t (AUROC / Spearman, out of fold). | informative only |

Caveats fixed in advance:
- Ceiling (ii) includes questions answered correctly at an early loop by chance. It is a loose upper bound, so D1 is a
  necessary condition only; D2 is the informative screen.
- D2 passing means "worth developing", not "reachable": its estimate is cross-fitted on train and still optimistic.

## 8. Unseen benchmarks (correction D)
- **Task-aware policies on an unseen benchmark:** a rule fixed now, with no access to the new benchmark's outcomes. The
  unseen benchmark gets the **task-agnostic track's frozen (τ, m)** and statistic. No benchmark-specific threshold is ever
  assigned after looking at test outcomes.
- **OOD baseline:** the task-agnostic pooled fixed depth (and pooled native gate), frozen on train.
- **OOD criteria (separate from H3):** per OOD benchmark, loss ≤ 1 pp (point estimate, CI reported) and saving
  improvement over the pooled fixed-depth baseline ≥ 0.03 with CI lower bound > 0. These are reported per benchmark and
  are never pooled with the in-distribution result.

## 9. Fresh test data (to be inventoried and audited before freeze; **no audited fresh set exists yet**)
In-distribution candidates are the unscored remainders of the six benchmarks' other splits, after the Gate 2 audit
procedure (exemplar recording, identity and template-independent text-hash exclusion, Phase 1 hash reproduction):
- **HellaSwag train:** about 676 left (1,676 eligible in the Gate 2 audit, 1,000 used).
- **CommonsenseQA train:** about 1,435 left (2,435 eligible, 1,000 used).
- **ARC-Challenge train, ARC-Easy train, OpenBookQA train:** not yet audited (pool sizes before exclusion 1,119 / 2,251 /
  4,957; how many were drawn as few-shot exemplars is unknown).
- **MMLU:** no fresh in-distribution pool (test used by Phase 1, validation by Gate 2, dev holds the exemplars).
  `auxiliary_train` is a different distribution and would count as OOD.

Pre-registered handling:
- A benchmark with < 300 audit-clean fresh questions is excluded from the in-distribution test and listed as such. The
  primary H3 macro is over the benchmarks that remain, fixed at freeze time from the inventory, not from outcomes.
- MMLU is therefore expected to be absent from the primary test (stated now).
- OOD: audit PIQA and WinoGrande first; Social IQa in reserve. Benchmarks used, sizes and seeds are fixed at freeze.
- Power check before freeze: a bootstrap of the Gate 2 validation P−B difference gave a CI width of about 0.016 on
  5,870 questions. Expected widths at the inventoried sizes are reported with the inventory, so a too-small test set is
  known before it is spent.

## 10. Gates
- **G3a (feasibility):** D1-D4 on train and validation. Ends Phase 3 with a written negative if D1 or D2 fails.
- **G3b (development):** candidates of section 6 fitted and selected on cross-fitted train. Every hyperparameter,
  (τ_b, m_b), the chosen policy, its twin, the baselines, the task-agnostic policies and the OOD rule are frozen with
  sha256 before any test outcome. Validation is reported as an exposed development check.
- **G3c (one-time tests):** the frozen policies are applied once to the fresh in-distribution set (primary) and once to the
  OOD set (separate), by a guarded script in the Gate 2 pattern.

Verdict table (H3; exhaustive), written before G3c:
| budget: every in-dist benchmark loss ≤ 1 pp | effect: Δ = S_net,π − S_net,strongest eligible baseline ≥ 0.03 and CI lower > 0 | verdict |
|---|---|---|
| yes | yes | **PASS**: question-specific allocation beyond task-aware depth |
| yes | no | **FAIL (no decision value shown)**: Δ and CI reported |
| no | any | **FAIL (budget)** |

H3-h, H3-agn, H3-ood, gross saving and measured wall-clock saving are reported beside the verdict and never change it. Claims are limited to Ouro-1.4B and the evaluated benchmarks.

## 11. Not allowed
- Using any Gate 2 confirmation outcome; selecting anything on validation; adding candidates or features after the freeze.
- Changing the accuracy rule, the 0.03 margin or the stop rules after G3a starts.
- Assigning thresholds to unseen benchmarks from their outcomes.
- Reporting a policy without its task-aware (or task-agnostic) baseline selected under the same discipline.
- If a check fails: stop, keep the artifacts, investigate.

## 12. Order of work
1. Reviewer approves v1, then freeze (commit and tag).
2. Gate 2 confirmation verdict committed (independent of Phase 3; must come first).
3. Data inventory and audits for section 9 (CPU), then extraction of the audited sets (GPU, about 35 min per 5k questions).
   Outcomes stay sealed.
4. G3a diagnostics, then the reviewer decides GO/STOP.
5. G3b development and freeze, then reviewer sign-off on the implementation (before training this time).
6. G3c one-time tests, then `docs/PHASE3_REPORT.md`.

## Change log
- v1 → v1.1 (second reviewer note, which had seen only the draft file): D1 states explicitly that the oracle may keep a wrong early
  answer within the budget; the G3c comparator is the strongest eligible task-aware baseline and the effect is on **net** saving
  with controller overhead inside the metric (section 5a); the "commit to loop 4" vs "one more loop" distinction is explicit; the
  absence of an audited fresh set is stated. D2's procedure, ties and fallback are in sections 5 and 7 (unchanged).
- Draft (fb9f669) → v1, on reviewer decision (CONDITIONAL GO):
  - (A) D1 uses an accuracy-budget oracle, with the stable-final oracle kept for reference; both labelled retrospective.
  - (B) D2 is nested cross-fitted on train; validation is reported as exposed.
  - (C) every task-aware policy has an explicit fixed-depth fallback (τ = +∞, cap m_b), so B_pb is nested exactly and
    selected under the same discipline.
  - (D) in-distribution and OOD tests are separated, with an unseen-benchmark rule and OOD criteria.
  - (E) pre-registered: per-benchmark ≤ 1 pp primary with uncertainty reporting, macro as secondary, the 0.03 margin with
    CI > 0, and a closed candidate list including one sequential value-to-go controller (π4).
  - Reviewer decisions adopted: task-aware primary and task-agnostic secondary tracks; PIQA and WinoGrande audited first.
  - Thresholds and B are now selected on cross-fitted train, not on validation, because validation is exposed.
