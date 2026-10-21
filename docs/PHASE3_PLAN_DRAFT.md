# Phase 3 plan (DRAFT for reviewer discussion; nothing frozen, nothing run)

Working title: **decision-aware halting on top of task-aware depth.**

Status: draft. Phase 3 starts only after the Gate 2 confirmation verdict is written and committed. No number from the Gate 2
confirmation set is used to design anything here, and that set cannot serve as Phase 3's test set (it will have been opened).
All diagnostics below use train and validation only.

## 1. What Gate 2 taught (validation, frozen selection; not the Gate 2 verdict)
| policy | macro saving | macro loss (pp) |
|---|---|---|
| P, hidden state + cheap features, one pooled threshold | 0.388 | 0.94 |
| Q, cheap features, one pooled threshold | 0.403 | 0.53 |
| label-shuffled control | 0.406 | 0.39 |
| B, fixed depth per benchmark (2,2,2,3,3,2) | **0.417** | 0.78 |
| stable-final oracle (question-level, zero loss) | 0.571 | 0.00 |

Lessons, each with its evidence:
1. **The Gate 2 policy class could not represent B.** P and Q use one threshold pooled over six benchmarks; B is per
   benchmark. A probe with per-benchmark thresholds contains B (constant scores reproduce it), so it can only add
   question-level variation on top of task-aware depth.
2. **Question-level signal exists.** Within-benchmark AUROC for S_t reaches 0.90 (MMLU) and 0.94 (CSQA) for Q at loop 3,
   but is weak on ARC and HellaSwag (0.49-0.73 across P and Q and loops; control audit, section G).
3. **Headroom above B exists at oracle level:** 0.571 − 0.417 = 0.154 macro saving, and that oracle spends no loss budget.
   How much of it is reachable is unknown (step 0).
4. **Possible hidden-state niche: direction of change, not stability.** Separating wrong→correct (WC) from
   correct→wrong (CW) transitions: AUROC 0.666 with h_t vs 0.615 with cheap features (pooled validation, 2,099 WC vs 693 CW
   adjacent-loop transitions). Not yet checked within benchmark, so it may again be benchmark identity.
5. **Stability ≠ value.** Predicting "the answer will not change" is a proxy; what matters is the accuracy change from
   continuing. The loss-gain target G was poorly predicted (Spearman 0.34 pooled, ~0.05-0.07 at loop 3).
6. **The macro loss budget hid per-benchmark damage.** OpenBookQA gains (100 validation questions) offset HellaSwag losses
   of 2-3 pp. The Phase 3 constraint must be fixed before any result (section 5).

## 2. Hypothesis
**H3:** A per-question halting rule that starts from task-aware depth and stops on an estimate of the *accuracy value of
continuing* achieves a higher saving than B under the same, pre-registered accuracy constraint, on data never used in
development.
**H3-h (secondary):** hidden states improve that estimate beyond cheap features (most plausibly through WC-vs-CW direction).
Both are falsifiable by step 0 before any new model is trained.

## 3. Step 0: go/no-go diagnostics (train and validation only; hours, CPU)
| id | diagnostic | proposed stop rule (reviewer to set) |
|---|---|---|
| D1 | **Budget-matched oracle headroom over B.** Per benchmark, the question-level oracle that maximises saving subject to the Phase 3 accuracy constraint (knows every loop's correctness). Report oracle − B. | stop if oracle − B < 0.05 under the primary constraint |
| D2 | **Reachable headroom.** Existing Gate 2 scores (P, Q, native gate) with *per-benchmark* thresholds, thresholds chosen on out-of-fold train predictions, evaluated on validation. | stop if no score beats B on validation by ≥ 0.02 |
| D3 | **Within-benchmark WC-vs-CW AUROC**, h_t vs cheap features, per benchmark and loop. | H3-h dropped if h ≤ x within benchmarks |
| D4 | **Value target quality.** Predictability of Δ_t = c_4 − c_t (accuracy gain of running on) within benchmark. | informs section 4 only |

D2 uses train out-of-fold scores for thresholds so that validation stays an honest development estimate (Gate 2 used
validation for thresholds, which is part of why its validation numbers were optimistic).

## 4. Candidate policies (to be frozen as a list before training)
- **π1 task-aware probe:** Q score, per-benchmark thresholds. Nests B.
- **π2 task-aware hidden-state probe:** same with P's score.
- **π3 decision value:** at loop t, estimate the expected gain from continuing, V_t = P̂(c_4 = 1 | ·) − P̂(c_t = 1 | ·)
  (equivalently from WC/CW-to-final probes); halt at the first t with V_t ≤ λ_b, λ_b per benchmark. A cost-aware variant
  divides by the remaining loops (4 − t).
- **Baselines, all made task-aware for fairness:** B; per-benchmark fixed-k under the primary constraint; native gate
  with per-benchmark thresholds; fixed-3; Gate 2's frozen P and Q (pooled threshold) for continuity.
- **Controls:** labels shuffled within benchmark with the same per-benchmark thresholds (should collapse to ≈ B);
  benchmark-prior policy.

Per-benchmark thresholds multiply the selection choices (6 per policy). Thresholds are therefore chosen on out-of-fold
train predictions, not on the data that estimates performance.

## 5. Accuracy constraint (fix before any result)
Recommendation: **primary = each benchmark's loss ≤ 1 pp** (no benchmark may subsidise another); secondary = macro
≤ 1 pp, question-weighted pooled ≤ 1 pp. Under the per-benchmark constraint B becomes (3,3,3,4,3,2) with saving 0.25 on
validation, so the bar is lower but no longer bought by OpenBookQA. The reviewer may prefer macro as primary for continuity
with Gates 1-2; either way it is fixed in writing before step 0 runs.

## 6. Data budget (inventory required before freezing)
After Gate 2 confirmation, the unscored validation splits of ARC-C, ARC-E, MMLU and OpenBookQA are used. Phase 3 needs a
fresh, audited test set chosen *now*, before development. Candidates to audit (same contamination procedure as Gate 2
section 3a): the unscored remainder of HellaSwag and CommonsenseQA train; multiple-choice benchmarks not yet used with
Ouro (for example PIQA, WinoGrande, Social IQa); and, for external validity, a second looped model or a task where the
answer is built over several steps. Each costs one GPU extraction (the 4,869-question confirmation set took about 35 min on the RTX 5050). No candidate is
assumed clean until audited.

## 7. Gates
- **G3a (step 0):** D1 and D2 pass their stop rules, else Phase 3 ends with a written negative (no headroom / not reachable).
- **G3b (development):** policy list, constraint, thresholds and models frozen on train (+ out-of-fold) and validation.
- **G3c (one-time test):** frozen policies applied once to the fresh audited set; verdict table in the Gate 2 style
  (budget, effect vs the best task-aware baseline with paired-bootstrap CI, hidden-state benefit), written before running.

## 8. Not allowed
Using any Gate 2 confirmation outcome to choose anything; changing the constraint or margin after step 0; adding policies
after G3b; reporting a policy without its task-aware baseline; claiming generality beyond Ouro-1.4B and these benchmarks.

## 9. Open questions for the reviewer
1. Primary constraint: per-benchmark ≤ 1 pp (recommended) or macro ≤ 1 pp?
2. Stop-rule values for D1 (0.05) and D2 (0.02)?
3. Is B (task identity known at inference) an acceptable deployment assumption, or should a task-agnostic track also be
   kept with B as an upper reference?
4. Which fresh test source to audit first?
5. Margin for G3c (Gate 2 used 0.03)?
