# Exploratory study E1: why does Ouro's native gate beat the post-hoc probes? (plan rev 1; nothing run)

**Status: exploratory.** This is not part of Phase 3, which closed at G3a/D2 (`docs/PHASE3_REPORT.md`, fcc26c6). E1 makes
no confirmatory claim and selects no policy for any test. It produces a **diagnostic report, not a model**.

## Data and boundaries
- **Data.** Only data already seen: Phase 1 train plus DEV+ (20,250 questions; development) and Phase 1 validation (5,870;
  exposed). The same loaders and guards as G3a are used (`phase3_g3a.load_g3a`).
- **TEST.** The 8,517 sealed TEST questions are never read. The sealed-path guard, the DEV+ allow-list and the TEST id
  deny-list stay in force.
- **Nothing reruns Phase 3.** π3 and π4 are not trained. No Phase 3 threshold, rule or verdict is revisited.
- **Analysis list, fixed now.** Exploratory does not mean unconstrained. The analyses below are fixed before any of them is
  run, and **every one is reported**, whatever it shows. Anything added later is labelled "added after seeing E1 results".
- **Shared protocol.** Unless stated otherwise, every comparison uses the G3a protocol unchanged:
  - the same folds (`lh-fold-v1`) and the nested outer and inner folds;
  - the same per-benchmark rule (τ_b, m_b) and the same 51-point grid;
  - gross saving, the three-benchmark primary macro, and the six-benchmark macro reported beside it.
  So any difference comes from the factor being studied, not from the protocol.

## A fact that shapes the hypotheses
π1-x's features already include the native gate's statistic: Gate 2's cheap features contain `cdf`, Ouro's cumulative exit
probability. π1-x had the gate's signal and still lost to it (G3a: 0.346 vs 0.370 held-out). So "the gate has information
the probes lack" cannot be the whole story. Three candidate explanations, each testable on development data:
- **H-target.** π1 is trained to predict *stability* (S_t: the answer will not change again). The selection rule rewards
  *saving under an accuracy budget*. A stable but wrong answer, or an unstable answer that ends up correct either way, is
  mislabelled for that purpose.
- **H-scale.** π1 has a separately fitted logistic model per loop, each calibrated on its own, but the policy applies **one
  threshold τ_b to all loops**. The gate's CDF is cumulative and monotone in t, so a single threshold means the same thing
  at every loop. The per-loop probabilities need not.
- **H-info.** The gate carries information that the probes' training cannot use: for example, a nonlinearity the
  logistic model cannot represent, or the gate's joint training with the model. **Joint training cannot be tested in E1**
  without retraining Ouro. It stays a hypothesis, and E1 can only bound how much of the gap it could account for.

## Analyses (fixed list)
| id | question | method |
|---|---|---|
| **A1. Ablation ladder** | Where is the gap lost? | Same nested D2 protocol for: (a) native-pb; (b) **π1 on cdf alone** (per-loop logistic on the one feature); (c) π1-x **without** cdf; (d) π1-x; (e) π1-hx. If (b) already falls below (a), the loss comes from the probe pipeline (target or scale), not from the features **Paired question-level differences (rev 1):** for each rung against native-pb, on the same held-out questions, report the per-question difference in exit loop, the 2×2 table of correctness, and the paired-bootstrap 95% CI of the saving difference. |
| **A2. Target swap (H-target)** | Is stability the wrong target? | The same x features, models and frozen C, with the target replaced by **c_t** (correct now) and by **c_t ≥ c_4** (stopping now loses nothing against loop 4). Nested D2 for each **Prediction quality and stopping utility are reported separately (rev 1):** out-of-fold AUROC of each target is how well it is *predicted*; nested held-out saving and loss under the per-benchmark rule is its *stopping utility*. No target is assumed optimal, and a well-predicted target that gives less saving counts against it. |
| **A3. Scale fix (H-scale)** | Does one threshold across loops hurt the per-loop probes? | (i) Make π1-x cumulative: s_t = max over u ≤ t of P̂_u, so it is monotone like the CDF. (ii) A per-loop rank transform, where each loop's probe score becomes its within-loop quantile on the selection rows. Nested D2 for each **No future information (rev 1):** s_t may use only loops 1..t of the same question. The rank map for loop t is fitted on loop t's selection-row scores only, and is applied unchanged to held-out rows. A test asserts that s_t does not change when everything at loops > t is altered. The rank transform is monotone within a loop, so it keeps each loop's ordering and changes only how scores compare across loops. That makes it a scale diagnostic, not a new score. |
| **A4. Ranking quality** | Do the probes rank questions worse, or only threshold them worse? | Out of fold, within benchmark and loop: AUROC of native cdf vs π1-x vs π1-hx for **S_t**, for **c_t**, and for **"stopping now loses nothing"**. Calibration (reliability curves, ECE) of each score for its own target |
| **A5. Disagreement** | Are the errors complementary? | At each method's selected nested thresholds, cross-tabulate the per-question exit decisions of native-pb and π1-x. On the questions where they disagree, report which one saved loops, and which one lost accuracy. A simple **max(cdf, π1-x score)** combination goes through the same nested protocol, as information only |
| **A6. Budget behaviour** | Does the native gate win because it breaks the budget less, or more? | For each method and fold: held-out per-benchmark loss next to saving, as in G3a observation 2. Then the same comparison with a **risk-controlled selection applied identically to all methods**: per benchmark, the bootstrap upper bound of the loss must be ≤ 1 pp, with 2,000 resamples and seed 0, as in Gate 2. Risk control is applied to the baselines as well as the probes **Identical treatment (rev 1):** one selection function applied to every method: native-pb, B_pb, the seven readouts and every probe variant. Held-out violations (the number of folds and benchmarks above 1 pp, and the worst loss) are reported for both selections, point estimate and risk-controlled. |
| **A7. Exposed check** | Do the main A1-A3 contrasts hold on validation? | The validation estimate, selected on development data as in G3a. Reported only **Descriptive only (rev 1):** validation has already informed development (Gate 2), so A7 never chooses between analyses or hypotheses and never decides an interpretation. |

## How the results are read (fixed now, interpretive only)
- **(b) close to (a), and (d) below (b).** Adding features hurts: an overfitting or calibration problem in the probe
  pipeline. Next, A3 and A6.
- **(b) well below (a).** The logistic transformation of the gate's own statistic loses saving: a target or scale problem.
  If A2 or A3 closes most of the gap, the explanation is H-target or H-scale, not information.
- **A2 or A3 closes the gap to native-pb.** That points to a better *objective* or *decision scale*, not a better feature
  set, as the lever for any future method.
- **Nothing closes the gap, and A4 shows better ranking by the gate.** That is consistent with H-info. A future
  jointly-trained-gate study would then have a stated motivation, as a separate, preregistered study.
- **Native-pb also violates the budget heavily (A6), and its lead shrinks under risk control.** Part of its G3a advantage
  came from looser compliance.
- **No credible opportunity emerges.** The Phase 3 negative stands, and E1 is reported as such.

**No E1 result is a performance claim.** All of it is development-data evidence, chosen and evaluated on data that has been
seen before. A method suggested by E1 would need its own frozen protocol and its single confirmatory test on the sealed
TEST.

## Deliverable and cost
- **Report.** `docs/E1_REPORT.md` plus `results/exploratory/e1_report.json`, with every analysis above.
- **Cost.** CPU only. It reuses the G3a code (`phase3_g3a`, unchanged) through a separate module, `src/lookahead/e1.py`,
  with synthetic tests. Estimated runtime 20-60 min on the laptop.
- **Process.** Like G3a: code, then a reviewer check of the implementation, then the run, then the report.

## Revision 1: the reviewer's four safeguards and one recommendation (2026-10-26; direction approved, implementation on HOLD)
1. **A2:** the predictability of each target and its stopping utility are reported separately. No target is presumed
   optimal.
2. **A3:** no information from loops after t. The rank maps are fitted per loop on selection rows only, and a test checks
   both.
3. **A6:** one risk-controlled selection function, applied identically to the native gate, every baseline and every probe.
   Held-out violations are reported under both selections.
4. **A7:** descriptive only.
5. **A1:** adds paired question-level differences against native-pb, alongside the aggregate ladder.
