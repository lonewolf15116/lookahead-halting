# E1 report: why does Ouro's native gate beat the post-hoc probes? (EXPLORATORY; for reviewer audit)

**Status: exploratory. No confirmatory claim.** Phase 3 stays closed at G3a/D2 (`docs/PHASE3_REPORT.md`), and nothing here
reopens it. Every number comes from development data that had already been seen (Phase 1 train + DEV+, 20,250 questions) or
from the exposed Phase 1 validation set (5,870). The 8,517 sealed TEST questions were not read.

- **Run.** Laptop, commit 4a5d4cc (code clean, 87 tests passing first), 2.1 min.
- **Report.** `results/exploratory/e1_report.json` (sha256 0fa367b4076958f55e21ec5ae122135c56dd1663f7f679b7f34797254f7e7d35), committed unchanged.
- **Consistency with G3a.** native-pb 0.3701, π1-x 0.3465 and π1-hx 0.3211 reproduce G3a exactly.
- **Conventions.** All savings are gross, primary macro, held-out, nested over 5 outer folds, unless stated otherwise.

## Headline (exploratory)
1. **The native gate's lead depends on the selection rule.**
   - Under the frozen point-estimate rule, it leads every probe variant: π1-x −0.024.
   - Under the risk-controlled rule, the **same** rule for every method (A6: point loss ≤ 1 pp **and** bootstrap 95% upper
     bound ≤ 1 pp), the order **reverses**:

     | method | risk-rule saving | vs native-pb |
     |---|---|---|
     | native-pb | 0.204 | — |
     | π1-x | 0.258 | **+0.053** (all 5 folds positive: +0.037 to +0.083) |
     | π1-x@safe | 0.258 | +0.054 |
     | π1-x+cummax | 0.257 | +0.053 |

   - Held-out violations (folds × primary benchmarks above 1 pp) are 3 of 15 for native-pb and for π1-x under the risk rule.
     Under the point rule they are 6 and 7.
2. **The gate does not rank questions better (A4).** The out-of-fold AUROC of π1-x's score is **higher** than the native cdf's
   for every target (primary mean):

   | target | π1-x | native cdf |
   |---|---|---|
   | S_t | 0.689 | 0.603 |
   | c_t | 0.615 | 0.557 |
   | "stopping now loses nothing" | 0.658 | 0.582 |

   So the gate's point-rule win does not come from better ranking.
3. **Neither the target swap nor the scale fixes close the point-rule gap (A2, A3).**
   - c_t: −0.017. "safe": −0.022.
   - cummax: −0.024 (unchanged). Per-loop rank: −0.058 (worse).
   - In the plan's terms, **H-scale is not supported**, and **H-target is at most weakly supported** (c_t narrows the gap by
     0.007).
4. **The ablation (A1)** shows the probe pipeline loses about 0.01 even with the gate's own statistic alone:
   - **π1-cdf: −0.010**, paired 95% CI [−0.012, −0.008].
   - Adding the other cheap features loses more: π1-x −0.024.
   - Removing cdf loses more still: π1-x-nocdf −0.039. cdf is the most useful single feature.
   - Hidden states hurt: π1-hx −0.049.

## What this does and does not mean
- **It suggests** that, at G3a, part of native-pb's advantage came from the point-estimate selection rule, not from better
  question ranking. With the risk-controlled rule applied identically, the better-ranking probe saves more.
  - This fits the plan's pre-written reading: "native-pb also violates the budget heavily, and its lead shrinks under risk
    control → part of its G3a advantage came from looser compliance".
  - Here the lead does not just shrink; it reverses.
- **It does not establish a better method.** Six reasons:
  1. These are exploratory contrasts on already-seen development data. A6 was in the analysis list fixed before the run,
     but 18 methods × 2 rules were compared. The cleanest contrast is the pre-planned one: π1-x, the frozen G3a candidate,
     against native-pb under the risk rule.
  2. The risk rule cuts savings for everyone (native-pb 0.370 → 0.204; B_pb 0.25 → 0.08), and it still leaves 3 of 15
     held-out violations. It is **not** a compliance guarantee.
  3. Why the gate does so well under the point rule while ranking worse is **not** explained. One unverified possibility is
     selection at the 1 pp edge on a very skewed statistic: the chosen cdf thresholds range from 4e-5 to 0.09.
  4. The paired CIs are exploratory: dependence from overlapping training folds is not modelled (reviewer caution).
  5. A4 is descriptive, not causal.
  6. Joint training (H-info) remains untested.
- **The six-benchmark macro agrees in direction** under the risk rule: π1-x +0.024. The exposed validation set (A7, point rule
  only) agrees with the point-rule results: π1-x −0.029.

## Analysis-by-analysis (all reported, including null and uninformative ones)
| id | result |
|---|---|
| A1 ladder (point) | native 0.370; π1-cdf 0.361 (−0.010); π1-x 0.347 (−0.024); π1-x-nocdf 0.331 (−0.039); π1-hx 0.321 (−0.049). Paired: π1-x exits earlier than native on 1,048 questions and later on 1,490. Correctness differs on 252 of 9,693 primary held-out questions |
| A2 (point) | Prediction AUROC (own target, primary mean): S 0.689, safe 0.661, c_t 0.619. Stopping utility vs native: c_t −0.017, safe −0.022, S −0.024. The most predictable target is not the most useful one |
| A3 (point) | cummax −0.024 (no change); rank −0.058 (worse) |
| A4 | AUROC as in headline 2. ECE of the probes for S is 0.004-0.024 by loop, all benchmarks pooled (well calibrated) |
| A5 | On the questions where they disagree (2,538 of 9,693 primary): where π1-x exits earlier, it saves 1.02 loops and loses 2.0 pp; where it exits later, it spends 1.04 more loops and gains 2.2 pp. **combo-max is uninformative:** see below |
| A6 | Headline 1. Point rule, violations (of 15): native 6, π1-x 7, π1-hx 8, readouts 5-9; worst held-out loss up to 5.2 pp (π1-x-nocdf). Risk rule: 2-5 violations; worst up to 3.3 pp (r-dlogit). Among the readouts under the risk rule, only r-kl ≥ native (+0.003) |
| A7 (descriptive) | Validation vs native (point rule): π1-cdf −0.017, π1-x −0.029, π1-hx −0.049, c_t −0.020, safe −0.030, cummax −0.031, rank −0.067. The same direction as the development point-rule results |

## Two design defects, reported rather than fixed after the fact
1. **A5's combo-max is degenerate.** It takes the elementwise max of the native cdf and π1-x's probability, but the scales
   differ.
   - The native gate's selected thresholds are 4e-5 to 0.09; π1-x's are 0.84-0.91.
   - So max(cdf, π1-x) equals π1-x's score wherever a threshold could fire, and combo-max reproduces π1-x exactly (point
     −0.0236, risk +0.0533).
   - It says nothing about complementarity. A scale-aware combination would be a new analysis and would have to be
     labelled as added after seeing E1.
2. **The A6 risk rule is the combined point + bootstrap rule** that the reviewer accepted. It is not the literal
   upper-bound-only rule.

## What would follow (for discussion, not decided here)
- **E1's most specific finding** is the dependence on the selection rule. The point-estimate rule at a 1 pp edge appears to
  reward a statistic that ranks worse.
- **A future study** could take as its preregistered primary question "learned probe vs native gate under identical
  risk-controlled selection".
  - It would need its own frozen protocol.
  - It would need a stated reason the risk rule is the right deployment criterion.
  - Its single confirmatory test would be on the sealed TEST.
  - Prior work on risk control (Learn-then-Test, conformal risk control, CALM) sets the novelty bar for the selection
    procedure itself.
