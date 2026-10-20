# Gate 2 plan (DRAFT for reviewer; nothing here has been run)

Status: draft. Gate 1 is frozen (tag `gate1-final`: validation freeze f5fb931, test evaluation a6590fa). No probe has been trained. All thresholds below marked PROPOSED need reviewer sign-off before any training starts.

## 1. Question Gate 2 answers
Can a probe on the loop-t hidden state h_t (last context token, 2048-d, stored for loops 1-4) make accuracy-aware stopping decisions that beat Ouro's native gate and fixed-3 at matched accuracy loss, once probe overhead is paid?

## 2. Data and what is off limits
- Train: `train` split (about 18.6k natural questions). Fit all probe parameters here only.
- Validation: model/feature selection, hyperparameters, threshold selection. Threshold selection uses the Gate 1 rule (50-quantile grid, loops 1-3, macro loss) for every policy, probe and baselines alike.
- Gate 1 test split: now visible. It may be reported as a "post-Gate-1 development check" but is NOT the final confirmation of any new policy and is never used to tune architecture, thresholds or losses.
- Fresh confirmation set (OPEN, reviewer to choose): extract a new sample from questions we have never scored, e.g. HF `train` splits of ARC, CSQA, OBQA, HellaSwag (few-shot exemplars come from those splits, so exclude the questions used as exemplars). Cost: about 4 h extraction per 33k questions on the laptop; a 6-8k-question set is about 1 h. Alternative: report the existing test as exploratory only and state the limitation.
- Synthetic tasks are excluded (unsolved by Ouro-1.4B). Limitation to state: no clear multi-step reasoning benchmark remains.

## 3. Probe targets (kept from the original plan)
For t in {1,2,3}: next-loop KL D_t (regress log D_t), answer flip F_t (classify), loss gain G_t (regress, clipped). Added because the reviewer separated "change" from "improvement":
- S_t = 1[answer at loop t equals the answer at every later loop through 4] (stable-final: stopping at t preserves the final answer). Classify.
- C_t = 1[answer at loop t is correct] (classify), used for diagnostics only.
Primary policy target: P(S_t). Stop at the first t whose probe score passes the threshold.

## 4. Probe models (all frozen to this list before training)
1. Logistic/ridge on h_t (loop-specific, one model per t).
2. Same with a shared model plus loop embedding.
3. Two-layer MLP (hidden 256), loop-specific.
Controls: (a) cheap-features-only model (confidence, entropy, native gate cdf, t, n_options, context length); (b) h_t + cheap features; (c) label-shuffled h_t. A probe only counts as using h_t if (b) beats (a) on validation.
Overhead: report probe FLOPs and measured ms per question next to the loop cost (a linear probe is about 2k multiply-adds against about 1.4B per token for a loop).

## 5. Policy comparison (PROPOSED)
- Baselines: fixed-3, native gate, confidence, entropy, deployable measured-KL, with the same selection rule.
- Threshold selection with a risk margin (new, applied to ALL policies incl. baselines): pick the max-saving grid point whose bootstrap 90th-percentile validation macro loss is at most 1 pp. Reason: Gate 1 showed that point-estimate qualification on validation did not hold on test (native gate 0.61 to 1.62 pp).
- Matched-loss comparison: for each policy report saving at validation-selected threshold and at loss budgets 0.5, 1.0, 2.0 pp.
- Statistics: paired bootstrap, 2,000 resamples, stratified by benchmark, seed 0; macro over six benchmarks, equal weight; OpenBookQA reported in the primary verdict and without it as sensitivity.
- Cross-benchmark generalisation: leave-one-benchmark-out probe as a secondary check.

## 6. Gate 2 pass rule (PROPOSED, reviewer to set before training)
PASS iff, on the confirmation set at the validation-selected threshold: (i) probe macro loss is within the 1 pp budget (or no worse than the native gate's loss), AND (ii) probe saving exceeds the best loss-compliant baseline's saving, with the lower end of the paired-bootstrap 95% CI of the difference above 0 and a point estimate of at least 3 pp of loop budget, AND (iii) the h_t probe beats the cheap-features-only control. Everything else is reported as a negative or mixed result, not tuned further.

## 7. Open questions for the reviewer
1. Fresh confirmation set (section 2): extract new questions, or accept the visible test as exploratory?
2. Risk-margin threshold selection (section 5): approve applying it to all policies?
3. Pass rule numbers in section 6 (3 pp margin, CI above 0, control requirement).
4. Whether to include the leave-one-benchmark-out check in the verdict or keep it secondary.

## 8. Order of work once approved
1. Data loader for train/val with hidden states (CPU, memory-mapped). 2. Controls and baselines on validation. 3. Probe training (minutes on the RTX 5050, mostly CPU-bound). 4. Freeze validation selections and commit before reading any confirmation outcome. 5. One-time confirmation run, same guard pattern as `gate1_test.py`.
