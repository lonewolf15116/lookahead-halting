# Draft sections 3 to 6 (draft 1, from docs/PAPER_OUTLINE.md v2)

Status: first prose draft of sections 3 to 6. Every number is traced in "Number provenance" at the end. Open points are listed in "Notes to author", and two inline NOTE TO AUTHOR blocks sit in section 3.6.

**Evidence labels.** [C] confirmatory: the pre-registered primary endpoint, decided once on the sealed TEST set. [S] pre-specified secondary: listed in the frozen protocol before TEST was opened. [P] post hoc emphasis or interpretation, chosen or framed after seeing TEST. [E] exploratory or development-only: development data, including the G3a gate and E1.

**Number conventions.** Gross and net savings, and differences in savings, are fractions of the four-loop budget, reported to three decimals (four decimals for the r-kl comparison, where the difference is near zero). Accuracy losses and one-sided upper bounds are in percentage points (pp), to two decimals. Confidence intervals are written [lower, upper] at the precision of their estimate. Hashes are abbreviated as in the repository documents.

---

## 3. Setting: model, halting signals and selection rules

### 3.1 Model, scoring and decision unit

We use Ouro-1.4B [Zhu2025], frozen, in bf16, with its trained depth of 4 loops. No weight is changed. Every benchmark is multiple choice. Each answer option is scored by the summed log-likelihood of its continuation tokens, using packed scoring: the context and all options form one sequence, each option token may attend only to the context and to earlier tokens of its own option, and one forward pass per loop yields all option scores. The answer is the option with the highest score, using the length-normalised log-likelihood (the `acc_norm` metric) for OpenBookQA, HellaSwag and PIQA, and the raw log-likelihood (`acc`) for CommonsenseQA and Social IQa. OpenBookQA is evaluated 0-shot, CommonsenseQA and HellaSwag 10-shot, and the two out-of-distribution benchmarks 0-shot.

A halting policy makes one decision per question. After loop $t \in \{1,2,3\}$ it reads Ouro's exit-gate output, the hidden state and the full-vocabulary next-token entropy at the last context token, together with the option-level scores at loop $t$, and either stops or continues. Loop 4 is the reference depth. For a question that exits at loop $e \in \{1,2,3,4\}$ the answer is the loop-$e$ answer, and the accuracy loss of a policy on a benchmark is $\text{acc}(\text{loop }4) - \text{acc}(\text{policy})$, in percentage points.

### 3.2 Ouro's native gate and what its released implementation does

Ouro attaches an exit gate to each loop: $\lambda_t = \sigma(\text{Linear}(h_t))$. The exit distribution is $p_t = \lambda_t \prod_{j<t}(1-\lambda_j)$, with the last loop taking the remaining mass. Its cumulative sum is the CDF, and a Q-exit stops at the first loop whose CDF reaches a threshold $q$ [Zhu2025].

The released implementation (`modeling_ouro.py`, sha256 c5c68fbb…, in the pinned model revision) does not skip loops. Its forward pass runs all `total_ut_steps` loops, applies the gate (a `Linear(hidden, 1)`) at every token position, and, when an exit threshold is given, chooses for each token position the first loop whose cumulative exit probability reaches the threshold and then reads out that loop's hidden state through the language-model head. The released Q-exit therefore operates per token position and selects a readout; it saves no compute. The paper's text states the goal of exiting per token and per example but does not fix the inference unit.

Our use is an adaptation. We evaluate the gate's CDF at a single position, the last context token, which gives one decision per question, and we count the loops after the exit as saved. Two consequences follow, and we state them wherever savings are reported. First, our native-gate baseline uses Ouro's trained gate as a halting statistic inside our own protocol; it is not Ouro's released inference procedure. Second, our savings are properties of this adaptation. They are not measurements of the released inference.

### 3.3 Halting signals

All policies share one form. Each has a statistic $s_t$ for loops $t = 1,2,3$, oriented so that larger means "safer to stop", a threshold $\tau_b$ and a depth cap $m_b \in \{1,2,3,4\}$, both chosen per benchmark $b$:

$$e = \min\{\, t < m_b : s_t \ge \tau_b \,\} \quad \text{if this set is non-empty, and } e = m_b \text{ otherwise.}$$

A statistic that is undefined at loop 1 never fires there. Setting $\tau_b = +\infty$ gives fixed depth $m_b$. The suffix "-pb" marks per-benchmark selection of $(\tau_b, m_b)$.

**Native gate (native-pb).** $s_t = \text{CDF}_t = 1 - \prod_{j \le t}(1-\lambda_j)$, computed from Ouro's own gate logits at the last context token. It has no fitted component.

**Learned probe (π1-x).** A per-loop L2-regularised logistic regression, one model for each of loops 1, 2 and 3, with regularisation constant $C = 1.0$ (inherited from the Gate 2 configuration, not re-tuned) and standardised inputs. The input has nine columns:

- conf: $\max_k p_t(k)$, where $p_t$ is the softmax of the option scores;
- ent: the entropy of $p_t$;
- vent: the entropy of the full-vocabulary next-token distribution at the last context token;
- cdf: the native gate's CDF at loop $t$;
- n_options: the number of answer options;
- log_n_tokens: the logarithm of the packed sequence length (context plus all options);
- three loop indicators, which are constant within each per-loop model and so carry no information there.

The target is the stability label $S_t$: it equals 1 if the loop-$t$ predicted option is the same as the predicted option at every later loop, so that the answer never changes again. The label is an agreement criterion in the spirit of patience-based exits over layers [Zhou2020], used here as a training label for a probe over loops rather than as a patience counter. The score is $s_t = \hat{P}(S_t = 1)$. The probes were fitted on development data from all six development benchmarks. For selection, $s_t$ is the five-fold cross-fitted out-of-fold score over all development data; for TEST, the final models were refitted on all development data and saved.

**Untrained readouts (r-\*).** Seven rules computed from the stored per-loop option scores $z_t$ (the scores whose softmax is $p_t$) or hidden states $h_t$ at the last context token, following Popescu et al. [Popescu2026] and Ouro's hidden-state rule [Zhu2025]:

| Rule | Statistic $s_t$ | Loops with $s_t$ defined |
|---|---|---|
| r-conf | $\max_k p_t(k)$ | 1–3 |
| r-margin | $z_t^{(1)} - z_t^{(2)}$, the gap between the top two option scores | 1–3 |
| r-ent | $-H(p_t)$ | 1–3 |
| r-kl | $-\mathrm{KL}(p_t \,\|\, p_{t-1})$ | 2–3 |
| r-dlogit | $-\lVert z_t - z_{t-1} \rVert_2$ | 2–3 |
| r-hl2 | $-\lVert h_t - h_{t-1} \rVert_2$ | 2–3 |
| r-hcos | $\cos(h_t, h_{t-1})$ | 2–3 |

**Fixed depth (B_pb).** $\tau_b = +\infty$, with the depth $m_b$ chosen per benchmark by the same rule as the adaptive policies. It is a reference, not the primary comparator.

### 3.4 Selection of $(\tau_b, m_b)$

Selection uses development data only, separately for each benchmark. The candidate set for a statistic is the product of a threshold grid and $m_b \in \{1,2,3,4\}$. The grid holds up to 50 distinct quantile values of the statistic, pooled over the usable questions of all development benchmarks and the loops where it is defined, plus $+\infty$. This gives up to 51 thresholds and 204 candidates per benchmark and policy. Two rules decide which candidates qualify.

- **Point-only rule** (the G3a rule). A candidate qualifies if its point accuracy loss on the selection data is at most 1 pp, with a numerical tolerance of 1e-12.
- **Bootstrap-screened rule** (called the "combined risk rule" in the C1 protocol, and implemented as `select_risk` in the E1 code). A candidate qualifies if its point loss is at most 1 pp and the one-sided 95% bootstrap upper bound of its loss on the selection data is also at most 1 pp. The bound is the 95th percentile of the loss over 2,000 resamples of the selection questions of that benchmark, with a fixed random seed per benchmark and the same resampled draws for every policy.

Among qualifying candidates the rule takes the maximum gross saving; ties within 1e-12 are broken by smaller point loss, then by preferring fixed depth ($\tau = +\infty$), then by larger $\tau$, then by smaller $m$. Fixed depth 4 has zero loss and always qualifies.

The bootstrap-screened rule is a heuristic screen. It carries no formal guarantee of risk control, and we do not describe it as risk control. Prior work selects early-exit thresholds with finite-sample statements: Fast yet Safe uses a high-probability upper confidence bound on a performance gap under i.i.d. calibration data [Jazbec2024]; Learn then Test frames threshold selection as multiple testing with family-wise error control [Angelopoulos2025LTT]; conformal risk control bounds the expected value of a monotone loss [Angelopoulos2024CRC]. Our rule differs in four ways: the bootstrap is asymptotic, not a finite-sample bound; there is no multiplicity correction across the candidates; selection and bound use the same data, so the selected candidate's bound is optimistic; and no $(\alpha, \delta)$ statement is made. The bootstrap also treats the cross-fitted scores as fixed, so it ignores the variability from fitting the probe. The held-out record is consistent with a screen that does not control held-out risk: in the exploratory study E1 the screened rule still left 3 of 15 held-out benchmark-by-fold cells above 1 pp for both native-pb and π1-x [E].

### 3.5 Gross and net saving

For a question exiting at loop $e$, the gross saving is $1 - e/4$. A benchmark's gross saving is the mean over its questions, and the macro saving is the unweighted mean over the three primary benchmarks.

The net saving charges π1-x for the probe's extra work. The native gate's CDF comes from the model's exit head, which is shared and counted for no policy, so native-pb and fixed depth have $\rho = 0$. For π1-x two overheads are charged in units of one loop's time: $\rho_{\text{score}}$, the cost of scoring the options at a loop where the controller evaluates but does not stop, and $\rho_{\text{rest}}$, the cost of the vocabulary entropy, the host transfer, the nine features and the probe, charged at every evaluation. Scoring at the halting loop is not charged, because it is the scoring that produces the answer and every controller needs it. For a question that halts by its score at loop $e < m_b$, the evaluations number $e$ and the extra scorings number $e-1$; for a question that reaches the cap, both number $m_b - 1$. The net saving of a question is

$$1 - \frac{e + \rho_{\text{score}}\, n_{\text{score}} + \rho_{\text{rest}}\, n_{\text{eval}}}{4}.$$

The overheads $\rho_{\text{score}}$ and $\rho_{\text{rest}}$ were measured once on the GPU at freeze time, on 200 development questions per primary benchmark at batch size 1, using extraction's own code path (Table 1). The net saving is an estimated deployment cost under this accounting boundary. It is not a wall-clock measurement.

| Benchmark | $\rho_{\text{score}}$ | $\rho_{\text{rest}}$ | $\rho_{\text{total}}$ | One loop (ms) |
|---|---|---|---|---|
| OpenBookQA | 0.036 | 0.014 | 0.050 | 20.7 |
| CommonsenseQA | 0.010 | 0.004 | 0.014 | 79.9 |
| HellaSwag | 0.013 | 0.003 | 0.015 | 148.8 |

*Table 1. Freeze-time overhead of the probe, as a fraction of one loop's time. Measured on one GPU, batch size 1, 200 development questions per benchmark.*

### 3.6 Retrospective evaluation

All stopping policies in this paper were evaluated retrospectively, offline, from full-depth per-loop traces. They were not executed with truncated computation. The extraction stage ran every question through all four loops once and stored, for each loop, the option scores and distributions, the prediction and its correctness, the gate logit and CDF, the vocabulary entropy and the hidden state at the last context token. The evaluation code then applies a policy to those stored arrays: it finds the exit loop from the stored statistics and reads the correctness at that loop from the stored correctness array. No halting policy was run through the model.

Accuracy and exit-loop outcomes obtained this way are exact for a truncated run, provided that every quantity a policy reads at loop $t$ depends only on loops up to $t$. The code satisfies this. The loop runner repeats the layer stack on a single packed sequence, the input to loop $t$ is the normalised output of loop $t-1$, attention at loop $t$ uses keys and values computed from that loop's own input (the runner keeps no cache across loops), and the gate CDF at loop $t$ is a function of the gate logits at loops up to $t$. The statistics used by the policies at loop $t$ (option scores, $p_t$, entropies, probe features, the divergence from $p_{t-1}$, hidden-state differences) are functions of loops up to $t$. The stability label $S_t$ looks at later loops, but only as a training label on development data; it is never an input at decision time.

Compute savings are of a different kind. They are simulated from exit-loop counts, $1 - e/4$, and the probe overhead $\rho$ was measured separately at freeze time, in isolation, on development questions (Table 1). They are therefore an estimate of the savings a truncated implementation could realise, under the assumption that a skipped loop costs what a measured loop costs. They are not observed end-to-end savings, and no wall-clock time of a halting policy was measured.

> **NOTE TO AUTHOR (retrospective exactness).** The statement that accuracy and exit loops are exact under truncation rests on reading the code (`ouro_adapter.py`, `packed.py`, `extract.py`, `c1_test.py`, and the released `modeling_ouro.py`), not on an experiment. Three qualifications belong in the text or the limitations. (i) No test in the repository runs a truncated policy and compares it with the stored traces; the adapter test in `tests/` compares the loop-by-loop runner with the stock forward pass on a tiny random model. I did not run it. (ii) The stored values come from the packed pass in bf16. The repository documents record a bf16 difference between the packed pass and a context-only pass on the 0-shot sets (relative L2 of the last-context state 0.030 to 0.036 for OpenBookQA, PIQA and Social IQa, and 0 on CommonsenseQA and HellaSwag), so a deployment that decided from a context-only pass before scoring options could behave slightly differently from the stored traces. The Phase 3 check D-b, which was to measure the agreement of halting decisions between numerical paths, was to run only if G3a allowed Phase 3 to continue; I found no D-b result in the documents. The sensitivity of halting decisions to this numerical path is therefore unmeasured. (iii) Same-input re-runs of the full four-loop extraction reproduced the stored states and gates exactly (difference 0), which supports determinism of the full-depth pass but is not evidence about a truncated pass. Also, `docs/PROTOCOL.md` says that exiting per question "saves real prefill compute"; that is a design rationale, and the paper should not repeat it as a result.

> **NOTE TO AUTHOR (what the code does not contradict).** I found nothing in the code that contradicts the retrospective description. `ouro_adapter.OuroLoopRunner` was written so that a policy could genuinely skip loops, and `PackedScorer.loops` accepts a loop count, but no C1 code path uses either to execute a halting policy: `extract.extract_question` runs all loops, `c1_rho.measure_rho` times loops 1 to 3 on fixed questions without any halting, and `c1_test.apply_policy` reads stored arrays. The overhead $\rho$ excludes the exit head and CDF for every policy, so even native-pb's controller cost is taken as zero by definition.

---

## 4. Experimental design and pre-registration

### 4.1 Data

Development data are the Phase 1 training questions (17,615) and the Phase 3 development allocation, DEV+ (2,635), together 20,250 questions across six benchmarks: OpenBookQA, CommonsenseQA, HellaSwag, ARC-Challenge, ARC-Easy and MMLU. They were used to fit the probes, select every $(\tau_b, m_b)$, measure $\rho$ and freeze the study. A further 5,870 Phase 1 validation questions were exposed earlier and were used for a report-only estimate in G3a; the C1 protocol excludes them from every choice (the protocol's reviewer decision D-a, "development data only").

The TEST set is a sealed allocation of 8,517 questions, disjoint from the development data by a salted hash and fixed in the manifest (sha256 b2df1930…). The primary in-distribution benchmarks are OpenBookQA (3,000), CommonsenseQA (1,000) and HellaSwag (750), 4,750 questions in all. The out-of-distribution benchmarks, PIQA (1,837) and Social IQa (1,930), are reported separately and never pooled. ARC and MMLU have no fresh rows and are not part of any claim. Every allocated row was removed from the few-shot exemplar pool, and the out-of-distribution training splits were never used.

| Set | Questions | Role |
|---|---|---|
| Development (Phase 1 train + DEV+) | 17,615 + 2,635 = 20,250 | fitting, selection, freeze |
| TEST, OpenBookQA | 3,000 | primary |
| TEST, CommonsenseQA | 1,000 | primary |
| TEST, HellaSwag | 750 | primary |
| TEST, PIQA | 1,837 | out of distribution |
| TEST, Social IQa | 1,930 | out of distribution |
| TEST, total | 8,517 | one-time confirmatory run |

*Table 2. Data split.*

### 4.2 Sequence of studies

The studies were run in a fixed order, and each later one was written after the earlier one was closed.

1. **G3a.** The pre-registered development gate of Phase 3 (plan v1.1) compared π1-x and a hidden-state variant with the strongest baseline under the point-only rule, using nested cross-validation on development data. It stopped at rule D2, because neither candidate improved on native-pb by the required 0.02. The decision was computed by the frozen code and confirmed by the reviewer.
2. **E1 [E].** An exploratory analysis on development data asked why the native gate won. Its main finding was that the ordering depended on the selection rule: under the point-only rule π1-x trailed native-pb (−0.024), whereas under the bootstrap-screened rule applied identically to both it led (+0.053, positive in all five folds).
3. **C1 protocol.** E1 motivated a new, separately pre-registered study, C1, whose single confirmatory test is on the sealed TEST set. The protocol was reviewed (conditional approval, decisions D-a to D-e), then frozen (sha256 4f38b8b7…, tag `c1-protocol`).
4. **Freeze on development data.** One script fitted π1-x, ran the selections for both policies, the fixed-depth reference and the pooled out-of-distribution selection, measured $\rho$, and wrote `frozen.json` (65728e16…) and the model file (72f15508…).
5. **One-time guarded TEST run.** The run applied the frozen selections once to the sealed TEST set.

The guard around the TEST run is part of the methodological rigour of the study. We do not present it as a contribution. Before reading any outcome, the runner verified the sha256 of the protocol, the frozen selections, the model file, the manifest and the TEST extraction check; verified the code files recorded at freeze; verified that the 24-file code bundle matched the approved hash (c0902af3…); refused to start if the marker, the report or a failure record already existed; and checked the TEST extraction's metadata (identifier lists, counts, file checksums) without reading any prediction or label. It then created the marker `TEST_OPENED.json` by exclusive creation, so that it could never be overwritten, and only then read the TEST extraction, restricted to the TEST directory. The report was written by exclusive creation. A second invocation after the successful run was refused at preflight, because the marker existed, and TEST was not read again.

### 4.3 Frozen decision rule, quoted from the protocol

The endpoint and the budget are quoted from `docs/C1_PROTOCOL.md` (sha256 4f38b8b7…), sections C1.3 to C1.5 (adjacent bullets are joined into paragraphs; the wording is unchanged). The hypothesis statement in C1.1 is paraphrased, not quoted, below.

> **Primary.** Δ = S_π1-x − S_native, the paired difference in **gross** loop saving, macro-averaged with equal weight over the three primary benchmarks on TEST. **Minimum meaningful improvement δ = 0.02** (2 percentage points of loops; the same size as Phase 3's D2 screen).
>
> **Budget, per primary benchmark.** loss_b = acc_b(loop 4) − acc_b(policy) ≤ 1.00 pp, point estimate on TEST, with the Gate 1 tolerance of 1e-12. **π1-x must meet the budget on all three benchmarks.** If it does not, the result is **FAIL (budget)**, whatever Δ is. A saving advantage alone never counts as success.
>
> **Interval.** Paired bootstrap over TEST questions, resampled within each benchmark, so each benchmark keeps its own n and the macro stays equal-weight. B = 10,000, seed `default_rng(2026)` (**D-d**). Two-sided 95% percentile interval [L, U] for Δ.
>
> **Verdict (exhaustive; the first matching row applies):**
>
> | condition | verdict |
> |---|---|
> | π1-x loss > 1 pp on any primary benchmark | **FAIL (budget)** |
> | Δ ≥ δ **and** L > 0 | **PASS**: a statistically supported positive improvement (L > 0) with an observed effect of at least δ = 0.02 (point estimate) |
> | U < δ | **FAIL (no meaningful advantage)**: improvements of size δ or more are ruled out |
> | otherwise | **INCONCLUSIVE**: Δ and [L, U] reported, no claim either way |
>
> **What PASS does and does not claim (reviewer clarification).** PASS establishes, at 95% confidence, that the improvement is **positive**, and that the **observed** effect is at least 0.02. It does **not** establish with 95% confidence that the true improvement is at least 0.02; that would need L > 0.02, which is not the criterion. Reports must keep the two apart. Whether L > 0.02 also holds is reported as a secondary fact.
>
> **One primary test.** Everything else is secondary and never changes the verdict.

In words: the comparison is between π1-x and native-pb, each selected on development data under the same bootstrap-screened rule. The protocol also fixed, before TEST was opened, the secondary analyses reported in sections 5 and 6: the same comparison under the point-only rule; π1-x against every other baseline, with native-pb as the sole primary comparator; the net saving; the out-of-distribution benchmarks, each on its own, with one task-agnostic $(\tau, m)$ per policy selected under the same rule over all six development benchmarks; and a table of held-out losses for every policy. The protocol also stated before the test that demonstrating a loss of at most 1 pp would be out of reach for CommonsenseQA (n = 1,000) and HellaSwag (n = 750), even for a policy that loses nothing: with a true loss of zero and 5% to 10% of questions changing correctness, the expected one-sided upper bound is 1.16 to 1.65 pp for CommonsenseQA and 1.34 to 1.90 pp for HellaSwag (planning assumptions from amendment A1.5). The expected half-width of the 95% interval for Δ at the TEST sizes was computed at freeze from development data as 0.0043.

### 4.4 Deviations and disclosures

- **Phase 3 numerical diagnostic D-a (PIQA).** (The label "D-a" here is the Phase 3 diagnostic, not the C1 reviewer decision of the same name.) Before the full extraction, a float32 check compared the packed pass with a context-only pass on 100 never-used PIQA training rows and 100 Social IQa rows, against a tolerance of 1e-4. Social IQa passed (maximum difference 3.1e-5). PIQA failed (1.17e-4). The tolerance was not changed and the failure stands as recorded. A follow-up, D-a2, repeated the comparison in float64 on the CPU reference path for the 10 PIQA questions with the largest float32 difference, with a pre-fixed criterion of at most 1e-9; the maximum difference was 1.67e-13, and the result was classified numerical. The reviewer accepted a documented PIQA deviation and released the extraction. A procedural deviation within D-a2 (a one-line fix was run before reviewer confirmation) is recorded in the decision log.
- **Console transcript.** After the successful TEST run, the runner was started a second time by mistake and was refused at preflight, before the marker step. Its console transcript overwrote the transcript file of the successful run, so the file holds only the refused attempt. The console output of the successful run survives as a paste in the working conversation. The run itself is fully recorded in `test_report.json` and the marker.
- **Smaller disclosures.** One test, the symlink-escape test for unsealing, was skipped on the Windows machine that ran the TEST script (it passes on Linux). The package snapshots before and after the run differ in one line, the editable install of this repository (the commit before and after the pull of the approved commit); all third-party versions are identical. `frozen.json` records code hashes but not a git commit.
- **Not run.** The Phase 3 check D-b, on the sensitivity of halting decisions to the bf16 numerical path, has no recorded result (see the note in section 3.6).

---

## 5. Results: RQ1, the preferred method under two selection rules

RQ1 asks whether the accuracy-risk selection rule changes which halting method is preferred. Each rule selects its own operating points $(\tau_b, m_b)$ for each method, so the comparison is between rule-dependent selections, not between fixed policies whose order flips.

### 5.1 Primary and secondary comparisons

| Rule and data | π1-x − native-pb (gross macro saving) | 95% CI | Evidence |
|---|---|---|---|
| Point-only, G3a nested held-out folds on development data | −0.024 | per-fold range −0.044 to −0.014 | G3a STOP, [E] |
| Point-only, G3a exposed Phase 1 validation | −0.029 | none computed | [E] |
| Point-only, TEST | −0.022 | [−0.026, −0.018] | [S] |
| **Bootstrap-screened, TEST** | **+0.053** | **[+0.049, +0.057]** | **[C], verdict PASS** |
| Bootstrap-screened, TEST, net of estimated probe overhead | +0.038 | [+0.033, +0.042] | [S], estimated cost |

*Table 3. π1-x minus native-pb under the two selection rules. The first two rows are development-data results; the last three are on the sealed TEST set. The G3a value to four decimals is −0.0236.*

Under the bootstrap-screened rule, π1-x stayed within the point budget on every primary benchmark (point losses 0.17, 0.60 and 0.40 pp for OpenBookQA, CommonsenseQA and HellaSwag), the observed difference was at least $\delta = 0.02$, and the lower bound was above 0, so the verdict was PASS [C]. The macro saving was 0.239 for π1-x and 0.186 for native-pb. Here the lower bound (0.049) also exceeded $\delta$, which the protocol records as a secondary fact [S]. As the protocol states, PASS is not a 95% claim that the true effect is at least 0.02. The realised half-width of the interval, 0.0042, matched the 0.0043 expected at freeze [S]. The result is limited to Ouro-1.4B with 4 loops, to these three benchmarks, to gross saving and net saving as estimated here, and to this selection rule.

Under the point-only rule, the order is the opposite: π1-x saves 0.022 less than native-pb on TEST [S], replicating on sealed data the direction found in G3a and E1 on development data [E]. At G3a, π1-x improved on the comparator by −0.0236 on nested held-out folds; this was below the required +0.02, and Phase 3 stopped. In the point-only TEST selection, both policies exceed the 1 pp point budget on some benchmarks (see Table 5), so the point-only comparison on TEST compares policies that do not meet the budget. It is reported as a secondary replication of the rule dependence, not as a statement about compliant policies [S].

### 5.2 Per benchmark

| Benchmark | Method | $m_b$ | Gross saving | Loss (pp) | Upper bound (pp) | Exits at loop 1 / 2 / 3 / 4 |
|---|---|---|---|---|---|---|
| OpenBookQA (3,000) | π1-x | 4 | 0.157 | 0.17 | 0.53 | 6 / 243 / 1,377 / 1,374 |
| OpenBookQA | native-pb | 4 | 0.136 | 0.20 | 0.67 | 0 / 26 / 1,579 / 1,395 |
| OpenBookQA | B_pb | 4 | 0.000 | 0.00 | 0.00 | 0 / 0 / 0 / 3,000 |
| CommonsenseQA (1,000) | π1-x | 4 | 0.282 | 0.60 | **1.10** | 8 / 349 / 405 / 238 |
| CommonsenseQA | native-pb | 4 | 0.161 | −0.10 | 0.30 | 0 / 1 / 643 / 356 |
| CommonsenseQA | B_pb | 4 | 0.000 | 0.00 | 0.00 | 0 / 0 / 0 / 1,000 |
| HellaSwag (750) | π1-x | 3 | 0.279 | 0.40 | **1.60** | 0 / 88 / 662 / 0 |
| HellaSwag | native-pb | 3 | 0.262 | 0.13 | **1.33** | 0 / 36 / 714 / 0 |
| HellaSwag | B_pb | 3 | 0.250 | 0.53 | **1.60** | 0 / 0 / 750 / 0 |

*Table 4. TEST results under the bootstrap-screened rule, per primary benchmark [S, descriptive]. Savings are gross. The upper bound is the one-sided 95% paired-bootstrap bound of the loss on TEST. Bold marks bounds above 1 pp.*

The macro advantage is not spread evenly. Per benchmark, π1-x saved more than native-pb by 0.021 on OpenBookQA, 0.121 on CommonsenseQA and 0.017 on HellaSwag (differences of the entries in Table 4; no interval was computed for these individual differences) [P]. The difference is therefore dominated by CommonsenseQA. On OpenBookQA both adaptive policies split between loops 3 and 4. On HellaSwag, where the cap is 3, most questions exit at loop 3 (662 of 750 for π1-x, 714 of 750 for native-pb). On CommonsenseQA π1-x exits at loop 2 for 349 of 1,000 questions, where native-pb almost never does (1 of 1,000) [S].

### 5.3 Upper bounds above 1 pp on TEST

The selection rule required each selected candidate's bootstrap upper bound to be at most 1 pp on development data. On TEST this did not hold everywhere. All nine point losses are within the 1 pp budget, but four of the nine one-sided upper bounds exceed 1 pp [S]:

- π1-x: CommonsenseQA 1.10 pp and HellaSwag 1.60 pp (OpenBookQA 0.53 pp is below);
- native-pb: HellaSwag 1.33 pp;
- B_pb: HellaSwag 1.60 pp.

At selection time the bounds of the frozen candidates were between 0.75 and 0.98 pp for the two adaptive methods. HellaSwag, with $n = 750$, exceeds 1 pp for every method including fixed depth, so the small sample is part of the cause, as the protocol anticipated (section 4.3). The exceedance for π1-x on CommonsenseQA is specific to π1-x [S]. We interpret the pattern as what one expects from a heuristic screen: the bound used for selection is asymptotic, uncorrected for the number of candidates, computed on the data that chose the candidate, and blind to the variability of the fitted probe (section 3.4). This interpretation is [P], and no analysis here isolates the cause. The practical reading is that the point budget was met on TEST and the demonstrated, upper-bound budget was not met everywhere. The PASS verdict rests on the point budget, as pre-registered.

| Rule | Method | OpenBookQA saving / loss (pp) | CommonsenseQA saving / loss (pp) | HellaSwag saving / loss (pp) | Macro saving |
|---|---|---|---|---|---|
| Point-only | π1-x | 0.305 / **1.13** | 0.357 / **1.10** | 0.351 / **1.07** | 0.337 |
| Point-only | native-pb | 0.403 / **1.57** | 0.359 / **1.20** | 0.316 / 0.67 | 0.359 |
| Bootstrap-screened | π1-x | 0.157 / 0.17 | 0.282 / 0.60 | 0.279 / 0.40 | 0.239 |
| Bootstrap-screened | native-pb | 0.136 / 0.20 | 0.161 / −0.10 | 0.262 / 0.13 | 0.186 |

*Table 5. TEST operating points selected by each rule [S]. Bold marks point losses above 1 pp. The macro savings under the point-only rule are means of the per-benchmark entries.*

Under the point-only rule on TEST, π1-x exceeds 1 pp on all three benchmarks (1.07 to 1.13 pp) and native-pb on two (OpenBookQA 1.57 pp and CommonsenseQA 1.20 pp) [S].

### 5.4 Rule-dependent preference

Which method is preferred depends on which selection rule is applied, and the two rules select different operating points. We do not claim that a fixed pair of policies reverses its order. Table 5 shows what changes. Moving from the point-only rule to the bootstrap-screened rule, each method retreats to a lower-saving operating point, and native-pb retreats further: its macro saving falls from 0.359 to 0.186, a drop of 0.173, against a drop from 0.337 to 0.239, or 0.098, for π1-x [P, differences of the Table 5 macro entries]. The largest single change is native-pb on OpenBookQA, which moves from 0.403 at 1.57 pp loss to 0.136 at 0.20 pp loss.

The mechanism is not established by our data. One candidate explanation, which we flag as [P] and as untested, is that the native CDF is a skewed statistic with little resolution near the point where the screen stops accepting thresholds, whereas π1-x's probability is graded more finely there. The only supporting evidence is exploratory: on development data the out-of-fold AUROC of π1-x's score for the stability label exceeded that of the native CDF (0.689 against 0.603), and in the E1 report the point-only thresholds selected for the native CDF were extremely small for some benchmarks [E]. Neither observation shows that a better ranking is what produces the larger saving under the screened rule; E1 itself records that the ranking quality does not explain the point-only win of the native gate. Neither rule compares the methods at matched loss, and no matched-loss comparison was pre-registered.

> **Figure 1 (placeholder).** `figures/fig1_rule_dependent_operating_points.png`, to be produced from `results/c1/test_report.json` (`per_benchmark` and `secondary_point_rule.per_benchmark`).
>
> **Caption (exact text).** *Figure 1. TEST operating points selected by each rule, one panel per primary benchmark (OpenBookQA, CommonsenseQA, HellaSwag). Horizontal axis: accuracy loss relative to loop 4, in percentage points. Vertical axis: gross loop saving, as a fraction of the four-loop budget. Circles: π1-x. Squares: native-pb. Filled markers: bootstrap-screened rule. Open markers: point-only rule. The dashed vertical line marks the 1 pp budget. Each marker is one frozen $(\tau_b, m_b)$ selected on development data; the figure shows four operating points per panel and is not a saving-loss frontier, so it does not support a comparison at matched loss. Macro-averaged over the three benchmarks, the screened-rule difference π1-x − native-pb is +0.053 [+0.049, +0.057] [C] and the point-only difference is −0.022 [−0.026, −0.018] [S].*

### 5.5 Net saving

Net of the estimated probe overhead, π1-x saves 0.038 more than native-pb, 95% CI [+0.033, +0.042] [S, estimated cost]. The per-benchmark net savings of π1-x are 0.125 (OpenBookQA), 0.274 (CommonsenseQA) and 0.272 (HellaSwag), against gross savings of 0.157, 0.282 and 0.279; the overhead costs 0.031 on OpenBookQA, where loops are short and the overhead relative to one loop is largest, and 0.007 on each of the other two (differences of the stored values) [S]. The overhead reduces the advantage but does not remove it. This is an accounting estimate, not a wall-clock measurement, and native-pb is credited with zero controller cost by definition (section 3.5).

---

## 6. Results: RQ2 and RQ3

### 6.1 RQ2: against the native gate and against untrained baselines

RQ2 asks whether an added trained probe beats (a) Ouro's trained native gate and (b) a training-free exit based on a divergence between successive loops. For (a) the primary comparison in section 5.1 is the answer [C]: under the bootstrap-screened rule, π1-x saved more than native-pb. For (b), and for every other baseline, the comparisons were pre-specified as secondary [S]. The emphasis we place on the comparison with r-kl is post hoc [P], because r-kl was identified as the strongest baseline using TEST outcomes (below).

| Baseline (selected under the same rule) | Macro gross saving | Δ = π1-x − baseline | 95% CI |
|---|---|---|---|
| B_pb | 0.083 | +0.156 | [+0.151, +0.161] |
| native-pb (primary) | 0.186 | +0.053 | [+0.049, +0.057] |
| r-conf | 0.171 | +0.068 | [+0.063, +0.073] |
| r-margin | 0.170 | +0.069 | [+0.064, +0.075] |
| r-ent | 0.179 | +0.060 | [+0.055, +0.065] |
| r-dlogit | 0.162 | +0.077 | [+0.071, +0.083] |
| r-hcos | 0.174 | +0.065 | [+0.059, +0.072] |
| r-hl2 | 0.149 | +0.090 | [+0.084, +0.096] |
| **r-kl** | **0.238** | **+0.0012** | **[−0.0041, +0.0065]** |

*Table 6. TEST comparison of π1-x (macro saving 0.239) with every baseline, paired and benchmark-stratified bootstrap, 10,000 resamples [S]. The r-kl row is given to four decimals. All baselines were selected on development data under the bootstrap-screened rule.*

All the readouts other than r-kl trail π1-x by between 0.060 and 0.090, and fixed depth trails by 0.156 [S]. Every baseline's TEST point loss stayed within 1 pp (the largest was 0.93 pp, for r-dlogit on HellaSwag) [S]; upper bounds were computed only for π1-x, native-pb and B_pb.

**r-kl.** The difference between π1-x and r-kl is +0.0012, with a 95% interval of [−0.0041, +0.0065] [S]. The strongest baseline was determined on TEST: in a bootstrap that re-selected the strongest baseline in every resample, r-kl was selected in 10,000 of 10,000 resamples, and the re-selected interval is the same as r-kl's own. Because the comparator was chosen with TEST outcomes, this comparison was always secondary and never touched the verdict. The result is non-superiority: we did not demonstrate that the added trained probe is better than r-kl. It is not a demonstration that the two perform alike. No margin for declaring them interchangeable was pre-specified or tested, and the interval states only what differences are unlikely: a gain for π1-x of more than about 0.65 pp of saving, or a gain for r-kl of more than about 0.41 pp [S]. The prominence we give this comparison is post hoc [P].

The near-tie was visible before TEST. On the development data used for selection, the in-sample macro savings were 0.246 for π1-x and 0.243 for r-kl, against 0.193 for native-pb [E, in-sample, not a held-out estimate]. In the nested E1 estimate under the same rule, r-kl was 0.207 against 0.204 for native-pb [E]. On TEST, r-kl exceeded native-pb by 0.052 in macro saving (0.238 against 0.186; a difference of the stored macro values, with no interval computed) [P]. The supported claim is therefore that a stability-based halting signal, trained or not, saved more than Ouro's native gate under the screened rule, and not that the learned probe adds value over a simple stability statistic.

**Relation to the KL exit of Geiping et al.** The rule r-kl is close in spirit to the convergence exit of Geiping et al. [Geiping2025, section 6.1], which stops a recurrent-depth model when the KL divergence between successive steps falls below 5×10⁻⁴, once per generated token and with a fixed threshold. Our r-kl differs in the unit and in the threshold. The unit is the question, and the distribution compared is the distribution over the answer options, not over the vocabulary for a generated token. The threshold is not fixed: for each benchmark, a threshold and a cap are selected on development data under the screened rule. The frozen thresholds correspond to halting when $\mathrm{KL}(p_t \| p_{t-1})$ falls to or below 3.5×10⁻³ on OpenBookQA (cap 3), 3.7×10⁻² on CommonsenseQA (cap 4) and 2.7×10⁻⁴ on HellaSwag (cap 3) [S, frozen at freeze], so only one of the three is of the order of the Geiping et al. value. We therefore do not describe r-kl as a reproduction of that exit, and our comparison does not test it.

### 6.2 RQ3: out-of-distribution budget

RQ3 asks whether development-selected policies keep their accuracy budget on unfamiliar benchmarks. For PIQA and Social IQa, each policy used one task-agnostic pair $(\tau, m)$, selected on the development data under the same bootstrap-screened rule, qualifying only if every one of the six development benchmarks met the rule: π1-x with $m = 4$ and $\tau = 0.961$, native-pb with $m = 4$ and $\tau = 0.386$ (development macro saving 0.159 for each). Both benchmarks are evaluated 0-shot, which differs from the few-shot prompts of most development benchmarks and is part of the shift. The protocol listed this analysis as a secondary one with a verdict-style point-estimate table per benchmark, and never pooled across benchmarks [S].

| Benchmark (n) | Method | Gross saving | Loss (pp) | Upper bound (pp) | Exits at loop 1 / 2 / 3 / 4 |
|---|---|---|---|---|---|
| PIQA (1,837) | π1-x | 0.308 | 0.71 | **1.52** | 0 / 433 / 1,399 / 5 |
| PIQA | native-pb | 0.126 | 0.11 | 0.60 | 0 / 0 / 923 / 914 |
| PIQA | B_pb | 0.000 | 0.00 | 0.00 | 0 / 0 / 0 / 1,837 |
| Social IQa (1,930) | π1-x | 0.481 | **3.26** | **4.72** | 0 / 1,785 / 143 / 2 |
| Social IQa | native-pb | 0.028 | −0.05 | 0.16 | 0 / 0 / 214 / 1,716 |
| Social IQa | B_pb | 0.000 | 0.00 | 0.00 | 0 / 0 / 0 / 1,930 |

*Table 7. Out-of-distribution TEST results with one task-agnostic $(\tau, m)$ per policy [S]. Bold marks losses and upper bounds above 1 pp. The B_pb row is full depth.*

On PIQA, π1-x met the point budget (0.71 pp) but its upper bound (1.52 pp) was above 1 pp, and it saved 0.183 more than native-pb [S] (95% CI [+0.174, +0.191]). On Social IQa, π1-x did not stay within the budget: it lost 3.26 pp, with an upper bound of 4.72 pp, and it halted at loop 2 for 1,785 of 1,930 questions. Its large saving there (0.481) comes from halting early on most questions, and the accuracy cost is real: accuracy fell from 43.32% at loop 4 to 40.05% [S]. The saving difference against native-pb, +0.453 (95% CI [+0.449, +0.458]), should not be read as an advantage, because it is paid for by the budget violation. The native gate stayed within the budget on both benchmarks, at a much smaller saving (0.126 and 0.028).

We draw no general conclusion from this. It is one failure on one benchmark, with development-selected, task-agnostic thresholds, on two benchmarks that were also 0-shot and differ from the development data in several ways at once. The repository records a bf16 numerical difference between the packed and context-only passes on the 0-shot sets, and a documented PIQA numerical deviation (section 4.4), but no analysis here links either to the Social IQa result, and we do not offer an explanation. The finding is a reason to report an out-of-distribution check next to any development-selected halting policy, and it is not a statement about out-of-distribution robustness in general [S].

---

## Number provenance

Every numeric value in sections 3 to 6 is listed. "TEST" values come from `results/c1/test_report.json`, freeze values from `results/c1/frozen.json`, G3a values from `results/phase3/g3a_report.json`. Key paths are written with dots; `*` stands for each key. "Derived" means computed from the stated keys (differences, sums, means, counts, or conversion to pp) and was recomputed from the JSON files while drafting; every JSON-sourced row below was machine-checked against the number as written in the text (rounding: half to even on the stored binary value, three decimals for savings, two for pp). Rows marked "document" were read from the named document section and located by text search, not recomputed.

| Number as written | Where used | Source file | JSON key path or document section |
|---|---|---|---|
| 0.036 | Table 1 OBQA rho_score | `results/c1/frozen.json` | rho.openbookqa.rho_score |
| 0.014 | Table 1 OBQA rho_rest | `results/c1/frozen.json` | rho.openbookqa.rho_rest |
| 0.050 | Table 1 OBQA rho_total | `results/c1/frozen.json` | rho.openbookqa.rho_total |
| 20.7 | Table 1 OBQA loop ms | `results/c1/frozen.json` | rho.openbookqa.loop_ms |
| 0.010 | Table 1 CSQA rho_score | `results/c1/frozen.json` | rho.commonsense_qa.rho_score |
| 0.004 | Table 1 CSQA rho_rest | `results/c1/frozen.json` | rho.commonsense_qa.rho_rest |
| 0.014 | Table 1 CSQA rho_total | `results/c1/frozen.json` | rho.commonsense_qa.rho_total |
| 79.9 | Table 1 CSQA loop ms | `results/c1/frozen.json` | rho.commonsense_qa.loop_ms |
| 0.013 | Table 1 HS rho_score | `results/c1/frozen.json` | rho.hellaswag.rho_score |
| 0.003 | Table 1 HS rho_rest | `results/c1/frozen.json` | rho.hellaswag.rho_rest |
| 0.015 | Table 1 HS rho_total | `results/c1/frozen.json` | rho.hellaswag.rho_total |
| 148.8 | Table 1 HS loop ms | `results/c1/frozen.json` | rho.hellaswag.loop_ms |
| 200 | 3.5 / 4.2 n questions | `results/c1/frozen.json` | rho.openbookqa.n_questions (same for all three) |
| 2,000 | 3.4 bootstrap resamples | `results/c1/frozen.json` | bootstrap_selection.n |
| 17,615 | 4.1 / Table 2 | `results/phase3/g3a_report.json` | n_by_source.p1_train |
| 2,635 | 4.1 / Table 2 | `results/phase3/g3a_report.json` | n_by_source.devplus |
| 20,250 | 4.1 / Table 2 | `results/phase3/g3a_report.json` | sum of n.* (six benchmarks); also frozen.json n_dev sum |
| 5,870 | 4.1 | `docs/PHASE3_G3A_RESULT.md` | Data bullet ("5,870 Phase 1 validation questions") (document) |
| 3,000 | 4.1 / Table 2 / Tables 4,7 headers | `results/c1/test_report.json` | provenance.test_metadata.openbookqa.n (also data/manifests/phase3_questions.json sets.test/openbookqa.n_rows) |
| 1,000 | 4.1 / Table 2 / Tables 4,7 headers | `results/c1/test_report.json` | provenance.test_metadata.commonsense_qa.n (also data/manifests/phase3_questions.json sets.test/commonsense_qa.n_rows) |
| 750 | 4.1 / Table 2 / Tables 4,7 headers | `results/c1/test_report.json` | provenance.test_metadata.hellaswag.n (also data/manifests/phase3_questions.json sets.test/hellaswag.n_rows) |
| 1,837 | 4.1 / Table 2 / Tables 4,7 headers | `results/c1/test_report.json` | provenance.test_metadata.piqa.n (also data/manifests/phase3_questions.json sets.test/piqa.n_rows) |
| 1,930 | 4.1 / Table 2 / Tables 4,7 headers | `results/c1/test_report.json` | provenance.test_metadata.social_iqa.n (also data/manifests/phase3_questions.json sets.test/social_iqa.n_rows) |
| 8,517 | 4.1 / Table 2 | `data/manifests/phase3_questions.json` | sum of sets.test/*.n_rows; docs/C1_REPORT.md; results/phase3/allocation.json n_test_rows |
| 4,750 | 4.1 | `results/c1/test_report.json` | sum of n.openbookqa + n.commonsense_qa + n.hellaswag (derived) |
| b2df1930 | 4.1 manifest hash | `results/c1/TEST_OPENED.json` | pinned.data/manifests/phase3_questions.json |
| 4f38b8b7 | 4.2 protocol hash | `results/c1/TEST_OPENED.json` | pinned.docs/C1_PROTOCOL.md |
| 65728e16 | 4.2 frozen hash | `results/c1/TEST_OPENED.json` | pinned.results/c1/frozen.json |
| 72f15508 | 4.2 model hash | `results/c1/TEST_OPENED.json` | pinned.results/c1/pi1x_models.npz |
| c0902af3 | 4.2 bundle hash | `results/c1/TEST_OPENED.json` | test_code_bundle_sha256 |
| 24 | 4.2 bundle size | `results/c1/TEST_OPENED.json` | len(test_code_bundle_files) |
| c5c68fbb | 3.2 modeling_ouro hash | `hf_cache/ouro-1.4b/modeling_ouro.py` | sha256 of file (computed) |
| −0.024 | Table 3 G3a | `results/phase3/g3a_report.json` | D2.candidates.pi1-x.improvement_vs_comparator |
| −0.0236 | Table 3 note / 5.1 | `results/phase3/g3a_report.json` | D2.candidates.pi1-x.improvement_vs_comparator (4 dp) |
| −0.044 | Table 3 per-fold min | `results/phase3/g3a_report.json` | D2.candidates.pi1-x.per_fold (min) |
| −0.014 | Table 3 per-fold max | `results/phase3/g3a_report.json` | D2.candidates.pi1-x.per_fold (max) |
| −0.029 | Table 3 validation | `results/phase3/g3a_report.json` | D2_validation_exposed.improvement_vs_comparator.pi1-x |
| −0.022 | Table 3 / 5.1 / Fig 1 point-only TEST | `results/c1/test_report.json` | secondary_point_rule.delta |
| −0.026 | Table 3 / Fig 1 CI lower | `results/c1/test_report.json` | secondary_point_rule.ci95[0] |
| −0.018 | Table 3 / Fig 1 CI upper | `results/c1/test_report.json` | secondary_point_rule.ci95[1] |
| +0.053 | Table 3 / 5.1 / 6.1 / Fig 1 primary Δ | `results/c1/test_report.json` | primary.delta_gross |
| +0.049 | Table 3 / 6.1 / Fig 1 primary CI lower | `results/c1/test_report.json` | primary.ci95[0] |
| +0.057 | Table 3 / 6.1 / Fig 1 primary CI upper | `results/c1/test_report.json` | primary.ci95[1] |
| 0.049 | 5.1 lower bound 0.049 | `results/c1/test_report.json` | primary.ci95[0] |
| +0.038 | Table 3 / 5.5 net Δ | `results/c1/test_report.json` | secondary_net.delta_net |
| +0.033 | Table 3 net CI lower | `results/c1/test_report.json` | secondary_net.ci95[0] |
| +0.042 | Table 3 net CI upper | `results/c1/test_report.json` | secondary_net.ci95[1] |
| 0.0042 | 5.1 realised half-width | `results/c1/test_report.json` | (primary.ci95[1] - primary.ci95[0]) / 2 (derived) |
| 0.0043 | 4.3 / 5.1 expected half-width | `results/c1/frozen.json` | expected_precision.expected_half_width_95 |
| 0.02 | 5.1 / 4.3 delta | `results/c1/test_report.json` | delta_min (also protocol C1.3) |
| 0.17 | 5.1 point losses π1-x OBQA | `results/c1/test_report.json` | per_benchmark.pi1-x.openbookqa.loss_pp |
| 0.60 | 5.1 / Table 4 π1-x CSQA loss | `results/c1/test_report.json` | per_benchmark.pi1-x.commonsense_qa.loss_pp |
| 0.40 | 5.1 / Table 4 π1-x HS loss | `results/c1/test_report.json` | per_benchmark.pi1-x.hellaswag.loss_pp |
| 0.239 | 5.1 / Table 5 / 6.1 macro π1-x | `results/c1/test_report.json` | macro_gross_saving.pi1-x |
| 0.186 | 5.1 / Table 5 / 5.4 macro native-pb | `results/c1/test_report.json` | macro_gross_saving.native-pb |
| 0.157 | Table 4 pi1-x OBQA saving | `results/c1/test_report.json` | per_benchmark.pi1-x.openbookqa.saving |
| 0.17 | Table 4 pi1-x OBQA loss | `results/c1/test_report.json` | per_benchmark.pi1-x.openbookqa.loss_pp |
| 0.53 | Table 4 pi1-x OBQA upper bound | `results/c1/test_report.json` | budget_table.pi1-x.openbookqa.upper_bound_95_pp |
| 6 / 243 / 1,377 / 1,374 | Table 4 pi1-x OBQA exit hist | `results/c1/test_report.json` | per_benchmark.pi1-x.openbookqa.exit_hist.1-4 |
| 0.282 | Table 4 pi1-x CSQA saving | `results/c1/test_report.json` | per_benchmark.pi1-x.commonsense_qa.saving |
| 0.60 | Table 4 pi1-x CSQA loss | `results/c1/test_report.json` | per_benchmark.pi1-x.commonsense_qa.loss_pp |
| 1.10 | Table 4 pi1-x CSQA upper bound | `results/c1/test_report.json` | budget_table.pi1-x.commonsense_qa.upper_bound_95_pp |
| 8 / 349 / 405 / 238 | Table 4 pi1-x CSQA exit hist | `results/c1/test_report.json` | per_benchmark.pi1-x.commonsense_qa.exit_hist.1-4 |
| 0.279 | Table 4 pi1-x HS saving | `results/c1/test_report.json` | per_benchmark.pi1-x.hellaswag.saving |
| 0.40 | Table 4 pi1-x HS loss | `results/c1/test_report.json` | per_benchmark.pi1-x.hellaswag.loss_pp |
| 1.60 | Table 4 pi1-x HS upper bound | `results/c1/test_report.json` | budget_table.pi1-x.hellaswag.upper_bound_95_pp |
| 0 / 88 / 662 / 0 | Table 4 pi1-x HS exit hist | `results/c1/test_report.json` | per_benchmark.pi1-x.hellaswag.exit_hist.1-4 |
| 0.136 | Table 4 native-pb OBQA saving | `results/c1/test_report.json` | per_benchmark.native-pb.openbookqa.saving |
| 0.20 | Table 4 native-pb OBQA loss | `results/c1/test_report.json` | per_benchmark.native-pb.openbookqa.loss_pp |
| 0.67 | Table 4 native-pb OBQA upper bound | `results/c1/test_report.json` | budget_table.native-pb.openbookqa.upper_bound_95_pp |
| 0 / 26 / 1,579 / 1,395 | Table 4 native-pb OBQA exit hist | `results/c1/test_report.json` | per_benchmark.native-pb.openbookqa.exit_hist.1-4 |
| 0.161 | Table 4 native-pb CSQA saving | `results/c1/test_report.json` | per_benchmark.native-pb.commonsense_qa.saving |
| −0.10 | Table 4 native-pb CSQA loss | `results/c1/test_report.json` | per_benchmark.native-pb.commonsense_qa.loss_pp |
| 0.30 | Table 4 native-pb CSQA upper bound | `results/c1/test_report.json` | budget_table.native-pb.commonsense_qa.upper_bound_95_pp |
| 0 / 1 / 643 / 356 | Table 4 native-pb CSQA exit hist | `results/c1/test_report.json` | per_benchmark.native-pb.commonsense_qa.exit_hist.1-4 |
| 0.262 | Table 4 native-pb HS saving | `results/c1/test_report.json` | per_benchmark.native-pb.hellaswag.saving |
| 0.13 | Table 4 native-pb HS loss | `results/c1/test_report.json` | per_benchmark.native-pb.hellaswag.loss_pp |
| 1.33 | Table 4 native-pb HS upper bound | `results/c1/test_report.json` | budget_table.native-pb.hellaswag.upper_bound_95_pp |
| 0 / 36 / 714 / 0 | Table 4 native-pb HS exit hist | `results/c1/test_report.json` | per_benchmark.native-pb.hellaswag.exit_hist.1-4 |
| 0.000 | Table 4 B_pb OBQA saving | `results/c1/test_report.json` | per_benchmark.B_pb.openbookqa.saving |
| 0.00 | Table 4 B_pb OBQA loss | `results/c1/test_report.json` | per_benchmark.B_pb.openbookqa.loss_pp |
| 0.00 | Table 4 B_pb OBQA upper bound | `results/c1/test_report.json` | budget_table.B_pb.openbookqa.upper_bound_95_pp |
| 0 / 0 / 0 / 3,000 | Table 4 B_pb OBQA exit hist | `results/c1/test_report.json` | per_benchmark.B_pb.openbookqa.exit_hist.1-4 |
| 0.000 | Table 4 B_pb CSQA saving | `results/c1/test_report.json` | per_benchmark.B_pb.commonsense_qa.saving |
| 0.00 | Table 4 B_pb CSQA loss | `results/c1/test_report.json` | per_benchmark.B_pb.commonsense_qa.loss_pp |
| 0.00 | Table 4 B_pb CSQA upper bound | `results/c1/test_report.json` | budget_table.B_pb.commonsense_qa.upper_bound_95_pp |
| 0 / 0 / 0 / 1,000 | Table 4 B_pb CSQA exit hist | `results/c1/test_report.json` | per_benchmark.B_pb.commonsense_qa.exit_hist.1-4 |
| 0.250 | Table 4 B_pb HS saving | `results/c1/test_report.json` | per_benchmark.B_pb.hellaswag.saving |
| 0.53 | Table 4 B_pb HS loss | `results/c1/test_report.json` | per_benchmark.B_pb.hellaswag.loss_pp |
| 1.60 | Table 4 B_pb HS upper bound | `results/c1/test_report.json` | budget_table.B_pb.hellaswag.upper_bound_95_pp |
| 0 / 0 / 750 / 0 | Table 4 B_pb HS exit hist | `results/c1/test_report.json` | per_benchmark.B_pb.hellaswag.exit_hist.1-4 |
| 4 | Table 4 m_b pi1-x OBQA | `results/c1/frozen.json` | primary.pi1-x.openbookqa.m |
| 4 | Table 4 m_b pi1-x CSQA | `results/c1/frozen.json` | primary.pi1-x.commonsense_qa.m |
| 3 | Table 4 m_b pi1-x HS | `results/c1/frozen.json` | primary.pi1-x.hellaswag.m |
| 4 | Table 4 m_b native-pb OBQA | `results/c1/frozen.json` | primary.native-pb.openbookqa.m |
| 4 | Table 4 m_b native-pb CSQA | `results/c1/frozen.json` | primary.native-pb.commonsense_qa.m |
| 3 | Table 4 m_b native-pb HS | `results/c1/frozen.json` | primary.native-pb.hellaswag.m |
| 4 | Table 4 B_pb m_b OBQA | `results/c1/frozen.json` | reference_B_pb.openbookqa.m |
| 4 | Table 4 B_pb m_b CSQA | `results/c1/frozen.json` | reference_B_pb.commonsense_qa.m |
| 3 | Table 4 B_pb m_b HS | `results/c1/frozen.json` | reference_B_pb.hellaswag.m |
| 0.021 | 5.2 π1-x minus native-pb saving OBQA | `results/c1/test_report.json` | per_benchmark.pi1-x.openbookqa.saving - per_benchmark.native-pb.openbookqa.saving (derived) |
| 0.120 | 5.2 π1-x minus native-pb saving CSQA | `results/c1/test_report.json` | per_benchmark.pi1-x.commonsense_qa.saving - per_benchmark.native-pb.commonsense_qa.saving (derived) |
| 0.017 | 5.2 π1-x minus native-pb saving HS | `results/c1/test_report.json` | per_benchmark.pi1-x.hellaswag.saving - per_benchmark.native-pb.hellaswag.saving (derived) |
| 0.75 | 5.3 dev UB min | `results/c1/frozen.json` | min of primary.{pi1-x,native-pb}.*.upper_bound (derived) |
| 0.98 | 5.3 dev UB max | `results/c1/frozen.json` | max of primary.{pi1-x,native-pb}.*.upper_bound (derived) |
| 0.305 | Table 5 point-only pi1-x OBQA saving | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.pi1-x.openbookqa.saving |
| 1.13 | Table 5 point-only pi1-x OBQA loss | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.pi1-x.openbookqa.loss_pp |
| 0.357 | Table 5 point-only pi1-x CSQA saving | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.pi1-x.commonsense_qa.saving |
| 1.10 | Table 5 point-only pi1-x CSQA loss | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.pi1-x.commonsense_qa.loss_pp |
| 0.351 | Table 5 point-only pi1-x HS saving | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.pi1-x.hellaswag.saving |
| 1.07 | Table 5 point-only pi1-x HS loss | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.pi1-x.hellaswag.loss_pp |
| 0.337 | Table 5 / 5.4 point-only macro pi1-x | `results/c1/test_report.json` | mean of secondary_point_rule.per_benchmark.pi1-x.*.saving (derived) |
| 0.403 | Table 5 point-only native-pb OBQA saving | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.native-pb.openbookqa.saving |
| 1.57 | Table 5 point-only native-pb OBQA loss | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.native-pb.openbookqa.loss_pp |
| 0.359 | Table 5 point-only native-pb CSQA saving | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.native-pb.commonsense_qa.saving |
| 1.20 | Table 5 point-only native-pb CSQA loss | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.native-pb.commonsense_qa.loss_pp |
| 0.316 | Table 5 point-only native-pb HS saving | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.native-pb.hellaswag.saving |
| 0.67 | Table 5 point-only native-pb HS loss | `results/c1/test_report.json` | secondary_point_rule.per_benchmark.native-pb.hellaswag.loss_pp |
| 0.359 | Table 5 / 5.4 point-only macro native-pb | `results/c1/test_report.json` | mean of secondary_point_rule.per_benchmark.native-pb.*.saving (derived) |
| 0.173 | 5.4 drop native-pb | `results/c1/test_report.json` | macro point-only native-pb minus macro_gross_saving.native-pb (derived) |
| 0.098 | 5.4 drop π1-x | `results/c1/test_report.json` | macro point-only π1-x minus macro_gross_saving.pi1-x (derived) |
| 0.022 | 5.1 point-only gap (0.022 as magnitude) | `results/c1/test_report.json` | abs(secondary_point_rule.delta) |
| 0.125 | 5.5 net saving π1-x OBQA | `results/c1/test_report.json` | per_benchmark.pi1-x.openbookqa.net_saving |
| 0.157 | 5.5 gross saving π1-x OBQA | `results/c1/test_report.json` | per_benchmark.pi1-x.openbookqa.saving |
| 0.274 | 5.5 net saving π1-x CSQA | `results/c1/test_report.json` | per_benchmark.pi1-x.commonsense_qa.net_saving |
| 0.282 | 5.5 gross saving π1-x CSQA | `results/c1/test_report.json` | per_benchmark.pi1-x.commonsense_qa.saving |
| 0.272 | 5.5 net saving π1-x HS | `results/c1/test_report.json` | per_benchmark.pi1-x.hellaswag.net_saving |
| 0.279 | 5.5 gross saving π1-x HS | `results/c1/test_report.json` | per_benchmark.pi1-x.hellaswag.saving |
| 0.031 | 5.5 overhead OBQA | `results/c1/test_report.json` | saving - net_saving, per_benchmark.pi1-x.openbookqa (derived) |
| 0.007 | 5.5 overhead CSQA and HS | `results/c1/test_report.json` | saving - net_saving, per_benchmark.pi1-x.{commonsense_qa,hellaswag} (derived; 0.0074 and 0.0073) |
| 0.083 | Table 6 B_pb macro saving | `results/c1/test_report.json` | macro_gross_saving.B_pb |
| +0.156 | Table 6 B_pb Δ | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.B_pb.delta |
| +0.151 | Table 6 B_pb CI lower | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.B_pb.ci95[0] |
| +0.161 | Table 6 B_pb CI upper | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.B_pb.ci95[1] |
| 0.186 | Table 6 native-pb macro saving | `results/c1/test_report.json` | macro_gross_saving.native-pb |
| 0.171 | Table 6 r-conf macro saving | `results/c1/test_report.json` | macro_gross_saving.r-conf |
| +0.068 | Table 6 r-conf Δ | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-conf.delta |
| +0.063 | Table 6 r-conf CI lower | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-conf.ci95[0] |
| +0.073 | Table 6 r-conf CI upper | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-conf.ci95[1] |
| 0.170 | Table 6 r-margin macro saving | `results/c1/test_report.json` | macro_gross_saving.r-margin |
| +0.069 | Table 6 r-margin Δ | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-margin.delta |
| +0.064 | Table 6 r-margin CI lower | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-margin.ci95[0] |
| +0.075 | Table 6 r-margin CI upper | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-margin.ci95[1] |
| 0.179 | Table 6 r-ent macro saving | `results/c1/test_report.json` | macro_gross_saving.r-ent |
| +0.060 | Table 6 r-ent Δ | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-ent.delta |
| +0.055 | Table 6 r-ent CI lower | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-ent.ci95[0] |
| +0.065 | Table 6 r-ent CI upper | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-ent.ci95[1] |
| 0.162 | Table 6 r-dlogit macro saving | `results/c1/test_report.json` | macro_gross_saving.r-dlogit |
| +0.077 | Table 6 r-dlogit Δ | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-dlogit.delta |
| +0.071 | Table 6 r-dlogit CI lower | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-dlogit.ci95[0] |
| +0.083 | Table 6 r-dlogit CI upper | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-dlogit.ci95[1] |
| 0.174 | Table 6 r-hcos macro saving | `results/c1/test_report.json` | macro_gross_saving.r-hcos |
| +0.065 | Table 6 r-hcos Δ | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-hcos.delta |
| +0.059 | Table 6 r-hcos CI lower | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-hcos.ci95[0] |
| +0.072 | Table 6 r-hcos CI upper | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-hcos.ci95[1] |
| 0.149 | Table 6 r-hl2 macro saving | `results/c1/test_report.json` | macro_gross_saving.r-hl2 |
| +0.090 | Table 6 r-hl2 Δ | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-hl2.delta |
| +0.084 | Table 6 r-hl2 CI lower | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-hl2.ci95[0] |
| +0.096 | Table 6 r-hl2 CI upper | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-hl2.ci95[1] |
| 0.238 | Table 6 r-kl macro saving | `results/c1/test_report.json` | macro_gross_saving.r-kl |
| +0.0012 | Table 6 r-kl Δ | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-kl.delta |
| −0.0041 | Table 6 r-kl CI lower | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-kl.ci95[0] |
| +0.0065 | Table 6 r-kl CI upper | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.r-kl.ci95[1] |
| 0.238 | Table 6 / 6.1 r-kl macro | `results/c1/test_report.json` | macro_gross_saving.r-kl |
| 0.052 | 6.1 r-kl exceeds native | `results/c1/test_report.json` | macro_gross_saving.r-kl - macro_gross_saving.native-pb (derived) |
| 0.060 | 6.1 other readouts trail (min) | `results/c1/test_report.json` | min of secondary_all_baselines.per_baseline.{r-conf,r-margin,r-ent,r-dlogit,r-hcos,r-hl2}.delta |
| 0.090 | 6.1 other readouts trail (max) | `results/c1/test_report.json` | max of same |
| 0.156 | 6.1 B_pb trails | `results/c1/test_report.json` | secondary_all_baselines.per_baseline.B_pb.delta |
| 0.93 | 6.1 largest baseline TEST point loss | `results/c1/test_report.json` | max of secondary_violations.*.* (r-dlogit, hellaswag) |
| 10,000 | 6.1 selection frequency | `results/c1/test_report.json` | secondary_all_baselines.selection_frequency.r-kl (of n_boot) |
| 10,000 | 6.1 / 4.3 bootstrap size | `results/c1/test_report.json` | n_boot |
| 0.65 | 6.1 pp upper | `results/c1/test_report.json` | 100 * secondary_all_baselines.per_baseline.r-kl.ci95[1] (derived) |
| 0.41 | 6.1 pp lower | `results/c1/test_report.json` | 100 * abs(secondary_all_baselines.per_baseline.r-kl.ci95[0]) (derived) |
| 0.246 | 6.1 dev in-sample π1-x | `results/c1/frozen.json` | selection_macro_saving_dev.pi1-x |
| 0.243 | 6.1 dev in-sample r-kl | `results/c1/frozen.json` | selection_macro_saving_dev.r-kl |
| 0.193 | 6.1 dev in-sample native | `results/c1/frozen.json` | selection_macro_saving_dev.native-pb |
| 3.5×10⁻³ | 6.1 r-kl threshold OBQA | `results/c1/frozen.json` | secondary_risk_all_baselines.r-kl.openbookqa.tau (−0.00350) |
| 3.7×10⁻² | 6.1 r-kl threshold CSQA | `results/c1/frozen.json` | secondary_risk_all_baselines.r-kl.commonsense_qa.tau (−0.0375) |
| 2.7×10⁻⁴ | 6.1 r-kl threshold HS | `results/c1/frozen.json` | secondary_risk_all_baselines.r-kl.hellaswag.tau (−0.000271) |
| 3 | 6.1 r-kl caps OBQA / HS | `results/c1/frozen.json` | secondary_risk_all_baselines.r-kl.{openbookqa,hellaswag}.m |
| 4 | 6.1 r-kl cap CSQA | `results/c1/frozen.json` | secondary_risk_all_baselines.r-kl.commonsense_qa.m |
| 0.308 | Table 7 PIQA pi1-x saving | `results/c1/test_report.json` | secondary_ood.piqa.summary.pi1-x.saving |
| 0.71 | Table 7 PIQA pi1-x loss | `results/c1/test_report.json` | secondary_ood.piqa.summary.pi1-x.loss_pp |
| 1.52 | Table 7 PIQA pi1-x upper bound | `results/c1/test_report.json` | secondary_ood.piqa.upper_bound_95_pp.pi1-x |
| 0 / 433 / 1,399 / 5 | Table 7 PIQA pi1-x exit hist | `results/c1/test_report.json` | secondary_ood.piqa.summary.pi1-x.exit_hist.1-4 |
| 0.126 | Table 7 PIQA native-pb saving | `results/c1/test_report.json` | secondary_ood.piqa.summary.native-pb.saving |
| 0.11 | Table 7 PIQA native-pb loss | `results/c1/test_report.json` | secondary_ood.piqa.summary.native-pb.loss_pp |
| 0.60 | Table 7 PIQA native-pb upper bound | `results/c1/test_report.json` | secondary_ood.piqa.upper_bound_95_pp.native-pb |
| 0 / 0 / 923 / 914 | Table 7 PIQA native-pb exit hist | `results/c1/test_report.json` | secondary_ood.piqa.summary.native-pb.exit_hist.1-4 |
| 0.000 | Table 7 PIQA B_pb saving | `results/c1/test_report.json` | secondary_ood.piqa.summary.B_pb.saving |
| 0.00 | Table 7 PIQA B_pb loss | `results/c1/test_report.json` | secondary_ood.piqa.summary.B_pb.loss_pp |
| 0.00 | Table 7 PIQA B_pb upper bound | `results/c1/test_report.json` | secondary_ood.piqa.upper_bound_95_pp.B_pb |
| 0 / 0 / 0 / 1,837 | Table 7 PIQA B_pb exit hist | `results/c1/test_report.json` | secondary_ood.piqa.summary.B_pb.exit_hist.1-4 |
| 0.183 | 6.2 PIQA Δ vs native | `results/c1/test_report.json` | secondary_ood.piqa.delta_pi1x_vs_native |
| +0.174 | 6.2 PIQA CI lower | `results/c1/test_report.json` | secondary_ood.piqa.ci95[0] |
| +0.191 | 6.2 PIQA CI upper | `results/c1/test_report.json` | secondary_ood.piqa.ci95[1] |
| 0.481 | Table 7 SIQA pi1-x saving | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.pi1-x.saving |
| 3.26 | Table 7 SIQA pi1-x loss | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.pi1-x.loss_pp |
| 4.72 | Table 7 SIQA pi1-x upper bound | `results/c1/test_report.json` | secondary_ood.social_iqa.upper_bound_95_pp.pi1-x |
| 0 / 1,785 / 143 / 2 | Table 7 SIQA pi1-x exit hist | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.pi1-x.exit_hist.1-4 |
| 0.028 | Table 7 SIQA native-pb saving | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.native-pb.saving |
| −0.05 | Table 7 SIQA native-pb loss | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.native-pb.loss_pp |
| 0.16 | Table 7 SIQA native-pb upper bound | `results/c1/test_report.json` | secondary_ood.social_iqa.upper_bound_95_pp.native-pb |
| 0 / 0 / 214 / 1,716 | Table 7 SIQA native-pb exit hist | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.native-pb.exit_hist.1-4 |
| 0.000 | Table 7 SIQA B_pb saving | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.B_pb.saving |
| 0.00 | Table 7 SIQA B_pb loss | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.B_pb.loss_pp |
| 0.00 | Table 7 SIQA B_pb upper bound | `results/c1/test_report.json` | secondary_ood.social_iqa.upper_bound_95_pp.B_pb |
| 0 / 0 / 0 / 1,930 | Table 7 SIQA B_pb exit hist | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.B_pb.exit_hist.1-4 |
| +0.453 | 6.2 SIQA Δ vs native | `results/c1/test_report.json` | secondary_ood.social_iqa.delta_pi1x_vs_native |
| +0.449 | 6.2 SIQA CI lower | `results/c1/test_report.json` | secondary_ood.social_iqa.ci95[0] |
| +0.458 | 6.2 SIQA CI upper | `results/c1/test_report.json` | secondary_ood.social_iqa.ci95[1] |
| 43.32% | 6.2 SIQA accuracy loop 4 | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.pi1-x.acc4_pp |
| 40.05% | 6.2 SIQA accuracy π1-x | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.pi1-x.acc_pp |
| 0.961 | 6.2 pooled π1-x tau | `results/c1/frozen.json` | ood_pooled.pi1-x.tau |
| 0.386 | 6.2 pooled native tau | `results/c1/frozen.json` | ood_pooled.native-pb.tau |
| 4 | 6.2 pooled m | `results/c1/frozen.json` | ood_pooled.{pi1-x,native-pb}.m |
| 0.159 | 6.2 pooled dev macro saving | `results/c1/frozen.json` | ood_pooled.pi1-x.macro_saving and ood_pooled.native-pb.macro_saving (0.1588, 0.1591) |
| +0.053 | 3.4 / 4.2 E1 +0.053 | `docs/E1_REPORT.md` | Headline 1 table (π1-x vs native-pb, risk rule), "all 5 folds positive" (document) |
| −0.024 | 4.2 E1 −0.024 | `docs/E1_REPORT.md` | Headline 1, "π1-x −0.024" (point rule) (document) |
| 3 of 15 | 3.4 E1 violations | `docs/E1_REPORT.md` | Headline 1, "Held-out violations ... 3 of 15 for native-pb and for π1-x under the risk rule" (document) |
| 0.689 | 5.4 E1 AUROC π1-x | `docs/E1_REPORT.md` | Headline 2 table, target S_t, π1-x (document) |
| 0.603 | 5.4 E1 AUROC native | `docs/E1_REPORT.md` | Headline 2 table, target S_t, native cdf (document) |
| 0.207 | 6.1 nested E1 r-kl | `docs/C1_FREEZE_AUDIT.md` | "Facts the reviewer should see" item 1 (r-kl 0.207 against native-pb 0.204) (document) |
| 0.204 | 6.1 nested E1 native | `docs/C1_FREEZE_AUDIT.md` | same (document) |
| 1.16 | 4.3 A1.5 CSQA bounds | `docs/PHASE3_AMENDMENT_A1.md` | A1.5 table, CommonsenseQA, d = 0.05 (document) |
| 1.65 | 4.3 A1.5 CSQA bounds | `docs/PHASE3_AMENDMENT_A1.md` | A1.5 table, CommonsenseQA, d = 0.10 (document) |
| 1.34 | 4.3 A1.5 HS bounds | `docs/PHASE3_AMENDMENT_A1.md` | A1.5 table, HellaSwag, d = 0.05 (document) |
| 1.90 | 4.3 A1.5 HS bounds | `docs/PHASE3_AMENDMENT_A1.md` | A1.5 table, HellaSwag, d = 0.10 (document) |
| 5% to 10% | 4.3 A1.5 d values | `docs/PHASE3_AMENDMENT_A1.md` | A1.5, d = 0.05 and d = 0.10 (document) |
| 100 | 4.4 D-a n rows | `docs/DECISIONS.md` | D-a entry: "100 never-used, unallocated TRAIN rows per benchmark" (document) |
| 3.1e-5 | 4.4 D-a SIQA max | `docs/DECISIONS.md` | D-a run entry: Social IQa PASS (float32 max 3.1e-5) (document) |
| 1.17e-4 | 4.4 D-a PIQA max | `docs/DECISIONS.md` | D-a run entry: PIQA FAIL (float32 max 1.17e-4) (document) |
| 1e-4 | 4.4 D-a tolerance | `docs/DECISIONS.md` | D-a run entry (tolerance unchanged) (document) |
| 10 | 4.4 D-a2 rows | `docs/DECISIONS.md` | D-a2 description: 10 PIQA questions with largest float32 difference (document) |
| 1e-9 | 4.4 D-a2 criterion | `docs/DECISIONS.md` | D-a2 criterion (document) |
| 1.67e-13 | 4.4 D-a2 max | `docs/DECISIONS.md` | D-a2 result entry (document) |
| 0.030 to 0.036 | 3.6 bf16 gap range | `docs/DECISIONS.md` | Full Phase 3 extraction entry: bf16 context-only state gap relative L2 0.030-0.036 on OBQA, PIQA, SIQA (document) |
| 50 | 3.4 grid size | `src/lookahead/gate1.py` | N_GRID = 50 (quantile points); docs/PHASE3_PLAN.md section 5 grid bullet (document) |
| 51 | 3.4 grid size | `docs/PHASE3_PLAN.md` | Section 5: "50 quantiles ... plus +∞. So 51 × 4 = 204 candidates" (document) |
| 204 | 3.4 candidates | `docs/PHASE3_PLAN.md` | same (document) |
| 1e-12 | 3.4 tolerance | `src/lookahead/gate1.py` | TOL = 1e-12 (document) |
| 1 pp | 3.4 / 3.3 / 5.1 point budget | `src/lookahead/gate1.py` | LOSS_LIMIT = 1.0 (pp) (document) |
| 1.0 | 3.3 probe C | `src/lookahead/phase3_g3a.py; results/c1/frozen.json` | FROZEN_C["x"] = 1.0; pi1x.C (document) |
| four of the nine | 5.3 count of upper bounds above 1 pp | `results/c1/test_report.json` | count of budget_table.*.*.upper_bound_over_1pp == true (derived; 4 of 9) |
| All nine point losses within budget | 5.3 count of point losses above 1 pp | `results/c1/test_report.json` | count of budget_table.*.*.point_over_1pp == true is 0 (derived) |
| 1 of 1,000 | 5.2 native-pb CSQA exits at loop 2 | `results/c1/test_report.json` | per_benchmark.native-pb.commonsense_qa.exit_hist.2 (= 1) and n |
| 349 of 1,000 | 5.2 π1-x CSQA exits at loop 2 | `results/c1/test_report.json` | per_benchmark.pi1-x.commonsense_qa.exit_hist.2 |
| 662 of 750; 714 of 750 | 5.2 HS exits at loop 3 | `results/c1/test_report.json` | per_benchmark.{pi1-x,native-pb}.hellaswag.exit_hist.3 |
| 1,785 of 1,930 | 6.2 SIQA π1-x exits at loop 2 | `results/c1/test_report.json` | secondary_ood.social_iqa.summary.pi1-x.exit_hist.2 and n |
| 10,000 of 10,000 | 6.1 r-kl selected in resamples | `results/c1/test_report.json` | secondary_all_baselines.selection_frequency.r-kl and n_boot |
| all five folds | 5.1 five folds (E1, G3a) | `docs/E1_REPORT.md; results/phase3/g3a_report.json` | E1 Headline 1 (+0.037 to +0.083 across folds); G3a D2.candidates.pi1-x.per_fold has 5 entries (document) |

### Structural and definitional numbers

| Number or class of numbers | Source | Where defined |
|---|---|---|
| 4 loops; "loop 4"; "four-loop budget"; 1 − e/4; e in {1,2,3,4} | `configs/model.yaml` | total_ut_steps: 4 |
| Ouro-1.4B ("1.4") | `configs/model.yaml` | repo_id: ByteDance/Ouro-1.4B (model name) |
| loops t = 1,2,3 for statistics; 9 columns; three loop indicators; "nine features" | `src/lookahead/gate2.py` | CHEAP_NAMES (9 names, last three loop1..loop3); PROBE_LOOPS = 3 |
| seven r-* rules | `src/lookahead/phase3_readouts.py` | READOUTS (7 names) |
| six development benchmarks; three primary benchmarks | `results/c1/frozen.json; src/lookahead/phase3_g3a.py` | n_dev has 6 keys; PRIMARY has 3 names |
| five-fold cross-fitting; five folds | `src/lookahead/gate2.py` | N_FOLDS = 5 (fold_of: hash mod N_FOLDS); G3a per_fold has 5 entries |
| m_b in {1,2,3,4}; 51-point grid; 50 quantile values | `src/lookahead/phase3_readouts.py; src/lookahead/gate1.py` | exit_loops cap in 1..4; grid_thresholds N_GRID = 50, plus +inf |
| one-sided 95% bootstrap bound; 95th percentile | `src/lookahead/e1.py` | RISK_QUANTILE = 0.95 |
| 95% confidence intervals (two-sided); "95%" | `docs/C1_PROTOCOL.md; src/lookahead/c1_test.py` | C1.5 interval; paired_macro uses quantiles 0.025 and 0.975 |
| OpenBookQA 0-shot; CommonsenseQA and HellaSwag 10-shot | `src/lookahead/questions.py` | SPECS shots: openbookqa 0, commonsense_qa 10, hellaswag 10 |
| OOD benchmarks 0-shot | `docs/PHASE3_AMENDMENT_A1.md` | A1.4 "Shots" |
| 1 pp budget; "1 pp" | `src/lookahead/gate1.py` | LOSS_LIMIT = 1.0 (pp) |
| batch size 1 | `docs/C1_PROTOCOL.md; docs/C1_FREEZE_AUDIT.md` | C1.3 (PackedScorer, batch size 1); audit item 4 |
| $1 - e/4$ gross saving; $\tau = +\infty$; $\rho = 0$ | `docs/C1_PROTOCOL.md` | C1.3 |
| Geiping et al. threshold 5×10⁻⁴; "section 6.1" | `docs/PAPER_OUTLINE.md` | Reference verification table, Geiping row (not re-verified here) |
| Section, table, figure and list numerals (3.1, 3.6, 4.2, Table 1 to Table 7, Figure 1, "Phase 3", "Gate 1/2", "D-a to D-e", "A1.5", "C1.3 to C1.5", "G3a", numbered lists) | `-` | Labels, not measured quantities |
| Hash prefixes (c5c68fbb, b2df1930, 4f38b8b7, 65728e16, 72f15508, c0902af3) | `see rows above` | Computed or read from the named file |

## Notes to author

Inconsistencies between documents and JSON files, and claims I could not verify. Items 1 to 6 affect wording in the draft.

1. **G3a value is a nested development result, not a validation result.** The outline's RQ1 table says "point-only (G3a, validation) −0.024". In `g3a_report.json` the −0.0236 is the nested held-out improvement of π1-x over the comparator across five development folds (`D2.candidates.pi1-x.improvement_vs_comparator`). The exposed Phase 1 validation figure is −0.0289 (`D2_validation_exposed.improvement_vs_comparator.pi1-x`). Table 3 gives both and labels them separately. I labelled both [E] because the outline's scheme has no letter for a pre-registered gate decided on development data; you may prefer a different label.
2. **Grid size.** The outline says "51-point pooled grid plus ∞", which would be 52 thresholds. The code (`gate1.N_GRID = 50`, with exact repeats removed, plus +∞) and the plan and A2 documents give 51 including +∞, hence 51 × 4 = 204 candidates. The draft follows the code ("up to 50 distinct quantile values ... plus +∞ ... up to 51 thresholds").
3. **"Nine cheap features".** Nine columns enter each per-loop model, but the three loop indicators are constant within a per-loop model (the standardised column is zero), so six features carry information. Also `log_n_tokens` is the log of the packed sequence length (context plus all options), not the context length (`gate2.cheap_features` docstring). The draft says both.
4. **Terminology in the frozen protocol.** The title line of `C1_PROTOCOL.md` and the hypothesis and selection wording in C1.1 and C1.2 apply a term to the rule that this paper must not use. The draft quotes only C1.3 to C1.5 verbatim (they contain no such term), paraphrases C1.1, and calls the rule "bootstrap-screened (the protocol's 'combined risk rule')". Say once, in the paper, that the protocol's own wording is stronger than the paper's. The one use of the word "guarantee" in section 3.4 is the mandated statement that the rule carries none.
5. **Two different "D-a" labels.** The C1 protocol's reviewer decisions are D-a to D-e (D-a: development data only; D-b: δ = 0.02). The Phase 3 numerical diagnostics are also called D-a, D-a2 and D-b. The draft says "the protocol's reviewer decision D-a" in section 4.1 and "Phase 3 numerical diagnostic D-a" in section 4.4. Consider renaming one set in the paper.
6. **Wording in `C1_REPORT.md` that I did not carry over.** (a) "The ranking reverses under the point-only rule": replaced by "rule-dependent preference between different operating points", per outline v2. (b) "π1-x exits mostly at loop 3 on OBQA (1,377 of 3,000)": at loop 4 it exits for 1,374, so the draft says "split between loops 3 and 4". (c) The report says the supported claim is that "a stability-based halting signal beats Ouro's native gate", which rests on r-kl − native-pb; `test_report.json` has no interval for that difference. The draft gives only the derived point difference (0.052) and labels it [P].
7. **The ±0.65 pp statement in outline section 6 is asymmetric.** The r-kl interval is [−0.0041, +0.0065], so differences favouring π1-x by more than about 0.65 pp, or favouring r-kl by more than about 0.41 pp, are unlikely. The draft states both limits.
8. **In-sample near-tie.** The outline says the near-tie "was also seen on development data (0.246 vs 0.243)". Those are in-sample selection-data savings (`frozen.json selection_macro_saving_dev`), not held-out. The held-out E1 estimate is r-kl 0.207 against native-pb 0.204 (`C1_FREEZE_AUDIT.md`). The draft labels both and does not describe the in-sample pair as held-out evidence.
9. **What `test_report.json` does not contain.** Per-benchmark TEST savings and exit histograms for the r-* rules, upper bounds for any baseline other than π1-x, native-pb and B_pb, and per-benchmark intervals for π1-x − native-pb. Section 5.2's per-benchmark differences (0.021, 0.121, 0.017) are differences of stored point savings without intervals, labelled [P]. Producing the missing items would require re-reading the sealed TEST arrays, which the protocol treats as spent; that would be a post hoc [P] analysis.
10. **Figure 1.** Only four selected operating points per benchmark exist in the TEST report, so the placeholder describes those, not saving-loss frontiers. A development-data frontier would be [E] and would need the development extraction, which is not in this repository snapshot. I did not create the figure or any figures directory.
11. **References.** (a) [Popescu2026] (arXiv 2607.20519) is not in the outline's reference table; it is cited in the code (`phase3_readouts.py`) and in `docs/PHASE3_LITERATURE.md` as the source of the readouts, and it was not verified in the outline's check. (b) The keys [Zhu2025], [Geiping2025], [Jazbec2024], [Angelopoulos2025LTT], [Zhou2020] follow your examples; [Angelopoulos2024CRC] and [Popescu2026] are my construction. (c) The Geiping et al. threshold (5×10⁻⁴), section (6.1) and per-token unit are taken from the outline's reference table and were not re-checked. (d) The statement that the Ouro paper "states the goal of exiting per token and per example but does not fix the inference unit" comes from the outline; the paper is not in the repository. I did verify the released-code description against `hf_cache/ouro-1.4b/modeling_ouro.py` (sha256 c5c68fbb…): the forward pass runs all loops, the gate is applied at every position, and the threshold path gathers one loop's hidden state per position.
12. **Retrospective exactness (section 3.6).** Established from the code, not by experiment; see the two NOTE blocks. The unrun Phase 3 check D-b is the relevant open item. I found no recorded D-b result, but I cannot rule out that one exists outside the repository.
13. **Cost accounting assumptions (section 3.5).** ρ was measured at batch size 1 on the laptop GPU (RTX 5050, per `C1_FREEZE_AUDIT.md`) on 200 development questions per benchmark (the first five per benchmark are excluded as warm-up in `c1_rho.py`). It divides by one measured loop on the same question, so net saving assumes loop cost is constant across loops and that a skipped loop saves its full measured time, with no fixed per-question cost. The exit head and CDF are charged to no policy by construction. The draft says "estimated deployment cost" throughout.
14. **Unverified commit facts.** `C1_REPORT.md` says the TEST bundle hash corresponds to commit b217523 and that the run, at 196a882, adds documentation only; the marker records 196a882. I was not allowed to run git and did not verify either. The draft relies only on the bundle hash, which the run script checked.
15. **Selection-statistic mismatch (not a defect, worth one sentence in the paper).** Selection used π1-x's out-of-fold scores, while TEST used models refitted on all development data. The two score distributions are not identical, which is a possible contributor to the TEST upper bounds exceeding the selection-time bounds; the draft does not claim it.
16. **Terms avoided.** I avoided the words you listed. The only mandated use of "guarantee" is in section 3.4 (and in items 4 and 16 of this list). Where the outline uses the opposite of non-superiority, the draft says the two signals were not shown to be interchangeable and that no margin for that was pre-specified.
17. **Outline numbers checked against the JSON.** The outline's RQ1 table (apart from the labelling in item 1), its RQ2 row and its RQ3 table agree with `test_report.json`, as does the net-saving row. The outline's upper-bound list (π1-x 1.10 and 1.60, native-pb 1.33, B_pb 1.60) also agrees. I found no outline number that disagrees with a JSON value other than the points in items 1, 7 and 8.
