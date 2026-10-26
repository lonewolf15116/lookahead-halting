# Draft: Sections 1–2

Evidence labels: [C] confirmatory, [S] pre-specified secondary, [P] post hoc emphasis or interpretation, [E] exploratory.

---

## 1. Introduction

Looped, or recurrent-depth, language models apply one block of layers several times. The number of loops therefore becomes a choice made at
inference time [Zhu2025; Geiping2025; Bae2025]. If a model could stop looping once further loops no longer change its answer, it could
spend less compute on easy inputs. Ouro [Zhu2025] ships a learned exit gate for this purpose. Recent work has shown that simple post-hoc
readouts of the loop trajectory, such as confidence, entropy, and divergence or displacement between loops, often match or beat that gate at
validation-selected operating points [Popescu2026]. Popescu et al. conclude that adaptive depth is "a trajectory–readout problem, not a
gate-learning problem".

Comparisons of this kind have one feature in common. Every method is reduced to a single operating point, chosen on held-out data by some
selection rule, such as a target accuracy or a budget on accuracy loss, and the methods are then compared at those points. The rule is usually
treated as a detail of the protocol. This paper asks whether it is one. We study three questions on a frozen Ouro-1.4B:

- **RQ1.** Does the selection rule change which halting method is preferred?
- **RQ2.** Does a trained probe added on top of the frozen model save more than (a) Ouro's own trained gate and (b) a training-free
  divergence between successive loops?
- **RQ3.** Do policies selected on development data keep their accuracy budget on benchmarks that were not used for development?

We answer these questions with a sequence of pre-registered studies. A first study (G3a) selected operating points with a point-only rule:
accuracy loss of at most 1 percentage point (pp) relative to the full four loops. Under that rule, a trained probe on cheap trajectory features
(π1-x) saved less than Ouro's gate, and the study stopped [E]. A second, confirmatory study (C1) was pre-registered after an exploratory analysis.
It compared the same two signals under a stricter, bootstrap-screened rule, which also required the bootstrap 95% upper bound of the loss to be
at most 1 pp, and it was evaluated once on a sealed test set of 8,517 questions.

**Findings.**
1. Under the bootstrap-screened rule, π1-x saved more of the four-loop budget than Ouro's gate: +0.053 in gross macro saving, 95% CI
   [+0.049, +0.057], over OpenBookQA, CommonsenseQA and HellaSwag [C]. Under the point-only rule, applied to the same test set, the
   preference reversed: −0.022 [−0.026, −0.018] [S]. The two rules select different operating points, so what reverses is the preference that each rule produces, not the
   order of fixed policies. The point-only test operating points also exceeded the 1 pp budget on most benchmarks.
2. π1-x showed no demonstrated advantage over an untrained KL divergence between successive loops' answer distributions: +0.0012
   [−0.0041, +0.0065]. This comparison was pre-specified [S]. The emphasis we give it was decided after seeing the test results [P], and the
   result is non-superiority, not equivalence.
3. On one of two out-of-distribution benchmarks (Social IQa), π1-x broke its accuracy budget by halting early on most questions: loss 3.26
   pp, upper bound 4.72 pp [S]. Ouro's gate stayed within the budget.
4. On the test set, four of nine upper bounds exceeded 1 pp even though every point loss stayed within it [S]. This is consistent with our
   screen being a heuristic, not a formal risk-control procedure [P].

**Contributions.** We do not propose a new halting signal, and we do not claim that post-hoc readouts beating Ouro's gate is new; that was
shown by Popescu et al. Our contributions are empirical and methodological:
- A pre-registered, sealed-test evaluation of halting signals on a frozen looped LM, showing that the preferred method depends on the
  operating-point selection rule (RQ1).
- Evidence that a trained probe adds no demonstrated value over a training-free convergence readout under that rule (RQ2), and an
  out-of-distribution budget failure for the trained probe (RQ3).
- Six recommendations for evaluating halting methods (Section 7.4). The central one is to treat each comparison as a comparison of signal
  plus selection procedure.

**Scope.** All results are for Ouro-1.4B with four loops, multiple-choice scoring, and one halting decision per question. The policies were
evaluated from stored full-depth traces, not run with truncated computation. Our loop savings are therefore estimates, not measured time,
and they are not measurements of Ouro's released inference path, which selects a readout per token after computing all loops (Section 3.2).

## 2. Related work

**Learned halting for recurrent computation.**
- ACT [Graves2016] adds a sigmoid halting unit to an RNN. It halts when the cumulative halting probability reaches 1−ε and trains with a
  ponder cost.
- PonderNet [Banino2021] replaces this with a learned conditional halting distribution and a geometric prior.
- Ouro's exit gate [Zhu2025] belongs to the same family. A linear head on each loop's state gives a per-loop exit probability, and Q-exit
  stops at the first loop whose cumulative exit probability reaches a threshold q. The gate is trained jointly with the model and then
  refined on the loss improvement from an extra loop.
- Ouro's own analysis on MMLU found the trained gate best at every budget. A hidden-state difference threshold came within one to two
  points of it [Zhu2025, section 5.4.1].
- Joint training of depth controllers is also studied in Mixture-of-Recursions [Bae2025] (per-token routers over recursion depth), in
  AdaPonderLM [Song2026] (per-iteration gates with a ponder penalty) and in Think-at-Hard [Fu2025] (a per-token iteration decider). We train
  nothing in the backbone or the gate; our probe sits on top of a frozen model.

**Post-hoc readouts on Ouro.** The closest work is Popescu et al. [Popescu2026]:
- They evaluated the released Ouro-1.4B and 2.6B on six benchmarks, including the three primary benchmarks we use.
- They compared the pretrained gate with post-hoc readouts: entropy, top-1 confidence, margin, predictive KL, logit change, and hidden-state
  displacement and cosine change.
- They chose thresholds on validation data and evaluated on held-out data.
- They found the gate "competitive but not uniformly Pareto-optimal".
- Our readout definitions follow theirs [Popescu2026, appendix B.6]. Our threshold grid has a similar size but a different construction:
  - ours: 50 quantiles over the full range plus a never-exit point, inherited from earlier stages of this project;
  - theirs: 51 quantiles between the 0.5th and 99.5th percentiles [Popescu2026, appendix B.5].
- Our work differs from theirs in four respects:
  1. a trained probe on the frozen trajectory;
  2. per-benchmark accuracy-loss budgets with a fixed-depth baseline;
  3. a comparison of two selection rules;
  4. a pre-registered evaluation, run once on a sealed test set.

**Convergence and stability exits.**
- Geiping et al. [Geiping2025] stop a recurrent-depth model once the KL divergence between successive steps falls below 5×10⁻⁴, per
  generated token. Our r-kl applies the same kind of signal to the answer-option distribution, once per question, with thresholds selected
  per benchmark.
- Pappone et al. [Pappone2025] propose a second-order ("acceleration") criterion on the step size and report that it beats the KL exit in a
  small recurrent-depth model.
- PABEE [Zhou2020] stops a fixed-depth encoder once consecutive layer classifiers agree for a set number of layers. It is the layer-wise
  ancestor of the stability target we train π1-x on.
- CALM [Schuster2022] exits decoder layers per token using softmax confidence, hidden-state saturation or a trained classifier.

**Risk-controlled early exit.** Choosing exit thresholds with statistical guarantees is established:
- CALM calibrates its thresholds with Learn then Test [Schuster2022; Angelopoulos2025LTT].
- Fast yet Safe [Jazbec2024] selects early-exit thresholds with a UCB bound. It controls the performance gap to the full model with
  probability 1−δ under i.i.d. calibration data, and leaves distribution shift to future work.
- Conformal risk control [Angelopoulos2024CRC] controls the expected value of a monotone loss.
- Kubaty et al. [Kubaty2026] show that calibration alone does not characterise early-exit quality.

Our bootstrap-screened rule is none of these procedures. Its bound is asymptotic, uncorrected for the number of candidate thresholds, and
computed on the data that selected the candidate. Our results therefore inherit none of their guarantees (Section 3.4). Part of our
contribution is to show what happens on a sealed test set when such a heuristic screen is used instead.

**Monitoring and engineering evaluations of Ouro's exit.**
- Olszowiec et al. [Olszowiec2026] find that in most of eight looped models the large majority of generated tokens settle on their final
  top-1 prediction at the first iteration, and that Ouro's gate fires several iterations after the answer is fixed.
- An engineering evaluation of Ouro-1.4B's native exit on GSM8K reported a 7.35 pp accuracy loss at about 0.55 of the loops for one gate
  setting [vllmrlt56]. This is a GitHub issue, not peer-reviewed.

Both suggest that the native gate leaves savings unused in generation. Our setting, a single per-question decision in multiple-choice
scoring, is different and does not test that claim.

**Early exit of reasoning chains.** SpecExit [Yang2026], NEAT [Liu2026] and hidden-state correctness probes for chain-of-thought [Zhang2025]
shorten the generated reasoning text, not the latent depth. We cite them as adjacent work and do not compare numbers.

---

## Notes to author

1. **Correction to an earlier note.** The 51-point threshold grid was not adopted from Popescu et al. It comes from `gate1.grid_thresholds`
   (N_GRID = 50, quantiles from 0 to 1, commit ee89407, 2026-10-20), plus ∞. That predates the literature check of 2026-10-25. Popescu's grid
   is 51 quantiles between the 0.5th and 99.5th percentiles (appendix B.5). What was adopted from Popescu is the readout definitions
   (appendix B.6, cited in `phase3_readouts.py`). The text above states this accurately.
2. TaH2 (You et al., arXiv 2609.35748) is not cited: its full text could not be opened to confirm the label ablation. Think-at-Hard (Fu et al.,
   arXiv 2511.08577) is cited instead for the per-token decider.
3. New citation keys: [Bae2025] Mixture-of-Recursions (NeurIPS 2025); [Fu2025] Think-at-Hard (preprint); [Pappone2025] Two-Scale Latent Dynamics
   (preprint); [Schuster2022] CALM (NeurIPS 2022); [Olszowiec2026] (preprint); [Zhang2025] (preprint); [vllmrlt56] (GitHub issue, cite with
   access date).
4. Pin the Ouro version cited: v1 (2025-10-29) versus v6 (2026-10-11). The results in section 5.4.1 may differ between versions.
