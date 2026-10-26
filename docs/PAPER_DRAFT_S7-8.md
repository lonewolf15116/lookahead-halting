# Draft: Sections 7–8 (discussion, limitations, threats to validity, conclusion)

Evidence labels as in the outline: [C] confirmatory, [S] pre-specified secondary, [P] post hoc emphasis or interpretation, [E] exploratory.
Every number is taken from Sections 5–6 of `docs/PAPER_DRAFT_S3-6.md` and inherits that file's provenance table.

---

## 7. Discussion

### 7.1 What the results support

The evidence supports three statements, each at a different strength.

1. **Under the bootstrap-screened rule, π1-x saved more loops than Ouro's native gate on the three primary benchmarks** [C]. The difference
   was +0.053 in gross macro saving, 95% CI [+0.049, +0.057], and +0.038 after the estimated probe overhead [S].
2. **The preferred method depends on the selection rule** [C] together with [S]. Under the point-only rule, the native gate was preferred on development data (G3a, −0.024 [E])
   and on TEST (−0.022 [−0.026, −0.018] [S]). Under the bootstrap-screened rule, π1-x was preferred [C]. Each rule selects its own
   operating points, so this is a statement about comparisons made through a selection procedure, not about fixed policies changing order.
   The point-only preference for the native gate on TEST compares operating points that broke the observed 1 pp budget on TEST: native-pb on two of
   three benchmarks, π1-x on all three (Table 5). That comparison therefore shows which method a point-only rule prefers. It does not show which method
   is better among budget-compliant policies.
3. **The added trained probe showed no demonstrated advantage over an untrained KL-between-loops readout** [S], and this comparison is
   emphasised post hoc [P]. Separately, one out-of-distribution benchmark showed a budget failure for π1-x [S].

What the evidence does not support:
- a claim that π1-x is a better halting signal than prior art;
- a claim that π1-x and r-kl are equivalent;
- any statement about latency or about Ouro's released inference path;
- any statement about models, loop counts or task formats other than those tested;
- any general statement about robustness to distribution shift.

### 7.2 Relation to prior work

**Post-hoc readouts on Ouro.** Popescu et al. [Popescu2026] already reported that post-hoc trajectory readouts can match or beat Ouro's
pretrained gate at validation-selected operating points. They describe the gate as "competitive but not uniformly Pareto-optimal" and
conclude that adaptive depth is "a trajectory–readout problem, not a gate-learning problem". Our primary result is consistent with that view
and should be read as a pre-registered, sealed-test instance of it under one selection rule, not as a new finding that readouts can beat the
gate. What we add is the following:
- the preference depends on the selection rule (a point-only rule favours the native gate on our data);
- the readout that matches the trained probe is a divergence between loops rather than a confidence readout;
- a development-selected policy can fail its budget on an out-of-distribution benchmark.

**Risk control for early exit.** Fast yet Safe [Jazbec2024] selects early-exit thresholds with a formal guarantee (a UCB bound at level 1−δ under
i.i.d. calibration data). Learn then Test [Angelopoulos2025LTT] and Conformal Risk Control [Angelopoulos2024CRC] give the general machinery. Our
selection rule is not one of these procedures and inherits none of their guarantees (Section 3.4). The TEST upper bounds that exceeded 1 pp
(Section 5.3) are what one might expect from a heuristic screen [P]. They argue for using a formal procedure, not for treating our screen as
one. A natural follow-up is to repeat the comparison with LTT or UCB selection, on calibration data separate from the data used for fitting.

**Calibration and ranking quality.** Kubaty et al. [Kubaty2026] argue that calibration alone does not characterise early-exit quality. Our
exploratory data fit with this. Better ranking of the stability label (π1-x AUROC 0.689 against 0.603 for the native CDF [E]) did not stop
the native gate from winning under the point-only rule [E], so ranking quality alone does not predict which method a given selection rule will prefer.

**Convergence exits.** Our r-kl is close in spirit to the KL convergence exit of Geiping et al. [Geiping2025]. It differs in unit (the question,
over answer options), in the distribution compared, and in threshold choice (selected per benchmark, not fixed). We therefore claim no
reproduction of their exit. The near-tie shows that a convergence statistic was as useful as a trained probe in our setting [S], [P].

### 7.3 Why the preference might depend on the rule

We have no causal analysis. The following are hypotheses for future work [P]:

- **Resolution near the boundary.** The two rules accept different regions of the threshold grid. The point-only rule accepts riskier operating
  points, and there the native CDF reached higher savings (macro 0.359 against 0.337). Adding the upper-bound screen removed those points for both
  methods, and the native gate lost more (−0.173 against −0.098). A statistic with coarse resolution near the acceptance boundary would behave this
  way. We did not test this.
- **Variance of the bound.** The bootstrap upper bound penalises policies whose loss is driven by few questions. Policies that exit early on a
  small, consistent subset may pass the screen more easily than policies that exit early on many questions with small per-question risk.
- **Selection noise.** Both rules select one of 51 thresholds per benchmark on the same development data used to judge them. Part of any
  difference between rules may be selection noise rather than a property of the signals. The disagreement between the in-sample freeze
  numbers and the cross-fitted E1 estimate for r-kl (Section 6.1) shows that this noise is not negligible.

### 7.4 Implications for evaluating halting methods

1. Treat a halting comparison as a comparison of signal plus selection procedure. Report results under more than one operating rule, and report
   the selected operating points, not only the macro saving.
2. Report upper bounds on TEST alongside point losses. In our study the point budget held everywhere on TEST, but four of nine upper bounds did not.
3. Where a guarantee is wanted, use a formal procedure (LTT, CRC, UCB as in Fast yet Safe) with held-out calibration data, and state its assumptions.
4. Include strong untrained baselines, especially a divergence between successive loops, before attributing gains to a trained controller.
5. Include at least one out-of-distribution check of a development-selected policy. A policy can save the most precisely by halting where it should not.
6. Keep simulated loop savings apart from measured latency, and state whether policies were executed or evaluated from stored traces.

### 7.5 Limitations

- **One model, one size, four loops.** Every result is for Ouro-1.4B with 4 loops in bf16. Ouro-2.6B, more loops and other looped architectures were not tested.
- **Multiple-choice scoring, not generation.** Each question has one decision, at the last context token, over a packed sequence with all
  options. Per-token halting during generation, the setting of Ouro's released code and of Geiping et al., is a different problem.
- **Our decision unit is an adaptation.** Ouro's released code computes every loop and selects a readout for each token position. Our
  per-question stopping, with the skipped loops counted as saved, is not Ouro's mechanism, and our savings are not measurements of Ouro's
  released inference.
- **Retrospective evaluation and simulated savings.** Policies were applied to stored full-depth traces. Exactness under truncation rests on the
  structure of the code (Section 3.6), not on an executed truncated run. Savings are loop counts plus an overhead measured separately at freeze time.
  No end-to-end time was measured. A loop is assumed to cost the same whether or not later loops follow, and the native gate is assigned zero
  controller cost.
- **Numerical path.** The planned check of whether halting decisions agree across bf16 numerical paths (Phase 3 check D-b) never ran, because
  Phase 3 stopped at G3a. A bf16 difference between the packed and context-only passes is documented on the 0-shot sets. Its effect on halting
  decisions is unmeasured.
- **No formal guarantee.** The bootstrap screen is asymptotic, uncorrected for the 51 candidates, and computed on the data that selected the
  candidate.
- **Benchmark identity is used.** The primary policies select $(\tau_b, m_b)$ per benchmark, so a deployment would need to know the task. The
  out-of-distribution analysis used one task-agnostic pair and is the closer model of deployment. That is where π1-x failed.
- **Sample size.** HellaSwag has $n = 750$, which limits the precision of its upper bound. Every method, including fixed depth, exceeded 1 pp there.
- **Out-of-distribution scope.** There are two benchmarks, both 0-shot, which differ from the development data in format and in task at once,
  so the source of the Social IQa failure cannot be isolated.
- **The TEST set is spent.** No further confirmatory analysis can use it. Any claim about r-kl against the native gate, or about an
  out-of-distribution guard, needs new held-out data.

### 7.6 Threats to validity

- **Internal.**
  - r-kl was singled out using TEST outcomes. Its comparison with π1-x was pre-specified, but its prominence is not.
  - The in-sample freeze numbers and the cross-fitted E1 estimate disagree about r-kl.
  - The Phase 3 diagnostic on PIQA found a float32 difference between packed and context-only scoring, attributed to numerical causes (D-a2,
    float64 maximum difference 1.67×10⁻¹³), and was accepted as a documented deviation.
  - The console transcript of the TEST run was overwritten. The run's outputs are fully recorded in the report and the marker.
- **Construct.**
  - Gross saving counts loops, not FLOPs. Net saving uses ρ timed in isolation and is an accounting estimate.
  - The 1 pp budget is relative to loop-4 accuracy on the same questions, not to an external standard.
- **Statistical.**
  - The primary interval is a paired, benchmark-stratified percentile bootstrap. It does not reflect variability from refitting the probe or
    reselecting thresholds, so it conditions on the frozen policies.
- **External.**
  - The results cover one model family, multiple-choice scoring and five benchmarks.

## 8. Conclusion

On a frozen Ouro-1.4B, a pre-registered comparison on a sealed test set found that a trained stability probe saved more loops than Ouro's
native gate when operating points were chosen with a bootstrap-screened accuracy rule: +0.053 gross macro saving [C], +0.038 net (estimated) [S].
The same comparison under a point-only rule preferred the native gate, both on development data [E] and on the sealed test set [S]. On the test set,
however, the point-only operating points of both methods exceeded the observed 1 pp budget on most benchmarks, so the reversal concerns which method each
rule prefers, not which method is better among compliant policies.

The trained probe showed no demonstrated advantage over an untrained KL-between-loops readout. That comparison was pre-specified [S], and we
give it post hoc emphasis [P]. On one out-of-distribution benchmark the probe broke its accuracy budget [S].

These results are consistent with prior evidence that simple trajectory readouts compete with Ouro's learned gate [Popescu2026]. Their main
lesson is methodological: which halting method looks better depends on how operating points are selected. Halting methods should be compared
under explicit selection procedures, preferably ones with formal risk control, alongside strong untrained baselines and out-of-distribution
checks.

Future work needs new held-out data:
- a pre-registered comparison of KL exit and the native gate;
- formal risk-controlled selection;
- an out-of-distribution guard;
- Ouro-2.6B;
- an executed, truncated implementation with measured time.

---

## Notes to author

1. **[Popescu2026]** (Popescu, Sáez de Ocáriz Borde, Liò, *Adaptive Depth in Looped Transformers: Diagnosing Learned Halting Gates and
   Trajectory Readouts*, arXiv:2607.20519, 8 July 2026, a preprint) is verified from the arXiv PDF through a summarising fetch. It studies
   Ouro-1.4B and 2.6B. Correction: our threshold grid was not taken from it. It comes from `gate1.grid_thresholds` (Gate 1, 2026-10-20). Only the
   readout definitions follow Popescu (appendix B.6). See `PAPER_DRAFT_S1-2.md`, note 1. It must be cited in Section 2 as the closest prior work, and the introduction's novelty statement must be written with it in view.
2. The numbers in Section 7.3 (0.359 vs 0.337; drops of −0.173 and −0.098) come from Table 5 of Sections 5–6.
3. Section 7.6 attributes the D-a2 result to "numerical causes" and gives a float64 difference of 1.67×10⁻¹³. Check the exact wording against `docs/DECISIONS.md` / the D-a2 report before submission.
4. Sections 1 and 2 are not yet drafted. The abstract should be written last, keeping the [C]/[S]/[P] distinctions in its sentences.
