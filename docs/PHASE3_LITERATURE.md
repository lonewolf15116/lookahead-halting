# Lookahead halting: literature and novelty check (2026-10-26)

Scope: work on adaptive depth and early exit that a Phase 3 result would be compared with. This is a targeted search, not a
systematic review. Google Scholar was not queried, and several papers were read only in part (noted below). Nothing here
changes the frozen protocol. Section 3 feeds the proposed amendment A2.

## 1. The closest work

**Popescu, Sáez de Ocáriz Borde and Liò (Cambridge), "Adaptive Depth in Looped Transformers: Diagnosing Learned Halting
Gates and Trajectory Readouts", arXiv 2607.20519, July 2026.** This is the closest prior work. I read it through the PDF,
and Sections 5.4-5.6 only via a summary of the PDF.
- They evaluate the released **Ouro-1.4B and Ouro-2.6B**, unmodified, on **our six benchmarks with our shot counts**:
  MMLU 5, ARC-C 25, ARC-E 8, OBQA 0, HellaSwag 10, CSQA 10.
- They compare the pretrained ponder gate against post-hoc **threshold readouts**: entropy, top-1 probability, margin,
  predictive KL, logit change, and hidden-state L2 displacement and cosine change.
- Thresholds are selected on validation data (20% of validation for HellaSwag and CSQA) and evaluated on held-out data.
- **Finding:** the gate is "competitive but not uniformly Pareto-optimal". Simple readouts, most often hidden cosine or
  hidden L2, match or beat it on several benchmarks. Examples for 1.4B HellaSwag: hidden L2 gives 72.9% at 3.13 loops, the
  gate 72.8% at 3.94. Most results are single runs, and differences are often under 1 point.
- **What they do not do on Ouro:** they train no learned probe or forecaster on the Ouro trajectory (their learned gates on
  frozen trajectories are MANO-only). They report no per-benchmark fixed-depth baseline, no per-benchmark accuracy-loss
  budget, and no preregistered, fresh, one-time test.

**Ouro (ByteDance Seed and collaborators), "Scaling Latent Reasoning via Looped Language Models", arXiv 2510.25741, Section 5.4.1.**
Ouro compares static exit, the learned Q-exit gate (exit when the cumulative exit probability passes q, with and without
the adaptive-exit loss), and a **hidden-state difference threshold**: exit when Δh_t = ‖h_t − h_{t−1}‖₂ < ε. The
comparison is on MMLU only. The trained gate is best at every budget, and the hidden-state rule is within about 1-2 points
of it at 2-3 average loops.

**You et al. (Tsinghua and Yale), "TaH2 / Improving Test-Time Scaling with Adaptive Looped Transformers", arXiv
2609.35748, September 2026.**
- A learned per-token "decider" chooses whether to run another iteration.
- It is trained on **gain-based labels**: the measured change in next-token loss from one more iteration. Replacing these
  with top-1-mismatch labels lowers accuracy (15.5% → 14.4%).
- The backbone and decider are trained jointly (post-trained Qwen3-1.7B). Ouro is used only as a baseline.
- So "predict the benefit of another loop" is published, but for a jointly trained model, not for a frozen pretrained
  looped LM.

**Olszowiec et al. (Samsung AI Center Warsaw), "Monitoring latent reasoning in looped language models", arXiv 2610.18127,
October 2026.** It uses no probes and tests no stopping. Its relevant findings:
- In most models, 76-89% of tokens are already settled at the first iteration.
- Ouro's exit gate fires 3.2-4.1 iterations after the answer is fixed.
- Extra iterations can hurt (the Ouro-Thinking variants are best at iteration 1).

**vllm-rlt issue #56 (InferMatrixLab, September-October 2026).** An engineering evaluation of Ouro-1.4B's native exit on
GSM8K with a pre-registered rule. At τ = 0.2, the native synchronous exit loses 7.35 pp for 0.55× the loops. A "trained
lookahead gate" is explicitly a non-goal.

**Earlier lines that any write-up must cite:**
- Halting and adaptive computation: ACT (Graves 2016); PonderNet (Banino et al. 2021). These, PABEE and Geiping et al. are cited from memory; their references must be checked before any write-up.
- Early exit in non-looped transformers: patience and stability exits such as PABEE (Zhou et al. 2020); CALM (Schuster
  et al., NeurIPS 2022), which uses softmax confidence, hidden-state saturation and a learned early-exit classifier with
  calibrated thresholds.
- Recurrent depth and adaptive recursion: Geiping et al. 2025 (recurrent depth with convergence-based exit);
  Mixture-of-Recursions (Bae et al., arXiv 2507.10524), which uses learned per-token routers over recursion depth.
- Hidden-state probes that forecast correctness in order to stop early, in chain-of-thought rather than loops: "Reasoning
  Models Know When They're Right" (Zhang et al., arXiv 2504.05419).

## 2. What this means for novelty

**Not novel by itself:**
- Halting a looped LM early.
- Using hidden-state convergence or confidence to do it (Ouro 5.4.1; Popescu et al.; CALM).
- Learning to predict the benefit of one more iteration (TaH2, which uses gain labels).
- Probing hidden states to forecast correctness (Zhang et al.).

**Defensible as new**, to our knowledge and subject to a fuller search, is the combination Phase 3 tests:
1. A **frozen** pretrained looped LM, with no retraining of the backbone or gate.
2. **Learned forecasters** of stability, final-loop gain and sequential value-to-go (π1, π3, π4), trained on the loop-t
   state.
3. A comparison against a **task-aware fixed-depth baseline**. Gate 2 found that this baseline (B) beat the probes and
   Ouro's gate. Popescu et al. report no such control.
4. **Per-benchmark accuracy budgets**, with net saving that includes controller overhead.
5. A **preregistered, one-time, fresh test**, with transfer to unseen benchmarks reported separately.

**Gate 2's negative result has value given this literature.** Popescu et al. conclude that simple readouts often match
learned gates. Gate 2 adds two things on the same model and benchmarks: a learned hidden-state probe did not beat
cheap-feature probes, and benchmark identity alone (fixed per-benchmark depth) explained most of the attainable saving. A
careful negative result with that baseline is publishable as an empirical study. It is not a new algorithm.

**Risk.** If Phase 3's forecasters do not beat the post-hoc readouts that Popescu et al. already published, a Phase 3 PASS
against B_pb and the native gate alone would overstate novelty. That is the reason for amendment A2.

## 3. Consequence for Phase 3 (proposed, not adopted)
The frozen comparator set (plan section 5a: B_pb and native-pb) leaves out the readouts that the closest prior work found
strongest on this exact model and these benchmarks. Draft A2 (`docs/PHASE3_AMENDMENT_A2_DRAFT.md`) proposes adding them as
task-aware baselines under the same selection discipline, before any Phase 3 outcome exists.

## Sources
- [Popescu et al., arXiv 2607.20519](https://arxiv.org/html/2607.20519v1) · [PDF](https://arxiv.org/pdf/2607.20519v1)
- [Ouro, arXiv 2510.25741](https://arxiv.org/abs/2510.25741) · [PDF](https://arxiv.org/pdf/2510.25741)
- [TaH2, arXiv 2609.35748](https://www.alphaxiv.org/abs/2609.35748)
- [Olszowiec et al., arXiv 2610.18127](https://arxiv.org/html/2610.18127)
- [vllm-rlt issue #56](https://github.com/InferMatrixLab/vllm-rlt/issues/56)
- [CALM, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/6fac9e316a4ae75ea244ddcef1982c71-Abstract.html)
- [Mixture-of-Recursions, arXiv 2507.10524](https://arxiv.org/abs/2507.10524v2)
- [Zhang et al., arXiv 2504.05419](https://arxiv.org/abs/2504.05419v1)
- [Two-Scale Latent Dynamics for Recurrent-Depth Transformers, arXiv 2509.23314](https://arxiv.org/pdf/2509.23314)
