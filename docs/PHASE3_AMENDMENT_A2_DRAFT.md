# Phase 3 amendment A2: published post-hoc readouts as baselines (DRAFT for review; nothing frozen)

This amends `docs/PHASE3_PLAN.md` v1.1 and `docs/PHASE3_AMENDMENT_A1.md` (tag phase3-a1). Neither file is edited.

**Timing.** No Phase 3 outcome exists. The full extraction has not run, and nothing has been computed on DEV+ or TEST. The
amendment comes from the literature check (`docs/PHASE3_LITERATURE.md`), not from any result.

## A2.1 Why
- The frozen H3 comparator is "the strongest eligible task-aware baseline", drawn from B_pb and native-pb (plan section
  5a).
- The closest prior work (Popescu et al. 2026, arXiv 2607.20519) evaluated the same released Ouro-1.4B on the same six
  benchmarks with the same shot counts. Simple post-hoc readouts there matched or beat Ouro's gate on several benchmarks;
  hidden cosine and hidden L2 were most often the best.
- Ouro itself (Section 5.4.1) proposes the hidden-state difference threshold.
- A Phase 3 PASS that did not beat these published rules could not claim decision value beyond known methods. A2 therefore
  adds them to the baseline pool. That only **raises** the bar; it cannot make a PASS easier.

## A2.2 The readouts (closed list)
All are computed from records the extraction already stores. No new GPU work is needed. Let h_t be the stored hidden state
after loop t at the last context token, and p_t the option distribution at loop t.

| id | statistic s_t at loop t (halt when s_t ≥ τ) | defined at | source |
|---|---|---|---|
| r-hl2 | −‖h_t − h_{t−1}‖₂ | t = 2, 3 | Ouro 5.4.1 (hidden-state difference threshold) |
| r-hcos | cos(h_t, h_{t−1}) | t = 2, 3 | Popescu et al. (hidden cosine) |
| r-kl | −KL(p_{t−1} ‖ p_t), the stored `D` of loop t−1 | t = 2, 3 | predictive KL readout (Gate 2's deployable KL rule) |
| r-conf | max_k p_t(k), the stored `conf` | t = 1, 2, 3 | top-1 probability |
| r-margin | p_t(1st) − p_t(2nd), from the stored `p` | t = 1, 2, 3 | margin |
| r-ent | −H(p_t), the stored `ent` | t = 1, 2, 3 | entropy |

- The convergence readouts are undefined at loop 1, because no h_0 or p_0 is stored. Under them a question cannot halt at
  loop 1 by score; it can stop there only through the cap m_b = 1. This is fixed now.
- h_t is the float16 copy of the bf16 state, as stored. Ouro does not state which position its Δh uses. We use the last
  context token, where all options branch, and say so in the report.

## A2.3 How they are selected and used (the same discipline as every other policy)
- **Task-aware form:** each readout is a policy in the plan's section 5 class. Per benchmark there is a threshold τ_b on the
  51-point grid (50 quantiles of the readout's pooled selection-data statistic over ok questions and the loops where it is
  defined, plus +∞) and a cap m_b ∈ {1, 2, 3, 4}. The selection rule, ties and fallback are those of plan section 5, on the
  same selection data (Phase 1 train plus DEV+). Readouts have no fitted parameters, so cross-fitting does not apply; their
  (τ_b, m_b) are chosen on the full selection data, and this is stated. τ_b = +∞ nests B_pb exactly, as for every policy.
- **Comparator (amends plan section 5a):** the strongest eligible task-aware baseline is taken over **B_pb, native-pb and
  the six task-aware readouts**. It is re-determined inside every bootstrap resample, as already specified.
- **Net saving:** every readout's ρ_b is measured like any other controller's (section 5a). The two hidden-state readouts
  need only a vector norm. The probability readouts need the option scores at each loop, which the learned policies need
  as well.
- **Task-agnostic track and OOD (plan section 8):** the pooled form of each readout (one (τ, m), constraint over all six
  training benchmarks) joins the task-agnostic baselines. The strongest of them joins the pooled fixed depth and pooled
  native gate as the OOD comparator.

## A2.4 Choice for the reviewer: the D2 screen
- Frozen D2 stops Phase 3 if the nested-train improvement over **B_pb** is below 0.02. If the readouts beat B_pb on
  development data, D2 could pass while H3 is bound to fail against a readout.
- **Option (a), recommended:** D2's comparator becomes the strongest of B_pb and the six task-aware readouts, chosen the
  same nested way. The threshold stays 0.02. This screens for what H3 actually requires.
- **Option (b):** keep D2 against B_pb, and report D2 against the strongest readout as information only.
- D1 (the oracle ceiling over B_pb) is unchanged under both options, because it bounds headroom rather than comparing
  policies.

## A2.5 What A2 does not change
- The candidate policies (π1, π3, π4 × x, hx), their features and hyperparameters, the allocation, TEST and DEV+, the
  accuracy rule, the 0.03 margin, the bootstrap, and the verdict table.
- The readouts are baselines, not candidates. A readout can never be "the selected policy".
- H3-h (hidden state versus cheap features) still compares the selected policy with its twin.

## A2.6 Reporting
- Every readout's selection-data and (at G3c) test saving and loss are reported per benchmark, beside B_pb and native-pb.
- So is the identity of the strongest baseline in the point estimate, and how often each baseline is the strongest across
  bootstrap resamples.
