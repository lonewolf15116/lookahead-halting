# Phase 3 amendment A2: published post-hoc readouts as baselines (FROZEN, tag `phase3-a2`)

**Status: frozen.** Reviewer decision "Phase 3 Amendment A2 — Revision 2 Review" (2026-10-26): APPROVED, GO for freeze. This
approval does not establish that the implementation is correct or that the Phase 3 hypothesis succeeds. Phase 3 has produced
no experimental outcome. Full extraction stays on HOLD until the Stage-build report has been reviewed, and D1-D4 and
training stay on HOLD until extraction is verified. Later changes need a further documented amendment.

**Interpretation the reviewer asked to keep explicit.** The D2 comparator is the strongest baseline selected using the
training (selection) folds. It is not necessarily the strongest baseline on the held-out fold.

**Implementation checks retained at freeze** (tests added in `tests/test_phase3_readouts.py`, 13 passing):
1. **Grid nesting.** τ = +∞ with cap m reproduces fixed depth m for every ok question and every cap m ∈ {1,2,3,4},
   including statistics that are undefined at loop 1. The selected point is never worse than the best qualifying fixed
   depth, so every readout nests every per-benchmark fixed-depth vector, B_pb included.
2. **Eligibility.** Eligibility means "qualified under the per-benchmark ≤ 1 pp rule on the selection (development) data".
   Because τ = +∞ is in the grid and fixed depth 4 has zero loss, **every readout and B_pb always qualify at selection**. So
   eligibility can never exclude a baseline: the comparator is effectively the best of all baselines on the selection data.
   A baseline that exceeds 1 pp on a held-out fold or a TEST benchmark is not removed: it still sets the bar, as in plan
   section 5a and A2.5. A test checks that held-out values never enter the eligibility flag or the comparator choice.

**Frozen artifacts (sha256):**
| file | sha256 |
|---|---|
| src/lookahead/phase3_readouts.py | 1b3a88a3519df5fd913340c828b88d158db59f7c18df1111fd3b5fdd0e372389 |
| tests/test_phase3_readouts.py | a7358555488982bfcf2c4a40908008f13ada5eb4a2165e9696d99925f66a6fe5 |
| docs/PHASE3_LITERATURE.md (motivation) | 087f9ebe17f4072e90108677530d65cd44bcb9835dfd8e4964e53e0c4bb358c6 |
| docs/PHASE3_AMENDMENT_A1.md (amended, unchanged) | 6b7e73284f40e8fe93854dfe6097bb4017b3a63b3ab9c3e5d0665908965231bf |
| docs/PHASE3_PLAN.md v1.1 (amended, unchanged) | e599f07c688f040467494a42d7efcaa36e13f437502da860ec8952944053b28f |

This amends `docs/PHASE3_PLAN.md` v1.1 and `docs/PHASE3_AMENDMENT_A1.md` (tag phase3-a1). Neither file is edited.

**Rev 2** applies the reviewer's "Phase 3 Amendment A2 — Reviewer Decision" (CONDITIONAL GO, freeze on HOLD). The five
required revisions are:
1. D2 uses a nested, held-out comparison against the strongest eligible baseline, with native-pb included (A2.4).
2. The claim that readouts need no cross-fitting is withdrawn (A2.3).
3. The loop-1 behaviour is frozen, and the stored signs and indexing are verified (A2.2, A2.7).
4. The mapping from the published definitions to our implementation is documented, together with three corrections it
   forced (A2.2).
5. The bootstrap comparator estimand is clarified (A2.5).
D1 also gets a non-binding companion quantity (A2.4).

**Timing.** No Phase 3 outcome exists. The full extraction has not run, and nothing has been computed on DEV+ or TEST. The
motivation is the literature check (`docs/PHASE3_LITERATURE.md`), not any result.

## A2.1 Why
- The frozen H3 comparator (plan section 5a) is the strongest eligible baseline among B_pb and native-pb.
- The closest prior work evaluated simple post-hoc readouts on the same released Ouro-1.4B, on the same six benchmarks
  with the same shot counts. That is Popescu, Sáez de Ocáriz Borde and Liò, arXiv 2607.20519, July 2026. The readouts
  matched or beat Ouro's gate on several benchmarks.
- Ouro itself (Section 5.4.1) proposes a hidden-state difference rule.
- A2 adds these readouts to the baseline pool. That can only **raise** the bar for a PASS.

## A2.2 The readouts: published definition, our operationalization, and the corrections it forced
Records stored per question and loop t = 1..4:
- p_t, the option distribution: the softmax of the option scores (`p`).
- z_t, the option scores: `opt_lln` for acc_norm tasks and `opt_ll` for acc tasks, i.e. exactly the scores whose softmax is
  p_t.
- h_t, the float16 copy of the hidden state at the **last context token**, where all options branch.

Every statistic is oriented so that the policy halts when s_t ≥ τ.

| id | published definition (source) | our s_t | defined at |
|---|---|---|---|
| r-conf | C_t = max_k p_t(k) (Popescu App. B.6) | max_k p_t(k) | loops 1-3 |
| r-margin | M_t = z_t,(1) − z_t,(2), the gap between the top two **logits** (B.6) | z_t(1st) − z_t(2nd) over option scores | loops 1-3 |
| r-ent | H_t = −Σ p log p; exit when H_t ≤ γ (B.6) | −H(p_t) | loops 1-3 |
| r-kl | ΔKL(t) = KL(p_t ‖ p_{t−1}), current versus previous loop (B.6) | −KL(p_t ‖ p_{t−1}), recomputed from the stored p | loops 2-3 |
| r-dlogit | Δlogit(t) = ‖z_t − z_{t−1}‖₂ (B.6) | −‖z_t − z_{t−1}‖₂ over option scores | loops 2-3 |
| r-hl2 | Δh_t = ‖h_t − h_{t−1}‖₂ < ε (Ouro 5.4.1); Δhid on a "pooled hidden representation" (B.6) | −‖h_t − h_{t−1}‖₂ at the last context token | loops 2-3 |
| r-hcos | "hidden cosine" is named in Popescu's tables but **not defined** in the paper | cos(h_t, h_{t−1}) | loops 2-3 |

**Corrections against draft rev 1**, each found by checking the published definitions:
- **KL direction.** Rev 1 used the stored `D`, which is KL(p_{t−1} ‖ p_t). The paper's direction is KL(p_t ‖ p_{t−1}),
  and rev 2 recomputes it from the stored p. A test asserts that the two directions differ, so the stored field cannot be
  used by mistake.
- **Margin.** Rev 1 used a probability gap. The paper defines a logit gap, and rev 2 uses the option-score gap.
- **Logit change.** The paper lists it and rev 1 omitted it. It is added, so the closed list now has **seven** readouts.

**Operationalization, not reproduction.** The paper does not specify:
- for Ouro, whether its distributions run over the vocabulary or the answer options;
- the layer, token position and pooling of its hidden representation;
- the form of "hidden cosine";
- its multiple-choice scoring.
Ouro 5.4.1 does not state the token position either. Ours are option-level distributions under our lm_eval-matched
scoring (acc or acc_norm per benchmark), the last context token, and cosine of consecutive states. These are reported as
**our operationalization of the published readouts**, not as exact reproductions.

**Exit rule (frozen; the reviewer's wording):**
- At loop 1 the convergence statistics (r-kl, r-dlogit, r-hl2, r-hcos) are unavailable.
- If the cap m_b = 1, the policy stops at loop 1. Otherwise it continues to loop 2.
- At loops 2 and 3, it halts if s_t ≥ τ_b.
- At the cap m_b, it stops regardless of the threshold.
- The confidence readouts (r-conf, r-margin, r-ent) may fire from loop 1, as in the paper ("min{t : S_t ≥ γ}"). The
  convergence readouts fire only from loop 2, as in the paper ("min{t ≥ 2 : Δ(t) ≤ ε}").
- An undefined statistic never fires. A cosine involving a zero vector is undefined.
- Failed questions run 4 loops and count wrong.

## A2.3 Selection: nested, like every other threshold-based controller (corrected)
Rev 1 said that readouts need no cross-fitting because they have no fitted parameters. **That is withdrawn.** A readout's
(τ_b, m_b) are chosen from outcomes, so they are fitted.

- **Grid and rule.** The grid has 51 points: 50 quantiles of the readout's statistic, pooled over ok questions and the loops
  where it is defined (exact repeats removed), plus +∞. The rule is that of plan section 5: per benchmark, max saving
  subject to loss ≤ 1 pp. Ties go, in order, to smaller loss, then fixed depth (τ = +∞), then larger τ, then smaller m. Since
  τ = +∞ is in the grid, every readout nests B_pb exactly.
- **In D2 (G3a):** each readout's (τ_b, m_b) are selected on the four selection folds and evaluated on the fifth, rotated,
  exactly as for π1 and B_pb.
- **For G3c:** after G3b, each readout's (τ_b, m_b) are re-selected once on the full permitted development data (Phase 1
  train plus DEV+). They are frozen with sha256 beside the candidate policies, before any TEST outcome.

## A2.4 D2 and D1 (G3a)
**D2 comparator (option (a), with the reviewer's correction).** In each outer fold k:
1. Every baseline's (τ_b, m_b) is selected on the folds other than k. The baselines are B_pb, native-pb and the seven
   readouts.
2. A baseline is **eligible** if it qualifies there under the same preregistered per-benchmark rule as the candidate. B_pb
   always qualifies, because fixed depth 4 has zero loss.
3. The **strongest eligible baseline for fold k** is the eligible one with the highest macro net saving **on the selection
   folds**, with ties broken by name. Only its **held-out** saving on fold k enters the comparison.
4. The D2 improvement is the mean over folds of (candidate held-out macro net saving − chosen comparator's held-out macro
   net saving). The candidates are π1-x and π1-hx, as frozen.

The threshold stays at **0.02**. Because no held-out value influences which comparator is chosen, the choice adds no
optimism. As information only, D2 also reports the improvement over the best baseline chosen on held-out values, which is
conservative.

**D1.** Unchanged and binding: ceiling (ii) − B_pb < 0.05 stops Phase 3. Beside it, a **non-binding** quantity is reported:
Δ_oracle-readout = S_oracle(ii) − S_best eligible baseline, where the best eligible baseline is chosen as in D2. It gives
context on headroom over the readouts and never stops or continues anything. The oracle remains a retrospective diagnostic,
never a deployable competitor.

## A2.5 H3 comparator and the bootstrap estimand (clarified)
- **Pool.** The strongest eligible task-aware baseline is drawn from B_pb, native-pb and the seven task-aware readouts.
  Eligible means qualified at selection on the development data. As in plan section 5a, a baseline that exceeds 1 pp on a
  TEST benchmark still sets the bar.
- **Verdict rule (frozen, preserved).** The effect criterion uses the paired bootstrap, with the strongest eligible baseline
  **re-selected inside every resample**.
- **Two quantities, reported separately:**
  - **(i) Point estimate:** Δ = S_net,π − S_net,c*, where c* is the strongest eligible baseline on the original TEST sample.
    A fixed-comparator 95% CI is reported beside it, with c* held fixed across resamples, as **secondary**.
  - **(ii) Re-selecting bootstrap:** the distribution of S_net,π − max over eligible c of S_net,c. It is an interval for the
    **difference to the best of several baselines**, not an ordinary fixed-comparator confidence interval, and is described
    as such. Its lower bound decides the effect criterion, as frozen.
- **Selection frequency.** How often each baseline is the strongest across resamples is reported.

## A2.6 Net saving, task-agnostic track, OOD
- **Net saving.** Each readout's ρ_b is measured with the same convention as the candidate policies (plan section 5a: same
  GPU, batch size 1, same code path, measured once on development data at G3b and frozen).
  - Hidden readouts: the vector norm or cosine on the GPU tensor.
  - Probability and score readouts: the option scoring at each loop, as the learned policies also need.
- **Task-agnostic and OOD.** The pooled form of each readout (one (τ, m), constrained over all six training benchmarks) joins
  the task-agnostic baselines. The strongest of them joins the pooled fixed depth and pooled native gate as the OOD
  comparator.

## A2.7 Implementation and tests
- `src/lookahead/phase3_readouts.py` contains the statistics, the exit rule, the grid, the per-benchmark selection and the
  D2 `nested_comparator`. It reads no data.
- `tests/test_phase3_readouts.py` has 11 tests, all passing:
  - Signs and indexing are checked against the extractor's own `option_record` and `kl_logp`, for acc and acc_norm. Stored
    `conf` = max p_t; stored `ent` = H(p_t) ≥ 0; stored `D` of loop t−1 = KL(p_{t−1} ‖ p_t), so r-kl must not use it.
  - The hidden readouts: values, undefined at loop 1, and a zero-vector cosine.
  - The exit rule: undefined at loop 1 means continue; cap 1 stops at loop 1; the cap stops regardless; τ = +∞ gives fixed
    depth; failed questions run 4 loops.
  - Grid construction, selection ties and budget, and that the nested comparator chooses on selection folds and never on
    held-out values.
- Nothing here has touched real outcomes.

## A2.8 What A2 does not change
- The candidate policies, features and hyperparameters.
- The allocation, TEST and DEV+.
- The accuracy rule, the 0.03 margin, the D1 threshold and the D2 threshold.
- The bootstrap settings and the verdict table.
- Readouts are baselines only. A readout can never be the selected policy. H3-h still compares the selected policy with its
  twin.

## A2.9 Reviewer decision
Approved for freeze (Revision 2 Review, 2026-10-26). Every item was accepted: the seven readouts, the corrected KL direction
and margin, logit change, the loop-1 and cap behaviour, the nested D2 selection including native-pb, D1 binding against
B_pb, the distinguished bootstrap estimands, and the pooled OOD readout baselines.
