# Phase 3 amendment A1: allocation, narrowed population, D1-D4 population (DRAFT for review; nothing frozen)

Amends the frozen plan `docs/PHASE3_PLAN.md` v1.1 (sha256 e599f07c688f…, tag phase3-plan-v1.1). The plan file is not
edited. A1 is frozen as its own file after reviewer approval, before any extraction or outcome analysis.

Inputs: inventory `results/phase3/inventory.json` (sha256 7ec501ae…, commit 848d1b9); reviewer decision "Phase 3
Inventory — Reviewer Decision" (2026-10-22). No Phase 3 outcome has been computed. No Gate 2 confirmation outcome is used.

**Disclosure.** Outcomes on Phase 1 train and validation for all six benchmarks were seen in Gates 1 and 2. For example, the
Gate 2 control audit gave within-benchmark AUROC of 0.94 for CommonsenseQA and 0.49-0.73 for HellaSwag. The narrowing below
is forced by the inventory (which benchmarks have unused data). It was not chosen from those outcomes, and it is fixed
before any Phase 3 diagnostic is run.

## A1.1 Two separate claims (never combined)
| claim | population | data | criterion |
|---|---|---|---|
| **H3, task-aware adaptation on known benchmarks** (primary) | OpenBookQA, CommonsenseQA, HellaSwag only | fresh in-distribution test (A1.3) | plan section 10 verdict table, with every "benchmark" read as these three |
| **H3-ood, transfer to unseen benchmarks** (separate) | the OOD benchmarks of A1.4, each on its own | OOD validation splits | plan section 8 criteria, per benchmark, never pooled |

H3-h and H3-agn (plan section 2) are evaluated on the H3 test set and population. Claims are limited to Ouro-1.4B and these
benchmarks. The H3 claim does **not** extend to ARC-Challenge, ARC-Easy or MMLU. Those three have no clean fresh rows: the
ARC train splits were used as few-shot exemplars, and the exemplar exclusion is kept strict by reviewer decision. MMLU's
splits were all used.

## A1.2 Primary population and what the six-benchmark parts of the plan now mean
- **Macro quantities** in H3, the final policy choice (plan section 6) and the G3a stop rules are taken over the three primary
  benchmarks. The thresholds are unchanged: D1 0.05, D2 0.02, effect margin 0.03, per-benchmark loss 1 pp.
- **Fitting** may still use all six benchmarks' Phase 1 train data, as Gate 2 did. Per-loop models gain from more data, and
  fitting uses no test row. Each benchmark's (τ_b, m_b) is selected on that benchmark's own data (plan section 5; the
  problem separates by benchmark). The three non-primary benchmarks therefore cannot change the primary benchmarks'
  thresholds, only the shared model weights.
- **Task-agnostic track and the unseen-benchmark rule** (plan sections 5 and 8): the single shared (τ, m) is selected under
  "every benchmark's loss ≤ 1 pp" over **all six** training benchmarks. It is meant to transfer, so the wider constraint is
  the relevant one. This is the rule applied to every OOD benchmark. No benchmark-specific value is ever assigned to an
  unseen benchmark.
- **Six-benchmark versions** of D1, D2 and the cross-fitted selections are reported beside the primary ones as information.
  They never decide anything.

## A1.3 In-distribution allocation (fresh rows; disjoint dev and test)
Every inventoried clean row of a primary benchmark is assigned to exactly one of **TEST** (one-time G3c test, sealed) and
**DEV+** (extra development data, used like Phase 1 train: fitting and cross-fitted selection, with folds assigned by the
frozen `lh-fold-v1` hash).

| benchmark | clean rows (inventory) | TEST | DEV+ | Phase 1 train (existing dev) | Phase 1 validation (exposed; reported only) |
|---|---|---|---|---|---|
| OpenBookQA | 4,952 | 3,000 | 1,952 | 300 | 100 |
| CommonsenseQA | 1,435 | 1,000 | 435 | 733 | 244 |
| HellaSwag | 676 under rule v1; **999** under the proposed rule v2 (below) | 750 (if v2) / 500 (if v1) | rest | 6,025 | 2,008 |

- **Selection procedure.** Rank the clean rows of each benchmark by sha256(`lh-phase3-alloc-v1|benchmark|task|row|text_hash`),
  ascending. The first N go to TEST and the rest to DEV+. Frozen with A1: the exact TEST and DEV+ id lists (CSV),
  their sha256, the inventory sha256 and the dataset revisions (A1.6).
- **HellaSwag grouping.** HellaSwag rows that share a `source_id` (the same video or article, different segment) are
  allocated as one group, keyed by the hash of their smallest row. This keeps TEST and DEV+ from sharing a source. The TEST
  count is then the smallest number of whole groups reaching ≥ N. The other two datasets have no source key.
- **HellaSwag repeated-id audit (done, `results/phase3/hellaswag_id_audit.json`).** HellaSwag's train ids come in at most
  pairs: 32,599 distinct ids over 39,905 rows, with 14,612 rows in repeated-id groups. All **820** id-only exclusions are
  **unrelated** to the used row that shares their id: different context, different source video or article, and different
  activity in every case (0 of 820 share any of the three). They are identifier collisions, not overlaps. The audit also
  found a different and real proximity risk that the v1 rule missed. Among rows never used by text, 488 share a
  `source_id` with a used row (another segment of the same video or article), and 16 share a context.
- **Proposed audit rule v2 for HellaSwag** (reviewer decision; applies to HellaSwag only, because only its ids repeat):
  1. Drop the native-id criterion, since its matches are collisions.
  2. Keep the text-hash criterion.
  3. **Add** a source criterion: exclude any row that shares a `source_id` or context with a used row.
  This is stricter on what matters (no segment of a used video or article enters the test) and less strict on what does not
  (an id number). It gives **999** clean rows under the within-train check, against 676 under v1. Before freezing, the
  allocation script must apply the source criterion against **every** used HellaSwag row, including the Phase 1 validation
  split (the audit compared within train only). So the final count can only fall from 999, and it is recorded at freeze.
- **HellaSwag TEST size.** If v2 is accepted: 750 TEST (two-sided paired half-width about 2.3 pp at d = 0.10, against 2.8 pp at
  500), and the rest DEV+. If v1 is kept: the reviewer's 500 of 676.
- **Why DEV+ is needed (reviewer concern on independent development data).** The binding case is not HellaSwag but
  **OpenBookQA**. It has only 300 Phase 1 train questions, so 1 pp is 3 questions, and inside a nested 4-of-5-fold D2 it is
  about 2. Per-benchmark thresholds selected at that resolution are unreliable. With DEV+ the OpenBookQA development set
  is 2,252 questions (1 pp ≈ 22 questions). CommonsenseQA goes from 733 to 1,168. HellaSwag's development data is
  essentially Phase 1 train (6,025), and its small remainder is not treated as a resource.
- **Cost.** DEV+ and TEST together come to about 7,063 questions to extract, roughly 50 GPU-minutes at the measured
  ~35 min per 5k. TEST outcomes are extracted but stay sealed, as the Gate 2 confirmation outcomes were.

## A1.4 OOD allocation
| benchmark | test split | rows | status |
|---|---|---|---|
| PIQA | validation | 1,837 | **primary OOD** |
| Social IQa | validation | 1,930 | **primary OOD**; promoted from reserve, because WinoGrande is excluded (below) |
| WinoGrande | validation | 1,267 | **excluded, a pipeline incompatibility found before any outcome** |

- **WinoGrande exclusion.** lm_eval scores WinoGrande by substituting each option into the context and scoring a shared
  continuation, so the options do not share one context. Our packed scoring and the h_t features assume one context per
  question; `load_lm_eval_questions` raises "options do not share one context" (checked on a local WinoGrande-format task
  with lm_eval's own preprocessing). Supporting it needs a new scoring path and a new feature definition (whose h_t?).
  That is a design change, not an evaluation. PIQA and Social IQa build without error.
- **OOD train splits are not used at all in Phase 3.** The reviewer note suggested keeping them "for development if
  needed". Any use of them would make these benchmarks seen, and turn the transfer test into an in-distribution test. So
  they stay sealed.
- **Shots.** OOD benchmarks are evaluated 0-shot, which is lm_eval's default for PIQA and Social IQa. That avoids drawing
  exemplars. The in-distribution benchmarks keep their Ouro shot counts. The resulting prompt-format difference is part of
  the distribution shift and is stated as such.
- **Policy on an unseen benchmark:** the task-agnostic track's frozen statistic and (τ, m) (A1.2), applied unchanged. The
  baselines are the pooled fixed depth and the pooled native gate frozen on train (plan section 8).

## A1.5 Observed versus demonstrated per-benchmark compliance
- **Observed compliance** (primary, unchanged from plan section 4): the point estimate loss_b ≤ 1.00 pp. The verdict table uses
  this.
- **Demonstrated compliance** (reported per benchmark, never changes the verdict): the paired-bootstrap one-sided 95% upper
  bound of loss_b ≤ 1.00 pp.
- **Feasibility, stated before the test.** If the true loss is 0, the one-sided upper bound is about 1.645·sqrt(d/n), where d
  is the fraction of questions whose correctness differs between the policy and loop 4. Demonstrating ≤ 1 pp then needs
  n ≥ 1,354 at d = 0.05, or n ≥ 2,707 at d = 0.10. So at the proposed sizes:

| TEST set | n | upper bound at true loss 0, d = 0.05 / 0.10 | two-sided half-width, d = 0.05 / 0.10 |
|---|---|---|---|
| OpenBookQA | 3,000 | 0.67 / 0.95 pp | 0.80 / 1.13 pp |
| CommonsenseQA | 1,000 | 1.16 / 1.65 pp | 1.39 / 1.96 pp |
| HellaSwag | 500 | 1.65 / 2.33 pp | 1.96 / 2.77 pp |
| PIQA (OOD) | 1,837 | 0.86 / 1.21 pp | 1.02 / 1.45 pp |
| Social IQa (OOD) | 1,930 | 0.84 / 1.18 pp | 1.00 / 1.41 pp |

  Demonstrated compliance is therefore expected to be out of reach for CommonsenseQA and HellaSwag even for a policy that
  loses nothing. That is said now, so that a PASS cannot be read as a demonstrated ≤ 1 pp on those two benchmarks. The d
  values are planning assumptions. At G3b freeze, the expected widths are recomputed from the cross-fitted train
  discordance of the frozen policy (no test data), and recorded.
- **Effect precision.** At G3b freeze, the expected CI width of the macro Δ net saving is also recomputed, from the frozen
  policy's per-question saving differences on cross-fitted train. If the expected half-width is ≥ 0.03, the test cannot
  distinguish the margin. In that case it is reported to the reviewer before G3c, and the test is not spent automatically.

## A1.6 Frozen identities, seeds, exclusions
- Allocation salt `lh-phase3-alloc-v1`; folds `lh-fold-v1`; bootstrap 2,000 resamples, stratified by benchmark, paired, seed 0
  (as in Gate 2).
- Dataset revisions (inventory): ai2_arc 210d026f…, commonsense_qa 94630fe3…, hellaswag 218ec52e…, mmlu c30699e8…,
  openbookqa 388097ea…, piqa (baber/piqa) 142f6d73…, social_iqa (lighteval/siqa) 54c6a1f8…. Extraction refuses a
  different revision.
- **Excluded from H3:** ARC-Challenge and ARC-Easy (0 clean; exemplar exclusion kept strict); MMLU (0 clean); HellaSwag test
  split (unlabeled). **Excluded from H3-ood:** WinoGrande (A1.4); all OOD test splits (unlabeled); all OOD train splits
  (sealed).
- Unresolved contamination risks are carried over unchanged from `docs/PHASE3_INVENTORY.md`. In particular, Ouro's
  pretraining exposure is unknowable from the data.

## A1.7 G3a (D1-D4) under A1
- **Data:** Phase 1 train plus DEV+ for the three primary benchmarks. Phase 1 validation is reported as exposed. TEST rows
  are never read: the G3a code takes the frozen DEV+ id list as an allow-list and refuses TEST ids.
- **D1-D4 definitions:** as in plan section 7, unchanged. The stop rules are computed on the three-benchmark macro:
  D1 ceiling (ii) − B_pb < 0.05 stops; D2 nested improvement over B_pb < 0.02 stops. D3's rule ("h > x in at least half of
  the cells with ≥ 30 transitions") is counted over the primary benchmarks' cells.
- **Order:** A1 frozen (commit and tag) → DEV+ and TEST extraction (TEST sealed, Gate 2 pattern) → G3a on Phase 1 train + DEV+ →
  reviewer GO/STOP → G3b → reviewer implementation sign-off → G3c.

## A1.8 For the reviewer to decide before freezing
1. Accept the three-benchmark H3 population and the macro definitions of A1.2. That includes fitting on all six benchmarks
   with selection per benchmark, and the task-agnostic rule constrained over all six.
2. Accept DEV+ (fresh non-test rows as development data) and the TEST sizes, including HellaSwag grouping by `source_id`.
3. HellaSwag audit rule v2 (drop the colliding-id criterion; add the source/context criterion against every used HellaSwag row) and TEST 750, or keep v1 (676 rows, TEST 500).
4. OOD: PIQA and Social IQa primary, WinoGrande excluded, OOD train splits sealed, 0-shot.
5. Observed versus demonstrated compliance as in A1.5, including the pre-test check that the effect is resolvable.
