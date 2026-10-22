# Phase 3 amendment A1: allocation, narrowed population, D1-D4 population (DRAFT rev 2; nothing frozen)

Rev 2 applies the reviewer's "Phase 3 A1 — Revised Amendment Review" (CONDITIONAL GO): HellaSwag rule v2 and 750 TEST accepted subject to the cross-split audit (A1.9); effect-precision rule rewritten (A1.5); tables reconciled; D3 change declared as an amendment (A1.7).

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
| HellaSwag | **998** under rule v2 (cross-split, A1.9) | 750 (749 sources) | 248 | 6,025 | 2,008 |

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
- **HellaSwag TEST size: 750** (rule v2, reviewer-accepted conditionally). If the cross-split audit (A1.9) leaves fewer than 750
  clean rows, or source grouping cannot reach 750, the allocation script writes no lists and stops with RETURN_FOR_REVIEW.
  Neither the target nor DEV+ is reduced silently. Because whole source groups are taken, TEST can exceed 750 by a few rows;
  the exact count is recorded at freeze.
- **Why DEV+ is needed (reviewer concern on independent development data).** The binding case is not HellaSwag but
  **OpenBookQA**. It has only 300 Phase 1 train questions, so 1 pp is 3 questions, and inside a nested 4-of-5-fold D2 it is
  about 2. Per-benchmark thresholds selected at that resolution are unreliable. With DEV+ the OpenBookQA development set
  is 2,252 questions (1 pp ≈ 22 questions). CommonsenseQA goes from 733 to 1,168. HellaSwag's development data is
  essentially Phase 1 train (6,025), and its small remainder is not treated as a resource.
- **Cost (corrected; the first draft left out the OOD sets).** In-distribution DEV+ and TEST come to at most 7,386 questions
  (4,952 + 1,435 + 998), and the OOD TEST sets to 3,767 (1,837 + 1,930). That is 11,152 questions (8,517 TEST, 2,635 DEV+),
  about 78 GPU-minutes at the measured ~35 min per 5k (`allocation.json`). TEST outcomes are extracted but stay sealed, as the
  Gate 2 confirmation outcomes were.

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
| HellaSwag | 750 | 1.34 / 1.90 pp | 1.60 / 2.26 pp |
| PIQA (OOD) | 1,837 | 0.86 / 1.21 pp | 1.02 / 1.45 pp |
| Social IQa (OOD) | 1,930 | 0.84 / 1.18 pp | 1.00 / 1.41 pp |

  Demonstrated compliance is therefore expected to be out of reach for CommonsenseQA and HellaSwag even for a policy that
  loses nothing. That is said now, so that a PASS cannot be read as a demonstrated ≤ 1 pp on those two benchmarks. The d
  values are planning assumptions. At G3b freeze, the expected widths are recomputed from the cross-fitted train
  discordance of the frozen policy (no test data), and recorded.
- **Effect precision (reviewer wording, replacing the first draft's incorrect "half-width ≥ 0.03" rule).** At G3b freeze,
  estimate the standard error and expected confidence-interval width for the primary macro difference in net saving, using
  cross-fitted development predictions. Report the minimum detectable improvement above the preregistered 0.03 margin at the
  chosen confidence level and statistical power. If precision is inadequate for a practically meaningful effect, obtain
  reviewer approval before spending the sealed test. This assessment does not change the success threshold.

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
  D1 ceiling (ii) − B_pb < 0.05 stops; D2 nested improvement over B_pb < 0.02 stops. **D3 (declared amendment A1-D3).** The rule
  is present in frozen v1.1 section 7: "H3-h and the hx variants are kept only if h > x within at least half of the
  (benchmark, loop) cells with ≥ 30 transitions". v1.1 counts cells over all six benchmarks. A1 amends only the population:
  the cells are counted over the three primary benchmarks, so that the D3 decision concerns the benchmarks the H3 claim covers.
  The "half", "≥ 30 transitions" and "h > x" parts are unchanged. Six-benchmark D3 is reported beside it and decides nothing.
- **Order:** A1 frozen (commit and tag) → DEV+ and TEST extraction (TEST sealed, Gate 2 pattern) → G3a on Phase 1 train + DEV+ →
  reviewer GO/STOP → G3b → reviewer implementation sign-off → G3c.

## A1.9 result (run at a99bd6b on the PC; real tokenizer; used-set verification passed)
- `results/phase3/allocation.json` sha256 1327e1cca5360f9b80de7759360b6d06519e8c881aad984e988a723417ea9a1b. The TEST and DEV+
  list hashes match the files. No row is in both lists, and no HellaSwag source is split between them. Every non-HellaSwag
  row is an inventory-clean row.
- **HellaSwag v2, cross-split:** 39,905 train rows. Excluded: 38,409 used text, 488 shared source, 10 shared context; 0
  unlabeled, 0 other-pool, 0 duplicate. Clean: **998**. The used rows behind the source and context checks are 38,409 in train
  plus all 10,042 in Phase 1 validation. **0** rows were excluded only through validation-split sources, and the cross-split
  context check removed one row more than the within-train count (999 → 998).
- **v2 versus v1:** only 513 of the 676 v1-clean rows survive v2. The other 163 share a source or context with a used row,
  which v1 did not check. So v2 is stricter than v1 on proximity, and more permissive only on colliding ids.
- **Allocation:** OpenBookQA 3,000 TEST / 1,952 DEV+; CommonsenseQA 1,000 / 435; HellaSwag 750 / 248; PIQA 1,837 TEST;
  Social IQa 1,930 TEST. TEST 8,517 rows, DEV+ 2,635 rows.

## A1.8 Reviewer decisions (Revised Amendment Review, 2026-10-22)
| item | status |
|---|---|
| Three-benchmark H3 population (A1.2) | accepted |
| DEV+ / TEST allocation | accepted conditionally (final lists from A1.9) |
| HellaSwag rule v2 | accepted conditionally (cross-split audit, A1.9) |
| HellaSwag TEST 750 | accepted if the final pool and source grouping permit (A1.9 stops otherwise) |
| PIQA and Social IQa OOD; WinoGrande excluded | accepted |
| Observed versus demonstrated compliance; effect precision as reworded | accepted |
| Freeze | HOLD until A1.9 has run and the reviewer has seen `allocation.json` |

## A1.9 Final allocation run (implements A1.3, A1.4 and the HellaSwag cross-split audit)
- `scripts/phase3_allocate.py`, with the pure logic in `src/lookahead/phase3_allocation.py` (9 synthetic tests). CPU only. It reads
  the committed inventory (sha256-checked against 7ec501ae…), the HellaSwag dataset and the manifests. It reads no model
  output.
- **HellaSwag cross-split audit.** The used-set is rebuilt and verified exactly as in the inventory. A train row is HellaSwag-clean
  only if all of these hold:
  - it has a gold answer;
  - its text was never used;
  - it shares no `source_id` or context with **any** used HellaSwag row, in the train or the Phase 1 validation split;
  - its text is not a clean row of another pool.
  Internal duplicates keep one row. The run reports counts by reason, including how many rows are excluded **only** through
  validation-split sources: what the within-train audit could not see.
- **Outputs.** `results/phase3/allocation.json` (counts, rule versions, list hashes, precision at the final sizes, extraction
  estimate), plus `allocation_test_ids.csv` and `allocation_dev_ids.csv`. The script refuses to overwrite them. If HellaSwag
  falls short of 750 it writes only the json, with RETURN_FOR_REVIEW (exit 3).
- **Freeze.** After the reviewer has seen `allocation.json`, these are committed and tagged together as `phase3-a1`, with their
  sha256 values recorded here: this file renamed to `docs/PHASE3_AMENDMENT_A1.md`, the allocation outputs, and the actual
  counts filled into A1.3 and A1.5. Only then are extraction and G3a authorized.
