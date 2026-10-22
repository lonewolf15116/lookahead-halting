# Phase 3 fresh-data inventory (generated; inventory only)

Audit rule `phase3-inventory-v1`; eligibility filter: at least 300 clean questions. The filter is not evidence of statistical adequacy: see the precision column and decide per pool before spending it.

No model outputs, correctness labels of any model, hidden states, thresholds or confirmation predictions were read. Counts below are rows; `primary` exclusions add up to `raw - clean`.

## In-distribution candidates

| pool | raw | used (P1/CONF eval) | used as exemplar | unlabeled | duplicates | clean | status | paired half-width at 10% discordance |
|---|---|---|---|---|---|---|---|---|
| arc_challenge/test | 1172 | 1172 | 0 | 0 | 0 | 0 | **ineligible** |  |
| arc_challenge/train | 1119 | 0 | 1119 | 0 | 0 | 0 | **ineligible** |  |
| arc_challenge/validation | 299 | 299 | 0 | 0 | 0 | 0 | **ineligible** |  |
| arc_easy/test | 2376 | 2376 | 0 | 0 | 0 | 0 | **ineligible** |  |
| arc_easy/train | 2251 | 2 | 2249 | 0 | 0 | 0 | **ineligible** |  |
| arc_easy/validation | 570 | 570 | 0 | 0 | 0 | 0 | **ineligible** |  |
| commonsense_qa/test | 1140 | 0 | 0 | 1140 | 0 | 0 | **ineligible** |  |
| commonsense_qa/train | 9741 | 1000 | 7306 | 0 | 0 | 1435 | **eligible** | ±0.0164 |
| commonsense_qa/validation | 1221 | 1221 | 0 | 0 | 0 | 0 | **ineligible** |  |
| hellaswag/test | n/a | | | | | n/a | **unresolved** | |
| hellaswag/train | 39905 | 1000 | 38229 | 0 | 0 | 676 | **eligible** | ±0.0238 |
| hellaswag/validation | 10042 | 10042 | 0 | 0 | 0 | 0 | **ineligible** |  |
| mmlu/dev | 285 | 5 | 280 | 0 | 0 | 0 | **ineligible** |  |
| mmlu/test | 14042 | 14042 | 0 | 0 | 0 | 0 | **ineligible** |  |
| mmlu/validation | 1531 | 1530 | 1 | 0 | 0 | 0 | **ineligible** |  |
| openbookqa/test | 500 | 500 | 0 | 0 | 0 | 0 | **ineligible** |  |
| openbookqa/train | 4957 | 0 | 0 | 0 | 5 | 4952 | **eligible** | ±0.0088 |
| openbookqa/validation | 500 | 500 | 0 | 0 | 0 | 0 | **ineligible** |  |

## New-benchmark (OOD) candidates

| pool | raw | used (P1/CONF eval) | used as exemplar | unlabeled | duplicates | clean | status | paired half-width at 10% discordance |
|---|---|---|---|---|---|---|---|---|
| piqa/test | 3084 | 0 | 0 | 3084 | 0 | 0 | **ineligible** |  |
| piqa/train | 16113 | 0 | 0 | 0 | 6 | 16107 | **eligible** | ±0.0049 |
| piqa/validation | 1838 | 0 | 0 | 0 | 1 | 1837 | **eligible** | ±0.0145 |
| social_iqa/train | 33410 | 0 | 0 | 0 | 29 | 33381 | **eligible** | ±0.0034 |
| social_iqa/validation | 1954 | 0 | 0 | 0 | 24 | 1930 | **eligible** | ±0.0141 |
| winogrande/test | 1767 | 0 | 0 | 1767 | 0 | 0 | **ineligible** |  |
| winogrande/train | 40398 | 0 | 0 | 0 | 4 | 40394 | **eligible** | ±0.0031 |
| winogrande/validation | 1267 | 0 | 0 | 0 | 0 | 1267 | **eligible** | ±0.0174 |

## Unresolved pools

- `hellaswag/test`: split could not be loaded: hellaswag: ValueError: invalid literal for int() with base 10: ''

## Unresolved contamination risks

- Pretraining and SFT contamination of Ouro-1.4B is unknowable from the data: its training mixture is not itemised, so any benchmark's train or validation split may have been seen by the model. The inventory removes overlap with what OUR pipeline used; it cannot remove overlap with what the model saw. A finding that depends on a pool being unseen by the model is not supported by this audit.
- Text matching is exact after lower-casing and whitespace normalisation. Paraphrases and near-duplicates (for example ARC items that differ in one word) are not detected.
- Native ids are only unique within a dataset revision; HellaSwag ids repeat inside train, so an id match can exclude more rows than were used (conservative; `excluded_by_id_only` shows how many).
- A pool that is clean today is clean for one dataset revision. The revisions recorded here must be pinned when the fresh allocation is frozen.
- Few-shot exemplars for the future fresh tests are not drawn yet; the sampler draws them at build time. For new benchmarks the report states the few-shot source split and the overlap of each candidate pool with it.

## Verification of the rebuilt used-set

- confirmation ids reproduce the committed sha256: True
- Phase 1 arc_challenge: doc ids equal splits.csv: True, n matches manifest: True, token-hash check: True
- Phase 1 arc_easy: doc ids equal splits.csv: True, n matches manifest: True, token-hash check: True
- Phase 1 hellaswag: doc ids equal splits.csv: True, n matches manifest: True, token-hash check: True
- Phase 1 mmlu: doc ids equal splits.csv: True, n matches manifest: True, token-hash check: True
- Phase 1 commonsense_qa: doc ids equal splits.csv: True, n matches manifest: True, token-hash check: True
- Phase 1 openbookqa: doc ids equal splits.csv: True, n matches manifest: True, token-hash check: True

Report: `results/phase3/inventory.json`. Clean-row identities: `results/phase3/inventory_clean_ids.csv` (sha256 in the report). The fresh allocation (which clean rows are dev, which are the one-time test) is NOT decided here.
