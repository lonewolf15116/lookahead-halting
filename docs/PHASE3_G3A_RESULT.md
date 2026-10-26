# Phase 3 G3a (D1-D4) result: STOP at D2 (for reviewer verdict)

Run on the laptop at commit 49295ea (code clean, all 66 tests passing first). CPU, 5.0 min. Report:
`results/phase3/g3a_report.json` (sha256 49ee7ace3337ec828d8771a18861c96d5dc6f1b0feedc91a1d83e6462b1ac7ec), committed unchanged.
- **Inputs.** Every frozen input hash matched: plan v1.1, A1, the readout module, the DEV+ check report, and Gate 2's frozen
  and recorded C.
- **Data.** 17,615 Phase 1 train questions plus 2,635 DEV+ = 20,250 development questions. 5,870 Phase 1 validation questions
  were used for the exposed estimate only. 0 failed questions. TEST was not read.

The verdict belongs to the reviewer. Below are the numbers as the frozen rules compute them, and three observations.

## Stop rules (primary macro: OpenBookQA, CommonsenseQA, HellaSwag; gross saving, OP1)
| rule | value | threshold | outcome |
|---|---|---|---|
| D1: ceiling (ii) − B_pb | 0.7036 − 0.2500 = **0.4536** | stop if < 0.05 | pass |
| D2: π1-x, nested improvement over the A2 comparator | **−0.0236** (per fold −0.025, −0.044, −0.014, −0.021, −0.015) | stop iff both < 0.02 | |
| D2: π1-hx, nested improvement over the A2 comparator | **−0.0490** (per fold −0.030, −0.068, −0.043, −0.028, −0.076) | | **STOP** |
| D3 (indirect; OP5): h > x | 10 of 18 counted cells | ≥ half | hx kept (moot after the D2 stop) |

**Decision computed by the frozen code: STOP (D2).** Under plan section 10, G3a ends Phase 3 with a written negative if D1 or D2
fails. Nothing was retried or changed.

## D1 detail
- B_pb chose fixed depth **3 on every benchmark** (gross saving 0.25; per-benchmark selection loss 0.07-0.61 pp).
- Ceiling (ii), the accuracy-budget oracle: 0.674-0.733 per benchmark.
- Ceiling (i), stable-final: primary macro 0.580.
- Six-benchmark headroom (ii): 0.448 (information only).
- Non-binding (A2.4): the best baseline on all development data is **native-pb**, at 0.369. Oracle (ii) − native-pb = 0.335.

## D2 detail (held-out macro saving, mean over the 5 outer folds)
| method | saving | method | saving |
|---|---|---|---|
| native-pb | **0.370** | r-margin | 0.340 |
| r-kl | 0.353 | r-hcos | 0.328 |
| π1-x | 0.347 | r-conf | 0.324 |
| π1-hx | 0.321 | r-dlogit | 0.318 |
| B_pb | 0.250 | r-hl2 / r-ent | 0.311 / 0.310 |

- **Comparator.** native-pb, in all five folds, chosen on the selection folds.
- **Against B_pb alone.** Both candidates beat it: π1-x by +0.097 and π1-hx by +0.071. But B_pb is no longer the strongest
  baseline: Ouro's own gate with per-benchmark thresholds is.
- **Exposed Phase 1 validation, reported only.** The comparator is again native-pb, at 0.368. π1-x is −0.029 against it and
  π1-hx −0.049. Same sign, similar size.
- **Six-benchmark macro, information only.** π1-x −0.009, π1-hx −0.024.

## D4 (informative): the gain from continuing is barely predictable
Out of fold, within benchmark. The first two columns are AUROC; the last is the Spearman correlation of the predicted
P(c_4) − P(c_t) with the actual Δ_t = c_4 − c_t.

| benchmark | AUROC c_t (x / hx) | AUROC c_4 (x / hx) | Spearman Δ_t (x / hx) |
|---|---|---|---|
| OpenBookQA, loops 1-3 | 0.55-0.58 / 0.58-0.59 | 0.53-0.57 / 0.56-0.59 | −0.03 to 0.01 / −0.01 to 0.01 |
| CommonsenseQA, loops 1-3 | 0.68-0.78 / 0.69-0.79 | 0.56-0.78 / 0.61-0.79 | 0.04-0.20 / −0.01 to 0.22 |
| HellaSwag, loops 1-3 | 0.54-0.58 / 0.58-0.62 | 0.52-0.54 / 0.61-0.62 | 0.02-0.14 / −0.01 to 0.11 |

- D1 shows a large retrospective headroom (0.45 over B_pb, 0.33 over native-pb).
- From loop 2 on, Δ_t is essentially unpredictable from these features: Spearman ≈ 0 on OpenBookQA and HellaSwag, and small
  on CommonsenseQA.
- Taken together, this is the "observable in hindsight, not forecastable" pattern, measured here for these features and
  models only. It does not show that no controller could predict the gain.

## Observations the reviewer should weigh (none changes the computed decision)
1. **D3 double-counts loop 3.** The adjacent transition 3→4 and the to-final transition 3→4 are the same transition, so the
   loop-3 cells appear twice among the 18 (OP6 did not exclude this).
   - Deduplicated, h > x in **8 of 15** cells, which is still ≥ half. The conclusion "hx kept" is unchanged.
   - Many of the AUROCs are near or below 0.5. Ten of the 18 primary cells are below 0.55 for both h and x. So WC vs CW is
     barely separable either way, and "h > x" is often the better of two weak scores.
   - This is moot after the D2 stop.
2. **Held-out loss often exceeds 1 pp.** Thresholds chosen at the 1 pp edge on four folds give held-out per-benchmark losses
   above 1 pp in most folds, for every adaptive method (the fold maximum is 1.2-4.2 pp). B_pb's fold maximum is 0.5-1.6 pp.
   - The held-out folds are small: about 230 CommonsenseQA and 450 OpenBookQA questions each.
   - G3a's stop rules use saving only (the frozen design). But this means "≤ 1 pp at selection" does not transfer reliably at
     these sizes, which matters for any future work.
3. **What beat the candidates.** It was Ouro's own native gate with per-benchmark thresholds (0.370). In Gate 2, the pooled
   native gate lost to fixed depth. Per-benchmark threshold selection, the change plan section 5 introduced, made the
   pretrained gate the strongest controller here, ahead of every published readout and both learned probes.

## Scope
Ouro-1.4B; six-benchmark development data, with the stop rules on the three primary benchmarks; gross saving; nested
cross-validation on development data. TEST (8,517 questions) stays sealed and unused. π3 and π4 were never trained: under the
frozen plan they are developed only at G3b, which G3a does not reach.
