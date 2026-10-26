# C1 freeze: artifacts and self-audit (for reviewer sign-off before the one-time TEST run)

**Run.** Laptop, 2026-10-26, working tree at fed36b0. The C1 code is byte-identical to the tagged `c1-protocol` commit
8f0eee2; the only later change is one line in DECISIONS.md. 102 tests passed first. Development data only; TEST unopened.

| artifact | sha256 |
|---|---|
| results/c1/frozen.json | 65728e16113e94114554fb28a951eee2b93af4b3fb5964bf32cf12845560b25f |
| results/c1/pi1x_models.npz | 72f1550813e12f242128345b4b20097eca0c83ebe843b358f5649a8ed8938bad |
| docs/C1_PROTOCOL.md (recorded in frozen.json) | 4f38b8b71479bdf2236f3887eac4dd56026c2b20b7edac10ed581529d00a2050 |

## Checks done
- **Protocol.** frozen.json records the approved protocol sha256, and the freeze refused any other.
- **Inputs.** The frozen G3a inputs are pinned: plan, A1, readout module, DEV+ check, Gate 2 C values.
- **Code.** Its six code hashes (c1_freeze, c1, c1_rho, e1, phase3_g3a, phase3_readouts) equal the repository files.
  - Gap: frozen.json does not record a git commit. The code hashes identify the code instead.
- **Model.** The model file hash in frozen.json equals the committed `pi1x_models.npz`.
- **Data.** 17,615 Phase 1 train + 2,635 DEV+ questions; 0 failed questions.
- **Selection.** Every primary selection meets the combined rule: point loss ≤ 1 pp and upper bound ≤ 1 pp.

## Frozen primary selections (combined risk rule; identical bootstrap draws, fingerprint 8dca6cc7…)
| benchmark | π1-x: m, τ | saving / loss / UB (dev) | native-pb: m, τ | saving / loss / UB (dev) |
|---|---|---|---|---|
| OpenBookQA | 4, 0.9399 | 0.154 / 0.31 / 0.75 | 4, 0.2959 | 0.142 / 0.44 / 0.98 |
| CommonsenseQA | 4, 0.9312 | 0.292 / 0.43 / 0.94 | 4, 0.3603 | 0.165 / 0.17 / 0.86 |
| HellaSwag | 3, 0.9312 | 0.292 / 0.49 / 0.86 | 3, 0.1883 | 0.273 / 0.38 / 0.77 |

π1-x uses τ = 0.9312 on two benchmarks: the 51-point grid is pooled over all development benchmarks (OP3), so two
benchmarks can select the same grid point. The savings differ (0.29195 at m = 4 vs 0.29197 at m = 3).

## Facts the reviewer should see before authorizing TEST
1. **In-sample development macro saving** (selection data; not a TEST estimate): π1-x 0.246, native-pb 0.193, r-kl 0.243,
   B_pb 0.083.
   - **r-kl is nearly level with π1-x in-sample.** It is a secondary comparator only (D-e).
   - In E1's nested estimate under the same rule, r-kl was 0.207 against native-pb's 0.204, so its in-sample figure is
     probably optimistic. Secondary analysis 2 will show the TEST picture.
2. **B_pb under the risk rule is fixed depth 4** (saving 0) on OpenBookQA and CommonsenseQA, and 3 on HellaSwag. As a
   reference it is weak.
3. **Expected 95% half-width at the TEST sizes: 0.0043**, from development discordance. δ = 0.02 is about 4.6 half-widths,
   so the design has ample precision. INCONCLUSIVE is unlikely unless the true Δ is near 0.02.
4. **Measured overhead** (RTX 5050, batch size 1, 200 DEV+ questions per benchmark):

   | benchmark | ρ_score | ρ_rest | loop | ρ_total |
   |---|---|---|---|---|
   | OpenBookQA | 0.036 | 0.014 | 20.7 ms | 0.050 |
   | CommonsenseQA | 0.010 | 0.004 | 79.9 ms | 0.014 |
   | HellaSwag | 0.013 | 0.003 | 148.8 ms | 0.015 |

   OpenBookQA's ρ is the largest because its loops are short (0-shot prompts). At most about 0.05 loop-equivalents per
   evaluation, so a gross advantage of the E1 size would survive in net terms. That is a prediction, not a result.
   - Net saving is an estimated deployment cost under the defined accounting boundary, not measured end-to-end latency
     (reviewer qualification).
5. **Secondary, OOD pooled selection** (one shared (τ, m) over all six development benchmarks): π1-x m = 4, τ = 0.961,
   saving 0.159; native-pb m = 4, τ = 0.386, saving 0.159. Level in-sample.
6. **Secondary, point rule:**
   - native-pb: OpenBookQA m = 4, 0.403; CommonsenseQA m = 3, 0.365; HellaSwag m = 3, 0.338.
   - π1-x: 0.303; 0.363; 0.363.
   - This is the same direction as E1, where native-pb led under the point rule.

## Next (needs the reviewer's sign-off)
- **The guarded one-time TEST script.** It refuses to run unless the frozen hashes match, writes `results/c1/TEST_OPENED.json`
  before reading any outcome, reads `data/extracted_phase3_test` once, and computes the C1.5 verdict table and every C1.6
  secondary analysis.
- **It is not written yet.** It will be submitted for review before it runs.
