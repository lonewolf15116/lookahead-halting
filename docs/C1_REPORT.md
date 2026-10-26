# C1 confirmatory study — result

Protocol: `docs/C1_PROTOCOL.md` (sha256 `4f38b8b7…`, tag `c1-protocol`). Frozen selections: `results/c1/frozen.json` (`65728e16…`),
`results/c1/pi1x_models.npz` (`72f15508…`). TEST code bundle sha256 `c0902af3…` (24 files, commit `b217523`; the run was at
`196a882`, which adds documentation only). The run was a one-time execution on the sealed Phase 3 TEST set, after the reviewer's GO.

- Marker `results/c1/TEST_OPENED.json`: sha256 `b22e2d1624f131b04664222b1449eed3c98cb29011831b8e4768d243ea24b965`, opened 2026-10-26T18:18:05Z.
- Report `results/c1/test_report.json`: sha256 `02ba4b265f03fc1e44a30bbfd36473c8a216d3d1141f7051782eb5c15e8379b4`.
- Primary TEST sizes: OBQA 3000, CSQA 1000, HellaSwag 750. Out-of-distribution sizes: PIQA 1837, SIQA 1930.

## Primary verdict: PASS (as defined in C1 protocol section C1.5)

| | value |
|---|---|
| Δ gross macro saving, π1-x − native-pb | **+0.0529** |
| paired benchmark-stratified bootstrap 95% CI (10,000 resamples, seed 2026) | [+0.0488, +0.0571] |
| macro saving π1-x / native-pb | 0.2393 / 0.1864 |
| π1-x point loss ≤ 1 pp on every primary benchmark | yes (OBQA 0.17, CSQA 0.60, HellaSwag 0.40 pp) |

PASS means that π1-x stayed within the 1 pp point budget, the observed Δ ≥ δ = 0.02, and the lower bound L > 0. It is **not** a 95% claim that the effect is ≥ 0.02,
although here L = 0.049 is above δ as well. The claim is limited to this rule: both methods selected on development data under the same
combined risk rule (point loss ≤ 1 pp AND bootstrap 95% upper bound ≤ 1 pp), on Ouro-1.4B (4 loops), on OBQA, CSQA and HellaSwag.

## Mandatory secondary: net saving (an estimated deployment cost, not measured latency)

Δ net = **+0.0376**, 95% CI [+0.0334, +0.0419]. π1-x net saving per benchmark: OBQA 0.125, CSQA 0.274, HellaSwag 0.272. The ρ values are the
freeze-time GPU measurements. This is an accounting estimate under the C1.3 boundary, not a wall-clock measurement.

## Qualifications that must accompany the PASS

1. **On TEST, the risk guarantee did not hold for every benchmark.** Selection required each benchmark's 95% upper bound on loss to be ≤ 1 pp on
   development data. On TEST, the upper bounds are:

   | method | OBQA | CSQA | HellaSwag |
   |---|---|---|---|
   | π1-x | 0.53 | **1.10** | **1.60** |
   | native-pb | 0.67 | 0.30 | **1.33** |
   | B_pb | 0.00 | 0.00 | **1.60** |

   The protocol's budget FAIL is on the point loss, and every point loss is ≤ 1 pp. The upper bounds are reported, not used for the verdict.
   HellaSwag (n = 750) exceeds 1 pp for every method, including the fixed-depth baseline, so the small sample is part of the cause.
   π1-x exceeding 1 pp on CSQA is specific to π1-x.
2. **π1-x is not better than the simplest stability rule.** r-kl, the KL divergence between consecutive loops' answer distributions with no
   training, reaches 0.2380 against π1-x's 0.2393. Δ = +0.0012, 95% CI [−0.0041, +0.0065]. The bootstrap that re-selects the
   strongest baseline picked r-kl in 10,000 of 10,000 resamples. Because r-kl was chosen using TEST outcomes, this comparison is secondary and
   never affected the verdict. The development data showed the same near-tie (0.243 vs 0.246). The supported claim is therefore "a stability-based
   halting signal beats Ouro's native gate under the risk rule", not "the learned probe adds value over a simple stability statistic".
3. **The ranking reverses under the point-only rule.** Under the G3a-style rule (point loss ≤ 1 pp only, no upper bound):
   - π1-x − native-pb = **−0.022**, 95% CI [−0.026, −0.018]. This replicates the G3a STOP on TEST.
   - Both methods exceed 1 pp point loss on TEST under this rule: native-pb on OBQA (1.57) and CSQA (1.20), π1-x on all three
     benchmarks (1.07–1.13).

   Which method saves more depends on the selection rule.
4. **Out of distribution, π1-x does not stay within the budget.** Out-of-distribution results:

   | benchmark | method | loss (pp) | upper bound (pp) | saving |
   |---|---|---|---|---|
   | PIQA | π1-x | 0.71 | 1.52 | 0.308 |
   | PIQA | native-pb | 0.11 | 0.60 | 0.126 |
   | SIQA | π1-x | **3.26** | **4.72** | 0.481 |
   | SIQA | native-pb | −0.05 | 0.16 | 0.028 |

   On SIQA, π1-x halts at loop 2 for 1,785 of 1,930 questions. Its large out-of-distribution saving comes from halting too early, and the
   accuracy cost is real. The native gate stays within budget on both. This connects to the D-a PIQA numerical deviation and the SIQA short-context
   behaviour recorded in Phase 3.

## Other secondaries

- Each baseline compared with π1-x, by Δ gross with 95% CI:

  | baseline | Δ | 95% CI |
  |---|---|---|
  | B_pb | +0.156 | [0.151, 0.161] |
  | r-conf | +0.068 | |
  | r-dlogit | +0.077 | |
  | r-ent | +0.060 | |
  | r-hcos | +0.065 | |
  | r-hl2 | +0.090 | |
  | r-margin | +0.069 | |
  | r-kl | +0.001 | [−0.004, 0.007] |

  The full CIs are in the report.
- Exit histograms are in `test_report.json` → `per_benchmark`. π1-x exits mostly at loop 3 on OBQA (1,377 of 3,000 questions) and HellaSwag
  (662 of 750), and at loops 2–3 on CSQA.

## Execution record

- The tests before the run gave 127 passed and 1 skipped on Windows. The skipped test is the symlink-escape test for unsealing, because Windows does
  not have symlinks available. It passes on Linux.
- After the successful run, the runner was started a second time by mistake. Preflight refused it before the marker step
  ("refusing: results/c1/TEST_OPENED.json exists"). TEST was not read again, and nothing was written to `results/c1`. This is the one-time guard
  working as designed.
- The second `Start-Transcript` overwrote the transcript file, so `results/c1_local_backup/c1_test_console.txt` holds only the refused second
  attempt. The console output of the successful run survives as the user's paste in the conversation. The run itself is fully recorded in
  `test_report.json` and the marker.
- Package snapshot before the run: sha256 `436fa628…`. The snapshot after the run is still to be taken; the existing post file predates the run and is not evidence.
- All protocol-required analyses are reported above. None of the frozen decisions was changed after TEST was opened.

## Status of the research claim

- G3a (pre-registered point rule): STOP. Replicated on TEST as a secondary result (−0.022).
- C1 (pre-registered combined risk rule): PASS, Δ +0.053 gross and +0.038 net (estimated).
- Every write-up must report the two results together, along with the r-kl tie, the upper-bound exceedances on TEST and the out-of-distribution
  budget violation of π1-x. The literature references (ACT, PonderNet, PABEE, Geiping et al., SpecExit, NEAT, AdaPonderLM) still need to be verified.
