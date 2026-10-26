# Phase 3: Negative Result at G3a (D2)

**Project:** Lookahead Halting — Ouro-1.4B  
**Status:** Phase 3 CLOSED; frozen STOP at D2  
**Run:** 2026-10-26, code commit `49295ea30fe10a7f475555848d2930c626b32ce4`  
**Decision record:** `521be0a` (reviewer's final verdict, "Independent G3a review — Final verdict": STOP confirmed)  
**Primary machine-readable result:** `results/phase3/g3a_report.json`  
**Report SHA-256:** `49ee7ace3337ec828d8771a18861c96d5dc6f1b0feedc91a1d83e6462b1ac7ec`

## 1. Research question and outcome

Can a post-hoc learned stopping controller for Ouro-1.4B deliver greater reasoning-loop savings than the strongest eligible stopping baseline while meeting the frozen accuracy-loss selection criterion? The G3a screening answer is **no for the tested π1-x and π1-hx controllers**. Both improve gross saving relative to a fixed-three-loop policy, but neither improves on Ouro's native exit gate with per-benchmark thresholds. The frozen D2 rule (with OP2, locked before the run) requires at least one candidate to improve over the A2-selected comparator by 0.02 or more; both fall below this threshold. **Phase 3 stops at G3a/D2.** This does not establish that adaptive halting is impossible, nor that no other feature or policy can work.

## 2. Frozen design and data boundaries

Phase 3 followed the frozen plan v1.1, Amendment A1, and the A2 comparator and G3a implementation decisions. Frozen input hashes and code-file hashes are recorded in the JSON provenance. The recorded G3a run used code commit `49295ea`, with clean code, 66 passing tests, no failed questions, and approximately five minutes of CPU time. This report describes the recorded hashes; independent verification against all original local inputs was **not** possible from the two exported report files alone. The frozen protocol documents are tagged `phase3-plan-v1.1`, `phase3-a1` and `phase3-a2` on GitHub (verified when this report was committed), so each pinned hash can be re-checked from the tagged commits in the repository.

The Phase 1 extraction contained 33,353 questions. G3a used 17,615 Phase 1 training questions plus 2,635 Phase 3 DEV+ questions (**20,250 development questions**) for fitting and development, and 5,870 Phase 1 validation questions for an **exposed, report-only** estimate. Phase 3 extracted 11,152 questions overall, divided into 2,635 DEV+ and **8,517 sealed TEST** questions. The sealed TEST questions were **not accessed**. The primary macro aggregates OpenBookQA, CommonsenseQA, and HellaSwag equally; the other three benchmarks are informative. G3a screening uses **gross loop saving** (overhead ratio ρ = 0), not measured end-to-end net savings.

## 3. Context: Gate 2 to Phase 3

Earlier Gate 2 testing did not establish a deployable learned halting advantage. Phase 3 revised the comparison protocol to permit benchmark-specific threshold selection and nested development-data evaluation, while retaining a sealed confirmatory TEST. This matters: a pooled native gate had been uncompetitive in Gate 2, but the native Ouro exit gate became the strongest comparator once benchmark-specific thresholds were permitted. This comparison change is part of the frozen Phase 3 protocol, not a post-result adjustment.

## 4. Frozen decision gates and findings

| Gate | Recorded finding | Frozen decision |
|---|---|---|
| D1: oracle headroom | Accuracy-budget oracle (ii) saving 0.703645 versus fixed-depth `B_pb` 0.250000; difference **0.453645** | PASS; stop only if headroom < 0.05 |
| D2: π1-x | Nested improvement over strongest eligible comparator **−0.023623** | Below +0.02 |
| D2: π1-hx | Nested improvement **−0.049032** | **STOP**, because both candidates are below +0.02 |
| D3: hidden versus cheap features | Indirect h > x in **10/18** counted cells; after deduplicating loop-3 overlap, **8/15** | Informative/moot after D2 STOP |
| D4: future gain predictability | Low out-of-fold Spearman association between predicted and observed `c4 − ct` on OpenBookQA and HellaSwag, particularly from loop 2 onward; small on CommonsenseQA | Informative only |

### D1: opportunity in hindsight

The primary-macro stable-final oracle (i) saving is **0.579851**; the accuracy-budget oracle (ii) saving is **0.703645**. Against the fixed-three-loop baseline, the latter offers **0.453645** gross-saving headroom. Against the strongest baseline selected descriptively on all development data, native-pb at **0.368838**, the nonbinding oracle gap is **0.334807**. Both oracle values depend on information unavailable to an online controller and must **not** be interpreted as achievable deployable savings.

### D2: the decisive comparison

| Method | Nested held-out primary-macro gross saving |
|---|---:|
| Native Ouro gate (`native-pb`) | **0.370088** |
| Learned π1-x | **0.346465** |
| Learned π1-hx | **0.321056** |
| Fixed depth 3 (`B_pb`) | **0.250000** |

The comparator was `native-pb` in **all five** outer folds, selected from the eligible baselines on the corresponding selection folds. π1-x's fold-level improvements were approximately −0.0253, −0.0438, −0.0135, −0.0208, −0.0147; π1-hx's were −0.0295, −0.0684, −0.0427, −0.0280, −0.0764. Both learned methods beat fixed depth by approximately +0.0965 and +0.0711, respectively, but not the native gate. The exposed Phase 1 validation comparison agrees in sign: π1-x −0.0289 and π1-hx −0.0491 against native-pb. Validation was not used to revise the frozen decision.

**Important limitation — held-out accuracy-budget violations:** The frozen selection procedure permitted up to **1 percentage point** of accuracy loss on selection data, but this did **not** reliably hold on held-out folds. The factual result write-up reports that **47 of 50 adaptive-method/fold comparisons** exceeded 1 pp on at least one held-out primary benchmark; held-out worst-benchmark losses reached roughly **0.8–4.2 pp**. The native gate also suffered held-out violations. Therefore, the gross-saving ranking must **not** be presented as a demonstrated accuracy-compliant deployment comparison. The frozen D2 stopping rule uses gross saving as specified, and this limitation does not reverse its STOP decision.

### D3: hidden-state discrimination, with double-counting disclosed

The recorded indirect comparison counted h > x in **10 of 18** primary cells. At loop 3, the adjacent transition (3→4) and the to-final transition (3→4) are identical; counting both duplicates three cells. Deduplicating yields **8 of 15**, still at least half, so the recorded keep-hx conclusion would not change. The indirect to-final estimate also extrapolates the Gate 2 adjacent-transition calibration rather than validating it directly for the to-final task. Many AUROCs are near chance; relative superiority of h over x does not imply strong absolute discrimination. In any case, D3 is **moot after D2 STOP**.

### D4: retrospective gain versus prospective prediction

D4 predicts correctness at the current and final loop and examines `Δt = c4 − ct`. Across OpenBookQA, Spearman correlations of predicted versus realized gain were approximately −0.03 to +0.01 with cheap features and −0.01 to +0.01 with hidden-state augmentation; on HellaSwag, approximately +0.02 to +0.14 and −0.01 to +0.11; on CommonsenseQA, approximately +0.04 to +0.20 and −0.01 to +0.22. These ranges cover loops 1–3 and should not be interpreted as identical at each loop. In the reported later-loop analyses, gain predictability is especially weak on OpenBookQA and HellaSwag. This supports a **bounded** observation: retrospective savings were substantial, but gain was difficult to forecast using **these particular features, fitted models, and benchmarks**. It does not prove an information-theoretic impossibility or exclude jointly trained or more informative controllers.

## 5. Interpretation and limitations

1. **Strong native baseline:** Ouro's own exit gate, with benchmark-specific thresholds allowed by the frozen protocol, outperformed all tested readouts and both post-hoc learned probes in the primary gross-saving comparison. Its joint-training history is a plausible hypothesis for its advantage, **not a demonstrated causal explanation**.
2. **Oracle is not a controller:** Oracle savings depend on hindsight. The 0.335 gap to native-pb is opportunity under an oracle, not an attainable guarantee.
3. **Accuracy compliance is unresolved:** Selection-time 1 pp constraints often failed on held-out folds (47/50 method-fold cases). A future method and native-gate comparator would both need the **same** risk-control treatment.
4. **Gross versus net:** Controller overhead and measured runtime savings were not established at G3a.
5. **D3 duplicate:** Loop-3 adjacent/to-final cells are the same; corrected count is 8/15, without changing the stopping decision.
6. **Restricted scope:** Ouro-1.4B, the tested six benchmarks (three primary), the recorded post-hoc features, and the specified nested cross-validation. No claims are made about other models, jointly trained policies, or broader deployment settings.
7. **Provenance:** The report records pinned hashes and a clean code state; independent byte-for-byte re-verification requires the original repository and data files.

## 6. Protocol disposition

The frozen decision is **STOP (D2)**, and Phase 3 ends with this written negative result. **π3 and π4 were never trained**, no thresholds were retroactively changed, and **all 8,517 TEST questions remain sealed and unused**. This report is descriptive; it does not authorize a new confirmatory run or retrospectively change the experiment's scope.

## 7. Open questions for a separately preregistered study

**Why is the native gate competitive?** Explore on already-seen development data whether the native gate's advantage reflects joint training, feature information, calibration, benchmark-specific thresholding, or another mechanism. Observational comparisons cannot isolate causality; any causal claim requires a designed experiment.

**Can future-loop benefit be predicted at acceptable cost?** Explore additional information sources, signal-to-noise, transition dynamics, and compute overhead before designing a new controller. Risk-controlled stopping is not itself novel; related prior work includes learn-then-test/conformal risk control and early-exit risk calibration (including CALM). Any new risk-controlled policy must be compared with a **risk-controlled native gate under the same selection and evaluation rules**. A robust minimax game is speculative until an empirically justified uncertainty set and useful guarantee are identified.

Any new investigation must have its own written, frozen protocol before a single confirmatory test. Development on previously exposed data is permissible if labelled exploratory; the sealed TEST cannot be used for iterative design or debugging.

## 8. Evidence and audit trail

- `results/phase3/g3a_report.json`: primary numerical evidence, fold-level results, D1–D4 diagnostics, hashes, and frozen decision.
- `docs/PHASE3_G3A_RESULT.md`: contemporaneous factual summary and documented caveats.
- `docs/PHASE3_G3A_IMPLEMENTATION.md`: implementation, frozen-spec crosswalk, OP1–OP8 and their reviewer resolutions.
- `docs/DECISIONS.md`: the append-only decision log, including every reviewer decision.
- Decision-record commit: `521be0a` (reviewer's final STOP verdict).
- Frozen code/run commit: `49295ea30fe10a7f475555848d2930c626b32ce4`.
- Frozen tags (present on GitHub): `phase3-plan-v1.1`, `phase3-a1`, `phase3-a2`.

**Bottom line:** G3a provides a controlled negative finding for two post-hoc learned stopping controllers against a stronger benchmark-calibrated native gate. The gap between hindsight oracle and forecastable benefit remains an open research question, not a demonstrated new algorithm.
