# Step 2: Scientific claims and novelty audit

- Date: 2026-10-27.
- Auditor: Claude, Opus 5.5.
- Manuscript: `docs/PAPER_DRAFT.md` at commit `5126009`. `paper/body.tex` has the same text, and line numbers below refer to `PAPER_DRAFT.md`.
- Status: PROPOSED. Nothing in the manuscript has been changed. Every change in section 6 needs approval first.

Verdict scale:
- **SUPPORTED**: the evidence and the stated scope match.
- **NEEDS QUALIFICATION**: true, but worded more strongly or more broadly than the evidence allows, or missing a caveat a reader needs.
- **UNSUPPORTED**: the evidence does not back the statement, or the statement contradicts the manuscript itself.

Ground rules:
- No frozen artifact, number, label or preregistration record is changed by any proposal below.
- No TEST data are re-read. Every number cited here already appears in the manuscript, which was checked in `docs/PAPER_NUMBER_AUDIT.md`. Numbers marked "derived" are arithmetic on manuscript values.

Benchmark for "enough contribution": TMLR's acceptance criteria are
1. "Are the claims made in the submission supported by accurate and convincing evidence?" (called the most important criterion);
2. "Would some individuals in TMLR's audience be interested in the findings…?"

TMLR states that novelty "is not a necessary criterion for acceptance". So the audit asks two things: whether every claim is supported, and whether the distinct findings are clearly separated from Popescu et al. It does not ask whether the paper is novel enough for a conference.

---

## 0. Summary

| # | Claim (location) | Verdict |
|---|---|---|
| N1 | Readouts beating Ouro's gate is not claimed as new (§1 l.59, §7.2) | SUPPORTED |
| N2 | Contribution: a pre-registered sealed-test evaluation showing that the preferred method depends on the selection rule (§1 l.61, §7.2 l.463–468) | NEEDS QUALIFICATION |
| N3 | The trained probe adds no demonstrated value over a training-free readout (§1 l.63, §6.1) | SUPPORTED |
| N4 | The readout that matches the probe is a divergence, not a confidence readout (§7.2 l.469) | NEEDS QUALIFICATION |
| N5 | Out-of-distribution budget failure (§1 l.52, §6.2) | SUPPORTED |
| N6 | Six recommendations (§7.4) | NEEDS QUALIFICATION (item 4) |
| N7 | Title: "When Does Adaptive Halting Win?" | NEEDS QUALIFICATION |
| P1 | [C] PASS: +0.053 [+0.049, +0.057] (§5.1) | SUPPORTED |
| P2 | "The preferred method depends on the selection rule [C] together with [S]" (§7.1 l.440) | NEEDS QUALIFICATION |
| P3 | The advantage holds "on three benchmarks" (abstract l.10) | NEEDS QUALIFICATION |
| S1 | Interpretation of PASS and of L > δ (§5.1 l.318) | SUPPORTED |
| S2 | Interval width and its precision (§5.1, §7.6) | NEEDS QUALIFICATION |
| S3 | r-kl interval stated in "pp of saving" (§6.1 l.405) | NEEDS QUALIFICATION |
| S4 | Four of nine upper bounds above 1 pp (§5.3, §1, §7.4) | SUPPORTED |
| S5 | Nine secondary baseline comparisons, unadjusted (§6.1) | NEEDS QUALIFICATION |
| S6 | "a stability-based halting signal, trained or not, saved more than Ouro's native gate" (§6.1 l.407) | NEEDS QUALIFICATION |
| S7 | "a convergence statistic was as useful as a trained probe" (§7.2 l.484) | **UNSUPPORTED** |
| S8 | Kubaty "fit with" our AUROC observation (§7.2 l.478–480) | NEEDS QUALIFICATION |
| L1 | "Exactness under truncation rests on the structure of the code, … not on an executed truncated run" (§7.5 l.518–519) | **UNSUPPORTED** (contradicts §3.6) |
| L2 | Prominence of the adapted native gate (abstract, §3.2, §7.5) | NEEDS QUALIFICATION |
| L3 | No formal risk-control guarantee (abstract, §3.4, §7.5) | SUPPORTED |
| L4 | Origin of the screened rule (§1 l.40, §4.2) | NEEDS QUALIFICATION |
| L5 | Retrospective savings and the scope paragraph (abstract, §1 l.68–70) | NEEDS QUALIFICATION (minor) |
| L6 | Cross-references and future work (§4.4 l.298, §8 l.574) | NEEDS QUALIFICATION (minor) |

There are two UNSUPPORTED statements, S7 and L1. Both are wording errors that are easy to fix, not problems with the evidence. No finding is UNSUPPORTED at the level of the evidence.

---

## 1. Novelty against Popescu et al. (2026), read adversarially

What Popescu et al. already establish. Sources: the arXiv v1 PDF in `refs_local/`; quotes were checked in the 2026-10-27 PDF log.
- On Ouro-1.4B and 2.6B, post-hoc readouts are "highly competitive" with the pretrained gate. The best readout is selected on validation data and evaluated on held-out data.
- In 5 of 12 model–benchmark pairs the selected readout has higher held-out accuracy than the gate.
- Their full per-benchmark curves (Figures 19–20) interleave. So a dependence of the preferred method on the operating point is already visible in their data.
- They name hidden-state cosine similarity, hidden-state displacement, maximum probability, entropy and logit margin as "the more promising post-hoc readouts".
- They train no probe, and they state no selection criterion beyond "validation-selected".
- They report estimated latency on Ouro.

The hostile reading: "When curves cross, any single-point comparison depends on how the point is chosen. That is a geometric fact, and it is visible in Popescu's figures. The paper confirms on one model that two rules pick different points. The 'reversal' is the expected consequence of crossing curves, and the point-only side of it compares points that break the budget."

What survives:
1. **The two rules are concrete and pre-specified, and the evaluation is on a sealed test set.** Both rules are the kind of accuracy-budget rule people actually use, and they produce opposite preferences there. Popescu et al. show crossing curves. They do not show that a reasonable, pre-specified selection protocol flips the conclusion, and they do not state their own criterion. This is the paper's main distinct contribution. It is narrow.
2. **The rule's own guarantee-like screen does not transfer.** Bounds of at most 1 pp at selection became 4 of 9 bounds above 1 pp on TEST, including fixed depth on HellaSwag. Popescu et al. report no bounds.
3. **A trained probe does not beat the best untrained readout.** Popescu et al. train no probe, so this is new evidence, though it is non-superiority from a [S] comparison with [P] emphasis.
4. **The strongest readout here (r-kl, a divergence) is not one of Popescu's named promising readouts.** The hidden-state readouts they highlight were the weakest here: r-hl2 0.149 and r-hcos 0.174 macro, against r-kl 0.238. This is a real contrast, but it is [P]. r-kl was identified on TEST, and the protocols differ: per-question multiple-choice option distributions here, against their readout definitions. It is not evidence that one finding is wrong.
5. **An out-of-distribution budget failure with task-agnostic thresholds.** This is a single observation. Fast yet Safe already names distribution shift as an open problem, so it is illustrative, not novel.

**Verdict on novelty.** The paper makes a distinct contribution beyond Popescu et al., but a narrow one. It is a carefully controlled, pre-registered demonstration of a methodological hazard, plus three secondary observations. That meets TMLR's interest criterion if the claims are scoped as below. It would not meet a "new finding" bar. The current text already disclaims the main finding (N1). The remaining risks are wording that makes the rule dependence sound like a discovery, rather than a sealed-test demonstration of something already visible in crossing curves (N2), and the title (N7).

**N2: NEEDS QUALIFICATION.**
- §7.2 (l.465–468) already concedes that the dependence is visible in Popescu's data. That is good.
- The contributions list in §1 (l.61–62) does not concede it, and reads as if the dependence were new. Add the concession there too (change C3).

**N4: NEEDS QUALIFICATION.** In §7.2 l.469 the bullet is an "addition" with no label, and r-kl was chosen using TEST outcomes. Add [P], add the contrast with Popescu's named readouts, and say that r-kl was identified on TEST (change C8).

**N6: NEEDS QUALIFICATION, item 4 only.** "Especially a divergence between successive loops" generalises from one [S]/[P] result on one model. Rephrase to "in our study, the strongest untrained baseline was a divergence between successive loops" (change C12). The other five recommendations are standard good practice, illustrated by our data. That is acceptable as long as they are not presented as new methodology. The text does not present them that way.

**N7: NEEDS QUALIFICATION.**
- "When does adaptive halting win?" promises a characterisation of the conditions under which halting wins. The paper gives one pair of rules on one model.
- The subtitle carries the actual finding.
- Suggested alternatives, for the author to choose:
  - "Selection Rules Decide Halting Comparisons: A Pre-registered Study on a Looped Language Model";
  - "Which Halting Method Wins Depends on the Selection Rule: Evidence from Ouro-1.4B".
- This is optional, but reviewers often quote the title back.

---

## 2. Primary claim

**P1: SUPPORTED.** The verdict follows mechanically from the frozen rule:
- π1-x point losses were 0.17, 0.60 and 0.40 pp, all at most 1 pp;
- Δ = +0.053, which is at least δ = 0.02;
- L = +0.049, which is above 0.

The text quotes the protocol's own limit on what PASS means (l.287, l.318). The label [C] is used only for this comparison. Correct.

**P2: NEEDS QUALIFICATION. This is the most important item.** "The preferred method depends on the selection rule" is a contrast between a [C] result and an [S] result. C1 tested only the screened-rule comparison. The rule dependence was never a pre-registered hypothesis with its own test.

The point-only side also compares operating points that broke the budget on TEST (5 of 6 benchmark-method cells; Table 5). native-pb's point-only "win" (0.359 against 0.337 macro) comes with larger losses: 1.57 and 1.20 pp, against 1.13, 1.10 and 1.07 pp for π1-x. So part of that win is extra accuracy loss, not a better trade-off, and no matched-loss comparison exists. The text says much of this already (l.320, l.371, l.443–445, l.558–559).

What still overstates:
- the [C] label on the rule-dependence statement (l.440);
- the contributions bullet (l.61), which says "showing" without scope.

Proposed changes:
- The rule dependence is labelled [S, contrasted with the [C] result].
- It is stated as "under two pre-specified rules, on one model, the preferred method differed", with the non-compliance attached in the same sentence wherever the reversal is stated (changes C2, C4).

**P3: NEEDS QUALIFICATION.**
- The macro advantage is concentrated. Per benchmark it is 0.021 on OpenBookQA, 0.121 on CommonsenseQA and 0.017 on HellaSwag, so CommonsenseQA supplies 0.040 of the 0.053 (derived, about 76%).
- CommonsenseQA is also the benchmark where π1-x's TEST upper bound broke (1.10 pp) while native-pb's did not (0.30 pp).
- §5.2 states the concentration (l.338). §5.3 states the CommonsenseQA exceedance (l.344, l.348). They are never put together.
- A reviewer will put them together: "the advantage comes from the benchmark where the probe's demonstrated budget failed."
- The abstract's "on three benchmarks" also invites the reading "on each of three".
- Proposed changes: the abstract says "macro-averaged over three benchmarks, driven mainly by CommonsenseQA", and §5.3 adds one sentence linking the two facts (changes C1, C6).

---

## 3. Statistical interpretation

**S1: SUPPORTED.** The difference between PASS and "true effect ≥ 0.02" is kept apart (l.287, l.318), and L > δ is labelled secondary.

**S2: NEEDS QUALIFICATION.**
- The primary interval (half-width 0.0042) is a test-sampling interval that conditions on the frozen policies. §7.6 says this (l.548–549).
- The manuscript's own numbers show that selection and refit variability are much larger. The E1 held-out macro savings against TEST were:
  - π1-x: 0.258 against 0.239;
  - r-kl: 0.207 against 0.238;
  - native-pb: 0.204 against 0.186.
  These are shifts of 0.018 to 0.031, about 4 to 7 times the half-width (derived).
- A reader who sees only §5.1 will take [+0.049, +0.057] as the uncertainty about the method difference. It is the uncertainty for these two frozen policies on this test set.
- Proposed change: one sentence in §5.1 after the PASS paragraph, and the qualifier "for the frozen policies" in §7.1 item 1 (change C5). This does not weaken the [C] result, which is about the frozen policies, as pre-registered.

**S3: NEEDS QUALIFICATION.**
- "A gain for π1-x of more than about 0.65 pp of saving, or a gain for r-kl of more than about 0.41 pp" uses "pp", which the paper otherwise reserves for accuracy loss.
- "Unlikely" is also loose wording for a percentile interval.
- Proposed text: "the interval excludes differences larger than 0.0065 in favour of π1-x or 0.0041 in favour of r-kl, in units of the four-loop budget" (change C7).

**S4: SUPPORTED.**
- "Four of nine" counts π1-x, native-pb and B_pb over three benchmarks (Table 4).
- §1 item 4 and §7.4 item 2 are now scoped to the primary rule.
- The causes are listed as inseparable [P].

**S5: NEEDS QUALIFICATION.**
- Table 6 reports nine unadjusted intervals, and the text does not say they are unadjusted.
- The conclusions do not depend on adjustment: the eight non-r-kl gaps are at least 0.060, with intervals far from 0, and the r-kl result is non-superiority, which adjustment would only widen.
- Proposed change: one sentence under Table 6 (change C7).

**S6: NEEDS QUALIFICATION.**
- l.407 says "The supported claim is therefore that a stability-based halting signal, trained or not, saved more than Ouro's native gate under the screened rule".
- For π1-x this is [C]. For r-kl it rests on a difference of stored macro values with no interval (0.052, [P]), for a comparator chosen using TEST.
- Proposed text: "The confirmatory claim concerns π1-x. Descriptively, the untrained r-kl also saved more than native-pb on TEST (0.238 against 0.186, no interval) [P]. So the evidence does not show that the learned probe adds value over a simple stability statistic." (change C7.)

**S7: UNSUPPORTED.**
- l.484 says "The near-tie shows that a convergence statistic was as useful as a trained probe in our setting".
- "As useful as" is an equivalence claim, which §6.1, §7.1 and the abstract all explicitly disclaim.
- Proposed text: "In our setting, we found no demonstrated advantage of the trained probe over a convergence statistic [S], [P]." (change C9.)

**S8: NEEDS QUALIFICATION.**
- Kubaty et al. concern calibration as a measure of early-exit quality. Our observation concerns ranking quality (AUROC) failing to predict which method a selection rule prefers.
- These are related properties, not the same one, so "our exploratory data fit with this" overstates the link.
- Proposed text: "Kubaty et al. show that calibration alone is insufficient to measure the quality of early-exit classifiers. We see an analogous gap for ranking quality: …" (change C10).

---

## 4. Limitations: prominence and accuracy

**L1: UNSUPPORTED.**
- §7.5 (l.518–519) says "Exactness under truncation rests on the structure of the code (Section 3.6), not on an executed truncated run."
- This was true before D-t and is now false. §3.6 reports an executed truncated run on 600 development questions, with 1,800 of 1,800 agreements.
- Proposed text: "Exactness under truncation was checked by an executed truncated run on 600 development questions (D-t; all 1,800 decisions agreed), not on the test set, and the per-loop statistics were not bitwise reproducible across GPU runs (Section 3.6)." (change C11.)

**L2: NEEDS QUALIFICATION.** The adaptation is stated in the abstract, in §3.2 and in §7.5. Two things a reviewer will raise are not stated together:
1. The gate was trained for Ouro's own objective: per token, on the language-modelling loss improvement. It is used here as a per-question statistic at one position, for multiple-choice stability. So "π1-x beats native-pb" says little about the gate in its intended use.
2. π1-x has the native CDF as one of its nine inputs (§3.3 l.167). The primary comparison is therefore between the native statistic and a model that contains it. A probe that includes the gate signal and also sees option scores (conf, ent) should be able to do at least as well, at least in ranking. This makes the [C] result less surprising, and it is exactly why the r-kl comparison matters.

Proposed changes: one limitations bullet in §7.5, and one sentence in §6.1 where RQ2(a) is answered (change C11).

**L3: SUPPORTED.** The missing guarantee is stated in the abstract, defined in §3.4 with four specific differences from LTT, UCB and CRC, and repeated in §7.5.

**L4: NEEDS QUALIFICATION.** This is the second most important item.
- The screened rule was adopted for C1 after E1 showed, on development data, that π1-x leads under it (§4.2 item 2–3).
- That is disclosed, but only in §4.2. The abstract and §1 say "pre-registered", which a reader may take to mean that the rule was fixed before any outcome was known.
- The sealed test confirms that a development finding held on new data. That is legitimate, and it is the value of the design. It does not show that the rule was chosen blind.
- Proposed changes:
  - §1, l.40: "C1 was pre-registered after an exploratory analysis (E1) found, on development data, that π1-x led under the screened rule; C1 tested whether that held on a sealed test set." The abstract needs at most a clause (changes C1, C3).
  - §7.6, Internal threats: one bullet (change C11).

**L5: NEEDS QUALIFICATION (minor).** The scope paragraph (l.68–70) says the policies were "not run with truncated computation". That is still true for the reported results, but the D-t check now exists. Add "a truncated-execution check on 600 development questions agreed on every decision (Section 3.6)" (change C3).

**L6: NEEDS QUALIFICATION (minor).**
- §4.4 l.298 says "(see the note in section 3.6)". §3.6 now points to §7.5, so this reference should point to §7.5.
- In §8, l.574, the future-work item "an executed, truncated implementation with measured time" is partly done by D-t, but on development data with descriptive timing only. Rephrase to "a truncated implementation with measured end-to-end time on held-out data" (change C13).

---

## 5. Strongest reasons a TMLR reviewer might reject, ranked

Each objection is followed by how the current text answers it and the change that closes it.

1. **"The reversal is an artifact of boundary overfitting, not a property of the signals."**
   - The point-only rule picks boundary points that break the budget on TEST. native-pb's point-only lead is paid for with 1.57 and 1.20 pp losses, and there is no matched-loss comparison.
   - Current answer: partly answered (l.320, l.371, l.443–445).
   - Close it with: C2 and C4 (the non-compliance attached to every statement of the reversal, and [S] labelling).
   - Optional, needs a GO: an exploratory saving-against-loss frontier for both statistics on development data only (E1 folds). It would show whether the curves cross near 1 pp. TEST must not be used.
2. **"You chose the rule under which your method won."**
   - Current answer: disclosed only in §4.2.
   - Close it with: C1, C3 and C11 (L4).
3. **"The advantage comes from the one benchmark where the probe's own bound failed."**
   - Current answer: the two facts are stated separately.
   - Close it with: C1 and C6 (P3).
4. **"The baseline is a strawman."**
   - The native gate is used off its training objective at one position, and π1-x contains its signal.
   - Current answer: partly answered (§3.2).
   - Close it with: C11 (L2). The strongest reply is already in the paper: the untrained r-kl matches π1-x, so the [C] result is not mainly about training.
5. **"Novelty is thin against Popescu et al.; one model, four loops, multiple choice."**
   - Current answer: the disclaimer is in place, and TMLR does not require novelty.
   - Close it with: C3 and C8, and optionally the title change (N7). The cover letter should cite TMLR's interest criterion.
6. **"The intervals overstate precision."**
   - Current answer: §7.6 only.
   - Close it with: C5 (S2).
7. **"The paper contradicts itself about truncated execution."**
   - Close it with: C11 (L1). This must be fixed before external review.
8. **"Equivalence language crept in despite the disclaimers."**
   - Close it with: C9 (S7).
9. **"Savings are simulated."**
   - Current answer: well answered (abstract, §3.6, D-t, §7.4 item 6). Popescu et al. report estimated latency on Ouro, so a reviewer may ask for timing.
   - No change. The D-t timing is descriptive [E], and adding it to the results is not recommended.

---

## 6. Proposed changes

These are text-only changes. They change no number, no evidence label of a [C] result and no frozen record. Each one must be mirrored in `paper/body.tex`.

| ID | Priority | Location | Change |
|---|---|---|---|
| C1 | must | Abstract l.9–11 | After "+0.053 gross macro saving, 95% CI [+0.049, +0.057]", add "macro-averaged over three benchmarks and driven mainly by CommonsenseQA, where the probe's test upper bound exceeded 1 pp". Optionally, after "pre-registered studies", add "the confirmatory one designed after an exploratory finding on development data". |
| C2 | must | §7.1 item 2, l.440 | Change "[C] together with [S]" to "[S], contrasted with the [C] result". Start the item with "Under two pre-specified rules, on this model, …". |
| C3 | must | §1, l.40–42, l.61–62, l.68–70 | (a) State that C1 was designed after E1 found that π1-x led under the screened rule on development data (L4). (b) Contributions bullet: "showing, under two pre-specified selection rules on one model, opposite preferences, a dependence already visible in the crossing curves of Popescu et al. but not previously tested under pre-specified rules" (N2). (c) Scope: mention the D-t check (L5). |
| C4 | should | §5.4 l.369 and §8 l.557 | Wherever the reversal is stated, attach in the same sentence that the point-only operating points broke the budget on 5 of 6 benchmark-method cells. §8 already does this; check that §5.4 does too. |
| C5 | must | §5.1 after l.318; §7.1 item 1 | Add: "The interval reflects test sampling for the frozen policies only. Development-to-test shifts in macro saving were 0.018 to 0.031 (E1 held-out against TEST), several times this half-width, so the interval should not be read as the uncertainty of the method difference beyond these policies." In §7.1 item 1, add "for the frozen policies". |
| C6 | must | §5.3 after l.348 | Add: "CommonsenseQA, where π1-x's upper bound exceeded 1 pp and native-pb's did not, is also where most of the macro advantage arises (0.121 of the per-benchmark differences; Section 5.2)." |
| C7 | must | §6.1 l.405, under Table 6, l.407 | S3: restate the r-kl interval in units of the four-loop budget, using "excludes". S5: state that the nine intervals are unadjusted, and that conclusions are unchanged. S6: replace the "supported claim" sentence as in section 3 above. |
| C8 | should | §7.2 l.469 | N4: add "[P]; r-kl was identified on TEST". Add the contrast: "Popescu et al. list hidden-state cosine and displacement among their more promising readouts on Ouro; in our per-question protocol these were the weakest (r-hl2 0.149, r-hcos 0.174 macro)." |
| C9 | must | §7.2 l.484 | S7: replace "was as useful as a trained probe" with "In our setting we found no demonstrated advantage of the trained probe over a convergence statistic [S], [P]." |
| C10 | should | §7.2 l.478–480 | S8: "fit with this" becomes "We see an analogous gap for ranking quality". |
| C11 | must | §7.5, §7.6, §6.1 | L1: rewrite the retrospective bullet (text in section 4 above). L2: add the bullet "The native-gate baseline is off its training objective, and π1-x takes the native CDF as an input, so the primary comparison is against a statistic the probe contains; the untrained r-kl result (Section 6.1) is the more informative comparison." L4: add an Internal-threats bullet on the rule being adopted after E1. Add one sentence in §6.1 under RQ2(a) on the nesting. |
| C12 | should | §7.4 item 4 | N6: "especially a divergence between successive loops" becomes "in our study the strongest untrained baseline was a divergence between successive loops". |
| C13 | should | §4.4 l.298, §8 l.574 | L6: change the cross-reference to Section 7.5, and change the future-work wording. |
| C14 | optional | Title | N7: choose between the current title and the two alternatives in section 1. This is the author's decision. |

Not proposed:
- Any change to the [C] verdict, to the numbers, or to the D-t P-2 FAIL label.
- Any new analysis on TEST.
- The optional development-only frontier analysis in section 5 item 1. It would be new exploratory work, and it needs its own GO.

## 7. Overall verdict

- **Claims.** The confirmatory result is supported as pre-registered. Two sentences are unsupported as worded (S7, L1), and both have direct fixes. The remaining issues are about prominence and scope, not evidence.
- **Novelty.** The paper is distinct from Popescu et al. in a narrow, defensible way: a sealed-test, pre-registered demonstration that two concrete selection rules lead to opposite conclusions, and that the screen's bounds do not transfer. It also has three secondary observations. It should not be framed as discovering rule dependence.
- **Readiness.** After the "must" changes (C1–C3, C5–C7, C9, C11), the manuscript should be ready for external human review. The two highest-risk reviewer objections, the boundary artifact and the rule chosen after E1, are best answered by putting the existing disclosures in the abstract and introduction.
