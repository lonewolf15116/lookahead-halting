# Publication Next-Step Decision — Independent GitHub Readiness Review (2026-11-25)

**Scope:** read-only comparison of `lookahead-halting`, `remat-oracle-pilot`, `dtr-regime-switching`, and `space-001` README/scientific release files, plus public arXiv and current TMLR instructions. The previous CSLG-001 methods/discovery track is parked; this is a new triage, **not** a rerun of frozen data.

## Decision

**Prioritize an author-controlled TMLR submission audit for `When Does Adaptive Halting Win? Selection Rules Matter in Looped Language Models`.** This is the closest to a complete independently traceable manuscript package, but **do not claim the submission has happened** or issue an unconditional ready-to-submit certification without a fresh source/PDF/package verification.

### Critical update to repository's older checklist

The code repository's `docs/PUBLICATION_CHECKLIST.md` still treats arXiv posting as future, but **the named manuscript is already publicly indexed as arXiv:2611.02333**, with a first-posting date in early November 2026. **Do not submit a duplicate arXiv entry.** If an arXiv revision is necessary, compare source version/public preprint, then use the existing record and revision mechanism.

Official public paper: https://arxiv.org/abs/2611.02333
Official TMLR submission instructions: https://jmlr.org/tmlr/submissions.html
Official author guide: https://www.jmlr.org/tmlr/author-guide.html
Official editorial policy (including originality/dual submission): https://jmlr.org/tmlr/editorial-policies.html

TMLR permits existing arXiv preprints but maintains a double-blind **submission itself**. Check anonymity of the submitted PDF and anonymized supplement, and do not insert a self-identifying preprint link within the anonymized PDF. Public availability may compromise practical anonymity but is not itself a ban. Do not infer that TMLR submission has already been completed.

## Why this choice

| Project | Audited repo evidence | Research maturity | Next gate |
|---|---|---|---|
| Lookahead halting | Frozen C1/TEST result files, 394 numeric checks with 0 FAIL (393 PASS, 1 qualified), 98 provenance checks 0 FAIL, final consistency 40/40, Windows bitwise dev reruns; TMLR anonymous paper and supplement packaging described; code README notes unreleased development features and sealed TEST arrays | Manuscript/preprint already public; TMLR *submission readiness audit* plausible | Check actual current Git commit, built PDF and supplement match, anonymity, preprint overlap, material availability and disclosures; author submission only after audit |
| Remat oracle pilot | README states M1/M2a/M2b complete; M2c windowed real-trace oracle and M3 DTR replay **not complete**. Its real-trace `oracle_ub` is an upper bound, not verified global optimum | Promising new system study, not yet complete | Finish oracle semantics/proofs then independent baseline comparison before asserting quantitative oracle gap |
| DTR regime-switching | Has reproducibility scripts and public simulator-limited results; related public preprints are indexed (arXiv:2609.31250, 2610.14369) | Already published as preprint, simulator domain only | Do not republish same phenomenon as new discovery |
| SPACE-001 | README G1 only, synthetic workloads and no independent reference implementation or optimality, no flight validation | Earlier-stage simulation research | Scientific validation before publication claims |

**Scope of novelty:** The halting paper's contribution is an **evaluation selection-rule effect**, not proof that the trained probe generally dominates, not real measured inference latency, and not a guarantee of accuracy-risk control. The observed advantage under the pre-registered screened selection rule flips under point-only selection; an untrained KL baseline is competitive. Confirmatory vs secondary vs exploratory labels must remain unchanged.

## Conditions for responsible submission

1. **Author-controlled state:** Check whether the paper has *already been submitted to OpenReview/TMLR* before any new attempt; no such submission status was verified by this audit. Ensure active OpenReview profile and coauthor/affiliation conflict declarations as required.
2. **Build/artefact state:** In a clean clone, verify current commit, reproduce `make check-numbers` / non-mutating audit commands, test anonymous PDF text and its hyperlink targets for identifying information, inspect ZIP omissions and anonymization notes, and compare submitted PDF to exact audited source.
3. **Scientific limitations:** TEST feature arrays deliberately unreleased, so third parties cannot recompute stored TEST intervals from the repo alone; development feature files also unreleased. This must be explicit in the TMLR data/software availability statement, not disguised as full numerical reproduction from public files. No new TEST analysis: spent set.
4. **Policy and overlap:** Existing arXiv:2611.02333 is permissible under the TMLR policy, subject to the anonymized manuscript and the journal's current instructions. Verify exact bibliographic overlap and avoid duplicate arXiv upload.
5. **Source licensing:** Recheck TMLR style/template distribution rights and the licenses of model/datasets; only code under MIT is necessarily redistributed. Do not release sealed TEST arrays without separate decision.
6. **Submission gate:** The author explicitly approves the final PDF/supplement, authorship/AI disclosures and journal submission after the audit. This record does not upload or submit anything.

## Specific next execution

A new **read-only TMLR release audit** should inspect `paper/main_anon.tex`, `paper/body.tex`, `docs/PAPER_DRAFT.md`, package script and latest `results/audit/publication` entries; validate whether corresponding PDFs exist in Git or need regenerating locally. **Do not modify** `results/c1/*`, `TEST_OPENED.json`, frozen protocols or sealed test data. If release artefacts missing, prepare commands for the owner to generate in new local `dist/` without overwriting frozen files.

## Decision

`PRIORITIZE_LOOKAHEAD_TMLR_READ_ONLY_FINAL_AUDIT / ARXIV_ALREADY_POSTED / TMLR_SUBMISSION_STATUS_UNVERIFIED / NO_UPLOAD_OR_PUBLICATION_ACTION_AUTHORIZED`.
