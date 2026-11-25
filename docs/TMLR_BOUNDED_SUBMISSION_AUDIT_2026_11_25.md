# Adaptive Halting — Bounded TMLR Submission Audit (2026-11-25)

**Decision:** `CORRECTIONS_REQUIRED` (artifact-level / account-state verification, **not** scientific redesign).
**Scope:** read-only; no edits to paper, frozen protocols, result files, sealed TEST arrays, or submissions.

## Examined, directly in current audit

- GitHub `paper/main_anon.tex` (blob `b6b55dc3d5a5b1b328cd26f6e229f44b8a06a6d4`) is anonymous; `paper/body.tex` (blob `c335d21d904e35f38c37ab629eb3b5db8a7c5625`) contains reported results and limitations.
- Anonymous PDF exists in GitHub at `paper/main_anon.pdf` (blob `568041072289c1aec0f85930aee0985f08158d6a`, 426207 bytes, magic PDF-1.5, EOF/xref). Direct byte inspection found PDF metadata `/Author()`, `/Title()`, `/Creator(LaTeX with hyperref)`, `/Producer(pdfTeX-1.40.22)`, build timestamp `2026-10-29T07:31:15Z`. Direct PDF **rendered text and embedded hyperlink targets have not been newly extracted or visually inspected in this session**; do not claim otherwise.
- GitHub `results/audit/publication/step4_manuscript_consistency.json` exists (blob `b419b7363a39d65e07e193926f0af79a5ed8625e`) and reports **40/40 PASS** from previous audit, including 21 pages, PDF citations, fonts, metadata, no identifying source/PDF text, and Markdown/LaTeX/PDF salient-number agreement. This is verified **as an existing signed-off audit record**, not a new run.
- GitHub `scripts/make_anonymous_package.py` exists and deterministically builds archive from tracked Git files; excludes named preprint/citation and files with identifying strings; documentation-only redactions; produces `ANONYMIZATION_NOTES.md`; intended output is `anonymous_code_package.zip`. Prior Addendum D reports 259 files, 26 excluded, five redacted; 393/394 numeric checks PASS and 66 sampled tests PASS, two skipped. **Current ZIP absent from GitHub**; actual current package bytes, size, exact omissions, runnable tests and anonymity **cannot be verified now**. Local regenerated artifact required.
- `docs/PAPER_DRAFT.md` / `paper/body.tex`: availability statement is candid that dev features ~0.9GB and TEST arrays are unreleased; third parties can recompute stored-result arithmetic, NOT independent TEST bootstrap intervals. Manuscript states loop saving is retrospective and not measured end-to-end latency. Confirmatory screened-rule +0.053 [+0.049,+0.057]; point-only reversal -0.022 [-0.026,-0.018] is secondary; KL comparison +0.00125 [-0.0041,+0.0065] is non-superiority; no formal risk-control guarantee.
- `docs/STEP4_REPRODUCIBILITY_AUDIT.md` Addendum D: 393/394 numeric PASS, 1 qualified; 97/98 provenance PASS, 1 qualified; consistency 40/40 PASS; Windows dev reruns bitwise, Linux numeric only, sealed TEST intervals not independently rederived. Current independent reruns NOT performed.
- External TMLR author guidelines as of 2026-11-25: anonymized TMLR-template PDF, supplementary ZIP/PDF max 100MB, complete OpenReview profiles/conflicts; signed preprint on arXiv is allowed when the **submitted PDF/supplement** does not link to identified version. Official https://www.jmlr.org/tmlr/author-guide.html and https://jmlr.org/tmlr/editorial-policies.html .
- The existing arXiv preprint is `2611.02333`, first posted 2026-11-02. Do NOT create a duplicate arXiv submission.
- Connected Gmail searches for manuscript title/arXiv ID with OpenReview/TMLR and OpenReview sender after 2026-10-01 found **no TMLR submission confirmation**; one OpenReview alternate-address confirmation on 2026-10-18. This cannot rule out submission from a different account or an invitation/decision outside mail search. Authenticated TMLR OpenReview dashboard status **NOT VERIFIED**. No public anonymous submission search used for de-anonymization.

## Remaining blockers — exactly three

**B1 PACKAGE BYTE VALIDATION (hard gate):** Generate `anonymous_code_package.zip` from a *recorded git commit* on owner's computer without overwriting frozen results; capture SHA-256, size (must be <=100MB), `ANONYMIZATION_NOTES.md`, exclusions and anonymization scan. Optionally reproduce the documented numeric audit/test subset from an isolated extraction. No ZIP is supplied in the present workspace.

**B2 CURRENT PDF BINARY CHECK (hard gate):** The committed PDF bytes and metadata were checked, and previous consistency tests passed; before upload, render/read the **exact PDF being uploaded** to confirm all 21 pages/figure/font/citation layout, external link targets and absence of name, email, affiliation, preprint/GitHub identity and embedded identifying metadata. Capture PDF SHA-256 and match the file sent to OpenReview. The current GitHub PDF timestamp is Oct 29, 2026; if sources changed since that build, rebuild from a clean checked-out commit and rerun consistency. No need to rerun sealed TEST data.

**B3 OPENREVIEW AUTHOR STATUS (hard gate):** Sign in to https://openreview.net/group?id=TMLR and inspect the author's own active/past submissions and OpenReview profile; confirm no existing submission of this manuscript, complete author/conflict/ethics details and author authorization before any new upload. No submission confirmation was located in searched connected mail, which is not conclusive.

## Non-blocking but material disclosure

Availability statement candidly states unreleased dev features and sealed TEST arrays; this limits complete third-party reproduction of the TEST bootstrap intervals. Do not claim comprehensive independent end-to-end reproducibility. Discuss whether to supply development features/anonymized hashes after rights review, but **do not release TEST arrays without the owner's explicit separate rights/integrity decision**. TMLR requires CC BY 4.0 license for submitted manuscript; this is distinct from MIT licensing of author code and third-party dataset/model terms.

## Final decision

`CORRECTIONS_REQUIRED / SCIENTIFIC_CLAIMS_ACCEPTABLE_AS_SCOPED / PDF_PRESENT_PRIOR_AUDIT_PASS / ZIP_NOT_AVAILABLE_FOR_BYTE_VERIFICATION / OPENREVIEW_STATUS_UNVERIFIED / NO_SUBMISSION_ACTION`.

No additional study, manuscript redesign, new theory, or unsealing is authorized or required by this audit.
