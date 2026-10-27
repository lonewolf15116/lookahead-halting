# Step 4 — Independent numerical and reproducibility audit

Audited commit: `ff0728c` (X1 closed). Audit date: 2026-10-27. Scope: the 20-page manuscript (`docs/PAPER_DRAFT.md`, `paper/body.tex`, both PDFs), `docs/X1_RESULT.md`, and every committed artifact they cite.
Rules followed: no frozen artifact modified; no TEST feature file opened or hashed; no new TEST analysis; no tolerance or threshold changed after seeing a result; **no manuscript or artifact corrected** (all remedies below are proposals awaiting approval).
Status labels: **PASS**, **FAIL**, **NEEDS QUALIFICATION (NQ)**. Basis labels: *counts* (recomputed from histograms/accuracies), *arithmetic* (from stored values), *stored* (stored statistic that cannot be recomputed without re-reading the sealed TEST set).

## 1. Overall verdict

**CONDITIONAL PASS — not yet clean.** Every scientific claim and every decision-relevant number reproduces; the PASS verdict, selections, thresholds and X1 reading are exactly as frozen. But the acceptance criterion "no unexplained numerical discrepancies" is not met as stated: **5 numeric checks FAIL** (4 distinct manuscript/documentation errors, all fully explained, none changes a conclusion) and several items need qualification. One reproducibility component (**E1 rerun on the original Windows environment**) is **PENDING** and is not marked PASS.

| Area | Checks | PASS | NQ | FAIL |
|---|---|---|---|---|
| Numerical integrity (`step4_numeric_audit.json`) | 394 | 388 | 1 | 5 |
| Provenance / protocol (`step4_provenance_audit.json`) | 98 | 97 | 1 | 0 |
| Manuscript consistency (`step4_manuscript_consistency.json`) | 41 | 38 | 3 | 0 |
| Release readiness (`step4_release_readiness.json`) | inventory | — | — | see §6 |
| Existing test suite (`pytest`) | 405 | 401 | — | 0 (4 skipped) |

No violation of the C1 protocol, the Phase 3 amendment A1 or the X1 rev 2 protocol was found. No unsupported confirmatory claim was found (see §5).

## 2. Verification table (what was checked and against what)

| Claim group | Source of truth | Basis | Result |
|---|---|---|---|
| Table 3 macro results; Table 4 exit histograms; gross saving; 5.5 net saving | `results/c1/test_report.json`, §3.5 formula | counts / arithmetic | 83/84 PASS, **1 FAIL (D1)** |
| Table 5 point-only vs screened; 5.3 bounds | `test_report.json`, `frozen.json` | arithmetic / stored | 32/33 PASS, **1 NQ (D4)** |
| TEST bootstrap CIs (+0.0529 [0.0488, 0.0571], point-only −0.022, r-kl +0.00125) | `test_report.json` | **stored** | Transcription PASS; **not recomputable** without re-reading the sealed TEST set (§4) |
| Table 1 ρ, 4.1 counts, 4.4 constants | `frozen.json`, manifests | arithmetic | 35/35 PASS |
| Table 6 baselines; Table 7 OOD (SIQA 3.26 pp, PIQA 0.71 pp) | `test_report.json` | counts | 107/107 PASS |
| E1, G3a, D-t, D-t2, D-a, D-a2, A1.5 planning bounds (z = 1.645, A1 line 145) | `results/exploratory`, `results/phase3`, `results/diagnostics` | counts / stored | 44/44 PASS |
| 7.3 in-sample bound margins | `frozen.json` primary | arithmetic | 8/9 PASS, **1 FAIL (D3)** |
| 7.3 X1 text numbers; `X1_RESULT.md` table and absolute rates | `x1_flips.json` | arithmetic / stored | 79/82 PASS (table 36/36), **3 FAIL (D2, D5a, D5b)** |
| Full SHA-256 of every frozen artifact; git ancestry; protocol-before-run ordering | git, files | — | 97/98 PASS, 1 NQ (D6) |
| TEST bundle hash recomputed from `git ls-tree` at the sealing commit; decision rule truth table; dev/TEST disjointness; model revision `574fa66c…` | `c1_test.verdict()`, manifests | — | PASS |
| md ↔ tex ↔ both PDFs: 435/412/454/454 salient numeric tokens, integers, label counts, cross-references, captions, citation keys, anonymisation | pdftotext | — | 38/41 PASS, 3 NQ (D8–D10) |

Frozen hashes confirmed unchanged (full SHA-256 in `step4_provenance_audit.json`): C1 protocol `4f38b8b7…`, `frozen.json` `65728e16…`, `pi1x_models.npz` `72f15508…`, `test_report.json` `02ba4b26…`, `TEST_OPENED.json` `b22e2d16…`, D-t report `1601a7e7…`, `x1_flips.json` `457617c7…`, X1 protocol rev 2 `b327cedc…`.

## 3. Discrepancy register

| ID | Status | File / location | Calculation | Proposed remedy (not applied) | Affects a conclusion? |
|---|---|---|---|---|---|
| D1 | **FAIL** | Manuscript Table 4 (md + tex), B_pb HellaSwag exit histogram printed `0 / 0 / 0 / 750` | `test_report.json` exit_hist gives `0 / 0 / 750 / 0` (all 750 questions exit at loop 3). The reported Table 3 savings recompute from the stored histogram, so only the Table 4 row is mistyped. | Change the row to `0 / 0 / 750 / 0` in md and tex; rebuild both PDFs. | No |
| D2 | **FAIL** | §7.3 text, "3.4 flips per 100" (native-pb, saving 0.20) | `x1_flips.macro` φ = 3.3471 per 100 → rounds to **3.3** | "3.4" → "3.3" (md + tex). | No (reading and ordering unchanged) |
| D3 | **FAIL** | §7.3, margin "0.69" (native-pb CSQA, in-sample bound margin) | `frozen.primary`: 0.68493 → **0.68** | "0.69" → "0.68". | No |
| D4 | NQ | §5.4, "−0.002" point-only π1-x − native-pb on CSQA | Exact value −0.00275 (→ −0.003). The printed −0.002 equals the difference of the two printed table entries, not the exact difference. | Either print −0.003, or add "(difference of the rounded entries)"; reword so the exact value is stated. | No |
| D5a/b | **FAIL** (×2) | `docs/X1_RESULT.md` absolute rates, native-pb φ at 0.20 and 0.30: `0.034`, `0.051` | `x1_flips.json`: 0.033471 → 0.033; 0.050496 → **0.050** | Correct both entries in the documentation. (This file is documentation, not the frozen `x1_flips.json`.) | No |
| D6 | NQ | `x1_flips.json` provenance: `git_dirty_code = True` at commit 5461ca7 | `X1_RESULT.md` documents one untracked old dry-run report and no differing code file; consistent with the audit's reading. | Keep the explanation in the manuscript's reproducibility note; no artifact change. | No |
| D7 | NQ | TEST CIs (§1 table) | Stored-only; see §4. | State in the paper that the TEST per-question arrays are required to regenerate the intervals; release them (see RR.test_unrepro). | No |
| D8 | NQ | Both PDFs | Fonts: CM + lmodern are present, but Nimbus Roman/Sans (Times stand-in) and Inter also embedded, i.e. the PDFs are not purely in the final TMLR fonts. | Rebuild PDFs on the laptop with the final TMLR style files. | No |
| D9 | NQ | `paper/main_preprint.tex` / preprint PDF | Placeholder e-mail `TODO@aston.ac.uk` present. | Insert the real address before posting the preprint. | No |
| D10 | NQ | `paper/body.tex`, `paper/main_preprint.tex` | TODO comments remain (comments only; none in the anonymous PDF text). | Remove before submission; confirm the AI-assistance statement. | No |
| D11 | PENDING | E1 rerun | Not run in the Linux VM; to be run in the original Windows environment. | Run `scripts/run_step4_rerun_windows.ps1` (§7). | — |

Discrepancies are not silently corrected: the numeric audit script reports the manuscript's printed value, the recomputed value, the rounding rule and the source for each, so each one reruns.

## 4. Reproducibility

Reruns used development data only (the script aborts if provenance mentions the sealed directory). The reruns were done on a different platform from the original (Linux, Python 3.10.12, numpy 2.2.6, scikit-learn 1.7.2) than the original (Windows, Python 3.12.10, numpy 2.5.4, scikit-learn 1.9.1).

| Rerun | Committed artifact | Leaves compared | Bit-identical | Numeric differences | Max abs diff | Non-numeric mismatches |
|---|---|---|---|---|---|---|
| C1 freeze (`C1.freeze(dev)`) | `results/c1/frozen.json` | 328 | 318 | 9 | 4.8e-07 | 1 (model SHA-256 only) |
| X1 (`run_x1(dev)`) | `results/exploratory/x1_flips.json` | 10,148 | 9,228 | 825 | 3.2e-06 | 95 (SHA-256 fingerprints only) |
| E1 | `results/exploratory/e1_report.json` | — | — | — | — | **PENDING** (Windows) |

Probe weights differ from the committed `pi1x_models.npz` by at most 1.2e-05 (means and SDs identical). Every non-numeric mismatch is a byte-hash fingerprint of a float array, which differs when the last bits differ. All selections, operating points, folds, thresholds' decision outcomes and the X1 reading are unchanged; the differences are the expected cross-platform BLAS/solver noise. **Not claimed:** bitwise reproduction across platforms. The same-environment bit-exactness test is the Windows run in §7. Summary: `results/audit/rerun_linux/rerun_summary.json`.

Not recomputable by design: the TEST bootstrap CIs and TEST point values (the TEST set is spent and its feature files must not be re-read); they are verified by transcription, by the pinned file hashes in `test_report.json`, and by recomputing the sealing-time bundle hash from git history. GPU feature extraction is not bitwise deterministic (D-t2: 16/20 questions differ across default repeats); stored features, not re-extraction, are the reproducible unit.

## 5. Protocol, provenance and label discipline

- C1: rules, δ = 0.02, B = 10,000, `default_rng(2026)`, paired benchmark-stratified bootstrap, screened-rule settings (2,000 resamples, `default_rng([0, 99, benchmark index])`, 51-point grid) and the verdict truth table match the frozen protocol; the PASS verdict follows from the stored statistics. The TEST bundle hash recomputed at the sealing commit equals the recorded one.
- X1: run only after its rev 2 protocol was committed; uses E1's held-out folds; dev-only; labelled exploratory [E]. The manuscript never calls X1 pre-registered or confirmatory ([C] appears 10 times, all on the primary comparison).
- Dev/TEST boundary: dev and TEST question sets are disjoint; the audit did not open or hash any file in `data/extracted_phase3_test`.

## 6. Release readiness (inventory; no changes made)

| Severity | Finding | Remedy |
|---|---|---|
| BLOCKER | Extracted features are git-ignored and unreleased; without them every number needs GPU re-extraction | Release dev features (and TEST arrays when appropriate) with the recorded SHA-256 |
| MAJOR | TEST numbers/CIs not recomputable by outsiders (70 pinned TEST hashes, no files) | Release TEST per-question arrays |
| MAJOR | Machine-specific paths in scripts; runners are Windows PowerShell only | Repo-relative paths; add a Makefile/`run_all.sh` |
| MAJOR | README still at Phase 0, no reproduction table | Rewrite with claim → command → artifact → hash |
| MAJOR | `requirements.txt` pins only transformers and lm_eval | Commit a lock file; state the tested range |
| MAJOR | No LICENSE; Figure 1 script not committed; TMLR anonymous code mirror missing | Add each |
| MINOR | No release tag; tables typed by hand; untracked working-tree files | Tag from the laptop; ship the number checker as `make check-numbers` |

## 7. Reproduction commands

```
# audit tooling (read-only, no GPU, ~minutes); outputs overwrite results/audit/*.json
python scripts/step4_numeric_audit.py          # exit 0 iff no FAIL (currently 5 FAIL = D1,D2,D3,D5a,D5b)
python scripts/step4_provenance_audit.py
python scripts/step4_manuscript_consistency.py
python scripts/step4_release_readiness.py
python -m pytest -q                            # 401 passed, 4 skipped

# development-data reruns (write only to the given new directory)
python scripts/step4_rerun_dev.py c1freeze --out results/audit/rerun_windows/c1freeze
python scripts/step4_rerun_dev.py e1       --out results/audit/rerun_windows/e1
python scripts/step4_rerun_dev.py x1       --out results/audit/rerun_windows/x1
# all of the above on Windows:
powershell -ExecutionPolicy Bypass -File scripts\run_step4_rerun_windows.ps1
```
`PYTHONPATH=src` is required (the runner sets it).

## 8. Audit-tooling note

One check of my own (`C.labels.x1_confirm`, added during the audit) initially failed because its exemption phrase was case-sensitive and the flagged sentence begins "After the confirmatory study". It was a bug in the check, not a manuscript issue; the check was fixed (case-insensitive) and rerun without changing what it tests. No thresholds or tolerances in any other check were altered.

## 9. Conclusion

The paper's claims are traceable to frozen artifacts and, where recomputable, correct. Before submission: fix D1–D5 (a rounding/transcription pass of ~6 edits), rebuild PDFs (D8), resolve D9–D10, run the Windows rerun (D11), and address the release items in §6. No re-analysis is needed.

---

# Addendum A — corrections applied (2026-10-27, after reviewer decision "CONDITIONAL PASS")

Sections 1–9 above are the immutable original record. This addendum records what was changed afterwards. Post-correction machine reports are in `results/audit/post_correction/` (the originals in `results/audit/` are untouched). No frozen artifact, protocol or C1/TEST result was modified; C1 was not rerun.

| ID | Reviewer decision | Action | Result |
|---|---|---|---|
| D1 | approve | **Retracted: audit error, not a manuscript error.** The expected string for Table 4 B_pb HellaSwag in `scripts/step4_numeric_audit.py` was mistyped (`0/0/0/750`); the manuscript (md and tex) already read `0 / 0 / 750 / 0`, matching `test_report.json`. The script was corrected; no manuscript edit. | PASS |
| D2 | approve | §7.3 "3.4" → "3.3" flips per 100 (md, tex) | PASS (3.347) |
| D3 | approve | §7.3 margin "0.69" → "0.68" (md, tex) | PASS (0.6849) |
| D4 | qualify | Kept −0.002, now labelled "differences of the rounded Table 5 entries; the unrounded CommonsenseQA difference is −0.00275" (md, tex) | Still reported NQ by the checker, by design: the printed value is not the exact difference; the qualification is now in the text |
| D5 | approve | `docs/X1_RESULT.md`: 0.034 → 0.033 and 0.051 → 0.050 for native-pb at 0.20 and 0.30. `x1_flips.json` untouched | PASS |

For D2, D3 and D5 the expected printed values hard-coded in the audit script were updated to the corrected text. This is a change of what the text says, not of any tolerance or recomputed value.

Post-correction results: numeric 394 checks, 393 PASS, 1 NQ (D4), 0 FAIL; provenance 98 checks, 97 PASS, 1 NQ (D6), 0 FAIL; consistency 41 checks, 36 PASS, 3 NQ, **2 FAIL**. The two consistency FAILs (`C.tex_pdf.anon.numbers`, `C.tex_pdf.preprint.numbers`) are expected and genuine: the LaTeX now says 0.68 and 0.00275 while both PDFs still print 0.69 (the PDFs are built on the laptop). They clear when the PDFs are rebuilt. Markdown ↔ LaTeX agreement passes.

Still open: **E1 Windows rerun** (D11); **PDF rebuild** with final fonts (D8, also clears the two FAILs above); **D9** the placeholder e-mail needs the real address for the preprint version (the anonymous version must stay without one); **D10** TODO comments in `paper/*.tex`, to be removed together with the rebuild; release items in §6 pending separate decisions (TEST arrays are not to be published without checking redistribution rights and the evaluation boundary; LICENSE and anonymous mirror need decisions). D6 and D7 remain documented, as the reviewer directed.

---

# Addendum B — E1/C1/X1 Windows rerun, PDF rebuild, final audit counts (2026-10-27)

Earlier sections and Addendum A are unchanged. New reports are in `results/audit/rerun_windows/` and `results/audit/post_rebuild/`; earlier report folders are untouched.

## B.1 Same-environment rerun (D11 closed)

Run with `scripts/run_step4_rerun_windows.ps1` on the original Windows environment (Python 3.12.10, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, pyarrow 26.0.0), at commit `6f833f1`, development data only, no sealed TEST file read, outputs written to a new folder (console log `results/audit/rerun_windows/console.txt`).

| Rerun | Committed artifact | Leaves compared | Bit-identical | Numeric differences | Non-numeric mismatches |
|---|---|---|---|---|---|
| C1 freeze | `results/c1/frozen.json` | 328 | **328** | 0 | 0 |
| E1 | `results/exploratory/e1_report.json` | 3,379 | **3,379** | 0 | 0 |
| X1 | `results/exploratory/x1_flips.json` | 10,148 | **10,148** | 0 | 0 |

All 12 probe-weight arrays are bit-identical to the committed `pi1x_models.npz`, and the model SHA-256 fingerprints match. Verdict: **bitwise reproduction in the original environment** (comparison excludes provenance, timestamps, timings and the GPU `rho` entries, which were not rerun). The earlier Linux rerun (Addendum summary in §4) remains **numerical agreement only** (max 1.2e-5 in probe weights, differing byte-hash fingerprints); the two must not be conflated. Not covered: E1/X1/C1-freeze reruns that start from GPU feature extraction (stored features are the reproducible unit).

The same run executed the numeric and provenance audits on Windows: 394 checks (393 PASS, 1 NQ, 0 FAIL) and 98 checks (97 PASS, 1 NQ, 0 FAIL), identical to the Linux counts.

Two defects of the runner were found by this run and fixed afterwards (no effect on any result): printing "−" crashed on a cp1252 console after the JSON had been written, and `SHA256SUMS.txt` could not be built because the transcript still held `console.txt` open. The scripts now force UTF-8 output and stop the transcript before hashing. `SHA256SUMS.txt` for this run therefore does not exist; file hashes can be recomputed from the committed copies.

## B.2 PDF rebuild (D8, D10 partly)

`latexmk` is not installed on the Windows laptop, so both PDFs were built in the linked laptop's Linux VM (TeX Live, pdfTeX 1.40.22, lmodern installed), from a copy of `paper/`; the repository on the laptop was not modified. Build logs: no undefined references or citations. Fonts (`results/audit/post_rebuild/*.pdffonts.txt`): Latin Modern (LMRoman, LMSans, LMMono, LMMath) plus DejaVu Sans (Figure 1) only; **no Nimbus/Times and no Inter**. D2, D3 and D4 render correctly (`3.3 flips per 100`, `0.68 pp`, `−0.00275` with the rounded-entries note; no `0.69 pp` remains). The anonymous PDF contains no author name, affiliation or handle. Both PDFs are 21 pages (the earlier 20 pages were with the Times substitute); the page-count expectation in the consistency script was updated accordingly and is the only expectation changed.

## B.3 Final counts after rebuild (`results/audit/post_rebuild/`)

| Audit | Checks | PASS | NQ | FAIL |
|---|---|---|---|---|
| Numerical | 394 | 393 | 1 (D4, by design) | 0 |
| Provenance | 98 | 97 | 1 (D6) | 0 |
| Manuscript consistency | 41 | 39 | 2 (placeholder e-mail D9, TODO comments D10) | 0 |

Closed: D1 (retracted), D2, D3, D5, D8, D11. D4 qualified in the text. D6 and D7 remain documented.

## B.4 Open items

- **D9** placeholder e-mail in `main_preprint` awaits the address authorised by the author. The anonymous build is unaffected.
- **D10** two TODO comments remain on purpose: the AI-assistance statement in `paper/body.tex` ("DRAFT --- author to review", printed in both PDFs) and the e-mail comment. The statement is the author's attestation, not an editorial note, so it is kept visible until the author supplies the final wording. Not scientific TODOs; none hides a methodological issue.
- **Release readiness** (`step4_release_readiness.json`): resolved: README, pinned environment lock, Figure 1 script, portable commands. Open: BLOCKER feature files not released; MAJOR TEST arrays not recomputable by outsiders, machine-specific paths in the older Windows runners and one test, LICENSE, anonymous mirror (the last three and any TEST release await explicit decisions); MINOR release tag, hand-typed tables, untracked files in the working tree.
- The PDFs must be rebuilt once more after the e-mail and the AI statement are final; use `make paper` or the VM procedure above.

Close-out recommendation: the numerical, protocol and reproducibility gates are met (0 FAIL; same-environment bitwise reproduction; PDF consistency clean). Step 4 can close once the e-mail and AI statement are settled and the PDFs are rebuilt, with the open release items explicitly accepted or scheduled.

---

# Addendum C — publication close-out (2026-10-27)

Earlier sections and Addenda A and B are unchanged. Final reports: `results/audit/final/`.

**Author decisions applied.** (1) The author-approved AI-assistance disclosure replaces the draft statement in `paper/body.tex` (both builds) and is added to `docs/PAPER_DRAFT.md`; the "DRAFT" label and its TODO comment are removed. (2) The preprint e-mail `mahesh15116@gmail.com`, supplied by the author, appears only in `paper/main_preprint.tex`; the anonymous build contains no name, affiliation or e-mail (checked in the rendered text). No TODO marker remains in `paper/*.tex`.

**Rebuild.** Both PDFs were rebuilt from these exact sources (md5 of `body.tex` and `main_preprint.tex` identical on the build machine and in the repository), in the laptop VM's TeX Live as in B.2: Latin Modern plus DejaVu Sans (Figure 1) only, no undefined references or citations, 21 pages each; the preprint shows the e-mail once, the anonymous PDF does not; the disclosure appears in both.

**Final counts**

| Audit | Checks | PASS | NQ | FAIL |
|---|---|---|---|---|
| Numerical | 394 | 393 | 1 (D4, qualified in the text) | 0 |
| Provenance | 98 | 97 | 1 (D6, dirty flag explained) | 0 |
| Manuscript consistency | 41 | 41 | 0 | 0 |

**Step 4 status.** All discrepancies are closed or documented (D1 retracted; D2, D3, D5, D8, D9, D10, D11 closed; D4, D6, D7 documented). Reproducibility evidence: Windows bitwise reproduction of C1 freeze, E1 and X1 on development data; Linux numerical agreement only. Not established: GPU feature-extraction reproducibility; independent recomputation of the sealed TEST intervals. Open release items, none of them executed: licence (MIT recommended, not authorised), anonymous code mirror (not authorised), TEST-array release (deferred pending rights and evaluation-integrity checks), release of development feature files (blocker for third-party reproduction; a data-availability statement must say they are not publicly regenerable until released), a release tag, and machine-specific paths in the older Windows runners.
