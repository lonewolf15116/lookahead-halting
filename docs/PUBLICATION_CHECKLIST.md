# Publication checklist (TMLR submission + arXiv preprint)

Nothing in this list has been executed on the author's behalf: uploading, posting, making the repository public and tagging are the author's actions.
Evidence that the manuscript, numbers and reproducibility claims are consistent: `docs/STEP4_REPRODUCIBILITY_AUDIT.md` (Addenda A to D) and `results/audit/publication/`.

## Files to submit

| Target | File | How it was produced |
|---|---|---|
| TMLR (OpenReview), main PDF | `paper/main_anon.pdf` | `paper/main_anon.tex` (anonymous mode), built with `latexmk`, Latin Modern + DejaVu Sans fonts only |
| TMLR, supplementary code | `anonymous_code_package.zip` | `python scripts/make_anonymous_package.py` (deterministic; omitted and redacted files are listed in its `ANONYMIZATION_NOTES.md`) |
| arXiv, source | `dist/arxiv_source.zip` (+ `dist/arxiv_test_build.pdf` as the expected result) | `python scripts/make_submission_packages.py arxiv --out dist` (needs `latexmk`, `lmodern`; compiled from the shipped `.bbl` with plain `pdflatex`, no undefined references) |
| arXiv, reference PDF | `paper/main_preprint.pdf` | `paper/main_preprint.tex` (named preprint) |

`dist/` and the zip are build outputs (not committed); regenerate them from the commit you tag.

## Before you submit

1. **Code link.** The preprint cites `https://github.com/lonewolf15116/lookahead-halting`, which is private. Make the repository public before posting (or change `\codeavailability` in `paper/main_preprint.tex` and rebuild). The anonymous TMLR PDF refers to supplementary material and contains no link or name.
2. **Tag the release** from your laptop (the cloud session cannot push tags): `git tag paper-v1 <commit> && git push origin paper-v1`.
3. **Check TMLR's current policy** on posting an arXiv version while the paper is under review, and its current submission requirements (template version, reviewer suggestions, conflict-of-interest and AI-use disclosure fields). Re-read the instructions on OpenReview on the day.
4. **Check third-party material.** Datasets and Ouro-1.4B are not redistributed; `tmlr.sty`, `tmlr.bst` and `fancyhdr.sty` are bundled in `paper/` and in the arXiv source (check the TMLR template licence if arXiv questions it).
5. **Do not release TEST arrays.** They stay unreleased (redistribution rights and evaluation boundary unchecked).

## arXiv metadata (suggested; confirm each field in the arXiv form)

- Title: When Does Adaptive Halting Win? Selection Rules Matter in Looped Language Models
- Author: Mahesh Reddy Pagadala (Aston University); contact mahesh15116@gmail.com
- Primary category: cs.LG (secondary: cs.CL, optional)
- Comments: 21 pages, 1 figure. Code: see repository. Evidence labels [C]/[S]/[P]/[E] in the text; the X1 analysis is exploratory.
- License: your choice in the arXiv form (not decided here).
- Abstract (1821 characters; arXiv allows 1,920):

Looped language models can stop iterating early, and recent work shows that post-hoc readouts of the loop trajectory often rival the learned exit gate of the looped model Ouro. Such comparisons reduce each method to one operating point chosen by a selection rule. We ask whether that rule decides the outcome. On a frozen Ouro-1.4B with four loops, we compared a trained stability probe (π1-x) with Ouro's native gate, adapted to a per-question stopping decision, in pre-registered studies; the confirmatory study was designed after an exploratory finding on development data and evaluated once on a sealed test set of 8,517 questions. In the pre-registered confirmatory comparison, which selected operating points with a bootstrap-screened 1 pp accuracy-loss rule, the probe saved more of the four-loop budget than the native gate (+0.053 gross macro saving over three benchmarks, 95% CI [+0.049, +0.057]), mostly from CommonsenseQA, where the probe's test upper bound on accuracy loss exceeded 1 pp. Under a point-only rule, a pre-specified secondary analysis on the same test set reversed the preference (−0.022 [−0.026, −0.018]), with operating points that exceeded the budget. In pre-specified secondary comparisons, the probe showed no demonstrated advantage over an untrained KL divergence between successive loops (+0.00125 [−0.0041, +0.0065]); we emphasise this result post hoc and claim no equivalence. The probe also broke its budget on one of two out-of-distribution benchmarks (3.26 pp loss on Social IQa). Savings are estimated from stored full-depth traces, not measured inference time, and our screen carries no formal risk-control guarantee. We conclude that halting methods should be compared as signals together with selection procedures, with strong untrained baselines and out-of-distribution checks.

## After submission

- Record the arXiv and OpenReview identifiers in `docs/DECISIONS.md`.
- Anything that changes the manuscript after this commit requires rebuilding both PDFs and rerunning `python scripts/step4_manuscript_consistency.py` and `python scripts/step4_numeric_audit.py` into a new directory.
