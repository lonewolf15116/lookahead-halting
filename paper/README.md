# LaTeX paper (TMLR style)

Generated from `docs/PAPER_DRAFT.md` (formatting-only conversion).

## Build
    cd paper
    latexmk -pdf main_preprint.tex   # named / arXiv version (tmlr [preprint])
    latexmk -pdf main_anon.tex       # anonymized TMLR submission version
Both `\input{body.tex}` (shared body, incl. the AI-assistance statement) and use `refs.bib` with `tmlr.bst`.
Requires a TeX distribution with `latexmk` and `lmodern`. The PDFs in this folder were rebuilt on 2026-10-27 with the final Latin Modern fonts (TeX Live, pdfTeX 1.40.22); the only other embedded font is DejaVu Sans, from Figure 1. The preprint PDF still carries a placeholder e-mail until the real address is supplied.
`figures/fig1_rule_dependent_operating_points.pdf` is a copy of `../figures/...`.

## TODO items
- `main_preprint.tex`: replace placeholder email `TODO@aston.ac.uk`.
- `body.tex` start: the italic evidence-label legend is intentional and kept.
- `body.tex` end: "Statement on AI assistance" is a DRAFT for the author to edit.
- `refs.bib`: Fu2025, Olszowiec2026, Pappone2025 list surnames only (as in the Markdown); years of Fu2025/Graves2016/etc. taken from the Markdown entries/keys. Check bib details against the sources.
- `main_*.tex`: `\month`, `\year`, `\openreview` are TMLR placeholders (only used for the accepted version).
