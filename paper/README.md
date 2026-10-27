# LaTeX paper (TMLR style)

Generated from `docs/PAPER_DRAFT.md` (formatting-only conversion).

## Build
    cd paper
    latexmk -pdf main_preprint.tex   # named / arXiv version (tmlr [preprint])
    latexmk -pdf main_anon.tex       # anonymized TMLR submission version
Both `\input{body.tex}` (shared body, incl. the AI-assistance statement) and use `refs.bib` with `tmlr.bst`.
Requires a TeX Live with `lmodern` (loaded by `tmlr.sty`). The PDFs currently in this folder were built in a
sandbox without `lmodern`, using a Times substitute (mathptmx/helvet); rebuild locally for the true TMLR fonts.
`figures/fig1_rule_dependent_operating_points.pdf` is a copy of `../figures/...`.

## TODO items
- `main_preprint.tex`: replace placeholder email `TODO@aston.ac.uk`.
- `body.tex` start: the italic evidence-label legend is intentional and kept.
- `body.tex` end: "Statement on AI assistance" is a DRAFT for the author to edit.
- `refs.bib`: Fu2025, Olszowiec2026, Pappone2025 list surnames only (as in the Markdown); years of Fu2025/Graves2016/etc. taken from the Markdown entries/keys. Check bib details against the sources.
- `main_*.tex`: `\month`, `\year`, `\openreview` are TMLR placeholders (only used for the accepted version).
