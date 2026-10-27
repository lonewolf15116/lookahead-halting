# When Does Adaptive Halting Win? Selection Rules Matter in Looped Language Models

Code, frozen protocols and result artifacts for a frozen Ouro-1.4B study (4 loops) comparing a trained stability probe (π1-x) with Ouro's native exit gate (native-pb) and an r-kl baseline, under two operating-point selection rules (point-only and bootstrap-screened).

- Manuscript sources: `docs/PAPER_DRAFT.md` (Markdown) and `paper/` (LaTeX, TMLR style; separate `main_anon.tex` and `main_preprint.tex` targets). Build notes: `paper/README.md`.
- Pre-registration: `docs/C1_PROTOCOL.md` (sha256 `4f38b8b7…`); amendment `docs/PHASE3_AMENDMENT_A1.md`; exploratory follow-up protocol `docs/X1_FLIP_PROTOCOL.md` (rev 2).
- Decision log: `docs/DECISIONS.md`. Audit: `docs/STEP4_REPRODUCIBILITY_AUDIT.md` (with addenda) and `results/audit/`.
- Evidence labels in the paper: [C] confirmatory, [S] pre-specified secondary, [P] post hoc, [E] exploratory. Only the π1-x vs native-pb comparison is [C]. X1 is exploratory.

## What can and cannot be reproduced from this repository

| Level | What | Needs | Status |
|---|---|---|---|
| 1 | Every number in the manuscript recomputed from committed artifacts | CPU, Python | **Reproducible now** (`make check-numbers`) |
| 2 | Frozen C1 selections, E1 and X1 recomputed from per-question development features | development feature files | **Not released yet** (git-ignored, about 0.9 GB); hash-pinned in `results/c1/frozen.json` |
| 3 | C1 TEST statistics and bootstrap intervals | the sealed TEST feature files | **Not recomputable by outsiders**; the TEST set is spent (opened once, `results/c1/TEST_OPENED.json`). The results are verified against pinned file hashes and the sealing-time bundle hash |
| 4 | Feature extraction from the model | CUDA GPU (RTX 5050 8 GB, bf16) | Not bitwise reproducible (GPU nondeterminism, see manuscript 3.6); stored features are the reproducible unit |

Data availability: benchmark datasets are public (Hugging Face hub, revisions recorded in `data/manifests/` and `results/phase3/inventory.json`; check each dataset's licence before redistribution). Extracted features are not yet redistributed. Whether and when to release them (including any TEST arrays) is an open decision pending a check of dataset redistribution rights and of the evaluation boundary. **No licence has been chosen yet**, so no reuse rights are granted by default.

## Environment

Results were produced on Windows 11, Python 3.12.10, torch 2.11.0+cu128, numpy 2.5.4, scipy 1.18.1, scikit-learn 1.9.1, pandas 3.0.6, pyarrow 26.0.0, transformers 4.54.1 (exact list: `requirements-lock.txt`; recorded in `results/gate0/environment.json`). `requirements.txt` gives the loose install list. Model: `ByteDance/Ouro-1.4B`, revision `574fa66cb8bf5abdc979642d01cf2b79b16bfab1` (needs `trust_remote_code`). The audit also ran on Linux, Python 3.10.12, numpy 2.2.6, scikit-learn 1.7.2: decision-relevant outputs were identical and numeric values agreed to at most 1.2e-5 (probe weights) and 3.2e-6 (thresholds), but byte hashes of float arrays differ. That is numerical agreement, not bitwise reproduction.

```
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt  # or the pinned versions in requirements-lock.txt
export PYTHONPATH=src            # Windows: $env:PYTHONPATH = "src"
```

## Claim → command → artifact

| Paper element | Command | Artifact |
|---|---|---|
| All tables, results and discussion numbers (Level 1) | `python scripts/step4_numeric_audit.py --out results/audit/latest/numeric.json` (or `make check-numbers`) | `results/c1/test_report.json`, `results/c1/frozen.json`, `results/exploratory/*.json` |
| Provenance, hashes, TEST bundle, dev/TEST boundary | `python scripts/step4_provenance_audit.py` | `results/audit/step4_provenance_audit.json` |
| Markdown vs LaTeX vs PDFs | `python scripts/step4_manuscript_consistency.py` | `results/audit/step4_manuscript_consistency.json` |
| Figure 1 | `python scripts/make_fig1.py` | `figures/fig1_*.png/.pdf` from `results/c1/test_report.json` |
| C1 freeze (selections, thresholds, probe) [Level 2] | `python scripts/c1_freeze.py` | `results/c1/frozen.json`, `pi1x_models.npz` |
| C1 TEST run (one time only, spent) | `python scripts/c1_test_run.py` | `results/c1/test_report.json`, `TEST_OPENED.json` |
| E1 held-out diagnostics [E] | `python scripts/e1_run.py` | `results/exploratory/e1_report.json` |
| X1 correctness flips [E] | `python scripts/x1_flip_check.py` | `results/exploratory/x1_flips.json`, `docs/X1_RESULT.md` |
| Truncation / determinism diagnostics | `python scripts/dt_truncation_check.py`, `python scripts/dt2_determinism_check.py` | `results/diagnostics/` |
| Reruns on development data into a new folder | `make rerun-dev` / `scripts\run_step4_rerun_windows.ps1` | `results/audit/rerun_*/` |
| Unit tests | `python -m pytest -q` (401 passed, 4 skipped at audit time) | |
| Paper PDFs | `scripts\build_paper_windows.ps1` or `make paper` | `paper/main_anon.pdf`, `paper/main_preprint.pdf` |

**Warning:** `c1_freeze.py`, `e1_run.py` and `x1_flip_check.py` write to the committed artifact paths shown above. To check reproducibility without overwriting anything use `scripts/step4_rerun_dev.py` (writes to a new folder). `c1_test_run.py` refuses to run again (one-time marker).

Feature extraction and the earlier phases (Gate 0 to Phase 3) are run by the `scripts/run_*_windows.ps1` wrappers in the order recorded in `docs/DECISIONS.md`. Those older wrappers assume the repository at `%USERPROFILE%\Projects\lookahead-halting` and a `.venv` inside it; the newer ones (`run_step4_rerun_windows.ps1`, `build_paper_windows.ps1`) locate the repository from their own path.

## Rules this repository follows

Frozen artifacts (protocols, `results/c1/*`, `results/exploratory/x1_flips.json`) are never edited after the run that produced them. The sealed TEST feature files are not read by any audit or rerun script. Corrections to the manuscript are logged in `docs/DECISIONS.md` and in the dated addenda of the audit.
