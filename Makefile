# Portable entry points (Linux/macOS/WSL; on Windows use the scripts\*.ps1 runners or run the python commands directly).
# Nothing here reads the sealed TEST data; GPU feature extraction is NOT part of these targets (see README.md).
PY ?= python
export PYTHONPATH := src

.PHONY: test check-numbers audit fig1 paper rerun-dev

test:            ## unit tests (CPU)
	$(PY) -m pytest -q

check-numbers:   ## recompute every manuscript number from the committed artifacts (exit 0 iff no FAIL)
	$(PY) scripts/step4_numeric_audit.py --out results/audit/latest/step4_numeric_audit.json

audit: check-numbers   ## all four audit scripts, written to results/audit/latest/
	$(PY) scripts/step4_provenance_audit.py --out results/audit/latest/step4_provenance_audit.json
	$(PY) scripts/step4_manuscript_consistency.py --out results/audit/latest/step4_manuscript_consistency.json
	$(PY) scripts/step4_release_readiness.py --out results/audit/latest/step4_release_readiness.json

fig1:            ## regenerate Figure 1 from results/c1/test_report.json
	$(PY) scripts/make_fig1.py

paper:           ## build both PDFs (needs latexmk + lmodern)
	cd paper && latexmk -pdf main_preprint.tex && latexmk -pdf main_anon.tex

rerun-dev:       ## development-data reruns into a NEW directory (needs the development feature files)
	@test ! -e results/audit/rerun_local || (echo "results/audit/rerun_local exists" && exit 1)
	$(PY) scripts/step4_rerun_dev.py c1freeze --out results/audit/rerun_local/c1freeze
	$(PY) scripts/step4_rerun_dev.py e1 --out results/audit/rerun_local/e1
	$(PY) scripts/step4_rerun_dev.py x1 --out results/audit/rerun_local/x1
