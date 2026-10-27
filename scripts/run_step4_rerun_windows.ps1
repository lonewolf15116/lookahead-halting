# Step 4: same-environment reproducibility rerun (development data only; nothing in results/ other than results/audit/rerun_windows is written).
# Run from the repo root after `git pull`:  powershell -ExecutionPolicy Bypass -File scripts\run_step4_rerun_windows.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$out = "results\audit\rerun_windows"
foreach ($w in @("c1freeze", "e1", "x1")) {
    python scripts\step4_rerun_dev.py $w --out "$out\$w"
}
python scripts\step4_numeric_audit.py
python scripts\step4_provenance_audit.py
python scripts\step4_manuscript_consistency.py
git status --short
