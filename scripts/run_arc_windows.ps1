# Finish Gate 0 on Windows: clean probe records, then ARC-Challenge 25-shot at loops 1-4 against the Ouro paper.
# Run from PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\run_arc_windows.ps1"
# Order: install, tests, the two probes, a 25-question smoke test (a few minutes), then the full test set
# (about an hour on the RTX 5050). Keep the laptop plugged in and awake. Resumable: run it again to continue.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). The error is printed above." }
}

& $py -m pip install --quiet -r requirements.txt; Check 'requirements install'
& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q; Check 'tests'

# Re-record the Gate 0 probes with the fixed scripts. A failure here is recorded, not fatal.
& $py scripts\smoke_stock.py; $smoke = $LASTEXITCODE
& $py scripts\verify_adapter.py; $verify = $LASTEXITCODE
Write-Host "Probes: stock smoke exit $smoke, adapter check exit $verify"

# Smoke test of the ARC pipeline, then the full run.
& $py scripts\reproduce_ouro.py --limit 25; Check 'ARC smoke test'
& $py scripts\reproduce_ouro.py; $arc = $LASTEXITCODE
if ($arc -eq 0) { Write-Host 'Gate 0 accuracy: within 2 points of the paper.' }
else { Write-Host "ARC run finished with exit $arc. If a table printed above, that means outside 2 points; see results\gate0\gate0_accuracy.json." }
Write-Host "Done. Results are in $Root\results\gate0"
