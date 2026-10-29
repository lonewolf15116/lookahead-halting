# Phase 3 A1: HellaSwag cross-split audit + TEST/DEV+ allocation. CPU only, ~5 min. Reads datasets and manifests only.
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\run_phase3_allocate_windows.ps1"
# Exit 3 = RETURN_FOR_REVIEW (HellaSwag pool below 750): send results\phase3\allocation.json. Refuses to overwrite earlier outputs.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
& $py -m pip install --quiet -e .
& $py -m pytest -q tests\test_phase3_allocation.py
if ($LASTEXITCODE -ne 0) { throw "allocation tests failed (exit $LASTEXITCODE)" }
& $py scripts\phase3_allocate.py --tokenizer; $rc = $LASTEXITCODE
if ($rc -eq 0) { Write-Host 'Allocation written. Paste the output above.' }
elseif ($rc -eq 3) { Write-Host 'RETURN FOR REVIEW (or used-set verification failed). Paste the output above.' }
else { throw "allocation refused or failed (exit $rc). The message is printed above." }
