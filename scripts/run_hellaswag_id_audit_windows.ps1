# Phase 3: HellaSwag repeated-id audit (reviewer request). CPU only, ~4 min. Reads the dataset and manifests only.
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\run_hellaswag_id_audit_windows.ps1"
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
& $py -m pytest -q tests\test_phase3_inventory.py -k "id_collision or hellaswag_id_audit"
if ($LASTEXITCODE -ne 0) { throw "audit tests failed (exit $LASTEXITCODE)" }
& $py scripts\phase3_hellaswag_id_audit.py --tokenizer
if ($LASTEXITCODE -ne 0) { throw "audit refused or failed (exit $LASTEXITCODE). The message is printed above." }
Write-Host 'Done. Paste the output above (results\phase3\hellaswag_id_audit.json is also written).'
