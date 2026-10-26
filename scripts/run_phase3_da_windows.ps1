# Phase 3 D-a (reviewer Numerical Fidelity Review, 2026-10-26): float32 packed-vs-context-only check on PIQA and Social IQa,
# 100 unused TRAIN rows each (never DEV+ or TEST). Prints only per-benchmark state differences and PASS/FAIL; no labels,
# predictions or accuracy. About 5 GPU-minutes.
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\run_phase3_da_windows.ps1"
$ErrorActionPreference = 'Stop'
$Root = Join-Path $HOME 'Projects\lookahead-halting'
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output; the report is kept." } }

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q tests\test_phase3_da.py; Check 'D-a tests'
& $py scripts\diagnose_short_context_ood.py; Check 'D-a diagnostic (see results\phase3\diagnose_short_context_ood.json)'
Write-Host 'D-a done. Send the output and results\phase3\diagnose_short_context_ood.json for review before -Stage full.'
