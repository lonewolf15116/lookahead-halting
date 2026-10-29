# D-t2 GPU determinism check (docs/DT_TRUNCATION_RESULT.md; reviewer GO 2026-10-27). Diagnostic; development data only; TEST never opened.
# Two separate processes, because CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialises. Outputs are created exclusively.
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_dt2_windows.ps1"
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output." } }

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q tests\test_dt2_determinism.py tests\test_dt_truncation.py; Check 'D-t2 tests'
Remove-Item Env:CUBLAS_WORKSPACE_CONFIG -ErrorAction SilentlyContinue
& $py scripts\dt2_determinism_check.py --mode default; Check 'D-t2 default mode'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $py scripts\dt2_determinism_check.py --mode deterministic; Check 'D-t2 deterministic mode'
Remove-Item Env:CUBLAS_WORKSPACE_CONFIG -ErrorAction SilentlyContinue
& $py scripts\dt2_determinism_check.py --summarize; Check 'D-t2 summary'
Write-Host 'D-t2 done. Send the output; the reports are in results\diagnostics\dt2_*.json.'
