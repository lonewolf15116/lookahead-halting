# Gate 2 shuffled-control audit + fixed-depth analysis. Train and validation only, CPU, a few minutes.
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & ".\scripts\run_control_audit_windows.ps1"
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
& $py scripts\gate2_control_audit.py --root .
if ($LASTEXITCODE -ne 0) { throw "audit failed (exit $LASTEXITCODE)" }
Write-Host 'Done. Send the output above (results\gate2\control_audit.json is also written).'
