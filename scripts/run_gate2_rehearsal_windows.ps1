# Gate 2 confirmation pipeline REHEARSAL on VALIDATION data (opens no confirmation data). ~5 min, CPU.
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & ".\scripts\run_gate2_rehearsal_windows.ps1"
# It must print "every frozen validation number ... reproduced". If it prints REHEARSAL FAILED, stop and send the output.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
& $py scripts\gate2_confirm.py --rehearsal-on-validation
if ($LASTEXITCODE -ne 0) { throw "rehearsal failed (exit $LASTEXITCODE)" }
