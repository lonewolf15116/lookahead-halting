# Phase 3 G3a (D1-D4) on development data. ONLY AFTER the reviewer's implementation sign-off. CPU only; TEST is never read.
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_phase3_g3a_windows.ps1"
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
Add-Type -Namespace Lh3g -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Lh3g.Power]::SetThreadExecutionState([uint32]2147483649)
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output." } }

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q tests\test_phase3_g3a.py tests\test_phase3_readouts.py; Check 'G3a tests'
$t0 = Get-Date
& $py scripts\phase3_g3a.py; Check 'G3a (see results\phase3\g3a_report.json)'
Write-Host ("G3a took {0:N1} min. Send the output and results\phase3\g3a_report.json for the reviewer's GO/STOP." -f ((Get-Date) - $t0).TotalMinutes)
[void][Lh3g.Power]::SetThreadExecutionState([uint32]2147483648)
