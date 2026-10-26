# Exploratory study E1 (diagnostics on development data; no confirmatory claim). ONLY AFTER the reviewer's implementation
# review. CPU only; the sealed TEST data is never read.
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_e1_windows.ps1"
$ErrorActionPreference = 'Stop'
$Root = Join-Path $HOME 'Projects\lookahead-halting'
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
Add-Type -Namespace LhE1 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][LhE1.Power]::SetThreadExecutionState([uint32]2147483649)
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output." } }

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q tests\test_e1.py tests\test_phase3_g3a.py tests\test_phase3_readouts.py; Check 'E1 tests'
$t0 = Get-Date
& $py scripts\e1_run.py; Check 'E1 (see results\exploratory\e1_report.json)'
Write-Host ("E1 took {0:N1} min. Send the output and results\exploratory\e1_report.json for the reviewer." -f ((Get-Date) - $t0).TotalMinutes)
[void][LhE1.Power]::SetThreadExecutionState([uint32]2147483648)
