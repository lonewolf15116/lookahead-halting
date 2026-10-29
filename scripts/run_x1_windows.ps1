# Exploratory study X1 (correctness flips at matched saving; development data only, no confirmatory claim). ONLY AFTER the
# reviewer's implementation review. CPU only; the sealed TEST data is never read.
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_x1_windows.ps1"
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
Add-Type -Namespace LhX1 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][LhX1.Power]::SetThreadExecutionState([uint32]2147483649)
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output." } }

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q tests\test_x1_flips.py tests\test_e1.py; Check 'X1 and E1 tests'
$t0 = Get-Date
& $py scripts\x1_flip_check.py; Check 'X1 (see results\exploratory\x1_flips.json)'
Write-Host ("X1 took {0:N1} min. Send the output and results\exploratory\x1_flips.json for the reviewer." -f ((Get-Date) - $t0).TotalMinutes)
[void][LhX1.Power]::SetThreadExecutionState([uint32]2147483648)
