# D-t truncated-execution check (docs/DT_TRUNCATION_PLAN.md rev 1). Diagnostic [E]; development data only, TEST is never opened.
# Laptop GPU; about 600 questions x (1 full + 3 truncated) runs, expected a few minutes. Creates
# results\diagnostics\dt_truncation_report.json exclusively (the run refuses to overwrite; there is no re-run to "fix" a result).
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_dt_truncation_windows.ps1"
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Add-Type -Namespace LhDt -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][LhDt.Power]::SetThreadExecutionState([uint32]2147483649)
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output." } }

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q tests\test_dt_truncation.py tests\test_c1_test.py tests\test_c1.py; Check 'D-t tests'
& $py scripts\dt_truncation_check.py; Check 'D-t truncation check (see results\diagnostics\dt_truncation_report.json)'
Write-Host 'D-t done. Send the output and results\diagnostics\dt_truncation_report.json.'
[void][LhDt.Power]::SetThreadExecutionState([uint32]2147483648)
