# Phase 3 A1: build the DEV+ and sealed TEST questions, then extract them (GPU). TEST outcomes stay sealed: nothing here
# prints or analyses correctness, transitions, savings or accuracy; the verifier prints only PASS/FAIL, counts and record status.
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\run_phase3_extraction_windows.ps1" -Stage build
#   ... reviewer verifies the build report and smoke checks ...
#   & "$HOME\Projects\lookahead-halting\scripts\run_phase3_extraction_windows.ps1" -Stage full
# Stage build: tests, question build (CPU, refuses to overwrite), 40-question smoke extraction + checks for DEV+ and TEST.
# Stage full: full DEV+ and TEST extraction + checks (about 80 GPU-minutes in total). Re-running resumes; completed parts are kept.
param([Parameter(Mandatory = $true)][ValidateSet('build', 'full')][string]$Stage)
$ErrorActionPreference = 'Stop'
$Root = Join-Path $HOME 'Projects\lookahead-halting'
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Add-Type -Namespace Lh3 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Lh3.Power]::SetThreadExecutionState([uint32]2147483649)
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output; reports are kept." } }

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q tests\test_phase3_build.py tests\test_phase3_allocation.py tests\test_phase3_inventory.py; Check 'phase 3 tests'
if ($Stage -eq 'build') {
    & $py scripts\prepare_phase3.py; Check 'question build (see results\phase3\build_report.json)'
    foreach ($s in 'phase3_dev', 'phase3_test') {
        & $py scripts\extract_phase1.py --set $s --limit 40; Check "$s smoke extraction"
        & $py scripts\verify_extraction.py --set $s --limit 40; Check "$s smoke checks"
    }
    Write-Host 'Build and smoke checks done. Send the output, results\phase3\build_report.json and data\manifests\phase3_questions.json for review before -Stage full.'
} else {
    foreach ($s in 'phase3_dev', 'phase3_test') {
        $t0 = Get-Date
        & $py scripts\extract_phase1.py --set $s; Check "$s full extraction"
        Write-Host ("$s extraction took {0:N2} h" -f ((Get-Date) - $t0).TotalHours)
        & $py scripts\verify_extraction.py --set $s; Check "$s full checks"
    }
    Write-Host 'Full extraction and checks done. Send the output; checks are in results\phase3\extraction_check_phase3_*.json.'
}
[void][Lh3.Power]::SetThreadExecutionState([uint32]2147483648)
