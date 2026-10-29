# Gate 2 confirmation set: extraction and acceptance checks (GPU, about 1 h). Outcomes stay sealed: nothing here prints
# or analyses correctness, transitions, savings or accuracy; the verifier prints only PASS/FAIL, question counts and
# record status counts.
# Run from PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & ".\scripts\run_confirm_extraction_windows.ps1"
# Order: tests, a 40-question-per-benchmark smoke extraction + checks (data\extracted_confirm_smoke), then, only if
# every check passes, the full extraction (data\extracted_confirm) + checks. Re-running resumes; completed parts are kept.
# Extraction refuses to start unless data\manifests\confirmation.json, confirmation_ids.csv and the passed audit match.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

Add-Type -Namespace Lh -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Lh.Power]::SetThreadExecutionState([uint32]2147483649)

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output; failing reports are kept." }
}

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q; Check 'tests'

& $py scripts\extract_phase1.py --set confirmation --limit 40; Check 'confirmation smoke extraction'
& $py scripts\verify_extraction.py --set confirmation --limit 40; Check 'confirmation smoke checks'

$t0 = Get-Date
& $py scripts\extract_phase1.py --set confirmation; Check 'confirmation full extraction'
Write-Host ("Confirmation extraction took {0:N2} h" -f ((Get-Date) - $t0).TotalHours)
& $py scripts\verify_extraction.py --set confirmation; Check 'confirmation full checks'
[void][Lh.Power]::SetThreadExecutionState([uint32]2147483648)
Write-Host "Done. Records in $Root\data\extracted_confirm, checks in $Root\results\gate2\extraction_check_confirm.json"
