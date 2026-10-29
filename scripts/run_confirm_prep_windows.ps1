# Gate 2, confirmation set: build it and run the contamination audit (CPU only; no extraction happens here).
# Run from PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\run_confirm_prep_windows.ps1"
# Order: tests (~7 min), the native-id diagnostic (a few min), a --limit 20 smoke run, then the real build. The real build rebuilds all Phase 1 questions
# with exemplar recording (the same work as prepare_phase1.py, roughly 10-20 min) and downloads the unused splits.
# Exit 2 from the real build means the audit failed: stop and send results\gate2\confirmation_audit.json.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). The error is printed above." }
}

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q; Check 'tests'
& $py scripts\diagnose_confirm_ids.py; Check 'native-id diagnostic'
& $py scripts\prepare_confirmation.py --limit 20; Check 'confirmation smoke run'
& $py scripts\prepare_confirmation.py; $rc = $LASTEXITCODE
if ($rc -eq 0) { Write-Host 'Contamination audit PASSED. Manifest written; send the output above. Extraction has NOT started.' }
elseif ($rc -eq 2) { Write-Host 'Contamination audit FAILED. No manifest written. Send results\gate2\confirmation_audit.json.' }
else { throw "prepare_confirmation.py failed (exit ${rc})." }
