# C1 ONE-TIME TEST RUN (docs/C1_PROTOCOL.md C1.7 step 3). ONLY after the reviewer approves the TEST script; it cannot be re-run.
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_c1_test_windows.ps1" -ApprovedCodeSha <sha256 from the review>
param([Parameter(Mandatory = $true)][string]$ApprovedCodeSha)
$ErrorActionPreference = 'Stop'
$Root = Join-Path $HOME 'Projects\lookahead-halting'
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output." } }

& $py -m pip install --quiet -e .; Check 'package install'
# tests run BEFORE the marker exists; they never read TEST
& $py -m pytest -q tests\test_c1_test.py tests\test_c1.py tests\test_e1.py tests\test_phase3_g3a.py tests\test_phase3_readouts.py; Check 'C1 tests'
& $py scripts\c1_test_run.py --approved-code-sha $ApprovedCodeSha; Check 'C1 TEST run (see results\c1\test_report.json)'
Write-Host 'C1 TEST run done. Send the output, results\c1\test_report.json and results\c1\TEST_OPENED.json.'
