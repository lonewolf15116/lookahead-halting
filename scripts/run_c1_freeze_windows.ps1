# C1 freeze (docs/C1_PROTOCOL.md C1.7 step 2). ONLY after the reviewer approves the final protocol + this implementation and
# the protocol is tagged c1-protocol. Development data only (CPU selection) + rho timing on the GPU. TEST is never read.
#   powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_c1_freeze_windows.ps1" -ProtocolSha <sha256 of the approved docs/C1_PROTOCOL.md>
param([Parameter(Mandatory = $true)][string]$ProtocolSha)
$ErrorActionPreference = 'Stop'
$Root = Join-Path $HOME 'Projects\lookahead-halting'
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Add-Type -Namespace LhC1 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][LhC1.Power]::SetThreadExecutionState([uint32]2147483649)
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output." } }

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q tests\test_c1.py tests\test_e1.py tests\test_phase3_g3a.py tests\test_phase3_readouts.py; Check 'C1 tests'
& $py scripts\c1_freeze.py --protocol-sha $ProtocolSha; Check 'C1 freeze (see results\c1\frozen.json)'
Write-Host 'C1 freeze done. Send the output, results\c1\frozen.json and results\c1\pi1x_models.npz for the reviewer sign-off.'
[void][LhC1.Power]::SetThreadExecutionState([uint32]2147483648)
