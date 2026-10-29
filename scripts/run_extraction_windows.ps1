# Phase 1 extraction on Windows: tests, a small end-to-end extraction and its acceptance checks, then (only if
# every check passes) the full extraction (about 4 h on the RTX 5050) and the same checks on it.
# Run from PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & ".\scripts\run_extraction_windows.ps1"
# Keep the laptop plugged in. The script asks Windows not to sleep while it runs. If it stops for any reason,
# run it again: completed parts are kept and the run continues where it stopped.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

# Keep the system awake for the lifetime of this PowerShell process (ES_CONTINUOUS | ES_SYSTEM_REQUIRED).
Add-Type -Namespace Lh -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Lh.Power]::SetThreadExecutionState([uint32]2147483649)

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). The error is printed above." }
}

& $py -m pip install --quiet -e .; Check 'package install'
& $py -m pytest -q; Check 'tests'

# Small end-to-end extraction: first 40 questions of every benchmark, then the six acceptance checks on what was stored.
& $py scripts\extract_phase1.py --limit 40; Check 'smoke extraction'
& $py scripts\verify_extraction.py --limit 40; Check 'smoke acceptance checks'

# Full extraction, only reached if everything above passed.
$t0 = Get-Date
& $py scripts\extract_phase1.py; Check 'full extraction'
Write-Host ("Full extraction took {0:N1} h" -f ((Get-Date) - $t0).TotalHours)
& $py scripts\verify_extraction.py; Check 'full acceptance checks'
[void][Lh.Power]::SetThreadExecutionState([uint32]2147483648)
Write-Host "Done. Records in $Root\data\extracted, checks in $Root\results\phase1\extraction_check.json"
