# Phase 3 D-a2 (reviewer D-a Failure Review, 2026-10-26): investigate the PIQA float32 D-a failure. Same 100 PIQA and 100
# Social IQa train rows as D-a; float32 per-loop localisation on the GPU, then a float64 CPU reference for the 10 PIQA
# questions with the largest float32 difference. Needs >= 16 GB free RAM (close browsers etc. first). Prints only state
# differences and the interpretation; no labels, predictions or accuracy. Roughly 10-20 minutes, mostly CPU.
#   cmd:  powershell -NoProfile -ExecutionPolicy Bypass -File "%USERPROFILE%\Projects\lookahead-halting\scripts\run_phase3_da2_windows.ps1"
# Option 2 (only after reviewer approval): add  -Mode stream-gpu  (bf16 weights on the GPU, float64 per module call; ~4 GB RAM)
param([ValidateSet('full-cpu', 'stream-gpu')][string]$Mode = 'full-cpu')
$ErrorActionPreference = 'Stop'
$Root = Join-Path $HOME 'Projects\lookahead-halting'
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Add-Type -Namespace Lh3a2 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Lh3a2.Power]::SetThreadExecutionState([uint32]2147483649)
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send the output; any report is kept." } }

& $py -m pip install --quiet -e .; Check 'package install'
if ($Mode -eq 'stream-gpu') { $env:LH_REQUIRE_CUDA = '1' }  # CUDA tests must run, not skip
& $py -m pytest -q -rs tests\test_phase3_da2.py tests\test_phase3_da.py; Check 'D-a2 tests'
& $py scripts\diagnose_short_context_ood_a2.py --mode $Mode; Check 'D-a2 (see results\phase3\diagnose_short_context_ood_a2.json)'
Write-Host 'D-a2 done. Send the output and results\phase3\diagnose_short_context_ood_a2.json for review.'
[void][Lh3a2.Power]::SetThreadExecutionState([uint32]2147483648)
