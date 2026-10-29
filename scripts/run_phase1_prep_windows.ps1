# Phase 1, steps 1-4 on Windows: tests, synthetic data, question build + token counts + split manifest, packed pilot.
# Run from PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & ".\scripts\run_phase1_prep_windows.ps1"
# Order: tests (~1-2 min), synthetic sets (seconds), prepare (CPU; first run downloads the datasets, MMLU is the
# largest, roughly 10-20 min), pilot on ARC-C (GPU, roughly 15-25 min). No full extraction happens here.
# Keep the laptop plugged in and awake. Each step can be re-run on its own.
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
& $py scripts\make_synthetic.py; Check 'synthetic data'
& $py scripts\prepare_phase1.py --limit 20; Check 'prepare smoke test'
& $py scripts\prepare_phase1.py; Check 'prepare (questions, token counts, splits)'
& $py scripts\pilot_packed.py --limit 25; Check 'pilot smoke test'
& $py scripts\pilot_packed.py; $pilot = $LASTEXITCODE
if ($pilot -eq 0) { Write-Host 'Packed pilot: all hard criteria passed.' }
else { Write-Host "Packed pilot finished with exit ${pilot}: at least one criterion failed; see results\phase1\pilot_packed.json." }
Write-Host "Done. Results are in $Root\results\phase1 and $Root\data\manifests"
