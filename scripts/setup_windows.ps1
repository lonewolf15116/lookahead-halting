# Phase 0 setup and first look at Ouro on Windows. Run from PowerShell:
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\setup_windows.ps1"
# Safe to run again: it skips what exists. Results go to results\gate0\ inside the repo folder.
$ErrorActionPreference = 'Stop'
$Root = Join-Path $HOME 'Projects\lookahead-halting'
Set-Location $Root

function Check($what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). The error is printed above." }
}

# 1. Python 3.10 or newer
$v = & python -c "import sys; print('%d.%d' % sys.version_info[:2])"
if ([version]$v -lt [version]'3.10') { throw "Python $v found; need 3.10 or newer" }

# 2. Environment
if (-not (Test-Path '.venv')) { python -m venv .venv }
$py = Join-Path $Root '.venv\Scripts\python.exe'
& $py -m pip install --quiet --upgrade pip; Check 'pip upgrade'
# The RTX 5050 is a Blackwell GPU, so it needs a PyTorch build made with CUDA 12.8 or newer.
& $py -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128; Check 'PyTorch install'
& $py -m pip install --quiet -r requirements.txt; Check 'requirements install'
& $py -m pip install --quiet -e .; Check 'package install'

# 3. Tests
& $py -m pytest -q; Check 'tests'

# 4. Gate 0 probes. The download comes first because it needs no GPU.
& $py scripts\inspect_ouro.py; Check 'Ouro download'
& $py scripts\check_environment.py; Check 'environment check'
& $py scripts\smoke_stock.py; Check 'stock Ouro smoke test'

Write-Host "Phase 0 setup done. Results are in $Root\results\gate0"
