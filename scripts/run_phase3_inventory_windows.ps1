# Phase 3 fresh-data inventory and audit. CPU only, reads datasets and manifests, no model outputs. Needs internet (Hugging Face).
#   Set-ExecutionPolicy -Scope Process Bypass -Force; & "$HOME\Projects\lookahead-halting\scripts\run_phase3_inventory_windows.ps1"
# Uses the real Ouro tokenizer so the Phase 1 question-set hashes are verified too. Refuses to overwrite an earlier inventory.
# Exit 3 from the inventory = it wrote a report with UNRESOLVED pools (not an error): send results\phase3\inventory.json and docs\PHASE3_INVENTORY.md.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
& $py -m pip install --quiet -e .
if ($LASTEXITCODE -ne 0) { throw "package install failed" }
& $py -m pytest -q tests\test_phase3_inventory.py
if ($LASTEXITCODE -ne 0) { throw "inventory tests failed (exit $LASTEXITCODE)" }
& $py scripts\phase3_inventory.py --tokenizer; $rc = $LASTEXITCODE
if ($rc -eq 0) { Write-Host 'Inventory complete, every pool resolved. Send results\phase3\inventory.json and docs\PHASE3_INVENTORY.md.' }
elseif ($rc -eq 3) { Write-Host 'Inventory written WITH UNRESOLVED POOLS. Send results\phase3\inventory.json and docs\PHASE3_INVENTORY.md.' }
else { throw "inventory refused or failed (exit $rc). The message is printed above." }
