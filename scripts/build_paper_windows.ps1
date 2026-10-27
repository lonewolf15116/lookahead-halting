# Build both manuscript PDFs (separate targets) and run the font / consistency checks.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_paper_windows.ps1
# Needs a TeX distribution with latexmk and lmodern (MiKTeX or TeX Live) and poppler (pdffonts, pdftotext) on PATH.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root 'paper')
foreach ($t in @('main_preprint', 'main_anon')) {
    latexmk -pdf -interaction=nonstopmode -halt-on-error $t.tex
    if ($LASTEXITCODE -ne 0) { throw "latexmk $t failed" }
}
Set-Location $Root
foreach ($t in @('main_preprint', 'main_anon')) {
    "--- fonts in $t.pdf"
    pdffonts "paper\$t.pdf"
    $bad = (pdffonts "paper\$t.pdf") -match 'Nimbus|Inter|Times|Helvet'
    if ($bad) { Write-Warning "temporary fonts still embedded in $t.pdf" }
}
Select-String -Path paper\main_preprint.log, paper\main_anon.log -Pattern 'undefined|Citation .* undefined|Label.*multiply' | Select-Object -First 20
$py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
& $py scripts\step4_manuscript_consistency.py --out results\audit\post_rebuild\step4_manuscript_consistency.json
