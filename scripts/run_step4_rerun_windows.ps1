# Step 4 same-environment reproducibility rerun (development data only; the sealed TEST data is never read).
# Writes ONLY to results\audit\rerun_windows\ (created exclusively) - no committed artifact is touched.
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_step4_rerun_windows.ps1
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$py = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$env:PYTHONPATH = Join-Path $Root 'src'
$out = Join-Path $Root 'results\audit\rerun_windows'
if (Test-Path $out) { throw "$out already exists; move it away first (outputs are never overwritten)." }
New-Item -ItemType Directory -Path $out | Out-Null
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE). Stop here and send results\audit\rerun_windows\console.txt." } }
Start-Transcript -Path (Join-Path $out 'console.txt') | Out-Null
git rev-parse HEAD
git status --short
& $py -c "import sys,numpy,scipy,sklearn,pandas,pyarrow;print(sys.version);print('numpy',numpy.__version__,'scipy',scipy.__version__,'sklearn',sklearn.__version__,'pandas',pandas.__version__,'pyarrow',pyarrow.__version__)"
foreach ($w in @('c1freeze', 'e1', 'x1')) {
    $t0 = Get-Date
    & $py scripts\step4_rerun_dev.py $w --out (Join-Path $out $w); Check "rerun $w"
    Write-Host ("{0} took {1:N1} min" -f $w, ((Get-Date) - $t0).TotalMinutes)
}
# post-rerun audits go to the rerun folder, never over the committed reports
& $py scripts\step4_numeric_audit.py --out (Join-Path $out 'step4_numeric_audit.json') | Select-Object -First 1
& $py scripts\step4_provenance_audit.py --out (Join-Path $out 'step4_provenance_audit.json') | Select-Object -First 1
Get-ChildItem -Recurse $out -File | ForEach-Object { '{0}  {1}' -f (Get-FileHash $_.FullName -Algorithm SHA256).Hash.ToLower(), $_.FullName.Substring($Root.Length + 1) } | Set-Content (Join-Path $out 'SHA256SUMS.txt')
Stop-Transcript | Out-Null
Write-Host "Done. Send results\audit\rerun_windows\ (console.txt, *_compare.json, SHA256SUMS.txt)."
