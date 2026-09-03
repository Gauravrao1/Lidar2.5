# PS-26053: Full metrics evaluation (PowerShell)
# Runs the pipeline on all available data and generates the complete metrics report

$ErrorActionPreference = "Stop"

Write-Host "============================================"
Write-Host "  PS-26053 LiDAR Mapping - Metrics Eval"
Write-Host "============================================"

$projDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projDir ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "ERROR: Virtual environment not found. Run setup first." -ForegroundColor Red
    exit 1
}

# Ensure sample data exists
$sampleDir = Join-Path $projDir "data\sample\velodyne"
if (-not (Test-Path $sampleDir)) {
    Write-Host "[0/3] Generating sample data..." -ForegroundColor Cyan
    & $python (Join-Path $projDir "scripts\generate_sample_data.py")
}

Write-Host "[1/3] Running unit tests..." -ForegroundColor Cyan
Set-Location $projDir
& $python -m pytest tests/ -v --tb=short 2>&1 | Tee-Object -Variable testOutput
Write-Host $testOutput

Write-Host "[2/3] Running full pipeline evaluation..." -ForegroundColor Cyan
& $python -m src.pipeline.standalone_runner --data-dir (Join-Path $projDir "data\sample") --warmup 2
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Pipeline failed" -ForegroundColor Red; exit 1 }

Write-Host "[3/3] Results:" -ForegroundColor Green
$reportsDir = Join-Path $projDir "reports"
if (Test-Path (Join-Path $reportsDir "metrics_table_filled.md")) {
    Get-Content (Join-Path $reportsDir "metrics_table_filled.md")
} else {
    Write-Host "WARNING: Metrics table not generated" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Reports saved to: $reportsDir" -ForegroundColor Cyan
Write-Host "  - latency.csv"
Write-Host "  - per_frame_log.csv"
Write-Host "  - metrics_table_filled.md"
