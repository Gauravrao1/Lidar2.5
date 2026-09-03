# PS-26053: One-command full demo (PowerShell)
# Generates sample data, runs pipeline, and launches dashboard

$ErrorActionPreference = "Stop"

Write-Host "============================================"
Write-Host "  PS-26053 LiDAR Mapping - Full Demo"
Write-Host "============================================"

$projDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projDir ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "ERROR: Virtual environment not found. Run setup first." -ForegroundColor Red
    exit 1
}

Write-Host "[1/4] Generating sample data..." -ForegroundColor Cyan
& $python (Join-Path $projDir "scripts\generate_sample_data.py")
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Sample data generation failed" -ForegroundColor Red; exit 1 }

Write-Host "[2/4] Running pipeline on sample data..." -ForegroundColor Cyan
Set-Location $projDir
& $python -m src.pipeline.standalone_runner --data-dir (Join-Path $projDir "data\sample") --max-frames 10
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Pipeline failed" -ForegroundColor Red; exit 1 }

Write-Host "[3/4] Launching dashboard..." -ForegroundColor Cyan
Write-Host "Open http://localhost:8050 in your browser" -ForegroundColor Green
& $python -m src.viz.standalone_dashboard --data-dir (Join-Path $projDir "data\sample")
