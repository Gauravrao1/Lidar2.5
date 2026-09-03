@echo off
REM PS-26053: One-command full demo
REM Generates sample data, runs pipeline, and launches dashboard

echo ============================================
echo  PS-26053 LiDAR Mapping - Full Demo
echo ============================================

set PROJ_DIR=%~dp0..
set PYTHON=%PROJ_DIR%\.venv\Scripts\python.exe

echo [1/4] Generating sample data...
%PYTHON% "%PROJ_DIR%\scripts\generate_sample_data.py"
if %ERRORLEVEL% neq 0 (echo ERROR: Sample data generation failed & exit /b 1)

echo [2/4] Running pipeline on sample data...
%PYTHON% -m src.pipeline.standalone_runner --data-dir "%PROJ_DIR%\data\sample" --max-frames 10
if %ERRORLEVEL% neq 0 (echo ERROR: Pipeline failed & exit /b 1)

echo [3/4] Launching dashboard...
echo Open http://localhost:8050 in your browser
%PYTHON% -m src.viz.standalone_dashboard --data-dir "%PROJ_DIR%\data\sample"
