@echo off
setlocal

REM 1. Run from the repository root and name this study.
cd /d "%~dp0"
set "study=%~1"
if not defined study set "study=depth_01"

REM 2. Download data once before starting the six training runs.
uv run runs/prepare.py
if errorlevel 1 exit /b 1
uv run runs/train.py +experiment=sweep_models "study=%study%"
if errorlevel 1 exit /b 1

REM 3. Summarize the completed runs in the same study.
uv run runs/report.py "study=%study%"
exit /b %errorlevel%
