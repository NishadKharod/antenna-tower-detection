@echo off
REM Quick-start server on Windows
REM Usage: run_server.bat [path\to\best.pt]

setlocal
set WEIGHTS=%1
if "%WEIGHTS%"=="" set WEIGHTS=runs\tower_detection\yolo11_tower_detector\weights\best.pt
if "%HOST%"==""  set HOST=127.0.0.1
if "%PORT%"==""  set PORT=5000

cd /d "%~dp0"
set ATD_WEIGHTS=%WEIGHTS%
python src\app.py --weights "%WEIGHTS%" --host %HOST% --port %PORT%
endlocal
