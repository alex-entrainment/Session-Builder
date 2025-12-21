@echo off
cd /d "%~dp0"
echo Starting Session Builder...
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup-run.ps1"
if %errorlevel% neq 0 (
    echo.
    echo Application exited with error code %errorlevel%.
    pause
)
