@echo off
setlocal
cd /d "%~dp0"

set "PY=%CD%\.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo Virtual environment python not found: "%PY%"
  echo Please re-run setup-run.ps1
  pause
  exit /b 1
)

"%PY%" -m src.audio.session_builder_launcher --binaural-preset-dir src/presets --noise-preset-dir src/presets 

if errorlevel 1 (
  echo.
  echo The application exited with an error.
  pause
)

