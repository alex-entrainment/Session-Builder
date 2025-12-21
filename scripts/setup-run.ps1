# setup.ps1 - All-in-One Setup & Launcher for Windows
# 1. Checks/Installs Python & Rust via winget.
# 2. Sets up Python Virtual Environment.
# 3. Builds the Rust Backend.
# 4. Launches the Application.

$ErrorActionPreference = "Stop"

Write-Host "=== Session Builder Standalone Launcher ==="
Write-Host "Checking system requirements..."

# Move to project root
Set-Location "$PSScriptRoot/.."


# --- 1. System Requirements Check (Python & Rust) ---

# Check Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Python not found. Attempting to install via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install -e --id Python.Python.3.13
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install Python. Please install Python 3 manually from python.org."
        }
        Write-Warning "Python installed. Please restart this script for the changes to take effect."
        exit 0
    } else {
        Write-Error "Winget not found. Please install Python 3 manually from python.org."
    }
} else {
    Write-Host "Python found."
}

# Check Rust (Cargo)
if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host "Rust (Cargo) not found. This is required for building the audio engine."
    Write-Host "Attempting to install via winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        winget install -e --id Rustlang.Rustup
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install Rust. Please install Rust manually from rustup.rs."
        }
        Write-Warning "Rust installed. Please restart this script for the changes to take effect."
        exit 0
    } else {
        Write-Error "Winget not found. Please install Rust manually from rustup.rs."
    }
} else {
    Write-Host "Rust found."
}

# --- 2. Environment Setup ---

$VenvDir = ".venv"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment..."
    python -m venv $VenvDir
}

$PythonExe = ".\$VenvDir\Scripts\python.exe"

Write-Host "Updating dependencies..."
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install maturin
& $PythonExe -m pip install -r requirements.txt

# Run audio initialization if present
if (Test-Path "scripts/setup_audio.py") {
    & $PythonExe scripts/setup_audio.py
}

# --- 3. Build Backend ---

Write-Host "Building Audio Backend (this may take a moment)..."
# Navigate to backend source
    # Build from root using pyproject.toml configuration
    $MaturinExe = ".\$VenvDir\Scripts\maturin.exe"
    if (-not (Test-Path $MaturinExe)) {
         # Fallback if installed globally or path issue, but preference is venv
         $MaturinExe = "maturin" 
    }
    
    & $MaturinExe develop --release
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to build audio backend."
    }

# --- 4. Run Application ---

Write-Host "Starting Session Builder..."
$Env:PYTHONPATH = "$PWD"
& $PythonExe -m src.audio.session_builder_launcher --binaural-preset-dir src/presets --noise-preset-dir src/presets @args
