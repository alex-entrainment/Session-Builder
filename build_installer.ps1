<#
.SYNOPSIS
    Build Session Builder installer for Windows.

.DESCRIPTION
    This script builds the Session Builder application for Windows distribution.
    It compiles the Rust backend using maturin and packages everything with PyInstaller.

.PARAMETER SkipRust
    Skip Rust backend compilation (use existing build).

.PARAMETER Debug
    Build with console window for debugging.

.PARAMETER Clean
    Clean previous build artifacts before building.

.EXAMPLE
    .\build_installer.ps1
    Build the installer with default settings.

.EXAMPLE
    .\build_installer.ps1 -Debug
    Build with console window for debugging.

.EXAMPLE
    .\build_installer.ps1 -SkipRust -Clean
    Clean and build without recompiling Rust backend.
#>

param(
    [switch]$SkipRust,
    [switch]$Debug,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== Session Builder Windows Build Script ===" -ForegroundColor Cyan
Write-Host "Base Directory: $ScriptDir"

# Check Python
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python: $pythonVersion"
} catch {
    Write-Host "Error: Python not found. Please install Python 3.9+" -ForegroundColor Red
    exit 1
}

# Check Rust (if building Rust backend)
if (-not $SkipRust) {
    try {
        $rustVersion = rustc --version 2>&1
        Write-Host "Rust: $rustVersion"
    } catch {
        Write-Host "Error: Rust not found. Please install from https://rustup.rs/" -ForegroundColor Red
        exit 1
    }
}

# Activate virtual environment if it exists
$venvPath = Join-Path $ScriptDir ".venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..."
    . $venvPath
}

# Install build dependencies
Write-Host "`n=== Installing Build Dependencies ===" -ForegroundColor Yellow
pip install maturin pyinstaller --quiet

# Clean if requested
if ($Clean) {
    Write-Host "`n=== Cleaning Build Artifacts ===" -ForegroundColor Yellow
    $dirsToClean = @("build", "dist", "__pycache__")
    foreach ($dir in $dirsToClean) {
        $path = Join-Path $ScriptDir $dir
        if (Test-Path $path) {
            Write-Host "Removing $path"
            Remove-Item -Path $path -Recurse -Force
        }
    }
}

# Build Rust backend
if (-not $SkipRust) {
    Write-Host "`n=== Building Rust Backend ===" -ForegroundColor Yellow

    Push-Location $ScriptDir
    try {
        # Build the wheel
        Write-Host "Running maturin build..."
        maturin build --release -o dist

        # Install in development mode to get the .pyd in place
        Write-Host "Running maturin develop..."
        maturin develop --release

        # Find and copy the .pyd file
        $pydFiles = Get-ChildItem -Path $ScriptDir -Filter "realtime_backend*.pyd" -Recurse
        if ($pydFiles.Count -gt 0) {
            $pydFile = $pydFiles[0]
            Write-Host "Found Rust backend: $($pydFile.FullName)" -ForegroundColor Green

            # Ensure it's in the src directory
            $destPath = Join-Path $ScriptDir "src\$($pydFile.Name)"
            if ($pydFile.FullName -ne $destPath) {
                Copy-Item -Path $pydFile.FullName -Destination $destPath -Force
                Write-Host "Copied to: $destPath"
            }
        } else {
            Write-Host "Warning: Rust backend .pyd file not found" -ForegroundColor Yellow
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "`n=== Skipping Rust Backend Build ===" -ForegroundColor Yellow
    $pydFiles = Get-ChildItem -Path (Join-Path $ScriptDir "src") -Filter "realtime_backend*.pyd" -ErrorAction SilentlyContinue
    if ($pydFiles.Count -gt 0) {
        Write-Host "Using existing Rust backend: $($pydFiles[0].FullName)"
    } else {
        Write-Host "Warning: No existing Rust backend found" -ForegroundColor Yellow
    }
}

# Build with PyInstaller
Write-Host "`n=== Building Installer with PyInstaller ===" -ForegroundColor Yellow

$specFile = Join-Path $ScriptDir "session_builder.spec"
if (-not (Test-Path $specFile)) {
    Write-Host "Error: Spec file not found: $specFile" -ForegroundColor Red
    exit 1
}

$pyinstallerArgs = @("--clean", "--noconfirm", $specFile)

Push-Location $ScriptDir
try {
    python -m PyInstaller @pyinstallerArgs
} finally {
    Pop-Location
}

# Check result
$exePath = Join-Path $ScriptDir "dist\SessionBuilder\SessionBuilder.exe"
if (Test-Path $exePath) {
    Write-Host "`n=== Build Successful! ===" -ForegroundColor Green
    Write-Host "Executable: $exePath"
    Write-Host "Distribution folder: $(Join-Path $ScriptDir 'dist\SessionBuilder')"
    Write-Host "`nTo run the application:"
    Write-Host "  .\dist\SessionBuilder\SessionBuilder.exe"
} else {
    Write-Host "`n=== Build may have failed ===" -ForegroundColor Red
    Write-Host "Check the output above for errors."
    exit 1
}
