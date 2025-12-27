#!/usr/bin/env bash
#
# Build Session Builder installer for Linux/macOS.
#
# This script builds the Session Builder application for distribution.
# It compiles the Rust backend using maturin and packages everything with PyInstaller.
#
# Usage:
#   ./build_installer.sh [OPTIONS]
#
# Options:
#   --skip-rust     Skip Rust backend compilation (use existing build)
#   --debug         Build with console window for debugging
#   --clean         Clean previous build artifacts before building
#   --help          Show this help message

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_RUST=false
DEBUG=false
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-rust)
            SKIP_RUST=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --help)
            head -30 "$0" | tail -n +2 | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Session Builder Linux/macOS Build Script ==="
echo "Base Directory: $SCRIPT_DIR"

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
    EXT_SUFFIX=".so"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
    EXT_SUFFIX=".so"
else
    echo "Warning: Unknown platform $OSTYPE, assuming Linux"
    PLATFORM="linux"
    EXT_SUFFIX=".so"
fi
echo "Platform: $PLATFORM"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found. Please install Python 3.9+"
    exit 1
fi
echo "Python: $($PYTHON --version)"

# Check Rust (if building Rust backend)
if [[ "$SKIP_RUST" == "false" ]]; then
    if ! command -v rustc &> /dev/null; then
        echo "Error: Rust not found. Please install from https://rustup.rs/"
        exit 1
    fi
    echo "Rust: $(rustc --version)"
fi

# Activate virtual environment if it exists
if [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/.venv/bin/activate"
elif [[ -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
    echo "Activating virtual environment..."
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Install build dependencies
echo ""
echo "=== Installing Build Dependencies ==="
$PYTHON -m pip install maturin pyinstaller --quiet

# Clean if requested
if [[ "$CLEAN" == "true" ]]; then
    echo ""
    echo "=== Cleaning Build Artifacts ==="
    for dir in build dist __pycache__; do
        path="$SCRIPT_DIR/$dir"
        if [[ -d "$path" ]]; then
            echo "Removing $path"
            rm -rf "$path"
        fi
    done
    find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$SCRIPT_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true
fi

# Build Rust backend
if [[ "$SKIP_RUST" == "false" ]]; then
    echo ""
    echo "=== Building Rust Backend ==="

    cd "$SCRIPT_DIR"

    # Build the wheel
    echo "Running maturin build..."
    $PYTHON -m maturin build --release -o dist

    # Install in development mode to get the .so in place
    echo "Running maturin develop..."
    $PYTHON -m maturin develop --release || true

    # Find the .so file
    SO_FILE=$(find "$SCRIPT_DIR" -name "realtime_backend*$EXT_SUFFIX" -type f 2>/dev/null | head -1)
    if [[ -n "$SO_FILE" ]]; then
        echo "Found Rust backend: $SO_FILE"

        # Ensure it's in the src directory
        DEST_PATH="$SCRIPT_DIR/src/$(basename "$SO_FILE")"
        if [[ "$SO_FILE" != "$DEST_PATH" ]]; then
            cp "$SO_FILE" "$DEST_PATH"
            echo "Copied to: $DEST_PATH"
        fi
    else
        echo "Warning: Rust backend .so file not found"
    fi
else
    echo ""
    echo "=== Skipping Rust Backend Build ==="
    SO_FILE=$(find "$SCRIPT_DIR/src" -name "realtime_backend*$EXT_SUFFIX" -type f 2>/dev/null | head -1)
    if [[ -n "$SO_FILE" ]]; then
        echo "Using existing Rust backend: $SO_FILE"
    else
        echo "Warning: No existing Rust backend found"
    fi
fi

# Build with PyInstaller
echo ""
echo "=== Building Installer with PyInstaller ==="

SPEC_FILE="$SCRIPT_DIR/session_builder.spec"
if [[ ! -f "$SPEC_FILE" ]]; then
    echo "Error: Spec file not found: $SPEC_FILE"
    exit 1
fi

cd "$SCRIPT_DIR"
$PYTHON -m PyInstaller --clean --noconfirm "$SPEC_FILE"

# Check result
EXE_PATH="$SCRIPT_DIR/dist/SessionBuilder/SessionBuilder"
if [[ -f "$EXE_PATH" ]]; then
    echo ""
    echo "=== Build Successful! ==="
    echo "Executable: $EXE_PATH"
    echo "Distribution folder: $SCRIPT_DIR/dist/SessionBuilder"
    echo ""
    echo "To run the application:"
    echo "  ./dist/SessionBuilder/SessionBuilder"

    # Make executable
    chmod +x "$EXE_PATH"
else
    echo ""
    echo "=== Build may have failed ==="
    echo "Check the output above for errors."
    exit 1
fi
