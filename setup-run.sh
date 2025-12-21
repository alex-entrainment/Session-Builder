#!/bin/bash
set -e

echo "=== Session Builder Standalone Launcher ==="

# --- 1. Helper Functions for Installation ---

OS="$(uname -s)"
echo "Detected OS: $OS"

install_python_linux() {
    if ! command -v python3 &> /dev/null; then
        echo "Python 3 not found. Installing..."
        if [ -x "$(command -v apt-get)" ]; then
            sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv build-essential
        elif [ -x "$(command -v dnf)" ]; then
            sudo dnf install -y python3 python3-pip
        elif [ -x "$(command -v pacman)" ]; then
            sudo pacman -S --noconfirm python python-pip
        else
            echo "Could not detect package manager. Please install Python 3 manually."
            exit 1
        fi
    fi
}

install_rust_linux() {
    if ! command -v cargo &> /dev/null; then
        echo "Rust not found. Installing via rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        # Source env immediately for this script
        source "$HOME/.cargo/env"
    fi
}

install_mac_deps() {
    if ! command -v brew &> /dev/null; then
        echo "Homebrew must be installed manually first: https://brew.sh/"
        exit 1
    fi
    if ! command -v python3 &> /dev/null; then
         brew install python
    fi
    if ! command -v cargo &> /dev/null; then
         brew install rust
    fi
}

# --- 2. System Check & Install ---

if [ "$OS" = "Darwin" ]; then
    install_mac_deps
elif [ "$OS" = "Linux" ]; then
    install_python_linux
    install_rust_linux
else
    echo "Warning: Unknown OS $OS. Assuming dependencies are met."
fi

# Double check cargo existence after install attempts
if ! command -v cargo &> /dev/null; then
    # Try one more time to source cargo env if it was just installed
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    if ! command -v cargo &> /dev/null; then
        echo "Error: Cargo (Rust) could not be found. Please restart your terminal or install Rust manually."
        exit 1
    fi
fi

# --- 3. Environment Setup ---

VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Updating dependencies..."
pip install --upgrade pip
pip install maturin
pip install -r requirements.txt

if [ -f "setup_audio.py" ]; then
    python3 setup_audio.py
fi

# --- 4. Build Backend ---

echo "Building Audio Backend..."
maturin develop --release

# --- 5. Run Application ---

echo "Starting Session Builder..."
export PYTHONPATH="$SCRIPT_DIR"
python3 -m src.audio.session_builder_launcher --binaural-preset-dir src/presets --noise-preset-dir src/presets "$@"
