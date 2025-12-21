#!/bin/bash
set -e

# Setup script for building the binauralbuilder-core package and its Rust extension.

# Check for Rust installation
if ! command -v rustc &> /dev/null; then
    echo "Rust is not installed. Please install Rust via rustup:"
    echo "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Check for Python virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No Python virtual environment detected."
    # echo "It is recommended to run this script within a virtualenv."
fi

echo "Installing build dependencies..."
pip install --upgrade pip maturin

echo "Building Rust backend..."
# Use maturin to build the extension module
# develop --release installs it into the current venv in editable mode (mostly)
# or just builds the wheel and installs it.
maturin develop --release

echo "Build complete. You can now run the application."
