# Session Builder Standalone

A standalone version of the Binaural Session Builder, featuring a high-performance Rust audio backend for real-time binaural beat and noise generation.

## Prerequisites

- **Internet Connection**: Required for the first run to download dependencies.
- **Administrator/Sudo Access**: May be required to install system dependencies (Python/Rust).

## How to Run

### Windows

1.  Open PowerShell in this directory.
2.  Run:
    ```powershell
    .\setup.ps1
    ```
    This script handles everything:
    - Checks for & installs Python/Rust (via winget).
    - Checks for & installs Python dependencies.
    - Builds the high-performance audio engine.
    - Launches the application.

### Linux / macOS

1.  Open a terminal in this directory.
2.  Run:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    This script handles everything:
    - Installs Python/Rust via your package manager (`apt`, `brew`, `pacman`) if missing.
    - Sets up the environment.
    - Builds the audio engine.
    - Launches the application.

## Troubleshooting

- **Audio Backend Build Failed**: Ensure you have a C compiler installed (e.g., `build-essential` on Linux, Xcode on Mac, VS Build Tools on Windows). The setup script attempts to handle this, but specific system configurations may vary.
- **Python Not Found**: If the automatic install fails, please install Python 3.9+ manually.

## Development

This repository is configured for packaging with `maturin`.

### Structrue
- `session_builder/`: Main Python package source.
- `binauralbuilder_core/`: Dependencies.
- `setup.ps1` / `setup.sh`: Unified setup and launch scripts.
