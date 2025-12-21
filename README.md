# Session Builder Standalone

A standalone version of the Binaural Session Builder, featuring a high-performance Rust audio backend for real-time binaural beat and noise generation.

<img width="1380" height="890" alt="image" src="https://github.com/user-attachments/assets/7fbaa3f7-add8-4582-b0d2-69c93ee9e738" />


## 🚀 Quick Start (Recommended)

### Windows
Double-click **`Session Builder Launcher.bat`**. 
This will automatically:
1.  Check for and install Python & Rust (via Winget) if missing.
2.  Set up the environment and dependencies.
3.  Build the audio engine.
4.  Launch the application.

*Note: You may be asked to approve administrative privileges for installing dependencies.*

### Linux / macOS
Run the launcher script from your terminal (or double-click if your file manager supports it):
```bash
./"Session Builder Launcher.sh"
```

## Features 

### Stream or Generate your own custom binaural sessions 
- Select from a range of Focus Level binaural presets
- Select from a range of background "Noise" profiles
- Load other audio files to play in the background
- Save sessions for loading later, or export the session to an audio file!
- 

---

## ⚡ One-Line Installation

**Windows (PowerShell)**
```powershell
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Write-Host "Installing Git..."; winget install --id Git.Git -e --source winget }; git clone https://github.com/alex-entrainment/Session-Builder.git; cd Session-Builder; .\scripts\setup-run.ps1
```

**Linux / macOS**
```bash
(command -v git >/dev/null || (echo "Installing Git..." && (command -v brew >/dev/null && brew install git) || (command -v apt >/dev/null && sudo apt update && sudo apt install git -y) || (command -v pacman >/dev/null && sudo pacman -S git --noconfirm) || (command -v dnf >/dev/null && sudo dnf install git -y))) && git clone https://github.com/alex-entrainment/Session-Builder.git && cd Session-Builder && chmod +x scripts/setup-run.sh && ./scripts/setup-run.sh
```

---

## 💻 Manual Terminal Usage

If you prefer using the command line or need to debug, you can run the setup scripts directly from the `scripts/` folder.

**Windows (PowerShell):**
```powershell
.\scripts\setup-run.ps1
```

**Linux / macOS (Bash):**
```bash
chmod +x scripts/setup-run.sh
./scripts/setup-run.sh
```

---


## 🔧 Prerequisites

*   **Internet Connection**: Required for the first run to download Python packages and Rust crates.
*   **Git**: Required to clone the repository (and for the Web Installer).
*   **System Dependencies**:
    *   **Windows**: The launcher attempts to install Python 3 and Rust via `winget`.
    *   **Linux**: Requires `python3`, `python3-pip`, `python3-venv`, and `build-essential`. The script attempts to install these via `apt`/`dnf`/`pacman`.
    *   **macOS**: Requires Homebrew installed. The script uses `brew` to install Python and Rust.

## 🛠️ Development Structure

*   `src/`: Main application source code.
*   `binauralbuilder_core/`: Core audio logic and synthesis functions.
*   `scripts/`: Setup and maintenance scripts.
*   `config/` (Created on first run): User preferences and presets.
