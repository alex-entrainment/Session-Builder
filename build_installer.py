#!/usr/bin/env python3
"""
Cross-platform build script for Session Builder.

This script:
1. Compiles the Rust realtime_backend using maturin
2. Copies the compiled module to the correct location
3. Runs PyInstaller to create the distributable application

Usage:
    python build_installer.py [--skip-rust] [--debug] [--onefile]

Options:
    --skip-rust     Skip Rust backend compilation (use existing build)
    --debug         Build with console window for debugging
    --onefile       Create a single executable file (slower startup)
    --clean         Clean previous build artifacts before building
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def get_platform_info():
    """Get platform-specific information."""
    system = platform.system()
    is_windows = system == 'Windows'
    is_linux = system == 'Linux'
    is_macos = system == 'Darwin'

    # Determine the extension module suffix
    if is_windows:
        ext_suffix = '.pyd'
        # Get Python version for suffix
        py_version = f'cp{sys.version_info.major}{sys.version_info.minor}'
        arch = 'win_amd64' if platform.machine().endswith('64') else 'win32'
        module_pattern = f'realtime_backend.{py_version}-{arch}.pyd'
    else:
        import sysconfig
        ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
        module_pattern = f'realtime_backend*{ext_suffix}'

    return {
        'system': system,
        'is_windows': is_windows,
        'is_linux': is_linux,
        'is_macos': is_macos,
        'ext_suffix': ext_suffix,
        'module_pattern': module_pattern,
    }


def run_command(cmd, cwd=None, check=True):
    """Run a command and print output."""
    print(f"\n>>> {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(
        cmd,
        cwd=cwd,
        shell=isinstance(cmd, str),
        capture_output=False,
    )
    if check and result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def find_rust_module(base_dir, pattern):
    """Find the compiled Rust module."""
    import glob as glob_module

    # Search locations in order of preference
    search_paths = [
        base_dir / 'src',
        base_dir,
        base_dir / 'target' / 'release',
        base_dir / 'target' / 'wheels',
    ]

    for search_path in search_paths:
        if search_path.exists():
            matches = list(search_path.glob(pattern))
            if matches:
                return matches[0]

    return None


def build_rust_backend(base_dir, platform_info):
    """Build the Rust realtime_backend using maturin."""
    print("\n=== Building Rust Backend ===")

    rust_dir = base_dir / 'src' / 'realtime_backend'

    if not rust_dir.exists():
        print(f"Error: Rust backend directory not found: {rust_dir}")
        sys.exit(1)

    # Check if Rust is installed
    result = subprocess.run(['rustc', '--version'], capture_output=True)
    if result.returncode != 0:
        print("Error: Rust is not installed. Please install Rust from https://rustup.rs/")
        sys.exit(1)

    # Check if maturin is installed
    result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'maturin'], capture_output=True)
    if result.returncode != 0:
        print("Installing maturin...")
        run_command([sys.executable, '-m', 'pip', 'install', 'maturin'])

    # Build with maturin from the project root
    # Using 'develop' to build and install in-place
    print("Building Rust extension with maturin...")
    run_command(
        [sys.executable, '-m', 'maturin', 'build', '--release', '-o', 'dist'],
        cwd=base_dir
    )

    # Also try maturin develop to get the .pyd/.so in the right place
    print("Installing Rust extension with maturin develop...")
    run_command(
        [sys.executable, '-m', 'maturin', 'develop', '--release'],
        cwd=base_dir,
        check=False  # Don't fail if this doesn't work
    )

    # Find the built module
    module_path = find_rust_module(base_dir, platform_info['module_pattern'])

    if module_path:
        print(f"Rust backend built successfully: {module_path}")

        # Copy to src/ directory if not already there
        dest_path = base_dir / 'src' / module_path.name
        if module_path != dest_path:
            print(f"Copying {module_path} -> {dest_path}")
            shutil.copy2(module_path, dest_path)

        return dest_path
    else:
        # Check in the wheel
        wheel_dir = base_dir / 'dist'
        if wheel_dir.exists():
            wheels = list(wheel_dir.glob('*.whl'))
            if wheels:
                print(f"Note: Wheel built at {wheels[0]}")
                print("You may need to extract the module from the wheel manually.")

        print("Warning: Could not find compiled Rust module after build.")
        return None


def clean_build(base_dir):
    """Clean previous build artifacts."""
    print("\n=== Cleaning Build Artifacts ===")

    dirs_to_clean = [
        base_dir / 'build',
        base_dir / 'dist',
        base_dir / '__pycache__',
    ]

    for dir_path in dirs_to_clean:
        if dir_path.exists():
            print(f"Removing {dir_path}")
            shutil.rmtree(dir_path)

    # Clean .pyc files
    for pyc in base_dir.rglob('*.pyc'):
        pyc.unlink()

    # Clean __pycache__ directories
    for pycache in base_dir.rglob('__pycache__'):
        if pycache.is_dir():
            shutil.rmtree(pycache)


def build_installer(base_dir, platform_info, debug=False, onefile=False):
    """Build the installer using PyInstaller."""
    print("\n=== Building Installer with PyInstaller ===")

    # Check if PyInstaller is installed
    result = subprocess.run([sys.executable, '-m', 'pip', 'show', 'pyinstaller'], capture_output=True)
    if result.returncode != 0:
        print("Installing PyInstaller...")
        run_command([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

    spec_file = base_dir / 'session_builder.spec'

    if not spec_file.exists():
        print(f"Error: Spec file not found: {spec_file}")
        sys.exit(1)

    # Build PyInstaller command
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--clean',
        '--noconfirm',
        str(spec_file),
    ]

    if debug:
        # Modify spec to enable console
        print("Note: Debug mode enabled - console window will be visible")

    run_command(cmd, cwd=base_dir)

    # Check if build succeeded
    dist_dir = base_dir / 'dist' / 'SessionBuilder'
    if platform_info['is_windows']:
        exe_path = dist_dir / 'SessionBuilder.exe'
    else:
        exe_path = dist_dir / 'SessionBuilder'

    if exe_path.exists():
        print(f"\n=== Build Successful! ===")
        print(f"Executable: {exe_path}")
        print(f"Distribution folder: {dist_dir}")
        return True
    else:
        print("\n=== Build may have failed ===")
        print("Check the output above for errors.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Build Session Builder installer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--skip-rust',
        action='store_true',
        help='Skip Rust backend compilation'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Build with console window for debugging'
    )
    parser.add_argument(
        '--onefile',
        action='store_true',
        help='Create single executable (not recommended)'
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean build artifacts before building'
    )

    args = parser.parse_args()

    # Get base directory
    base_dir = Path(__file__).parent.resolve()
    print(f"Base directory: {base_dir}")

    # Get platform info
    platform_info = get_platform_info()
    print(f"Platform: {platform_info['system']}")
    print(f"Python: {sys.version}")

    # Clean if requested
    if args.clean:
        clean_build(base_dir)

    # Build Rust backend
    if not args.skip_rust:
        module_path = build_rust_backend(base_dir, platform_info)
        if not module_path:
            print("\nWarning: Rust backend not found. The application will use Python fallback.")
            print("Continue building anyway? [y/N] ", end='')
            response = input().strip().lower()
            if response != 'y':
                sys.exit(1)
    else:
        print("\n=== Skipping Rust Backend Build ===")
        module_path = find_rust_module(base_dir, platform_info['module_pattern'])
        if module_path:
            print(f"Using existing Rust backend: {module_path}")
        else:
            print("Warning: No existing Rust backend found.")

    # Build installer
    success = build_installer(base_dir, platform_info, debug=args.debug, onefile=args.onefile)

    if success:
        print("\n=== Build Complete ===")
        print("To run the application:")
        if platform_info['is_windows']:
            print("  dist\\SessionBuilder\\SessionBuilder.exe")
        else:
            print("  ./dist/SessionBuilder/SessionBuilder")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
