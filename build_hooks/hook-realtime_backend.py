"""
PyInstaller hook for the realtime_backend Rust module.

This hook ensures the compiled Rust extension module is properly included
in the PyInstaller bundle.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs

# The realtime_backend is a compiled Rust extension
# It should be collected as a binary
hiddenimports = []

# Try to collect any dynamic libraries the module might depend on
try:
    binaries = collect_dynamic_libs('realtime_backend')
except Exception:
    binaries = []
