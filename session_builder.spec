# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Session Builder.

This configuration bundles the Session Builder application with the Rust
realtime_backend for Windows and Linux platforms.

Build instructions:
    # First, ensure the Rust backend is compiled:
    # Windows: maturin build --release
    # Linux: maturin build --release

    # Then run PyInstaller:
    # pyinstaller session_builder.spec --clean
"""

import os
import sys
import glob
import platform
from pathlib import Path

# Determine platform-specific settings
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

# Base paths
SPEC_DIR = os.path.dirname(os.path.abspath(SPEC))
SRC_DIR = os.path.join(SPEC_DIR, 'src')
CORE_DIR = os.path.join(SPEC_DIR, 'binauralbuilder_core')
RUST_BACKEND_DIR = os.path.join(SRC_DIR, 'realtime_backend')

# Find the compiled Rust backend module
def find_rust_backend():
    """Find the compiled realtime_backend module (.pyd or .so)."""
    patterns = []

    if IS_WINDOWS:
        # Windows: look for .pyd files
        patterns = [
            os.path.join(SRC_DIR, 'realtime_backend*.pyd'),
            os.path.join(SPEC_DIR, 'realtime_backend*.pyd'),
            os.path.join(SPEC_DIR, 'target', 'release', 'realtime_backend*.pyd'),
        ]
    else:
        # Linux/macOS: look for .so files
        patterns = [
            os.path.join(SRC_DIR, 'realtime_backend*.so'),
            os.path.join(SPEC_DIR, 'realtime_backend*.so'),
            os.path.join(SPEC_DIR, 'target', 'release', 'librealtime_backend*.so'),
        ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None

RUST_BACKEND_PATH = find_rust_backend()
if RUST_BACKEND_PATH:
    print(f"Found Rust backend: {RUST_BACKEND_PATH}")
else:
    print("WARNING: Rust backend module not found! The application will fall back to Python-only mode.")

# Main entry point
ENTRY_POINT = os.path.join(SRC_DIR, 'audio', 'session_builder_launcher.py')

# Data files to include
datas = [
    # Rust backend config
    (os.path.join(RUST_BACKEND_DIR, 'config.toml'), 'realtime_backend'),
]

# Binary files to include
binaries = []

# Add the Rust backend module if found
if RUST_BACKEND_PATH:
    # The module needs to be at the root level for import to work
    backend_filename = os.path.basename(RUST_BACKEND_PATH)
    binaries.append((RUST_BACKEND_PATH, '.'))

# Hidden imports - packages that PyInstaller may not detect automatically
hiddenimports = [
    # Core packages
    'src',
    'src.audio',
    'src.audio.session_builder_launcher',
    'src.audio.session_model',
    'src.audio.session_engine',
    'src.audio.session_stream',
    'src.audio.rust_stream_player',
    'src.ui',
    'src.ui.themes',
    'src.ui.session_builder_window',
    'src.ui.preferences_dialog',
    'src.ui.defaults_dialog',
    'src.ui.html_delegate',
    'src.models',
    'src.utils',
    'src.presets',

    # Binauralbuilder core
    'binauralbuilder_core',
    'binauralbuilder_core.assembly',
    'binauralbuilder_core.session',
    'binauralbuilder_core.synthesis',
    'binauralbuilder_core.synth_functions',
    'binauralbuilder_core.synth_functions.binaural_beat',
    'binauralbuilder_core.synth_functions.isochronic_tone',
    'binauralbuilder_core.synth_functions.qam_beat',
    'binauralbuilder_core.synth_functions.qam_beat_transition',
    'binauralbuilder_core.synth_functions.monaural_beat_stereo_amps',
    'binauralbuilder_core.synth_functions.dual_pulse_binaural',
    'binauralbuilder_core.synth_functions.spatial_ambi2d',
    'binauralbuilder_core.synth_functions.spatial_angle_modulation',
    'binauralbuilder_core.synth_functions.spatial_itd_stable',
    'binauralbuilder_core.synth_functions.stereo_am_independent',
    'binauralbuilder_core.synth_functions.wave_shape_stereo_am',
    'binauralbuilder_core.synth_functions.rhythmic_waveshaping',
    'binauralbuilder_core.synth_functions.fx_flanger',
    'binauralbuilder_core.synth_functions.noise_flanger',
    'binauralbuilder_core.synth_functions.subliminals',
    'binauralbuilder_core.synth_functions.common',
    'binauralbuilder_core.synth_functions.audio_engine',
    'binauralbuilder_core.synth_functions.sound_creator',
    'binauralbuilder_core.utils',
    'binauralbuilder_core.utils.noise_file',
    'binauralbuilder_core.utils.voice_file',
    'binauralbuilder_core.utils.colored_noise',
    'binauralbuilder_core.utils.binaural_processing',
    'binauralbuilder_core.utils.amp_utils',
    'binauralbuilder_core.utils.preferences',
    'binauralbuilder_core.utils.numba_status',
    'binauralbuilder_core.utils.settings_file',
    'binauralbuilder_core.utils.timeline_visualizer',

    # PyQt5 modules
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'PyQt5.QtMultimedia',
    'PyQt5.sip',

    # NumPy and related
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.lib.format',
    'numpy.fft',
    'numpy.random',

    # SciPy
    'scipy',
    'scipy.signal',
    'scipy.fft',
    'scipy.io',
    'scipy.io.wavfile',
    'scipy.interpolate',
    'scipy.ndimage',
    'scipy.special',
    'scipy.linalg',

    # Numba (optional JIT)
    'numba',
    'numba.core',
    'numba.np',
    'numba.np.ufunc',

    # Audio libraries
    'soundfile',
    'slab',

    # Plotting
    'matplotlib',
    'matplotlib.pyplot',
    'matplotlib.backends',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.figure',
    'plotly',
    'plotly.graph_objects',
    'plotly.express',

    # Data handling
    'pandas',
    'pandas.core',

    # Other dependencies
    'json',
    'pathlib',
    'logging',
    'typing',
    'dataclasses',
    'argparse',
    'time',
    'threading',
    'queue',
]

# Exclude packages that are not needed or cause issues
excludes = [
    'tkinter',
    'tcl',
    'tk',
    '_tkinter',
    'unittest',
    'test',
    'tests',
    'pytest',
    'IPython',
    'jupyter',
    'notebook',
    'sphinx',
    'docutils',
]

# Collect submodules for packages that need full collection
from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs

# Collect all numpy submodules
hiddenimports += collect_submodules('numpy')

# Collect scipy submodules (selectively to reduce size)
hiddenimports += collect_submodules('scipy.signal')
hiddenimports += collect_submodules('scipy.fft')
hiddenimports += collect_submodules('scipy.io')

# Collect PyQt5 submodules
hiddenimports += collect_submodules('PyQt5')

# Collect soundfile dependencies
hiddenimports += collect_submodules('soundfile')

# Collect matplotlib backends
hiddenimports += collect_submodules('matplotlib.backends')

# Try to collect numba (it's optional)
try:
    hiddenimports += collect_submodules('numba')
except Exception:
    print("Note: numba not available, skipping numba submodules")

# Collect dynamic libraries for soundfile
try:
    binaries += collect_dynamic_libs('soundfile')
except Exception:
    print("Note: Could not collect soundfile dynamic libs")

# Runtime hook for environment setup
RUNTIME_HOOK = os.path.join(SPEC_DIR, 'build_hooks', 'rthook_session_builder.py')
runtime_hooks = [RUNTIME_HOOK] if os.path.exists(RUNTIME_HOOK) else []

# Analysis configuration
a = Analysis(
    [ENTRY_POINT],
    pathex=[SPEC_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[os.path.join(SPEC_DIR, 'build_hooks')],
    hooksconfig={},
    runtime_hooks=runtime_hooks,
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

# Add source packages to the analysis
# This ensures all Python source files are included
for pkg_path, pkg_name in [(SRC_DIR, 'src'), (CORE_DIR, 'binauralbuilder_core')]:
    if os.path.isdir(pkg_path):
        for root, dirs, files in os.walk(pkg_path):
            # Skip __pycache__ directories
            dirs[:] = [d for d in dirs if d != '__pycache__']

            for file in files:
                if file.endswith('.py'):
                    full_path = os.path.join(root, file)
                    # Calculate the destination path relative to package root
                    rel_path = os.path.relpath(full_path, SPEC_DIR)
                    dest_dir = os.path.dirname(rel_path)
                    a.datas.append((rel_path, full_path, 'DATA'))

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Executable configuration
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SessionBuilder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging, False for release
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=os.path.join(SPEC_DIR, 'assets', 'icon.ico') if os.path.exists(os.path.join(SPEC_DIR, 'assets', 'icon.ico')) else None,
)

# Collect all files for distribution
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SessionBuilder',
)
