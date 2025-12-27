"""
PyInstaller hook for the src package.

This hook ensures all audio, ui, models, and utils submodules are included.
"""

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all submodules
hiddenimports = collect_submodules('src')

# Explicitly include key modules
hiddenimports += [
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
    'src.init',
]

# Collect data files (presets, etc.)
datas = collect_data_files('src')
