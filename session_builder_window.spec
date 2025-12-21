# session_builder_window.spec
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

hiddenimports = []
datas = []
binaries = []

# Qt
hiddenimports += collect_submodules("PyQt5")

# slab data (KEMAR etc.)
hiddenimports += collect_submodules("slab")
datas += collect_data_files("slab", include_py_files=True)

# realtime_backend (Rust/PyO3 extension + dependent DLLs)
hiddenimports += collect_submodules("realtime_backend")
binaries += collect_dynamic_libs("realtime_backend")

binaries += [
    (
        r"C:\Users\alexb\Downloads\binauralbuilder\bb2\Lib\site-packages\realtime_backend\realtime_backend.cp313-win_amd64.pyd",
        ".",
    ),
]


a = Analysis(
    ["src/audio/session_builder_launcher.py"],
    pathex=["src"],
    binaries=binaries,
    datas=[
        ('src/presets', 'src/presets'),
        ('audio_config.ini', '.'),
    ],
    hiddenimports=hiddenimports,
)

pyz = PYZ(a.pure)


exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Session Builder",
    console=True,
    debug=False,
    strip=False,
    upx=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Session Builder",
)