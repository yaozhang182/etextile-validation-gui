# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the E-Textile Validation GUI.

Build (must run ON Windows — PyInstaller cannot cross-compile):

    pyinstaller packaging/etextile.spec --noconfirm

Output: dist/ETextileValidation/ETextileValidation.exe plus its support files.

Two things break a naive build and are handled explicitly below:

1. MediaPipe ships its graphs and models as data files (modules/**/*.tflite and
   *.binarypb). PyInstaller's dependency analysis only follows imports, so
   without collect_data_files these are missing and the app crashes at runtime
   the first time a video is processed, not at startup.

2. Torch must come from the CPU-only wheel index. The default wheel bundles CUDA
   and is ~1.2 GB, which would dominate the installer for no benefit — the
   toolchain trains on CPU by design.

One-folder mode is deliberate: one-file mode unpacks several hundred MB to a
temp directory on every launch, which makes startup take tens of seconds.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
project_root = os.path.abspath(os.path.join(SPECPATH, '..'))

datas = []
binaries = []
hiddenimports = []

# --- MediaPipe: graphs + tflite models must be shipped verbatim -------------
datas += collect_data_files('mediapipe', include_py_files=False)
hiddenimports += collect_submodules('mediapipe.python')

# --- Scientific stack: a few modules are imported dynamically ---------------
for pkg in ('sklearn', 'scipy', 'matplotlib', 'seaborn'):
    hiddenimports += collect_submodules(pkg)
datas += collect_data_files('matplotlib')

# Only the Qt backend is used; excluding the others keeps the bundle smaller.
hiddenimports += ['matplotlib.backends.backend_qtagg']

# --- Application resources --------------------------------------------------
datas += [(os.path.join(project_root, 'config'), 'config')]

# The stylesheet is a data file, not an import, so PyInstaller will not find it
# on its own. Without it the app starts unstyled — gui/theme.py degrades rather
# than crashing, and app.py --selftest asserts it is present so CI catches this.
datas += [(os.path.join(project_root, 'gui', 'style.qss'), 'gui')]

# Icon SVGs, same reasoning. gui/icons.py returns an empty QIcon if they are
# missing, so a lost datas entry costs blank buttons rather than a crash — which
# is exactly why CI checks for them explicitly.
datas += [(os.path.join(project_root, 'gui', 'icons'), os.path.join('gui', 'icons'))]

a = Analysis(
    [os.path.join(project_root, 'app.py')],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Never imported by this app; each pulls in a large dependency tree.
        # NOTE: do NOT exclude submodules of torch itself. torch/__init__.py
        # imports several of them eagerly (distributions, fx, jit, ...), so
        # dropping one makes `import torch` fail with
        #   cannot import name '<sub>' from partially initialized module 'torch'
        # torchvision/torchaudio are separate top-level packages and are safe.
        'tkinter', 'PyQt5', 'PySide2', 'PySide6', 'IPython', 'jupyter',
        'notebook', 'pytest', 'torchvision', 'torchaudio',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ETextileValidation',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX corrupts some Qt and MediaPipe DLLs
    console=False,      # windowed app; see packaging/README.md for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ETextileValidation',
)
