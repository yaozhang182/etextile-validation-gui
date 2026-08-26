# Packaging a Windows app

Goal: hand a textile researcher a zip they can unpack and double-click — no
Python, no conda, no terminal. This is the concrete answer to the reviewer's
objection that "no-code" still meant installing a Python environment.

## Status

The spec and workflow here are **written but not yet run**. They were authored on
Linux, and PyInstaller cannot cross-compile — a Windows machine or runner has to
produce the binary. Budget one or two debug iterations on the first real build.

## Recommended route: GitHub Actions

No Windows machine needed.

1. Create the repository and push:

   ```bash
   git init
   git add .
   git commit -m "E-Textile Validation GUI"
   gh repo create etextile-validation-gui --private --source=. --push
   ```

2. In the repo's **Actions** tab, run **Build Windows app** (`workflow_dispatch`).
3. Download the `ETextileValidation-windows` artifact when it finishes.

Tagging a release builds automatically and attaches the zip:

```bash
git tag v0.2.0 && git push --tags
```

## Alternative: build on your own Windows machine

```bat
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pyinstaller
pyinstaller packaging/etextile.spec --noconfirm
```

Result: `dist\ETextileValidation\ETextileValidation.exe`. Zip the whole
`ETextileValidation` folder — the .exe does not work on its own.

## What the spec handles, and why

| Problem | Handling |
|---|---|
| MediaPipe's `.tflite` / `.binarypb` assets are data, not imports, so PyInstaller misses them — the app then builds fine and crashes the first time a video is processed | `collect_data_files('mediapipe')`, plus a workflow step that fails the build if `pose_landmark*.tflite` is absent |
| Default torch wheel bundles CUDA (~1.2 GB) for no benefit | installed from `https://download.pytorch.org/whl/cpu` |
| One-file mode unpacks hundreds of MB to temp on every launch | one-**folder** mode (`COLLECT`) |
| UPX compression corrupts some Qt and MediaPipe DLLs | `upx=False` |
| `sklearn`/`scipy` import submodules dynamically | `collect_submodules` for each |

Expected size: roughly 500–700 MB unpacked, 200–300 MB zipped. Most of it is
torch and MediaPipe; there is no way around that short of dropping a dependency.

## Debugging a failed build

The app is built windowed (`console=False`), so a crash shows nothing. To see the
traceback, set `console=True` in `etextile.spec`, rebuild, and run the .exe from
`cmd`.

Failure modes in rough order of likelihood:

- **Crash on "Extract Skeleton"** — MediaPipe assets missing. Check
  `dist\ETextileValidation\_internal\mediapipe\modules\` contains the `.tflite`
  files; the CI step above catches this.
- **`ImportError` on launch** — a dynamically imported module was not detected.
  Add it to `hiddenimports` in the spec.
- **Missing DLL** — usually the MSVC runtime; install the Visual C++
  Redistributable on the target machine.
- **Antivirus quarantines the .exe** — unsigned PyInstaller binaries are a known
  false-positive source. Code signing fixes it properly; otherwise document it.

## Still worth doing regardless

Even without the .exe, `requirements.txt` now points at the CPU-only torch
wheel, which cuts a fresh install from ~1.4 GB to ~250 MB.
