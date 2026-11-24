# Project page

The source of https://yaozhang182.github.io/etextile-validation-gui/, published by
`.github/workflows/pages.yml` whenever this folder changes on `main`. Plain HTML, CSS
and JavaScript; nothing to build. Preview it locally with

    python -m http.server -d website 8000

and open http://localhost:8000 (the pose demo loads its data with `fetch`, which needs
a server rather than a `file://` URL).

| Folder | What | Made by |
|---|---|---|
| `media/tabs/`, `media/loop_*`, `media/hero.*` | screenshots and loops of the app | `tools/app_assets.py`, from `docs/video/capture.py` |
| `media/demo/` | the pose comparison data and its video | `tools/app_assets.py` |
| `media/paper/` | photos and layouts from the EICS 2026 paper (CC BY), faces masked | `tools/paper_figures.py`, from the paper's figures |
| `media/cover.jpg` | the image shown when the page is shared | a screenshot of the page |
