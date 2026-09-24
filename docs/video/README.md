# Tutorial video

The [5-minute tutorial](https://www.youtube.com/watch?v=PFFmzeKfwpI) is generated
from the real application, so it can be regenerated whenever the interface changes.

| File | What it does |
|---|---|
| `script.py` | The narration, beat by beat: what is said, which screen it plays over, and which control is highlighted, clicked or zoomed. **Edit this to change the video.** |
| `capture.py` | Runs the application headlessly on the example data — imports the sessions, runs one real extraction and two real training runs — and saves every screen state at 1920×1080 with the pixel rectangle of each control. About 20 minutes. |
| `tts.py` | Speaks each beat with Microsoft Edge text-to-speech (`en-US-AndrewNeural`; needs a network connection). |
| `render.py` | Composes the frames — cursor, highlights, zooms, captions — adds the narration, and writes `etextile_tutorial.mp4` and a matching `.srt`. |

## Regenerating

```bash
pip install edge-tts imageio-ffmpeg      # in addition to the app's own requirements
python docs/video/capture.py             # screens -> docs/video/build/shots/
python docs/video/tts.py                 # narration -> docs/video/build/audio/
python docs/video/render.py              # -> docs/video/build/etextile_tutorial.mp4
```

Everything is written to `docs/video/build/` (ignored by git); set `VIDEO_BUILD` to
use another folder. `PREVIEW=0,7,14 python docs/video/render.py` renders only the
listed beats, which is much faster when adjusting one part.

Change only the wording in `script.py` → re-run `tts.py` and `render.py`. Change the
interface → re-run all three. `tts.py` skips clips that already exist, so delete
`build/audio/` after editing narration.

`capture.py` limits PyTorch to two threads by default. With one thread per core on
a machine that grants the process only a few cores, training slows dramatically.
