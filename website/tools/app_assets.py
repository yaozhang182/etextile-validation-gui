"""
Build the project page's media from the application itself.

1. Screenshots of the six tabs with the position of every control the page
   points at, taken from the tutorial video's capture
   (docs/video/capture.py writes docs/video/build/shots/).
2. Short silent loops of the same capture: scrubbing the skeleton, the Tab 3
   cursor, and the Pose tab.
3. The data for the page's own pose comparison: example session 3 as the test
   recording of a model trained, through the GUI, on sessions 1, 2 and 4, with
   the predicted right arm drawn into the skeleton exactly as Tab 5 draws it.

    python docs/video/capture.py          # once, ~20 min
    QT_QPA_PLATFORM=offscreen python website/tools/app_assets.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SHOTS = Path(os.environ.get('VIDEO_BUILD', ROOT / 'docs' / 'video' / 'build')) / 'shots'
OUT = ROOT / 'website' / 'media'
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import imageio_ffmpeg  # noqa: E402
FF = imageio_ffmpeg.get_ffmpeg_exe()

shots = json.load(open(SHOTS / 'shots.json'))


def rect_pct(r, size=(1920, 1080)):
    """[x, y, w, h] in pixels -> percentages of the screenshot."""
    if isinstance(r[0], list):                 # a list of rects: their union
        x0 = min(a[0] for a in r); y0 = min(a[1] for a in r)
        x1 = max(a[0] + a[2] for a in r); y1 = max(a[1] + a[3] for a in r)
        r = [x0, y0, x1 - x0, y1 - y0]
    return [round(100 * r[0] / size[0], 2), round(100 * r[1] / size[1], 2),
            round(100 * r[2] / size[0], 2), round(100 * r[3] / size[1], 2)]


# --- 1. tab screenshots and their hotspots ----------------------------------
TABS = {
    'tab1': ('tab1_tables', ['train_table', 'test_table', 'extract']),
    'tab2': ('tab2_still', ['skeleton', 'slider', 'tq']),
    'tab3': ('tab3_selected', ['angles', 'sensors', 'plots', 'info']),
    'tab4': ('tab4_done', []),
    'tab5': ('tab5_still', ['pred_ax', 'ghost', 'values', 'scrub']),
    'tab6': ('tab6_top', ['headline', 'table', 'pred']),
}
(OUT / 'tabs').mkdir(parents=True, exist_ok=True)
hotspots = {}
for key, (shot, names) in TABS.items():
    im = Image.open(shots[shot]['file']).convert('RGB')
    im.resize((1600, 900), Image.LANCZOS).save(OUT / 'tabs' / f'{key}.jpg', quality=84,
                                               optimize=True)
    rects = shots[shot]['rects']
    if key == 'tab4':
        # the training tab: summary, settings, start button and loss curve
        names = [n for n in ('summary', 'hyper', 'start', 'loss') if n in rects]
        if 'summary' not in rects:
            rects = {**rects, **shots['tab4_ready']['rects']}
            names = ['summary', 'hyper', 'start', 'loss']
    hotspots[key] = {n: rect_pct(rects[n]) for n in names if n in rects}
json.dump(hotspots, open(OUT / 'tabs' / 'hotspots.json', 'w'), indent=1)
print('tabs:', {k: list(v) for k, v in hotspots.items()})


# --- 2. loops ---------------------------------------------------------------
def loop(prefix, n, name, crop=None, fps=12, width=1280, bounce=True):
    """A silent H.264 loop from a numbered capture sequence."""
    frames = [Image.open(SHOTS / f'{prefix}{k:02d}.png').convert('RGB') for k in range(n)]
    if crop:
        frames = [f.crop(crop) for f in frames]
    if bounce:                                  # forward then back: a seamless loop
        frames = frames + frames[-2:0:-1]
    w = width; h = round(frames[0].height * w / frames[0].width / 2) * 2
    cmd = [FF, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{w}x{h}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
           '-pix_fmt', 'yuv420p', '-crf', '26', '-preset', 'slow', '-movflags', '+faststart',
           str(OUT / name)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.resize((w, h), Image.LANCZOS).tobytes())
    p.stdin.close(); p.wait()
    frames[0].resize((w, h), Image.LANCZOS).save(OUT / name.replace('.mp4', '.jpg'), quality=82)
    print('loop', name, w, h, (OUT / name).stat().st_size // 1024, 'kB')


def union_px(*names, shot):
    rs = [shots[shot]['rects'][n] for n in names]
    x0 = min(r[0] for r in rs); y0 = min(r[1] for r in rs)
    x1 = max(r[0] + r[2] for r in rs); y1 = max(r[1] + r[3] for r in rs)
    return (x0, y0, x1, y1)


loop('tab5_pose', 48, 'hero.mp4', crop=union_px('video', 'canvas', shot='tab5_pose00'),
     width=1600)
loop('tab2_f', 35, 'loop_tab2.mp4', width=1280)
loop('tab3_cur', 36, 'loop_tab3.mp4', width=1280)
loop('tab5_pose', 48, 'loop_tab5.mp4', width=1280)


# --- 3. the pose comparison data ---------------------------------------------
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import torch  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402
torch.manual_seed(1); np.random.seed(1)
app = QApplication.instance() or QApplication(sys.argv)
from gui.main_window import MainWindow  # noqa: E402
from gui.state import make_subject  # noqa: E402
from gui.panels.data_import import read_csv_auto  # noqa: E402

w = MainWindow()
E = ROOT / 'input_data' / 'examples_data'
for split, n in (('train', 1), ('train', 2), ('train', 4), ('test', 3)):
    d = E / f'session_{n}'
    s = make_subject(f'S{n}', str(d / 'video.mp4'), str(d / 'sensor.csv'), str(d / 'frames.csv'))
    s['sensor_df'] = read_csv_auto(s['sensor_path']); s['video_df'] = read_csv_auto(s['video_csv_path'])
    w.state[f'{split}_subjects'].append(s)
    w.tab_import._load_cached_angles(s)
t3 = w.tab_comparison
t3.refresh()
for bone in ('right_upper_arm', 'right_forearm'):
    t3._on_bone_clicked(bone)
tr = w.tab_training
tr.epochs_spin.setValue(50)
tr._start_training()
while tr._worker is not None and tr._worker.isRunning():
    app.processEvents(); time.sleep(0.05)
app.processEvents()
p = w.tab_pose
p.refresh()
d = p._data
names = w.state['config']['target_names']
metrics = w.state['metrics']
owner = d['owner']
have = np.flatnonzero(owner >= 0)
first, last = int(have[0]), int(have[-1])
joints = d['joints']
skel = w.tab_skeleton
from gui.panels.skeleton_viewer import SKELETON_BONES  # noqa: E402
from core.bones import BONES  # noqa: E402


def r3(a):
    return np.round(np.asarray(a, float), 3).tolist()


readout = {}
for name, kind, vals in d['readout']:
    if kind == 'dir':
        readout[name] = {'kind': 'direction', 'error': np.round(vals, 1).tolist()}
    else:
        readout[name] = {'kind': 'angle', 'pred': np.round(vals[0], 1).tolist(),
                         'true': np.round(vals[1], 1).tolist()}
data = {
    'note': 'Example session 3 as test recording; model trained on sessions 1, 2 and 4 '
            'through the GUI (HybridCNNLSTM, 50 epochs). Example data only, not a result.',
    'fps': 30, 'first': first, 'last': last,
    'bones': [list(b) for b in SKELETON_BONES],
    'predicted': {b: [BONES[b][1], BONES[b][2]] for b in sorted(d['bones'])},
    'owner': owner[first:last + 1].tolist(),
    'true': [r3(joints[f]) for f in range(first, last + 1)],
    'recon': [r3(x) for x in d['recon']],
    'readout': readout,
    'offset': r3(d['offset']), 'extent': round(float(d['extent']), 3),
    'mpjae': {k: round(v, 1) for k, v in metrics['MPJAE_per_joint'].items()},
}
(OUT / 'demo').mkdir(exist_ok=True)
json.dump(data, open(OUT / 'demo' / 'pose.json', 'w'), separators=(',', ':'))
print('pose.json', (OUT / 'demo' / 'pose.json').stat().st_size // 1024, 'kB',
      'frames', first, last, 'mpjae', data['mpjae'])

# the test video, frames first..last, re-timed to exactly 30 fps so that
# frame k of the clip is at k / 30 s
import cv2  # noqa: E402
cap = cv2.VideoCapture(str(E / 'session_3' / 'video.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES, first)
W, H = 640, 360
cmd = [FF, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}',
       '-r', '30', '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '27',
       '-preset', 'slow', '-movflags', '+faststart', '-g', '15', str(OUT / 'demo' / 'session3.mp4')]
proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
for k in range(first, last + 1):
    ok, frame = cap.read()
    if not ok:
        break
    proc.stdin.write(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA).tobytes())
proc.stdin.close(); proc.wait()
print('session3.mp4', (OUT / 'demo' / 'session3.mp4').stat().st_size // 1024, 'kB')
