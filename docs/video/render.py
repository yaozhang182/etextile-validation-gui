"""Compose the tutorial from captured shots, narration clips and the beat script."""
import json, math, os, re, subprocess, sys, wave
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(__file__))
import imageio_ffmpeg, matplotlib
from script import BEATS

W, H, FPS = 1920, 1080, 30
ROOT = str(Path(__file__).resolve().parents[2])
V = os.environ.get('VIDEO_BUILD', os.path.join(ROOT, 'docs', 'video', 'build'))
OUTFILE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(V, 'etextile_tutorial.mp4')
PREVIEW = os.environ.get('PREVIEW')          # render only a few beats, for checking
FF = imageio_ffmpeg.get_ffmpeg_exe()
FD = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
def font(size, bold=False):
    return ImageFont.truetype(os.path.join(FD, 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'), size)

ORANGE = (240, 138, 36); NAVY = (22, 32, 43); ACCENT = (11, 92, 138); BG = (233, 237, 242)
shots = json.load(open(f'{V}/shots/shots.json'))
durs = json.load(open(f'{V}/audio/durations.json'))

# ------------------------------------------------------------------ cards
def text_center(d, y, s, f, fill):
    w = d.textlength(s, font=f); d.text(((W - w) / 2, y), s, font=f, fill=fill)

def card_title():
    im = Image.new('RGB', (W, H), NAVY); d = ImageDraw.Draw(im)
    d.rectangle([0, H - 14, W, H], fill=ACCENT)
    text_center(d, 360, 'E-Textile Validation GUI', font(96, True), (255, 255, 255))
    text_center(d, 500, 'Validate a motion-sensing garment from a webcam and a sensor CSV', font(40), (200, 214, 228))
    text_center(d, 560, 'no motion capture, no code', font(40), (200, 214, 228))
    sys.path.insert(0, ROOT); from app import __version__
    text_center(d, 700, f'Tutorial  ·  version {__version__}', font(30), (140, 160, 180))
    return im

def card_pipeline():
    im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
    fig = Image.open(f'{ROOT}/docs/img/pipeline_overview.png').convert('RGB')
    fig.thumbnail((700, 840)); im.paste(fig, (200, 60))
    x = 1000
    d.text((x, 200), 'What the tool does', font=font(58, True), fill=NAVY)
    steps = ['Webcam video + sensor CSV, on one clock', 'Body pose → joint angles (ground truth)',
             'Model learns: sensors → joint angles', 'Per-joint accuracy and sensor importance']
    for i, s in enumerate(steps):
        y = 330 + i * 110
        d.ellipse([x, y, x + 60, y + 60], fill=ACCENT)
        d.text((x + 30, y + 30), str(i + 1), font=font(32, True), fill='white', anchor='mm')
        d.text((x + 90, y + 12), s, font=font(36), fill=NAVY)
    return im

def card_files():
    import pandas as pd, cv2
    im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
    text_center(d, 80, 'One session = three files', font(60, True), NAVY)
    s = f'{ROOT}/input_data/examples_data/session_1'
    cap = cv2.VideoCapture(f'{s}/video.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, 160); ok, fr = cap.read()
    thumb = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).crop((360, 0, 980, 720)); thumb.thumbnail((500, 420))
    cols = [(120, 'video.mp4', 'the webcam recording'), (720, 'sensor.csv', 'one timestamp + one column per sensor'),
            (1320, 'frames.csv', 'the timestamp of every video frame')]
    for x, name, sub in cols:
        d.rounded_rectangle([x, 200, x + 520, 780], 14, fill='white', outline=(197, 204, 214), width=2)
        d.text((x + 30, 222), name, font=font(40, True), fill=ACCENT)
        d.text((x + 30, 276), sub, font=font(24), fill=(91, 104, 118))
    im.paste(thumb, (120 + 10 + (500 - thumb.width) // 2, 330))
    mono = ImageFont.truetype(os.path.join(FD, 'DejaVuSansMono.ttf'), 22)
    sys.path.insert(0, ROOT)
    from gui.panels.data_import import read_csv_auto            # the app's own reader
    sd = read_csv_auto(f'{s}/sensor.csv'); vd = read_csv_auto(f'{s}/frames.csv')
    tables = [(720, ['EpochTime', 'S1', 'S2', 'S3'], sd, ['{:.3f}', '{:.0f}', '{:.0f}', '{:.0f}'], [16, 7, 7, 7]),
              (1320, ['FrameIndex', 'EpochTime'], vd, ['{:.0f}', '{:.3f}'], [11, 16])]
    for x, cols_, df, fmts, widths in tables:
        head = ''.join(f'{c:<{wd}}' for c, wd in zip(cols_ + ([' ...'] if len(cols_) > 2 else []), widths + [4]))
        d.text((x + 24, 335), head, font=mono, fill=ACCENT)
        for k in range(9):
            row = ''.join(f'{fm.format(float(df[c].iloc[k])):<{wd}}' for c, fm, wd in zip(cols_, fmts, widths))
            d.text((x + 24, 375 + k * 40), row, font=mono, fill=NAVY)
    d.rounded_rectangle([700, 812, 1860, 886], 12, fill=ACCENT)
    msg = 'Both CSVs use the same clock (Unix time in seconds)'; bf = font(34, True)
    d.text((1280 - d.textlength(msg, font=bf) / 2, 828), msg, font=bf, fill=(255, 255, 255))
    d.line([(980, 812), (980, 780)], fill=ACCENT, width=5); d.line([(1580, 812), (1580, 780)], fill=ACCENT, width=5)
    return im

def card_end():
    im = Image.new('RGB', (W, H), NAVY); d = ImageDraw.Draw(im)
    d.rectangle([0, H - 14, W, H], fill=ACCENT)
    text_center(d, 330, 'github.com/yaozhang182/etextile-validation-gui', font(64, True), (255, 255, 255))
    text_center(d, 470, 'Windows download  ·  User manual  ·  Example data', font(40), (200, 214, 228))
    text_center(d, 560, 'Report a bug  ·  ★ Star the project', font(40), (200, 214, 228))
    return im

CARDS = {'card_title': card_title, 'card_pipeline': card_pipeline, 'card_files': card_files, 'card_end': card_end}
_cache = {}
def base_image(name):
    if name not in _cache:
        _cache[name] = CARDS[name]() if name in CARDS else Image.open(shots[name]['file']).convert('RGB')
        assert _cache[name].size == (W, H), (name, _cache[name].size)
    return _cache[name]

def rects(beat, key):
    shot = beat['shot']
    if isinstance(shot, tuple):
        shot = f"{shot[1]}00"
    r = shots.get(shot, {}).get('rects', {}).get(key)
    if r is None:
        raise KeyError(f"{shot}: no rect {key}")
    return r if isinstance(r[0], list) else [r]

def union(rs):
    x0 = min(r[0] for r in rs); y0 = min(r[1] for r in rs)
    return [x0, y0, max(r[0] + r[2] for r in rs) - x0, max(r[1] + r[3] for r in rs) - y0]

# ------------------------------------------------------------------ timing
LEAD, TAIL = 0.35, 0.55
timeline, t = [], 0.0
for i, b in enumerate(BEATS):
    dur = max(LEAD + durs[i] + TAIL + b.get('hold', 0.0), 2.4)
    timeline.append((t, dur)); t += dur
TOTAL = t
if PREVIEW:
    keep = [int(x) for x in PREVIEW.split(',')]
    sel = [(i, BEATS[i]) for i in keep]
else:
    sel = list(enumerate(BEATS))

def ease(x):
    x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)

def view_for(beat):
    if not beat.get('zoom'):
        return [0, 0, W, H]
    r = union(rects(beat, beat['zoom']))
    cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
    vw = max(r[2] * 1.6, r[3] * 1.6 * W / H, W / 2.1)          # never more than ~2x
    vw = min(vw, W); vh = vw * H / W
    x0 = min(max(cx - vw / 2, 0), W - vw); y0 = min(max(cy - vh / 2, 0), H - vh)
    return [x0, y0, vw, vh]

def lerp(a, b, k):
    return [a[i] + (b[i] - a[i]) * k for i in range(len(a))]

# captions: sentence chunks of at most ~84 characters, timed by length
def chunks(text, limit=100):
    """Sentences, split further only where needed, preferring a comma or colon."""
    out = []
    for sent in re.split(r'(?<=[.!?])\s+', text.strip()):
        while len(sent) > limit:
            cands = [m.end() for m in re.finditer(r'[,:;] ', sent) if 30 <= m.end() <= limit]
            cut = cands[-1] if cands else sent.rfind(' ', 0, limit)
            out.append(sent[:cut].strip()); sent = sent[cut:].strip()
        out.append(sent)
    return out

CAPS = []
for i, b in enumerate(BEATS):
    st, _ = timeline[i]; cs = chunks(b['say']); total = sum(len(c) for c in cs); acc = 0
    for c in cs:
        a = st + LEAD + durs[i] * acc / total; acc += len(c)
        CAPS.append((a, st + LEAD + durs[i] * acc / total + 0.15, c, i))

def srt_time(x):
    h, x = divmod(x, 3600); m, x = divmod(x, 60); s = int(x); ms = int(round((x - s) * 1000))
    return f'{int(h):02d}:{int(m):02d}:{s:02d},{min(ms,999):03d}'

# ------------------------------------------------------------------ drawing
CUR = [(0, 0), (0, 34), (9, 26), (15, 40), (21, 37), (15, 24), (27, 24)]
def draw_cursor(d, x, y, s=1.25):
    pts = [(x + px * s, y + py * s) for px, py in CUR]
    d.polygon([(p[0] + 2, p[1] + 3) for p in pts], fill=(0, 0, 0, 90))
    d.polygon(pts, fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))

capfont = font(38)
def draw_caption(im, text):
    """One line if it fits, otherwise two lines broken near the middle."""
    d = ImageDraw.Draw(im, 'RGBA')
    if d.textlength(text, font=capfont) <= 1560:
        lines = [text]
    else:
        words = text.split(); best = None
        for k in range(1, len(words)):
            a, b = ' '.join(words[:k]), ' '.join(words[k:])
            cost = max(d.textlength(a, font=capfont), d.textlength(b, font=capfont))
            if best is None or cost < best[0]:
                best = (cost, [a, b])
        lines = best[1]
    lh = 52; bh = lh * len(lines) + 28; y0 = H - 60 - bh
    bw = max(d.textlength(l, font=capfont) for l in lines) + 56
    d.rounded_rectangle([(W - bw) / 2, y0, (W + bw) / 2, y0 + bh], 14, fill=(12, 18, 26, 205))
    for k, l in enumerate(lines):
        tw = d.textlength(l, font=capfont)
        d.text(((W - tw) / 2, y0 + 14 + k * lh), l, font=capfont, fill=(255, 255, 255, 255))

def render_beat_frame(i, b, tl, cur_from, view_from):
    """One frame of beat i at local time tl (seconds)."""
    shot = b['shot']
    if isinstance(shot, tuple):
        _, prefix, n, fps = shot
        k = min(int(tl * fps), n - 1); name = f'{prefix}{k:02d}'
    else:
        name = shot
    im = base_image(name).copy().convert('RGBA')
    is_card = name.startswith('card_')
    over = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(over)

    target = b.get('click') or b.get('point')
    move_end = 0.25 + 0.75 if target else 0.25
    hl_alpha = ease((tl - move_end) / 0.3)
    hls = [r for key in b.get('hl', []) for r in rects(b, key)]
    if hls and not is_card and hl_alpha > 0:
        mask = Image.new('L', (W, H), int(80 * hl_alpha)); md = ImageDraw.Draw(mask)
        for r in hls:
            md.rounded_rectangle([r[0] - 10, r[1] - 10, r[0] + r[2] + 10, r[1] + r[3] + 10], 12, fill=0)
        shade = Image.new('RGBA', (W, H), (10, 16, 24, 255)); shade.putalpha(mask)
        over = Image.alpha_composite(over, shade); d = ImageDraw.Draw(over)
        for r in hls:
            d.rounded_rectangle([r[0] - 10, r[1] - 10, r[0] + r[2] + 10, r[1] + r[3] + 10], 12,
                                outline=ORANGE + (int(255 * hl_alpha),), width=5)
        if b.get('label'):
            r = hls[0]; lf = font(30, True); tw = d.textlength(b['label'], font=lf)
            lx = min(max(r[0] - 10, 20), W - tw - 60); ly = r[1] - 70 if r[1] > 110 else r[1] + r[3] + 22
            d.rounded_rectangle([lx, ly, lx + tw + 36, ly + 50], 10, fill=ORANGE + (int(245 * hl_alpha),))
            d.text((lx + 18, ly + 8), b['label'], font=lf, fill=(255, 255, 255, int(255 * hl_alpha)))
    if target:
        r = union(rects(b, target)); tx, ty = r[0] + r[2] * 0.55, r[1] + r[3] * 0.55
        k = ease((tl - 0.25) / 0.75)
        cx, cy = cur_from[0] + (tx - cur_from[0]) * k, cur_from[1] + (ty - cur_from[1]) * k
        if b.get('click'):
            rt = tl - move_end
            if 0 < rt < 0.45:
                rad = 12 + 46 * (rt / 0.45)
                d.ellipse([tx - rad, ty - rad, tx + rad, ty + rad], outline=ORANGE + (int(230 * (1 - rt / 0.45)),), width=5)
        draw_cursor(d, cx, cy)
    im = Image.alpha_composite(im, over).convert('RGB')

    vk = ease(tl / 0.8)
    v = lerp(view_from, view_for(b), vk)
    if v != [0, 0, W, H]:
        im = im.resize((W, H), Image.BICUBIC, box=(v[0], v[1], v[0] + v[2], v[1] + v[3]))
    return im, (target and (union(rects(b, target))[0] + union(rects(b, target))[2] * 0.55,
                            union(rects(b, target))[1] + union(rects(b, target))[3] * 0.55))

# ------------------------------------------------------------------ encode
def main():
    # audio
    sr = 48000
    total = TOTAL if not PREVIEW else sum(timeline[i][1] for i, _ in sel)
    audio = np.zeros(int(total * sr) + sr, dtype=np.float32)
    off = 0.0
    starts = {}
    for i, b in sel:
        starts[i] = off
        with wave.open(f'{V}/audio/beat_{i:02d}.wav') as f:
            pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16).astype(np.float32) / 32768
        a = int((off + LEAD) * sr); audio[a:a + len(pcm)] += pcm
        off += timeline[i][1]
    wav_path = f'{V}/narration.wav'
    with wave.open(wav_path, 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(sr)
        f.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())

    # captions file (full render only)
    if not PREVIEW:
        with open(OUTFILE[:-4] + '.srt', 'w') as f:
            for n, (a, z, c, _) in enumerate(CAPS, 1):
                f.write(f'{n}\n{srt_time(a)} --> {srt_time(z)}\n{c}\n\n')

    proc = subprocess.Popen([FF, '-y', '-loglevel', 'error',
                             '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                             '-i', wav_path, '-c:v', 'libx264', '-preset', 'medium', '-crf', '19',
                             '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-shortest',
                             '-movflags', '+faststart', OUTFILE], stdin=subprocess.PIPE)
    cursor = (W * 0.62, H * 0.72); view = [0, 0, W, H]; prev_last = None; prev_shot = None
    frames = 0
    for i, b in sel:
        dur = timeline[i][1]; n = int(round(dur * FPS))
        caps = [c for c in CAPS if c[3] == i]
        shot_key = b['shot'] if not isinstance(b['shot'], tuple) else b['shot'][1]
        fade = prev_last is not None and shot_key != prev_shot
        last_static = None
        for fi in range(n):
            tl = fi / FPS
            animating = tl < 1.4 or isinstance(b['shot'], tuple)
            if animating or last_static is None:
                im, tgt = render_beat_frame(i, b, tl, cursor, view)
                if not animating:
                    last_static = im
            else:
                im = last_static
            if fade and tl < 0.35:
                im = Image.blend(prev_last, im, ease(tl / 0.35))
            gt = timeline[i][0] + tl
            cap = next((c for c in caps if c[0] <= gt < c[1]), None)
            frame = im.copy()
            if cap:
                draw_caption(frame, cap[2])
            proc.stdin.write(frame.tobytes()); frames += 1
        prev_last = im; prev_shot = shot_key
        if tgt:
            cursor = tgt
        view = view_for(b)
        print(f'beat {i:2d} done ({frames / FPS:6.1f} s)', flush=True)
    proc.stdin.close(); proc.wait()
    print('wrote', OUTFILE, f'{frames / FPS:.1f} s')

if __name__ == '__main__':
    main()
