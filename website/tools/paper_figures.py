"""
Crop the photos and layout drawings for the project page out of the figures
of the EICS 2026 paper (Liu, Zhang et al., PACMHCI 10(4) EICS019, CC BY).

The figures are rendered as whole pages and then cropped, so everything drawn
on top of a photo in the paper (the boxes over faces, the labels, the motion
arrows) is kept. The raw photos embedded in the PDFs are never extracted.

    python website/tools/paper_figures.py <folder with the paper's figures/>

Needs PyMuPDF and Pillow. The paper's sources are not in this repository;
only the crops in website/media/paper/ are.
"""
import io
import sys
from pathlib import Path

import fitz
from PIL import Image

SRC = Path(sys.argv[1])
OUT = Path(__file__).resolve().parents[1] / 'media' / 'paper'
OUT.mkdir(parents=True, exist_ok=True)
ZOOM = 3.0


def render(name):
    page = fitz.open(SRC / f'{name}.pdf')[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
    return Image.open(io.BytesIO(pix.tobytes('png'))).convert('RGB'), page


def crop(img, box, pad=0):
    x0, y0, x1, y1 = box
    return img.crop((max(0, (x0 - pad) * ZOOM), max(0, (y0 - pad) * ZOOM),
                     min(img.width, (x1 + pad) * ZOOM), min(img.height, (y1 + pad) * ZOOM)))


def save(im, name, width):
    im = im.copy()
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(OUT / name, quality=86, optimize=True)
    print('wrote', name, im.size)


# --- the recording setup and the sensor -----------------------------------
img, page = render('hardware_setup')
save(crop(img, (0, 0, 267, 356)), 'setup.jpg', 800)
save(crop(img, (275, 0, 410, 356)), 'sensor.jpg', 500)

# --- the six motions ---------------------------------------------------------
img, page = render('all_motion')
photos = sorted((i['bbox'] for i in page.get_image_info() if i['width'] == 964),
                key=lambda b: b[0])
for k, b in enumerate(photos):
    save(crop(img, (b[0], max(b[1], 0), b[2], b[3])), f'motion_{k + 1}.jpg', 420)

# --- sensor layouts: one crop per design, front view above back view --------
img, page = render('all_design')
boxes = [i['bbox'] for i in page.get_image_info() if i['width'] in (336, 352)]
columns = []                               # [(is_top_block, [boxes])]
for b in sorted(boxes, key=lambda b: b[0]):
    top = b[1] < 800                      # upper block: H series, lower: LH series
    for is_top, bs in columns:
        if is_top == top and abs(bs[0][0] - b[0]) < 40:
            bs.append(b)
            break
    else:
        columns.append((top, [b]))
for top, bs in columns:
    x0 = min(b[0] for b in bs); x1 = max(b[2] for b in bs)
    y0 = min(b[1] for b in bs); y1 = max(b[3] for b in bs)
    tag = 'H' if top else 'LH'
    # stop above the next row's labels (H) and the panel captions (LH)
    y1 = min(y1, 738 if top else 1405)
    save(crop(img, (x0, y0, x1, y1)), f'layout_{tag}_{int(x0)}.jpg', 300)

# file names by designer and iteration, as in the paper's figure
NAMES = {'H_9': 'A_H0', 'H_219': 'A_H1', 'H_429': 'A_H2', 'LH_16': 'L0', 'LH_219': 'A_LH1',
         'H_681': 'B_H0', 'H_885': 'B_H1', 'H_1103': 'B_H2', 'LH_891': 'B_LH1', 'LH_1113': 'B_LH2',
         'H_1392': 'C_H0', 'H_1607': 'C_H1', 'H_1827': 'C_H2', 'LH_1601': 'C_LH1',
         'LH_1835': 'C_LH2'}
for f in OUT.glob('layout_*_*.jpg'):
    key = f.stem[len('layout_'):]
    if key in NAMES:
        f.replace(OUT / f'layout_{NAMES[key]}.jpg')
    elif key.count('_') == 1 and key.split('_')[0] in ('H', 'LH'):
        f.unlink()                        # L0 repeated in each designer's block
