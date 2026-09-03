"""
Icon loading.

Icons are Lucide SVGs (ISC licence, see gui/icons/LICENSE) bundled in
gui/icons/. They are drawn with stroke="currentColor", which Qt's SVG renderer
does **not** resolve — it renders such strokes as black — so the colour is
substituted textually before rendering. That also lets one file serve the normal
and disabled states.

Icons are decoration here, never the only label: every button keeps its text, so
a missing icon file costs nothing but a blank space. That is deliberate for an
audience that has not used the tool before.
"""

import sys
from pathlib import Path

from PyQt6.QtCore import QByteArray, Qt, QSize
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

from gui import design

DEFAULT_SIZE = 16

# (name, colour) -> QIcon. Rendering is cheap but happens on every panel build.
_cache = {}


def _icons_dir():
    """gui/icons in a checkout, or inside a PyInstaller bundle."""
    candidates = [Path(__file__).resolve().parent / 'icons']
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        candidates.insert(0, Path(bundle) / 'gui' / 'icons')
        candidates.insert(1, Path(bundle) / 'icons')
    for path in candidates:
        if path.is_dir():
            return path
    return None


def available():
    """True if the icon files were found. Used by the packaging self-test."""
    directory = _icons_dir()
    return bool(directory and any(directory.glob('*.svg')))


def _render(svg_text, colour, size):
    svg = svg_text.replace('currentColor', colour)
    renderer = QSvgRenderer(QByteArray(svg.encode('utf-8')))
    if not renderer.isValid():
        return None

    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter)
    painter.end()
    return pixmap


def icon(name, colour=None, size=DEFAULT_SIZE):
    """
    Load a bundled icon, recoloured.

    Returns an empty QIcon if the file is missing, so a caller never has to
    guard — the button simply shows its text alone.
    """
    colour = colour or design.TEXT
    key = (name, colour, size)
    if key in _cache:
        return _cache[key]

    directory = _icons_dir()
    path = directory / f'{name}.svg' if directory else None
    if path is None or not path.is_file():
        _cache[key] = QIcon()
        return _cache[key]

    try:
        svg_text = path.read_text(encoding='utf-8')
    except Exception:
        _cache[key] = QIcon()
        return _cache[key]

    result = QIcon()
    normal = _render(svg_text, colour, size)
    if normal is not None:
        result.addPixmap(normal, QIcon.Mode.Normal)
    faint = _render(svg_text, design.TEXT_FAINT, size)
    if faint is not None:
        result.addPixmap(faint, QIcon.Mode.Disabled)

    _cache[key] = result
    return result


def decorate(button, name, colour=None, size=DEFAULT_SIZE):
    """Give a button an icon beside its existing text. Returns the button."""
    button.setIcon(icon(name, colour, size))
    button.setIconSize(QSize(size, size))
    return button


def decorate_accent(button, name, size=DEFAULT_SIZE):
    """For a primary button: the icon must read against the accent fill."""
    return decorate(button, name, design.TEXT_ON_ACCENT, size)
