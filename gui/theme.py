"""
Loads style.qss, substitutes the design tokens, and applies it.

The stylesheet is a data file, so it has to be findable both from a source
checkout and from inside a PyInstaller bundle (where it is unpacked under
sys._MEIPASS). A missing stylesheet degrades to unstyled Qt rather than
crashing — an ugly window is better than an app that will not start.
"""

import re
import sys
from pathlib import Path

from gui import design

_TOKEN = re.compile(r'\{\{([A-Z_0-9]+)\}\}')


def _stylesheet_path():
    """Locate style.qss in a checkout or inside a packaged bundle."""
    candidates = [Path(__file__).resolve().parent / 'style.qss']
    bundle = getattr(sys, '_MEIPASS', None)
    if bundle:
        candidates.insert(0, Path(bundle) / 'gui' / 'style.qss')
        candidates.insert(1, Path(bundle) / 'style.qss')
    for path in candidates:
        if path.is_file():
            return path
    return None


def _tokens():
    """Public UPPER_CASE names from gui.design, rendered as strings."""
    out = {}
    for name in dir(design):
        if name.startswith('_') or not name.isupper():
            continue
        value = getattr(design, name)
        if isinstance(value, (str, int, float)):
            out[name] = str(value)
    return out


def build_stylesheet():
    """Return the substituted QSS, or '' if the file is unavailable."""
    path = _stylesheet_path()
    if path is None:
        return ''

    tokens = _tokens()
    missing = set()

    def replace(match):
        key = match.group(1)
        if key in tokens:
            return tokens[key]
        missing.add(key)
        return match.group(0)

    qss = _TOKEN.sub(replace, path.read_text(encoding='utf-8'))
    if missing:
        # Loud during development, harmless in production: an unsubstituted
        # token makes Qt discard the whole rule silently, which is hard to spot.
        print(f"[theme] warning: no design token for {sorted(missing)}",
              file=sys.stderr)
    return qss


def apply_theme(app):
    """Apply the stylesheet and the matplotlib defaults to a QApplication."""
    from gui.plot_style import apply_plot_style
    apply_plot_style()

    qss = build_stylesheet()
    if qss:
        app.setStyleSheet(qss)
    return bool(qss)


def set_style_property(widget, name, value):
    """
    Set a QSS-selector property and force Qt to restyle the widget.

    Qt resolves property selectors like QCheckBox[lowConfidence="true"] when a
    widget is polished. Changing the property afterwards has no visible effect
    until the widget is unpolished and polished again, so a value that changes
    at runtime must go through here.
    """
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    style = widget.style()
    if style is not None:
        style.unpolish(widget)
        style.polish(widget)
