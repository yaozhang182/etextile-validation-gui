"""
A front-view body outline whose bones can be clicked to pick regression
targets (Tab 3).

The outline is a fixed drawing, not the subject's pose: a seated or turned
participant would make bones overlap and hard to hit. It is drawn as seen by
the camera, so the person's right is on the left of the screen.

Each bone is coloured by its tracking confidence; a picked bone gets a blue
halo, dashed when only some of its targets are ticked (e.g. after unticking one
in the list view).
"""

import numpy as np
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QSizePolicy

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import cm

from core.bones import BONES, selection_state
from gui import design

# Template joints, front view: x = the person's left (so their right is on
# the screen's left), y = up, metres.
_P = {
    0: (0.0, 0.95), 1: (0.10, 0.90), 2: (-0.10, 0.90),
    4: (0.12, 0.50), 5: (-0.12, 0.50), 7: (0.13, 0.11), 8: (-0.13, 0.11),
    10: (0.21, 0.05), 11: (-0.21, 0.05),
    12: (0.0, 1.47), 15: (0.0, 1.62),
    16: (0.19, 1.43), 17: (-0.19, 1.43), 18: (0.30, 1.17), 19: (-0.30, 1.17),
    20: (0.40, 0.94), 21: (-0.40, 0.94),
}
#: Drawn for shape only, not clickable: the shoulder and hip lines.
_CONNECTORS = [(12, 16), (12, 17), (0, 1), (0, 2)]

_HALO = design.ACCENT
_UNKNOWN = '#9aa5b3'


def _confidence_colour(conf):
    if conf is None or not np.isfinite(conf):
        return _UNKNOWN
    # muted red -> amber -> green over the 0.3 .. 0.9 range that matters
    return cm.RdYlGn(float(np.clip((conf - 0.3) / 0.6, 0, 1)) * 0.8 + 0.1)


class BonePicker(FigureCanvas):
    """Emits boneClicked(name) when a bone is clicked."""
    boneClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(3.2, 3.6))
        super().__init__(self.fig)
        self.setParent(parent)
        # Small on purpose: a cramped column shrinks the whole figure (see _fit)
        # instead of cutting the legs off; it grows with the column.
        self.setMinimumHeight(140)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._selected = set()
        self._confidence = {}
        self.lines = {}          # bone -> Line2D (the bone itself)
        self.halos = {}          # bone -> Line2D (selection halo underneath)
        self._build()
        self.mpl_connect('pick_event', self._on_pick)
        self.mpl_connect('motion_notify_event', self._on_motion)
        self.mpl_connect('resize_event', lambda e: self._fit())

    # --- public ---------------------------------------------------------

    def set_selection(self, selected):
        self._selected = set(selected)
        self._restyle()

    def set_confidence(self, confidence):
        """{bone: mean tracking confidence, or None}."""
        self._confidence = dict(confidence)
        self._restyle()

    def state(self, bone):
        return selection_state(bone, self._selected)

    def set_summary(self, text):
        """One line under the figure saying what is picked."""
        self._summary.set_text(text)
        self.draw_idle()

    def summary_text(self):
        return self._summary.get_text()

    # --- drawing --------------------------------------------------------

    def _build(self):
        ax = self.ax = self.fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_aspect('equal')
        for a, b in _CONNECTORS:
            ax.plot(*zip(_P[a], _P[b]), color=design.BORDER, lw=3, solid_capstyle='round',
                    zorder=1)
        for name, (label, a, b, _, _) in BONES.items():
            xy = list(zip(_P[a], _P[b]))
            halo, = ax.plot(*xy, color=_HALO, lw=15, alpha=0.0, solid_capstyle='round',
                            zorder=2)
            line, = ax.plot(*xy, color=_UNKNOWN, lw=6, solid_capstyle='round', zorder=3,
                            picker=True, pickradius=9)
            line.bone = name
            self.lines[name], self.halos[name] = line, halo
        self._head = ax.scatter([_P[15][0]], [_P[15][1] + 0.04], s=300, facecolor='none',
                                edgecolor=design.BORDER, lw=2, zorder=1)
        ax.text(-0.42, 1.62, "R", ha='center', fontsize=11, color=design.TEXT_MUTED,
                fontweight='bold')
        ax.text(0.42, 1.62, "L", ha='center', fontsize=11, color=design.TEXT_MUTED,
                fontweight='bold')
        self._summary = self.fig.text(0.5, 0.015, "", ha='center', va='bottom',
                                      fontsize=8.5, color=design.TEXT_MUTED, wrap=True)
        self._fit()
        self._restyle()

    def _fit(self):
        """Keep the whole figure in view whatever the widget's aspect ratio."""
        w, h = self.fig.get_size_inches()
        x0, x1, y0, y1 = -0.52, 0.52, -0.16, 1.84
        need = (x1 - x0) / (y1 - y0)
        if w / h > need:            # wide: pad x
            pad = ((y1 - y0) * w / h - (x1 - x0)) / 2
            x0, x1 = x0 - pad, x1 + pad
        else:                       # tall: pad y
            pad = ((x1 - x0) * h / w - (y1 - y0)) / 2
            y0, y1 = y0 - pad, y1 + pad
        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)

    def _restyle(self):
        for name, line in self.lines.items():
            line.set_color(_confidence_colour(self._confidence.get(name)))
            halo = self.halos[name]
            st = selection_state(name, self._selected)
            halo.set_alpha({'all': 0.7, 'some': 0.5, 'none': 0.0}[st])
            halo.set_linestyle('--' if st == 'some' else '-')
        self.draw_idle()

    # --- interaction ----------------------------------------------------

    def _on_pick(self, event):
        bone = getattr(event.artist, 'bone', None)
        if bone is not None and getattr(event.mouseevent, 'button', 1) == 1:
            self.boneClicked.emit(bone)

    def _on_motion(self, event):
        hit = None
        if event.inaxes is self.ax:
            for name, line in self.lines.items():
                if line.contains(event)[0]:
                    hit = name
                    break
        if hit is None:
            self.setToolTip("Click a bone to predict it")
            return
        label, _, _, _, targets = BONES[hit]
        conf = self._confidence.get(hit)
        lines = [f"<b>{label}</b>"] + [f"&nbsp;&nbsp;{t}" for t in targets]
        if conf is not None and np.isfinite(conf):
            lines.append(f"tracking confidence {conf:.2f}")
        lines.append({'all': "picked: click to remove",
                      'some': "partly picked: click to pick all",
                      'none': "click to predict"}[self.state(hit)])
        self.setToolTip("<br>".join(lines))
