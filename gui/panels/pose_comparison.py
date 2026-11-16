"""
Tab 5: Pose Comparison — the tracked skeleton next to the same skeleton with
the predicted bones redrawn, over the test recording, with its video.

Left is the ground truth from the video. Right copies it, except that every
bone with a direction target is redrawn where the model said it pointed
(core.bones.reconstruct). One slider moves the video and both skeletons, and
turning either skeleton turns the other.
"""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.bones import BONES, predicted_directions, reconstruct
from core.targets import angle_between, is_vector, target_slices
from gui import design, help as help_ui
from gui.plot_style import style_3d_axes
from gui.panels.skeleton_viewer import SKELETON_BONES
from gui.panels.video_scrubber import VideoScrubber
from gui.state import ready_subjects

PRED = design.DANGER
TRUE = design.ACCENT
GHOST = '#9aa5b3'

#: A frame without a prediction of its own borrows the nearest one this many
#: frames away at most (aligning to sensor samples predicts fewer frames).
_NEAREST = 3


def _xyz(points):
    """humanml3d (X the person's left, Y up, Z forward) -> plot axes.
    A proper rotation (no mirror), so turning the view behaves like the body."""
    p = np.atleast_2d(points)
    return -p[:, 0], p[:, 2], p[:, 1]


class PoseComparisonPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._data = None          # everything for the shown subject
        self._series_id = None
        self._init_ui()

    # --- layout -----------------------------------------------------------

    def _init_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Test subject:"))
        self.subject_combo = QComboBox()
        self.subject_combo.currentIndexChanged.connect(self._load_subject)
        top.addWidget(self.subject_combo)
        self.video_toggle = QCheckBox("Show video")
        self.video_toggle.setChecked(True)
        self.video_toggle.toggled.connect(self._on_video_toggled)
        top.addWidget(self.video_toggle)
        self.ghost_toggle = QCheckBox("Show true position of predicted bones")
        self.ghost_toggle.setChecked(True)
        self.ghost_toggle.toggled.connect(lambda _: self._draw_frame(self.scrubber.frame()))
        top.addWidget(self.ghost_toggle)
        top.addStretch()
        help_ui.attach(top, 'pose_comparison')
        layout.addLayout(top)

        self.info_label = QLabel("Train a model in Tab 4 first.")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        # The scrubber owns decoding and real-time playback; its picture goes
        # beside the skeletons and its controls underneath all three, so one
        # slider visibly drives everything.
        self.scrubber = VideoScrubber(video_height=380)
        self.scrubber.frameChanged.connect(self._on_frame_changed)
        video = self.scrubber.video_label
        self.scrubber.layout().removeWidget(video)
        video.setMinimumWidth(260)

        row = QHBoxLayout()
        row.addWidget(video, stretch=3)
        self.fig = Figure(figsize=(9, 4.6))
        # each skeleton gets exactly half the canvas, edge to edge: mplot3d's
        # default margins left them small in a sea of white
        self.ax_true = self.fig.add_axes([0.0, 0.0, 0.5, 0.93], projection='3d')
        self.ax_pred = self.fig.add_axes([0.5, 0.0, 0.5, 0.93], projection='3d')
        # titles above the axes, where no bone can reach them
        self.fig.text(0.25, 0.965, "Tracked from the video (ground truth)",
                      ha='center', va='center', fontsize=10, color=design.TEXT)
        self.fig.text(0.75, 0.965, "Predicted bones redrawn (red)",
                      ha='center', va='center', fontsize=10, color=design.TEXT)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(380)
        # The rendered image can come out a pixel narrower than the widget;
        # fill the widget first so that column is blank, not a stale smear.
        self.canvas.setAutoFillBackground(True)
        self.canvas.setStyleSheet(f"background-color: {design.SURFACE};")
        row.addWidget(self.canvas, stretch=6)
        layout.addLayout(row, stretch=1)
        layout.addWidget(self.scrubber)

        self.values_label = QLabel("")
        self.values_label.setWordWrap(True)
        self.values_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.values_label)

        self.canvas.mpl_connect('motion_notify_event', self._sync_views)
        self.canvas.mpl_connect('button_release_event', self._sync_views)
        self._reset_axes()

    def _reset_axes(self):
        for ax in (self.ax_true, self.ax_pred):
            ax.clear()
            style_3d_axes(ax)
            ax.set_axis_off()            # the body is the reference; boxes only clutter
            ax.set_box_aspect((1, 1, 1), zoom=1.7)
            ax.view_init(elev=10, azim=70)

    # --- tab life cycle ------------------------------------------------------

    def refresh(self):
        series = self.state.get('predictions_per_subject') or {}
        if not series or self.state.get('metrics') is None:
            self._clear("Train a model in Tab 4 first. This tab then shows the "
                        "test recording with the predicted bones drawn into "
                        "the skeleton.")
            return
        if id(series) == self._series_id and self._data is not None:
            return
        self._series_id = id(series)
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        self.subject_combo.addItems(list(series))
        self.subject_combo.blockSignals(False)
        self._load_subject()

    def reset(self):
        self._series_id = None
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        self.subject_combo.blockSignals(False)
        self._clear("Train a model in Tab 4 first.")

    def hideEvent(self, event):
        scrubber = getattr(self, 'scrubber', None)
        if scrubber is not None:
            scrubber.stop()
        super().hideEvent(event)

    def _clear(self, message):
        self._data = None
        self.scrubber.clear()
        self._reset_axes()
        self.canvas.draw_idle()
        self.values_label.setText("")
        self.info_label.setText(message)

    # --- data ---------------------------------------------------------------

    def _load_subject(self):
        sid = self.subject_combo.currentText()
        series = (self.state.get('predictions_per_subject') or {}).get(sid)
        subject = next((s for s in ready_subjects(self.state, 'test') if s['id'] == sid), None)
        if series is None or subject is None:
            self._clear("No predictions for this subject.")
            return
        skeleton = subject.get('skeleton') or {}
        joints = skeleton.get('smplh_joints')
        if joints is None:
            self._clear(f"{sid}: the 3D skeleton is not in this cache. Press "
                        "'Extract Skeleton & Compute Angles' in Tab 1 to compute it.")
            return
        joints = np.asarray(joints, dtype=float)
        names = list((self.state.get('config') or {}).get('target_names')
                     or self.state.get('selected_angles') or [])
        truth, pred = np.asarray(series['truth']), np.asarray(series['pred'])
        frames = np.asarray(series.get('frames', np.full(len(pred), -1)), dtype=int)

        ok = (frames >= 0) & (frames < len(joints))
        ok[ok] = np.isfinite(joints[frames[ok]]).all(axis=(1, 2))
        idx = np.flatnonzero(ok)
        if not len(idx):
            self._clear(f"{sid}: no prediction could be matched to a video frame.")
            return

        dirs = predicted_directions(pred, names)
        recon = reconstruct(joints[frames[idx]], {b: d[idx] for b, d in dirs.items()})

        # each frame -> its prediction (index into idx), or the nearest one
        n_frames = len(joints)
        owner = np.full(n_frames, -1)
        owner[frames[idx]] = np.arange(len(idx))
        have = np.flatnonzero(owner >= 0)
        near = np.clip(np.searchsorted(have, np.arange(n_frames)), 1, len(have) - 1) \
            if len(have) > 1 else np.zeros(n_frames, dtype=int)
        if len(have) > 1:
            left, right = have[near - 1], have[near]
            pick = np.where(np.arange(n_frames) - left <= right - np.arange(n_frames), left, right)
        else:
            pick = np.full(n_frames, have[0])
        fill = (owner < 0) & (np.abs(pick - np.arange(n_frames)) <= _NEAREST)
        owner[fill] = owner[pick[fill]]

        # per-sample readout: direction error, or predicted vs true value
        readout = []
        for name, sl in target_slices(names):
            if is_vector(name):
                readout.append((name, 'dir', angle_between(truth[idx, sl], pred[idx, sl])))
            else:
                readout.append((name, 'val', (pred[idx, sl.start], truth[idx, sl.start])))

        # Frame the body, not the pelvis: a seated person's pelvis sits low,
        # and an arm raised overhead must stay in view. The box is the space
        # the body (and the redrawn bones) take up over the whole recording,
        # held fixed so the view does not breathe from frame to frame.
        tracked = joints[np.isfinite(joints).all(axis=(1, 2))]
        both = np.concatenate([tracked, recon]) if len(tracked) else recon
        rel = both - both[:, :1]                         # relative to the pelvis
        lo, hi = np.nanpercentile(rel.min(axis=1), 2, axis=0), \
            np.nanpercentile(rel.max(axis=1), 98, axis=0)
        offset = (lo + hi) / 2
        half = 0.55 * float(np.max(hi - lo))
        self._data = {
            'sid': sid, 'joints': joints, 'recon': recon, 'owner': owner,
            'sample_frames': frames[idx], 'bones': set(dirs), 'readout': readout,
            'offset': offset,
            'extent': half if np.isfinite(half) and half > 0.2 else 1.0,
        }
        self._build_artists()
        n_dir = len(dirs)
        self.info_label.setText(
            f"{sid}: {len(idx)} predicted frames of {n_frames}. "
            + (f"{n_dir} bone(s) redrawn from their predicted direction."
               if n_dir else
               "No direction target was trained, so no bone can be redrawn; the "
               "predicted angles are listed below the video."))
        fps = skeleton.get('fps') or 30.0
        start = int(frames[idx[0]])
        self.scrubber.set_source(subject.get('video_path'), n_frames, fps, frame=start)
        self._draw_frame(start)

    # --- drawing ------------------------------------------------------------

    def _build_artists(self):
        self._reset_axes()
        d = self._data
        predicted_pairs = {(BONES[b][1], BONES[b][2]) for b in d['bones']}
        art = {'true': {}, 'pred': {}, 'ghost': {}, 'err': {}}
        for a, b in SKELETON_BONES:
            art['true'][(a, b)], = self.ax_true.plot([], [], [], color=TRUE, lw=2.2)
            hot = (a, b) in predicted_pairs
            art['pred'][(a, b)], = self.ax_pred.plot(
                [], [], [], color=PRED if hot else TRUE, lw=3.4 if hot else 2.2)
        for bone in d['bones']:
            a, b = BONES[bone][1], BONES[bone][2]
            art['ghost'][bone], = self.ax_pred.plot([], [], [], color=GHOST, lw=2.2, ls='--')
            art['err'][bone] = self.ax_pred.text(
                0, 0, 0, "", color=PRED, fontsize=9, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85))
        art['pts_true'] = self.ax_true.plot([], [], [], 'o', ms=3, color=TRUE)[0]
        art['pts_pred'] = self.ax_pred.plot([], [], [], 'o', ms=3, color=TRUE)[0]
        art['none'] = self.ax_pred.text2D(0.5, 0.04, "", transform=self.ax_pred.transAxes,
                                          ha='center', fontsize=9, color=design.TEXT_MUTED)
        d['art'] = art

    def _on_frame_changed(self, frame):
        self._draw_frame(frame)

    def _draw_frame(self, frame):
        d = self._data
        if d is None or not (0 <= frame < len(d['joints'])):
            return
        art = d['art']
        true = d['joints'][frame]
        s = d['owner'][frame]
        have_true = np.isfinite(true).all()
        recon = d['recon'][s] if s >= 0 else (true if have_true else None)

        pelvis = true[0] if have_true else (recon[0] if recon is not None else np.zeros(3))
        centre = pelvis + d['offset']
        e = d['extent']
        cx, cy, cz = _xyz(centre)
        for ax in (self.ax_true, self.ax_pred):
            ax.set_xlim(cx[0] - e, cx[0] + e)
            ax.set_ylim(cy[0] - e, cy[0] + e)
            ax.set_zlim(cz[0] - e, cz[0] + e)

        def put(lines, points, joints):
            for (a, b), line in lines.items():
                if joints is None:
                    line.set_data_3d([], [], [])
                else:
                    x, y, z = _xyz(joints[[a, b]])
                    line.set_data_3d(x, y, z)
            if joints is None:
                points.set_data_3d([], [], [])
            else:
                points.set_data_3d(*_xyz(joints))

        put(art['true'], art['pts_true'], true if have_true else None)
        put(art['pred'], art['pts_pred'], recon)

        ghost_on = self.ghost_toggle.isChecked() and s >= 0 and have_true
        errs = {name: vals for name, kind, vals in d['readout'] if kind == 'dir'}
        for bone, line in art['ghost'].items():
            a, b = BONES[bone][1], BONES[bone][2]
            text = art['err'][bone]
            if s < 0 or recon is None:
                line.set_data_3d([], [], [])
                text.set_text("")
                continue
            if ghost_on:
                line.set_data_3d(*_xyz(true[[a, b]]))   # where the bone really was
            else:
                line.set_data_3d([], [], [])
            # just past the end of the bone, along it, so the number never
            # sits on the bone or on the joint where two labels meet
            vec = recon[b] - recon[a]
            tip = recon[b] + 0.35 * vec
            x, y, z = _xyz(tip)
            text.set_position((x[0], y[0]))
            text.set_3d_properties(z[0], None)
            text.set_text(f"{errs[BONES[bone][3]][s]:.0f}°")
        art['none'].set_text("" if s >= 0 else
                             "No prediction at this frame: outside the time both "
                             "streams cover, or within the first sequence-length "
                             "frames.")
        self.canvas.draw_idle()
        self._show_values(s)

    def _show_values(self, s):
        d = self._data
        if s < 0:
            self.values_label.setText("")
            return
        parts = []
        for name, kind, vals in d['readout']:
            if kind == 'dir':
                parts.append(f"{name} <b style='color:{PRED}'>off by {vals[s]:.1f}°</b>")
            else:
                p, t = vals
                parts.append(f"{name} <b style='color:{PRED}'>{p[s]:.1f}</b> "
                             f"predicted, <b>{t[s]:.1f}</b> true")
        self.values_label.setText("&nbsp;&nbsp;·&nbsp;&nbsp;".join(parts))

    # --- interaction --------------------------------------------------------

    def _on_video_toggled(self, shown):
        self.scrubber.video_label.setVisible(shown)

    def _sync_views(self, event):
        """Turning one skeleton turns the other."""
        if event.inaxes not in (self.ax_true, self.ax_pred):
            return
        if event.name == 'motion_notify_event' and not event.button:
            return
        src = event.inaxes
        dst = self.ax_pred if src is self.ax_true else self.ax_true
        if (dst.elev, dst.azim) != (src.elev, src.azim):
            dst.view_init(elev=src.elev, azim=src.azim)
            self.canvas.draw_idle()
