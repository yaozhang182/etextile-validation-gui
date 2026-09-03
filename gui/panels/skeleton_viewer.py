"""
Tab 2: Skeleton Viewer — 3D skeleton wireframe + video frame, timeline slider.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton,
    QGroupBox, QComboBox, QDialog, QTextEdit, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QTimer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.state import get_subjects
from gui.plot_style import style_3d_axes
from gui import help as help_ui, icons

# SMPLH bone connections for skeleton wireframe
SKELETON_BONES = [
    # Spine
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # Left leg
    (0, 1), (1, 4), (4, 7), (7, 10),
    # Right leg
    (0, 2), (2, 5), (5, 8), (8, 11),
    # Left arm
    (12, 13), (13, 16), (16, 18), (18, 20),
    # Right arm
    (12, 14), (14, 17), (17, 19), (19, 21),
]

JOINT_NAMES_22 = [
    "pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee",
    "spine2", "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot",
    "neck", "L_collar", "R_collar", "head",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
]


class SkeletonViewerPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._playing = False
        self._cap = None
        self._cap_path = None
        self._timer = QTimer()
        self._timer.timeout.connect(self._advance_frame)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Data selector
        top = QHBoxLayout()
        top.addWidget(QLabel("Data:"))
        self.data_combo = QComboBox()
        self.data_combo.addItems(["Train", "Test"])
        self.data_combo.currentIndexChanged.connect(self._on_split_changed)
        top.addWidget(self.data_combo)

        top.addWidget(QLabel("Subject:"))
        self.subject_combo = QComboBox()
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        top.addWidget(self.subject_combo)
        top.addStretch()

        self.quality_btn = icons.decorate(
            QPushButton("Tracking Quality..."), 'triangle-alert')
        self.quality_btn.setToolTip(
            "How confident MediaPipe is in the body landmarks it produced, and "
            "which joint angles that makes unreliable."
        )
        self.quality_btn.clicked.connect(self._show_quality)
        top.addWidget(self.quality_btn)
        help_ui.attach(top, 'joint_colours')
        layout.addLayout(top)

        # Matplotlib figure with two subplots: video frame + 3D skeleton
        self.fig = Figure(figsize=(12, 5))
        self.ax_video = self.fig.add_subplot(121)
        self.ax_skeleton = self.fig.add_subplot(122, projection='3d')
        style_3d_axes(self.ax_skeleton)   # mplot3d ignores rcParams
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # Controls
        ctrl = QHBoxLayout()
        self.play_btn = icons.decorate(QPushButton("Play"), 'play')
        self.play_btn.clicked.connect(self._toggle_play)
        ctrl.addWidget(self.play_btn)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self._update_plot)
        ctrl.addWidget(self.frame_slider)

        self.frame_label = QLabel("Frame: 0 / 0")
        ctrl.addWidget(self.frame_label)
        layout.addLayout(ctrl)

        self.info_label = QLabel("Load data in Tab 1 first.")
        layout.addWidget(self.info_label)

    @property
    def _split(self):
        return 'train' if self.data_combo.currentIndex() == 0 else 'test'

    def _current_subject(self):
        subjects = get_subjects(self.state, self._split)
        idx = self.subject_combo.currentIndex()
        if 0 <= idx < len(subjects):
            return subjects[idx]
        return None

    def _on_split_changed(self):
        self._populate_subjects()

    def _on_subject_changed(self):
        if getattr(self, '_cap', None) is not None:
            self._cap.release()
            self._cap = None
            self._cap_path = None
        subject = self._current_subject()
        if subject is not None and subject.get('skeleton') is not None:
            self.frame_slider.setMaximum(max(subject['skeleton']['total_frames'] - 1, 0))
        self.frame_slider.setValue(0)
        self._update_plot()

    def _populate_subjects(self):
        """Rebuild the subject dropdown for the selected split."""
        subjects = get_subjects(self.state, self._split)
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        self.subject_combo.addItems([s['id'] for s in subjects])
        self.subject_combo.blockSignals(False)
        self._on_subject_changed()

    def refresh(self):
        """Called when tab becomes active."""
        self._populate_subjects()

        subject = self._current_subject()
        skeleton = subject.get('skeleton') if subject else None
        if skeleton is not None:
            total = skeleton['total_frames']
            self.info_label.setText(
                f"{self._split.capitalize()} / {subject['id']}: {total} frames, "
                f"{skeleton['fps']:.1f} fps, "
                f"{skeleton['valid_frames']}/{total} poses detected"
            )
        elif subject is not None:
            self.info_label.setText(
                f"{subject['id']} has no skeleton yet "
                f"(status: {subject['status']}). Run extraction in Tab 1."
            )
        else:
            self.info_label.setText(
                f"No {self._split} subjects. Add one in Tab 1 first."
            )

    def _show_quality(self):
        """Plain-language summary of how far the pose tracking can be trusted."""
        from core.confidence import tracking_report, LOW_CONFIDENCE, is_synthesised

        subject = self._current_subject()
        if subject is None:
            return
        report = tracking_report(subject.get('skeleton'))

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Tracking Quality — {subject['id']}")
        dlg.setMinimumSize(620, 460)
        box = QVBoxLayout(dlg)

        text = QTextEdit()
        text.setReadOnly(True)

        if not report['available']:
            text.setPlainText(report['reason'])
        else:
            colour = {'good': '#2e7d32', 'fair': '#ef6c00',
                      'poor': '#c62828'}.get(report['label'], '#555')
            o = report['overall']
            html = [
                f"<h3>Overall: <span style='color:{colour}'>{report['label']}</span></h3>",
                f"<p>Mean landmark confidence <b>{o['mean']:.2f}</b>, "
                f"worst frame <b>{o['min']:.2f}</b>. "
                f"<b>{o['frac_low'] * 100:.1f}%</b> of frames fall below "
                f"{LOW_CONFIDENCE:.1f}, where MediaPipe considers a landmark "
                f"occluded or inaccurate.</p>",
                "<p><i>Confidence is MediaPipe's own per-landmark visibility "
                "score. It reflects occlusion, not absolute angular accuracy — "
                "a well-tracked pose can still differ from marker-based "
                "capture.</i></p>",
                "<h4>Least reliable body landmarks</h4><ul>",
            ]
            for name, mean in report['worst_landmarks']:
                html.append(f"<li>{name}: {mean:.2f}</li>")
            html.append("</ul>")

            html.append("<h4>Joint angle channels ranked by confidence</h4>"
                        "<table cellpadding='3'><tr><th align='left'>Channel</th>"
                        "<th>Mean</th><th>% low</th><th align='left'>Note</th></tr>")
            ranked = sorted(report['per_angle'].items(), key=lambda kv: kv[1]['mean'])
            for name, s in ranked:
                note = ("depends on the synthesised pelvis/spine"
                        if is_synthesised(name) else "")
                row_colour = '#c62828' if s['mean'] < LOW_CONFIDENCE else '#000'
                html.append(
                    f"<tr><td style='color:{row_colour}'>{name}</td>"
                    f"<td align='right'>{s['mean']:.2f}</td>"
                    f"<td align='right'>{s['frac_low'] * 100:.0f}%</td>"
                    f"<td>{note}</td></tr>"
                )
            html.append("</table>")
            html.append(
                "<p><i>MediaPipe provides no pelvis landmark, so the pelvis and "
                "spine are reconstructed geometrically. Channels marked above "
                "inherit that assumption on top of their visibility score.</i></p>"
            )
            text.setHtml(''.join(html))

        box.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        box.addWidget(buttons)
        dlg.exec()

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._timer.stop()
            self.play_btn.setText("Play")
            icons.decorate(self.play_btn, 'play')
        else:
            self._playing = True
            self.play_btn.setText("Pause")
            icons.decorate(self.play_btn, 'pause')
            subject = self._current_subject()
            skeleton = subject.get('skeleton') if subject else None
            fps = skeleton['fps'] if skeleton else 25
            self._timer.start(int(1000 / (fps or 25)))

    def _advance_frame(self):
        val = self.frame_slider.value()
        mx = self.frame_slider.maximum()
        if val < mx:
            self.frame_slider.setValue(val + 1)
        else:
            self._toggle_play()

    def _draw_video_frame(self, subject, frame_idx):
        """
        Show one video frame, reusing the open file handle.

        Reopening a multi-hundred-megabyte MP4 on every slider step makes
        scrubbing crawl, which reads as the window hanging.
        """
        self.ax_video.clear()
        video_path = subject.get('video_path') if subject else None
        if video_path:
            try:
                import cv2
                if getattr(self, '_cap_path', None) != video_path:
                    if getattr(self, '_cap', None) is not None:
                        self._cap.release()
                    self._cap = cv2.VideoCapture(video_path)
                    self._cap_path = video_path
                if self._cap is not None and self._cap.isOpened():
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ok, frame = self._cap.read()
                    if ok:
                        self.ax_video.imshow(frame[:, :, ::-1])
            except Exception:
                pass    # a missing or unreadable video must not break the tab
        self.ax_video.set_title(f"Video Frame {frame_idx}")
        self.ax_video.axis('off')

    def _joint_confidence(self, skeleton):
        """(T, 22) confidence for the current skeleton, cached per subject."""
        visibility = skeleton.get('visibility') if skeleton else None
        if visibility is None:
            return None
        key = id(skeleton)
        if getattr(self, '_conf_key', None) != key:
            from core.confidence import mediapipe_visibility_to_smplh22
            self._conf_key = key
            self._conf_cache = mediapipe_visibility_to_smplh22(visibility)
        return self._conf_cache

    def _update_plot(self):
        subject = self._current_subject()
        skeleton = subject.get('skeleton') if subject else None
        if skeleton is None:
            return

        frame_idx = self.frame_slider.value()
        total = skeleton.get('total_frames', 0)
        self.frame_label.setText(f"Frame: {frame_idx} / {total}")

        # A subject restored from a cache written before 3D joints were stored
        # has angles but no skeleton geometry. Say so instead of raising — an
        # exception here propagates out of a Qt slot and kills the app.
        smplh = skeleton.get('smplh_joints')
        if smplh is None:
            self._draw_video_frame(subject, frame_idx)
            self.ax_skeleton.clear()
            self.ax_skeleton.set_axis_off()
            self.ax_skeleton.text2D(
                0.5, 0.5,
                "3D skeleton not in this cache.\n\n"
                "Press 'Extract Skeleton & Compute Angles' in Tab 1\n"
                "to compute it. Joint angles, training and results\n"
                "all work without it.",
                transform=self.ax_skeleton.transAxes,
                ha='center', va='center', fontsize=10, color='#666',
            )
            self.fig.tight_layout()
            self.canvas.draw()
            return

        self._draw_video_frame(subject, frame_idx)

        # Draw 3D skeleton
        # SMPL convention: X=left/right, Y=up, Z=forward
        # matplotlib 3D:   X=right,      Z=up, Y=depth
        # Swap data Y↔Z so the person stands upright in the plot.
        self.ax_skeleton.clear()
        style_3d_axes(self.ax_skeleton)
        if frame_idx < len(smplh):
            joints = smplh[frame_idx]  # (22, 3)
            if not np.any(np.isnan(joints)):
                xs = joints[:, 0]   # X → plot X (left/right)
                ys = joints[:, 2]   # Z → plot Y (forward/depth)
                zs = joints[:, 1]   # Y → plot Z (up)

                # Colour each joint by MediaPipe's confidence for that frame, so
                # occluded limbs are obvious while scrubbing.
                conf = self._joint_confidence(skeleton)
                if conf is not None and frame_idx < len(conf):
                    self.ax_skeleton.scatter(
                        xs, ys, zs, c=conf[frame_idx], cmap='RdYlGn',
                        vmin=0.0, vmax=1.0, s=28, depthshade=True,
                        edgecolors='black', linewidths=0.3,
                    )
                else:
                    self.ax_skeleton.scatter(xs, ys, zs, c='red', s=20, depthshade=True)

                for i, name in enumerate(JOINT_NAMES_22):
                    self.ax_skeleton.text(xs[i], ys[i], zs[i], name, fontsize=5, alpha=0.7)

                for a, b in SKELETON_BONES:
                    if a < len(joints) and b < len(joints):
                        self.ax_skeleton.plot(
                            [xs[a], xs[b]], [ys[a], ys[b]], [zs[a], zs[b]],
                            'b-', linewidth=1.5
                        )

                # Set consistent axes
                center = np.array([xs.mean(), ys.mean(), zs.mean()])
                all_coords = np.column_stack([xs, ys, zs])
                extent = max(all_coords.max(axis=0) - all_coords.min(axis=0)) * 0.6
                self.ax_skeleton.set_xlim(center[0] - extent, center[0] + extent)
                self.ax_skeleton.set_ylim(center[1] - extent, center[1] + extent)
                self.ax_skeleton.set_zlim(center[2] - extent, center[2] + extent)

        self.ax_skeleton.set_xlabel('X')
        self.ax_skeleton.set_ylabel('Z (forward)')
        self.ax_skeleton.set_zlabel('Y (up)')
        self.ax_skeleton.view_init(elev=20, azim=-90)
        title = "Skeleton (SMPLH 22)"
        conf = self._joint_confidence(skeleton)
        if conf is not None and frame_idx < len(conf):
            title += f"\nframe confidence: worst {conf[frame_idx].min():.2f} (green = high)"
        self.ax_skeleton.set_title(title, fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()
