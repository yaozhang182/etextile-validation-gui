"""
Tab 3: Angle & Sensor Comparison — Select angles/sensors, overlay time-series plot.
The selection here determines what goes into ML training (Tab 4).
"""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QCheckBox, QScrollArea, QComboBox, QPushButton,
)
from PyQt6.QtCore import Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gui.state import (
    get_subjects, common_sensor_columns, inconsistent_sensor_subjects,
)
from gui.theme import set_style_property
from gui import help as help_ui

# Angle grouping by body part
ANGLE_GROUPS = {
    'Pelvis': ['pelvis_flexion', 'pelvis_adduction', 'pelvis_rotation'],
    'Lumbar': ['lumbar_extension', 'lumbar_bending', 'lumbar_rotation'],
    'Hip L': ['left_hip_flexion', 'left_hip_abduction', 'left_hip_rotation'],
    'Hip R': ['right_hip_flexion', 'right_hip_abduction', 'right_hip_rotation'],
    'Knee': ['left_knee_flexion', 'right_knee_flexion'],
    'Ankle': ['left_ankle_flexion', 'right_ankle_flexion'],
    'Shoulder L': ['left_shoulder_flexion', 'left_shoulder_abduction', 'left_shoulder_rotation'],
    'Shoulder R': ['right_shoulder_flexion', 'right_shoulder_abduction', 'right_shoulder_rotation'],
    'Elbow': ['left_elbow_flexion', 'right_elbow_flexion'],
    'Neck': ['neck_flexion', 'neck_bending'],
}


def _pin_scroll_content(scroll):
    """
    Stop a scroll area from squeezing its contents instead of scrolling.

    QScrollArea with setWidgetResizable(True) sets the inner widget's geometry
    directly, and QWidget::setGeometry clamps only to an *explicit* minimum —
    minimumSizeHint is advisory. So a list rebuilt while the tab is hidden keeps
    the widget at the old (viewport) height, and its layout then compresses the
    rows below their own minimum: ten 21px checkboxes rendered 11px tall, text
    clipped to illegible stubs and nothing scrollable to reach the rest.

    Pinning the explicit minimum to what the layout actually needs makes the
    scrollbar appear, which is what a scroll area is for.

    The rows must be made visible before they are measured, or the number this
    pins is wrong in the one case that matters. A layout leaves hidden widgets
    out of its minimum, and rows added while their tab is not current stay
    hidden until the event loop shows them — so measuring there yields the
    height of an empty column, and pinning that is worse than not pinning at
    all, because it survives as an explicit minimum once the rows do appear.
    """
    inner = scroll.widget()
    layout = inner.layout()

    inner.ensurePolished()
    for i in range(layout.count()):
        child = layout.itemAt(i).widget()
        if child is not None:
            child.setVisible(True)
            child.ensurePolished()

    layout.invalidate()
    layout.activate()
    inner.setMinimumHeight(layout.minimumSize().height())


def _frame_epochs(video_df, n_frames):
    """
    Epoch timestamp for each extracted video frame.

    The timestamps CSV is the authority on when each frame was taken — not the
    frame index divided by a nominal rate, which assumes a perfectly constant
    capture rate the camera does not deliver.

    The CSV routinely holds one row fewer than the video has decoded frames, so
    the trailing frame is extrapolated at the measured mean period rather than
    left clamped on the last timestamp.
    """
    from core.data_alignment import find_timestamp_column

    ts_col = find_timestamp_column(video_df)
    if ts_col is None:
        return None

    epoch = np.asarray(video_df[ts_col].values, dtype=float)
    if 'FrameIndex' in video_df.columns:
        index = np.asarray(video_df['FrameIndex'].values, dtype=float)
    else:
        index = np.arange(len(epoch), dtype=float)
    if len(epoch) < 2:
        return None

    want = np.arange(n_frames, dtype=float)
    out = np.interp(want, index, epoch)

    # np.interp clamps beyond the ends; continue at the measured rate instead.
    period = (epoch[-1] - epoch[0]) / (index[-1] - index[0])
    after = want > index[-1]
    out[after] = epoch[-1] + (want[after] - index[-1]) * period
    before = want < index[0]
    out[before] = epoch[0] - (index[0] - want[before]) * period
    return out


def _shared_clock(subject, n_angle_frames):
    """
    Put the joint angles and the sensor stream on one time axis.

    Both recordings are stamped with the same Unix clock, which is the whole
    premise of the tool — but each stream starts when its own recorder was
    started, and those moments differ by seconds. Plotting each as "seconds
    since my own first sample" therefore slides one panel against the other by
    exactly that difference, and a reader comparing the two by eye sees the
    sensor responding before or after a movement that it was in fact tracking.

    Returns (angle_t, sensor_t, aligned): seconds from the first sample of
    whichever stream began first, so a vertical line means one instant in both
    panels. ``aligned`` is true only when both streams supplied real timestamps
    — the caller must not claim the panels are comparable otherwise.
    """
    from core.data_alignment import find_timestamp_column

    video_df = subject.get('video_df') if subject else None
    sensor_df = subject.get('sensor_df') if subject else None

    angle_epoch = None
    if video_df is not None and n_angle_frames:
        angle_epoch = _frame_epochs(video_df, n_angle_frames)

    sensor_epoch = None
    if sensor_df is not None:
        ts_col = find_timestamp_column(sensor_df)
        if ts_col is not None:
            sensor_epoch = np.asarray(sensor_df[ts_col].values, dtype=float)

    starts = [e[0] for e in (angle_epoch, sensor_epoch) if e is not None and len(e)]
    if not starts:
        return None, None, False

    origin = min(starts)
    aligned = angle_epoch is not None and sensor_epoch is not None
    return (None if angle_epoch is None else angle_epoch - origin,
            None if sensor_epoch is None else sensor_epoch - origin,
            aligned)


def _clear_layout(layout):
    """
    Empty a layout, removing its widgets from the display immediately.

    deleteLater() alone is not enough: it defers destruction to the event loop,
    and a widget taken out of a layout keeps its parent and its last geometry, so
    it goes on painting until then. The result is old and new labels drawn on top
    of each other. setParent(None) detaches and hides it at once; deleteLater
    then frees it. Nested layouts are handled too, or the buttons inside them
    would be orphaned the same way.
    """
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()
        else:
            child = item.layout()
            if child is not None:
                _clear_layout(child)
                child.setParent(None)


class AngleComparisonPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.angle_checkboxes = {}
        self.sensor_checkboxes = {}
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)

        # --- Left: Selection panel ---
        left_panel = QVBoxLayout()

        # Data selector
        dh = QHBoxLayout()
        dh.addWidget(QLabel("Data:"))
        self.data_combo = QComboBox()
        self.data_combo.addItems(["Train", "Test"])
        self.data_combo.currentIndexChanged.connect(self._on_split_changed)
        dh.addWidget(self.data_combo)
        left_panel.addLayout(dh)

        sh = QHBoxLayout()
        sh.addWidget(QLabel("Subject:"))
        self.subject_combo = QComboBox()
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        sh.addWidget(self.subject_combo)
        left_panel.addLayout(sh)

        # Joint angle checkboxes
        angle_group = QGroupBox("Joint Angles (targets)")
        angle_header = QVBoxLayout()
        angle_header.addWidget(help_ui.labelled(
            "What the model will predict", 'angle_targets'))
        angle_header.addWidget(help_ui.labelled(
            "(0.00) = tracking confidence", 'angle_confidence'))
        angle_scroll = self.angle_scroll = QScrollArea()
        angle_scroll.setWidgetResizable(True)
        # A QScrollArea's minimum size hint is tiny, so in a cramped column it is
        # the first thing Qt shrinks — down to nothing, leaving only its
        # scrollbar and no way to pick a target. Give it a floor.
        angle_scroll.setMinimumHeight(200)
        angle_inner = QWidget()
        angle_layout = QVBoxLayout(angle_inner)

        for group_name, angle_names in ANGLE_GROUPS.items():
            group_label = QLabel(f"<b>{group_name}</b>")
            angle_layout.addWidget(group_label)
            for name in angle_names:
                cb = QCheckBox(name)
                cb.stateChanged.connect(self._on_selection_changed)
                angle_layout.addWidget(cb)
                self.angle_checkboxes[name] = cb

        angle_layout.addStretch()
        angle_scroll.setWidget(angle_inner)
        _pin_scroll_content(angle_scroll)
        ag = QVBoxLayout()
        ag.addLayout(angle_header)
        ag.addWidget(angle_scroll, stretch=1)

        # Quick select buttons for angles
        abtn = QHBoxLayout()
        btn_all = QPushButton("All")
        btn_all.clicked.connect(lambda: self._set_all_angles(True))
        btn_none = QPushButton("None")
        btn_none.clicked.connect(lambda: self._set_all_angles(False))
        btn_shoulder = QPushButton("Shoulder")
        btn_shoulder.clicked.connect(self._select_shoulder_only)
        abtn.addWidget(btn_all)
        abtn.addWidget(btn_none)
        abtn.addWidget(btn_shoulder)
        ag.addLayout(abtn)

        angle_group.setLayout(ag)
        left_panel.addWidget(angle_group, stretch=3)

        # Sensor channel checkboxes
        self.sensor_group = QGroupBox("Sensor Channels (inputs)")
        sensor_outer = QVBoxLayout(self.sensor_group)
        sensor_outer.addWidget(help_ui.labelled(
            "What the model may use as input", 'sensor_inputs'))
        # Scrolled for the same reason as the angles: a plain column of one
        # checkbox per channel has no upper bound, and a garment with many
        # channels would otherwise push the angle list off the panel.
        sensor_scroll = self.sensor_scroll = QScrollArea()
        sensor_scroll.setWidgetResizable(True)
        sensor_scroll.setMinimumHeight(120)
        sensor_inner = QWidget()
        self.sensor_layout = QVBoxLayout(sensor_inner)
        sensor_scroll.setWidget(sensor_inner)
        sensor_outer.addWidget(sensor_scroll, stretch=1)
        self.sensor_layout.addWidget(QLabel("Load sensor data in Tab 1 first."))
        self.sensor_layout.addStretch()
        _pin_scroll_content(sensor_scroll)

        self.sensor_buttons = QWidget()
        sbtn = QHBoxLayout(self.sensor_buttons)
        sbtn.setContentsMargins(0, 0, 0, 0)
        btn_sall = QPushButton("All")
        btn_sall.clicked.connect(lambda: self._set_all_sensors(True))
        btn_snone = QPushButton("None")
        btn_snone.clicked.connect(lambda: self._set_all_sensors(False))
        sbtn.addWidget(btn_sall)
        sbtn.addWidget(btn_snone)
        self.sensor_buttons.setVisible(False)
        sensor_outer.addWidget(self.sensor_buttons)
        left_panel.addWidget(self.sensor_group, stretch=2)

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMinimumWidth(340)
        left_widget.setMaximumWidth(380)
        layout.addWidget(left_widget)

        # --- Right: Plot area ---
        right_panel = QVBoxLayout()

        self.fig = Figure(figsize=(10, 6))
        self.ax_angle = self.fig.add_subplot(211)
        self.ax_sensor = self.fig.add_subplot(212, sharex=self.ax_angle)
        self.canvas = FigureCanvas(self.fig)
        right_panel.addWidget(self.canvas)

        self.info_label = QLabel("Select angles and sensors, then view the overlay plot.")
        right_panel.addWidget(self.info_label)

        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        layout.addWidget(right_widget, stretch=1)

    def showEvent(self, event):
        """Re-pin on show: metrics settle only once the tab is really visible."""
        super().showEvent(event)
        _pin_scroll_content(self.angle_scroll)
        _pin_scroll_content(self.sensor_scroll)

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
        self._annotate_angle_confidence()
        self._update_plot()

    def _on_subject_changed(self):
        self._annotate_angle_confidence()
        self._update_plot()

    def _populate_subjects(self):
        subjects = get_subjects(self.state, self._split)
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        self.subject_combo.addItems([s['id'] for s in subjects])
        self.subject_combo.blockSignals(False)

    def refresh(self):
        """Rebuild subject list and sensor checkboxes from loaded data."""
        self._populate_subjects()
        self._rebuild_sensor_checkboxes()
        self._annotate_angle_confidence()
        self._update_plot()

    def reset(self):
        """Clear the selections that define the learning problem."""
        for cb in self.angle_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.state['selected_angles'] = []
        self.state['selected_sensors'] = []
        self.refresh()

    def _annotate_angle_confidence(self):
        """
        Label each angle checkbox with its tracking confidence.

        Users pick regression targets on this tab, so this is the point where
        knowing a channel is poorly tracked actually changes a decision.
        """
        from core.confidence import (
            tracking_report, is_synthesised, LOW_CONFIDENCE,
        )

        subject = self._current_subject()
        report = tracking_report(subject.get('skeleton')) if subject else {'available': False}
        per_angle = report.get('per_angle', {}) if report.get('available') else {}

        for name, cb in self.angle_checkboxes.items():
            summary = per_angle.get(name)
            notes = []
            if is_synthesised(name):
                notes.append("synthesised pelvis/spine")

            if summary is None:
                cb.setText(name)
                set_style_property(cb, "lowConfidence", False)
                cb.setToolTip(
                    "Extract a skeleton in Tab 1 to see tracking confidence."
                    + (f"\nNote: {notes[0]}." if notes else "")
                )
                continue

            mean = summary['mean']
            cb.setText(f"{name}  ({mean:.2f})")
            set_style_property(cb, "lowConfidence", mean < LOW_CONFIDENCE)
            tip = [
                f"Mean tracking confidence {mean:.2f}, worst frame {summary['min']:.2f}.",
                f"{summary['frac_low'] * 100:.0f}% of frames below {LOW_CONFIDENCE:.1f}.",
            ]
            if mean < LOW_CONFIDENCE:
                tip.append("Poorly tracked — treat results for this channel with caution.")
            tip += [f"Note: {n}." for n in notes]
            cb.setToolTip("\n".join(tip))

    def _rebuild_sensor_checkboxes(self):
        """
        Offer only sensor channels present in EVERY ready subject, so a
        selection can never reference a channel some subject is missing.
        """
        self.sensor_checkboxes.clear()
        _clear_layout(self.sensor_layout)

        sensor_cols = common_sensor_columns(self.state)
        self.sensor_buttons.setVisible(bool(sensor_cols))
        if not sensor_cols:
            self.sensor_layout.addWidget(QLabel("No sensor data loaded."))
            self.sensor_layout.addStretch()
            _pin_scroll_content(self.sensor_scroll)
            return

        mismatched = inconsistent_sensor_subjects(self.state)
        if mismatched:
            detail = "; ".join(f"{split}/{sid}: {', '.join(extra)}"
                               for split, sid, extra in mismatched)
            warn = QLabel(f"<span style='color:#c62828;'>Channels ignored "
                          f"(not in all subjects) — {detail}</span>")
            warn.setWordWrap(True)
            self.sensor_layout.addWidget(warn)

        for col in sensor_cols:
            cb = QCheckBox(col)
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_selection_changed)
            self.sensor_layout.addWidget(cb)
            self.sensor_checkboxes[col] = cb

        self.sensor_layout.addStretch()
        _pin_scroll_content(self.sensor_scroll)

    def _set_all_angles(self, checked):
        for cb in self.angle_checkboxes.values():
            cb.setChecked(checked)

    def _set_all_sensors(self, checked):
        for cb in self.sensor_checkboxes.values():
            cb.setChecked(checked)

    def _select_shoulder_only(self):
        shoulder_names = set()
        for g in ('Shoulder L', 'Shoulder R'):
            shoulder_names.update(ANGLE_GROUPS[g])
        for name, cb in self.angle_checkboxes.items():
            cb.setChecked(name in shoulder_names)

    def _on_selection_changed(self):
        # Store selections in shared state
        self.state['selected_angles'] = [
            name for name, cb in self.angle_checkboxes.items() if cb.isChecked()
        ]
        self.state['selected_sensors'] = [
            name for name, cb in self.sensor_checkboxes.items() if cb.isChecked()
        ]
        self._update_plot()

    @staticmethod
    def _overlap(angle_time, sensor_time):
        """The window both streams cover, which is all training can ever use."""
        if (angle_time is None or sensor_time is None
                or not len(angle_time) or not len(sensor_time)):
            return None
        start = max(angle_time[0], sensor_time[0])
        stop = min(angle_time[-1], sensor_time[-1])
        return (start, stop) if stop > start else None

    def _update_plot(self):
        subject = self._current_subject()
        angles = subject.get('angles') if subject else None
        sensor_df = subject.get('sensor_df') if subject else None

        # Both panels must read off one clock, or comparing them by eye is
        # misleading — see _shared_clock.
        n_frames = len(next(iter(angles.values()))) if angles else 0
        angle_time, sensor_time, aligned = _shared_clock(subject, n_frames)
        if angle_time is None:
            angle_time = subject.get('time') if subject else None

        self.ax_angle.clear()
        self.ax_sensor.clear()

        sel_angles = [n for n, cb in self.angle_checkboxes.items() if cb.isChecked()]
        sel_sensors = [n for n, cb in self.sensor_checkboxes.items() if cb.isChecked()]

        has_angles = angles is not None and len(sel_angles) > 0
        has_sensors = sensor_df is not None and len(sel_sensors) > 0

        if not has_angles and not has_sensors:
            self.ax_angle.text(0.5, 0.5, "Select angles and/or sensors to plot",
                        transform=self.ax_angle.transAxes, ha='center', va='center', fontsize=14)
            self.canvas.draw()
            return

        # Top: Joint angles, on the session clock
        if has_angles:
            for name in sel_angles:
                if name in angles:
                    vals = np.asarray(angles[name], dtype=float)
                    t = angle_time
                    if t is None or len(t) != len(vals):
                        t = np.arange(len(vals))
                    valid = ~np.isnan(vals)
                    self.ax_angle.plot(t[valid], vals[valid], label=name, linewidth=1)
            self.ax_angle.legend(loc='upper right', fontsize=7)
        self.ax_angle.set_ylabel("Joint Angle (deg)", fontsize=10)
        title = f"{self._split.capitalize()} Data"
        if subject is not None:
            title += f" — {subject['id']}"
        self.ax_angle.set_title(title, fontsize=11)
        self.ax_angle.grid(True, alpha=0.3)

        # Bottom: Sensor channels, on the same clock
        if has_sensors:
            if sensor_time is None or len(sensor_time) != len(sensor_df):
                sensor_time = np.arange(len(sensor_df))

            for col in sel_sensors:
                if col in sensor_df.columns:
                    self.ax_sensor.plot(sensor_time, sensor_df[col].values,
                                       label=col, linewidth=1)
            self.ax_sensor.legend(loc='upper right', fontsize=7)
        self.ax_sensor.set_ylabel("Sensor Value", fontsize=10)
        self.ax_sensor.set_xlabel(
            "Time (s from the start of the recording session)" if aligned
            else "Time — the two panels are NOT on a common clock",
            fontsize=10)
        self.ax_sensor.grid(True, alpha=0.3)

        n_ang = len(sel_angles)
        n_sen = len(sel_sensors)
        info = f"Selected: {n_ang} angle(s), {n_sen} sensor(s)"
        warn = False
        if has_angles and has_sensors:
            if not aligned:
                # Saying nothing here would leave the reader assuming the
                # panels line up, which is the mistake this path exists for.
                info += ("   ·   timestamps missing, so the two panels are"
                         " not on a common clock — do not compare them"
                         " by eye")
                warn = True
            else:
                overlap = self._overlap(angle_time, sensor_time)
                if overlap is not None:
                    info += (f"   ·   both streams cover "
                             f"{overlap[0]:.1f}–{overlap[1]:.1f} s — only"
                             f" this window is used for training")
        set_style_property(self.info_label, "warning", warn)
        self.info_label.setText(info)

        self.fig.tight_layout()
        self.canvas.draw()
