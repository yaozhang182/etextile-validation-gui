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
from gui.panels.video_scrubber import VideoScrubber

# Regression targets and display-only angles, grouped by body part. Shoulder
# and hip flexion / abduction / rotation can be plotted but not trained on; see
# core/targets.py for why.
from core.targets import (  # noqa: E402
    TARGET_GROUPS, DISPLAY_GROUPS, DISPLAY_ONLY_REASON, VECTOR_TARGETS,
    COMPONENTS, is_vector,
)
ANGLE_GROUPS = TARGET_GROUPS


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
        self.angle_checkboxes = {}       # regression targets
        self.display_checkboxes = {}     # plotted only, never trained on
        self.sensor_checkboxes = {}
        # What the cursor needs to read values off the current plot; filled by
        # _update_plot, used by _update_cursor.
        self._cursor = None
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
        angle_group = QGroupBox("Targets")
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

        for group_name, angle_names in TARGET_GROUPS.items():
            group_label = QLabel(f"<b>{group_name}</b>")
            angle_layout.addWidget(group_label)
            for name in angle_names:
                cb = QCheckBox(name)
                cb.stateChanged.connect(self._on_selection_changed)
                angle_layout.addWidget(cb)
                self.angle_checkboxes[name] = cb

        # Angles that describe the movement well but cannot be trained on.
        display_label = QLabel("<b>Display only</b> (not used for training)")
        display_label.setWordWrap(True)
        display_label.setToolTip(DISPLAY_ONLY_REASON)
        angle_layout.addWidget(display_label)
        for group_name, angle_names in DISPLAY_GROUPS.items():
            angle_layout.addWidget(QLabel(f"<i>{group_name}</i>"))
            for name in angle_names:
                cb = QCheckBox(name)
                cb.stateChanged.connect(self._update_plot)
                angle_layout.addWidget(cb)
                self.display_checkboxes[name] = cb

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

        # Video with a cursor: scrubbing the video moves a vertical line across
        # both plots and shows every selected channel's value at that instant;
        # clicking a plot jumps the video there. That is how a suspicious stretch
        # of a curve gets checked against what the person was actually doing.
        vrow = QHBoxLayout()
        self.video_toggle = QCheckBox("Show video")
        self.video_toggle.setChecked(True)
        self.video_toggle.toggled.connect(self._on_video_toggled)
        vrow.addWidget(self.video_toggle)
        help_ui.attach(vrow, 'angle_cursor')
        self.cursor_label = QLabel("")
        vrow.addWidget(self.cursor_label, stretch=1)
        right_panel.addLayout(vrow)
        # Values at the cursor, coloured like their curves. Kept out of the
        # matplotlib legends on purpose: redrawing a legend costs ~18 ms, which
        # alone would make playback stutter.
        self.values_label = QLabel("")
        self.values_label.setWordWrap(True)
        self.values_label.setTextFormat(Qt.TextFormat.RichText)
        right_panel.addWidget(self.values_label)
        self.scrubber = VideoScrubber(video_height=200)
        self.scrubber.frameChanged.connect(self._on_frame_changed)
        right_panel.addWidget(self.scrubber)

        self.fig = Figure(figsize=(10, 6))
        self.ax_angle = self.fig.add_subplot(211)
        self.ax_sensor = self.fig.add_subplot(212, sharex=self.ax_angle)
        self.ax_dir = None
        self._time_axes = [self.ax_angle, self.ax_sensor]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self._on_plot_clicked)
        # The cursor is drawn by blitting: the plots are rendered once into a
        # cached background and only the lines and legends are redrawn per
        # frame. A full redraw takes ~100 ms, too slow to follow playback.
        self._background = None
        self.canvas.mpl_connect('draw_event', self._on_draw)
        right_panel.addWidget(self.canvas, stretch=1)

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

    def hideEvent(self, event):
        """Leaving the tab pauses the video; it would otherwise play unseen."""
        scrubber = getattr(self, 'scrubber', None)   # hidden once while being built
        if scrubber is not None:
            scrubber.stop()
        super().hideEvent(event)

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
        self._load_video()
        self._update_plot()

    def _load_video(self):
        """Point the player at the current subject, at the frame last viewed."""
        subject = self._current_subject()
        angles = subject.get('angles') if subject else None
        if not angles or not subject.get('video_path'):
            self.scrubber.clear()
            return
        n = len(next(iter(angles.values())))
        skeleton = subject.get('skeleton') or {}
        self.scrubber.set_source(subject['video_path'], n, skeleton.get('fps') or 30.0,
                                 frame=subject.get('view_frame', 0))

    def _on_video_toggled(self, shown):
        self.scrubber.video_label.setVisible(shown)
        if not shown:
            self.scrubber.stop()

    def _on_frame_changed(self, frame):
        subject = self._current_subject()
        if subject is not None:
            subject['view_frame'] = int(frame)
        self._update_cursor(frame)

    def _on_plot_clicked(self, event):
        """Jump the video to the instant clicked in either plot."""
        c = self._cursor
        if c is None or event.xdata is None or event.button != 1:
            return
        if event.inaxes is self.ax_sensor and not c['aligned']:
            return      # the sensor panel is on its own clock; no frame there
        if event.inaxes not in self._time_axes:
            return
        if self.canvas.toolbar is not None and getattr(self.canvas.toolbar, 'mode', ''):
            return      # zooming or panning, not picking a moment
        t = c['angle_t']
        frame = int(np.clip(np.searchsorted(t, event.xdata), 0, len(t) - 1))
        if frame > 0 and abs(t[frame - 1] - event.xdata) < abs(t[frame] - event.xdata):
            frame -= 1
        self.scrubber.set_frame(frame)

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
        self._load_video()
        self._update_plot()

    def reset(self):
        """Clear the selections that define the learning problem."""
        for cb in list(self.angle_checkboxes.values()) + list(self.display_checkboxes.values()):
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.state['selected_angles'] = []
        self.state['selected_sensors'] = []
        self.scrubber.clear()
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

        boxes = {**self.angle_checkboxes, **self.display_checkboxes}
        for name, cb in boxes.items():
            summary = per_angle.get(name)
            notes = []
            if is_synthesised(name):
                notes.append("synthesised pelvis/spine")
            if name in self.display_checkboxes:
                notes.append(DISPLAY_ONLY_REASON.rstrip('.'))
            elif is_vector(name):
                notes.append("unit vector in the chest (arm) or pelvis (leg) frame, "
                             "components forward / up / outward")

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
        for side in ('left', 'right'):
            shoulder_names.update([f'{side}_shoulder_elevation',
                                   f'{side}_upper_arm_direction'])
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

        self._cursor = None
        self._background = None
        series_a, series_s = [], []      # (label, x, values, colour, unit) as plotted

        sel_angles = [n for n, cb in self.angle_checkboxes.items() if cb.isChecked()]
        shown = sel_angles + [n for n, cb in self.display_checkboxes.items() if cb.isChecked()]
        sel_sensors = [n for n, cb in self.sensor_checkboxes.items() if cb.isChecked()]
        scalars = [n for n in shown if not is_vector(n)]
        vectors = [n for n in shown if is_vector(n)]

        # Directions are unit vectors (-1..1) and would be flattened on a
        # degree axis, so they get a panel of their own when selected.
        self.fig.clf()
        rows = [2, 1.4, 2] if vectors else [2, 2]
        grid = self.fig.add_gridspec(len(rows), 1, height_ratios=rows)
        self.ax_angle = self.fig.add_subplot(grid[0])
        self.ax_dir = (self.fig.add_subplot(grid[1], sharex=self.ax_angle)
                       if vectors else None)
        self.ax_sensor = self.fig.add_subplot(grid[-1], sharex=self.ax_angle)
        self._time_axes = [ax for ax in (self.ax_angle, self.ax_dir, self.ax_sensor)
                           if ax is not None]

        has_angles = angles is not None and len(shown) > 0
        has_sensors = sensor_df is not None and len(sel_sensors) > 0

        if not has_angles and not has_sensors:
            self.ax_angle.text(0.5, 0.5, "Select angles and/or sensors to plot",
                        transform=self.ax_angle.transAxes, ha='center', va='center', fontsize=14)
            self.cursor_label.setText("")
            self.values_label.setText("")
            self.canvas.draw()
            return

        def plot_channel(ax, key, label, style='-', color=None, unit='°'):
            vals = np.asarray(angles[key], dtype=float)
            t = angle_time
            if t is None or len(t) != len(vals):
                t = np.arange(len(vals))
            valid = ~np.isnan(vals)
            line, = ax.plot(t[valid], vals[valid], style, label=label,
                            linewidth=1, color=color)
            series_a.append((label, t, vals, line.get_color(), unit))
            return line.get_color()

        # Top: angles (degrees), on the session clock
        if has_angles:
            for name in scalars:
                if name in angles:
                    plot_channel(self.ax_angle, name, name)
            if scalars:
                self.ax_angle.legend(loc='upper right', fontsize=7)
        self.ax_angle.set_ylabel("Angle (deg)", fontsize=10)

        # Middle: limb directions, one colour per limb, one line style per axis
        if self.ax_dir is not None and angles is not None:
            styles = dict(zip(COMPONENTS, ('-', '--', ':')))
            for name in vectors:
                color = None
                for comp, key in zip(COMPONENTS, VECTOR_TARGETS[name]):
                    if key in angles:
                        color = plot_channel(self.ax_dir, key, f"{name}.{comp}",
                                             styles[comp], color, unit='')
            self.ax_dir.set_ylim(-1.08, 1.08)
            self.ax_dir.set_ylabel("Direction\n(unit vector)", fontsize=10)
            self.ax_dir.legend(loc='upper right', fontsize=6, ncol=max(1, len(vectors)))
            self.ax_dir.grid(True, alpha=0.3)
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
                    line, = self.ax_sensor.plot(sensor_time, sensor_df[col].values,
                                                label=col, linewidth=1)
                    series_s.append((col, np.asarray(sensor_time, dtype=float),
                                     np.asarray(sensor_df[col].values, dtype=float),
                                     line.get_color(), ''))
            self.ax_sensor.legend(loc='upper right', fontsize=7)
        self.ax_sensor.set_ylabel("Sensor Value", fontsize=10)
        self.ax_sensor.set_xlabel(
            "Time (s from the start of the recording session)" if aligned
            else "Time — the two panels are NOT on a common clock",
            fontsize=10)
        self.ax_sensor.grid(True, alpha=0.3)

        n_ang = len(sel_angles)
        n_sen = len(sel_sensors)
        info = f"Selected: {n_ang} target(s), {n_sen} sensor(s)"
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
        self._install_cursor(subject, angle_time, aligned, series_a, series_s)
        self.canvas.draw()

    def _install_cursor(self, subject, angle_time, aligned, series_a, series_s):
        """Add the vertical cursor lines and remember how to read values off."""
        angles = subject.get('angles') if subject else None
        if not angles:
            return
        n = len(next(iter(angles.values())))
        t = angle_time
        if t is None or len(t) != n:
            t = np.arange(n, dtype=float)    # same fallback as the angle plot
        self._cursor = {
            'angle_t': np.asarray(t, dtype=float),
            'aligned': aligned,
            'angles': series_a,
            'sensors': series_s,
            'lines': [ax.axvline(t[0], color='#d32f2f', linewidth=1.2, zorder=5,
                                 animated=True)
                      for ax in self._time_axes if ax is not self.ax_sensor],
        }
        if aligned:
            self._cursor['lines'].append(self.ax_sensor.axvline(
                t[0], color='#d32f2f', linewidth=1.2, zorder=5, animated=True))
        self._background = None
        self._update_cursor(self.scrubber.frame(), draw=False)

    def _cursor_artists(self):
        c = self._cursor
        if c is None:
            return []
        return c['lines']

    def _on_draw(self, event):
        """After a full redraw: cache the plots without the cursor, then add it."""
        self._background = self.canvas.copy_from_bbox(self.fig.bbox)
        for artist in self._cursor_artists():
            self.fig.draw_artist(artist)

    def _blit(self):
        if self._background is None:
            self.canvas.draw_idle()       # _on_draw will cache and draw the cursor
            return
        self.canvas.restore_region(self._background)
        for artist in self._cursor_artists():
            self.fig.draw_artist(artist)
        # Repaint only the two plot areas, not the whole figure.
        for ax in self._time_axes:
            self.canvas.blit(ax.bbox)

    @staticmethod
    def _value_at(x, values, at):
        """Linear interpolation inside the series, None outside it or on NaN."""
        if len(x) == 0 or at < x[0] or at > x[-1]:
            return None
        ok = np.isfinite(values)
        if not ok.any():
            return None
        v = float(np.interp(at, x[ok], values[ok]))
        return v

    def _update_cursor(self, frame, draw=True):
        """Move the cursor to a video frame and show each channel's value there."""
        c = self._cursor
        if c is None:
            self.cursor_label.setText("")
            self.values_label.setText("")
            return
        t = c['angle_t']
        frame = int(np.clip(frame, 0, len(t) - 1))
        now = t[frame]
        for line in c['lines']:
            line.set_xdata([now, now])

        def fmt(v, unit):
            return "—" if v is None else f"{v:.1f}{unit}"

        parts = []
        for name, x, vals, color, unit in c['angles']:
            v = vals[frame] if frame < len(vals) and np.isfinite(vals[frame]) else None
            parts.append((name, "—" if v is None else
                          (f"{v:.1f}{unit}" if unit else f"{v:.2f}"), color))
        for name, x, vals, color, unit in c['sensors']:
            v = self._value_at(x, vals, now) if c['aligned'] else None
            parts.append((name, fmt(v, unit), color))
        self.values_label.setText("&nbsp;&nbsp; ".join(
            f"<span style='color:{color};'>&#9632;</span>&nbsp;{name}&nbsp;<b>{value}</b>"
            for name, value, color in parts))

        self.cursor_label.setText(f"Frame {frame}   ·   t = {now:.2f} s")
        if draw:
            self._blit()
