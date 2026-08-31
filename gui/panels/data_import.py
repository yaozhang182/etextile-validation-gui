"""
Tab 1: Data Import — Manage train/test subjects, extract skeletons.

Each subject is one recording session and bundles three files that belong
together: a video, a sensor CSV, and a video-timestamp CSV sharing a global
clock. Train and test each hold their own list of subjects.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QLineEdit, QProgressBar,
    QGroupBox, QTextEdit, QMessageBox, QDialog, QDialogButtonBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
)
from PyQt6.QtCore import QThread, pyqtSignal

from gui.state import make_subject, get_subjects, next_subject_id

VIDEO_FILTER = "Video (*.mp4 *.avi *.mov)"
CSV_FILTER = "CSV (*.csv)"

# Columns in the subject tables
COL_ID, COL_VIDEO, COL_SENSOR, COL_TIMES, COL_FRAMES, COL_STATUS = range(6)


def read_csv_auto(path):
    """Read a CSV that may be ';' or ',' separated."""
    try:
        df = pd.read_csv(path, sep=';')
        # A ','-separated file read with sep=';' collapses into a single column
        if df.shape[1] == 1:
            raise ValueError("wrong separator")
        return df
    except Exception:
        return pd.read_csv(path)


def short_path(path):
    """
    Display a file as "parent/name".

    Datasets commonly give every session identically named files
    (session_1/video.mp4, session_2/video.mp4, ...), so the bare filename would
    make every row look the same. The parent directory is what distinguishes them.
    """
    p = Path(path)
    return f"{p.parent.name}/{p.name}" if p.parent.name else p.name


def angle_cache_path(video_path):
    """Where per-subject computed angles are cached, next to the video."""
    return Path(video_path).with_suffix('.angles.npz')


def cache_description(video_path):
    """
    What a existing cache can restore, so the user knows what to expect.

    Caches written before 3D joints were stored still give angles, training and
    results — only the Skeleton Viewer needs a re-extraction.
    """
    try:
        with np.load(angle_cache_path(video_path), allow_pickle=True) as data:
            if '__smplh__' in data.files:
                return "Cached angles and 3D skeleton found — extraction will be skipped."
            return ("Cached angles found (no 3D skeleton in this older cache). "
                    "Everything works except the Skeleton Viewer; re-extract to "
                    "restore it.")
    except Exception:
        return "A cache file exists but could not be read; it will be re-extracted."


class SkeletonWorker(QThread):
    """Background thread for skeleton extraction of a single subject."""
    progress = pyqtSignal(int, int)  # frame_idx, total
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path

    def run(self):
        try:
            from core.smpl_extraction import extract_skeleton_from_video
            result = extract_skeleton_from_video(
                self.video_path,
                progress_callback=lambda f, t: self.progress.emit(f, t),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class AddSubjectDialog(QDialog):
    """Asks for the three files that make up one subject."""

    def __init__(self, parent, split, suggested_id):
        super().__init__(parent)
        self.setWindowTitle(f"Add {split.capitalize()} Subject")
        self.setMinimumWidth(640)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "A subject is one recording session. All three files must share the "
            "same global clock."
        ))

        grid = QGridLayout()

        grid.addWidget(QLabel("Subject ID:"), 0, 0)
        self.id_edit = QLineEdit(suggested_id)
        grid.addWidget(self.id_edit, 0, 1, 1, 2)

        self.video_edit = self._add_row(grid, 1, "Video (MP4):", VIDEO_FILTER)
        self.sensor_edit = self._add_row(grid, 2, "Sensor CSV:", CSV_FILTER)
        self.times_edit = self._add_row(grid, 3, "Video Timestamps CSV:", CSV_FILTER)

        layout.addLayout(grid)

        self.hint = QLabel("")
        self.hint.setStyleSheet("color: #888;")
        layout.addWidget(self.hint)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.id_edit.textChanged.connect(self._validate)
        self._validate()

    def _add_row(self, grid, row, label, filter_str):
        grid.addWidget(QLabel(label), row, 0)
        edit = QLineEdit()
        edit.setReadOnly(True)
        grid.addWidget(edit, row, 1)
        btn = QPushButton("Browse...")
        btn.clicked.connect(lambda: self._browse(edit, filter_str))
        grid.addWidget(btn, row, 2)
        return edit

    def _browse(self, edit, filter_str):
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", filter_str)
        if path:
            edit.setText(path)
            self._validate()

    def _validate(self):
        """OK stays disabled until all three files and an ID are provided."""
        complete = bool(
            self.id_edit.text().strip()
            and self.video_edit.text()
            and self.sensor_edit.text()
            and self.times_edit.text()
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(complete)

        if complete and angle_cache_path(self.video_edit.text()).exists():
            self.hint.setText(cache_description(self.video_edit.text()))
        elif not complete:
            self.hint.setText("All three files are required.")
        else:
            self.hint.setText("")

    def values(self):
        return (
            self.id_edit.text().strip(),
            self.video_edit.text(),
            self.sensor_edit.text(),
            self.times_edit.text(),
        )


class DataImportPanel(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._workers = []
        self._pending_extractions = []
        self.tables = {}
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        for split in ('train', 'test'):
            layout.addWidget(self._build_split_group(split))

        # --- Actions ---
        action_layout = QHBoxLayout()
        self.extract_btn = QPushButton("Extract Skeleton && Compute Angles")
        self.extract_btn.setStyleSheet("font-weight: bold; padding: 8px;")
        self.extract_btn.clicked.connect(self._extract_all)
        action_layout.addWidget(self.extract_btn)
        layout.addLayout(action_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setMaximumHeight(160)
        layout.addWidget(self.preview)

    def _build_split_group(self, split):
        group = QGroupBox(f"{split.capitalize()}ing Subjects" if split == 'train'
                          else "Test Subjects")
        vbox = QVBoxLayout(group)

        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(
            ["Subject", "Video", "Sensor CSV", "Timestamps CSV", "Frames", "Status"]
        )
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setMaximumHeight(160)
        header = table.horizontalHeader()
        for col in (COL_VIDEO, COL_SENSOR, COL_TIMES):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Stretch)
        vbox.addWidget(table)
        self.tables[split] = table

        btns = QHBoxLayout()
        add_btn = QPushButton("+ Add Subject")
        add_btn.clicked.connect(lambda _, s=split: self._add_subject(s))
        btns.addWidget(add_btn)

        rm_btn = QPushButton("Remove Selected")
        rm_btn.clicked.connect(lambda _, s=split: self._remove_subject(s))
        btns.addWidget(rm_btn)
        btns.addStretch()
        vbox.addLayout(btns)

        return group

    # ------------------------------------------------------------------
    # Subject management
    # ------------------------------------------------------------------

    def _add_subject(self, split):
        dlg = AddSubjectDialog(self, split, next_subject_id(self.state, split))
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        subject_id, video, sensor, times = dlg.values()

        if any(s['id'] == subject_id for s in get_subjects(self.state, split)):
            QMessageBox.warning(self, "Duplicate ID",
                                f"A {split} subject named '{subject_id}' already exists.")
            return

        subject = make_subject(subject_id, video, sensor, times)

        try:
            subject['sensor_df'] = read_csv_auto(sensor)
            subject['video_df'] = read_csv_auto(times)
        except Exception as e:
            QMessageBox.critical(self, "Cannot Read CSV", str(e))
            return

        if 'EpochTime' not in subject['video_df'].columns:
            QMessageBox.warning(
                self, "Missing Column",
                f"{Path(times).name} has no 'EpochTime' column, which is needed to "
                f"align the sensor stream with the video."
            )

        get_subjects(self.state, split).append(subject)
        self._load_cached_angles(subject)
        self._refresh_table(split)
        self._describe(split, subject)

    def _remove_subject(self, split):
        table = self.tables[split]
        rows = sorted({idx.row() for idx in table.selectedIndexes()}, reverse=True)
        if not rows:
            QMessageBox.information(self, "No Selection",
                                    "Select a row to remove.")
            return
        subjects = get_subjects(self.state, split)
        for row in rows:
            if 0 <= row < len(subjects):
                subjects.pop(row)
        self._refresh_table(split)

    def _load_cached_angles(self, subject):
        """Reuse previously computed angles so extraction can be skipped."""
        cache = angle_cache_path(subject['video_path'])
        if not cache.exists():
            return
        reserved = {'__meta__', '__visibility__', '__smplh__'}
        try:
            with np.load(cache, allow_pickle=True) as data:
                angles = {k: data[k] for k in data.files if k not in reserved}
                if not angles:
                    return
                subject['angles'] = angles
                n = len(next(iter(angles.values())))
                fps = float(data['__meta__'][0]) if '__meta__' in data.files else 30.0
                subject['time'] = np.arange(n) / (fps or 30.0)

                # Rebuild the skeleton record. Caches written before 3D joints
                # were stored contain no '__smplh__'; the Skeleton Viewer detects
                # the missing key and says so rather than failing.
                skeleton = {'fps': fps, 'total_frames': n, 'valid_frames': n}
                if '__visibility__' in data.files:
                    skeleton['visibility'] = data['__visibility__']
                if '__smplh__' in data.files:
                    joints = data['__smplh__']
                    skeleton['smplh_joints'] = joints
                    skeleton['total_frames'] = int(len(joints))
                subject['skeleton'] = skeleton
                subject['status'] = 'ready'
        except Exception:
            pass  # Corrupt cache is not fatal — just re-extract.

    def _describe(self, split, subject):
        """Log a short summary of the newly added files."""
        lines = [f"[{split}/{subject['id']}] added"]
        try:
            import cv2
            cap = cv2.VideoCapture(subject['video_path'])
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            lines.append(f"  video: {frames} frames @ {fps:.1f} fps")
        except Exception:
            pass

        sdf, vdf = subject['sensor_df'], subject['video_df']
        if sdf is not None:
            lines.append(f"  sensor: {len(sdf)} rows, columns {list(sdf.columns)}")
        if sdf is not None and vdf is not None and 'EpochTime' in vdf.columns:
            from core.data_alignment import find_timestamp_column
            tcol = find_timestamp_column(sdf)
            if tcol:
                overlap = (min(sdf[tcol].max(), vdf['EpochTime'].max())
                           - max(sdf[tcol].min(), vdf['EpochTime'].min()))
                if overlap <= 0:
                    lines.append("  WARNING: sensor and video timestamps do not overlap")
                else:
                    lines.append(f"  usable overlap: {overlap:.1f} s")
        if subject['status'] == 'ready':
            lines.append("  cached angles loaded — no extraction needed")

        self.preview.append('\n'.join(lines))

    def _refresh_table(self, split):
        table = self.tables[split]
        subjects = get_subjects(self.state, split)
        table.setRowCount(len(subjects))
        for row, s in enumerate(subjects):
            frames = ''
            if s.get('skeleton'):
                frames = str(s['skeleton'].get('total_frames', ''))
            elif s.get('angles'):
                frames = str(len(next(iter(s['angles'].values()))))

            values = {
                COL_ID: s['id'],
                COL_VIDEO: short_path(s['video_path']),
                COL_SENSOR: short_path(s['sensor_path']),
                COL_TIMES: short_path(s['video_csv_path']),
                COL_FRAMES: frames,
                COL_STATUS: s['error'] if s['status'] == 'error' else s['status'],
            }
            for col, text in values.items():
                item = QTableWidgetItem(str(text))
                if col in (COL_VIDEO, COL_SENSOR, COL_TIMES):
                    item.setToolTip(str(
                        {COL_VIDEO: s['video_path'],
                         COL_SENSOR: s['sensor_path'],
                         COL_TIMES: s['video_csv_path']}[col]
                    ))
                table.setItem(row, col, item)
        table.resizeColumnsToContents()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_all(self):
        """Queue every subject that has not been processed yet."""
        pending = []
        for split in ('train', 'test'):
            for subject in get_subjects(self.state, split):
                if subject['status'] != 'ready':
                    pending.append((split, subject))

        if not pending:
            if not get_subjects(self.state, 'train') and not get_subjects(self.state, 'test'):
                QMessageBox.warning(self, "No Subjects",
                                    "Add at least one subject first.")
            else:
                QMessageBox.information(self, "Nothing To Do",
                                        "All subjects already have joint angles.")
            return

        self.extract_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self._pending_extractions = pending
        self._run_next_extraction()

    def _run_next_extraction(self):
        if not self._pending_extractions:
            self.extract_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            self.progress_label.setText("Done! Switch to Tab 2 or 3 to view results.")
            return

        split, subject = self._pending_extractions.pop(0)
        subject['status'] = 'extracting'
        self._refresh_table(split)
        self.progress_label.setText(f"[{split}/{subject['id']}] extracting skeleton...")

        worker = SkeletonWorker(subject['video_path'])
        worker.progress.connect(lambda f, t, s=split, sub=subject: self._on_progress(f, t, s, sub))
        worker.finished.connect(lambda r, s=split, sub=subject: self._on_extraction_done(r, s, sub))
        worker.error.connect(lambda m, s=split, sub=subject: self._on_extraction_error(m, s, sub))
        self._workers.append(worker)
        worker.start()

    def _on_progress(self, frame, total, split, subject):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(frame)
        self.progress_label.setText(
            f"[{split}/{subject['id']}] frame {frame}/{total}..."
        )

    def _on_extraction_done(self, result, split, subject):
        from core.joint_angles import extract_all_joint_angles
        from core.data_alignment import estimate_fps_from_frames

        # The MP4 container's declared frame rate is only a tag and is often
        # wrong (the example recordings claim 25 fps but were captured at ~30).
        # The per-frame timestamps are authoritative, so prefer them for the
        # time axis. Stream alignment uses FrameIndex/EpochTime and is unaffected.
        container_fps = result['fps']
        fps = estimate_fps_from_frames(subject.get('video_df'), fallback=container_fps)
        if container_fps and abs(fps - container_fps) / container_fps > 0.02:
            self.preview.append(
                f"[{split}/{subject['id']}] video file reports "
                f"{container_fps:.1f} fps but its timestamps indicate "
                f"{fps:.1f} fps; using the timestamps."
            )
        result['fps'] = fps
        result['container_fps'] = container_fps

        subject['skeleton'] = result
        try:
            angles, time = extract_all_joint_angles(
                result['smplh_joints'], fps=fps
            )
        except Exception as e:
            self._on_extraction_error(f"joint angle computation failed: {e}", split, subject)
            return

        subject['angles'] = angles
        subject['time'] = time
        subject['status'] = 'ready'
        subject['error'] = None

        try:
            extra = {}
            if result.get('visibility') is not None:
                extra['__visibility__'] = result['visibility']
            # Cache the 3D joints as well, otherwise the Skeleton Viewer has
            # nothing to draw when a subject is restored from cache. (T, 22, 3)
            # float32 is ~0.8 MB for a two-minute recording — cheap next to
            # re-running MediaPipe over every frame.
            if result.get('smplh_joints') is not None:
                extra['__smplh__'] = np.asarray(result['smplh_joints'],
                                                dtype=np.float32)
            np.savez_compressed(
                angle_cache_path(subject['video_path']),
                __meta__=np.array([result['fps']]), **extra, **angles,
            )
        except Exception:
            pass  # Caching is best-effort.

        summary = (f"[{split}/{subject['id']}] {result['valid_frames']}/"
                   f"{result['total_frames']} frames tracked, "
                   f"{len(angles)} angle channels")
        try:
            from core.confidence import tracking_report
            report = tracking_report(result)
            if report['available']:
                summary += (f", tracking quality {report['label']} "
                            f"(mean confidence {report['overall']['mean']:.2f})")
        except Exception:
            pass
        self.preview.append(summary)
        self._refresh_table(split)
        self._run_next_extraction()

    def _on_extraction_error(self, msg, split, subject):
        subject['status'] = 'error'
        subject['error'] = msg
        self._refresh_table(split)
        self.preview.append(f"[{split}/{subject['id']}] ERROR: {msg}")
        QMessageBox.critical(self, "Extraction Error", f"[{subject['id']}] {msg}")
        self._run_next_extraction()

    def refresh(self):
        for split in ('train', 'test'):
            self._refresh_table(split)
