"""
A modal progress window for the two slow steps.

Pose extraction and model training each run for minutes. They already ran in a
background thread with a progress bar on their tab, but the rest of the window
stayed clickable: users switch tabs, press buttons again, and — because the main
thread is busy delivering progress signals — conclude the application has hung
and start clicking harder.

This blocks input to the main window for the duration, says what is happening and
how long is left, and offers a cancel. It is shown with show() rather than exec()
so the existing signal-driven flow is untouched: the worker keeps emitting, the
slots keep updating, and nothing has to be restructured around a nested event
loop.

The one rule when using it: **close it on every exit path** — success, failure and
cancellation alike. A dialog left open behind a finished job locks the window.
"""

import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout,
)


def _duration(seconds):
    """'45 s' / '2 min 05 s'. Coarse on purpose: false precision is noise."""
    if seconds is None or seconds < 0 or seconds != seconds:   # NaN-safe
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds} s"
    return f"{seconds // 60} min {seconds % 60:02d} s"


class BusyDialog(QDialog):
    """
    Modal progress window with elapsed time, an estimate, and optional cancel.

    Call setProgress()/setDetail() as work advances and finish() when it ends.
    """

    def __init__(self, parent, title, message, cancel_text=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        # No close button: the only way out is Cancel or completion, so the
        # dialog cannot be dismissed while the thread is still running.
        self.setWindowFlags(
            Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint
            | Qt.WindowType.WindowTitleHint
        )
        self.setMinimumWidth(460)

        self._cancelled = False
        self._started = time.monotonic()
        self._cancel_callback = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self.message_label = QLabel(message)
        self.message_label.setWordWrap(True)
        self.message_label.setProperty("headline", True)
        layout.addWidget(self.message_label)

        self.detail_label = QLabel("Starting…")
        self.detail_label.setWordWrap(True)
        self.detail_label.setProperty("hint", True)
        layout.addWidget(self.detail_label)

        self.bar = QProgressBar()
        self.bar.setRange(0, 0)          # indeterminate until told otherwise
        layout.addWidget(self.bar)

        self.timing_label = QLabel("")
        self.timing_label.setProperty("hint", True)
        layout.addWidget(self.timing_label)

        row = QHBoxLayout()
        row.addStretch()
        self.cancel_btn = QPushButton(cancel_text or "Cancel")
        self.cancel_btn.setVisible(bool(cancel_text))
        self.cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(self.cancel_btn)
        layout.addLayout(row)

    # -- state -------------------------------------------------------------

    @property
    def cancelled(self):
        return self._cancelled

    def set_cancel_callback(self, callback):
        self._cancel_callback = callback

    def _on_cancel(self):
        self._cancelled = True
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Stopping…")
        self.detail_label.setText("Finishing the current step, please wait…")
        if self._cancel_callback is not None:
            self._cancel_callback()

    # -- updates -----------------------------------------------------------

    def set_message(self, text):
        self.message_label.setText(text)

    def set_detail(self, text):
        self.detail_label.setText(text)

    def set_progress(self, current, total):
        """Determinate progress, with an estimate once there is enough to go on."""
        if not total or total <= 0:
            self.bar.setRange(0, 0)
            self.timing_label.setText(
                f"Elapsed {_duration(time.monotonic() - self._started)}")
            return

        current = max(0, min(int(current), int(total)))
        if self.bar.maximum() != int(total):
            self.bar.setRange(0, int(total))
        self.bar.setValue(current)

        elapsed = time.monotonic() - self._started
        # Below a few percent the estimate swings wildly and reads as noise.
        if current > 0 and elapsed > 1.0 and current / total > 0.02:
            remaining = elapsed * (total - current) / current
            self.timing_label.setText(
                f"Elapsed {_duration(elapsed)}   ·   "
                f"about {_duration(remaining)} left")
        else:
            self.timing_label.setText(f"Elapsed {_duration(elapsed)}")

    def finish(self):
        """Close the dialog. Safe to call more than once."""
        if self.isVisible():
            self.accept()

    def reject(self):
        """Esc must not dismiss a dialog whose work is still running."""
        if self._cancelled:
            super().reject()
        else:
            self._on_cancel()
