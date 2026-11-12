"""
A small video player for scrubbing through a recording frame by frame.

Shows one frame of the video in a QLabel and offers Play / slider / frame label.
It knows nothing about joint angles or sensors: it emits ``frameChanged(int)``
and the owner decides what that frame means.
"""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from gui import icons


class VideoScrubber(QWidget):
    frameChanged = pyqtSignal(int)

    def __init__(self, parent=None, video_height=200):
        super().__init__(parent)
        self._cap = None
        self._cap_path = None
        self._next_frame = None      # frame the decoder will return on read()
        self._video_path = None
        self._fps = 30.0
        self._play_start = None      # (wall-clock time, frame) when Play was pressed
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("No video")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedHeight(video_height)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        self.video_label.setStyleSheet("background: #111; color: #aaa;")
        layout.addWidget(self.video_label)

        ctrl = QHBoxLayout()
        self.play_btn = icons.decorate(QPushButton("Play"), 'play')
        self.play_btn.clicked.connect(self.toggle_play)
        ctrl.addWidget(self.play_btn)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_value_changed)
        self.slider.sliderPressed.connect(self.stop)   # grabbing the slider takes over from Play
        ctrl.addWidget(self.slider, stretch=1)
        self.frame_label = QLabel("Frame 0 / 0")
        ctrl.addWidget(self.frame_label)
        layout.addLayout(ctrl)

    # --- public -------------------------------------------------------------

    @property
    def playing(self):
        return self._timer.isActive()

    def frame(self):
        return self.slider.value()

    def set_source(self, video_path, n_frames, fps=30.0, frame=0):
        """Point the player at a recording; does not emit frameChanged."""
        self.stop()
        if video_path != self._video_path:
            self._release()
        self._video_path = video_path
        self._fps = float(fps) if fps else 30.0
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(int(n_frames) - 1, 0))
        self.slider.setValue(min(max(int(frame), 0), self.slider.maximum()))
        self.slider.blockSignals(False)
        self._show(self.slider.value())

    def set_frame(self, frame):
        """Move to a frame (emits frameChanged if it changes)."""
        self.slider.setValue(int(frame))

    def toggle_play(self):
        if self.playing:
            self.stop()
            return
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        import time
        self._play_start = (time.perf_counter(), self.slider.value())
        self._timer.start(max(int(1000 / self._fps), 10))
        self.play_btn.setText("Pause")
        icons.decorate(self.play_btn, 'pause')

    def stop(self):
        self._timer.stop()
        self.play_btn.setText("Play")
        icons.decorate(self.play_btn, 'play')

    def clear(self):
        self.stop()
        self._release()
        self._video_path = None
        self.slider.blockSignals(True)
        self.slider.setMaximum(0)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.frame_label.setText("Frame 0 / 0")
        self.video_label.clear()
        self.video_label.setText("No video")

    # --- internals ----------------------------------------------------------

    def _advance(self):
        """
        Show the frame that belongs to the time elapsed since Play, so playback
        keeps real time on a slow computer by skipping frames instead of
        drifting ever further behind.
        """
        import time
        t0, f0 = self._play_start
        target = f0 + int((time.perf_counter() - t0) * self._fps)
        if target >= self.slider.maximum():
            self.slider.setValue(self.slider.maximum())
            self.stop()
        elif target > self.slider.value():
            self.slider.setValue(target)

    def _on_value_changed(self, value):
        self._show(value)
        self.frameChanged.emit(value)

    def _release(self):
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._cap_path = None
        self._next_frame = None

    def _read(self, frame_idx):
        """
        Decode one frame. Seeking an MP4 is slow (it decodes from the previous
        keyframe), so during playback, when the wanted frame is simply the next
        one, read on without seeking.
        """
        import cv2
        if self._cap_path != self._video_path:
            self._release()
            self._cap = cv2.VideoCapture(self._video_path)
            self._cap_path = self._video_path
        if self._cap is None or not self._cap.isOpened():
            return None
        if self._next_frame != frame_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, image = self._cap.read()
        self._next_frame = frame_idx + 1 if ok else None
        return image if ok else None

    def _show(self, frame_idx):
        total = self.slider.maximum() + 1 if self._video_path else 0
        self.frame_label.setText(f"Frame {frame_idx} / {max(total - 1, 0)}")
        if not self._video_path:
            return
        image = None
        try:
            image = self._read(frame_idx)
        except Exception:
            image = None      # an unreadable video must not break the tab
        if image is None:
            self.video_label.setText("Video frame not available")
            return
        # Shrink with OpenCV first: scaling a 720p frame in Qt costs ~10 ms,
        # a third of the time one frame has during playback.
        import cv2
        h, w = image.shape[:2]
        scale = min(self.video_label.width() / w, self.video_label.height() / h, 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (max(int(w * scale), 1), max(int(h * scale), 1)),
                               interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg.copy()))

    def closeEvent(self, event):
        self.clear()
        super().closeEvent(event)
