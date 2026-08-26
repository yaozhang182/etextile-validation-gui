# Example dataset

Four recording sessions of one participant seated, performing upper-body arm
movements while wearing the instrumented garment. Recorded 2025-10-03 with a
single webcam ~1.5–3 m in front of the participant, full body in frame.

These replace an earlier placeholder set in which the video and sensor files did
not correspond to each other.

## Layout

Each session is one *subject* in the GUI's Data Import tab — the three files
belong together and must be added as a unit:

```
session_N/
  video.mp4     webcam recording, 1280x720
  frames.csv    FrameIndex ; EpochTime ; Marker   (per video frame)
  sensor.csv    EpochTime ; S1 … S10              (10 garment channels)
```

All three streams share one global clock (Unix epoch seconds), which is how the
toolchain aligns them. `frames.csv` maps each video frame index to the wall-clock
instant it was captured; the same timestamp is also burned into the top-left
corner of each video frame, so the correspondence can be checked by eye.

## Sessions

| Session | Video frames | Duration | Sensor rows | Sensor rate | Usable overlap |
|---------|--------------|----------|-------------|-------------|----------------|
| 1 | 454  | 18.2 s  | 143 | 8.9 Hz | 12.9 s  |
| 2 | 575  | 23.0 s  | 237 | 8.9 Hz | 19.0 s  |
| 3 | 463  | 18.5 s  | 160 | 8.9 Hz | 15.3 s  |
| 4 | 3171 | 126.8 s | 936 | 8.9 Hz | 105.2 s |

"Usable overlap" is the window in which both streams have data; the toolchain
crops to it automatically.

A suggested split is sessions **1, 2 and 4 for training** and **session 3 held
out for testing**, which keeps the two longest recordings on the training side.
Any split works — choose it in the GUI by adding sessions to the Training or Test
list.

## Two things to know about this data

**The MP4 frame rate tag is wrong.** Every video declares 25 fps in its
container, but the timestamps show the true capture rate is ~30.1 fps. The
toolchain detects this and uses the timestamps
(`core.data_alignment.estimate_fps_from_frames`), reporting the discrepancy in
the Data Import log. Stream alignment was never affected — it is driven by
`FrameIndex` and `EpochTime` rather than the container tag — but the displayed
time axis would otherwise be stretched by ~21 %.

**The sensor stream is much slower than the video** (8.9 Hz vs ~30 fps). With the
default `upsample_sensor` alignment the sensor readings are linearly interpolated
onto the video frame timestamps, so the number of training samples follows the
video rate.

## Not included

The original archive also contained VIBE-derived joint angles and joint
positions (`joint_angle/`, `joint_position/`, `output/*/vibe_output.pkl`, ~420 MB).
Those come from a different pose pipeline than the MediaPipe one this toolchain
uses, so they are omitted to avoid two conflicting sources of ground truth. They
remain in `example_data.zip` if a comparison is ever wanted —
`joint_position/motion_N.npy` is `(T, 22, 3)` in the same SMPLH layout that
`core.joint_angles.extract_all_joint_angles()` accepts.
