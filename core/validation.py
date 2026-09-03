"""
Input validation for one recording session.

The three files a subject is built from can be wrong in many ways, and until now
most of those surfaced late and in the wrong language: an unsupported video was
only noticed minutes into extraction, a sensor CSV without a timestamp column
failed at training time with "Could not find timestamp column", and swapping the
sensor and timestamp files produced no complaint at all because both are valid
CSVs. For a tool aimed at people who do not read tracebacks, that is the worst
possible failure mode.

validate_session() checks everything cheaply up front and returns Findings, each
carrying what is wrong *and* what to do about it. ERROR blocks the subject from
being added; WARNING lets it through but says what to expect.

Nothing here raises: a validator that crashes on malformed input is no better
than no validator.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ERROR = 'error'
WARNING = 'warning'
INFO = 'info'

#: Unix timestamps outside this range almost certainly are not wall-clock
#: seconds — usually a recording that logged time relative to its own start.
_EPOCH_MIN = 1_000_000_000     # 2001-09-09
_EPOCH_MAX = 4_000_000_000     # 2096-10-02

VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv', '.m4v'}

_TIME_NAMES = {'epochtime', 'epoctime', 'epoch', 'time', 'timestamp'}
_NON_CHANNEL = _TIME_NAMES | {'frameindex', 'marker', 'frame', 'index'}


@dataclass
class Finding:
    level: str
    title: str
    detail: str
    fix: str = ''

    @property
    def blocking(self):
        return self.level == ERROR


def _read_csv(path):
    """Read a ';' or ',' separated CSV. Returns (df, error_message)."""
    try:
        df = pd.read_csv(path, sep=';')
        if df.shape[1] == 1:            # wrong separator collapses to one column
            df = pd.read_csv(path)
        return df, None
    except Exception:
        try:
            return pd.read_csv(path), None
        except Exception as e:
            return None, str(e)


def _timestamp_column(df):
    for col in df.columns:
        if str(col).strip().lower() in _TIME_NAMES:
            return col
    for col in df.columns:                       # looser second pass
        lc = str(col).strip().lower()
        if 'time' in lc or 'epoch' in lc:
            return col
    return None


def _channel_columns(df):
    """Numeric columns that are neither timestamps nor bookkeeping."""
    out = []
    for col in df.columns:
        if str(col).strip().lower() in _NON_CHANNEL:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            out.append(col)
    return out


def _looks_like_frames_log(df):
    """A per-frame timestamp log: an index column, a clock, and no real channels."""
    lower = {str(c).strip().lower() for c in df.columns}
    return 'frameindex' in lower and not _channel_columns(df)


def _describe_video(path):
    """(info dict, error_message). Cheap: opens the container, reads metadata."""
    try:
        import cv2
    except Exception as e:
        return None, f"OpenCV is unavailable ({e})"

    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None, "the file could not be opened as a video"
        info = {
            'frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        # Metadata can be present while the stream is undecodable, so confirm
        # one frame actually comes back.
        ok, _ = cap.read()
        if not ok:
            return info, "the video opened but no frame could be decoded"
        return info, None
    except Exception as e:
        return None, str(e)
    finally:
        if cap is not None:
            cap.release()


def validate_session(video_path, sensor_path, frames_path):
    """
    Check one subject's three files.

    Returns a list of Finding, most severe first. An empty list means everything
    looked fine.
    """
    findings = []
    add = findings.append

    # ---------------------------------------------------------------- files
    paths = {'Video': video_path, 'Sensor CSV': sensor_path,
             'Timestamps CSV': frames_path}
    for label, raw in paths.items():
        p = Path(raw) if raw else None
        if not raw or not p.exists():
            add(Finding(ERROR, f"{label} not found",
                        f"There is no file at:\n{raw}",
                        "Pick the file again."))
        elif p.is_dir():
            add(Finding(ERROR, f"{label} is a folder",
                        f"{p.name} is a directory, not a file.",
                        "Select the file inside it."))
        elif p.stat().st_size == 0:
            add(Finding(ERROR, f"{label} is empty",
                        f"{p.name} is 0 bytes.",
                        "The recording probably failed; use a different file."))
    if any(f.blocking for f in findings):
        return findings

    # ---------------------------------------------------------------- video
    suffix = Path(video_path).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        add(Finding(WARNING, "Unusual video extension",
                    f"{Path(video_path).name} does not end in one of "
                    f"{', '.join(sorted(VIDEO_SUFFIXES))}.",
                    "It will still be tried, but MP4 is what the tool expects."))

    video, video_error = _describe_video(video_path)
    if video_error:
        add(Finding(ERROR, "Video cannot be read",
                    f"{Path(video_path).name}: {video_error}.",
                    "Re-export it as an H.264 MP4 — that is the format the "
                    "pose tracker is tested against."))
    elif video['frames'] <= 0:
        add(Finding(ERROR, "Video contains no frames",
                    f"{Path(video_path).name} reports {video['frames']} frames.",
                    "The recording is truncated or corrupt; re-record it."))

    # ------------------------------------------------------------ CSV parse
    sensor_df, sensor_error = _read_csv(sensor_path)
    if sensor_error:
        add(Finding(ERROR, "Sensor CSV cannot be parsed",
                    f"{Path(sensor_path).name} is not readable as a CSV "
                    f"({sensor_error}).",
                    "Open it in a spreadsheet and re-save as CSV. Columns may "
                    "be separated by ';' or ','."))
    frames_df, frames_error = _read_csv(frames_path)
    if frames_error:
        add(Finding(ERROR, "Timestamps CSV cannot be parsed",
                    f"{Path(frames_path).name} is not readable as a CSV "
                    f"({frames_error}).",
                    "It needs one row per video frame, with FrameIndex and "
                    "EpochTime columns."))
    if sensor_df is None or frames_df is None:
        return _ordered(findings)

    # ------------------------------------------------------- swapped files?
    # Both are valid CSVs, so nothing else would catch this, and it is an easy
    # mistake to make in a dialog with three Browse buttons.
    if _looks_like_frames_log(sensor_df) and not _looks_like_frames_log(frames_df):
        add(Finding(ERROR, "The two CSVs look swapped",
                    f"{Path(sensor_path).name} looks like a per-frame timestamp "
                    f"log (it has FrameIndex and no sensor channels), while "
                    f"{Path(frames_path).name} does not.",
                    "Swap the Sensor CSV and Timestamps CSV selections."))
        return _ordered(findings)

    # ------------------------------------------------------------ sensor CSV
    sensor_time = _timestamp_column(sensor_df)
    channels = _channel_columns(sensor_df)
    if sensor_time is None:
        add(Finding(ERROR, "Sensor CSV has no timestamp column",
                    f"Columns found: {', '.join(map(str, sensor_df.columns))}.",
                    "Add a column named EpochTime holding Unix seconds for each "
                    "sample. Without it the sensor stream cannot be aligned to "
                    "the video."))
    if not channels:
        add(Finding(ERROR, "Sensor CSV has no sensor channels",
                    f"No numeric data columns besides the timestamp were found "
                    f"in {Path(sensor_path).name}.",
                    "Each sensing element needs its own numeric column, e.g. "
                    "S1, S2, S3."))
    elif len(sensor_df) < 20:
        add(Finding(WARNING, "Very short sensor recording",
                    f"{Path(sensor_path).name} has only {len(sensor_df)} rows.",
                    "Training needs more samples than the sequence length; "
                    "expect to lower it or record for longer."))

    # ------------------------------------------------------------ frames CSV
    if 'EpochTime' not in frames_df.columns:
        found = _timestamp_column(frames_df)
        add(Finding(ERROR, "Timestamps CSV has no EpochTime column",
                    f"Columns found: {', '.join(map(str, frames_df.columns))}."
                    + (f" A column named {found!r} looks close." if found else ""),
                    "The column must be called exactly EpochTime and hold the "
                    "Unix time at which each video frame was captured."))
    if 'FrameIndex' not in frames_df.columns:
        add(Finding(WARNING, "Timestamps CSV has no FrameIndex column",
                    f"{Path(frames_path).name} has no FrameIndex.",
                    "Rows will be assumed to start at the first video frame. "
                    "Add FrameIndex if the log does not begin at frame 0."))

    if any(f.blocking for f in findings):
        return _ordered(findings)

    # ------------------------------------------------- clocks and overlap
    s_time = pd.to_numeric(sensor_df[sensor_time], errors='coerce').dropna()
    f_time = pd.to_numeric(frames_df['EpochTime'], errors='coerce').dropna()

    if s_time.empty or f_time.empty:
        add(Finding(ERROR, "Timestamps are not numeric",
                    "A timestamp column could not be read as numbers.",
                    "Timestamps must be plain Unix seconds, e.g. "
                    "1759494812.93 — not a formatted date."))
        return _ordered(findings)

    for label, series, name in (("Sensor CSV", s_time, Path(sensor_path).name),
                                ("Timestamps CSV", f_time, Path(frames_path).name)):
        if not (_EPOCH_MIN < float(series.median()) < _EPOCH_MAX):
            add(Finding(ERROR, f"{label} timestamps are not wall-clock time",
                        f"{name} has timestamps around {float(series.median()):.1f}, "
                        f"which is not a plausible Unix time.",
                        "Both files must use the same absolute clock (Unix "
                        "seconds). Time measured from the start of each "
                        "recording cannot be aligned."))
    if any(f.blocking for f in findings):
        return _ordered(findings)

    overlap = min(s_time.max(), f_time.max()) - max(s_time.min(), f_time.min())
    if overlap <= 0:
        gap = abs(max(s_time.min(), f_time.min()) - min(s_time.max(), f_time.max()))
        add(Finding(ERROR, "Sensor and video do not overlap in time",
                    f"The sensor covers "
                    f"{s_time.min():.0f}–{s_time.max():.0f} and the video covers "
                    f"{f_time.min():.0f}–{f_time.max():.0f}; they miss each other "
                    f"by {gap:.0f} s.",
                    "These two files are from different recordings. Pick the "
                    "sensor CSV recorded at the same time as this video."))
        return _ordered(findings)

    if overlap < 5:
        add(Finding(WARNING, "Very little usable overlap",
                    f"Only {overlap:.1f} s of the recording has both video and "
                    f"sensor data.",
                    "Everything outside the overlap is discarded, so this "
                    "session will contribute few training samples."))

    # frames log vs actual video length
    if video and video['frames'] > 0:
        rows = len(frames_df)
        if rows > video['frames'] + 1:
            add(Finding(WARNING, "More timestamp rows than video frames",
                        f"{Path(frames_path).name} has {rows} rows but the video "
                        f"has {video['frames']} frames.",
                        "The extra rows are ignored. Check the two files come "
                        "from the same recording."))
        elif rows < video['frames'] * 0.5:
            add(Finding(WARNING, "Timestamp log covers only part of the video",
                        f"{Path(frames_path).name} has {rows} rows for a "
                        f"{video['frames']}-frame video.",
                        "Only the logged frames can be used as ground truth; "
                        "the rest of the video is ignored."))

        # A wrong container frame-rate tag is common and handled, but say so.
        span = float(f_time.max() - f_time.min())
        if span > 0 and video['fps'] > 0:
            true_fps = (rows - 1) / span
            if abs(true_fps - video['fps']) / video['fps'] > 0.05:
                add(Finding(INFO, "Video frame-rate tag looks wrong",
                            f"The file declares {video['fps']:.1f} fps but its "
                            f"timestamps imply {true_fps:.1f} fps.",
                            "The timestamps are used instead, so this is "
                            "handled — no action needed."))

    if not any(f.level in (ERROR, WARNING) for f in findings):
        detail = f"{overlap:.1f} s of overlap, {len(channels)} sensor channel(s)"
        if video:
            detail += f", {video['frames']} video frames"
        add(Finding(INFO, "Files look consistent", detail + "."))

    return _ordered(findings)


def _ordered(findings):
    rank = {ERROR: 0, WARNING: 1, INFO: 2}
    return sorted(findings, key=lambda f: rank.get(f.level, 3))


def has_errors(findings):
    return any(f.blocking for f in findings)


def format_findings(findings, html=True):
    """Render findings for display in the GUI."""
    if not findings:
        return ""
    if not html:
        return "\n".join(
            f"[{f.level.upper()}] {f.title}: {f.detail}"
            + (f" -> {f.fix}" if f.fix else "")
            for f in findings
        )

    colour = {ERROR: '#c62828', WARNING: '#ef6c00', INFO: '#2e7d32'}
    label = {ERROR: 'Problem', WARNING: 'Warning', INFO: 'OK'}
    parts = []
    for f in findings:
        parts.append(
            f"<p style='margin:2px 0 8px 0'>"
            f"<b style='color:{colour.get(f.level, '#555')}'>"
            f"{label.get(f.level, f.level)}: {f.title}</b><br>"
            f"{f.detail}"
            + (f"<br><i>{f.fix}</i>" if f.fix else "")
            + "</p>"
        )
    return "".join(parts)
