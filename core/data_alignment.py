"""
Data loading and alignment module.

Handles sensor-frame time alignment, label synchronization,
interpolation, and PyTorch dataset creation.

Consolidated from shoulderLLM:
  - src/data_loader.py
  - src/utils.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import interpolate

# Heavy imports (torch, sklearn) are deferred to avoid blocking the GUI
# when only lightweight functions like find_timestamp_column are needed.


# =============================================================================
# PyTorch Dataset
# =============================================================================

class ShoulderDataset:
    """
    PyTorch Dataset for shoulder motion prediction.
    Creates sliding windows over sensor data for temporal modeling.
    Compatible with torch DataLoader (__len__ + __getitem__).
    """

    def __init__(self, sensor_data, labels, seq_len=40):
        """
        Args:
            sensor_data: (N, num_sensors) array
            labels: (N, 3) array [flexion, abduction, rotation]
            seq_len: sliding window length
        """
        import torch
        self.sensor = torch.FloatTensor(sensor_data)
        self.labels = torch.FloatTensor(labels)
        self.seq_len = seq_len

    def __len__(self):
        # Clamped: a recording shorter than one window yields no samples.
        # Returning a negative length raises ValueError in len().
        return max(0, len(self.sensor) - self.seq_len)

    def __getitem__(self, idx):
        x = self.sensor[idx: idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]
        return x, y


# =============================================================================
# Timestamp Utilities
# =============================================================================

def find_timestamp_column(df):
    """Auto-detect the timestamp column in a dataframe."""
    candidates = {col: col.lower() for col in df.columns}

    for col, lc in candidates.items():
        if 'epo' in lc and 'epoch' in lc:
            return col
    for col, lc in candidates.items():
        if 'epoch' in lc:
            return col
    for col, lc in candidates.items():
        if 'time' in lc or 'timestamp' in lc:
            return col
    return None


def align_sensor_frame_time_window(sensor_df, frames_df):
    """
    Crop sensor_df and frames_df to their overlapping EpochTime window.

    Returns:
        (sensor_time_col, sensor_df_cropped, frames_df_cropped, overlap_start, overlap_end)
    """
    sensor_time_col = find_timestamp_column(sensor_df)
    if sensor_time_col is None:
        raise ValueError("Could not find timestamp column in sensor dataframe")
    if 'EpochTime' not in frames_df.columns:
        raise ValueError("frames dataframe must contain 'EpochTime' column")

    sensor_start = float(sensor_df[sensor_time_col].min())
    sensor_end = float(sensor_df[sensor_time_col].max())
    frame_start = float(frames_df['EpochTime'].min())
    frame_end = float(frames_df['EpochTime'].max())

    overlap_start = max(sensor_start, frame_start)
    overlap_end = min(sensor_end, frame_end)

    if overlap_end <= overlap_start:
        raise ValueError(
            f"No time overlap between sensor and frames: "
            f"[{sensor_start}, {sensor_end}] vs [{frame_start}, {frame_end}]"
        )

    sensor_mask = (
        (sensor_df[sensor_time_col] >= overlap_start) &
        (sensor_df[sensor_time_col] <= overlap_end)
    )
    frames_mask = (
        (frames_df['EpochTime'] >= overlap_start) &
        (frames_df['EpochTime'] <= overlap_end)
    )

    sensor_df_cropped = sensor_df.loc[sensor_mask].reset_index(drop=True)
    frames_df_cropped = frames_df.loc[frames_mask].reset_index(drop=True)

    return sensor_time_col, sensor_df_cropped, frames_df_cropped, overlap_start, overlap_end


def align_labels_to_frames(joint_angles, frames_df):
    """
    Trim joint_angles array to match frames_df rows, using FrameIndex if available.

    Returns:
        (trimmed_joint_angles, frames_df_trimmed)
    """
    n_frames = len(frames_df)

    if 'FrameIndex' in frames_df.columns:
        start_frame = int(frames_df['FrameIndex'].iloc[0])
        max_needed = start_frame + n_frames

        if len(joint_angles) < max_needed:
            safe_len = len(joint_angles) - start_frame
            if safe_len <= 0:
                raise ValueError("Label arrays shorter than first FrameIndex in cropped frames")
            frames_df = frames_df.iloc[:safe_len].reset_index(drop=True)
            n_frames = len(frames_df)
            max_needed = start_frame + n_frames

        ja_trimmed = joint_angles[start_frame:max_needed]
        return ja_trimmed, frames_df.reset_index(drop=True)
    else:
        if len(joint_angles) >= n_frames:
            return joint_angles[:n_frames], frames_df.reset_index(drop=True)
        else:
            frames_df = frames_df.iloc[:len(joint_angles)].reset_index(drop=True)
            return joint_angles, frames_df


def interpolate_timeseries(source_timestamps, source_data, target_timestamps, kind='linear'):
    """
    Interpolate time series from source to target timestamps.

    Args:
        source_timestamps: (N,) source time points
        source_data: (N, D) or (N,) data
        target_timestamps: (M,) target time points
        kind: interpolation method

    Returns:
        (M, D) or (M,) interpolated data
    """
    if source_data.ndim == 1:
        f = interpolate.interp1d(
            source_timestamps, source_data,
            kind=kind, bounds_error=False, fill_value='extrapolate'
        )
        return f(target_timestamps)
    else:
        return np.stack([
            interpolate.interp1d(
                source_timestamps, source_data[:, d],
                kind=kind, bounds_error=False, fill_value='extrapolate'
            )(target_timestamps)
            for d in range(source_data.shape[1])
        ], axis=1)


# =============================================================================
# High-Level Data Loading
# =============================================================================

def align_and_load(sensor_df, video_df, joint_angles, method='downsample_labels',
                   sensor_cols=None):
    """
    Align sensor data with joint angles using EpochTime overlap.

    Args:
        sensor_df: DataFrame with sensor data + timestamp column
        video_df: DataFrame with EpochTime (and optional FrameIndex)
        joint_angles: (N, 3) array
        method: 'downsample_labels' or 'upsample_sensor'
        sensor_cols: explicit list of sensor columns to use. When None, all
            non-timestamp columns are auto-detected (previous behaviour).

    Returns:
        (aligned_sensor, aligned_labels, timestamps)
    """
    sensor_time_col = find_timestamp_column(sensor_df)
    if sensor_time_col is None:
        raise ValueError("Could not find timestamp column in sensor data")
    if 'EpochTime' not in video_df.columns:
        raise ValueError("Video dataframe must contain 'EpochTime' column")

    sensor_time_col, sensor_df_cropped, video_df_cropped, overlap_start, overlap_end = \
        align_sensor_frame_time_window(sensor_df, video_df)

    labels_trimmed, video_df_trimmed = align_labels_to_frames(joint_angles, video_df_cropped)

    if sensor_cols is None:
        exclude_cols = ['epoctime', 'epoch', 'time', 'timestamp', 'frameindex',
                        sensor_time_col.lower()]
        sensor_cols = [c for c in sensor_df_cropped.columns if c.lower() not in exclude_cols]
    else:
        missing = [c for c in sensor_cols if c not in sensor_df_cropped.columns]
        if missing:
            raise ValueError(f"Sensor columns not found in data: {missing}")

    sensor_timestamps = sensor_df_cropped[sensor_time_col].values.astype(float)
    label_timestamps = video_df_trimmed['EpochTime'].values.astype(float)

    if method == 'downsample_labels':
        target_timestamps = sensor_timestamps
        aligned_sensor = sensor_df_cropped[sensor_cols].values
        aligned_labels = interpolate_timeseries(
            label_timestamps, labels_trimmed, target_timestamps, kind='linear'
        )
    elif method == 'upsample_sensor':
        target_timestamps = label_timestamps
        aligned_labels = labels_trimmed
        sensor_data = sensor_df_cropped[sensor_cols].values
        aligned_sensor = interpolate_timeseries(
            sensor_timestamps, sensor_data, target_timestamps, kind='linear'
        )
    else:
        raise ValueError(f"Unknown alignment method: {method}")

    return aligned_sensor, aligned_labels, target_timestamps


def estimate_fps_from_frames(video_df, fallback=None):
    """
    Recover the true capture rate from the video-timestamp CSV.

    An MP4 container's declared frame rate is only a tag and is often wrong: the
    example recordings are tagged 25 fps but were actually captured at ~30 fps.
    The per-frame timestamps are the authoritative clock, so derive the rate from
    them whenever they are available.

    Note this affects only the displayed time axis. Stream alignment is driven by
    FrameIndex and EpochTime directly and is unaffected by a wrong container tag.

    Args:
        video_df: DataFrame with EpochTime (and ideally FrameIndex)
        fallback: value to return when the rate cannot be derived

    Returns:
        frames per second, or `fallback`
    """
    if video_df is None or 'EpochTime' not in getattr(video_df, 'columns', []):
        return fallback

    t = np.asarray(video_df['EpochTime'].values, dtype=float)
    if len(t) < 2:
        return fallback

    span = float(t[-1] - t[0])
    if not np.isfinite(span) or span <= 0:
        return fallback

    # Prefer FrameIndex: it tolerates a timestamp log that skipped frames.
    if 'FrameIndex' in video_df.columns:
        idx = np.asarray(video_df['FrameIndex'].values, dtype=float)
        n_frames = float(idx[-1] - idx[0])
    else:
        n_frames = float(len(t) - 1)

    if n_frames <= 0:
        return fallback

    fps = n_frames / span
    return fps if 1.0 < fps < 1000.0 else fallback


def build_supervised_arrays(sensor_df, video_df, angles_dict, angle_names,
                            sensor_cols, method='upsample_sensor'):
    """
    Build time-aligned (X, Y) training arrays for ONE subject/recording.

    This is the GUI's entry point into the alignment machinery. It stacks the
    user-selected angle channels into a label matrix, runs the shared global
    clock alignment, and drops rows where the pose tracker produced NaNs.

    Args:
        sensor_df: DataFrame with sensor data + timestamp column
        video_df: DataFrame with EpochTime (and optional FrameIndex)
        angles_dict: {angle_name: (T,) array} as produced by extract_all_joint_angles
        angle_names: list of angle channel names to use as targets
        sensor_cols: list of sensor column names to use as inputs
        method: 'upsample_sensor' interpolates the sensor stream onto the video
            frame timestamps (default, matches the per-frame ground truth);
            'downsample_labels' interpolates angles onto the sensor timestamps.

    Returns:
        (X, Y, t): X (N, n_sensors), Y (N, n_angles), t (N,) epoch seconds
    """
    missing = [n for n in angle_names if n not in angles_dict]
    if missing:
        raise ValueError(f"Angle channels not found: {missing}")

    labels = np.stack([np.asarray(angles_dict[n], dtype=float) for n in angle_names], axis=1)

    X, Y, t = align_and_load(
        sensor_df, video_df, labels, method=method, sensor_cols=sensor_cols,
    )

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    t = np.asarray(t, dtype=float)

    valid = ~(np.any(np.isnan(X), axis=1) | np.any(np.isnan(Y), axis=1))
    return X[valid], Y[valid], t[valid]


def load_iteration_data(data_dir, alignment_method='downsample_labels'):
    """
    Load and prepare train/test data from an iteration directory.

    NOTE: legacy flat layout, used by main_pipeline.py. It expects joint angles
    to have been computed already and saved as .npy. The GUI and
    benchmarks/run_benchmark.py instead use the session_N/ layout of
    input_data/examples_data and compute angles from video via MediaPipe.

    Expected files:
        train_sensor.csv, train_video.csv, train_angle.npy
        test_sensor.csv, test_video.csv, test_angle.npy

    Args:
        data_dir: Path to iteration directory
        alignment_method: 'downsample_labels' or 'upsample_sensor'

    Returns:
        (train_sensor, train_labels, test_sensor, test_labels, scaler)
    """
    data_dir = Path(data_dir)

    # Load and align training data
    train_sensor_df = pd.read_csv(data_dir / 'train_sensor.csv', sep=';')
    train_video_df = pd.read_csv(data_dir / 'train_video.csv', sep=';')
    train_angles = np.load(data_dir / 'train_angle.npy')

    train_sensor, train_labels, _ = align_and_load(
        train_sensor_df, train_video_df, train_angles, method=alignment_method
    )

    # Load and align test data
    test_sensor_df = pd.read_csv(data_dir / 'test_sensor.csv', sep=';')
    test_video_df = pd.read_csv(data_dir / 'test_video.csv', sep=';')
    test_angles = np.load(data_dir / 'test_angle.npy')

    test_sensor, test_labels, _ = align_and_load(
        test_sensor_df, test_video_df, test_angles, method=alignment_method
    )

    # Normalize sensor data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_sensor = scaler.fit_transform(train_sensor)
    test_sensor = scaler.transform(test_sensor)

    return train_sensor, train_labels, test_sensor, test_labels, scaler


# =============================================================================
# Metadata I/O
# =============================================================================

def save_metadata(output_dir, metadata_dict):
    """Save metadata to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata_dict, f, indent=2)


def load_metadata(input_dir):
    """Load metadata from JSON."""
    with open(Path(input_dir) / 'metadata.json', 'r') as f:
        return json.load(f)
