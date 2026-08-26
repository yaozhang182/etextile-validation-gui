#!/usr/bin/env python3
"""
Regression tests for multi-subject import and time alignment.

The bug these guard against was silent: the GUI used to pair sensor rows with
video frames by array index, so a 9.6 Hz sensor stream was matched against 25 fps
joint angles and every reported metric was computed from misaligned data. Nothing
crashed — the numbers were just wrong. These tests assert the structural
invariants instead of any particular accuracy.

Run headless (no display needed, no training performed):

    QT_QPA_PLATFORM=offscreen PYTHONPATH=. python tests/test_multisubject.py
"""

import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from PyQt6.QtWidgets import QApplication, QDialogButtonBox  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from core.data_alignment import (  # noqa: E402
    ShoulderDataset, align_and_load, build_supervised_arrays,
    estimate_fps_from_frames,
)
from gui.main_window import MainWindow  # noqa: E402
from gui.panels.data_import import (  # noqa: E402
    AddSubjectDialog, angle_cache_path, read_csv_auto,
)
from gui.state import (  # noqa: E402
    common_sensor_columns, inconsistent_sensor_subjects, make_subject,
    next_subject_id,
)

SENSOR_HZ = 9.6
VIDEO_FPS = 30.0


def fake_subject(sid, n_sensor, n_frames, t0, cols=('S1', 'S2', 'S3', 'S4'),
                 sensor_t0=None):
    """A ready-to-use subject with realistic mismatched sensor/video rates."""
    st = (t0 if sensor_t0 is None else sensor_t0) + np.arange(n_sensor) / SENSOR_HZ
    vt = t0 + np.arange(n_frames) / VIDEO_FPS
    s = make_subject(sid, f'/tmp/{sid}.mp4', f'/tmp/{sid}_s.csv', f'/tmp/{sid}_v.csv')
    s['sensor_df'] = pd.DataFrame(
        {'EpochTime': st, **{c: np.sin(st * (i + 1)) for i, c in enumerate(cols)}}
    )
    s['video_df'] = pd.DataFrame({'FrameIndex': np.arange(n_frames), 'EpochTime': vt})
    s['angles'] = {'a1': np.sin(np.arange(n_frames) / 30.) * 60,
                   'a2': np.cos(np.arange(n_frames) / 25.) * 45}
    s['time'] = np.arange(n_frames) / VIDEO_FPS
    s['skeleton'] = {'total_frames': n_frames, 'fps': VIDEO_FPS,
                     'valid_frames': n_frames,
                     'smplh_joints': np.zeros((n_frames, 22, 3))}
    s['status'] = 'ready'
    return s


def test_alignment_resamples_onto_the_chosen_clock():
    """Sensor and angle streams must end up on one clock, not paired by index."""
    s = fake_subject('S1', 300, 900, 1e9)
    names, cols = ['a1', 'a2'], ['S1', 'S2', 'S3', 'S4']

    X, Y, t = build_supervised_arrays(s['sensor_df'], s['video_df'], s['angles'],
                                      names, cols, method='upsample_sensor')
    assert len(X) == len(Y) == len(t)
    assert abs(len(t) / (t[-1] - t[0]) - VIDEO_FPS) < 1.0, "should land on video rate"

    X2, Y2, t2 = build_supervised_arrays(s['sensor_df'], s['video_df'], s['angles'],
                                         names, cols, method='downsample_labels')
    assert abs(len(t2) / (t2[-1] - t2[0]) - SENSOR_HZ) < 1.0, "should land on sensor rate"

    # Both cover the same wall-clock window, only the sample rate differs.
    assert abs((t[-1] - t[0]) - (t2[-1] - t2[0])) < 0.5


def test_selected_sensor_columns_are_honoured():
    s = fake_subject('S1', 300, 900, 1e9)
    X, _, _ = build_supervised_arrays(s['sensor_df'], s['video_df'], s['angles'],
                                      ['a1'], ['S2', 'S4'])
    assert X.shape[1] == 2

    try:
        build_supervised_arrays(s['sensor_df'], s['video_df'], s['angles'],
                                ['a1'], ['S1', 'MISSING'])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown sensor column should raise")


def test_windows_never_straddle_two_subjects():
    """The core multi-subject invariant: window per subject, then concatenate."""
    from torch.utils.data import ConcatDataset

    seq = 20
    a = fake_subject('S1', 200, 600, 1e9)
    b = fake_subject('S2', 150, 450, 2e9)   # a different session, far away in time
    parts = [build_supervised_arrays(s['sensor_df'], s['video_df'], s['angles'],
                                     ['a1', 'a2'], ['S1', 'S2', 'S3', 'S4'])
             for s in (a, b)]

    sets = [ShoulderDataset(X, Y, seq) for X, Y, _ in parts]
    cat = ConcatDataset(sets)

    expected = sum(max(0, len(X) - seq) for X, _, _ in parts)
    assert len(cat) == expected, (len(cat), expected)
    # Pooling first would yield seq extra windows spanning the boundary.
    pooled = np.vstack([X for X, _, _ in parts])
    assert len(cat) < max(0, len(pooled) - seq)


def test_short_recording_yields_no_windows_instead_of_raising():
    """len() must not go negative when a recording is shorter than one window."""
    ds = ShoulderDataset(np.zeros((5, 4)), np.zeros((5, 2)), seq_len=40)
    assert len(ds) == 0


def test_subject_with_no_time_overlap_is_skipped_not_fatal():
    w = MainWindow()
    w.state['train_subjects'] += [
        fake_subject('BAD', 100, 300, 1e9, sensor_t0=9e9),   # disjoint clocks
        fake_subject('GOOD', 200, 600, 2e9),
    ]
    parts = w.tab_training._align_split(
        'train', ['a1', 'a2'], ['S1', 'S2', 'S3', 'S4'], 'upsample_sensor'
    )
    assert [p[0] for p in parts] == ['GOOD']
    log = w.tab_training.log.toPlainText()
    assert 'BAD' in log and 'skipped' in log


def test_only_channels_common_to_all_subjects_are_offered():
    w = MainWindow()
    w.state['train_subjects'] += [
        fake_subject('A', 100, 300, 1e9, cols=('S1', 'S2', 'S3', 'S4')),
        fake_subject('B', 100, 300, 2e9, cols=('S1', 'S2')),
    ]
    assert common_sensor_columns(w.state) == ['S1', 'S2']
    assert inconsistent_sensor_subjects(w.state) == [('train', 'A', ['S3', 'S4'])]

    w.tab_comparison.refresh()
    assert list(w.tab_comparison.sensor_checkboxes) == ['S1', 'S2']


def test_subject_ids_autoincrement_per_split():
    w = MainWindow()
    w.state['train_subjects'] += [fake_subject('S1', 10, 30, 1e9),
                                  fake_subject('S3', 10, 30, 2e9)]
    assert next_subject_id(w.state, 'train') == 'S2'
    assert next_subject_id(w.state, 'test') == 'S1'


def test_every_tab_refreshes_with_no_data():
    w = MainWindow()
    for tab in (w.tab_import, w.tab_skeleton, w.tab_comparison,
                w.tab_training, w.tab_results):
        tab.refresh()


def test_add_subject_dialog_requires_all_three_files():
    parent = MainWindow()   # must outlive the dialog, or Qt deletes its children
    dlg = AddSubjectDialog(parent, 'train', 'S1')
    ok = dlg.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()
    for edit in (dlg.video_edit, dlg.sensor_edit):
        edit.setText('/tmp/x')
        dlg._validate()
        assert not ok.isEnabled()
    dlg.times_edit.setText('/tmp/x')
    dlg._validate()
    assert ok.isEnabled()
    dlg.id_edit.setText('')
    assert not ok.isEnabled()


def test_angle_cache_roundtrip_and_corruption():
    w = MainWindow()
    d = tempfile.mkdtemp()
    video = os.path.join(d, 'sub.mp4')
    open(video, 'wb').close()

    angles = {'a1': np.arange(300, dtype=float), 'a2': np.arange(300, dtype=float) * 2}
    np.savez_compressed(angle_cache_path(video), __meta__=np.array([VIDEO_FPS]), **angles)

    s = make_subject('S1', video, '/tmp/s.csv', '/tmp/t.csv')
    w.tab_import._load_cached_angles(s)
    assert s['status'] == 'ready'
    assert set(s['angles']) == {'a1', 'a2'}, "meta key must not leak into angles"
    assert abs(s['time'][-1] - 299 / VIDEO_FPS) < 1e-9

    with open(angle_cache_path(video), 'wb') as f:
        f.write(b'not an npz')
    s2 = make_subject('S2', video, '/tmp/s.csv', '/tmp/t.csv')
    w.tab_import._load_cached_angles(s2)
    assert s2['status'] == 'pending', "corrupt cache must fall back to extraction"


def test_csv_separator_autodetection():
    d = tempfile.mkdtemp()
    semi = os.path.join(d, 'semi.csv')
    comma = os.path.join(d, 'comma.csv')
    open(semi, 'w').write("EpochTime;S1\n1.0;2.0\n")
    open(comma, 'w').write("EpochTime,S1\n1.0,2.0\n")
    assert list(read_csv_auto(semi).columns) == ['EpochTime', 'S1']
    assert list(read_csv_auto(comma).columns) == ['EpochTime', 'S1']


def test_align_and_load_stays_backwards_compatible():
    """main_pipeline.py calls this without sensor_cols."""
    s = fake_subject('X', 300, 900, 1e9)
    labels = np.stack([s['angles']['a1'], s['angles']['a2']], axis=1)
    X, _, _ = align_and_load(s['sensor_df'], s['video_df'], labels)
    assert X.shape[1] == 4, "all non-timestamp columns auto-detected"


def test_fps_is_derived_from_timestamps_not_the_container():
    """
    The example MP4s declare 25 fps but were captured at ~30. The timestamps are
    authoritative; a wrong container tag would stretch the displayed time axis.
    """
    n_frames = 600
    true_fps = 30.0
    video_df = pd.DataFrame({
        'FrameIndex': np.arange(n_frames),
        'EpochTime': 1e9 + np.arange(n_frames) / true_fps,
    })
    assert abs(estimate_fps_from_frames(video_df) - true_fps) < 0.05

    # Without FrameIndex it falls back to row spacing.
    assert abs(estimate_fps_from_frames(video_df[['EpochTime']]) - true_fps) < 0.05

    # A timestamp log that skipped frames: FrameIndex keeps the rate honest.
    sparse = video_df.iloc[::2].reset_index(drop=True)
    assert abs(estimate_fps_from_frames(sparse) - true_fps) < 0.05


def test_fps_estimation_degrades_safely():
    for bad in (None,
                pd.DataFrame({'EpochTime': [1.0]}),                  # one row
                pd.DataFrame({'nope': [1.0, 2.0]}),                  # no EpochTime
                pd.DataFrame({'EpochTime': [5.0, 5.0, 5.0]})):       # zero span
        assert estimate_fps_from_frames(bad, fallback=25.0) == 25.0

    # Implausible rates are rejected rather than propagated.
    absurd = pd.DataFrame({'EpochTime': 1e9 + np.arange(100) / 1e6})
    assert estimate_fps_from_frames(absurd, fallback=25.0) == 25.0


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
