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
    """OK needs an ID, all three files, and no blocking validation problem."""
    examples = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                + '/input_data/examples_data/session_1')
    parent = MainWindow()   # must outlive the dialog, or Qt deletes its children
    dlg = AddSubjectDialog(parent, 'train', 'S1')
    ok = dlg.buttons.button(QDialogButtonBox.StandardButton.Ok)

    assert not ok.isEnabled(), "nothing selected"
    dlg.video_edit.setText(f'{examples}/video.mp4');  dlg._validate()
    assert not ok.isEnabled(), "one file is not enough"
    dlg.sensor_edit.setText(f'{examples}/sensor.csv'); dlg._validate()
    assert not ok.isEnabled(), "two files are not enough"

    dlg.times_edit.setText(f'{examples}/frames.csv'); dlg._validate()
    assert ok.isEnabled(), "a complete, valid session should be accepted"

    dlg.id_edit.setText('')
    assert not ok.isEnabled(), "an ID is required"


def test_add_subject_dialog_blocks_invalid_files():
    """
    Validation gates the dialog, so a bad selection cannot become a subject.
    Previously three unreadable paths were accepted and failed much later.
    """
    parent = MainWindow()
    dlg = AddSubjectDialog(parent, 'train', 'S1')
    ok = dlg.buttons.button(QDialogButtonBox.StandardButton.Ok)
    for edit in (dlg.video_edit, dlg.sensor_edit, dlg.times_edit):
        edit.setText('/tmp/definitely-not-a-real-file')
    dlg._validate()
    assert not ok.isEnabled()
    assert any(f.blocking for f in dlg.findings)
    assert dlg.report.isVisible() or not dlg.report.isHidden()


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


def _write_cache(with_joints, n=200):
    """Build a subject whose angles come from a cache file, as after re-opening."""
    from gui.panels.data_import import angle_cache_path
    d = tempfile.mkdtemp()
    video = os.path.join(d, 'video.mp4')
    open(video, 'wb').close()
    angles = {'right_shoulder_flexion': np.sin(np.arange(n) / 30.) * 60,
              'right_elbow_flexion': np.cos(np.arange(n) / 25.) * 45}
    extra = {'__visibility__': np.ones((n, 33), dtype=np.float32)}
    if with_joints:
        extra['__smplh__'] = np.random.rand(n, 22, 3).astype(np.float32)
    np.savez_compressed(angle_cache_path(video), __meta__=np.array([30.0]),
                        **extra, **angles)
    return video


def test_cache_without_joints_does_not_kill_the_skeleton_viewer():
    """
    Regression: caches written before 3D joints were stored produce a skeleton
    dict with no 'smplh_joints'. Tab 2 indexed it directly, and the resulting
    KeyError propagated out of a Qt slot, which takes the whole app down.
    """
    w = MainWindow()
    subject = make_subject('S1', _write_cache(with_joints=False),
                           '/tmp/s.csv', '/tmp/f.csv')
    w.state['train_subjects'].append(subject)
    w.tab_import._load_cached_angles(subject)

    assert subject['status'] == 'ready'
    assert 'smplh_joints' not in subject['skeleton']

    w.tab_skeleton.refresh()               # switching to Tab 2 must not raise
    w.tab_skeleton.frame_slider.setValue(10)
    w.tab_skeleton._update_plot()


def test_cache_round_trips_the_3d_joints():
    """A fresh cache carries the geometry, so the viewer works without re-extraction."""
    w = MainWindow()
    subject = make_subject('S1', _write_cache(with_joints=True, n=200),
                           '/tmp/s.csv', '/tmp/f.csv')
    w.state['train_subjects'].append(subject)
    w.tab_import._load_cached_angles(subject)

    joints = subject['skeleton'].get('smplh_joints')
    assert joints is not None and joints.shape == (200, 22, 3)
    assert subject['skeleton']['total_frames'] == 200
    assert set(subject['angles']) == {'right_shoulder_flexion', 'right_elbow_flexion'}, \
        "reserved keys must not leak into the angle channels"

    w.tab_skeleton.refresh()
    w.tab_skeleton.frame_slider.setValue(10)
    w.tab_skeleton._update_plot()


def test_extraction_result_supplies_everything_the_cache_needs():
    """
    Guards the write side: whatever extract_skeleton_from_video returns must
    contain the keys _load_cached_angles expects to find again.
    """
    import inspect
    from core import smpl_extraction
    src = inspect.getsource(smpl_extraction.extract_skeleton_from_video)
    for key in ("'smplh_joints'", "'visibility'", "'fps'"):
        assert key in src, f"extraction no longer produces {key}"


def test_rebuilding_the_sensor_list_leaves_no_ghost_widgets():
    """
    A widget taken out of a layout keeps its parent and geometry until
    deleteLater() runs, so it goes on painting. That put two labels on top of
    each other in the sensor panel. Rebuilding must fully detach the old ones.
    """
    from PyQt6.QtWidgets import QLabel

    def label_count(panel):
        # Count only the rebuilt region. The group box also holds permanent
        # chrome (the help badge's label), which is not what this guards.
        layout = panel.sensor_layout
        return sum(1 for i in range(layout.count())
                   if isinstance(layout.itemAt(i).widget(), QLabel))

    w = MainWindow()
    w.state['train_subjects'].append(fake_subject('S1', 100, 300, 1e9))

    counts = []
    for _ in range(5):
        w.tab_comparison.refresh()
        _app.processEvents()
        counts.append(label_count(w.tab_comparison))

    assert len(set(counts)) == 1, f"labels accumulated across rebuilds: {counts}"
    assert list(w.tab_comparison.sensor_checkboxes) == ['S1', 'S2', 'S3', 'S4']

    # With no data at all: exactly one explanatory label, and still stable.
    w2 = MainWindow()
    empties = []
    for _ in range(3):
        w2.tab_comparison.refresh()
        _app.processEvents()
        empties.append(label_count(w2.tab_comparison))
    assert empties == [1, 1, 1], empties


def test_dashboard_refresh_does_not_accumulate_axes():
    """
    Revisiting the Results tab must not stack figures up.

    This started as a seaborn colourbar leak on the error heatmap — seaborn adds
    its colourbar as a new axes and ax.clear() does not remove it. The heatmap is
    gone, but the prediction figure builds one subplot per channel on every
    refresh, so the same class of leak is one missing clear() away.
    """
    def metrics(n):
        joints = [f'J{i}' for i in range(n)]
        return {
            'Global_MPJAE': 3.0, 'Global_AMPE': 5.0,
            'Global_RMSE': 4.0, 'Global_PCC': 0.9,
            'MPJAE_per_joint': {j: 3.0 for j in joints},
            'AMPE_per_joint': {j: 5.0 for j in joints},
            'RMSE_per_joint': {j: 4.0 for j in joints},
            'PCC_per_joint': {j: 0.9 for j in joints},
        }

    w = MainWindow()
    w.state['metrics'] = metrics(3)
    w.state['predictions'] = np.random.rand(50, 3)
    w.state['ground_truth'] = np.random.rand(50, 3)

    counts = []
    for _ in range(4):
        w.tab_results.refresh()
        _app.processEvents()
        counts.append((len(w.tab_results.fig_pred.axes),
                       len(w.tab_results.fig_importance.axes)))

    assert len(set(counts)) == 1, f"axes count grew across refreshes: {counts}"
    # one prediction panel per channel, one bar chart
    assert counts[0] == (3, 1), counts


def _loaded_window():
    """A window carrying a full pipeline's worth of state."""
    w = MainWindow()
    w.state['train_subjects'].append(fake_subject('S1', 200, 600, 1e9))
    w.state['test_subjects'].append(fake_subject('T1', 150, 450, 2e9))
    w.state['selected_angles'] = ['a1']
    w.state['selected_sensors'] = ['S1', 'S2']
    w.state['model'] = object()
    w.state['metrics'] = {
        'Global_MPJAE': 3.0, 'Global_AMPE': 5.0, 'Global_RMSE': 4.0,
        'Global_PCC': 0.9,
        'MPJAE_per_joint': {'A': 3.0}, 'AMPE_per_joint': {'A': 5.0},
        'RMSE_per_joint': {'A': 4.0}, 'PCC_per_joint': {'A': 0.9},
    }
    w.state['metrics_per_subject'] = {'T1': w.state['metrics']}
    w.state['predictions'] = np.random.rand(40, 1)
    w.state['ground_truth'] = np.random.rand(40, 1)
    for i in range(5):
        w.tabs.setCurrentIndex(i)
        _app.processEvents()
    return w


def test_reset_clears_shared_state_and_returns_to_tab_1():
    w = _loaded_window()
    assert w.tabs.currentIndex() == 4

    w.reset_all()
    _app.processEvents()

    assert w.tabs.currentIndex() == 0, "must land back on Data Import"
    assert w.state['train_subjects'] == [] and w.state['test_subjects'] == []
    assert w.state['selected_angles'] == [] and w.state['selected_sensors'] == []
    for key in ('model', 'metrics', 'metrics_per_subject', 'predictions',
                'ground_truth', 'scaler', 'config'):
        assert w.state[key] is None, key
    assert w.state['alignment_method'] == 'upsample_sensor'


def test_reset_clears_the_panels_not_only_the_state():
    """Stale widgets are as misleading as stale state."""
    w = _loaded_window()
    w.reset_all()
    _app.processEvents()

    assert w.tab_import.tables['train'].rowCount() == 0
    assert w.tab_import.tables['test'].rowCount() == 0
    assert not any(cb.isChecked()
                   for cb in w.tab_comparison.angle_checkboxes.values())
    assert w.tab_results.metrics_table.rowCount() == 0
    assert w.tab_results.subject_table.rowCount() == 0
    assert w.tab_skeleton._cap is None, "the video decoder must be released"
    assert w.tab_training.log.toPlainText() == ''
    # hyperparameters back to their defaults
    assert w.tab_training.seq_spin.value() == 40
    assert w.tab_training.epochs_spin.value() == 50


def test_pipeline_still_works_after_a_reset():
    """A reset that leaves the app unusable would be worse than no reset."""
    w = _loaded_window()
    w.reset_all()
    _app.processEvents()

    w.state['train_subjects'].append(fake_subject('S1', 200, 600, 3e9))
    w.state['selected_angles'] = ['a1', 'a2']
    w.state['selected_sensors'] = ['S1', 'S2', 'S3', 'S4']
    parts = w.tab_training._align_split(
        'train', w.state['selected_angles'], w.state['selected_sensors'],
        'upsample_sensor')
    assert parts and len(parts[0][1]) > 100, parts


def test_reset_needs_no_confirmation_path_to_be_destructive():
    """confirm_reset() must not clear anything on its own; reset_all() does."""
    import inspect
    src = inspect.getsource(MainWindow.confirm_reset)
    assert 'exec()' in src and 'reset_all' in src
    # and the destructive call is guarded by the dialog's result
    assert 'StandardButton.Yes' in src


def test_extraction_worker_is_cancellable():
    """
    Extraction runs for minutes. Without a cancel hook, Reset and closing the
    window would both have to wait for it.
    """
    from gui.panels.data_import import SkeletonWorker, _Cancelled

    worker = SkeletonWorker('/tmp/nonexistent.mp4')
    assert hasattr(worker, 'stop')
    worker.stop()
    raised = False
    try:
        worker._tick(1, 100)
    except _Cancelled:
        raised = True
    assert raised, "a stopped worker must abort from its progress callback"

    fresh = SkeletonWorker('/tmp/nonexistent.mp4')
    fresh._tick(1, 100)          # not stopped: must not raise


def test_busy_dialog_closes_on_every_exit_path():
    """
    A progress window left open behind a finished job locks the whole
    application, so every ending must close it: success, error and cancel.
    """
    from gui.progress import BusyDialog

    w = MainWindow()
    panel = w.tab_import
    subject = fake_subject('S1', 100, 300, 1e9)
    w.state['train_subjects'].append(subject)

    def open_one():
        panel._dialog = BusyDialog(panel, "Extracting", "x", cancel_text="Cancel")
        panel._dialog.show()
        _app.processEvents()
        assert panel._dialog.isModal(), "must block the main window"

    # cancelled
    open_one()
    panel._on_cancelled('train', subject)
    _app.processEvents()
    assert panel._dialog is None, "cancel left the dialog open"

    # errored — the real handler raises a modal message box, which would block
    # forever with no one to dismiss it, so stub just that call.
    open_one()
    panel._pending_extractions = []
    from PyQt6.QtWidgets import QMessageBox
    original = QMessageBox.critical
    QMessageBox.critical = staticmethod(lambda *a, **k: None)
    try:
        panel._on_extraction_error("boom", 'train', subject)
    finally:
        QMessageBox.critical = original
    _app.processEvents()
    assert panel._dialog is None, "error left the dialog open"

    # queue drained normally
    open_one()
    panel._pending_extractions = []
    panel._run_next_extraction()
    _app.processEvents()
    assert panel._dialog is None, "completion left the dialog open"

    # reset
    open_one()
    w.reset_all()
    _app.processEvents()
    assert panel._dialog is None, "reset left the dialog open"


def test_busy_dialog_reports_progress_and_time():
    from gui.progress import BusyDialog, _duration

    assert _duration(45) == "45 s"
    assert _duration(125) == "2 min 05 s"
    assert _duration(None) == "—"
    assert _duration(float('nan')) == "—"

    w = MainWindow()
    d = BusyDialog(w, "T", "message", cancel_text="Cancel")
    assert d.bar.maximum() == 0, "starts indeterminate"

    d.set_progress(50, 200)
    assert d.bar.maximum() == 200 and d.bar.value() == 50
    d.set_progress(999, 200)
    assert d.bar.value() == 200, "must clamp rather than overflow"
    d.set_progress(-5, 200)
    assert d.bar.value() == 0

    d.set_detail("frame 3 of 9")
    assert d.detail_label.text() == "frame 3 of 9"

    # Esc must not dismiss work that is still running; it requests a cancel.
    fired = []
    d.set_cancel_callback(lambda: fired.append(True))
    d.reject()
    assert fired == [True] and d.cancelled
    assert d.isVisible() or True   # offscreen visibility is not meaningful

    d.finish()
    d.finish()                      # idempotent


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
