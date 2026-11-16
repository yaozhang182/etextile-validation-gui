"""
Tests for picking bones (Tab 3) and the pose comparison (Tab 5).

Run: QT_QPA_PLATFORM=offscreen python tests/test_pose.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication  # noqa: E402
_app = QApplication.instance() or QApplication(sys.argv)

from core.bones import (  # noqa: E402
    BONES, DIRECTION_BONE, bone_of_target, predicted_directions, reconstruct,
    selection_state,
)
from core.joint_angles import extract_all_joint_angles  # noqa: E402
from core.targets import COMPONENTS, TARGET_CHANNELS, angle_between, expand_targets  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from gui.panels.data_import import read_csv_auto  # noqa: E402
from gui.state import make_subject  # noqa: E402
from test_joint_angles import EXAMPLE, MIRROR_PERM  # noqa: E402

SESSION = os.path.join(ROOT, 'input_data', 'examples_data', 'session_3')
R_SH, R_EL, R_WR = 17, 19, 21


def _true_directions(joints):
    a, _ = extract_all_joint_angles(joints)
    return a, {b: np.stack([a[f'{BONES[b][3]}_{c}'] for c in COMPONENTS], axis=1)
               for b in DIRECTION_BONE.values()}


def _turn(v, deg):
    """Turn each unit vector of (N, 3) by exactly `deg` degrees (about an axis
    perpendicular to it)."""
    v = np.asarray(v, float)
    k = np.cross(v, [0.0, 0.0, 1.0])
    small = np.linalg.norm(k, axis=1) < 1e-6
    k[small] = np.cross(v[small], [1.0, 0.0, 0.0])
    k /= np.linalg.norm(k, axis=1, keepdims=True)
    t = np.radians(deg)
    return v * np.cos(t) + np.cross(k, v) * np.sin(t)


# --- core.bones -------------------------------------------------------------

def test_every_target_belongs_to_exactly_one_bone():
    for name in TARGET_CHANNELS:
        owners = [b for b, spec in BONES.items() if name in spec[4]]
        assert len(owners) == 1, (name, owners)
        assert bone_of_target(name) == owners[0]
    assert bone_of_target('right_shoulder_flexion') is None, "display-only has no bone"


def test_true_directions_rebuild_the_true_skeleton():
    joints = np.load(EXAMPLE)['__smplh__'].astype(float)
    for j in (joints, joints[:, MIRROR_PERM] * [-1, 1, 1]):      # also mirrored
        _, dirs = _true_directions(j)
        assert np.abs(reconstruct(j, dirs) - j).max() < 1e-9


def test_a_wrong_upper_arm_carries_the_forearm_along():
    joints = np.load(EXAMPLE)['__smplh__'].astype(float)[:50]
    _, dirs = _true_directions(joints)
    wrong = _turn(dirs['right_upper_arm'], 30)      # 30 deg off
    out = reconstruct(joints, {'right_upper_arm': wrong})
    upper = out[:, R_EL] - out[:, R_SH]
    true_upper = joints[:, R_EL] - joints[:, R_SH]
    assert np.allclose(np.linalg.norm(upper, axis=1), np.linalg.norm(true_upper, axis=1))
    assert np.allclose(angle_between(upper, true_upper), 30, atol=1e-6)
    # the forearm keeps its true direction and length, hung from the new elbow
    assert np.allclose(out[:, R_WR] - out[:, R_EL], joints[:, R_WR] - joints[:, R_EL])
    # everything else untouched
    others = [k for k in range(22) if k not in (R_EL, R_WR)]
    assert np.array_equal(out[:, others], joints[:, others])


def test_predicted_directions_are_read_from_the_model_output():
    names = ['right_shoulder_elevation', 'right_upper_arm_direction', 'right_elbow_flexion']
    pred = np.arange(10 * len(expand_targets(names)), dtype=float).reshape(10, -1)
    d = predicted_directions(pred, names)
    assert list(d) == ['right_upper_arm'] and np.array_equal(d['right_upper_arm'], pred[:, 1:4])


def test_selection_state():
    assert selection_state('right_upper_arm', []) == 'none'
    assert selection_state('right_upper_arm', ['right_upper_arm_direction']) == 'some'
    assert selection_state('right_upper_arm', BONES['right_upper_arm'][4]) == 'all'


# --- Tab 3 --------------------------------------------------------------------

def _subject(sid='S3'):
    s = make_subject(sid, os.path.join(SESSION, 'video.mp4'),
                     os.path.join(SESSION, 'sensor.csv'), os.path.join(SESSION, 'frames.csv'))
    s['sensor_df'] = read_csv_auto(s['sensor_path'])
    s['video_df'] = read_csv_auto(s['video_csv_path'])
    return s


def _click_bone(picker, bone):
    from matplotlib.backend_bases import MouseEvent
    line = picker.lines[bone]
    xy = np.mean(np.asarray(line.get_xydata()), axis=0)
    x, y = picker.ax.transData.transform(xy)
    picker.callbacks.process('button_press_event',
                             MouseEvent('button_press_event', picker, x, y, button=1))
    _app.processEvents()


def test_clicking_a_bone_picks_its_targets_and_clicking_again_drops_them():
    w = MainWindow()
    w.resize(1280, 860)
    w.show()
    tab = w.tab_comparison
    w.tabs.setCurrentWidget(tab)
    _app.processEvents()
    tab.bone_picker.draw()
    _click_bone(tab.bone_picker, 'right_upper_arm')
    assert w.state['selected_angles'] == ['right_shoulder_elevation',
                                          'right_upper_arm_direction']
    assert tab.angle_checkboxes['right_upper_arm_direction'].isChecked()
    assert 'Right upper arm' in tab.bone_picker.summary_text()
    _click_bone(tab.bone_picker, 'right_forearm')
    assert set(w.state['selected_angles']) >= {'right_forearm_direction',
                                               'right_elbow_flexion'}
    _click_bone(tab.bone_picker, 'right_upper_arm')
    assert w.state['selected_angles'] == ['right_forearm_direction', 'right_elbow_flexion']


def test_list_and_skeleton_show_the_same_selection():
    w = MainWindow()
    tab = w.tab_comparison
    tab.angle_checkboxes['right_upper_arm_direction'].setChecked(True)
    assert tab.bone_picker.state('right_upper_arm') == 'some'
    assert tab.bone_picker.halos['right_upper_arm'].get_linestyle() == '--'
    tab.angle_checkboxes['right_shoulder_elevation'].setChecked(True)
    assert tab.bone_picker.state('right_upper_arm') == 'all'
    # clicking a partly picked bone picks all of it (here: already all -> drops)
    tab._on_bone_clicked('right_upper_arm')
    assert not tab.angle_checkboxes['right_upper_arm_direction'].isChecked()
    tab.view_buttons.button(1).click()
    assert tab.target_stack.currentIndex() == 1


# --- Tab 5 --------------------------------------------------------------------

def _window_with_predictions(names, rotate_upper_arm=20.0):
    """
    The pose tab fed as training would feed it: test subject S3, a prediction
    for every frame, equal to the truth except the right upper arm, which is
    turned by `rotate_upper_arm` degrees.
    """
    w = MainWindow()
    w.resize(1400, 950)
    w.show()
    s = _subject()
    w.state['test_subjects'].append(s)
    w.tab_import._load_cached_angles(s)
    assert s['status'] == 'ready'
    cols = expand_targets(names)
    truth = np.stack([s['angles'][c] for c in cols], axis=1)
    keep = np.isfinite(truth).all(axis=1)
    from gui.panels.angle_comparison import _frame_epochs
    n = len(truth)
    times = _frame_epochs(s['video_df'], n)[keep]
    truth = truth[keep]
    pred = truth.copy()
    if 'right_upper_arm_direction' in names:
        k = cols.index('right_upper_arm_direction_fwd')
        pred[:, k:k + 3] = _turn(truth[:, k:k + 3], rotate_upper_arm)
    frames = w.tab_training._frames_of('S3', times)
    w.state['config'] = {'target_names': names}
    w.state['metrics'] = {'Global_MPJAE': 0.0}
    w.state['predictions_per_subject'] = {
        'S3': {'truth': truth, 'pred': pred, 'times': times, 'frames': frames}}
    w.tabs.setCurrentWidget(w.tab_pose)
    _app.processEvents()
    return w, s, frames


def test_predictions_are_matched_to_their_video_frames():
    w, s, frames = _window_with_predictions(['right_upper_arm_direction'])
    n = len(s['angles']['right_elbow_flexion'])
    keep = np.isfinite(s['angles']['right_upper_arm_direction_fwd'])
    assert np.array_equal(frames, np.flatnonzero(keep)), "every frame maps to itself"
    assert frames.max() < n


def test_the_predicted_bone_is_redrawn_and_the_rest_copied():
    w, s, frames = _window_with_predictions(
        ['right_shoulder_elevation', 'right_upper_arm_direction', 'right_elbow_flexion'])
    p = w.tab_pose
    assert p._data is not None and p._data['bones'] == {'right_upper_arm'}
    f = int(frames[len(frames) // 2])
    p.scrubber.set_frame(f)
    _app.processEvents()
    art = p._data['art']
    def seg(lines, pair):
        x, y, z = lines[pair].get_data_3d()
        # back to humanml3d: plot (-x, z, y)
        return np.array([[-x[i], z[i], y[i]] for i in range(2)])
    true = s['skeleton']['smplh_joints'][f].astype(float)
    shown_true = seg(art['true'], (R_SH, R_EL))
    assert np.allclose(shown_true, true[[R_SH, R_EL]], atol=1e-6)
    shown_pred = seg(art['pred'], (R_SH, R_EL))
    err = angle_between((shown_pred[1] - shown_pred[0])[None], (true[R_EL] - true[R_SH])[None])
    assert abs(err[0] - 20) < 1e-3, err      # turned by 20 deg in the body frame
    assert np.allclose(seg(art['pred'], (2, 5)), seg(art['true'], (2, 5))), "legs copied"
    assert "off by 20.0" in p.values_label.text(), p.values_label.text()
    assert "predicted" in p.values_label.text() and "right_elbow_flexion" in p.values_label.text()
    assert art['err']['right_upper_arm'].get_text() == "20°"


def test_turning_one_skeleton_turns_the_other():
    from matplotlib.backend_bases import MouseEvent
    w, s, frames = _window_with_predictions(['right_upper_arm_direction'])
    p = w.tab_pose
    p.canvas.draw()
    p.ax_true.view_init(elev=33, azim=-20)
    x, y = p.ax_true.transAxes.transform((0.5, 0.5))
    ev = MouseEvent('button_release_event', p.canvas, x, y, button=1)
    p.canvas.callbacks.process('button_release_event', ev)
    assert (p.ax_pred.elev, p.ax_pred.azim) == (33, -20)


def test_angle_only_targets_are_listed_not_drawn():
    w, s, frames = _window_with_predictions(['right_elbow_flexion'])
    p = w.tab_pose
    assert p._data['bones'] == set()
    assert "No direction target" in p.info_label.text()
    p.scrubber.set_frame(int(frames[10]))
    _app.processEvents()
    assert "right_elbow_flexion" in p.values_label.text()


def test_the_tab_explains_itself_before_training():
    w = MainWindow()
    w.tabs.setCurrentWidget(w.tab_pose)
    _app.processEvents()
    assert "Train a model in Tab 4" in w.tab_pose.info_label.text()
    w.tab_pose.reset()


def main():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback; traceback.print_exc()
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
