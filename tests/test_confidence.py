#!/usr/bin/env python3
"""
Tests for tracking-confidence propagation.

Reviewer 1 asked that confidence be applied "throughout" rather than only to the
one channel already flagged. These tests pin the property that makes that true:
a confidence problem at one landmark must surface on exactly the joint-angle
channels that depend on it, and on no others.

Run headless:

    QT_QPA_PLATFORM=offscreen PYTHONPATH=. python tests/test_confidence.py
"""

import os
import sys
import warnings

import numpy as np

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

from PyQt6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv)

from core.confidence import (  # noqa: E402
    ANGLE_JOINT_DEPENDENCIES, LOW_CONFIDENCE, angle_confidence, is_synthesised,
    mediapipe_visibility_to_smplh22, quality_label, summarise, tracking_report,
)
from core.joint_angles import SMPLH_JOINT_NAMES  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from gui.panels.angle_comparison import ANGLE_GROUPS  # noqa: E402

# MediaPipe landmark indices used by the tests
MP_LEFT_WRIST, MP_RIGHT_KNEE = 15, 26


def synthetic_visibility(T=400, seed=0):
    """Well-tracked body with an occluded left arm, like a seated recording."""
    rng = np.random.default_rng(seed)
    vis = np.clip(rng.normal(0.95, 0.05, (T, 33)), 0, 1).astype(np.float32)
    vis[:, [13, 15, 17, 19, 21]] = np.clip(rng.normal(0.70, 0.15, (T, 5)), 0, 1)
    return vis


def test_every_gui_angle_has_a_dependency_map():
    """A channel with no map would silently show no confidence at all."""
    gui_angles = {a for group in ANGLE_GROUPS.values() for a in group}
    assert not gui_angles - set(ANGLE_JOINT_DEPENDENCIES), \
        f"unmapped: {gui_angles - set(ANGLE_JOINT_DEPENDENCIES)}"
    assert not set(ANGLE_JOINT_DEPENDENCIES) - gui_angles, \
        f"mapped but never shown: {set(ANGLE_JOINT_DEPENDENCIES) - gui_angles}"


def test_joint_confidence_takes_the_worst_contributing_landmark():
    vis = np.ones((10, 33), dtype=np.float32)
    vis[:, MP_LEFT_WRIST] = 0.2
    joint_conf = mediapipe_visibility_to_smplh22(vis)

    assert abs(joint_conf[0, 20] - 0.2) < 1e-6, "L_WRIST must inherit its landmark"
    assert abs(joint_conf[0, 16] - 1.0) < 1e-6, "L_SHOULDER must be unaffected"

    # A synthesised joint is gated by the worst of its several sources.
    vis2 = np.ones((10, 33), dtype=np.float32)
    vis2[:, 23] = 0.3                      # left hip landmark
    assert abs(mediapipe_visibility_to_smplh22(vis2)[0, 0] - 0.3) < 1e-6


def test_confidence_reaches_dependent_channels_only():
    vis = np.ones((10, 33), dtype=np.float32)
    vis[:, MP_LEFT_WRIST] = 0.2
    conf = angle_confidence(mediapipe_visibility_to_smplh22(vis))

    for affected in ('left_elbow_flexion', 'left_shoulder_rotation',
                     'left_shoulder_flexion'):
        assert abs(conf[affected][0] - 0.2) < 1e-6, affected
    for unaffected in ('right_elbow_flexion', 'right_knee_flexion',
                       'right_shoulder_flexion'):
        assert abs(conf[unaffected][0] - 1.0) < 1e-6, unaffected


def test_leg_and_arm_chains_are_independent():
    vis = np.ones((10, 33), dtype=np.float32)
    vis[:, MP_RIGHT_KNEE] = 0.1
    conf = angle_confidence(mediapipe_visibility_to_smplh22(vis))
    assert abs(conf['right_knee_flexion'][0] - 0.1) < 1e-6
    assert abs(conf['right_hip_flexion'][0] - 0.1) < 1e-6
    assert abs(conf['right_elbow_flexion'][0] - 1.0) < 1e-6
    assert abs(conf['left_knee_flexion'][0] - 1.0) < 1e-6


def test_synthesised_pelvis_channels_are_flagged():
    """MediaPipe has no pelvis landmark; channels relying on it carry a caveat."""
    for name in ('lumbar_extension', 'pelvis_flexion', 'left_hip_flexion'):
        assert is_synthesised(name), name
    for name in ('right_knee_flexion', 'right_elbow_flexion', 'left_ankle_flexion'):
        assert not is_synthesised(name), name


def test_summary_statistics():
    s = summarise(np.array([1.0, 1.0, 0.0, 0.0]))
    assert s['mean'] == 0.5 and s['min'] == 0.0 and s['frac_low'] == 0.5
    empty = summarise(np.array([]))
    assert all(np.isnan(v) for v in empty.values())
    assert quality_label(0.9) == 'good'
    assert quality_label(0.6) == 'fair'
    assert quality_label(0.1) == 'poor'


def test_tracking_report_structure():
    report = tracking_report({'visibility': synthetic_visibility()})
    assert report['available']
    assert set(report['per_joint']) == set(SMPLH_JOINT_NAMES[:22])
    assert set(report['per_angle']) == set(ANGLE_JOINT_DEPENDENCIES)
    assert len(report['worst_landmarks']) == 5
    # The occluded arm should dominate the "least reliable" ranking.
    worst = {n for n, _ in report['worst_landmarks']}
    assert worst & {'left_elbow', 'left_wrist', 'left_shoulder'}, worst


def test_missing_visibility_degrades_gracefully():
    """Skeletons cached before confidence tracking existed must not crash."""
    report = tracking_report({'fps': 30.0})
    assert not report['available'] and 'Re-run extraction' in report['reason']
    assert not tracking_report(None)['available']


def test_tab3_annotates_channels_with_confidence():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_multisubject import fake_subject

    w = MainWindow()
    subject = fake_subject('S1', 200, 400, 1e9)
    subject['skeleton']['visibility'] = synthetic_visibility(T=400)
    w.state['train_subjects'].append(subject)
    w.tab_comparison.refresh()

    boxes = w.tab_comparison.angle_checkboxes
    # Keys stay canonical so the selection logic is unaffected by relabelling.
    assert 'right_elbow_flexion' in boxes
    assert '(' in boxes['right_elbow_flexion'].text()
    assert boxes['lumbar_extension'].toolTip()

    good = boxes['right_elbow_flexion'].text()
    poor = boxes['left_shoulder_rotation'].text()
    parse = lambda t: float(t.split('(')[1].rstrip(') '))
    assert parse(poor) < parse(good), (poor, good)

    w.state['selected_angles'] = []
    boxes['right_elbow_flexion'].setChecked(True)
    assert 'right_elbow_flexion' in w.state['selected_angles']


def test_low_confidence_channels_are_styled_as_a_warning():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_multisubject import fake_subject

    w = MainWindow()
    subject = fake_subject('S1', 200, 400, 1e9)
    vis = np.ones((400, 33), dtype=np.float32)
    vis[:, MP_LEFT_WRIST] = 0.1          # well below LOW_CONFIDENCE
    subject['skeleton']['visibility'] = vis
    w.state['train_subjects'].append(subject)
    w.tab_comparison.refresh()

    boxes = w.tab_comparison.angle_checkboxes
    # Styling moved from inline stylesheets to a QSS property selector, so the
    # flag is now carried by the property that style.qss keys off.
    assert boxes['left_elbow_flexion'].property('lowConfidence') is True
    assert boxes['right_elbow_flexion'].property('lowConfidence') is False
    assert str(LOW_CONFIDENCE) in boxes['left_elbow_flexion'].toolTip()

    # The property is only meaningful if the stylesheet actually targets it.
    from gui.theme import build_stylesheet
    assert 'lowConfidence' in build_stylesheet(), \
        "style.qss has no rule for the lowConfidence property"


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
