"""
Tests for the continuous regression targets (core/targets.py).

Run: python tests/test_targets.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.joint_angles import extract_all_joint_angles  # noqa: E402
from core.targets import (  # noqa: E402
    COMPONENTS, DISPLAY_GROUPS, DISPLAY_ONLY, ELEVATION_TARGETS, TARGET_CHANNELS,
    VECTOR_TARGETS, angle_between, display_only_reason, evaluate_targets, expand_targets, postprocess, target_slices,
)
from test_joint_angles import EXAMPLE, MIRROR_PERM, standing_skeleton, _place_right_arm  # noqa: E402

SIDES = ('left', 'right')


def _example():
    joints = np.load(EXAMPLE)['__smplh__'].astype(float)
    return joints, extract_all_joint_angles(joints)[0]


def _direction(angles, limb):
    return np.stack([angles[f'{limb}_direction_{c}'] for c in COMPONENTS], axis=1)


def test_directions_and_elevations_agree_with_flexion_and_abduction():
    """Where v4 defines flexion, direction = (cos a sin f, -cos a cos f, -sin a)."""
    joints, a = _example()
    for side in SIDES:
        for joint, limb in (('shoulder', f'{side}_upper_arm'), ('hip', f'{side}_thigh')):
            f = np.radians(a[f'{side}_{joint}_flexion'])
            ab = np.radians(a[f'{side}_{joint}_abduction'])
            defined = np.abs(np.degrees(ab)) < 66.4      # v4 interpolates beyond
            expected = np.stack([np.cos(ab) * np.sin(f), -np.cos(ab) * np.cos(f),
                                 -np.sin(ab)], axis=1)
            d = _direction(a, limb)
            assert np.allclose(d[defined], expected[defined], atol=1e-9), limb
            elev = np.degrees(np.arccos(np.cos(ab) * np.cos(f)))
            assert np.allclose(a[f'{side}_{joint}_elevation'][defined], elev[defined],
                               atol=1e-6), limb


def test_targets_are_unit_vectors_and_elevations_in_range():
    joints, a = _example()
    for name, comps in VECTOR_TARGETS.items():
        d = np.stack([a[c] for c in comps], axis=1)
        assert np.allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-9), name
    for name in ELEVATION_TARGETS:
        assert a[name].min() >= 0 and a[name].max() <= 180, name


def test_side_raise_is_continuous_where_flexion_is_not():
    """
    A side raise through horizontal: flexion is undefined there (and steps from
    about 0 to 180), elevation and direction move smoothly.
    """
    base = standing_skeleton()
    frames = []
    for t in np.linspace(0, 175, 60):
        f = base.copy()
        r = np.radians(t)
        _place_right_arm(f, [-np.sin(r), -np.cos(r), 0.0], [-np.sin(r), -np.cos(r), 0.01])
        frames.append(f)
    a, _ = extract_all_joint_angles(np.array(frames))
    elev = a['right_shoulder_elevation']
    assert np.max(np.abs(np.diff(elev))) < 5, np.max(np.abs(np.diff(elev)))
    assert abs(elev[0] - 0) < 3 and abs(elev[-1] - 175) < 3, (elev[0], elev[-1])
    d = _direction(a, 'right_upper_arm')
    step = angle_between(d[1:], d[:-1])
    assert step.max() < 5, step.max()
    # outward first grows then shrinks; it never points inward
    assert d[:, 2].min() > -0.05 and d[:, 2].max() > 0.99


def test_mirrored_motion_gives_mirrored_targets():
    """Both sides read the same: + outward means away from the midline."""
    joints, a = _example()
    m = joints[:, MIRROR_PERM].copy()
    m[..., 0] *= -1
    b, _ = extract_all_joint_angles(m)
    for name in list(ELEVATION_TARGETS) + [c for comps in VECTOR_TARGETS.values() for c in comps]:
        other = name.replace('left_', 'X').replace('right_', 'left_').replace('X', 'right_')
        assert np.allclose(a[name], b[other], atol=1e-6), name


def test_display_only_channels_are_never_targets():
    assert not set(DISPLAY_ONLY) & set(TARGET_CHANNELS)
    for side in SIDES:
        for dof in ('flexion', 'abduction', 'rotation'):
            assert f'{side}_shoulder_{dof}' in DISPLAY_ONLY


def test_trunk_channels_a_webcam_cannot_measure_are_display_only():
    """
    MediaPipe has no spine: the reconstructed pelvis lies on the hip-to-neck
    line, so lumbar extension is identically 0, and the pelvis angles are the
    trunk's orientation relative to the camera.
    """
    joints, a = _example()
    assert np.max(np.abs(a['lumbar_extension'])) < 1e-3   # float32 joints
    for name in ('lumbar_extension', 'pelvis_flexion', 'pelvis_adduction',
                 'pelvis_rotation'):
        assert name in DISPLAY_ONLY and name not in TARGET_CHANNELS, name
    for name in ('lumbar_bending', 'lumbar_rotation', 'neck_flexion'):
        assert name in TARGET_CHANNELS, name
    assert 'always 0' in display_only_reason('lumbar_extension')
    assert 'camera' in display_only_reason('pelvis_flexion')
    assert display_only_reason('right_elbow_flexion') == ''


def test_every_channel_is_either_a_target_or_display_only():
    from core.joint_angles import ANGLE_CHANNELS
    shown = [n for names in DISPLAY_GROUPS.values() for n in names]
    assert sorted(shown) == sorted(DISPLAY_ONLY)
    assert all(display_only_reason(n) for n in DISPLAY_ONLY)
    for name in ANGLE_CHANNELS:
        assert (name in TARGET_CHANNELS) != (name in DISPLAY_ONLY), name


def test_expansion_and_slices():
    names = ['left_shoulder_elevation', 'left_upper_arm_direction', 'left_elbow_flexion']
    cols = expand_targets(names)
    assert cols == ['left_shoulder_elevation', 'left_upper_arm_direction_fwd',
                    'left_upper_arm_direction_up', 'left_upper_arm_direction_out',
                    'left_elbow_flexion']
    assert [(n, s.start, s.stop) for n, s in target_slices(names)] == [
        ('left_shoulder_elevation', 0, 1), ('left_upper_arm_direction', 1, 4),
        ('left_elbow_flexion', 4, 5)]


def test_postprocess_normalises_directions_only():
    names = ['left_elbow_flexion', 'left_upper_arm_direction']
    pred = np.array([[30.0, 0.0, -2.0, 0.0], [45.0, 3.0, 0.0, 4.0]])
    out = postprocess(pred, names)
    assert np.allclose(out[:, 0], [30.0, 45.0])
    assert np.allclose(out[0, 1:], [0, -1, 0]) and np.allclose(out[1, 1:], [0.6, 0, 0.8])


def test_metrics_measure_the_angle_between_directions():
    names = ['right_upper_arm_direction', 'right_elbow_flexion']
    n = 200
    t = np.linspace(0, np.pi, n)
    truth_dir = np.stack([np.sin(t), -np.cos(t), np.zeros(n)], axis=1)
    # rotate every predicted direction by exactly 10 degrees about the outward axis
    r = np.radians(10)
    pred_dir = np.stack([np.sin(t + r), -np.cos(t + r), np.zeros(n)], axis=1)
    truth = np.column_stack([truth_dir, np.full(n, 40.0)])
    pred = np.column_stack([pred_dir * 3.0, np.full(n, 43.0)])   # unnormalised output
    m = evaluate_targets(truth, pred, names)
    assert abs(m['MPJAE_per_joint']['Right Upper Arm Direction'] - 10) < 1e-6
    assert abs(m['MPJAE_per_joint']['Right Elbow Flexion'] - 3) < 1e-9
    assert abs(m['Global_MPJAE'] - 6.5) < 1e-6
    assert m['target_kinds']['Right Upper Arm Direction'] == 'direction'
    perfect = evaluate_targets(truth, truth, names)
    assert perfect['Global_MPJAE'] < 1e-6


def main():
    tests = [v for k, v in globals().items() if k.startswith('test_') and callable(v)]
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
