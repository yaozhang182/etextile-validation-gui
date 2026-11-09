"""
Tests for the joint angle channels (core/joint_angles.py, SMD v4 definitions).

Run: python tests/test_joint_angles.py
"""

import hashlib
import importlib.util
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core import smd_angle_calculator as v4  # noqa: E402
from core.joint_angles import (  # noqa: E402
    ANGLE_CHANNELS, SMD_ANGLE_SOURCE_MD5, V4_NAMES, WRAPPABLE_ANGLES,
    extract_all_joint_angles, unwrap_angles,
)

SOURCE = ('/scratch/elec/t412-humanmotion/yaozhang/smd_reasoning/'
          'smd_core/angle_calculator.py')
EXAMPLE = os.path.join(ROOT, 'input_data', 'examples_data', 'session_1',
                       'video.angles.npz')

# SMPL 22-joint indices
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT = 1, 2, 4, 5, 7, 8, 10, 11
NECK, L_COLLAR, R_COLLAR, HEAD = 12, 13, 14, 15
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 16, 17, 18, 19, 20, 21
MIRROR_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]


def _md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def standing_skeleton():
    """A neutral standing pose, X = the person's left, Y = up, Z = forward (metres)."""
    j = np.zeros((22, 3))
    j[0] = [0, 0.95, 0]
    j[L_HIP], j[R_HIP] = [0.09, 0.87, 0], [-0.09, 0.87, 0]
    j[3], j[6], j[9] = [0, 1.05, 0], [0, 1.18, 0], [0, 1.32, 0]
    j[L_KNEE], j[R_KNEE] = [0.10, 0.48, 0.02], [-0.10, 0.48, 0.02]
    j[L_ANKLE], j[R_ANKLE] = [0.10, 0.08, 0], [-0.10, 0.08, 0]
    j[L_FOOT], j[R_FOOT] = [0.12, 0.02, 0.13], [-0.12, 0.02, 0.13]
    j[NECK], j[HEAD] = [0, 1.50, 0], [0, 1.62, 0.05]
    j[L_COLLAR], j[R_COLLAR] = [0.07, 1.45, 0], [-0.07, 1.45, 0]
    j[L_SH], j[R_SH] = [0.18, 1.42, 0], [-0.18, 1.42, 0]
    j[L_EL], j[R_EL] = [0.20, 1.14, 0], [-0.20, 1.14, -0.0]
    j[L_WR], j[R_WR] = [0.20, 0.98, 0.20], [-0.20, 0.98, 0.20]   # elbows bent 90
    return j


def _place_right_arm(frame, direction, forearm):
    """Put the right upper arm along `direction` with the given forearm direction."""
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    f = np.asarray(forearm, float); f /= np.linalg.norm(f)
    frame[R_EL] = frame[R_SH] + 0.28 * d
    frame[R_WR] = frame[R_EL] + 0.25 * f


# --- the vendored copy --------------------------------------------------------

def test_vendored_calculator_is_the_recorded_version():
    """core/smd_angle_calculator.py must stay a verbatim copy; edit the source instead."""
    assert _md5(v4.__file__) == SMD_ANGLE_SOURCE_MD5


def test_vendored_calculator_matches_the_source_when_available():
    """Warns (does not fail) if the single source of truth moved on: re-copy it."""
    if not os.path.exists(SOURCE):
        print('    (source not reachable here, skipped)')
        return
    if _md5(SOURCE) != SMD_ANGLE_SOURCE_MD5:
        print('    WARNING: smd_reasoning angle_calculator.py has changed since it '
              'was copied; see core/joint_angles.py')


def test_channels_equal_the_v4_reference_on_real_data():
    """Every channel equals the v4 calculator (mod 360 for unwrapped channels)."""
    joints = np.load(EXAMPLE)['__smplh__'].astype(float)
    ref = v4.compute_all_joint_angles(joints, 'humanml3d', fix_chirality=True,
                                      fix_flex_singularity=True, fix_angle_defs=True)
    if os.path.exists(SOURCE):            # and the source itself, not just our copy
        spec = importlib.util.spec_from_file_location('smd_src', SOURCE)
        src = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(src)
        ref_src = src.compute_all_joint_angles(joints, 'humanml3d', True, True, None, True)
        for name in V4_NAMES.values():
            assert np.allclose(ref[name], ref_src[name], atol=1e-9), name

    angles, time = extract_all_joint_angles(joints, fps=30.0)
    assert set(angles) == set(ANGLE_CHANNELS) and len(ANGLE_CHANNELS) == 26
    for name, v4_name in V4_NAMES.items():
        diff = angles[name] - ref[v4_name]
        if name in WRAPPABLE_ANGLES:
            diff = (diff + 180.0) % 360.0 - 180.0
        assert np.max(np.abs(diff)) < 1e-6, name
    assert np.allclose(np.diff(time), 1 / 30.0)


def test_example_skeleton_is_not_mirrored():
    """MediaPipe -> SMPLH mapping gives a right-handed body (chirality +1)."""
    joints = np.load(EXAMPLE)['__smplh__'].astype(float)
    info = {}
    extract_all_joint_angles(joints, info=info)
    assert info['chirality_score'] > 0.5 and not info['mirrored'], info


# --- behaviour ----------------------------------------------------------------

def test_flexion_has_no_360_jumps_through_overhead_and_side_raise():
    """
    The bug this replaces: shoulder flexion was unwrapped over the whole series,
    so one bad frame (arm out to the side, where flexion is undefined) offset
    everything after it by 360. v4 keeps flexion in [-90, 270) and interpolates
    the undefined frames.
    """
    base = standing_skeleton()
    frames = []
    # forward raise to overhead and back, then a side raise to overhead and back
    for t in np.r_[np.linspace(0, 175, 40), np.linspace(175, 0, 40)]:
        f = base.copy(); a = np.radians(t)
        _place_right_arm(f, [0, -np.cos(a), np.sin(a)], [0, 0.3, 1])
        frames.append(f)
    for t in np.r_[np.linspace(0, 175, 40), np.linspace(175, 0, 40)]:
        f = base.copy(); a = np.radians(t)
        _place_right_arm(f, [-np.sin(a), -np.cos(a), 0.02], [0, 0.3, 1])
        frames.append(f)
    angles, _ = extract_all_joint_angles(np.array(frames))
    flex = angles['right_shoulder_flexion']
    assert np.all(np.isfinite(flex))
    assert flex.min() >= -90 and flex.max() < 270, (flex.min(), flex.max())
    assert np.max(np.abs(np.diff(flex))) < 30, np.max(np.abs(np.diff(flex)))
    assert flex[:80].max() > 160, "overhead must read as ~175"
    assert abs(flex[0]) < 10 and abs(flex[79]) < 10 and abs(flex[-1]) < 20
    add = angles['right_shoulder_abduction']
    assert add[80:].min() < -70, "a side raise reads as a large negative value"


def test_rotation_signs_agree_on_both_sides():
    """+ = internal rotation on both sides (the old left side had the opposite sign)."""
    base = standing_skeleton()
    frames = []
    for t in np.linspace(-60, 60, 25):
        f = base.copy(); a = np.radians(t)
        # forearm swings toward the belly (+t = internal) on both arms
        f[R_WR] = f[R_EL] + 0.25 * np.array([np.sin(a), 0, np.cos(a)])
        f[L_WR] = f[L_EL] + 0.25 * np.array([-np.sin(a), 0, np.cos(a)])
        frames.append(f)
    angles, _ = extract_all_joint_angles(np.array(frames))
    r, l = angles['right_shoulder_rotation'], angles['left_shoulder_rotation']
    assert np.allclose(r, l, atol=1e-6)
    assert r[-1] - r[0] > 100, "rotation must follow the full 120 deg sweep"


def test_mirrored_motion_gives_mirrored_channels():
    """Swap left and right in a real recording: L and R channels swap, lateral ones flip sign."""
    joints = np.load(EXAMPLE)['__smplh__'].astype(float)
    mirrored = joints[:, MIRROR_PERM].copy()
    mirrored[..., 0] *= -1
    a, _ = extract_all_joint_angles(joints)
    b, _ = extract_all_joint_angles(mirrored)
    for name in ANGLE_CHANNELS:
        if name.startswith(('left_', 'right_')):
            other = name.replace('left_', 'X').replace('right_', 'left_').replace('X', 'right_')
            assert np.allclose(a[name], b[other], atol=1e-6), name
        elif name in ('pelvis_flexion', 'lumbar_extension', 'neck_flexion'):
            assert np.allclose(a[name], b[name], atol=1e-6), name
        else:   # list, heading, side bend, twist, head tilt
            assert np.allclose(a[name], -b[name], atol=1e-6), name


def test_untracked_frames_stay_nan():
    joints = np.load(EXAMPLE)['__smplh__'].astype(float)[:100].copy()
    joints[40:45] = np.nan
    angles, time = extract_all_joint_angles(joints)
    for name, values in angles.items():
        assert np.all(np.isnan(values[40:45])), name
        assert np.all(np.isfinite(values[:40])) and np.all(np.isfinite(values[45:])), name
    assert len(time) == 100


def test_unwrap_removes_jumps_and_recentres():
    x = np.array([170.0, 179.0, -178.0, -170.0, np.nan, -160.0])
    u = unwrap_angles(x)
    assert np.isnan(u[4])
    finite = u[np.isfinite(u)]
    assert np.max(np.abs(np.diff(finite))) < 20
    assert -180 <= np.median(finite) < 180
    assert np.allclose((u[np.isfinite(u)] - x[np.isfinite(x)]) % 360, 0)
    assert np.allclose(unwrap_angles(np.array([500.0, 510.0])), [140.0, 150.0])


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
