"""
Joint angle computation.

The angles are the SMD v4 joint angles (definitions after the 2026-10-10 angle
audit). The maths lives in ``core/smd_angle_calculator.py``, a verbatim copy of
the single source of truth::

    /scratch/elec/t412-humanmotion/yaozhang/smd_reasoning/smd_core/angle_calculator.py
    md5 302070cf2d0c01b729ad32b0a3049cc4 (2026-10-10)

Do not edit the copy. When the source changes, copy it again and update the md5
above and ``SMD_ANGLE_SOURCE_MD5`` below; ``tests/test_joint_angles.py`` checks
that they agree.

This module keeps the GUI's channel names (``right_shoulder_flexion`` ...) and
maps each one onto a v4 angle. What v4 fixes compared with the angles the GUI
computed before:

- shoulder and hip flexion are kept in [-90, 270) and frames where flexion is
  undefined (limb out to the side) are interpolated, instead of unwrapping the
  whole series, where one bad frame offset everything after it by 360 degrees;
- pelvis tilt / list no longer swap when the person turns (heading-first Euler);
- lumbar rotation uses the collar line instead of the right shoulder alone;
- shoulder and hip rotation are twists about the limb axis, + = internal on
  both sides, and are no longer shrunk by elbow / knee bending;
- left / right sign conventions agree (the right hip abduction channel and the
  left hip and shoulder rotation channels had the opposite sign of their other
  side);
- a mirrored skeleton is detected (chirality score) and un-mirrored.
"""

import numpy as np

from core import smd_angle_calculator as _v4

#: md5 of the vendored calculator; tests compare it with the file on disk.
SMD_ANGLE_SOURCE_MD5 = '302070cf2d0c01b729ad32b0a3049cc4'

#: Stored in every angle cache. A cache with another (or no) version was
#: computed with older definitions and is recomputed from its cached joints.
ANGLE_DEFINITION_VERSION = 'smd-v4-2026-10-10'


# =============================================================================
# SMPLH Joint Names (52 joints: 22 body + 30 hand)
# =============================================================================

SMPLH_JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    # Left hand (22-36)
    "left_index1", "left_index2", "left_index3",
    "left_middle1", "left_middle2", "left_middle3",
    "left_pinky1", "left_pinky2", "left_pinky3",
    "left_ring1", "left_ring2", "left_ring3",
    "left_thumb1", "left_thumb2", "left_thumb3",
    # Right hand (37-51)
    "right_index1", "right_index2", "right_index3",
    "right_middle1", "right_middle2", "right_middle3",
    "right_pinky1", "right_pinky2", "right_pinky3",
    "right_ring1", "right_ring2", "right_ring3",
    "right_thumb1", "right_thumb2", "right_thumb3",
]


# =============================================================================
# SMPLX -> SMPLH Conversion
# =============================================================================

def convert_smplx_to_smplh(smplx_joints):
    """
    Convert SMPLX 137-joint format to SMPLH 52-joint format.

    Args:
        smplx_joints: numpy array of shape (T, 137, 3) or (137, 3)

    Returns:
        smplh_joints: numpy array of shape (T, 52, 3) or (52, 3)
    """
    if smplx_joints.ndim == 3:
        num_frames = smplx_joints.shape[0]
        smplh_all = np.zeros((num_frames, 52, 3), dtype=smplx_joints.dtype)
        for i in range(num_frames):
            smplh_all[i] = _convert_single_frame(smplx_joints[i])
        return smplh_all
    else:
        return _convert_single_frame(smplx_joints)


def _convert_single_frame(smplx_joints):
    """Convert a single frame from SMPLX 137-joint to SMPLH 52-joint format."""
    smplh = np.zeros((52, 3), dtype=smplx_joints.dtype)

    # Direct body joint mappings: smplh_idx -> smplx_idx
    mapping = {
        0: 0,    # pelvis
        1: 1,    # left_hip
        2: 2,    # right_hip
        4: 3,    # left_knee
        5: 4,    # right_knee
        7: 5,    # left_ankle
        8: 6,    # right_ankle
        12: 7,   # neck
        16: 8,   # left_shoulder
        17: 9,   # right_shoulder
        18: 10,  # left_elbow
        19: 11,  # right_elbow
        20: 12,  # left_wrist
        21: 13,  # right_wrist
    }
    for smplh_idx, smplx_idx in mapping.items():
        smplh[smplh_idx] = smplx_joints[smplx_idx]

    # Interpolated spine joints
    pelvis = smplx_joints[0]
    neck = smplx_joints[7]
    smplh[3] = pelvis + (neck - pelvis) * (1 / 4)   # spine1
    smplh[6] = pelvis + (neck - pelvis) * (2 / 4)   # spine2
    smplh[9] = pelvis + (neck - pelvis) * (3 / 4)   # spine3

    # Foot / collar / head
    smplh[10] = smplx_joints[14]  # left_foot <- L_Big_toe
    smplh[11] = smplx_joints[17]  # right_foot <- R_Big_toe
    smplh[13] = smplx_joints[8]   # left_collar <- L_Shoulder
    smplh[14] = smplx_joints[9]   # right_collar <- R_Shoulder
    smplh[15] = smplx_joints[24]  # head <- Nose

    # Left hand joints (SMPLH 22-36)
    smplh[22] = smplx_joints[29]; smplh[23] = smplx_joints[30]; smplh[24] = smplx_joints[31]
    smplh[25] = smplx_joints[33]; smplh[26] = smplx_joints[34]; smplh[27] = smplx_joints[35]
    smplh[28] = smplx_joints[41]; smplh[29] = smplx_joints[42]; smplh[30] = smplx_joints[43]
    smplh[31] = smplx_joints[37]; smplh[32] = smplx_joints[38]; smplh[33] = smplx_joints[39]
    smplh[34] = smplx_joints[25]; smplh[35] = smplx_joints[26]; smplh[36] = smplx_joints[27]

    # Right hand joints (SMPLH 37-51)
    smplh[37] = smplx_joints[49]; smplh[38] = smplx_joints[50]; smplh[39] = smplx_joints[51]
    smplh[40] = smplx_joints[53]; smplh[41] = smplx_joints[54]; smplh[42] = smplx_joints[55]
    smplh[43] = smplx_joints[61]; smplh[44] = smplx_joints[62]; smplh[45] = smplx_joints[63]
    smplh[46] = smplx_joints[57]; smplh[47] = smplx_joints[58]; smplh[48] = smplx_joints[59]
    smplh[49] = smplx_joints[45]; smplh[50] = smplx_joints[46]; smplh[51] = smplx_joints[47]

    # Coordinate system transformation: flip y and z
    smplh[:, 1] *= -1
    smplh[:, 2] *= -1

    return smplh


# =============================================================================
# Channels
# =============================================================================

#: GUI channel name -> SMD v4 angle name. Signs follow v4 (see docs/manual.md):
#: the "abduction" channels are + = toward the midline on both sides.
V4_NAMES = {
    'pelvis_flexion':           'pelvis_tilt',
    'pelvis_adduction':         'pelvis_list',
    'pelvis_rotation':          'pelvis_rotation',
    'left_hip_flexion':         'hip_flexion_l',
    'left_hip_abduction':       'hip_adduction_l',
    'left_hip_rotation':        'hip_rotation_l',
    'right_hip_flexion':        'hip_flexion_r',
    'right_hip_abduction':      'hip_adduction_r',
    'right_hip_rotation':       'hip_rotation_r',
    'left_knee_flexion':        'knee_angle_l',
    'right_knee_flexion':       'knee_angle_r',
    'left_ankle_flexion':       'ankle_angle_l',
    'right_ankle_flexion':      'ankle_angle_r',
    'lumbar_extension':         'lumbar_extension',
    'lumbar_bending':           'lumbar_bending',
    'lumbar_rotation':          'lumbar_rotation',
    'left_shoulder_flexion':    'arm_flex_l',
    'left_shoulder_abduction':  'arm_add_l',
    'left_shoulder_rotation':   'arm_rot_l',
    'right_shoulder_flexion':   'arm_flex_r',
    'right_shoulder_abduction': 'arm_add_r',
    'right_shoulder_rotation':  'arm_rot_r',
    'left_elbow_flexion':       'elbow_flex_l',
    'right_elbow_flexion':      'elbow_flex_r',
    'neck_flexion':             'neck_flexion',
    'neck_bending':             'neck_adduction',
}

ANGLE_CHANNELS = list(V4_NAMES)

#: Channels unwrapped across +/-180 deg: the set the SMD v4 text layer unwraps.
#: Shoulder and hip flexion are deliberately absent; v4 keeps them in [-90, 270).
WRAPPABLE_ANGLES = {
    'pelvis_flexion', 'pelvis_adduction', 'pelvis_rotation',
    'left_hip_rotation', 'right_hip_rotation',
    'lumbar_rotation',
    'left_shoulder_rotation', 'right_shoulder_rotation',
    'neck_flexion', 'neck_bending',
}


def unwrap_angles(values):
    """
    Remove +/-360 deg jumps from an angle series (shortest path between
    consecutive frames), then shift the whole series by a multiple of 360 so its
    median lies in [-180, 180). NaN frames are skipped and stay NaN.
    """
    out = np.asarray(values, dtype=float).copy()
    ok = np.flatnonzero(np.isfinite(out))
    if ok.size == 0:
        return out
    x = out[ok]
    steps = (np.diff(x) + 180.0) % 360.0 - 180.0
    x = x[0] + np.concatenate([[0.0], np.cumsum(steps)])
    x -= 360.0 * np.floor((np.median(x) + 180.0) / 360.0)
    out[ok] = x
    return out


# =============================================================================
# High-level API
# =============================================================================

def _body_joints(joint_positions):
    """(T, 22 or 52, 3) -> (T, 22, 3) float64 body joints."""
    j = np.asarray(joint_positions, dtype=float)
    if j.ndim != 3 or j.shape[2] != 3:
        raise ValueError(f"Expected a (T, N, 3) array, got shape {j.shape}")
    if j.shape[1] == 52:
        j = j[:, :22]
    elif j.shape[1] != 22:
        raise ValueError(f"Expected 22 or 52 joints, got {j.shape[1]}")
    return j


def extract_all_joint_angles(joint_positions, fps=30.0, info=None):
    """
    Compute the 26 joint angle channels from SMPLH body joints.

    Args:
        joint_positions: (T, 22, 3) or (T, 52, 3), SMPLH joint order, axes
            X = the person's left, Y = up, Z = forward (what
            ``smpl_extraction.mediapipe_to_smplh22`` produces; the SMD
            "humanml3d" convention). Frames that contain NaN (no pose
            detected) are allowed.
        fps: frame rate, only used for the returned time axis.
        info: optional dict; receives the v4 diagnostics (chirality score,
            whether the clip was un-mirrored, interpolated frame counts).

    Returns:
        angles: {channel: (T,) degrees}, NaN where no pose was detected
        time: (T,) seconds, frame i at i / fps
    """
    joints = _body_joints(joint_positions)
    T = joints.shape[0]
    valid = np.isfinite(joints).all(axis=(1, 2))
    angles = {name: np.full(T, np.nan) for name in ANGLE_CHANNELS}

    # The v4 maths cannot take NaN frames. Compute on the tracked frames only
    # (its interpolation then bridges the gaps) and leave the rest NaN.
    if valid.sum() >= 2:
        raw = _v4.compute_all_joint_angles(
            joints[valid], 'humanml3d', fix_chirality=True,
            fix_flex_singularity=True, info=info, fix_angle_defs=True,
        )
        for name, v4_name in V4_NAMES.items():
            angles[name][valid] = raw[v4_name]

    for name in WRAPPABLE_ANGLES:
        angles[name] = unwrap_angles(angles[name])

    fps = float(fps) if fps else 30.0
    return angles, np.arange(T) / fps


#: v4 diagnostic key -> (channel, why the value is undefined in those frames)
_INTERP_KEYS = {
    'interp_rot_arm_r': ('right_shoulder_rotation', 'elbow nearly straight or arm overhead'),
    'interp_rot_arm_l': ('left_shoulder_rotation', 'elbow nearly straight or arm overhead'),
    'interp_arm_r': ('right_shoulder_flexion', 'arm out to the side'),
    'interp_arm_l': ('left_shoulder_flexion', 'arm out to the side'),
    'interp_hip_r': ('right_hip_flexion', 'leg out to the side'),
    'interp_hip_l': ('left_hip_flexion', 'leg out to the side'),
    'interp_pelvis': ('pelvis_rotation', 'body lying down'),
}


def interpolation_warnings(info, n_frames, threshold=0.2):
    """
    Channels whose value is undefined, and therefore interpolated, in more than
    `threshold` of the frames, as readable lines for the import log. Such a
    channel is a poor training target for this recording.
    """
    lines = []
    for key, (channel, reason) in _INTERP_KEYS.items():
        n = info.get(key) or 0
        if n_frames and n / n_frames > threshold:
            lines.append(f"{channel} is undefined in {100 * n / n_frames:.0f}% of "
                         f"frames ({reason}); those frames are interpolated")
    return lines


def smplx_to_joint_angles(smplx_joints, fps=30.0):
    """SMPLX (T, 137, 3) joints -> (angles, time)."""
    return extract_all_joint_angles(convert_smplx_to_smplh(smplx_joints), fps=fps)
