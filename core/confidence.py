"""
Tracking-confidence propagation.

MediaPipe emits a per-landmark ``visibility`` score in [0, 1] — a classifier
trained to flag landmarks that are occluded or whose position is deemed
inaccurate (Bazarevsky et al., BlazePose, 2020, Sec. 2). That score is the only
honest signal the tool has about how far to trust its ground truth, so it is
carried all the way through to the individual joint-angle channels the user
selects as regression targets.

Three levels, each derived from the one before:

  landmark confidence  (T, 33)   straight from MediaPipe
  joint confidence     (T, 22)   after mapping onto the SMPLH body joints
  angle confidence     per channel, min over the joints that angle depends on

The minimum (not the mean) is used when combining, because a joint angle is only
as trustworthy as the worst landmark feeding it.

Separately, MediaPipe provides no pelvis landmark, so the pelvis and spine are
synthesised geometrically. Angles that depend on them carry a structural caveat
that has nothing to do with visibility, tracked here as SYNTHESISED_JOINTS.
"""

import numpy as np

# --- SMPLH 22-joint indices -------------------------------------------------
PELVIS, L_HIP, R_HIP, SPINE1, L_KNEE, R_KNEE = 0, 1, 2, 3, 4, 5
SPINE2, L_ANKLE, R_ANKLE, SPINE3, L_FOOT, R_FOOT = 6, 7, 8, 9, 10, 11
NECK, L_COLLAR, R_COLLAR, HEAD = 12, 13, 14, 15
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 16, 17, 18, 19, 20, 21

# Joints MediaPipe does not observe; reconstructed from other landmarks in
# mediapipe_to_smplh22(). Their confidence is inherited, but the geometry is an
# assumption, which is a different kind of uncertainty.
SYNTHESISED_JOINTS = {PELVIS, SPINE1, SPINE2, SPINE3, NECK, L_COLLAR, R_COLLAR}

# Which MediaPipe landmarks each SMPLH joint is built from. Directly mapped
# joints list one landmark; synthesised joints list every contributor.
_MP_SOURCES = {
    PELVIS:     [23, 24, 11, 12],   # hip midpoint nudged toward the neck
    L_HIP:      [23],
    R_HIP:      [24],
    SPINE1:     [23, 24, 11, 12],
    L_KNEE:     [25],
    R_KNEE:     [26],
    SPINE2:     [23, 24, 11, 12],
    L_ANKLE:    [27],
    R_ANKLE:    [28],
    SPINE3:     [23, 24, 11, 12],
    L_FOOT:     [31],
    R_FOOT:     [32],
    NECK:       [11, 12],           # shoulder midpoint
    L_COLLAR:   [11, 12],
    R_COLLAR:   [11, 12],
    HEAD:       [0],                # nose
    L_SHOULDER: [11],
    R_SHOULDER: [12],
    L_ELBOW:    [13],
    R_ELBOW:    [14],
    L_WRIST:    [15],
    R_WRIST:    [16],
}

# Every angle is expressed in a body frame built from the pelvis, hips and
# shoulders, so those joints gate all of them.
_BODY_FRAME = [PELVIS, L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]

# Angles built on the shoulder frame additionally inherit the lumbar chain.
_SHOULDER_FRAME = _BODY_FRAME + [NECK, R_SHOULDER]

#: Which SMPLH joints each angle channel is computed from. Mirrors the maths in
#: joint_angles.extract_all_joint_angles().
ANGLE_JOINT_DEPENDENCIES = {
    'pelvis_flexion':           _BODY_FRAME,
    'pelvis_adduction':         _BODY_FRAME,
    'pelvis_rotation':          _BODY_FRAME,

    'left_hip_flexion':         _BODY_FRAME + [L_HIP, L_KNEE, L_ANKLE],
    'left_hip_abduction':       _BODY_FRAME + [L_HIP, L_KNEE, L_ANKLE],
    'left_hip_rotation':        _BODY_FRAME + [L_HIP, L_KNEE, L_ANKLE],
    'right_hip_flexion':        _BODY_FRAME + [R_HIP, R_KNEE, R_ANKLE],
    'right_hip_abduction':      _BODY_FRAME + [R_HIP, R_KNEE, R_ANKLE],
    'right_hip_rotation':       _BODY_FRAME + [R_HIP, R_KNEE, R_ANKLE],

    # Pure three-point angles — no body frame involved.
    'left_knee_flexion':        [L_HIP, L_KNEE, L_ANKLE],
    'right_knee_flexion':       [R_HIP, R_KNEE, R_ANKLE],
    'left_ankle_flexion':       [L_KNEE, L_ANKLE, L_FOOT],
    'right_ankle_flexion':      [R_KNEE, R_ANKLE, R_FOOT],

    'lumbar_extension':         _BODY_FRAME + [NECK, R_SHOULDER],
    'lumbar_bending':           _BODY_FRAME + [NECK, R_SHOULDER],
    'lumbar_rotation':          _BODY_FRAME + [NECK, R_SHOULDER],

    'left_shoulder_flexion':    _SHOULDER_FRAME + [L_SHOULDER, L_ELBOW, L_WRIST],
    'left_shoulder_abduction':  _SHOULDER_FRAME + [L_SHOULDER, L_ELBOW, L_WRIST],
    'left_shoulder_rotation':   _SHOULDER_FRAME + [L_SHOULDER, L_ELBOW, L_WRIST],
    'right_shoulder_flexion':   _SHOULDER_FRAME + [R_SHOULDER, R_ELBOW, R_WRIST],
    'right_shoulder_abduction': _SHOULDER_FRAME + [R_SHOULDER, R_ELBOW, R_WRIST],
    'right_shoulder_rotation':  _SHOULDER_FRAME + [R_SHOULDER, R_ELBOW, R_WRIST],

    'left_elbow_flexion':       [L_SHOULDER, L_ELBOW, L_WRIST],
    'right_elbow_flexion':      [R_SHOULDER, R_ELBOW, R_WRIST],

    'neck_flexion':             _SHOULDER_FRAME + [NECK, HEAD],
    'neck_bending':             _SHOULDER_FRAME + [NECK, HEAD],
}

#: Confidence at or below this is surfaced as unreliable in the GUI.
LOW_CONFIDENCE = 0.5


def mediapipe_visibility_to_smplh22(visibility):
    """
    Map MediaPipe's (T, 33) landmark visibility onto the 22 SMPLH body joints.

    A synthesised joint is only as good as the worst landmark it was built from.

    Args:
        visibility: (T, 33) or (33,) array in [0, 1]

    Returns:
        (T, 22) or (22,) array
    """
    visibility = np.asarray(visibility, dtype=np.float32)
    single = visibility.ndim == 1
    if single:
        visibility = visibility[np.newaxis]

    T = visibility.shape[0]
    joint_conf = np.zeros((T, 22), dtype=np.float32)
    for joint, sources in _MP_SOURCES.items():
        joint_conf[:, joint] = visibility[:, sources].min(axis=1)

    return joint_conf[0] if single else joint_conf


def angle_confidence(joint_confidence, angle_names=None):
    """
    Per-frame confidence for each joint-angle channel.

    Args:
        joint_confidence: (T, 22) array from mediapipe_visibility_to_smplh22()
        angle_names: channels to compute; defaults to all known channels

    Returns:
        {angle_name: (T,) array in [0, 1]}
    """
    joint_confidence = np.asarray(joint_confidence, dtype=np.float32)
    if angle_names is None:
        angle_names = list(ANGLE_JOINT_DEPENDENCIES)

    out = {}
    for name in angle_names:
        joints = ANGLE_JOINT_DEPENDENCIES.get(name)
        if not joints:
            continue
        out[name] = joint_confidence[:, sorted(set(joints))].min(axis=1)
    return out


def is_synthesised(angle_name):
    """True if the angle depends on a joint MediaPipe does not actually observe."""
    joints = ANGLE_JOINT_DEPENDENCIES.get(angle_name, [])
    return bool(set(joints) & SYNTHESISED_JOINTS)


def summarise(confidence, threshold=LOW_CONFIDENCE):
    """
    Condense a (T,) confidence trace into numbers a non-technical user can read.

    Returns a dict with mean, min, and the fraction of frames below `threshold`.
    """
    confidence = np.asarray(confidence, dtype=float)
    if confidence.size == 0:
        return {'mean': float('nan'), 'min': float('nan'), 'frac_low': float('nan')}
    return {
        'mean': float(np.nanmean(confidence)),
        'min': float(np.nanmin(confidence)),
        'frac_low': float(np.nanmean(confidence < threshold)),
    }


def quality_label(mean_confidence):
    """Traffic-light wording for a mean confidence value."""
    if not np.isfinite(mean_confidence):
        return 'unknown'
    if mean_confidence >= 0.85:
        return 'good'
    if mean_confidence >= LOW_CONFIDENCE:
        return 'fair'
    return 'poor'


def tracking_report(skeleton, angle_names=None):
    """
    Build the data behind the GUI's "Tracking Quality" view.

    Args:
        skeleton: result dict from extract_skeleton_from_video()
        angle_names: angle channels to report on (default: all)

    Returns:
        dict with 'available' plus, when visibility was recorded:
        'joint_confidence' (T, 22), 'per_joint' summaries, 'per_angle'
        summaries, 'overall' summary, and 'worst_landmarks'.
    """
    from core.joint_angles import SMPLH_JOINT_NAMES

    visibility = skeleton.get('visibility') if skeleton else None
    if visibility is None:
        return {
            'available': False,
            'reason': 'This skeleton was extracted before confidence tracking '
                      'was added. Re-run extraction to see tracking quality.',
        }

    visibility = np.asarray(visibility, dtype=np.float32)
    joint_conf = mediapipe_visibility_to_smplh22(visibility)

    per_joint = {
        SMPLH_JOINT_NAMES[j]: summarise(joint_conf[:, j])
        for j in range(joint_conf.shape[1])
    }
    per_angle = {
        name: summarise(trace)
        for name, trace in angle_confidence(joint_conf, angle_names).items()
    }

    overall = summarise(joint_conf.min(axis=1))
    worst = sorted(per_joint.items(), key=lambda kv: kv[1]['mean'])[:5]

    return {
        'available': True,
        'joint_confidence': joint_conf,
        'per_joint': per_joint,
        'per_angle': per_angle,
        'overall': overall,
        'label': quality_label(overall['mean']),
        'worst_landmarks': [(n, s['mean']) for n, s in worst],
    }
