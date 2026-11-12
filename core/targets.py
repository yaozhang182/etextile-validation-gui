"""
Regression targets.

Shoulder and hip flexion, abduction and rotation are fine for describing a
movement but poor targets for frame-by-frame regression: two angles cannot
describe a limb direction without a singular point (flexion is undefined with
the limb straight out to the side) and a cut where the value jumps by 360
(flexion -90 / 270, limb level and behind the body). A loss on such a channel
sees a jump of hundreds of degrees where the limb hardly moved.

The targets offered for training are therefore continuous everywhere:

- hinge angles (knee, elbow, ankle) and the trunk and neck angles, which never
  reach a singular point in normal movement, as computed by core.joint_angles;
- the **elevation** of the upper arm and of the thigh: the angle between the
  limb and its hanging-down direction, 0 (hanging) to 180 (straight up),
  whatever the direction of the raise;
- the **direction** of the upper arm, forearm, thigh and shank as a unit vector
  in the frame of the body segment they hang from (chest for the arm, pelvis
  for the leg). Upper arm plus forearm direction describe the whole arm, twist
  included, and stay defined with a straight elbow, where shoulder rotation is
  not.

Shoulder and hip flexion / abduction / rotation and the body heading
(pelvis_rotation) stay available for display only.

The frames are exactly those of the SMD v4 angle calculator, so for every frame
where flexion and abduction are defined:
    elevation = arccos(cos(abduction) * cos(flexion))
    direction = (cos a * sin f, -cos a * cos f, -sin a)    (forward, up, outward)
with f = flexion and a = the abduction channel (negative = away from the body).
"""

import numpy as np

from core import smd_angle_calculator as _v4

# --- channel names ---------------------------------------------------------

#: Direction components, in the order (forward, up, outward). "Outward" is the
#: person's right for right limbs and left for left limbs, so both sides read
#: the same: + = away from the body's midline.
COMPONENTS = ('fwd', 'up', 'out')
COMPONENT_LABELS = {'fwd': 'forward', 'up': 'up', 'out': 'outward'}

_LIMBS = {
    # name: (parent frame, start joint, end joint) — SMPL indices
    'left_upper_arm':  ('chest', 16, 18),
    'right_upper_arm': ('chest', 17, 19),
    'left_forearm':    ('chest', 18, 20),
    'right_forearm':   ('chest', 19, 21),
    'left_thigh':      ('pelvis', 1, 4),
    'right_thigh':     ('pelvis', 2, 5),
    'left_shank':      ('pelvis', 4, 7),
    'right_shank':     ('pelvis', 5, 8),
}

#: vector target -> its three component channels
VECTOR_TARGETS = {
    f'{limb}_direction': [f'{limb}_direction_{c}' for c in COMPONENTS]
    for limb in _LIMBS
}

#: elevation channel -> the limb it measures
ELEVATION_TARGETS = {
    'left_shoulder_elevation': 'left_upper_arm',
    'right_shoulder_elevation': 'right_upper_arm',
    'left_hip_elevation': 'left_thigh',
    'right_hip_elevation': 'right_thigh',
}

#: Channels kept for display but never offered as training targets.
DISPLAY_ONLY = (
    [f'{side}_{joint}_{dof}' for side in ('left', 'right')
     for joint in ('shoulder', 'hip')
     for dof in ('flexion', 'abduction', 'rotation')]
    + ['pelvis_rotation']
)

#: Why each kind of display-only channel cannot be a target (tooltips, docs).
DISPLAY_ONLY_REASON = (
    "Display only: this angle has a singular point (undefined with the limb "
    "straight out to the side, or with the elbow straight for rotation) and a "
    "point where it jumps by 360 degrees, which a regression loss cannot handle. "
    "Use the elevation and direction targets instead."
)

#: How the targets are grouped in Tab 3.
TARGET_GROUPS = {
    'Shoulder L': ['left_shoulder_elevation', 'left_upper_arm_direction',
                   'left_forearm_direction'],
    'Shoulder R': ['right_shoulder_elevation', 'right_upper_arm_direction',
                   'right_forearm_direction'],
    'Elbow': ['left_elbow_flexion', 'right_elbow_flexion'],
    'Hip L': ['left_hip_elevation', 'left_thigh_direction', 'left_shank_direction'],
    'Hip R': ['right_hip_elevation', 'right_thigh_direction', 'right_shank_direction'],
    'Knee': ['left_knee_flexion', 'right_knee_flexion'],
    'Ankle': ['left_ankle_flexion', 'right_ankle_flexion'],
    'Trunk': ['pelvis_flexion', 'pelvis_adduction', 'lumbar_extension',
              'lumbar_bending', 'lumbar_rotation'],
    'Neck': ['neck_flexion', 'neck_bending'],
}

DISPLAY_GROUPS = {
    'Shoulder L': ['left_shoulder_flexion', 'left_shoulder_abduction',
                   'left_shoulder_rotation'],
    'Shoulder R': ['right_shoulder_flexion', 'right_shoulder_abduction',
                   'right_shoulder_rotation'],
    'Hip L': ['left_hip_flexion', 'left_hip_abduction', 'left_hip_rotation'],
    'Hip R': ['right_hip_flexion', 'right_hip_abduction', 'right_hip_rotation'],
    'Pelvis': ['pelvis_rotation'],
}

TARGET_CHANNELS = [n for names in TARGET_GROUPS.values() for n in names]

#: Every channel this module adds to the angle dict.
EXTRA_CHANNELS = list(ELEVATION_TARGETS) + [c for comps in VECTOR_TARGETS.values()
                                            for c in comps]


def is_vector(name):
    return name in VECTOR_TARGETS


def expand_targets(names):
    """Target names -> the scalar columns they occupy (a vector takes three)."""
    cols = []
    for name in names:
        cols.extend(VECTOR_TARGETS.get(name, [name]))
    return cols


def target_slices(names):
    """[(name, slice into the expanded columns)] in the same order."""
    out, start = [], 0
    for name in names:
        width = 3 if is_vector(name) else 1
        out.append((name, slice(start, start + width)))
        start += width
    return out


def display_name(name):
    return name.replace('_', ' ').title()


def unit(raw):
    """Normalise each row of an (N, 3) array; zero rows stay zero."""
    raw = np.asarray(raw, dtype=float)
    norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    return np.divide(raw, norm, out=np.zeros_like(raw), where=norm > 1e-9)


def postprocess(pred, names):
    """Model output -> valid targets: each predicted direction is normalised."""
    pred = np.array(pred, dtype=float, copy=True)
    for name, sl in target_slices(names):
        if is_vector(name):
            pred[:, sl] = unit(pred[:, sl])
    return pred


# --- computation ----------------------------------------------------------

def _v4_frames(joints):
    """
    The pelvis and chest frames exactly as the SMD v4 calculator builds them
    (compute_all_joint_angles with every fix on), in its axes: X forward,
    Y up, Z the person's right. Returns (ric, pelvis_frame, chest_frame); frame
    rows are the axes, so frame @ v gives v's coordinates.
    """
    ric = joints[:, :, [2, 1, 0]].copy()      # "humanml3d" -> calculator axes
    ric[:, :, 2] *= -1
    if _v4.chirality_score(ric) < _v4.CHIRALITY_THR:
        ric = _v4.unmirror_keep_heading(ric)
    pelvis = _v4.compute_body_coordinate(ric[:, 2], ric[:, 1], ric[:, 17],
                                         ric[:, 16], ric[:, 0])
    trunk = ric[:, 12] - ric[:, 0]
    ext = _v4.calculate_joint_angles(trunk, pelvis, ref_axis="pos_y")[:, 0]
    bend = _v4.compute_signed_angle_xy_projection(trunk, pelvis)
    _, neck = _v4.compute_lumbar_rotation(np.stack([ext, bend], axis=1),
                                          ric[:, 12], ric[:, 17], pelvis)
    twist = _v4.lumbar_rotation_collar(neck, ric[:, 13], ric[:, 14])
    chest = _v4.compute_shoulder_coordinate(pelvis, ext, bend, twist, neck)
    return ric, pelvis, chest


def compute_target_channels(joints):
    """
    (T, 22, 3) body joints without NaN, in the "humanml3d" axes (X the
    person's left, Y up, Z forward) -> {channel: (T,)} for every elevation and
    direction component.
    """
    ric, pelvis, chest = _v4_frames(np.asarray(joints, dtype=float))
    frames = {'pelvis': pelvis, 'chest': chest}
    out, directions = {}, {}
    for limb, (frame, a, b) in _LIMBS.items():
        d = unit(np.einsum('nij,nj->ni', frames[frame], ric[:, b] - ric[:, a]))
        if limb.startswith('left_'):
            d[:, 2] = -d[:, 2]            # outward = the person's left
        directions[limb] = d
        for k, comp in enumerate(COMPONENTS):
            out[f'{limb}_direction_{comp}'] = d[:, k]
    for name, limb in ELEVATION_TARGETS.items():
        out[name] = np.degrees(np.arccos(np.clip(-directions[limb][:, 1], -1.0, 1.0)))
    return out


# --- evaluation -----------------------------------------------------------

def angle_between(a, b):
    """Angle in degrees between corresponding rows of two (N, 3) arrays."""
    cos = np.sum(unit(a) * unit(b), axis=1)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _pcc(x, y):
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def evaluate_targets(y_true, y_pred, names):
    """
    Metrics per target, in the layout of core.evaluator.calculate_metrics so the
    dashboard can show any mix of targets.

    Scalar targets: absolute error (MPJAE), RMSE, Pearson r, and AMPE (MPJAE as
    a percentage of the range covered, at least 5 degrees).
    Direction targets: the error of a frame is the angle between the predicted
    and the true direction; MPJAE and RMSE are its mean and root mean square,
    the range is twice the largest angle between a true direction and their
    mean, and Pearson r is the mean over the three components.
    Everything is in degrees; the Global values average over targets.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = postprocess(y_pred, names)
    keys = ('MPJAE', 'AMPE', 'RMSE', 'PCC', 'Range_of_motion')
    per = {k: {} for k in keys}
    errors = []
    for name, sl in target_slices(names):
        label = display_name(name)
        t, p = y_true[:, sl], y_pred[:, sl]
        if is_vector(name):
            err = angle_between(t, p)
            centre = unit(t.mean(axis=0, keepdims=True))
            rom = 2.0 * float(np.max(angle_between(t, np.repeat(centre, len(t), 0))))
            pcc = float(np.mean([_pcc(t[:, k], p[:, k]) for k in range(3)]))
        else:
            err = np.abs(t[:, 0] - p[:, 0])
            rom = float(np.max(t) - np.min(t))
            pcc = _pcc(t[:, 0], p[:, 0])
        rom = max(rom, 5.0)
        per['MPJAE'][label] = float(np.mean(err))
        per['RMSE'][label] = float(np.sqrt(np.mean(err ** 2)))
        per['AMPE'][label] = 100.0 * per['MPJAE'][label] / rom
        per['PCC'][label] = pcc
        per['Range_of_motion'][label] = rom
        errors.append(err)

    metrics = {f'{k}_per_joint': v for k, v in per.items() if k != 'Range_of_motion'}
    metrics['Range_of_motion'] = per['Range_of_motion']
    for k in ('MPJAE', 'AMPE', 'PCC'):
        metrics[f'Global_{k}'] = float(np.mean(list(per[k].values())))
    all_err = np.concatenate(errors)
    metrics['Global_RMSE'] = float(np.sqrt(np.mean(all_err ** 2)))
    metrics['MAE_std'] = float(np.std(all_err))
    metrics['Max_error'] = float(np.max(all_err))
    metrics['target_kinds'] = {display_name(n): ('direction' if is_vector(n) else 'angle')
                               for n in names}
    return metrics
