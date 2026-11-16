"""
Bones: what the user picks in Tab 3, and how a predicted direction is drawn
back into the skeleton in Tab 5.

Each bone maps to the regression targets that describe it (core.targets).
Arm and leg segments carry a direction target, so their predicted pose can be
drawn: the predicted end of the bone is

    true start joint + true bone length * predicted direction,

with the direction taken out of the chest (arm) or pelvis (leg) frame of the
true skeleton. Down a limb the moved joints are carried along: with the upper
arm and the forearm both predicted, the forearm starts at the predicted elbow,
so an error at the shoulder shows at the wrist too, as it would on the body.
A segment that is not predicted keeps its true direction and length and is only
moved with its parent.

The foot, the trunk and the head have angle targets only; an angle does not fix
a direction (the elbow angle says nothing about where the forearm points), so
those bones are not redrawn and their predictions are shown as numbers.
"""

import numpy as np

from core.targets import _v4_frames, target_slices, unit

# SMPL 22-joint indices
PELVIS, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT = 0, 1, 2, 4, 5, 7, 8, 10, 11
NECK, HEAD, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 12, 15, 16, 17, 18, 19, 20, 21

#: name -> (label, start joint, end joint, direction target or None, targets)
#: The first target is the one that defines the bone; picking the bone ticks
#: all of them.
BONES = {}
for _side, _S in (('left', 'L'), ('right', 'R')):
    _j = {'L': (L_SH, L_EL, L_WR, L_HIP, L_KNEE, L_ANKLE, L_FOOT),
          'R': (R_SH, R_EL, R_WR, R_HIP, R_KNEE, R_ANKLE, R_FOOT)}[_S]
    sh, el, wr, hip, knee, ankle, foot = _j
    BONES.update({
        f'{_side}_upper_arm': (f'{_side.title()} upper arm', sh, el,
                               f'{_side}_upper_arm_direction',
                               [f'{_side}_upper_arm_direction', f'{_side}_shoulder_elevation']),
        f'{_side}_forearm': (f'{_side.title()} forearm', el, wr,
                             f'{_side}_forearm_direction',
                             [f'{_side}_forearm_direction', f'{_side}_elbow_flexion']),
        f'{_side}_thigh': (f'{_side.title()} thigh', hip, knee,
                           f'{_side}_thigh_direction',
                           [f'{_side}_thigh_direction', f'{_side}_hip_elevation']),
        f'{_side}_shank': (f'{_side.title()} shank', knee, ankle,
                           f'{_side}_shank_direction',
                           [f'{_side}_shank_direction', f'{_side}_knee_flexion']),
        f'{_side}_foot': (f'{_side.title()} foot', ankle, foot, None,
                          [f'{_side}_ankle_flexion']),
    })
BONES['trunk'] = ('Trunk', PELVIS, NECK, None, ['lumbar_bending', 'lumbar_rotation'])
BONES['head'] = ('Head and neck', NECK, HEAD, None, ['neck_flexion', 'neck_bending'])

#: Down each limb, in order: (bone, joint it moves). Joints further down a
#: chain move with it.
_CHAINS = {
    side: [(f'{side}_upper_arm', el), (f'{side}_forearm', wr)]
    for side, el, wr in (('left', L_EL, L_WR), ('right', R_EL, R_WR))
}
_CHAINS.update({
    f'{side}_leg': [(f'{side}_thigh', knee), (f'{side}_shank', ankle), (f'{side}_foot', foot)]
    for side, knee, ankle, foot in (('left', L_KNEE, L_ANKLE, L_FOOT),
                                    ('right', R_KNEE, R_ANKLE, R_FOOT))
})

#: direction target -> its bone
DIRECTION_BONE = {spec[3]: name for name, spec in BONES.items() if spec[3]}


def bone_targets(name):
    return list(BONES[name][4])


def bone_of_target(target):
    """The bone a target belongs to, or None (display-only channels have none)."""
    for name, spec in BONES.items():
        if target in spec[4]:
            return name
    return None


def selection_state(name, selected):
    """'all', 'some' or 'none' of the bone's targets are in `selected`."""
    targets = BONES[name][4]
    n = sum(t in selected for t in targets)
    return 'all' if n == len(targets) else ('some' if n else 'none')


def _frame_of(bone):
    return 'chest' if ('arm' in bone) else 'pelvis'


def directions_to_world(joints, local, bone):
    """
    (T, 22, 3) true joints (humanml3d axes: X the person's left, Y up, Z
    forward) and (T, 3) directions in the bone's body frame (forward, up,
    outward) -> (T, 3) unit directions in the joints' axes.
    """
    _, pelvis, chest, mirror = _v4_frames(np.asarray(joints, float), return_mirror=True)
    frame = chest if _frame_of(bone) == 'chest' else pelvis
    d = unit(local).copy()
    if bone.startswith('left_'):
        d[:, 2] = -d[:, 2]                      # outward = the person's left
    c = np.einsum('nji,nj->ni', frame, d)       # body frame -> calculator axes
    if mirror is not None:
        c = c - 2.0 * (c @ mirror)[:, None] * mirror
    return np.stack([-c[:, 2], c[:, 1], c[:, 0]], axis=1)   # -> humanml3d axes


def reconstruct(joints, predicted):
    """
    The true skeleton with the predicted bones swapped in.

    joints: (T, 22, 3) true joints, no NaN.
    predicted: {bone name: (T, 3) predicted direction in the bone's frame}.
    Returns (T, 22, 3).
    """
    joints = np.asarray(joints, float)
    out = joints.copy()
    world = {b: directions_to_world(joints, d, b) for b, d in predicted.items()}
    for chain in _CHAINS.values():
        for bone, end in chain:
            start = BONES[bone][1]
            true_vec = joints[:, end] - joints[:, start]
            if bone in world:
                length = np.linalg.norm(true_vec, axis=1, keepdims=True)
                out[:, end] = out[:, start] + length * world[bone]
            else:
                out[:, end] = out[:, start] + true_vec
    return out


def predicted_directions(pred, names):
    """{bone: (N, 3)} for every direction target among the model's outputs."""
    out = {}
    for name, sl in target_slices(names):
        if name in DIRECTION_BONE:
            out[DIRECTION_BONE[name]] = np.asarray(pred)[:, sl]
    return out
