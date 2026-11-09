"""
Joint angle calculator for 22-joint SMPL skeleton.

Computes biomechanical joint angles (flexion, adduction, rotation) from
3D joint positions using body coordinate systems and Euler angles.

Adapted from motion2angle_22joints_v1/joint_angle_cal.py

Joint index mapping (SMPL 22-joint):
    0: pelvis, 1: left_hip, 2: right_hip, 3: spine1,
    4: left_knee, 5: right_knee, 6: spine2,
    7: left_ankle, 8: right_ankle, 9: spine3,
    10: left_foot, 11: right_foot, 12: neck,
    13: left_collar, 14: right_collar, 15: head,
    16: left_shoulder, 17: right_shoulder,
    18: left_elbow, 19: right_elbow,
    20: left_wrist, 21: right_wrist
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def compute_body_coordinate(right_hip, left_hip, right_shoulder, left_shoulder, pelvis):
    """Compute body coordinate system from hip and shoulder positions."""
    across = right_hip - left_hip
    across = across / np.linalg.norm(across, axis=-1, keepdims=True)

    hip_mid = (right_hip + left_hip) / 2
    up = pelvis - hip_mid
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)

    # Make up perpendicular to across
    up = up - np.sum(up * across, axis=-1, keepdims=True) * across
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)

    # Forward = up x across
    forward = np.cross(up, across)
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)

    # Stack: (3, T, 3) -> (T, 3, 3)
    human_coordinate = np.array([forward, up, across])
    return np.transpose(human_coordinate, (1, 0, 2))


# ---------------------------------------------------------------------------------------------------------------------------
# Optional fixes (plans/ROUTE1_COT.md §9.4 / §9.5). Both default OFF, so existing callers and SMD are unchanged until we decide.
#   fix_chirality:   some sources (HumanML3D's humanact12 subset, ...) store a mirrored skeleton with the left/right labels kept.
#                    The body frame's "forward" then points backwards and every sagittal angle flips sign. Detected per clip by
#                    the knee-hinge handedness (chirality_score) < CHIRALITY_THR; repaired by negating the lateral axis (labels unchanged).
#   fix_flex_singularity: flexion = atan2 of the limb's sagittal projection is undefined when the limb points along the
#                    across axis (arm raised to the side). Put flexion in [-90, 270) (overhead = 180 is interior, -90 = limb
#                    straight back, anatomically unreachable) and, where the sagittal projection is < SING_THR of the limb
#                    length, linearly interpolate flexion between the neighbouring well-conditioned frames.
CHIRALITY_THR = -0.5
SING_THR = 0.4                     # v4 value (plans/diag/FIX_SHOULDER_SWEEP.md); was 0.3 in the first trial


def chirality_score(ric, return_parts=False):
    """+1 = normal right-handed body, -1 = mirrored skeleton. Primary cue: the knee hinge. For frames with knee flexion > 20 deg,
    the normal of (thigh x shank) points to the body's left in a normal body, i.e. against across = right_hip - left_hip; the
    score is the mean of -sign(normal . across) over those frames of both legs. Fallback when the knees hardly bend
    (< 10 such frames): median of (horizontal body forward . horizontal toe direction)."""
    bc = compute_body_coordinate(ric[:, 2], ric[:, 1], ric[:, 17], ric[:, 16], ric[:, 0])
    across = bc[:, 2]
    s = []
    for h, k, a in [(2, 5, 8), (1, 4, 7)]:
        th = ric[:, k] - ric[:, h]; sh = ric[:, a] - ric[:, k]
        cosang = (th * sh).sum(1) / (np.linalg.norm(th, axis=1) * np.linalg.norm(sh, axis=1) + 1e-9)
        m = np.degrees(np.arccos(np.clip(cosang, -1, 1))) > 20
        s.append(-np.sign((np.cross(th, sh) * across).sum(1))[m])
    s = np.concatenate(s); knee = float(s.mean()) if len(s) >= 10 else None
    fwd = bc[:, 0].copy(); fwd[:, 1] = 0
    toe = (ric[:, 10] - ric[:, 7]) + (ric[:, 11] - ric[:, 8]); toe[:, 1] = 0
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True) + 1e-9; toe /= np.linalg.norm(toe, axis=1, keepdims=True) + 1e-9
    toe_s = float(np.median((fwd * toe).sum(1)))
    score = knee if knee is not None else toe_s
    return (score, knee, toe_s) if return_parts else score


def sagittal_fraction(vectors, coordinate_axes):
    """|sagittal (XY) projection| / |vector| of each frame's limb vector in the given frame."""
    ca = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)
    v = np.einsum('nij,nj->ni', ca, vectors); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return np.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)


def regularize_flexion(flex, sag, thr=None):
    """flex (T,) deg from atan2 -> [-90, 270), then frames with sag < thr are replaced by linear interpolation between the
    nearest well-conditioned frames (held at the clip ends). Returns (flex, n_interpolated)."""
    thr = SING_THR if thr is None else thr
    f = (np.asarray(flex, float) + 90.0) % 360.0 - 90.0
    bad = sag < thr
    if not bad.any() or bad.all():
        return f, 0
    t = np.arange(len(f)); good = ~bad
    f[bad] = np.interp(t[bad], t[good], f[good])
    return f, int(bad.sum())


def calculate_pelvis_euler_angles(body_coordinate_vectors):
    """Calculate pelvis orientation using intrinsic ZXY Euler angles."""
    n_frame = body_coordinate_vectors.shape[0]
    euler_angles = np.zeros((n_frame, 3))

    for i in range(n_frame):
        rotation_matrix = body_coordinate_vectors[i].T
        rot = R.from_matrix(rotation_matrix)
        euler_angles[i] = rot.as_euler('ZXY', degrees=True)

    return euler_angles


def calculate_joint_angles(vectors, coordinate_axes, ref_axis="neg_y"):
    """
    Calculate signed angles between vectors and coordinate planes.

    Args:
        vectors: (n_frame, 3) vectors to measure.
        coordinate_axes: (n_frame, 3, 3) coordinate system per frame.
        ref_axis: reference axis for angle computation.

    Returns:
        (n_frame, 2) angles: [flexion, adduction] in degrees.
    """
    n_frame = vectors.shape[0]
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    angles = np.zeros((n_frame, 2))
    for i in range(n_frame):
        v = vectors_transformed[i]
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            continue
        v = v / norm

        # Flexion: angle in XY plane
        v_xy = np.array([v[0], v[1]])
        if np.linalg.norm(v_xy) > 1e-8:
            v_xy /= np.linalg.norm(v_xy)
            if ref_axis == "neg_y":
                angles[i, 0] = np.degrees(np.arctan2(v_xy[0], -v_xy[1]))
            else:  # pos_y (for lumbar/neck)
                angles[i, 0] = np.degrees(np.arctan2(v_xy[0], v_xy[1]))

        # Adduction: signed angle between vector and XY projection
        v_yz = np.array([v[1], v[2]])
        if np.linalg.norm(v_yz) > 1e-8:
            v_yz /= np.linalg.norm(v_yz)
            if ref_axis == "neg_y":
                angles[i, 1] = np.degrees(np.arctan2(v_yz[1], -v_yz[0]))
            else:
                angles[i, 1] = np.degrees(np.arctan2(v_yz[1], v_yz[0]))

    return angles


def compute_signed_angle_xy_projection(vectors, coordinate_axes):
    """Compute signed angle between vector and its XY projection."""
    n_frame = vectors.shape[0]
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    projections_xy = np.copy(vectors_transformed)
    projections_xy[:, 2] = 0

    vector_norms = np.linalg.norm(vectors_transformed, axis=1)
    projection_norms = np.linalg.norm(projections_xy, axis=1)

    valid_mask = projection_norms > 1e-8
    angles = np.zeros(n_frame)

    if np.any(valid_mask):
        cos_theta = np.clip(
            np.sum(vectors_transformed[valid_mask] * projections_xy[valid_mask], axis=1) /
            (vector_norms[valid_mask] * projection_norms[valid_mask]),
            -1.0, 1.0
        )
        angles[valid_mask] = np.degrees(np.arccos(cos_theta))
        angles[valid_mask] *= np.sign(vectors_transformed[valid_mask, 2])

    return angles


def compute_hip_rotation(hip_angles, hip_positions, knee_positions, ankle_positions, body_coordinate):
    """Compute hip rotation angle using hip flexion/adduction to derive rotated coordinate."""
    n_frames = hip_angles.shape[0]
    hip_rotation_angles = np.zeros(n_frames)

    for i in range(n_frames):
        flexion = np.radians(hip_angles[i, 0])
        adduction = np.radians(-hip_angles[i, 1])

        body_axes = body_coordinate[i].T
        axis_z = body_axes[:, 2]

        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()
        R_knee_flexion = rot1 @ body_axes

        axis_x = R_knee_flexion[:, 0]
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_knee = rot2 @ R_knee_flexion
        knee_axes = R_knee.T

        ankle_knee_vec = ankle_positions[i] - knee_positions[i]
        ankle_knee_in_knee_coord = knee_axes @ ankle_knee_vec

        proj_xy = np.copy(ankle_knee_in_knee_coord)
        proj_xy[2] = 0

        norm_vec = np.linalg.norm(ankle_knee_in_knee_coord)
        norm_proj = np.linalg.norm(proj_xy)

        if norm_proj < 1e-8 or norm_vec < 1e-8:
            hip_rotation_angles[i] = 0
        else:
            dot = np.dot(ankle_knee_in_knee_coord, proj_xy)
            angle = np.degrees(np.arccos(np.clip(dot / (norm_vec * norm_proj), -1.0, 1.0)))
            hip_rotation_angles[i] = angle * np.sign(ankle_knee_in_knee_coord[2])

    return hip_rotation_angles


def compute_lumbar_rotation(lumbar_angles, neck, right_shoulder, body_coordinate):
    """Compute lumbar rotation from neck-to-shoulder vector under rotated coordinate."""
    n_frames = lumbar_angles.shape[0]
    rotation_angles = np.zeros(n_frames)
    neck_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(-lumbar_angles[i, 0])
        adduction = np.radians(lumbar_angles[i, 1])

        body_axes = body_coordinate[i].T
        axis_z = body_axes[:, 2]
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()
        R_lumbar_flexion = rot1 @ body_axes
        axis_x = R_lumbar_flexion[:, 0]
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_lumbar = rot2 @ R_lumbar_flexion
        neck_axes = R_lumbar.T
        neck_coordinate.append(neck_axes)

        neck_shoulder_vec = right_shoulder[i] - neck[i]
        vec_in_neck = neck_axes @ neck_shoulder_vec

        proj_yz = np.copy(vec_in_neck)
        proj_yz[0] = 0

        norm_vec = np.linalg.norm(vec_in_neck)
        norm_proj = np.linalg.norm(proj_yz)

        if norm_vec < 1e-8 or norm_proj < 1e-8:
            rotation_angles[i] = 0
        else:
            dot = np.dot(vec_in_neck, proj_yz)
            angle = np.degrees(np.arccos(np.clip(dot / (norm_vec * norm_proj), -1.0, 1.0)))
            rotation_angles[i] = angle * np.sign(vec_in_neck[0])

    return rotation_angles, np.array(neck_coordinate)


def compute_shoulder_coordinate(body_coordinate, lumbar_extension, lumbar_bending, lumbar_rotation, neck_coordinate):
    """Compute shoulder coordinate system from body + lumbar rotations."""
    n_frames = body_coordinate.shape[0]
    shoulder_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(-lumbar_extension[i])
        adduction = np.radians(lumbar_bending[i])
        rotation = np.radians(lumbar_rotation[i])

        body_axes = body_coordinate[i].T
        axis_z = body_axes[:, 2]
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()
        R_body_flexion = rot1 @ body_axes
        axis_x = R_body_flexion[:, 0]
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_body_fa = rot2 @ R_body_flexion
        axis_y = R_body_fa[:, 1]
        rot3 = R.from_rotvec(rotation * axis_y).as_matrix()
        R_shoulder = rot3 @ R_body_fa
        shoulder_coordinate.append(R_shoulder.T)

    return np.array(shoulder_coordinate)


def compute_shoulder_rotation(shoulder_angles, elbow_positions, wrist_positions, shoulder_coordinate):
    """Compute shoulder internal/external rotation."""
    n_frames = shoulder_angles.shape[0]
    rotation_angles = np.zeros(n_frames)

    for i in range(n_frames):
        flexion = np.radians(shoulder_angles[i, 0])
        adduction = np.radians(-shoulder_angles[i, 1])

        shoulder_axes = shoulder_coordinate[i].T
        axis_z = shoulder_axes[:, 2]

        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()
        R_sf = rot1 @ shoulder_axes
        axis_x = R_sf[:, 0]
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_shoulder = rot2 @ R_sf
        shoulder_axes = R_shoulder.T

        elbow_wrist_vec = wrist_positions[i] - elbow_positions[i]
        vec_in_shoulder = shoulder_axes @ elbow_wrist_vec

        proj_xy = np.copy(vec_in_shoulder)
        proj_xy[2] = 0

        norm_vec = np.linalg.norm(vec_in_shoulder)
        norm_proj = np.linalg.norm(proj_xy)

        if norm_vec < 1e-8 or norm_proj < 1e-8:
            rotation_angles[i] = 0
        else:
            dot = np.dot(vec_in_shoulder, proj_xy)
            angle = np.degrees(np.arccos(np.clip(dot / (norm_vec * norm_proj), -1.0, 1.0)))
            rotation_angles[i] = angle * np.sign(vec_in_shoulder[2])

    return rotation_angles


# ---------------------------------------------------------------------------------------------------------------------------
# Angle-definition fixes of the v4 audit (plans/ANGLE_AUDIT.md). Default OFF (fix_angle_defs); smd_v4 turns them on.
#   unmirror:      reflect across the vertical plane normal to the frame-0 LABELLED forward, so the clip keeps the dataset's own
#                  heading (HumanML3D humanact12: 0 instead of 180; MotionFix: the partner clip's heading).
#   pelvis:        intrinsic YZX = heading first, then tilt / list about the body's own axes (ZXY put them in the start frame).
#                  |list| <= 90 (tilt in (-180, 180], so a handstand reads tilt ~ 180); frames with |tilt| within
#                  90 - PELVIS_GIMBAL_THR of 90 (lying prone / supine) have ill-defined heading and list, which are interpolated.
#   lumbar rot:    twist of the left -> right COLLAR line (SMPL 13 / 14, rigid on spine3) instead of neck -> right shoulder.
#   shoulder rot:  twist of the forearm about the upper-arm axis (swing-twist from the hanging arm, limb_frames), + = internal
#                  on both sides; elbow < ELBOW_ROT_THR or arm within LIMB_UP_THR of straight up interpolated (all frames -> 0).
#   hip rot:       twist of the lower leg's medio-lateral axis about the thigh axis (swing-twist), + = internal on both sides. The axis is the
#                  knee hinge (thigh x shank) when the knee bends, the ankle hinge (foot x shank, + HIP_ROT_ANKLE_OFFSET) when it
#                  is straight, blended over knee KNEE_AXIS_LO..KNEE_AXIS_HI.
PELVIS_GIMBAL_THR = 80.0
ELBOW_ROT_THR = 20.0
LIMB_UP_THR = 25.0                  # deg from straight up within which the arm / thigh twist is interpolated
KNEE_AXIS_LO, KNEE_AXIS_HI = 20.0, 40.0
HIP_ROT_ANKLE_OFFSET = 0.0          # deg added to the ankle-axis estimate; measured knee - ankle offset is +0..+2.5 on all corpora (ANGLE_FIX_CHECK_calib.md)


def unmirror_keep_heading(ric):
    bc = compute_body_coordinate(ric[:, 2], ric[:, 1], ric[:, 17], ric[:, 16], ric[:, 0])
    f = bc[0, 0].copy(); f[1] = 0.0; f /= np.linalg.norm(f) + 1e-12
    return ric - 2.0 * np.einsum("tjk,k->tj", ric, f)[..., None] * f


def interp_bad(x, bad):
    """linear interpolation of x over frames `bad` (held at the ends); unchanged if none / all are bad."""
    x = np.asarray(x, float).copy()
    if not bad.any() or bad.all():
        return x, 0
    t = np.arange(len(x)); x[bad] = np.interp(t[bad], t[~bad], x[~bad])
    return x, int(bad.sum())


def wrap180(a):
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def pelvis_euler_yzx(body_coordinate):
    """-> (tilt, list, rotation) in deg; signs as the old ZXY output (tilt + back, list + right, rotation + left)."""
    e = R.from_matrix(np.transpose(body_coordinate, (0, 2, 1))).as_euler("YZX", degrees=True)
    rot, tilt, lst = e[:, 0], e[:, 1], e[:, 2]
    # equivalent solution (rot + 180, 180 - tilt, list + 180): keep |list| <= 90 so pitching over (handstand, roll) reads as
    # |tilt| > 90 at a continuous heading instead of list ~ 180 with the heading flipped
    flip = np.abs(lst) > 90
    rot, tilt, lst = np.where(flip, rot + 180, rot), wrap180(np.where(flip, 180 - tilt, tilt)), wrap180(np.where(flip, lst + 180, lst))
    bad = np.abs(np.abs(tilt) - 90) < 90 - PELVIS_GIMBAL_THR
    rot, n = interp_bad(np.unwrap(rot, period=360.0), bad)
    lst, _ = interp_bad(np.unwrap(lst, period=360.0), bad)
    return tilt, wrap180(lst), wrap180(rot), n


def limb_frames(frame, limb_dir, side):
    """(T,3,3) frame rows (x fwd, y up, z right) -> per-frame (x, y, z) world axes of the limb, y = -limb direction. The transverse
    reference x is the frame's forward axis carried along by the SWING (shortest-arc rotation) from the hanging direction (-y) to
    the measured limb direction. For a limb raised in the sagittal or the frontal plane this is the frame of the flexion-then-
    adduction Euler angles (ISB order), but it does not depend on those angles, whose flexion is singular (and interpolated)
    with the limb out to the side. The swing is singular with the limb pointing straight up (see LIMB_UP_THR)."""
    d0 = np.array([0.0, -1.0, 0.0]); x0 = np.array([1.0, 0.0, 0.0])
    d = limb_dir / (np.linalg.norm(limb_dir, axis=1, keepdims=True) + 1e-12)
    dl = np.einsum("nij,nj->ni", frame, d)                            # limb direction in frame coordinates
    k = np.cross(np.broadcast_to(d0, dl.shape), dl); c = dl @ d0
    xl = x0 * c[:, None] + np.cross(k, x0) + k * ((k @ x0) / (1.0 + c + 1e-9))[:, None]
    x = np.einsum("nji,nj->ni", frame, xl); y = -d
    x = x - (x * y).sum(1, keepdims=True) * y; x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x, y, np.cross(x, y)


def limb_up_mask(frame, limb_dir):
    """frames whose limb points within LIMB_UP_THR deg of the frame's +y (twist undefined there)."""
    d = limb_dir / (np.linalg.norm(limb_dir, axis=1, keepdims=True) + 1e-12)
    return np.einsum("nij,nj->ni", frame, d)[:, 1] > np.cos(np.radians(LIMB_UP_THR))


def interp_or_zero(x, bad):
    """interp_bad, but a clip whose frames are all bad (e.g. arms straight throughout) gets 0 = undefined, not noise."""
    if bad.all():
        return np.zeros(len(x)), int(bad.sum())
    return interp_bad(x, bad)


def _limb_twist(frame, limb_dir, vec, side, arm):
    x, _, z = limb_frames(frame, limb_dir, side)
    if arm:                                                           # forearm direction; raw + = toward the body's right
        a = np.degrees(np.arctan2((vec * z).sum(1), (vec * x).sum(1))); return -a if side == "r" else a
    a = np.degrees(np.arctan2(-(vec * x).sum(1), -(vec * z).sum(1)))  # lower-leg medio-lateral axis (points to the body's left)
    return a if side == "r" else -a


def _neutral_twist(side, arm):
    """twist read in the neutral pose (limb straight down; forearm forward / knee axis pointing left), subtracted so it reads 0."""
    f = np.eye(3)[None]; down = np.array([[0.0, -1.0, 0.0]])
    return float(_limb_twist(f, down, np.array([[1.0, 0.0, 0.0]]) if arm else np.array([[0.0, 0.0, -1.0]]), side, arm)[0])


def shoulder_rotation_atan2(frame, shoulder, elbow, wrist, side, elbow_angle):
    """+ = internal on both sides; 0 = forearm pointing forward with the upper arm hanging. Elbow < ELBOW_ROT_THR interpolated."""
    a = wrap180(_limb_twist(frame, elbow - shoulder, wrist - elbow, side, True) - _neutral_twist(side, True))
    a, n = interp_or_zero(np.unwrap(a, period=360.0), (elbow_angle < ELBOW_ROT_THR) | limb_up_mask(frame, elbow - shoulder))
    return wrap180(a), n


def hip_rotation_axes(frame, hip, knee, ankle, foot, side, knee_angle, return_parts=False):
    """+ = internal on both sides; 0 = knee / ankle hinge axis pointing to the body's left with the thigh hanging."""
    off = _neutral_twist(side, False); sgn = 1.0 if side == "r" else -1.0
    a_knee = wrap180(_limb_twist(frame, knee - hip, np.cross(knee - hip, ankle - knee), side, False) - off)
    a_ankle = wrap180(_limb_twist(frame, knee - hip, np.cross(foot - ankle, ankle - knee), side, False) - off + sgn * HIP_ROT_ANKLE_OFFSET)
    w = np.clip((knee_angle - KNEE_AXIS_LO) / (KNEE_AXIS_HI - KNEE_AXIS_LO), 0.0, 1.0)
    a = wrap180(a_ankle + w * wrap180(a_knee - a_ankle))
    a = wrap180(interp_or_zero(np.unwrap(a, period=360.0), limb_up_mask(frame, knee - hip))[0])
    return (a, a_knee, a_ankle) if return_parts else a


def lumbar_rotation_collar(neck_coordinate, left_collar, right_collar):
    """twist of the left -> right collar line in the lumbar-flexed / bent frame; + = shoulders twisted to the left."""
    v = np.einsum("nij,nj->ni", neck_coordinate, right_collar - left_collar)
    return np.degrees(np.arctan2(v[:, 0], v[:, 2]))


def calculate_hinge_angle(parent, joint, child):
    """Calculate hinge joint angle (knee, elbow, ankle)."""
    v1 = joint - parent
    v2 = child - joint
    dot = np.einsum('ij,ij->i', v1, v2)
    mag1 = np.linalg.norm(v1, axis=1)
    mag2 = np.linalg.norm(v2, axis=1)
    denom = np.clip(mag1 * mag2, 1e-8, None)
    return np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))


def compute_all_joint_angles(joints: np.ndarray, coord_system: str = "humanml3d", fix_chirality: bool = False,
                             fix_flex_singularity: bool = False, info: dict = None, fix_angle_defs: bool = False) -> dict:
    """
    Compute all biomechanical joint angles from 22-joint positions.

    Args:
        joints: (T, 22, 3) joint positions.
        coord_system: input coordinate system.
            "humanml3d": X=left, Y=up, Z=forward (HumanML3D new_joints)
            "babel":     X=right, Y=forward, Z=up (BABEL-QA, HuMMan-QA, AMASS)

    Returns:
        dict with angle name -> (T,) array of angles in degrees.
    """
    # Target (ric_data): X=forward, Y=up, Z=right (right-hand system)
    # Ref: motion2angle_22joints_v1/joint_angle_cal.py line 965-966
    #
    # Output pelvis_pos mapping:
    #   pelvis_pos_x = forward
    #   pelvis_pos_y = up/height
    #   pelvis_pos_z = right
    if coord_system == "humanml3d":
        # HumanML3D: X=left, Y=up, Z=forward
        # → [2,1,0]: X'=Z(fwd), Y'=Y(up), Z'=X(left) → Z*=-1: Z'=right
        ric_data = joints[:, :, [2, 1, 0]].copy()
        ric_data[:, :, 2] *= -1
    elif coord_system == "babel":
        # BABEL/AMASS: X=right, Y=forward, Z=up
        # → [1,2,0]: X'=Y(fwd), Y'=Z(up), Z'=X(right)
        ric_data = joints[:, :, [1, 2, 0]].copy()
    else:
        raise ValueError(f"Unknown coord_system: {coord_system}. Use 'humanml3d' or 'babel'.")

    if fix_chirality:
        score, knee_s, toe_s = chirality_score(ric_data, return_parts=True)
        if info is not None: info.update(chirality_score=score, chirality_knee=knee_s, chirality_toe=toe_s, mirrored=score < CHIRALITY_THR)
        if score < CHIRALITY_THR:
            if fix_angle_defs:
                ric_data = unmirror_keep_heading(ric_data)               # un-mirror, keep the labelled frame-0 heading
            else:
                ric_data = ric_data.copy(); ric_data[:, :, 2] *= -1      # un-mirror the geometry; joint labels unchanged

    # Extract joint positions
    pelvis = ric_data[:, 0, :]
    left_hip = ric_data[:, 1, :]
    right_hip = ric_data[:, 2, :]
    left_knee = ric_data[:, 4, :]
    right_knee = ric_data[:, 5, :]
    left_ankle = ric_data[:, 7, :]
    right_ankle = ric_data[:, 8, :]
    left_foot = ric_data[:, 10, :]
    right_foot = ric_data[:, 11, :]
    neck = ric_data[:, 12, :]
    left_collar = ric_data[:, 13, :]
    right_collar = ric_data[:, 14, :]
    head = ric_data[:, 15, :]
    left_shoulder = ric_data[:, 16, :]
    right_shoulder = ric_data[:, 17, :]
    left_elbow = ric_data[:, 18, :]
    right_elbow = ric_data[:, 19, :]
    left_wrist = ric_data[:, 20, :]
    right_wrist = ric_data[:, 21, :]

    # Body coordinate system
    body_coord = compute_body_coordinate(right_hip, left_hip, right_shoulder, left_shoulder, pelvis)

    # Pelvis Euler angles
    if fix_angle_defs:
        tilt, lst, rot, n = pelvis_euler_yzx(body_coord)
        pelvis_angles = np.stack([tilt, lst, rot], axis=1)
        if info is not None: info["interp_pelvis"] = n
    else:
        pelvis_angles = calculate_pelvis_euler_angles(body_coord)

    # Hip angles
    r_hip_angles = calculate_joint_angles(right_knee - right_hip, body_coord, ref_axis="neg_y")
    r_hip_angles[:, 1] = compute_signed_angle_xy_projection(right_knee - right_hip, body_coord)
    if fix_flex_singularity:
        r_hip_angles[:, 0], n = regularize_flexion(r_hip_angles[:, 0], sagittal_fraction(right_knee - right_hip, body_coord))
        if info is not None: info["interp_hip_r"] = n
    r_hip_rotation = compute_hip_rotation(r_hip_angles[:, :2], right_hip, right_knee, right_ankle, body_coord)

    l_hip_angles = calculate_joint_angles(left_knee - left_hip, body_coord, ref_axis="neg_y")
    l_hip_angles[:, 1] = compute_signed_angle_xy_projection(left_knee - left_hip, body_coord)
    if fix_flex_singularity:
        l_hip_angles[:, 0], n = regularize_flexion(l_hip_angles[:, 0], sagittal_fraction(left_knee - left_hip, body_coord))
        if info is not None: info["interp_hip_l"] = n
    l_hip_rotation = compute_hip_rotation(l_hip_angles[:, :2], left_hip, left_knee, left_ankle, body_coord)

    # Knee and ankle
    r_knee_angle = calculate_hinge_angle(right_hip, right_knee, right_ankle)
    l_knee_angle = calculate_hinge_angle(left_hip, left_knee, left_ankle)
    r_ankle_angle = calculate_hinge_angle(right_knee, right_ankle, right_foot)
    l_ankle_angle = calculate_hinge_angle(left_knee, left_ankle, left_foot)

    # Lumbar
    lumbar_vector = neck - pelvis
    lumbar_angles_fa = calculate_joint_angles(lumbar_vector, body_coord, ref_axis="pos_y")
    lumbar_extension = lumbar_angles_fa[:, 0]
    lumbar_bending = compute_signed_angle_xy_projection(lumbar_vector, body_coord)
    lumbar_angles_2d = np.stack([lumbar_extension, lumbar_bending], axis=1)
    lumbar_rotation, neck_coord = compute_lumbar_rotation(lumbar_angles_2d, neck, right_shoulder, body_coord)
    if fix_angle_defs:
        lumbar_rotation = lumbar_rotation_collar(neck_coord, left_collar, right_collar)

    # Shoulder coordinate
    shoulder_coord = compute_shoulder_coordinate(body_coord, lumbar_extension, lumbar_bending, lumbar_rotation, neck_coord)

    # Shoulder angles
    r_shoulder_angles = calculate_joint_angles(right_elbow - right_shoulder, shoulder_coord, ref_axis="neg_y")
    r_shoulder_angles[:, 1] = compute_signed_angle_xy_projection(right_elbow - right_shoulder, shoulder_coord)
    if fix_flex_singularity:
        r_shoulder_angles[:, 0], n = regularize_flexion(r_shoulder_angles[:, 0], sagittal_fraction(right_elbow - right_shoulder, shoulder_coord))
        if info is not None: info["interp_arm_r"] = n
    r_shoulder_rotation = compute_shoulder_rotation(r_shoulder_angles[:, :2], right_elbow, right_wrist, shoulder_coord)

    l_shoulder_angles = calculate_joint_angles(left_elbow - left_shoulder, shoulder_coord, ref_axis="neg_y")
    l_shoulder_angles[:, 1] = compute_signed_angle_xy_projection(left_elbow - left_shoulder, shoulder_coord)
    if fix_flex_singularity:
        l_shoulder_angles[:, 0], n = regularize_flexion(l_shoulder_angles[:, 0], sagittal_fraction(left_elbow - left_shoulder, shoulder_coord))
        if info is not None: info["interp_arm_l"] = n
    l_shoulder_rotation = compute_shoulder_rotation(l_shoulder_angles[:, :2], left_elbow, left_wrist, shoulder_coord)

    # Elbow
    r_elbow_angle = calculate_hinge_angle(right_shoulder, right_elbow, right_wrist)
    l_elbow_angle = calculate_hinge_angle(left_shoulder, left_elbow, left_wrist)

    # Neck
    neck_angles = calculate_joint_angles(head - neck, shoulder_coord, ref_axis="pos_y")
    neck_angles[:, 1] = compute_signed_angle_xy_projection(head - neck, shoulder_coord)

    # v4 audit fixes of the rotation DOFs (need knee / elbow angles)
    if fix_angle_defs:
        hip_rot_r = hip_rotation_axes(body_coord, right_hip, right_knee, right_ankle, right_foot, "r", r_knee_angle)
        hip_rot_l = hip_rotation_axes(body_coord, left_hip, left_knee, left_ankle, left_foot, "l", l_knee_angle)
        arm_rot_r, n_r = shoulder_rotation_atan2(shoulder_coord, right_shoulder, right_elbow, right_wrist, "r", r_elbow_angle)
        arm_rot_l, n_l = shoulder_rotation_atan2(shoulder_coord, left_shoulder, left_elbow, left_wrist, "l", l_elbow_angle)
        if info is not None: info.update(interp_rot_arm_r=n_r, interp_rot_arm_l=n_l)
    else:
        hip_rot_r, hip_rot_l = r_hip_rotation, -l_hip_rotation
        arm_rot_r, arm_rot_l = -r_shoulder_rotation, -l_shoulder_rotation

    # Global trajectory info (in angle_calculator coords: X=forward, Y=up, Z=across)
    # ric_data pelvis = ric_data[:, 0, :] which is already in target coords
    pelvis_global = ric_data[:, 0, :]  # (T, 3) in (X-forward, Y-up, Z-across)
    # Per-frame velocity (displacement between consecutive frames)
    pelvis_vel = np.zeros_like(pelvis_global)
    pelvis_vel[1:] = pelvis_global[1:] - pelvis_global[:-1]

    return {
        # Global trajectory (X=forward, Y=up, Z=across)
        "pelvis_pos_x": pelvis_global[:, 0],   # forward position
        "pelvis_pos_y": pelvis_global[:, 1],   # height
        "pelvis_pos_z": pelvis_global[:, 2],   # lateral position
        "pelvis_vel_x": pelvis_vel[:, 0],      # forward velocity
        "pelvis_vel_y": pelvis_vel[:, 1],      # vertical velocity
        "pelvis_vel_z": pelvis_vel[:, 2],      # lateral velocity
        # Joint angles
        "pelvis_tilt": pelvis_angles[:, 0],
        "pelvis_list": pelvis_angles[:, 1],
        "pelvis_rotation": pelvis_angles[:, 2],
        "hip_flexion_r": r_hip_angles[:, 0],
        "hip_adduction_r": -r_hip_angles[:, 1],
        "hip_rotation_r": hip_rot_r,
        "knee_angle_r": r_knee_angle,
        "ankle_angle_r": r_ankle_angle - 90,
        "hip_flexion_l": l_hip_angles[:, 0],
        "hip_adduction_l": l_hip_angles[:, 1],
        "hip_rotation_l": hip_rot_l,
        "knee_angle_l": l_knee_angle,
        "ankle_angle_l": l_ankle_angle - 90,
        "lumbar_extension": -lumbar_extension,
        "lumbar_bending": lumbar_bending,
        "lumbar_rotation": lumbar_rotation,
        "arm_flex_r": r_shoulder_angles[:, 0],
        "arm_add_r": -r_shoulder_angles[:, 1],
        "arm_rot_r": arm_rot_r,
        "arm_flex_l": l_shoulder_angles[:, 0],
        "arm_add_l": l_shoulder_angles[:, 1],
        "arm_rot_l": arm_rot_l,
        "elbow_flex_r": r_elbow_angle,
        "elbow_flex_l": l_elbow_angle,
        "neck_flexion": neck_angles[:, 0],
        "neck_adduction": neck_angles[:, 1],
    }
