"""
Joint angle computation module.

Handles:
1. SMPLX-137 -> SMPLH-52 joint format conversion
2. Clinical joint angle extraction for ALL body joints:
   - Pelvis (3 angles: flexion, adduction, rotation via Euler)
   - Hip L/R (3 each: flexion, abduction, rotation)
   - Knee L/R (1 each: flexion)
   - Ankle L/R (1 each: flexion)
   - Lumbar (3: extension, bending, rotation)
   - Shoulder L/R (3 each: flexion, abduction, rotation)
   - Elbow L/R (1 each: flexion)
   - Neck (2: flexion, bending)

Consolidated from shoulderLLM:
  - position2angle/convert_smplx_to_smplh.py
  - position2angle/joint_angle_cal_nohand.py
  - utils/extract_shoulder_angles.py
"""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import scipy.ndimage.filters as filters
import warnings


def _safe_normalize(v, axis=-1):
    """Normalize vectors, returning zero for degenerate (zero-length) vectors."""
    norms = np.linalg.norm(v, axis=axis, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)  # avoid division by zero
    return v / norms


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
# Coordinate System Validation
# =============================================================================

def is_right_hand_coordinate(coordinate_axes):
    """Check whether the given coordinate system is right-handed."""
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=1, keepdims=True)
    dot_xy = np.dot(coordinate_axes[0], coordinate_axes[1])
    dot_yz = np.dot(coordinate_axes[1], coordinate_axes[2])
    dot_zx = np.dot(coordinate_axes[2], coordinate_axes[0])
    orthogonality = np.allclose([dot_xy, dot_yz, dot_zx], [0, 0, 0], atol=1e-6)
    cross_product = np.cross(coordinate_axes[0], coordinate_axes[1])
    right_hand = np.allclose(cross_product, coordinate_axes[2], atol=1e-3)
    return orthogonality and right_hand


# =============================================================================
# Body Coordinate System
# =============================================================================

def compute_body_coordinate(right_hip, left_hip, right_shoulder, left_shoulder,
                            pelvis, smooth_forward=False):
    """
    Compute body-centric coordinate system from joint positions.

    Returns:
        (T, 3, 3) array where each frame has [forward, up, across] axes.
    """
    across = right_hip - left_hip
    across = _safe_normalize(across)

    hip_mid = (right_hip + left_hip) / 2
    up = pelvis - hip_mid
    up = _safe_normalize(up)

    # Make up perpendicular to across
    up = up - np.sum(up * across, axis=-1, keepdims=True) * across
    up = _safe_normalize(up)

    forward = np.cross(up, across)
    forward = _safe_normalize(forward)

    if smooth_forward:
        forward = filters.gaussian_filter1d(forward, 20, axis=0, mode='nearest')

    human_coordinate = np.array([forward, up, across])
    return np.transpose(human_coordinate, (1, 0, 2))


# =============================================================================
# Pelvis Angles
# =============================================================================

def calculate_pelvis_euler_angles(body_coordinate_vectors):
    """
    Calculate pelvis orientation using intrinsic ZXY Euler angles.

    Returns:
        (T, 3) Euler angles in degrees: [flexion (X), adduction (Y), rotation (Z)]
    """
    n_frame = body_coordinate_vectors.shape[0]
    euler_angles = np.zeros((n_frame, 3))
    for i in range(n_frame):
        rotation_matrix = body_coordinate_vectors[i].T
        if np.any(np.isnan(rotation_matrix)):
            euler_angles[i] = np.nan
            continue
        try:
            rot = R.from_matrix(rotation_matrix)
            euler_angles[i] = rot.as_euler('ZXY', degrees=True)
        except (np.linalg.LinAlgError, ValueError):
            euler_angles[i] = np.nan
    return euler_angles


# =============================================================================
# Hip Angles
# =============================================================================

def calculate_hip_angles(vectors, coordinate_axes):
    """Calculate hip flexion and abduction angles."""
    n_frame = vectors.shape[0]
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    angles = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])
        if np.linalg.norm(v_xy) > 1e-8:
            v_xy /= np.linalg.norm(v_xy)
            angles[i, 0] = np.degrees(np.arctan2(v_xy[0], -v_xy[1]))
        v_yz = np.array([v[1], v[2]])
        if np.linalg.norm(v_yz) > 1e-8:
            v_yz /= np.linalg.norm(v_yz)
            angles[i, 1] = np.degrees(np.arctan2(v_yz[1], -v_yz[0]))
    return angles


def compute_hip_rotation(hip_angles, hip_positions, knee_positions, ankle_positions,
                         body_coordinate):
    """Compute hip rotation angle using hip flexion/adduction and knee-ankle vector."""
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


# =============================================================================
# Knee / Ankle / Elbow (simple 2-segment angles)
# =============================================================================

def calculate_knee_angle(hip, knee, ankle):
    """Calculate knee flexion angle from hip-knee-ankle chain."""
    thigh = knee - hip
    shank = ankle - knee
    dot = np.einsum('ij,ij->i', thigh, shank)
    mag_thigh = np.linalg.norm(thigh, axis=1)
    mag_shank = np.linalg.norm(shank, axis=1)
    denom = np.clip(mag_thigh * mag_shank, 1e-8, None)
    return np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))


def calculate_ankle_angle(knee, ankle, foot):
    """Calculate ankle joint angle from knee-ankle-foot chain."""
    shank = ankle - knee
    foot_vec = foot - ankle
    dot = np.einsum('ij,ij->i', shank, foot_vec)
    mag_shank = np.linalg.norm(shank, axis=1)
    mag_foot = np.linalg.norm(foot_vec, axis=1)
    denom = np.clip(mag_shank * mag_foot, 1e-8, None)
    return np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))


def calculate_elbow_angle(shoulder, elbow, wrist):
    """Calculate elbow flexion angle from shoulder-elbow-wrist chain."""
    upper_arm = elbow - shoulder
    forearm = wrist - elbow
    dot = np.einsum('ij,ij->i', upper_arm, forearm)
    mag_upper = np.linalg.norm(upper_arm, axis=1)
    mag_fore = np.linalg.norm(forearm, axis=1)
    denom = np.clip(mag_upper * mag_fore, 1e-8, None)
    return np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))


# =============================================================================
# Lumbar Angles
# =============================================================================

def calculate_lumbar_angles(vectors, coordinate_axes):
    """Calculate lumbar flexion and bending angles."""
    n_frame = vectors.shape[0]
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    angles = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])
        if np.linalg.norm(v_xy) > 1e-8:
            v_xy /= np.linalg.norm(v_xy)
            angles[i, 0] = np.degrees(np.arctan2(v_xy[0], v_xy[1]))
        v_yz = np.array([v[1], v[2]])
        if np.linalg.norm(v_yz) > 1e-8:
            v_yz /= np.linalg.norm(v_yz)
            angles[i, 1] = np.degrees(np.arctan2(v_yz[1], v_yz[0]))
    return angles


def compute_lumbar_rotation(lumbar_angles, neck_positions, shoulder_positions, body_coordinate):
    """Compute lumbar rotation from neck-to-shoulder vector under rotated coordinate system."""
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

        neck_shoulder_vec = shoulder_positions[i] - neck_positions[i]
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


# =============================================================================
# Neck Angles
# =============================================================================

def calculate_neck_angles(vectors, coordinate_axes):
    """Calculate neck flexion and bending (same JCS as shoulder)."""
    n_frame = vectors.shape[0]
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    angles = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])
        if np.linalg.norm(v_xy) > 1e-8:
            v_xy /= np.linalg.norm(v_xy)
            angles[i, 0] = np.degrees(np.arctan2(v_xy[0], v_xy[1]))
        v_yz = np.array([v[1], v[2]])
        if np.linalg.norm(v_yz) > 1e-8:
            v_yz /= np.linalg.norm(v_yz)
            angles[i, 1] = np.degrees(np.arctan2(v_yz[1], v_yz[0]))
    return angles


# =============================================================================
# Shoulder Angles
# =============================================================================

def calculate_shoulder_angles(vectors, coordinate_axes):
    """Calculate shoulder flexion and abduction from upper-arm vector in shoulder frame."""
    n_frame = vectors.shape[0]
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    angles = np.zeros((n_frame, 3))
    for i in range(n_frame):
        v = vectors_transformed[i] / np.linalg.norm(vectors_transformed[i])
        v_xy = np.array([v[0], v[1]])
        if np.linalg.norm(v_xy) > 1e-8:
            v_xy /= np.linalg.norm(v_xy)
            angles[i, 0] = np.degrees(np.arctan2(v_xy[0], -v_xy[1]))
        v_yz = np.array([v[1], v[2]])
        if np.linalg.norm(v_yz) > 1e-8:
            v_yz /= np.linalg.norm(v_yz)
            angles[i, 1] = np.degrees(np.arctan2(v_yz[1], -v_yz[0]))
    return angles


def compute_signed_angle_xy_projection(vectors, coordinate_axes):
    """Compute signed angle between vector and its XY-plane projection."""
    n_frame = vectors.shape[0]
    coordinate_axes = coordinate_axes / np.linalg.norm(coordinate_axes, axis=2, keepdims=True)
    vectors_transformed = np.einsum('nij,nj->ni', coordinate_axes, vectors)

    projections_xy = np.copy(vectors_transformed)
    projections_xy[:, 2] = 0

    vector_norms = np.linalg.norm(vectors_transformed, axis=1)
    projection_norms = np.linalg.norm(projections_xy, axis=1)

    valid_mask = projection_norms > 1e-8
    angles = np.full(n_frame, np.nan)

    if np.any(valid_mask):
        cos_theta = np.clip(
            np.sum(vectors_transformed[valid_mask] * projections_xy[valid_mask], axis=1) /
            (vector_norms[valid_mask] * projection_norms[valid_mask]),
            -1.0, 1.0
        )
        angles[valid_mask] = np.degrees(np.arccos(cos_theta))
        angles[valid_mask] *= np.sign(vectors_transformed[valid_mask, 2])

    return angles


def compute_shoulder_coordinate(body_coordinate, lumbar_flexion, lumbar_adduction,
                                lumbar_rotation, neck_coordinate):
    """Compute shoulder coordinate system from body frame and lumbar angles."""
    n_frames = body_coordinate.shape[0]
    shoulder_coordinate = []

    for i in range(n_frames):
        flexion = np.radians(-lumbar_flexion[i])
        adduction = np.radians(lumbar_adduction[i])
        rotation = np.radians(lumbar_rotation[i])

        body_axes = body_coordinate[i].T
        axis_z = body_axes[:, 2]
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()
        R_body_flexion = rot1 @ body_axes
        axis_x = R_body_flexion[:, 0]
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_body_flexion_adduction = rot2 @ R_body_flexion
        axis_y = R_body_flexion_adduction[:, 1]
        rot3 = R.from_rotvec(rotation * axis_y).as_matrix()
        R_shoulder = rot3 @ R_body_flexion_adduction
        shoulder_coordinate.append(R_shoulder.T)

    return np.array(shoulder_coordinate)


def compute_shoulder_rotation(shoulder_angles, elbow_positions, wrist_positions,
                              shoulder_coordinate):
    """Compute shoulder internal/external rotation angle."""
    n_frames = shoulder_angles.shape[0]
    rotation_angles = np.zeros(n_frames)

    for i in range(n_frames):
        flexion = np.radians(shoulder_angles[i, 0])
        adduction = np.radians(-shoulder_angles[i, 1])

        shoulder_axes = shoulder_coordinate[i].T
        axis_z = shoulder_axes[:, 2]
        rot1 = R.from_rotvec(flexion * axis_z).as_matrix()
        R_shoulder_flexion = rot1 @ shoulder_axes
        axis_x = R_shoulder_flexion[:, 0]
        rot2 = R.from_rotvec(adduction * axis_x).as_matrix()
        R_shoulder = rot2 @ R_shoulder_flexion
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


# =============================================================================
# OpenSim .sto Export
# =============================================================================

def generate_sto_file(time_array, joint_data_dict, output_filename):
    """
    Generate an OpenSim .sto file from joint angle data.

    Args:
        time_array: (T,) time values
        joint_data_dict: dict of {joint_name: (T,) angle array}
        output_filename: output file path
    """
    length = len(time_array)
    for joint, data in joint_data_dict.items():
        if len(data) != length:
            raise ValueError(f"Length mismatch: {joint} has {len(data)}, expected {length}")

    data = {'time': time_array}
    data.update(joint_data_dict)
    df = pd.DataFrame(data)

    header = [
        "inDegrees=yes",
        f"name={output_filename}",
        "DataType=double",
        "version=3",
        "OpenSimVersion=4.5",
        "endheader",
    ]

    with open(output_filename, 'w') as f:
        for line in header:
            f.write(line + "\n")
        f.write('\t'.join(df.columns) + "\n")
        df.to_csv(f, sep='\t', index=False, header=False, float_format='%.6f')


# =============================================================================
# Angle Unwrapping
# =============================================================================

def unwrap_angles(values):
    """
    Unwrap angle time series to remove +/-180 deg discontinuities.
    Uses shortest-path difference on the circle.
    NaN values are skipped (preserved as-is).
    """
    unwrapped = np.copy(values).astype(float)
    for i in range(1, len(unwrapped)):
        if np.isnan(unwrapped[i]) or np.isnan(unwrapped[i - 1]):
            continue
        diff = unwrapped[i] - unwrapped[i - 1]
        diff = ((diff + 180) % 360) - 180
        unwrapped[i] = unwrapped[i - 1] + diff
    return unwrapped


# Angles that may wrap around +/-180 deg (rotational DOFs and large-range flexions)
WRAPPABLE_ANGLES = {
    'pelvis_flexion', 'pelvis_adduction', 'pelvis_rotation',
    'left_hip_rotation', 'right_hip_rotation',
    'lumbar_rotation',
    'left_shoulder_flexion', 'right_shoulder_flexion',
    'left_shoulder_rotation', 'right_shoulder_rotation',
    'neck_flexion', 'neck_bending',
}


# =============================================================================
# High-Level API: Extract ALL Joint Angles
# =============================================================================

def _prepare_joint_positions(joint_positions):
    """Validate and prepare joint positions, converting to body-only 22-joint format."""
    if joint_positions.ndim != 3:
        raise ValueError(f"Expected 3D array (T, N, 3), got shape {joint_positions.shape}")

    num_joints = joint_positions.shape[1]
    if num_joints == 52:
        joint_positions = joint_positions[:, :22, :]
    elif num_joints != 22:
        raise ValueError(f"Expected 22 or 52 joints, got {num_joints}")

    # Reorder coordinates: (x, y, z) -> (z, y, -x), the ric_data need x-forward, y-up, z-right
    ric_data = joint_positions[:, :, [2, 1, 0]]
    ric_data[:, :, 2] *= -1
    return ric_data


def extract_all_joint_angles(joint_positions, fps=20):
    """
    Extract ALL body joint angles from SMPLH joint positions.

    Args:
        joint_positions: (T, 22, 3) or (T, 52, 3) SMPLH joint positions
        fps: frames per second

    Returns:
        dict: {angle_name: (T,) array in degrees}
              Keys include:
                pelvis_flexion, pelvis_adduction, pelvis_rotation,
                left_hip_flexion, left_hip_abduction, left_hip_rotation,
                right_hip_flexion, right_hip_abduction, right_hip_rotation,
                left_knee_flexion, right_knee_flexion,
                left_ankle_flexion, right_ankle_flexion,
                lumbar_extension, lumbar_bending, lumbar_rotation,
                left_shoulder_flexion, left_shoulder_abduction, left_shoulder_rotation,
                right_shoulder_flexion, right_shoulder_abduction, right_shoulder_rotation,
                left_elbow_flexion, right_elbow_flexion,
                neck_flexion, neck_bending
        time: (T,) time array in seconds
    """
    ric_data = _prepare_joint_positions(joint_positions)
    T = ric_data.shape[0]

    # Extract all joint positions (SMPLH body indices)
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
    head = ric_data[:, 15, :]
    left_shoulder = ric_data[:, 16, :]
    right_shoulder = ric_data[:, 17, :]
    left_elbow = ric_data[:, 18, :]
    right_elbow = ric_data[:, 19, :]
    left_wrist = ric_data[:, 20, :]
    right_wrist = ric_data[:, 21, :]

    # --- Body coordinate system ---
    body_coord = compute_body_coordinate(
        right_hip, left_hip, right_shoulder, left_shoulder, pelvis, smooth_forward=False
    )

    # --- Pelvis ---
    pelvis_angles = calculate_pelvis_euler_angles(body_coord)

    # --- Hip ---
    right_hip_angles = calculate_hip_angles(right_knee - right_hip, body_coord)
    right_hip_angles[:, 1] = compute_signed_angle_xy_projection(right_knee - right_hip, body_coord)
    right_hip_angles[:, 2] = compute_hip_rotation(
        right_hip_angles[:, :2], right_hip, right_knee, right_ankle, body_coord
    )

    left_hip_angles = calculate_hip_angles(left_knee - left_hip, body_coord)
    left_hip_angles[:, 1] = compute_signed_angle_xy_projection(left_knee - left_hip, body_coord)
    left_hip_angles[:, 2] = compute_hip_rotation(
        left_hip_angles[:, :2], left_hip, left_knee, left_ankle, body_coord
    )

    # --- Knee ---
    right_knee_angle = calculate_knee_angle(right_hip, right_knee, right_ankle)
    left_knee_angle = calculate_knee_angle(left_hip, left_knee, left_ankle)

    # --- Ankle ---
    right_ankle_angle = calculate_ankle_angle(right_knee, right_ankle, right_foot)
    left_ankle_angle = calculate_ankle_angle(left_knee, left_ankle, left_foot)

    # --- Lumbar ---
    lumbar_vector = neck - pelvis
    lumbar_extension = calculate_lumbar_angles(lumbar_vector, body_coord)[:, 0]
    lumbar_bending = compute_signed_angle_xy_projection(lumbar_vector, body_coord)
    lumbar_angles_stacked = np.stack([lumbar_extension, lumbar_bending], axis=1)
    lumbar_rot, neck_coord = compute_lumbar_rotation(
        lumbar_angles_stacked, neck, right_shoulder, body_coord
    )

    # --- Shoulder coordinate system (shared by shoulder and neck) ---
    shoulder_coord = compute_shoulder_coordinate(
        body_coord, lumbar_extension, lumbar_bending, lumbar_rot, neck_coord
    )

    # --- Shoulder (right) ---
    r_shoulder_angles = calculate_shoulder_angles(right_elbow - right_shoulder, shoulder_coord)
    r_shoulder_angles[:, 1] = compute_signed_angle_xy_projection(
        right_elbow - right_shoulder, shoulder_coord
    )
    r_shoulder_angles[:, 2] = compute_shoulder_rotation(
        r_shoulder_angles[:, :2], right_elbow, right_wrist, shoulder_coord
    )
    # Sign corrections for right shoulder
    r_shoulder_angles[:, 1] = -r_shoulder_angles[:, 1]
    r_shoulder_angles[:, 2] = -r_shoulder_angles[:, 2]

    # --- Shoulder (left) ---
    l_shoulder_angles = calculate_shoulder_angles(left_elbow - left_shoulder, shoulder_coord)
    l_shoulder_angles[:, 1] = compute_signed_angle_xy_projection(
        left_elbow - left_shoulder, shoulder_coord
    )
    l_shoulder_angles[:, 2] = compute_shoulder_rotation(
        l_shoulder_angles[:, :2], left_elbow, left_wrist, shoulder_coord
    )
    # Sign corrections for left shoulder
    l_shoulder_angles[:, 2] = -l_shoulder_angles[:, 2]

    # --- Elbow ---
    right_elbow_angle = calculate_elbow_angle(right_shoulder, right_elbow, right_wrist)
    left_elbow_angle = calculate_elbow_angle(left_shoulder, left_elbow, left_wrist)

    # --- Neck ---
    neck_angles = calculate_neck_angles(head - neck, shoulder_coord)
    neck_angles[:, 1] = compute_signed_angle_xy_projection(head - neck, shoulder_coord)

    # --- Assemble result dict ---
    angles = {
        'pelvis_flexion': pelvis_angles[:, 0],
        'pelvis_adduction': pelvis_angles[:, 1],
        'pelvis_rotation': pelvis_angles[:, 2],

        'left_hip_flexion': left_hip_angles[:, 0],
        'left_hip_abduction': left_hip_angles[:, 1],
        'left_hip_rotation': left_hip_angles[:, 2],

        'right_hip_flexion': right_hip_angles[:, 0],
        'right_hip_abduction': right_hip_angles[:, 1],
        'right_hip_rotation': right_hip_angles[:, 2],

        'left_knee_flexion': left_knee_angle,
        'right_knee_flexion': right_knee_angle,

        'left_ankle_flexion': left_ankle_angle,
        'right_ankle_flexion': right_ankle_angle,

        'lumbar_extension': lumbar_extension,
        'lumbar_bending': lumbar_bending,
        'lumbar_rotation': lumbar_rot,

        'left_shoulder_flexion': l_shoulder_angles[:, 0],
        'left_shoulder_abduction': l_shoulder_angles[:, 1],
        'left_shoulder_rotation': l_shoulder_angles[:, 2],

        'right_shoulder_flexion': r_shoulder_angles[:, 0],
        'right_shoulder_abduction': r_shoulder_angles[:, 1],
        'right_shoulder_rotation': r_shoulder_angles[:, 2],

        'left_elbow_flexion': left_elbow_angle,
        'right_elbow_flexion': right_elbow_angle,

        'neck_flexion': neck_angles[:, 0],
        'neck_bending': neck_angles[:, 1],
    }

    # Unwrap angles that may have +/-180 deg discontinuities
    for name in WRAPPABLE_ANGLES:
        if name in angles:
            angles[name] = unwrap_angles(angles[name])

    time = np.linspace(0, T / fps, T)
    return angles, time


def extract_shoulder_angles(joint_positions, fps=20, left_shoulder=True):
    """
    Extract shoulder angles only (convenience wrapper).

    Args:
        joint_positions: (T, 22, 3) or (T, 52, 3)
        fps: frames per second
        left_shoulder: True for left, False for right

    Returns:
        shoulder_angles: (T, 3) [flexion, abduction, rotation] in degrees
        time: (T,) time array
    """
    all_angles, time = extract_all_joint_angles(joint_positions, fps=fps)

    side = 'left' if left_shoulder else 'right'
    shoulder_angles = np.stack([
        all_angles[f'{side}_shoulder_flexion'],
        all_angles[f'{side}_shoulder_abduction'],
        all_angles[f'{side}_shoulder_rotation'],
    ], axis=1)

    return shoulder_angles, time


def smplx_to_joint_angles(smplx_joints, fps=20):
    """
    End-to-end: SMPLX joints -> all joint angles.

    Args:
        smplx_joints: (T, 137, 3) SMPLX joint positions
        fps: frames per second

    Returns:
        angles: dict of {angle_name: (T,) array}
        time: (T,) time array
    """
    smplh = convert_smplx_to_smplh(smplx_joints)
    return extract_all_joint_angles(smplh, fps=fps)


def smplx_to_shoulder_angles(smplx_joints, fps=20, left_shoulder=True):
    """
    End-to-end: SMPLX joints -> shoulder angles only.

    Args:
        smplx_joints: (T, 137, 3) SMPLX joint positions
        fps: frames per second
        left_shoulder: which shoulder

    Returns:
        shoulder_angles: (T, 3) [flexion, abduction, rotation]
        time: (T,) time array
    """
    smplh = convert_smplx_to_smplh(smplx_joints)
    return extract_shoulder_angles(smplh, fps=fps, left_shoulder=left_shoulder)
