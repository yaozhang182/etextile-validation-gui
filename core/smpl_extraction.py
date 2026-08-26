"""
Skeleton extraction from video using MediaPipe Pose.

Lightweight alternative to heavy GPU-based SMPL models.
Runs on CPU, single `pip install mediapipe` dependency.

Pipeline:
  Video (MP4) -> MediaPipe Pose (33 landmarks per frame)
  -> Convert to SMPLH 22-joint format -> joint angle computation

MediaPipe Pose provides 33 3D world landmarks covering all major body joints.
We map these to the SMPLH 22-joint format expected by joint_angles.py.
"""

import numpy as np
from pathlib import Path


# =============================================================================
# MediaPipe -> SMPLH 22-joint Mapping
# =============================================================================

# MediaPipe Pose landmark indices
MP_NOSE = 0
MP_LEFT_SHOULDER = 11
MP_RIGHT_SHOULDER = 12
MP_LEFT_ELBOW = 13
MP_RIGHT_ELBOW = 14
MP_LEFT_WRIST = 15
MP_RIGHT_WRIST = 16
MP_LEFT_HIP = 23
MP_RIGHT_HIP = 24
MP_LEFT_KNEE = 25
MP_RIGHT_KNEE = 26
MP_LEFT_ANKLE = 27
MP_RIGHT_ANKLE = 28
MP_LEFT_FOOT_INDEX = 31
MP_RIGHT_FOOT_INDEX = 32


def mediapipe_to_smplh22(mp_landmarks):
    """
    Convert MediaPipe 33 world landmarks to SMPLH 22-joint format.

    Args:
        mp_landmarks: (T, 33, 3) or (33, 3)
            MediaPipe world coords: X=right, Y=down, Z=towards_camera

    Returns:
        (T, 22, 3) or (22, 3) in SMPLH convention (X=right, Y=up, Z=forward)
    """
    single_frame = mp_landmarks.ndim == 2
    if single_frame:
        mp_landmarks = mp_landmarks[np.newaxis]

    T = mp_landmarks.shape[0]
    smplh = np.zeros((T, 22, 3), dtype=np.float32)
    mp = mp_landmarks

    # Direct mappings
    smplh[:, 1] = mp[:, MP_LEFT_HIP]          # left_hip
    smplh[:, 2] = mp[:, MP_RIGHT_HIP]         # right_hip
    smplh[:, 4] = mp[:, MP_LEFT_KNEE]         # left_knee
    smplh[:, 5] = mp[:, MP_RIGHT_KNEE]        # right_knee
    smplh[:, 7] = mp[:, MP_LEFT_ANKLE]        # left_ankle
    smplh[:, 8] = mp[:, MP_RIGHT_ANKLE]       # right_ankle
    smplh[:, 10] = mp[:, MP_LEFT_FOOT_INDEX]  # left_foot
    smplh[:, 11] = mp[:, MP_RIGHT_FOOT_INDEX] # right_foot
    smplh[:, 16] = mp[:, MP_LEFT_SHOULDER]    # left_shoulder
    smplh[:, 17] = mp[:, MP_RIGHT_SHOULDER]   # right_shoulder
    smplh[:, 18] = mp[:, MP_LEFT_ELBOW]       # left_elbow
    smplh[:, 19] = mp[:, MP_RIGHT_ELBOW]      # right_elbow
    smplh[:, 20] = mp[:, MP_LEFT_WRIST]       # left_wrist
    smplh[:, 21] = mp[:, MP_RIGHT_WRIST]      # right_wrist

    # Synthesized joints
    hip_mid = (mp[:, MP_LEFT_HIP] + mp[:, MP_RIGHT_HIP]) / 2.0
    neck = (mp[:, MP_LEFT_SHOULDER] + mp[:, MP_RIGHT_SHOULDER]) / 2.0

    # Pelvis must NOT equal hip_mid — compute_body_coordinate needs pelvis
    # slightly above hip_mid to derive the "up" direction. In SMPLX, pelvis is
    # a distinct joint ~2-5% along the spine from hip_mid. We offset by 5%.
    pelvis = hip_mid + 0.05 * (neck - hip_mid)

    smplh[:, 0] = pelvis                                  # pelvis
    smplh[:, 12] = neck                                   # neck
    smplh[:, 15] = mp[:, MP_NOSE]                         # head
    smplh[:, 13] = (neck + mp[:, MP_LEFT_SHOULDER]) / 2.0   # left_collar
    smplh[:, 14] = (neck + mp[:, MP_RIGHT_SHOULDER]) / 2.0  # right_collar

    # Spine: interpolate between pelvis and neck
    spine_vec = neck - pelvis
    smplh[:, 3] = pelvis + spine_vec * 0.25   # spine1
    smplh[:, 6] = pelvis + spine_vec * 0.50   # spine2
    smplh[:, 9] = pelvis + spine_vec * 0.75   # spine3

    # Coordinate transform: MediaPipe (Y-down, Z-backward) -> SMPLH (Y-up, Z-forward)
    smplh[:, :, 1] *= -1
    smplh[:, :, 2] *= -1

    if single_frame:
        return smplh[0]
    return smplh


# =============================================================================
# Video -> Skeleton Extraction
# =============================================================================

def extract_skeleton_from_video(video_path, output_path=None,
                                progress_callback=None,
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5):
    """
    Extract 3D skeleton from video using MediaPipe Pose.

    Args:
        video_path: path to MP4/AVI video
        output_path: save result as NPZ (optional)
        progress_callback: callable(frame_idx, total_frames) for GUI
        min_detection_confidence: MediaPipe detection threshold
        min_tracking_confidence: MediaPipe tracking threshold

    Returns:
        dict with:
            'landmarks_3d': (T, 33, 3) MediaPipe world landmarks
            'smplh_joints': (T, 22, 3) SMPLH format joints
            'visibility': (T, 33) per-landmark confidence in [0, 1]
            'fps': video frame rate
            'total_frames': int
            'valid_frames': int (frames where pose was detected)
    """
    import cv2
    import mediapipe as mp_lib

    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    T_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mp_pose = mp_lib.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,  # 0=lite, 1=full, 2=heavy (best accuracy)
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    all_landmarks = []
    all_visibility = []
    valid_count = 0

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_world_landmarks:
            lm = results.pose_world_landmarks.landmark
            coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32)
            # MediaPipe's own occlusion/accuracy classifier. Kept so the GUI can
            # tell the user which channels to distrust.
            vis = np.array([l.visibility for l in lm], dtype=np.float32)
            all_landmarks.append(coords)
            all_visibility.append(vis)
            valid_count += 1
        else:
            all_landmarks.append(np.full((33, 3), np.nan, dtype=np.float32))
            all_visibility.append(np.zeros(33, dtype=np.float32))

        frame_idx += 1
        if progress_callback:
            progress_callback(frame_idx, T_total)

    cap.release()
    pose.close()

    if valid_count == 0:
        raise RuntimeError(f"No pose detected in any frame of: {video_path}")

    landmarks_3d = np.stack(all_landmarks, axis=0)  # (T, 33, 3)
    visibility = np.stack(all_visibility, axis=0)   # (T, 33)
    smplh_joints = mediapipe_to_smplh22(landmarks_3d)  # (T, 22, 3)

    result = {
        'landmarks_3d': landmarks_3d,
        'smplh_joints': smplh_joints,
        'visibility': visibility,
        'fps': fps,
        'total_frames': T_total,
        'valid_frames': valid_count,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(output_path), **result)

    return result


def video_to_joint_angles(video_path, progress_callback=None):
    """
    End-to-end: Video -> ALL joint angles.

    Args:
        video_path: path to video file
        progress_callback: callable(frame_idx, total_frames)

    Returns:
        angles: dict of {angle_name: (T,) array in degrees}
        time: (T,) time array in seconds
        skeleton_data: raw extraction result dict
    """
    from core.joint_angles import extract_all_joint_angles

    skeleton_data = extract_skeleton_from_video(
        video_path, progress_callback=progress_callback,
    )

    smplh_joints = skeleton_data['smplh_joints']
    fps = skeleton_data['fps']
    angles, time = extract_all_joint_angles(smplh_joints, fps=fps)

    return angles, time, skeleton_data
