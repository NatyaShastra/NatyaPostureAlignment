import os
import numpy as np
import cv2
import mediapipe as mp

NUM_LANDMARKS = 33
FEATURE_DIM   = 174

def pad_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w:
        return image
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    return cv2.copyMakeBorder(
        image, pad_h, size - h - pad_h, pad_w, size - w - pad_w,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

_pose_landmarker = None

def get_pose_landmarker(model_path: str = "pose_landmarker_heavy.task"):
    global _pose_landmarker
    if _pose_landmarker is None:
        _pose_landmarker = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=0, 
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return _pose_landmarker

def extract_landmarks_from_video(
    video_path: str,
    num_frames: int = 120,
    model_path: str = "pose_landmarker_heavy.task",
) -> np.ndarray | None:
    landmarker = get_pose_landmarker(model_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    target_indices = set(indices)
    max_idx = max(target_indices) if target_indices else -1

    raw_seq_dict = {}
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx > max_idx:
            break

        if frame_idx in target_indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_padded = pad_to_square(rgb)
            result = landmarker.process(rgb_padded)

            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                raw_seq_dict[frame_idx] = np.array([[l.x, l.y, l.visibility] for l in lm])

        frame_idx += 1

    cap.release()
    seq = []
    for idx in indices:
        if idx in raw_seq_dict:
            seq.append(raw_seq_dict[idx])
        else:
            seq.append(seq[-1] if seq else np.zeros((NUM_LANDMARKS, 3)))

    if not seq:
        return None
    return np.array(seq)

def normalise_landmarks(seq: np.ndarray) -> np.ndarray:
    seq = seq.copy()
    hip_mid      = (seq[:, 23, :2] + seq[:, 24, :2]) / 2          
    shoulder_mid = (seq[:, 11, :2] + seq[:, 12, :2]) / 2          
    scale        = np.linalg.norm(shoulder_mid - hip_mid, axis=1)  
    scale        = np.maximum(scale, 1e-6)[:, np.newaxis]          
    seq[:, :, :2] = (seq[:, :, :2] - hip_mid[:, np.newaxis, :]) / scale[:, np.newaxis, :]
    return seq

def compute_symmetry_features(angles_mean: np.ndarray) -> np.ndarray:
    from .angles import ANGLE_NAMES
    SYMMETRY_PAIRS = [
        (ANGLE_NAMES.index('left_shoulder'), ANGLE_NAMES.index('right_shoulder')),
        (ANGLE_NAMES.index('left_elbow'),    ANGLE_NAMES.index('right_elbow')),
        (ANGLE_NAMES.index('left_wrist'),    ANGLE_NAMES.index('right_wrist')),
        (ANGLE_NAMES.index('left_hip'),      ANGLE_NAMES.index('right_hip')),
        (ANGLE_NAMES.index('left_knee'),     ANGLE_NAMES.index('right_knee')),
        (ANGLE_NAMES.index('left_ankle'),    ANGLE_NAMES.index('right_ankle')),
    ]
    return np.array([abs(angles_mean[l] - angles_mean[r]) for l, r in SYMMETRY_PAIRS])

def build_feature_vector(seq_norm: np.ndarray, angles_seq: np.ndarray) -> np.ndarray:
    coords = seq_norm[:, :, :2]
    coord_mean = coords.mean(axis=0).flatten()
    coord_std  = coords.std(axis=0).flatten()
    angle_mean = angles_seq.mean(axis=0)
    angle_std  = angles_seq.std(axis=0)
    angle_vel  = np.abs(np.diff(angles_seq, axis=0)).mean(axis=0) if len(angles_seq) > 1 else np.zeros_like(angle_mean)
    sym = compute_symmetry_features(angle_mean)
    return np.concatenate([coord_mean, coord_std, angle_mean, angle_std, angle_vel, sym])

def extract_mid_frame_rgb(video_path: str, target_idx: int) -> tuple[np.ndarray | None, int]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(target_idx + 1):
            ret, frame = cap.read()
            if not ret: break
    cap.release()
    if not ret or frame is None:
        return None, 0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pad_to_square(rgb), target_idx

def extract_frames_rgb(video_path: str, target_indices: list[int]) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results: dict[int, np.ndarray] = {}
    if total <= 0 or not target_indices:
        cap.release()
        return results
    for idx in sorted(set(target_indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for _ in range(idx + 1):
                ret, frame = cap.read()
                if not ret: break
        if ret and frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results[idx] = pad_to_square(rgb)
    cap.release()
    return results