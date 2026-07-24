"""
inference/pose.py — MediaPipe pose extraction utilities.

Key design: PoseLandmarker is expensive to initialise (~1–2 s).
Instantiate once at startup via get_pose_landmarker() which caches
the instance module-globally. Never call it per-request.
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

NUM_LANDMARKS = 33
FEATURE_DIM   = 163   # 132 + 27 + 4

def pad_to_square(image: np.ndarray) -> np.ndarray:
    """Pad an image to a 1:1 square aspect ratio using black borders."""
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

# Module-level singleton — initialised lazily on first call
_pose_landmarker = None


def get_pose_landmarker(model_path: str = "pose_landmarker_heavy.task") -> vision.PoseLandmarker:
    """
    Return the module-level PoseLandmarker, creating it if needed.
    Call once at app startup:  get_pose_landmarker(model_path)
    Subsequent calls return the cached instance regardless of model_path.
    """
    global _pose_landmarker
    if _pose_landmarker is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MediaPipe model not found at '{model_path}'. "
                "Run:  wget -q https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
            )
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            num_poses=1,
        )
        _pose_landmarker = vision.PoseLandmarker.create_from_options(options)
    return _pose_landmarker


def extract_landmarks_from_video(
    video_path: str,
    num_frames: int = 30,
    model_path: str = "pose_landmarker_heavy.task",
) -> np.ndarray | None:
    """
    Sample num_frames evenly from the video and run MediaPipe pose detection.

    Returns:
        np.ndarray of shape (num_frames, 33, 3) — (x, y, visibility) per joint.
        None if the video cannot be opened or pose detection fails on every frame.
    """
    landmarker = get_pose_landmarker(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    seq = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            seq.append(seq[-1] if seq else np.zeros((NUM_LANDMARKS, 3)))
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        W = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        H = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        aspect_ratio = W / H if H > 0 else 1.0

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            seq.append(np.array([[l.x * aspect_ratio, l.y, l.visibility] for l in lm]))
        else:
            seq.append(seq[-1] if seq else np.zeros((NUM_LANDMARKS, 3)))

    cap.release()

    if not seq:
        return None

    return np.array(seq)  # (T, 33, 3)


def normalise_landmarks(seq: np.ndarray) -> np.ndarray:
    """
    Make landmarks camera- and distance-invariant.
    Origin  → hip midpoint
    Scale   → torso length (hip-mid to shoulder-mid)
    seq: (T, 33, 3)  returns same shape (visibility col unchanged)
    """
    seq = seq.copy()
    hip_mid      = (seq[:, 23, :2] + seq[:, 24, :2]) / 2          # (T, 2)
    shoulder_mid = (seq[:, 11, :2] + seq[:, 12, :2]) / 2          # (T, 2)
    scale        = np.linalg.norm(shoulder_mid - hip_mid, axis=1)  # (T,)
    scale        = np.maximum(scale, 1e-6)[:, np.newaxis]          # (T, 1)

    seq[:, :, :2] = (seq[:, :, :2] - hip_mid[:, np.newaxis, :]) / scale[:, np.newaxis, :]
    return seq


def compute_symmetry_features(angles_mean: np.ndarray) -> np.ndarray:
    from .angles import ANGLE_NAMES
    SYMMETRY_PAIRS = [
        (ANGLE_NAMES.index('left_knee'),     ANGLE_NAMES.index('right_knee')),
        (ANGLE_NAMES.index('left_hip'),      ANGLE_NAMES.index('right_hip')),
        (ANGLE_NAMES.index('left_elbow'),    ANGLE_NAMES.index('right_elbow')),
        (ANGLE_NAMES.index('left_shoulder'), ANGLE_NAMES.index('right_shoulder')),
    ]
    return np.array([
        abs(angles_mean[l] - angles_mean[r]) for l, r in SYMMETRY_PAIRS
    ])

def build_feature_vector(seq_norm: np.ndarray, angles_seq: np.ndarray) -> np.ndarray:
    """
    Convert (T, 33, 3) landmark sequence and angles into a 163-dim feature vector.
    """
    coords = seq_norm[:, :, :2]
    coord_mean = coords.mean(axis=0).flatten()
    coord_std  = coords.std(axis=0).flatten()
    
    angle_mean = angles_seq.mean(axis=0)
    angle_std  = angles_seq.std(axis=0)
    angle_vel  = np.abs(np.diff(angles_seq, axis=0)).mean(axis=0) if len(angles_seq) > 1 else np.zeros_like(angle_mean)
    
    sym = compute_symmetry_features(angle_mean)
    return np.concatenate([coord_mean, coord_std, angle_mean, angle_std, angle_vel, sym])


def extract_mid_frame_rgb(video_path: str) -> tuple[np.ndarray | None, int]:
    """
    Extract the middle frame of a video as an RGB image.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None, 0

    mid = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    
    # Fallback: OpenCV CAP_PROP_POS_FRAMES often fails on mobile/VFR videos.
    # If seeking fails, read sequentially from the start to reach the mid frame.
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(mid + 1):
            ret, frame = cap.read()
            if not ret:
                break
                
    cap.release()

    if not ret or frame is None:
        return None, mid

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), mid