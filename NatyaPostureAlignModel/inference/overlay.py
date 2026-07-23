"""
inference/overlay.py — Skeleton overlay image generation.

Draws MediaPipe connections on a dimmed video frame.
Green = joint within threshold, Red = flagged joint.
Returns the image as a BGR numpy array and/or saves to disk.
"""

from __future__ import annotations
import base64
import os

import cv2
import numpy as np

from .angles import ANGLE_DEFS

# ---------------------------------------------------------------------------
# Colour scheme (BGR for OpenCV)
# ---------------------------------------------------------------------------

COLOR_GOOD        = (50,  200,  50)    # green
COLOR_BAD         = (50,   50, 220)    # red
COLOR_NEUTRAL     = (180, 180, 180)    # grey — face / non-scored joints
COLOR_BONE_GOOD   = (100, 220, 100)
COLOR_BONE_BAD    = (100,  80, 220)

# MediaPipe skeleton connections
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]

# Landmark indices that are part of any scored angle
SCORED_LANDMARKS: set[int] = set()
for _, a, v, c in ANGLE_DEFS:
    SCORED_LANDMARKS.update([a, v, c])


# ---------------------------------------------------------------------------
# Core drawing
# ---------------------------------------------------------------------------

def draw_skeleton_overlay(
    frame_rgb: np.ndarray,
    landmarks_frame: np.ndarray,
    flagged_joint_names: set[str],
    img_size: tuple[int, int] = (480, 480),
    adavu_label: str = "",
) -> np.ndarray:
    """
    Render an annotated skeleton on a dimmed background.

    Args:
        frame_rgb:           (H, W, 3) RGB video frame
        landmarks_frame:     (33, 3) normalised (x, y, visibility) landmarks
        flagged_joint_names: set of joint names that were flagged
        img_size:            (W, H) output size in pixels
        adavu_label:         text to overlay at the top of the image

    Returns:
        (H, W, 3) BGR annotated image
    """
    orig_H, orig_W = frame_rgb.shape[:2]
    # Scale image so max dimension is 480 pixels
    scale = 480.0 / max(orig_H, orig_W)
    new_W, new_H = int(orig_W * scale), int(orig_H * scale)
    
    bg = cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (new_W, new_H))
    canvas = (bg * 0.35).astype(np.uint8)

    lm = landmarks_frame   # (33, 3)
    
    # Landmarks were computed on a padded square of size = max(orig_H, orig_W)
    # We must un-pad them to match the original frame dimensions!
    size = max(orig_H, orig_W)
    pad_h = (size - orig_H) // 2
    pad_w = (size - orig_W) // 2

    # Map flagged joint names → landmark indices
    flagged_indices: set[int] = set()
    for name, a, v, c in ANGLE_DEFS:
        if name in flagged_joint_names:
            flagged_indices.update([a, v, c])

    def lm_px(idx: int) -> tuple[int, int]:
        px_in_orig = (lm[idx, 0] * size) - pad_w
        py_in_orig = (lm[idx, 1] * size) - pad_h
        return (int(px_in_orig * scale), int(py_in_orig * scale))

    def joint_color(idx: int) -> tuple[int, int, int]:
        if idx not in SCORED_LANDMARKS:
            return COLOR_NEUTRAL
        return COLOR_BAD if idx in flagged_indices else COLOR_GOOD

    def bone_color(a: int, b: int) -> tuple[int, int, int]:
        if a in flagged_indices or b in flagged_indices:
            return COLOR_BONE_BAD
        if a not in SCORED_LANDMARKS or b not in SCORED_LANDMARKS:
            return COLOR_NEUTRAL
        return COLOR_BONE_GOOD

    # Bones
    for a, b in POSE_CONNECTIONS:
        if lm[a, 2] < 0.3 or lm[b, 2] < 0.3:
            continue
        cv2.line(canvas, lm_px(a), lm_px(b), bone_color(a, b), 2, cv2.LINE_AA)

    # Joints
    for i in range(33):
        if lm[i, 2] < 0.3:
            continue
        px  = lm_px(i)
        col = joint_color(i)
        r   = 6 if i in SCORED_LANDMARKS else 3
        cv2.circle(canvas, px, r, col,          -1, cv2.LINE_AA)
        cv2.circle(canvas, px, r, (255, 255, 255), 1, cv2.LINE_AA)

    # Adavu label at top
    if adavu_label:
        cv2.putText(
            canvas, adavu_label, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    # Legend
    legend_y = new_H - 60
    cv2.rectangle(canvas, (8, legend_y - 10), (220, new_H - 8), (0, 0, 0), -1)
    cv2.circle(canvas,  (22, legend_y + 6),   5, COLOR_GOOD, -1)
    cv2.putText(canvas, "Within range",     (32, legend_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_GOOD, 1, cv2.LINE_AA)
    cv2.circle(canvas,  (22, legend_y + 24), 5, COLOR_BAD,  -1)
    cv2.putText(canvas, "Needs correction", (32, legend_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_BAD,  1, cv2.LINE_AA)

    return canvas


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_overlay_image(
    video_path: str,
    seq: np.ndarray,
    flagged_joints: list[dict],
    adavu_class: str,
    out_dir: str = "/tmp",
    img_size: tuple[int, int] = (480, 480),
) -> str | None:
    """
    Generate the overlay for the mid-video frame and save as JPEG.

    Returns:
        Path to the saved image, or None on failure.
    """
    from .pose import extract_mid_frame_rgb

    flagged_names = {j["joint"] for j in flagged_joints}
    frame_rgb, _ = extract_mid_frame_rgb(video_path)
    if frame_rgb is None:
        return None

    mid = len(seq) // 2
    canvas = draw_skeleton_overlay(
        frame_rgb, seq[mid], flagged_names, img_size, adavu_label=adavu_class
    )

    safe = adavu_class.replace(" ", "_")
    out_path = os.path.join(out_dir, f"overlay_{safe}.jpg")
    cv2.imwrite(out_path, canvas)
    return out_path


def overlay_to_base64(image_bgr: np.ndarray) -> str:
    """Encode a BGR image as a base64 JPEG string (for JSON responses)."""
    _, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode("utf-8")