"""
inference/__init__.py

Exports the high-level run_coach_v2() pipeline function and the
startup initialisation helper load_model_and_refs().

Usage (FastAPI lifespan or script):
    from inference import load_model_and_refs, run_coach_v2
    load_model_and_refs(
        checkpoint_path="checkpoints/dance_coach_model.pt",
        features_cache="checkpoints/adavu_features.npz",
        mediapipe_model="pose_landmarker_heavy.task",
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )
    result = run_coach_v2("student_video.mp4")
"""

from __future__ import annotations
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.classifier import AdavuClassifier
from .pose     import (
    get_pose_landmarker,
    extract_landmarks_from_video,
    build_feature_vector,
    extract_mid_frame_rgb,
    normalise_landmarks,
)
from .angles   import build_angle_refs, compute_angles_from_sequence, detect_mistakes
from .scoring  import compute_score
from .feedback import init_groq_client, get_llm_feedback
from .overlay  import save_overlay_image, overlay_to_base64, draw_skeleton_overlay

# ---------------------------------------------------------------------------
# Module-level model state (populated by load_model_and_refs)
# ---------------------------------------------------------------------------

_model:   AdavuClassifier | None = None
_le       = None        # sklearn LabelEncoder
_X_mean:  np.ndarray | None = None
_X_std:   np.ndarray | None = None
_device   = "cpu"


# ---------------------------------------------------------------------------
# Startup initialiser
# ---------------------------------------------------------------------------

def load_model_and_refs(
    checkpoint_path: str = "checkpoints/dance_coach_model.pt",
    features_cache:  str = "checkpoints/adavu_features.npz",
    mediapipe_model: str = "pose_landmarker_heavy.task",
    groq_api_key:    str | None = None,
) -> None:
    """
    Load all inference dependencies. Call once at application startup.

    # --- Feature cache → angle reference distributions --------------------
    if not os.path.exists(features_cache):
        raise FileNotFoundError(f"Feature cache not found: {features_cache}")
        
    data = np.load(features_cache, allow_pickle=True)
    build_angle_refs(data["X"], data["y"])
    print(f"[startup] Angle refs built from {len(data['X'])} cached samples")

    # --- MediaPipe ---------------------------------------------------------
    get_pose_landmarker(mediapipe_model)
    print(f"[startup] MediaPipe PoseLandmarker ready")

    # --- Groq --------------------------------------------------------------
    init_groq_client(groq_api_key)
    if groq_api_key or os.environ.get("GROQ_API_KEY"):
        print(f"[startup] Groq client initialised")
    else:
        print(f"[startup] No GROQ_API_KEY — will use template feedback")


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

def run_coach_v2(
    video_path:      str,
    target_adavu:    str  = "Thattadavu",
    num_frames:      int  = 30,
    top_k:           int  = 3,
    return_overlay:  bool = True,
    overlay_out_dir: str  = "/tmp",
) -> dict:
    """
    Full Dance Coach pipeline: pose → classify → angle analysis → score → feedback → overlay.
    """
    # --- Step 1: Pose extraction ------------------------------------------
    seq = extract_landmarks_from_video(video_path, num_frames)
    if seq is None:
        return {"error": "Could not extract pose from video. Check that the video contains a visible person."}

    # --- Step 2: Skip classification and use target_adavu -----------------
    seq_norm = normalise_landmarks(seq)
    
    adavu_class = target_adavu
    confidence  = 1.0
    top_preds   = [(target_adavu, 1.0)]

    # --- Step 3: Angular analysis -----------------------------------------
    angles            = compute_angles_from_sequence(seq_norm)
    student_mean      = angles.mean(axis=0)
    flagged, region_scores, _ = detect_mistakes(student_mean, adavu_class)

    # --- Step 4: Scoring --------------------------------------------------
    score_result = compute_score(region_scores, adavu_class)

    # --- Step 5: LLM feedback ---------------------------------------------
    feedback_text, feedback_source = get_llm_feedback(
        adavu_class, score_result, flagged, confidence
    )

    # --- Step 6: Overlay --------------------------------------------------
    overlay_b64 = None
    if return_overlay:
        try:
            from .overlay import draw_skeleton_overlay, overlay_to_base64
            flagged_names = {j["joint"] for j in flagged}
            frame_rgb, _ = extract_mid_frame_rgb(video_path)
            if frame_rgb is not None:
                mid    = len(seq) // 2
                canvas = draw_skeleton_overlay(
                    frame_rgb, seq[mid], flagged_names, adavu_label=adavu_class
                )
                overlay_b64 = overlay_to_base64(canvas)
        except Exception as e:
            print(f"  Overlay generation failed: {e}")

    # --- Assemble result --------------------------------------------------
    return {
        "adavu_class":       adavu_class,
        "confidence":        confidence,
        "top_k_predictions": top_preds,
        "overall_score":     score_result["overall_score"],
        "region_scores":     score_result["region_scores"],
        "passed":            score_result["passed"],
        "grade":             score_result["grade"],
        "grade_message":     score_result["grade_message"],
        "pass_threshold":    score_result["pass_threshold"],
        "needed_to_pass":    score_result["needed_to_pass"],
        "flagged_joints":    flagged,
        "coaching_feedback": feedback_text,
        "feedback_source":   feedback_source,
        "overlay_image_b64": overlay_b64,
    }


__all__ = ["load_model_and_refs", "run_coach_v2"]