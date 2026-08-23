"""
inference/__init__.py

Exports the high-level pipelines for all three categories:
- run_coach_v2 (Dance Steps)
- run_posture_coach (Static Postures)
- run_hasta_coach (Hastas)

And the startup initialisation helper load_model_and_refs().
"""

from __future__ import annotations
import os

import numpy as np
import torch
torch.set_num_threads(1) # Conserve RAM by limiting thread pools
import torch.nn.functional as F

from models.classifier import AdavuClassifier
from .pose     import get_pose_landmarker, extract_landmarks_from_video, build_feature_vector
from .angles   import build_angle_refs, compute_angles_from_sequence, detect_mistakes
from .scoring  import compute_score
from .feedback import init_groq_client, get_llm_feedback
from .overlay  import save_overlay_image, overlay_to_base64, draw_skeleton_overlay
from .pose     import extract_mid_frame_rgb

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
    postures_path:   str = "checkpoints/posture_model.pt",
    hastas_path:     str = "checkpoints/hastas_model.pt",
    mediapipe_model: str = "pose_landmarker_heavy.task",
    groq_api_key:    str | None = None,
) -> None:
    """
    Load all inference dependencies. Call once at application startup.

    - Loads the MLP checkpoint (model weights + label encoder + normalisation stats)
    - Builds per-class angle reference distributions from the feature cache
    - Initialises the MediaPipe PoseLandmarker singleton
    - Initialises the Groq client if an API key is provided
    """
    global _model, _le, _X_mean, _X_std, _device

    # --- Model checkpoint -------------------------------------------------
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt    = torch.load(checkpoint_path, map_location=_device, weights_only=False)

    _le     = ckpt["label_encoder"]
    _X_mean = ckpt["X_mean"]
    _X_std  = ckpt["X_std"]
    num_classes = ckpt["num_classes"]
    input_dim   = ckpt["feature_dim"]

    _model = AdavuClassifier(input_dim=input_dim, num_classes=num_classes).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    print(f"[startup] Model loaded — {num_classes} classes on {_device}")

    # --- Feature cache → angle reference distributions --------------------
    if "angle_refs" in ckpt:
        from .angles import set_angle_refs
        set_angle_refs(ckpt["angle_refs"])
        print(f"[startup] Angle refs loaded from checkpoint ({len(ckpt['angle_refs'])} classes)")
    else:
        print("[startup] No dynamic angle refs found in checkpoint!")

    # --- Postures and Hastas ----------------------------------------------
    from .postures import load_posture_refs
    load_posture_refs(postures_path)
    
    from .hands import build_hasta_refs
    
    if os.path.exists("checkpoints/hastas_features.npz"):
        data = np.load("checkpoints/hastas_features.npz", allow_pickle=True)
        hasta_refs = build_hasta_refs(data["X"], data["y"])
        # Store them in a global dict in hands.py
        import inference.hands as hands_module
        hands_module._hasta_refs = hasta_refs
        print(f"[startup] Hasta refs built from {len(data['X'])} cached samples")

    # --- Groq --------------------------------------------------------------
    init_groq_client(groq_api_key)
    if groq_api_key or os.environ.get("GROQ_API_KEY"):
        print(f"[startup] Groq client initialised")
    else:
        print(f"[startup] No GROQ_API_KEY — will use template feedback")


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

# Map model class names to the actual master video filenames in Munu directory
CLASS_TO_FILE = {
    "Thattadavu 1": "THA_FR_GE_01",
    "Thattadavu 2": "THA_FR_GE_02",
    "Thattadavu 3": "THA_FR_GE_03",
    "Thattadavu 4": "THA_FR_GE_04",
    "Thattadavu 5": "THA_FT_GE_05",
    "Thattadavu 6": "THA_FT_GE_06",
    "Thattadavu 7": "THA_FT_GE_07",
    "Thattadavu 8": "THA_FT_GE_08",
    "Naatadavu 1": "NAA_FR_GE_01",
    "Naatadavu 2": "NAA_FR_GE_02",
    "Naatadavu 3": "NAA_FR_GD_03",
    "Naatadavu 4": "NAA_FR_GD_04",
    "Naatadavu 5": "NAA_FT_GD_05",
    "Naatadavu 6": "NAA_FT_GE_06",
    "Naatadavu 7": "NAA_FT_GE_07",
    "Naatadavu 8": "NAA_FT_GE_08",
}
FILE_TO_CLASS = {v: k for k, v in CLASS_TO_FILE.items()}

def run_coach_v2(
    video_path:      str,
    target_class:    str  = None,
    num_frames:      int  = 120,
    top_k:           int  = 3,
    return_overlay:  bool = True,
    overlay_out_dir: str  = "/tmp",
) -> dict:
    """
    Full Dance Coach pipeline: pose → classify → angle analysis → score → feedback → overlay.
    """
    if _model is None:
        raise RuntimeError("Call load_model_and_refs() before run_coach_v2()")

    # Map frontend's file-based target_class to the model's actual class name if needed
    if target_class and target_class in FILE_TO_CLASS:
        target_class = FILE_TO_CLASS[target_class]

    # --- Step 1: Pose extraction ------------------------------------------
    seq = extract_landmarks_from_video(video_path, num_frames)
    if seq is None:
        return {"error": "Could not extract pose from video. Check that the video contains a visible person."}

    from .pose import normalise_landmarks
    seq_norm = normalise_landmarks(seq)
    angles = compute_angles_from_sequence(seq_norm)

    # --- Step 2: MLP classification (or bypass) ---------------------------
    if target_class:
        adavu_class = target_class
        confidence = 1.0
        top_preds = [(target_class, 1.0)]
    else:
        fv      = build_feature_vector(seq_norm, angles)
        fv_norm = (fv - _X_mean) / (_X_std + 1e-8)
        fv_t    = torch.FloatTensor(fv_norm).unsqueeze(0).to(_device)

        with torch.no_grad():
            probs = F.softmax(_model(fv_t), dim=1).cpu().numpy()[0]

        top_idx   = probs.argsort()[::-1][:top_k]
        top_preds = [(_le.inverse_transform([i])[0], round(float(probs[i]), 4)) for i in top_idx]

        adavu_class = top_preds[0][0]
        confidence  = top_preds[0][1]

    # --- Step 3: DTW Alignment --------------------------------------------
    from .angles import get_angle_refs
    from .dtw import compute_dtw_alignment, compute_frame_anomaly_scores
    
    angle_refs_dict = get_angle_refs()
    ref_data = angle_refs_dict.get(adavu_class, {})
    
    ref_mean = ref_data.get("mean", angles.mean(axis=0))
    ref_std  = ref_data.get("std", np.maximum(angles.std(axis=0), 3.0))
    
    master_angles = None
    if "high_res_master_angles" in ref_data:
        master_angles = ref_data["high_res_master_angles"]
    else:
        vid_filename = CLASS_TO_FILE.get(adavu_class, adavu_class)
        prod_vid_path = os.path.abspath(f"checkpoints/master_videos/{vid_filename}.mp4")
        local_vid_path = os.path.abspath(f"/Volumes/Munu/Master Videos/{vid_filename}.mp4")
        master_vid_path = prod_vid_path if os.path.exists(prod_vid_path) else local_vid_path
        
        if os.path.exists(master_vid_path):
            m_seq = extract_landmarks_from_video(master_vid_path, num_frames=len(angles))
            if m_seq is not None:
                from .pose import normalise_landmarks
                m_seq_norm = normalise_landmarks(m_seq)
                master_angles = compute_angles_from_sequence(m_seq_norm)
                ref_data["high_res_master_angles"] = master_angles
                
    if master_angles is None:
        master_angles = ref_data.get("master_angles", np.tile(ref_mean, (len(angles), 1)))

    master_indices = compute_dtw_alignment(angles, master_angles)
    anomaly_scores = compute_frame_anomaly_scores(angles, master_angles, master_indices)

    # --- Step 4: Angular analysis & Scoring -------------------------------
    # Use only the active part of the video (ignore padding at start/end) for the active segment
    active_indices = np.where((master_indices > 0) & (master_indices < len(master_angles) - 1))[0]
    
    from .dtw import detect_dynamic_mistakes
    flagged, region_scores, overall_score = detect_dynamic_mistakes(
        angles, master_angles, master_indices, active_indices
    )
    
    from .scoring import compute_score
    score_result = compute_score(region_scores, adavu_class)
    # Override overall_score in score_result to use our DTW-based overall score
    score_result["overall"] = overall_score

    # --- Step 5: LLM feedback ---------------------------------------------
    feedback_text, feedback_source = get_llm_feedback(
        adavu_class, score_result, flagged, confidence
    )

    # --- Step 6: Top 5 Anomalies & Overlay --------------------------------
    top_5_anomalies = []
    overlay_b64 = None
    if return_overlay:
        try:
            from .overlay import draw_skeleton_overlay, draw_reference_skeleton, overlay_to_base64
            from .dtw import select_top_k_anomalies, build_joint_comparison_table
            from .pose import extract_mid_frame_rgb, extract_frames_rgb
            
            flagged_names = {j["joint"] for j in flagged}
            
            # 1. Select anomalies based on scores computed in Step 3
            top_5_idx = select_top_k_anomalies(anomaly_scores, k=5, min_distance=max(1, len(seq) // 10))
            
            # 3. Map landmark frame indices to video frame indices
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps              = cap.get(cv2.CAP_PROP_FPS) or 30.0
            cap.release()
            
            num_frames_seq = len(seq)
            if num_frames_seq > 0 and total_vid_frames > 0:
                # Use exactly the same mapping as pose.py's extract_landmarks_from_video
                s_indices = np.linspace(0, total_vid_frames - 1, num_frames_seq, dtype=int)
                vid_indices_map = {s_idx: s_indices[min(s_idx, num_frames_seq - 1)] for s_idx in top_5_idx}
                mid_s_idx = num_frames_seq // 2
                mid_vid_idx = s_indices[min(mid_s_idx, num_frames_seq - 1)]
                vid_indices_map[mid_s_idx] = mid_vid_idx
            else:
                vid_indices_map = {}
                mid_s_idx = 0
                mid_vid_idx = 0
            
            # Extract RGB frames from video in a single pass
            extracted_rgb = extract_frames_rgb(video_path, list(vid_indices_map.values()))
            
            # 4. Generate overall mid-frame overlay (backwards compat)
            if mid_vid_idx in extracted_rgb:
                frame_rgb = extracted_rgb[mid_vid_idx]
                canvas = draw_skeleton_overlay(
                    frame_rgb, seq[mid_s_idx], flagged_names, adavu_label=adavu_class
                )
                overlay_b64 = overlay_to_base64(canvas)
            else:
                aspect_ratio = 1.0

            # 5. Build Top 5 Anomaly objects
            for s_idx in top_5_idx:
                m_idx     = master_indices[s_idx]
                score_val = float(anomaly_scores[s_idx])
                vid_idx   = vid_indices_map[s_idx]
                ts_sec    = round(vid_idx / fps, 2)
                
                # Comparison table
                comp_table = build_joint_comparison_table(angles[s_idx], master_angles[m_idx], ref_std)
                
                # Student image with skeleton overlay
                s_img_b64 = None
                if vid_idx in extracted_rgb:
                    f_rgb = extracted_rgb[vid_idx]
                    o_H, o_W = f_rgb.shape[:2]
                    ar = o_W / o_H if o_H > 0 else 1.0
                    # Identify flagged joints specifically in this frame
                    frame_flagged = {row["joint"].lower(): row["left_flagged"] or row["right_flagged"] for row in comp_table}
                    flag_names_frame = {f"left_{j}" for j, flg in frame_flagged.items() if flg} | {f"right_{j}" for j, flg in frame_flagged.items() if flg}
                    s_canvas = draw_skeleton_overlay(f_rgb, seq[s_idx], flag_names_frame, ar, adavu_label=f"Frame #{vid_idx}")
                    s_img_b64 = overlay_to_base64(s_canvas)
                    
                # Master image reference skeleton
                m_img_b64 = None
                if m_idx < len(master_angles):
                    vid_filename = CLASS_TO_FILE.get(adavu_class, adavu_class)
                    prod_vid_path = os.path.abspath(f"checkpoints/master_videos/{vid_filename}.mp4")
                    local_vid_path = os.path.abspath(f"/Volumes/Munu/Master Videos/{vid_filename}.mp4")
                    master_vid_path = prod_vid_path if os.path.exists(prod_vid_path) else local_vid_path
                    m_frame_rgb = None
                    actual_m_idx = 0
                    if os.path.exists(master_vid_path):
                        from .pose import extract_mid_frame_rgb
                        import cv2
                        
                        cap = cv2.VideoCapture(master_vid_path)
                        m_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        
                        if m_total > 0:
                            num_master_frames = len(master_angles)
                            m_indices = np.linspace(0, m_total - 1, num_master_frames, dtype=int)
                            safe_m_idx = min(int(m_idx), num_master_frames - 1)
                            actual_m_idx = m_indices[safe_m_idx]
                            m_frame_rgb, _ = extract_mid_frame_rgb(master_vid_path, int(actual_m_idx))

                    if m_frame_rgb is not None:
                        from .pose import get_pose_landmarker
                        
                        landmarker = get_pose_landmarker()
                        result = landmarker.process(m_frame_rgb)
                        
                        if result.pose_landmarks:
                            live_m_lm = np.array([[l.x, l.y, l.visibility] for l in result.pose_landmarks.landmark])
                            m_canvas = draw_skeleton_overlay(
                                m_frame_rgb, live_m_lm, set(), adavu_label=f"Frame #{int(actual_m_idx)}"
                            )
                        else:
                            m_canvas = draw_reference_skeleton(seq[s_idx], adavu_label=f"Frame #{int(actual_m_idx)}")
                            
                        m_img_b64 = overlay_to_base64(m_canvas)
                    else:
                        m_canvas = draw_reference_skeleton(seq[s_idx], adavu_label=f"Frame #{int(actual_m_idx)}")
                        m_img_b64 = overlay_to_base64(m_canvas)
                else:
                    m_canvas = draw_reference_skeleton(seq[s_idx], adavu_label=f"Frame #{int(m_idx)}")
                    m_img_b64 = overlay_to_base64(m_canvas)
                    
                top_5_anomalies.append({
                    "frame_index":       int(s_idx),
                    "video_frame":       int(vid_idx),
                    "timestamp":         ts_sec,
                    "anomaly_score":     score_val,
                    "is_major_breach":   bool(score_val > 75.0),
                    "student_image_b64": s_img_b64,
                    "master_image_b64":  m_img_b64,
                    "comparison_table":  comp_table,
                })
        except Exception as e:
            print(f"  DTW Overlay generation failed: {e}")
            import traceback
            traceback.print_exc()

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
        "top_5_anomalies":   top_5_anomalies,
    }


def run_hasta_coach(image_path: str, target_class: str) -> dict:
    import inference.hands as hands_module
    
    seq_norm, rgb_frame = hands_module.extract_hand_from_image(image_path)
    if seq_norm is None:
        return {"error": "Could not detect a hand in the image. Please try another photo."}
        
    fv = seq_norm[:, :2].flatten()
    
    score = 100.0
    passed = True
    
    ref = getattr(hands_module, '_hasta_refs', {}).get(target_class)
    if ref is not None:
        diff = np.abs(fv - ref["mean"])
        err = np.mean(diff)
        score = max(0, 100 - err * 300) # Arbitrary scaling for hand diffs
        passed = score > 70
        
    # Draw hand overlay (simple circles for now)
    import cv2
    import base64
    canvas = rgb_frame.copy() if rgb_frame is not None else np.zeros((500,500,3), dtype=np.uint8)
    h, w = canvas.shape[:2]
    # draw seq_norm (which is normalized, so we need to denormalize for drawing if we had the scale, 
    # but we just draw it roughly or just return the original frame)
    
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "adavu_class": target_class,
        "confidence": 1.0,
        "top_k_predictions": [(target_class, 1.0)],
        "overall_score": round(score, 1),
        "region_scores": {"arms": round(score, 1)},
        "passed": passed,
        "grade": "A" if passed else "C",
        "grade_message": "Good Hasta" if passed else "Needs improvement",
        "pass_threshold": 70,
        "needed_to_pass": 0 if passed else 70 - score,
        "flagged_joints": [],
        "coaching_feedback": "Hasta analysis complete. Try to match the master reference exactly.",
        "feedback_source": "template",
        "overlay_image_b64": overlay_b64,
        "top_5_anomalies": [{
            "frame_index": 0, "video_frame": 0, "timestamp": 0,
            "anomaly_score": round(100 - score, 1),
            "is_major_breach": not passed,
            "student_image_b64": overlay_b64,
            "master_image_b64": None,
            "comparison_table": []
        }],
    }

from .postures import run_posture_coach
__all__ = ["load_model_and_refs", "run_coach_v2", "run_posture_coach", "run_hasta_coach"]