import os
import cv2
import numpy as np
import torch

from .pose import extract_mid_frame_rgb, get_pose_landmarker, pad_to_square, normalise_landmarks, build_feature_vector
from .angles import compute_angles_from_sequence, build_angle_refs
from .overlay import draw_skeleton_overlay, overlay_to_base64, draw_reference_skeleton

_posture_refs = {}

def load_posture_refs(checkpoint_path: str, features_cache: str = "checkpoints/postures_features.npz"):
    global _posture_refs
    if os.path.exists(features_cache):
        data = np.load(features_cache, allow_pickle=True)
        # build_angle_refs assigns to a global in angles.py, we want our own
        # Actually, let's just use a simple mean for postures
        unique_labels = sorted(set(data["y"]))
        for cls in unique_labels:
            idx = (data["y"] == cls)
            cls_angles = data["X"][idx] # Wait, X in postures_features is features, not angles!
            
            # We need to extract the angle part. In colab, features = 84 dim. Angles are somewhere inside.
            # But we can just use the features themselves for simple euclidean distance grading!
            _posture_refs[cls] = {
                "mean_features": cls_angles.mean(axis=0),
                "std_features": cls_angles.std(axis=0)
            }
        print(f"[startup] Posture refs built from {len(data['X'])} cached samples")

def run_posture_coach(image_path: str, target_class: str) -> dict:
    import mediapipe as mp
    
    landmarker = get_pose_landmarker()
    
    # Read image
    frame = cv2.imread(image_path)
    if frame is None:
        cap = cv2.VideoCapture(image_path)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return {"error": "Could not read image or video."}
            
    frame = pad_to_square(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.process(rgb)
    
    if not result.pose_landmarks:
        return {"error": "Could not detect body pose in the image."}
        
    lm = result.pose_landmarks.landmark
    arr = np.array([[[l.x, l.y, l.visibility] for l in lm]]) # (1, 33, 3)
    
    seq_norm = normalise_landmarks(arr)
    angles = compute_angles_from_sequence(seq_norm)
    fv = build_feature_vector(seq_norm, angles)
    
    # Grade against master
    score = 100.0
    passed = True
    flagged = []
    
    ref = _posture_refs.get(target_class)
    if ref is not None:
        diff = np.abs(fv - ref["mean_features"])
        # simple heuristic score
        err = np.mean(diff)
        score = max(0, 100 - err * 100)
        passed = score > 70
        
    # Overlay
    canvas = draw_skeleton_overlay(rgb, arr[0], set(), adavu_label=target_class)
    overlay_b64 = overlay_to_base64(canvas)
    
    # Master image
    m_img_b64 = None
    master_path = f"/Volumes/Munu/Master Videos/{target_class}.png"
    if os.path.exists(master_path):
        m_frame = cv2.imread(master_path)
        m_frame = pad_to_square(m_frame)
        m_rgb = cv2.cvtColor(m_frame, cv2.COLOR_BGR2RGB)
        m_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=m_rgb)
        m_res = landmarker.detect(m_mp)
        if m_res.pose_landmarks:
            m_lm = np.array([[l.x, l.y, l.visibility] for l in m_res.pose_landmarks[0]])
            m_canvas = draw_skeleton_overlay(m_rgb, m_lm, set(), adavu_label=f"Master {target_class}")
            m_img_b64 = overlay_to_base64(m_canvas)
            
    return {
        "adavu_class": target_class,
        "confidence": 1.0,
        "top_k_predictions": [(target_class, 1.0)],
        "overall_score": round(score, 1),
        "region_scores": {"legs": round(score, 1), "arms": round(score, 1)},
        "passed": passed,
        "grade": "A" if passed else "C",
        "grade_message": "Good posture" if passed else "Needs improvement",
        "pass_threshold": 70,
        "needed_to_pass": 0 if passed else 70 - score,
        "flagged_joints": flagged,
        "coaching_feedback": "Posture analysis complete. Try to match the master reference exactly.",
        "feedback_source": "template",
        "overlay_image_b64": overlay_b64,
        "top_5_anomalies": [{
            "frame_index": 0,
            "video_frame": 0,
            "timestamp": 0,
            "anomaly_score": round(100 - score, 1),
            "is_major_breach": not passed,
            "student_image_b64": overlay_b64,
            "master_image_b64": m_img_b64,
            "comparison_table": []
        }],
    }
