import os
import cv2
import numpy as np
import mediapipe as mp

_hand_landmarker = None

def get_hand_landmarker():
    global _hand_landmarker
    if _hand_landmarker is None:
        _hand_landmarker = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
    return _hand_landmarker

def pad_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w: return image
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    return cv2.copyMakeBorder(image, pad_h, size - h - pad_h, pad_w, size - w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def normalise_hand_landmarks(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    wrist = arr[0, :2]
    middle_mcp = arr[9, :2]
    scale = np.linalg.norm(middle_mcp - wrist)
    scale = max(scale, 1e-6)
    arr[:, :2] = (arr[:, :2] - wrist) / scale
    return arr

def extract_hand_from_image(image_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    landmarker = get_hand_landmarker()
    frame = cv2.imread(image_path)
    if frame is None:
        return None, None
        
    frame = pad_to_square(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = landmarker.process(rgb)
    
    if not result.multi_hand_landmarks:
        return None, frame
        
    lm = result.multi_hand_landmarks[0].landmark
    arr = np.array([[l.x, l.y, l.z] for l in lm])
    return normalise_hand_landmarks(arr), frame

def build_hasta_refs(features: np.ndarray, labels: np.ndarray) -> dict:
    refs = {}
    unique_labels = sorted(set(labels))
    for cls in unique_labels:
        idx = (labels == cls)
        cls_features = features[idx]
        mean_feat = cls_features.mean(axis=0)
        std_feat = cls_features.std(axis=0)
        
        mean_lm = mean_feat.reshape(21, 2)
        master_lm = np.zeros((21, 3))
        master_lm[:, :2] = mean_lm
        
        refs[cls] = {
            "mean": mean_feat,
            "std": std_feat,
            "master_landmarks": master_lm
        }
    return refs
