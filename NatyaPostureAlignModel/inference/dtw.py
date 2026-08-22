"""
inference/dtw.py — Dynamic Time Warping (DTW) and Temporal Anomaly Engine.

Provides temporal alignment between a student's variable-speed dance execution
and the stored Gold-Standard Master reference trajectory. Also performs
Non-Maximum Suppression (NMS) to select the Top 5 distinct anomaly frames.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


def get_active_segment(seq: np.ndarray, threshold_ratio: float = 0.15) -> tuple:
    """
    Identifies the start and end indices of the active movement in a sequence
    by thresholding the smoothed frame-to-frame angular velocity.
    """
    T = seq.shape[0]
    diffs = np.sum(np.abs(np.diff(seq, axis=0)), axis=1)
    diffs = np.concatenate(([0], diffs))
    
    window = min(10, max(1, T // 10))
    smoothed = np.convolve(diffs, np.ones(window)/window, mode='same')
    
    threshold = np.max(smoothed) * threshold_ratio
    active_idx = np.where(smoothed > threshold)[0]
    
    if len(active_idx) > 10:
        start = max(0, active_idx[0] - window)
        end = min(T - 1, active_idx[-1] + window)
    else:
        start = 0
        end = T - 1
        
    return start, end


def compute_dtw_alignment(student_seq: np.ndarray, master_seq: np.ndarray) -> np.ndarray:
    """
    Align student time series (T_s, D) to master reference time series (T_m, D)
    using Euclidean Dynamic Time Warping (DTW) with motion-based cropping.
    
    Cropping BOTH sequences based on angular velocity prevents 
    standing-still padding periods from pushing the videos out of phase.
    """
    T_s, D = student_seq.shape
    T_m, _ = master_seq.shape

    # 1. Motion-based cropping to isolate the active dancing segments
    start_s, end_s = get_active_segment(student_seq)
    start_m, end_m = get_active_segment(master_seq)
        
    active_student = student_seq[start_s:end_s+1]
    active_master  = master_seq[start_m:end_m+1]
    
    T_s_active = active_student.shape[0]
    T_m_active = active_master.shape[0]

    # 2. Standard DTW on the active segments
    dist_matrix = cdist(active_student, active_master, metric="euclidean")

    cost = np.full((T_s_active, T_m_active), np.inf)
    cost[0, 0] = dist_matrix[0, 0]

    for i in range(1, T_s_active):
        cost[i, 0] = cost[i - 1, 0] + dist_matrix[i, 0]
    for j in range(1, T_m_active):
        cost[0, j] = cost[0, j - 1] + dist_matrix[0, j]

    for i in range(1, T_s_active):
        for j in range(1, T_m_active):
            cost[i, j] = dist_matrix[i, j] + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])

    i, j = T_s_active - 1, T_m_active - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            options = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            best_idx = np.argmin(options)
            if best_idx == 0:
                i -= 1; j -= 1
            elif best_idx == 1:
                i -= 1
            else:
                j -= 1
        path.append((i, j))

    path.reverse()

    # Map each student frame in active segment to the active master frame
    active_master_indices = np.zeros(T_s_active, dtype=int)
    for idx_in_active in range(T_s_active):
        matches = [m_idx for (s, m_idx) in path if s == idx_in_active]
        if matches:
            active_master_indices[idx_in_active] = matches[len(matches) // 2]
        else:
            active_master_indices[idx_in_active] = active_master_indices[max(0, idx_in_active - 1)]

    # 3. Reconstruct full indices array for all frames
    master_indices = np.zeros(T_s, dtype=int)
    
    # Map student padding to master padding (or frame 0 / T_m-1)
    master_indices[:start_s] = 0
    master_indices[end_s+1:] = T_m - 1
    
    # Map the active segment back to the full master indices
    for idx_in_active, m_active_idx in enumerate(active_master_indices):
        master_indices[start_s + idx_in_active] = start_m + m_active_idx

    return master_indices


def compute_frame_anomaly_scores(
    student_angles: np.ndarray,
    master_angles: np.ndarray,
    master_indices: np.ndarray,
    ref_std: np.ndarray,
) -> np.ndarray:
    """
    Compute a 0.0 to 100.0 anomaly severity score for every student frame.

    A score over 75% indicates a severe form breakdown across key limbs.
    """
    T_s, D = student_angles.shape
    scores = np.zeros(T_s)

    for i in range(T_s):
        s_frame = student_angles[i]
        m_frame = master_angles[master_indices[i]]

        # Deviation relative to standard deviation (3 sigma = 100% anomaly for that joint)
        dev_ratios = np.abs(s_frame - m_frame) / (3.0 * np.maximum(ref_std, 3.0))
        dev_ratios = np.minimum(dev_ratios, 1.0)  # cap at 1.0 (100%)

        # Average severity across the 12 joints × 100
        scores[i] = round(float(np.mean(dev_ratios) * 100.0), 1)

    return scores


def select_top_k_anomalies(
    scores: np.ndarray,
    k: int = 5,
    min_distance: int = 12,
) -> list[int]:
    """
    Select up to k distinct frame indices with the highest anomaly scores
    using Non-Maximum Suppression (NMS) to ensure temporal diversity.
    """
    T_s = len(scores)
    sorted_indices = np.argsort(scores)[::-1]  # descending order of severity
    selected: list[int] = []

    for idx in sorted_indices:
        # Check if this frame is too close to an already selected frame
        if any(abs(idx - sel) < min_distance for sel in selected):
            continue
        selected.append(int(idx))
        if len(selected) == k:
            break

    # Sort selected indices chronologically for dashboard presentation
    selected.sort()
    return selected


def build_joint_comparison_table(
    student_frame: np.ndarray,
    master_frame: np.ndarray,
    ref_std: np.ndarray,
) -> list[dict]:
    """
    Build Left/Right joint comparison table for a specific aligned frame.
    """
    from .angles import ANGLE_NAMES
    joints = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
    table: list[dict] = []
    
    for j in joints:
        l_idx = ANGLE_NAMES.index(f"left_{j}")
        r_idx = ANGLE_NAMES.index(f"right_{j}")
        
        l_s = round(float(student_frame[l_idx]), 1)
        l_m = round(float(master_frame[l_idx]), 1)
        l_diff = round(float(abs(l_s - l_m)), 1)
        l_flag = (l_diff / np.maximum(ref_std[l_idx], 3.0)) > 1.5
        
        r_s = round(float(student_frame[r_idx]), 1)
        r_m = round(float(master_frame[r_idx]), 1)
        r_diff = round(float(abs(r_s - r_m)), 1)
        r_flag = (r_diff / np.maximum(ref_std[r_idx], 3.0)) > 1.5
        
        table.append({
            "joint": j.capitalize(),
            "left_student": l_s,
            "left_master": l_m,
            "left_diff": l_diff,
            "left_flagged": bool(l_flag),
            "right_student": r_s,
            "right_master": r_m,
            "right_diff": r_diff,
            "right_flagged": bool(r_flag),
        })
        
    return table
