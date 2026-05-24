"""
inference/angles.py — Joint angle computation and mistake detection.

Reference angle distributions are derived from the training set
feature cache (adavu_features.npz) at startup and stored in module
state via build_angle_refs(). Must be called once before detect_mistakes().
"""

from __future__ import annotations
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# (name, joint_a, vertex, joint_c) — all MediaPipe landmark indices
ANGLE_DEFS: list[tuple[str, int, int, int]] = [
    ("left_knee",       23, 25, 27),   # left hip → left knee → left ankle
    ("right_knee",      24, 26, 28),   # right hip → right knee → right ankle
    ("left_hip",        11, 23, 25),   # left shoulder → left hip → left knee
    ("right_hip",       12, 24, 26),   # right shoulder → right hip → right knee
    ("left_elbow",      11, 13, 15),   # left shoulder → left elbow → left wrist
    ("right_elbow",     12, 14, 16),   # right shoulder → right elbow → right wrist
    ("left_shoulder",   13, 11, 23),   # left elbow → left shoulder → left hip
    ("right_shoulder",  14, 12, 24),   # right elbow → right shoulder → right hip
    ("spine_lean",      23, 11, 24),   # left hip → left shoulder → right shoulder
]

ANGLE_NAMES: list[str] = [d[0] for d in ANGLE_DEFS]
NUM_ANGLES:  int        = len(ANGLE_DEFS)

# Body region → angle names mapping (for weighted scoring)
REGIONS: dict[str, list[str]] = {
    "legs":  ["left_knee", "right_knee", "left_hip", "right_hip"],
    "arms":  ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder"],
    "torso": ["spine_lean"],
}

# Deviations larger than this (in σ) are flagged
THRESHOLD_SIGMA: float = 1.5

# Per-class reference distributions — populated by build_angle_refs()
_angle_refs: dict[str, dict[str, np.ndarray]] = {}


# ---------------------------------------------------------------------------
# Angle computation
# ---------------------------------------------------------------------------

def _angle_between(pa: np.ndarray, pv: np.ndarray, pc: np.ndarray) -> float:
    """Compute the angle at vertex pv, in degrees."""
    v1 = pa - pv
    v2 = pc - pv
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def compute_angles_from_sequence(seq: np.ndarray) -> np.ndarray:
    """
    Compute joint angles for every frame in a landmark sequence.

    Args:
        seq: (T, 33, 3) array of landmarks

    Returns:
        angles: (T, NUM_ANGLES) array of angles in degrees
    """
    T = seq.shape[0]
    angles = np.zeros((T, NUM_ANGLES))
    for t in range(T):
        frame = seq[t]
        for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
            angles[t, i] = _angle_between(frame[a, :2], frame[v, :2], frame[c, :2])
    return angles


def _angles_from_mean_pose(mean_pose: np.ndarray) -> np.ndarray:
    """
    Compute angles from a single (33, 3) representative pose.
    Used when building reference distributions from cached features.
    """
    angles = np.zeros(NUM_ANGLES)
    for i, (_, a, v, c) in enumerate(ANGLE_DEFS):
        angles[i] = _angle_between(mean_pose[a, :2], mean_pose[v, :2], mean_pose[c, :2])
    return angles


# ---------------------------------------------------------------------------
# Reference distribution builder (called once at startup)
# ---------------------------------------------------------------------------

def build_angle_refs(X: np.ndarray, y: np.ndarray) -> None:
    """
    Build per-class angle reference distributions from the training cache.

    The cache stores 297-dim feature vectors where the first 99 dims are
    flattened mean landmarks (33 joints × 3 coords). We reshape these back
    to (33, 3) and treat each as a representative pose for that sample.

    Args:
        X: (N, 297) feature matrix from adavu_features.npz
        y: (N,) array of string class labels
    """
    global _angle_refs
    _angle_refs = {}

    for cls in np.unique(y):
        mask = y == cls
        mean_poses = X[mask, :99].reshape(-1, 33, 3)   # (N_cls, 33, 3)

        cls_angles = np.array([_angles_from_mean_pose(p) for p in mean_poses])  # (N_cls, NUM_ANGLES)

        _angle_refs[cls] = {
            "mean": cls_angles.mean(axis=0),
            # Floor std at 3° to avoid division-by-zero on very consistent classes
            "std":  np.maximum(cls_angles.std(axis=0), 3.0),
        }


def get_angle_refs() -> dict:
    """Return the current reference distributions (for serialisation / inspection)."""
    return _angle_refs


# ---------------------------------------------------------------------------
# Mistake detection
# ---------------------------------------------------------------------------

def detect_mistakes(
    student_angles_mean: np.ndarray,
    adavu_class: str,
) -> tuple[list[dict], dict[str, float], float]:
    """
    Compare the student's mean angles against the reference distribution.

    Args:
        student_angles_mean: (NUM_ANGLES,) mean angle per joint from the student video
        adavu_class:         predicted adavu class name

    Returns:
        flagged_joints: list of dicts with joint, measured, reference, deviation fields
        region_scores:  {region_name: 0–100} — % of joints within threshold per region
        overall_score:  float 0–100
    """
    if adavu_class not in _angle_refs:
        # No reference available — return perfect score
        return [], {r: 100.0 for r in REGIONS}, 100.0

    ref_mean = _angle_refs[adavu_class]["mean"]
    ref_std  = _angle_refs[adavu_class]["std"]

    deviations_sigma = np.abs(student_angles_mean - ref_mean) / ref_std

    flagged_joints: list[dict] = []
    for i, (name, *_) in enumerate(ANGLE_DEFS):
        if deviations_sigma[i] > THRESHOLD_SIGMA:
            flagged_joints.append({
                "joint":           name,
                "measured":        round(float(student_angles_mean[i]), 1),
                "reference":       round(float(ref_mean[i]), 1),
                "deviation":       round(float(deviations_sigma[i]), 2),
                "deviation_deg":   round(float(abs(student_angles_mean[i] - ref_mean[i])), 1),
            })

    # Per-region score: fraction of joints within threshold × 100
    region_scores: dict[str, float] = {}
    for region, joint_names in REGIONS.items():
        idxs = [ANGLE_NAMES.index(j) for j in joint_names if j in ANGLE_NAMES]
        within = float(np.sum(deviations_sigma[idxs] <= THRESHOLD_SIGMA))
        region_scores[region] = round(within / len(idxs) * 100, 1)

    overall_score = round(float(np.mean(deviations_sigma <= THRESHOLD_SIGMA)) * 100, 1)

    return flagged_joints, region_scores, overall_score