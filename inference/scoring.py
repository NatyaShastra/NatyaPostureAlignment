"""
inference/scoring.py — Weighted scoring, pass/fail gate, and grade bands.

Region weights:
  legs  50%  (footwork is the primary criterion in Bharatanatyam)
  arms  30%
  torso 20%
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_PASS_THRESHOLD: float = 70.0

# Per-adavu overrides — complex / physically demanding adavus get a lower bar
ADAVU_PASS_THRESHOLDS: dict[str, float] = {
    "MandiAdavu":               65.0,
    "KudithuMettuAdavu":        65.0,
    "ParavalAdavu":             65.0,
    "ThahathaJamTharithaAdavu": 60.0,
    "Nattadavu":                72.0,
    "Thattadavu":               72.0,
}

REGION_WEIGHTS: dict[str, float] = {
    "legs":   0.50,
    "arms":   0.30,
    "torso":  0.20,
}

# (min_score, grade_label, message)
GRADE_BANDS: list[tuple[float, str, str]] = [
    (90.0, "A", "Excellent — your form is very close to reference. Maintain this quality."),
    (75.0, "B", "Good — strong execution overall. Focus on the flagged joints to advance."),
    (DEFAULT_PASS_THRESHOLD, "C", "Passing — you've cleared the threshold but there is clear room to improve."),
    (55.0, "D", "Needs work — more practice before progressing. See corrections below."),
    (0.0,  "F", "Retry — focus on the fundamentals and revisit this adavu."),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(
    region_scores: dict[str, float],
    adavu_class: str | None = None,
) -> dict:
    """
    Compute the weighted overall score and pass/fail result.

    Args:
        region_scores: {"legs": float, "arms": float, "torso": float}  (0–100)
        adavu_class:   optional — used to look up per-adavu threshold

    Returns dict with:
        overall_score, region_scores, passed, grade, grade_message,
        pass_threshold, needed_to_pass
    """
    overall = sum(
        region_scores.get(r, 100.0) * w
        for r, w in REGION_WEIGHTS.items()
    )
    overall = round(overall, 1)

    threshold = (
        ADAVU_PASS_THRESHOLDS.get(adavu_class, DEFAULT_PASS_THRESHOLD)
        if adavu_class else DEFAULT_PASS_THRESHOLD
    )
    passed     = overall >= threshold
    needed     = max(0.0, round(threshold - overall, 1))

    # Find grade
    grade, grade_message = GRADE_BANDS[-1][1], GRADE_BANDS[-1][2]
    for min_score, g, msg in GRADE_BANDS:
        if overall >= min_score:
            grade, grade_message = g, msg
            break

    return {
        "overall_score":  overall,
        "region_scores":  region_scores,
        "passed":         passed,
        "grade":          grade,
        "grade_message":  grade_message,
        "pass_threshold": threshold,
        "needed_to_pass": needed,
    }