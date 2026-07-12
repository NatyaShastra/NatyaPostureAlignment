"""
data/ingest.py — One-time dataset ingestion pipeline.

Downloads videos from HuggingFace one at a time (no bulk 12 GB download),
extracts MediaPipe pose features, and saves the result to adavu_features.npz.

This script runs ONCE to build the feature cache used by inference.
It does NOT run at inference time.

Usage:
    python -m data.ingest --output checkpoints/adavu_features.npz
    python -m data.ingest --output checkpoints/adavu_features.npz --max-per-class 3  # quick test
"""

from __future__ import annotations
import argparse
import os
import re
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

# Add parent directory to path so we can import inference modules
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pose import (
    get_pose_landmarker,
    extract_landmarks_from_video,
    build_feature_vector,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REPO_ID      = "vibhuti16/bharatnatyam_adavus"
NUM_FRAMES   = 30
MIN_VIDEOS   = 4     # drop classes with fewer than this many samples

# Canonical name mapping — normalises messy folder names in the HF dataset
CLASS_ALIASES: dict[str, str] = {
    "thattimettuadavu":               "ThattimettuAdavu",
    "thatimettuadavu":                "ThattimettuAdavu",
    "kudhittamettuadavu":             "KudithuMettuAdavu",
    "kudhittamettu":                  "KudithuMettuAdavu",
    "kudithumettuadavu":              "KudithuMettuAdavu",
    "kudhitthamettuadavu":            "KudithuMettuAdavu",
    "korvaiadavu":                    "KorvaiAdavu",
    "karthariadavu":                  "KarthariAdavu",
    "thahathajamtharithaadavu":       "ThaHathaJhamTharithamAdavu",
    "thahathajhamtharithamadavu":     "ThaHathaJhamTharithamAdavu",
    "thahathajanjtharithaadavu":      "ThaHathaJhamTharithamAdavu",
    "thaithaithathamadavu":           "ThaiThaiThaThamAdavu",
    "thaithaithatham":                "ThaiThaiThaThamAdavu",
    "mandiadavu":                     "MandiAdavu",
    "sarukkaladavu":                  "SarukkalAdavu",
    "thathaithaha":                   "ThathaiThaha",
    "thathaithahaadavu":              "ThathaiThaha",
    "thatheitheiha":                  "ThaTheiTheiTha",
    "thaiyathaihiadavu":              "ThaiyaThaihi",
    "thaiyathaihi":                   "ThaiyaThaihi",
    "theermaanamadavu":               "TheermanaAdavu",
    "theermanaadavu":                 "TheermanaAdavu",
    "theermanaadavulearnandpractice": "TheermanaAdavu",
    "theermanaadavulearn":            "TheermanaAdavu",
    "theerumanamadavu":               "TheermanaAdavu",
    "uthsangaadavu":                  "UthsangaAdavu",
    "uthplavanaadavu":                "UtplavanadaAdavu",
}


def normalize_class(name: str) -> str:
    key = re.sub(r"[\s_\-]+", "", name).lower()
    key = re.sub(r"kalakshetrastyle$", "", key)
    key = re.sub(r"learn(andpractice)?$", "", key)
    return CLASS_ALIASES.get(key, name)


def get_class(path: str) -> str:
    filename = path.split("/")[-1].replace(".mp4", "")
    match = re.match(r"^([A-Za-z\s]+?)(\d.*)?$", filename)
    raw   = match.group(1).strip() if match else filename
    return normalize_class(raw)


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def build_dataset(
    output_path: str,
    max_per_class: int | None = None,
    mediapipe_model: str = "pose_landmarker_heavy.task",
    hf_cache_dir: str = "/tmp/hf_cache",
) -> None:
    """
    Stream all videos from HuggingFace, extract features, save .npz cache.

    Args:
        output_path:     Where to write adavu_features.npz
        max_per_class:   Cap videos per class (None = all). Set to 3 for a quick test.
        mediapipe_model: Path to pose_landmarker_heavy.task
        hf_cache_dir:    Temp dir for downloaded videos (each deleted after processing)
    """
    if os.path.exists(output_path):
        print(f"Cache already exists at {output_path}. Delete it to re-run.")
        return

    # Warm up MediaPipe
    get_pose_landmarker(mediapipe_model)

    # List all videos in the HF repo
    print(f"Listing videos in {REPO_ID}...")
    all_files   = list(list_repo_files(REPO_ID, repo_type="dataset"))
    video_files = [f for f in all_files if f.startswith("videos/") and f.endswith(".mp4")]
    print(f"Found {len(video_files)} videos")

    # Count and filter classes
    class_counts: dict[str, int] = {}
    for f in video_files:
        cls = get_class(f)
        class_counts[cls] = class_counts.get(cls, 0) + 1

    valid_classes = {cls for cls, cnt in class_counts.items() if cnt >= MIN_VIDEOS}
    filtered      = [f for f in video_files if get_class(f) in valid_classes]

    print(f"Keeping {len(valid_classes)} classes (≥{MIN_VIDEOS} videos each)")
    print(f"Keeping {len(filtered)} / {len(video_files)} videos\n")

    # Group by class
    class_files: dict[str, list[str]] = defaultdict(list)
    for f in filtered:
        class_files[get_class(f)].append(f)

    features, labels, failed = [], [], []

    for cls, files in class_files.items():
        subset = files[:max_per_class] if max_per_class else files
        print(f"Processing {cls} ({len(subset)} videos)")

        for hf_path in tqdm(subset, desc=cls, leave=False):
            tmp_path = None
            try:
                tmp_path = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=hf_path,
                    repo_type="dataset",
                    local_dir=hf_cache_dir,
                )
                lm = extract_landmarks_from_video(tmp_path, NUM_FRAMES)
                if lm is None or len(lm) == 0:
                    failed.append(hf_path)
                    continue
                features.append(build_feature_vector(lm))
                labels.append(cls)
            except Exception as e:
                failed.append(hf_path)
                print(f"  FAILED: {hf_path} — {e}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

    X = np.array(features)
    y = np.array(labels)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, X=X, y=y, label_names=sorted(set(y)))

    print(f"\nDone: {len(X)} samples, {len(set(y))} classes, {len(failed)} failed")
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build adavu feature cache from HuggingFace dataset")
    parser.add_argument("--output",        default="checkpoints/adavu_features.npz")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap videos per class (e.g. 3 for a quick test)")
    parser.add_argument("--mediapipe",     default="pose_landmarker_heavy.task",
                        help="Path to pose_landmarker_heavy.task")
    args = parser.parse_args()

    build_dataset(
        output_path=args.output,
        max_per_class=args.max_per_class,
        mediapipe_model=args.mediapipe,
    )


if __name__ == "__main__":
    main()