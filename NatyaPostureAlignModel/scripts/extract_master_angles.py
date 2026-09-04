"""
scripts/extract_master_angles.py — Offline precomputation of high-resolution master angle trajectories.

Extracts 120-frame normalized joint angle sequences for all 16 Adavu classes from local master videos,
and saves them to checkpoints/master_angles.npz.
"""

from __future__ import annotations
import os
import sys
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.pose import extract_landmarks_from_video, normalise_landmarks
from inference.angles import compute_angles_from_sequence

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

def main():
    print("[precompute] Extracting high-res master angle trajectories...")
    master_dict = {}
    
    for cls, fn in CLASS_TO_FILE.items():
        vid_path = os.path.abspath(f"checkpoints/master_videos/{fn}.mp4")
        if not os.path.exists(vid_path):
            print(f"  [WARNING] Missing video for {cls}: {vid_path}")
            continue
            
        print(f"  Extracting {cls} ({fn}.mp4)...")
        seq = extract_landmarks_from_video(vid_path, num_frames=120)
        if seq is None:
            print(f"  [ERROR] Failed to extract landmarks for {cls}")
            continue
            
        seq_norm = normalise_landmarks(seq)
        angles = compute_angles_from_sequence(seq_norm)
        
        # Verify finite values
        if not np.all(np.isfinite(angles)):
            print(f"  [WARNING] Non-finite values detected in {cls}")
            
        master_dict[cls] = angles
        print(f"    [OK] {cls}: shape {angles.shape}")

    out_path = os.path.abspath("checkpoints/master_angles.npz")
    np.savez_compressed(out_path, **master_dict)
    print(f"\n[precompute] Saved {len(master_dict)} classes to {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
