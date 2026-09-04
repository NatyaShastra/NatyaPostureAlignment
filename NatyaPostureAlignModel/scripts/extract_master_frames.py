"""
scripts/extract_master_frames.py

Extracts frame JPEGs from master MP4 videos for Bunny.net CDN hosting.

Verified Indexing Rule:
In inference/__init__.py:
    m_indices = np.linspace(0, m_total - 1, num_master_frames, dtype=int)
    actual_m_idx = m_indices[safe_m_idx]
OpenCV reads video frame-by-frame with a zero-based counter (`frame_counter = 0`).
Therefore, OpenCV frame 0 maps directly to frame_000.jpg.
`actual_m_idx` (e.g. 250) corresponds exactly to `frame_250.jpg`.
"""

import os
import cv2
import sys

MASTER_VIDEOS_DIR = os.path.abspath("checkpoints/master_videos")
OUTPUT_DIR        = os.path.abspath("master_frames_output")

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

def extract_all_master_frames(dry_run: bool = True) -> None:
    if not os.path.exists(MASTER_VIDEOS_DIR):
        print(f"ERROR: Master videos directory not found at {MASTER_VIDEOS_DIR}")
        sys.exit(1)

    print(f"--- Master Frame Extraction Plan ---")
    print(f"Source Directory: {MASTER_VIDEOS_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Mode: {'DRY RUN (Preview Only)' if dry_run else 'EXECUTE'}\n")

    summary = []

    for class_name, file_stem in CLASS_TO_FILE.items():
        mp4_path = os.path.join(MASTER_VIDEOS_DIR, f"{file_stem}.mp4")
        if not os.path.exists(mp4_path):
            print(f"⚠️  MISSING: {file_stem}.mp4 for class '{class_name}'")
            continue

        cap = cv2.VideoCapture(mp4_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        out_folder = os.path.join(OUTPUT_DIR, file_stem)
        summary.append({
            "class_name": class_name,
            "file_stem": file_stem,
            "total_frames": total_frames,
            "fps": fps,
            "sample_first_file": f"{file_stem}/frame_000.jpg",
            "sample_last_file": f"{file_stem}/frame_{total_frames - 1:03d}.jpg",
            "output_path": out_folder
        })

    print(f"{'Adavu Class':<15} | {'File Stem':<15} | {'Frames':<8} | {'FPS':<5} | {'Sample First File':<22} | {'Sample Last File':<22}")
    print("-" * 100)
    for s in summary:
        print(f"{s['class_name']:<15} | {s['file_stem']:<15} | {s['total_frames']:<8} | {s['fps']:<5.1f} | {s['sample_first_file']:<22} | {s['sample_last_file']:<22}")

    print("\n--- Folder Structure Preview for Bunny.net ---")
    print(f"{OUTPUT_DIR}/")
    for s in summary[:2]:
        print(f"|-- {s['file_stem']}/")
        print(f"|   |-- frame_000.jpg")
        print(f"|   |-- frame_001.jpg")
        print(f"|   +-- ...")
        print(f"|   +-- {os.path.basename(s['sample_last_file'])}")
    print("+-- ... (16 Adavu master video folders in total)\n")

    if not dry_run:
        print("Extracting frames to disk...")
        for s in summary:
            mp4_path = os.path.join(MASTER_VIDEOS_DIR, f"{s['file_stem']}.mp4")
            out_folder = s["output_path"]
            os.makedirs(out_folder, exist_ok=True)
            cap = cv2.VideoCapture(mp4_path)
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out_file = os.path.join(out_folder, f"frame_{frame_idx:03d}.jpg")
                cv2.imwrite(out_file, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                frame_idx += 1
            cap.release()
            print(f"[OK] Extracted {frame_idx} frames for {s['file_stem']}")

if __name__ == "__main__":
    dry_run = "--execute" not in sys.argv
    extract_all_master_frames(dry_run=dry_run)
