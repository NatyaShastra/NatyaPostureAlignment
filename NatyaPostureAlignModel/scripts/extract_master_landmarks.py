"""
scripts/extract_master_landmarks.py

Extracts raw unnormalized [0, 1] MediaPipe Pose landmarks for every physical JPEG frame in master_frames_output/.
Saves the resulting dictionary to checkpoints/master_landmarks_unnorm.npz.

Lookup structure:
master_landmarks_unnorm[master_video_name] -> numpy array of shape (total_frames, 33, 3)
where master_landmarks_unnorm[master_video_name][physical_frame_index] gives the (33, 3) raw [x, y, vis] landmarks.
"""

import os
import sys
import glob
import cv2
import numpy as np

# Add repo root to sys.path to reuse inference/pose.py utilities
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from inference.pose import get_pose_landmarker, pad_to_square, NUM_LANDMARKS

MASTER_FRAMES_DIR = os.path.join(REPO_ROOT, "master_frames_output")
CHECKPOINT_OUT = os.path.join(REPO_ROOT, "checkpoints", "master_landmarks_unnorm.npz")

def extract_all_master_landmarks() -> None:
    if not os.path.exists(MASTER_FRAMES_DIR):
        print(f"ERROR: master_frames_output directory not found at {MASTER_FRAMES_DIR}")
        sys.exit(1)

    folders = sorted([
        f for f in os.listdir(MASTER_FRAMES_DIR)
        if os.path.isdir(os.path.join(MASTER_FRAMES_DIR, f))
    ])

    print(f"--- Master Landmark Offline Extraction ---")
    print(f"Source Directory: {MASTER_FRAMES_DIR}")
    print(f"Found {len(folders)} master video folders\n")

    landmarker = get_pose_landmarker()
    result_dict = {}

    total_physical_frames_processed = 0
    total_missing_detections = 0

    for folder in folders:
        folder_path = os.path.join(MASTER_FRAMES_DIR, folder)
        image_paths = sorted(glob.glob(os.path.join(folder_path, "frame_*.jpg")))

        if not image_paths:
            print(f"⚠️  No frame JPEGs found in {folder}")
            continue

        frame_landmarks_list = []
        missing_count = 0

        for idx, img_path in enumerate(image_paths):
            frame_bgr = cv2.imread(img_path)
            if frame_bgr is None:
                print(f"⚠️ Could not read image: {img_path}")
                lm_arr = frame_landmarks_list[-1] if frame_landmarks_list else np.zeros((NUM_LANDMARKS, 3))
                frame_landmarks_list.append(lm_arr)
                missing_count += 1
                continue

            frame_square = pad_to_square(frame_bgr)
            rgb = cv2.cvtColor(frame_square, cv2.COLOR_BGR2RGB)

            res = landmarker.process(rgb)

            if res.pose_landmarks:
                lm_arr = np.array([[l.x, l.y, l.visibility] for l in res.pose_landmarks.landmark])
            else:
                missing_count += 1
                # Fallback to previous frame if available, else zeros
                lm_arr = frame_landmarks_list[-1] if frame_landmarks_list else np.zeros((NUM_LANDMARKS, 3))

            frame_landmarks_list.append(lm_arr)

        folder_arr = np.array(frame_landmarks_list)  # (total_frames, 33, 3)
        result_dict[folder] = folder_arr

        num_frames = len(image_paths)
        total_physical_frames_processed += num_frames
        total_missing_detections += missing_count

        print(f"  [OK] {folder:<16s}: {num_frames:4d} frames processed (shape: {folder_arr.shape}), fallback detections: {missing_count}")

    print("\n--- Summary ---")
    print(f"Master Video Folders Processed: {len(result_dict)}")
    print(f"Total Physical Frames Extracted: {total_physical_frames_processed}")
    print(f"Total Fallback Detections: {total_missing_detections}")

    # Save dictionary to npz file
    os.makedirs(os.path.dirname(CHECKPOINT_OUT), exist_ok=True)
    np.savez_compressed(CHECKPOINT_OUT, **result_dict)
    print(f"Saved unnormalized master landmarks to: {CHECKPOINT_OUT}")

if __name__ == "__main__":
    extract_all_master_landmarks()
