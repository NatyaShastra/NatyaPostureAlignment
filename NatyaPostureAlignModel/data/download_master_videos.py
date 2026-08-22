import os
import torch
from huggingface_hub import hf_hub_download
import shutil

CKPT_PATH = 'checkpoints/dance_coach_model.pt'
REPO_ID = 'vibhuti16/bharatnatyam_adavus'
OUT_DIR = 'checkpoints/master_videos'

def download_master_videos():
    if not os.path.exists(CKPT_PATH):
        print(f"Error: {CKPT_PATH} not found. Train the model first.")
        return

    print("Loading checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=False)
    angle_refs = ckpt.get('angle_refs', {})

    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Found {len(angle_refs)} classes. Downloading master videos...")
    for cls, ref in angle_refs.items():
        hf_path = ref.get('master_hf_path')
        if not hf_path:
            print(f"Skipping {cls} - no master video path found in checkpoint.")
            continue
            
        out_path = os.path.join(OUT_DIR, f"{cls}.mp4")
        if os.path.exists(out_path):
            print(f"Already exists: {out_path}")
            continue
            
        print(f"Downloading {cls} -> {hf_path}")
        try:
            tmp = hf_hub_download(
                repo_id=REPO_ID, 
                filename=hf_path, 
                repo_type='dataset'
            )
            shutil.copy(tmp, out_path)
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == '__main__':
    download_master_videos()
