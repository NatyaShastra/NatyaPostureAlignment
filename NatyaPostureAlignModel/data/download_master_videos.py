import os
import zipfile
import gdown

OUT_DIR = 'checkpoints/master_videos'
GDRIVE_FILE_ID = '14RBl75ketothB0I-cvN-STsnaD4t3U2b'
ZIP_PATH = 'checkpoints/master_videos.zip'

def download_master_videos():
    os.makedirs('checkpoints', exist_ok=True)
    
    if os.path.exists(OUT_DIR) and len(os.listdir(OUT_DIR)) > 0:
        print(f"Master videos already exist in {OUT_DIR}. Skipping download.")
        return
        
    print(f"Downloading master videos from Google Drive (ID: {GDRIVE_FILE_ID})...")
    url = f'https://drive.google.com/uc?id={GDRIVE_FILE_ID}'
    
    try:
        gdown.download(url, ZIP_PATH, quiet=False)
    except Exception as e:
        print(f"Failed to download from Google Drive: {e}")
        return
        
    if not os.path.exists(ZIP_PATH):
        print("Download failed or file not found.")
        return
        
    print(f"Extracting {ZIP_PATH} to {OUT_DIR}...")
    os.makedirs(OUT_DIR, exist_ok=True)
    
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            # Extract files directly into OUT_DIR, ignoring any folder structure inside the zip
            for member in zip_ref.namelist():
                filename = os.path.basename(member)
                if not filename: # Skip directories
                    continue
                
                source = zip_ref.open(member)
                target_path = os.path.join(OUT_DIR, filename)
                with open(target_path, "wb") as target:
                    target.write(source.read())
        
        print("Extraction complete. Cleaning up zip file...")
        os.remove(ZIP_PATH)
        print("Master videos are ready for production!")
    except Exception as e:
        print(f"Failed to extract zip file: {e}")

if __name__ == '__main__':
    download_master_videos()
