import re
with open("train.py", "r") as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if line.startswith("%%capture") or line.startswith("!pip") or line.startswith("!wget") or "google.colab" in line:
        continue
    if "DRIVE_DIR = '/content/drive" in line or "os.makedirs(DRIVE_DIR" in line or "print(f'Drive output folder" in line:
        continue
    if line.startswith("FEATURES_CACHE ="):
        new_lines.append("FEATURES_CACHE = 'checkpoints/adavu_features.npz'\n")
        continue
    if line.startswith("CKPT_PATH      ="):
        new_lines.append("CKPT_PATH      = 'checkpoints/dance_coach_model.pt'\n")
        continue
    if "plot_path = os.path.join(DRIVE_DIR" in line:
        new_lines.append("plot_path = 'checkpoints/training_curves.png'\n")
        continue
    if "cm_path = os.path.join(DRIVE_DIR" in line:
        new_lines.append("cm_path = 'checkpoints/confusion_matrix.png'\n")
        continue
    if "drive_ckpt =" in line or "shutil.copy(CKPT_PATH" in line or "print(f'Checkpoint copied" in line:
        continue
    if "drive_npz =" in line or "shutil.copy(FEATURES_CACHE" in line or "print(f'Features cache copied" in line:
        continue
    if "hf_token = input" in line: # truncate here
        break
        
    new_lines.append(line)

code = "".join(new_lines)

# Add pad_to_square
pad_func = """
def pad_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w: return image
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    return cv2.copyMakeBorder(image, pad_h, size - h - pad_h, pad_w, size - w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
"""
code = code.replace("def extract_landmarks_from_video", pad_func + "\ndef extract_landmarks_from_video")

# Use pad_to_square in extract_landmarks
old_extract = """        ret, frame = cap.read()
        if not ret:
            continue

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)"""
new_extract = """        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = pad_to_square(frame)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)"""
code = code.replace(old_extract, new_extract)

with open("train.py", "w") as f:
    f.write(code)

