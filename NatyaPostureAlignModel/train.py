import os
# All outputs go here — change the folder name if you prefer


print('MediaPipe model ready.')


import os, re, json, warnings
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from huggingface_hub import hf_hub_download, list_repo_files
from collections import defaultdict
warnings.filterwarnings('ignore')

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
REPO_ID      = 'vibhuti16/bharatnatyam_adavus'
NUM_FRAMES   = 60          # more frames → better median
NUM_LANDMARKS = 33
MIN_VISIBILITY = 0.50      # frames with more low-conf joints than this are dropped
MAX_BAD_JOINT_FRAC = 0.20  # drop frame if >20% joints below MIN_VISIBILITY
MIN_VIDEOS   = 4           # classes with fewer videos are excluded
MAX_PER_CLASS = None       # set to small int (e.g. 3) for a quick smoke-test

# Paths
FEATURES_CACHE = 'checkpoints/adavu_features.npz'
CKPT_PATH      = 'checkpoints/dance_coach_model.pt'

# Angle definitions  (name, joint_a, vertex, joint_c) — MediaPipe indices
ANGLE_DEFS = [
    ('left_knee',      23, 25, 27),
    ('right_knee',     24, 26, 28),
    ('left_hip',       11, 23, 25),
    ('right_hip',      12, 24, 26),
    ('left_elbow',     11, 13, 15),
    ('right_elbow',    12, 14, 16),
    ('left_shoulder',  13, 11, 23),
    ('right_shoulder', 14, 12, 24),
    ('spine_lean',     23, 11, 24),
]
ANGLE_NAMES = [d[0] for d in ANGLE_DEFS]
NUM_ANGLES  = len(ANGLE_DEFS)

# Feature dimension breakdown:
#   normalised coords: 33 joints × 2 (x,y only) × 2 stats (mean, std) = 132
#   angles:            9 angles  × 3 stats (mean, std, velocity)       =  27
#   symmetry:          4 pairs   × 1 (L-R angle diff)                  =   4
FEATURE_DIM = 132 + 27 + 4   # = 163

print(f'Device : {DEVICE}')
print(f'Feature dim : {FEATURE_DIM}')


CLASS_ALIASES = {
    'thattimettuadavu':               'ThattimettuAdavu',
    'thatimettuadavu':                'ThattimettuAdavu',
    'kudhittamettuadavu':             'KudithuMettuAdavu',
    'kudhittamettu':                  'KudithuMettuAdavu',
    'kudithumettuadavu':              'KudithuMettuAdavu',
    'korvaiadavu':                    'KorvaiAdavu',
    'karthariadavu':                  'KarthariAdavu',
    'thahathajamtharithaadavu':       'ThaHathaJhamTharithamAdavu',
    'thahathajhamtharithamadavu':     'ThaHathaJhamTharithamAdavu',
    'thahathajanjtharithaadavu':      'ThaHathaJhamTharithamAdavu',
    'thaithaithathamadavu':           'ThaiThaiThaThamAdavu',
    'thaithaithatham':                'ThaiThaiThaThamAdavu',
    'mandiadavu':                     'MandiAdavu',
    'sarukkaladavu':                  'SarukkalAdavu',
    'thathaithaha':                   'ThathaiThaha',
    'thathaithahaadavu':              'ThathaiThaha',
    'thatheitheiha':                  'ThaTheiTheiTha',
    'thaiyathaihi':                   'ThaiyaThaihi',
    'thaiyathaihi adavu':             'ThaiyaThaihi',
    'theermaanamadavu':               'TheermanaAdavu',
    'theermanaadavu':                 'TheermanaAdavu',
    'theermanaadavulearnandpractice': 'TheermanaAdavu',
    'theermanaadavulearn':            'TheermanaAdavu',
    'theerumanamadavu':               'TheermanaAdavu',
    'uthsangaadavu':                  'UthsangaAdavu',
    'uthplavanaadavu':                'UtplavanadaAdavu',
    'kudhitthamettuadavu':            'KudithuMettuAdavu',
}

def normalize_class(name):
    key = re.sub(r'[\s_\-]+', '', name).lower()
    key = re.sub(r'kalakshetrastyle$', '', key)
    key = re.sub(r'learn(andpractice)?$', '', key)
    return CLASS_ALIASES.get(key, name)

def get_class(path):
    filename = path.split('/')[-1].replace('.mp4', '')
    match = re.match(r'^([A-Za-z\s]+?)(\d.*)?$', filename)
    raw = match.group(1).strip() if match else filename
    return normalize_class(raw)

print('Class normalisation ready.')


print('Listing repo files...')
all_files   = list(list_repo_files(REPO_ID, repo_type='dataset'))
video_files = [f for f in all_files if f.startswith('videos/') and f.endswith('.mp4')]
print(f'Total video files found: {len(video_files)}')

class_counts = defaultdict(int)
for f in video_files:
    class_counts[get_class(f)] += 1

print('\nClass counts (raw):')
for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
    print(f'  {cls:<40s} {cnt:3d}')

valid_classes       = {cls for cls, cnt in class_counts.items() if cnt >= MIN_VIDEOS}
video_files_filtered = [f for f in video_files if get_class(f) in valid_classes]

print(f'\nKept {len(valid_classes)} classes with >= {MIN_VIDEOS} videos')
print(f'Kept {len(video_files_filtered)} / {len(video_files)} videos')


# ── MediaPipe initialisation ─────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options      = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False,
    num_poses=1,
)
mp_pose = vision.PoseLandmarker.create_from_options(options)


# ── Angle helper ─────────────────────────────────────────────────────────
def _angle_between(pa, pv, pc):
    v1 = pa - pv;  v2 = pc - pv
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_a)))


def compute_angles_for_frame(frame):
    """frame: (33, 3) — returns (NUM_ANGLES,) in degrees"""
    return np.array([
        _angle_between(frame[a, :2], frame[v, :2], frame[c, :2])
        for _, a, v, c in ANGLE_DEFS
    ])


# ── Landmark normalisation ────────────────────────────────────────────────
def normalise_landmarks(seq):
    """
    Make landmarks camera- and distance-invariant.
    Origin  → hip midpoint
    Scale   → torso length (hip-mid to shoulder-mid)
    seq: (T, 33, 3)  returns same shape (visibility col unchanged)
    """
    seq = seq.copy()
    hip_mid      = (seq[:, 23, :2] + seq[:, 24, :2]) / 2          # (T, 2)
    shoulder_mid = (seq[:, 11, :2] + seq[:, 12, :2]) / 2          # (T, 2)
    scale        = np.linalg.norm(shoulder_mid - hip_mid, axis=1)  # (T,)
    scale        = np.maximum(scale, 1e-6)[:, np.newaxis]          # (T, 1)

    seq[:, :, :2] = (seq[:, :, :2] - hip_mid[:, np.newaxis, :]) / scale[:, np.newaxis, :]
    return seq


# ── Core extraction ───────────────────────────────────────────────────────

def pad_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w: return image
    size = max(h, w)
    pad_h = (size - h) // 2
    pad_w = (size - w) // 2
    return cv2.copyMakeBorder(image, pad_h, size - h - pad_h, pad_w, size - w - pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def extract_landmarks_from_video(video_path, num_frames=NUM_FRAMES):
    """
    Sample num_frames evenly, filter low-confidence frames, return median pose.

    Returns:
        seq_norm  : (T_kept, 33, 3) normalised landmark sequence
        angles_seq: (T_kept, NUM_ANGLES) per-frame angles (degrees)
        None, None on failure
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return None, None

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    raw_seq = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = pad_to_square(frame)
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = mp_pose.detect(mp_image)

        if not result.pose_landmarks:
            continue

        lm = result.pose_landmarks[0]
        arr = np.array([[l.x, l.y, l.visibility] for l in lm])  # (33, 3)

        # Drop frame if too many joints are low-confidence
        bad_frac = np.mean(arr[:, 2] < MIN_VISIBILITY)
        if bad_frac > MAX_BAD_JOINT_FRAC:
            continue

        raw_seq.append(arr)

    cap.release()

    if len(raw_seq) < 5:   # too few usable frames
        return None, None

    raw_seq   = np.array(raw_seq)            # (T_kept, 33, 3)
    seq_norm  = normalise_landmarks(raw_seq) # (T_kept, 33, 3)
    angles_seq = np.array([
        compute_angles_for_frame(seq_norm[t]) for t in range(len(seq_norm))
    ])                                       # (T_kept, NUM_ANGLES)

    return seq_norm, angles_seq


# ── Symmetry features ─────────────────────────────────────────────────────
# Pairs: (left_angle_idx, right_angle_idx)
SYMMETRY_PAIRS = [
    (ANGLE_NAMES.index('left_knee'),     ANGLE_NAMES.index('right_knee')),
    (ANGLE_NAMES.index('left_hip'),      ANGLE_NAMES.index('right_hip')),
    (ANGLE_NAMES.index('left_elbow'),    ANGLE_NAMES.index('right_elbow')),
    (ANGLE_NAMES.index('left_shoulder'), ANGLE_NAMES.index('right_shoulder')),
]

def compute_symmetry_features(angles_mean):
    """angles_mean: (NUM_ANGLES,) — returns (4,) L-R differences"""
    return np.array([
        abs(angles_mean[l] - angles_mean[r]) for l, r in SYMMETRY_PAIRS
    ])


# ── Feature vector builder ────────────────────────────────────────────────
def build_feature_vector(seq_norm, angles_seq):
    """
    seq_norm   : (T, 33, 3)
    angles_seq : (T, NUM_ANGLES)
    Returns    : (FEATURE_DIM,) = 132 + 27 + 4
    """
    # 1. Normalised coordinate stats — x,y only (drop visibility)
    coords = seq_norm[:, :, :2]              # (T, 33, 2)
    coord_mean = coords.mean(axis=0).flatten()   # 66
    coord_std  = coords.std(axis=0).flatten()    # 66  → total 132

    # 2. Angle stats
    angle_mean = angles_seq.mean(axis=0)         # 9
    angle_std  = angles_seq.std(axis=0)          # 9
    angle_vel  = np.abs(np.diff(angles_seq, axis=0)).mean(axis=0)  # 9 → total 27

    # 3. Symmetry features
    sym = compute_symmetry_features(angle_mean)  # 4

    return np.concatenate([coord_mean, coord_std, angle_mean, angle_std, angle_vel, sym])


print('Pose extraction utilities ready.')
print(f'Expected feature dim: {66+66+9+9+9+4} (should be {FEATURE_DIM})')


if os.path.exists(FEATURES_CACHE):
    print(f'Cache found → {FEATURES_CACHE}')
    data        = np.load(FEATURES_CACHE, allow_pickle=True)
    X           = data['X']
    y           = data['y']
    label_names = list(data['label_names'])
    angle_means = data['angle_means']
    angle_stds  = data['angle_stds']
    print(f'Loaded {len(X)} samples, {len(label_names)} classes.')
else:
    class_files = defaultdict(list)
    for f in video_files_filtered:
        class_files[get_class(f)].append(f)
    class_files = defaultdict(list)
    for f in video_files_filtered:
        class_files[get_class(f)].append(f)

    features, labels, angle_means_list, angle_stds_list, failed = [], [], [], [], []

    for cls, files in class_files.items():
        if cls not in valid_classes:
            continue
        subset = files[:MAX_PER_CLASS] if MAX_PER_CLASS else files
        print(f'Processing {cls} ({len(subset)} videos)')

        for hf_path in tqdm(subset, desc=cls, leave=False):
            tmp = None
            try:
                tmp = hf_hub_download(
                    repo_id=REPO_ID, filename=hf_path,
                    repo_type='dataset', local_dir='/tmp/hf_cache'
                )
                seq_norm, angles_seq = extract_landmarks_from_video(tmp)

                if seq_norm is None:
                    failed.append(hf_path)
                    continue

                features.append(build_feature_vector(seq_norm, angles_seq))
                labels.append(cls)
                angle_means_list.append(angles_seq.mean(axis=0))
                angle_stds_list.append(angles_seq.std(axis=0))

            except Exception as e:
                failed.append(hf_path)
                print(f'  FAILED: {hf_path} — {e}')
            finally:
                if tmp and os.path.exists(tmp):
                    os.remove(tmp)

    X           = np.array(features)
    y           = np.array(labels)
    label_names = sorted(set(y))
    angle_means = np.array(angle_means_list)   # (N, NUM_ANGLES)
    angle_stds  = np.array(angle_stds_list)    # (N, NUM_ANGLES)

    np.savez(
        FEATURES_CACHE,
        X=X, y=y, label_names=label_names,
        angle_means=angle_means, angle_stds=angle_stds,
    )
    print(f'\nSaved cache → {FEATURES_CACHE}')
    print(f'Samples: {len(X)} | Classes: {len(label_names)} | Failed: {len(failed)}')

    # Copy to Drive immediately
    import shutil
    print(f'Copied to Drive.')

print('\nClass distribution:')
for c in label_names:
    print(f'  {c:<40s} {np.sum(y == c):3d}')


# Mirror/flip: swap left↔right landmark indices
# MediaPipe left/right pairs
LR_PAIRS = [
    (11, 12), (13, 14), (15, 16),   # shoulders, elbows, wrists
    (17, 18), (19, 20), (21, 22),   # hand landmarks
    (23, 24), (25, 26), (27, 28),   # hips, knees, ankles
    (29, 30), (31, 32),             # heel, foot index
    (1, 4), (2, 5), (3, 6),         # face
    (7, 8), (9, 10),
]

def flip_sequence(seq_norm):
    """
    Horizontally mirror a normalised (T, 33, 3) sequence.
    Flips x → -x and swaps left/right landmark indices.
    """
    flipped = seq_norm.copy()
    flipped[:, :, 0] *= -1          # mirror x
    for l, r in LR_PAIRS:
        flipped[:, [l, r]] = flipped[:, [r, l]]
    return flipped


def add_noise(seq_norm, sigma=0.01):
    """Add small gaussian noise to (x, y) coords only."""
    noisy = seq_norm.copy()
    noisy[:, :, :2] += np.random.normal(0, sigma, noisy[:, :, :2].shape)
    return noisy


def speed_warp(seq_norm, angles_seq, factor=0.8):
    """
    Resample the sequence in time by `factor`.
    factor < 1  →  fewer frames (faster)
    factor > 1  →  more frames (slower)
    Both outputs are resampled back to the original length.
    """
    T = len(seq_norm)
    new_T   = max(5, int(T * factor))
    old_idx = np.linspace(0, T - 1, new_T)
    new_idx = np.linspace(0, new_T - 1, T)

    def interp_seq(s):
        out = np.zeros_like(s)
        for j in range(s.shape[1]):
            for k in range(s.shape[2]):
                sampled = np.interp(old_idx, np.arange(T), s[:, j, k])
                out[:, j, k] = np.interp(new_idx, np.arange(new_T), sampled)
        return out

    def interp_ang(a):
        out = np.zeros_like(a)
        for j in range(a.shape[1]):
            sampled = np.interp(old_idx, np.arange(T), a[:, j])
            out[:, j] = np.interp(new_idx, np.arange(new_T), sampled)
        return out

    return interp_seq(seq_norm), interp_ang(angles_seq)


def augment_sample(seq_norm, angles_seq):
    """
    Return a list of (feature_vec, angle_mean, angle_std) tuples —
    one per augmentation variant (not including the original).
    """
    variants = []

    # 1. Horizontal flip
    flipped     = flip_sequence(seq_norm)
    flip_angles = np.array([compute_angles_for_frame(flipped[t]) for t in range(len(flipped))])
    variants.append((
        build_feature_vector(flipped, flip_angles),
        flip_angles.mean(axis=0),
        flip_angles.std(axis=0),
    ))

    # 2. Gaussian noise
    noisy       = add_noise(seq_norm)
    noisy_angles = np.array([compute_angles_for_frame(noisy[t]) for t in range(len(noisy))])
    variants.append((
        build_feature_vector(noisy, noisy_angles),
        noisy_angles.mean(axis=0),
        noisy_angles.std(axis=0),
    ))

    # 3. Speed warp slow (1.2×)
    sw_seq, sw_ang = speed_warp(seq_norm, angles_seq, factor=1.2)
    variants.append((
        build_feature_vector(sw_seq, sw_ang),
        sw_ang.mean(axis=0),
        sw_ang.std(axis=0),
    ))

    # 4. Speed warp fast (0.8×)
    sw_seq2, sw_ang2 = speed_warp(seq_norm, angles_seq, factor=0.8)
    variants.append((
        build_feature_vector(sw_seq2, sw_ang2),
        sw_ang2.mean(axis=0),
        sw_ang2.std(axis=0),
    ))

    return variants


print('Augmentation functions ready.')
print('Variants per sample: 4 (flip, noise, slow, fast)')


# Count samples per class in original dataset
class_sample_counts = {cls: int(np.sum(y == cls)) for cls in label_names}
max_count = max(class_sample_counts.values())
print(f'Original dataset: {len(X)} samples')
print(f'Max class size  : {max_count}')
print(f'\nPer-class counts:')
for c, n in sorted(class_sample_counts.items(), key=lambda x: -x[1]):
    print(f'  {c:<40s} {n:3d}')

# We need to re-extract raw sequences for augmentation.
# Strategy: re-download each video once, store (seq_norm, angles_seq)
# alongside its label, then augment minority classes to reach max_count.

print('\nRe-downloading videos to build augmented dataset...')
print('(This re-uses /tmp/hf_cache so already-cached files are fast)')

raw_samples = []   # list of (feature_vec, angle_mean, angle_std, label)

for cls, files in class_files.items():
    if cls not in valid_classes:
        continue
    subset = files[:MAX_PER_CLASS] if MAX_PER_CLASS else files

    for hf_path in tqdm(subset, desc=cls, leave=False):
        tmp = None
        try:
            tmp = hf_hub_download(
                repo_id=REPO_ID, filename=hf_path,
                repo_type='dataset', local_dir='/tmp/hf_cache'
            )
            seq_norm, angles_seq = extract_landmarks_from_video(tmp)
            if seq_norm is None:
                continue

            fv    = build_feature_vector(seq_norm, angles_seq)
            a_mu  = angles_seq.mean(axis=0)
            a_sig = angles_seq.std(axis=0)
            raw_samples.append((fv, a_mu, a_sig, cls, seq_norm, angles_seq))

        except Exception as e:
            print(f'  FAILED: {hf_path} — {e}')
        finally:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)

print(f'Raw samples collected: {len(raw_samples)}')

# ── Balance by augmenting minority classes ───────────────────────────────
X_aug, y_aug, am_aug, as_aug = [], [], [], []

class_raw = defaultdict(list)
for item in raw_samples:
    class_raw[item[3]].append(item)

for cls in label_names:
    items = class_raw[cls]
    # Always add originals
    for fv, a_mu, a_sig, _, seq_norm, angles_seq in items:
        X_aug.append(fv); y_aug.append(cls)
        am_aug.append(a_mu); as_aug.append(a_sig)

    needed = max_count - len(items)
    if needed <= 0:
        continue

    # Cycle through originals and augment until we reach max_count
    pool = items.copy()
    added = 0
    while added < needed:
        src = pool[added % len(pool)]
        fv, a_mu, a_sig, _, seq_norm, angles_seq = src
        variants = augment_sample(seq_norm, angles_seq)
        for v_fv, v_amu, v_asig in variants:
            if added >= needed:
                break
            X_aug.append(v_fv); y_aug.append(cls)
            am_aug.append(v_amu); as_aug.append(v_asig)
            added += 1

X_aug = np.array(X_aug)
y_aug = np.array(y_aug)
am_aug = np.array(am_aug)
as_aug = np.array(as_aug)

print(f'\nAugmented dataset: {len(X_aug)} samples')
print(f'Per-class after balancing:')
for c in label_names:
    print(f'  {c:<40s} {np.sum(y_aug == c):3d}')


# Per-class angle reference: computed from ORIGINAL samples only (not augmented)
# so the reference reflects real human execution, not synthetic variants.

REGIONS = {
    'legs':  ['left_knee', 'right_knee', 'left_hip', 'right_hip'],
    'arms':  ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder'],
    'torso': ['spine_lean'],
}

angle_refs = {}
for cls in label_names:
    # Use only original (non-augmented) samples — am_aug[:len(X)] corresponds to originals
    orig_mask = (y_aug == cls)[:len(X)]   # mask within original portion
    orig_mus  = am_aug[:len(X)][orig_mask]  # (N_cls, NUM_ANGLES)

    if len(orig_mus) == 0:
        continue

    angle_refs[cls] = {
        'mean': orig_mus.mean(axis=0),
        # Floor std at 3° — avoids division by zero on very consistent classes
        'std':  np.maximum(orig_mus.std(axis=0), 3.0),
    }

print('Angle reference distributions built:')
for cls, ref in angle_refs.items():
    print(f'  {cls}')
    for i, name in enumerate(ANGLE_NAMES):
        print(f'    {name:<20s}  mean={ref["mean"][i]:6.1f}°  std={ref["std"][i]:5.1f}°')
    print()


le = LabelEncoder()
le.fit(label_names)
y_enc = le.transform(y_aug)
NUM_CLASSES = len(le.classes_)

X_mean = X_aug.mean(axis=0)
X_std  = X_aug.std(axis=0) + 1e-8
X_norm = (X_aug - X_mean) / X_std

try:
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
except ValueError:
    print('Warning: stratified split failed — using random split.')
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_enc, test_size=0.2, random_state=42
    )

print(f'Train: {len(X_train)} | Val: {len(X_val)} | Classes: {NUM_CLASSES}')
print('Classes:', list(le.classes_))


class AdavuClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=256, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, hidden // 4),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden // 4, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class AdavuDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


model = AdavuClassifier(FEATURE_DIM, NUM_CLASSES).to(DEVICE)
total = sum(p.numel() for p in model.parameters())
print(model)
print(f'\nTotal parameters: {total:,}')


EPOCHS = 200
BATCH  = 32
LR     = 3e-4
WD     = 1e-4

train_loader = DataLoader(AdavuDataset(X_train, y_train), batch_size=BATCH, shuffle=True,  drop_last=True)
val_loader   = DataLoader(AdavuDataset(X_val,   y_val),   batch_size=BATCH, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Class weights from TRAINING labels only (y_train is already balanced so weights ~ equal,
# but keep it in case some classes still have slight imbalance after augmentation)
counts_tr = np.array([np.sum(y_train == i) for i in range(NUM_CLASSES)], dtype=float)
cw = 1.0 / (counts_tr + 1e-8)
cw = cw / cw.sum() * NUM_CLASSES
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(cw).to(DEVICE))

train_losses, val_losses, val_accs = [], [], []
best_val_acc, best_state = 0.0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running += loss.item() * len(Xb)
    train_losses.append(running / len(X_train))
    scheduler.step()

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            val_loss += criterion(logits, yb).item() * len(Xb)
            correct  += (logits.argmax(1) == yb).sum().item()
    val_losses.append(val_loss / len(X_val))
    acc = correct / len(X_val)
    val_accs.append(acc)

    if acc > best_val_acc:
        best_val_acc = acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 25 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{EPOCHS}  train_loss={train_losses[-1]:.4f}  '
              f'val_loss={val_losses[-1]:.4f}  val_acc={acc:.2%}')

model.load_state_dict(best_state)
print(f'\nBest val accuracy: {best_val_acc:.2%}')


fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(train_losses, label='Train loss')
axes[0].plot(val_losses,   label='Val loss')
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend()

axes[1].plot(val_accs)
axes[1].axhline(best_val_acc, color='r', linestyle='--', label=f'Best {best_val_acc:.2%}')
axes[1].set_title('Val Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend()

plt.tight_layout()
plot_path = 'checkpoints/training_curves.png'
plt.savefig(plot_path, dpi=150)
print(f'Plot saved → {plot_path}')


model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for Xb, yb in val_loader:
        preds = model(Xb.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

print(classification_report(all_true, all_preds, target_names=le.classes_))

cm = confusion_matrix(all_true, all_preds)
fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_,
            yticklabels=le.classes_, ax=ax, cmap='Blues')
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title('Confusion Matrix — v1 model')
plt.tight_layout()
cm_path = 'checkpoints/confusion_matrix.png'
plt.savefig(cm_path, dpi=150)
print(f'Confusion matrix saved → {cm_path}')


import shutil

# ── Save checkpoint ──────────────────────────────────────────────────────
ckpt = {
    'model_state':  best_state,
    'label_encoder': le,
    'X_mean':       X_mean,
    'X_std':        X_std,
    'num_classes':  NUM_CLASSES,
    'feature_dim':  FEATURE_DIM,
    'angle_refs':   angle_refs,        # baked-in reference distributions
    'angle_names':  ANGLE_NAMES,
    'angle_defs':   ANGLE_DEFS,
    'regions':      REGIONS,
    'best_val_acc': best_val_acc,
}

torch.save(ckpt, CKPT_PATH)
print(f'Checkpoint saved -> {CKPT_PATH}')
