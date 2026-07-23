"""
api/app.py — FastAPI backend for Dance Coach AI.

Endpoints:
  POST /analyse   — Upload a dance video, get back a full coaching report
  GET  /health    — Liveness probe (used by cron job to keep Space warm)

This file is also the HF Spaces entrypoint.
The Dockerfile sets:  CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]

Environment variables:
  GROQ_API_KEY          — Groq API key for LLM feedback (optional, falls back to templates)
  HF_MODEL_REPO         — HF model repo ID to download checkpoints from  (default: see below)
  DANCE_COACH_MODEL_PT  — local path to checkpoint   (default: checkpoints/dance_coach_model.pt)
  DANCE_COACH_FEATURES  — local path to feature cache (default: checkpoints/adavu_features.npz)
  MEDIAPIPE_MODEL       — local path to .task file    (default: pose_landmarker_heavy.task)
"""

from __future__ import annotations
import os
import sys
import tempfile
from contextlib import asynccontextmanager
import urllib.request

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure repo root is on the path when running from /app
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import load_model_and_refs, run_coach_v2

# ---------------------------------------------------------------------------
# Startup — download artefacts from HF if not present, then load
# ---------------------------------------------------------------------------

CHECKPOINT_PATH  = os.environ.get("DANCE_COACH_MODEL_PT",  "checkpoints/dance_coach_model.pt")
FEATURES_PATH    = os.environ.get("DANCE_COACH_FEATURES",  "checkpoints/adavu_features.npz")
MEDIAPIPE_PATH   = os.environ.get("MEDIAPIPE_MODEL",       "pose_landmarker_heavy.task")
HF_MODEL_REPO    = os.environ.get("HF_MODEL_REPO",        "theusefulnerd/dance-coach-model")

MEDIAPIPE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

MAX_VIDEO_BYTES = 100 * 1024 * 1024   # 100 MB hard cap on uploads


def _download_checkpoints() -> None:
    """Download model checkpoint and feature cache from GitHub if not already present or if they are just LFS pointers."""
    os.makedirs("checkpoints", exist_ok=True)

    def needs_download(path: str) -> bool:
        return not os.path.exists(path) or os.path.getsize(path) < 1024

    if needs_download(CHECKPOINT_PATH) or needs_download(FEATURES_PATH):
        print("[startup] Downloading model artefacts from GitHub (LFS pointers detected)...")
        try:
            # Using the direct raw GitHub links to fetch the actual LFS binaries
            base_url = "https://github.com/NatyaShastra/NatyaPostureAlignment/raw/feature/upgrades/NatyaPostureAlignModel/checkpoints/"
            
            if needs_download(CHECKPOINT_PATH):
                print(f"[startup] Downloading dance_coach_model.pt...")
                urllib.request.urlretrieve(base_url + "dance_coach_model.pt", CHECKPOINT_PATH)
                print(f"[startup] Downloaded dance_coach_model.pt")
                
            if needs_download(FEATURES_PATH):
                print(f"[startup] Downloading adavu_features.npz...")
                urllib.request.urlretrieve(base_url + "adavu_features.npz", FEATURES_PATH)
                print(f"[startup] Downloaded adavu_features.npz")
        except Exception as e:
            print(f"[startup] WARNING: Could not download from GitHub: {e}")
            print("[startup] Continuing — will fail at inference if files are missing or invalid")

    if not os.path.exists(MEDIAPIPE_PATH):
        print("[startup] Downloading MediaPipe pose model...")
        
        urllib.request.urlretrieve(MEDIAPIPE_URL, MEDIAPIPE_PATH)
        print(f"[startup] MediaPipe model ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all inference dependencies once at startup."""
    _download_checkpoints()
    load_model_and_refs(
        checkpoint_path=CHECKPOINT_PATH,
        features_cache=FEATURES_PATH,
        mediapipe_model=MEDIAPIPE_PATH,
        groq_api_key=os.environ.get("GROQ_API_KEY"),
    )
    print("[startup] Dance Coach ready ✓")
    yield
    # Teardown (nothing to clean up)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Dance Coach AI",
    description="Bharatanatyam adavu analysis — pose classification, scoring, and coaching feedback",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS# Read allowed origins from env, defaulting to localhost for dev and netlify for prod
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5500,https://natya-posture-alignment.netlify.app").split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """
    Liveness probe.
    Returns 200 so that the HF Spaces cron job can keep the container warm.
    """
    return {"status": "ok", "service": "dance-coach-api"}


@app.post("/analyse")
async def analyse(video: UploadFile = File(...)):
    """
    Analyse a Bharatanatyam dance video.

    - Accepts: multipart/form-data with a `video` file field (.mp4 / .mov / .avi)
    - Returns: JSON coaching report (see result dict schema in handoff doc)

    The overlay image is returned as a base64-encoded JPEG string in
    `overlay_image_b64`. The frontend should decode and display it as an
    <img src="data:image/jpeg;base64,..."> tag.
    """
    # Validate content type
    if video.content_type not in ("video/mp4", "video/quicktime", "video/x-msvideo", "application/octet-stream"):
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {video.content_type}")

    # Read and size-check
    data = await video.read()
    if len(data) > MAX_VIDEO_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Video too large ({len(data) // 1024 // 1024} MB). Maximum is 100 MB.",
        )

    # Write to temp file (MediaPipe needs a file path, not bytes)
    suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp_path = f.name

    try:
        result = run_coach_v2(tmp_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])

    return result