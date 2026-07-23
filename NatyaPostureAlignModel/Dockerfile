FROM python:3.11-slim

# System dependencies — OpenCV and MediaPipe need these
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgles2 \
    libegl1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860

# The cron job hits GET /health to keep the container alive.
# Uvicorn with 1 worker — inference is CPU-bound so multiple workers
# don't help on the free tier and just waste RAM.
CMD sh -c "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"