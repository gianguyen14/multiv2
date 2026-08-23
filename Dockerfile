# ==============================================================================
# AIC 2026 Multimodal Video Retrieval System — Production Dockerfile
# Base: Python 3.12 Slim (Debian Bookworm)
# ==============================================================================

FROM python:3.12-slim

# Prevent bytecode compilation and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    VIDEO_PROCESSED_ROOT=/data/processed \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    TORCH_HOME=/cache/torch \
    TMPDIR=/tmp

# Install essential OS packages:
# - build-essential, gcc, g++: C/C++ compilation for native extensions / PyTorch JIT
# - git: VCS provenance inspection
# - ffmpeg: Video demuxing, decoding, and audio extraction
# - tesseract-ocr (eng, vie): Optical Character Recognition engines
# - libgl1, libglib2.0-0: OpenCV and Pillow image rendering runtimes
# - curl: Lightweight healthcheck utility
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-vie \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root application user
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Create runtime directories with non-root ownership
RUN mkdir -p /data/videos /data/processed /models /cache/huggingface /cache/torch /logs /tmp \
    && chown -R appuser:appuser /data /models /cache /logs /tmp /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements/base.txt requirements/base.txt
COPY pyproject.toml .

# Install Python runtime dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/base.txt

# Copy entrypoint script and set executable permissions
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Copy application source code
COPY backend backend
COPY frontend frontend
COPY projectctl.py .

# Ensure app directory permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose default API port
EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]

# Production server: 1 worker to prevent ML model memory duplication, no auto-reload
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
