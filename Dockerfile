FROM python:3.12-slim

# Install system dependencies:
# - build-essential, gcc, g++: C/C++ compiler required for the native retrieval core and PyTorch/Triton JIT
# - git: VCS executable for provenance tracking
# - FFmpeg: Video decoding and demuxing
# - Tesseract OCR with English and Vietnamese packs: Production OCR engine
# - OpenCV runtime libs: libgl1, libglib2.0-0
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching.
COPY requirements/base.txt requirements/base.txt
COPY pyproject.toml .

# Install Python runtime dependencies.
RUN pip install --no-cache-dir -r requirements/base.txt

# Copy native build inputs and application source.
COPY setup.py .
COPY native native
COPY backend backend

# Native acceleration is mandatory for this experimental C++ image. The Python
# source still contains fallbacks, but a broken compiler/ABI must fail the image build.
RUN UVR_NATIVE_STRICT_BUILD=1 python setup.py build_ext --inplace \
    && UVR_NATIVE_CORE=cpp python -c "from backend.app.native import native_status; s=native_status(); assert s['available'] and s['backend']=='cpp', s; print('native core:', s)"

# Copy entrypoint script and remaining application source code.
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

COPY frontend frontend
COPY projectctl.py .

# Environment defaults.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UVR_NATIVE_CORE=auto \
    VIDEO_PROCESSED_ROOT=/data/processed \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    TMPDIR=/tmp

# Create mount points and ensure permissive directory access for non-root host execution.
RUN mkdir -p /data/videos /data/processed /models /tmp \
    && chmod -R 777 /data/processed /models /tmp

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]

# Default command runs the FastAPI backend service.
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
