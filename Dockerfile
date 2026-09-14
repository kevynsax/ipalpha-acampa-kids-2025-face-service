# Default (and what TeamCity builds): CPU-only ONNX Runtime.
# Multi-arch friendly — builds on the amd64 agents and on an arm64 Mac.
# For the GPU variant see Dockerfile.gpu.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FACE_MODEL_ROOT=/app/data/insightface

# build-essential + python3-dev: insightface ships only an sdist and compiles on install
# libgl1 + libglib2.0-0: OpenCV runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential python3-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-cpu.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install "numpy==1.26.4" "cython==3.0.11" \
    && pip install -r requirements-cpu.txt

COPY face_service ./face_service
EXPOSE 8000
CMD ["uvicorn", "face_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
