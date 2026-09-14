FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:$PATH \
    FACE_MODEL_ROOT=/app/data/insightface

# build-essential + python3-dev: insightface ships only an sdist and compiles on install
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential python3 python3-dev python3-venv libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-gpu.txt ./
RUN python3 -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel \
    && pip install "numpy==1.26.4" "cython==3.0.11" \
    && pip install -r requirements-gpu.txt

COPY face_service ./face_service
EXPOSE 8000
CMD ["uvicorn", "face_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
