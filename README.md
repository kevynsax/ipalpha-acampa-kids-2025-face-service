# Acampa Kids — face service 🙂

Private HTTP service that turns an image into **face embeddings**. It is the
only component that ever sees a parent's reference photo, and it stores
nothing: bytes in, vectors out.

Used by the camping backend (`FACE_SERVICE_URL`) to

1. index every gallery photo after upload, and
2. match a parent's reference picture against those stored vectors.

## Stack

- **Detection + embeddings**: InsightFace `buffalo_l` (SCRFD detector + ArcFace, 512 floats)
- **Runtime**: ONNX Runtime — CUDA when a GPU is present, CPU otherwise
- **API**: FastAPI + Uvicorn
- The model pack downloads on first start into `FACE_MODEL_ROOT` (a PVC in production)

## Endpoints

| Method | Path | Body | Result |
|---|---|---|---|
| GET | `/health` | — | `{ status, version, model, provider, ready }` |
| POST | `/embed` | multipart `file` | `{ model, provider, width, height, faces: [{ embedding, detScore, bbox }] }` |

`faces` is sorted by bounding-box area (largest first), so `faces[0]` is the
subject of a reference picture. Embeddings are L2-normalised: comparing two of
them is a plain dot product.

The service never decides *who* someone is — it has no names, no database and
no state. Thresholding and matching live in the backend.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `FACE_MODEL` | `buffalo_l` | InsightFace model pack |
| `FACE_MODEL_ROOT` | `/app/data/insightface` | where the pack is cached |
| `FACE_DET_SIZE` | `640` | detector input edge |
| `FACE_MAX_BYTES` | `8388608` | request size guard |

## Run it locally (Docker, CPU)

`Dockerfile` is the CUDA production image; `Dockerfile.cpu` is the same code on
CPU-only ONNX Runtime and builds on an arm64 Mac.

```bash
docker build -f Dockerfile.cpu -t acampa-face:cpu .
docker run --rm -p 8000:8000 -v "$PWD/.models:/app/data/insightface" acampa-face:cpu
```

The first start downloads the model pack (~300 MB) into `.models/`; keep the
volume so restarts are instant. Then:

```bash
curl -s localhost:8000/health
curl -s -F file=@/path/to/photo.jpg localhost:8000/embed | head -c 400
```

Point the backend at it with `FACE_SERVICE_URL=http://localhost:8000`.

### Without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt
FACE_MODEL_ROOT=.models uvicorn face_service.app:app --port 8000
```

## Deployment

Published by the parent `camping/` folder's `./publish` (repo
`ipalpha-acampa-kids-2025-face-service`, image
`ip-alpha/kids/acampa-2025-face`), manifest
`k8s/ipalpha/kids/acampa-2025/face-service.yaml`. The version lives in
`face_service/version.py`.

The GPU variant asks for one time-sliced `nvidia.com/gpu`; dropping that limit
and `runtimeClassName: nvidia` falls back to CPU (seconds per photo instead of
fractions).

> Face embeddings of children are biometric data. Keep this service
> cluster-internal — never behind a public ingress.
