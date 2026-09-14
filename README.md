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

## Run it locally (Docker)

`Dockerfile` is the CPU-only build — what CI builds and what production runs;
it works on an arm64 Mac too. `Dockerfile.gpu` is the CUDA variant.

```bash
docker build -t acampa-face:cpu .
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

| | |
|---|---|
| Repo | `kevynsax/ipalpha-acampa-kids-2025-face-service` |
| CI | TeamCity `IpAlpha_Kids_Acampa2025FaceService_Build` — VCS trigger on every push to `master`; reads the version, refuses to rebuild an existing tag, builds and pushes |
| Image | `registry.kevyn.com.br/ip-alpha/kids/acampa-2025-face:<version>` |
| Manifest | `k8s/ipalpha/kids/acampa-2025/face-service.yaml` (namespace `ipalpha-kids`) |
| Version | `face_service/version.py` — bumped by the parent folder's `./publish` |

Production runs the **CPU** image: ~1 s per photo, and indexing happens in the
background after upload, so a GPU buys nothing for a camp-sized album. To move
to the GPU later: build `Dockerfile.gpu`, then add `runtimeClassName: nvidia`
and a `nvidia.com/gpu: "1"` limit to the deployment.

The model pack lives on a PVC (`/mnt/k8s-data/ipalpha/kids/acampa-2025/face-models`)
so restarts do not re-download it.

> Face embeddings of children are biometric data. Keep this service
> cluster-internal — never behind a public ingress.
