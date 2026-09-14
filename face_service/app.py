import asyncio
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from insightface.app import FaceAnalysis

from .version import version

MODEL_NAME = os.getenv("FACE_MODEL", "buffalo_l")
MODEL_ROOT = os.getenv("FACE_MODEL_ROOT", "/app/data/insightface")
DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))
MAX_BYTES = int(os.getenv("FACE_MAX_BYTES", str(8 * 1024 * 1024)))

analyser: FaceAnalysis | None = None
provider = "CPUExecutionProvider"
# One model instance, one picture at a time: the queue keeps GPU memory flat and
# avoids concurrent reuse of the session's internal buffers.
inference_lock = asyncio.Semaphore(1)


def _load() -> tuple[FaceAnalysis, str]:
    available = set(ort.get_available_providers())
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in available
        else ["CPUExecutionProvider"]
    )
    face_app = FaceAnalysis(name=MODEL_NAME, root=MODEL_ROOT, providers=providers)
    face_app.prepare(ctx_id=0 if providers[0] == "CUDAExecutionProvider" else -1, det_size=(DET_SIZE, DET_SIZE))
    return face_app, providers[0]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global analyser, provider
    # Blocks startup (and the k8s startup probe) until the model pack is on disk.
    analyser, provider = await run_in_threadpool(_load)
    print(f"face service {version} ready: model={MODEL_NAME} provider={provider}", flush=True)
    yield
    analyser = None


app = FastAPI(title="Acampa Kids face embeddings", version=version, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "version": version, "model": MODEL_NAME, "provider": provider, "ready": analyser is not None}


def _embed(image: np.ndarray) -> list[dict]:
    assert analyser is not None
    faces = []
    for face in analyser.get(image):
        bbox = [float(v) for v in np.asarray(face.bbox, dtype=np.float32).tolist()]
        faces.append(
            {
                "embedding": np.asarray(face.normed_embedding, dtype=np.float32).tolist(),
                "detScore": float(face.det_score),
                "bbox": bbox,
            }
        )
    # biggest face first: the subject of a reference picture is the one that matters
    faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
    return faces


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    if analyser is None:
        raise HTTPException(status_code=503, detail="model is not ready")
    raw = await file.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="image is empty or too large")

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=415, detail="invalid image")

    async with inference_lock:
        faces = await run_in_threadpool(_embed, image)
    return {
        "model": MODEL_NAME,
        "provider": provider,
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
        "faces": faces,
    }
