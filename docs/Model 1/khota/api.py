"""HTTP API around the Khota pipeline, for the mobile app.

Run (from the folder that contains khota/ and weights/):
    python -m uvicorn khota.api:app --host 0.0.0.0 --port 8000
Call:
    curl -F "image=@photo.jpg" http://localhost:8000/predict

Environment variables
    KHOTA_WEIGHTS        weights folder (default: weights)
    KHOTA_DEVICE         "cpu" to force CPU; default: GPU if available
    KHOTA_MAX_UPLOAD_MB  largest accepted upload in MB (default: 25)
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from . import __version__
from .pipeline import KhotaPipeline

_pipe: KhotaPipeline | None = None
_lock = threading.Lock()  # one prediction at a time: the models are shared and the GPU is used by one request
MAX_BYTES = int(float(os.environ.get("KHOTA_MAX_UPLOAD_MB", "25")) * 1024 * 1024)


def pipe() -> KhotaPipeline:
    global _pipe
    if _pipe is None:
        over = {"device": os.environ["KHOTA_DEVICE"]} if os.environ.get("KHOTA_DEVICE") else {}
        _pipe = KhotaPipeline.from_dir(os.environ.get("KHOTA_WEIGHTS", "weights"), **over)
    return _pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load every model once at start-up and run one warm-up prediction, so the first real request is not slow
    pipe().predict(Image.new("RGB", (1280, 960), (128, 128, 128)))
    yield


app = FastAPI(title="Khota sidewalk-defect API", version=__version__, lifespan=lifespan)


@app.get("/health")
def health():
    p = pipe()
    return {"status": "ok", "api_version": __version__, "model_version": p.model_version,
            "detectors": len(p.detectors), "classifiers": len(p.classifiers), "device": str(p.device)}


@app.post("/predict")
def predict(image: UploadFile = File(...)):
    data = image.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"image larger than {MAX_BYTES // (1024 * 1024)} MB")
    t = time.perf_counter()
    try:
        with _lock:
            res = pipe().predict(data)
    except Exception as e:  # noqa: BLE001  (not an image, corrupt file, ...)
        raise HTTPException(400, f"could not process image: {e}") from e
    res["image"] = image.filename
    res["processing_ms"] = round((time.perf_counter() - t) * 1000)
    return res
