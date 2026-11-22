"""HTTP API around the Khota models, for the mobile app.

Run (from the folder that contains khota/ and weights/):
    python -m uvicorn khota.api:app --host 0.0.0.0 --port 8000
Call:
    curl -F "image=@photo.jpg" http://localhost:8000/predict                    # integrated: detection + severity
    curl -F "image=@photo.jpg" http://localhost:8000/detect                     # detection model only
    curl -F "image=@photo.jpg" -F "boxes=[[100,200,400,500]]" http://localhost:8000/severity   # severity model only

Environment variables
    KHOTA_WEIGHTS        weights folder (default: weights)
    KHOTA_DEVICE         "cpu" to force CPU; default: GPU if available
    KHOTA_MAX_UPLOAD_MB  largest accepted upload in MB (default: 25)
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

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


def _read(image: UploadFile) -> bytes:
    data = image.file.read(MAX_BYTES + 1)
    if not data:
        raise HTTPException(400, "empty upload")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"image larger than {MAX_BYTES // (1024 * 1024)} MB")
    return data


def _run(fn, image: UploadFile, *args) -> dict:
    data = _read(image)
    t = time.perf_counter()
    try:
        with _lock:
            res = fn(data, *args)
    except UnidentifiedImageError as e:
        raise HTTPException(400, "the upload is not a readable image (JPEG or PNG expected)") from e
    except ValueError as e:  # bad box
        raise HTTPException(400, str(e)) from e
    except Exception as e:  # noqa: BLE001  (not an image, corrupt file, ...)
        raise HTTPException(400, f"could not process image: {e}") from e
    res["image"] = image.filename
    res["processing_ms"] = round((time.perf_counter() - t) * 1000)
    return res


@app.get("/health")
def health():
    p = pipe()
    return {"status": "ok", "api_version": __version__, "model_version": p.model_version,
            "detectors": len(p.detectors), "classifiers": len(p.classifiers), "device": str(p.device)}


@app.post("/predict")
def predict(image: UploadFile = File(...)):
    """Integrated model: every defect with its box, type and severity."""
    return _run(pipe().predict, image)


@app.post("/detect")
def detect(image: UploadFile = File(...)):
    """Detection model only: every defect with its box and type, no severity."""
    return _run(pipe().predict_detection, image)


@app.post("/severity")
def severity(image: UploadFile = File(...), boxes: str | None = Form(None)):
    """Severity model only. With `boxes` (JSON list of [x0, y0, x1, y1] in photo pixels): severity of each box,
    cropped from the photo as in training. Without `boxes`: the image is one crop around a single defect."""
    parsed = None
    if boxes is not None and boxes.strip():
        try:
            parsed = json.loads(boxes)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"boxes is not valid JSON: {e}") from e
        if not isinstance(parsed, list) or not parsed:
            raise HTTPException(400, "boxes must be a non-empty JSON list of [x0, y0, x1, y1]")
    return _run(pipe().predict_severity, image, parsed)
