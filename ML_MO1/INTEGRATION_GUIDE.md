# Khota model — app integration guide

Model version **v1-2026-11-21**. This package contains everything needed to run the Khota sidewalk-defect model as a
service for the app: the code, every model file, an HTTP API, a command-line tool and client examples.

**Updating later is a folder swap.** All model-specific settings live in `weights/pipeline.json`. When a newer
model is delivered, only the `weights/` folder is replaced; the code, the API and the response format stay the
same (section 7).

---

## 1. What the model does

For each sidewalk photo it returns every defect found, with:

* **box**: where the defect is in the photo;
* **class**: `Curb Ramp Issue` or `Surface Damage`;
* **severity**: `High Severity` or `Low–Medium Severity`.

A photo with no defect returns `"result": "No Target Defect"` and an empty `defects` list.

Internally, the photo is downscaled to 1280 px. Seven YOLO detectors find the defects and their boxes are fused.
Each defect is then cropped from the full-resolution photo and rated by six ResNet-18 severity classifiers.

## 2. Package contents

| Path | What |
|---|---|
| `khota/` | Python package: `pipeline.py` (end to end), `ensemble.py` (box fusion), `classifier.py`, `crops.py`, `api.py` (HTTP API) |
| `weights/` | `detector_1..7.pt`, `classifier_1..6.pt`, `pipeline.json` (thresholds, file list, model version) |
| `predict.py` | command-line tool |
| `requirements.txt` | dependencies |
| `examples/` | `sample_photo.jpg` and its real API response (`sample_response.json`), `python_client.py`, `flutter_client.dart` |

## 3. Requirements and installation

* Python 3.11 (tested), Windows or Linux.
* About 2 GB RAM. A GPU is optional.
* Measured time for one 4032 px phone photo: about **0.45 s on an NVIDIA RTX 3070 laptop GPU** and about
  **1.7 s on that laptop's CPU**.
* Models load once at start-up (a few seconds).

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
# PyTorch first, ONE of:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu     # CPU only
pip install -r requirements.txt
```

Quick check:

```bash
python predict.py examples/sample_photo.jpg
```

## 4. Run the API

From this folder (the one that contains `khota/` and `weights/`):

```bash
python -m uvicorn khota.api:app --host 0.0.0.0 --port 8000
```

| Environment variable | Default | Meaning |
|---|---|---|
| `KHOTA_WEIGHTS` | `weights` | weights folder |
| `KHOTA_DEVICE` | GPU if available | set to `cpu` to force CPU |
| `KHOTA_MAX_UPLOAD_MB` | `25` | largest accepted photo |

* **Concurrency:** the server handles one prediction at a time and queues the rest.
* **More throughput:** run several server instances behind a load balancer.
* **Security:** the API has no authentication. Keep it behind your backend or a reverse proxy that checks an API
  key, and do not expose it directly to the internet.

## 5. API reference

### `GET /health`

```json
{"status": "ok", "api_version": "1.0.0", "model_version": "v1-2026-11-21", "detectors": 7, "classifiers": 6, "device": "cuda"}
```

### `POST /predict`

`multipart/form-data` with one file field named **`image`** (JPEG or PNG; the original phone photo is best).

```bash
curl -F "image=@photo.jpg" http://localhost:8000/predict
```

Response (the real output for `examples/sample_photo.jpg`):

```json
{
  "image": "sample_photo.jpg",
  "model_version": "v1-2026-11-21",
  "width": 2268,
  "height": 4032,
  "result": "Defects found",
  "defects": [
    {
      "box_xyxy": [
        1324.0,
        1939.7,
        1733.7,
        2304.5
      ],
      "box_xywhn": [
        0.674098,
        0.526311,
        0.180622,
        0.09047
      ],
      "class_id": 0,
      "class": "Curb Ramp Issue",
      "confidence": 0.3129,
      "severity": "High Severity",
      "severity_label": "HIGH",
      "p_high": 0.993
    },
    {
      "box_xyxy": [
        497.4,
        2190.9,
        719.0,
        2475.7
      ],
      "box_xywhn": [
        0.26817,
        0.578704,
        0.097734,
        0.070628
      ],
      "class_id": 1,
      "class": "Surface Damage",
      "confidence": 0.2377,
      "severity": "High Severity",
      "severity_label": "HIGH",
      "p_high": 0.9999
    }
  ],
  "thresholds": {
    "detection_conf": 0.12,
    "high_severity": 0.29
  },
  "processing_ms": 450
}
```

| Field | Meaning |
|---|---|
| `model_version` | version of the weights that produced the result |
| `width`, `height` | photo size in pixels after EXIF rotation, i.e. the coordinate space of the boxes |
| `result` | `"Defects found"` or `"No Target Defect"` |
| `defects[]` | one entry per defect, sorted by `confidence`, highest first |
| `box_xyxy` | `[x0, y0, x1, y1]` in pixels of the photo (top-left origin) |
| `box_xywhn` | `[x_center, y_center, width, height]` as fractions 0–1 of the photo |
| `class_id`, `class` | `0` = `Curb Ramp Issue`, `1` = `Surface Damage` |
| `confidence` | detection confidence 0–1 (fused over the 7 detectors) |
| `severity`, `severity_label` | `"High Severity"` / `"HIGH"` or `"Low–Medium Severity"` / `"LM"` |
| `p_high` | probability that the defect is High severity (0–1); `HIGH` when `p_high` ≥ `thresholds.high_severity` |
| `thresholds` | the thresholds used (from `weights/pipeline.json`) |
| `processing_ms` | server processing time |

**Errors:** `400` empty or unreadable image · `413` image too large · `422` no `image` field in the request.

**Drawing boxes in the app.** Coordinates refer to the photo *as displayed with its EXIF orientation applied*
(`width` × `height`). To draw on a scaled view, multiply by `shown_width / width`. Alternatively, use
`box_xywhn`, which is independent of the photo size.

## 6. Client examples

* **Flutter:** `examples/flutter_client.dart`. It uses the `http` package to send the photo as multipart field
  `image`, and shows how to scale the boxes.
* **Python:** `examples/python_client.py`.
* **Python without the HTTP API:**

```python
import sys; sys.path.insert(0, "path/to/this/folder")
from khota.pipeline import KhotaPipeline
pipe = KhotaPipeline.from_dir("path/to/this/folder/weights")   # load once, reuse
result = pipe.predict("photo.jpg")                              # also accepts bytes or a PIL image
```

## 7. Updating to a newer model (no code change)

1. Stop the API server.
2. Rename the current `weights/` folder (e.g. to `weights_v1-2026-11-21/`). This is your rollback.
3. Copy the new `weights/` folder in its place.
4. Start the server. Check with `GET /health` that `model_version` shows the new version.

The request, the response fields and the API stay the same, so the app needs no change. The number of model
files and the thresholds may change between versions; the code reads them from `weights/pipeline.json`. If a
future update ever needs a code change, it will be delivered as a complete new package and stated explicitly.

## 8. Settings you may tune (`weights/pipeline.json`)

| Key | Current | Effect |
|---|---|---|
| `det_conf` | 0.12 | detection confidence threshold. Higher = fewer false alarms but more missed defects; lower = the reverse. |
| `high_threshold` | 0.29 | High-severity threshold on `p_high`. Higher = fewer, surer High results; lower = catches more High defects. |

Leave every other key unchanged; they must match the model files. A new model package comes with its own values.

## 9. About this version (v1-2026-11-21)

Measured on held-out test photos from locations never seen in training:

* **Curb Ramp Issue:** mAP@0.5 0.61.
* **Surface Damage:** mAP@0.5 0.17.
* **High severity:** recall 0.71 (17 of 24 High defects found); Macro-F1 0.75.
* **Photo level:** 56% of photos with a labelled defect get at least one detection; 9% of photos without one
  raise a false alarm.

Surface Damage detection is the weak point of this version. The next version is trained on reviewed Surface
Damage labels.
