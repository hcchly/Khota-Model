# Khota model — app integration guide

Model version **v2-2026-11-22**. This package contains everything needed to run the Khota sidewalk-defect model as a
service for the app: the inference code, an HTTP API, a command-line tool and API examples. The weight files are distributed separately.

The model can be used in three ways, all from the same files:

| Use | API endpoint | Python class | Command line |
|---|---|---|---|
| **Integrated** (detection + severity) | `POST /predict` | `KhotaPipeline` | `python predict.py photo.jpg` |
| **Detection model only** | `POST /detect` | `KhotaDetector` | `python predict.py photo.jpg --mode detect` |
| **Severity model only** | `POST /severity` | `KhotaSeverity` | `python predict.py photo.jpg --mode severity --boxes "[[x0,y0,x1,y1]]"` |

`/predict` and its response are unchanged from API 1.0.0; `/detect` and `/severity` were added in API 1.1.0.

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

Internally, the photo is downscaled to 1280 px. An **RF-DETR** detector (Roboflow's real-time detection
transformer with a DINOv2 backbone, fine-tuned on the Khota photos; run on the photo and its mirror image) and seven
YOLO detectors find the defects, and their boxes are fused (RF-DETR weighted 4×). Each defect is then cropped from
the full-resolution photo and rated by six ResNet-18 severity classifiers.

The two stages are also usable on their own:

* **Detection model:** photo in, boxes with type and confidence out. Uses `detector_1.pt` through `detector_7.pt`, `detector_8.pth` and `pipeline.json`.
* **Severity model:** photo plus boxes in, severity per box out. Each box is expanded by 20% on every side and cut
  from the full-resolution photo, exactly as in training. A single image that is already a crop around one
  defect is also accepted. Needs only `classifier_*.pt` and `pipeline.json`.

## 2. Package contents

| Path | What |
|---|---|
| `khota/` | Python package: `pipeline.py` (`KhotaPipeline`, `KhotaDetector`, `KhotaSeverity`), `ensemble.py` (box fusion), `classifier.py`, `crops.py`, `api.py` (HTTP API) |
| `weights/` | `detector_1..7.pt` (YOLO), `detector_8.pth` (RF-DETR), `classifier_1..6.pt`, `pipeline.json` (thresholds, file list, model version) |
| `predict.py` | command-line tool |
| `requirements.txt` | dependencies |
| `examples/` | `sample_photo.jpg` and its real responses (`sample_response.json` for `/predict`, `sample_response_detect.json`, `sample_response_severity.json`), `python_api_example.py`, `flutter_api_example.dart` |

All weights for this package are included in `weights/`. See `weights/README.md` for the file list and checksum instructions.

## 3. Requirements and installation

* Python 3.11 (tested), Windows or Linux.
* About 2 GB RAM. A GPU is optional.
* Measured time for one 4032 px phone photo on an NVIDIA RTX 3070 laptop GPU: about **1.0 s**
  (`processing_ms` in the sample response). The CPU works too, but is several times slower.
* Models load once at start-up (a few seconds).

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
# PyTorch first, ONE of:
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu     # Linux/Windows CPU wheels
# macOS: pip install torch==2.6.0 torchvision==0.21.0
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
{"status": "ok", "api_version": "1.2.0", "model_version": "v2-2026-11-22", "detectors": 8, "classifiers": 6, "device": "cuda"}
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
  "model_version": "v2-2026-11-22",
  "width": 4032,
  "height": 3024,
  "result": "Defects found",
  "defects": [
    {
      "box_xyxy": [
        215.4,
        1914.6,
        575.0,
        2116.2
      ],
      "box_xywhn": [
        0.098017,
        0.666473,
        0.08917,
        0.066678
      ],
      "class_id": 1,
      "class": "Surface Damage",
      "confidence": 0.3517,
      "severity": "Low–Medium Severity",
      "severity_label": "LM",
      "p_high": 0.0816
    },
    {
      "box_xyxy": [
        2704.5,
        1523.2,
        3826.1,
        2365.2
      ],
      "box_xywhn": [
        0.809853,
        0.642922,
        0.27818,
        0.278416
      ],
      "class_id": 1,
      "class": "Surface Damage",
      "confidence": 0.3448,
      "severity": "Low–Medium Severity",
      "severity_label": "LM",
      "p_high": 0.0387
    },
    {
      "box_xyxy": [
        1205.3,
        934.1,
        1522.3,
        1048.1
      ],
      "box_xywhn": [
        0.33824,
        0.327737,
        0.078608,
        0.037686
      ],
      "class_id": 1,
      "class": "Surface Damage",
      "confidence": 0.2917,
      "severity": "Low–Medium Severity",
      "severity_label": "LM",
      "p_high": 0.0159
    },
    {
      "box_xyxy": [
        1540.3,
        967.3,
        2659.6,
        1422.2
      ],
      "box_xywhn": [
        0.52083,
        0.395097,
        0.277599,
        0.150422
      ],
      "class_id": 0,
      "class": "Curb Ramp Issue",
      "confidence": 0.2887,
      "severity": "High Severity",
      "severity_label": "HIGH",
      "p_high": 0.8055
    }
  ],
  "thresholds": {
    "detection_conf": 0.15,
    "high_severity": 0.29
  },
  "processing_ms": 995
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
| `confidence` | detection confidence 0–1 |
| `severity`, `severity_label` | `"High Severity"` / `"HIGH"` or `"Low–Medium Severity"` / `"LM"` |
| `p_high` | probability that the defect is High severity (0–1); `HIGH` when `p_high` ≥ `thresholds.high_severity` |
| `thresholds` | the thresholds used (from `weights/pipeline.json`) |
| `processing_ms` | server processing time |

**Errors:** `400` empty or unreadable image · `413` image too large · `422` no `image` field in the request.
The same errors apply to `/detect` and `/severity`.

**Drawing boxes in the app.** Coordinates refer to the photo *as displayed with its EXIF orientation applied*
(`width` × `height`). To draw on a scaled view, multiply by `shown_width / width`. Alternatively, use
`box_xywhn`, which is independent of the photo size.

### `POST /detect` (detection model only)

Same request as `/predict`. The response has the same fields except the severity fields (`severity`,
`severity_label`, `p_high`, `thresholds.high_severity`). Real output for `examples/sample_photo.jpg`:

```bash
curl -F "image=@photo.jpg" http://localhost:8000/detect
```

```json
{
  "image": "sample_photo.jpg",
  "model_version": "v2-2026-11-22",
  "width": 4032,
  "height": 3024,
  "result": "Defects found",
  "defects": [
    {
      "box_xyxy": [
        215.4,
        1914.6,
        575.0,
        2116.2
      ],
      "box_xywhn": [
        0.098017,
        0.666473,
        0.08917,
        0.066678
      ],
      "class_id": 1,
      "class": "Surface Damage",
      "confidence": 0.3517
    },
    {
      "box_xyxy": [
        2704.5,
        1523.2,
        3826.1,
        2365.2
      ],
      "box_xywhn": [
        0.809853,
        0.642922,
        0.27818,
        0.278416
      ],
      "class_id": 1,
      "class": "Surface Damage",
      "confidence": 0.3448
    },
    {
      "box_xyxy": [
        1205.3,
        934.1,
        1522.3,
        1048.1
      ],
      "box_xywhn": [
        0.33824,
        0.327737,
        0.078608,
        0.037686
      ],
      "class_id": 1,
      "class": "Surface Damage",
      "confidence": 0.2917
    },
    {
      "box_xyxy": [
        1540.3,
        967.3,
        2659.6,
        1422.2
      ],
      "box_xywhn": [
        0.52083,
        0.395097,
        0.277599,
        0.150422
      ],
      "class_id": 0,
      "class": "Curb Ramp Issue",
      "confidence": 0.2887
    }
  ],
  "thresholds": {
    "detection_conf": 0.15
  },
  "processing_ms": 865
}
```

### `POST /severity` (severity model only)

`multipart/form-data` with the file field **`image`** and an optional text field **`boxes`**:

* **With `boxes`:** a JSON list of `[x0, y0, x1, y1]` boxes in pixels of the photo, in the same coordinate space
  as `box_xyxy` from `/detect` (after EXIF rotation). Each box is rated. Send the original full photo, not a resized
  one, so the crops match training.
* **Without `boxes`:** the image is treated as one crop around a single defect, with about 20% margin on every side.
  The response then has one entry with `box_xyxy: null`.

```bash
curl -F "image=@photo.jpg" -F 'boxes=[[215.4, 1914.6, 575.0, 2116.2], [2704.5, 1523.2, 3826.1, 2365.2], [1205.3, 934.1, 1522.3, 1048.1], [1540.3, 967.3, 2659.6, 1422.2]]' http://localhost:8000/severity
```

```json
{
  "image": "sample_photo.jpg",
  "model_version": "v2-2026-11-22",
  "width": 4032,
  "height": 3024,
  "mode": "photo+boxes",
  "defects": [
    {
      "box_xyxy": [
        215.4,
        1914.6,
        575.0,
        2116.2
      ],
      "box_xywhn": [
        0.098016,
        0.666468,
        0.089187,
        0.066667
      ],
      "severity": "Low–Medium Severity",
      "severity_label": "LM",
      "p_high": 0.0895
    },
    {
      "box_xyxy": [
        2704.5,
        1523.2,
        3826.1,
        2365.2
      ],
      "box_xywhn": [
        0.809846,
        0.642923,
        0.278175,
        0.278439
      ],
      "severity": "Low–Medium Severity",
      "severity_label": "LM",
      "p_high": 0.0387
    },
    {
      "box_xyxy": [
        1205.3,
        934.1,
        1522.3,
        1048.1
      ],
      "box_xywhn": [
        0.338244,
        0.327745,
        0.078621,
        0.037698
      ],
      "severity": "Low–Medium Severity",
      "severity_label": "LM",
      "p_high": 0.0159
    },
    {
      "box_xyxy": [
        1540.3,
        967.3,
        2659.6,
        1422.2
      ],
      "box_xywhn": [
        0.520821,
        0.395089,
        0.277604,
        0.15043
      ],
      "severity": "High Severity",
      "severity_label": "HIGH",
      "p_high": 0.8055
    }
  ],
  "thresholds": {
    "high_severity": 0.29
  },
  "processing_ms": 297
}
```

`mode` is `"photo+boxes"` or `"crop"`. A malformed `boxes` value, or a box outside the photo, returns `400` with
the reason.

## 6. API examples

* **Flutter:** `examples/flutter_api_example.dart`. It uses the `http` package to send the photo as multipart field
  `image`, and shows how to scale the boxes.
* **Python:** `examples/python_api_example.py`.
* **Python without the HTTP API:**

```python
import sys; sys.path.insert(0, "path/to/this/folder")
from khota.pipeline import KhotaPipeline, KhotaDetector, KhotaSeverity
W = "path/to/this/folder/weights"

pipe = KhotaPipeline.from_dir(W)          # integrated; load once, reuse
result = pipe.predict("photo.jpg")        # also accepts bytes or a PIL image

det = KhotaDetector.from_dir(W)           # detection model only
boxes = [d["box_xyxy"] for d in det.predict("photo.jpg")["defects"]]

sev = KhotaSeverity.from_dir(W)           # severity model only
sev.predict("photo.jpg", boxes=boxes)     # boxes in photo pixels
sev.predict("defect_crop.jpg")            # or one crop around a single defect
```

Each class loads only its own model files, so a detection-only or severity-only service uses less memory.

## 7. Updating to a newer model (no code change)

1. Stop the API server.
2. Rename the current `weights/` folder (e.g. to `weights_v2-2026-11-22/`). This is your rollback.
3. Copy the new `weights/` folder in its place.
4. Start the server. Check with `GET /health` that `model_version` shows the new version.

The request, the response fields and the API stay the same, so the app needs no change. The number of model
files and the thresholds may change between versions; the code reads them from `weights/pipeline.json`. If a
future update ever needs a code change, it will be delivered as a complete new package and stated explicitly.

## 8. Settings you may tune (`weights/pipeline.json`)

| Key | Current | Effect |
|---|---|---|
| `det_conf` | 0.15 | detection confidence threshold. Higher = fewer false alarms but more missed defects; lower = the reverse. |
| `high_threshold` | 0.29 | High-severity threshold on `p_high`. Higher = fewer, surer High results; lower = catches more High defects. |

Leave every other key unchanged; they must match the model files. A new model package comes with its own values.

## 9. About this version (v2-2026-11-22)

**Test split (scored once, with the thresholds below):** mAP@0.5 0.375 (Curb Ramp 0.557, Surface Damage 0.193); severity Macro-F1 0.761, High recall 0.708; 89% of the photos with a defect flagged, 22% false alarms on photos without one. Details: `results/test_metrics_v2.json` in the repository.

**Speed and memory:** CPU only (11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz) 1.8 s per photo (median), peak RAM 1.7 GB; GPU (NVIDIA GeForce RTX 3070 Laptop GPU) 0.6 s per photo. Full pipeline on original photos (`results/benchmark_v2_cpu.json`).

**Licenses:** `docs/LICENSES.md` in the repository. The YOLO detectors run on Ultralytics, which is AGPL-3.0; RF-DETR is Apache-2.0.

Measured on the validation photos (locations never seen in training), with the same AP code as before:

| | previous version (v1, 7 YOLO detectors) | **this version (v2, RF-DETR + 7 YOLO)** |
|---|---|---|
| mAP@0.5 Curb Ramp Issue | 0.611 | **0.659** |
| mAP@0.5 Surface Damage | 0.147 | **0.233** |
| mAP@0.5, both classes | 0.379 | **0.446** |

The severity classifiers are unchanged. Test-split results for this version are reported in the project report.

**Updating from v1 (API 1.0.0 / 1.1.0):** this version needs the code in this package (API 1.2.0, which adds
RF-DETR support) and one new dependency, `rfdetr` (run `pip install -r requirements.txt` again). Replace the
`khota/` folder, `predict.py` and the `weights/` folder. The request, the response fields and the API stay the same,
so the app needs no change.
