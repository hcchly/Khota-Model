"""Minimal Python example for the Khota API."""
import json
from pathlib import Path

import requests

API = "http://localhost:8000"
PHOTO = Path(__file__).with_name("sample_photo.jpg")

print(requests.get(f"{API}/health", timeout=10).json())
with PHOTO.open("rb") as f:
    r = requests.post(f"{API}/predict", files={"image": (PHOTO.name, f, "image/jpeg")}, timeout=120)
r.raise_for_status()
result = r.json()
print(result["model_version"], result["result"])
for d in result["defects"]:
    print(d["class"], d["severity"], d["box_xyxy"], d["confidence"])

# detection model only (no severity)
with PHOTO.open("rb") as f:
    det = requests.post(f"{API}/detect", files={"image": (PHOTO.name, f, "image/jpeg")}, timeout=120).json()

# severity model only, for boxes you already have (pixels of the photo, e.g. det["defects"][i]["box_xyxy"])
boxes = [d["box_xyxy"] for d in det["defects"]]
if boxes:
    with PHOTO.open("rb") as f:
        sev = requests.post(f"{API}/severity", files={"image": (PHOTO.name, f, "image/jpeg")},
                            data={"boxes": json.dumps(boxes)}, timeout=120).json()
    for d in sev["defects"]:
        print(d["box_xyxy"], d["severity"], d["p_high"])
