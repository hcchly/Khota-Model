"""Minimal Python client for the Khota API."""
import requests

API = "http://localhost:8000"

print(requests.get(f"{API}/health", timeout=10).json())
with open("sample_photo.jpg", "rb") as f:
    r = requests.post(f"{API}/predict", files={"image": ("sample_photo.jpg", f, "image/jpeg")}, timeout=120)
r.raise_for_status()
result = r.json()
print(result["model_version"], result["result"])
for d in result["defects"]:
    print(d["class"], d["severity"], d["box_xyxy"], d["confidence"])
