"""Khota end-to-end inference: image -> detector -> 20%-expanded crops -> severity classifier -> JSON.

    from khota.pipeline import KhotaPipeline
    pipe = KhotaPipeline.from_dir("weights")          # detector(s), classifier(s) and pipeline.json in weights/
    result = pipe.predict("photo.jpg")                # dict, JSON-serialisable

Output format
{
  "image": "photo.jpg", "width": 4032, "height": 3024,
  "result": "Defects found" | "No Target Defect",
  "defects": [
    {"box_xyxy": [x0, y0, x1, y1],            # pixels in the (EXIF-corrected) original image
     "box_xywhn": [xc, yc, w, h],             # normalised YOLO format
     "class_id": 1, "class": "Surface Damage", "confidence": 0.81,
     "severity": "High Severity", "severity_label": "HIGH", "p_high": 0.73}
  ],
  "thresholds": {"detection_conf": 0.25, "high_severity": 0.5}
}
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .classifier import SEVERITIES, SEVERITY_NAMES, build_model, eval_transform
from .crops import crop_defect
from .ensemble import predict_members, wbf

CLASS_NAMES = {0: "Curb Ramp Issue", 1: "Surface Damage"}


def load_image(src) -> tuple[Image.Image, str]:
    """Accepts a path, raw bytes, a file-like object or a PIL image. Applies EXIF orientation
    (the training images were auto-oriented by Roboflow, phone photos usually carry EXIF rotation)."""
    if isinstance(src, Image.Image):
        im, name = src, "<PIL image>"
    elif isinstance(src, (bytes, bytearray)):
        im, name = Image.open(io.BytesIO(src)), "<bytes>"
    elif hasattr(src, "read"):
        im, name = Image.open(src), getattr(src, "name", "<stream>")
    else:
        im, name = Image.open(src), Path(src).name
    return ImageOps.exif_transpose(im).convert("RGB"), name


class KhotaPipeline:
    def __init__(self, detector_weights: str | Path | list, classifier_weights: str | Path | list,
                 det_conf: float = 0.25, high_threshold: float | None = None, iou: float = 0.6,
                 imgsz: int = 640, tta: bool = False, device: str | None = None, det_input_max: int | None = 1280,
                 wbf_iou: float = 0.55, member_conf: float = 0.001, model_version: str = "unversioned"):
        """detector_weights: one YOLO checkpoint, or a list whose predictions are fused with weighted box fusion
        (each member at confidence >= member_conf, fused boxes kept if their fused confidence >= det_conf).
        classifier_weights: one checkpoint, or a list whose High probabilities are averaged (seed ensemble).
        tta: also classify the horizontally flipped crop and average.
        det_input_max: before detection the photo is downscaled (LANCZOS) to this long side, exactly as the detector's
        training images were prepared (scripts/data/05_build_datasets.py); feeding a 4032 px photo directly lets YOLO
        shrink it ~6x with a linear filter, which costs recall. Boxes are returned in original-photo pixels and the
        severity crops are cut from the full-resolution photo, as in training."""
        from ultralytics import YOLO  # imported lazily so the classifier can be used without it

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dets = detector_weights if isinstance(detector_weights, (list, tuple)) else [detector_weights]
        self.detectors = [YOLO(str(p)) for p in dets]
        self.detector = self.detectors[0]
        self.wbf_iou, self.member_conf = wbf_iou, member_conf
        paths = classifier_weights if isinstance(classifier_weights, (list, tuple)) else [classifier_weights]
        self.classifiers = []
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = build_model(ck["arch"], pretrained=False, dropout=ck.get("dropout", 0.0))
            m.load_state_dict(ck["model"])
            self.classifiers.append(m.eval().to(self.device))
        self.tf = eval_transform()
        self.tta = tta
        self.det_conf = det_conf
        self.high_threshold = high_threshold if high_threshold is not None else float(ck.get("high_threshold", 0.5))
        self.iou = iou
        self.imgsz = imgsz
        self.det_input_max = det_input_max
        self.model_version = model_version

    @classmethod
    def from_dir(cls, weights_dir: str | Path = "weights", **overrides) -> "KhotaPipeline":
        """weights_dir holds detector.pt, the classifier checkpoint(s) and pipeline.json
        ({"det_conf", "high_threshold", "imgsz", "iou", "tta", "classifiers": ["classifier.pt", ...]})."""
        d = Path(weights_dir)
        cfg = json.loads((d / "pipeline.json").read_text()) if (d / "pipeline.json").exists() else {}
        cfg.update(overrides)
        clf = [d / f for f in cfg.pop("classifiers", ["classifier.pt"])]
        det = [d / f for f in cfg.pop("detectors", ["detector.pt"])]
        cfg = {k: v for k, v in cfg.items() if k in ("det_conf", "high_threshold", "iou", "imgsz", "tta", "device",
                                                     "det_input_max", "wbf_iou", "member_conf", "model_version")}
        return cls(det if len(det) > 1 else det[0], clf, **cfg)

    @torch.no_grad()
    def classify_crops(self, crops: list[Image.Image]) -> list[float]:
        if not crops:
            return []
        x = torch.stack([self.tf(c) for c in crops]).to(self.device)
        k = SEVERITIES.index("HIGH")
        p, n = 0.0, 0
        for m in self.classifiers:
            for xx in ([x, torch.flip(x, dims=[3])] if self.tta else [x]):
                p = p + torch.softmax(m(xx).float(), 1)[:, k]
                n += 1
        return (p / n).cpu().tolist()

    def detect(self, image) -> tuple[list, list, list]:
        """Boxes (normalised xyxy), confidences and class ids for one (already downscaled) image."""
        dev = 0 if self.device.type == "cuda" else "cpu"
        if len(self.detectors) == 1:
            r = self.detector.predict(image, imgsz=self.imgsz, conf=self.det_conf, iou=self.iou, device=dev,
                                      verbose=False)[0]
            return (r.boxes.xyxyn.cpu().tolist(), r.boxes.conf.cpu().tolist(),
                    r.boxes.cls.cpu().int().tolist())
        fb, fs, fc = wbf(predict_members(self.detectors, image, self.imgsz, self.member_conf, self.iou, dev),
                         self.wbf_iou)
        keep = fs >= self.det_conf
        return fb[keep].tolist(), fs[keep].tolist(), fc[keep].tolist()

    def predict(self, src) -> dict:
        im, name = load_image(src)
        W, H = im.size
        det_im = im
        if self.det_input_max and max(W, H) > self.det_input_max:
            s = self.det_input_max / max(W, H)
            det_im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
        xyxyn, scores, classes = self.detect(det_im)
        xyxy = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in xyxyn]  # original-photo pixels
        crops = [crop_defect(im, tuple(b), size=None) for b in xyxy]   # resize to 224 happens in self.tf
        p_high = self.classify_crops(crops)
        defects = []
        for b, cls_id, conf, ph in zip(xyxy, classes, scores, p_high):
            sev = "HIGH" if ph >= self.high_threshold else "LM"
            x0, y0, x1, y1 = b
            defects.append({
                "box_xyxy": [round(v, 1) for v in b],
                "box_xywhn": [round((x0 + x1) / 2 / W, 6), round((y0 + y1) / 2 / H, 6),
                              round((x1 - x0) / W, 6), round((y1 - y0) / H, 6)],
                "class_id": int(cls_id), "class": CLASS_NAMES[int(cls_id)], "confidence": round(float(conf), 4),
                "severity": SEVERITY_NAMES[sev], "severity_label": sev, "p_high": round(float(ph), 4),
            })
        defects.sort(key=lambda d: -d["confidence"])
        return {"image": name, "model_version": self.model_version, "width": W, "height": H,
                "result": "Defects found" if defects else "No Target Defect",
                "defects": defects,
                "thresholds": {"detection_conf": self.det_conf, "high_severity": self.high_threshold}}
