"""Khota inference: the integrated pipeline and the two models on their own.

    from khota.pipeline import KhotaPipeline, KhotaDetector, KhotaSeverity
    pipe = KhotaPipeline.from_dir("weights")      # detection + severity (integrated)
    result = pipe.predict("photo.jpg")            # dict, JSON-serialisable

    det = KhotaDetector.from_dir("weights")       # detection only: boxes + defect type, no severity
    det.predict("photo.jpg")

    sev = KhotaSeverity.from_dir("weights")       # severity only
    sev.predict("photo.jpg", boxes=[[x0, y0, x1, y1]])   # boxes in photo pixels; crops are cut as in training
    sev.predict("defect_crop.jpg")                # or one image that is already a crop around a single defect

All three read the same weights/ folder (pipeline.json). The detector needs only detector_*.pt, the severity model
only classifier_*.pt.

Output format of KhotaPipeline.predict
{
  "image": "photo.jpg", "model_version": "...", "width": 4032, "height": 3024,
  "result": "Defects found" | "No Target Defect",
  "defects": [
    {"box_xyxy": [x0, y0, x1, y1],            # pixels in the (EXIF-corrected) original image
     "box_xywhn": [xc, yc, w, h],             # normalised YOLO format
     "class_id": 1, "class": "Surface Damage", "confidence": 0.81,
     "severity": "High Severity", "severity_label": "HIGH", "p_high": 0.73}
  ],
  "thresholds": {"detection_conf": 0.25, "high_severity": 0.5}
}
KhotaDetector.predict returns the same without the severity fields; KhotaSeverity.predict returns the same
"defects" list with only the box and severity fields.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from .classifier import SEVERITIES, SEVERITY_NAMES, build_model, eval_transform
from .crops import crop_defect
from .ensemble import RFDETRMember, class_map, merge_subclasses, predict_members, wbf

CLASS_NAMES = {0: "Curb Ramp Issue", 1: "Surface Damage"}
_DET_KEYS = ("det_conf", "iou", "imgsz", "device", "det_input_max", "wbf_iou", "member_conf", "model_version",
             "rfdetr")
_SEV_KEYS = ("high_threshold", "tta", "device", "model_version")


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


def _read_dir(weights_dir, overrides: dict) -> tuple[list, list, dict]:
    """weights_dir holds the model files and pipeline.json; returns detector paths, classifier paths, settings."""
    d = Path(weights_dir)
    cfg = json.loads((d / "pipeline.json").read_text()) if (d / "pipeline.json").exists() else {}
    cfg.update(overrides)
    clf = [d / f for f in cfg.pop("classifiers", ["classifier.pt"])]
    det = [d / f for f in cfg.pop("detectors", ["detector.pt"])]
    return det, clf, cfg


def _pick(cfg: dict, keys) -> dict:
    return {k: v for k, v in cfg.items() if k in keys}


def _box_fields(b, W: int, H: int) -> dict:
    x0, y0, x1, y1 = b
    return {"box_xyxy": [round(float(v), 1) for v in b],
            "box_xywhn": [round((x0 + x1) / 2 / W, 6), round((y0 + y1) / 2 / H, 6),
                          round((x1 - x0) / W, 6), round((y1 - y0) / H, 6)]}


class _DetectorPart:
    """Detector ensemble: 1..n YOLO models, fused with weighted box fusion."""

    def _init_detector(self, detector_weights, det_conf, iou, imgsz, det_input_max, wbf_iou, member_conf,
                       rfdetr: dict | None = None):
        from ultralytics import YOLO  # imported lazily so the severity model can be used without it

        dets = detector_weights if isinstance(detector_weights, (list, tuple)) else [detector_weights]
        rfdetr = rfdetr or {}  # file name -> {"size": ..., "names": [...]} for RF-DETR members (pipeline.json)
        dev = "cuda" if self.device.type == "cuda" else "cpu"
        self.detectors = [RFDETRMember(p, rfdetr[Path(p).name]["size"], rfdetr[Path(p).name]["names"], dev,
                                       rfdetr[Path(p).name].get("flip", False), rfdetr[Path(p).name].get("weight", 1))
                          if Path(p).name in rfdetr else YOLO(str(p)) for p in dets]
        self.detector = self.detectors[0]
        self.det_conf, self.iou, self.imgsz = det_conf, iou, imgsz
        self.det_input_max = det_input_max
        self.wbf_iou, self.member_conf = wbf_iou, member_conf

    def detect(self, image) -> tuple[list, list, list]:
        """Boxes (normalised xyxy), confidences and class ids for one (already downscaled) image."""
        dev = 0 if self.device.type == "cuda" else "cpu"
        if len(self.detectors) == 1 and hasattr(self.detector, "predict_passes"):  # single RF-DETR detector
            members = predict_members(self.detectors, image, self.imgsz, self.member_conf, self.iou, dev)
            fb, fs, fc = wbf(members, self.wbf_iou) if len(members) > 1 else members[0]
            keep = fs >= self.det_conf
            return fb[keep].tolist(), fs[keep].tolist(), fc[keep].tolist()
        if len(self.detectors) == 1:
            r = self.detector.predict(image, imgsz=self.imgsz, conf=self.det_conf, iou=self.iou, device=dev,
                                      verbose=False)[0]
            b, s, c = merge_subclasses(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy().astype(int), class_map(self.detector), self.iou)
            return b.tolist(), s.tolist(), c.tolist()
        fb, fs, fc = wbf(predict_members(self.detectors, image, self.imgsz, self.member_conf, self.iou, dev),
                         self.wbf_iou)
        keep = fs >= self.det_conf
        return fb[keep].tolist(), fs[keep].tolist(), fc[keep].tolist()

    def _detect_photo(self, im: Image.Image) -> tuple[list, list, list]:
        """Full-resolution photo -> boxes in photo pixels, confidences, class ids.
        The photo is first downscaled (LANCZOS) to det_input_max, exactly as the detector's training images were
        prepared; feeding a 4032 px photo directly lets YOLO shrink it ~6x with a linear filter, which costs recall."""
        W, H = im.size
        det_im = im
        if self.det_input_max and max(W, H) > self.det_input_max:
            s = self.det_input_max / max(W, H)
            det_im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
        xyxyn, scores, classes = self.detect(det_im)
        return [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in xyxyn], scores, classes

    def predict_detection(self, src) -> dict:
        """Detection only: every defect's box, type and confidence (no severity)."""
        im, name = load_image(src)
        W, H = im.size
        xyxy, scores, classes = self._detect_photo(im)
        defects = [{**_box_fields(b, W, H), "class_id": int(c), "class": CLASS_NAMES[int(c)],
                    "confidence": round(float(s), 4)} for b, s, c in zip(xyxy, scores, classes)]
        defects.sort(key=lambda d: -d["confidence"])
        return {"image": name, "model_version": self.model_version, "width": W, "height": H,
                "result": "Defects found" if defects else "No Target Defect", "defects": defects,
                "thresholds": {"detection_conf": self.det_conf}}


class _SeverityPart:
    """Severity ensemble: 1..n ResNet-18 classifiers whose High probabilities are averaged."""

    def _init_severity(self, classifier_weights, high_threshold, tta):
        paths = classifier_weights if isinstance(classifier_weights, (list, tuple)) else [classifier_weights]
        self.classifiers = []
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = build_model(ck["arch"], pretrained=False, dropout=ck.get("dropout", 0.0))
            m.load_state_dict(ck["model"])
            self.classifiers.append(m.eval().to(self.device))
        self.tf = eval_transform()
        self.tta = tta
        self.high_threshold = high_threshold if high_threshold is not None else float(ck.get("high_threshold", 0.5))

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

    def _severity_fields(self, p_high: float) -> dict:
        sev = "HIGH" if p_high >= self.high_threshold else "LM"
        return {"severity": SEVERITY_NAMES[sev], "severity_label": sev, "p_high": round(float(p_high), 4)}

    def predict_severity(self, src, boxes: list | None = None) -> dict:
        """Severity only.
        boxes given: [[x0, y0, x1, y1], ...] in pixels of the photo (after EXIF rotation, i.e. the box_xyxy that
            the detector returns). Each box is expanded by 20% on every side and cut from the full-resolution photo,
            exactly as in training.
        boxes None: the image itself is one defect crop (cut around a single defect with ~20% margin)."""
        im, name = load_image(src)
        W, H = im.size
        if boxes is None:
            p = self.classify_crops([im])[0]
            defects = [{"box_xyxy": None, "box_xywhn": None, **self._severity_fields(p)}]
            mode = "crop"
        else:
            clean = []
            for b in boxes:
                if not isinstance(b, (list, tuple)) or len(b) != 4:
                    raise ValueError(f"each box must be [x0, y0, x1, y1], got {b!r}")
                x0, y0, x1, y1 = (float(v) for v in b)
                x0, x1 = max(0.0, min(x0, W)), max(0.0, min(x1, W))
                y0, y1 = max(0.0, min(y0, H)), max(0.0, min(y1, H))
                if x1 - x0 < 1 or y1 - y0 < 1:
                    raise ValueError(f"box {b!r} is empty or outside the {W}x{H} photo")
                clean.append([x0, y0, x1, y1])
            ph = self.classify_crops([crop_defect(im, tuple(b), size=None) for b in clean])
            defects = [{**_box_fields(b, W, H), **self._severity_fields(p)} for b, p in zip(clean, ph)]
            mode = "photo+boxes"
        return {"image": name, "model_version": self.model_version, "width": W, "height": H, "mode": mode,
                "defects": defects, "thresholds": {"high_severity": self.high_threshold}}


def _device(device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


class KhotaDetector(_DetectorPart):
    """Standalone detection model: photo -> defect boxes, type (Curb Ramp Issue / Surface Damage), confidence."""

    def __init__(self, detector_weights, det_conf: float = 0.25, iou: float = 0.6, imgsz: int = 640,
                 device: str | None = None, det_input_max: int | None = 1280, wbf_iou: float = 0.55,
                 member_conf: float = 0.001, model_version: str = "unversioned", rfdetr: dict | None = None):
        self.device = _device(device)
        self.model_version = model_version
        self._init_detector(detector_weights, det_conf, iou, imgsz, det_input_max, wbf_iou, member_conf, rfdetr)

    @classmethod
    def from_dir(cls, weights_dir="weights", **overrides) -> "KhotaDetector":
        det, _, cfg = _read_dir(weights_dir, overrides)
        return cls(det if len(det) > 1 else det[0], **_pick(cfg, _DET_KEYS))

    def predict(self, src) -> dict:
        return self.predict_detection(src)


class KhotaSeverity(_SeverityPart):
    """Standalone severity model: defect (photo + box, or a crop) -> High / Low-Medium and p_high."""

    def __init__(self, classifier_weights, high_threshold: float | None = None, tta: bool = False,
                 device: str | None = None, model_version: str = "unversioned"):
        self.device = _device(device)
        self.model_version = model_version
        self._init_severity(classifier_weights, high_threshold, tta)

    @classmethod
    def from_dir(cls, weights_dir="weights", **overrides) -> "KhotaSeverity":
        _, clf, cfg = _read_dir(weights_dir, overrides)
        return cls(clf, **_pick(cfg, _SEV_KEYS))

    def predict(self, src, boxes: list | None = None) -> dict:
        return self.predict_severity(src, boxes)


class KhotaPipeline(_DetectorPart, _SeverityPart):
    """Integrated model: detection, then severity for every detected defect."""

    def __init__(self, detector_weights: str | Path | list, classifier_weights: str | Path | list,
                 det_conf: float = 0.25, high_threshold: float | None = None, iou: float = 0.6,
                 imgsz: int = 640, tta: bool = False, device: str | None = None, det_input_max: int | None = 1280,
                 wbf_iou: float = 0.55, member_conf: float = 0.001, model_version: str = "unversioned",
                 rfdetr: dict | None = None):
        """detector_weights: one YOLO checkpoint, or a list whose predictions are fused with weighted box fusion
        (each member at confidence >= member_conf, fused boxes kept if their fused confidence >= det_conf).
        classifier_weights: one checkpoint, or a list whose High probabilities are averaged (seed ensemble).
        tta: also classify the horizontally flipped crop and average.
        det_input_max: before detection the photo is downscaled to this long side (see _detect_photo). Boxes are
        returned in original-photo pixels and the severity crops are cut from the full-resolution photo."""
        self.device = _device(device)
        self.model_version = model_version
        self._init_detector(detector_weights, det_conf, iou, imgsz, det_input_max, wbf_iou, member_conf, rfdetr)
        self._init_severity(classifier_weights, high_threshold, tta)

    @classmethod
    def from_dir(cls, weights_dir: str | Path = "weights", **overrides) -> "KhotaPipeline":
        """weights_dir holds the detector(s), the classifier(s) and pipeline.json
        ({"det_conf", "high_threshold", "imgsz", "iou", "tta", "detectors": [...], "classifiers": [...], ...})."""
        det, clf, cfg = _read_dir(weights_dir, overrides)
        return cls(det if len(det) > 1 else det[0], clf, **_pick(cfg, set(_DET_KEYS) | set(_SEV_KEYS)))

    def predict(self, src) -> dict:
        im, name = load_image(src)
        W, H = im.size
        xyxy, scores, classes = self._detect_photo(im)
        crops = [crop_defect(im, tuple(b), size=None) for b in xyxy]   # resize to 224 happens in self.tf
        p_high = self.classify_crops(crops)
        defects = [{**_box_fields(b, W, H), "class_id": int(c), "class": CLASS_NAMES[int(c)],
                    "confidence": round(float(s), 4), **self._severity_fields(p)}
                   for b, c, s, p in zip(xyxy, classes, scores, p_high)]
        defects.sort(key=lambda d: -d["confidence"])
        return {"image": name, "model_version": self.model_version, "width": W, "height": H,
                "result": "Defects found" if defects else "No Target Defect",
                "defects": defects,
                "thresholds": {"detection_conf": self.det_conf, "high_severity": self.high_threshold}}
