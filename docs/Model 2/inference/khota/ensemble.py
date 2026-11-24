"""Detector ensemble: weighted box fusion (WBF, Solovyev et al. 2021) of several YOLO models' predictions.

Boxes of the same class from all models are clustered (IoU > iou_thr with the cluster's fused box); the fused box
is the confidence-weighted mean of its members and the fused confidence is sum(conf) / n_models, so a defect
found by only one of seven models gets a low score.
"""
from __future__ import annotations

import numpy as np


def iou_one_to_many(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i + 1e-9)


def wbf(per_model: list[tuple[np.ndarray, np.ndarray, np.ndarray]], iou_thr: float = 0.55,
        classes=(0, 1)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """per_model: one (boxes xyxy normalised (n,4), scores (n,), classes (n,)) tuple per model, for ONE image.
    Returns fused (boxes, scores, classes), sorted by score."""
    M = len(per_model)
    fb, fs, fc = [], [], []
    for c in classes:
        items = [(float(s), np.asarray(b, float)) for boxes, scores, cls in per_model
                 for b, s, k in zip(boxes, scores, cls) if int(k) == c]
        items.sort(key=lambda t: -t[0])
        w_sum, wb_sum, n, fused = [], [], [], []
        for s, b in items:
            j = -1
            if fused:
                ious = iou_one_to_many(b, np.asarray(fused))
                k = int(ious.argmax())
                if ious[k] > iou_thr:
                    j = k
            if j < 0:
                w_sum.append(s); wb_sum.append(s * b); n.append(1); fused.append(b.copy())
            else:
                w_sum[j] += s; wb_sum[j] = wb_sum[j] + s * b; n[j] += 1; fused[j] = wb_sum[j] / w_sum[j]
        for sw, nn, box in zip(w_sum, n, fused):
            fb.append(box); fs.append(sw / M if nn <= M else sw / nn); fc.append(c)
    if not fb:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    order = np.argsort(-np.asarray(fs))
    return np.asarray(fb)[order], np.asarray(fs)[order], np.asarray(fc, int)[order]


TYPES = {"Curb Ramp Issue": 0, "Surface Damage": 1}


def class_map(model) -> np.ndarray | None:
    """Detectors trained with sub-classes name them "<type>|<sub-class>" (e.g. "Surface Damage|HIGH"); returns the
    array sub-class id -> output class ID (0 Curb Ramp Issue, 1 Surface Damage), or None for a plain 2-class model."""
    names = model.names
    if not any("|" in str(n) for n in names.values()):
        return None
    return np.array([TYPES[str(names[i]).split("|")[0]] for i in range(len(names))], int)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy NMS; returns the kept indices, highest score first."""
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        order = rest[iou_one_to_many(boxes[i], boxes[rest]) <= iou_thr]
    return np.asarray(keep, int)


def merge_subclasses(boxes: np.ndarray, scores: np.ndarray, cls: np.ndarray, cmap: np.ndarray | None,
                     iou_thr: float = 0.6):
    """Map sub-class predictions to the two output classes; where two sub-classes predicted the same defect, keep the
    more confident box (per-class NMS after mapping), so each model still gives one box per defect."""
    if cmap is None or not len(cls):
        return boxes, scores, cls
    cls = cmap[cls]
    keep = np.concatenate([np.flatnonzero(cls == c)[nms(boxes[cls == c], scores[cls == c], iou_thr)]
                           for c in np.unique(cls)])
    keep = keep[np.argsort(-scores[keep])]
    return boxes[keep], scores[keep], cls[keep]


class RFDETRMember:
    """An RF-DETR checkpoint (Roboflow, DINOv2 backbone, Apache-2.0) used as an ensemble member next to the YOLO
    models. Needs the rfdetr package. Class names may use the "<type>|<sub-class>" convention like the YOLO models."""

    SIZES = ("nano", "small", "medium", "base", "large")

    def __init__(self, path, size: str, names: list, device: str | None = None, flip: bool = False, weight: int = 1):
        """flip: also predict on the mirrored photo (test-time augmentation; fused like a second model).
        weight: the member's predictions count `weight` times in the box fusion (chosen on validation)."""
        self.flip, self.weight = flip, int(weight)
        import rfdetr
        cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
               "base": rfdetr.RFDETRBase, "large": rfdetr.RFDETRLarge}[size]
        self.model = cls(pretrain_weights=str(path), num_classes=len(names), **({"device": device} if device else {}))
        self.names = dict(enumerate(names))

    def predict_arrays(self, image, conf: float):
        from PIL import Image
        im = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        return self._predict(im, conf)

    def predict_passes(self, image, conf: float) -> list:
        """All passes of this member (plain, + mirrored if flip), each repeated `weight` times."""
        from PIL import Image, ImageOps
        im = image if isinstance(image, Image.Image) else Image.open(image).convert("RGB")
        passes = [self._predict(im, conf)]
        if self.flip:
            b, s, c = self._predict(ImageOps.mirror(im), conf)
            b = b.copy()
            b[:, [0, 2]] = 1 - b[:, [2, 0]]  # mirror the boxes back
            passes.append((b, s, c))
        return passes * self.weight

    def _predict(self, im, conf: float):
        W, H = im.size
        d = self.model.predict(im, threshold=conf)
        b = np.asarray(d.xyxy, float).reshape(-1, 4) / np.array([W, H, W, H])
        s, c = np.asarray(d.confidence, float), np.asarray(d.class_id, int)
        k = (c >= 0) & (c < len(self.names))
        return np.clip(b[k], 0, 1), s[k], c[k]


def predict_members(models, image, imgsz: int | list, conf: float = 0.001, iou: float = 0.6, device=None):
    """Run every model on one image (path, PIL or array); returns the per-model tuples wbf() expects."""
    sizes = imgsz if isinstance(imgsz, (list, tuple)) else [imgsz] * len(models)
    out = []
    for m, sz in zip(models, sizes):
        if hasattr(m, "predict_passes"):  # RF-DETR member: one or more passes (flip, weight)
            out.extend(merge_subclasses(b, s, c, class_map(m), iou) for b, s, c in m.predict_passes(image, conf))
            continue
        r = m.predict(image, imgsz=sz, conf=conf, iou=iou, max_det=300, device=device, verbose=False)[0]
        b, s, c = r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)
        out.append(merge_subclasses(b, s, c, class_map(m), iou))
    return out
