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


def predict_members(models, image, imgsz: int | list, conf: float = 0.001, iou: float = 0.6, device=None):
    """Run every model on one image (path, PIL or array); returns the per-model tuples wbf() expects."""
    sizes = imgsz if isinstance(imgsz, (list, tuple)) else [imgsz] * len(models)
    out = []
    for m, sz in zip(models, sizes):
        r = m.predict(image, imgsz=sz, conf=conf, iou=iou, max_det=300, device=device, verbose=False)[0]
        out.append((r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)))
    return out
