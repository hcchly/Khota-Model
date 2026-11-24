"""Per-class Average Precision at several IoU thresholds (supplementary to the standard mAP@0.5).

Standard mAP@0.5 counts a detection as correct only if its box overlaps the labelled box by IoU >= 0.5. For damage
whose extent is drawn inconsistently in the labels (where a patch of damage starts and ends, one large box vs. several
small ones) a detection on the right damage can still fail that test. AP at IoU 0.1 / 0.3 shows how much of the gap
is box extent rather than missed or false defects. Same AP definition as ultralytics (101-point interpolation of the
precision envelope); AP@0.5 here is a sanity check against the ultralytics number (greedy matching by confidence,
so it can differ slightly).

Usage: python detection/scripts/ap_by_iou.py --weights runs/detector_v2/<run>/weights/best.pt --split val [--imgsz 640]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("Curb Ramp Issue", "Surface Damage")


def _iou(a, B):
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i + 1e-9)


def _ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision, [0.0])))))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def collect(model: YOLO, split_dir: Path, imgsz: int):
    """Predictions (conf 0.001, NMS IoU 0.6 as in validation) and labels, normalised xyxy."""
    preds, gts = {0: [], 1: []}, {0: {}, 1: {}}
    for ip in sorted((split_dir / "images").glob("*.jpg")):
        r = model.predict(str(ip), imgsz=imgsz, conf=0.001, iou=0.6, max_det=300, verbose=False)[0]
        for b, c, f in zip(r.boxes.xyxyn.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int), r.boxes.conf.cpu().numpy()):
            preds[int(c)].append((float(f), ip.stem, b))
        lab = split_dir / "labels" / f"{ip.stem}.txt"
        for line in (lab.read_text().splitlines() if lab.exists() else []):
            if line.strip():
                c, xc, yc, w, h = line.split()[:5]
                xc, yc, w, h = map(float, (xc, yc, w, h))
                gts[int(c)].setdefault(ip.stem, []).append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return preds, gts


def ap_table(preds, gts, thresholds=(0.1, 0.3, 0.5)) -> dict:
    out = {}
    for c in (0, 1):
        npos = sum(len(v) for v in gts[c].values())
        P = sorted(preds[c], key=lambda t: -t[0])
        row = {"n_labels": npos}
        for t in thresholds:
            used = {k: np.zeros(len(v), bool) for k, v in gts[c].items()}
            tp = np.zeros(len(P))
            for i, (_, img, b) in enumerate(P):
                G = np.asarray(gts[c].get(img, []))
                if len(G):
                    ious = _iou(b, G)
                    ious[used[img]] = -1
                    j = int(ious.argmax())
                    if ious[j] >= t:
                        used[img][j] = True
                        tp[i] = 1
            ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
            rec, prec = ctp / max(npos, 1), ctp / np.maximum(ctp + cfp, 1e-9)
            row[f"AP@{t}"] = _ap(rec, prec) if len(P) else 0.0
            row[f"max_recall@{t}"] = float(rec[-1]) if len(P) else 0.0
        out[NAMES[c]] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "yolo"))
    ap.add_argument("--out")
    a = ap.parse_args()
    res = ap_table(*collect(YOLO(a.weights), Path(a.data_dir) / a.split, a.imgsz))
    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
