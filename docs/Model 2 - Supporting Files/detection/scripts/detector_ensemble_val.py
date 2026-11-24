"""Inference-side detector improvements, measured on VALIDATION only:
  * test-time augmentation (ultralytics augment=True: flips + scales),
  * an ensemble of several detectors fused with weighted box fusion (WBF).
Every candidate is scored with the same AP code (detection/scripts/ap_by_iou.py), so the comparison is like for like.

Usage: python detection/scripts/detector_ensemble_val.py --runs runs/detector_v2/y26s640_s42 runs/detector_v2/y26s640_s43 ...
       [--split val] [--out results/val_detector_ensemble.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / d / "scripts") for d in ("detection", "severity", "evaluation")]
sys.path.insert(0, str(ROOT / "inference"))
from ap_by_iou import _iou, ap_table  # noqa: E402
from khota.ensemble import class_map, merge_subclasses, wbf  # noqa: E402


def load_gts(split_dir: Path) -> dict:
    gts = {0: {}, 1: {}}
    for lab in sorted((split_dir / "labels").glob("*.txt")):
        for line in lab.read_text().splitlines():
            if line.strip():
                c, xc, yc, w, h = line.split()[:5]
                xc, yc, w, h = map(float, (xc, yc, w, h))
                gts[int(c)].setdefault(lab.stem, []).append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return gts


def predict(model: YOLO, imgs: list[Path], imgsz: int, augment: bool) -> dict:
    """stem -> (boxes xyxyn (n,4), scores (n,), classes (n,)) at conf 0.001 / NMS IoU 0.6 (as in validation)."""
    out, cmap = {}, class_map(model)
    for ip in imgs:
        r = model.predict(str(ip), imgsz=imgsz, conf=0.001, iou=0.6, max_det=300, augment=augment, verbose=False)[0]
        out[ip.stem] = merge_subclasses(r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                        r.boxes.cls.cpu().numpy().astype(int), cmap)  # sub-class models -> 2 classes
    return out


def to_table(preds: dict) -> dict:
    P = {0: [], 1: []}
    for img, (boxes, scores, cls) in preds.items():
        for b, s, c in zip(boxes, scores, cls):
            P[int(c)].append((float(s), img, b))
    return P


def best_conf(P: dict, gts: dict, grid=np.round(np.arange(0.01, 0.81, 0.01), 2)) -> tuple[float, dict]:
    """Confidence threshold maximising the mean F1 of the two classes (IoU >= 0.5, greedy by confidence)."""
    best = (-1.0, None, None)
    for t in grid:
        f1s, per = [], {}
        for c in (0, 1):
            npos = sum(len(v) for v in gts[c].values())
            used = {k: np.zeros(len(v), bool) for k, v in gts[c].items()}
            tp = fp = 0
            for s_, img, b in sorted((x for x in P[c] if x[0] >= t), key=lambda x: -x[0]):
                G = np.asarray(gts[c].get(img, []))
                hit = False
                if len(G):
                    ious = _iou(b, G); ious[used[img]] = -1; j = int(ious.argmax())
                    if ious[j] >= 0.5:
                        used[img][j] = True; hit = True
                tp += hit; fp += not hit
            pr, rc = tp / max(tp + fp, 1), tp / max(npos, 1)
            f1 = 2 * pr * rc / max(pr + rc, 1e-9)
            f1s.append(f1); per[c] = {"precision": pr, "recall": rc, "f1": f1}
        if np.mean(f1s) > best[0]:
            best = (float(np.mean(f1s)), float(t), per)
    return best[1], {"mean_f1": best[0], "per_class": best[2]}


def summary(name: str, t: dict) -> dict:
    cr, sd = t["Curb Ramp Issue"]["AP@0.5"], t["Surface Damage"]["AP@0.5"]
    print(f"{name:42s} mAP50 {(cr + sd) / 2:.3f}  CR {cr:.3f}  SD {sd:.3f}  | SD AP@0.1 {t['Surface Damage']['AP@0.1']:.3f}",
          flush=True)
    return {"mAP50": (cr + sd) / 2, "CR": cr, "SD": sd, "table": t}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "val_detector_ensemble.json"))
    a = ap.parse_args()
    if a.split == "test":
        sys.exit("validation-only tool")
    split_dir = ROOT / "data" / "yolo" / a.split
    imgs = sorted((split_dir / "images").glob("*.jpg"))
    gts = load_gts(split_dir)
    res, plain, tta = {}, [], []
    for r in map(Path, a.runs):
        imgsz = json.loads((r / "val_metrics.json").read_text())["config"]["imgsz"]
        m = YOLO(str(r / "weights" / "best.pt"))
        p = predict(m, imgs, imgsz, False); plain.append(p)
        res[r.name] = summary(r.name, ap_table(to_table(p), gts))
        if not a.no_tta:
            try:
                q = predict(m, imgs, imgsz, True); tta.append(q)
                res[r.name + "+TTA"] = summary(r.name + " + TTA", ap_table(to_table(q), gts))
            except Exception as e:  # some heads do not support augment
                print(f"TTA not available for {r.name}: {str(e)[:120]}")
    for name, group in (("ensemble", plain), ("ensemble+TTA", tta)):
        if len(group) > 1:
            fused = {img: wbf([g[img] for g in group]) for img in group[0]}
            res[name] = summary(f"WBF {name} ({len(group)} models)", ap_table(to_table(fused), gts))
            if name == "ensemble":
                conf, info = best_conf(to_table(fused), gts)
                res[name]["best_conf_by_mean_f1"] = conf
                res[name]["at_best_conf"] = info
                print(f"   best detection confidence on {a.split}: {conf:.2f}  mean F1 {info['mean_f1']:.3f}  "
                      + "  ".join(f"{('CR', 'SD')[c]} P {v['precision']:.2f} R {v['recall']:.2f}" for c, v in info["per_class"].items()))
    Path(a.out).write_text(json.dumps({k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
