"""Cache raw detector predictions once per (run, input size, split), so ensembles and fusion settings can be compared
on the CPU in seconds (detection/scripts/ens_eval.py) instead of re-running every model.

Predictions are made exactly like detector_ensemble_val.py: the downscaled split images (data/yolo/<split>/images),
conf 0.001, NMS IoU 0.6, max 300 boxes, sub-class models merged to the two output classes.
Saved to runs/pred_cache/<split>/<run name>@<imgsz>.npz (one entry per image stem: boxes xyxyn, scores, classes).

Usage: python detection/scripts/pred_cache.py --runs runs/detector_v2/y26s640_s42 ... [--imgsz 640 800] [--split val] [--device 0]
The test split is refused (it is scored once, at the end, by evaluate.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
from khota.ensemble import class_map, merge_subclasses  # noqa: E402

CACHE = ROOT / "runs" / "pred_cache"


def cache_path(split: str, run: Path, imgsz: int) -> Path:
    return CACHE / split / f"{run.parent.name}__{run.name}@{imgsz}.npz"


def load(split: str, run: Path, imgsz: int) -> dict:
    z = np.load(cache_path(split, run, imgsz), allow_pickle=True)
    return z["preds"].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--imgsz", nargs="*", type=int, help="input sizes (default: the run's training size)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default="best.pt")
    a = ap.parse_args()
    if a.split == "test":
        sys.exit("validation-only tool")
    from ultralytics import YOLO
    imgs = sorted((ROOT / "data" / "yolo" / a.split / "images").glob("*.jpg"))
    for r in map(Path, a.runs):
        train_sz = json.loads((r / "val_metrics.json").read_text())["config"]["imgsz"] if (r / "val_metrics.json").exists() else 640
        m = YOLO(str(r / "weights" / a.weights))
        cmap = class_map(m)
        for sz in (a.imgsz or [train_sz]):
            out = cache_path(a.split, r, sz)
            if out.exists():
                print(f"cached {out.name}")
                continue
            preds = {}
            for ip in imgs:
                res = m.predict(str(ip), imgsz=sz, conf=0.001, iou=0.6, max_det=300, device=a.device, verbose=False)[0]
                preds[ip.stem] = merge_subclasses(res.boxes.xyxyn.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                                  res.boxes.cls.cpu().numpy().astype(int), cmap)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, preds=np.array(preds, dtype=object))
            print(f"wrote {out.name} ({sum(len(v[1]) for v in preds.values())} boxes)", flush=True)


if __name__ == "__main__":
    main()
