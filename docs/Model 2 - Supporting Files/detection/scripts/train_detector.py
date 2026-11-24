"""Train the YOLO defect detector (Curb Ramp Issue / Surface Damage).

No-Target-Defect images are in the dataset with empty label files -> used as background images.
After training, the model is evaluated on the VALIDATION split and the detection confidence
threshold is chosen there (max mean F1 over classes). The test split is not used here.

Example (baseline):
  python detection/scripts/train_detector.py --name v8n_640_default --model yolov8n.pt --imgsz 640 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "yolo" / "khota.yaml"


def val_report(model: YOLO, imgsz: int, split: str, batch: int, out_dir: Path, data: str = str(DATA)) -> dict:
    m = model.val(data=data, split=split, imgsz=imgsz, batch=batch, conf=0.001, iou=0.6,
                  plots=True, project=str(out_dir), name=f"eval_{split}", exist_ok=True, verbose=False)
    names = m.names
    box = m.box
    per_class = {}
    for i, c in enumerate(box.ap_class_index):
        per_class[names[int(c)]] = {
            "precision": float(box.p[i]), "recall": float(box.r[i]),
            "mAP50": float(box.ap50[i]), "mAP50-95": float(box.ap[i]),
        }
    # confidence threshold that maximises the mean F1 over classes on THIS split
    f1 = np.asarray(box.f1_curve)  # (nc, 1000) over confidence grid box.px
    px = np.asarray(box.px)
    mean_f1 = f1.mean(0)
    k = int(mean_f1.argmax())
    return {
        "split": split,
        "overall": {"precision": float(box.mp), "recall": float(box.mr),
                    "mAP50": float(box.map50), "mAP50-95": float(box.map)},
        "per_class": per_class,
        "best_conf_by_mean_f1": float(px[k]), "mean_f1_at_best_conf": float(mean_f1[k]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra", default="{}", help='JSON dict of extra ultralytics train args, e.g. {"mosaic":0.5}')
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--project", default=str(ROOT / "runs" / "detector"))
    a = ap.parse_args()

    out = Path(a.project).resolve()  # absolute: ultralytics nests relative project paths under runs/detect/
    model = YOLO(a.model)
    extra = json.loads(a.extra)
    args = dict(data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, patience=a.patience,
                seed=a.seed, deterministic=True, workers=a.workers, project=str(out), name=a.name,
                exist_ok=True, cos_lr=True, close_mosaic=10, plots=True)
    args.update(extra)  # --extra overrides the defaults above
    model.train(**args)
    best = out / a.name / "weights" / "best.pt"
    rep = {"config": vars(a), "val": val_report(YOLO(str(best)), a.imgsz, "val", a.batch, out / a.name, a.data)}
    (out / a.name / "val_metrics.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["val"], indent=2))


if __name__ == "__main__":
    main()
