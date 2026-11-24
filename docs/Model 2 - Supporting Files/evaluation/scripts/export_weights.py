"""Assemble the deployable model folder: detector + severity classifier checkpoint(s) + pipeline.json.

The detection confidence threshold defaults to the value maximising mean F1 on VALIDATION (from the detector
run's val_metrics.json); the High threshold must be the one chosen on validation (severity/scripts/select_classifier.py).
Usage:
  python evaluation/scripts/export_weights.py --detector runs/detector_v2/<run> \
      --classifiers runs/classifier_v2/r18_default_s4* runs/classifier_v2/r18_balanced_s4* \
      --high-threshold 0.29 --tta --out inference/weights
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", nargs="*", default=[],
                    help="detector run folder(s) (weights/best.pt + val_metrics.json); several = WBF ensemble")
    ap.add_argument("--wbf-iou", type=float, default=0.55)
    ap.add_argument("--rfdetr", nargs="*", default=[], help="RF-DETR run folders (runs/rfdetr/<name>) added as members")
    ap.add_argument("--rfdetr-size", default="medium")
    ap.add_argument("--rfdetr-flip", action="store_true", help="also run RF-DETR on the mirrored photo")
    ap.add_argument("--rfdetr-weight", type=int, default=1, help="RF-DETR passes count this many times in the fusion")
    ap.add_argument("--rfdetr-data", default="data/yolo3/data.yaml", help="data.yaml the RF-DETR runs were trained on")
    ap.add_argument("--classifiers", nargs="+", required=True, help="classifier run folders (best.pt each)")
    ap.add_argument("--high-threshold", type=float, required=True)
    ap.add_argument("--det-conf", type=float)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--out", default="inference/weights")
    ap.add_argument("--model-version", required=True, help="e.g. v1-2026-11-21; returned by the API with every result")
    a = ap.parse_args()
    dets, out = [Path(d) for d in a.detector], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("classifier*.pt")) + list(out.glob("detector*.pt")) + list(out.glob("detector*.pth")):
        old.unlink(missing_ok=True)
    if (len(dets) > 1 or a.rfdetr) and a.det_conf is None:
        raise SystemExit("an ensemble needs --det-conf (chosen on validation by detection/scripts/detector_ensemble_val.py)")
    vm = json.loads((dets[0] / "val_metrics.json").read_text()) if dets else {"config": {"imgsz": 640}, "val": {}}
    if len({json.loads((d / "val_metrics.json").read_text())["config"]["imgsz"] for d in dets}) > 1:
        raise SystemExit("ensemble members must share one input size")
    det_names = []
    for i, d in enumerate(dets, 1):
        n = f"detector_{i}.pt" if len(dets) > 1 else "detector.pt"
        shutil.copy2(d / "weights" / "best.pt", out / n)
        det_names.append(n)
    rf = {}
    if a.rfdetr:
        line = [l for l in Path(a.rfdetr_data).read_text().splitlines() if l.startswith("names:")][0]
        rf_names = json.loads(line.split(":", 1)[1])
        for j, r in enumerate(map(Path, a.rfdetr), len(dets) + 1):
            ck = sorted(r.glob("checkpoint_best*.pth"))[0]
            n = f"detector_{j}.pth"
            shutil.copy2(ck, out / n)
            det_names.append(n)
            rf[n] = {"size": a.rfdetr_size, "names": rf_names, "flip": a.rfdetr_flip, "weight": a.rfdetr_weight,
                     "run": f"{r.parent.name}/{r.name}/{ck.name}"}
    names = []
    for i, c in enumerate(sorted(a.classifiers), 1):
        n = f"classifier_{i}.pt" if len(a.classifiers) > 1 else "classifier.pt"
        shutil.copy2(Path(c) / "best.pt", out / n)
        names.append(n)
    cfg = {"model_version": a.model_version, "det_conf": round(a.det_conf if a.det_conf is not None else vm["val"]["best_conf_by_mean_f1"], 3),
           "high_threshold": a.high_threshold, "imgsz": vm["config"]["imgsz"], "iou": 0.6, "tta": a.tta,
           "det_input_max": 1280,  # photos are downscaled like the detector's training images (05_build_datasets)
           "classifiers": names, "detectors": det_names, "wbf_iou": a.wbf_iou, "member_conf": 0.001,
           **({"rfdetr": rf} if rf else {}),
           "source": {"detector_runs": [f"{d.parent.name}/{d.name}" for d in dets],
                      "detector_models": [Path(json.loads((d / "val_metrics.json").read_text())["config"]["model"]).name
                                          for d in dets],
                      "classifier_runs": [Path(c).name for c in sorted(a.classifiers)]}}
    (out / "pipeline.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))


if __name__ == "__main__":
    main()
