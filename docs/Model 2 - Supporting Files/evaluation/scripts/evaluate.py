"""Final evaluation on one split (default: TEST — run once, after every choice was made on validation).

  1. Detector: per-class precision, recall, mAP@0.5, mAP@0.5:0.95 (ultralytics, conf 0.001 for the PR curves)
     plus P / R / F1 at the deployed confidence threshold.
  2. Severity classifier on the ground-truth boxes (20% crops): accuracy, Macro-F1, High recall/precision,
     confusion matrix and the always-Low–Medium majority baseline, at the threshold chosen on validation.
  3. End to end (detector -> classifier) on the original photos: severity accuracy on detections that match a
     labelled box, photo-level flag / false-alarm rates, detector recall per class and severity.
  Supplementary: per-class AP at IoU 0.1 / 0.3 / 0.5 (detection/scripts/ap_by_iou.py).

Usage: python evaluation/scripts/evaluate.py --weights weights --split test --out results/test
       (weights/ holds the detector(s), classifier(s) and pipeline.json with det_conf / high_threshold / imgsz)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
from ultralytics import YOLO

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
sys.path[:0] = [str(ROOT / d / "scripts") for d in ("detection", "severity", "evaluation")]
from khota.classifier import CropDataset, eval_transform, metrics_binary  # noqa: E402
from khota.pipeline import KhotaPipeline  # noqa: E402
from ap_by_iou import ap_table, collect  # noqa: E402
from detector_ensemble_val import best_conf, load_gts  # noqa: E402
from khota.ensemble import predict_members, wbf  # noqa: E402
from train_detector import val_report  # noqa: E402


def classifier_probs(pipe: KhotaPipeline, crops: Path, split: str):
    """High probability of every ground-truth crop of the split, exactly as the deployed pipeline computes it
    (same checkpoint(s), same flip TTA setting)."""
    ds = CropDataset(crops, (split,), eval_transform())
    ps = []
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(ds, 64, num_workers=0):
            x = x.to(pipe.device)
            p, n = 0.0, 0
            for m in pipe.classifiers:
                for xx in ([x, torch.flip(x, dims=[3])] if pipe.tta else [x]):
                    p = p + torch.softmax(m(xx).float(), 1)[:, 1]
                    n += 1
            ps.append((p / n).cpu().numpy())
    return ds.labels, np.concatenate(ps), ds.rows


def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i][j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax.set_xticks([0, 1], ["Low–Medium", "High"]); ax.set_yticks([0, 1], ["Low–Medium", "High"])
    ax.set_xlabel("predicted"); ax.set_ylabel("label"); ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "inference" / "weights"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=str(ROOT / "data" / "yolo" / "khota.yaml"))
    ap.add_argument("--crops", default=str(ROOT / "data" / "crops"))
    ap.add_argument("--labels", default=str(ROOT / "data" / "annotations" / "boxes_final.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "test"))
    ap.add_argument("--limit", type=int, help="debug only: end-to-end on the first N photos")
    a = ap.parse_args()
    W, out = Path(a.weights), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((W / "pipeline.json").read_text())
    res = {"split": a.split, "pipeline": cfg}

    pipe = KhotaPipeline.from_dir(W)
    split_dir = Path(a.data).parent / a.split
    # 1. detector
    if len(pipe.detectors) == 1 and not hasattr(pipe.detector, "predict_passes"):  # single YOLO: ultralytics val
        det = val_report(pipe.detector, cfg["imgsz"], a.split, 8, out, a.data)
        # supplementary: AP at IoU 0.1 / 0.3 / 0.5 (how much of the gap is box extent)
        res["detector_ap_by_iou"] = ap_table(*collect(pipe.detector, split_dir, cfg["imgsz"]))
    else:
        # WBF ensemble: fused predictions on the split's dataset images (1280 px, as in training), scored with the
        # same AP code as detection/scripts/detector_ensemble_val.py (101-point AP, greedy matching by confidence)
        P, gts = {0: [], 1: []}, load_gts(split_dir)
        dev = 0 if pipe.device.type == "cuda" else "cpu"
        for ip in sorted((split_dir / "images").glob("*.jpg")):
            members = predict_members(pipe.detectors, str(ip), pipe.imgsz, pipe.member_conf, pipe.iou, dev)
            fb, fs, fc = wbf(members, pipe.wbf_iou) if len(members) > 1 else members[0]  # one model: no fusion
            for bb, sc, cc in zip(fb, fs, fc):
                P[int(cc)].append((float(sc), ip.stem, bb))
        tab = ap_table(P, gts)
        res["detector_ap_by_iou"] = tab
        _, at = best_conf(P, gts, grid=[cfg["det_conf"]])
        names = ("Curb Ramp Issue", "Surface Damage")
        det = {"split": a.split, "metric": f"WBF ensemble of {len(pipe.detectors)} detectors, own AP code",
               "overall": {"mAP50": float(np.mean([tab[n]["AP@0.5"] for n in names]))},
               "per_class": {n: {"mAP50": tab[n]["AP@0.5"], **{k: at["per_class"][i][k] for k in ("precision", "recall", "f1")}}
                             for i, n in enumerate(names)},
               "at_deployed_conf": {"conf": cfg["det_conf"], "mean_f1": at["mean_f1"]}}
    res["detector"] = det

    # 2. classifier on ground-truth crops
    y, p, rows = classifier_probs(pipe, Path(a.crops), a.split)
    thr = cfg["high_threshold"]
    m = metrics_binary(y, p, thr)
    res["classifier"] = {"at_threshold": m, "at_0.5": metrics_binary(y, p, 0.5),
                         "majority_baseline": metrics_binary(y, np.zeros_like(p), 0.5), "n": int(len(y)),
                         "n_high": int(y.sum())}
    plot_cm(m["confusion_matrix"], f"Severity, {a.split} (threshold {thr:.2f})", out / "severity_confusion_matrix.png")
    with open(out / "severity_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "box", "label", "p_high", "pred"])
        for r, yy, pp in zip(rows, y, p):
            w.writerow([r["image_id"], r["box"], "HIGH" if yy else "LM", f"{pp:.4f}", "HIGH" if pp >= thr else "LM"])

    # 3. end to end on the split's images
    gt = {}
    for r in csv.DictReader(open(a.labels, encoding="utf-8")):
        gt.setdefault(r["image_id"], []).append(r)
    raw = {r["image_id"]: ROOT / "data" / "raw" / f"batch{r['batch']}" / "train" / "images" / r["file"]
           for r in csv.DictReader(open(Path(a.labels).parent / "images_final.csv", encoding="utf-8"))}
    imgs = sorted((Path(a.data).parent / a.split / "images").glob("*.jpg"))[:a.limit]  # split ids; ORIGINAL photos used
    matched, sev_ok, n_gt, n_det = 0, 0, 0, 0
    e2e_y, e2e_p = [], []
    THR = (0.1, 0.3, 0.5)
    loc = {c: {t: [0, 0, 0, 0] for t in THR} for c in (0, 1)}  # per class/IoU: gt found, n gt, det matched, n det
    sev_rec = {(c, sv): {t: [0, 0] for t in THR} for c in (0, 1) for sv in ("HIGH", "LM")}  # detector recall by severity
    img_level = {"photos_with_defect": 0, "flagged": 0, "photos_without_defect": 0, "false_alarm": 0}
    for ip in imgs:
        pred = pipe.predict(str(raw[ip.stem]))
        Wd, Hd = pred["width"], pred["height"]
        G = [g for g in gt.get(ip.stem, [])]
        n_gt += len(G); n_det += len(pred["defects"])
        # localisation-tolerant recall / precision at the deployed confidence threshold (supplementary)
        db = [(d["class_id"], [d["box_xyxy"][0] / Wd, d["box_xyxy"][1] / Hd, d["box_xyxy"][2] / Wd,
                                     d["box_xyxy"][3] / Hd]) for d in pred["defects"]]
        gb = [(int(g["class_id"]), [float(g["xc"]) - float(g["w"]) / 2, float(g["yc"]) - float(g["h"]) / 2,
                                    float(g["xc"]) + float(g["w"]) / 2, float(g["yc"]) + float(g["h"]) / 2]) for g in G]
        for g, (k, b) in zip(G, gb):
            if g["severity"] in ("HIGH", "LM"):
                for t in THR:
                    sev_rec[(k, g["severity"])][t][0] += any(iou(b, d) >= t for kk, d in db if kk == k)
                    sev_rec[(k, g["severity"])][t][1] += 1
        for c in (0, 1):
            for t in THR:
                gc, dc = [b for k, b in gb if k == c], [b for k, b in db if k == c]
                loc[c][t][0] += sum(any(iou(g, d) >= t for d in dc) for g in gc); loc[c][t][1] += len(gc)
                loc[c][t][2] += sum(any(iou(d, g) >= t for g in gc) for d in dc); loc[c][t][3] += len(dc)
        if G:
            img_level["photos_with_defect"] += 1; img_level["flagged"] += int(bool(pred["defects"]))
        else:
            img_level["photos_without_defect"] += 1; img_level["false_alarm"] += int(bool(pred["defects"]))
        used = set()
        for d in sorted(pred["defects"], key=lambda d: -d["confidence"]):
            bx = [d["box_xyxy"][0] / Wd, d["box_xyxy"][1] / Hd, d["box_xyxy"][2] / Wd, d["box_xyxy"][3] / Hd]
            best, bj = 0.5, None
            for j, g in enumerate(G):
                if j in used or int(g["class_id"]) != d["class_id"]:
                    continue
                xc, yc, w, h = (float(g[k]) for k in ("xc", "yc", "w", "h"))
                v = iou(bx, [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
                if v >= best:
                    best, bj = v, j
            if bj is not None and G[bj]["severity"] in ("HIGH", "LM"):
                used.add(bj); matched += 1
                e2e_y.append(int(G[bj]["severity"] == "HIGH")); e2e_p.append(d["p_high"])
                sev_ok += int(d["severity_label"] == G[bj]["severity"])
    res["end_to_end"] = {"images": len(imgs), "labelled_boxes": n_gt, "detections": n_det,
                         "detections_matching_a_label_iou50": matched,
                         "severity_accuracy_on_matched": sev_ok / max(matched, 1),
                         "severity_metrics_on_matched": metrics_binary(np.array(e2e_y), np.array(e2e_p), thr) if e2e_y else None,
                         "photo_level": {**img_level,
                                         "flag_rate_on_defect_photos": img_level["flagged"] / max(img_level["photos_with_defect"], 1),
                                         "false_alarm_rate_on_no_defect_photos": img_level["false_alarm"] / max(img_level["photos_without_defect"], 1)},
                         "localisation_tolerant_at_deployed_conf": {
                             ("Curb Ramp Issue", "Surface Damage")[c]: {f"IoU>={t}": {"recall": v[0] / max(v[1], 1), "precision": v[2] / max(v[3], 1)}
                                                                         for t, v in loc[c].items()} for c in (0, 1)},
                         "detector_recall_by_severity_at_deployed_conf": {
                             f"{('Curb Ramp Issue', 'Surface Damage')[c]} / {sv}": {f"IoU>={t}": v[0] / max(v[1], 1) for t, v in d.items()}
                             | {"n": d[0.5][1]} for (c, sv), d in sev_rec.items()}}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({"detector": det["overall"], "per_class": det["per_class"],
                      "classifier": {k: m[k] for k in ("accuracy", "macro_f1", "high_recall", "high_precision")},
                      "end_to_end": {k: v for k, v in res["end_to_end"].items() if k != "severity_metrics_on_matched"}},
                     indent=2))


if __name__ == "__main__":
    main()
