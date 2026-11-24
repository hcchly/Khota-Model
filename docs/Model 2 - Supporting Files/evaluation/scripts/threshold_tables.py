"""Threshold tables for the deployed pipeline (weights/), no retraining.

Runs the deployed pipeline ONCE per photo at the lowest detection threshold (fused 7-detector boxes, each detection
classified by the deployed severity ensemble), then scores every threshold from those saved predictions:

  1. Severity threshold (high_threshold) on the labelled defects (ground-truth crops), as reported before:
     High recall, High precision, Macro-F1, accuracy, High caught / missed, Low-Medium flagged as High.
  2. Detection confidence (det_conf) on the original photos: % of photos with a labelled defect that get a
     detection, % of photos without one that raise a false alarm, precision / recall per class (IoU >= 0.5;
     IoU >= 0.1 in a separate sheet).
  3. End to end High recall for every (det_conf, high_threshold) pair: a High-labelled defect counts as caught only
     if it is detected (IoU >= 0.5, same class) AND that detection is rated High.

Writes results/thresholds/threshold_tables_<split>.xlsx and .json.
Usage: python evaluation/scripts/threshold_tables.py --split val      (thresholds must be chosen on val)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "Model 2" / "inference"))
sys.path[:0] = [str(ROOT / d / "scripts") for d in ("detection", "severity", "evaluation")]
from evaluate import classifier_probs  # noqa: E402
from khota.classifier import metrics_binary  # noqa: E402
from khota.crops import crop_defect  # noqa: E402
from khota.pipeline import KhotaPipeline, load_image  # noqa: E402

SEV_T = [0.05, 0.10, 0.15, 0.20, 0.25, 0.29, 0.35, 0.40, 0.50]   # initial grid + 0.05 / 0.10 to locate 90% recall
DET_T = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
E2E_SEV = [0.05, 0.10, 0.15, 0.20, 0.25, 0.29, 0.35, 0.40, 0.50]
NAMES = {0: "Curb Ramp", 1: "Surface Damage"}


def iou(a, B):
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i + 1e-9)


def collect(pipe: KhotaPipeline, split: str):
    """Per photo: labelled boxes and every fused detection >= the lowest det threshold with its p_high."""
    gt = {}
    for r in csv.DictReader(open(ROOT / "data" / "annotations" / "boxes_final.csv", encoding="utf-8")):
        xc, yc, w, h = (float(r[k]) for k in ("xc", "yc", "w", "h"))
        gt.setdefault(r["image_id"], []).append((int(r["class_id"]), r["severity"], [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]))
    raw = {r["image_id"]: ROOT / "data" / "raw" / f"batch{r['batch']}" / "train" / "images" / r["file"]
           for r in csv.DictReader(open(ROOT / "data" / "annotations" / "images_final.csv", encoding="utf-8"))}
    ids = sorted(p.stem for p in (ROOT / "data" / "yolo" / split / "images").glob("*.jpg"))
    pipe.det_conf = min(DET_T)
    photos = []
    for iid in ids:
        im, _ = load_image(str(raw[iid]))
        W, H = im.size
        det_im = im
        if pipe.det_input_max and max(W, H) > pipe.det_input_max:
            s = pipe.det_input_max / max(W, H)
            det_im = im.resize((round(W * s), round(H * s)), Image.LANCZOS)
        boxes, scores, classes = pipe.detect(det_im)  # exactly the deployed detection path, filtered at min(DET_T)
        crops = [crop_defect(im, (b[0] * W, b[1] * H, b[2] * W, b[3] * H), size=None) for b in boxes]
        ph = pipe.classify_crops(crops)
        photos.append({"id": iid, "gt": gt.get(iid, []),
                       "det": [(int(c), float(s_), list(map(float, b)), float(p)) for b, s_, c, p in zip(boxes, scores, classes, ph)]})
    return photos


def match(photos, t, iou_thr):
    """Greedy (by confidence) same-class matching at det_conf t. Returns per-class TP/FP/FN and, per labelled
    box, the p_high of the detection matched to it (None if missed)."""
    st = {c: [0, 0, 0] for c in (0, 1)}
    matched = []  # (gt_severity, p_high or None)
    flags = {"defect_photos": 0, "flagged": 0, "clean_photos": 0, "false_alarm": 0}
    for ph in photos:
        dets = [d for d in ph["det"] if d[1] >= t]
        if ph["gt"]:
            flags["defect_photos"] += 1; flags["flagged"] += bool(dets)
        else:
            flags["clean_photos"] += 1; flags["false_alarm"] += bool(dets)
        for c in (0, 1):
            G = [g for g in ph["gt"] if g[0] == c]
            used = np.zeros(len(G), bool)
            hit_p = [None] * len(G)
            for _, s_, b, p in sorted((d for d in dets if d[0] == c), key=lambda d: -d[1]):
                if G:
                    v = iou(np.array(b), np.array([g[2] for g in G]))
                    v[used] = -1
                    j = int(v.argmax())
                    if v[j] >= iou_thr:
                        used[j] = True; hit_p[j] = p; st[c][0] += 1
                        continue
                st[c][1] += 1
            st[c][2] += int((~used).sum())
            matched += [(g[1], hp) for g, hp in zip(G, hit_p)]
    return st, matched, flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--weights", default=str(ROOT.parent / "Model 2" / "inference" / "weights"))
    a = ap.parse_args()
    out = ROOT / "results" / "thresholds"
    out.mkdir(parents=True, exist_ok=True)
    pipe = KhotaPipeline.from_dir(a.weights)
    cfg = json.loads((Path(a.weights) / "pipeline.json").read_text())

    # 1. severity on the labelled defects
    y, p, _ = classifier_probs(pipe, ROOT / "data" / "crops", a.split)
    sev = []
    for t in SEV_T:
        m = metrics_binary(y, p, t)
        (tn, fp), (fn, tp) = m["confusion_matrix"]
        sev.append({"high_threshold": t, "high_recall": m["high_recall"], "high_precision": m["high_precision"],
                    "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "high_caught": tp, "high_missed": fn,
                    "lm_flagged_high": fp, "n_high": tp + fn, "n_lm": tn + fp})

    # 2./3. detection and end to end
    photos = collect(pipe, a.split)
    det, det01, e2e, e2e01 = [], [], [], []
    for t in DET_T:
        for iou_thr, table in ((0.5, det), (0.1, det01)):
            st, matched, fl = match(photos, t, iou_thr)
            row = {"det_conf": t, "defect_photos_flagged": fl["flagged"] / fl["defect_photos"],
                   "clean_photos_false_alarm": fl["false_alarm"] / fl["clean_photos"],
                   "flagged": f"{fl['flagged']}/{fl['defect_photos']}", "false_alarms": f"{fl['false_alarm']}/{fl['clean_photos']}"}
            for c in (0, 1):
                tp, fp, fn = st[c]
                row[f"{NAMES[c]} precision"] = tp / max(tp + fp, 1)
                row[f"{NAMES[c]} recall"] = tp / max(tp + fn, 1)
                row[f"{NAMES[c]} found/labelled"] = f"{tp}/{tp + fn}"
            highs = [hp for sv, hp in matched if sv == "HIGH"]
            row["High defects detected"] = sum(hp is not None for hp in highs) / len(highs)
            table.append(row)
            (e2e if iou_thr == 0.5 else e2e01).append(
                {"det_conf": t, **{f"{h:.2f}": sum(hp is not None and hp >= h for hp in highs) / len(highs)
                                   for h in E2E_SEV}, "n_high": len(highs)})
    res = {"split": a.split, "model_version": cfg.get("model_version"), "current": {"det_conf": cfg["det_conf"],
           "high_threshold": cfg["high_threshold"]}, "severity": sev, "detection_iou50": det, "detection_iou10": det01,
           "end_to_end_high_recall_iou50": e2e, "end_to_end_high_recall_iou10": e2e01}
    (out / f"threshold_tables_{a.split}.json").write_text(json.dumps(res, indent=2))
    write_xlsx(res, out / f"threshold_tables_{a.split}.xlsx")
    print(json.dumps({"severity": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in sev]}, indent=0)[:3000])
    for r in det:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    for name, tab in (("e2e IoU>=0.5", e2e), ("e2e IoU>=0.1", e2e01)):
        print(name)
        for r in tab:
            print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})


def write_xlsx(res, path):
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    wb = Workbook()
    split = "VALIDATION" if res["split"] == "val" else "TEST"
    head_fill = PatternFill("solid", fgColor="DDE7F5")
    cur_fill = PatternFill("solid", fgColor="FFF2CC")

    def sheet(ws, title, note, cols, rows, cur_key=None, cur_val=None, pct=()):
        ws.append([f"{title} — {split} split — model {res['model_version']}"]); ws["A1"].font = Font(bold=True, size=12)
        ws.append([note]); ws.append([])
        ws.append([c[0] for c in cols])
        for c in ws[4]:
            c.font = Font(bold=True); c.fill = head_fill
        for r in rows:
            ws.append([r[k] for _, k in cols])
            if cur_key and abs(r[cur_key] - cur_val) < 1e-9:
                for c in ws[ws.max_row]:
                    c.fill = cur_fill
        for col_idx, (_, k) in enumerate(cols, 1):
            ws.column_dimensions[ws.cell(4, col_idx).column_letter].width = max(12, len(cols[col_idx - 1][0]) + 2)
            if k in pct:
                for row in ws.iter_rows(min_row=5, min_col=col_idx, max_col=col_idx):
                    row[0].number_format = "0.0%"

    ws = wb.active; ws.title = "1 Severity"
    sheet(ws, "Severity threshold (high_threshold)",
          "On the labelled defects (ground-truth boxes, cropped as in the pipeline). Highlighted = current setting.",
          [("high_threshold", "high_threshold"), ("High recall", "high_recall"), ("High precision", "high_precision"),
           ("Macro-F1", "macro_f1"), ("Accuracy", "accuracy"), ("High caught", "high_caught"),
           ("High missed", "high_missed"), ("Low–Medium flagged as High", "lm_flagged_high")],
          res["severity"], "high_threshold", res["current"]["high_threshold"],
          pct=("high_recall", "high_precision", "accuracy"))
    dcols = [("det_conf", "det_conf"), ("Defect photos with a detection", "defect_photos_flagged"), ("(count)", "flagged"),
             ("Clean photos with a false alarm", "clean_photos_false_alarm"), ("(count)", "false_alarms"),
             ("Curb Ramp precision", "Curb Ramp precision"), ("Curb Ramp recall", "Curb Ramp recall"),
             ("(found/labelled)", "Curb Ramp found/labelled"),
             ("Surface Damage precision", "Surface Damage precision"), ("Surface Damage recall", "Surface Damage recall"),
             ("(found/labelled)", "Surface Damage found/labelled"), ("High-severity defects detected", "High defects detected")]
    pc = ("defect_photos_flagged", "clean_photos_false_alarm", "Curb Ramp precision", "Curb Ramp recall",
          "Surface Damage precision", "Surface Damage recall", "High defects detected")
    sheet(wb.create_sheet("2 Detection"), "Detection confidence (det_conf)",
          "Original photos through the deployed pipeline. A detection matches a labelled box of the same class at "
          "IoU >= 0.5. Highlighted = current setting.", dcols, res["detection_iou50"], "det_conf",
          res["current"]["det_conf"], pct=pc)
    e2e_cols = [("det_conf  \\  high_threshold", "det_conf")] + [(k, k) for k in res["end_to_end_high_recall_iou50"][0]
                                                                  if k not in ("det_conf", "n_high")]
    sheet(wb.create_sheet("3 End-to-end High recall"), "End-to-end High recall (detected AND rated High)",
          f"Share of the {res['end_to_end_high_recall_iou50'][0]['n_high']} High-labelled defects that the pipeline "
          "both detects (IoU >= 0.5, same class) and rates High, for each pair of thresholds.",
          e2e_cols, res["end_to_end_high_recall_iou50"], pct=tuple(k for _, k in e2e_cols[1:]))
    sheet(wb.create_sheet("4 Detection IoU 0.1"), "Detection confidence, lenient matching (IoU >= 0.1)",
          "Same as sheet 2 but a detection only has to overlap the labelled defect (IoU >= 0.1); Surface Damage boxes "
          "are drawn inconsistently, so this shows how many defects are actually found.", dcols,
          res["detection_iou10"], "det_conf", res["current"]["det_conf"], pct=pc)
    sheet(wb.create_sheet("5 End-to-end IoU 0.1"), "End-to-end High recall, lenient matching (IoU >= 0.1)",
          "Same as sheet 3 but the detection only has to overlap the High-labelled defect (IoU >= 0.1).",
          e2e_cols, res["end_to_end_high_recall_iou10"], pct=tuple(k for _, k in e2e_cols[1:]))
    wb.save(path)


if __name__ == "__main__":
    main()
