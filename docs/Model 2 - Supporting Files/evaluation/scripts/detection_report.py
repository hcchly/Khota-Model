"""Detection report of the delivered pipeline on one split: precision-recall curves, detection confusion matrix, AP at
IoU 0.1 / 0.3 / 0.5 and, optionally, annotated examples of correct detections and of every error type.

  python evaluation/scripts/detection_report.py --split test                  # PR curves + confusion matrix -> results/detection_test/
  python evaluation/scripts/detection_report.py --split val --examples        # + example candidates (full pipeline, original photos)
  python evaluation/scripts/detection_report.py --split val --render results/detection_val/examples_selected.json

Detection is run exactly as in evaluation/scripts/evaluate.py (the fused detector ensemble on the split's 1280 px dataset images,
101-point AP with greedy matching by confidence), so the AP values equal those in metrics.json. The confusion matrix
uses the delivered detection threshold and IoU >= 0.5 with class-agnostic matching (a box of the wrong class counts as
a class confusion). Examples use the full pipeline on the original photos at the delivered thresholds, as the app does:
  TP   detection matches a labelled box (same class, IoU >= 0.5) and the severity is right
  SEV  detection matches a labelled box, wrong severity
  LOC  detection overlaps a labelled box of its class only partly (0.1 <= IoU < 0.5) and that box is not found otherwise
  FP   detection overlaps no labelled box (IoU < 0.1)
  FN   labelled box that no detection overlaps (IoU < 0.1)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageDraw, ImageFont, ImageOps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
sys.path[:0] = [str(ROOT / d / "scripts") for d in ("detection", "severity", "evaluation")]
from ap_by_iou import NAMES, _ap, _iou, ap_table  # noqa: E402
from detector_ensemble_val import load_gts  # noqa: E402
from khota.ensemble import predict_members, wbf  # noqa: E402
from khota.pipeline import KhotaPipeline  # noqa: E402

SHORT = ("Curb Ramp", "Surface Damage")
SEVN = {"HIGH": "High", "LM": "Low–Medium"}
BLUE, GREY, GREEN, RED = "#2a78d6", "#8c8b85", (40, 200, 70), (255, 70, 40)


def font(n):
    for f in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, n)
        except OSError:
            pass
    return ImageFont.load_default()


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


# ------------------------------------------------------------------------------------------------ detection metrics
def detect_split(pipe, split_dir: Path):
    P, dets = {0: [], 1: []}, {}
    dev = 0 if pipe.device.type == "cuda" else "cpu"
    for ip in sorted((split_dir / "images").glob("*.jpg")):
        members = predict_members(pipe.detectors, str(ip), pipe.imgsz, pipe.member_conf, pipe.iou, dev)
        fb, fs, fc = wbf(members, pipe.wbf_iou) if len(members) > 1 else members[0]
        dets[ip.stem] = (np.asarray(fb, float).reshape(-1, 4), np.asarray(fs, float), np.asarray(fc, int))
        for bb, sc, cc in zip(*dets[ip.stem]):
            P[int(cc)].append((float(sc), ip.stem, bb))
    return P, dets


def pr_curve(P, gts, c, t):
    """Same matching as ap_by_iou.ap_table: greedy by confidence, each labelled box used once."""
    npos = sum(len(v) for v in gts[c].values())
    Pc = sorted(P[c], key=lambda x: -x[0])
    used = {k: np.zeros(len(v), bool) for k, v in gts[c].items()}
    tp = np.zeros(len(Pc))
    for i, (_, img, b) in enumerate(Pc):
        G = np.asarray(gts[c].get(img, []))
        if len(G):
            ious = _iou(b, G); ious[used[img]] = -1; j = int(ious.argmax())
            if ious[j] >= t:
                used[img][j] = True; tp[i] = 1
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / max(npos, 1), ctp / np.maximum(ctp + cfp, 1e-9)
    return rec, prec, np.array([x[0] for x in Pc]), (_ap(rec, prec) if len(Pc) else 0.0), npos


def plot_pr(P, gts, conf, split, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    op = {}
    for ax, c in zip(axes, (0, 1)):
        for t, ls, col in ((0.5, "-", BLUE), (0.1, "--", GREY)):
            rec, prec, sc, ap, npos = pr_curve(P, gts, c, t)
            env = np.flip(np.maximum.accumulate(np.flip(prec))) if len(prec) else prec
            ax.plot(np.r_[0, rec], np.r_[env[0] if len(env) else 1, env], ls, color=col, lw=1.8,
                    label=f"IoU ≥ {t}: AP {ap:.3f}")
            if t == 0.5 and len(sc) and (sc >= conf).any():
                k = int(np.where(sc >= conf)[0][-1])
                op[NAMES[c]] = {"conf": conf, "precision": float(prec[k]), "recall": float(rec[k])}
                ax.plot(rec[k], prec[k], "o", ms=7, mfc="white", mec=BLUE, mew=2,
                        label=f"threshold {conf}: P {prec[k]:.2f}, R {rec[k]:.2f}")
        ax.set_title(f"{NAMES[c]} ({npos} labelled boxes)", fontsize=10)
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, lw=0.6); ax.legend(loc="upper right", fontsize=8, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(f"Precision–recall curves, {split} split (v2 detector ensemble)", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return op


def confusion(dets, gts, conf, t=0.5):
    """3 x 3: rows = label (Curb Ramp, Surface Damage, background), cols = prediction; class-agnostic matching."""
    G_all = {}
    for c in (0, 1):
        for stem, bs in gts[c].items():
            G_all.setdefault(stem, []).extend((c, b) for b in bs)
    M = np.zeros((3, 3), int)
    for stem, (b, s, c) in dets.items():
        keep = s >= conf
        b, s, c = b[keep], s[keep], c[keep]
        G = G_all.get(stem, [])
        used = np.zeros(len(G), bool)
        for i in np.argsort(-s):
            best, bj = t, -1
            for j, (_, gb) in enumerate(G):
                if not used[j]:
                    v = iou(b[i], gb)
                    if v >= best:
                        best, bj = v, j
            if bj >= 0:
                used[bj] = True; M[G[bj][0], int(c[i])] += 1
            else:
                M[2, int(c[i])] += 1
        for j, (gc, _) in enumerate(G):
            if not used[j]:
                M[gc, 2] += 1
    return M


def plot_confusion(M, conf, split, path):
    lab = ["Curb Ramp", "Surface Damage", "background"]
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    show = M.astype(float); show[2, 2] = np.nan
    ax.imshow(show, cmap="Blues", vmin=0, vmax=np.nanmax(show))
    for i in range(3):
        for j in range(3):
            if (i, j) != (2, 2):
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=13,
                        color="white" if M[i, j] > np.nanmax(show) / 2 else "black")
    ax.set_xticks(range(3), lab); ax.set_yticks(range(3), lab)
    ax.set_xlabel("predicted (detection confidence ≥ %.2f)" % conf); ax.set_ylabel("label")
    ax.set_title(f"Detection confusion matrix, {split} split (IoU ≥ 0.5)", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


# ------------------------------------------------------------------------------------------------ examples
def end_to_end(pipe, split):
    gt = {}
    for r in csv.DictReader(open(ROOT / "data" / "annotations" / "boxes_final.csv", encoding="utf-8")):
        gt.setdefault(r["image_id"], []).append(r)
    raw = {r["image_id"]: ROOT / "data" / "raw" / f"batch{r['batch']}" / "train" / "images" / r["file"]
           for r in csv.DictReader(open(ROOT / "data" / "annotations" / "images_final.csv", encoding="utf-8"))}
    recs = []
    for ip in sorted((ROOT / "data" / "yolo" / split / "images").glob("*.jpg")):
        pred = pipe.predict(str(raw[ip.stem]))
        W, H = pred["width"], pred["height"]
        G = [{"box": g["box"], "class_id": int(g["class_id"]), "severity": g["severity"],
              "xyxy": [(float(g["xc"]) - float(g["w"]) / 2) * W, (float(g["yc"]) - float(g["h"]) / 2) * H,
                       (float(g["xc"]) + float(g["w"]) / 2) * W, (float(g["yc"]) + float(g["h"]) / 2) * H]}
             for g in gt.get(ip.stem, [])]
        D = [{k: d[k] for k in ("class_id", "confidence", "severity_label", "p_high", "box_xyxy")} for d in pred["defects"]]
        recs.append({"image_id": ip.stem, "path": str(raw[ip.stem].relative_to(ROOT)), "W": W, "H": H, "gt": G, "det": D})
    return recs


def categorize(rec):
    G, D = rec["gt"], rec["det"]
    used, match, items = set(), {}, []
    for i in sorted(range(len(D)), key=lambda i: -D[i]["confidence"]):
        best, bj = 0.5, None
        for j, g in enumerate(G):
            if j not in used and g["class_id"] == D[i]["class_id"]:
                v = iou(D[i]["box_xyxy"], g["xyxy"])
                if v >= best:
                    best, bj = v, j
        if bj is not None:
            used.add(bj); match[i] = (bj, best)
    for i, d in enumerate(D):
        if i in match:
            j, v = match[i]
            items.append({"cat": "TP" if d["severity_label"] == G[j]["severity"] else "SEV", "det": i, "gt": j, "iou": v})
            continue
        same = [(iou(d["box_xyxy"], g["xyxy"]), j) for j, g in enumerate(G) if g["class_id"] == d["class_id"]]
        anyc = [iou(d["box_xyxy"], g["xyxy"]) for g in G]
        if same and max(same)[0] >= 0.1 and max(same)[1] not in used:
            items.append({"cat": "LOC", "det": i, "gt": max(same)[1], "iou": max(same)[0]})
        elif not anyc or max(anyc) < 0.1:
            items.append({"cat": "FP", "det": i, "gt": None, "iou": max(anyc, default=0.0)})
        else:
            items.append({"cat": "OTHER", "det": i, "gt": None, "iou": max(anyc)})
    for j, g in enumerate(G):
        if j not in used and all(iou(d["box_xyxy"], g["xyxy"]) < 0.1 for d in D):
            items.append({"cat": "FN", "det": None, "gt": j, "iou": 0.0})
    return items


def det_label(d):
    return f"pred {SHORT[d['class_id']]} {SEVN[d['severity_label']]}  conf {d['confidence']:.2f}  P(High) {d['p_high']:.2f}"


def gt_label(g):
    return f"label {SHORT[g['class_id']]} {SEVN[g['severity']]}"


def panel(rec, items, size=(640, 480), header=None, caption=None):
    """Crop of the original photo around the boxes of the items: labels green, predictions red."""
    im = ImageOps.exif_transpose(Image.open(ROOT / rec["path"])).convert("RGB")
    sx, sy = im.width / rec["W"], im.height / rec["H"]
    boxes = []
    for it in items:
        if it.get("gt") is not None:
            g = rec["gt"][it["gt"]]; boxes.append(([g["xyxy"][0] * sx, g["xyxy"][1] * sy, g["xyxy"][2] * sx, g["xyxy"][3] * sy], GREEN, gt_label(g)))
        if it.get("det") is not None:
            d = rec["det"][it["det"]]; b = d["box_xyxy"]
            boxes.append(([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], RED, det_label(d)))
    x0 = min(b[0][0] for b in boxes); y0 = min(b[0][1] for b in boxes)
    x1 = max(b[0][2] for b in boxes); y1 = max(b[0][3] for b in boxes)
    w, h = max(x1 - x0, im.width * 0.3), max(y1 - y0, im.height * 0.3)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = w * 1.5, h * 1.5
    ar = size[0] / size[1]
    w, h = (h * ar, h) if w / h < ar else (w, w / ar)
    w, h = min(w, im.width), min(h, im.height)
    cx0 = min(max(cx - w / 2, 0), im.width - w); cy0 = min(max(cy - h / 2, 0), im.height - h)
    crop = im.crop((int(cx0), int(cy0), int(cx0 + w), int(cy0 + h))).resize(size, Image.LANCZOS)
    k = size[0] / w
    dr = ImageDraw.Draw(crop)
    f = font(15)
    for (b, col, lab) in boxes:
        bb = [(b[0] - cx0) * k, (b[1] - cy0) * k, (b[2] - cx0) * k, (b[3] - cy0) * k]
        dr.rectangle(bb, outline=col, width=4)
    ty = size[1] - 22 * len(boxes) - 6
    dr.rectangle([0, ty - 4, size[0], size[1]], fill=(0, 0, 0))
    for n, (_, col, lab) in enumerate(boxes):
        dr.text((8, ty + 22 * n), lab, fill=col, font=f)
    if header:
        dr.rectangle([0, 0, size[0], 26], fill=(0, 0, 0))
        dr.text((8, 4), header, fill=(255, 255, 255), font=font(16))
    if caption:
        lines = textwrap.wrap(caption, 74)
        out = Image.new("RGB", (size[0], size[1] + 22 * len(lines) + 14), (255, 255, 255))
        out.paste(crop, (0, 0))
        d2 = ImageDraw.Draw(out)
        for n, line in enumerate(lines):
            d2.text((8, size[1] + 7 + 22 * n), line, fill=(20, 20, 20), font=font(16))
        return out
    return crop


def sheet(panels, cols=2):
    w = max(p.width for p in panels); h = max(p.height for p in panels)
    rows = (len(panels) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols - 1) * 10, rows * h + (rows - 1) * 10), (255, 255, 255))
    for n, p in enumerate(panels):
        out.paste(p, ((n % cols) * (w + 10), (n // cols) * (h + 10)))
    return out


ORDER = {"TP": "-conf", "FP": "-conf", "LOC": "-conf", "FN": "-area", "SEV": "sev"}


def candidates(recs, out: Path, k=4):
    pool = {c: [] for c in ORDER}
    stats = {c: 0 for c in list(ORDER) + ["OTHER"]}
    for r in recs:
        for it in categorize(r):
            stats[it["cat"]] += 1
            if it["cat"] in pool:
                d = r["det"][it["det"]] if it["det"] is not None else None
                g = r["gt"][it["gt"]] if it["gt"] is not None else None
                if it["cat"] == "FN":  # High first, then the largest (most visible) missed boxes
                    key = -((g["xyxy"][2] - g["xyxy"][0]) * (g["xyxy"][3] - g["xyxy"][1]) / (r["W"] * r["H"])) - (g["severity"] == "HIGH")
                elif it["cat"] == "SEV":  # most confident wrong severity first
                    key = -abs(d["p_high"] - 0.29)
                else:  # TP, FP, LOC: most confident detections first
                    key = -d["confidence"]
                pool[it["cat"]].append((key, r["image_id"], it))
    by_id = {r["image_id"]: r for r in recs}
    chosen = {}
    for cat, lst in pool.items():
        lst.sort(key=lambda x: x[0])
        groups = {}  # by class, so both classes are shown where possible
        for key, iid, it in lst:
            r = by_id[iid]
            c = (r["det"][it["det"]] if it["det"] is not None else r["gt"][it["gt"]])["class_id"]
            groups.setdefault(c, []).append((iid, it))
        pick, seen = [], set()
        while len(pick) < k and any(groups.values()):
            for c in sorted(groups):
                while groups[c]:
                    iid, it = groups[c].pop(0)
                    if iid not in seen:
                        pick.append({"image_id": iid, **it}); seen.add(iid)
                        break
                if len(pick) == k:
                    break
        chosen[cat] = pick
        if pick:
            ps = [panel(by_id[p["image_id"]], [p], header=f"[{cat}{n + 1}] {p['image_id']}  IoU {p['iou']:.2f}")
                  for n, p in enumerate(pick)]
            sheet(ps).save(out / f"candidates_{cat}.jpg", quality=85)
    (out / "examples_candidates.json").write_text(json.dumps(chosen, indent=1))
    return stats


def render(selection: Path, recs, out_dir: Path):
    """selection: [{"cat", "image_id", "det", "gt", "title", "cause"}] -> one figure per category (2 panels)."""
    sel = json.loads(selection.read_text(encoding="utf-8"))
    by_id = {r["image_id"]: r for r in recs}
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {"TP": "1_true_positives", "FP": "2_false_positives", "FN": "3_false_negatives",
             "LOC": "4_localisation_errors", "SEV": "5_severity_errors"}
    for cat, fname in names.items():
        items = [s for s in sel if s["cat"] == cat]
        if items:
            ps = [panel(by_id[s["image_id"]], [s], size=(640, 480), header=f"{s['image_id']} — {s['title']}",
                        caption=s["cause"]) for s in items]
            sheet(ps, cols=len(ps)).save(out_dir / f"{fname}.jpg", quality=88)
    print("examples ->", out_dir)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "inference" / "weights"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--out")
    ap.add_argument("--examples", action="store_true", help="also run the full pipeline on the original photos")
    ap.add_argument("--render", help="examples_selected.json -> final annotated example figures")
    ap.add_argument("--reuse", action="store_true", help="reuse the saved end-to-end predictions of the split")
    a = ap.parse_args()
    out = Path(a.out or ROOT / "results" / f"detection_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    pipe = KhotaPipeline.from_dir(a.weights)
    cfg = json.loads((Path(a.weights) / "pipeline.json").read_text())
    e2e = out / "end_to_end_predictions.json"
    if a.render:
        render(Path(a.render), json.loads(e2e.read_text()), out / "examples")
        return
    split_dir = ROOT / "data" / "yolo" / a.split
    gts = load_gts(split_dir)
    P, dets = detect_split(pipe, split_dir)
    tab = ap_table(P, gts)
    op = plot_pr(P, gts, cfg["det_conf"], a.split, out / f"pr_curves_{a.split}.png")
    M = confusion(dets, gts, cfg["det_conf"])
    plot_confusion(M, cfg["det_conf"], a.split, out / f"detection_confusion_matrix_{a.split}.png")
    res = {"split": a.split, "model_version": cfg.get("model_version"), "det_conf": cfg["det_conf"],
           "ap_by_iou": tab, "mAP50": float(np.mean([tab[n]["AP@0.5"] for n in NAMES])),
           "mAP10": float(np.mean([tab[n]["AP@0.1"] for n in NAMES])), "operating_point": op,
           "confusion_matrix": {"rows_label": ["Curb Ramp Issue", "Surface Damage", "background"],
                                "cols_predicted": ["Curb Ramp Issue", "Surface Damage", "background"],
                                "iou": 0.5, "det_conf": cfg["det_conf"], "matrix": M.tolist()}}
    if a.examples:
        if a.reuse and e2e.exists():
            recs = json.loads(e2e.read_text())
        else:
            recs = end_to_end(pipe, a.split)
            e2e.write_text(json.dumps(recs))
        res["example_counts"] = candidates(recs, out)
    (out / f"detection_report_{a.split}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("mAP50", "mAP10", "operating_point")} | {"cm": M.tolist()}
                     | ({"examples": res["example_counts"]} if a.examples else {}), indent=1))


if __name__ == "__main__":
    main()
