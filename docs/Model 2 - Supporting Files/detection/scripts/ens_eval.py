"""Evaluate detector ensembles from cached predictions (detection/scripts/pred_cache.py) on VALIDATION, on the CPU.

Every result uses the same AP code as detector_ensemble_val.py / evaluate.py (101-point AP, greedy matching).
Members are given as "<run dir>@<imgsz>" (e.g. runs/detector_v2/y26s640_s42@640).

  python detection/scripts/ens_eval.py --members A@640 B@640 ...                          # one ensemble, current settings
  python detection/scripts/ens_eval.py --members ... --wbf-iou 0.4 0.5 0.55 0.6 0.7         # sweep fusion IoU
  python detection/scripts/ens_eval.py --members ... --skip 0.001 0.01 0.05                 # sweep member confidence floor
  python detection/scripts/ens_eval.py --members ... --sd-scale 0.9 1.0 1.1 1.2             # scale fused Surface Damage boxes
  python detection/scripts/ens_eval.py --members ... --groups "old7=A,B,C" "new=D,E"         # named member subsets
Prints mAP@0.5 (CR, SD), SD AP@0.1 and the best mean F1 over detection thresholds (the deployed operating point).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / d / "scripts") for d in ("detection", "severity", "evaluation")]
sys.path.insert(0, str(ROOT / "inference"))
from ap_by_iou import ap_table  # noqa: E402
from khota.ensemble import wbf  # noqa: E402
from pred_cache import load  # noqa: E402


def load_gts(split_dir: Path) -> dict:
    gts = {0: {}, 1: {}}
    for lab in sorted((split_dir / "labels").glob("*.txt")):
        for line in lab.read_text().splitlines():
            if line.strip():
                c, xc, yc, w, h = line.split()[:5]
                xc, yc, w, h = map(float, (xc, yc, w, h))
                gts[int(c)].setdefault(lab.stem, []).append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return gts


def to_table(preds: dict) -> dict:
    P = {0: [], 1: []}
    for img, (boxes, scores, cls) in preds.items():
        for b, s, c in zip(boxes, scores, cls):
            P[int(c)].append((float(s), img, b))
    return P


def scale_boxes(boxes, cls, f: float, c: int = 1):
    if f == 1.0 or not len(boxes):
        return boxes
    b = boxes.copy()
    m = cls == c
    cx, cy = (b[m, 0] + b[m, 2]) / 2, (b[m, 1] + b[m, 3]) / 2
    w, h = (b[m, 2] - b[m, 0]) * f, (b[m, 3] - b[m, 1]) * f
    b[m] = np.clip(np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1), 0, 1)
    return b


def fuse(members: list[dict], iou: float, skip: float, sd_scale: float) -> dict:
    out = {}
    for img in members[0]:
        per = []
        for m in members:
            b, s, c = m[img]
            k = s >= skip
            per.append((b[k], s[k], c[k]))
        if len(per) == 1:
            fb, fs, fc = per[0]
        else:
            fb, fs, fc = wbf(per, iou)
        out[img] = (scale_boxes(np.asarray(fb).reshape(-1, 4), np.asarray(fc, int), sd_scale), np.asarray(fs), np.asarray(fc, int))
    return out


def best_f1(P: dict, gts: dict) -> tuple[float, float]:
    from detector_ensemble_val import best_conf
    t, info = best_conf(P, gts)
    return t, info["mean_f1"]


def score(name: str, preds: dict, gts: dict, f1: bool = True) -> dict:
    P = to_table(preds)
    t = ap_table(P, gts)
    cr, sd = t["Curb Ramp Issue"]["AP@0.5"], t["Surface Damage"]["AP@0.5"]
    r = {"mAP50": (cr + sd) / 2, "CR": cr, "SD": sd, "SD@0.1": t["Surface Damage"]["AP@0.1"]}
    line = f"{name:58s} mAP50 {r['mAP50']:.3f}  CR {cr:.3f}  SD {sd:.3f}  SD@0.1 {r['SD@0.1']:.3f}"
    if f1:
        r["best_conf"], r["best_mean_f1"] = best_f1(P, gts)
        line += f"  | best F1 {r['best_mean_f1']:.3f} @ {r['best_conf']:.2f}"
    print(line, flush=True)
    return r


def parse_member(spec: str) -> tuple[Path, int]:
    run, sz = spec.rsplit("@", 1)
    return Path(run), int(sz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--groups", nargs="*", default=[], help='"name=idx,idx,..." subsets of --members (0-based) or "name=all"')
    ap.add_argument("--wbf-iou", nargs="*", type=float, default=[0.55])
    ap.add_argument("--skip", nargs="*", type=float, default=[0.001])
    ap.add_argument("--sd-scale", nargs="*", type=float, default=[1.0])
    ap.add_argument("--singles", action="store_true", help="also score every member alone")
    ap.add_argument("--split", default="val")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.split == "test":
        sys.exit("validation-only tool")
    gts = load_gts(ROOT / "data" / "yolo" / a.split)
    specs = [parse_member(s) for s in a.members]
    cache = [load(a.split, r, sz) for r, sz in specs]
    names = [f"{r.parent.name}/{r.name}@{sz}" for r, sz in specs]
    res = {}
    if a.singles:
        for n, c in zip(names, cache):
            res[n] = score(n, c, gts, f1=False)
    groups = {"all": list(range(len(cache)))}
    for g in a.groups:
        k, v = g.split("=", 1)
        groups[k] = list(range(len(cache))) if v == "all" else [int(i) for i in v.split(",")]
    for gname, idx in groups.items():
        for iou, skip, sc in itertools.product(a.wbf_iou, a.skip, a.sd_scale):
            label = f"{gname} ({len(idx)}) iou {iou} skip {skip} sd_scale {sc}"
            res[label] = score(label, fuse([cache[i] for i in idx], iou, skip, sc), gts)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
