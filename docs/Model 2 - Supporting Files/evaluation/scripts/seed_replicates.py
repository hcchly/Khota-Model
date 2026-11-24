"""Seed variability (mean ± std over seeds 42 / 43 / 44) of the chosen recipes on one split.

Severity classifier, per seed s:
  * each single configuration (e.g. r18_default_s42), and
  * the recipe ensemble of that seed (mean High probability of r18_default_s + r18_balanced_s, flip TTA),
  with the threshold chosen on VALIDATION by the same rule as the deployed model
  (max Macro-F1 subject to High recall >= --min-recall; the lowest grid threshold if no threshold reaches it).
  The deployed model averages all six checkpoints; its own numbers come from evaluation/scripts/evaluate.py.
Detector: every given run's best.pt evaluated on the split (per-class P / R / mAP50 / mAP50-95), grouped by
<runs folder>/<configuration> (seed suffix removed).

No choice is made here: thresholds come from validation, runs are given on the command line.
Usage:
  python evaluation/scripts/seed_replicates.py --split test --out results/test/seed_replicates.json \
      --det-runs runs/detector_v2/y26s640_s42 runs/detector_v2/y26s640_s43 runs/detector_v2/y26s640_s44
  python evaluation/scripts/seed_replicates.py --split val --cls-configs --det-runs runs/detector/v8n640_s4? --out ...  (detector only)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
sys.path[:0] = [str(ROOT / d / "scripts") for d in ("detection", "severity", "evaluation")]
from khota.classifier import CropDataset, eval_transform, metrics_binary  # noqa: E402
from select_classifier import operating_points, probs  # noqa: E402
from train_detector import val_report  # noqa: E402

KEYS = ("accuracy", "macro_f1", "high_recall", "high_precision")


def mean_std(rows: list[dict], keys) -> dict:
    return {k: {"mean": float(np.mean([r[k] for r in rows])),
                "std": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0} for k in keys}


def classifier_seeds(a, dev) -> dict:
    dv, ds = (CropDataset(Path(a.crops), (s,), eval_transform()) for s in ("val", a.split))
    yv, ys = dv.labels, ds.labels
    pv, ps = {}, {}
    for c in a.cls_configs:
        for s in a.seeds:
            ck = Path(a.cls_runs) / f"{c}_s{s}" / "best.pt"
            pv[(c, s)], ps[(c, s)] = probs(ck, dv, dev), probs(ck, ds, dev)
    recipes = {c: lambda s, c=c: [(c, s)] for c in a.cls_configs}
    if len(a.cls_configs) > 1:
        recipes["+".join(a.cls_configs)] = lambda s: [(c, s) for c in a.cls_configs]
    out = {}
    for name, members in recipes.items():
        per_seed = []
        for s in a.seeds:
            p_val = np.mean([pv[m] for m in members(s)], 0)
            p_split = np.mean([ps[m] for m in members(s)], 0)
            thr = a.fixed_threshold if a.fixed_threshold is not None else operating_points(yv, p_val, a.min_recall)[1]
            m = metrics_binary(ys, p_split, thr)
            per_seed.append({"seed": s, "threshold": float(thr), **{k: m[k] for k in KEYS}})
        out[name] = {"per_seed": per_seed, **mean_std(per_seed, KEYS)}
    return {"classifier": out,
            "classifier_majority_baseline": {k: metrics_binary(ys, np.zeros(len(ys)), 0.5)[k] for k in KEYS},
            "classifier_n": {"crops": int(len(ys)), "high": int(ys.sum())}}


def detector_seeds(a, out_dir: Path) -> dict:
    rows = []
    for r in map(Path, a.det_runs):
        cfg = json.loads((r / "val_metrics.json").read_text())["config"]
        v = val_report(YOLO(str(r / "weights" / "best.pt")), cfg["imgsz"], a.split, 8,
                       out_dir / f"{r.parent.name}_{r.name}", a.data)
        row = {"run": f"{r.parent.name}/{r.name}", "group": f"{r.parent.name}/{r.name.rsplit('_s', 1)[0]}",
               "mAP50": v["overall"]["mAP50"], "mAP50-95": v["overall"]["mAP50-95"],
               "P": v["overall"]["precision"], "R": v["overall"]["recall"]}
        for cname, short in (("Curb Ramp Issue", "CR"), ("Surface Damage", "SD")):
            pc = v["per_class"].get(cname, {})
            row.update({f"{k}_{short}": pc.get(kk, 0.0) for k, kk in
                        (("mAP50", "mAP50"), ("mAP50-95", "mAP50-95"), ("P", "precision"), ("R", "recall"))})
        rows.append(row)
        print(row["run"], f"mAP50 {row['mAP50']:.3f}  CR {row['mAP50_CR']:.3f}  SD {row['mAP50_SD']:.3f}", flush=True)
    groups = {}
    for row in rows:
        groups.setdefault(row["group"], []).append(row)
    num = [k for k in rows[0] if k not in ("run", "group")]
    return {g: {"per_seed": rs, **mean_std(rs, num)} for g, rs in groups.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--cls-runs", default=str(ROOT / "runs" / "classifier_v2"))
    ap.add_argument("--cls-configs", nargs="*", default=["r18_default", "r18_balanced"], help="none given: skip")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--min-recall", type=float, default=0.8)
    ap.add_argument("--fixed-threshold", type=float, help="use this High threshold instead of the validation rule "
                                                            "(e.g. 0.5 = the requester's default argmax)")
    ap.add_argument("--crops", default=str(ROOT / "data" / "crops"))
    ap.add_argument("--det-runs", nargs="*", default=[])
    ap.add_argument("--data", default=str(ROOT / "data" / "yolo" / "khota.yaml"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"split": a.split, "cls_runs": a.cls_runs,
           "threshold_rule": f"fixed {a.fixed_threshold}" if a.fixed_threshold is not None
           else f"val: max Macro-F1 s.t. High recall >= {a.min_recall}"}
    if a.cls_configs:
        res.update(classifier_seeds(a, dev))
    if a.det_runs:
        res["detector"] = detector_seeds(a, out.parent.resolve() / f"seed_runs_{a.split}")
    out.write_text(json.dumps(res, indent=2))
    for name, d in res.get("classifier", {}).items():
        print(f"{name:28s} " + "  ".join(f"{k} {d[k]['mean']:.3f}±{d[k]['std']:.3f}" for k in KEYS))
    for g, d in res.get("detector", {}).items():
        print(f"{g:28s} " + "  ".join(f"{k} {d[k]['mean']:.3f}±{d[k]['std']:.3f}" for k in ("mAP50", "mAP50_CR", "mAP50_SD")))


if __name__ == "__main__":
    main()
