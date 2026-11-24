"""Collect validation metrics of every run into one table (every configuration tried is reported).

Usage: python evaluation/scripts/summarize_runs.py [--classifier runs/classifier_v2] [--detector runs/detector_v2]
Writes results/val_runs_classifier.csv and results/val_runs_detector.csv and prints mean ± std per config
(configs = run names without the _s<seed> suffix).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def table(rows, keys, title, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "config", "seed"] + keys)
        w.writeheader()
        w.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    print(f"\n{title}  (val; mean ± std over seeds)")
    print(f"{'config':28s} n  " + "  ".join(f"{k:>14s}" for k in keys))
    for c, rs in sorted(by.items(), key=lambda kv: -np.mean([r[keys[0]] for r in kv[1]])):
        cells = []
        for k in keys:
            v = np.array([r[k] for r in rs], dtype=float)
            cells.append(f"{v.mean():.3f}±{v.std(ddof=1):.3f}" if len(v) > 1 else f"{v.mean():.3f}")
        print(f"{c:28s} {len(rs)}  " + "  ".join(f"{s:>14s}" for s in cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default=str(ROOT / "runs" / "classifier_v2"))
    ap.add_argument("--detector", default=str(ROOT / "runs" / "detector_v2"))
    a = ap.parse_args()
    seed = re.compile(r"_s(\d+)$")

    rows = []
    for f in sorted(Path(a.classifier).glob("*/val_metrics.json")):
        m = json.loads(f.read_text())
        t = m["val_at_tuned"]
        name = f.parent.name
        rows.append({"run": name, "config": seed.sub("", name), "seed": (seed.search(name) or [0, ""])[1],
                     "macro_f1": t["macro_f1"], "high_recall": t["high_recall"], "high_precision": t["high_precision"],
                     "accuracy": t["accuracy"], "ap_high": m["val_ap_high"], "threshold": t["threshold"]})
    if rows:
        table(rows, ["macro_f1", "high_recall", "high_precision", "accuracy", "ap_high", "threshold"],
              "SEVERITY CLASSIFIER", ROOT / "results" / "val_runs_classifier.csv")

    rows = []
    for f in sorted(Path(a.detector).glob("*/val_metrics.json")):
        v = json.loads(f.read_text())["val"]
        name = f.parent.name
        pc = v["per_class"]
        rows.append({"run": name, "config": seed.sub("", name), "seed": (seed.search(name) or [0, ""])[1],
                     "mAP50": v["overall"]["mAP50"], "mAP50_SD": pc["Surface Damage"]["mAP50"],
                     "mAP50_CR": pc["Curb Ramp Issue"]["mAP50"], "mAP50-95": v["overall"]["mAP50-95"],
                     "P": v["overall"]["precision"], "R": v["overall"]["recall"],
                     "R_SD": pc["Surface Damage"]["recall"], "R_CR": pc["Curb Ramp Issue"]["recall"]})
    if rows:
        table(rows, ["mAP50", "mAP50_SD", "mAP50_CR", "mAP50-95", "P", "R", "R_SD", "R_CR"],
              "DETECTOR", ROOT / "results" / "val_runs_detector.csv")


if __name__ == "__main__":
    main()
