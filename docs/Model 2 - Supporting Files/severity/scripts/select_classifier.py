"""Compare severity-classifier configurations on VALIDATION: single seeds, seed ensembles (mean probability of the
3 seeds) and cross-architecture ensembles, at two operating points chosen on validation:
  balanced     threshold maximising Macro-F1
  high-recall  threshold maximising Macro-F1 subject to High recall >= --min-recall
Usage: python severity/scripts/select_classifier.py --runs runs/classifier_v2 [--min-recall 0.8]
Writes results/val_severity_selection.csv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
from khota.classifier import CropDataset, build_model, eval_transform, metrics_binary  # noqa: E402

GRID = np.round(np.linspace(0.05, 0.95, 91), 2)


def probs(ck_path, ds, dev):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], pretrained=False, dropout=ck.get("dropout", 0.0))
    m.load_state_dict(ck["model"])
    m.to(dev).eval()
    out = []
    with torch.no_grad():
        for x, _ in torch.utils.data.DataLoader(ds, 64, num_workers=0):
            x = x.to(dev)
            p = torch.softmax(m(x).float(), 1)[:, 1] + torch.softmax(m(torch.flip(x, dims=[3])).float(), 1)[:, 1]
            out.append((p / 2).cpu().numpy())  # + horizontal-flip TTA
    return np.concatenate(out)


def operating_points(y, p, min_recall):
    f1 = [(metrics_binary(y, p, t)["macro_f1"], -abs(t - 0.5), t) for t in GRID]
    bal = max(f1)[2]
    ok = [(metrics_binary(y, p, t)["macro_f1"], t) for t in GRID if metrics_binary(y, p, t)["high_recall"] >= min_recall]
    hr = max(ok)[1] if ok else float(GRID[0])
    return bal, hr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "runs" / "classifier_v2"))
    ap.add_argument("--crops", default=str(ROOT / "data" / "crops"))
    ap.add_argument("--min-recall", type=float, default=0.8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = CropDataset(Path(a.crops), ("val",), eval_transform())
    y = ds.labels
    P, groups = {}, defaultdict(list)
    for ck in sorted(Path(a.runs).glob("*/best.pt")):
        name = ck.parent.name
        P[name] = probs(ck, ds, dev)
        groups[re.sub(r"_s\d+$", "", name)].append(name)
    cands = dict(P)
    for g, names in groups.items():
        if len(names) > 1:
            cands[f"ENS[{g} x{len(names)}]"] = np.mean([P[n] for n in names], 0)
    full = [g for g, n in groups.items() if len(n) >= 3]
    for i, g1 in enumerate(full):
        for g2 in full[i + 1:]:
            cands[f"ENS[{g1}+{g2}]"] = np.mean([P[n] for n in groups[g1] + groups[g2]], 0)

    rows = []
    for name, p in cands.items():
        bal, hr = operating_points(y, p, a.min_recall)
        mb, mh = metrics_binary(y, p, bal), metrics_binary(y, p, hr)
        rows.append({"model": name, "auc": roc_auc_score(y, p),
                     "thr_bal": bal, "f1_bal": mb["macro_f1"], "rec_bal": mb["high_recall"], "prec_bal": mb["high_precision"],
                     "acc_bal": mb["accuracy"], "thr_hr": hr, "f1_hr": mh["macro_f1"], "rec_hr": mh["high_recall"],
                     "prec_hr": mh["high_precision"], "acc_hr": mh["accuracy"]})
    rows.sort(key=lambda r: -r["f1_bal"])
    out = ROOT / "results" / "val_severity_selection.csv"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"val: {len(y)} crops, {int(y.sum())} High | high-recall point: recall >= {a.min_recall}")
    print(f"{'model':40s}  auc   | thr  F1    recH  precH acc  | thr  F1    recH  precH acc")
    for r in rows:
        print(f"{r['model'][:40]:40s} {r['auc']:.3f} | {r['thr_bal']:.2f} {r['f1_bal']:.3f} {r['rec_bal']:.2f} {r['prec_bal']:.2f} "
              f"{r['acc_bal']:.2f} | {r['thr_hr']:.2f} {r['f1_hr']:.3f} {r['rec_hr']:.2f} {r['prec_hr']:.2f} {r['acc_hr']:.2f}")


if __name__ == "__main__":
    main()
