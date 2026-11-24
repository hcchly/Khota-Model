"""Train the severity classifier (Low–Medium vs High) on 20%-expanded defect crops.

Checkpoint selection and the High-probability threshold use the VALIDATION split only.
The test split is never touched here (see evaluation/scripts/evaluate.py).

Example (baseline configuration):
  python severity/scripts/train_classifier.py --name r18_default --arch resnet18 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
from khota.classifier import (CropDataset, best_threshold, build_model, eval_transform,  # noqa: E402
                              metrics_binary, train_transform)


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor | None, gamma: float = 2.0, smoothing: float = 0.0):
        super().__init__()
        self.alpha, self.gamma, self.smoothing = alpha, gamma, smoothing

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.alpha, reduction="none", label_smoothing=self.smoothing)
        pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
        return ((1 - pt) ** self.gamma * ce).mean()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--aug", default="medium", choices=["none", "light", "medium", "strong"])
    ap.add_argument("--jitter", type=float, default=0.0, help="box jitter (fraction of box size) in training")
    ap.add_argument("--loss", default="weighted_ce", choices=["ce", "weighted_ce", "focal"])
    ap.add_argument("--sampler", default="none", choices=["none", "balanced"])
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--freeze-epochs", type=int, default=0, help="train only the head for N epochs first")
    ap.add_argument("--init", help="start from this checkpoint's weights (e.g. a model pre-trained on public road crops)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--crops", default=str(ROOT / "data" / "crops"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "classifier"))
    a = ap.parse_args()

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out) / f"{a.name}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=2))

    tr = CropDataset(a.crops, ("train",), train_transform(a.aug), jitter=a.jitter)
    va = CropDataset(a.crops, ("val",), eval_transform())
    counts = np.bincount(tr.labels, minlength=2)
    print(f"train {len(tr)} (LM {counts[0]}, HIGH {counts[1]}) | val {len(va)} (HIGH {va.labels.sum()})")

    g = torch.Generator().manual_seed(a.seed)
    if a.sampler == "balanced":
        w = (1.0 / counts)[tr.labels]
        sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.double), len(tr), generator=g)
        tl = torch.utils.data.DataLoader(tr, a.bs, sampler=sampler, num_workers=a.workers, pin_memory=True,
                                         persistent_workers=a.workers > 0, drop_last=True)
    else:
        tl = torch.utils.data.DataLoader(tr, a.bs, shuffle=True, generator=g, num_workers=a.workers,
                                         pin_memory=True, persistent_workers=a.workers > 0, drop_last=True)
    vl = torch.utils.data.DataLoader(va, 64, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)

    model = build_model(a.arch, pretrained=True, dropout=a.dropout)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["model"])
    model = model.to(device)
    cw = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device)
    if a.loss == "ce":
        crit = nn.CrossEntropyLoss(label_smoothing=a.smoothing)
    elif a.loss == "weighted_ce":
        crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=a.smoothing)
    else:
        crit = FocalLoss(cw, gamma=2.0, smoothing=a.smoothing)

    head = [p for n, p in model.named_parameters() if n.startswith(("fc", "classifier"))]
    body = [p for n, p in model.named_parameters() if not n.startswith(("fc", "classifier"))]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": head, "lr": a.lr * 10}], weight_decay=a.wd)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[a.lr, a.lr * 10], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    hist, best = [], {"val_ap": -1}
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        for p in body:
            p.requires_grad_(ep > a.freeze_epochs)
        tot, n = 0.0, 0
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        yv, pv = predict(model, vl, device)
        ap_ = float(average_precision_score(yv, pv))
        auc = float(roc_auc_score(yv, pv))
        m05 = metrics_binary(yv, pv, 0.5)
        thr = best_threshold(yv, pv)
        mt = metrics_binary(yv, pv, thr)
        row = {"epoch": ep, "train_loss": tot / n, "val_ap_high": ap_, "val_auc": auc,
               "val_macro_f1@0.5": m05["macro_f1"], "val_acc@0.5": m05["accuracy"],
               "val_high_recall@0.5": m05["high_recall"], "val_macro_f1@tuned": mt["macro_f1"],
               "tuned_thr": thr, "sec": time.time() - t0}
        hist.append(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        # checkpoint selection: threshold-free High average precision on val (more stable than F1 on a small val set)
        if ap_ > best["val_ap"]:
            best = {"val_ap": ap_, "epoch": ep}
            torch.save({"model": model.state_dict(), "arch": a.arch, "dropout": a.dropout, "epoch": ep}, out / "best.pt")

    with open(out / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)

    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    yv, pv = predict(model, vl, device)
    thr = best_threshold(yv, pv)
    res = {
        "name": a.name, "seed": a.seed, "best_epoch": best["epoch"],
        "val_ap_high": float(average_precision_score(yv, pv)), "val_auc": float(roc_auc_score(yv, pv)),
        "val_at_0.5": metrics_binary(yv, pv, 0.5),
        "val_at_tuned": metrics_binary(yv, pv, thr),
        "high_threshold": thr,
        "majority_baseline_val": metrics_binary(yv, np.zeros_like(pv), 0.5),
    }
    ck["high_threshold"] = thr
    torch.save(ck, out / "best.pt")
    (out / "val_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "majority_baseline_val"}, indent=2))


if __name__ == "__main__":
    main()
