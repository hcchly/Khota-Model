"""Severity classifier: model, dataset and transforms shared by training, evaluation and inference."""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from .crops import CROP_SIZE, EXPAND

SEVERITIES = ["LM", "HIGH"]          # index 1 = High (positive class for recall / thresholds)
SEVERITY_NAMES = {"LM": "Low–Medium Severity", "HIGH": "High Severity"}
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_model(arch: str = "resnet18", pretrained: bool = True, dropout: float = 0.0) -> nn.Module:
    if arch == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, len(SEVERITIES)))
    elif arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, len(SEVERITIES)))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(max(dropout, 0.2)), nn.Linear(m.classifier[1].in_features, len(SEVERITIES)))
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(SEVERITIES))
    elif arch == "inception_v3":  # 224 px input works (adaptive pooling); the aux head needs >= 299 px, so it is dropped
        m = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None,
                                 aux_logits=True, init_weights=False, transform_input=True)
        m.aux_logits, m.AuxLogits = False, None
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, len(SEVERITIES)))
    else:
        raise ValueError(arch)
    return m


def eval_transform() -> transforms.Compose:
    """Applied to an already-cut 20%-expanded crop. Identical in evaluation and inference."""
    return transforms.Compose([
        transforms.Resize((CROP_SIZE, CROP_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def train_transform(strength: str = "medium") -> transforms.Compose:
    if strength == "none":
        return eval_transform()
    jitter = {"light": (0.15, 0.15, 0.1, 0.02), "medium": (0.3, 0.3, 0.2, 0.03), "strong": (0.4, 0.4, 0.3, 0.05)}[strength]
    scale = {"light": (0.85, 1.0), "medium": (0.75, 1.0), "strong": (0.6, 1.0)}[strength]
    return transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=scale, ratio=(0.75, 1.333),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
        transforms.ColorJitter(*jitter),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def cut_crop(ctx: Image.Image, box: tuple[float, float, float, float], expand: float = EXPAND,
             jitter: float = 0.0, rng: random.Random | None = None) -> Image.Image:
    """Cut the 20%-expanded crop out of a cached context crop.

    `box` is the defect box in context-crop pixels. With jitter > 0 the box centre is shifted and its
    size scaled by up to +-jitter of the box size (simulates imperfect detector boxes).
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if jitter > 0:
        r = rng or random
        cx = (x0 + x1) / 2 + r.uniform(-jitter, jitter) * w
        cy = (y0 + y1) / 2 + r.uniform(-jitter, jitter) * h
        w *= 1 + r.uniform(-jitter, jitter)
        h *= 1 + r.uniform(-jitter, jitter)
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    W, H = ctx.size
    ex0, ey0 = max(0, round(x0 - expand * w)), max(0, round(y0 - expand * h))
    ex1, ey1 = min(W, round(x1 + expand * w)), min(H, round(y1 + expand * h))
    ex1, ey1 = max(ex1, ex0 + 1), max(ey1, ey0 + 1)
    return ctx.crop((ex0, ey0, ex1, ey1))


class CropDataset(torch.utils.data.Dataset):
    """Reads data/crops/crops.csv rows for the given split(s)."""

    def __init__(self, crops_dir: str | Path, splits: tuple[str, ...], transform, jitter: float = 0.0):
        self.dir = Path(crops_dir)
        rows = list(csv.DictReader(open(self.dir / "crops.csv", encoding="utf-8")))
        self.rows = [r for r in rows if r["split"] in splits]
        self.transform = transform
        self.jitter = jitter
        self.labels = np.array([SEVERITIES.index(r["severity"]) for r in self.rows])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ctx = Image.open(self.dir / r["crop"]).convert("RGB")
        box = (float(r["bx0"]), float(r["by0"]), float(r["bx1"]), float(r["by1"]))
        crop = cut_crop(ctx, box, jitter=self.jitter)
        return self.transform(crop), int(self.labels[i])


def metrics_binary(y_true: np.ndarray, p_high: np.ndarray, thr: float = 0.5) -> dict:
    y_pred = (p_high >= thr).astype(int)
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for c in (0, 1):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(np.mean(f1s)),
        "f1_lm": float(f1s[0]), "f1_high": float(f1s[1]),
        "high_recall": float(cm[1, 1] / cm[1].sum()) if cm[1].sum() else 0.0,
        "high_precision": float(cm[1, 1] / cm[:, 1].sum()) if cm[:, 1].sum() else 0.0,
        "confusion_matrix": cm.tolist(),   # rows = true [LM, HIGH], cols = predicted [LM, HIGH]
        "threshold": float(thr),
        "n": int(len(y_true)), "n_high": int(y_true.sum()),
    }


def best_threshold(y_true: np.ndarray, p_high: np.ndarray) -> float:
    """High-probability threshold maximising macro-F1 (ties -> threshold closest to 0.5). Val only.

    Searched on a fixed 0.05..0.95 grid (step 0.01) rather than at the observed probabilities, so a small
    validation set cannot push the threshold to an extreme value that only separates a few val crops."""
    cands = np.round(np.linspace(0.05, 0.95, 91), 2)
    best, best_f1 = 0.5, -1.0
    for t in cands:
        f1 = metrics_binary(y_true, p_high, t)["macro_f1"]
        if f1 > best_f1 + 1e-9 or (abs(f1 - best_f1) <= 1e-9 and abs(t - 0.5) < abs(best - 0.5)):
            best, best_f1 = float(t), f1
    return best
