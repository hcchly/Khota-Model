"""Fine-tune RF-DETR (Roboflow; DINOv2 backbone, Apache-2.0) on Khota, then cache its VALIDATION predictions in the
pred_cache format so detection/scripts/ens_eval.py can score it alone and inside the WBF ensemble.

Data: YOLO format with data.yaml (data/yolo3 = Surface Damage split into LM / HIGH sub-classes; merged back to the
two project classes through the "<type>|<sub-class>" class names, exactly like the YOLO sub-class models).

Usage: python detection/scripts/train_rfdetr.py --size medium --name med_c3_s42 [--epochs 40] [--predict-only]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "Model 2" / "inference"))
from khota.ensemble import TYPES, merge_subclasses  # noqa: E402


def model_class(size: str):
    import rfdetr
    return {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium,
            "base": rfdetr.RFDETRBase, "large": rfdetr.RFDETRLarge}[size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="medium")
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "yolo3"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    out = ROOT / "runs" / "rfdetr" / a.name
    cls = model_class(a.size)
    if not a.predict_only:
        out.mkdir(parents=True, exist_ok=True)
        cls().train(dataset_dir=a.data, output_dir=str(out), epochs=a.epochs, batch_size=a.batch,
                    grad_accum_steps=a.accum, early_stopping=True, early_stopping_patience=10,
                    **({"seed": a.seed} if a.seed is not None else {}))
    ck = sorted(out.glob("checkpoint_best*.pth")) or sorted(out.glob("*.pth"))
    print("checkpoints:", [c.name for c in ck])
    m = cls(pretrain_weights=str(ck[0]))
    names = [l.strip() for l in (Path(a.data) / "data.yaml").read_text().splitlines() if l.startswith("names:")][0]
    names = eval(names.split(":", 1)[1])  # list written by us
    cmap = np.array([TYPES[n.split("|")[0]] for n in names])
    res_sz = m.model_config.resolution if hasattr(m, "model_config") else 0
    preds, hist = {}, np.zeros(len(names) + 2, int)
    for ip in sorted((Path(a.data) / "val" / "images").glob("*.jpg")):
        im = Image.open(ip).convert("RGB")
        W, H = im.size
        d = m.predict(im, threshold=0.001)
        b = np.asarray(d.xyxy, float).reshape(-1, 4) / np.array([W, H, W, H])
        s = np.asarray(d.confidence, float)
        c = np.asarray(d.class_id, int)
        np.add.at(hist, np.clip(c, 0, len(hist) - 1), 1)
        keep = (c >= 0) & (c < len(names))
        preds[ip.stem] = merge_subclasses(np.clip(b[keep], 0, 1), s[keep], c[keep], cmap)
    print("raw class-id histogram:", hist.tolist())
    p = ROOT / "runs" / "pred_cache" / "val" / f"rfdetr__{a.name}@{res_sz}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, preds=np.array(preds, dtype=object))
    print("wrote", p)


if __name__ == "__main__":
    main()
