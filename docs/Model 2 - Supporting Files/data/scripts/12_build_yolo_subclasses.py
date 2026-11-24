"""Step 12: detector dataset with Surface Damage split by severity (sub-classes), for training only.

Surface Damage mixes hairline cracks (Low-Medium) with missing paving and holes (High), which look very different.
A detector may learn the two visually consistent sub-classes more easily than the mixed class. The sub-classes are
merged back into the two requested output classes at prediction time (khota.ensemble.merge_subclasses, driven by the model's
class names "<type>|<severity>"), so outputs, metrics and the API stay exactly as before.

Classes: 0 Curb Ramp Issue | 1 Surface Damage|LM | 2 Surface Damage|HIGH   (Curb Ramp is not split: 39 High boxes)
Images are the same downscaled photos as data/yolo (copied); labels come from data/annotations/boxes_final.csv in the
same line order as data/yolo, so only the class id differs.

Usage: python data/scripts/12_build_yolo_subclasses.py [--out data/yolo3]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "yolo"
NAMES = ["Curb Ramp Issue", "Surface Damage|LM", "Surface Damage|HIGH"]


def sub_class(b: dict) -> int:
    if int(b["class_id"]) == 0:
        return 0
    return 2 if b["severity"] == "HIGH" else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "yolo3"))
    out = Path(ap.parse_args().out)
    boxes = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "data" / "annotations" / "boxes_final.csv", encoding="utf-8")):
        boxes[r["image_id"]].append(r)
    if out.exists():
        shutil.rmtree(out)
    stats = Counter()
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        for img in sorted((SRC / split / "images").glob("*.jpg")):
            iid = img.stem
            src_lines = [l for l in (SRC / split / "labels" / f"{iid}.txt").read_text().splitlines() if l.strip()]
            bx = boxes.get(iid, [])
            assert len(src_lines) == len(bx), f"{iid}: {len(src_lines)} label lines vs {len(bx)} boxes"
            lines = []
            for line, b in zip(src_lines, bx):
                c, *coords = line.split()
                assert int(c) == int(b["class_id"]), f"{iid}: class mismatch"
                # same geometry as data/yolo (checked), only the class id changes
                assert abs(float(coords[0]) - float(b["xc"])) < 1e-5 and abs(float(coords[2]) - float(b["w"])) < 1e-5, iid
                k = sub_class(b)
                lines.append(" ".join([str(k)] + coords))
                stats[(split, NAMES[k])] += 1
            shutil.copy2(img, out / split / "images" / img.name)
            (out / split / "labels" / f"{iid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    names = "\n".join(f"  {i}: \"{n}\"" for i, n in enumerate(NAMES))
    (out / "khota3.yaml").write_text(f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                                     f"names:\n{names}\n")
    # data.yaml: the same dataset for RF-DETR (train_rfdetr.py / export_weights.py read the one-line names list)
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
                                   f"nc: {len(NAMES)}\nnames: {json.dumps(NAMES)}\n")
    for k in sorted(stats):
        print(k, stats[k])


if __name__ == "__main__":
    main()
