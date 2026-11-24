"""Step 5 of data preparation: build the training datasets from the corrected labels.

Inputs
  data/annotations/boxes_final.csv   one row per final box (output of step 4)
      image_id, box, class_id, xc, yc, w, h, severity (HIGH|LM|UNDECIDED), batch, file
      UNDECIDED boxes go into the detector data only (no severity crop)
  data/annotations/images_final.csv  one row per image used (annotated + No Target Defect)
      image_id, batch, file, n_boxes
  data/splits/split.csv               frozen location-based split

Outputs
  data/yolo/{train,val,test}/images/*.jpg   images resized to <= 1280 px long side (labels are
  data/yolo/{train,val,test}/labels/*.txt   normalised, so they are unchanged); empty label file for
  data/yolo/khota.yaml                      No-Target-Defect images (background examples)
  data/crops/{train,val,test}/{LM,HIGH}/<image>_<box>.jpg   CONTEXT crops: the box expanded 60% per
                                            side (clipped to the image), long side <= 800 px. The
                                            classifier loader cuts the exact 20%-expanded crop out of
                                            this (optionally with box jitter for training) and resizes
                                            it to 224x224, so it sees the same crop as the pipeline.
  data/crops/crops.csv                      index of crops: split, class, severity and the box
                                            position (bx0,by0,bx1,by1 px) inside the context crop

Usage:
  python data/scripts/05_build_datasets.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "inference"))
from khota.crops import expand_box_xyxy, yolo_to_xyxy  # noqa: E402

RAW = ROOT / "data" / "raw"
FINAL = ROOT / "data" / "annotations"
SPLIT = ROOT / "data" / "splits" / "split.csv"
YOLO = ROOT / "data" / "yolo"
CROPS = ROOT / "data" / "crops"
YOLO_LONG = 1280
CONTEXT = 0.60
CROP_LONG = 800
CLASS_NAMES = ["Curb Ramp Issue", "Surface Damage"]  # class ids as in the Roboflow export


def build_one(job):
    iid, batch, fname, split, boxes, yolo_dir, crops_dir = job
    YOLO, CROPS = Path(yolo_dir), Path(crops_dir)  # passed explicitly: spawned workers don't see main()'s globals
    im = ImageOps.exif_transpose(Image.open(RAW / f"batch{batch}" / "train" / "images" / fname)).convert("RGB")
    W, H = im.size
    # YOLO image + label
    s = min(1.0, YOLO_LONG / max(W, H))
    small = im.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else im
    small.save(YOLO / split / "images" / f"{iid}.jpg", quality=95)
    with open(YOLO / split / "labels" / f"{iid}.txt", "w") as f:
        for b in boxes:
            f.write(f"{int(b['class_id'])} {float(b['xc']):.6f} {float(b['yc']):.6f} {float(b['w']):.6f} {float(b['h']):.6f}\n")
    # context crops from the ORIGINAL resolution image
    rows = []
    for b in boxes:
        if b["severity"] not in ("LM", "HIGH"):  # no severity label -> detector only
            continue
        x0, y0, x1, y1 = yolo_to_xyxy(float(b["xc"]), float(b["yc"]), float(b["w"]), float(b["h"]), W, H)
        cx0, cy0, cx1, cy1 = expand_box_xyxy(x0, y0, x1, y1, W, H, expand=CONTEXT)
        c = im.crop((cx0, cy0, cx1, cy1))
        cs = min(1.0, CROP_LONG / max(c.size))
        if cs < 1:
            c = c.resize((max(1, round(c.size[0] * cs)), max(1, round(c.size[1] * cs))), Image.LANCZOS)
        rel = Path(split) / b["severity"] / f"{iid}_{b['box']}.jpg"
        c.save(CROPS / rel, quality=95)
        rows.append({"crop": rel.as_posix(), "image_id": iid, "box": b["box"], "split": split,
                     "class_id": b["class_id"], "severity": b["severity"],
                     # box inside the saved context crop, in its pixel coordinates
                     "bx0": round((x0 - cx0) * cs, 2), "by0": round((y0 - cy0) * cs, 2),
                     "bx1": round((x1 - cx0) * cs, 2), "by1": round((y1 - cy0) * cs, 2),
                     # where the full image would end, in context-crop pixels (for clipping the 20% crop)
                     "img_x0": round(-cx0 * cs, 2), "img_y0": round(-cy0 * cs, 2),
                     "img_x1": round((W - cx0) * cs, 2), "img_y1": round((H - cy0) * cs, 2),
                     "box_w_px": round(x1 - x0), "box_h_px": round(y1 - y0)})
    return rows


def main():
    global YOLO, CROPS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", default=str(FINAL), help="folder with boxes_final.csv / images_final.csv")
    ap.add_argument("--yolo-out", default=str(YOLO))
    ap.add_argument("--crops-out", default=str(CROPS))
    ap.add_argument("--limit", type=int, default=0, help="only first N images per split (smoke tests)")
    a = ap.parse_args()
    final, YOLO, CROPS = Path(a.final), Path(a.yolo_out), Path(a.crops_out)

    split = {r["image_id"]: r["split"] for r in csv.DictReader(open(SPLIT))}
    images = list(csv.DictReader(open(final / "images_final.csv", encoding="utf-8")))
    if a.limit:
        per = defaultdict(list)
        for im in images:
            per[split[im["image_id"]]].append(im)
        images = [im for v in per.values() for im in v[:a.limit]]
    boxes = defaultdict(list)
    for r in csv.DictReader(open(final / "boxes_final.csv", encoding="utf-8")):
        assert r["severity"] in ("HIGH", "LM", "UNDECIDED"), r
        boxes[r["image_id"]].append(r)

    for d in (YOLO, CROPS):
        if d.exists():
            shutil.rmtree(d)
    for s in ("train", "val", "test"):
        (YOLO / s / "images").mkdir(parents=True)
        (YOLO / s / "labels").mkdir(parents=True)
        for sev in ("LM", "HIGH"):
            (CROPS / s / sev).mkdir(parents=True)

    jobs = []
    for im in images:
        iid = im["image_id"]
        assert iid in split, f"{iid} has no split"
        jobs.append((iid, int(im["batch"]), im["file"], split[iid], boxes.get(iid, []), str(YOLO), str(CROPS)))
    crop_rows = []
    with ProcessPoolExecutor(8) as ex:
        for rows in ex.map(build_one, jobs, chunksize=4):
            crop_rows.extend(rows)

    with open(CROPS / "crops.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(crop_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(crop_rows, key=lambda r: r["crop"]))
    (YOLO / "khota.yaml").write_text(
        f"path: {YOLO.as_posix()}\ntrain: train/images\nval: val/images\ntest: test/images\n"
        f"names:\n  0: {CLASS_NAMES[0]}\n  1: {CLASS_NAMES[1]}\n")

    print("images per split:", Counter(j[3] for j in jobs))
    print("empty images per split:", Counter(j[3] for j in jobs if not j[4]))
    print("boxes per split/class:", Counter((r["split"], CLASS_NAMES[int(r["class_id"])]) for r in crop_rows))
    print("crops per split/severity:", Counter((r["split"], r["severity"]) for r in crop_rows))


if __name__ == "__main__":
    main()
