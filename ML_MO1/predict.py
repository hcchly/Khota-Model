"""Run the Khota model on a photo or a folder of photos.

  python predict.py photo.jpg                        # prints the JSON result
  python predict.py photos/ --out results.jsonl       # one JSON line per photo
  python predict.py photo.jpg --draw annotated.jpg    # also save the photo with boxes drawn
  options: --weights weights   --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from khota.pipeline import KhotaPipeline  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--weights", default=str(HERE / "weights"))
    ap.add_argument("--device", help="cpu or cuda (default: GPU if available)")
    ap.add_argument("--out", help="write JSON lines here instead of printing")
    ap.add_argument("--draw", help="save annotated copy (file for one photo, folder for a folder)")
    a = ap.parse_args()
    pipe = KhotaPipeline.from_dir(a.weights, **({"device": a.device} if a.device else {}))
    src = Path(a.source)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXT) if src.is_dir() else [src]
    sink = open(a.out, "w", encoding="utf-8") if a.out else None
    for f in files:
        res = pipe.predict(str(f))
        if sink:
            sink.write(json.dumps(res) + "\n")
        else:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        if a.draw:
            im = ImageOps.exif_transpose(Image.open(f)).convert("RGB")
            d = ImageDraw.Draw(im)
            lw = max(3, im.width // 300)
            for x in res["defects"]:
                c = (230, 30, 30) if x["severity_label"] == "HIGH" else (255, 190, 0)
                d.rectangle(x["box_xyxy"], outline=c, width=lw)
                d.text((x["box_xyxy"][0] + lw, x["box_xyxy"][1] + lw), f"{x['class']} | {x['severity_label']}", fill=c)
            dst = Path(a.draw) / f.name if src.is_dir() else Path(a.draw)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, quality=90)
    if sink:
        sink.close()
        print(f"wrote {len(files)} results to {a.out}")


if __name__ == "__main__":
    main()
