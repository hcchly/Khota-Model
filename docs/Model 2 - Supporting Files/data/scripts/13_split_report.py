"""Step 13: dataset counts per split from the labels, for every label version -> data/splits/split_report.json

The split itself (data/splits/split.csv) is frozen and is not changed here; only the counts are recomputed.

Label versions
  final      data/annotations/boxes_final.csv: requester labels + all 160 logged changes (22 Nov 2026)
  reviewed   the labels of 18 Nov 2026 (requester labels + the first 111 changes), reconstructed exactly from the final
             labels by undoing the 49 Surface Damage relabelling changes in data/annotations/change_log.csv
  original   the labels as delivered (counts from the first split report, 15 Nov 2026)

Per split: photos, photos with / without a target defect, location groups, boxes, boxes by class, by severity and by
class x severity.
Usage: python data/scripts/13_split_report.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINAL = ROOT / "data" / "annotations"
SPLIT = ROOT / "data" / "splits" / "split.csv"
OUT = ROOT / "data" / "splits" / "split_report.json"
CHANGES = ROOT / "data" / "annotations" / "change_log.csv"
SPLITS = ("train", "val", "test")
CLS = {"CR": "Curb Ramp Issue", "SD": "Surface Damage"}
SEV = {"HIGH": "High", "LM": "Low-Medium"}
ORIGINAL = {  # first split report (labels as delivered, 15 Nov 2026): boxes, High boxes, Curb Ramp boxes, empty photos
    "train": {"boxes": 713, "high": 109, "curb_ramp": 111, "photos_without_defect": 135},
    "val": {"boxes": 143, "high": 24, "curb_ramp": 26, "photos_without_defect": 27},
    "test": {"boxes": 154, "high": 22, "curb_ramp": 23, "photos_without_defect": 32},
}


def counts(boxes: list, photos: dict, split: dict, groups: dict) -> dict:
    """boxes: (image_id, class name, severity) ; photos: image_id -> number of boxes."""
    out = {}
    for s in SPLITS + ("total",):
        ids = [i for i in photos if s == "total" or split[i] == s]
        bx = [b for b in boxes if s == "total" or split[b[0]] == s]
        cs = Counter((b[1], b[2]) for b in bx)
        out[s] = {
            "photos": len(ids),
            "photos_with_defect": sum(photos[i] > 0 for i in ids),
            "photos_without_defect": sum(photos[i] == 0 for i in ids),
            "location_groups": len({groups[i] for i in ids}),
            "boxes": len(bx),
            "boxes_by_class": {c: sum(v for (k, _), v in cs.items() if k == c) for c in CLS.values()},
            "boxes_by_severity": {v_: sum(v for (_, k), v in cs.items() if k == v_) for v_ in SEV.values()},
            "boxes_by_class_and_severity": {c: {v_: cs[(c, v_)] for v_ in SEV.values()} for c in CLS.values()},
        }
    return out


def main():
    rows = list(csv.DictReader(open(SPLIT, encoding="utf-8")))
    split = {r["image_id"]: r["split"] for r in rows}
    groups = {r["image_id"]: r["location_group"] for r in rows}
    images = list(csv.DictReader(open(FINAL / "images_final.csv", encoding="utf-8")))
    final = list(csv.DictReader(open(FINAL / "boxes_final.csv", encoding="utf-8")))
    assert all(b["severity"] in SEV for b in final), "undecided severities left"
    fin_boxes = [(b["image_id"], b["class_name"], SEV[b["severity"]]) for b in final]
    fin_photos = {r["image_id"]: 0 for r in images}
    for b in final:
        fin_photos[b["image_id"]] += 1
    assert all(int(r["n_boxes"]) == fin_photos[r["image_id"]] for r in images)

    # reviewed labels (18 Nov) = final labels with the 49 relabelling changes undone
    by_key = {(b["image_id"], b["box"]): b for b in final}
    rev = [list(x) for x in fin_boxes]
    idx = {(b["image_id"], b["box"]): k for k, b in enumerate(final)}
    rev_photos, dropped = dict(fin_photos), set()
    relabel = [c for c in csv.DictReader(open(CHANGES, encoding="utf-8")) if c["source"].startswith("relabel")]
    for c in relabel:
        key = (c["image_id"], c["box"])
        if c["type"] == "delete":  # box existed before the relabelling
            k, sv = c["old"].split()
            rev.append([c["image_id"], CLS[k], SEV[sv]])
            rev_photos[c["image_id"]] += 1
        elif c["type"] == "add":  # box did not exist before
            k, sv = c["new"].split()
            b = by_key[key]
            assert (b["class_name"], b["severity"]) == (CLS[k], sv), c
            dropped.add(idx[key])
            rev_photos[c["image_id"]] -= 1
        elif c["type"] == "class":
            b = by_key[key]
            assert b["class_name"] == CLS[c["new"]], c
            rev[idx[key]][1] = CLS[c["old"]]
        elif c["type"] == "severity":
            b = by_key[key]
            assert b["severity"] == c["new"], c
            rev[idx[key]][2] = SEV[c["old"]]
        else:
            assert c["type"] == "geometry", c  # box redrawn: no change in the counts
    rev = [tuple(x) for k, x in enumerate(rev) if k not in dropped]

    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    old = old.get("split_method", old)  # first report: top level; regenerated report: "split_method"
    method = {k: old[k] for k in ("link_m", "seed", "trials", "n_groups", "largest_group_images",
                                  "min_cross_split_distance_m") if k in old}
    orig = {s: dict(v) for s, v in ORIGINAL.items()}
    orig["total"] = {k: sum(v[k] for v in ORIGINAL.values()) for k in ORIGINAL["train"]}
    report = {
        "split_method": method | {"note": "frozen location split (data/splits/split.csv); photos within link_m metres "
                                          "of each other are always in the same split"},
        "labels": "final labels (data/annotations/boxes_final.csv): requester labels + all 160 logged changes, 22 Nov 2026",
        "per_split": counts(fin_boxes, fin_photos, split, groups),
        "label_versions": {
            "final_2026-11-22": "per_split above (160 logged changes)",
            "reviewed_2026-11-18": {"note": "requester labels + the first 111 logged changes; the final labels with the 49 "
                                            "relabelling changes of 22 Nov undone",
                                    "per_split": counts(rev, rev_photos, split, groups)},
            "original_as_delivered": {"note": "labels as delivered (first split report, 15 Nov 2026)",
                                      "per_split": orig},
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, per in (("final", report["per_split"]), ("reviewed", report["label_versions"]["reviewed_2026-11-18"]["per_split"])):
        for s in SPLITS + ("total",):
            d = per[s]
            print(f"{name:9s} {s:5s} photos {d['photos']:3d} (with defect {d['photos_with_defect']:3d}, without "
                  f"{d['photos_without_defect']:3d}, groups {d['location_groups']:3d}) boxes {d['boxes']:4d} | "
                  f"CR {d['boxes_by_class']['Curb Ramp Issue']:3d} ({d['boxes_by_class_and_severity']['Curb Ramp Issue']['High']} High) "
                  f"SD {d['boxes_by_class']['Surface Damage']:3d} ({d['boxes_by_class_and_severity']['Surface Damage']['High']} High) "
                  f"| High {d['boxes_by_severity']['High']:3d} LM {d['boxes_by_severity']['Low-Medium']:3d}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
