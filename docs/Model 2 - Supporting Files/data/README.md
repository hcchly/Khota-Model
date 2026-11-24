# Data

| path | in git | what |
|---|---|---|
| `splits/split.csv` | yes | frozen train / val / test split, one row per photo; photos within 200 m of each other share a split |
| `splits/split_report.json` | yes | photos, boxes and High boxes per split |
| `annotations/Khota_Model_Final_reviewed.xlsx` | yes | the project workbook; every label change is logged in its `Change_Log` sheet |
| `annotations/boxes_final.csv` | yes | final labels, one row per box: photo, type, severity, box (YOLO format, normalised) |
| `annotations/images_final.csv` | yes | every photo used, including photos without a target defect |
| `annotations/change_log.csv` | yes | the 160 label changes (same as the `Change_Log` sheet) |
| `scripts/` | yes | dataset builders |
| `raw/batch1/` … `raw/batch3/` | no | the photos (Roboflow exports, below) |
| `yolo/`, `yolo3/`, `crops/` | no | built by the scripts |

## Counts (final labels)

High-severity boxes in brackets. The same counts are in `splits/split_report.json`, the report and the README.

| split | photos | with a defect | without a defect | location groups | Curb Ramp boxes | Surface Damage boxes | all boxes |
|---|---|---|---|---|---|---|---|
| train | 482 | 345 | 137 | 112 | 112 (28) | 620 (115) | 732 (143) |
| val | 104 | 77 | 27 | 43 | 27 (6) | 120 (20) | 147 (26) |
| test | 103 | 71 | 32 | 28 | 23 (5) | 133 (19) | 156 (24) |
| total | 689 | 493 | 196 | 183 | 162 (39) | 873 (154) | 1,035 (193) |

## Labels used by each delivered model

Every v2 result is scored on the final labels (`annotations/boxes_final.csv`, 22 Nov 2026). The delivered models were trained on these label versions; the reviewed labels of 18 Nov 2026 are the final labels without the 49 relabelling changes in `annotations/change_log.csv` (source “relabel in Roboflow”). Checkpoints were chosen on the validation photos of the same version.

| delivered model | trained on | training boxes (train split, 482 photos) |
|---|---|---|
| `detector_1–7` (YOLO) | reviewed labels, 18 Nov 2026 | 739: Curb Ramp 114, Surface Damage 625; 136 photos without a defect as background |
| `detector_8` (RF-DETR) | final labels, 22 Nov 2026 | 732: Curb Ramp 112, Surface Damage Low–Medium 505, Surface Damage High 115; 137 photos without a defect |
| `classifier_1–6` (ResNet-18) | reviewed labels, 18 Nov 2026 | 739 defect crops: Low–Medium 597, High 142 |

Retraining from this repository uses the final labels.


## 1. Photos

Export these three dataset versions from Roboflow in **YOLO26** format and unzip each export into its folder:

| folder | Roboflow project | version |
|---|---|---|
| `data/raw/batch1/` | khota-batch1 | v5 (batch1_team_reviewed2) |
| `data/raw/batch2/` | khota-batch2 | v6 (batch2_edited) |
| `data/raw/batch3/` | khota-batch3 | v4 (batch3_team_reviewed2) |

Each folder then holds `data.yaml` and `train/images/*.jpg`; the file names in `annotations/images_final.csv` are the
ones of these versions. Only the photos are taken from the exports; the labels come from `annotations/boxes_final.csv`.

## 2. Build the datasets

```bash
python data/scripts/05_build_datasets.py          # data/yolo  (detector) + data/crops (severity crops)
python data/scripts/12_build_yolo_subclasses.py   # data/yolo3 (RF-DETR: Surface Damage as Low–Medium / High)
```
