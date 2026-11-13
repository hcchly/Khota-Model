# Khota Model | خُطى

Computer vision models for **Khota**, a graduation project at King Saud University
for citizen reporting of sidewalk and curb ramp issues in Riyadh.

The pipeline has two stages:

1. **Detection** – a YOLOv8 detector localizes two issue classes:
   *Surface Damage* and *Curb Ramp Issue*.
2. **Severity classification** – a ResNet-18 classifier assigns each detected
   issue a severity of *Low–Medium* or *High*, using a crop around its box.

The mobile application is in a separate repository:
https://github.com/hcchly/Khota

## Team

- Haifa Alromaih
- Hala Alduhayan
- Rand Aldahmash
- Rahaf Alshalwi

Supervisor: Prof. Reham Alabduljabbar

## Repository Structure

```
Khota-Model/
├── data/                 Dataset files (not included; see "Data")
│   ├── raw/              Images
│   ├── annotations/      YOLO labels and the severity workbook
│   └── splits/           Location-grouped train / val / test lists
├── detection/
│   ├── configs/          YOLOv8 training settings
│   └── scripts/          Training and evaluation scripts
├── severity/
│   ├── configs/          Classifier settings (runs A, B, C)
│   └── scripts/          Crop extraction, training, and evaluation
├── scripts/              Shared tools (label checks, dataset split)
├── notebooks/            Exploration and analysis notebooks
├── results/
│   ├── detection/        Detection metrics
│   └── severity/         Severity metrics and the selected model summary
├── docs/                 Severity rubric and reason codes
├── requirements.txt
└── README.md
```

## Data

The Riyadh dataset contains 689 images and 1,010 annotated boxes
(850 Surface Damage, 160 Curb Ramp Issue; 872 Low–Medium, 138 High) and
194 images with no target defect. Images were collected for the project,
faces and licence plates were blurred, and access is limited to the project
team, so the images are **not** included in this repository.

Expected layout once the data is available:

```
data/raw/            RYD-0001.jpg, RYD-0002.jpg, ...
data/annotations/    YOLO .txt labels and Khota_Model_Final.xlsx
data/splits/         train.txt, val.txt, test.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Split

Images within 50 m of each other are chained into location groups
(272 groups), and whole groups are assigned to train / validation / test
in an approximately 70 / 15 / 15 ratio (seed 42). The test split is locked
and is not used for training, tuning, threshold selection, or model choice.

## Severity Classification

- Input: box expanded by 20% on each side, cropped from the original image,
  resized to 224 × 224, ImageNet normalization.
- Model: ImageNet-pretrained ResNet-18 with a two-class head.
- Training: AdamW (lr 1e-4, weight decay 1e-4), batch size 32, up to 20 epochs,
  early stopping after 5 epochs without improvement in validation macro-F1.
- Augmentation (training only): horizontal flip, rotation ±10°,
  brightness and contrast ±20%.
- Selected configuration (Run C): augmentation + weighted random sampler,
  uniform loss weights, decision threshold 0.50, seeds 42–44.

## Detection

- Model: COCO-pretrained YOLOv8n, input size 640, batch size 8, AdamW,
  up to 100 epochs with early stopping (patience 20).
- Background images (no target defect) are included in training.
- Main metric: mAP@0.5, with precision, recall, and mAP@0.5:0.95 per class.

## Results (validation, preliminary)

| Model | Metric | Value |
|---|---|---|
| Majority baseline | Macro-F1 | 0.465 |
| Severity – Run C (3 seeds) | Macro-F1 | 0.798 ± 0.005 |

Results on the locked test split will be added after the final evaluation.

## Repository

https://github.com/hcchly/Khota-model
