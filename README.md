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

The Riyadh dataset contains 689 images and 1,010 annotated boxes.
Images were collected for the project,
faces and licence plates were blurred, and access is limited to the project
team, so the images are **not** included in this repository.



