# Khota — sidewalk defect detection and severity classification

**Model v2-2026-11-22** · API 1.2.0 · previous version in [`archive/v1`](archive/v1)

A sidewalk photo goes in; every defect found comes out with its box, type and severity:

1. **Detection** — `Curb Ramp Issue` or `Surface Damage`: an RF-DETR Medium detector (run on the photo and on its
   mirror image) and 7 YOLO detectors; their boxes are fused (weighted box fusion).
2. **Severity** — `Low–Medium` or `High` for each defect: 6 ResNet-18 classifiers on the defect crop (box + 20 % per
   side) and its mirror image, averaged; the High threshold catches at least 80 % of High defects on validation.

`photo → [{box, type, detection confidence, severity, P(High)}, …]` or `"No Target Defect"`

## Results

Validation split (104 photos at locations not used for training), final labels:

| | v1 (7 YOLO) | **v2 (RF-DETR + 7 YOLO)** |
|---|---|---|
| mAP@0.5 Curb Ramp Issue | 0.611 | **0.659** |
| mAP@0.5 Surface Damage | 0.147 | **0.233** |
| mAP@0.5, both classes | 0.379 | **0.446** |
| Severity Macro-F1 (labelled defects) | 0.788 | 0.788 |
| Severity High recall (labelled defects) | 0.81 | 0.81 |

Test split (103 photos), scored once per delivered version:

* **v2: mAP@0.5 0.375** (Curb Ramp 0.557, Surface Damage 0.193); severity Macro-F1 0.761, High recall 0.708, accuracy 0.859; end to end 89% of the photos with a defect flagged, 22% false alarms on photos without one ([`results/test_metrics_v2.json`](results/test_metrics_v2.json)).
* v1: mAP@0.5 0.389 (Curb Ramp 0.610, Surface Damage 0.168); severity Macro-F1 0.753, High recall 0.708 ([`archive/v1/results`](archive/v1/results)).

On test, v2 improves Surface Damage (0.168 → 0.193) but not Curb Ramp (0.610 → 0.557), so the validation gain does not carry over as a whole. v1 was scored before the last relabelling (3 test boxes differ).

Full report: [`docs/Khota_Project_Report.pdf`](docs/Khota_Project_Report.pdf) (also as [`docs/REPORT.md`](docs/REPORT.md)); result files: [`results/`](results).

## Use the model (app integration)

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # or the CPU wheels
pip install -r inference/requirements.txt
unzip khota_weights_v2-2026-11-22.zip     # model files (Releases page); Windows: tar -xf khota_weights_v2-2026-11-22.zip
cd inference
python predict.py examples/sample_photo.jpg                     # one photo -> JSON
python -m uvicorn khota.api:app --host 0.0.0.0 --port 8000      # HTTP API for the app
```

The model files are not in git: see [`inference/weights/README.md`](inference/weights/README.md). The API, the response format, the Python / Flutter API examples and how to update the model: [`inference/INTEGRATION_GUIDE.md`](inference/INTEGRATION_GUIDE.md).

**Speed and memory** (full v2 pipeline on original photos, [`results/benchmark_v2_cpu.json`](results/benchmark_v2_cpu.json)):

* CPU only (11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz): 1.8 s per photo (median), peak RAM 1.7 GB
* GPU (NVIDIA GeForce RTX 3070 Laptop GPU): 0.6 s per photo (median), peak RAM 2.0 GB

**Licenses:** [`docs/LICENSES.md`](docs/LICENSES.md). The YOLO detectors run on Ultralytics, which is AGPL-3.0; RF-DETR is Apache-2.0.

## Reproduce the results

Run everything from the repository root. Training needs an NVIDIA GPU (used: RTX 3070, 8 GB).

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**1. Data.** The photos are not in git: export the three Roboflow datasets into `data/raw/` as described in [`data/README.md`](data/README.md), then build the datasets. The split and the final labels are committed and fixed.

```bash
python data/scripts/05_build_datasets.py          # data/yolo (detector) + data/crops (severity crops)
python data/scripts/12_build_yolo_subclasses.py   # data/yolo3 (RF-DETR)
```

**2. Evaluate the delivered model** (model files in `inference/weights/`):

```bash
python evaluation/scripts/evaluate.py --split val --out results/val_v2   # detection mAP, severity, end to end
python evaluation/scripts/threshold_tables.py --split val                 # results/thresholds/
```

The test split is scored once, after every choice is made on validation: `--split test --out results/test_v2`.

**3. Train again (optional).** The command behind every delivered model file; the full settings of each run are in `detection/configs/` and `severity/configs/`. GPU training is not bit-for-bit repeatable, so retrained models land within the usual run-to-run spread rather than on identical numbers. Retraining uses the final labels; the delivered YOLO detectors and severity classifiers were trained on the reviewed labels of 18 Nov 2026, RF-DETR on the final labels ([`data/README.md`](data/README.md)).

| file | model | command |
|---|---|---|
| `detector_1.pt` | YOLO26-s detector, 640 px, seed 42 | `python detection/scripts/train_detector.py --name y26s640_s42 --model yolo26s.pt --imgsz 640 --epochs 120 --patience 40 --seed 42 --project runs/detector_v2` |
| `detector_2.pt` | YOLO26-s detector, 640 px, seed 43 | `python detection/scripts/train_detector.py --name y26s640_s43 --model yolo26s.pt --imgsz 640 --epochs 120 --patience 40 --seed 43 --project runs/detector_v2` |
| `detector_3.pt` | YOLO26-s detector, 640 px, seed 44 | `python detection/scripts/train_detector.py --name y26s640_s44 --model yolo26s.pt --imgsz 640 --epochs 120 --patience 40 --seed 44 --project runs/detector_v2` |
| `detector_4.pt` | YOLOv8n detector, 640 px, seed 42 | `python detection/scripts/train_detector.py --name v8n640_s42 --model yolov8n.pt --imgsz 640 --epochs 120 --patience 40 --seed 42 --project runs/detector_v2` |
| `detector_5.pt` | YOLOv8n detector, 640 px, seed 43 | `python detection/scripts/train_detector.py --name v8n640_s43 --model yolov8n.pt --imgsz 640 --epochs 120 --patience 40 --seed 43 --project runs/detector_v2` |
| `detector_6.pt` | YOLOv8n detector, 640 px, seed 44 | `python detection/scripts/train_detector.py --name v8n640_s44 --model yolov8n.pt --imgsz 640 --epochs 120 --patience 40 --seed 44 --project runs/detector_v2` |
| `detector_7.pt` | YOLO26-m detector, 640 px, seed 42 | `python detection/scripts/train_detector.py --name y26m640_s42 --model yolo26m.pt --imgsz 640 --epochs 120 --patience 40 --batch 8 --seed 42 --project runs/detector_v2` |
| `detector_8.pth` | RF-DETR Medium detector, 576 px, Surface Damage sub-classes; mirror pass, weight 4 in the fusion | `python detection/scripts/train_rfdetr.py --size medium --name med_c3_s42` |
| `classifier_1.pt` | ResNet-18, class-balanced sampling, seed 42 | `python severity/scripts/train_classifier.py --name r18_balanced --loss ce --sampler balanced --seed 42 --out runs/classifier_v2` |
| `classifier_2.pt` | ResNet-18, class-balanced sampling, seed 43 | `python severity/scripts/train_classifier.py --name r18_balanced --loss ce --sampler balanced --seed 43 --out runs/classifier_v2` |
| `classifier_3.pt` | ResNet-18, class-balanced sampling, seed 44 | `python severity/scripts/train_classifier.py --name r18_balanced --loss ce --sampler balanced --seed 44 --out runs/classifier_v2` |
| `classifier_4.pt` | ResNet-18, class-weighted loss, seed 42 | `python severity/scripts/train_classifier.py --name r18_default --seed 42 --out runs/classifier_v2` |
| `classifier_5.pt` | ResNet-18, class-weighted loss, seed 43 | `python severity/scripts/train_classifier.py --name r18_default --seed 43 --out runs/classifier_v2` |
| `classifier_6.pt` | ResNet-18, class-weighted loss, seed 44 | `python severity/scripts/train_classifier.py --name r18_default --seed 44 --out runs/classifier_v2` |

Then assemble `inference/weights/` (thresholds chosen on validation):

```bash
python evaluation/scripts/export_weights.py \
  --detector runs/detector_v2/y26s640_s42 runs/detector_v2/y26s640_s43 runs/detector_v2/y26s640_s44 runs/detector_v2/v8n640_s42 runs/detector_v2/v8n640_s43 runs/detector_v2/v8n640_s44 runs/detector_v2/y26m640_s42 \
  --rfdetr runs/rfdetr/med_c3_s42 --rfdetr-flip --rfdetr-weight 4 \
  --classifiers runs/classifier_v2/r18_balanced_s42 runs/classifier_v2/r18_balanced_s43 runs/classifier_v2/r18_balanced_s44 runs/classifier_v2/r18_default_s42 runs/classifier_v2/r18_default_s43 runs/classifier_v2/r18_default_s44 \
  --high-threshold 0.29 --det-conf 0.15 --tta --model-version v2-2026-11-22 --out inference/weights
```

## Repository layout

```
khota-model/
├── README.md
├── requirements.txt          training + evaluation environment
├── inference/                deployable package (API, command line, app examples)
│   ├── khota/                pipeline.py, api.py, ensemble.py, classifier.py, crops.py
│   ├── predict.py
│   ├── requirements.txt      inference only
│   ├── INTEGRATION_GUIDE.md
│   ├── examples/             sample_photo.jpg, sample responses, Python and Flutter API examples
│   └── weights/              pipeline.json (v2-2026-11-22), SHA256SUMS, README.md (model-file download)
├── data/
│   ├── README.md             how to get the photos and build the datasets
│   ├── splits/               split.csv (frozen location split), split_report.json
│   ├── annotations/          Khota_Model_Final_reviewed.xlsx, boxes_final.csv, images_final.csv, change_log.csv
│   └── scripts/              dataset builders, incl. the severity crop extraction
├── detection/
│   ├── configs/              training settings of the 8 delivered detectors
│   └── scripts/              training + evaluation
├── severity/
│   ├── configs/              training settings of the 6 delivered classifiers
│   └── scripts/              training + model selection
├── evaluation/
│   └── scripts/              end-to-end evaluation, threshold tables, model export
├── results/                  test + validation results, threshold tables, training runs, plots/, examples/
├── docs/                     Khota_Project_Report.pdf, REPORT.md (same content), LICENSES.md
└── archive/
    └── v1/                   previous version (v1-2026-11-21), kept for reference
```

## Scripts

| script | what it does |
|---|---|
| `data/scripts/05_build_datasets.py` | Builds `data/yolo` (photos downscaled to 1280 px + YOLO labels, per split) and the severity crops `data/crops` (box + 20 % per side) from the final labels and the frozen split |
| `data/scripts/12_build_yolo_subclasses.py` | Builds `data/yolo3`: Surface Damage as Low–Medium / High sub-classes, used to train RF-DETR (merged back into the two output classes at prediction) |
| `data/scripts/13_split_report.py` | Counts per split (photos, photos without a defect, boxes by class and by severity) for the final labels and the earlier label versions -> `data/splits/split_report.json` |
| `detection/scripts/train_detector.py` | Trains one YOLO detector; chooses its detection threshold on validation |
| `detection/scripts/train_rfdetr.py` | Fine-tunes RF-DETR on `data/yolo3`; caches its validation predictions |
| `detection/scripts/ap_by_iou.py` | The AP code behind every detection result (101-point AP; IoU 0.1 / 0.3 / 0.5) |
| `detection/scripts/detector_ensemble_val.py` | Scores YOLO detectors and their fused ensembles on validation |
| `detection/scripts/pred_cache.py` | Caches detector predictions per split, so ensembles can be compared on the CPU |
| `detection/scripts/ens_eval.py` | Scores ensembles and fusion settings from the cached predictions (validation) |
| `severity/scripts/train_classifier.py` | Trains one severity classifier on the crops; checkpoint chosen on validation |
| `severity/scripts/select_classifier.py` | Compares severity configurations and ensembles on validation; chooses the High threshold |
| `evaluation/scripts/evaluate.py` | Evaluates the delivered pipeline on one split: detection mAP per class, severity on the labelled defects, end to end on the original photos |
| `evaluation/scripts/detection_report.py` | Precision–recall curves, detection confusion matrix, AP at IoU 0.1 / 0.3 / 0.5 and annotated examples of correct detections and of each error type |
| `evaluation/scripts/threshold_tables.py` | Threshold tables (detection confidence, High threshold) for the delivered pipeline |
| `evaluation/scripts/benchmark.py` | Inference time per photo and memory of the delivered pipeline (CPU or GPU) |
| `evaluation/scripts/export_weights.py` | Assembles `inference/weights` (model files + `pipeline.json`) from trained runs |
| `evaluation/scripts/seed_replicates.py` | Seed-to-seed variation of the chosen recipes (mean ± std) |
| `evaluation/scripts/summarize_runs.py` | Collects the validation metrics of every training run into one table |

The inference code itself (box fusion, crops, severity, API) is in `inference/khota/`.

## Notes

* **Split.** `data/splits/split.csv` is frozen: photos within 200 m of each other are always in the same split, so no location appears in two splits. Every result uses it.
* **Validation and test.** Every model and threshold choice was made on the validation split; the test split is scored once per delivered version.
* **Not in git** (`.gitignore`): photos, generated datasets, training runs and model files.
