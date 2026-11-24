# Results

| file | what |
|---|---|
| `val_metrics_v2.json` | v2 on the validation split: detection AP per class (also at IoU 0.1 / 0.3), severity on the labelled defects, end to end (`evaluation/scripts/evaluate.py`) |
| `val_severity_predictions_v2.csv` | P(High) and the predicted severity of every labelled defect in validation |
| `threshold_tables_val.xlsx / .json` | effect of every detection-confidence and High threshold on validation (`evaluation/scripts/threshold_tables.py`) |
| `val_runs_detector.csv, val_runs_classifier.csv` | validation metrics of every training run (`evaluation/scripts/summarize_runs.py`) |
| `val_severity_selection.csv` | severity configurations and ensembles compared on validation (`severity/scripts/select_classifier.py`) |
| `detection_report_train.json, _val.json, _test.json` | per split: AP at IoU 0.1 / 0.3 / 0.5, precision and recall at the delivered threshold, detection confusion matrix (`evaluation/scripts/detection_report.py`) |
| `plots/` | precision–recall curves and detection confusion matrices per split; severity confusion matrices |
| `examples/` | annotated validation examples: correct detections, false positives, false negatives, localisation and severity errors, each with its likely cause (also in `examples_val.json`) |
| `benchmark_v2_cpu.json, benchmark_v2_cuda.json` | time per photo and memory, CPU only and GPU (`evaluation/scripts/benchmark.py`) |
| `test_metrics_v2.json` | v2 on the test split, scored once with the thresholds chosen on validation (0.15 / 0.29): detection mAP@0.5 per class, severity, end to end (`evaluation/scripts/evaluate.py --split test`) |
| `test_severity_predictions_v2.csv` | P(High) and the predicted severity of every labelled test defect |

Results of the previous version: `archive/v1/results/`.
