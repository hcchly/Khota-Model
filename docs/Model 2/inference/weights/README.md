# Model files — v2-2026-11-22

Place the 14 v2 checkpoints (526 MB) in this folder alongside `pipeline.json`. The repository ignores checkpoint binaries. Verify the files against the supplied manifest before running inference.

| file | model | MB |
|---|---|---|
| `classifier_1.pt` | ResNet-18, class-balanced sampling, seed 42 | 44.8 |
| `classifier_2.pt` | ResNet-18, class-balanced sampling, seed 43 | 44.8 |
| `classifier_3.pt` | ResNet-18, class-balanced sampling, seed 44 | 44.8 |
| `classifier_4.pt` | ResNet-18, class-weighted loss, seed 42 | 44.8 |
| `classifier_5.pt` | ResNet-18, class-weighted loss, seed 43 | 44.8 |
| `classifier_6.pt` | ResNet-18, class-weighted loss, seed 44 | 44.8 |
| `detector_1.pt` | YOLO26-s detector, 640 px, seed 42 | 20.3 |
| `detector_2.pt` | YOLO26-s detector, 640 px, seed 43 | 20.3 |
| `detector_3.pt` | YOLO26-s detector, 640 px, seed 44 | 20.3 |
| `detector_4.pt` | YOLOv8n detector, 640 px, seed 42 | 6.2 |
| `detector_5.pt` | YOLOv8n detector, 640 px, seed 43 | 6.2 |
| `detector_6.pt` | YOLOv8n detector, 640 px, seed 44 | 6.2 |
| `detector_7.pt` | YOLO26-m detector, 640 px, seed 42 | 44.0 |
| `detector_8.pth` | RF-DETR Medium detector, 576 px, Surface Damage sub-classes; mirror pass, weight 4 in the fusion | 133.8 |

Check the files against `SHA256SUMS`: `sha256sum -c SHA256SUMS` on Linux or `shasum -a 256 -c SHA256SUMS` on macOS, run from this folder, or `certutil -hashfile <file> SHA256` (Windows).

`pipeline.json` lists these files and holds every threshold; the code reads everything from it.
