# Model files — v1-2026-11-21

The model files (392 MB) are not stored in git. Download **`khota_weights_v1-2026-11-21.zip`** (363 MB) from the repository's **Releases** page (release `v1-2026-11-21`) or from the shared Drive folder, and unzip it in the repository root; it puts the files below into `archive/v1/weights/`, next to `pipeline.json`.

```bash
unzip khota_weights_v1-2026-11-21.zip        # Windows: tar -xf khota_weights_v1-2026-11-21.zip
```

| file | model | MB |
|---|---|---|
| `classifier_1.pt` | ResNet-18, class-balanced sampling, seed 42 | 44.8 |
| `classifier_2.pt` | ResNet-18, class-balanced sampling, seed 43 | 44.8 |
| `classifier_3.pt` | ResNet-18, class-balanced sampling, seed 44 | 44.8 |
| `classifier_4.pt` | ResNet-18, class-weighted loss, seed 42 | 44.8 |
| `classifier_5.pt` | ResNet-18, class-weighted loss, seed 43 | 44.8 |
| `classifier_6.pt` | ResNet-18, class-weighted loss, seed 44 | 44.8 |
| `detector_1.pt` | YOLO26-s detector, seed 42 | 20.3 |
| `detector_2.pt` | YOLO26-s detector, seed 43 | 20.3 |
| `detector_3.pt` | YOLO26-s detector, seed 44 | 20.3 |
| `detector_4.pt` | YOLOv8n detector, seed 42 | 6.2 |
| `detector_5.pt` | YOLOv8n detector, seed 43 | 6.2 |
| `detector_6.pt` | YOLOv8n detector, seed 44 | 6.2 |
| `detector_7.pt` | YOLO26-m detector, seed 42 | 44.0 |

Check the download against `SHA256SUMS`: `sha256sum -c SHA256SUMS` in this folder (Linux / macOS), or `certutil -hashfile <file> SHA256` (Windows).

`pipeline.json` (in git) lists these files and holds every threshold; the code reads everything from it.
