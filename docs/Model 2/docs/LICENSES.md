# Licenses and notices

Licenses of the delivered models, of the software they run on and of the data behind them. The versions are the ones in `requirements.txt` and `inference/requirements.txt`.

## Models

| model | origin | license |
|---|---|---|
| detector_1–7.pt — YOLO detectors (YOLO26-s, YOLOv8n, YOLO26-m) | fine-tuned from Ultralytics COCO-pretrained weights | AGPL-3.0 (Ultralytics; Enterprise License available) |
| detector_8.pth — RF-DETR Medium detector | fine-tuned from Roboflow's RF-DETR Medium weights (DINOv2 backbone by Meta) | Apache-2.0 |
| classifier_1–6.pt — ResNet-18 severity classifiers | fine-tuned from torchvision ImageNet-1k weights | BSD-3-Clause (torchvision) |

## Software

| package | version | license | used for |
|---|---|---|---|
| ultralytics | 8.4.84 | AGPL-3.0 | inference (YOLO detectors) |
| ultralytics-thop | 2.2.5 | AGPL-3.0 | inference (dependency of ultralytics) |
| rfdetr | 1.12.1 | Apache-2.0 | inference (RF-DETR detector) |
| transformers | 5.24.0 | Apache-2.0 | inference (dependency of rfdetr) |
| supervision | 0.32.0 | MIT | inference (dependency of rfdetr) |
| pydeprecate | 0.9.0 | Apache-2.0 | inference (dependency of rfdetr) |
| torch | 2.6.0 | BSD-3-Clause | inference + training |
| torchvision | 0.21.0 | BSD-3-Clause | inference + training |
| numpy | 2.4.6 | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | inference + training |
| pillow | 13.0.0 | MIT-CMU | inference + training |
| opencv-python | 5.0.0.93 | Apache-2.0 | inference (dependency of ultralytics) |
| polars | 2.1.0 | MIT | inference (dependency of ultralytics) |
| matplotlib | 3.11.2 | Matplotlib License (PSF-based) | inference (dependency of ultralytics) + plots |
| pyyaml | 6.0.3 | MIT | inference + training |
| psutil | 7.2.2 | BSD-3-Clause | inference (dependency of ultralytics) + benchmark |
| requests | 2.34.2 | Apache-2.0 | inference (dependency) |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | inference (dependency) |
| fastapi | 0.163.1 | MIT | HTTP API |
| starlette | 1.10.0 | BSD-3-Clause | HTTP API (dependency of fastapi) |
| pydantic | 2.14.3 | MIT | HTTP API (dependency of fastapi) |
| uvicorn | 0.54.0 | BSD-3-Clause | HTTP API server |
| python-multipart | 0.0.33 | Apache-2.0 | HTTP API (photo upload) |
| scipy | 1.17.1 | BSD-3-Clause | training / evaluation |
| scikit-learn | 1.9.1 | BSD-3-Clause | training / evaluation |
| pandas | 3.1.1 | BSD-3-Clause | training / evaluation |
| openpyxl | 3.1.5 | MIT | training / evaluation (Excel files) |
| pytorch-lightning | 2.6.6 | Apache-2.0 | training of RF-DETR (rfdetr[train]) |
| torchmetrics | 1.8.2 | Apache-2.0 | training of RF-DETR (rfdetr[train]) |
| peft | 0.22.0 | Apache-2.0 | training of RF-DETR (rfdetr[train]) |
| pycocotools | 2.0.11 | BSD-2-Clause | training of RF-DETR (rfdetr[train]) |

## Data

| data | terms | used for |
|---|---|---|
| Khota photos and labels | data owned by its original provider | training and evaluation of every delivered model |
| COCO | annotations CC BY 4.0; images under the Flickr terms of use | pre-training of the Ultralytics and RF-DETR weights by their publishers |
| ImageNet-1k | ImageNet terms of access | pre-training of the torchvision ResNet-18 weights by their publisher |
| RDD2022 | CC BY 4.0 | one experiment only (report 3.11); not used by any delivered model |

## Required notices

* **Ultralytics (AGPL-3.0).** The YOLO detectors 1–7 run on the Ultralytics package and were fine-tuned from Ultralytics weights, which Ultralytics distributes under AGPL-3.0. Under AGPL-3.0, software that includes this code and is distributed, or offered to users over a network (for example the app's API), must make its complete source code available under AGPL-3.0. For a closed-source product, Ultralytics sells an Enterprise License. A third option is to run v2 without the YOLO detectors: RF-DETR alone (Apache-2.0) scored mAP@0.5 0.418 on validation, against 0.446 for the full v2 ensemble. That needs a small code change, because the pipeline imports Ultralytics.
* **Apache-2.0** (RF-DETR and its weights, DINOv2, transformers, OpenCV, python-multipart, requests and others): when redistributing, include the license text and any NOTICE file of the package, and mark files you changed.
* **BSD, MIT, MIT-CMU, PSF-based and Zlib** (PyTorch, torchvision, NumPy, Pillow, FastAPI, uvicorn and others): keep the copyright notice and the license text in redistributions.
* **MPL-2.0** (part of tqdm): changes to tqdm's own files must be published under MPL-2.0. Using tqdm unchanged creates no obligation.
* **Pre-training data.** The pre-trained weights come from their publishers (Ultralytics, Roboflow, Meta, torchvision). Their training data (COCO, ImageNet-1k, DINOv2's LVD-142M) carries its own terms; the publishers release the weights under the licenses above.
* This list is a summary, not legal advice. Please have it confirmed by your legal team before a commercial release.
