"""Defect cropping shared by classifier training, evaluation and the inference pipeline.

Client specification (Scope Clarifications §2): each box is expanded by 20% of its width/height
on each side, cropped from the ORIGINAL image, and resized to 224 x 224.

Training and inference must import this module so the crops are identical.
"""
from __future__ import annotations

from PIL import Image

EXPAND = 0.20
CROP_SIZE = 224


def expand_box_xyxy(x0: float, y0: float, x1: float, y1: float, W: int, H: int,
                    expand: float = EXPAND) -> tuple[int, int, int, int]:
    """Expand a pixel box by `expand` x its size on every side and clip to the image."""
    bw, bh = x1 - x0, y1 - y0
    ex0 = max(0, int(round(x0 - expand * bw)))
    ey0 = max(0, int(round(y0 - expand * bh)))
    ex1 = min(W, int(round(x1 + expand * bw)))
    ey1 = min(H, int(round(y1 + expand * bh)))
    if ex1 <= ex0:
        ex1 = min(W, ex0 + 1)
    if ey1 <= ey0:
        ey1 = min(H, ey0 + 1)
    return ex0, ey0, ex1, ey1


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, W: int, H: int) -> tuple[float, float, float, float]:
    return (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H


def crop_defect(img: Image.Image, box_xyxy: tuple[float, float, float, float],
                expand: float = EXPAND, size: int | None = CROP_SIZE) -> Image.Image:
    """Crop one defect from an RGB PIL image (pixel xyxy box). Returns an RGB crop.

    The crop is resized (aspect ratio NOT preserved, as specified: "resized to 224 x 224") when
    `size` is given; pass size=None to keep the native resolution (used when caching crops so
    augmentation can operate before the final resize).
    """
    W, H = img.size
    c = img.crop(expand_box_xyxy(*box_xyxy, W, H, expand=expand))
    if size is not None:
        c = c.resize((size, size), Image.BICUBIC)
    return c
