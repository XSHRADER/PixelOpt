"""The labelled picture: every found object boxed and named, the chosen one tinted."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import Found

# The app's chart colours, so the boxes belong to the same page.
PALETTE = ((92, 200, 184), (245, 165, 36), (123, 156, 255),
           (255, 107, 107), (195, 155, 255), (126, 224, 122))
TINT = (92, 200, 184)
INK = (20, 21, 23)


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1 has only the fixed bitmap font
        return ImageFont.load_default()


def overlay(image: np.ndarray, objects: Sequence[Found] = (), chosen: Optional[int] = None,
            mask: Optional[np.ndarray] = None) -> np.ndarray:
    """`image` with each object boxed and labelled, and `mask` tinted.

    Line and label sizes follow the image, so a 1600 px photo and a 400 px
    one are equally readable once scaled to the page.
    """
    out = image.copy()
    scale = max(image.shape[:2]) / 1000.0
    line = max(2, int(round(2 * scale)))

    if mask is not None and mask.any():
        tint = np.array(TINT, dtype=np.float32)
        out[mask] = np.rint(out[mask].astype(np.float32) * 0.5 + tint * 0.5).astype(np.uint8)
        kernel = np.ones((2 * line + 1, 2 * line + 1), np.uint8)
        rim = mask & ~cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        out[rim] = TINT

    if not objects:
        return out
    picture = Image.fromarray(out)
    pen = ImageDraw.Draw(picture)
    font = _font(max(12, int(round(15 * scale))))
    for index, item in enumerate(objects):
        colour = PALETTE[index % len(PALETTE)]
        x0, y0, x1, y1 = item.box
        pen.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], outline=colour,
                      width=line * 2 if index == chosen else line)
        left, top, right, bottom = pen.textbbox((0, 0), item.name, font=font)
        wide, tall = right - left, bottom - top
        pad = max(2, line)
        # Above the box when there is room, otherwise just inside its top edge.
        label_top = y0 - tall - 2 * pad if y0 - tall - 2 * pad >= 0 else y0
        pen.rectangle([x0, label_top, x0 + wide + 2 * pad, label_top + tall + 2 * pad], fill=colour)
        pen.text((x0 + pad - left, label_top + pad - top), item.name, fill=INK, font=font)
    return np.asarray(picture)
