"""Exact outlines: SAM 2.1 turns a box into a mask, and two helpers shape it.

A mask straight from a segmenter hugs the object, which is wrong for removal:
anti-aliased edge pixels and soft shadows sit just outside it and survive as a
ghostly rim. So the mask is grown first, and the fill is blended back in
*inside* that grown margin -- over background, never over the object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from .types import Box

if TYPE_CHECKING:
    from PIL import Image


def grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    """The mask widened by `pixels` in every direction (morphological dilation)."""
    solid = np.asarray(mask).astype(bool)
    if pixels <= 0 or not solid.any():
        return solid
    side = 2 * int(pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    return cv2.dilate(solid.astype(np.uint8), kernel).astype(bool)


def inner_feather(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Blend weights for a fill: 1 deep inside the mask, easing to 0 at its edge.

    The weight is exactly 0 everywhere outside the mask, which is what lets
    the caller promise that nothing outside it changes. It comes from the
    distance to the mask's edge rather than from blurring the mask, because a
    blur spills weight outwards.
    """
    solid = np.asarray(mask).astype(bool)
    if pixels <= 0:
        return solid.astype(np.float32)
    distance = cv2.distanceTransform(solid.astype(np.uint8), cv2.DIST_L2, 3)
    return np.clip(distance / float(pixels), 0.0, 1.0).astype(np.float32)


def bounding_box(mask: np.ndarray) -> Optional[Box]:
    """The tightest box around the mask, or None when it is empty."""
    rows = np.flatnonzero(np.asarray(mask).any(axis=1))
    columns = np.flatnonzero(np.asarray(mask).any(axis=0))
    if rows.size == 0 or columns.size == 0:
        return None
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def share(mask: np.ndarray) -> float:
    """The fraction of the image the mask covers."""
    solid = np.asarray(mask)
    return float(solid.sum()) / solid.size if solid.size else 0.0


def box_mask(shape, box: Box) -> np.ndarray:
    """A mask that is simply the box: the fallback when no outline is found."""
    mask = np.zeros(shape[:2], dtype=bool)
    x0, y0, x1, y1 = box
    mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return mask


class Segmenter:
    """SAM 2.1, prompted with a box."""

    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor

    def mask_for(self, image: "Image.Image", box: Box) -> np.ndarray:
        import torch

        inputs = self.processor(images=image, input_boxes=[[[float(v) for v in box]]],
                                return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.processor.post_process_masks(outputs.pred_masks.cpu(),
                                                  inputs["original_sizes"])[0]
        mask = np.asarray(masks[0, 0]).astype(bool)
        # An empty answer would make "remove" a silent no-op; the box is a
        # crude outline, but it is the right place.
        return mask if mask.any() else box_mask(mask.shape, box)
