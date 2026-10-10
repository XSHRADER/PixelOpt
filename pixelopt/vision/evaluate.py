"""Measuring a fill: hide a region whose true pixels are known, then compare.

A removal cannot be scored on a real photo, because nobody knows what was
behind the object. So the test runs the other way round: take an image, punch
a hole in a place where the truth is known, fill it, and compare the fill with
what was really there.

Read the numbers with one caveat. PSNR and SSIM reward being close on
average, and a blurry smear is closer on average than a sharp, plausible, but
different texture. So a classical fill can outscore a generated one that any
viewer would prefer. That is the perception-distortion trade-off, and it is a
property of the metrics, not a bug in either fill.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def score(original: np.ndarray, restored: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """PSNR and SSIM of a fill against the truth, inside the mask only."""
    solid = np.asarray(mask).astype(bool)
    if not solid.any():
        return {"psnr": float("inf"), "ssim": 1.0, "pixels": 0}
    truth = original[solid].astype(np.float64)
    guess = restored[solid].astype(np.float64)
    mse = float(np.mean((truth - guess) ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)
    # SSIM is a mean over a local map; keeping the map lets the mean be taken
    # over the hole alone instead of being diluted by the untouched pixels.
    _, local = structural_similarity(
        cv2.cvtColor(original, cv2.COLOR_RGB2GRAY),
        cv2.cvtColor(restored, cv2.COLOR_RGB2GRAY),
        data_range=255, full=True,
    )
    return {"psnr": psnr, "ssim": float(local[solid].mean()), "pixels": int(solid.sum())}


def blob_mask(shape: Tuple[int, ...], seed: int = 0, size: float = 0.22) -> np.ndarray:
    """A reproducible, off-centre elliptical hole about `size` of the short side wide."""
    height, width = shape[:2]
    rng = np.random.default_rng(seed)
    short = min(height, width)
    axes = (max(4, int(short * size * rng.uniform(0.8, 1.2) / 2)),
            max(4, int(short * size * rng.uniform(0.5, 0.9) / 2)))
    centre = (int(width * rng.uniform(0.3, 0.7)), int(height * rng.uniform(0.3, 0.7)))
    canvas = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(canvas, centre, axes, float(rng.uniform(0, 180)), 0, 360, 255, -1)
    return canvas.astype(bool)
