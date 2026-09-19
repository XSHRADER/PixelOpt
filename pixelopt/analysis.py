"""Where quality was lost: a per-pixel damage map from local SSIM.

A single SSIM number says how much an encode lost; it cannot say *where*. The
same 0.97 can be damage spread thinly everywhere or a ruined face on an
otherwise perfect frame, and those are very different results. SSIM is a mean
of a local similarity map, so the map is already computed on the way to the
score -- this module keeps it instead of throwing it away.

The display scale is fixed rather than normalised per image. Normalising would
stretch a near-perfect encode's tiny losses across the whole colour range and
make it look ruined; with a fixed ceiling, a clean encode looks clean and a bad
one looks bad, and two heatmaps can be compared by eye.
"""

from __future__ import annotations

import math
from typing import Dict

import cv2
import numpy as np
from skimage.metrics import structural_similarity

# Local loss (1 - SSIM) at which the heatmap saturates.
DISPLAY_CEILING = 0.25

# Local SSIM below 0.9 is where compression damage starts to be visible at 1:1.
VISIBLE_LOSS = 0.10

# The map is measured at FULL resolution up to this size, and the pixels are
# only resampled beyond it. Resampling first destroys the thing being measured:
# INTER_AREA averages away blocking and ringing before SSIM can see them. On a
# faded 2.2 MP scan the damaged share read 86% at full resolution, 52% after a
# barely noticeable downscale to 2 MP, and 0.4% at 400k pixels -- the same file.
# Full resolution is affordable anyway (~0.2s at 2 MP, ~1.9s at 12 MP), and the
# loss map, not the image, is what gets shrunk for display.
MAP_PIXEL_CAP = 24_000_000


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image


def damage_map(reference: np.ndarray, output: np.ndarray,
               max_pixels: int = MAP_PIXEL_CAP) -> np.ndarray:
    """Per-pixel quality loss in [0, 1], where 0 means locally identical.

    A downscaled output is scaled back up first, matching how the quality
    metrics treat it: the question is how good a replacement it is.
    """
    reference = reference.astype(np.uint8)
    output = output.astype(np.uint8)
    height, width = reference.shape[:2]
    if output.shape[:2] != (height, width):
        shrinking = output.shape[0] > height or output.shape[1] > width
        output = cv2.resize(output, (width, height),
                            interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)

    ref_gray, out_gray = _to_gray(reference), _to_gray(output)
    if height * width > max_pixels:
        scale = math.sqrt(max_pixels / (height * width))
        size = (max(7, int(width * scale)), max(7, int(height * scale)))
        ref_gray = cv2.resize(ref_gray, size, interpolation=cv2.INTER_AREA)
        out_gray = cv2.resize(out_gray, size, interpolation=cv2.INTER_AREA)

    if min(ref_gray.shape) < 7:
        return np.zeros(ref_gray.shape, dtype=np.float32)

    _, local = structural_similarity(ref_gray, out_gray, data_range=255, full=True)
    return np.clip(1.0 - local, 0.0, 1.0).astype(np.float32)


def damage_summary(loss: np.ndarray, grid: int = 12) -> Dict[str, object]:
    """Numbers to go with the picture, including the worst region."""
    cells = cv2.resize(loss, (grid, grid), interpolation=cv2.INTER_AREA)
    row, col = np.unravel_index(int(np.argmax(cells)), cells.shape)
    return {
        "mean_loss": float(loss.mean()),
        "p95_loss": float(np.percentile(loss, 95)),
        "damaged_share": float((loss > VISIBLE_LOSS).mean()),
        "worst_loss": float(cells[row, col]),
        # Fractions of width and height, so it applies at any display size.
        # Plain floats: this goes to the browser as JSON.
        "worst_box": (float(col / grid), float(row / grid),
                      float((col + 1) / grid), float((row + 1) / grid)),
    }


def heatmap_rgba(loss: np.ndarray, ceiling: float = DISPLAY_CEILING) -> np.ndarray:
    """Coloured damage layer with alpha that follows the damage.

    Undamaged pixels are fully transparent, so laid over the output it only
    tints the places that actually lost something.
    """
    norm = np.clip(loss / ceiling, 0.0, 1.0)
    shades = (np.power(norm, 0.7) * 255).astype(np.uint8)
    colour = cv2.applyColorMap(shades, cv2.COLORMAP_INFERNO)[..., ::-1]
    alpha = (np.clip(norm * 1.6, 0.0, 1.0) * 235).astype(np.uint8)
    return np.dstack([colour, alpha])


def heatmap_overlay(reference: np.ndarray, loss: np.ndarray,
                    ceiling: float = DISPLAY_CEILING) -> np.ndarray:
    """The damage layer composited over a dimmed reference, as one RGB image."""
    height, width = reference.shape[:2]
    layer = heatmap_rgba(loss, ceiling)
    if layer.shape[:2] != (height, width):
        layer = cv2.resize(layer, (width, height), interpolation=cv2.INTER_LINEAR)
    base = reference.astype(np.float32) if reference.ndim == 3 else np.dstack(
        [reference] * 3).astype(np.float32)
    base *= 0.55  # dim the photo so the damage reads first
    alpha = layer[..., 3:4].astype(np.float32) / 255.0
    blended = base * (1 - alpha) + layer[..., :3].astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)
