"""Filling the hole a removed object leaves: one classical way, one generative.

**Classical** (OpenCV, Telea's fast marching method) carries the surrounding
colours inwards along the hole's edge. It is fast and honest about what it
is, and it cannot invent anything: behind a tree's leaves it paints sky.

**Generative** asks a diffusion model to paint what a description says is
there. That can produce the branches -- but they are invented, a plausible
picture and not a recovery of what the camera never saw.

Both obey one rule the tests pin down: pixels outside the mask come back
identical to the input. For the generative fill that is not the model's
doing -- a diffusion model repaints its whole canvas slightly -- so its answer
is composited back through the mask and everything else it did is discarded.
"""

from __future__ import annotations

from typing import Callable, Tuple

import cv2
import numpy as np
from PIL import Image

from .segment import bounding_box, inner_feather
from .types import Box

# (picture, mask picture with white = paint here, prompt, seed) -> painted picture
Painter = Callable[[Image.Image, Image.Image, str, int], Image.Image]

# Context around the mask, as a fraction of its longer side. Too little and
# the model cannot tell what scene it is continuing.
CROP_MARGIN = 0.4
# The smallest window worth handing to a model trained on 512-1024 px images.
CROP_MINIMUM = 256


def classical(image: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Fill the mask from its surroundings (Telea's method)."""
    solid = np.asarray(mask).astype(bool)
    if not solid.any():
        return image.copy()
    return cv2.inpaint(image, solid.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)


def crop_box(mask: np.ndarray, margin: float = CROP_MARGIN, minimum: int = CROP_MINIMUM) -> Box:
    """A window around the mask with room for context, clamped to the image.

    Square where the image allows. It always contains the whole mask.
    """
    height, width = mask.shape[:2]
    box = bounding_box(mask)
    if box is None:
        return 0, 0, width, height
    x0, y0, x1, y1 = box
    side = int(round(max(x1 - x0, y1 - y0) * (1.0 + 2.0 * margin)))
    side = max(side, minimum)
    wide, tall = min(side, width), min(side, height)
    left = int(round((x0 + x1) / 2.0 - wide / 2.0))
    top = int(round((y0 + y1) / 2.0 - tall / 2.0))
    left = min(max(left, 0), width - wide)
    top = min(max(top, 0), height - tall)
    return left, top, left + wide, top + tall


def work_size(width: int, height: int, target: int) -> Tuple[int, int]:
    """(width, height) scaled so the longer side is `target`, in multiples of 8.

    Diffusion models work on a latent grid one eighth the size of the image,
    so both sides have to divide by 8.
    """
    scale = float(target) / max(width, height, 1)
    return (max(64, int(round(width * scale / 8.0)) * 8),
            max(64, int(round(height * scale / 8.0)) * 8))


def composite(original: np.ndarray, painted: np.ndarray, mask: np.ndarray,
              feather: int = 0) -> np.ndarray:
    """`painted` inside the mask, `original` outside, eased over `feather` px.

    The easing happens inside the mask, so outside pixels are not merely
    close to the original: they are the original.
    """
    solid = np.asarray(mask).astype(bool)
    alpha = inner_feather(solid, feather)[..., None]
    blended = original.astype(np.float32) * (1.0 - alpha) + painted.astype(np.float32) * alpha
    out = np.rint(blended).astype(np.uint8)
    out[~solid] = original[~solid]
    return out


def generative(image: np.ndarray, mask: np.ndarray, prompt: str, painter: Painter,
               seed: int = 0, target: int = 1024, feather: int = 6) -> np.ndarray:
    """Fill the mask with what `painter` paints for `prompt`.

    The model only ever sees a window around the mask, resized to the
    resolution it was trained at, and its answer is composited back through
    the mask. Whatever it did to the rest of the window is thrown away.
    """
    solid = np.asarray(mask).astype(bool)
    if not solid.any():
        return image.copy()
    x0, y0, x1, y1 = crop_box(solid)
    window = image[y0:y1, x0:x1]
    window_mask = solid[y0:y1, x0:x1]
    size = work_size(x1 - x0, y1 - y0, target)

    # The model must not see the thing it is replacing: an inpainting pipeline
    # starts from a noised copy of its input, and a trace of green leaves is
    # enough to pull the answer back towards leaves. So the hole is filled
    # classically first, and the model starts from "nothing there".
    blank = classical(np.ascontiguousarray(window), window_mask)
    picture = Image.fromarray(blank).resize(size, Image.LANCZOS)
    hole = Image.fromarray(window_mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)
    painted = painter(picture, hole, prompt, int(seed)).convert("RGB")
    restored = np.asarray(painted.resize((x1 - x0, y1 - y0), Image.LANCZOS))

    out = image.copy()
    out[y0:y1, x0:x1] = composite(window, restored, window_mask, feather)
    return out
