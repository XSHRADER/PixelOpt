"""Enhance, then compress -- and measure each against the right reference.

This module exists because of one problem. The compressor's quality metric is
SSIM against the image it was given. Enhancement deliberately *changes* the
image, so if enhancement and compression share a reference, the metric starts
punishing the enhancement and the search fights it.

So the reference splits in two:

    original ──enhance──> reference ──compress──> output
                 │                        │
                 │                        └── fidelity: SSIM(reference, output)
                 │                            how much the ENCODER lost
                 └── drift: SSIM(original, reference)
                     how much the ENHANCEMENT changed

Fidelity is the number the search optimises and the one worth quoting as
"compression quality". Drift is not an error -- a large drift on a noisy photo
is the denoiser doing its job -- but it is worth showing, because it is also
what a too-aggressive filter looks like. Neither number alone tells the truth,
which is exactly why the old single-SSIM design could not be extended.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PIL import Image

from .adaptive_compressor import AdaptiveImageCompressor
from .enhance import Enhancements, apply, auto_enhancements, estimate_noise
from .image_features import compute_quality_metrics, load_image


def process(
    image_input,
    target_size_kb: float,
    enhancements: Optional[Enhancements] = None,
    image_format: Optional[str] = None,
    compressor: Optional[AdaptiveImageCompressor] = None,
    measure_drift: bool = True,
) -> Dict[str, object]:
    """Run the full publish pipeline and report both halves of the quality story.

    `enhancements` of None means do nothing. Pass `auto_enhancements(image)` to
    let the noise measurement decide.
    """
    original = load_image(image_input)
    if original.ndim != 3:
        raise ValueError("Only RGB images are supported.")

    settings = enhancements or Enhancements()
    reference = apply(original, settings) if settings.any_enabled() else original

    compressor = compressor or AdaptiveImageCompressor()
    result = compressor.compress(
        Image.fromarray(reference), target_size_kb, image_format=image_format
    )

    # The compressor already measured the output against what it was handed,
    # which is the reference. Relabel rather than recompute.
    result["fidelity_ssim"] = result["ssim"]
    result["fidelity_psnr"] = result["psnr"]
    result["fidelity_mse"] = result["mse"]

    if settings.any_enabled() and measure_drift:
        drift = compute_quality_metrics(original, reference)
        result["drift_ssim"] = drift["ssim"]
        result["drift_psnr"] = drift["psnr"]
        result["noise_before"] = estimate_noise(original)
        result["noise_after"] = estimate_noise(reference)
    else:
        result["drift_ssim"] = 1.0
        result["drift_psnr"] = float("inf")
        result["noise_before"] = result["noise_after"] = estimate_noise(original)

    result["enhancements"] = settings
    result["enhancement_steps"] = settings.describe()
    result["enhanced"] = settings.any_enabled()
    result["original_image"] = Image.fromarray(original)
    result["reference_image"] = Image.fromarray(reference)

    return result


def rate_distortion_curve(
    image_input,
    targets_kb,
    enhancements: Optional[Enhancements] = None,
    image_format: Optional[str] = None,
) -> list:
    """Quality as a function of budget, for one image.

    Every point of this is already computed and thrown away during a normal
    run. Surfacing it lets someone pick the knee of their own curve instead of
    guessing a number -- most images have an obvious one, past which extra
    kilobytes buy almost nothing.
    """
    compressor = AdaptiveImageCompressor()
    original = load_image(image_input)
    settings = enhancements or Enhancements()
    reference = apply(original, settings) if settings.any_enabled() else original
    picture = Image.fromarray(reference)

    points = []
    for target in sorted(targets_kb):
        result = compressor.compress(picture, float(target), image_format=image_format)
        points.append(
            {
                "target_kb": float(target),
                "actual_kb": float(result["actual_size_kb"]),
                "ssim": float(result["ssim"]),
                "psnr": float(result["psnr"]),
                "format": result["format"],
                "width": result["width"],
                "height": result["height"],
                "resize_factor": float(result["resize_factor"]),
                "quality": int(result["quality"]),
                "capped": bool(
                    result["resize_factor"] >= 0.999 and int(result["quality"]) >= 95
                ),
            }
        )
    return points


def knee_point(points: list) -> Optional[dict]:
    """The budget past which more bytes stop buying meaningful quality.

    Finds the point of maximum distance from the straight line joining the
    cheapest and most expensive results -- the standard elbow heuristic. With
    fewer than three points there is no elbow to find.
    """
    usable = [p for p in points if not p["capped"]]
    if len(usable) < 3:
        return None

    xs = np.array([p["actual_kb"] for p in usable], dtype=float)
    ys = np.array([p["ssim"] for p in usable], dtype=float)
    if np.ptp(xs) < 1e-9 or np.ptp(ys) < 1e-9:
        return None

    xn = (xs - xs.min()) / np.ptp(xs)
    yn = (ys - ys.min()) / np.ptp(ys)
    # Perpendicular distance from the first-to-last chord.
    dx, dy = xn[-1] - xn[0], yn[-1] - yn[0]
    norm = np.hypot(dx, dy)
    if norm < 1e-9:
        return None
    distance = np.abs(dy * (xn - xn[0]) - dx * (yn - yn[0])) / norm
    return usable[int(np.argmax(distance))]
