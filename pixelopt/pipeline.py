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

import io
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .adaptive_compressor import AdaptiveImageCompressor
from .enhance import Enhancements, apply, auto_enhancements, estimate_noise
from .image_features import compute_quality_metrics, load_image
from .passthrough import try_passthrough
from .quality_target import compress_to_quality


def _source_bytes(image_input) -> Optional[bytes]:
    """The encoded file behind an input, when there is one.

    Passthrough needs the original bytes, not decoded pixels. Paths, raw
    bytes and file-like objects (web uploads) carry them; arrays and PIL
    images do not, and simply never pass through.
    """
    if isinstance(image_input, (bytes, bytearray, memoryview)):
        return bytes(image_input)
    if isinstance(image_input, (str, Path)):
        try:
            return Path(image_input).read_bytes()
        except OSError:
            return None
    if hasattr(image_input, "read"):
        try:
            position = image_input.tell() if hasattr(image_input, "tell") else None
            data = image_input.read()
            if position is not None and hasattr(image_input, "seek"):
                image_input.seek(position)
            return bytes(data) if data else None
        except Exception:
            return None
    return None


def _prepare(image_input, enhancements: Optional[Enhancements]):
    raw = _source_bytes(image_input)
    original = load_image(io.BytesIO(raw) if raw is not None else image_input)
    if original.ndim != 3:
        raise ValueError("Only RGB images are supported.")
    settings = enhancements or Enhancements()
    reference = apply(original, settings) if settings.any_enabled() else original
    return raw, original, settings, reference


def _finish(result, original, reference, settings, measure_drift) -> Dict[str, object]:
    # The encoder measured the output against what it was handed, which is
    # the reference. Relabel rather than recompute.
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

    result.setdefault("passthrough", False)
    result["enhancements"] = settings
    result["enhancement_steps"] = settings.describe()
    result["enhanced"] = settings.any_enabled()
    result["original_image"] = Image.fromarray(original)
    result["reference_image"] = Image.fromarray(reference)
    return result


def process(
    image_input,
    target_size_kb: float,
    enhancements: Optional[Enhancements] = None,
    image_format: Optional[str] = None,
    compressor: Optional[AdaptiveImageCompressor] = None,
    measure_drift: bool = True,
    allow_passthrough: bool = True,
) -> Dict[str, object]:
    """Run the full publish pipeline and report both halves of the quality story.

    `enhancements` of None means do nothing. Pass `auto_enhancements(image)` to
    let the noise measurement decide.

    When nothing changes the pixels and the original file already fits, the
    original is returned with only its metadata removed -- see passthrough.py.
    `passthrough_reason` says why it was or was not used.
    """
    raw, original, settings, reference = _prepare(image_input, enhancements)

    result = None
    reason = "enhancement changes the pixels" if settings.any_enabled() else         "the input was not an encoded file"
    if allow_passthrough and raw is not None and not settings.any_enabled():
        result, reason = try_passthrough(
            raw, reference, max(target_size_kb * 1024.0, 1.0), image_format
        )

    if result is None:
        compressor = compressor or AdaptiveImageCompressor()
        result = compressor.compress(
            Image.fromarray(reference), target_size_kb, image_format=image_format
        )
    result["target_size_kb"] = float(target_size_kb)
    result["passthrough_reason"] = reason
    return _finish(result, original, reference, settings, measure_drift)


def process_to_quality(
    image_input,
    min_ssim: float,
    enhancements: Optional[Enhancements] = None,
    image_format: Optional[str] = None,
    measure_drift: bool = True,
    allow_passthrough: bool = True,
) -> Dict[str, object]:
    """The smallest file whose fidelity is at least `min_ssim`.

    The untouched original is exact, so it meets any target; it wins whenever
    it is also the smallest file that does -- typically an upload that was
    already compressed, which a re-encode could only make larger.
    """
    raw, original, settings, reference = _prepare(image_input, enhancements)
    result = compress_to_quality(reference, min_ssim, image_format=image_format)

    reason = "enhancement changes the pixels" if settings.any_enabled() else         "the input was not an encoded file"
    if allow_passthrough and raw is not None and not settings.any_enabled():
        kept, reason = try_passthrough(raw, reference, float("inf"), image_format)
        if kept is not None:
            # The original is exact, so it always meets the target. It wins if
            # it is smaller -- or if no re-encode reached the target at all.
            if not result.get("met", True):
                kept.update(target_ssim=float(min_ssim), met=True)
                result, reason = kept, "no re-encode reached the target, but the original does"
            elif len(kept["raw_bytes"]) <= len(result["raw_bytes"]):
                kept.update(target_ssim=float(min_ssim), met=True)
                result, reason = kept, "the original is already the smallest file that meets the target"
            else:
                reason = "a re-encode that meets the target is smaller than the original"
    result["passthrough_reason"] = reason
    return _finish(result, original, reference, settings, measure_drift)


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
