"""Quality target: the smallest file that still meets a minimum SSIM.

The size-budget search answers "how good can this be in N KB". Plenty of
people think the other way round -- "as small as possible, but it must still
look right" -- and forcing them to guess a budget means they either overshoot
and waste bytes or undershoot and ship damage.

For each encoder, the search walks resolutions downward from full size. At
each one it first checks whether even the top quality setting can reach the
target; if not, no smaller resolution will either, so that encoder stops.
Otherwise it bisects quality for the lowest setting that meets the target.

Scoring every probe on the whole image would be too slow -- full-size SSIM
is about 0.25 s at 2 MP and 3.8 s at 12 MP. The first version scored on a
downscaled copy instead, and that was a mistake worth recording: shrinking
both images smooths compression damage away before SSIM sees it, so
downscaled candidates sailed through the proxy and then failed the real
check. On a 12 MP image that meant 50 full-size corrections and 240 seconds.

So probes are scored on tiles at full resolution -- a stratified grid of
128-pixel squares spread over the frame. SSIM is the mean of a local map, and
tiles sample that map directly without resampling anything, so the estimate
leans toward neither answer. Images small enough are scored whole, which is
exact. The winner is still verified on the full image with the same metric
the result reports, and raised in quality if it falls short. Candidates are
verified smallest first, and the loop stops once the next one cannot beat the
best verified file -- correcting only ever makes a file larger.

Lossless PNG always meets any target, so under Auto a result always exists.
It is only tried up to PNG_PIXEL_CAP: encoding a 12 MP PNG costs seconds, and
a photograph that large is never smaller as PNG than a lossy encode that
meets the target. When the encoders are restricted and nothing reaches the
target, the highest-quality full-resolution encode is returned with `met`
set to False.
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .adaptive_compressor import (
    LOSSLESS_FORMATS,
    QUALITY_MAX,
    QUALITY_MIN,
    SUPPORTED_FORMATS,
    WEBP_MAX_SIDE,
    AdaptiveImageCompressor,
)
from .image_features import compute_quality_metrics, load_image

# Lower resolutions rarely reach a useful SSIM target, since resolution lost
# is detail lost; the downward walk usually stops after one or two steps.
TARGET_RESIZE_GRID: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4)

# Pixels of full-resolution tiles scored per probe. Images at or under this
# are scored whole, which is exact.
SAMPLE_PIXEL_CAP = 600_000
TILE = 128

# Above this many pixels, lossless PNG is not tried under Auto.
PNG_PIXEL_CAP = 4_000_000

MIN_TARGET, MAX_TARGET = 0.5, 0.9999

_EXTENSION = {"JPEG": "jpg", "WEBP": "webp", "PNG": "png"}
_MIME = {"JPEG": "image/jpeg", "WEBP": "image/webp", "PNG": "image/png"}


class _Probe:
    """One encode, with its estimated score and (once checked) its full metrics."""

    __slots__ = ("fmt", "resize", "quality", "data", "pixels", "estimate", "full")

    def __init__(self, fmt, resize, quality, data, pixels, estimate):
        self.fmt, self.resize, self.quality = fmt, resize, quality
        self.data, self.pixels, self.estimate = data, pixels, estimate
        self.full: Optional[Dict[str, float]] = None

    @property
    def size(self) -> int:
        return len(self.data)


class _Estimator:
    """SSIM of an encode against the reference, from full-resolution tiles.

    Decoding and scaling back up follow compute_quality_metrics exactly, so
    on small images, where the whole frame is scored, the two agree.
    """

    def __init__(self, image: np.ndarray):
        self.height, self.width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.boxes: Optional[List[Tuple[int, int]]] = None
        if self.height * self.width > SAMPLE_PIXEL_CAP and min(self.height, self.width) >= TILE:
            per_side = max(2, int(math.sqrt(SAMPLE_PIXEL_CAP / (TILE * TILE))))
            ys = np.linspace(0, self.height - TILE, per_side).astype(int)
            xs = np.linspace(0, self.width - TILE, per_side).astype(int)
            self.boxes = [(int(y), int(x)) for y in ys for x in xs]
            self.reference = [gray[y:y + TILE, x:x + TILE] for y, x in self.boxes]
        else:
            self.reference = gray

    def score(self, data: bytes) -> float:
        decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        if decoded.shape[:2] != (self.height, self.width):
            shrinking = decoded.shape[0] > self.height or decoded.shape[1] > self.width
            decoded = cv2.resize(
                decoded, (self.width, self.height),
                interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
            )
        gray = cv2.cvtColor(decoded, cv2.COLOR_RGB2GRAY)
        if self.boxes is None:
            return float(structural_similarity(self.reference, gray, data_range=255))
        return float(np.mean([
            structural_similarity(ref, gray[y:y + TILE, x:x + TILE], data_range=255)
            for ref, (y, x) in zip(self.reference, self.boxes)
        ]))


class _Search:
    def __init__(self, image: np.ndarray, target: float):
        self.image = image
        self.target = target
        self.estimator = _Estimator(image)
        self.encodes = 0
        self.full_checks = 0

    def probe(self, fmt: str, resize: float, quality: int) -> _Probe:
        pixels, data = AdaptiveImageCompressor._encode(self.image, resize, quality, fmt)
        self.encodes += 1
        return _Probe(fmt, resize, quality, data, pixels, self.estimator.score(data))

    def verify(self, probe: _Probe) -> float:
        if probe.full is None:
            decoded = np.asarray(Image.open(io.BytesIO(probe.data)).convert("RGB"))
            probe.full = compute_quality_metrics(self.image, decoded)
            self.full_checks += 1
        return probe.full["ssim"]

    def lowest_passing(self, fmt: str, resize: float, bound: float):
        """Lowest quality whose estimated score meets the target.

        Returns (probe, reachable). `probe` is None when nothing here can
        beat `bound`; `reachable` is False when even top quality misses, which
        tells the caller to stop walking to lower resolutions.
        """
        # The floor first. Smooth images often pass even there, which ends
        # the search in one probe. And file size only grows with quality, so a
        # floor already bigger than the best file found so far means nothing
        # at this resolution can win -- skip it without searching.
        floor = self.probe(fmt, resize, QUALITY_MIN)
        if floor.estimate >= self.target:
            return floor, True
        if floor.size >= bound:
            return None, True
        top = self.probe(fmt, resize, QUALITY_MAX)
        if top.estimate < self.target:
            return None, False
        # Bracketed search, alternating interpolation on the scores with
        # plain bisection: interpolation lands close when the curve is
        # smooth, and the bisection steps guarantee it cannot stall.
        low, high, interpolate = floor, top, True
        while high.quality - low.quality > 1:
            if interpolate and high.estimate > low.estimate:
                fraction = (self.target - low.estimate) / (high.estimate - low.estimate)
                quality = round(low.quality + fraction * (high.quality - low.quality))
            else:
                quality = (low.quality + high.quality) // 2
            quality = min(max(quality, low.quality + 1), high.quality - 1)
            interpolate = not interpolate
            candidate = self.probe(fmt, resize, quality)
            if candidate.estimate >= self.target:
                high = candidate
            else:
                if candidate.size >= bound:
                    # It misses the target and is already too big; every
                    # passing quality is higher still, so larger again.
                    return None, True
                low = candidate
        return high, True

    def correct(self, probe: _Probe) -> Optional[_Probe]:
        """Raise quality until the FULL-size score meets the target, or give up.

        The tile estimate is not exact, so a candidate near the boundary can
        miss by a hair. Step up a little first -- the gap is usually small --
        before resorting to a search.
        """
        if self.verify(probe) >= self.target:
            return probe
        for step in (1, 2, 4):
            quality = min(QUALITY_MAX, probe.quality + step)
            candidate = self.probe(probe.fmt, probe.resize, quality)
            if self.verify(candidate) >= self.target:
                return candidate
            if quality == QUALITY_MAX:
                return None
        top = self.probe(probe.fmt, probe.resize, QUALITY_MAX)
        if self.verify(top) < self.target:
            return None
        best, low, high = top, probe.quality + 5, QUALITY_MAX - 1
        while low <= high:
            mid = (low + high) // 2
            candidate = self.probe(probe.fmt, probe.resize, mid)
            if self.verify(candidate) >= self.target:
                best, high = candidate, mid - 1
            else:
                low = mid + 1
        return best


def compress_to_quality(
    image_input,
    min_ssim: float,
    image_format: Optional[str] = None,
    formats: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Smallest encode whose SSIM against the input is at least `min_ssim`."""
    if not (MIN_TARGET <= float(min_ssim) <= MAX_TARGET):
        raise ValueError(f"The SSIM target must be between {MIN_TARGET} and {MAX_TARGET}.")
    image = load_image(image_input)
    if image.ndim != 3:
        raise ValueError("Only RGB images are supported.")

    chosen = [image_format.upper()] if image_format else list(formats or SUPPORTED_FORMATS)
    unknown = sorted(set(chosen) - set(SUPPORTED_FORMATS))
    if unknown:
        raise ValueError("Unsupported format(s): " + ", ".join(unknown))
    if max(image.shape[:2]) > WEBP_MAX_SIDE:
        chosen = [f for f in chosen if f != "WEBP"] or ["JPEG"]

    search = _Search(image, float(min_ssim))
    candidates: List[_Probe] = []
    # JPEG first: it encodes about ten times faster than WebP at full size,
    # so it sets a size bound cheaply that lets WebP skip whole resolutions.
    chosen.sort(key=lambda f: {"JPEG": 0, "PNG": 1, "WEBP": 2}.get(f, 3))

    for fmt in chosen:
        if fmt in LOSSLESS_FORMATS:
            if image_format is None and image.shape[0] * image.shape[1] > PNG_PIXEL_CAP:
                continue
            # Exact at full resolution, so it always meets the target.
            pixels, data = AdaptiveImageCompressor._encode(image, 1.0, QUALITY_MAX, fmt)
            search.encodes += 1
            lossless = _Probe(fmt, 1.0, QUALITY_MAX, data, pixels, 1.0)
            lossless.full = {"ssim": 1.0, "psnr": float("inf"), "mse": 0.0}
            candidates.append(lossless)
            continue
        smallest = math.inf
        worse_in_a_row = 0
        for resize in TARGET_RESIZE_GRID:
            bound = min((c.size for c in candidates), default=math.inf)
            found, reachable = search.lowest_passing(fmt, resize, bound)
            if not reachable:
                break  # even top quality misses; lower resolutions will too
            if found is None:
                continue  # cannot beat the best so far; a smaller size might
            candidates.append(found)
            if found.size < smallest:
                smallest, worse_in_a_row = found.size, 0
            else:
                worse_in_a_row += 1
                if worse_in_a_row >= 2:
                    break  # files are growing as resolution falls; stop

    winner: Optional[_Probe] = None
    for probe in sorted(candidates, key=lambda p: p.size):
        if winner is not None and probe.size >= winner.size:
            break  # correcting can only grow a candidate, so none can win now
        corrected = search.correct(probe)
        if corrected is not None and (winner is None or corrected.size < winner.size):
            winner = corrected

    met = winner is not None
    if winner is None:
        # Restricted encoders and an unreachable target: the best that exists.
        fallbacks = [search.probe(fmt, 1.0, QUALITY_MAX) for fmt in chosen]
        winner = max(fallbacks, key=search.verify)
    search.verify(winner)

    metrics = winner.full
    decoded = Image.open(io.BytesIO(winner.data)).convert("RGB")
    height, width = winner.pixels.shape[:2]
    return {
        "raw_bytes": winner.data,
        "format": winner.fmt,
        "extension": _EXTENSION[winner.fmt],
        "mime": _MIME[winner.fmt],
        "width": width,
        "height": height,
        "original_shape": image.shape,
        "resized_shape": (width, height),
        "compressed_shape": winner.pixels.shape,
        "resize_factor": float(winner.resize),
        "quality": int(winner.quality),
        "actual_size_kb": winner.size / 1024.0,
        "ssim": float(metrics["ssim"]),
        "psnr": float(metrics["psnr"]),
        "mse": float(metrics["mse"]),
        "target_ssim": float(min_ssim),
        "met": bool(met),
        "encodes": search.encodes,
        "full_checks": search.full_checks,
        "candidates": len(candidates),
        "resized_image": Image.fromarray(winner.pixels.astype(np.uint8)),
        "image": decoded,
        "passthrough": False,
    }
