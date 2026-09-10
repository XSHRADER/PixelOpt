from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .image_features import compute_quality_metrics, load_image

# Resize factors the search is allowed to pick from.
RESIZE_GRID: Tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.22, 0.15)

QUALITY_MIN = 10
QUALITY_MAX = 95

# Bits per pixel the encoder is aimed at. Below roughly this rate a block
# transform codec spends its whole budget on artefacts, and downscaling first
# buys back more real detail than the lost resolution costs. Measured on the
# benchmark corpus the crossover sits between 0.25 and 0.5 bpp; above ~0.5
# full resolution always won, so the rule caps resize at 1.0 there.
BPP_TARGET = 0.55

# Ranking candidates against each other does not need full-resolution SSIM,
# which costs ~3.8s on a 12MP image. Rank on a capped proxy, then measure the
# winner properly once.
RANK_PIXEL_CAP = 1_000_000

# SSIM differences smaller than this are not worth trading resolution for.
SCORE_TOLERANCE = 0.002

# WebP cannot encode a side longer than this.
WEBP_MAX_SIDE = 16383

SUPPORTED_FORMATS: Tuple[str, ...] = ("JPEG", "WEBP")

_EXTENSION = {"JPEG": "jpg", "WEBP": "webp"}
_MIME = {"JPEG": "image/jpeg", "WEBP": "image/webp"}


@dataclass
class Attempt:
    """One encoded candidate."""

    fmt: str
    resize: float
    quality: int
    data: bytes
    pixels: np.ndarray
    score: float = float("nan")

    @property
    def size(self) -> int:
        return len(self.data)


@dataclass
class _Budget:
    """Counts encoder calls so the benchmark can report search cost."""

    encodes: int = 0


class AdaptiveImageCompressor:
    """Fit an image into a byte budget at the highest measured quality.

    The search has two parts. Quality is solved exactly: encoded size grows
    smoothly and monotonically with the quality setting, so interpolating on
    log-size converges in about five encodes. Resolution is chosen by the
    bits-per-pixel rule above, then confirmed by scoring the seed factor, its
    two neighbours and full resolution with SSIM -- because "largest file
    under the budget" is not the same thing as "best looking file under it".
    """

    RESIZE_GRID = RESIZE_GRID
    QUALITY_MIN = QUALITY_MIN
    QUALITY_MAX = QUALITY_MAX

    MAX_PROBES = 7

    def __init__(self, formats: Optional[Sequence[str]] = None):
        chosen = tuple(f.upper() for f in (formats or SUPPORTED_FORMATS))
        unknown = sorted(set(chosen) - set(SUPPORTED_FORMATS))
        if unknown:
            raise ValueError("Unsupported format(s): " + ", ".join(unknown))
        self.formats = chosen

    # ---------------------------------------------------------------- encode

    @staticmethod
    def _resize(image_rgb: np.ndarray, resize: float) -> np.ndarray:
        if resize >= 0.999:
            return image_rgb
        height, width = image_rgb.shape[:2]
        return cv2.resize(
            image_rgb,
            (max(1, int(width * resize)), max(1, int(height * resize))),
            interpolation=cv2.INTER_AREA,
        )

    @staticmethod
    def _encode(
        image_rgb: np.ndarray, resize: float, quality: int, fmt: str = "JPEG"
    ) -> Tuple[np.ndarray, bytes]:
        """Resize the ORIGINAL image by `resize` and encode it.

        Always scaling from the original matters: an earlier version resized
        the already-resized image, so the two factors multiplied and the
        search could only ever shrink, never climb back toward a larger target.
        """
        candidate = AdaptiveImageCompressor._resize(image_rgb, resize)
        buffer = io.BytesIO()
        picture = Image.fromarray(candidate.astype(np.uint8))
        if fmt == "WEBP":
            picture.save(buffer, format="WEBP", quality=int(quality), method=4)
        else:
            # progressive + optimize cost nothing at encode time and buy
            # roughly 2% SSIM at the same byte budget, plus progressive
            # rendering in browsers.
            picture.save(
                buffer,
                format="JPEG",
                quality=int(quality),
                progressive=True,
                optimize=True,
            )
        return candidate, buffer.getvalue()

    def _usable_formats(self, image_rgb: np.ndarray) -> List[str]:
        height, width = image_rgb.shape[:2]
        if max(height, width) <= WEBP_MAX_SIDE:
            return list(self.formats)
        return [f for f in self.formats if f != "WEBP"] or ["JPEG"]

    # ---------------------------------------------------------------- search

    @staticmethod
    def seed_resize(target_bytes: float, pixels: int) -> float:
        """Pick a resize factor straight from the bits-per-pixel budget.

        After downscaling by r the image has r^2 * N pixels, so the encoder
        works at bpp / r^2. Solving for the rate we want gives
        r = sqrt(bpp / BPP_TARGET), capped at 1.0 -- upscaling never helps.
        """
        bpp = target_bytes * 8.0 / max(pixels, 1)
        ideal = min(1.0, math.sqrt(bpp / BPP_TARGET))
        return min(RESIZE_GRID, key=lambda r: abs(r - ideal))

    @staticmethod
    def _candidates(seed: float) -> List[float]:
        """The seed factor, one step either side of it, and full resolution.

        Full resolution is always worth a look. The bpp rule assumes the image
        needs BPP_TARGET bits per pixel, but an easily compressed one needs far
        fewer -- a flat or text-like image often fits the budget untouched. On
        the benchmark, testing 1.0 anyway lifted an image from 0.956 to 0.999
        SSIM *and* made the file smaller. It costs two encodes to rule out.
        """
        index = RESIZE_GRID.index(seed)
        factors = [
            RESIZE_GRID[i]
            for i in (index - 1, index, index + 1)
            if 0 <= i < len(RESIZE_GRID)
        ]
        if 1.0 not in factors:
            factors.append(1.0)
        return factors

    def _best_quality(
        self,
        image_rgb: np.ndarray,
        resize: float,
        target_bytes: float,
        fmt: str,
        budget: _Budget,
    ) -> Optional[Attempt]:
        """Highest quality whose encode still fits, by interpolation on log-size.

        Encoded size is monotonic in the quality setting and roughly
        exponential in it, so interpolating between two bracketing probes in
        log space lands near the answer immediately. Bisecting the same range
        needs about seven probes per resize factor; this needs about three.
        """
        low, high = QUALITY_MIN, QUALITY_MAX

        pixels, low_data = self._encode(image_rgb, resize, low, fmt)
        budget.encodes += 1
        if len(low_data) > target_bytes:
            return None  # does not fit even at the quality floor

        best = Attempt(fmt, resize, low, low_data, pixels)

        _, high_data = self._encode(image_rgb, resize, high, fmt)
        budget.encodes += 1
        if len(high_data) <= target_bytes:
            return Attempt(fmt, resize, high, high_data, pixels)

        low_size, high_size = len(low_data), len(high_data)

        for _ in range(self.MAX_PROBES):
            span = math.log(high_size) - math.log(low_size)
            if span > 1e-9:
                fraction = (math.log(target_bytes) - math.log(low_size)) / span
            else:
                fraction = 0.5  # flat image: every quality encodes the same, bisect
            guess = int(round(low + fraction * (high - low)))
            guess = min(max(guess, low + 1), high - 1)

            _, data = self._encode(image_rgb, resize, guess, fmt)
            budget.encodes += 1
            size = len(data)

            if size <= target_bytes:
                if guess > best.quality:
                    best = Attempt(fmt, resize, guess, data, pixels)
                low, low_size = guess, size
            else:
                high, high_size = guess, size

            if high - low <= 1:
                break

        return best

    @staticmethod
    def _rank_proxy(image_rgb: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        height, width = image_rgb.shape[:2]
        scale = min(1.0, math.sqrt(RANK_PIXEL_CAP / max(height * width, 1)))
        size = (max(1, int(width * scale)), max(1, int(height * scale)))
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        return gray, size

    @staticmethod
    def _score(proxy: np.ndarray, size: Tuple[int, int], data: bytes) -> float:
        decoded = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
        if (decoded.shape[1], decoded.shape[0]) != size:
            shrinking = decoded.shape[0] > size[1] or decoded.shape[1] > size[0]
            decoded = cv2.resize(
                decoded,
                size,
                interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
            )
        return float(structural_similarity(proxy, decoded, data_range=255))

    def _smallest_possible(
        self, image_rgb: np.ndarray, formats: Sequence[str], budget: _Budget
    ) -> Attempt:
        """Nothing fits the budget -- return the smallest file we can make."""
        smallest = None
        for fmt in formats:
            pixels, data = self._encode(image_rgb, RESIZE_GRID[-1], QUALITY_MIN, fmt)
            budget.encodes += 1
            attempt = Attempt(fmt, RESIZE_GRID[-1], QUALITY_MIN, data, pixels)
            if smallest is None or attempt.size < smallest.size:
                smallest = attempt
        return smallest

    def _search(
        self, image_rgb: np.ndarray, target_bytes: float, budget: _Budget
    ) -> Attempt:
        height, width = image_rgb.shape[:2]
        formats = self._usable_formats(image_rgb)
        seed = self.seed_resize(target_bytes, width * height)

        fitting: List[Attempt] = []
        for fmt in formats:
            for resize in self._candidates(seed):
                found = self._best_quality(image_rgb, resize, target_bytes, fmt, budget)
                if found is not None:
                    fitting.append(found)

        if not fitting:
            # The seed and its neighbours were all too big even at the quality
            # floor. Walk down the rest of the grid until something fits.
            for fmt in formats:
                for resize in RESIZE_GRID:
                    found = self._best_quality(
                        image_rgb, resize, target_bytes, fmt, budget
                    )
                    if found is not None:
                        fitting.append(found)
                        break

        if not fitting:
            return self._smallest_possible(image_rgb, formats, budget)

        proxy, proxy_size = self._rank_proxy(image_rgb)
        for attempt in fitting:
            attempt.score = self._score(proxy, proxy_size, attempt.data)

        # Measured quality decides, but only past a margin. The proxy is a
        # downscaled copy of the original, which slightly flatters downscaled
        # candidates -- on a near-flat image a 0.9 encode "beat" full
        # resolution by 0.0004 SSIM purely because both had been smoothed the
        # same way. Inside the margin, prefer resolution, then fewer bytes:
        # equal quality for a smaller file is a strict win.
        best = max(attempt.score for attempt in fitting)
        close = [a for a in fitting if a.score >= best - SCORE_TOLERANCE]
        return max(close, key=lambda a: (a.resize, -a.size))

    # --------------------------------------------------------------- public

    def compress(
        self,
        image_input,
        target_size_kb: float,
        image_format: Optional[str] = None,
    ) -> Dict[str, object]:
        image = load_image(image_input)
        if image.ndim != 3:
            raise ValueError("Only RGB images are supported.")

        previous = self.formats
        if image_format is not None:
            self.formats = (image_format.upper(),)
        try:
            budget = _Budget()
            target_bytes = max(target_size_kb * 1024.0, 1.0)
            best = self._search(image, target_bytes, budget)
        finally:
            self.formats = previous

        resized_image = Image.fromarray(best.pixels.astype(np.uint8))
        compressed_image = Image.open(io.BytesIO(best.data)).convert("RGB")
        metrics = compute_quality_metrics(image, np.asarray(compressed_image))

        height, width = best.pixels.shape[:2]
        return {
            "original_shape": image.shape,
            "resized_shape": (width, height),
            "compressed_shape": best.pixels.shape,
            "width": width,
            "height": height,
            "resize_factor": float(best.resize),
            "quality": int(best.quality),
            "format": best.fmt,
            "extension": _EXTENSION[best.fmt],
            "mime": _MIME[best.fmt],
            "target_size_kb": float(target_size_kb),
            "actual_size_kb": best.size / 1024.0,
            "encodes": budget.encodes,
            "ssim": metrics["ssim"],
            "psnr": metrics["psnr"],
            "mse": metrics["mse"],
            "resized_image": resized_image,
            "image": compressed_image,
            "raw_bytes": best.data,
        }
