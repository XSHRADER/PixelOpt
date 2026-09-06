"""
benchmark.py
Measures what the compressor actually delivers, so the README can report
numbers instead of claims.

The headline claim of this project is that it predicts resize factor and JPEG
quality *for a requested file size*, so the metric that matters most is
size-targeting error: how close the encoded output lands to the target the
user asked for. SSIM and PSNR are reported alongside it to show what that
size costs in fidelity.

The test images are generated, not photographs -- they cover the cases that
stress a compressor differently (smooth gradients, hard edges and text-like
glyphs, dense high-frequency detail, near-flat colour). Size-targeting error
transfers to real photos; the absolute SSIM/PSNR figures are only meaningful
relative to each other, and the README says so.

Usage:
    python benchmark.py
    python benchmark.py --targets 100 200 500
"""

import argparse
import statistics
import time

import numpy as np
from PIL import Image

from src.adaptive_compressor import AdaptiveImageCompressor

SIZE = 1200
RNG = np.random.default_rng(20260906)  # fixed so runs are comparable


def img_gradient() -> np.ndarray:
    """Smooth gradient — the easy case; JPEG handles it well."""
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    image = np.zeros((SIZE, SIZE, 3), dtype=np.float64)
    image[..., 0] = x / SIZE * 255
    image[..., 1] = y / SIZE * 255
    image[..., 2] = ((x + y) / (2 * SIZE)) * 255
    return image.clip(0, 255).astype(np.uint8)


def img_edges() -> np.ndarray:
    """Hard edges and glyph-like blocks — ringing artefacts show up here."""
    image = np.full((SIZE, SIZE, 3), 245, dtype=np.uint8)
    for i in range(0, SIZE, 80):
        image[i : i + 30, :, :] = [30, 30, 30]
    for j in range(0, SIZE, 140):
        image[:, j : j + 18, :] = [200, 40, 60]
    for _ in range(120):
        y, x = RNG.integers(0, SIZE - 40, size=2)
        image[y : y + 28, x : x + 24] = RNG.integers(0, 60, size=3)
    return image


def img_detail() -> np.ndarray:
    """Dense high-frequency noise over structure — the expensive case."""
    y, x = np.mgrid[0:SIZE, 0:SIZE]
    base = (np.sin(x / 9.0) * np.cos(y / 11.0) + 1) * 110
    noise = RNG.normal(0, 34, (SIZE, SIZE))
    channel = (base + noise).clip(0, 255)
    return np.dstack([channel, np.roll(channel, 60, 0), np.roll(channel, 60, 1)]).astype(
        np.uint8
    )


def img_flat() -> np.ndarray:
    """Near-flat colour — compresses far below any sane target."""
    image = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
    image[..., 0] += RNG.integers(-3, 4, (SIZE, SIZE), dtype=np.int16).astype(np.uint8)
    return image


IMAGES = {
    "gradient": img_gradient,
    "edges/text": img_edges,
    "high detail": img_detail,
    "near-flat": img_flat,
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the adaptive compressor.")
    parser.add_argument(
        "--targets", type=float, nargs="+", default=[100, 200, 500],
        help="Target sizes in KB (default: 100 200 500)",
    )
    args = parser.parse_args()

    compressor = AdaptiveImageCompressor()
    rows = []

    header = (
        f"{'image':<13}{'target':>8}{'actual':>9}{'error':>9}"
        f"{'SSIM':>8}{'PSNR':>8}{'ratio':>8}{'time':>8}"
    )
    print(f"\n{SIZE}x{SIZE} test images, {len(args.targets)} target sizes\n")
    print(header)
    print("-" * len(header))

    for name, make in IMAGES.items():
        array = make()
        original_kb = array.size / 1024.0  # raw RGB bytes
        for target in args.targets:
            started = time.monotonic()
            result = compressor.compress(Image.fromarray(array), target)
            elapsed = time.monotonic() - started

            actual = float(result["actual_size_kb"])
            error = (actual - target) / target * 100.0
            ssim = float(result["ssim"])
            psnr = float(result["psnr"])
            ratio = original_kb / max(actual, 1e-6)

            # At full resolution and maximum quality the encoder cannot
            # produce more bytes -- a smooth gradient has nothing left to
            # encode. Undershooting there is a property of the image, not
            # a miss by the search, so it is counted separately.
            capped = result["resize_factor"] >= 0.999 and int(result["quality"]) >= 95

            rows.append(
                {
                    "image": name, "target": target, "actual": actual,
                    "error": error, "ssim": ssim, "psnr": psnr, "ratio": ratio, "capped": capped,
                }
            )
            print(
                f"{name:<13}{target:>7.0f}K{actual:>8.1f}K{error:>8.1f}%"
                f"{ssim:>8.3f}{psnr:>8.1f}{ratio:>7.0f}x{elapsed:>7.2f}s" + ('  (at max quality)' if capped else '')
            )

    print("-" * len(header))
    reachable = [r for r in rows if not r["capped"]]
    capped_rows = [r for r in rows if r["capped"]]
    errors = [abs(r["error"]) for r in reachable]
    undershoot = [r for r in rows if r["actual"] <= r["target"]]

    print()
    if errors:
        print(
            "size-targeting error (reachable targets): "
            f"median {statistics.median(errors):.1f}%  "
            f"mean {statistics.mean(errors):.1f}%  max {max(errors):.1f}%"
        )
    print(
        f"targets unreachable at max quality: {len(capped_rows)}/{len(rows)}"
    )
    print(f"stayed at or under target: {len(undershoot)}/{len(rows)}")
    print(
        f"SSIM: median {statistics.median(r['ssim'] for r in rows):.3f}  "
        f"min {min(r['ssim'] for r in rows):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
