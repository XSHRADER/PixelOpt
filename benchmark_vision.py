"""Measure the two fills: hide a region whose true pixels are known, fill it, score it.

    python benchmark_vision.py               # classical and generated: needs the vision extra and a GPU
    python benchmark_vision.py --classical   # classical only: runs anywhere

A removal on a real photo cannot be scored, because nobody knows what was
behind the object. So this runs the other way round: punch a hole where the
truth is known, fill it, and compare the fill with what was really there.

Read the table with pixelopt/vision/evaluate.py's caveat in mind. PSNR and
SSIM reward being close on average, and a smooth smear is closer on average
than a sharp, plausible, different texture. A generated fill can lose on
these numbers and still be the one a viewer prefers.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image
from skimage import data

from pixelopt.vision import evaluate, inpaint, models, segment

# Photographs that ship inside scikit-image, so the run needs no download.
IMAGES = ("astronaut", "coffee", "chelsea", "rocket")
HOLE_SEEDS = (1, 2)
FALLBACK_PROMPT = "a seamless continuation of the surroundings"


def load(name: str) -> np.ndarray:
    return np.ascontiguousarray(getattr(data, name)()[..., :3]).astype(np.uint8)


def first_sentence(text: str) -> str:
    return text.split(".")[0].strip()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classical", action="store_true",
                        help="score only the classical fill (no GPU or vision extra needed)")
    args = parser.parse_args(argv)

    status = models.availability()
    generated = not args.classical
    if generated and not status.cuda:
        print("No CUDA GPU or vision extra here, so only the classical fill is scored.\n")
        generated = False
    hub = models.Hub() if generated else None

    print(f"{'image':<10} {'hole':>6}  {'method':<10} {'PSNR':>7} {'SSIM':>7} {'seconds':>8}")
    rows: List[dict] = []
    for name in IMAGES:
        image = load(name)
        prompt = FALLBACK_PROMPT
        if hub is not None:
            prompt = first_sentence(hub.describe(Image.fromarray(image)).caption) or FALLBACK_PROMPT
        for seed in HOLE_SEEDS:
            mask = evaluate.blob_mask(image.shape, seed=seed)
            fills = [("classical", lambda: inpaint.classical(image, mask))]
            if hub is not None:
                fills.append(("generated", lambda: inpaint.generative(
                    image, mask, prompt, hub.paint, seed=0, feather=0)))
            for method, fill in fills:
                started = time.perf_counter()
                restored = fill()
                seconds = time.perf_counter() - started
                result = evaluate.score(image, restored, mask)
                rows.append({"method": method, "seconds": seconds, **result})
                print(f"{name:<10} {segment.share(mask) * 100:>5.1f}%  {method:<10} "
                      f"{result['psnr']:>7.2f} {result['ssim']:>7.3f} {seconds:>8.2f}")

    print()
    for method in ("classical", "generated"):
        chosen = [row for row in rows if row["method"] == method]
        if chosen:
            print(f"{method:<10} mean PSNR {statistics.mean(r['psnr'] for r in chosen):.2f}  "
                  f"mean SSIM {statistics.mean(r['ssim'] for r in chosen):.3f}  "
                  f"median {statistics.median(r['seconds'] for r in chosen):.2f} s per fill")
    if hub is not None:
        print(f"generated fills painted by: {models.LABELS.get(hub.painted_with, 'nothing')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
