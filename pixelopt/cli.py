"""Command-line entry point.

    pixelopt photo.jpg --target 200
    pixelopt *.jpg --target 150 --out-dir web/ --format webp
    pixelopt scan.png --target 300 --enhance scan

A size-targeting compressor earns its keep in batch jobs, which is exactly
what a web upload form cannot do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .adaptive_compressor import AdaptiveImageCompressor
from .enhance import PRESETS, Enhancements, auto_enhancements, preset, with_overrides
from .image_features import load_image
from .pipeline import process, process_to_quality

READABLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixelopt",
        description=(
            "Fit images into a byte budget at the highest measured quality. "
            "Output never exceeds the target; it may fall well under when the "
            "image cannot fill the budget."
        ),
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="image files to compress")
    goal = parser.add_mutually_exclusive_group(required=True)
    goal.add_argument(
        "-t", "--target", type=float, metavar="KB",
        help="maximum output size in KB: the best quality that fits",
    )
    goal.add_argument(
        "-s", "--min-ssim", type=float, metavar="SSIM",
        help="minimum fidelity, e.g. 0.98: the smallest file that meets it",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "-o", "--out", type=Path, help="output file (single input only)"
    )
    destination.add_argument(
        "-d", "--out-dir", type=Path,
        help="directory to write into (default: alongside each input)",
    )
    parser.add_argument(
        "-f", "--format", choices=("auto", "jpeg", "webp", "png"), default="auto",
        help="encoder to use; auto tries all and keeps the best at the budget",
    )

    group = parser.add_argument_group(
        "enhancement",
        "Applied before encoding. 'auto' measures the image noise and denoises "
        "only when it is worth doing -- denoising first improved quality in "
        "every tested condition, because noise is the most expensive thing an "
        "encoder can be asked to store.",
    )
    group.add_argument(
        "-e", "--enhance", default="none",
        choices=tuple(sorted(PRESETS)) + ("auto",),
        help="enhancement preset (default: none)",
    )
    group.add_argument("--denoise", type=float, metavar="N", help="filter strength, 0 to disable")
    group.add_argument("--sharpen", type=float, metavar="N", help="unsharp amount, 0 to disable")
    group.add_argument("--contrast", type=float, metavar="N", help="CLAHE clip limit, 0 to disable")
    group.add_argument("--saturation", type=float, metavar="N", help="1.0 leaves colour unchanged")
    group.add_argument("--white-balance", action="store_true", default=None, help="gray-world correction")
    group.add_argument("--auto-level", action="store_true", default=None, help="stretch the tonal range")

    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing output files instead of skipping them",
    )
    parser.add_argument(
        "--always-reencode", action="store_true",
        help="re-encode even when the original already fits (by default it is "
             "kept, minus its metadata)",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="only report failures")
    return parser


def _destination(
    source: Path, extension: str, out: Optional[Path], out_dir: Optional[Path]
) -> Path:
    if out is not None:
        return out
    folder = out_dir if out_dir is not None else source.parent
    return folder / f"{source.stem}_compressed.{extension}"


def _settings_for(args, source: Path) -> Enhancements:
    """Resolve preset plus explicit flags into one settings object.

    'auto' has to look at the pixels, so it is resolved per image rather than
    once for the batch -- a run of photos from different cameras will not all
    want the same filter strength.
    """
    if args.enhance == "auto":
        base = auto_enhancements(load_image(source))
    else:
        base = preset(args.enhance)
    return with_overrides(
        base,
        denoise=args.denoise,
        sharpen=args.sharpen,
        contrast=args.contrast,
        saturation=args.saturation,
        white_balance=args.white_balance,
        auto_level=args.auto_level,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.out is not None and len(args.inputs) > 1:
        parser.error("--out takes a single input; use --out-dir for several")
    if args.target is not None and args.target <= 0:
        parser.error("--target must be greater than zero")
    if args.min_ssim is not None and not (0.5 <= args.min_ssim <= 0.9999):
        parser.error("--min-ssim must be between 0.5 and 0.9999")

    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        parser.error("no such file: " + ", ".join(missing))

    image_format = None if args.format == "auto" else args.format.upper()
    compressor = AdaptiveImageCompressor()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    rows: List[str] = []

    for source in args.inputs:
        if source.suffix.lower() not in READABLE_SUFFIXES:
            print(f"{source}: unsupported file type", file=sys.stderr)
            failures += 1
            continue
        try:
            settings = _settings_for(args, source)
            if args.min_ssim is not None:
                result = process_to_quality(
                    source, args.min_ssim, enhancements=settings,
                    image_format=image_format,
                    allow_passthrough=not args.always_reencode,
                )
            else:
                result = process(
                    source, args.target, enhancements=settings,
                    image_format=image_format, compressor=compressor,
                    allow_passthrough=not args.always_reencode,
                )
        except Exception as error:  # a bad file should not abort the batch
            print(f"{source}: {error}", file=sys.stderr)
            failures += 1
            continue

        target_path = _destination(
            source, str(result["extension"]), args.out, args.out_dir
        )
        if target_path.exists() and not args.overwrite:
            print(f"{target_path}: exists, skipping (use --overwrite)", file=sys.stderr)
            failures += 1
            continue

        target_path.write_bytes(result["raw_bytes"])

        original_kb = source.stat().st_size / 1024.0
        actual_kb = float(result["actual_size_kb"])
        dimensions = f"{result['width']}x{result['height']}"
        if result.get("passthrough"):
            removed = int(result.get("metadata_removed_bytes", 0))
            note = (f"original kept, {removed:,} bytes of metadata removed"
                    if removed else "original kept unchanged")
        elif actual_kb > original_kb * 1.005:
            # A re-encode can still come out bigger, e.g. under --always-reencode
            # or a forced format. Say so plainly rather than "0.2x smaller".
            note = "LARGER than input"
        elif actual_kb >= original_kb * 0.995:
            note = "same size as input"
        else:
            note = f"{original_kb / max(actual_kb, 1e-9):.1f}x smaller"
        rows.append(
            f"{source.name:<24}{original_kb:>9.1f}K ->{actual_kb:>8.1f}K"
            f"{str(result['format']):>6}{dimensions:>12}"
            f"   SSIM {float(result['fidelity_ssim']):.3f}   {note}"
        )
        if args.min_ssim is not None and not result.get("met", True):
            rows.append(f"{'':<24}  SSIM {args.min_ssim:g} is out of reach with this "
                        "encoder; this is the closest it gets")
        if result["enhanced"]:
            # Drift is not an error -- on a noisy photo a large drift is the
            # denoiser working -- but it is also what overprocessing looks
            # like, so it goes next to the steps that caused it.
            rows.append(
                f"{'':<24}  enhanced: {', '.join(result['enhancement_steps'])}"
                f"  (drift {float(result['drift_ssim']):.3f}, "
                f"noise {float(result['noise_before']):.1f} -> "
                f"{float(result['noise_after']):.1f})"
            )

    if rows and not args.quiet:
        goal = (f"target {args.target:g} KB" if args.target is not None
                else f"target SSIM >= {args.min_ssim:g}")
        print(goal + "\n")
        for row in rows:
            print(row)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
