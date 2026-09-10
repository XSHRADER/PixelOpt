"""Command-line entry point.

    pixelopt photo.jpg --target 200
    pixelopt *.jpg --target 150 --out-dir web/ --format webp

A size-targeting compressor earns its keep in batch jobs, which is exactly
what a web upload form cannot do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .adaptive_compressor import SUPPORTED_FORMATS, AdaptiveImageCompressor

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
    parser.add_argument(
        "-t",
        "--target",
        type=float,
        required=True,
        metavar="KB",
        help="maximum output size in KB",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "-o", "--out", type=Path, help="output file (single input only)"
    )
    destination.add_argument(
        "-d",
        "--out-dir",
        type=Path,
        help="directory to write into (default: alongside each input)",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=("auto", "jpeg", "webp"),
        default="auto",
        help="encoder to use; auto tries both and keeps the better one",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output files instead of skipping them",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="only report failures"
    )
    return parser


def _destination(
    source: Path, extension: str, out: Optional[Path], out_dir: Optional[Path]
) -> Path:
    if out is not None:
        return out
    folder = out_dir if out_dir is not None else source.parent
    return folder / f"{source.stem}_compressed.{extension}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.out is not None and len(args.inputs) > 1:
        parser.error("--out takes a single input; use --out-dir for several")
    if args.target <= 0:
        parser.error("--target must be greater than zero")

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
            result = compressor.compress(source, args.target, image_format=image_format)
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
        if actual_kb >= original_kb:
            # Re-encoding a file that already fits can make it bigger -- a
            # lossless PNG of flat or glyph-like art beats any lossy encoder.
            # Say so plainly rather than reporting "0.2x smaller".
            note = "LARGER than input; the original already fits"
        else:
            note = f"{original_kb / max(actual_kb, 1e-9):.1f}x smaller"
        rows.append(
            f"{source.name:<24}{original_kb:>9.1f}K ->{actual_kb:>8.1f}K"
            f"{str(result['format']):>6}{dimensions:>12}"
            f"   SSIM {float(result['ssim']):.3f}   {note}"
        )

    if rows and not args.quiet:
        print(f"target {args.target:g} KB\n")
        for row in rows:
            print(row)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
