"""Run PixelOpt on every sample and write the outputs plus a contact sheet.

    python samples/make_samples.py     # once, to build the inputs
    python samples/run_samples.py

The contact sheet shows a 1:1 crop of each input next to the same crop of its
output, enlarged 2x. Full-image thumbnails are not enough to judge compression
-- at fit-to-screen a good and a bad encode look identical -- so the crops are
the part worth looking at.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from pixelopt.enhance import Enhancements, auto_enhancements, preset
from pixelopt.image_features import load_image
from pixelopt.pipeline import process

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
OUTPUTS = HERE / "outputs"


@dataclass
class Case:
    slug: str
    source: str
    target_kb: float
    enhance: str          # "none", "auto", a preset name, or "white_balance"
    expect: str
    image_format: Optional[str] = None


CASES = [
    Case("01a_noisy_plain", "01_noisy_photo.jpg", 150, "none",
         "Baseline: the encoder spends its budget storing the noise."),
    Case("01b_noisy_auto", "01_noisy_photo.jpg", 150, "auto",
         "Auto denoise: same budget, visibly cleaner; noise estimate drops sharply."),
    Case("02_clean_auto", "02_clean_landscape.jpg", 60, "auto",
         "Clean image: auto must do nothing (no enhancement line, drift 1.000)."),
    Case("03_screenshot", "03_screenshot_text.png", 100, "none",
         "Flat UI + text: lossless PNG should win, SSIM 1.000, crisp text."),
    Case("04_logo", "04_logo_transparent.png", 30, "none",
         "Transparency: output composited onto WHITE, never black."),
    Case("05_portrait", "05_portrait_exif_rotated.jpg", 80, "none",
         "EXIF rotation: output must be TALL (900x1400) with the label upright."),
    Case("06_large_12mp", "06_large_12mp_detail.jpg", 200, "none",
         "0.13 bpp: resolution rule should downscale rather than smear 12 MP."),
    Case("07a_scan_plain", "07_faded_scan.png", 250, "none",
         "Faded, yellowed scan left as it is."),
    Case("07b_scan_preset", "07_faded_scan.png", 250, "scan",
         "Scan preset: levels + contrast + denoise; text darker, paper whiter."),
    Case("08_blue_cast_wb", "08_blue_cast_photo.jpg", 80, "white_balance",
         "White balance: the blue cast is neutralised."),
]


def settings_for(case: Case, image) -> Enhancements:
    if case.enhance == "auto":
        return auto_enhancements(image)
    if case.enhance == "white_balance":
        return Enhancements(white_balance=True)
    return preset(case.enhance)


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ("segoeuib.ttf", "arialbd.ttf") if bold else ("segoeui.ttf", "arial.ttf")
    for name in names:
        for candidate in (name, f"C:/Windows/Fonts/{name}", "DejaVuSans.ttf"):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default(size=size)


def fit(picture: Image.Image, box: tuple) -> Image.Image:
    copy = picture.convert("RGB")
    copy.thumbnail(box, Image.LANCZOS)
    return copy


def crop_pair(source: Image.Image, output: Image.Image, size: int = 170, zoom: int = 2):
    """The same 1:1 region from input and output, enlarged with nearest-neighbour
    so compression artefacts are magnified rather than smoothed away."""
    source = source.convert("RGB")
    output = output.convert("RGB")
    if output.size != source.size:
        # A downscaled output is compared the way a viewer would see it:
        # scaled back up to the original dimensions.
        output = output.resize(source.size, Image.LANCZOS)
    cx, cy = source.width // 2, int(source.height * 0.62)
    box = (max(0, cx - size // 2), max(0, cy - size // 2))
    box = box + (min(source.width, box[0] + size), min(source.height, box[1] + size))
    enlarge = lambda im: im.crop(box).resize(
        ((box[2] - box[0]) * zoom, (box[3] - box[1]) * zoom), Image.NEAREST
    )
    return enlarge(source), enlarge(output)


def main() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    rows = []

    for case in CASES:
        source_path = INPUTS / case.source
        if not source_path.exists():
            raise SystemExit(f"missing {source_path} -- run samples/make_samples.py first")

        image = load_image(source_path)
        settings = settings_for(case, image)
        started = time.monotonic()
        result = process(source_path, case.target_kb, enhancements=settings,
                         image_format=case.image_format)
        elapsed = time.monotonic() - started

        out_path = OUTPUTS / f"{case.slug}.{result['extension']}"
        out_path.write_bytes(result["raw_bytes"])

        rows.append({
            "case": case,
            "result": result,
            "seconds": elapsed,
            "input_kb": source_path.stat().st_size / 1024.0,
            "out_path": out_path,
            "input_image": Image.fromarray(image),
        })
        print(
            f"{case.slug:<18} {rows[-1]['input_kb']:>8.1f}K -> {result['actual_size_kb']:>7.1f}K "
            f"(target {case.target_kb:g}K)  {result['format']:<4} "
            f"{result['width']}x{result['height']:<5} fidelity {result['fidelity_ssim']:.3f}  "
            f"drift {result['drift_ssim']:.3f}  {elapsed:5.1f}s"
            + (f"  [{', '.join(result['enhancement_steps'])}]" if result["enhanced"] else "")
        )

    write_table(rows)
    write_sheet(rows)


def write_table(rows) -> None:
    lines = [
        "| case | input | target | output | encoder | dimensions | fidelity | drift | enhancement | time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        r, c = row["result"], row["case"]
        lines.append(
            f"| `{c.slug}` | {row['input_kb']:.1f} KB | {c.target_kb:g} KB | "
            f"{r['actual_size_kb']:.1f} KB | {r['format']} | {r['width']}x{r['height']} | "
            f"{r['fidelity_ssim']:.3f} | {r['drift_ssim']:.3f} | "
            f"{', '.join(r['enhancement_steps']) or '-'} | {row['seconds']:.1f}s |"
        )
    (OUTPUTS / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sheet(rows) -> None:
    thumb_w, thumb_h, crop, gap = 300, 200, 340, 18
    text_w = 560
    row_h = max(thumb_h, crop) + gap
    width = gap + thumb_w + gap + crop + gap + crop + gap + text_w + gap
    header_h = 96
    sheet = Image.new("RGB", (width, header_h + row_h * len(rows) + gap), (246, 247, 250))
    draw = ImageDraw.Draw(sheet)

    draw.text((gap, 20), "PixelOpt sample run", fill=(29, 36, 51), font=font(34, bold=True))
    columns = [
        (gap, "input (full)"),
        (gap * 2 + thumb_w, "input, 1:1 crop x2"),
        (gap * 3 + thumb_w + crop, "output, same crop x2"),
        (gap * 4 + thumb_w + crop * 2, "result"),
    ]
    for x, label in columns:
        draw.text((x, 66), label, fill=(98, 108, 128), font=font(18, bold=True))

    for index, row in enumerate(rows):
        top = header_h + index * row_h
        r, c = row["result"], row["case"]
        output_image = Image.open(io.BytesIO(r["raw_bytes"]))
        # Crops compare against what the encoder was handed (the reference),
        # so enhancement shows up as a difference here -- which is the point.
        in_crop, out_crop = crop_pair(row["input_image"], output_image)

        thumb = fit(row["input_image"], (thumb_w, thumb_h))
        sheet.paste(thumb, (gap, top))
        sheet.paste(in_crop, (gap * 2 + thumb_w, top))
        sheet.paste(out_crop, (gap * 3 + thumb_w + crop, top))

        tx = gap * 4 + thumb_w + crop * 2
        draw.text((tx, top), c.slug, fill=(29, 36, 51), font=font(22, bold=True))
        details = [
            f"{row['input_kb']:.1f} KB  ->  {r['actual_size_kb']:.1f} KB   (target {c.target_kb:g} KB)",
            f"{r['format']}  ·  {r['width']}x{r['height']}  ·  quality {r['quality']}",
            f"fidelity {r['fidelity_ssim']:.3f}   drift {r['drift_ssim']:.3f}",
            ("enhanced: " + ", ".join(r["enhancement_steps"])) if r["enhanced"] else "no enhancement",
        ]
        for line_no, text in enumerate(details):
            draw.text((tx, top + 36 + line_no * 27), text, fill=(52, 60, 78), font=font(18))
        draw.text((tx, top + 36 + 4 * 27 + 8), "check: " + c.expect,
                  fill=(59, 91, 219), font=font(16))

    sheet.save(OUTPUTS / "contact_sheet.png", optimize=True)
    print(f"\nwrote {OUTPUTS / 'contact_sheet.png'} and {OUTPUTS / 'results.md'}")


if __name__ == "__main__":
    main()
