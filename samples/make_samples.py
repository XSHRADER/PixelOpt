"""Generate the sample inputs in this folder.

Each image is built to exercise one specific behaviour, so a wrong result is
easy to spot. They are synthetic -- no photographs are downloaded -- which
makes them reproducible, but it also means the absolute quality numbers are
only meaningful relative to one another.

    python samples/make_samples.py
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
RNG = np.random.default_rng(20260913)


def font(size: int, name: str = "arial.ttf") -> ImageFont.ImageFont:
    for candidate in (name, f"C:/Windows/Fonts/{name}", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def scene(width: int, height: int, seed: int) -> np.ndarray:
    """A smooth, photo-like scene: sky gradient, hills, soft light."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float64)
    sky = np.dstack([
        110 + 90 * (y / height),
        150 + 60 * (y / height),
        235 - 40 * (y / height),
    ])
    ridge = height * (0.55 + 0.08 * np.sin(x / (width / 5.3)) + 0.04 * np.sin(x / (width / 13.1)))
    hills = y > ridge
    shade = np.clip((y - ridge) / (height * 0.45), 0, 1)
    ground = np.dstack([70 + 40 * shade, 120 - 30 * shade, 60 - 20 * shade])
    image = np.where(hills[..., None], ground, sky)
    # soft sun glow and low-frequency texture so there is detail worth keeping
    glow = np.exp(-(((x - width * 0.72) ** 2) + ((y - height * 0.25) ** 2)) / (2 * (width * 0.08) ** 2))
    image += glow[..., None] * np.array([120, 90, 30])
    texture = cv2.GaussianBlur(rng.normal(0, 30, (height, width)), (0, 0), 6)
    image += texture[..., None] * hills[..., None]
    return np.clip(image, 0, 255).astype(np.uint8)


def noisy_photo() -> None:
    """Sensor noise on a clean scene. Auto enhancement should denoise it."""
    clean = scene(1600, 1067, seed=1)
    noise = RNG.normal(0, 18, clean.shape)
    Image.fromarray(np.clip(clean + noise, 0, 255).astype(np.uint8)).save(
        INPUTS / "01_noisy_photo.jpg", quality=95
    )


def clean_landscape() -> None:
    """The same kind of scene with no noise. Auto should leave it alone."""
    Image.fromarray(scene(1600, 1067, seed=2)).save(
        INPUTS / "02_clean_landscape.jpg", quality=95
    )


def screenshot() -> None:
    """Flat UI with crisp text. Lossless PNG should win outright."""
    canvas = Image.new("RGB", (1280, 800), (246, 247, 250))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, 1280, 64], fill=(33, 41, 60))
    draw.text((28, 18), "PixelOpt - Dashboard", fill=(255, 255, 255), font=font(26, "segoeui.ttf"))
    for i, (label, value, colour) in enumerate(
        [("Images processed", "12,408", (79, 124, 255)),
         ("Bytes saved", "3.2 GB", (46, 160, 67)),
         ("Median SSIM", "0.993", (210, 105, 30))]
    ):
        left = 28 + i * 412
        draw.rounded_rectangle([left, 96, left + 384, 236], 14, fill=(255, 255, 255), outline=(225, 229, 238), width=2)
        draw.text((left + 24, 116), label, fill=(98, 108, 128), font=font(22, "segoeui.ttf"))
        draw.text((left + 24, 156), value, fill=colour, font=font(48, "segoeui.ttf"))
    draw.rounded_rectangle([28, 264, 1252, 772], 14, fill=(255, 255, 255), outline=(225, 229, 238), width=2)
    code = [
        "def seed_resize(target_bytes, pixels):",
        "    bpp = target_bytes * 8.0 / max(pixels, 1)",
        "    ideal = min(1.0, math.sqrt(bpp / BPP_TARGET))",
        "    return min(RESIZE_GRID, key=lambda r: abs(r - ideal))",
        "",
        "# resolution from arithmetic, not a guess",
    ]
    for row, line in enumerate(code):
        draw.text((56, 296 + row * 40), line, fill=(40, 44, 52), font=font(26, "consola.ttf"))
    canvas.save(INPUTS / "03_screenshot_text.png", optimize=True)


def transparent_logo() -> None:
    """Real alpha. Expect the transparency warning and a white background."""
    size = 900
    logo = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(logo)
    draw.ellipse([90, 90, 810, 810], fill=(79, 124, 255, 255))
    draw.ellipse([230, 230, 670, 670], fill=(0, 0, 0, 0))
    draw.rectangle([420, 60, 480, 840], fill=(255, 176, 32, 235))
    draw.text((250, 400), "PIXEL", fill=(33, 41, 60, 255), font=font(110, "segoeui.ttf"))
    logo.save(INPUTS / "04_logo_transparent.png")


def rotated_portrait() -> None:
    """Pixels stored landscape with an EXIF 'rotate 90' flag, like a phone.

    Opened correctly it is a TALL image with the label reading upright. If
    orientation handling ever breaks, the output comes back sideways.
    """
    upright = Image.fromarray(scene(900, 1400, seed=3))
    draw = ImageDraw.Draw(upright)
    draw.rectangle([0, 0, 900, 120], fill=(33, 41, 60))
    draw.text((40, 24), "THIS SIDE UP", fill=(255, 255, 255), font=font(64, "segoeui.ttf"))
    stored = upright.transpose(Image.Transpose.ROTATE_90)  # what the sensor wrote
    exif = stored.getexif()
    exif[274] = 6  # orientation: rotate 90 CW to display
    stored.save(INPUTS / "05_portrait_exif_rotated.jpg", quality=95, exif=exif)


def large_detail() -> None:
    """12 MP of structured detail. At 200 KB that is ~0.13 bpp, so the
    resolution rule should downscale first rather than smear full-res blocks."""
    width, height = 4000, 3000
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = scene(width, height, seed=4).astype(np.float32)
    fabric = 22 * np.sin(x / 3.1) * np.cos(y / 3.7) + 16 * np.sin((x + y) / 9.0)
    image = np.clip(base + fabric[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(image).save(INPUTS / "06_large_12mp_detail.jpg", quality=92)


def faded_scan() -> None:
    """A washed-out, yellowed, slightly noisy document. Try the scan preset."""
    page = Image.new("RGB", (1240, 1754), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    draw.text((110, 120), "Quarterly Compression Report", fill=(0, 0, 0), font=font(56))
    body = font(30)
    lines = [
        "Median size-targeting error fell from 74.1% to 0.7% once the search",
        "stopped rescaling an already-rescaled image.",
        "",
        "Denoising before encoding improved fidelity in every measured",
        "condition, because noise is the most expensive thing to store.",
    ]
    for row, line in enumerate(lines * 5):
        draw.text((110, 260 + row * 52), line, fill=(20, 20, 20), font=body)
    array = np.asarray(page).astype(np.float32)
    # fade toward grey, add a yellow cast, then scanner noise
    array = 70 + array * 0.62
    array[..., 2] *= 0.86
    array += RNG.normal(0, 7, array.shape)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(INPUTS / "07_faded_scan.png")


def blue_cast() -> None:
    """A photo with a strong cool cast. Try white balance."""
    array = scene(1600, 1067, seed=5).astype(np.float32)
    array[..., 0] *= 0.72
    array[..., 1] *= 0.88
    array[..., 2] = np.clip(array[..., 2] * 1.15 + 18, 0, 255)
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(
        INPUTS / "08_blue_cast_photo.jpg", quality=95
    )


def main() -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    for build in (noisy_photo, clean_landscape, screenshot, transparent_logo,
                  rotated_portrait, large_detail, faded_scan, blue_cast):
        build()
        print(f"built {build.__name__}")
    for path in sorted(INPUTS.iterdir()):
        with Image.open(path) as picture:
            print(f"  {path.name:<30} {picture.size[0]:>5}x{picture.size[1]:<5} "
                  f"{picture.mode:<5} {path.stat().st_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
