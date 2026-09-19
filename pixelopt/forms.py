"""Form photo mode: exact pixel dimensions and a file-size *range*.

Online application forms ask for something the general compressor was never
built for -- "a JPEG of exactly 200 x 230 pixels, between 20 and 50 KB". That
is three constraints the normal search does not have:

- **Fixed dimensions.** There is no resolution to search; the image has to be
  cropped or padded to the exact aspect, then resized.
- **A minimum as well as a maximum.** Portals set a floor to reject images
  that have been crushed, and the main search only ever enforces a ceiling.
- **Plain baseline JPEG.** Some upload validators choke on progressive JPEG,
  so this path never emits it.

**Which kilobyte?** Portals disagree on whether a KB is 1000 or 1024 bytes,
and a file that is "50 KB" by one reading is 51.2 KB by the other -- a very
common reason for a rejected upload. By default the limits are applied so the
file passes under *both* readings: the maximum in 1000-byte KB and the minimum
in 1024-byte KB. Only when a range is too narrow for that are 1024-byte KB
used for both, and the result says so.

**When the image is too simple to reach the minimum**, raising JPEG quality
runs out before the byte count does -- a plain background encodes to a few KB
even at quality 100. The file is then padded with JPEG comment segments until
it is inside the range. The pixels are untouched and the file stays a valid
JPEG, and the result reports exactly how many bytes were added. It is only a
last resort: a real encoding that meets the minimum is always preferred when
one exists within a hair of the best quality.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

from .image_features import load_image

# A real encoding within this much SSIM of the best one is preferred over
# padding the best one up to the minimum.
SCORE_TOLERANCE = 0.002

# JPEG comment segments: 2 marker bytes, 2 length bytes, up to 65533 payload.
_SEGMENT_MAX = 65537
_SEGMENT_MIN = 4
_FILLER = b"PixelOpt size padding. "


@dataclass(frozen=True)
class FormSpec:
    """What the form asks for."""

    width: int
    height: int
    min_kb: float
    max_kb: float
    fit: str = "crop"            # "crop" to the aspect, or "pad" onto white
    grayscale: bool = False
    dpi: Optional[int] = None
    label: str = "Custom"

    def validate(self) -> None:
        if not (16 <= self.width <= 8000 and 16 <= self.height <= 8000):
            raise ValueError("Dimensions must be between 16 and 8000 pixels.")
        if self.max_kb <= 0:
            raise ValueError("The maximum size must be greater than zero.")
        if self.min_kb < 0 or self.min_kb > self.max_kb:
            raise ValueError("The minimum size must be between 0 and the maximum.")
        if self.fit not in ("crop", "pad"):
            raise ValueError("fit must be crop or pad.")
        if self.dpi is not None and not (36 <= self.dpi <= 1200):
            raise ValueError("DPI must be between 36 and 1200.")


# Common shapes of request. These are typical values, not any organisation's
# official specification -- the form being filled in is always the authority.
FORM_PRESETS: Dict[str, FormSpec] = {
    "photo": FormSpec(200, 230, 20, 50, "crop", label="Application photo"),
    "signature": FormSpec(140, 60, 10, 20, "pad", label="Signature"),
    "passport": FormSpec(413, 531, 30, 100, "crop", dpi=300, label="Passport 35 x 45 mm"),
    "square": FormSpec(600, 600, 50, 200, "crop", dpi=300, label="Square 2 x 2 in"),
    "thumbnail": FormSpec(150, 150, 5, 15, "crop", label="Profile thumbnail"),
}


def byte_bounds(spec: FormSpec) -> Tuple[int, int, bool]:
    """Byte limits for the range, and whether they satisfy both KB readings."""
    low = math.ceil(spec.min_kb * 1024)
    high = math.floor(spec.max_kb * 1000)
    if low <= high:
        return low, high, True
    return math.ceil(spec.min_kb * 1024), math.floor(spec.max_kb * 1024), False


# ------------------------------------------------------------------ framing


@lru_cache(maxsize=1)
def _face_cascade():
    try:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    except (AttributeError, cv2.error):
        return None
    return None if cascade.empty() else cascade


def _anchor(image: np.ndarray) -> Tuple[float, float, str]:
    """Where the crop should be centred, in source pixels, and why.

    A face wins if one is found: for a form photo the face is the subject. The
    fallback is the centroid of edge energy, which keeps the detailed part of
    the frame instead of a blank strip of background. A featureless image
    simply centres.
    """
    height, width = image.shape[:2]
    scale = min(1.0, 640 / max(height, width))
    small = (
        cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))),
                   interpolation=cv2.INTER_AREA)
        if scale < 1.0 else image
    )
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    cascade = _face_cascade()
    if cascade is not None:
        side = max(24, min(gray.shape) // 8)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                         minSize=(side, side))
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
            return (x + w / 2) / scale, (y + h / 2) / scale, "face"

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 3)
    total = float(energy.sum())
    if total < 1e-6:
        return width / 2, height / 2, "centre"
    ys, xs = np.indices(energy.shape)
    return (
        float((energy * xs).sum() / total) / scale,
        float((energy * ys).sum() / total) / scale,
        "detail",
    )


def prepare(image: np.ndarray, spec: FormSpec) -> Tuple[np.ndarray, Dict[str, object]]:
    """Crop or pad to the exact aspect, then resize to the exact dimensions."""
    height, width = image.shape[:2]
    info: Dict[str, object] = {"anchor": "pad", "upscaled": False,
                               "crop_box": (0, 0, width, height)}

    if spec.fit == "pad":
        scale = min(spec.width / width, spec.height / height)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        resized = cv2.resize(
            image, (new_w, new_h),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
        )
        framed = np.full((spec.height, spec.width, 3), 255, dtype=np.uint8)
        top, left = (spec.height - new_h) // 2, (spec.width - new_w) // 2
        framed[top:top + new_h, left:left + new_w] = resized
        info["upscaled"] = scale > 1.0001
    else:
        ratio = spec.width / spec.height
        if width / height > ratio:
            crop_w, crop_h = min(width, max(1, round(height * ratio))), height
        else:
            crop_w, crop_h = width, min(height, max(1, round(width / ratio)))
        cx, cy, how = _anchor(image)
        left = int(round(cx - crop_w / 2))
        # A face goes a little above centre, the way a portrait is framed.
        top = int(round(cy - (0.42 if how == "face" else 0.5) * crop_h))
        left = min(max(left, 0), width - crop_w)
        top = min(max(top, 0), height - crop_h)
        crop = image[top:top + crop_h, left:left + crop_w]
        framed = cv2.resize(
            crop, (spec.width, spec.height),
            interpolation=cv2.INTER_AREA if crop_w > spec.width else cv2.INTER_CUBIC,
        )
        info.update(anchor=how, crop_box=(left, top, left + crop_w, top + crop_h),
                    upscaled=crop_w < spec.width or crop_h < spec.height)

    if spec.grayscale:
        gray = cv2.cvtColor(framed, cv2.COLOR_RGB2GRAY)
        framed = np.dstack([gray] * 3)
    return framed, info


# ----------------------------------------------------------------- encoding


def _encode(picture: Image.Image, quality: int, subsampling: Optional[int],
            dpi: Optional[int]) -> bytes:
    buffer = io.BytesIO()
    # Baseline, not progressive: some upload validators reject progressive.
    options = {"format": "JPEG", "quality": int(quality), "optimize": True}
    if subsampling is not None:
        options["subsampling"] = subsampling
    if dpi:
        options["dpi"] = (dpi, dpi)
    picture.save(buffer, **options)
    return buffer.getvalue()


def _largest_fitting(picture, max_bytes, subsampling, dpi, counter):
    """Highest quality whose encode fits, by bisection. Form images are small,
    so a plain bisection over 1-100 costs a handful of milliseconds."""
    floor = _encode(picture, 1, subsampling, dpi)
    counter[0] += 1
    if len(floor) > max_bytes:
        return None, floor
    best, low, high = (1, floor), 2, 100
    while low <= high:
        mid = (low + high) // 2
        data = _encode(picture, mid, subsampling, dpi)
        counter[0] += 1
        if len(data) <= max_bytes:
            best, low = (mid, data), mid + 1
        else:
            high = mid - 1
    return best, floor


def _score(reference_gray: np.ndarray, data: bytes) -> float:
    decoded = np.asarray(Image.open(io.BytesIO(data)).convert("L"))
    return float(structural_similarity(reference_gray, decoded, data_range=255))


def pad_jpeg(data: bytes, total_bytes: int) -> bytes:
    """Grow a JPEG to `total_bytes` with comment segments, pixels untouched.

    The segments go after any APPn headers so JFIF and EXIF stay where strict
    parsers expect them. Any padding of 4 bytes or more lands exactly; a
    shortfall of 1-3 bytes rounds up to 4, the smallest legal segment.
    """
    if data[:2] != b"\xff\xd8":
        raise ValueError("Not a JPEG.")
    need = total_bytes - len(data)
    if need <= 0:
        return data

    position = 2
    while (position + 4 <= len(data) and data[position] == 0xFF
           and 0xE0 <= data[position + 1] <= 0xEF):
        position += 2 + int.from_bytes(data[position + 2:position + 4], "big")

    sizes: List[int] = []
    remaining = need
    while remaining > _SEGMENT_MAX:
        sizes.append(_SEGMENT_MAX)
        remaining -= _SEGMENT_MAX
    if remaining:
        if remaining >= _SEGMENT_MIN:
            sizes.append(remaining)
        elif sizes:
            sizes[-1] -= _SEGMENT_MIN - remaining  # borrow so the total is exact
            sizes.append(_SEGMENT_MIN)
        else:
            sizes.append(_SEGMENT_MIN)

    segments = bytearray()
    for size in sizes:
        payload = size - _SEGMENT_MIN
        filler = (_FILLER * (payload // len(_FILLER) + 1))[:payload]
        segments += b"\xff\xfe" + (payload + 2).to_bytes(2, "big") + filler
    return data[:position] + bytes(segments) + data[position:]


def encode_for_form(image_input, spec: FormSpec) -> Dict[str, object]:
    """Produce a JPEG that meets the spec, and report honestly if it cannot."""
    spec.validate()
    original = load_image(image_input)
    if original.ndim != 3:
        raise ValueError("Only RGB images are supported.")

    reference, framing = prepare(original, spec)
    low, high, strict = byte_bounds(spec)
    reference_gray = reference[..., 0] if spec.grayscale else cv2.cvtColor(
        reference, cv2.COLOR_RGB2GRAY)
    picture = Image.fromarray(reference_gray if spec.grayscale else reference)

    counter = [0]
    candidates = []
    smallest: Optional[Tuple[Optional[int], bytes]] = None
    # 4:2:0 wins at tight budgets; 4:4:4 keeps colour edges sharp when there
    # are bytes to spare. Try both and let the measurement choose.
    for subsampling in ([None] if spec.grayscale else [2, 0]):
        best, floor = _largest_fitting(picture, high, subsampling, spec.dpi, counter)
        if smallest is None or len(floor) < len(smallest[1]):
            smallest = (subsampling, floor)
        if best is not None:
            quality, data = best
            candidates.append({"subsampling": subsampling, "quality": quality,
                               "data": data, "ssim": _score(reference_gray, data)})

    messages: List[str] = []
    if not strict:
        messages.append(
            f"The {spec.min_kb:g}-{spec.max_kb:g} KB range is too narrow to hold "
            "under both 1000- and 1024-byte kilobytes, so 1024-byte KB were "
            "used for both limits."
        )
    if framing["upscaled"]:
        messages.append(
            f"The source is smaller than {spec.width} x {spec.height}, so it was "
            "enlarged; expect some softness."
        )

    if candidates:
        top = max(c["ssim"] for c in candidates)
        genuine = [c for c in candidates
                   if len(c["data"]) >= low and c["ssim"] >= top - SCORE_TOLERANCE]
        pool = genuine or candidates
        chosen = max(pool, key=lambda c: (c["ssim"], len(c["data"])))
    else:
        subsampling, floor = smallest
        chosen = {"subsampling": subsampling, "quality": 1, "data": floor,
                  "ssim": _score(reference_gray, floor)}
        messages.append(
            f"Even at the lowest quality this is {len(floor) / 1024:.1f} KB, over "
            f"the {spec.max_kb:g} KB limit at {spec.width} x {spec.height}. Try "
            "grayscale, or smaller dimensions if the form allows them."
        )

    natural = chosen["data"]
    data = natural
    if candidates and len(natural) < low:
        target = low + min(256, max(0, high - low) // 4)
        data = pad_jpeg(natural, min(target, high) if high >= low else low)
        messages.append(
            f"The best encoding that fits is {len(natural) / 1024:.1f} KB, under "
            f"the {spec.min_kb:g} KB minimum, so {len(data) - len(natural):,} "
            "bytes of JPEG comment data were added. The pixels are unchanged."
        )

    decoded = Image.open(io.BytesIO(data))
    decoded.load()
    subsampling_label = {None: "grayscale", 2: "4:2:0", 0: "4:4:4"}[chosen["subsampling"]]

    return {
        "raw_bytes": data,
        "unpadded_bytes": natural,
        "format": "JPEG",
        "extension": "jpg",
        "mime": "image/jpeg",
        "width": spec.width,
        "height": spec.height,
        "quality": int(chosen["quality"]),
        "subsampling": subsampling_label,
        "ssim": float(chosen["ssim"]),
        "size_bytes": len(data),
        "size_kb": len(data) / 1024.0,
        "size_kb_decimal": len(data) / 1000.0,
        "low_bytes": low,
        "high_bytes": high,
        "strict_units": strict,
        "meets_min": len(data) >= low,
        "meets_max": len(data) <= high,
        "padded_bytes": len(data) - len(natural),
        "anchor": framing["anchor"],
        "crop_box": framing["crop_box"],
        "upscaled": bool(framing["upscaled"]),
        "messages": messages,
        "encodes": counter[0],
        "spec": spec,
        "original_image": Image.fromarray(original),
        "reference_image": Image.fromarray(reference),
        "image": decoded.convert("RGB"),
    }
