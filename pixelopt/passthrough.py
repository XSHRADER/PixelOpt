"""Keep the original file when it already fits.

Re-encoding a JPEG that is already under budget is pure loss: decoding and
encoding again throws away detail a second time -- generation loss -- to save
bytes nobody asked to save, and the old CLI could even report the result as
LARGER than the input. When the upload already fits and nothing about its
pixels needs to change, the best possible output is the upload itself.

It is not returned verbatim, though. Phone photos carry EXIF with GPS
coordinates, and re-encoding used to strip that as a side effect. So the file
is rewritten losslessly at the container level instead: metadata segments and
chunks are removed, the compressed image data is copied byte for byte, and
the colour profile is kept because dropping it changes how the picture looks.

Two guards make this safe rather than hopeful:

- The stripped file is decoded and compared with the reference pixel for
  pixel. Anything that does not match exactly is refused. That one check also
  covers the case this design cannot handle -- a JPEG whose EXIF orientation
  flag rotates it on display. Stripping the flag would turn it sideways, the
  decoded pixels stop matching, and the file is re-encoded as before.
- Transparent and animated images are refused outright. The app promises an
  opaque, single-frame result, and a passthrough must not quietly break that.
"""

from __future__ import annotations

import io
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

from .image_features import has_transparency, load_image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# PNG ancillary chunks that carry text or EXIF and nothing that affects pixels.
_PNG_DROP = {b"tEXt", b"zTXt", b"iTXt", b"eXIf", b"tIME"}

_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_EXTENSION = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}


class NotEligible(ValueError):
    """The file cannot be passed through; the message says why."""


def sniff(data: bytes) -> Optional[str]:
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if data[:8] == PNG_SIGNATURE:
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


# ------------------------------------------------------------------- JPEG


def _jpeg_scan_end(data: bytes, start: int) -> int:
    """Offset just past the EOI marker, starting from the first SOS.

    Walks marker segments by their length fields instead of searching for the
    FFD9 byte pair, which can legitimately appear inside a table segment.
    Inside entropy-coded data, 0xFF is either stuffed (FF00), a restart marker
    (FFD0-FFD7) or fill (FFFF). Anything after EOI -- such as the secondary
    images of an iPhone multi-picture file -- is dropped.
    """
    j = start
    size = len(data)
    while j + 1 < size:
        if data[j] != 0xFF:
            j += 1
            continue
        marker = data[j + 1]
        if marker == 0xFF:
            j += 1
        elif marker == 0x00 or 0xD0 <= marker <= 0xD7:
            j += 2
        elif marker == 0xD9:
            return j + 2
        else:
            if j + 4 > size:
                break
            j += 2 + int.from_bytes(data[j + 2:j + 4], "big")
    return size


def strip_jpeg(data: bytes) -> bytes:
    """Remove EXIF, XMP, IPTC and comments; keep JFIF, ICC and Adobe segments."""
    if data[:2] != b"\xff\xd8":
        raise NotEligible("not a JPEG")
    out = bytearray(b"\xff\xd8")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise NotEligible("unexpected byte in JPEG header")
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte before a marker
            i += 1
            continue
        if marker == 0xDA:  # start of scan: the image data, copied verbatim
            out += data[i:_jpeg_scan_end(data, i)]
            return bytes(out)
        length = int.from_bytes(data[i + 2:i + 4], "big")
        if length < 2 or i + 2 + length > len(data):
            raise NotEligible("truncated JPEG segment")
        segment = data[i:i + 2 + length]
        keep = True
        if 0xE1 <= marker <= 0xEF and marker != 0xEE:  # APP1-APP15 except APP14 (Adobe)
            keep = marker == 0xE2 and segment[4:16] == b"ICC_PROFILE\x00"
        elif marker == 0xFE:  # comment
            keep = False
        if keep:
            out += segment
        i += 2 + length
    raise NotEligible("JPEG has no image data")


# -------------------------------------------------------------------- PNG


def strip_png(data: bytes) -> bytes:
    if data[:8] != PNG_SIGNATURE:
        raise NotEligible("not a PNG")
    out = bytearray(PNG_SIGNATURE)
    i = 8
    while i + 12 <= len(data):
        length = int.from_bytes(data[i:i + 4], "big")
        kind = data[i + 4:i + 8]
        end = i + 12 + length
        if end > len(data):
            raise NotEligible("truncated PNG chunk")
        if kind == b"acTL":
            raise NotEligible("animated PNG")
        if kind not in _PNG_DROP:
            out += data[i:end]  # CRC travels with the chunk, so it stays valid
        i = end
        if kind == b"IEND":
            return bytes(out)
    raise NotEligible("PNG has no end chunk")


# ------------------------------------------------------------------- WebP


def strip_webp(data: bytes) -> bytes:
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise NotEligible("not a WebP")
    chunks = []
    i = 12
    while i + 8 <= len(data):
        fourcc = data[i:i + 4]
        size = int.from_bytes(data[i + 4:i + 8], "little")
        end = i + 8 + size + (size & 1)  # chunks are padded to even length
        if i + 8 + size > len(data):
            raise NotEligible("truncated WebP chunk")
        chunk = data[i:min(end, len(data))]
        if fourcc in (b"ANIM", b"ANMF"):
            raise NotEligible("animated WebP")
        if fourcc == b"VP8X":
            # Clear the EXIF (0x08) and XMP (0x04) flags to match the chunks
            # being dropped; ICC and alpha flags are left alone.
            chunk = chunk[:8] + bytes([chunk[8] & ~0x0C]) + chunk[9:]
        if fourcc not in (b"EXIF", b"XMP "):
            chunks.append(chunk)
        i = end
    body = b"WEBP" + b"".join(chunks)
    return b"RIFF" + len(body).to_bytes(4, "little") + body


_STRIPPERS = {"JPEG": strip_jpeg, "PNG": strip_png, "WEBP": strip_webp}


# ------------------------------------------------------------------ entry


def try_passthrough(
    raw: bytes,
    reference: np.ndarray,
    max_bytes: float,
    image_format: Optional[str] = None,
) -> Tuple[Optional[Dict[str, object]], str]:
    """The original, minus metadata, if it can stand in for an encode.

    Returns (result, reason). `result` is None when the file is not eligible,
    and `reason` says why in words fit for a user.
    """
    fmt = sniff(raw)
    if fmt is None:
        return None, "the original is not a JPEG, PNG or WebP"
    if image_format is not None and image_format.upper() != fmt:
        return None, f"the original is {fmt}, but {image_format.upper()} was requested"

    try:
        stripped = _STRIPPERS[fmt](raw)
        with Image.open(io.BytesIO(stripped)) as picture:
            picture.load()
            if getattr(picture, "n_frames", 1) > 1:
                return None, "the original is animated"
            if has_transparency(picture):
                return None, "the original has transparency"
            if fmt == "JPEG" and picture.mode not in ("RGB", "L"):
                return None, f"the original is a {picture.mode} JPEG"
        decoded = load_image(io.BytesIO(stripped))
    except NotEligible as reason:
        return None, str(reason)
    except Exception as error:  # a file Pillow cannot re-read is not eligible
        return None, f"the original could not be re-read ({type(error).__name__})"

    if len(stripped) > max_bytes:
        return None, "the original is over the budget"
    if decoded.shape != reference.shape or not np.array_equal(decoded, reference):
        return None, "the original has an orientation flag or other state that changes its pixels"

    height, width = reference.shape[:2]
    picture = Image.fromarray(reference)
    return {
        "raw_bytes": stripped,
        "format": fmt,
        "extension": _EXTENSION[fmt],
        "mime": _MIME[fmt],
        "width": width,
        "height": height,
        "original_shape": reference.shape,
        "resized_shape": (width, height),
        "compressed_shape": reference.shape,
        "resize_factor": 1.0,
        "quality": None,
        "actual_size_kb": len(stripped) / 1024.0,
        "encodes": 0,
        # Identical pixels, verified above.
        "ssim": 1.0,
        "psnr": float("inf"),
        "mse": 0.0,
        "resized_image": picture,
        "image": picture,
        "passthrough": True,
        "metadata_removed_bytes": len(raw) - len(stripped),
    }, "the original already fits"
