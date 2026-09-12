from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def flatten_transparency(picture: Image.Image) -> Image.Image:
    """Composite any alpha channel onto white and return an RGB image.

    JPEG has no alpha channel at all. A plain .convert("RGB") does not
    composite -- it simply drops the alpha, keeping whatever colour happened
    to sit under fully transparent pixels, which is usually black and
    sometimes arbitrary. Compositing makes the result defined.
    """
    if picture.mode == "P" and "transparency" in picture.info:
        picture = picture.convert("RGBA")
    if picture.mode in ("RGBA", "LA"):
        background = Image.new("RGBA", picture.size, (255, 255, 255, 255))
        picture = Image.alpha_composite(background, picture.convert("RGBA"))
    return picture.convert("RGB")


def has_transparency(picture: Image.Image) -> bool:
    """Whether an image actually uses alpha the opaque output cannot preserve.

    Having an alpha *channel* is not the same as using it. Plenty of PNGs are
    saved RGBA with every pixel fully opaque -- anything that went through a
    canvas, for one -- and warning about those is a false alarm, so check the
    channel contents rather than just the mode.
    """
    if picture.mode == "P":
        return "transparency" in picture.info
    if picture.mode not in ("RGBA", "LA"):
        return False
    minimum, _ = picture.getchannel("A").getextrema()
    return minimum < 255


def load_image(image_source) -> np.ndarray:
    """Load an image into a NumPy RGB array, honouring EXIF orientation.

    Cameras record rotation as an EXIF flag rather than rotating the pixels,
    so a portrait phone photo is stored landscape. Without exif_transpose the
    whole pipeline -- and the downloaded file -- comes out sideways.
    """
    if isinstance(image_source, (bytes, bytearray, memoryview)):
        image_source = io.BytesIO(bytes(image_source))

    # A path, or any file-like object. Web uploads and in-memory buffers arrive
    # as the latter, which is the common case for the app and was the one
    # originally missing here.
    if isinstance(image_source, (str, Path)) or hasattr(image_source, "read"):
        with Image.open(image_source) as handle:
            return np.array(flatten_transparency(ImageOps.exif_transpose(handle)))

    if isinstance(image_source, Image.Image):
        return np.array(flatten_transparency(ImageOps.exif_transpose(image_source)))

    if isinstance(image_source, np.ndarray):
        return image_source.astype(np.uint8)

    raise TypeError(
        "Unsupported image input type: " + type(image_source).__name__
    )


def image_stats(image: np.ndarray) -> dict:
    """Describe the image content that drives how expensive it is to encode."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    gradients_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.hypot(gradients_x, gradients_y)

    # One histogram, reused -- this used to be computed twice on the same data.
    counts = np.histogram(gray, 256, (0, 256))[0] / max(gray.size, 1)
    entropy = float(-np.sum(counts * np.log2(np.clip(counts, 1e-12, None))))

    return {
        "mean_intensity": float(gray.mean()),
        "brightness_std": float(gray.std()),
        "edge_density": float(edges.mean()),
        "sharpness": float(np.var(laplacian)),
        "texture": float(np.mean(gradient_mag**2)),
        "color_var": float(np.var(image.reshape(-1, 3), axis=0).mean()),
        "entropy": entropy,
    }


def compute_quality_metrics(original: np.ndarray, compressed: np.ndarray) -> dict:
    """Compare a compressed result against the original at full resolution.

    The output is usually smaller than the input, so it is upscaled back
    before comparison: the question being answered is "how good a replacement
    for the original is this file", which has to include the resolution lost.
    """
    original = original.astype(np.uint8)
    compressed = compressed.astype(np.uint8)

    if original.shape[:2] != compressed.shape[:2]:
        shrinking = (
            compressed.shape[0] > original.shape[0]
            or compressed.shape[1] > original.shape[1]
        )
        compressed = cv2.resize(
            compressed,
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
        )

    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)

    mse = float(mean_squared_error(original_gray, compressed_gray))
    # A lossless encode reproduces the reference exactly, so the error is zero
    # and PSNR is genuinely infinite. Now that PNG is a candidate the search
    # can pick, that is a normal outcome rather than a curiosity -- compute it
    # directly instead of letting skimage divide by zero and warn every time.
    psnr = (
        float("inf")
        if mse <= 0
        else float(
            peak_signal_noise_ratio(original_gray, compressed_gray, data_range=255)
        )
    )

    return {
        "ssim": float(
            structural_similarity(original_gray, compressed_gray, data_range=255)
        ),
        "psnr": psnr,
        "mse": mse,
    }
