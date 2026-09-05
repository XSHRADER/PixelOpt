from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


def load_image(image_source) -> np.ndarray:
    """Load an image into a NumPy BGR array."""
    if isinstance(image_source, str):
        image = cv2.imread(image_source)
        if image is None:
            raise ValueError(f"Unable to read image from path: {image_source}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if isinstance(image_source, Image.Image):
        return np.array(image_source.convert("RGB"))

    if isinstance(image_source, np.ndarray):
        return image_source.astype(np.uint8)

    raise TypeError("Unsupported image input type")


def image_stats(image: np.ndarray) -> dict:
    """Compute basic image statistics relevant to compression."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    gradients_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradients_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.hypot(gradients_x, gradients_y)

    mean_intensity = float(gray.mean())
    brightness_std = float(gray.std())
    edge_density = float(edges.mean())
    sharpness = float(np.var(laplacian))
    texture = float(np.mean(gradient_mag**2))
    color_var = float(np.var(image.reshape(-1, 3), axis=0).mean())
    entropy = float(-np.sum((np.histogram(gray, 256, (0, 256))[0] / max(gray.size, 1))
                            * np.log2(np.clip((np.histogram(gray, 256, (0, 256))[0] / max(gray.size, 1)), 1e-12, None))))

    return {
        "mean_intensity": mean_intensity,
        "brightness_std": brightness_std,
        "edge_density": edge_density,
        "sharpness": sharpness,
        "texture": texture,
        "color_var": color_var,
        "entropy": entropy,
    }


def extract_feature_vector(image: np.ndarray, target_size_kb: float) -> np.ndarray:
    """Generate feature vector for model prediction."""
    height, width = image.shape[:2]
    aspect_ratio = width / max(height, 1)
    mean_color = image.mean(axis=(0, 1))
    rgb_std = image.std(axis=(0, 1))
    stats = image_stats(image)

    features = [
        width,
        height,
        width * height,
        aspect_ratio,
        target_size_kb,
        stats["mean_intensity"],
        stats["brightness_std"],
        stats["edge_density"],
        stats["sharpness"],
        stats["texture"],
        stats["color_var"],
        stats["entropy"],
        float(mean_color[0]),
        float(mean_color[1]),
        float(mean_color[2]),
        float(rgb_std[0]),
        float(rgb_std[1]),
        float(rgb_std[2]),
    ]
    return np.array(features, dtype=np.float32)


def rgb_to_hsv_stats(image: np.ndarray) -> Tuple[float, float, float]:
    rgb = image.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2HSV)
    return float(hsv[..., 0].mean()), float(hsv[..., 1].mean()), float(hsv[..., 2].mean())


def compute_quality_metrics(original: np.ndarray, compressed: np.ndarray) -> dict:
    """Compute SSIM, PSNR and MSE between original and reconstructed images.

    If the compressed image has different dimensions than the original, resize
    the compressed image to the original's dimensions before computing metrics.
    This prevents ValueError from skimage when shapes differ.
    """
    # Defensive casts to uint8
    original = original.astype(np.uint8)
    compressed = compressed.astype(np.uint8)

    # If shapes differ (height, width), resize the compressed image to match
    if original.shape[:2] != compressed.shape[:2]:
        compressed = cv2.resize(
            compressed,
            (original.shape[1], original.shape[0]),
            interpolation=(cv2.INTER_AREA if (compressed.shape[0] > original.shape[0] or compressed.shape[1] > original.shape[1]) else cv2.INTER_LINEAR),
        )

    original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    compressed_gray = cv2.cvtColor(compressed, cv2.COLOR_RGB2GRAY)

    ssim_value = structural_similarity(original_gray, compressed_gray, data_range=255)
    mse_value = mean_squared_error(original_gray, compressed_gray)
    psnr_value = peak_signal_noise_ratio(original_gray, compressed_gray, data_range=255)

    return {
        "ssim": float(ssim_value),
        "psnr": float(psnr_value),
        "mse": float(mse_value),
    }


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
