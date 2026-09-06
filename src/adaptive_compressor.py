from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .compression_model import AdaptiveCompressionModel
from .image_features import compute_quality_metrics, extract_feature_vector, load_image


class AdaptiveImageCompressor:
    def __init__(self, model_path: str = "models/adaptive_compression_model.joblib"):
        self.model = AdaptiveCompressionModel(model_path=model_path)
        self.model_cache = {}

    def ensure_model(self) -> bool:
        if self.model.model is not None:
            return True
        return self.model.load()

    def _generate_synthetic_dataset(self, image_path: str, target_size_kb: float, count: int = 20):
        image = load_image(image_path)
        height, width = image.shape[:2]
        features = []
        targets = []

        for i in range(count):
            resize_factor = 0.4 + (0.8 - 0.4) * (i / max(count - 1, 1))
            quality = 20 + (90 - 20) * (i / max(count - 1, 1))
            resized = cv2.resize(image, (max(1, int(width * resize_factor)), max(1, int(height * resize_factor))), interpolation=cv2.INTER_AREA)
            encoded = cv2.imencode(".jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
            image_bytes = encoded.tobytes()
            final_size_kb = len(image_bytes) / 1024.0
            target_rel = final_size_kb / max(target_size_kb, 1e-6)
            feature_vector = extract_feature_vector(resized, target_size_kb)
            features.append(feature_vector)
            targets.append(np.array([resize_factor, quality], dtype=np.float32))

        return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)

    def train_on_file(self, image_path: str, target_size_kb: float, samples: int = 20):
        features, targets = self._generate_synthetic_dataset(image_path, target_size_kb, count=samples)
        return self.model.train(features, targets)

    RESIZE_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.22, 0.15)
    QUALITY_MIN = 10
    QUALITY_MAX = 95

    @staticmethod
    def _encode(image_rgb: np.ndarray, resize: float, quality: int):
        """Resize the ORIGINAL image by `resize` and JPEG-encode it.

        Always scaling from the original matters: the previous version
        resized the already-resized image, so the model's resize factor and
        the search factor multiplied together and the search could only ever
        shrink further. A large target was unreachable no matter what.
        """
        if resize >= 0.999:
            candidate = image_rgb
        else:
            height, width = image_rgb.shape[:2]
            candidate = cv2.resize(
                image_rgb,
                (max(1, int(width * resize)), max(1, int(height * resize))),
                interpolation=cv2.INTER_AREA,
            )
        encoded = cv2.imencode(
            ".jpg",
            cv2.cvtColor(candidate, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
        )[1]
        return candidate, encoded.tobytes()

    def _search(self, image_rgb: np.ndarray, target_bytes: float, seed_resize: float):
        """Find the highest-fidelity encoding that still fits `target_bytes`.

        File size is monotonic in JPEG quality, so for each resize factor the
        best quality is found by binary search instead of a fixed +/-15 sweep
        around the model's guess -- which is what stopped the old loop from
        ever reaching the target when the guess was wrong.

        The grid is ordered outward from the model's predicted resize factor,
        so the prediction still steers the search; it just no longer bounds it.
        """
        grid = sorted(self.RESIZE_GRID, key=lambda r: abs(r - seed_resize))
        tried = []

        for resize in grid:
            low, high = self.QUALITY_MIN, self.QUALITY_MAX
            while low <= high:
                mid = (low + high) // 2
                candidate, data = self._encode(image_rgb, resize, mid)
                tried.append((len(data), resize, mid, candidate, data))
                if len(data) <= target_bytes:
                    low = mid + 1     # room to spare -- try higher quality
                else:
                    high = mid - 1    # too big -- back off

        fits = [t for t in tried if t[0] <= target_bytes]
        if fits:
            # Closest to the budget from below, preferring less downscaling
            # when two candidates land on the same size.
            return max(fits, key=lambda t: (t[0], t[1]))
        # Nothing fits even at the smallest settings; return the smallest.
        return min(tried, key=lambda t: t[0])

    def compress(self, image_input, target_size_kb: float) -> Dict[str, object]:
        image = load_image(image_input)
        if image.ndim != 3:
            raise ValueError("Only RGB images are supported.")

        if not self.ensure_model():
            self._train_fallback_model(image, target_size_kb)

        feature_vector = extract_feature_vector(image, target_size_kb)
        prediction = self.model.predict_single(feature_vector)
        resize_factor, quality = prediction

        target_bytes = max(target_size_kb * 1024, 1)
        size_bytes, best_resize, best_quality, best_image, final_bytes = self._search(
            image, target_bytes, float(resize_factor)
        )

        resized_image = Image.fromarray(np.asarray(best_image).astype(np.uint8))
        compressed_image = Image.open(io.BytesIO(final_bytes)).convert("RGB")
        metrics = compute_quality_metrics(image, np.asarray(compressed_image))

        return {
            "original_shape": image.shape,
            "resized_shape": best_image.shape[:2][::-1],
            "compressed_shape": best_image.shape,
            "resize_factor": float(best_resize),
            "quality": int(best_quality),
            "target_size_kb": float(target_size_kb),
            "actual_size_kb": len(final_bytes) / 1024.0,
            "ssim": metrics["ssim"],
            "psnr": metrics["psnr"],
            "mse": metrics["mse"],
            "resized_image": resized_image,
            "image": compressed_image,
            "prediction": (resize_factor, quality),
            "raw_bytes": final_bytes,
        }

    def _train_fallback_model(self, image: np.ndarray, target_size_kb: float):
        feature_list = []
        target_list = []
        height, width = image.shape[:2]
        for quality in np.linspace(20, 95, 25):
            for resize in np.linspace(0.3, 1.0, 15):
                resized = cv2.resize(
                    image,
                    (max(1, int(width * resize)), max(1, int(height * resize))),
                    interpolation=cv2.INTER_AREA,
                )
                encoded = cv2.imencode(".jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
                size_kb = len(encoded.tobytes()) / 1024.0
                if size_kb <= 0:
                    continue
                feature_list.append(extract_feature_vector(resized, target_size_kb))
                target_list.append(np.array([resize, quality], dtype=np.float32))

        if not feature_list:
            raise ValueError("Unable to build a fallback dataset from the input image.")

        self.model.train(feature_list, target_list)
        self.model.model = self.model.model
