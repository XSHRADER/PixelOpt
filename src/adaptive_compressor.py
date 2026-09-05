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

    def compress(self, image_input, target_size_kb: float) -> Dict[str, object]:
        image = load_image(image_input)
        if image.ndim != 3:
            raise ValueError("Only RGB images are supported.")

        if not self.ensure_model():
            self._train_fallback_model(image, target_size_kb)

        feature_vector = extract_feature_vector(image, target_size_kb)
        prediction = self.model.predict_single(feature_vector)
        resize_factor, quality = prediction

        resized = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
            (
                max(1, int(image.shape[1] * resize_factor)),
                max(1, int(image.shape[0] * resize_factor)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        resized_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        target = max(target_size_kb * 1024, 1)
        adjusted_quality = int(round(quality))
        adjusted_resize = resize_factor
        best_result = None
        best_error = float("inf")

        for q in range(max(10, adjusted_quality - 15), min(95, adjusted_quality + 15) + 1, 2):
            for r in np.linspace(max(0.2, adjusted_resize - 0.2), min(1.0, adjusted_resize + 0.2), 8):
                candidate = cv2.resize(
                    resized_rgb,
                    (
                        max(1, int(resized_rgb.shape[1] * r)),
                        max(1, int(resized_rgb.shape[0] * r)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
                encoded = cv2.imencode(".jpg", cv2.cvtColor(candidate, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])[1]
                size_bytes = len(encoded.tobytes())
                error = abs(size_bytes - target)
                if error < best_error:
                    best_error = error
                    best_result = {
                        "resize_factor": float(r),
                        "quality": int(q),
                        "size_bytes": size_bytes,
                        "image": candidate,
                    }

        if best_result is None:
            best_result = {
                "resize_factor": float(resize_factor),
                "quality": int(adjusted_quality),
                "size_bytes": 0,
                "image": resized_rgb,
            }

        compressed_rgb = best_result["image"]
        resized_image = Image.fromarray(np.asarray(compressed_rgb).astype(np.uint8))
        encoded = cv2.imencode(".jpg", cv2.cvtColor(compressed_rgb, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), int(best_result["quality"])])[1]
        final_bytes = encoded.tobytes()
        compressed_image = Image.open(io.BytesIO(final_bytes)).convert("RGB")
        metrics = compute_quality_metrics(image, np.asarray(compressed_image))

        return {
            "original_shape": image.shape,
            "resized_shape": resized_image.size[::-1] if hasattr(resized_image, "size") else compressed_rgb.shape[:2][::-1],
            "compressed_shape": compressed_rgb.shape,
            "resize_factor": float(best_result["resize_factor"]),
            "quality": int(best_result["quality"]),
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
