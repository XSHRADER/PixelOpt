from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from src.adaptive_compressor import AdaptiveImageCompressor
from src.image_features import extract_feature_vector


def synthetic_dataset(image_dir: str, target_size_kb: float = 200.0, samples_per_image: int = 40):
    features = []
    targets = []
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    for filename in os.listdir(image_dir):
        path = os.path.join(image_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            image = Image.open(path).convert("RGB")
            image_np = np.array(image)
        except Exception:
            continue

        height, width = image_np.shape[:2]
        for idx in range(samples_per_image):
            resize = 0.35 + (0.95 - 0.35) * (idx / max(samples_per_image - 1, 1))
            quality = 18 + (90 - 18) * (idx / max(samples_per_image - 1, 1))
            resized = cv2.resize(
                cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR),
                (max(1, int(width * resize)), max(1, int(height * resize))),
                interpolation=cv2.INTER_AREA,
            )
            encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
            if len(encoded.tobytes()) == 0:
                continue
            features.append(extract_feature_vector(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), target_size_kb))
            targets.append(np.array([resize, quality], dtype=np.float32))

    if not features:
        raise ValueError("No valid images were found to train on.")

    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def main():
    image_dir = os.path.join("data", "images")
    model = AdaptiveImageCompressor()

    if os.path.exists(image_dir):
        X, y = synthetic_dataset(image_dir)
        metrics = model.model.train(X, y)
        print(f"Training complete. MAE: {metrics['mae']:.4f}, R^2: {metrics['r2']:.4f}")
    else:
        print("No dataset directory found. Training a synthetic fallback model using sample patterns.")
        sample = np.zeros((800, 600, 3), dtype=np.uint8)
        sample[..., 0] = 120
        sample[..., 1] = 160
        sample[..., 2] = 200
        features = []
        targets = []
        for quality in np.linspace(20, 95, 25):
            for resize in np.linspace(0.3, 1.0, 15):
                resized = cv2.resize(sample, (max(1, int(600 * resize)), max(1, int(800 * resize))), interpolation=cv2.INTER_AREA)
                encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])[1]
                if not len(encoded.tobytes()):
                    continue
                features.append(extract_feature_vector(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), 200.0))
                targets.append(np.array([resize, quality], dtype=np.float32))
        metrics = model.model.train(features, targets)
        print(f"Fallback training complete. MAE: {metrics['mae']:.4f}, R^2: {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
