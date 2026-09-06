import unittest

import numpy as np
from PIL import Image

from src.adaptive_compressor import AdaptiveImageCompressor
from src.image_features import (
    compute_quality_metrics,
    extract_feature_vector,
    image_stats,
    load_image,
)

SIZE = 400
RNG = np.random.default_rng(7)


def detailed_image(size: int = SIZE) -> np.ndarray:
    """High-frequency content, so a bigger budget can actually buy more bytes."""
    y, x = np.mgrid[0:size, 0:size]
    base = (np.sin(x / 5.0) * np.cos(y / 7.0) + 1) * 110
    noise = RNG.normal(0, 40, (size, size))
    channel = (base + noise).clip(0, 255)
    return np.dstack([channel, np.roll(channel, 20, 0), np.roll(channel, 20, 1)]).astype(
        np.uint8
    )


def flat_image(size: int = SIZE) -> np.ndarray:
    return np.full((size, size, 3), 130, dtype=np.uint8)


class QualityMetricTests(unittest.TestCase):
    def test_identical_images_score_perfectly(self):
        image = detailed_image()
        metrics = compute_quality_metrics(image, image)
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=5)
        self.assertAlmostEqual(metrics["mse"], 0.0, places=5)

    def test_degraded_image_scores_worse(self):
        image = detailed_image()
        noisy = np.clip(image.astype(np.int16) + 45, 0, 255).astype(np.uint8)
        self.assertLess(
            compute_quality_metrics(image, noisy)["ssim"],
            compute_quality_metrics(image, image)["ssim"],
        )


class FeatureTests(unittest.TestCase):
    def test_stats_are_finite(self):
        for value in image_stats(detailed_image()).values():
            self.assertTrue(np.isfinite(value), f"non-finite stat: {value}")

    def test_feature_vector_is_fixed_length_and_finite(self):
        a = extract_feature_vector(detailed_image(), 100)
        b = extract_feature_vector(flat_image(), 100)
        self.assertEqual(a.shape, b.shape)
        self.assertTrue(np.all(np.isfinite(a)))

    def test_target_size_reaches_the_features(self):
        # The target is an input to the model, so it must change the vector.
        image = detailed_image()
        self.assertFalse(
            np.array_equal(
                extract_feature_vector(image, 50), extract_feature_vector(image, 500)
            )
        )

    def test_load_image_accepts_pil_and_array(self):
        array = detailed_image()
        self.assertEqual(load_image(Image.fromarray(array)).shape, array.shape)
        self.assertEqual(load_image(array).shape, array.shape)


class CompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compressor = AdaptiveImageCompressor()
        cls.image = Image.fromarray(detailed_image())

    def test_returns_the_documented_keys(self):
        result = self.compressor.compress(self.image, 60)
        for key in (
            "resize_factor", "quality", "target_size_kb", "actual_size_kb",
            "ssim", "psnr", "mse", "image", "resized_image", "raw_bytes",
        ):
            self.assertIn(key, result)

    def test_never_exceeds_the_target(self):
        # A size budget that gets overshot is not a budget.
        for target in (20, 60, 120):
            result = self.compressor.compress(self.image, target)
            self.assertLessEqual(
                result["actual_size_kb"], target,
                f"target {target}KB overshot at {result['actual_size_kb']:.1f}KB",
            )

    def test_a_bigger_budget_buys_a_bigger_file(self):
        # The regression this suite exists for: the search used to rescale an
        # already-rescaled image, so the two factors multiplied and it could
        # only ever shrink. Three different targets returned one identical
        # file. On an image with enough detail to fill them, they must differ.
        sizes = [
            self.compressor.compress(self.image, t)["actual_size_kb"]
            for t in (20, 60, 120)
        ]
        self.assertLess(sizes[0], sizes[1])
        self.assertLess(sizes[1], sizes[2])

    def test_quality_and_resize_stay_in_range(self):
        result = self.compressor.compress(self.image, 60)
        self.assertGreaterEqual(result["quality"], 10)
        self.assertLessEqual(result["quality"], 95)
        self.assertGreater(result["resize_factor"], 0.0)
        self.assertLessEqual(result["resize_factor"], 1.0)

    def test_encoded_bytes_match_the_reported_size(self):
        result = self.compressor.compress(self.image, 60)
        self.assertAlmostEqual(
            len(result["raw_bytes"]) / 1024.0, result["actual_size_kb"], places=3
        )

    def test_unreachable_target_returns_max_fidelity(self):
        # A flat image cannot fill a large budget; it should come back at full
        # resolution rather than being needlessly downscaled.
        result = AdaptiveImageCompressor().compress(Image.fromarray(flat_image()), 5000)
        self.assertEqual(result["resize_factor"], 1.0)
        self.assertEqual(result["quality"], 95)

    def test_encode_scales_from_the_original(self):
        array = detailed_image()
        candidate, _ = AdaptiveImageCompressor._encode(array, 0.5, 80)
        self.assertEqual(candidate.shape[0], array.shape[0] // 2)

    def test_rejects_non_rgb_input(self):
        with self.assertRaises(ValueError):
            self.compressor.compress(np.zeros((50, 50), dtype=np.uint8), 50)


if __name__ == "__main__":
    unittest.main()
