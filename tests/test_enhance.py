"""Tests for enhancement and the two-reference pipeline."""

import unittest

import cv2
import numpy as np
from PIL import Image

from pixelopt.enhance import (
    AUTO_DENOISE_THRESHOLD,
    Enhancements,
    apply,
    auto_enhancements,
    estimate_noise,
    preset,
    with_overrides,
)
from pixelopt.image_features import compute_quality_metrics
from pixelopt.pipeline import knee_point, process, rate_distortion_curve

SIZE = 320


def clean_image(seed: int = 3, size: int = SIZE) -> np.ndarray:
    """Smooth content with a little real structure worth preserving."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:size, 0:size]
    base = cv2.GaussianBlur(rng.normal(128, 60, (size, size)), (0, 0), 10)
    base = (base - base.mean()) / max(base.std(), 1e-6) * 45 + 128
    base += np.sin(x / 20.0) * 18
    channel = np.clip(base, 0, 255)
    return np.dstack([channel] * 3).astype(np.uint8)


def noisy(image: np.ndarray, sigma: float, seed: int = 77) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(
        image.astype(np.float64) + rng.normal(0, sigma, image.shape), 0, 255
    ).astype(np.uint8)


class NoiseEstimateTests(unittest.TestCase):
    def test_estimate_rises_with_noise(self):
        base = clean_image()
        readings = [estimate_noise(noisy(base, s)) for s in (0, 5, 15, 30)]
        self.assertEqual(readings, sorted(readings), f"not monotonic: {readings}")

    def test_clean_image_reads_near_zero(self):
        self.assertLess(estimate_noise(clean_image()), AUTO_DENOISE_THRESHOLD)

    def test_periodic_texture_is_not_mistaken_for_noise(self):
        # The kernel annihilates locally linear intensity, so even a tight
        # sinusoid should not read as noise. If this breaks, auto mode will
        # start denoising detailed images that do not need it.
        y, x = np.mgrid[0:SIZE, 0:SIZE]
        texture = np.clip((np.sin(x / 6.0) * np.cos(y / 7.0) + 1) * 110, 0, 255)
        stack = np.dstack([texture] * 3).astype(np.uint8)
        self.assertLess(estimate_noise(stack), AUTO_DENOISE_THRESHOLD)

    def test_estimate_ignores_image_size(self):
        # Noise is a sensor property, not a function of megapixels.
        small = noisy(clean_image(size=200), 18)
        large = noisy(clean_image(size=400), 18)
        self.assertAlmostEqual(estimate_noise(small), estimate_noise(large), delta=3.0)


class AutoEnhancementTests(unittest.TestCase):
    def test_clean_image_is_left_alone(self):
        # Silently altering an image that needs nothing is not an improvement.
        self.assertFalse(auto_enhancements(clean_image()).any_enabled())

    def test_noisy_image_gets_denoised(self):
        settings = auto_enhancements(noisy(clean_image(), 25))
        self.assertGreater(settings.denoise, 0)

    def test_strength_tracks_the_noise(self):
        light = auto_enhancements(noisy(clean_image(), 8)).denoise
        heavy = auto_enhancements(noisy(clean_image(), 30)).denoise
        self.assertLess(light, heavy)

    def test_auto_never_changes_the_look(self):
        # Contrast, white balance and saturation change how a photograph is
        # meant to look. That belongs to whoever took it, not to us.
        settings = auto_enhancements(noisy(clean_image(), 25))
        self.assertEqual(settings.contrast, 0.0)
        self.assertFalse(settings.white_balance)
        self.assertEqual(settings.saturation, 1.0)


class EnhancementOperationTests(unittest.TestCase):
    def test_denoising_reduces_measured_noise(self):
        image = noisy(clean_image(), 25)
        before = estimate_noise(image)
        after = estimate_noise(apply(image, Enhancements(denoise=12)))
        self.assertLess(after, before)

    def test_denoising_moves_toward_the_truth(self):
        # The claim the whole feature rests on, stated as a test: against a
        # clean ground truth neither path saw, denoising is an improvement.
        truth = clean_image()
        image = noisy(truth, 25)
        raw = compute_quality_metrics(truth, image)["ssim"]
        cleaned = compute_quality_metrics(
            truth, apply(image, Enhancements(denoise=14))
        )["ssim"]
        self.assertGreater(cleaned, raw)

    def test_sharpening_adds_high_frequency_detail(self):
        image = clean_image()
        sharper = apply(image, Enhancements(sharpen=0.8))
        self.assertGreater(
            cv2.Laplacian(cv2.cvtColor(sharper, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var(),
            cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var(),
        )

    def test_white_balance_neutralises_a_cast(self):
        image = clean_image()
        cast = image.copy()
        cast[..., 0] = np.clip(cast[..., 0].astype(np.int16) + 40, 0, 255)
        corrected = apply(cast, Enhancements(white_balance=True))
        spread_before = np.ptp(cast.reshape(-1, 3).mean(axis=0))
        spread_after = np.ptp(corrected.reshape(-1, 3).mean(axis=0))
        self.assertLess(spread_after, spread_before)

    def test_auto_level_widens_the_range(self):
        flat = np.full((SIZE, SIZE, 3), 128, dtype=np.uint8)
        flat[:, :100] = 120
        flat[:, 200:] = 136
        widened = apply(flat, Enhancements(auto_level=True))
        self.assertGreater(np.ptp(widened), np.ptp(flat))

    def test_operations_preserve_shape_and_dtype(self):
        image = noisy(clean_image(), 10)
        for settings in (
            Enhancements(denoise=8),
            Enhancements(sharpen=0.5),
            Enhancements(contrast=2.0),
            Enhancements(white_balance=True),
            Enhancements(auto_level=True),
            Enhancements(saturation=1.3),
            preset("photo"),
            preset("scan"),
        ):
            out = apply(image, settings)
            self.assertEqual(out.shape, image.shape)
            self.assertEqual(out.dtype, np.uint8)

    def test_nothing_enabled_is_a_no_op(self):
        image = clean_image()
        self.assertTrue(np.array_equal(apply(image, Enhancements()), image))

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError):
            preset("nope")

    def test_overrides_only_replace_supplied_fields(self):
        base = preset("photo")
        merged = with_overrides(base, denoise=1.0, sharpen=None)
        self.assertEqual(merged.denoise, 1.0)
        self.assertEqual(merged.sharpen, base.sharpen)


class PipelineTests(unittest.TestCase):
    def test_reports_both_references_separately(self):
        image = Image.fromarray(noisy(clean_image(), 22))
        result = process(image, 60, enhancements=Enhancements(denoise=12))
        # Fidelity is against the enhanced reference, drift against the
        # original. Conflating them is the bug this split exists to prevent.
        self.assertIn("fidelity_ssim", result)
        self.assertIn("drift_ssim", result)
        self.assertLess(result["drift_ssim"], 1.0)
        self.assertGreater(result["fidelity_ssim"], 0.5)

    def test_no_enhancement_means_no_drift(self):
        result = process(Image.fromarray(clean_image()), 60)
        self.assertEqual(result["drift_ssim"], 1.0)
        self.assertFalse(result["enhanced"])

    def test_denoising_lifts_fidelity_at_a_tight_budget(self):
        # Only at a tight budget. Fidelity measures how faithfully the encoder
        # reproduced what it was handed, so given plenty of bytes both paths
        # score ~0.99 and there is nothing to lift -- 6 KB on this image is
        # ~0.4 bpp, where noise genuinely competes with detail for bits.
        image = Image.fromarray(noisy(clean_image(), 25))
        plain = process(image, 6)["fidelity_ssim"]
        cleaned = process(image, 6, enhancements=Enhancements(denoise=14))[
            "fidelity_ssim"
        ]
        self.assertGreater(cleaned, plain)

    def test_noise_is_reported_before_and_after(self):
        result = process(
            Image.fromarray(noisy(clean_image(), 25)),
            60,
            enhancements=Enhancements(denoise=14),
        )
        self.assertLess(result["noise_after"], result["noise_before"])

    def test_budget_is_still_respected_with_enhancement(self):
        result = process(
            Image.fromarray(noisy(clean_image(), 20)), 40, enhancements=preset("scan")
        )
        self.assertLessEqual(result["actual_size_kb"], 40)

    def test_rejects_non_rgb_input(self):
        with self.assertRaises(ValueError):
            process(np.zeros((40, 40), dtype=np.uint8), 40)


class RateDistortionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = np.mgrid[0:SIZE, 0:SIZE]
        rng = np.random.default_rng(4)
        channel = np.clip(
            (np.sin(x / 7.0) * np.cos(y / 9.0) + 1) * 110
            + rng.normal(0, 30, (SIZE, SIZE)),
            0,
            255,
        )
        cls.image = Image.fromarray(np.dstack([channel] * 3).astype(np.uint8))

    def test_curve_is_ordered_and_complete(self):
        points = rate_distortion_curve(self.image, [10, 20, 40, 80])
        self.assertEqual(len(points), 4)
        self.assertEqual(
            [p["target_kb"] for p in points], sorted(p["target_kb"] for p in points)
        )

    def test_quality_does_not_fall_as_the_budget_grows(self):
        points = [
            p for p in rate_distortion_curve(self.image, [10, 20, 40, 80])
            if not p["capped"]
        ]
        scores = [p["ssim"] for p in points]
        for earlier, later in zip(scores, scores[1:]):
            self.assertLessEqual(earlier, later + 1e-6)

    def test_knee_needs_three_usable_points(self):
        self.assertIsNone(knee_point(rate_distortion_curve(self.image, [40])))

    def test_knee_lands_inside_the_curve(self):
        points = rate_distortion_curve(self.image, [5, 10, 20, 40, 80, 160])
        knee = knee_point(points)
        if knee is not None:
            budgets = [p["actual_kb"] for p in points if not p["capped"]]
            self.assertGreaterEqual(knee["actual_kb"], min(budgets))
            self.assertLessEqual(knee["actual_kb"], max(budgets))


if __name__ == "__main__":
    unittest.main()
