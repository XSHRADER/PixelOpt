import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pixelopt.adaptive_compressor import (
    BPP_TARGET,
    RESIZE_GRID,
    AdaptiveImageCompressor,
)
from pixelopt.cli import main as cli_main
from pixelopt.image_features import (
    compute_quality_metrics,
    has_transparency,
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

    def test_search_is_cheap(self):
        # The old exhaustive sweep spent ~69 encodes per call. Interpolating on
        # log-size costs ~3 per lossy candidate and 1 per lossless one, so with
        # three formats over four resize candidates the benchmark measures
        # mean 20, max 37. 45 leaves headroom without hiding a regression back
        # toward sweeping.
        result = self.compressor.compress(self.image, 60)
        self.assertLess(result["encodes"], 45, "search got expensive again")

    def test_reported_bytes_are_the_downloadable_bytes(self):
        # The app hands raw_bytes to the download button, so it has to be the
        # same file the metrics describe.
        result = self.compressor.compress(self.image, 60)
        decoded = Image.open(io.BytesIO(result["raw_bytes"]))
        self.assertEqual(decoded.size, (result["width"], result["height"]))

    def test_picks_a_supported_format(self):
        result = self.compressor.compress(self.image, 60)
        self.assertIn(result["format"], ("JPEG", "WEBP", "PNG"))

    def test_lossless_wins_on_flat_art(self):
        # A PNG of glyph-like art is both smaller and perfect. If the search
        # picks a lossy encoder here it is leaving free quality on the table.
        art = np.full((300, 300, 3), 245, dtype=np.uint8)
        art[40:80, :, :] = 30
        art[:, 120:150, :] = [200, 40, 60]
        result = AdaptiveImageCompressor().compress(Image.fromarray(art), 60)
        self.assertEqual(result["format"], "PNG")
        self.assertAlmostEqual(result["ssim"], 1.0, places=6)

    def test_lossless_is_not_used_when_nothing_fits(self):
        # When even the smallest encode overshoots, the goal is the smallest
        # possible file -- the worst possible moment to pick a lossless codec.
        result = AdaptiveImageCompressor().compress(self.image, 0.5)
        self.assertNotEqual(result["format"], "PNG")

    def test_format_can_be_forced(self):
        for fmt in ("JPEG", "WEBP"):
            result = self.compressor.compress(self.image, 60, image_format=fmt)
            self.assertEqual(result["format"], fmt)

    def test_rejects_unknown_format(self):
        with self.assertRaises(ValueError):
            AdaptiveImageCompressor(formats=["TIFF"])


class ResizeRuleTests(unittest.TestCase):
    def test_generous_budget_keeps_full_resolution(self):
        # 1 bpp is well above the crossover; downscaling only loses detail.
        pixels = 1000 * 1000
        target = pixels * 1.0 / 8
        self.assertEqual(AdaptiveImageCompressor.seed_resize(target, pixels), 1.0)

    def test_tight_budget_downscales(self):
        # 0.1 bpp: full resolution cannot hold detail, so shrink first.
        pixels = 4000 * 3000
        target = pixels * 0.1 / 8
        self.assertLess(AdaptiveImageCompressor.seed_resize(target, pixels), 0.5)

    def test_rule_lands_near_the_intended_bitrate(self):
        pixels = 2000 * 1500
        for bpp in (0.05, 0.1, 0.2, 0.4, 0.8):
            resize = AdaptiveImageCompressor.seed_resize(pixels * bpp / 8, pixels)
            if resize < 1.0:
                effective = bpp / (resize**2)
                self.assertAlmostEqual(effective, BPP_TARGET, delta=BPP_TARGET * 0.6)

    def test_seed_is_always_on_the_grid(self):
        for bpp in (0.01, 0.05, 0.3, 1.0, 20.0):
            pixels = 800 * 600
            self.assertIn(
                AdaptiveImageCompressor.seed_resize(pixels * bpp / 8, pixels),
                RESIZE_GRID,
            )


class OrientationTests(unittest.TestCase):
    def test_exif_rotation_is_applied(self):
        # A portrait phone photo is stored landscape with an orientation flag.
        # Ignoring it is how images come out sideways.
        array = detailed_image()[:200]  # 200 tall, 400 wide
        buffer = io.BytesIO()
        picture = Image.fromarray(array)
        exif = picture.getexif()
        exif[274] = 6  # rotate 90 CW on display
        picture.save(buffer, format="JPEG", exif=exif)
        buffer.seek(0)
        loaded = load_image(Image.open(buffer))
        self.assertEqual(loaded.shape[:2], (array.shape[1], array.shape[0]))

    def test_rejects_non_rgb_input(self):
        with self.assertRaises(ValueError):
            AdaptiveImageCompressor().compress(np.zeros((50, 50), dtype=np.uint8), 50)


class TransparencyTests(unittest.TestCase):
    @staticmethod
    def transparent_png() -> Image.Image:
        # Fully transparent pixels sitting on top of black.
        picture = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        picture.paste((200, 30, 30, 255), (16, 16, 48, 48))
        return picture

    def test_transparency_is_detected(self):
        self.assertTrue(has_transparency(self.transparent_png()))
        self.assertFalse(has_transparency(Image.fromarray(flat_image())))

    def test_transparent_areas_become_white_not_black(self):
        # .convert("RGB") would keep the black underneath the alpha. The app
        # promises white, so the loader has to actually composite.
        loaded = load_image(self.transparent_png())
        self.assertEqual(loaded.shape, (64, 64, 3))
        corner = loaded[0, 0]
        self.assertTrue((corner == 255).all(), f"corner was {corner}, expected white")
        self.assertLess(int(loaded[32, 32][1]), 100)  # the red square survived

    def test_transparent_image_compresses(self):
        result = AdaptiveImageCompressor().compress(self.transparent_png(), 50)
        self.assertLessEqual(result["actual_size_kb"], 50)


class QualitySearchTests(unittest.TestCase):
    def test_quality_search_finds_the_boundary(self):
        # The encode just above the chosen quality must not fit, otherwise the
        # interpolation stopped early and gave away quality it could have had.
        image = detailed_image()
        target = 40 * 1024
        result = AdaptiveImageCompressor(formats=["JPEG"]).compress(image, 40)
        quality = int(result["quality"])
        if quality < 95:
            _, bigger = AdaptiveImageCompressor._encode(
                image, result["resize_factor"], quality + 1, "JPEG"
            )
            self.assertGreater(
                len(bigger), target, "a higher quality also fit; search stopped short"
            )

    def test_webp_beats_jpeg_at_the_same_budget(self):
        # The reason WebP was added. If this ever fails, revisit the default.
        image = Image.fromarray(detailed_image())
        jpeg = AdaptiveImageCompressor().compress(image, 40, image_format="JPEG")
        webp = AdaptiveImageCompressor().compress(image, 40, image_format="WEBP")
        self.assertGreater(webp["ssim"], jpeg["ssim"])


class CommandLineTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.source = self.root / "sample.png"
        Image.fromarray(detailed_image()).save(self.source)

    def tearDown(self):
        self.folder.cleanup()

    def test_compresses_to_a_named_output(self):
        out = self.root / "out.jpg"
        code = cli_main([str(self.source), "--target", "30", "-o", str(out), "-q"])
        self.assertEqual(code, 0)
        self.assertTrue(out.exists())
        self.assertLessEqual(out.stat().st_size, 30 * 1024)

    def test_batch_writes_into_a_directory(self):
        second = self.root / "second.png"
        Image.fromarray(flat_image()).save(second)
        outdir = self.root / "web"
        code = cli_main(
            [str(self.source), str(second), "--target", "40", "-d", str(outdir), "-q"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(len(list(outdir.iterdir())), 2)

    def test_format_flag_picks_the_extension(self):
        outdir = self.root / "webp"
        cli_main(
            [str(self.source), "--target", "40", "-d", str(outdir), "-f", "webp", "-q"]
        )
        self.assertEqual([p.suffix for p in outdir.iterdir()], [".webp"])

    def test_existing_output_is_not_clobbered(self):
        out = self.root / "out.jpg"
        out.write_bytes(b"keep me")
        code = cli_main([str(self.source), "--target", "30", "-o", str(out), "-q"])
        self.assertEqual(code, 1)
        self.assertEqual(out.read_bytes(), b"keep me")

    def test_overwrite_flag_replaces_it(self):
        out = self.root / "out.jpg"
        out.write_bytes(b"replace me")
        code = cli_main(
            [str(self.source), "--target", "30", "-o", str(out), "--overwrite", "-q"]
        )
        self.assertEqual(code, 0)
        self.assertNotEqual(out.read_bytes(), b"replace me")

    def test_missing_file_is_an_error(self):
        with self.assertRaises(SystemExit):
            cli_main([str(self.root / "nope.png"), "--target", "30"])

    def test_out_with_many_inputs_is_rejected(self):
        with self.assertRaises(SystemExit):
            cli_main(
                [str(self.source), str(self.source), "--target", "30", "-o", "x.jpg"]
            )


if __name__ == "__main__":
    unittest.main()
