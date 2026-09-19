"""Tests for quality-target mode and keeping originals that already fit."""

import io
import struct
import unittest
import zlib

import numpy as np
from PIL import Image

from pixelopt.enhance import Enhancements
from pixelopt.image_features import compute_quality_metrics, load_image
from pixelopt.passthrough import strip_jpeg, strip_png, strip_webp, try_passthrough
from pixelopt.pipeline import process, process_to_quality
from pixelopt.quality_target import compress_to_quality


def photo(width=480, height=360, seed=2, noise=10):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = 120 + 60 * np.sin(x / 31.0) * np.cos(y / 23.0) + rng.normal(0, noise, (height, width))
    channel = np.clip(base, 0, 255)
    return np.dstack([channel, np.roll(channel, 7, 0), np.roll(channel, 7, 1)]).astype(np.uint8)


def jpeg(array, quality=85, exif=None):
    buffer = io.BytesIO()
    options = {"quality": quality}
    if exif is not None:
        options["exif"] = exif
    Image.fromarray(array).save(buffer, format="JPEG", **options)
    return buffer.getvalue()


def gps_exif(orientation=None):
    exif = Image.new("RGB", (1, 1)).getexif()
    exif[0x010F] = "PhoneMaker"
    if orientation is not None:
        exif[0x0112] = orientation
    gps = exif.get_ifd(0x8825)
    gps[1], gps[2], gps[3], gps[4] = "N", (28.0, 36.0, 0.0), "E", (77.0, 12.0, 0.0)
    return exif


def png_chunk(kind, data):
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))


def png_with_text(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    raw = buffer.getvalue()
    # Insert a tEXt chunk right after IHDR (8-byte signature + 25-byte IHDR).
    return raw[:33] + png_chunk(b"tEXt", b"Comment\x00home address") + raw[33:]


# ------------------------------------------------------------ passthrough


class PassthroughTests(unittest.TestCase):
    def test_fitting_jpeg_is_kept_and_gps_removed(self):
        array = photo()
        raw = jpeg(array, exif=gps_exif())
        self.assertTrue(Image.open(io.BytesIO(raw)).getexif().get_ifd(0x8825))
        result = process(raw, 200)
        self.assertTrue(result["passthrough"])
        kept = Image.open(io.BytesIO(result["raw_bytes"]))
        self.assertFalse(kept.getexif().get_ifd(0x8825), "GPS survived")
        self.assertEqual(len(kept.getexif()), 0)
        self.assertGreater(result["metadata_removed_bytes"], 0)
        self.assertTrue(np.array_equal(load_image(raw), load_image(result["raw_bytes"])))
        self.assertEqual(result["fidelity_ssim"], 1.0)

    def test_rotated_jpeg_is_reencoded_upright(self):
        # Stripping the orientation flag would turn it sideways, so the pixel
        # check must refuse it and the normal encode must run instead.
        array = photo(480, 360)
        raw = jpeg(array, exif=gps_exif(orientation=6))
        result = process(raw, 200)
        self.assertFalse(result["passthrough"])
        self.assertIn("orientation", result["passthrough_reason"])
        self.assertEqual((result["width"], result["height"]), (360, 480))

    def test_over_budget_is_reencoded(self):
        raw = jpeg(photo(noise=30), quality=95)
        result = process(raw, len(raw) / 1024 / 3)
        self.assertFalse(result["passthrough"])
        self.assertLessEqual(result["actual_size_kb"], len(raw) / 1024 / 3)

    def test_requested_format_must_match(self):
        result = process(jpeg(photo()), 500, image_format="WEBP")
        self.assertFalse(result["passthrough"])
        self.assertEqual(result["format"], "WEBP")

    def test_enhancement_disables_passthrough(self):
        result = process(jpeg(photo()), 500, enhancements=Enhancements(denoise=6))
        self.assertFalse(result["passthrough"])

    def test_arrays_never_pass_through(self):
        self.assertFalse(process(photo(), 500)["passthrough"])

    def test_png_text_chunks_are_removed(self):
        array = photo(200, 150)
        raw = png_with_text(array)
        self.assertIn(b"home address", raw)
        result = process(raw, 5000)
        self.assertTrue(result["passthrough"])
        self.assertNotIn(b"home address", result["raw_bytes"])
        self.assertTrue(np.array_equal(load_image(result["raw_bytes"]), array))

    def test_transparent_png_is_not_passed_through(self):
        picture = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
        picture.paste((200, 30, 30, 255), (20, 20, 60, 60))
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        result = process(buffer.getvalue(), 500)
        self.assertFalse(result["passthrough"])
        self.assertIn("transparency", result["passthrough_reason"])

    def test_webp_exif_is_removed(self):
        buffer = io.BytesIO()
        Image.fromarray(photo(200, 150)).save(buffer, format="WEBP", quality=80,
                                               exif=gps_exif().tobytes())
        raw = buffer.getvalue()
        self.assertIn(b"EXIF", raw)
        stripped = strip_webp(raw)
        self.assertNotIn(b"EXIF", stripped)
        self.assertTrue(np.array_equal(load_image(stripped), load_image(raw)))
        self.assertEqual(int.from_bytes(stripped[4:8], "little"), len(stripped) - 8)

    def test_data_after_jpeg_end_is_dropped(self):
        raw = jpeg(photo(120, 90)) + b"trailing secondary image bytes"
        stripped = strip_jpeg(raw)
        self.assertTrue(stripped.endswith(b"\xff\xd9"))
        self.assertTrue(np.array_equal(load_image(stripped), load_image(raw)))

    def test_corrupt_input_is_refused_not_raised(self):
        array = photo(64, 48)
        result, reason = try_passthrough(b"\xff\xd8\xff\xe1\x00", array, 1e9)
        self.assertIsNone(result)
        self.assertTrue(reason)
        with self.assertRaises(ValueError):
            strip_png(b"not a png")

    def test_cli_row_reports_the_kept_original(self):
        import tempfile
        from pathlib import Path

        from pixelopt.cli import main as cli_main

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "phone.jpg"
            source.write_bytes(jpeg(photo(), exif=gps_exif()))
            out = Path(folder) / "out.jpg"
            self.assertEqual(cli_main([str(source), "-t", "300", "-o", str(out), "-q"]), 0)
            self.assertFalse(Image.open(out).getexif().get_ifd(0x8825))


# ---------------------------------------------------------- quality target


class QualityTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image = photo(640, 480, noise=12)

    def full_ssim(self, result):
        return compute_quality_metrics(self.image, np.asarray(result["image"]))["ssim"]

    def test_meets_the_target_at_full_resolution(self):
        for target in (0.9, 0.95, 0.98):
            result = compress_to_quality(self.image, target)
            self.assertTrue(result["met"])
            self.assertGreaterEqual(self.full_ssim(result), target, f"target {target}")
            self.assertAlmostEqual(result["ssim"], self.full_ssim(result), places=6)

    def test_higher_target_never_gives_a_smaller_file(self):
        sizes = [compress_to_quality(self.image, t)["actual_size_kb"] for t in (0.9, 0.95, 0.98)]
        self.assertEqual(sizes, sorted(sizes))

    def test_large_image_is_scored_on_tiles_and_still_meets_the_target(self):
        big = photo(1400, 1000, noise=10)
        result = compress_to_quality(big, 0.96)
        full = compute_quality_metrics(big, np.asarray(result["image"]))["ssim"]
        self.assertGreaterEqual(full, 0.96)

    def test_unreachable_target_with_restricted_encoder_is_flagged(self):
        noisy = photo(320, 240, noise=45)
        result = compress_to_quality(noisy, 0.9999, image_format="JPEG")
        self.assertFalse(result["met"])
        self.assertEqual(result["quality"], 95)

    def test_lossless_is_available_under_auto(self):
        noisy = photo(200, 150, noise=45)
        result = compress_to_quality(noisy, 0.9999)
        self.assertTrue(result["met"])
        self.assertGreaterEqual(result["ssim"], 0.9999)

    def test_invalid_target_is_rejected(self):
        for bad in (0.2, 1.5):
            with self.assertRaises(ValueError):
                compress_to_quality(self.image, bad)

    def test_near_exact_target_keeps_the_original(self):
        # Reproducing a JPEG to 0.999 of itself costs more bytes than the
        # file already has, so the untouched original is the smallest answer.
        raw = jpeg(self.image, quality=40)
        result = process_to_quality(raw, 0.999)
        self.assertTrue(result["passthrough"])
        self.assertLessEqual(len(result["raw_bytes"]), len(raw))

    def test_looser_target_may_beat_the_original(self):
        # The original is exact, but exact is not required: at 0.97 a smaller
        # re-encode genuinely exists, and the search must find it rather than
        # settle for the original.
        raw = jpeg(self.image, quality=40)
        result = process_to_quality(raw, 0.97)
        self.assertFalse(result["passthrough"])
        self.assertLess(len(result["raw_bytes"]), len(raw))
        self.assertGreaterEqual(result["fidelity_ssim"], 0.97)

    def test_original_wins_when_no_reencode_reaches_the_target(self):
        # Forced JPEG at 0.9999 on a noisy quality-100 original: the search
        # tops out at quality 95 and cannot get there, but the untouched
        # original is exact. (A quality-95 source would not do -- re-encoding
        # it at 95 reproduces it almost perfectly.)
        noisy = photo(320, 240, noise=45)
        raw = jpeg(noisy, quality=100)
        result = process_to_quality(raw, 0.9999, image_format="JPEG")
        self.assertTrue(result["passthrough"])
        self.assertTrue(result["met"])
        self.assertEqual(result["fidelity_ssim"], 1.0)

    def test_pipeline_reports_fidelity_and_target(self):
        result = process_to_quality(self.image, 0.95)
        self.assertGreaterEqual(result["fidelity_ssim"], 0.95)
        self.assertEqual(result["target_ssim"], 0.95)
        self.assertIn("passthrough_reason", result)


if __name__ == "__main__":
    unittest.main()
