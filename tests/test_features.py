"""Tests for form photo mode, the damage heatmap and batch processing."""

import csv
import io
import unittest
import zipfile

import numpy as np
from PIL import Image

from pixelopt.analysis import damage_map, damage_summary, heatmap_overlay, heatmap_rgba
from pixelopt.batch import build_zip, report_csv, run_batch, summarize
from pixelopt.forms import (
    FORM_PRESETS,
    FormSpec,
    byte_bounds,
    encode_for_form,
    pad_jpeg,
)
from pixelopt.pipeline import process


def photo_like(width=640, height=480, seed=5):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = 120 + 60 * np.sin(x / 37.0) * np.cos(y / 29.0) + rng.normal(0, 14, (height, width))
    channel = np.clip(base, 0, 255)
    return np.dstack([channel, np.roll(channel, 9, 0), np.roll(channel, 9, 1)]).astype(np.uint8)


def flat(width=200, height=230, value=180):
    return np.full((height, width, 3), value, dtype=np.uint8)


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def open_bytes(data):
    picture = Image.open(io.BytesIO(data))
    picture.load()
    return picture


# ------------------------------------------------------------------- forms


class FormDimensionTests(unittest.TestCase):
    def test_crop_hits_exact_dimensions(self):
        result = encode_for_form(photo_like(), FORM_PRESETS["photo"])
        self.assertEqual(open_bytes(result["raw_bytes"]).size, (200, 230))

    def test_pad_hits_exact_dimensions_on_white(self):
        # 400x400 into 140x60 scales to 60x60, leaving white bands either side.
        result = encode_for_form(photo_like(400, 400), FORM_PRESETS["signature"])
        picture = open_bytes(result["raw_bytes"]).convert("RGB")
        self.assertEqual(picture.size, (140, 60))
        self.assertTrue(all(c > 235 for c in picture.getpixel((6, 30))))

    def test_crop_follows_the_detail(self):
        # Blank on the left, striped on the right: a centred crop would keep
        # mostly nothing, so the crop must slide toward the stripes.
        image = np.full((300, 900, 3), 250, dtype=np.uint8)
        image[:, 700:, :] = np.where((np.arange(200) // 6) % 2, 20, 230)[None, :, None]
        spec = FormSpec(300, 300, 5, 40, "crop")
        result = encode_for_form(image, spec)
        self.assertEqual(result["anchor"], "detail")
        self.assertGreaterEqual(result["crop_box"][0], 500)

    def test_small_source_is_flagged_as_upscaled(self):
        result = encode_for_form(photo_like(120, 120), FORM_PRESETS["square"])
        self.assertTrue(result["upscaled"])
        self.assertTrue(any("enlarged" in m for m in result["messages"]))


class FormSizeTests(unittest.TestCase):
    def test_output_passes_under_both_kilobyte_readings(self):
        spec = FORM_PRESETS["photo"]
        result = encode_for_form(photo_like(), spec)
        size = result["size_bytes"]
        # 20 KB minimum even if KB means 1024 bytes, 50 KB maximum even if it
        # means 1000 -- a portal using either reading accepts the file.
        self.assertGreaterEqual(size, 20 * 1024)
        self.assertLessEqual(size, 50 * 1000)
        self.assertTrue(result["meets_min"] and result["meets_max"])
        self.assertTrue(result["strict_units"])

    def test_flat_image_is_padded_without_touching_pixels(self):
        result = encode_for_form(flat(), FORM_PRESETS["photo"])
        self.assertGreater(result["padded_bytes"], 0)
        self.assertTrue(result["meets_min"] and result["meets_max"])
        padded = np.asarray(open_bytes(result["raw_bytes"]))
        natural = np.asarray(open_bytes(result["unpadded_bytes"]))
        self.assertTrue(np.array_equal(padded, natural))
        self.assertTrue(any("comment data" in m for m in result["messages"]))

    def test_impossible_budget_is_reported_not_hidden(self):
        rng = np.random.default_rng(1)
        noise = rng.integers(0, 256, (1500, 1500, 3), dtype=np.uint8)
        result = encode_for_form(noise, FormSpec(1500, 1500, 0, 5, "crop"))
        self.assertFalse(result["meets_max"])
        self.assertGreater(result["size_bytes"], 5000)
        self.assertTrue(result["messages"])

    def test_narrow_range_falls_back_to_binary_kilobytes(self):
        low, high, strict = byte_bounds(FormSpec(200, 200, 49, 50))
        self.assertFalse(strict)
        self.assertEqual((low, high), (49 * 1024, 50 * 1024))

    def test_reported_kilobytes_match_the_bytes(self):
        result = encode_for_form(photo_like(), FORM_PRESETS["photo"])
        self.assertAlmostEqual(result["size_kb"], len(result["raw_bytes"]) / 1024)


class FormEncodingTests(unittest.TestCase):
    def test_grayscale_produces_a_single_channel_jpeg(self):
        spec = FormSpec(200, 230, 5, 50, "crop", grayscale=True)
        self.assertEqual(open_bytes(encode_for_form(photo_like(), spec)["raw_bytes"]).mode, "L")

    def test_dpi_is_written_into_the_file(self):
        result = encode_for_form(photo_like(), FORM_PRESETS["passport"])
        dpi = open_bytes(result["raw_bytes"]).info.get("dpi")
        self.assertIsNotNone(dpi)
        self.assertAlmostEqual(float(dpi[0]), 300, delta=1)

    def test_output_is_baseline_not_progressive(self):
        info = open_bytes(encode_for_form(photo_like(), FORM_PRESETS["photo"])["raw_bytes"]).info
        self.assertFalse(info.get("progressive") or info.get("progression"))

    def test_invalid_specs_are_rejected(self):
        for spec in (FormSpec(200, 200, 60, 50), FormSpec(200, 200, 5, 50, "stretch"),
                     FormSpec(4, 200, 5, 50), FormSpec(200, 200, 5, 0)):
            with self.assertRaises(ValueError):
                spec.validate()

    def test_every_preset_is_valid(self):
        for spec in FORM_PRESETS.values():
            spec.validate()


class PaddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        buffer = io.BytesIO()
        Image.fromarray(flat(64, 64)).save(buffer, format="JPEG", quality=80)
        cls.jpeg = buffer.getvalue()
        cls.pixels = np.asarray(open_bytes(cls.jpeg))

    def assert_same_pixels(self, data):
        self.assertTrue(np.array_equal(np.asarray(open_bytes(data)), self.pixels))

    def test_exact_lengths_including_multi_segment(self):
        # 70000 spans two segments; 65537 + 2 exercises the borrow branch.
        for extra in (4, 5, 1000, 65537, 65537 + 2, 70000):
            padded = pad_jpeg(self.jpeg, len(self.jpeg) + extra)
            self.assertEqual(len(padded), len(self.jpeg) + extra, f"extra={extra}")
            self.assert_same_pixels(padded)

    def test_tiny_shortfall_rounds_up_to_the_smallest_segment(self):
        padded = pad_jpeg(self.jpeg, len(self.jpeg) + 2)
        self.assertEqual(len(padded), len(self.jpeg) + 4)
        self.assert_same_pixels(padded)

    def test_rejects_non_jpeg(self):
        with self.assertRaises(ValueError):
            pad_jpeg(png_bytes(flat(16, 16)), 10_000)


# ----------------------------------------------------------------- heatmap


class HeatmapTests(unittest.TestCase):
    def test_identical_images_show_no_damage(self):
        image = photo_like(300, 300)
        loss = damage_map(image, image)
        self.assertLess(float(loss.max()), 1e-3)
        self.assertEqual(damage_summary(loss)["damaged_share"], 0.0)

    def test_worst_region_is_where_the_damage_is(self):
        reference = photo_like(400, 400)
        output = reference.copy()
        rng = np.random.default_rng(3)
        output[:, :200] = np.clip(
            output[:, :200].astype(int) + rng.normal(0, 60, output[:, :200].shape), 0, 255
        ).astype(np.uint8)
        summary = damage_summary(damage_map(reference, output))
        self.assertLessEqual(summary["worst_box"][2], 0.55)
        self.assertGreater(summary["damaged_share"], 0.2)

    def test_downscaled_output_is_compared_at_reference_size(self):
        reference = photo_like(400, 400)
        smaller = np.asarray(Image.fromarray(reference).resize((200, 200)))
        self.assertEqual(damage_map(reference, smaller).shape, (400, 400))

    def test_large_images_are_capped(self):
        image = photo_like(400, 400)
        self.assertLess(damage_map(image, image, max_pixels=10_000).size, 400 * 400)

    def test_ordinary_photos_are_measured_without_resampling(self):
        # Resampling before SSIM averages the damage away: a scan read 86%
        # damaged at full resolution and 0.4% after downscaling. A 3 MP photo
        # must be measured on its own pixel grid.
        reference = photo_like(2000, 1500)
        buffer = io.BytesIO()
        Image.fromarray(reference).save(buffer, format="JPEG", quality=12)
        output = np.asarray(Image.open(io.BytesIO(buffer.getvalue())).convert("RGB"))
        loss = damage_map(reference, output)
        self.assertEqual(loss.shape, (1500, 2000))
        self.assertGreater(damage_summary(loss)["damaged_share"], 0.05)

    def test_undamaged_pixels_are_transparent(self):
        loss = np.zeros((20, 20), dtype=np.float32)
        loss[10:, :] = 0.3
        layer = heatmap_rgba(loss)
        self.assertEqual(layer.shape, (20, 20, 4))
        self.assertEqual(int(layer[0, 0, 3]), 0)
        self.assertGreater(int(layer[15, 5, 3]), 0)

    def test_overlay_matches_the_reference(self):
        reference = photo_like(300, 200)
        overlay = heatmap_overlay(reference, damage_map(reference, reference))
        self.assertEqual(overlay.shape, reference.shape)
        self.assertEqual(overlay.dtype, np.uint8)


# ------------------------------------------------------------------- batch


def _job(name, raw):
    return summarize(process(io.BytesIO(raw), 30))


class BatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jpeg = io.BytesIO()
        Image.fromarray(photo_like(320, 240, seed=8)).save(jpeg, format="JPEG", quality=95)
        cls.files = [
            ("holiday.png", png_bytes(photo_like(320, 240))),
            ("holiday.jpg", jpeg.getvalue()),   # same stem as the PNG
            ("broken.png", b"this is not an image"),
            ("logo.png", png_bytes(flat(120, 120))),
        ]
        cls.progress = []
        cls.items = run_batch(cls.files, _job,
                              on_progress=lambda done, total, item: cls.progress.append(done))

    def test_one_bad_file_does_not_abort_the_batch(self):
        by_name = {item.name: item for item in self.items}
        self.assertFalse(by_name["broken.png"].ok)
        self.assertTrue(by_name["broken.png"].error)
        self.assertEqual(sum(item.ok for item in self.items), 3)

    def test_outputs_get_unique_names(self):
        names = [item.output_name for item in self.items if item.ok]
        self.assertEqual(len(names), len({n.lower() for n in names}))

    def test_zip_holds_every_output_and_the_report(self):
        archive = zipfile.ZipFile(io.BytesIO(build_zip(self.items)))
        names = set(archive.namelist())
        self.assertIn("report.csv", names)
        for item in self.items:
            if item.ok:
                self.assertIn(item.output_name, names)
                # the exact measured bytes, not a re-encode
                self.assertEqual(archive.read(item.output_name), item.data)
                self.assertEqual(archive.getinfo(item.output_name).compress_type,
                                 zipfile.ZIP_STORED)

    def test_report_has_a_row_per_file(self):
        rows = list(csv.reader(io.StringIO(report_csv(self.items))))
        self.assertEqual(len(rows), 1 + len(self.files))
        statuses = {row[0]: row[1] for row in rows[1:]}
        self.assertEqual(statuses["broken.png"], "error")

    def test_progress_is_reported_for_every_file(self):
        self.assertEqual(self.progress, [1, 2, 3, 4])

    def test_every_output_respects_the_budget(self):
        for item in self.items:
            if item.ok:
                self.assertLessEqual(item.output_bytes, 30 * 1024)


if __name__ == "__main__":
    unittest.main()
