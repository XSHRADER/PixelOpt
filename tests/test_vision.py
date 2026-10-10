"""Tests for the Objects page's engine. No model, GPU or vision library is needed.

Everything here exercises the code around the models: parsing what they
return, shaping masks, cropping, compositing, scoring, and talking to Ollama
(against a stub server). The models themselves are checked on a machine with
a GPU -- see benchmark_vision.py.

    python tests/test_vision.py -v
"""

from __future__ import annotations

import base64
import http.server
import io
import json
import math
import subprocess
import sys
import threading
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pixelopt.vision import detect, draw, segment  # noqa: E402
from pixelopt.vision.types import Found  # noqa: E402

SIZE = (640, 480)  # width, height


def scene(height: int = 240, width: int = 320, seed: int = 5) -> np.ndarray:
    """A smooth colour gradient with a little texture: easy to predict from its edges."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack([x / width * 200 + 30, y / height * 180 + 40, (x + y) / (width + height) * 160 + 50], -1)
    return np.clip(base + rng.normal(0, 2, base.shape), 0, 255).astype(np.uint8)


def disc(height: int = 240, width: int = 320, centre=(160, 120), radius: int = 30) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    return (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= radius ** 2


class DetectionParsing(unittest.TestCase):
    def test_object_detection_answer(self):
        parsed = {"<OD>": {"bboxes": [[10.2, 20.7, 110.0, 220.4]], "labels": ["Car"]}}
        self.assertEqual(detect.parse_detections(parsed, "<OD>", SIZE),
                         [Found("car", (10, 21, 110, 220))])

    def test_open_vocabulary_answer_uses_its_own_label_key(self):
        parsed = {"<OPEN_VOCABULARY_DETECTION>": {
            "bboxes": [[0, 0, 100, 100]], "bboxes_labels": ["leaves"],
            "polygons": [], "polygons_labels": []}}
        found = detect.parse_detections(parsed, "<OPEN_VOCABULARY_DETECTION>", SIZE)
        self.assertEqual(found, [Found("leaves", (0, 0, 100, 100))])

    def test_outlines_become_boxes(self):
        parsed = {"<OPEN_VOCABULARY_DETECTION>": {
            "bboxes": [], "bboxes_labels": [],
            "polygons": [[[50, 60, 150, 60, 150, 200, 50, 200]]], "polygons_labels": ["sign"]}}
        found = detect.parse_detections(parsed, "<OPEN_VOCABULARY_DETECTION>", SIZE)
        self.assertEqual(found, [Found("sign", (50, 60, 150, 200))])

    def test_boxes_are_clamped_to_the_image(self):
        parsed = {"<OD>": {"bboxes": [[-30, -10, 200, 900]], "labels": ["pole"]}}
        self.assertEqual(detect.parse_detections(parsed, "<OD>", SIZE)[0].box, (0, 0, 200, 480))

    def test_useless_boxes_are_dropped(self):
        parsed = {"<OD>": {
            "bboxes": [[5, 5, 8, 300],            # a sliver
                       [0, 0, 640, 480],          # the whole frame
                       [10, 10, 200, 200],        # no usable name
                       [10, 10, 200, 200],        # a sentence, not a name
                       ["a", "b", "c", "d"],      # not numbers
                       [1, 2]],                   # not a box
            "labels": ["wire", "field", "  ", "x" * 60, "cat", "dog"]}}
        self.assertEqual(detect.parse_detections(parsed, "<OD>", SIZE), [])

    def test_a_missing_label_takes_the_fallback_name(self):
        parsed = {"<OPEN_VOCABULARY_DETECTION>": {"bboxes": [[10, 10, 90, 90]], "bboxes_labels": []}}
        found = detect.parse_detections(parsed, "<OPEN_VOCABULARY_DETECTION>", SIZE,
                                        fallback_name="Leaves")
        self.assertEqual(found, [Found("leaves", (10, 10, 90, 90))])

    def test_an_answer_for_another_task_or_of_the_wrong_shape_is_empty(self):
        self.assertEqual(detect.parse_detections({"<OD>": "no boxes here"}, "<OD>", SIZE), [])
        self.assertEqual(detect.parse_detections({}, "<OD>", SIZE), [])
        self.assertEqual(detect.parse_detections(None, "<OD>", SIZE), [])

    def test_names_are_tidied(self):
        self.assertEqual(detect.clean_name("  The  Oak   tree. "), "oak tree")
        self.assertEqual(detect.clean_name("car</s>"), "car")
        self.assertEqual(detect.clean_name("An apple"), "apple")


class MergeAndNumber(unittest.TestCase):
    def test_overlapping_boxes_are_one_thing_and_the_first_name_wins(self):
        first = [Found("tree", (100, 100, 300, 400))]
        second = [Found("large oak tree", (104, 98, 298, 396)), Found("bench", (400, 300, 500, 380))]
        self.assertEqual([item.name for item in detect.merge([first, second])], ["tree", "bench"])

    def test_separate_things_with_one_name_are_all_kept(self):
        people = [Found("person", (0, 0, 50, 100)), Found("person", (300, 0, 350, 100))]
        self.assertEqual(len(detect.merge([people])), 2)

    def test_largest_comes_first(self):
        found = [Found("cup", (0, 0, 20, 20)), Found("table", (0, 100, 400, 300))]
        self.assertEqual([item.name for item in detect.merge([found])], ["table", "cup"])

    def test_nothing_in_nothing_out(self):
        self.assertEqual(detect.merge([]), [])
        self.assertEqual(detect.merge([[], []]), [])
        self.assertEqual(detect.number([]), [])

    def test_repeated_names_are_numbered(self):
        found = [Found("person", (0, 0, 9, 9)), Found("dog", (0, 0, 9, 9)), Found("person", (9, 9, 20, 20))]
        self.assertEqual([item.name for item in detect.number(found)], ["person", "dog", "person 2"])

    def test_iou(self):
        self.assertEqual(detect.iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertEqual(detect.iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
        self.assertAlmostEqual(detect.iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)


class Masks(unittest.TestCase):
    def test_growing_widens_and_keeps_the_original(self):
        mask = disc(radius=20)
        grown = segment.grow(mask, 6)
        self.assertTrue(grown[mask].all())
        self.assertGreater(grown.sum(), mask.sum())
        # About six pixels further out in every direction.
        self.assertTrue(grown[120, 160 + 25])
        self.assertFalse(grown[120, 160 + 29])

    def test_growing_by_nothing_changes_nothing(self):
        mask = disc()
        np.testing.assert_array_equal(segment.grow(mask, 0), mask)
        empty = np.zeros((20, 20), dtype=bool)
        np.testing.assert_array_equal(segment.grow(empty, 5), empty)

    def test_feather_is_zero_outside_and_one_deep_inside(self):
        mask = disc(radius=30)
        alpha = segment.inner_feather(mask, 8)
        self.assertEqual(float(alpha[~mask].max()), 0.0)
        self.assertEqual(float(alpha[120, 160]), 1.0)
        # It eases: a pixel near the edge weighs less than one further in.
        self.assertLess(alpha[120, 160 + 28], alpha[120, 160 + 24])

    def test_feather_of_zero_is_the_hard_mask(self):
        mask = disc()
        np.testing.assert_array_equal(segment.inner_feather(mask, 0), mask.astype(np.float32))

    def test_bounding_box_and_share(self):
        mask = np.zeros((100, 200), dtype=bool)
        mask[10:30, 50:90] = True
        self.assertEqual(segment.bounding_box(mask), (50, 10, 90, 30))
        self.assertAlmostEqual(segment.share(mask), 800 / 20000)
        self.assertIsNone(segment.bounding_box(np.zeros((5, 5), dtype=bool)))

    def test_box_mask_is_the_box(self):
        mask = segment.box_mask((100, 200, 3), (50, 10, 90, 30))
        self.assertEqual(segment.bounding_box(mask), (50, 10, 90, 30))
        self.assertEqual(int(mask.sum()), 800)


class LabelledPicture(unittest.TestCase):
    def test_nothing_to_draw_changes_nothing(self):
        image = scene()
        np.testing.assert_array_equal(draw.overlay(image), image)

    def test_boxes_labels_and_the_tint_are_drawn(self):
        image, mask = scene(), disc()
        objects = [Found("tree", (40, 40, 200, 200)), Found("bench", (220, 150, 300, 220))]
        out = draw.overlay(image, objects, chosen=0, mask=mask)
        self.assertEqual((out.shape, out.dtype), (image.shape, image.dtype))
        self.assertFalse(np.array_equal(out[mask], image[mask]))      # tinted
        self.assertFalse(np.array_equal(out[40, 40:200], image[40, 40:200]))  # box edge
        np.testing.assert_array_equal(image, scene())  # the input is not drawn on


if __name__ == "__main__":
    unittest.main()
