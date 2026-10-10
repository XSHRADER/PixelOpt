"""The Streamlit pages render.

AppTest runs each page headlessly. It cannot upload a file, so these cover
the empty state of every page -- imports, widget arguments, layout calls --
and the loaded workbench is checked in a browser instead.

    python tests/test_app.py -v
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app_shared import png_bytes  # noqa: E402

APP = str(ROOT / "app.py")
PAGES = {
    "optimize": None,
    "form photo": "app_pages/form_photo.py",
    "batch": "app_pages/batch.py",
    "objects": "app_pages/objects.py",
}


def run_page(page=None) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=60).run()
    if page is not None:
        app.switch_page(page).run()
    return app


class EmptyStates(unittest.TestCase):
    def test_every_page_renders_without_an_exception(self):
        for name, page in PAGES.items():
            with self.subTest(page=name):
                app = run_page(page)
                self.assertFalse(app.exception, [e.value for e in app.exception])
                self.assertEqual(len(app.get("file_uploader")), 1)

    def test_header_has_no_ambient_motion(self):
        for name, page in PAGES.items():
            with self.subTest(page=name):
                bodies = " ".join(e.proto.body for e in run_page(page).get("html"))
                self.assertIn("po-title", bodies)
                for removed in ("po-marquee", "po-grid", "po-glow", "po-rise"):
                    self.assertNotIn(removed, bodies)

    def test_controls_wait_for_an_upload(self):
        # The control panel is part of the workbench, which needs a file.
        for name, page in PAGES.items():
            with self.subTest(page=name):
                app = run_page(page)
                self.assertEqual(len(app.slider), 0)
                self.assertEqual(len(app.number_input), 0)
                self.assertEqual(len(app.selectbox), 0)


class ReferenceDownload(unittest.TestCase):
    def test_png_bytes_is_lossless(self):
        array = np.random.default_rng(3).integers(0, 256, (37, 53, 3), dtype=np.uint8)
        data = png_bytes(Image.fromarray(array))
        self.assertTrue(data.startswith(b"\x89PNG"))
        decoded = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        np.testing.assert_array_equal(decoded, array)


class Suggestions(unittest.TestCase):
    def test_a_failed_suggestion_is_asked_for_again(self):
        # The local model may simply not be up yet. Remembering its silence
        # would make the Suggest button a dead end for that object.
        import app_shared

        buffer = io.BytesIO()
        Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8)).save(buffer, format="PNG")
        raw = buffer.getvalue()
        replies = [None, "bare branches"]
        original = app_shared.reason.suggest_fill
        app_shared.reason.suggest_fill = lambda *args, **kwargs: replies.pop(0)
        try:
            self.assertEqual(app_shared.vision_suggestion(raw, "leaves", (1, 2, 30, 40), "model"), "")
            self.assertEqual(app_shared.vision_suggestion(raw, "leaves", (1, 2, 30, 40), "model"),
                             "bare branches")
            # A real answer is remembered: nothing is left to ask.
            self.assertEqual(app_shared.vision_suggestion(raw, "leaves", (1, 2, 30, 40), "model"),
                             "bare branches")
        finally:
            app_shared.reason.suggest_fill = original


if __name__ == "__main__":
    unittest.main()
