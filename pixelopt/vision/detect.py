"""Naming what is in an image with Florence-2.

Florence-2 answers one task prompt at a time, and its processor parses the
text it returns into boxes and labels. Three passes make the list: a detailed
caption, the objects it detects outright, and the phrases of that caption
grounded back onto the image. The last pass is what finds things a fixed
label set has no word for. A fourth task finds a name the user types.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

from .types import Box, Found, Scene

if TYPE_CHECKING:
    from PIL import Image

CAPTION = "<MORE_DETAILED_CAPTION>"
DETECT = "<OD>"
GROUND = "<CAPTION_TO_PHRASE_GROUNDING>"
OPEN_VOCABULARY = "<OPEN_VOCABULARY_DETECTION>"

# Thinner than this and there is nothing to outline.
MIN_SIDE = 8
# Phrase grounding likes to box the whole frame for "the image" or "a field".
# Removing everything is not a removal, so those are dropped.
MAX_FRAME_SHARE = 0.95
# Dense captions can come back as whole sentences; a name is short.
MAX_NAME_LENGTH = 40
# Two boxes with one name are a repeat once they overlap this much.
SAME_NAME_IOU = 0.6
# Two boxes with different names are one thing only when they nearly coincide.
# A part often fills most of its whole -- on the photo this was tuned on, the
# leaves cover 78% of the tree's box -- and dropping the part would drop the
# very thing someone wants to remove.
SAME_BOX_IOU = 0.9

_ARTICLE = re.compile(r"^(a|an|the|some|this|that)\s+", re.IGNORECASE)
_TAG = re.compile(r"<[^>]*>")


def clean_name(text: object) -> str:
    """A label as a short lower-case name: "The  Oak tree." -> "oak tree"."""
    name = re.sub(r"\s+", " ", _TAG.sub("", str(text))).strip().strip(".,;:").lower()
    return _ARTICLE.sub("", name).strip()


def _clamp(raw: Sequence[object], size: Tuple[int, int]) -> Optional[Box]:
    width, height = size
    try:
        x0, y0, x1, y1 = (float(value) for value in raw[:4])
    except (TypeError, ValueError):
        return None
    x0, x1 = sorted((min(max(x0, 0.0), width), min(max(x1, 0.0), width)))
    y0, y1 = sorted((min(max(y0, 0.0), height), min(max(y1, 0.0), height)))
    box = (int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1)))
    wide, tall = box[2] - box[0], box[3] - box[1]
    if wide < MIN_SIDE or tall < MIN_SIDE:
        return None
    if wide * tall > MAX_FRAME_SHARE * width * height:
        return None
    return box


def parse_detections(parsed: object, task: str, size: Tuple[int, int],
                     fallback_name: str = "") -> List[Found]:
    """Florence-2's parsed answer for one task, as named boxes.

    `size` is (width, height). Boxes are clamped to the image; anything
    degenerate, nearly frame-sized or without a usable name is dropped.
    """
    answer = parsed.get(task) if isinstance(parsed, dict) else None
    if not isinstance(answer, dict):
        return []
    boxes = list(answer.get("bboxes") or [])
    labels = list(answer.get("labels") or answer.get("bboxes_labels") or [])
    # Open-vocabulary answers can come back as outlines instead of boxes.
    for polygons, label in zip(answer.get("polygons") or [], answer.get("polygons_labels") or []):
        points = [value for polygon in polygons for value in polygon]
        xs, ys = points[0::2], points[1::2]
        if xs and ys:
            boxes.append([min(xs), min(ys), max(xs), max(ys)])
            labels.append(label)

    found: List[Found] = []
    for index, raw in enumerate(boxes):
        if not isinstance(raw, (list, tuple)) or len(raw) < 4:
            continue
        box = _clamp(raw, size)
        name = clean_name(labels[index] if index < len(labels) else "") or clean_name(fallback_name)
        if box is None or not name or len(name) > MAX_NAME_LENGTH:
            continue
        found.append(Found(name, box))
    return found


def iou(a: Box, b: Box) -> float:
    """Intersection over union of two boxes."""
    wide = min(a[2], b[2]) - max(a[0], b[0])
    tall = min(a[3], b[3]) - max(a[1], b[1])
    if wide <= 0 or tall <= 0:
        return 0.0
    overlap = wide * tall
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def _repeats(found: Found, other: Found) -> bool:
    overlap = iou(found.box, other.box)
    return overlap >= (SAME_NAME_IOU if found.name == other.name else SAME_BOX_IOU)


def merge(passes: Iterable[Iterable[Found]]) -> List[Found]:
    """One list from several passes, without repeats, largest first.

    An item repeats an earlier one when it has the same name and a similar
    box, or a nearly identical box under any name. The earlier one is kept,
    so order the passes from most to least trusted.
    """
    kept: List[Found] = []
    for found in (item for group in passes for item in group):
        if not any(_repeats(found, other) for other in kept):
            kept.append(found)
    return sorted(kept, key=lambda item: (-item.area, item.name, item.box))


def number(found: Sequence[Found]) -> List[Found]:
    """Unique display names: the second "person" becomes "person 2"."""
    seen: Dict[str, int] = {}
    named: List[Found] = []
    for item in found:
        seen[item.name] = seen.get(item.name, 0) + 1
        count = seen[item.name]
        named.append(item if count == 1 else Found(f"{item.name} {count}", item.box))
    return named


class Detector:
    """Florence-2, asked one task at a time."""

    def __init__(self, model, processor) -> None:
        self.model = model
        self.processor = processor

    def _ask(self, image: "Image.Image", task: str, text: str = "") -> dict:
        import torch

        inputs = self.processor(text=task + text, images=image, return_tensors="pt")
        inputs = inputs.to(self.model.device, self.model.dtype)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=1024, num_beams=3, do_sample=False)
        raw = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        return self.processor.post_process_generation(raw, task=task, image_size=image.size)

    def describe(self, image: "Image.Image") -> Scene:
        """A caption and every named object, unnumbered and without repeats."""
        caption = str(self._ask(image, CAPTION).get(CAPTION, "")).strip()
        detected = parse_detections(self._ask(image, DETECT), DETECT, image.size)
        grounded = (parse_detections(self._ask(image, GROUND, caption), GROUND, image.size)
                    if caption else [])
        # Caption phrases win a tie. The detector answers from a fixed set of
        # labels and guesses when nothing fits: it called a lone oak a
        # "houseplant". A phrase was written about this image.
        return Scene(caption, tuple(merge([grounded, detected])))

    def find(self, image: "Image.Image", name: str) -> List[Found]:
        """Every place the typed name appears, all carrying that name."""
        name = clean_name(name)
        if not name:
            return []
        parsed = self._ask(image, OPEN_VOCABULARY, name)
        hits = parse_detections(parsed, OPEN_VOCABULARY, image.size, fallback_name=name)
        return merge([[Found(name, hit.box) for hit in hits]])
