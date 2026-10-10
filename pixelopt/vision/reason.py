"""Asking a local vision-language model what is probably behind an object.

The fill model paints whatever a description says; something has to write the
description. A local model served by Ollama looks at the photo, with the
object outlined, and answers in a phrase -- "bare branches against a blue
sky". The user can edit it, and types it themselves when Ollama is not
running.

Only the standard library is used, so this module costs the project no
dependency, and every failure -- Ollama not running, a timeout, a reply that
is not an answer -- comes back as None instead of an exception.
"""

from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
from typing import Iterable, List, Optional, Sequence

from PIL import Image, ImageDraw

OLLAMA_URL = "http://127.0.0.1:11434"

# Substrings of model names that can read images, best first.
VISION_HINTS = ("qwen2.5vl", "qwen3-vl", "vl", "llava", "vision", "minicpm-v", "moondream")

# The model only needs the gist of the scene; a small JPEG keeps the request quick.
MAX_SIDE = 768

_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_COUNTER = re.compile(r"\s+\d+$")


def models(url: str = OLLAMA_URL, timeout: float = 0.5) -> List[str]:
    """The models Ollama has, or [] when it is not running."""
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=timeout) as response:
            listing = json.load(response)
        return [str(entry["name"]) for entry in listing.get("models", []) if entry.get("name")]
    except Exception:
        return []


def pick_vision_model(names: Iterable[str]) -> Optional[str]:
    """The best image-reading model among `names`, or None."""
    available = list(names)
    for hint in VISION_HINTS:
        for name in available:
            if hint in name.lower():
                return name
    return None


def clean_phrase(text: object, limit: int = 120) -> str:
    """One tidy line from a model's reply: no reasoning block, quotes or full stop."""
    reply = _THINKING.sub("", str(text))
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return ""
    phrase = re.sub(r"\s+", " ", lines[0]).strip(" \"'`*").rstrip(".").strip()
    return phrase[:limit].rstrip()


def _encode(image: Image.Image, box: Optional[Sequence[int]]) -> str:
    picture = image.convert("RGB")
    if box is not None:
        picture = picture.copy()
        width = max(3, max(picture.size) // 200)
        ImageDraw.Draw(picture).rectangle([int(v) for v in box], outline=(255, 0, 0), width=width)
    if max(picture.size) > MAX_SIDE:
        picture.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def suggest_fill(image: Image.Image, name: str, model: str, box: Optional[Sequence[int]] = None,
                 url: str = OLLAMA_URL, timeout: float = 120.0) -> Optional[str]:
    """A short phrase for what would be visible if `name` were erased, or None.

    `keep_alive: 0` asks Ollama to unload the model as soon as it has
    answered: it holds about 6 GB of the GPU, which the fill model needs next.
    """
    plain = _COUNTER.sub("", str(name)).strip() or "object"
    area = "the area outlined in red" if box is not None else f"the area the {plain} covers"
    # Worded as a brief to a painter because that is what the answer becomes.
    # Asked plainly what "would be visible", the model says "sky" for a tree's
    # leaves; asked what they are attached to or hiding, it says "trunk and
    # branches". Negations are ruled out because an image model paints the
    # noun it is given: "a tree without leaves" gets leaves.
    prompt = (
        f"A painter must repaint {area} as if the {plain} had never been there. "
        f"Think about what the {plain} is attached to or hiding. In at most twelve words, "
        "name what the painter should paint there. Name only things that will be visible: "
        f'do not mention the {plain}, and do not use the words "no" or "without". '
        "Reply with the description only."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_encode(image, box)],
        "stream": False,
        "keep_alive": 0,
        "options": {"temperature": 0.2, "num_predict": 60},
    }
    request = urllib.request.Request(
        url + "/api/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            answer = json.load(response)
        phrase = clean_phrase(answer.get("response", ""))
    except Exception:
        return None
    return phrase or None
