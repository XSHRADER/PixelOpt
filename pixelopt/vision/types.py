"""The two small records the vision modules pass around."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# x0, y0, x1, y1 in pixels; x1 and y1 are exclusive.
Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Found:
    """One named thing and where it is."""

    name: str
    box: Box

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.box
        return max(0, x1 - x0) * max(0, y1 - y0)


@dataclass(frozen=True)
class Scene:
    """What the detector made of an image."""

    caption: str
    objects: Tuple[Found, ...]
