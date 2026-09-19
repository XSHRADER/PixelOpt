"""Run one job over many images and package the results.

Kept free of any UI so the app and the tests share it. Two rules shape it:
a bad file must never abort the batch, and the ZIP must hand back exactly the
bytes that were measured.
"""

from __future__ import annotations

import csv
import io
import math
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Callable, Dict, List, Optional, Sequence, Tuple


@dataclass
class BatchItem:
    name: str
    ok: bool
    error: str = ""
    input_bytes: int = 0
    output_bytes: int = 0
    format: str = ""
    width: int = 0
    height: int = 0
    fidelity: float = float("nan")
    drift: float = float("nan")
    damaged_share: float = float("nan")
    seconds: float = 0.0
    notes: str = ""
    extension: str = ""
    output_name: str = ""
    data: bytes = field(default=b"", repr=False)

    @property
    def saving(self) -> float:
        """Fraction of the input size saved; negative when the output grew."""
        if not self.ok or self.input_bytes <= 0:
            return float("nan")
        return 1.0 - self.output_bytes / self.input_bytes


def summarize(result: Dict[str, object],
              damaged_share: Optional[float] = None) -> Dict[str, object]:
    """Flatten a pipeline or form result into the fields a batch row needs."""
    notes = list(result.get("enhancement_steps") or []) + list(result.get("messages") or [])
    return {
        "output_bytes": len(result["raw_bytes"]),
        "format": str(result["format"]),
        "width": int(result["width"]),
        "height": int(result["height"]),
        "fidelity": float(result.get("fidelity_ssim", result.get("ssim", float("nan")))),
        "drift": float(result.get("drift_ssim", 1.0)),
        "damaged_share": float("nan") if damaged_share is None else float(damaged_share),
        "notes": "; ".join(notes),
        "extension": str(result["extension"]),
        "data": bytes(result["raw_bytes"]),
    }


def run_batch(
    files: Sequence[Tuple[str, bytes]],
    job: Callable[[str, bytes], Dict[str, object]],
    on_progress: Optional[Callable[[int, int, BatchItem], None]] = None,
) -> List[BatchItem]:
    """Run `job` over every file. A failure is recorded, never raised."""
    items: List[BatchItem] = []
    for index, (name, raw) in enumerate(files):
        started = time.monotonic()
        try:
            item = BatchItem(name=name, ok=True, input_bytes=len(raw), **job(name, raw))
        except Exception as error:  # one broken upload must not sink the rest
            item = BatchItem(name=name, ok=False, input_bytes=len(raw),
                             error=f"{type(error).__name__}: {error}")
        item.seconds = time.monotonic() - started
        items.append(item)
        if on_progress is not None:
            on_progress(index + 1, len(files), item)
    assign_output_names(items)
    return items


def assign_output_names(items: Sequence[BatchItem]) -> None:
    """Give every output a distinct archive name.

    Two uploads called photo.png and photo.jpg would otherwise both become
    photo_pixelopt.jpg, and one would silently overwrite the other in the ZIP.
    """
    taken = set()
    for item in items:
        if not item.ok:
            continue
        stem = PurePath(item.name).stem or "image"
        candidate, counter = f"{stem}_pixelopt.{item.extension}", 2
        while candidate.lower() in taken:
            candidate = f"{stem}_pixelopt-{counter}.{item.extension}"
            counter += 1
        taken.add(candidate.lower())
        item.output_name = candidate


def _cell(value: float, digits: int) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) \
        else f"{value:.{digits}f}"


def report_csv(items: Sequence[BatchItem]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow([
        "file", "status", "input_kb", "output_kb", "saving_pct", "format",
        "width", "height", "fidelity_ssim", "drift_ssim", "damaged_area_pct",
        "seconds", "output_name", "notes",
    ])
    for item in items:
        writer.writerow([
            item.name,
            "ok" if item.ok else "error",
            _cell(item.input_bytes / 1024, 1),
            _cell(item.output_bytes / 1024, 1) if item.ok else "",
            _cell(item.saving * 100, 1),
            item.format,
            item.width or "",
            item.height or "",
            _cell(item.fidelity, 4),
            _cell(item.drift, 4),
            _cell(item.damaged_share * 100, 2),
            _cell(item.seconds, 2),
            item.output_name,
            item.notes if item.ok else item.error,
        ])
    return buffer.getvalue()


def build_zip(items: Sequence[BatchItem]) -> bytes:
    """Every successful output plus report.csv.

    Images are stored rather than deflated: they are already compressed, so
    deflate would only burn CPU for a few bytes.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in items:
            if item.ok:
                archive.writestr(item.output_name, item.data,
                                 compress_type=zipfile.ZIP_STORED)
        archive.writestr("report.csv", report_csv(items))
    return buffer.getvalue()
