"""Streamlit-side helpers shared by the pages.

Kept out of the `pixelopt` package on purpose: the library must never import
Streamlit, so the CLI and the tests stay light.

Everything expensive is cached and keyed on the raw upload bytes plus the
settings that change the result. The comparison payload -- data URLs and the
heatmap -- is cached too, because rebuilding it on every rerun (the viewer
reruns the page each time its state is saved) would cost a WebP and a PNG
encode for nothing.
"""

from __future__ import annotations

import hashlib
import io
from typing import Callable, Dict, Optional

import cv2
import numpy as np
import streamlit as st

from pixelopt.adaptive_compressor import AdaptiveImageCompressor
from pixelopt.analysis import damage_map, damage_summary, heatmap_rgba
from pixelopt.batch import summarize
from pixelopt.enhance import Enhancements, auto_enhancements, estimate_noise, preset
from pixelopt.forms import FormSpec, encode_for_form
from pixelopt.image_features import load_image
from pixelopt.pipeline import process, process_to_quality, rate_distortion_curve
from ui_components import bytes_data_url, image_data_url, rgba_data_url

UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]

# The loss map can be 12 MP; the browser only needs a display-sized copy.
HEATMAP_DISPLAY_SIDE = 1400

CHIPS = (
    "Resolution from arithmetic", "Log-size interpolation", "SSIM-ranked winner",
    "JPEG · WebP · lossless PNG", "Measured denoising", "EXIF-aware",
    "Transparency onto white", "Damage heatmap", "Form photo mode",
    "Batch ZIP export",
)


def fingerprint(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()


@st.cache_data(show_spinner=False, max_entries=16)
def decode(raw: bytes):
    image = load_image(io.BytesIO(raw))
    return image, estimate_noise(image)


@st.cache_data(show_spinner=False, max_entries=24)
def run_process(raw: bytes, target_kb: float, settings: Enhancements, fmt: Optional[str]):
    return process(io.BytesIO(raw), target_kb, enhancements=settings, image_format=fmt)


@st.cache_data(show_spinner=False, max_entries=24)
def run_quality(raw: bytes, min_ssim: float, settings: Enhancements, fmt: Optional[str]):
    return process_to_quality(io.BytesIO(raw), min_ssim, enhancements=settings,
                              image_format=fmt)


@st.cache_data(show_spinner=False, max_entries=24)
def run_form(raw: bytes, spec: FormSpec):
    return encode_for_form(io.BytesIO(raw), spec)


def _view_payload(result: Dict[str, object]) -> Dict[str, object]:
    reference = result["reference_image"].convert("RGB")
    output = result["image"].convert("RGB")
    loss = damage_map(np.asarray(reference), np.asarray(output))
    summary = damage_summary(loss)  # from the full-resolution map
    # Shrink the LOSS map for display -- averaging loss is fine for a picture.
    # Shrinking the images before measuring is what made damage disappear.
    height, width = loss.shape
    if max(height, width) > HEATMAP_DISPLAY_SIDE:
        scale = HEATMAP_DISPLAY_SIDE / max(height, width)
        loss = cv2.resize(loss, (max(1, int(width * scale)), max(1, int(height * scale))),
                          interpolation=cv2.INTER_AREA)
    return {
        "before_url": image_data_url(reference),
        "after_url": bytes_data_url(result["raw_bytes"], str(result["mime"])),
        "heat_url": rgba_data_url(heatmap_rgba(loss)),
        "summary": summary,
        "aspect": reference.width / max(reference.height, 1),
    }


@st.cache_data(show_spinner=False, max_entries=16)
def process_view(raw: bytes, target_kb: float, settings: Enhancements, fmt: Optional[str]):
    return _view_payload(run_process(raw, target_kb, settings, fmt))


@st.cache_data(show_spinner=False, max_entries=16)
def quality_view(raw: bytes, min_ssim: float, settings: Enhancements, fmt: Optional[str]):
    return _view_payload(run_quality(raw, min_ssim, settings, fmt))


@st.cache_data(show_spinner=False, max_entries=16)
def form_view(raw: bytes, spec: FormSpec):
    return _view_payload(run_form(raw, spec))


@st.cache_data(show_spinner=False, max_entries=8)
def ceiling_kb(raw: bytes, settings: Enhancements, fmt: Optional[str]) -> float:
    """The largest file this image can produce, lossless excluded.

    A PNG ceiling is not representative of the lossy curve being plotted, and
    spacing the budget ladder off it pushed most points out of reach.
    """
    lossy = AdaptiveImageCompressor(formats=("JPEG", "WEBP")) if fmt is None else None
    result = process(io.BytesIO(raw), 1e9, enhancements=settings, image_format=fmt,
                     compressor=lossy)
    return float(result["actual_size_kb"])


@st.cache_data(show_spinner=False, max_entries=8)
def curve_points(raw: bytes, settings: Enhancements, fmt: Optional[str], budgets: tuple):
    return rate_distortion_curve(io.BytesIO(raw), list(budgets), enhancements=settings,
                                 image_format=fmt)


# ------------------------------------------------------------- batch jobs


def _damaged_share(result: Dict[str, object]) -> float:
    # Full resolution, same as the single-image view, so the Batch column and
    # the Optimize page agree about the same file.
    loss = damage_map(
        np.asarray(result["reference_image"].convert("RGB")),
        np.asarray(result["image"].convert("RGB")),
    )
    return damage_summary(loss)["damaged_share"]


def target_job(target_kb: Optional[float], fmt: Optional[str], enhancement: str,
               measure: bool, min_ssim: Optional[float] = None
               ) -> Callable[[str, bytes], Dict[str, object]]:
    """A size-limit job, or a quality-target job when `min_ssim` is given."""
    def job(name: str, raw: bytes) -> Dict[str, object]:
        image = load_image(io.BytesIO(raw))
        # Auto is resolved per image: a batch from different cameras will not
        # all want the same filter strength.
        settings = auto_enhancements(image) if enhancement == "Auto" else preset(enhancement)
        # The raw bytes, not the decoded array, so a file that already fits
        # can be kept as it is.
        if min_ssim is not None:
            result = process_to_quality(io.BytesIO(raw), float(min_ssim),
                                        enhancements=settings, image_format=fmt)
        else:
            result = process(io.BytesIO(raw), float(target_kb), enhancements=settings,
                             image_format=fmt)
        summary = summarize(result, _damaged_share(result) if measure else None)
        if result.get("passthrough"):
            removed = int(result.get("metadata_removed_bytes", 0))
            kept = ("original kept, " + f"{removed:,} bytes of metadata removed"
                    if removed else "original kept unchanged")
            summary["notes"] = kept + ("; " + summary["notes"] if summary["notes"] else "")
        if min_ssim is not None and not result.get("met", True):
            summary["notes"] = "TARGET MISSED; " + summary["notes"]
        return summary
    return job


def form_job(spec: FormSpec, measure: bool) -> Callable[[str, bytes], Dict[str, object]]:
    def job(name: str, raw: bytes) -> Dict[str, object]:
        result = encode_for_form(io.BytesIO(raw), spec)
        summary = summarize(result, _damaged_share(result) if measure else None)
        if not result["meets_max"]:
            summary["notes"] = "OVER LIMIT; " + summary["notes"]
        return summary
    return job
