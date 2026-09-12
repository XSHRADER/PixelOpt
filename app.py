"""Streamlit interface for PixelOpt.

Layout note: the controls and the explanation render before any compression
runs, because Streamlit streams elements top to bottom and greys out stale
ones while a slow step is in progress. Doing the work last keeps the page
from flickering on every rerun.
"""

from __future__ import annotations

import io

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from pixelopt.enhance import (
    PRESETS,
    Enhancements,
    auto_enhancements,
    estimate_noise,
    preset,
)
from pixelopt.adaptive_compressor import AdaptiveImageCompressor
from pixelopt.image_features import has_transparency, load_image
from pixelopt.pipeline import knee_point, process, rate_distortion_curve
from ui_components import compare_view

st.set_page_config(
    page_title="PixelOpt",
    page_icon=":material/compress:",
    layout="wide",
)


@st.cache_data(show_spinner=False, max_entries=8)
def _load(raw: bytes):
    """Decode once per uploaded file rather than on every rerun."""
    image = load_image(io.BytesIO(raw))
    return image, estimate_noise(image)


@st.cache_data(show_spinner=False, max_entries=24)
def _run(raw: bytes, target_kb: float, settings: Enhancements, fmt: str | None):
    """Compression is the expensive step; key it on everything that changes it."""
    return process(
        io.BytesIO(raw), target_kb, enhancements=settings, image_format=fmt
    )


@st.cache_data(show_spinner=False, max_entries=8)
def _ceiling_kb(raw: bytes, settings: Enhancements, fmt: str | None) -> float:
    """The largest file this image can produce: full resolution, max quality.

    The curve has to be sampled below this. Spacing budgets off the slider
    instead meant that a well-compressing image put most of the ladder above
    its own ceiling -- 5 of 7 points unreachable, and nothing left to plot.
    """
    # Lossless is excluded: a PNG ceiling is not representative of the
    # lossy rate-distortion curve this chart is about, and spacing the ladder
    # off it pushed most budgets above what JPEG or WebP can fill.
    lossy = (
        AdaptiveImageCompressor(formats=("JPEG", "WEBP")) if fmt is None else None
    )
    result = process(
        io.BytesIO(raw), 1e9, enhancements=settings, image_format=fmt,
        compressor=lossy,
    )
    return float(result["actual_size_kb"])


@st.cache_data(show_spinner=False, max_entries=8)
def _curve(raw: bytes, settings: Enhancements, fmt: str | None, budgets: tuple):
    return rate_distortion_curve(
        io.BytesIO(raw), list(budgets), enhancements=settings, image_format=fmt
    )


st.title(":material/compress: PixelOpt")
st.caption(
    "Fit an image into a byte budget at the highest measured quality — "
    "then check the result at 1:1 rather than taking the number on trust."
)

uploaded = st.file_uploader(
    "Image", type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"]
)

intro_slot = st.empty()

if uploaded is None:
    with intro_slot.container(horizontal=True, gap="medium"):
        with st.container(border=True):
            st.subheader(":material/straighten: Resolution from arithmetic")
            st.markdown(
                "Downscaling by `r` leaves `r²·N` pixels, so the encoder works "
                "at `bpp/r²`. Solving for a workable rate gives "
                "`r = √(bpp/0.55)` — no guessing, no model."
            )
        with st.container(border=True):
            st.subheader(":material/timeline: Quality by interpolation")
            st.markdown(
                "File size is monotonic and roughly exponential in the quality "
                "setting, so interpolating on log-size finds the budget "
                "boundary in about three encodes instead of a full sweep."
            )
        with st.container(border=True):
            st.subheader(":material/auto_fix_high: Denoise pays for itself")
            st.markdown(
                "Noise is the most expensive thing an encoder can store. "
                "Removing it first improved quality in **9 of 9** measured "
                "conditions — by **+380%** SSIM at high noise."
            )
    st.info(
        "Upload an image to begin.", icon=":material/upload_file:"
    )
    st.stop()

raw = uploaded.getvalue()
image, noise_level = _load(raw)
height, width = image.shape[:2]
megapixels = width * height / 1e6

if has_transparency(Image.open(io.BytesIO(raw))):
    st.warning(
        "This image has transparency. The output is opaque, so transparent "
        "areas are composited onto white.",
        icon=":material/texture:",
    )

# ---------------------------------------------------------------- controls

controls = st.container(border=True)
with controls:
    left, right = st.columns([3, 2], gap="large")

    with left:
        target_kb = st.slider(
            "Target size", min_value=10, max_value=2000, value=200, step=10,
            format="%d KB",
            help="The output never exceeds this. It may land well under when "
                 "the image cannot fill the budget.",
        )
        bpp = target_kb * 1024 * 8 / max(width * height, 1)
        regime = (
            (":material/hd: Above ~0.5 bpp — full resolution wins, "
             "so only quality is tuned.")
            if bpp >= 0.5
            else (":material/photo_size_select_small: Below ~0.35 bpp — "
                  "downscaling first preserves more detail than it costs.")
        )
        st.caption(
            f"{width} × {height} · {megapixels:.1f} MP · "
            f"**{bpp:.2f} bits per pixel** at this budget"
        )
        st.markdown(regime)

    with right:
        fmt_choice = st.segmented_control(
            "Encoder",
            ["Auto", "JPEG", "WebP", "PNG"],
            default="Auto",
            help="Auto encodes every format and keeps whichever scores highest "
                 "at the budget — including lossless PNG when it fits.",
        )
        image_format = None if fmt_choice == "Auto" else fmt_choice.upper()

        enhance_choice = st.selectbox(
            "Enhancement",
            ["Auto", *sorted(PRESETS)],
            index=0,
            help="Auto measures the image noise and denoises only when the "
                 "measurement justifies it.",
        )

        if noise_level > 3.0:
            st.caption(
                f":material/grain: Measured noise **{noise_level:.1f}** — "
                "denoising will free up bits for real detail."
            )
        else:
            st.caption(
                f":material/check_circle: Measured noise **{noise_level:.1f}** — "
                "clean enough to leave alone."
            )

settings = (
    auto_enhancements(image) if enhance_choice == "Auto" else preset(enhance_choice)
)

with st.expander(":material/tune: Fine-tune enhancement", expanded=False):
    cols = st.columns(3, gap="medium")
    with cols[0]:
        denoise = st.slider("Denoise", 0.0, 22.0, float(settings.denoise), 0.5)
        sharpen = st.slider("Sharpen", 0.0, 1.5, float(settings.sharpen), 0.05)
    with cols[1]:
        contrast = st.slider("Contrast", 0.0, 5.0, float(settings.contrast), 0.1)
        saturation = st.slider("Saturation", 0.5, 2.0, float(settings.saturation), 0.05)
    with cols[2]:
        white_balance = st.toggle("White balance", value=settings.white_balance)
        auto_lvl = st.toggle("Auto levels", value=settings.auto_level)
        st.caption(
            "Sharpening *adds* high-frequency detail, which the encoder then "
            "has to store. It is worth it after a downscale, but it is not free."
        )
    settings = Enhancements(
        denoise=denoise, sharpen=sharpen, contrast=contrast,
        saturation=saturation, white_balance=white_balance, auto_level=auto_lvl,
    )

# ------------------------------------------------------------------ result

with st.spinner("Searching for the best encoding within the budget…"):
    result = _run(raw, float(target_kb), settings, image_format)

original_kb = len(raw) / 1024.0
actual_kb = float(result["actual_size_kb"])

metrics = st.container(horizontal=True, gap="small")
with metrics:
    st.metric(
        "Output size", f"{actual_kb:.1f} KB",
        delta=f"{actual_kb - original_kb:+.1f} KB vs source",
        delta_color="inverse", border=True, width="stretch",
    )
    st.metric(
        "Fidelity (SSIM)", f"{float(result['fidelity_ssim']):.4f}",
        help="Against the enhanced reference — how much the *encoder* lost.",
        border=True, width="stretch",
    )
    st.metric(
        "Encoder", str(result["format"]),
        delta=f"quality {result['quality']}", delta_color="off",
        border=True, width="stretch",
    )
    st.metric(
        "Dimensions", f"{result['width']} × {result['height']}",
        delta=f"resize {float(result['resize_factor']):.2f}×", delta_color="off",
        border=True, width="stretch",
    )
    st.metric(
        "Search cost", f"{result['encodes']} encodes",
        help="The exhaustive sweep this replaced averaged 69.",
        border=True, width="stretch",
    )

if result["enhanced"]:
    st.caption(
        ":material/auto_fix_high: **Enhanced:** "
        + ", ".join(result["enhancement_steps"])
        + f" · drift {float(result['drift_ssim']):.3f} from the original"
        + f" · noise {float(result['noise_before']):.1f} → "
        f"{float(result['noise_after']):.1f}"
    )

# ---------------------------------------------------------------- compare

compare_tab, curve_tab, detail_tab = st.tabs(
    [":material/compare: Compare", ":material/show_chart: Budget curve",
     ":material/info: Details"]
)

with compare_tab:
    st.caption(
        "Drag the divider. Scroll to zoom, drag to pan, double-click to reset. "
        "**Zoom to 1:1 or past it** — at fit-to-screen a 200 KB and a 500 KB "
        "encode of the same photo look identical."
    )
    compare_view(
        result["reference_image"],
        result["raw_bytes"],
        str(result["mime"]),
        "Reference" + (" (enhanced)" if result["enhanced"] else " (original)"),
        f"{result['format']} · {actual_kb:.1f} KB",
        key="compare",
    )

    st.container(height=8, border=False)
    downloads = st.container(horizontal=True, gap="small")
    stem = uploaded.name.rsplit(".", 1)[0]
    with downloads:
        # The exact measured bytes. Re-encoding here would hand over a
        # different file from the one every number above describes.
        st.download_button(
            f"Download {result['format']}",
            data=result["raw_bytes"],
            file_name=f"{stem}_compressed.{result['extension']}",
            mime=str(result["mime"]),
            type="primary",
            icon=":material/download:",
        )
        reference_buffer = io.BytesIO()
        result["reference_image"].save(reference_buffer, format="PNG")
        st.download_button(
            "Download reference (lossless PNG)",
            data=reference_buffer.getvalue(),
            file_name=f"{stem}_reference.png",
            mime="image/png",
            icon=":material/image:",
        )

with curve_tab:
    st.caption(
        "Quality as a function of budget for this image. Most images have an "
        "obvious knee, past which extra kilobytes buy almost nothing."
    )
    # Streamlit computes the contents of hidden tabs too, so this is gated:
    # without the guard, eight extra compressions ran on every rerun even
    # while Compare was the tab actually on screen. st.stop() is no use here
    # either -- it would halt the whole script and take the Details tab with
    # it -- so the whole body is conditional instead.
    asked = st.button(
        "Measure the curve", icon=":material/play_arrow:", key="run_curve"
    )
    if asked:
        st.session_state["curve_done"] = True

    if not st.session_state.get("curve_done"):
        st.info(
            "Runs a compression at each of eight budgets, so it takes a few "
            "seconds longer than a single encode.",
            icon=":material/timer:",
        )
    else:
        with st.spinner("Measuring quality across budgets…"):
            ceiling = _ceiling_kb(raw, settings, image_format)
            ladder = {
                max(3, round(ceiling * f))
                for f in (0.04, 0.08, 0.15, 0.28, 0.45, 0.7, 1.0)
            }
            if target_kb < ceiling:      # keep the chosen budget on the chart
                ladder.add(int(target_kb))
            points = _curve(raw, settings, image_format, tuple(sorted(ladder)))

        frame = pd.DataFrame(points)
        reachable = frame[~frame["capped"]]
        knee = knee_point(points)

        if len(reachable) < 2:
            st.info(
                f"This image tops out at about {ceiling:.0f} KB at full "
                "resolution and maximum quality, so there is no meaningful "
                "range of budgets to plot — it is already as good as the "
                "encoder can make it.",
                icon=":material/check_circle:",
            )
        else:
            base = alt.Chart(reachable).encode(
                x=alt.X("actual_kb:Q", title="File size (KB)"),
                y=alt.Y(
                    "ssim:Q", title="Fidelity (SSIM)",
                    scale=alt.Scale(zero=False, nice=True),
                ),
            )
            chart = (
                base.mark_area(
                    line={"strokeWidth": 2.5}, opacity=0.18,
                    interpolate="monotone",
                )
                + base.mark_point(size=95, filled=True).encode(
                    tooltip=[
                        alt.Tooltip("actual_kb:Q", title="Size (KB)", format=".1f"),
                        alt.Tooltip("ssim:Q", title="SSIM", format=".4f"),
                        alt.Tooltip("format:N", title="Encoder"),
                        alt.Tooltip("quality:Q", title="Quality"),
                        alt.Tooltip("width:Q", title="Width"),
                    ]
                )
            )
            if knee is not None:
                chart += (
                    alt.Chart(pd.DataFrame([knee]))
                    .mark_point(
                        size=340, shape="diamond", filled=False, strokeWidth=2.5
                    )
                    .encode(x="actual_kb:Q", y="ssim:Q")
                )
            st.altair_chart(
                chart.properties(height=330).interactive(), width="stretch"
            )

            if knee is not None:
                st.info(
                    f"Best value near **{knee['actual_kb']:.0f} KB** "
                    f"({knee['format']}, SSIM {knee['ssim']:.4f}). "
                    "The diamond marks it.",
                    icon=":material/savings:",
                )
            capped = frame[frame["capped"]]
            if not capped.empty:
                st.caption(
                    f":material/block: {len(capped)} of {len(frame)} budgets "
                    "are unreachable — at full resolution and maximum quality "
                    "the encoder cannot produce more bytes, so they are "
                    "excluded from the curve."
                )

with detail_tab:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Encoding")
        st.table(
            {
                "Encoder": str(result["format"]),
                "Encoder quality": str(result["quality"]),
                "Resize factor": f"{float(result['resize_factor']):.3f}",
                "Output dimensions": f"{result['width']} × {result['height']}",
                "Target": f"{float(result['target_size_kb']):.0f} KB",
                "Actual": f"{actual_kb:.2f} KB",
                "Encoder calls": str(result["encodes"]),
            }
        )
    with right:
        st.subheader("Quality")
        st.table(
            {
                "Fidelity SSIM": f"{float(result['fidelity_ssim']):.4f}",
                "Fidelity PSNR": f"{float(result['fidelity_psnr']):.2f} dB",
                "Fidelity MSE": f"{float(result['fidelity_mse']):.4f}",
                "Drift SSIM": f"{float(result['drift_ssim']):.4f}",
                "Noise before": f"{float(result['noise_before']):.2f}",
                "Noise after": f"{float(result['noise_after']):.2f}",
            }
        )
        st.caption(
            "**Fidelity** is measured against the enhanced reference and says "
            "what the encoder lost. **Drift** is measured against the original "
            "and says what the enhancement changed. A large drift on a noisy "
            "photo is the denoiser working — but it is also what "
            "overprocessing looks like, which is why both are shown."
        )
