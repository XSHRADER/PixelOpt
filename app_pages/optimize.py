"""Optimize: fit one image into a byte budget, then inspect it at 1:1."""

from __future__ import annotations

import io

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from app_shared import (
    CHIPS,
    UPLOAD_TYPES,
    ceiling_kb,
    curve_points,
    decode,
    process_view,
    quality_view,
    run_process,
    run_quality,
)
from pixelopt.enhance import PRESETS, Enhancements, auto_enhancements, preset
from pixelopt.image_features import has_transparency
from pixelopt.pipeline import knee_point
from ui_components import compare_view, hero, metric_strip, tone

hero(
    eyebrow="Measured, not guessed",
    title="Fit any image into a byte budget.",
    subtitle=(
        "PixelOpt picks the resolution, encoder and quality that look best "
        "under your size limit, cleans up noise when it pays for itself, and "
        "shows you exactly where quality was lost."
    ),
    chips=CHIPS,
)

uploaded = st.file_uploader("Image", type=UPLOAD_TYPES, key="optimize_upload")

intro_slot = st.empty()
if uploaded is None:
    # st.empty keeps this slot stable, so the cards do not linger as stale
    # elements once a file arrives.
    with intro_slot.container(horizontal=True, gap="medium"):
        for key, icon, title, body in (
            ("res", ":material/straighten:", "Resolution from arithmetic",
             "Downscaling by r leaves r²·N pixels, so the encoder works at bpp/r². "
             "Solving for a workable rate gives r = √(bpp/0.55)."),
            ("q", ":material/timeline:", "Quality by interpolation",
             "File size is monotonic in the quality setting, so interpolating on "
             "log-size finds the boundary in about three encodes."),
            ("heat", ":material/blur_on:", "See where it lost",
             "The damage heatmap shows local quality loss, so a ruined face on a "
             "clean frame never hides behind a good average."),
        ):
            with st.container(border=True, key=f"po_card_intro_{key}"):
                st.markdown(f"#### {icon} {title}")
                st.caption(body)
    st.stop()

raw = uploaded.getvalue()
image, noise_level = decode(raw)
height, width = image.shape[:2]

if has_transparency(Image.open(io.BytesIO(raw))):
    st.warning(
        "This image has transparency. The output is opaque, so transparent areas "
        "are composited onto white.",
        icon=":material/texture:",
    )

# ---------------------------------------------------------------- controls

with st.container(border=True, key="po_card_controls"):
    left, right = st.columns([3, 2], gap="large")
    with left:
        goal = st.segmented_control(
            "Goal", ["size", "quality"], default="size", key="optimize_goal",
            format_func={"size": "Size limit", "quality": "Quality target"}.get,
            help="Size limit: the best quality that fits a budget. Quality target: "
                 "the smallest file that still meets a fidelity you choose.",
        ) or "size"
        if goal == "size":
            min_ssim = None
            target_kb = st.slider(
                "Target size", min_value=10, max_value=2000, value=200, step=10,
                format="%d KB", key="optimize_target",
                help="The output never exceeds this. It may land well under when "
                     "the image cannot fill the budget.",
            )
            bpp = target_kb * 1024 * 8 / max(width * height, 1)
            st.caption(
                f"{width} × {height} · {width * height / 1e6:.1f} MP · "
                f"**{bpp:.2f} bits per pixel** at this budget"
            )
            st.markdown(
                ":material/hd: Above ~0.5 bpp, full resolution wins — only quality is tuned."
                if bpp >= 0.5 else
                ":material/photo_size_select_small: Below ~0.35 bpp, downscaling first "
                "keeps more detail than it costs."
            )
        else:
            target_kb = None
            min_ssim = st.slider(
                "Minimum fidelity (SSIM)", min_value=0.900, max_value=0.999, value=0.980,
                step=0.001, format="%.3f", key="optimize_min_ssim",
                help="The smallest file whose SSIM against the reference is at least "
                     "this. Higher stays closer to the reference and costs more bytes.",
            )
            st.caption(
                f"{width} × {height} · {width * height / 1e6:.1f} MP · the search tries "
                "each encoder at falling resolutions and keeps the smallest file that "
                "passes. Large images take longer here than in size mode."
            )
    with right:
        fmt_choice = st.segmented_control(
            "Encoder", ["Auto", "JPEG", "WebP", "PNG"], default="Auto",
            key="optimize_format",
            help="Auto tries every format and keeps whichever scores highest at "
                 "the budget, including lossless PNG when it fits.",
        ) or "Auto"
        image_format = None if fmt_choice == "Auto" else fmt_choice.upper()
        enhance_choice = st.selectbox(
            "Enhancement", ["Auto", *sorted(PRESETS)], key="optimize_enhance",
            help="Auto measures the noise and denoises only when the measurement "
                 "justifies it.",
        )
        if noise_level > 3.0:
            st.caption(f":material/grain: Measured noise **{noise_level:.1f}** — "
                       "denoising will free up bits for real detail.")
        else:
            st.caption(f":material/check_circle: Measured noise **{noise_level:.1f}** — "
                       "clean enough to leave alone.")

settings = auto_enhancements(image) if enhance_choice == "Auto" else preset(enhance_choice)

with st.expander(":material/tune: Fine-tune enhancement"):
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
        st.caption("White balance assumes the average colour is grey, so it can "
                   "over-correct scenes that are mostly one colour, like a big sky.")
    settings = Enhancements(denoise=denoise, sharpen=sharpen, contrast=contrast,
                            saturation=saturation, white_balance=white_balance,
                            auto_level=auto_lvl)

# ------------------------------------------------------------------ result

if goal == "size":
    with st.spinner("Searching for the best encoding within the budget…"):
        result = run_process(raw, float(target_kb), settings, image_format)
        view = process_view(raw, float(target_kb), settings, image_format)
else:
    with st.spinner("Searching for the smallest file that meets the target…"):
        result = run_quality(raw, float(min_ssim), settings, image_format)
        view = quality_view(raw, float(min_ssim), settings, image_format)

original_kb = len(raw) / 1024.0
actual_kb = float(result["actual_size_kb"])
fidelity = float(result["fidelity_ssim"])
damage = view["summary"]
kept = bool(result.get("passthrough"))
met = bool(result.get("met", True))

if kept:
    removed = int(result.get("metadata_removed_bytes", 0))
    st.success(
        ("Your original already fits, so it was kept as it is"
         if goal == "size" else
         "Your original is already the smallest file that meets the target, so it was kept")
        + " — identical pixels, no second round of compression. "
        + (f"{removed:,} bytes of metadata (EXIF, GPS location, comments) were removed."
           if removed else "It carried no metadata to remove."),
        icon=":material/verified:",
    )
elif goal == "quality" and not met:
    st.warning(
        f"No {fmt_choice if image_format else ''} encode reaches SSIM {min_ssim:.3f}; "
        "this is the closest it gets. Choose Auto to allow lossless PNG, which always "
        "meets the target.",
        icon=":material/warning:",
    )

quality_label = "original" if result.get("quality") is None else f"q{result['quality']}"
metric_strip(
    [
        {"label": "Output size", "value": actual_kb, "decimals": 1, "suffix": " KB",
         "hint": f"{(1 - actual_kb / original_kb) * 100:.0f}% smaller than the source"
                 if actual_kb < original_kb else "not smaller than the source",
         "tone": "good" if actual_kb < original_kb else "warn"},
        {"label": "Fidelity (SSIM)", "value": fidelity, "decimals": 4,
         "hint": (f"target ≥ {min_ssim:.3f}" if goal == "quality" else "against the reference"),
         "tone": ("good" if met else "bad") if goal == "quality" else tone(fidelity, 0.98, 0.93)},
        {"label": "Damaged area", "value": damage["damaged_share"] * 100, "decimals": 1,
         "suffix": "%", "hint": "local SSIM below 0.9",
         "tone": tone(damage["damaged_share"], 0.02, 0.15, higher_is_better=False)},
        {"label": "Encoder", "text": f"{result['format']} · {quality_label}",
         "hint": f"resize {float(result['resize_factor']):.2f}×"},
        {"label": "Dimensions", "text": f"{result['width']} × {result['height']}",
         "hint": f"from {width} × {height}"},
        {"label": "Search cost", "value": int(result["encodes"]), "suffix": " encodes",
         "hint": "an exhaustive sweep averaged 69"},
    ],
    key="optimize_metrics",
)

if result["enhanced"]:
    st.caption(
        ":material/auto_fix_high: **Enhanced:** " + ", ".join(result["enhancement_steps"])
        + f" · drift {float(result['drift_ssim']):.3f} from the original"
        + f" · noise {float(result['noise_before']):.1f} → {float(result['noise_after']):.1f}"
    )

compare_tab, curve_tab, detail_tab = st.tabs(
    [":material/compare: Compare", ":material/show_chart: Budget curve",
     ":material/info: Details"]
)

with compare_tab:
    compare_view(
        before_url=view["before_url"],
        after_url=view["after_url"],
        aspect=view["aspect"],
        label_before="Reference" + (" (enhanced)" if result["enhanced"] else ""),
        label_after=f"{result['format']} · {actual_kb:.1f} KB",
        heat_url=view["heat_url"],
        worst_box=damage["worst_box"],
        key="optimize_compare",
    )
    stem = uploaded.name.rsplit(".", 1)[0]
    with st.container(horizontal=True, gap="small"):
        # The exact measured bytes; re-encoding here would hand over a
        # different file from the one every number above describes.
        st.download_button(
            f"Download {result['format']} · {actual_kb:.1f} KB",
            data=result["raw_bytes"],
            file_name=f"{stem}_pixelopt.{result['extension']}",
            mime=str(result["mime"]), type="primary", icon=":material/download:",
        )
        reference_buffer = io.BytesIO()
        result["reference_image"].save(reference_buffer, format="PNG")
        st.download_button(
            "Reference (lossless PNG)", data=reference_buffer.getvalue(),
            file_name=f"{stem}_reference.png", mime="image/png", icon=":material/image:",
        )

with curve_tab:
    st.caption("Quality as a function of budget. Most images have an obvious knee, "
               "past which extra kilobytes buy almost nothing.")
    # Streamlit computes hidden tabs too, so this is gated. st.stop() would
    # halt the whole page, so the body is conditional instead.
    if st.button("Measure the curve", icon=":material/play_arrow:", key="run_curve"):
        st.session_state["curve_done"] = True
    if not st.session_state.get("curve_done"):
        st.info("Runs a compression at each of eight budgets, so it takes a few "
                "seconds longer than a single encode.", icon=":material/timer:")
    else:
        with st.spinner("Measuring quality across budgets…"):
            ceiling = ceiling_kb(raw, settings, image_format)
            ladder = {max(3, round(ceiling * f)) for f in (0.04, 0.08, 0.15, 0.28, 0.45, 0.7, 1.0)}
            if target_kb is not None and target_kb < ceiling:
                ladder.add(int(target_kb))
            points = curve_points(raw, settings, image_format, tuple(sorted(ladder)))
        frame = pd.DataFrame(points)
        reachable = frame[~frame["capped"]]
        knee = knee_point(points)
        if len(reachable) < 2:
            st.info(f"This image tops out at about {ceiling:.0f} KB at full resolution and "
                    "maximum quality — there is no range of budgets worth plotting.",
                    icon=":material/check_circle:")
        else:
            base = alt.Chart(reachable).encode(
                x=alt.X("actual_kb:Q", title="File size (KB)"),
                y=alt.Y("ssim:Q", title="Fidelity (SSIM)", scale=alt.Scale(zero=False, nice=True)),
            )
            chart = (
                base.mark_area(line={"strokeWidth": 2.5}, opacity=0.16, interpolate="monotone")
                + base.mark_point(size=90, filled=True).encode(tooltip=[
                    alt.Tooltip("actual_kb:Q", title="Size (KB)", format=".1f"),
                    alt.Tooltip("ssim:Q", title="SSIM", format=".4f"),
                    alt.Tooltip("format:N", title="Encoder"),
                    alt.Tooltip("quality:Q", title="Quality"),
                ])
            )
            if knee is not None:
                chart += alt.Chart(pd.DataFrame([knee])).mark_point(
                    size=340, shape="diamond", filled=False, strokeWidth=2.5
                ).encode(x="actual_kb:Q", y="ssim:Q")
            st.altair_chart(chart.properties(height=320).interactive(), width="stretch")
            if knee is not None:
                st.info(f"Best value near **{knee['actual_kb']:.0f} KB** ({knee['format']}, "
                        f"SSIM {knee['ssim']:.4f}). The diamond marks it.",
                        icon=":material/savings:")

with detail_tab:
    left, middle, right = st.columns(3, gap="large")
    with left:
        st.markdown("#### Encoding")
        st.table({
            "Encoder": str(result["format"]),
            "Quality": quality_label,
            "Resize factor": f"{float(result['resize_factor']):.3f}",
            "Dimensions": f"{result['width']} × {result['height']}",
            "Target": (f"{target_kb} KB" if goal == "size" else f"SSIM >= {min_ssim:.3f}"),
            "Original kept": "yes, metadata removed" if kept else "no",
            "Actual": f"{actual_kb:.2f} KB",
            "Encoder calls": str(result["encodes"]),
        })
    with middle:
        st.markdown("#### Quality")
        st.table({
            "Fidelity SSIM": f"{fidelity:.4f}",
            "Fidelity PSNR": f"{float(result['fidelity_psnr']):.2f} dB",
            "Drift SSIM": f"{float(result['drift_ssim']):.4f}",
            "Noise before": f"{float(result['noise_before']):.2f}",
            "Noise after": f"{float(result['noise_after']):.2f}",
        })
    with right:
        st.markdown("#### Damage")
        x0, y0, x1, y1 = damage["worst_box"]
        st.table({
            "Damaged area": f"{damage['damaged_share'] * 100:.2f}%",
            "Mean local loss": f"{damage['mean_loss']:.4f}",
            "95th percentile": f"{damage['p95_loss']:.4f}",
            "Worst region loss": f"{damage['worst_loss']:.4f}",
            "Worst region": f"{x0 * 100:.0f}–{x1 * 100:.0f}% across, {y0 * 100:.0f}–{y1 * 100:.0f}% down",
        })
    st.caption(
        "**Fidelity** is measured against the enhanced reference: what the encoder lost. "
        "**Drift** is measured against the original: what enhancement changed. "
        "**Damaged area** is the share of the frame whose local SSIM fell below 0.9, "
        "roughly where compression damage becomes visible at 1:1."
    )
