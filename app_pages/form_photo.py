"""Form photo: exact pixel dimensions and a file-size range."""

from __future__ import annotations

import streamlit as st
from PIL import Image, ImageDraw

from app_shared import UPLOAD_TYPES, form_view, run_form
from pixelopt.forms import FORM_PRESETS, FormSpec
from ui_components import compare_view, hero, metric_strip, range_meter, tone

hero(
    eyebrow="Form photo mode",
    title="Exact pixels. Exact kilobytes.",
    subtitle=(
        "For upload forms that demand a precise size and a file-size range. "
        "Every file passes whether the portal counts a kilobyte as 1000 or 1024 "
        "bytes, and the crop follows the face or the detail rather than the centre."
    ),
    chips=("Exact dimensions", "Minimum and maximum size", "Both KB readings",
           "Face-aware crop", "Baseline JPEG", "DPI written in", "Honest padding"),
)

uploaded = st.file_uploader("Photo or signature", type=UPLOAD_TYPES, key="form_upload")

# ------------------------------------------------------------------- spec

with st.container(border=True, key="po_card_form_spec"):
    options = list(FORM_PRESETS) + ["custom"]
    choice = st.pills(
        "Preset", options, default="photo", key="form_preset",
        format_func=lambda k: "Custom" if k == "custom" else FORM_PRESETS[k].label,
    ) or "photo"
    base = FORM_PRESETS.get(choice, FORM_PRESETS["photo"])

    # Keys include the preset, so switching preset resets the fields to it.
    c1, c2, c3, c4 = st.columns(4)
    width = c1.number_input("Width (px)", 16, 8000, base.width, step=1, key=f"fw_{choice}")
    height = c2.number_input("Height (px)", 16, 8000, base.height, step=1, key=f"fh_{choice}")
    min_kb = c3.number_input("Minimum (KB)", 0.0, 20000.0, float(base.min_kb), step=1.0,
                             key=f"fmin_{choice}")
    max_kb = c4.number_input("Maximum (KB)", 1.0, 20000.0, float(base.max_kb), step=1.0,
                             key=f"fmax_{choice}")

    d1, d2, d3 = st.columns([2, 1, 1])
    fit = d1.segmented_control(
        "Framing", ["crop", "pad"], default=base.fit, key=f"ffit_{choice}",
        format_func={"crop": "Crop to fill", "pad": "Fit on white"}.get,
        help="Crop fills the frame and anchors on a face or the most detailed "
             "region. Fit on white keeps everything, which is safer for signatures.",
    ) or base.fit
    grayscale = d2.toggle("Grayscale", value=base.grayscale, key=f"fgray_{choice}")
    dpi_options = [None, 72, 150, 200, 300, 600]
    dpi = d3.selectbox(
        "DPI", dpi_options, index=dpi_options.index(base.dpi) if base.dpi in dpi_options else 0,
        format_func=lambda d: "Not set" if d is None else f"{d} dpi", key=f"fdpi_{choice}",
    )
    st.caption(
        ":material/info: Presets are common sizes, not any organisation's official "
        "specification — always check what your form asks for."
    )

spec = FormSpec(int(width), int(height), float(min_kb), float(max_kb), fit,
                bool(grayscale), dpi, label=base.label if choice != "custom" else "Custom")
try:
    spec.validate()
except ValueError as error:
    st.error(str(error), icon=":material/error:")
    st.stop()

if uploaded is None:
    with st.container(horizontal=True, gap="medium"):
        for key, icon, title, body in (
            ("units", ":material/rule:", "Both kilobyte readings",
             "Portals disagree on whether 1 KB is 1000 or 1024 bytes. The limits are "
             "applied so the file passes under either."),
            ("crop", ":material/center_focus_strong:", "A crop that keeps the subject",
             "Crops anchor on a detected face, or on the most detailed region, "
             "instead of blindly taking the centre."),
            ("pad", ":material/data_object:", "Honest about padding",
             "If an image is too simple to reach the minimum even at top quality, "
             "comment bytes are added and reported. Pixels are never touched."),
        ):
            with st.container(border=True, key=f"po_card_intro_{key}"):
                st.markdown(f"#### {icon} {title}")
                st.caption(body)
    st.stop()

raw = uploaded.getvalue()
with st.spinner("Framing and encoding…"):
    result = run_form(raw, spec)
    view = form_view(raw, spec)

# ----------------------------------------------------------------- status

if not result["meets_max"]:
    st.error(f"Cannot fit {spec.width} × {spec.height} under {spec.max_kb:g} KB.",
             icon=":material/error:")
elif result["padded_bytes"] > 0:
    st.warning(f"In range at {result['size_kb']:.1f} KB, with "
               f"{result['padded_bytes']:,} bytes of padding.", icon=":material/data_object:")
else:
    st.success(f"Ready: {result['size_kb']:.1f} KB, inside {spec.min_kb:g}–{spec.max_kb:g} KB "
               "under both kilobyte readings.", icon=":material/verified:")

range_meter(result["low_bytes"], result["high_bytes"], result["size_bytes"],
            len(result["unpadded_bytes"]), key="form_meter")

metric_strip(
    [
        {"label": "File size", "value": result["size_kb"], "decimals": 1, "suffix": " KB",
         "hint": f"{result['size_kb_decimal']:.1f} KB if 1 KB = 1000 bytes",
         "tone": "good" if result["meets_min"] and result["meets_max"] else "bad"},
        {"label": "JPEG quality", "value": result["quality"],
         "hint": f"chroma {result['subsampling']}"},
        {"label": "Fidelity (SSIM)", "value": result["ssim"], "decimals": 4,
         "hint": "against the framed reference", "tone": tone(result["ssim"], 0.98, 0.93)},
        {"label": "Damaged area", "value": view["summary"]["damaged_share"] * 100,
         "decimals": 1, "suffix": "%", "hint": "local SSIM below 0.9",
         "tone": tone(view["summary"]["damaged_share"], 0.02, 0.15, higher_is_better=False)},
        {"label": "Dimensions", "text": f"{spec.width} × {spec.height}",
         "hint": f"{spec.dpi} dpi" if spec.dpi else "DPI not set"},
    ],
    key="form_metrics",
)

for message in result["messages"]:
    st.caption(":material/info: " + message)

# ---------------------------------------------------------------- preview

framing_col, output_col = st.columns([1, 1], gap="large")

with framing_col:
    st.markdown("#### Framing")
    original = result["original_image"]
    preview = original.copy()
    preview.thumbnail((900, 900), Image.LANCZOS)
    scale = preview.width / original.width
    if spec.fit == "crop":
        left, top, right, bottom = (int(v * scale) for v in result["crop_box"])
        # Dim what gets cropped away, outline what is kept.
        shade = Image.new("RGBA", preview.size, (8, 9, 10, 150))
        ImageDraw.Draw(shade).rectangle([left, top, right, bottom], fill=(0, 0, 0, 0))
        preview = Image.alpha_composite(preview.convert("RGBA"), shade)
        ImageDraw.Draw(preview).rectangle([left, top, right, bottom],
                                          outline=(123, 127, 255, 255), width=3)
    st.image(preview, width="stretch")
    st.caption({
        "face": ":material/face: Crop anchored on the detected face.",
        "detail": ":material/filter_center_focus: Crop anchored on the most detailed region.",
        "centre": ":material/crop_free: Nothing stood out, so the crop is centred.",
        "pad": ":material/fit_screen: Fitted on white — nothing was cropped.",
    }[result["anchor"]])

with output_col:
    st.markdown("#### Output")
    compare_view(
        before_url=view["before_url"],
        after_url=view["after_url"],
        aspect=view["aspect"],
        label_before="Framed reference",
        label_after=f"JPEG · {result['size_kb']:.1f} KB",
        heat_url=view["heat_url"],
        worst_box=view["summary"]["worst_box"],
        key="form_compare",
    )
    stem = uploaded.name.rsplit(".", 1)[0]
    st.download_button(
        f"Download JPEG · {result['size_kb']:.1f} KB",
        data=result["raw_bytes"],
        file_name=f"{stem}_{spec.width}x{spec.height}.jpg",
        mime="image/jpeg", type="primary", icon=":material/download:",
        disabled=not result["meets_max"],
    )
