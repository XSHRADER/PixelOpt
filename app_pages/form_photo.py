"""Form photo: exact pixel dimensions and a file-size range."""

from __future__ import annotations

import streamlit as st
from PIL import Image, ImageDraw

from app_shared import UPLOAD_TYPES, decode, form_view, run_form
from pixelopt.forms import FORM_PRESETS, FormSpec
from ui_components import (
    compare_view,
    file_line,
    intro_cards,
    metric_strip,
    page_header,
    range_meter,
    tone,
    workbench,
)

# Space this page needs around the output viewer: the range meter, metrics,
# a message line, the viewer toolbar and the download button. Measured at
# 1440 x 900. The output is only a few hundred pixels tall, so a viewer of
# about that height already shows it near 1:1.
VIEWER_RESERVE_PX = 630
# The framing preview gets about the same height as the output viewer.
PREVIEW_MAX_HEIGHT = 300


def upload_widget():
    # Identical arguments in both layouts, so the file survives the move.
    return st.file_uploader("Photo or signature", type=UPLOAD_TYPES, key="form_upload")


if st.session_state.get("form_upload") is None:
    page_header(
        eyebrow="Form photo mode",
        title="Exact pixels. Exact kilobytes.",
        subtitle=(
            "For upload forms that demand a precise size and a file-size range. "
            "Every file passes whether the portal counts a kilobyte as 1000 or 1024 "
            "bytes, and the crop follows the face or the detail rather than the centre."
        ),
    )
    upload_widget()
    intro_cards([
        ("units", ":material/rule:", "Both kilobyte readings",
         "Portals disagree on whether 1 KB is 1000 or 1024 bytes. The limits are "
         "applied so the file passes under either."),
        ("crop", ":material/center_focus_strong:", "A crop that keeps the subject",
         "Crops anchor on a detected face, or on the most detailed region, "
         "instead of blindly taking the centre."),
        ("pad", ":material/data_object:", "Honest about padding",
         "If an image is too simple to reach the minimum even at top quality, "
         "comment bytes are added and reported. Pixels are never touched."),
    ])
    st.stop()

uploaded = st.session_state["form_upload"]
raw = uploaded.getvalue()
source, _ = decode(raw)
file_line(uploaded.name, f"{source.shape[1]} × {source.shape[0]} · {len(raw) / 1024:,.0f} KB")

panel, result_col = workbench("form")

# ------------------------------------------------------------------- spec

with panel:
    upload_widget()
    options = list(FORM_PRESETS) + ["custom"]
    choice = st.pills(
        "Preset", options, default="photo", key="form_preset",
        format_func=lambda k: "Custom" if k == "custom" else FORM_PRESETS[k].label,
    ) or "photo"
    base = FORM_PRESETS.get(choice, FORM_PRESETS["photo"])

    # Keys include the preset, so switching preset resets the fields to it.
    c1, c2 = st.columns(2)
    width = c1.number_input("Width (px)", 16, 8000, base.width, step=1, key=f"fw_{choice}")
    height = c2.number_input("Height (px)", 16, 8000, base.height, step=1, key=f"fh_{choice}")
    c3, c4 = st.columns(2)
    min_kb = c3.number_input("Minimum (KB)", 0.0, 20000.0, float(base.min_kb), step=1.0,
                             key=f"fmin_{choice}")
    max_kb = c4.number_input("Maximum (KB)", 1.0, 20000.0, float(base.max_kb), step=1.0,
                             key=f"fmax_{choice}")

    fit = st.segmented_control(
        "Framing", ["crop", "pad"], default=base.fit, key=f"ffit_{choice}",
        format_func={"crop": "Crop to fill", "pad": "Fit on white"}.get,
        help="Crop fills the frame and anchors on a face or the most detailed "
             "region. Fit on white keeps everything, which is safer for signatures.",
    ) or base.fit
    d1, d2 = st.columns(2, vertical_alignment="bottom")
    grayscale = d1.toggle("Grayscale", value=base.grayscale, key=f"fgray_{choice}")
    dpi_options = [None, 72, 150, 200, 300, 600]
    dpi = d2.selectbox(
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

# ----------------------------------------------------------------- result

with result_col:
    with st.spinner("Framing and encoding…"):
        result = run_form(raw, spec)
        view = form_view(raw, spec)

    # The range meter's status pill already says "in range" or "padded", and
    # the messages below explain any padding, so only failure gets an alert.
    if not result["meets_max"]:
        st.error(f"Cannot fit {spec.width} × {spec.height} under {spec.max_kb:g} KB.",
                 icon=":material/error:")

    range_meter(result["low_bytes"], result["high_bytes"], result["size_bytes"],
                len(result["unpadded_bytes"]), key="form_meter")

    metric_strip(
        [
            {"label": "File size", "value": result["size_kb"], "decimals": 1, "suffix": " KB",
             "hint": f"{result['size_kb_decimal']:.1f} KB if 1 KB = 1000 bytes",
             "tone": "good" if result["meets_min"] and result["meets_max"] else "bad"},
            {"label": "Fidelity (SSIM)", "value": result["ssim"], "decimals": 4,
             "hint": "against the framed reference", "tone": tone(result["ssim"], 0.98, 0.93)},
            {"label": "Damaged area", "value": view["summary"]["damaged_share"] * 100,
             "decimals": 1, "suffix": "%", "hint": "local SSIM below 0.9",
             "tone": tone(view["summary"]["damaged_share"], 0.02, 0.15, higher_is_better=False)},
            {"label": "JPEG quality", "value": result["quality"],
             "hint": f"chroma {result['subsampling']} · {spec.width} × {spec.height}"
                     + (f" · {spec.dpi} dpi" if spec.dpi else "")},
        ],
        key="form_metrics",
    )

    for message in result["messages"]:
        st.caption(":material/info: " + message)

    framing_col, output_col = st.columns(2, gap="medium")

    with framing_col:
        original = result["original_image"]
        preview = original.copy()
        preview.thumbnail((900, 900), Image.LANCZOS)
        scale = preview.width / original.width
        if spec.fit == "crop":
            left, top, right, bottom = (int(v * scale) for v in result["crop_box"])
            # Dim what gets cropped away, outline what is kept.
            shade = Image.new("RGBA", preview.size, (31, 32, 35, 150))
            ImageDraw.Draw(shade).rectangle([left, top, right, bottom], fill=(0, 0, 0, 0))
            preview = Image.alpha_composite(preview.convert("RGBA"), shade)
            ImageDraw.Draw(preview).rectangle([left, top, right, bottom],
                                              outline=(92, 200, 184, 255), width=3)
        st.image(preview, width=min(preview.width,
                                    int(PREVIEW_MAX_HEIGHT * preview.width / preview.height)))
        st.caption({
            "face": ":material/face: Crop anchored on the detected face.",
            "detail": ":material/filter_center_focus: Crop anchored on the most detailed region.",
            "centre": ":material/crop_free: Nothing stood out, so the crop is centred.",
            "pad": ":material/fit_screen: Fitted on white — nothing was cropped.",
        }[result["anchor"]])

    with output_col:
        compare_view(
            before_url=view["before_url"],
            after_url=view["after_url"],
            aspect=view["aspect"],
            label_before="Framed reference",
            label_after=f"JPEG · {result['size_kb']:.1f} KB",
            heat_url=view["heat_url"],
            worst_box=view["summary"]["worst_box"],
            reserve_px=VIEWER_RESERVE_PX,
            key="form_compare",
        )
        stem = uploaded.name.rsplit(".", 1)[0]
        st.download_button(
            f"Download JPEG · {result['size_kb']:.1f} KB",
            data=result["raw_bytes"],
            file_name=f"{stem}_{spec.width}x{spec.height}.jpg",
            mime="image/jpeg", type="primary", icon=":material/download:",
            disabled=not result["meets_max"], on_click="ignore",
        )
