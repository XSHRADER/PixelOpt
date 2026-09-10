from __future__ import annotations

import io

import streamlit as st
from PIL import Image

from pixelopt.adaptive_compressor import AdaptiveImageCompressor
from pixelopt.image_features import has_transparency, image_stats, load_image

st.set_page_config(page_title="Adaptive Image Compressor", layout="wide")
st.title("Adaptive Image Compression and Resizing Studio")

if "compressor" not in st.session_state:
    st.session_state.compressor = AdaptiveImageCompressor()

uploaded_file = st.file_uploader(
    "Upload an image to compress", type=["png", "jpg", "jpeg", "webp", "bmp"]
)

if uploaded_file is None:
    st.info("Upload an image to begin the compression workflow.")
else:
    original = Image.open(uploaded_file)

    # JPEG has no alpha channel, so transparency cannot survive. It is
    # composited onto white rather than silently dropped -- say so.
    if has_transparency(original):
        st.warning(
            "This image has transparency. The compressed output is opaque, so "
            "transparent areas are composited onto a white background."
        )

    image = load_image(original)
    pil_image = Image.fromarray(image)
    height, width = image.shape[:2]

    controls = st.columns([3, 2])
    with controls[0]:
        target_kb = st.slider(
            "Target output size (KB)", min_value=20, max_value=2000, value=200, step=10
        )
    with controls[1]:
        choice = st.selectbox(
            "Output format",
            ("Auto (best quality per byte)", "JPEG", "WebP"),
            help="Auto encodes both and keeps whichever scores higher at the same budget.",
        )
    image_format = None if choice.startswith("Auto") else choice.upper()

    bits_per_pixel = target_kb * 1024 * 8 / max(width * height, 1)
    st.caption(
        f"{width} x {height} ({width * height / 1e6:.1f} MP) at {target_kb} KB "
        f"= {bits_per_pixel:.2f} bits per pixel. "
        + (
            "Above ~0.5 bpp full resolution wins, so only quality is tuned."
            if bits_per_pixel >= 0.5
            else "Below ~0.35 bpp downscaling first preserves more detail than it costs."
        )
    )

    if st.button("Compress and resize image"):
        with st.spinner("Searching for the best encoding within the size budget..."):
            result = st.session_state.compressor.compress(
                pil_image, float(target_kb), image_format=image_format
            )

        st.subheader("Original vs resized and compressed output")
        col1, col2 = st.columns(2)
        with col1:
            st.image(
                pil_image,
                caption=f"Original image ({width} x {height})",
                use_container_width=True,
            )
        with col2:
            st.image(
                result["image"],
                caption=(
                    f"Compressed output ({result['width']} x {result['height']}, "
                    f"{result['format']})"
                ),
                use_container_width=True,
            )

        st.subheader("Compression summary")
        summary, analysis = st.columns(2)
        with summary:
            st.write(
                {
                    "Format chosen": result["format"],
                    "Resize factor": round(result["resize_factor"], 4),
                    "Encoder quality": result["quality"],
                    "Output dimensions": f"{result['width']} x {result['height']}",
                    "Target size (KB)": round(result["target_size_kb"], 2),
                    "Actual size (KB)": round(result["actual_size_kb"], 2),
                    "Encoder calls used": result["encodes"],
                }
            )
        with analysis:
            st.write(
                {
                    "SSIM": round(result["ssim"], 4),
                    "PSNR (dB)": round(result["psnr"], 2),
                    "MSE": round(result["mse"], 4),
                    **{
                        key.replace("_", " ").title(): round(value, 3)
                        for key, value in image_stats(image).items()
                    },
                }
            )

        st.subheader("Download outputs")
        stem = uploaded_file.name.rsplit(".", 1)[0]

        resized_buffer = io.BytesIO()
        result["resized_image"].save(resized_buffer, format="PNG")

        download_cols = st.columns(2)
        with download_cols[0]:
            st.download_button(
                label="Download resized image (lossless PNG)",
                data=resized_buffer.getvalue(),
                file_name=f"{stem}_resized.png",
                mime="image/png",
            )
        with download_cols[1]:
            # Hand over the exact bytes that were measured. Re-encoding here
            # would produce a different file from the one reported above.
            st.download_button(
                label=f"Download compressed image ({result['format']})",
                data=result["raw_bytes"],
                file_name=f"{stem}_compressed.{result['extension']}",
                mime=result["mime"],
            )

st.markdown(
    """
### How it works

The compressor fits an image into a byte budget at the highest measured quality.

1. **Resolution** comes from the bits-per-pixel budget: after downscaling by
   `r` the encoder works at `bpp / r^2`, so `r = sqrt(bpp / 0.55)` aims it at a
   rate where it produces detail rather than artefacts.
2. **Quality** is solved by interpolating on log file size, which converges in
   about five encodes instead of sweeping the whole range.
3. **The winner** is chosen by SSIM across the candidate resolutions and both
   formats, because the largest file under the budget is not the best-looking
   one.

### Quality metrics used

- SSIM: structural similarity index
- PSNR: peak signal-to-noise ratio
- MSE: mean squared error
"""
)
