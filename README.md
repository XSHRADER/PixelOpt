# ML-Based Adaptive Image Compression and Resizing

[![tests](https://github.com/XSHRADER/PixelOpt/actions/workflows/tests.yml/badge.svg)](https://github.com/XSHRADER/PixelOpt/actions/workflows/tests.yml)

This project implements an intelligent image compression workflow that predicts the best resize factor and JPEG quality for a target file size. It combines computer-vision feature extraction with a multi-output regression model to balance file-size reduction with visual quality preservation.

## Measured results

`python benchmark.py` runs four 1200x1200 test images across three target
sizes and reports what the compressor actually delivers.

| image | target | actual | error | SSIM | PSNR | ratio |
|---|---|---|---|---|---|---|
| gradient | 100 KB | 92.5 KB | −7.5% | 0.999 | 59.9 | 46x |
| edges/text | 100 KB | 99.6 KB | −0.4% | 0.988 | 42.7 | 42x |
| high detail | 100 KB | 99.1 KB | −0.9% | 0.524 | 21.9 | 43x |
| high detail | 200 KB | 199.6 KB | −0.2% | 0.612 | 22.7 | 21x |
| high detail | 500 KB | 497.3 KB | −0.5% | 0.902 | 27.0 | 8x |
| near-flat | 100 KB | 91.7 KB | −8.3% | 0.990 | 50.6 | 46x |

**Size-targeting error: median 0.7%, mean 3.0%, worst 8.3%.** Every output
stayed at or under its target — the point of a size budget is not to exceed it.

Six of the twelve runs are omitted above because their target was
**unreachable**: a smooth gradient encoded at full resolution and quality 95
is only 105 KB, so asking for 500 KB cannot produce 500 KB. In those cases the
compressor returns the maximum-fidelity encoding it can and the benchmark
reports them separately rather than counting them as error. Undershooting
there is a property of the image, not a miss by the search.

The test images are generated rather than photographed, and they deliberately
cover cases that stress a compressor differently. Size-targeting error
transfers to real photographs; the absolute SSIM/PSNR figures are only
meaningful relative to one another. The low SSIM on the high-detail image is
expected — dense high-frequency noise is exactly what JPEG discards first.

### What the benchmark caught

The original search was anchored to the model's prediction — it swept quality
within ±15 and resize within ±0.2 of it — and it resized an image that had
*already* been resized by the predicted factor. The two factors multiplied, so
the search could only ever shrink further and could never climb back toward a
larger target. A gradient asked for 100 KB, 200 KB and 500 KB returned the
same 28 KB file in all three cases.

Rebuilding the search to scale from the original image and binary-search
quality per resize factor took median size-targeting error from **74.1% to
0.7%**, moved every run to at-or-under target (was 9/12), lifted median SSIM
from 0.967 to 0.991, and roughly halved the runtime because the binary search
tries fewer encodings than the old fixed sweep.

## Features

- Extracts image statistics such as sharpness, edge density, color variance, and texture complexity.
- Trains a multi-output regressor to estimate the best resize factor and JPEG quality.
- Binary-searches JPEG quality per resize factor so the output lands as close under the target size as the image allows.
- Computes SSIM, PSNR, and MSE to compare the original and output images.
- Includes a Streamlit app for interactive use.
- Lets users download both the resized image and the final compressed image.
- Includes a training script to generate and fit the predictive model.

## Project Layout

- `app.py` — Streamlit web interface.
- `train_model.py` — trains or refreshes the regressor.
- `src/image_features.py` — image feature extraction and quality metrics.
- `src/compression_model.py` — model training, persistence, and prediction utilities.
- `src/adaptive_compressor.py` — full resize-and-compress pipeline.

## Quick Start

1. Open a terminal in the project folder.

2. Use the built-in launcher file on Windows:

   ```bat
   launch_app.bat
   ```

   Or run manually:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
   ```

3. Open the browser at:

   ```text
   http://127.0.0.1:8501
   ```

4. Upload an image and choose the target output size in KB.

5. Click “Compress and resize image”.

6. Download either the resized version or the final compressed JPEG.

## App startup notes

- The app is intended to run locally on your machine.
- If a trained model is not present, the app can build a fallback model from the uploaded image.
- For the simplest startup, use the included launcher script: `launch_app.bat`.

## How the app executes

1. `app.py` creates the Streamlit interface and starts the user workflow.
2. The uploaded image is passed to `AdaptiveImageCompressor.compress()`.
3. Feature extraction occurs in `src/image_features.py`.
4. The model predicts the resize factor and JPEG quality in `src/compression_model.py`.
5. The compressor tunes the resized output to match the requested size.
6. Quality metrics are calculated and displayed in the browser.
7. The user can download the resized image and the compressed image.

## Execution flow

```text
User uploads image
       ↓
app.py
       ↓
AdaptiveImageCompressor.compress()
       ↓
extract_feature_vector()
       ↓
Model predicts resize factor + JPEG quality
       ↓
Image is resized and tuned for target size
       ↓
JPEG is encoded and evaluated
       ↓
SSIM / PSNR / MSE are calculated
       ↓
Resized and compressed outputs are shown and can be downloaded
```

## Model strategy

The project uses a multi-output `CatBoostRegressor` (via `MultiOutputRegressor`) and predicts:

- resize factor
- JPEG quality

The prediction seeds a search that scales from the original image and binary-searches JPEG quality at each resize factor, keeping the largest encoding that still fits the requested size. The model steers the search; it no longer bounds it.

## Synthetic fallback dataset

When the project does not have a real image dataset available, it falls back to a synthetic dataset made from generated and sample image patterns. This allows the project to work immediately while still supporting real training with your own image directory.

## Typical usage

- Upload a photo or product image.
- Choose a target size such as 200 KB.
- The system predicts the best resize and quality settings.
- It generates resized and compressed outputs and compares them against the original with SSIM, PSNR, and MSE.
- Download the resized file or the final compressed version when ready.

## Future extensions

- Add WebP or AVIF support.
- Compare multiple regression models such as XGBoost or neural networks.
- Add automatic batch processing for directories.
- Integrate a more advanced perceptual-quality optimization loop.
