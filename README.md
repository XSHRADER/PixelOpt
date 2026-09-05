# ML-Based Adaptive Image Compression and Resizing

This project implements an intelligent image compression workflow that predicts the best resize factor and JPEG quality for a target file size. It combines computer-vision feature extraction with a multi-output regression model to balance file-size reduction with visual quality preservation.

## Features

- Extracts image statistics such as sharpness, edge density, color variance, and texture complexity.
- Trains a multi-output regressor to estimate the best resize factor and JPEG quality.
- Uses an iterative size-tuning loop so the final output stays close to the chosen target size.
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

The model is trained on image metadata and visual features plus the target size. The regressor output is then refined by a small iterative adjustment loop that keeps the encoded output near the requested target size.

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
