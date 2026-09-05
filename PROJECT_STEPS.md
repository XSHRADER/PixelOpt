# Project Steps: ML-Based Adaptive Image Compression and Resizing

## 1. Set up the environment

1. Open the project folder in VS Code.
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```
3. Activate the environment:
   - Windows PowerShell:
     ```bash
     .\.venv\Scripts\Activate.ps1
     ```
   - Command Prompt:
     ```bash
     .\.venv\Scripts\activate.bat
     ```
4. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## 2. Prepare the project dataset

1. Collect a set of reference images for training.
2. Put them in a folder such as:
   ```text
   data/images/
   ```

3. If no dataset is available, the project can still train using the built-in synthetic fallback pipeline.

## 3. Train the adaptive model

Run:
```bash
python train_model.py
```

This script:
- builds feature vectors from image characteristics,
- generates target pairs for resize factor and JPEG quality,
- trains the multi-output regression model,
- saves the trained model to the models folder.

## 4. Launch the Streamlit app

Run:
```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:
```text
http://localhost:8501
```

## 5. Use the app

1. Upload an image.
2. Set a target output size in KB.
3. Click “Compress and resize image”.
4. The system predicts the best resize factor and JPEG quality.
5. The app displays:
   - the original image,
   - the resized and compressed output,
   - the target size,
   - the actual size,
   - SSIM,
   - PSNR,
   - MSE.
6. Download either the resized image or the final compressed image.

## 6. Evaluate the output

Compare the final result against the original image using quality metrics:
- SSIM: higher is better
- PSNR: higher is better
- MSE: lower is better

## 7. Improve the system later

Possible future upgrades:
- Add WebP/AVIF support
- Compare Random Forest with XGBoost or neural networks
- Train on a larger real-world image dataset
- Add batch processing for many images
- Add baseline comparison against standard compression methods

## 8. Summary of the workflow

```text
Input image
   ↓
Feature extraction
   ↓
ML prediction of resize and quality
   ↓
Image resizing + JPEG tuning
   ↓
Quality evaluation (SSIM, PSNR, MSE)
   ↓
Resized and compressed outputs ready for download
```
