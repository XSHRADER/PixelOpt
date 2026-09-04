# Build Project from Scratch

This file explains how to build the ML-Based Adaptive Image Compression project from the beginning.

## 1. Create the project folder

Open a terminal and create a folder for the project:

```bash
mkdir ML_Adaptive_Image_Compression
cd ML_Adaptive_Image_Compression
```

## 2. Create the virtual environment

```bash
python -m venv .venv
```

Activate it:

### Windows PowerShell
```bash
.\.venv\Scripts\Activate.ps1
```

### Command Prompt
```bash
.\.venv\Scripts\activate.bat
```

## 3. Install required packages

```bash
python -m pip install --upgrade pip
python -m pip install numpy pandas pillow opencv-python scikit-image scikit-learn streamlit matplotlib joblib
```

## 4. Create project structure

Create these folders and files:

```text
ML_Adaptive_Image_Compression/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_STEPS.md
├── BUILD_FROM_SCRATCH.md
│
├── src/
│   ├── __init__.py
│   ├── image_features.py
│   ├── compression_model.py
│   └── adaptive_compressor.py
│
├── models/
│   └── (created automatically after training)
│
└── data/
    └── images/
```

## 5. Add the Python source files

Create:
- `src/image_features.py`
- `src/compression_model.py`
- `src/adaptive_compressor.py`
- `src/__init__.py`

These files contain:
- image loading,
- feature extraction,
- quality metrics,
- ML prediction logic,
- adaptive resizing and compression workflow.

## 6. Add the app interface

Create `app.py` with a Streamlit UI that lets the user:
- upload an image,
- choose a target output size,
- resize and compress the image,
- view results and quality metrics,
- download the resized image and the final compressed image.

## 7. Add the training script

Create `train_model.py` to:
- load images or synthetic data,
- generate examples of resize/quality combinations,
- train the regressor,
- save the model.

## 8. Add the README and documentation

Create `README.md` and `PROJECT_STEPS.md` so the project is easy to understand and run.

## 9. Train the model

Run:

```bash
python train_model.py
```

This creates the model in the `models/` folder.

## 10. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## 11. Use the project

1. Upload an image.
2. Set a target size in KB.
3. Click the compress button.
4. Compare the original and compressed images.
5. Check SSIM, PSNR, and MSE values.

## 12. Future improvements

You can later expand the project with:
- WebP or AVIF compression
- XGBoost or deep learning models
- larger datasets
- batch processing
- comparison with traditional compression methods

## Summary

This project is built from scratch by:
- setting up Python and dependencies,
- creating image feature extraction,
- training a regression model,
- building the adaptive compression logic,
- creating the Streamlit interface,
- verifying the app runs successfully.
