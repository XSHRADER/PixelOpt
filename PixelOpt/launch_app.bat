@echo off
cd /d "%~dp0"

if not exist ".venv" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501

if errorlevel 1 (
    echo.
    echo The app did not start successfully.
    pause
)
