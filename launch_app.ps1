$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
