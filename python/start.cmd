@echo off
REM Double-click launcher for OpenWritr (no console window).
setlocal
set "DIR=%~dp0"
if not exist "%DIR%..\.venv\Scripts\pythonw.exe" (
    echo Virtual env not found at %DIR%..\.venv
    echo Run: python -m venv .venv ^&^& .venv\Scripts\activate ^&^& pip install -r python\requirements.txt
    pause
    exit /b 1
)
if not exist "%LOCALAPPDATA%\OpenWritr\models\parakeet-tdt-0.6b-v3\encoder-model.int8.onnx" (
    echo Model not present. Fetching now (about 640 MB) ...
    "%DIR%..\.venv\Scripts\python.exe" "%DIR%fetch_model.py" || ( pause & exit /b 1 )
)
start "" "%DIR%..\.venv\Scripts\pythonw.exe" "%DIR%openwritr.py"
endlocal
