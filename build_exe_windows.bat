@echo off
setlocal
cd /d %~dp0
if not exist .venv (
  py -3.10 -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --clean --windowed --name WhisperSteg --collect-all PyQt6 --collect-all pywt app.py
echo.
echo Build complete: dist\WhisperSteg\WhisperSteg.exe
echo IMPORTANT: ffmpeg.exe and ffprobe.exe must still be available in PATH.
pause
endlocal
