@echo off
cd /d "%~dp0"
echo Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Install failed.
  pause
  exit /b 1
)
echo.
echo Starting QuadMeshTesser...
python -m app.main
