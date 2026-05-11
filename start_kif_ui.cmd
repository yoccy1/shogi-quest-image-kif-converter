@echo off
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
set "UI_URL=http://127.0.0.1:8765/"
set "MODEL=models\shogi_quest_ichimonji_piece_model.pkl"

if not exist "%MODEL%" (
  echo Model file was not found:
  echo   %CD%\%MODEL%
  echo.
  echo Please download the latest repository files again.
  pause
  exit /b 1
)

where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
  set "PYTHON_CMD=py -3"
) else (
  set "PYTHON_CMD=python"
)

echo shogi-gazo-desktop image analysis UI
echo.
echo Browser will open: %UI_URL%
echo Close this window to stop the local UI server.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process '%UI_URL%'"
%PYTHON_CMD% -m shogi_gazo_desktop.cli kif-ui --host 127.0.0.1 --port 8765 --out outputs\kif_ui --model "%MODEL%"

if errorlevel 1 (
  echo.
  echo Failed to start the UI.
  echo If this is the first run, open PowerShell in this folder and run:
  echo   py -m pip install -e .
)

echo.
echo The UI server stopped. Press any key to close this window.
pause >nul
