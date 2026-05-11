@echo off
setlocal

cd /d "%~dp0"
set "PYTHONPATH=%CD%\src"
set "UI_URL=http://127.0.0.1:8765/"

echo shogi-gazo-desktop image analysis UI
echo.
echo Browser will open: %UI_URL%
echo Close this window to stop the local UI server.
echo.

start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process '%UI_URL%'"
python -m shogi_gazo_desktop.cli kif-ui --host 127.0.0.1 --port 8765 --out outputs\kif_ui

echo.
echo The UI server stopped. Press any key to close this window.
pause >nul
