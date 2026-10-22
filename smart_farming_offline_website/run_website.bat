@echo off
setlocal
set "WEBSITE_DIR=%~dp0"
for %%I in ("%WEBSITE_DIR%..") do set "PROJECT_ROOT=%%~fI"
set "PYTHON=%PROJECT_ROOT%\venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo ERROR: Existing project Python was not found:
  echo   %PYTHON%
  echo Make sure this website folder is inside C:\Smart_Farming_Assistant.
  pause
  exit /b 1
)
cd /d "%WEBSITE_DIR%"
set "PYTHONPATH=%PROJECT_ROOT%;%WEBSITE_DIR%;%PYTHONPATH%"
"%PYTHON%" -m pip install -r "%WEBSITE_DIR%requirements-web.txt"
if errorlevel 1 (
  echo Could not install website wrapper packages. Check your environment.
  pause
  exit /b 1
)
echo.
echo Starting Smart Farming Assistant offline website...
echo Keep this window open. Open http://127.0.0.1:8000 in your browser.
echo Press Ctrl+C here to stop the website.
echo.
"%PYTHON%" -m uvicorn web_app:app --host 127.0.0.1 --port 8000
pause
