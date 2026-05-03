@echo off
REM Bypasses broken global pip.exe. Uses py -m pip (default Python first).
cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
  echo ERROR: ^`py^` launcher not found. Re-run Python installer and enable **py launcher**.
  pause
  exit /b 1
)

echo Trying default Python ^(py -m pip^)...
py -c "import sys" 2>nul
if not errorlevel 1 (
  echo Using py default
  py -m pip install --upgrade pip
  py -m pip install -r requirements.txt
  if errorlevel 1 goto fail
  goto ok
)

for %%V in (3.14 3.13 3.12 3.11 3.10) do (
  py -%%V -c "import sys" 2>nul
  if not errorlevel 1 (
    echo Using Python %%V
    py -%%V -m pip install --upgrade pip
    py -%%V -m pip install -r requirements.txt
    if errorlevel 1 goto fail
    goto ok
  )
)

where python >nul 2>&1
for /f "delims=" %%P in ('where python 2^>nul') do (
  echo %%P | findstr /i msys64 >nul
  if errorlevel 1 (
    echo Using %%P
    "%%P" -m pip install --upgrade pip
    "%%P" -m pip install -r requirements.txt
    if errorlevel 1 goto fail
    goto ok
  )
)

echo No usable Python found. Run: py -0p
echo Install Python from python.org with **py launcher** enabled.
pause
exit /b 1

:fail
echo pip install failed.
pause
exit /b 1

:ok
echo.
echo OK. Next: .\setup_windows_env.ps1 ^(optional venv^) or run live_stream_highlight_pipeline.py with this Python.
pause
exit /b 0
