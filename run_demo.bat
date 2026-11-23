@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -c "import struct, sys; sys.exit(struct.calcsize('P') != 8)" >nul 2>&1
    if errorlevel 1 (call :make_venv_from_path) else (py -3.11 -m venv .venv)
    if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -c "import struct, sys; sys.exit(sys.version_info[:2] != (3,11) or struct.calcsize('P') != 8)"
if errorlevel 1 (
    echo This demo needs a Python 3.11 64-bit virtual environment.
    goto :failed
)
if not exist ".venv\its-deps-v2.ready" (
    echo Installing demo dependencies. Internet access is needed for the first run.
    ".venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0+cpu
    if errorlevel 1 goto :failed
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
    type nul > ".venv\its-deps-v2.ready"
)
".venv\Scripts\python.exe" demo.py %*
if errorlevel 1 goto :failed
exit /b 0

:make_venv_from_path
python -c "import struct, sys; sys.exit(sys.version_info[:2] != (3,11) or struct.calcsize('P') != 8)" >nul 2>&1
if errorlevel 1 (
    echo Install Python 3.11 64-bit and enable Add python.exe to PATH.
    exit /b 1
)
python -m venv .venv
exit /b %errorlevel%

:failed
echo Demo failed. See the message above.
if /i not "%~1"=="--headless" pause
exit /b 1
