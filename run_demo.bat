@echo off
rem ==========================================================================
rem  POPS demo - double-click this file.
rem
rem  It needs nothing installed beforehand. In order, it looks for:
rem    1. an environment a previous run already built  (instant)
rem    2. a Python 3.12 or 3.11 already on this machine
rem    3. uv, which downloads a private Python 3.12 that touches nothing else
rem       (and installs uv itself, into your user folder, if it is missing)
rem
rem  Then it hands over to run_demo.py, which builds the environment if
rem  needed and opens the demo in your browser.
rem
rem  The first run downloads several GB and can take 10-20 minutes. Later runs
rem  start in seconds. An internet connection is required the first time.
rem ==========================================================================
setlocal
cd /d "%~dp0"
title POPS demo

echo.
set "PY="
set "UV="

rem --- 1. an environment a previous run already built -----------------------
for /d %%D in ("venv_*" "venv" ".venv") do call :try_venv "%%~fD"
if defined PY goto run

rem --- 2. a supported Python already installed ------------------------------
call :try_py 3.12
if defined PY goto found_python
call :try_py 3.11
if defined PY goto found_python

rem --- 3. uv, installing it first if this machine does not have it ----------
echo  No suitable Python found on this machine (3.11 or 3.12 is required).
echo  Using uv to fetch a private copy that will not affect anything else.
echo.

for /f "delims=" %%P in ('where uv 2^>nul') do set "UV=%%P"
if not defined UV if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"

if not defined UV (
    echo  Installing uv from https://astral.sh/uv ...
    echo.
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 ^| iex"
    if exist "%USERPROFILE%\.local\bin\uv.exe" set "UV=%USERPROFILE%\.local\bin\uv.exe"
)

if not defined UV (
    echo.
    echo  [PROBLEM] uv could not be installed automatically.
    echo.
    echo  Please install Python 3.12 from https://www.python.org/downloads/
    echo  - tick "Add python.exe to PATH" in the installer - then run this
    echo  file again.
    goto fail
)

echo  Downloading Python 3.12 ...
"%UV%" python install 3.12
for /f "delims=" %%P in ('"%UV%" python find 3.12 2^>nul') do set "PY=%%P"

if not defined PY (
    echo.
    echo  [PROBLEM] could not obtain a Python 3.12.
    echo  Check that this machine can reach the internet, then try again.
    goto fail
)

:found_python
echo  Using Python: %PY%
echo.

:run
"%PY%" run_demo.py %*
if errorlevel 1 goto fail

echo.
if "%~1"=="" echo  The demo has stopped.
goto end

:fail
echo.
echo  ======================================================================
echo    Setup did not finish. The messages above explain why.
echo    Send a photo of this window to whoever gave you this folder.
echo  ======================================================================

:end
echo.
pause
exit /b

rem --------------------------------------------------------------------------
rem  Helpers. Called, not jumped to, so a failure inside one does not abort
rem  the search - the next candidate still gets a turn.
rem --------------------------------------------------------------------------

:try_venv
if defined PY exit /b
if exist "%~1\Scripts\python.exe" set "PY=%~1\Scripts\python.exe"
exit /b

:try_py
if defined PY exit /b
for /f "delims=" %%P in ('py -%~1 -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%P"
exit /b
