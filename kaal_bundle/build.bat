@echo off
REM KAAL air-gapped executable builder (Windows)
REM Run from repo root: kaal_bundle\build.bat
REM
REM Produces: kaal_bundle\dist\kaal.exe  (Windows standalone executable)
REM Requirements: Python 3.10+, pip, PyInstaller 6.x
REM
REM Usage:
REM   kaal_bundle\build.bat
REM
REM After build, copy kaal_bundle\dist\kaal.exe to any Windows machine and run:
REM   kaal.exe --help
REM   kaal.exe audit --model model.pt --dataset .\images\

setlocal enabledelayedexpansion

set "REPO_ROOT=%~dp0.."
set "SPEC_FILE=%REPO_ROOT%\kaal_bundle\kaal.spec"
set "DIST_DIR=%REPO_ROOT%\kaal_bundle\dist"
set "BUILD_DIR=%REPO_ROOT%\kaal_bundle\build"

echo ========================================
echo   KAAL -- Building standalone executable
echo ========================================
echo   Repo root : %REPO_ROOT%
echo   Spec file : %SPEC_FILE%
echo   Output    : %DIST_DIR%\kaal.exe
echo.

REM [1/3] Install PyInstaller if not present
pyinstaller --version >nul 2>&1
if errorlevel 1 (
    echo [1/3] Installing PyInstaller...
    pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: pip install pyinstaller failed.
        exit /b 1
    )
) else (
    echo [1/3] PyInstaller already installed.
)

REM [2/3] Run PyInstaller
echo [2/3] Running PyInstaller...
pyinstaller ^
    "%SPEC_FILE%" ^
    --distpath "%DIST_DIR%" ^
    --workpath "%BUILD_DIR%" ^
    --clean ^
    --noconfirm

if errorlevel 1 (
    echo.
    echo ERROR: PyInstaller build failed.
    echo Check the output above for errors.
    exit /b 1
)

REM [3/3] Verify
echo [3/3] Verifying...
if exist "%DIST_DIR%\kaal.exe" (
    echo.
    echo ========================================
    echo   Build successful!
    echo   Executable: %DIST_DIR%\kaal.exe
    echo ========================================
    echo.
    echo   Quick test:
    echo     %DIST_DIR%\kaal.exe --help
    echo.
    echo   Full audit:
    echo     %DIST_DIR%\kaal.exe audit --model model.pt --dataset .\images\
    echo.
) else (
    echo.
    echo ERROR: Build failed -- %DIST_DIR%\kaal.exe not found.
    echo Check the PyInstaller output above for errors.
    exit /b 1
)

endlocal
