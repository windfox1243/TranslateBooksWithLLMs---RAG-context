@echo off
REM ============================================
REM TranslateBook - Build Executable
REM ============================================

echo.
echo ============================================
echo TranslateBook - Building Executable
echo ============================================
echo.

REM Check if virtual environment exists
if not exist "..\..\venv" (
    echo [ERROR] Virtual environment not found
    echo Please run start.bat first to set up the environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo [1/4] Activating virtual environment...
call ..\..\venv\Scripts\activate.bat

REM Install PyInstaller if not already installed
echo [2/4] Checking PyInstaller installation...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller
)
echo [OK] PyInstaller ready

REM Clean previous builds
echo [3/4] Cleaning previous builds...
if exist "..\..\dist" rmdir /s /q ..\..\dist
if exist "..\dist" rmdir /s /q ..\dist
if exist "..\TranslateBookWithLLM" rmdir /s /q ..\TranslateBookWithLLM
echo [OK] Cleaned

REM Build executable
echo [4/4] Building TranslateBook.exe...
echo This may take 5-10 minutes...
echo.
REM Call the venv interpreter by path rather than the activated `pyinstaller`.
REM If activation above silently fails, a bare `pyinstaller` resolves to a
REM global install, and PyInstaller follows imports into whatever that
REM interpreter has -- a machine with torch installed produced a 382 MB build
REM instead of 36 MB, with no error to say why.
..\..\venv\Scripts\python.exe -m PyInstaller --clean TranslateBook.spec

if errorlevel 1 (
    echo.
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo.
echo ============================================
echo Build Complete!
echo ============================================
echo.
echo Executable location: ..\..\dist\TranslateBook.exe
echo File size:
for %%A in (..\..\dist\TranslateBook.exe) do echo %%~zA bytes (approx. %%~zA / 1048576 MB)
echo.
echo You can now distribute this single .exe file
echo Users need to have Ollama installed separately
echo.
pause
