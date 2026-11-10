@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found.
  echo Run the setup commands in README.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" assembly_check.py --folder candidate_pack\test_images
if errorlevel 1 (
  echo Inspection failed. See the error above.
  pause
  exit /b 1
)
echo.
echo See the output folder for annotated images and reports.
pause
