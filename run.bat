@echo off
REM Check if venv exists
if not exist "venv" (
    echo Virtual environment not found. Please run installation first.
    pause
    exit /b 1
)

REM Activate virtual environment and run the app
call venv\Scripts\activate
python app.py
pause
