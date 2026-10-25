#!/bin/bash
# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run installation first."
    exit 1
fi

# Activate virtual environment and run the app
source venv/bin/activate
python3 app.py
