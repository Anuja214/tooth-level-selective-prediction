@echo off
cd /d "%~dp0"

if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
)

echo Open http://localhost:8501 after Streamlit starts.
python -m streamlit run app.py --browser.gatherUsageStats false
pause
