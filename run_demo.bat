@echo off
cd /d %~dp0
echo Open http://localhost:8501 after Streamlit starts.
python -m streamlit run app.py --browser.gatherUsageStats false
pause
