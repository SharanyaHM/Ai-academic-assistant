@echo off
title AI Academic Assistant - SIT Tumakuru
echo ============================================================
echo   AI Academic Assistant: Intelligent PDF Analysis
echo   and Scheme-Based Answer Generation
echo   Siddaganga Institute of Technology - Tumakuru (CSE)
echo ============================================================
echo.
echo Starting Streamlit Web Application...
cd /d "%~dp0"
python -m streamlit run app.py
pause
