@echo off
cd /d "%~dp0.."
start "" http://localhost:8765
"C:\tesis_data\.venv\Scripts\python.exe" scripts\web_clips.py
pause
