@echo off
cd /d "%~dp0"
echo Starting Dify API Server...
python api_server.py
pause
