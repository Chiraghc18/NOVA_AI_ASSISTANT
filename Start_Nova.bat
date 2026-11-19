_NOVA.bat
@echo off
title NOVA Server
cd /d "%~dp0"
 
REM Open the browser only after the server is actually up (models take a while to load)
start "" /min powershell -NoProfile -Command "while(-not (Test-NetConnection 127.0.0.1 -Port 8000 -InformationLevel Quiet)){Start-Sleep -Seconds 2}; Start-Process 'http://127.0.0.1:8000'"
 
echo Starting NOVA... loading models, please wait.
"%~dp0py310_env\Scripts\python.exe" server.py
 
echo.
echo NOVA has stopped.
pause
 