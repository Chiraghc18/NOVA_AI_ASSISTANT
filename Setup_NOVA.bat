@echo off
setlocal
title NOVA First-Time Setup
cd /d "%~dp0"

REM ---------- Settings ----------
set "ENV_DIR=%~dp0py310_env"
set "PY=%ENV_DIR%\Scripts\python.exe"
set "MODEL_FILE=%~dp0nova_model.txt"

echo ==========================================
echo   NOVA first-time setup
echo ==========================================
echo.

REM ---------- 1. Python virtual environment ----------
if exist "%PY%" goto env_ok

echo [1/4] Creating Python 3.10 environment...
where py >nul 2>&1
if errorlevel 1 goto no_python
py -3.10 -m venv "%ENV_DIR%"
if errorlevel 1 goto no_python
goto env_ok

:no_python
echo Python 3.10 was not found.
where winget >nul 2>&1
if errorlevel 1 goto manual_python
echo Installing Python 3.10 with winget...
winget install -e --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
echo.
echo Python was installed. Close this window and run Setup_NOVA.bat again.
pause
exit /b 1

:manual_python
echo Please install Python 3.10 from https://www.python.org/downloads/
echo Then run Setup_NOVA.bat again.
pause
exit /b 1

REM ---------- 2. Python packages ----------
:env_ok
echo [2/4] Installing packages. This can take a long time...
"%PY%" -m pip install --upgrade pip
if not exist requirements.txt goto skip_requirements
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto pip_failed
:skip_requirements
"%PY%" -m pip install fastapi "uvicorn[standard]"
if errorlevel 1 goto pip_failed
goto packages_ok

:pip_failed
echo.
echo Package installation failed. Read the error above, fix it, and run again.
pause
exit /b 1

REM ---------- 3. Ollama + model ----------
:packages_ok
echo [3/4] Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 goto no_ollama

ollama list >nul 2>&1
if not errorlevel 1 goto ollama_running
echo Starting Ollama...
start "" /min ollama serve
timeout /t 6 /nobreak >nul

:ollama_running
echo.
echo Which AI model should NOVA use?
echo Bigger models answer better, but need more RAM and respond slower.
echo Sizes and RAM needs below are rough guides.
echo.
echo   1 - TinyLlama      637 MB   fastest, weakest answers, runs on almost any PC
echo   2 - Llama 3.2 3B   2.0 GB   good balance, best pick for most laptops [recommended]
echo   3 - Phi-3 Mini     2.2 GB   small and good at reasoning
echo   4 - Mistral        4.4 GB   strong answers, wants about 8 GB RAM or more
echo   5 - Llama 3        4.7 GB   best quality here, wants 16 GB RAM or a GPU
echo   6 - Qwen 3.5 4B   3.3 GB   newer, but may think before answering, so test it first
echo   7 - Gemma 4 E4B   7 GB+    newer, made for laptops
echo   8 - Gemma 4 12B   7.7 GB   newest and best for conversation, wants 16 GB RAM
echo   9 - Skip, I already have a model
echo.
choice /c 123456789 /n /m "Type a number from 1 to 9: "
if errorlevel 9 goto model_ok
set "MODEL=gemma4:12b"
if errorlevel 8 goto pick_done
set "MODEL=gemma4:e4b"
if errorlevel 7 goto pick_done
set "MODEL=qwen3.5:4b"
if errorlevel 6 goto pick_done
set "MODEL=llama3:latest"
if errorlevel 5 goto pick_done
set "MODEL=mistral:latest"
if errorlevel 4 goto pick_done
set "MODEL=phi3:mini"
if errorlevel 3 goto pick_done
set "MODEL=llama3.2:3b"
if errorlevel 2 goto pick_done
set "MODEL=tinyllama:latest"

:pick_done
echo Selected: %MODEL%
>"%MODEL_FILE%" echo %MODEL%
ollama list | findstr /i /c:"%MODEL%" >nul
if not errorlevel 1 goto model_ok
echo Downloading model %MODEL% ...
ollama pull %MODEL%
if errorlevel 1 goto pull_failed
goto model_ok

:no_ollama
echo Ollama was not found.
where winget >nul 2>&1
if errorlevel 1 goto manual_ollama
echo Installing Ollama with winget...
winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
echo.
echo Ollama was installed. Close this window and run Setup_NOVA.bat again.
pause
exit /b 1

:manual_ollama
echo Please install Ollama from https://ollama.com/download
echo Then run Setup_NOVA.bat again.
pause
exit /b 1

:pull_failed
echo Could not download %MODEL%. Check your internet connection and run again.
pause
exit /b 1

REM ---------- 4. ffmpeg (needed by Whisper) ----------
:model_ok
echo [4/4] Checking ffmpeg...
where ffmpeg >nul 2>&1
if not errorlevel 1 goto ffmpeg_ok
where winget >nul 2>&1
if errorlevel 1 goto manual_ffmpeg
echo Installing ffmpeg with winget...
winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements
echo.
echo ffmpeg was installed. Close this window and run Setup_NOVA.bat again.
pause
exit /b 1

:manual_ffmpeg
echo Please install ffmpeg from https://ffmpeg.org/download.html and add it to PATH.
echo Then run Setup_NOVA.bat again.
pause
exit /b 1

:ffmpeg_ok
echo.
echo ==========================================
echo   Setup complete.
echo   Double-click Start_NOVA.bat to run NOVA.
echo ==========================================
pause