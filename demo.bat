@echo off
REM Obre el dashboard al navegador (http://localhost:8501).
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo No hi ha entorn virtual. Executa primer instal-lar.bat
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app\dashboard.py
