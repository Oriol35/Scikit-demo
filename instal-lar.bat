@echo off
REM Crea l'entorn virtual i instal·la les dependencies (cal Python 3.11 o superior i internet).
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  echo L'entorn .venv ja existeix.
) else (
  py -3 -m venv .venv 2>nul || python -m venv .venv
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
echo.
echo Instal·lacio completada. Executa demo.bat per obrir el dashboard.
pause
