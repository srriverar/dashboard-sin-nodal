@echo off
rem Arranca el dashboard 'Sin el Nino' (XM / Colombia).
rem Doble clic en el Explorador, o escribelo desde cmd dentro de C:\AppManuel.
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
where streamlit >nul 2>nul
if errorlevel 1 (
  echo No se encuentra Streamlit en este entorno.  Instalalo con:
  echo     pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)
streamlit run app.py
pause
