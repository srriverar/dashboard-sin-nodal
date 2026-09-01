@echo off
rem ==========================================================================
rem  Preparar_Git.bat - deja C:\AppManuel listo para versionar y publicar.
rem  Doble clic en el archivo, o escribrelo desde cmd dentro de C:\AppManuel.
rem  NO borra ni sobreescribe nada: solo crea lo que falte.
rem
rem  Por que existe: si "git add .gitignore" responde
rem     fatal: pathspec '.gitignore' did not match any files
rem  es que el archivo no esta ahi. El Explorador de Windows no deja crear un
rem  archivo llamado ".gitignore" a mano (lo termina guardando como
rem  .gitignore.txt, y git no lo ve), asi que o no existe o existe mal nombrado.
rem ==========================================================================
setlocal EnableExtensions
cd /d "%~dp0"

echo(
echo === 1 de 4: .gitignore y carpetas ===
if exist ".gitignore.txt" if not exist ".gitignore" (
    ren ".gitignore.txt" ".gitignore"
    echo     se renombro .gitignore.txt  =^>  .gitignore
)
if not exist ".gitignore" (
    echo     no existia: se crea con las reglas correctas para esta app
    call :escribir_gitignore
) else (
    echo     ya existe: se respeta tal cual
)
if not exist "data\raw" mkdir "data\raw"
if not exist "data\processed" mkdir "data\processed"
if not exist "data\reference" mkdir "data\reference"
if not exist "certs" mkdir "certs"
if not exist "data\raw\.gitkeep" type nul > "data\raw\.gitkeep"
if not exist "data\processed\.gitkeep" type nul > "data\processed\.gitkeep"
if not exist "certs\LEEME.txt" echo Copia aqui los .pem de XM o UPME. Esta carpeta NUNCA se sube a GitHub.> "certs\LEEME.txt"

echo(
echo === 2 de 4: git init ===
where git >nul 2>nul
if errorlevel 1 goto :sin_git
if exist ".git" (
    echo     el repositorio ya estaba inicializado
) else (
    git init
)

echo(
echo === 3 de 4: git add -A ===
echo     -A toma TODO lo que existe y respeta .gitignore; por eso ya no falla
echo     cuando faltan .streamlit, data\reference o tools.
git add -A

echo(
echo === 4 de 4: revision antes del commit ===
git status --short
echo(
set TOTAL=0
for /f %%A in ('git ls-files ^| find /c /v ""') do set TOTAL=%%A
echo     archivos en el indice: %TOTAL%
git ls-files | findstr /i /r /c:"^\.venv" /c:"^venv" /c:"__pycache__" /c:"^data/raw/.*\.csv" /c:"^data/processed/.*\.csv" /c:"^certs/.*\.pem" >nul
if not errorlevel 1 (
    echo     ADVERTENCIA: entrarian crudos, el .venv o los certificados. Abre .gitignore,
    echo     agrega la regla que falte y corre de nuevo este .bat.
) else (
    echo     bien: no entran .venv, __pycache__, crudos de data\raw ni certificados.
)

echo(
echo ---------------- pegar ahora, en esta misma ventana de cmd ----------------
echo     git commit -m "Dashboard XM v4: 42 figuras, El Nino, perfiles por hora y CO2"
echo     git branch -M main
echo     git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
echo     git push -u origin main
echo(
echo Alternativa minima (solo codigo, sin los csv de referencia):
echo     git add app.py README.md requirements.txt Iniciar_Dashboard.bat Preparar_Git.bat .gitignore .streamlit
echo(
pause
goto :eof

:sin_git
echo     GIT NO ESTA INSTALADO, o no esta en el PATH.
echo     Descargalo de https://git-scm.com/download/win e instalalo con las opciones
echo     por defecto. Cierra esta ventana, abre cmd otra vez y corre este .bat.
echo     (El paso 1 ya dejo .gitignore creado, asi que solo falta el resto.)
pause
goto :eof

:escribir_gitignore
> ".gitignore" echo # datos crudos y derivados: no se versionan, la app los regenera
>>".gitignore" echo data/raw/*.csv
>>".gitignore" echo data/processed/*.csv
>>".gitignore" echo data/processed/figuras_app/
>>".gitignore" echo !data/raw/.gitkeep
>>".gitignore" echo !data/processed/.gitkeep
>>".gitignore" echo # la referencia mensual SI se versiona: es pequena y deja arrancar sin red
>>".gitignore" echo !data/reference/*.csv
>>".gitignore" echo # certificados y llaves: nunca subir
>>".gitignore" echo certs/*.pem
>>".gitignore" echo certs/*.key
>>".gitignore" echo # python
>>".gitignore" echo __pycache__/
>>".gitignore" echo *.py[cod]
>>".gitignore" echo .venv/
>>".gitignore" echo venv/
>>".gitignore" echo .ipynb_checkpoints/
>>".gitignore" echo # streamlit y sistema
>>".gitignore" echo .streamlit/secrets.toml
>>".gitignore" echo .DS_Store
>>".gitignore" echo Thumbs.db
exit /b 0
