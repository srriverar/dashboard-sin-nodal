# Despliegue **en paralelo** (v4.2 con la pestaña 🕸️ de precios nodales)

Objetivo: una **carpeta nueva** en su PC, un **repo nuevo** en GitHub y una **URL nueva** en la
web, sin tocar `C:\AppManuel` ni el dashboard que ya está publicado. Todo desde `cmd` (símbolo del
sistema), en el orden en que funciona.

> **Regla de oro:** nunca corra `git` dentro de `C:\AppManuel` para esto. Ese `.git` ya apunta al
> repo publicado; un `git push` desde allí actualizaría la app que hoy funciona. Verifique siempre
> con `git remote -v` **dentro de la carpeta nueva** antes de hacer el primer push.

---

## 0 · Qué se copia y qué no

| Se lleva a la carpeta nueva | Se queda fuera (a propósito) |
|---|---|
| `app.py` (v4.2, con el módulo nodal), `README.md`, `requirements.txt`, `Iniciar_Dashboard.bat`, `Preparar_Git.bat`, `.gitignore`, `.streamlit\config.toml`, `tools\*.py`, `data\reference\*.csv` (semilla offline, 84 KB), `certs\LEEME.txt` | `data\raw\*.csv` (25 MB de caché: no se versionan, la app los regenera) · `data\processed\` · `certs\*.pem` · `.venv\` · `__pycache__\` · `_nodal_src.py` (espejo de pruebas del asistente, **no** lo importa la app) · `uploads\01_ETL_exploracion_v1.json` (5,6 MB, solo para la auditoría del §6) |

`app.py` localiza su carpeta raíz a partir de su propia ubicación (`_detectar_base_dir()`), así que
la copia es **independiente**: descarga, cachea y escribe en `C:\AppManuel_Nodal\data\raw`, nunca en
la carpeta vieja.

---

## 1 · Crear la carpeta nueva y traer esta versión

```bat
mkdir C:\AppManuel_Nodal
```

Descomprima el ZIP de esta versión dentro de `C:\AppManuel_Nodal` (clic derecho → **Extraer
todo…** → escriba `C:\AppManuel_Nodal`). Debe quedar `C:\AppManuel_Nodal\app.py`, no
`C:\AppManuel_Nodal\AppManuel\app.py`. Si quedó anidada:

```bat
cd /d C:\AppManuel_Nodal
move AppManuel_Nodal\* .
rmdir /s /q AppManuel_Nodal
```

Copie su caché actual para arrancar sin depender de la API (opcional, pero ahorra varios minutos):

```bat
robocopy C:\AppManuel\data\raw  C:\AppManuel_Nodal\data\raw  *.csv /NJH /NJS /NFL
```

*(robocopy devuelve 1 cuando copió archivos: no es un error. Solo `>= 8` es fallo.)*

Verifique la estructura y que no haya rastro de la carpeta vieja:

```bat
cd /d C:\AppManuel_Nodal
dir /b
where /r . *.pem
```

`where /r . *.pem` no debe encontrar nada (los certificados locales no viajan).

---

## 2 · Entorno y arranque local de la versión nueva

```bat
cd /d C:\AppManuel_Nodal
python -V
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```

Abra <http://localhost:8501>, entre a **🕸️ Precios nodales** y cambie el *Escenario predefinido*
(pruebe `1 · Base sin congestión` y `9 · El Niño con los Δ% medidos en XM`). Para apagar: `Ctrl+C`.

Con el venv ya creado, los arranques siguientes son solo `Iniciar_Dashboard.bat` (doble clic).

---

## 3 · Repo **nuevo** en GitHub

1. Navegador → <https://github.com/new>
   * **Repository name:** `dashboard-sin-nodal` (o el que quiera; ese nombre define la URL de la app)
   * **Public** (obligatorio si va a usar Streamlit Community Cloud en el plan gratis)
   * **Desmarque** *Add a README* / *.gitignore* / *license* → **Create repository**
   * Copie la URL: `https://github.com/<su-usuario>/dashboard-sin-nodal.git`
2. En `cmd`:

```bat
cd /d C:\AppManuel_Nodal
Preparar_Git.bat
```

El `.bat` crea `.gitignore` y las carpetas que falten, hace `git init`, `git add -A`, muestra
`git status --short` y **le avisa** si se cuelan `data\raw\*.csv`, `.venv` o `certs\*.pem`. Cierre
la pausa y pegue:

```bat
git config --global user.name  "Manuel Fernando Fajardo"
git config --global user.email "tu@correo.com"
git commit -m "v4.2 · pestaña de precios nodales: OPF DC calibrado, 23 figuras, 12 escenarios justificados"
git branch -M main
git remote add origin https://github.com/<su-usuario>/dashboard-sin-nodal.git
git remote -v
git push -u origin main
```

`git remote -v` debe decir **el repo nuevo** en `fetch` y en `push`. Si dice
`sin-el-nino-dashboard`, está en la carpeta equivocada: pare ahí.

**Credencial:** GitHub no acepta la contraseña de la cuenta. Cree un *personal access token* en
<https://github.com/settings/tokens> → *Generate new token (classic)* → scope `repo` → y cuando
Windows pida contraseña, pegue el token. (Alternativa: GitHub Desktop, *Add local repository* →
`C:\AppManuel_Nodal` → *Publish*.)

Al terminar, en el repo debería haber ~1-2 MB y estas rutas: `app.py`, `README.md`,
`requirements.txt`, `.streamlit/config.toml`, `data/reference/*.csv`, `tools/*.py`, `*.bat`,
`.gitignore`. **No** `data/raw/*.csv`.

---

## 4 · App web **nueva** en Streamlit Community Cloud

1. <https://share.streamlit.io> → inicie sesión con la **misma** cuenta de GitHub del repo nuevo →
   **Add new app** → **Blank app**.
2. Campos:
   * **GitHub repository:** `<su-usuario>/dashboard-sin-nodal` · **Branch:** `main`
   * **App entry point:** `app.py`
   * **Python version:** `3.11` o superior
   * **Advanced settings → Secrets:** *nada*. La API de XM (`servapibi.xm.com.co`) no pide token y
     el código no lee `st.secrets`: todo lo configurable son variables de entorno con valores por
     defecto (`APP_MANUEL_*`).
   * **Manage → Config file:** use el del repo (`.streamlit/config.toml`), no duplique tema ni puerto.
3. **Deploy** (primera vez tarda 1-3 min en construir el entorno). El log debe cerrar con
   `You can now view your Streamlit app`.
4. Su link nuevo: `https://dashboard-sin-nodal.streamlit.app` ( sale del nombre del repo; el viejo
   sigue en su URL de siempre, sin relación).

**Qué va a ver en la nube:** el disco de la plataforma es de solo lectura y no existe `data/raw`,
así que en el primer arranque baja la ventana desde XM (varios minutos si la API está lenta) y
cachea en memoria. Si se atranca o falla el botón *Retry*: sidebar → **Modo de datos =
`📦 Solo referencia (demo)`** → arranca al instante con `data/reference` (las pestañas de mercado
quedan vacías con aviso explícito). Y para datos frescos: `🌐 Forzar descarga API` + **♻️ Recargar**.

Otras plataformas (Render/Fly/Railway), si algún día prefiere no depender de Community Cloud, el
comando de arranque es:

```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

---

## 5 · Actualizar la app nueva sin tocar la vieja

```bat
cd /d C:\AppManuel_Nodal
.venv\Scripts\activate.bat
git status --short
git add -A
git commit -m "…"
git push
```

Community Cloud redepliega solo al detectar el push (1-2 min). Si no lo hace:
**Settings → Manage app → Restart**; si el log muestra un error de dependencias,
*Clear cache and rebuild* antes de culpar al código.

Para la app vieja, el flujo sigue siendo el de siempre desde `C:\AppManuel`; los dos repos no se
conocen entre sí. Cuando la v4.2 esté probada en la URL nueva y quiera promoverla, se copia
`app.py` + `README.md` de una carpeta a la otra y se hace push desde `C:\AppManuel` (eso sí toca la
app publicada: hágalo a sabiendas).

---

## 6 · Verificaciones (las mismas que se usaron en el desarrollo), opcionales pero recomendadas

Desde `C:\AppManuel_Nodal`, con el venv activado:

```bat
set APPMANUEL_TEST_TIMEOUT=900
python tools\test_app.py
python tools\smoke_test.py
python tools\verificar_celdas.py
```

* `test_app.py` → `✅ APP RENDERIZA SIN EXCEPCIONES`, `excepciones del script: 0`, 70 figuras
  plotly y las 10 pestañas con `🕸️ Precios nodales`.
* `smoke_test.py` → `✅ TODO EL PIPELINE Y LAS FIGURAS OK` y el bloque nodal (23 figuras
  registradas; alguna puede salir "sin datos" si su capa está apagada en el sidebar).
  Con `set APPMANUEL_SMOKE_MODE=cache` y `set APPMANUEL_SMOKE_DIAS=31` no toca la red si usted ya
  copió `data\raw`.
* `verificar_celdas.py` → exige que el JSON del notebook esté en `C:\uploads\01_ETL_exploracion_v1.json`
  o dentro de la carpeta del proyecto; si no está, avisa `notebook no encontrado` y no hace nada
  (no es un fallo de la app).

---

## 7 · Si algo no carga

| Síntoma | Causa y arreglo |
|---|---|
| `python` no se reconoce | instale Python 3.11+ marcando *Add to PATH*, o abra el *Anaconda Prompt* |
| `git` no se reconoce | <https://git-scm.com/download/win> con opciones por defecto; **cierre y reabra** `cmd` |
| `fatal: not a git repository` | no corrió `Preparar_Git.bat` (o está en otra carpeta): `cd /d C:\AppManuel_Nodal` |
| `pathspec '.gitignore' did not match any files` | Windows lo guardó como `.gitignore.txt` → `ren .gitignore.txt .gitignore` |
| `remote: Support for password authentication was removed` | use PAT (§3), no la contraseña de GitHub |
| `master` vs `main` en el push | `git branch -M main` antes del primer push |
| Cloud: `ModuleNotFoundError: streamlit` | no instaló el venv de la carpeta nueva, o el repo no tiene `requirements.txt` |
| Cloud: log da vueltas sin datos | salida HTTPS bloqueada o XM lento: `Solo referencia` primero y `Recargar` después |
| La app nueva muestra la app vieja | el *entry point* o el repo apuntan mal: Settings → revise *Repository* y *App entry point* = `app.py` |
