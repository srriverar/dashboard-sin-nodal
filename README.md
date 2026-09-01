# 🌊 Agua, precio y El Niño — el SIN colombiano bajo sequía

> Tablero operativo del mercado eléctrico de Colombia: extracción XM (Bienda) + ETL + análisis
> ENSO con el ONI de NOAA/CPC, ejecutando la lógica de las 212 celdas del notebook
> `01_ETL_exploracion_v1` (v1 · v2 · v3 · v4) dentro del navegador.

Aplicación **Streamlit autocontenida** que ejecuta, en vivo y desde cero, todo el pipeline del notebook
`01_ETL_exploracion_v1.ipynb` (212 celdas, v1 → v4) sobre la API pública de **XM Bienda**
(`https://servapibi.xm.com.co`) y el índice **ONI de NOAA/CPC**: extracción → ETL → modelo analítico →
44 figuras Plotly en 9 pestañas, **cada una con su leyenda** (qué muestra, qué decisión permite y qué significa bajo
El Niño).

| Capa | Contenido | Autor del notebook |
|---|---|---|
| v1 | Catálogos maestros (`ListadoRecursos/Agentes/Rios`), `dim_plantas`, EDA por tecnología | Manuel Fernando Fajardo Rodríguez |
| v2 | `fact_*`, `gen_enriquecida`, `sistema_h`, `resumen_diario`, balance/precio/perfiles/matrices | Manuel Fernando Fajardo Rodríguez |
| v3 | CEN, margen de reserva, factor de planta, emisiones/intensidad, **análisis ENSO** (ONI, eventos, lag, boxplots) | Prof. Sergio Rivera, PhD |
| v4 | Indisponibilidad, reservas, costo marginal y ENSO económico | Prof. Sergio Rivera, PhD |

## 1 · Estructura del proyecto

```
C:\AppManuel\
├── app.py                    ← todo el tablero (datos + ETL + figuras + UI)
├── Iniciar_Dashboard.bat     ← doble clic para arrancar en Windows (activa .venv + streamlit run)
├── Preparar_Git.bat          ← doble clic para publicar: crea .gitignore si falta, git init y git add -A
├── requirements.txt
├── .gitignore                ← que no suban .venv, data\raw\*.csv, __pycache__ ni certs\*.pem
├── .streamlit\config.toml    ← tema y puerto
├── certs\                    ← (opcional) upme_bundle.pem para TLS intermediado / UPME
├── data\
│   ├── raw\                  ← caché de las descargas de XM (CSV por métrica y ventana)
│   ├── processed\            ← exportaciones y figuras
│   └── reference\            ← serie ONI, Niño1+2/3.4, nino_mensual y v4_mensual (versionadas)
├── src\                      ← (opcional) módulos propios; el notebook los tenía aquí
└── tools\
    ├── build_seeds.py        ← regenera data\reference cuando se desactualice (1 comando)
    ├── smoke_test.py         ← ejecuta el pipeline + las 44 figuras y reporta fallos
    ├── verificar_celdas.py   ← **verificación celda por celda** del notebook contra app.py
    ├── mapa_celdas.py        ← reconstruye el mapa figura↔celda desde el JSON del notebook
    ├── test_app.py           ← renderiza la app con streamlit.testing (AppTest)
    ├── probe_units.py / probe_emisiones.py / probe_cen.py   ← chequeos de unidades
```

No hay que crear nada a mano: si `data\*` no existe, `app.py` lo crea al arrancar
(en Streamlit Cloud, donde el disco es de solo lectura, la app degrada con aviso y usa la memoria).

## 2 · Arranque local (Windows, `cmd`)

```bat
cd /d C:\AppManuel
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

> **⚠️ No uses `python app.py`.** Ese comando ejecuta el archivo "en crudo" (*bare mode*): no hay
> servidor ni sesión, los widgets no responden y la consola se llena de avisos
> `missing ScriptRunContext`. El comando de Streamlit es **`streamlit run app.py`** (la app ahora
> te lo recuerda y se detiene sola si te equivocas). Si quieres doble clic en vez de `cmd`,
> usa el archivo **`Iniciar_Dashboard.bat`** de esta carpeta: activa el `.venv` y ejecuta lo correcto.

```bat
start "" http://localhost:8501
```

Abre <http://localhost:8501>. Primera carga: ~20-60 s (14 consultas de la ventana + la siembra
del Niño si `data/reference` está vacío). Después, la caché de `data/raw` hace que arranque en segundos.

**No necesitas `data/reference` para arrancar.** La app funciona con solo `app.py`: si faltan las
series de referencia, siembra los últimos 4 meses de `PorcApor`, `PorcVoluUtilDiar` y
`PPPrecBolsNaci` desde la API y los deja cacheados. Con `tools\build_seeds.py nino` (o `all`)
obtienes la historia completa 2015→ y el arranque es instantáneo.

### Verificar antes de publicar

```bat
python tools\smoke_test.py                 :: pipeline + las 44 figuras (usa la API real)
set APPMANUEL_SMOKE_MODE=cache&& python tools\smoke_test.py   :: igual, sin red (solo caché)
python tools\test_app.py                   :: renderiza la app con AppTest (sin navegador)
python tools\verificar_celdas.py            :: 212 celdas del notebook clasificadas una por una
python tools\verificar_celdas.py --literal   :: re-emite CELDAS_NB / COBERTURA_NB / CALC_APP_NB
```

### Regenerar las series de referencia (se recomienda 1 vez al mes)

```bat
python tools\build_seeds.py all
```

Tarda unos minutos (140 meses × 3 métricas, en paralelo). Deja `data/reference\*.csv` al día:
`oni_noaa.csv` (NOAA), `nino34_mensual.csv` (Niño1+2/3/4/3.4), `nino_mensual.csv` (2015→) y
`v4_mensual.csv` (demanda, DemaMax, CEN y costo marginal 2016→).

### Variables de entorno útiles

| Variable | Efecto |
|---|---|
| `APP_MANUEL_HOME` | raíz de datos alternativa (por defecto: carpeta de `app.py`, o `C:\AppManuel`) |
| `APPMANUEL_NO_MAIN=1` | importa `app.py` como librería sin levantar la UI (para pruebas) |
| `APPMANUEL_BARE=1` | permite `python app.py` como *refresco de datos* sin interfaz (sin UI, sin widgets) |
| `APP_MANUEL_NINO_MESES` | meses que se bajan para sembrar la serie del Niño si falta la semilla (1-24, por defecto 4) |
| `APPMANUEL_FORCE_DEMO=1` | fuerza el modo sin red (usa solo caché/referencia) |
| `APP_MANUEL_OBREROS` | hilos de descarga de XM (1 = serie; 5 por defecto; 8 ya satura la API) |
| `APPMANUEL_SIN_HILO=1` | corre el pipeline en el hilo de la UI (sin tarjeta de progreso interactiva) |
| `APP_MANUEL_LIMITE_CARGA` | segundos antes de avisar que la carga lleva demasiado tiempo (600) |
| `SSL_CERT_FILE` | bundle CA si tu red hace inspección TLS (corp/VPN) |
| `SEED_V4_START` / `SEED_HISTORIC_START` / `SEED_WORKERS` | ventana y paralelismo de `tools\build_seeds.py` |

### 2b · Rendimiento: qué se carga, en qué etapa y por qué no se congela

La primera pasada del tablero ejecuta el pipeline completo en **4 etapas visibles** (la tarjeta de
progreso dice en cuál vas, qué está leyendo y cuánto va):

| Etapa | Qué hace | Tipo de dato que lee | Costo medido |
|---|---|---|---|
| **V1** | 14-19 consultas al catálogo y a las métricas, **en paralelo** (5 hilos) | API REST de XM (JSON) + CSV locales de `data/raw` | 20,8 s en serie → **6,7 s** en paralelo |
| **V2** | ETL: `wide→long`, unidades (fracción→%), 24 h por día, duplicados, NaN | memoria (pandas) | 1-2 s |
| **V3** | cruce con `dim_plantas`, resúmenes diarios y linaje | CSV locales de `ListadoRecursos`/`ListadoAgentes`/`ListadoRios` | < 1 s |
| **V4** | ONI + fases + eventos + rezagos + CEN + emisiones + capa v4 | CSV público de NOAA/CPC + semillas `data/reference` + API de XM | 1-3 s |
| — | 44 figuras de 9 pestañas | solo memoria | 4,2 s la 1.ª vez → **0,0 s** después |

Claves de la implementación:

* **Dice qué celda del notebook se está ejecutando.** Cada hito del progreso lleva el ancla
  `[nb celdas 7-20]` / `[nb celda 61]`, la tarjeta muestra el mapa de la etapa en curso
  (`V1 celdas 7-20 · V2 21-119 · V3 120-174 · V4 175-212`) y cada figura del tablero repite su celda en
  el encabezado del desplegable y en la leyenda. El número es el índice absoluto de celda dentro de
  `cells` del JSON (0-based, contando markdown y código); el `In [n]` de Jupyter no coincide porque
  solo numera las celdas de código.
* **No bloquea el hilo de la UI.** `cargar_contexto_con_progreso()` lanza el pipeline en un
  `ThreadPoolExecutor`, pinta la tarjeta de progreso y se auto-refresca con `st.rerun()`, así que el
  estado *Running* del navegador no queda pegado, los widgets responden y el spinner del navegador
  anima. Al terminar: `st.toast` + un bloque verde con el tiempo total y el número de hitos.
* **Dos cachés.** `st.cache_data(ttl=6 h, max_entries=4)` para el `ctx` (llave = días + métricas +
  modo + fecha) y `figura_cacheada()` para cada figura, indexada por la *firma* de los datos
  (`ctx["_firma"]`); cambiar la ventana o las métricas cambia la firma, así que nunca se sirve una
  figura de otros datos. En disco, `data/raw/{prefijo}_{inicio}_{fin}.csv` sobrevive reinicios.
* **No hay `while True` ni esperas infinitas:** `requests` con `timeout=180 s`, 4 reintentos con
  pausa, respaldo serial si el paralelismo falla, y corte del aviso a los `APP_MANUEL_LIMITE_CARGA`
  segundos (600 por defecto).
* No hay base de datos SQL: únicamente **API REST** (XM y NOAA) y **CSV** (caché `data/raw` y
  semillas `data/reference`).

### Si algo falla, mira esta tabla

| Lo que ves en la consola | Qué significa / qué hacer |
|---|---|
| `missing ScriptRunContext` (muchos) y `Warning: to view this Streamlit app on a browser` | Ejecutaste `python app.py`. Usa `streamlit run app.py`. |
| `No se pudo completar la extracción: HTTPError 503` | La API de XM está saturada. Reintenta en 1-2 min o pon **💾 Solo caché**. |
| `Sin semilla offline` / pestañas vacías en modo 📦 | Faltan los CSV de `data/reference`. Ejecuta `python tools\build_seeds.py all` o usa **🔄 Auto**. |
| `SSLCertVerificationError` / `unable to get local issuer certificate` | Tu red inspecciona TLS: coloca `certs\upme_bundle.pem` o define `SSL_CERT_FILE`. |
| `'python' no se reconoce` | Usa `py` en lugar de `python`, o revisa que el `.venv` esté activado (debe verse `(.venv)` en el prompt). |
| `NameError` / `AttributeError` en el arranque | Copiaste un `app.py` viejo: vuelve a copiar el actual (estos tres bugs están corregidos aquí). |
| `fatal: pathspec '.gitignore' did not match any files` | `.gitignore` no existe en `C:\AppManuel` (o el Explorador lo guardó como `.gitignore.txt`, que es lo típico: Windows no deja crear archivos que empiecen por punto). Doble clic en **`Preparar_Git.bat`**, que lo crea con las reglas correctas, renombra el `.gitignore.txt` si existía y hace `git add -A`. Manual: `ren .gitignore.txt .gitignore`. |
| `fatal: pathspec 'data/reference' did not match any files` (o `.streamlit`, `tools`) | Esa carpeta no la creaste en tu equipo. No hace falta: usa `git add -A` respeta el `.gitignore` y nunca reclama por rutas inexistentes. |
| `Please tell me who you are` al hacer commit | Falta tu firma de git: `git config --global user.name "Tu Nombre"` y `git config --global user.email "tu@correo.com"`. |
| Arranque lento la primera vez | Normal: 14 consultas + siembra ENSO. La segunda vez usa `data/raw` y tarda segundos. |

### 2c · Completitud: por qué la serie corta antes de hoy (y por qué el balance ya no se infla)

XM **no cierra el día el mismo día**: publica el día *D* en *D+1/D+2* y las primeras entregas traen
solo una fracción de las horas y de los agentes. Medido sobre la API real en la ventana
10→31 de agosto de 2026:

| Día | Suma de `DemaReal_Sistema` | Mediana de la ventana | Ratio | `Gene_Sistema` |
|---|---|---|---|---|
| 24-ago | 254,1 GWh | 252,3 GWh | 1,01 | completo |
| 25-ago | 264,7 GWh | 252,3 GWh | 1,05 | completo |
| 26-ago | 263,3 GWh | 252,3 GWh | 1,04 | completo |
| **27-ago** | **60,5 GWh** | 252,3 GWh | **0,24** | 269,9 GWh (¡completo!) |
| **28-ago** | **62,5 GWh** | 252,3 GWh | **0,25** | 271,3 GWh (¡completo!) |

Como las 24 columnas `Values_Hour01…24` vienen **no nulas**, un chequeo de «¿faltan horas?» no detecta
nada: hay que auditar la **magnitud**. `podar_dias_provisionales()` hace justo eso (umbral 60 % de la
mediana del propio feed, configurable con `APP_MANUEL_UMBRAL_DIA`) y **excluye esos días de todos los
marcos a la vez**. Sin ese paso, la app mostraba «la demanda cayó 75 %» y un
`Balance_GWh = Generación − Demanda` de **+209 GWh** en los dos últimos días: un artefacto de
publicación, no del sistema. Con él, el balance queda en 3-4 GWh/día y el eje de todas las figuras
termina en el **último día con dato cerrado** (`ctx["fin_efectiva"]`, visible en la métrica «Ventana analizada»
del encabezado, en el aviso amarillo de Resumen y en la tabla *Auditoría de completitud* de 📚 Datos).

Los feeds horarios llegan aún más tarde que el resto (en la misma consulta pedida hasta el 30-ago,
`DemaReal`/`Gene_Sistema`/`PrecBolsNaci` terminaban en 28-ago y `PorcVoluUtilDiar` sí llegaba a 30-ago):
**cada métrica tiene su propio rezago**, por eso la app usa un único «último día completo» común para no
mezclar días cerrados con días abiertos.

* Para inspeccionar el crudo tal como lo publica XM: barra lateral → **🩹 Incluir días provisionales de
  XM** (aparece un recuadro rojo de advertencia y las cifras vuelven a verse como arriba).
* Filtros retirados a pedido del autor: «Tecnología (filtra generación y emisiones)», «Agente /
  empresa» y «Tipo de recurso» —siempre estaban en *todas* y solo generaban confusión. El filtrado fino
  sigue en el **explorador de planta** y en el **explorador de recursos y cuencas**.
* Tampoco se despliegan ya las dos vistas redundantes de 🔥 CO₂ y economía ENSO (*dispersión ONI +
  rezagos 0-12 m* y el gráfico de *Δ% por episodio*): la tabla de impacto por episodio
  (celda 203 del notebook) se sigue calculando y se ve como **tabla**, con su descarga en CSV.

### 2d · Verificación celda por celda

```bat
python tools\verificar_celdas.py            :: 212 celdas clasificadas; sale 1 si algo queda sin espejo
```

El script clasifica cada celda del JSON como `figura` (63), `cálculo portado` (72),
`salida en notebook` (3), `config/imports` (1) o `documentación` (73 markdown) y exige que (i) cada
clave de `FIGURES` tenga ancla de celda, (ii) ninguna `graficar_*` del notebook quede huérfana y
(iii) ningún ancla apunte a una celda que no sea de código. Su salida **está empotrada en la app**:
📚 Datos → *6 · Cobertura del notebook, celda por celda* (tabla filtrable + CSV), junto a la
*Validación de las tablas analíticas* que portea la celda 61 (`validar_tablas_analiticas`: rango, nulos
y duplicados de grano por tabla, sin ocultar observaciones).

> **Repositorio de referencia (notebook v1-v4 y esta app):**
> <https://github.com/srriverar/sin-el-nino-dashboard>. Ahí está el `01_ETL_exploracion_v1.ipynb`
> completo (212 celdas) que este tablero reproduce celda por celda; cada figura de la app cita la
> celda que ejecuta (en el encabezado del desplegable: `· v3 · notebook celda 143`).

## 3 · Publicar en GitHub (`cmd`)

**La vía corta:** doble clic en **`Preparar_Git.bat`**. Crea `.gitignore` (y las carpetas
`data\*` con sus `.gitkeep`) si faltan, hace `git init`, `git add -A`, muestra qué va a subir y
avisa si se cuelan `.venv`, `__pycache__`, los crudos de `data/raw` o los `certs\*.pem`.
Después solo pegas el commit y el push.

**La vía a mano**, con el orden que sí funciona en `cmd`:

```bat
cd /d C:\AppManuel
git config --global user.name  "Manuel Fernando Fajardo"      :: solo la primera vez
git config --global user.email "tu@correo.com"                 :: solo la primera vez
notepad .gitignore                                             :: crea/verifica el archivo (guardar como "Todos los archivos")
git init
git add -A                                                     :: -A toma lo que existe y respeta .gitignore
git status --short                                             :: NO deben aparecer data\raw\*.csv, .venv ni certs\*.pem
git commit -m "Tablero SIN Colombia v4.2: XM + ETL + ENSO, 44 figuras, auditoría de días XM"
git branch -M main
git remote add origin https://github.com/<tu-usuario>/sin-el-nino-dashboard.git
git push -u origin main
```

No escribas `git add .gitignore app.py ... .streamlit data/reference tools`: esa lista exige que
**todas** esas rutas existan y una sola carpeta ausente aborta el comando (`pathspec ... did not
match any files`). Con `git add -A` + `.gitignore` subes exactamente lo mismo y no falla. Y ojo con
el Explorador de Windows: si intentas crear `.gitignore` con clic derecho, termina como
`.gitignore.txt` y git lo ignora; renómbralo desde cmd con `ren .gitignore.txt .gitignore`.

Si prefieres subir solo el código (sin los CSV de `data/reference`, que son la semilla offline):

```bat
git add app.py README.md requirements.txt Iniciar_Dashboard.bat Preparar_Git.bat .gitignore .streamlit
```

Si aún no existe el repo en GitHub: **New repository** → nombre `sin-el-nino-dashboard`, **privado o
público** (público si vas a usar Streamlit Cloud con el plan gratis), **sin README** → copia la URL y
úsala en `git remote add`. Para una app pública, deja el repo público para
que Streamlit Cloud pueda leerlo.

## 4 · Desplegar en la web (Streamlit Cloud, gratis)

1. Entra a <https://share.streamlit.io> → **Add new app** → **Blank app** → conecta tu repo y la rama `main`.
2. **App entry point**: `app.py`. **Python version**: `3.11` o superior.
3. En *Advanced settings* deja que instale `requirements.txt` (no añadas nada más).
4. **Deploy**. Log → debe terminar en `You can now view your Streamlit app`. La app detecta que el
   disco es de solo lectura: funciona con lo que hay en `data/reference` y, si la plataforma permite
   salida a internet (Sí en Community Cloud), descarga de XM en cada arranque y cachea en memoria.
5. Si prefieres otro proveedor (Render/Fly), el comando de arranque es:
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

### Si la app no carga datos en la nube

* La API de XM **no** requiere token, pero sí salida HTTPS saliente a `servapibi.xm.com.co:443`.
* Cambia el selector lateral **Modo de datos** a `Solo referencia` para una demo instantánea con la
  serie ENSO versionada en el repo (las pestañas de mercado quedarán vacías, con aviso explícito).
* Para forzar datos frescos: `Forzar descarga API` + botón **♻️ Recargar**.

## 4b · Mapa de pestañas (v4 del tablero)

| Pestaña | Contenido |
|---|---|
| 🎯 Resumen ejecutivo | balance diario, perfil horario, precio, embalses/aportes, CEN y margen, indisponibilidad, tabla diaria; **al final** el semáforo El Niño y los hallazgos automáticos |
| ⚡ Generación | matriz tecnología × hora (totales), correlaciones, perfiles, solares y factores de planta |
| 💧 Hidrología | embalses por zona y aportes contra la media histórica |
| 🌊 El Niño | KPIs del evento, ONI (últimos 7 años por defecto; 1950→ con el 🔭 del sidebar), ventana diaria por fase, ONI vs aportes/precio y Niño1+2 vs Niño3.4 |
| 💰 Costos y emisiones | escasez vs aportes, distribución del costo, heatmap precio, emisiones y Pareto |
| 🛠️ Disponibilidad (v4) | despacho térmico, matriz disponibilidad × precio y **El Niño en dinero** (funciona sin `v4_mensual.csv`: cae a la base diaria de la ventana) |
| 🕐 Perfiles × hora | cálculos del notebook uno por uno: heatmap 08 (promedio diario), pequeños múltiplos 08b con pico anotado, precio día × hora (09), costo marginal v4 (02/03), panel ENSO de 4 filas (v3 c21), climatología por fase con ★ (v3 c24) y el explorador de **ámbito operativo/geográfico** (08d/08e/08g/08h) |
| 🔥 CO₂ y economía ENSO | bloque de emisiones v3 (celdas 13-16: emisiones por combustible e intensidad, perfil horario vs solar, Pareto de emisoras, correlación extendida) y bloque v4 (celdas 14-17: impacto ENSO con rezagos 0-12 m, Δ% por episodio, tabla de impacto y hallazgos v4 en tabla legible) |
| 🕸️ Precios nodales | **módulo académico (v4.2), fuera del notebook y con coautor propio**: OPF DC calibrado con la ventana cargada, topología y regiones explicadas, LMP por nodo, reglas de liquidación, refuerzos hipotéticos de la red con su OPF por candidato, capa estocástica de dos etapas, GARCH/VaR, almacenamiento, coberturas, poder de mercado y **aprendizaje por refuerzo (Q-learning multiagente)** — 26 figuras en 11 grupos y 12 escenarios justificados (ver §4c) |
| 📚 Datos y bitácora | linaje de la extracción, tablas descargables, bitácora, exploradores (planta, río, catálogos) y JSON del contexto |

A pedido del autor del tablero se retiraron de la pestaña 🌊 El Niño las subpestañas *Resagos ONI →
variable*, *Caja-bigotes por fase* y *ONI vs embalses*; los rezagos y la climatología por fase se
conservan (mejor dibujados y con la ventana actual marcada) en las dos pestañas nuevas.
`data/reference/mapa_operativo.csv` (opcional, formato en `data/reference/LEEME_mapa_operativo.txt`)
activa los niveles **Área/Subárea operativa (SIMEM)** del explorador de ámbito.

**Política de ejes X.** Todo gráfico operativo se recorta a la ventana evaluada (y los índices
ENSO a los últimos 4-7 años) para que el presente se vea sin hacer zoom; el histórico completo
sigue estando a un clic del **🔭 «Ejes con todo el histórico»** de la barra lateral. El recorte es
de *vista*, no de datos: el zoom del lector puede abrirse hasta 1950.

**Compatibilidad de Streamlit.** `app.py` traduce solo `use_container_width=True` a
`width="stretch"` cuando la versión instalada ya usa la API nueva (1.49+) y deja el parámetro
antiguo en 1.32-1.48, de modo que el mismo archivo corre sin avisos de deprecación ni en
Streamlit 1.x ni cuando llegue la 2.x. Los `st.plotly_chart` llevan `key` única para que dos
figuras idénticas (p. ej. el heatmap «Todo el SIN» repetida en dos pestañas) no corten la página
con `StreamlitDuplicateElementId`.

## 4c · Pestaña 🕸️ Precios nodales (módulo académico, v4.2)

**No está en el notebook v1-v4.** Es el anexo académico del tablero, escrito por
**Libardo Acero García** (ingeniero eléctrico, docente y especialista en regulación energética, asesor de la CREG,
dos maestrías y Magíster en Finanzas, doctorando UNAL) como desarrollo de precios nodales dentro de
su trabajo de grado doctoral; la base de datos y las pestañas previas siguen siendo de Manuel
Fajardo (v1-v2) y del profesor Sergio Rivera (v3-v4).

El módulo implementa los **tres objetivos** que la propia pestaña enuncia al abrir — no se citan
documentos, secciones ni numeración de ecuaciones de ningún trabajo previo; cada figura responde a
un objetivo y declara de dónde sale cada número:

* **a** · cuánto cambian precio, congestión y costo del sistema cuando la señal pasa de uninodal a
  nodal (congestión, pérdidas, estocástica de solar/eólica/demanda y desconexión voluntaria).
* **b** · cómo responde esa señal a la configuración del sistema: topología, refuerzos de red,
  almacenamiento, flujos óptimos, despacho simplificado y reglas de formación de precio.
* **c** · cómo se comportan los LMP resultantes: dispersión y riesgo, cobertura ante volatilidad con
  alta penetración variable y comportamientos estratégicos (incluido el aprendizaje por refuerzo).

Todo corre sobre la **misma ventana ya cargada** por las demás pestañas (`ctx` común: no vuelve a
llamar a la API). El módulo vive al final de `app.py`, sección `# 10 · MÓDULO «SIN NODAL» (v4.2)`, y
se verifica solo: `tools/verificar_celdas.py` exige paridad exacta entre `FIGURES_NODAL` y
`NODAL_OBJETIVOS` (26 figuras, cada una con su etiqueta `objetivo a/b/c` y su fuente), revisa que
ningún texto del módulo cite documentos externos ni su numeración de secciones y ecuaciones,
y comprueba que existan los bloques exigidos en la revisión (`AUTOR_NODAL`, `_bloque_regiones_nodales`, `NODAL_REFUERZO_TEXTO`,
`NODAL_COMBINACIONES`, `despacho_estocastico`, `emparejar_cv`).

**Regiones y topología, explicadas en la propia pestaña** (`_bloque_regiones_nodales` +
`fig_nodal_topo`). Tres reglas mandan: (i) la **unidad espacial es la subregión XM**, porque es el
nivel al que `servapibi` publica demanda y generación; (ii) cada nodo se arma por **toponimia** de
las plantas (`KEYWORDS_ZONA`: se revisa el nombre de planta, de municipio y de unidad de generación
antes de asignar el nodo, y la pestaña reporta cuántas plantas quedaron sin nodo como control); (iii) cada nodo lleva el **peso calibrado** de `PESO_DEMANDA_NODAL` (CEN 26 %, NOR 24 %, CAR 21 %, OCC
12 %, SUR 8 %, GUJ 4,5 %, ORI 4,5 %), porque XM no publica demanda por subregión. Con eso, **10
nodos** es el punto donde la malla aún distingue los patrones reales (Caribe, Guajira, Nordeste,
Bogotá-Cundinamarca, OCC-SUR, Oriente) sin volver inmanejable el OPF: `subregion` baja a 7 nodos y
`fina` sube a 12 (separando Bogotá, Norte de Santander y los llanos); la figura de topología muestra
las 18 trazas (una por rama y su contraria), los círculos separados y el % de uso visible.

**Refuerzos hipotéticos de la red.** `refuerzos_candidatos()` genera el **conjunto** (5-6 obras) desde
la propia congestión medida — no un catálogo fijo: toma las ramas con mayor uso relativo, las ordena
por aporte a la congestión y propone dos intensidades (0,35 y 0,75 de la capacidad existente) más una
obra nueva `NUEVA` que crea rama con `x_pu = 0,03` y **recalcula la matriz PTDF** (`red_con_refuerzo`).
Cada candidato se liquida con el mismo OPF y devuelve congestión, λ medio, CO₂ y ahorro por día, con
payback sobre `0,32 M COP/MW` (orden de magnitud declarado, **no** cifra UPME/ISA/PER); si el ahorro
no pasa de 1,0 M COP/año el payback se muestra como no aplicable. El bloque cierra con el **paquete
completo** para leer los rendimientos decrecientes, y `fig_nodal_refuerzos` lo dibuja en abanico.

**Capa estocástica de dos etapas.** `gen_escenarios` simula (log-normal solar, Weibull eólica gamma de
demanda, ρ entre nodos y ρ cruce sol-viento), `emparejar_cv` corrige la dispersión con `x' = μ + α(x−μ)`
para que el CV eólico simulado iguale al medido (tope `α ≤ 12`, con panel que muestra por qué nodo y
por qué escena se modifica cada variable), `reducir_escenarios` cierra con k-medias, y
`despacho_estocastico` liquida **cada escenario con el OPF** (no con una fórmula): da Wait-and-See,
Here-and-Now y VSS, además de la banda p05/p50/p95 del precio horario y `p(escasez)`. La pestaña
permite apagar la segunda etapa para ver solo las bandas (rápido) o encenderla para el VSS real.

**Aprendizaje por refuerzo, en el centro de la pestaña.** `NODAL_REFUERZO_TEXTO` lo enuncia arriba:
`fig_nodal_ql` (Q-learning multiagente con λ endógeno: los agentes aprenden a ofertar contra el mismo
ascenso dual que usa el OPF, y el déficit se mide contra el cierre exacto del balance) y `fig_nodal_ap`
(mark-up aprendido por regla de liquidación: la señal nodal castiga el recargo porque el agente ya no
puede ponerlo a todo el sistema). Es el lente estratégico del objetivo *c*, no un adorno: los
controles del episodio, ε y la resolución están en la propia figura.

**Tres combinaciones de impacto** (`NODAL_COMBINACIONES`, bajo `fig_nodal_combinaciones`): red +
recurso + demanda interrumpible liquidadas con las cuatro reglas y el mismo OPF, con el delta de λ,
congestión, CO₂ y costo contra la línea de fondo; y las 3 obras estructurales de
`NODAL_OBRAS_ESTRUCTURALES` (ORI→SUR 700 MW, CAR→OCC 900 MW, GUJ→NOR 800 MW) entran como paquete.

**Convención de leyenda.** Cada figura abre con un bloque `📖` de cinco campos —qué muestra, decisión
o lectura, objetivo al que responde, supuestos/limitación y fuente de cada número— y el título del
desplegable termina en `· objetivo a/b/c`. Los campos los normaliza `_leyenda()` (con `_lim()` para
no dejar dobles espacios ni separadores colgando cuando un texto se acorta).

**Enfoque de la v4.2: el efecto positivo.** La pregunta no es si el precio sube o baja al pasar de
uninodal a nodal, sino **qué gana el sistema** cuando el precio lleva información de lugar. El
primer bloque de la pestaña (`fig_nodal_beneficios`) es un marcador con esas cifras medidas en el
escenario activo, no en la literatura: renta de congestión que pasa de 0 a estar identificada y
por tanto repartible (143 M COP/h en la ventana de prueba), de 1 a 10 precios distintos (la señal
que usa la inversión para decidir *dónde* conectar), mark-up de equilibrio aprendido por los
agentes que cae al pasar a liquidación nodal, y horas congestionadas/violación que bajan cuando la
demanda puede desconectarse. También dice lo incómodo: el desbalance absoluto de liquidación
**crece** bajo nodal (`a revisar` en el marcador), porque la señal separa pagos por nodo en vez de
promediarlos.

**Qué se resuelve.** Red de 7-10 nodos según topología (`subregion`/`media`/`fina`) con flujos de
distancia DC (`SF = diag(1/x)·A·pinv(Aᵀ·diag(1/x)·A)`), oferta por bloque `c1·P + c2·P²` calibrada
contra `Precio_Bolsa_Dia` de la ventana (κ y nivel por hora, error medio 0,09 %), LMP por
`λ_i = λ(1+LF_i) + Σ_k SF_ki·μ_k` con multiplicadores por ascenso dual adaptativo,
pérdidas iteradas, corte de variable y cuatro reglas de liquidación (`uninodal | zonal |
híbrido κ | nodal`). Encima: escenarios estocásticos (log-normal solar, Weibull eólica, gamma de
demanda, ρ entre nodos y ρ cruce sol-viento, reducción k-medias), GARCH(1,1) + VaR/ES con backtest
de Kupiec, cobertura de varianza mínima `h* = ρσ_S/σ_F`, IOR y η con efectos fijos horarios,
HHI/cuotas por agente y por subregión, Cournot/Bertrand con índice de Lerner, EMS de almacenamiento
(water-filling con η y SOC), Q-learning multiagente con λ endógeno y aprendizaje de mark-up por
regla. Todo con numpy/pandas: **sin scipy** (`requirements.txt` no cambia).

**Los 12 escenarios y su justificación** (selectbox *Escenario predefinido*; la pestaña muestra la
misma tabla con el caso activo marcado, y cada preset sólo fija los controles que declara — el
resto sigue siendo del lector):

| Escenario | Qué fija | Por qué ese número |
|---|---|---|
| `personalizado` | nada | la máquina completa con lo que usted ponga en los controles |
| `base` | ×1,00 de red, sin estrategia | **línea de fondo honesta:** con la malla nominal NO hay congestión (0 h, violación 0, OPF converge), luego zonal ≡ nodal. Todo lo demás se lee como desviación contra aquí |
| `estres-moderado` | ×0,70 + 15 % interrumpible | estrés mínimo con el que aparecen horas congestionadas en la topología `media` (objetivo b) |
| `estres-severo` | ×0,40 + 25 % interrumpible | corredores al límite: dispersión del orden de 1 000 COP/kWh y violación residual avisada |
| `renovables-2030` | solar y eólica ×2 | orden de la expansión del Plan Energético Nacional hacia 2030 sin cambiar la red; muestra el efecto rebote en la rampa de la tarde |
| `almacenamiento` | batería 6 % (4 h, η 0,88) en CEN+BOG y bombeo 10 % (12 h, η 0,80) | η de catálogo, no de oferta; 4 h = ancho del pico 18-21 h medido en la figura de calibración |
| `siting-errado` | el mismo activo en GUJ/MAG con EMS de alivio | es el caso que **debe** salir mal: reproduce que el almacenamiento mal sentado no captura renta locacional y pierde dinero |
| `poder-mercado` | markup 20 % + retiro de energía 25 % sobre el nodo-agente | orden del recargo con el que la SSPD abrió expedientes por poder de mercado (EPM 14/15-mar-2022, Emgesa 11/14/15-mar-2022); el retiro se pone en modo `energia` porque en `capacidad` no muerde si el bloque no está en su techo |
| `desconexion` | 30 % interrumpible con gatillo en 1 000 COP/kWh | objetivo a explícito: el gatillo queda sobre el precio medio (937), así que el corte solo aparece en las horas pico — que es lo que se quiere probar |
| `el-nino` | ×0,60 de red + fase ENSO | nivel re-escalado con el Δ% **medido** en la pestaña 🌊 El Niño (`impacto`, 46 meses): precio +61,3 %, aportes 76,0 vs 104,2 %, embalses 60,5 vs 64,6 % → demanda +6,8 %, recurso −2,2 %/−5,4 % |
| `la-nina` | ×0,85 de red + fase húmeda | espejo con los mismos multiplicadores medidos (197,1 vs 259,9 COP/kWh → ×0,76); muestra que la señal nodal no es un artificio de los meses secos |
| `hibrida-transicion` | κ = 0,35 + batería 3 % | la transición intermedia que discute la regulación en vez del salto al LMP pleno; κ = 0,35 recupera ~un tercio de la renta de congestión (argumento de diseño de CREG 143/2021) |

**Cómo se conecta el escenario de fase.** `_fase_enso()` no usa factores de libro: lee
`ctx["impacto"]` (la tabla Δ% por fase que construye la pestaña 🌊 El Niño) y devuelve
multiplicadores de nivel, demanda, solar y eólica; `_modelo_nodal` los aplica al perfil real y al
nivel calibrado, y `gen_escenarios` recibe el mismo `factor_fase` para que la capa estocástica no
quede en el régimen neutro. El resultado se guarda en `mdl["fase"]` y la pestaña lo enuncia en
cada leyenda —ningún número del escenario se muestra sin su origen.

**Objetivos → figuras.** *a* (comparativo uninodal/nodal con congestión, pérdidas, estocástica de
solar/eólica/demanda y desconexión voluntaria): `nodal_cong`, `nodal_descomp`, `nodal_esc`,
`nodal_flex` (malla flexibilidad × gatillo con los tres percentiles del propio escenario),
`nodal_reglas`, `nodal_beneficios`. *b* (múltiples configuraciones y topologías, almacenamiento,
precios por flujos óptimos, despacho simplificado y reglas de formación de precio): `nodal_topo`,
`nodal_lmp`, `nodal_flujos`, `nodal_desp`, `nodal_valida` (λ dual contra ∂Costo/∂D por diferencias
finitas, |error| ≤ 0,53 %), `nodal_ems`, las 4 reglas en `nodal_reglas`, `nodal_refuerzos`. *c* (comportamiento de los LMP,
cobertura ante volatilidad con alta penetración variable y comportamientos estratégicos ante la
señal): `nodal_spread`, `nodal_var`, `nodal_cov`, `nodal_ior`, `nodal_mec`, `nodal_ap`, `nodal_ql`,
`sensibilidad_penetracion` en `nodal_penetra`.

**Cifras de la ventana 2026-07-31 → 2026-08-26** (31 días, topología `media`, para que el lector
sepa qué esperar): λ medio 937,7 vs bolsa 937,9 (κ ≈ 0,9, error de calibración 0,09 %); 3 h
congestionadas a ×0,70 con violación 79 MW; CO₂ del día tipo 14 215 t; con el escenario El Niño
λ 1 482,8, dispersión 26,5, 6 h congestionadas y 33,5 GWh/día desconectados con 15 % de demanda
interrumpible; en la malla de desconexión (El Niño, 30 % al p40) λ baja a 1 492,5, la dispersión a
43,1, la violación a 260,8 MW y el CO₂ a 13 209 t (−885 t) a cambio de 11,2 GWh/día no servidos.

**Cifras del bloque de refuerzos y de la segunda etapa** (topología `media` × 0,70, ventana
2026-07-31 → 2026-08-30, 60 escenarios reducidos a 6 típicos, modelo en 4,1 s): base con 3 h
congestionadas, violación 79,45 MW, uso máximo 107,1 %, λ 937,68 COP/kWh y CO₂ 14 170 t; los
candidatos individuales mueven λ entre +0,00 y +0,05 COP/kWh con ahorro de −5,0 a +2,9 M COP/día;
la obra nueva GUJ→NOR +800 MW rinde +26,3 M COP/día (payback 0,03 años) y CAR→OCC +900 MW +2,9 M
(0,28 años); el **paquete completo** (9 obras, 1 596 M COP) deja **0 h congestionadas**, uso máximo
63,4 %, violación −79,45 MW y +12,3 M COP/día con payback 0,36 años — menos que la suma de las
partes, que es exactamente el punto. Estocástica: WS 7 367,7 y HN 8 094,5 M COP/h, **VSS +9,87 %**,
`p(escasez)` 7,2 %, dispersión 12,5 COP/kWh y banda p05/p50/p95 = 919,8 / 934,3 / 955,7 COP/kWh
frente al 937,8 COP/kWh del día real. El emparejamiento CV baja el CV eólico de 1,057 a 0,264 con
α = 0,250; la demanda simulada reproduce el nivel real en −0,15 % y su ρ₁ en 0,843 vs 0,950.

**Límites, declarados en la propia pestaña** (bloque *Supuestos del módulo*, 11 puntos). Red DC de
una sola etapa sin restricciones de rampa ni reserva rodante; `MW_max` y reactancias de orden de
magnitud público, no datos ISA/XM; demanda por nodo repartida con ponderaciones calibradas (XM no
publica demanda por subregión en `servapibi`); la forma intradía de la calibración es circular por
construcción y así se enuncia; el «futuro» de la cobertura se aproxima con la serie uninodal.
**Es un prototipo académico: no sustituye la resolución de XM/CREG.** Lo real es el anclaje
(CEN, demanda, bolsa, CMD, mezcla), y por eso sirve para dimensionar el efecto, no para facturar.

## 5 · Qué hace exactamente la app en cada arranque

1. Fija la ventana **último mes cerrado** (`D-1` hacia atrás 31 días por defecto) y **sondea** cada
   métrica: si XM aún no publicó la ventana, retrocede mes a mes (hasta 14) hasta encontrar datos —
   así conviven `Gene` (D-2) con `DemaMaxPot` (2-6 semanas de rezago) sin huecos falsos. Antes de
   agregar nada, **audita la completitud día a día** y aparta los días que XM aún publica preliminares
   (ver §2c); de ahí sale `fin_efectiva`, el día que usan todos los ejes.
2. Descarga en **bloques ≤ 31 días** (límite duro de la API) con 4 reintentos exponenciales por
   bloque, y guarda cada resultado en `data/raw\{metrica}_{inicio}_{fin}.csv`.
3. Transforma el formato ancho (`Values_Hour01…24`) al modelo dimensional (celdas 59-60 del notebook)
   y valida las tablas con la celda 61: `fact_generacion`,
   `dim_plantas`, `gen_enriquecida`, `sistema_h` (único por Timestamp, con verificación de integridad),
   `resumen_diario` (balance, participación térmica, precio ponderado/banda, embalses, escasez).
4. v3/v4: CEN por tecnología y planta, margen de reserva (banda crítica 15 %), factor de planta,
   emisiones e intensidad (con `factorEmisionCO2e` de XM), indisponibilidad y matriz de cruce.
5. ENSO: ONI + Niño1+2/3.4 de NOAA, fase por mes, detección de eventos (≥5 meses), rezagos 0-12 meses,
   tabla de impacto por fase y **semáforo del episodio activo** (hoy: El Niño iniciado en la primavera
   de 2026 y en fortalecimiento en **agosto de 2026**, con el aviso NOAA/CPC del 13-ago-2026 citado).
6. Exporta todas las tablas a CSV (botones de descarga en 📚 Datos) y deja la **bitácora** de la
   ejecución visible, como el `bitacora()` del notebook.

## 6 · Autoría y créditos

> La aplicación se desarrolló a partir del notebook analítico creado por **Manuel Fajardo**
> —versiones v1–v2, secciones 0–6 y anexos— y fue ampliada por el profesor **Sergio Rivera** en las
> versiones v3–v4 con nuevos indicadores y su implementación como tablero interactivo.

* **Manuel Fernando Fajardo Rodríguez** — Senior Electrical Engineer, Power Systems & Data Science;
  Maestría en Ingeniería Eléctrica, Universidad Nacional de Colombia. Autor de las celdas **v1 y v2**
  (Secciones 0-6 y Anexos), originadas en
  [`mffajardor/solar-generation-colombia-dash`](https://github.com/mffajardor/solar-generation-colombia-dash).
* **Prof. Sergio Rivera, PhD, SMIEEE** — Profesor Titular y ex-Director del Departamento de Ingeniería
  Eléctrica y Electrónica, UNAL. Ing. Eléctrico UNAL (2001), Especialista en Distribución (2004),
  PhD (Instituto de Energía Eléctrica, UNSJ, 2011); Postdoc Associate en MIT (2013-14), Postdoctoral
  Fellow en Masdar Institute (2014), Fulbright Visiting Scholar en la Universidad de Florida y Gambrinus
  Fellow en TU Dortmund. Autor de las celdas **v3 y v4** (Secciones 7-12): capacidad, factor de planta,
  emisiones, fenómeno El Niño/La Niña, indisponibilidad, margen y costo marginal.
* **v5 (previsto):** ventana arbitraria (`fecha_inicio`/`fecha_fin`), comparación entre dos periodos y
  informe exportable. La capa de datos ya está preparada (bloques ≤31 d, caché y dimensiones
  reconstruidas por periodo); solo falta cambiar el selector de días por dos `st.date_input`.

**Fuentes:** la herramienta usa **información operativa pública de XM** (API Bienda,
`servapibi.xm.com.co`) **y datos climáticos de NOAA/CPC** (`oni.ascii.txt`,
`ersst5.nino.mth.91-20.ascii`); los avisos ENSO citados son de CPC (13-ago-2026) e IRI (19-ago-2026).
Las consultas a XM emplean los mismos `metric_id` que expone la librería oficial
**`pydataxm.pydatasimem`** —esta app no la importa, llama directo a los mismos endpoints para poder
correr con `pandas`+`requests`+`plotly`+`streamlit` nada más, así que cualquier cifra es reproducible
con esa librería—. **La app es un prototipo académico independiente y no una herramienta oficial de
XM**: no sustituye las publicaciones del CNO/ASM ni la información oficial del operador.
