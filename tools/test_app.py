"""Prueba de renderizado real de la app con streamlit.testing (AppTest)."""
import os
import sys

os.environ.setdefault("APPMANUEL_SMOKE", "1")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")

at = AppTest.from_file(APP, default_timeout=int(os.environ.get("APPMANUEL_TEST_TIMEOUT", "900")))
at.run()

fallos = []
print("excepciones del script:", len(at.exception))
for e in at.exception:
    fallos.append(str(e.value)[:400])
    print("---- EXCEPCIÓN ----")
    print(str(e.value)[:2500])

for lbl, seq in (("error", at.error), ("warning", at.warning), ("info", at.info),
                 ("success", at.success)):
    for x in seq:
        print(f"[{lbl}] {str(x.value)[:500]}")
print(f"\nwidgets: títulos={len(at.title)} markdown={len(at.markdown)} métricas={len(at.metric)} "
      f"tabs={len(at.tabs)} figuras(plotly)={len(at.get('plotly_chart')) if hasattr(at,'get') else '·'} "
      f"botones={len(at.button)} selectores={len(at.selectbox)}")
print("título:", at.title[0].value if at.title else "—")
for m in at.metric[:8]:
    print(f"  métrica: {m.label} = {m.value} ({m.delta or '·'})")
# blob de texto con TODO lo renderizado (markdown, captions, subheaders, expanders)
def _txtos(seq):
    out = []
    for e in seq:
        try:
            out.append(str(getattr(e, "value", "") or getattr(e, "label", "") or ""))
        except Exception:                                     # noqa: BLE001
            continue
    return "\n".join(out)


textos = "\n".join([_txtos(at.markdown), _txtos(at.caption), _txtos(at.subheader),
                    _txtos(at.header), _txtos(at.title), _txtos(at.expander)])
for buscado in ("EL NIÑO ESTÁ ACTIVO", "AGOSTO", "v5", "Manuel Fernando Fajardo", "Sergio Rivera",
                "Hallazgos automáticos", "NOAA"):
    print(f"  contiene {buscado!r}: {buscado in textos}")

# --- reglas pedidas explícitamente: pestañas nuevas, eje X en la ventana, 'El Niño en dinero' ---
TABS_ESPERADAS = ["🎯 Resumen ejecutivo", "⚡ Generación", "💧 Hidrología", "🌊 El Niño",
                  "💰 Costos y emisiones", "🛠️ Disponibilidad (v4)", "🕐 Perfiles × hora",
                  "🔥 CO₂ y economía ENSO", "🕸️ Precios nodales", "📚 Datos y bitácora"]
etiquetas = " | ".join(str(getattr(t, "label", "") or "") for t in at.tabs) if at.tabs else ""
print("\npestañas:", etiquetas or "·")
for tab_esperada in TABS_ESPERADAS:
    if tab_esperada and tab_esperada not in etiquetas:
        fallos.append(f"falta la pestaña {tab_esperada!r}")

expanders = _txtos(at.expander)
for grafico in ("El Niño en dinero", "Climatología por fase ENSO", "Emisiones diarias por combustible",
                "Costo marginal", "Panel ENSO", "Top emisoras y curva de Pareto",
                "Perfiles horarios por tecnología",
                # módulo nodal (tesis): anclas que deben verse en la pestaña 🕸️
                "Topología del modelo y uso de las ramas",
                "Mapa de calor del precio marginal por nodo y hora",
                "Descomposición del LMP (Eq. 2-21)",
                "Uninodal · zonal · híbrido · nodal",
                "IOR real, elasticidad medida y concentración por nodo",
                "GARCH(1,1), VaR/ES condicional y backtest de Kupiec",
                "EMS de baterías y bombeo bajo precio nodal",
                "Mark-up aprendido según la regla de liquidación"):
    print(f"  figura '…{grafico}…' renderizada: {grafico in expanders}")
    if grafico not in expanders:
        fallos.append(f"no se renderizó la figura {grafico!r}")

charts = at.get("plotly_chart") if hasattr(at, "get") else []
print(f"figuras plotly en la página: {len(charts)}")
if len(charts) < 40:
    fallos.append(f"solo {len(charts)} figuras plotly (se esperaban ≥ 40)")
# 44 figuras del notebook + ≥ 18 del módulo nodal
if len(charts) < 60:
    fallos.append(f"solo {len(charts)} figuras plotly: la pestaña 🕸️ Precios nodales no renderizó "
                  "sus figuras (se esperaban ≥ 60 en total)")
errores_txt = " ".join(str(x.value) for x in at.error)
if "No se pudo construir el modelo" in errores_txt:
    fallos.append(f"el modelo nodal falló al renderizar: {errores_txt[:220]}")
if "No se pudo construir" in errores_txt:
    fallos.append(f"hubo figuras que no construyeron: {errores_txt[:260]}")
print("  sin errores de construcción de figuras:", "No se pudo construir" not in errores_txt)

# --- lo que el autor pidió RETIRAR no puede aparecer (figuras, filtros y desplegables) ---
infos_temprano = " ".join(str(x.value) for x in at.info) + " " + " ".join(str(x.value) for x in at.warning)
todo = (textos + "\n" + infos_temprano + "\n"
        + "\n".join(str(getattr(w, "label", "") or "") for w in
                     list(at.multiselect) + list(at.selectbox) + list(at.text_input) + list(at.slider)))
for vetado in ("Impacto ENSO: dispersión + rezagos", "Δ% por episodio ENSO (v4 c15)",
               "Tecnología (filtra generación y emisiones)", "Agente / empresa",
               "Tipo de recurso (filtra", "Filtro por tipo de recurso"):
    presente = vetado in todo
    print(f"  retirado {vetado!r}: {not presente}")
    if presente:
        fallos.append(f"sigue apareciendo {vetado!r}")
for clave_vetada in ("en_impacto", "en_eventos"):
    if clave_vetada in expanders or clave_vetada in todo:
        fallos.append(f"la clave {clave_vetada} sigue referenciada en la UI")

# --- lo que el autor pidió AÑADIR ---
for obligatorio in ("srriverar/sin-el-nino-dashboard",       # enlace al notebook v1-v4
                    "Auditoría de completitud",               # tabla de días provisionales
                    "Cobertura del notebook",                  # verificación celda por celda
                    "Validación de las tablas analíticas",     # espejo de la c61
                    "Composición de la CEN por tecnología",    # fig_v3_01 (celda 131)
                    "spread pico-valle",                       # celdas 84-85 (SAEB)
                    "Factor de planta diario × tecnología",    # celda 139
                    "Precio vs térmica / embalses / aportes",  # celdas 76-80
                    "notebook celda",                          # sello celda en las figuras
                    "Incluir días provisionales",              # toggle del sidebar
                    # pestaña nueva 🕸️ · precios nodales (módulo de la tesis doctoral)
                    "Precios marginales nodales en el SIN", "Simulación académica, no liquidación",
                    "Descomposición del LMP", "Supuestos, límites y qué NO demuestra",
                    "ocho preguntas de la propuesta"):          # toggle del sidebar
    presente = obligatorio in todo or any(obligatorio in str(x.label) for x in at.toggle)
    print(f"  presente {obligatorio!r}: {presente}")
    if not presente:
        fallos.append(f"falta {obligatorio!r} en la app")
etiquetas_fig = expanders
import re as _re
celdas_citadas = _re.findall(r"celdas? (\d+)(?:-(\d+))?", etiquetas_fig)
print(f"  expanders con número de celda: {len(celdas_citadas)}")
if len(celdas_citadas) < 25:
    fallos.append(f"solo {len(celdas_citadas)} figuras citan su celda del notebook")

import json as _json
from datetime import datetime as _dt
con_rango = 0
for el in charts:
    spec = (getattr(el.proto, "spec", "") or "").strip()
    if not spec:
        continue
    try:
        layout = (_json.loads(spec).get("layout") or {})
    except Exception:                                         # noqa: BLE001
        continue
    for ax in ("xaxis", "xaxis2", "xaxis3", "xaxis4"):
        r = (layout.get(ax) or {}).get("range")
        if not r:
            continue
        try:
            ini = _dt.fromisoformat(str(r[0])[:19])
        except Exception:                                     # noqa: BLE001
            continue
        con_rango += 1
        if ini.year < 2019:
            fallos.append(f"eje X recortado pero empieza en {ini:%Y-%m-%d}")
print(f"ejes X con rango fijo (ventana/últimos años): {con_rango}")
if con_rango < 5:
    fallos.append("ningún gráfico trae el eje X recortado a la ventana")

# --- avisos de carga, autoría y redacción pedida por el revisor ---
for frase in ("Procesamiento terminado", "Cómo leer esta barra lateral",
              "versiones v1–v2, secciones 0–6 y anexos", "pydataxm.pydatasimem",
              "prototipo académico independiente", "índice Niño-3.4", "NOAA/CPC",
              "V1 · Extracción", "V4 · ENSO"):
    presente = frase in textos or frase in " ".join(str(x.value) for x in at.success)
    print(f"  texto '{frase}': {presente}")
    if not presente and frase not in ("V1 · Extracción", "V4 · ENSO"):
        fallos.append(f"falta el texto {frase!r}")
if "<details>" in textos or "<summary>" in textos:
    fallos.append("quedan etiquetas <details>/<summary> literales en la barra lateral")
else:
    print("  sin etiquetas <details>/<summary> literales: True")

infos = " ".join(str(x.value) for x in at.info)
if "Sin datos suficientes para *El Niño en dinero*" in infos:
    fallos.append("'El Niño en dinero' sigue sin calcularse")
print("'El Niño en dinero' con datos:", "Sin datos suficientes para *El Niño en dinero*" not in infos)
for w in at.warning:
    if "use_container_width" in str(w.value):
        fallos.append("aviso de deprecación de use_container_width")

print("\n" + ("❌ HAY FALLOS" if fallos else "✅ APP RENDERIZA SIN EXCEPCIONES"))
sys.exit(1 if fallos else 0)
