#!/usr/bin/env python3
"""Prueba de humo del tablero: ejecuta el pipeline real y TODAS las figuras.

Uso:
    python tools/smoke_test.py            # ventana corta, descarga real (modo api)
    APPMANUEL_SMOKE_MODE=cache python tools/smoke_test.py   # sin red, solo data/raw
Salida: 0 si todo el ETL y todas las figuras construyen; 1 en caso contrario.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time as _time
import traceback
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]
os.environ["APPMANUEL_NO_MAIN"] = "1"
sys.path.insert(0, str(BASE))

spec = importlib.util.spec_from_file_location("app", BASE / "app.py")
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)

DIAS = int(os.environ.get("APPMANUEL_SMOKE_DIAS", "12"))
MODO = os.environ.get("APPMANUEL_SMOKE_MODE", "auto")

opciones = {
    "Gene_Sistema": True, "Gene_Recurso": True, "DemaReal_Sistema": True,
    "PrecBolsNaci_Sistema": True, "PPPrecBolsNaci_Sistema": True,
    "PorcVoluUtilDiar_Sistema": True, "PorcApor_Sistema": True, "AporEner_Sistema": True,
    "PrecEsca_Sistema": True, "CostMargDesp_Sistema": True, "CapEfecNeta_Recurso": True,
    "DemaMaxPot_Sistema": True, "EmisionesCO2_RecursoComb": True,
    "ConsCombustibleMBTU_Recurso": True, "factorEmisionCO2e_Sistema": True,
}

print(f"== construir_contexto(dias={DIAS}, modo={MODO}) ==")
ctx = app.construir_contexto(DIAS, opciones, MODO)

lin = ctx["linaje"]
print(lin[["métrica", "filas", "origen", "nota"]].to_string(index=False))

fallos: list[str] = []


def revisar(nombre, df, cols, minimo=1):
    if df is None or len(df) < minimo:
        fallos.append(f"{nombre}: vacío")
        return
    faltan = [c for c in cols if c not in df.columns]
    if faltan:
        fallos.append(f"{nombre}: faltan columnas {faltan}")


print("\n== tablas ==")
revisar("gen_enr", ctx["gen_enr"], ["Fecha", "Hora", "Codigo_Planta", "Generacion_kWh", "Tecnologia"], 1000)
revisar("sistema", ctx["sistema"], ["Fecha", "Hora", "Demanda_MW", "Embalses_Pct"], 100)
revisar("resumen", ctx["resumen"], ["Fecha", "Demanda_GWh", "Precio_Ponderado_COP_kWh", "Embalses_Pct"])
revisar("embalses", ctx["embalses"], ["Fecha", "Volumen_Pct"])
revisar("aportes_pct", ctx["aportes_pct"], ["Fecha", "Aportes_Pct"])
revisar("dim_plantas", ctx["dim_plantas"], ["Codigo_Planta", "Tecnologia", "Nombre_Planta"])
revisar("cen_recurso", ctx["cen_recurso"], ["Fecha", "Codigo_Planta", "Potencia_Confiable_kW"])
revisar("cen_diario", ctx["cen_diario"], ["Fecha", "CEN_Total_MW", "Margen_pct"])
revisar("fp", ctx["fp"], ["Mes", "Codigo_Planta", "FP_pct"])
revisar("emisiones", ctx["emisiones"], ["Fecha", "Hora", "Codigo_Planta", "Combustible", "tCO2"])
revisar("emisiones_dia", ctx["emisiones_dia"], ["Fecha", "tCO2_XM", "Intensidad_tCO2_MWh"])
revisar("nino", ctx["nino"], ["Mes", "ONI", "Fase", "Fase_operativa", "Embalses_Pct", "Aportes_Pct"])
revisar("v4_serie", ctx["v4_serie"], ["Mes", "Demanda_MW", "Costo_Marginal_COP_kWh"])
tamanos = []
for k in ("gen", "gen_enr", "gen_tec", "sistema", "resumen", "embalses", "aportes_pct", "escasez",
          "dim_plantas", "cen_recurso", "cen_diario", "dema_max", "disp_come", "indisp", "emisiones",
          "emisiones_dia", "combustible", "costo_marginal", "fp", "nino", "v4_serie", "cruce_v4"):
    v = ctx.get(k)
    tamanos.append(f"{k}={len(v) if v is not None and hasattr(v, '__len__') else '-'}")
print("filas: " + " · ".join(tamanos))

print("\n== controles de unidad / plausibilidad ==")
r = ctx["resumen"]
if not r.empty:
    if "Demanda_GWh" in r:
        med = float(r["Demanda_GWh"].median())
        print(f"demand media diaria = {med:,.0f} GWh  (esperado 200-280)")
        if not 120 < med < 400:
            fallos.append(f"demand fuera de rango plausible: {med}")
    if "Embalses_Pct" in r:
        e = float(r["Embalses_Pct"].median())
        print(f"embalses mediana = {e:.1f} %  (esperado 15-95)")
        if not 5 < e < 100:
            fallos.append(f"embalses fuera de rango: {e} (¿falta escalar fracción?)")
    if "Precio_Ponderado_COP_kWh" in r:
        p = float(r["Precio_Ponderado_COP_kWh"].median())
        print(f"precio ponderado mediana = {p:,.1f} COP/kWh (esperado 50-4000)")
        if not 1 < p < 20000:
            fallos.append(f"precio fuera de rango: {p}")
print("\n== balance generación vs demanda ==")
if not r.empty and "Generacion_Total_GWh" in r and "Demanda_GWh" in r:
    ge, de = float(r["Generacion_Total_GWh"].mean()), float(r["Demanda_GWh"].mean())
    print(f"generación media {ge:,.1f} GWh/día vs demanda media {de:,.1f} GWh/día "
          f"(diferencia {100*(ge-de)/de:+.1f} %)")
    if not 0.75 < ge / max(de, 1e-9) < 1.35:
        fallos.append(f"balance generación/demanda implausible: {ge:,.0f} vs {de:,.0f} GWh/día")
else:
    fallos.append("resumen sin Generacion_Total_GWh/Demanda_GWh (no se puede validar el balance)")
if not ctx["cen_diario"].empty and "CEN_Total_MW" in ctx["cen_diario"]:
    c = float(ctx["cen_diario"]["CEN_Total_MW"].median())
    print(f"CEN total mediana = {c:,.0f} MW (esperado 15 000-30 000)")
    if not 5000 < c < 60000:
        fallos.append(f"CEN fuera de rango: {c}")
if not ctx["nino"].empty:
    est = app.resumen_evento_actual(ctx["nino"])
    print("estado ENSO:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                     for k, v in est.items()}, ensure_ascii=False))
    if not est or est.get("inicio_oficial") is None:
        fallos.append("resumen_evento_actual sin inicio")
    if not ctx["nino_eventos"].empty:
        print(f"eventos detectados: {len(ctx['nino_eventos'])} "
              f"(último: {ctx['nino_eventos'].iloc[-1].to_dict()})")
    else:
        fallos.append("detección de eventos ENSO vacía")

print("\n== figuras ==")
for clave, (fn, tab, titulo) in app.FIGURES.items():
    try:
        out = fn(ctx)
    except Exception as exc:                                      # noqa: BLE001
        fallos.append(f"figura {clave}: {type(exc).__name__}: {exc}")
        print(f"  ✗ {clave:20s} {type(exc).__name__}: {str(exc)[:110]}")
        continue
    if out is None or out.get("fig") is None:
        print(f"  – {clave:20s} sin datos ({titulo})")
        continue
    n_trazas = len(out["fig"].data)
    if not out.get("leyenda"):
        fallos.append(f"figura {clave}: sin leyenda")
    print(f"  ✓ {clave:20s} trazas={n_trazas}  leyenda={len(out['leyenda'])} chars")

print("\n== módulo nodal (sección 10 de app.py · tesis doctoral) ==")
try:
    cfg_n = dict(app._NODAL_DEFECTOS)
    cfg_n.update(episodios_rl=800, episodios_ap=4, horas_valida=4, n_esc=24, n_tipicos=4,
                 sensibilidad=False, almacenar=True, bateria_pct=6.0, bombeo_pct=0.0)
    t_n = _time.time()
    mdl = app._modelo_nodal(ctx, cfg_n)
    dt_n = _time.time() - t_n
    lam_med = float(np.mean(mdl["res"]["lam"]))
    print(f"  modelo en {dt_n:.1f} s · topología {mdl['red']['granularidad']} · "
          f"{len(mdl['red']['nodos'])} nodos · {len(mdl['red']['ramas'])} ramas · "
          f"λ {lam_med:,.1f} COP/kWh · converged={bool(mdl['res']['converged'])}")
    if dt_n > 150:
        fallos.append(f"_modelo_nodal tardó {dt_n:.0f} s (¿se activó un bloque caro por defecto?)")
    if set(app.FIGURES_NODAL) != set(app.NODAL_TESIS):
        fallos.append("FIGURES_NODAL y NODAL_TESIS no tienen las mismas claves")
    if not 300.0 < lam_med < 3000.0:
        fallos.append(f"λ medio implausible: {lam_med:,.1f} COP/kWh")
    real_med = float(np.mean(np.asarray(mdl["pf"]["precio"], dtype=float)))
    if real_med > 0 and abs(lam_med - real_med) > 140.0:
        fallos.append(f"λ medio del modelo ({lam_med:,.0f}) se aleja >140 COP/kWh del precio de bolsa "
                      f"real ({real_med:,.0f}): la calibración nodal falló")
    sesgo = mdl["cal"].get("sesgo")
    if np.ndim(sesgo):
        sesgo = float(np.mean(sesgo))
    if sesgo is not None and abs(float(sesgo)) > 80.0:
        fallos.append(f"sesgo de calibración nodal {float(sesgo):.1f} COP/kWh")
    n_ok = n_none = 0
    for clave, (fn, _g, titulo) in app.FIGURES_NODAL.items():
        try:
            out = fn(ctx, cfg_n, mdl)
        except Exception as exc:                                  # noqa: BLE001
            fallos.append(f"figura nodal {clave}: {type(exc).__name__}: {exc}")
            print(f"  ✗ {clave:16s} {type(exc).__name__}: {str(exc)[:90]}")
            continue
        if out is None or out.get("fig") is None:
            n_none += 1
            print(f"  – {clave:16s} sin figura con esta configuración ({titulo[:36]})")
            continue
        if not out.get("leyenda"):
            fallos.append(f"figura nodal {clave}: sin leyenda")
        out["fig"].to_plotly_json()        # plotly valida propiedades del spec
        n_ok += 1
        print(f"  ✓ {clave:16s} trazas={len(out['fig'].data):2d}  leyenda={len(out['leyenda'])} chars")
    print(f"  figuras nodales: {n_ok} con gráfico · {n_none} sin datos (capas apagadas) · "
          f"{len(app.FIGURES_NODAL)} registradas")
    if n_ok < 18:
        fallos.append(f"sólo {n_ok} figuras nodales produjeron gráfico (se esperaban ≥ 18)")
except Exception as exc:                                          # noqa: BLE001
    fallos.append(f"módulo nodal: {type(exc).__name__}: {exc}")
    traceback.print_exc(limit=3)

try:
    hall = app.construir_hallazgos_v3(ctx)
    print(f"\nhallazgos v2 = {len(ctx['hallazgos'])} · v3 = {len(hall)}")
    for h in (ctx["hallazgos"] + hall):
        print("   •", h["titulo"])
    if not ctx["hallazgos"]:
        fallos.append("construir_hallazgos devolvió lista vacía")
except Exception as exc:                                          # noqa: BLE001
    fallos.append(f"construir_hallazgos_v3: {exc}")

print("\n" + ("❌ FALLOS:\n  - " + "\n  - ".join(fallos) if fallos else "✅ TODO EL PIPELINE Y LAS FIGURAS OK"))
sys.exit(1 if fallos else 0)
