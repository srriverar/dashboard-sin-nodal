"""Diagnóstico de la cadena CapEfecNeta -> cen_recurso -> cen_diario (unidades y duplicados)."""
import datetime as dt
import importlib.util
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
os.environ["APPMANUEL_NO_MAIN"] = "1"
spec = importlib.util.spec_from_file_location("app", BASE / "app.py")
app = importlib.util.module_from_spec(spec)
sys.modules["app"] = app
spec.loader.exec_module(app)

m = app.CATALOGOS["ListadoRecursos"]
cap = app.METRICAS_V3["CapEfecNeta_Recurso"]
d = dt.date(2026, 8, 20)
raw_cap, _ = app.descargar_metrica(cap, d, d, usar_cache=False)
raw_dim, _ = app.descargar_metrica(m, d, d, usar_cache=False)
print("raw CapEfecNeta:", raw_cap.shape, list(raw_cap.columns)[:6])
print("raw catálogo   :", raw_dim.shape, list(raw_dim.columns)[:8])

cen = app.daily_to_long_recurso(raw_cap, "Potencia_Confiable_kW")
print("\ncen_recurso:", cen.shape, "| suma kW =", format(cen["Potencia_Confiable_kW"].sum(), ",.0f"),
      "-> MW =", format(cen["Potencia_Confiable_kW"].sum() / 1e3, ",.1f"))
print("  filas por fecha:", cen.groupby("Fecha").size().to_dict())
print("  códigos duplicados en el día:", int((cen["Codigo_Planta"].value_counts() > 1).sum()))

dim = app.preparar_dim_plantas(raw_dim)
print("\ndim_plantas:", dim.shape, "códigos únicos:", dim["Codigo_Planta"].nunique())
print("  tecnologia top:", dim["Tecnologia"].value_counts().head(8).to_dict())

cen_d = app.resumen_capacidad(cen, None, dim)
print("\ncen_diario:", cen_d.shape)
print(cen_d.filter(like="CEN_").sum(numeric_only=True).to_string())
