"""Chequeo de unidades de EmisionesCO2 / ConsCombustibleMBTU contra la generación térmica."""
import glob
import pandas as pd

for pref in ("EmisionesCO2_RecursoComb", "ConsCombustibleMBTU_Recurso"):
    fs = glob.glob("data/raw/%s_*.csv" % pref)
    if not fs:
        print(pref, ": sin caché")
        continue
    d = pd.read_csv(fs[0])
    horas = [c for c in d.columns if "Hour" in c]
    num = d[horas].apply(pd.to_numeric, errors="coerce")
    dia = num.sum(axis=1)
    print("%-30s filas=%5d  suma/día media=%s  max=%s  entidad media=%s"
          % (pref, len(d), format(dia.mean(), ",.1f"), format(dia.max(), ",.1f"),
             format(dia.mean() / max(d.groupby("Date").size().mean(), 1), ",.2f")))
    print("   muestra:", d[["Date", "Values_Name", "Values_code"]].head(3).to_dict("records"))

g = sorted(glob.glob("data/raw/Gene_Sistema_*.csv"))
if g:
    d = pd.read_csv(g[0])
    horas = [c for c in d.columns if "Hour" in c]
    print("\nGene_Sistema: %d filas (fecha×recurso)" % len(d))
