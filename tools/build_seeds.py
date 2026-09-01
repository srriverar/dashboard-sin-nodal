"""
Herramienta de desarrollo (NO forma parte de la app Streamlit).

Construye los archivos de referencia ligeros que permiten que el dashboard
arranque al instante (y funcione sin conexión) sin volver a descargar años
de historia mes a mes:

  data/reference/oni_noaa.csv         -> ONI NOAA CPC (1950 -> hoy), 1 fila/temporada
  data/reference/nino_mensual.csv     -> medias mensuales 2015 -> hoy de
                                         PorcApor / PorcVoluUtilDiar / PPPrecBolsNaci
  data/reference/v4_mensual.csv       -> medias mensuales 2016 -> hoy de
                                         DemaReal / DemaMaxPot / CostMargDesp (Sistema)

Se ejecuta una sola vez:  python tools/build_seeds.py
El app.py regenera/actualiza estos mismos archivos por la vía incremental.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import os
import sys
import threading

import pandas as pd
import requests

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REF = os.path.join(BASE, "data", "reference")
os.makedirs(REF, exist_ok=True)

HISTORIC = os.environ.get("SEED_HISTORIC_START", "2015-01")
V4_START = os.environ.get("SEED_V4_START", "2016-01")
YERNO = dt.date.today() - dt.timedelta(days=1)
WORKERS = int(os.environ.get("SEED_WORKERS", "12"))

_tls = threading.local()


def _sesion() -> requests.Session:
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def _post(path: str, metric: str, entity: str, ini: dt.date, fin: dt.date, reintentos: int = 4):
    cuerpo = {
        "MetricId": metric,
        "StartDate": ini.isoformat(),
        "EndDate": fin.isoformat(),
        "Entity": entity,
        "Filter": [],
    }
    ultima = None
    for k in range(reintentos):
        try:
            r = _sesion().post(f"https://servapibi.xm.com.co/{path}", json=cuerpo, timeout=120)
            if r.status_code == 200:
                return r.json().get("Items", []) or []
            ultima = f"HTTP {r.status_code}"
        except Exception as exc:  # noqa: BLE001
            ultima = str(exc)[:120]
        import time

        time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"{metric}/{ini}->{fin}: {ultima}")


def _diarios(items, clave="DailyEntities"):
    out = []
    for it in items:
        fecha = it.get("Date")
        for ent in it.get(clave, []) or []:
            vals = ent.get("Values") if isinstance(ent.get("Values"), dict) else {}
            v = ent.get("Value", vals.get("Value"))
            if v is None:
                continue
            try:
                out.append((fecha, float(v)))
            except (TypeError, ValueError):
                pass
    return out


def _horarios(items, clave="HourlyEntities"):
    out = []
    for it in items:
        fecha = it.get("Date")
        for ent in it.get(clave, []) or []:
            vals = ent.get("Values") if isinstance(ent.get("Values"), dict) else {}
            code = vals.get("code") or vals.get("Code") or ent.get("Code") or "Sistema"
            for h in range(1, 25):
                raw = vals.get(f"Hour{h:02d}")
                try:
                    out.append((fecha, str(code), h, float(raw)))
                except (TypeError, ValueError):
                    continue
    return out


def _meses(inicio: str):
    return [p for p in pd.period_range(inicio, pd.PeriodIndex([YERNO], freq="M")[0], freq="M")
            if p.start_time.date() <= YERNO]


# ── ONI ──────────────────────────────────────────────────────────────────────
def build_oni():
    for url in ("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
                "https://psl.noaa.gov/data/correlation/oni.data"):
        try:
            txt = _sesion().get(url, timeout=90).text
            if "oni.ascii" in url:
                import io as _io

                df = pd.read_csv(_io.StringIO(txt), sep=r"\s+")
                df = df.rename(columns={"YR": "Anio", "SEAS": "Temporada", "ANOM": "ONI"})
                central = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7,
                           "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
                df["Mes"] = df["Temporada"].map(central)
                df["Fecha"] = pd.to_datetime(dict(year=df["Anio"], month=df["Mes"], day=1))
                df = df[["Fecha", "Anio", "Mes", "Temporada", "ONI"]].dropna(subset=["ONI"])
            else:
                filas = []
                for linea in txt.splitlines():
                    partes = linea.split()
                    if len(partes) != 13 or not partes[0][:4].isdigit():
                        continue
                    anio = int(partes[0][:4])
                    for mes, val in enumerate(partes[1:], start=1):
                        v = float(val)
                        if v > -90:
                            filas.append({"Fecha": pd.Timestamp(anio, mes, 1), "Anio": anio,
                                          "Mes": mes, "Temporada": "PSL", "ONI": v})
                df = pd.DataFrame(filas)
            if not df.empty:
                df = df.sort_values("Fecha").drop_duplicates("Fecha").reset_index(drop=True)
                df.to_csv(os.path.join(REF, "oni_noaa.csv"), index=False)
                print(f"  ✅ oni_noaa.csv  {len(df)} filas  ({df['Fecha'].min():%Y-%m} → {df['Fecha'].max():%Y-%m})")
                return
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠️  ONI {url}: {exc}")
    raise SystemExit("No se pudo construir el ONI")


# ── serie mensual ENSO (métrica diaria de Sistema: media del mes) ───────────
def _agg_diario(mes, metric, entity, path):
    ini = mes.start_time.date()
    fin = min(mes.end_time.date(), YERNO)
    items = _post(path, metric, entity, ini, fin)
    vals = [v for _, v in _diarios(items)]
    return str(mes), (sum(vals) / len(vals) if vals else float("nan")), len(vals)


def build_nino_mensual():
    filas = {}
    defs = {"Aportes_Pct": ("PorcApor", "Sistema", "daily"),
            "Embalses_Pct": ("PorcVoluUtilDiar", "Sistema", "daily"),
            "Precio_Ponderado_COP": ("PPPrecBolsNaci", "Sistema", "daily")}
    meses = _meses(HISTORIC)
    print(f"  📡 nino_mensual: {len(meses)} meses × {len(defs)} métricas")
    for col, (metric, entity, path) in defs.items():
        with cf.ThreadPoolExecutor(WORKERS) as pool:
            res = list(pool.map(lambda m: _agg_diario(m, metric, entity, path), meses))
        for mes, val, n in res:
            filas.setdefault(mes, {})[col] = val
        if col in ("Aportes_Pct", "Embalses_Pct"):  # XM entrega fracción -> *100
            serie = pd.Series({k: v.get(col) for k, v in filas.items()}, dtype=float).dropna()
            if len(serie) and serie.median() <= 2.0:
                for k in filas:
                    if filas[k].get(col) is not None:
                        filas[k][col] *= 100
        print(f"     · {col}: ok")
    df = pd.DataFrame.from_dict(filas, orient="index").rename_axis("Mes").reset_index().sort_values("Mes")
    df.to_csv(os.path.join(REF, "nino_mensual.csv"), index=False)
    print(f"  ✅ nino_mensual.csv  {len(df)} filas")


# ── serie mensual v4 (demanda / demanda máx / costo marginal) ───────────────
def _agg_mes_v4(mes):
    """Tres peticiones por mes -> todas las columnas mensuales de v4."""
    ini = mes.start_time.date()
    fin = min(mes.end_time.date(), YERNO)
    fila = {"Mes": str(mes)}

    items = _post("hourly", "DemaReal", "Sistema", ini, fin)
    vals = [v for _, _, _, v in _horarios(items)]
    fila["Demanda_MW"] = sum(vals) / len(vals) / 1e3 if vals else float("nan")

    items = _post("daily", "DemaMaxPot", "Sistema", ini, fin)
    vals = [v for _, v in _diarios(items)]
    if vals:
        fila["DemaMax_MW"] = max(vals) / 1e3
        fila["DemaMaxProm_MW"] = sum(vals) / len(vals) / 1e3
    else:
        fila["DemaMax_MW"] = fila["DemaMaxProm_MW"] = float("nan")

    items = _post("hourly", "CostMargDesp", "Sistema", ini, fin)
    vals = [v for _, _, _, v in _horarios(items)]
    fila["Costo_Marginal_COP_kWh"] = sum(vals) / len(vals) if vals else float("nan")

    # CEN media del mes (kW -> MW): suma diaria de recursos, media del mes
    items = _post("daily", "CapEfecNeta", "Recurso", ini, fin)
    por_dia: dict = {}
    for it in items:
        fecha = it.get("Date")
        for ent in it.get("DailyEntities", []) or []:
            try:
                v = float(ent.get("Value"))
            except (TypeError, ValueError):
                continue
            por_dia.setdefault(fecha, 0.0)
            por_dia[fecha] += v
    if por_dia:
        totales = list(por_dia.values())
        fila["CEN_MW"] = sum(totales) / len(totales) / 1e3
        fila["CEN_dias"] = len(totales)
    else:
        fila["CEN_MW"] = float("nan")
    return fila


def build_v4_mensual():
    meses = _meses(V4_START)
    print(f"  📡 v4_mensual: {len(meses)} meses × 3 métricas (Sistema, ligeras)")
    with cf.ThreadPoolExecutor(WORKERS) as pool:
        filas = list(pool.map(_agg_mes_v4, meses))
    df = pd.DataFrame(filas).sort_values("Mes").reset_index(drop=True)
    df = df[[c for c in ["Mes", "Demanda_MW", "DemaMax_MW", "DemaMaxProm_MW", "CEN_MW",
                          "Costo_Marginal_COP_kWh"] if c in df.columns]]
    df.to_csv(os.path.join(REF, "v4_mensual.csv"), index=False)
    print(f"  ✅ v4_mensual.csv  {len(df)} filas · margen no incluido (requiere DispoCome, pesado)")


def build_nino34():
    """Nino1+2/3/4/3.4 mensuales (ERSSTv5, base 1991-2020) -> reference/nino34_mensual.csv."""
    urls = ["https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
            "https://psl.noaa.gov/data/correlation/nina34.anom.data"]
    for u in urls:
        try:
            raw = requests.get(u, timeout=120).text
        except Exception as exc:
            print("  x", u, exc)
            continue
        filas = []
        for lin in raw.splitlines():
            parts = lin.split()
            if len(parts) < 10 or not parts[0].strip().isdigit():
                continue
            ano, mes = int(parts[0]), int(parts[1])
            if not (1950 <= ano <= 2100) or not (1 <= mes <= 12):
                continue
            try:
                n12, a12, n3, a3, n4, a4, n34, a34 = (float(x) for x in parts[2:10])
            except ValueError:
                continue
            filas.append({"Mes": f"{ano}-{mes:02d}", "Nino1_2": a12, "Nino3": a3,
                          "Nino4": a4, "Nino3_4": a34})
        if filas:
            df = pd.DataFrame(filas).sort_values("Mes").reset_index(drop=True)
            df.to_csv(os.path.join(REF, "nino34_mensual.csv"), index=False)
            print(f"  OK nino34_mensual.csv  {len(df)} filas  ({df['Mes'].iloc[0]} -> {df['Mes'].iloc[-1]})")
            return
    print("  x sin serie Nino3.4 mensual")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(f"🌱 BASE={BASE}  trabajadores={WORKERS}  ventana={HISTORIC}→{YERNO}")
    if what in ("all", "oni"):
        build_oni()
    if what in ("all", "nino34"):
        build_nino34()
    if what in ("all", "nino"):
        build_nino_mensual()
    if what in ("all", "v4"):
        build_v4_mensual()
    print("🏁 seeds listos en", REF)
