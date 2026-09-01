import requests
import pandas as pd

B = "https://servapibi.xm.com.co"


def post(ep, mid, ent, ini, fin):
    r = requests.post(B + "/" + ep, json={"MetricId": mid, "StartDate": ini,
                                          "EndDate": fin, "Entity": ent, "Filter": []}, timeout=120)
    if r.status_code != 200:
        print(mid, ent, "HTTP", r.status_code, r.text[:120])
        return []
    return r.json().get("Items") or []


it = post("daily", "CapEfecNeta", "Recurso", "2026-08-20", "2026-08-20")
d = pd.json_normalize(it, "DailyEntities", "Date", sep="_")
d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
print("CapEfecNeta  filas=%d  cols=%s" % (len(d), list(d.columns)))
print("  suma kW = %s   -> MW = %s" % (format(d["Value"].sum(), ",.0f"), format(d["Value"].sum() / 1e3, ",.1f")))
print(d.nlargest(6, "Value")[["Id", "Code", "Value"]].to_string(index=False))
print("  codes duplicados:", int((d["Code"].value_counts() > 1).sum()))
print(d.groupby("Id")["Value"].agg(["count", "sum"]).to_string())

h = post("hourly", "Gene", "Recurso", "2026-08-20", "2026-08-20")
g = pd.json_normalize(h, "HourlyEntities", "Date", sep="_")
g["H1"] = pd.to_numeric(g["Values_Hour01"], errors="coerce")
print("\nGene  filas=%d  suma Hour01 (kWh) = %s -> MW = %s" % (len(g), format(g["H1"].sum(), ",.0f"),
                                                                format(g["H1"].sum() / 1e3, ",.1f")))

dd = post("hourly", "DispoCome", "Recurso", "2026-08-20", "2026-08-20")
a = pd.json_normalize(dd, "HourlyEntities", "Date", sep="_")
a["H1"] = pd.to_numeric(a["Values_Hour01"], errors="coerce")
print("DispoCome  filas=%d  suma Hour01 = %s (kW -> MW %s)" % (len(a), format(a["H1"].sum(), ",.0f"),
                                                                format(a["H1"].sum() / 1e3, ",.1f")))
print(a.nlargest(4, "H1")[["Id", "Values_code", "H1"]].to_string(index=False))
