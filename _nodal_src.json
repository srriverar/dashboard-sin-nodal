def zona_por_toponimia(nombre: Any) -> str | None:
    """Subregión de un recurso por palabra clave en su nombre (None si no aparece)."""
    n = str(nombre).upper()
    for z, kws in KEYWORDS_ZONA.items():
        for k in kws:
            kk = k.strip().upper().rstrip("? ").strip()
            if not kk or kk.endswith("NO"):
                continue
            if kk in n:
                return z
    return None


def mapa_planta_zona(ctx: dict) -> pd.DataFrame:
    """`Codigo_Planta` → subregión + MW + cómo se asignó (y % de cobertura).

    Es la respuesta operativa a una brecha de información real del set abierto:
    la API de XM no publica coordenadas
    ni zona en `ListadoRecursos`, así que la única geografía presente en los
    datos abiertos es la toponimia del nombre de la planta.
    """
    dp, cen = ctx.get("dim_plantas"), ctx.get("cen_recurso")
    if not isinstance(dp, pd.DataFrame) or dp.empty or not isinstance(cen, pd.DataFrame) or cen.empty:
        return pd.DataFrame()
    pot = "Potencia_Confiable_kW" if "Potencia_Confiable_kW" in cen.columns else list(cen.columns)[-1]
    cap = cen.groupby("Codigo_Planta")[pot].max().div(1000.0).rename("CEN_MW").reset_index()
    cols = [c for c in ("Codigo_Planta", "Nombre_Planta", "Tecnologia", "Fuente_Energia",
                        "Tipo_Despacho", "Tipo_Recurso", "Codigo_Agente") if c in dp.columns]
    t = cap.merge(dp[cols].drop_duplicates("Codigo_Planta"), on="Codigo_Planta", how="inner")
    if t.empty:
        return t
    t["Zona"] = t["Nombre_Planta"].map(zona_por_toponimia)
    t["asignacion"] = np.where(t["Zona"].notna(), "toponimia", "reparto proporcional")
    tot = max(float(t["CEN_MW"].sum()), 1e-9)
    t["cobertura_MW_pct"] = round(100.0 * float(t.loc[t["Zona"].notna(), "CEN_MW"].sum()) / tot, 1)
    return t.sort_values("CEN_MW", ascending=False).reset_index(drop=True)


# --------------------- 10.2 · DC: PTDF y factores de pérdidas --------------

def matriz_ptdf(nodos: list[str], ramas: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """SF (sensibilidad del flujo de rama ante inyección neta) y `r` de las ramas.

    DC sin pérdidas: B·θ = P_inj y F = diag(b)·A·θ ⇒ SF = diag(b)·A·B⁺. El factor
    de pérdidas marginal del nodo i se reconstruye después como
    LF_i = 2·Σ_k r_k·F_k·SF_ki (depende del punto de operación): es el «factor de
    pérdidas marginales»: el término λ·LF_i más la contribución de congestión.
    """
    idx = {z: i for i, z in enumerate(nodos)}
    n, m = len(nodos), len(ramas)
    A = np.zeros((m, n))
    b = np.zeros(m)
    r = np.zeros(m)
    for k, row in enumerate(ramas.itertuples()):
        i, j = idx[row.de], idx[row.a]
        A[k, i], A[k, j] = 1.0, -1.0
        b[k] = 1.0 / max(float(row.x_pu), 1e-9)
        r[k] = max(float(row.r_pu), 0.0)
    B = A.T @ (A * b[:, None]) + np.eye(n) * 1e-9
    Binv = np.linalg.pinv(B, rcond=1e-9)
    return (A * b[:, None]) @ Binv, r


def variantes_topologia() -> pd.DataFrame:
    """Catálogo de las redes que se pueden simular (objetivo b)."""
    filas = [
        dict(clave="subregion", nodos=7, ramas=len(RAMAS_BASE), etiqueta="SIN-7 · un nodo por subregión",
             nota="Topología base: 7 nodos con la CEN real por subregión. Aquí zonal ≡ nodal: la "
                  "granularidad define el esquema, y el tablero lo dice en vez de disfrazarlo."),
        dict(clave="media", nodos=10, ramas=len(RAMAS_BASE) + 3, etiqueta="SIN-10 · 3 nodos desdoblados",
             nota="Centro/Nordeste/Caribe se parten en nodo de generación y nodo de carga: aparece la "
                  "diferencia entre precio nodal y zonal (sensibilidad que pide el objetivo b)."),
        dict(clave="fina", nodos=12, ramas=len(RAMAS_BASE) + 5, etiqueta="SIN-12 · malla con refuerzos",
             nota="SIN-10 + dos refuerzos hipotéticos (Antioquia–Valle y Caribe–Orinoquía): mide cuánto "
                  "de la congestión es removible con inversión en la red."),
    ]
    return pd.DataFrame(filas)


def construye_red(ctx: dict, *, granularidad: str = "subregion", escalon_lineas: float = 1.0,
                  solar_pct: float = 0.0, eolica_pct: float = 0.0,
                  curvatura: float = NODAL_CURVATURA) -> dict[str, Any]:
    """Red (nodos, ramas, bloques de oferta) calibrada con los datos de XM.

    Capacidad por nodo: `cen_recurso` agregado a subregión con `mapa_planta_zona`;
    demanda por nodo: `PESO_DEMANDA_NODAL` sobre el perfil real de `sistema`;
    nivel de oferta: calibrado contra `Precio_Bolsa_COP_kWh` en `calibra_oferta`.
    """
    mp = mapa_planta_zona(ctx)
    nodos = list(ZONAS_NODAL)
    ramas = [dict(de=d, a=a, x_pu=x, r_pu=r, MW_max=M * float(escalon_lineas), nombre=nb)
             for d, a, x, r, M, nb in RAMAS_BASE]
    pesos_dem = dict(PESO_DEMANDA_NODAL)
    if granularidad in ("media", "fina"):
        for hijo, padre in ZONA_PADRE_NODAL.items():
            nodos.append(hijo)
            ramas.append(dict(de=hijo, a=padre, x_pu=0.0040, r_pu=0.0004,
                              MW_max=(2400.0 if granularidad == "fina" else 1600.0) * float(escalon_lineas),
                              nombre=f"enlace de desdoblamiento {hijo}–{padre}"))
            base = pesos_dem.get(padre, 0.0)
            pesos_dem[hijo] = base * 0.55
            pesos_dem[padre] = base * 0.45
        if granularidad == "fina":
            ramas.append(dict(de="ANT", a="OCC", x_pu=0.0200, r_pu=0.0025,
                              MW_max=1200.0 * float(escalon_lineas),
                              nombre="refuerzo Antioquia–Valle (variante)"))
            ramas.append(dict(de="MAG", a="ORI", x_pu=0.0400, r_pu=0.0052,
                              MW_max=700.0 * float(escalon_lineas),
                              nombre="refuerzo Caribe–Orinoquía (variante)"))
    df_ram = pd.DataFrame(ramas)
    nombres = {**{k: v[0] for k, v in ZONAS_NODAL.items()}, **{k: v[0] for k, v in ZONAS_EXTRA_NODAL.items()}}
    coords = {**{k: v[1] for k, v in ZONAS_NODAL.items()}, **{k: v[1] for k, v in ZONAS_EXTRA_NODAL.items()}}

    tec_cols = ["HIDRO", "CARBON", "GAS_CC", "GAS_OXI", "BIO", "SOLAR", "EOLICA"]
    cap = pd.DataFrame(0.0, index=nodos, columns=tec_cols)
    cobertura = 0.0
    if not mp.empty:
        mp = mp.assign(tec=mp["Tecnologia"].map(TEC_XM_A_NODAL) if "Tecnologia" in mp else np.nan)
        asign = mp[mp["Zona"].notna() & mp["tec"].notna()]
        if not asign.empty:
            g = (asign.groupby(["Zona", "tec"])["CEN_MW"].sum().unstack(fill_value=0.0)
                 .reindex(index=nodos, columns=tec_cols).fillna(0.0))
            cap = cap.add(g, fill_value=0.0)
        resid = mp.loc[mp["Zona"].isna() & mp["tec"].notna()].groupby("tec")["CEN_MW"].sum()
        if not resid.empty:
            base = cap.sum(axis=1)
            w_res = (base / base.sum()) if float(base.sum()) > 0 else pd.Series(1.0 / len(nodos), index=nodos)
            for tec, val in resid.items():
                if tec in cap.columns:
                    cap[tec] = cap[tec] + float(val) * w_res
        tot = float(mp["CEN_MW"].sum())
        cobertura = 100.0 * float(mp.loc[mp["Zona"].notna(), "CEN_MW"].sum()) / max(tot, 1e-9)
    else:
        nominal = {
            "HIDRO": {"CEN": 4200, "NOR": 7100, "CAR": 900, "OCC": 1100, "SUR": 900, "ORI": 250, "GUJ": 60},
            "CARBON": {"CEN": 900, "NOR": 900, "CAR": 1500, "OCC": 700, "SUR": 60, "ORI": 120, "GUJ": 1400},
            "GAS_CC": {"CEN": 700, "NOR": 350, "CAR": 500, "OCC": 250, "SUR": 120, "ORI": 350, "GUJ": 300},
            "GAS_OXI": {"CEN": 450, "NOR": 250, "CAR": 420, "OCC": 180, "SUR": 90, "ORI": 180, "GUJ": 60},
            "BIO": {"CEN": 60, "NOR": 60, "CAR": 90, "OCC": 130, "SUR": 40, "ORI": 20, "GUJ": 10},
            "SOLAR": {"GUJ": 1900, "CAR": 900, "CEN": 500, "NOR": 350, "OCC": 250, "SUR": 100, "ORI": 80},
            "EOLICA": {"GUJ": 400, "CAR": 120, "OCC": 30, "NOR": 30, "CEN": 10, "SUR": 10, "ORI": 10},
        }
        for tec, dic in nominal.items():
            for z, v in dic.items():
                if z in cap.index:
                    cap.loc[z, tec] = float(v)
        cobertura = 0.0
    for extra, orig in ZONA_PADRE_NODAL.items():        # extra = nodo hijo, orig = zona padre
        if extra in cap.index and orig in cap.index and cap.index.is_unique:
            for tec in cap.columns:
                mover = cap.loc[orig, tec] * (0.30 if tec in ("HIDRO", "CARBON") else 0.55)
                cap.loc[orig, tec] -= mover
                cap.loc[extra, tec] += mover
    cap = cap.clip(lower=0.0)
    # --- refinamientos con lo que XM sí publica ---------------------------------
    # (1) La CEN llega con `Tecnologia = TERMICA` para gas, carbón y diésel. Sin
    #     separarlos, todo el térmico pagaría el factor de emisión del carbón y el CO₂
    #     del tablero saldría inflado; aquí se reparte la piscina térmica de cada nodo
    #     con la participación de `Fuente_Energia` medida en `dim_plantas`.
    notas_extra: dict[str, str] = {}
    if not mp.empty and {"Fuente_Energia", "Tecnologia"} <= set(mp.columns):
        term = mp[mp["Tecnologia"].astype(str).str.upper().isin(["TERMICA", "TÉRMICA"])]
        term = term[term["Zona"].notna()]

        def _grupo_fuente(f: Any) -> str:
            t = str(f).upper()
            if "GAS NATURAL" in t:
                return "GAS_CC"
            if any(k in t for k in ("ACPM", "DIÉSEL", "DIESEL", "FUEL", "RECICLAR", "TURBO", "COQUE")):
                return "GAS_OXI"
            return "CARBON"

        if not term.empty and "Fuente_Energia" in term.columns:
            fr = (term.assign(g=term["Fuente_Energia"].map(_grupo_fuente))
                      .groupby(["Zona", "g"])["CEN_MW"].sum().unstack(fill_value=0.0)
                      .reindex(index=list(cap.index)).fillna(0.0))
            fr = fr.div(fr.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
            piscina = cap["CARBON"] + cap["GAS_CC"] + cap["GAS_OXI"]
            for gcol in ("GAS_CC", "GAS_OXI"):
                if gcol in fr.columns and float(fr[gcol].max()) > 0:
                    add = piscina * fr[gcol].reindex(cap.index).fillna(0.0)
                    cap[gcol] = cap[gcol] + add
                    cap["CARBON"] = (cap["CARBON"] - add).clip(lower=0.0)
            share_gas = 100.0 * float((cap["GAS_CC"] + cap["GAS_OXI"]).sum()) / max(
                float(cap[["CARBON", "GAS_CC", "GAS_OXI"]].to_numpy().sum()), 1e-9)
            if share_gas > 0:
                notas_extra["nota_combustible"] = (
                    f"Piscina térmica repartida por `Fuente_Energia` de dim_plantas: "
                    f"{share_gas:.0f} % del CEN térmico es gas/diésel (0,20-0,24 tCO₂e/MWh) "
                    f"y el resto carbón (0,336).")
    # (2) `CapEfecNeta` no siempre trae EOLICA (en la ventana 2026-07-31→08-30 no viene
    #     ni una fila): la capacidad eólica se dimensiona desde la generación observada
    #     de XM y un factor de planta nominal de 35 %, con Guajira y Orinoquía como
    #     corredores. Se declara en la pestaña, no se esconde.
    if float(cap["EOLICA"].sum()) <= 0.5:
        pk_eol = 0.0
        for kk in ("sistema_v4", "sistema"):
            dd = ctx.get(kk)
            col = next((c for c in (dd.columns if isinstance(dd, pd.DataFrame) else [])
                        if str(c).upper() in ("GEN_EOLICA", "EOLICA_MW")), None) if isinstance(dd, pd.DataFrame) else None
            if col:
                v = pd.to_numeric(dd[col], errors="coerce").dropna().to_numpy(dtype=float)
                if v.size > 24:
                    raw = float(np.percentile(v, 99))
                    # `Gen_EOLICA` de la tabla v4 ya viene en MW; sólo se escala si
                    # llegara en GWh (valores < 1 en un sistema de 10 GW no tiene otra lectura).
                    pk_eol = raw * 1000.0 if raw < 1.0 else raw
                    break
        if pk_eol <= 0:
            try:
                perfil = np.asarray(perfiles_reales(ctx)["eolica"], dtype=float)
                pk_eol = float(np.max(perfil)) * 8.0                  # media diaria → pico horario
            except Exception:                                       # noqa: BLE001
                pk_eol = 0.0
        if pk_eol > 0:
            nom_mw = pk_eol / 0.35
            for z, fr_ in (("GUJ", 0.60), ("ORI", 0.25), ("CAR", 0.15)):
                if z in cap.index:
                    cap.loc[z, "EOLICA"] = float(cap.loc[z, "EOLICA"]) + nom_mw * fr_
            notas_extra["nota_eolica"] = (
                f"XM no publicó CEN eólica en la ventana: se dimensionó {nom_mw:,.0f} MW a "
                f"partir del percentil 99 de la generación eólica horaria observada "
                f"({pk_eol:,.0f} MW) ÷ FP 0,35, repartida GUJ 60 % / ORI 25 % / CAR 15 %.")
    if solar_pct:
        cap["SOLAR"] = cap["SOLAR"] * (1.0 + float(solar_pct) / 100.0)
    if eolica_pct:
        cap["EOLICA"] = cap["EOLICA"] * (1.0 + float(eolica_pct) / 100.0)

    filas_u = []
    for z in nodos:
        for tec in tec_cols:
            pmax = float(cap.loc[z, tec])
            if pmax <= 0.5:
                continue
            filas_u.append(dict(nodo=z, bloque=f"{z}-{tec}", tecnologia=tec, Pmax_MW=pmax,
                               Pmin_MW=(0.06 * pmax if tec in ("CARBON", "GAS_OXI", "GAS_CC") else 0.0),
                               prima=float(PRIMA_NODAL[tec]), variable=tec in ("SOLAR", "EOLICA"),
                               despachable=tec not in ("SOLAR", "EOLICA")))
    unidades = pd.DataFrame(filas_u)
    w = np.array([pesos_dem.get(z, 0.0) for z in nodos], dtype=float)
    if w.sum() <= 0:
        w = np.ones(len(nodos))
    w = w / w.sum()
    SF, r_pu = matriz_ptdf(nodos, df_ram)
    nodos_df = pd.DataFrame([dict(id=z, nombre=nombres.get(z, z), x=float(coords.get(z, (0.5, 0.5))[0]),
                                  y=float(coords.get(z, (0.5, 0.5))[1]), CEN_MW=float(cap.loc[z].sum()),
                                  share_demanda=float(w[i]),
                                  **{t: float(cap.loc[z, t]) for t in tec_cols})
                             for i, z in enumerate(nodos)])
    grupos: dict[str, list[int]] = {}
    for i, z in enumerate(nodos):
        grupos.setdefault(ZONA_PADRE_NODAL.get(z, z), []).append(i)
    return dict(nodos=nodos, nodos_df=nodos_df, ramas=df_ram, unidades=unidades, cap=cap,
                SF=SF, r_pu=r_pu, peso_demanda=w, cobertura_toponimia=float(cobertura),
                granularidad=granularidad, curvatura=float(curvatura), mapa=mp,
                grupos_zonales=dict(grupos), tec_cols=list(tec_cols), **notas_extra)


def valida_red(red: dict) -> pd.DataFrame:
    """Chequeos de cordura de la red antes de usarla (nunca revienta la app)."""
    filas = []
    nod, ram, un = red["nodos_df"], red["ramas"], red["unidades"]
    filas.append(dict(check="nodos con capacidad > 0", valor=int((nod["CEN_MW"] > 0).sum()),
                      esperado=len(red["nodos"]), ok=bool((nod["CEN_MW"] > 0).all())))
    filas.append(dict(check="ramas con MW_max > 0", valor=int((ram["MW_max"] > 0).sum()),
                      esperado=len(ram), ok=bool((ram["MW_max"] > 0).all())))
    filas.append(dict(check="bloques con Pmax ≥ Pmin", valor=int((un["Pmax_MW"] >= un["Pmin_MW"]).sum()),
                      esperado=len(un), ok=bool((un["Pmax_MW"] >= un["Pmin_MW"]).all())))
    filas.append(dict(check="share de demanda suma 1", valor=round(float(red["peso_demanda"].sum()), 6),
                      esperado=1.0, ok=bool(abs(float(red["peso_demanda"].sum()) - 1.0) < 1e-6)))
    sf_ok = bool(np.isfinite(red["SF"]).all())
    filas.append(dict(check="PTDF finita", valor=int(sf_ok), esperado=1, ok=sf_ok))
    filas.append(dict(check="|SF| ≤ 1,05 (PTDF acotada)", valor=round(float(np.abs(red["SF"]).max()), 4),
                      esperado="≤1,05", ok=bool(np.abs(red["SF"]).max() <= 1.05)))
    filas.append(dict(check="cobertura toponímica de la CEN (%)", valor=round(red["cobertura_toponimia"], 1),
                      esperado=">60", ok=bool(red["cobertura_toponimia"] > 60.0 or red["mapa"].empty)))
    return pd.DataFrame(filas)


# ------------------ 10.3 · perfiles reales y calibración de oferta ----------

def perfiles_reales(ctx: dict) -> dict[str, np.ndarray]:
    """(24,) de demanda, precio y mezcla tecnológica horarios promediados de la ventana.

    Es el anclaje a datos XM de todo el módulo: la forma del día (valle 04 h, rampa
    de la tarde, pico 20 h) y el nivel del precio uninodal salen de `sistema`, no de
    un perfil de libro. Si la ventana está vacía se usa un perfil nominal declarado.
    """
    sis = ctx.get("sistema")
    out = {k: np.zeros(24) for k in ("demanda", "precio", "solar", "hidro", "termica", "eolica",
                                     "demanda_sd", "precio_sd", "margen", "escasez")}
    out["fuente"] = None          # type: ignore[assignment]
    if not isinstance(sis, pd.DataFrame) or sis.empty or "Hora" not in sis.columns:
        base = np.array([9449, 9149, 8924, 8790, 8873, 9105, 9132, 9507, 9898, 10213, 10533,
                         10838, 10892, 11010, 11163, 11196, 11122, 11049, 11610, 11842, 11626,
                         11186, 10609, 10015], dtype=float)
        out["demanda"], out["demanda_sd"] = base, np.full(24, float(base.std()) * 0.30)
        out["precio"], out["precio_sd"] = np.full(24, 864.0), np.full(24, 55.0)
        out["solar"] = np.array([0, 0, 0, 0, 1, 4, 261, 1094, 1836, 2285, 2482, 2572, 2459, 2275,
                                 1927, 1427, 962, 264, 5, 2, 3, 0, 0, 0], dtype=float)
        out["termica"] = np.full(24, 2800.0)
        out["hidro"] = np.full(24, 6700.0)
        out["eolica"] = np.full(24, 28.0)
        out["fuente"] = "perfil nominal del tablero (sin `sistema` en la ventana)"   # type: ignore[assignment]
        return out
    g = sis.groupby("Hora")
    cols = set(sis.columns)

    def prom(col):
        if col not in cols:
            return np.zeros(24)
        return g[col].mean().reindex(range(1, 25)).fillna(0.0).to_numpy(dtype=float)

    def sd(col):
        if col not in cols:
            return np.zeros(24)
        return g[col].std().reindex(range(1, 25)).fillna(0.0).to_numpy(dtype=float)

    out["demanda"] = prom("Demanda_MW")
    out["demanda_sd"] = sd("Demanda_MW")
    if "Precio_Bolsa_COP_kWh" in cols:
        out["precio"] = prom("Precio_Bolsa_COP_kWh")
        out["precio_sd"] = sd("Precio_Bolsa_COP_kWh")
    elif "Costo_Marginal_COP_kWh" in cols:
        out["precio"] = prom("Costo_Marginal_COP_kWh")
        out["precio_sd"] = sd("Costo_Marginal_COP_kWh")
    for src, dst in (("Gen_SOLAR", "solar"), ("Gen_HIDRAULICA", "hidro"),
                     ("Gen_TERMICA", "termica"), ("Gen_EOLICA", "eolica"),
                     ("Costo_Marginal_COP_kWh", "margen"), ("Precio_Escasez_COP_kWh", "escasez")):
        out[dst] = prom(src)
    if float(out["demanda"].sum()) <= 0:
        return perfiles_reales({})                     # perfil nominal declarado
    out["fuente"] = f"`sistema` de XM · {int(sis['Fecha'].nunique())} día(s) × 24 h"   # type: ignore[assignment]
    return out


def fp_reales_por_hora(ctx: dict) -> pd.DataFrame:
    """Factor de planta real por tecnología y hora (con su dispersión), desde `gen_enr`.

    Con esto el modelo estocástico no inventa la variabilidad del recurso: la mide.
    `fp = Generacion_kWh / (CEN_kW·1000)` por planta y hora, agregado solo con horas
    cuya capacidad instalada es ≥ 0,5 MW; se reporta `n` para que se sepa cuánta
    muestra hay detrás de cada ajuste (la validación exige datos históricos).
    """
    g, cap = ctx.get("gen_enr"), ctx.get("cen_recurso")
    if not isinstance(g, pd.DataFrame) or g.empty or "Tecnologia" not in g.columns:
        return pd.DataFrame()
    if not isinstance(cap, pd.DataFrame) or "Potencia_Confiable_kW" not in cap.columns:
        return pd.DataFrame()
    c = cap.groupby("Codigo_Planta")["Potencia_Confiable_kW"].max().div(1000.0)
    gg = g.copy()
    gg["CEN_MW"] = gg["Codigo_Planta"].map(c)
    gg = gg[gg["CEN_MW"].fillna(0.0) > 0.5]
    if gg.empty:
        return pd.DataFrame()
    gg["fp"] = (gg["Generacion_kWh"] / (gg["CEN_MW"] * 1000.0)).clip(0.0, 1.35)
    gg = gg[np.isfinite(gg["fp"])]
    if gg.empty:
        return pd.DataFrame()
    out = (gg.groupby(["Tecnologia", "Hora"], dropna=False)["fp"]
             .agg(media="mean", sd="std", n="count", q05=lambda s: s.quantile(0.05),
                  q50=lambda s: s.median(), q95=lambda s: s.quantile(0.95),
                  pico=lambda s: s.max()).reset_index())
    out["cv"] = (out["sd"] / out["media"].replace(0.0, np.nan)).fillna(0.0)
    return out


def matriz_recursos(ctx: dict) -> pd.DataFrame:
    """Versión tabular simple: media/CV del FP real por tecnología × hora (o vacío)."""
    r = fp_reales_por_hora(ctx)
    if isinstance(r, tuple):
        return r[0]
    return pd.DataFrame()


def factores_de_planta_por_zona(ctx: dict, red: dict) -> pd.DataFrame:
    """FP medio por (subregión, hora, tecnología) con la red ya construida.

    Une la toponimia (planta→zona) con `gen_enr` (planta→hora) para que el recurso
    estocástico sea *por nodo*: la solar de Guajira no se comporta como la del
    Centro, y eso es justamente lo que un mercado nodal precia.
    """
    mp = red.get("mapa")
    g = ctx.get("gen_enr")
    if not isinstance(mp, pd.DataFrame) or mp.empty or not isinstance(g, pd.DataFrame) or g.empty:
        return pd.DataFrame()
    zmap = dict(zip(mp["Codigo_Planta"], mp["Zona"]))
    gg = g.copy()
    gg["Zona"] = gg["Codigo_Planta"].map(zmap)
    gg = gg[gg["Zona"].notna()]
    if "CEN_MW" not in gg.columns:
        cap = ctx.get("cen_recurso")
        if isinstance(cap, pd.DataFrame) and "Potencia_Confiable_kW" in cap.columns:
            c = cap.groupby("Codigo_Planta")["Potencia_Confiable_kW"].max().div(1000.0)
            gg["CEN_MW"] = gg["Codigo_Planta"].map(c)
    if "CEN_MW" not in gg.columns:
        return pd.DataFrame()
    gg = gg[gg["CEN_MW"].fillna(0.0) > 0.5]
    gg["fp"] = (gg["Generacion_kWh"] / (gg["CEN_MW"] * 1000.0)).clip(0.0, 1.3)
    gg["tec_n"] = gg["Tecnologia"].map(TEC_XM_A_NODAL)
    out = (gg.groupby(["Zona", "tec_n", "Hora"], dropna=False)["fp"]
             .agg(media="mean", sd="std", n="count", q10=lambda s: s.quantile(0.10),
                  q90=lambda s: s.quantile(0.90)).reset_index())
    out["cv"] = (out["sd"] / out["media"].replace(0.0, np.nan)).fillna(0.0)
    out = out[out["n"] >= 4]
    for z in red["nodos"]:                      # garantiza que todo nodo aparezca en la gráfica
        for tec in ("SOLAR", "EOLICA", "HIDRO", "CARBON"):
            if not ((out["Zona"] == z) & (out["tec_n"] == tec)).any():
                out.loc[len(out)] = [z, tec, 0, np.nan, np.nan, 0, np.nan, np.nan, 0.0]
    return out


def perfiles_nodales(red: dict, perfiles: dict[str, np.ndarray], *,
                     horas: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(demanda, solar, eólica) por nodo y hora (24 × n) a partir de los promedios de XM.

    Conveniente y declarado: el perfil horario del sistema se reparte entre nodos
    en proporción a la CEN de la tecnología en cada nodo (`red["cap"]`) y a la
    participación de demanda (`red["peso_demanda"]`). Es el escenario "día medio";
    `gen_escenarios` lo que hace es multiplicarlo por trayectorias estocásticas.
    """
    h = np.arange(24) if horas is None else np.asarray(horas, dtype=int) % 24
    n = len(red["nodos"])
    w = red["peso_demanda"]
    dem = np.asarray(perfiles["demanda"], dtype=float)[h][:, None] * w[None, :]
    out = [dem]
    for key, tec in (("solar", "SOLAR"), ("eolica", "EOLICA")):
        cap = red["cap"][tec].to_numpy(dtype=float)
        tot = float(cap.sum())
        prof = np.asarray(perfiles[key], dtype=float)[h]
        if tot > 0:
            fp = np.clip(prof / max(float(np.max(prof)), 1e-9), 0.0, 1.0)
            out.append(fp[:, None] * cap[None, :])
        else:
            out.append(np.zeros((len(h), n)))
    return out[0], out[1], out[2]


def calibra_oferta(red: dict, perfiles: dict[str, np.ndarray], *, use_opf: bool = True,
                   por_hora: bool = True, rondas: int = 9, tol_pct: float = 0.4,
                   **kw) -> dict[str, Any]:
    """Nivel de oferta (COP/kWh del bloque de referencia) que reproduce el precio real de XM.

    Dos etapas, ambas reportadas en la pestaña:

    1. Ajuste analítico por mínimos cuadrados: κ = Σ(p_real·λ0)/Σλ0² sobre la
       bisección uninodal (sin red), que fija forma y nivel de arranque.
    2. Si `use_opf`, se refina κ biseccionando el **nivel dentro del propio
       `opf_nodal`** para que la media del λ del modelo iguale la media del
       `Precio_Bolsa_COP_kWh` de la ventana. Sin esta etapa el precio del modelo
       sale sistemáticamente bajo, porque las variables despachan a costo ~0 y
       desplazan térmica: la calibración debe ver lo mismo que ve el liquidador.

    Devuelve además MAE, R², sesgo y el residuo de calibración en %, para que el
    lector juzgue el modelo.
    """
    precio_real = np.asarray(perfiles["precio"], dtype=float)
    valido = precio_real > 0.0
    nivel0 = float(np.median(precio_real[valido])) if np.any(valido) else 800.0
    if not np.any(valido):
        return dict(kappa=1.0, nivel=nivel0, nivel0=nivel0, mae=float("nan"), r2=float("nan"),
                    sesgo=float("nan"), n=0, curvatura=float(NODAL_CURVATURA), pred_medio=float("nan"),
                    real_medio=float("nan"), ok=False, nota="`sistema` sin Precio_Bolsa en la ventana")
    un = red["unidades"].loc[~red["unidades"]["variable"]].reset_index(drop=True)
    prima = un["prima"].to_numpy(dtype=float)

    # etapa 1 — analítica
    dem_tot = np.asarray(perfiles["demanda"], dtype=float)
    ren_tot = np.asarray(perfiles["solar"], dtype=float) + np.asarray(perfiles["eolica"], dtype=float)
    dem_neta = np.maximum(dem_tot - ren_tot, 1.0)
    c1_0 = nivel0 * prima
    lam0 = _lambda_biseccion(dem_neta, c1_0, np.maximum(c1_0 * NODAL_CURVATURA / np.maximum(un["Pmax_MW"].to_numpy(float), 1.0), 1e-7),
                             un["Pmax_MW"].to_numpy(dtype=float), un["Pmin_MW"].to_numpy(dtype=float))
    num = float(np.sum(precio_real * np.maximum(lam0, 1e-9)))
    den = float(np.sum(np.maximum(lam0, 1e-9) ** 2))
    kappa = num / max(den, 1e-9)
    nivel = float(nivel0 * kappa)
    real_medio = float(np.mean(precio_real))
    pred_medio = float(np.mean(lam0 * kappa))
    nota = "κ analítica (etapa 1)"

    # etapa 2a — curva κ_h: reproduce también la forma intradía del precio real
    nivel_h = None
    if por_hora:
        lam0_h = np.asarray(lam0, dtype=float)
        kap_h = np.clip(precio_real / np.maximum(lam0_h, 1e-6), 0.35, 3.0)
        nivel_h = nivel0 * kap_h
        pred_medio = float(np.mean(lam0_h * kap_h))
        nota = "κ_h por hora sobre la bisección uninodal (etapa 2a)"

    # etapa 2b — dentro del OPF
    if use_opf:
        dem, sol, eol = perfiles_nodales(red, perfiles)
        obj = real_medio

        def media_lambda(niv: float) -> float:
            r = opf_nodal(red, demanda=dem, solar=sol, eolica=eol, nivel_cop_kwh=niv, **kw)
            return float(np.mean(r["lam"]))

        if nivel_h is None:
            lo, hi = 0.25 * nivel0, 3.0 * nivel0
            for _ in range(int(rondas)):
                mid = 0.5 * (lo + hi)
                if media_lambda(mid) < obj:
                    lo = mid
                else:
                    hi = mid
                if abs(media_lambda(0.5 * (lo + hi)) - obj) / max(obj, 1e-9) * 100.0 < tol_pct:
                    break
            nivel = float(0.5 * (lo + hi))
            r = opf_nodal(red, demanda=dem, solar=sol, eolica=eol, nivel_cop_kwh=nivel, **kw)
            resid = precio_real - np.asarray(r["lam"], dtype=float)
            nota = "κ calibrada sobre el propio OPF (etapa 2b, nivel único)"
        else:
            # se preserva la forma κ_h y se escala en bloque hasta acertar la media
            esc_obj = 1.0
            lo, hi = 0.4, 2.5
            for _ in range(int(rondas)):
                esc = 0.5 * (lo + hi)
                if media_lambda(nivel_h * esc) < obj:
                    lo = esc
                else:
                    hi = esc
            esc_obj = 0.5 * (lo + hi)
            nivel_h = nivel_h * esc_obj
            r = opf_nodal(red, demanda=dem, solar=sol, eolica=eol, nivel_cop_kwh=nivel_h, **kw)
            resid = precio_real - np.asarray(r["lam"], dtype=float)
            nota = ("κ_h calibrada sobre el propio OPF (etapa 2b, curva horaria; "
                    f"escala global ×{esc_obj:.3f})")
        nivel = float(np.mean(nivel_h)) if nivel_h is not None else nivel
        pred_medio = float(np.mean(r["lam"]))
        mae = float(np.mean(np.abs(resid)))
        ss_tot = float(np.sum((precio_real - real_medio) ** 2))
    else:
        pmo = un["Pmax_MW"].to_numpy(dtype=float)
        pmi = un["Pmin_MW"].to_numpy(dtype=float)
        nivel_uso = (nivel0 * kap_h) if nivel_h is not None else nivel
        if np.ndim(nivel_uso) == 0:
            cc1 = float(nivel_uso) * prima
            cc2 = np.maximum(cc1 * NODAL_CURVATURA / np.maximum(pmo, 1.0), 1e-7)
            pred_h = np.atleast_1d(_lambda_biseccion(dem_neta, cc1, cc2, pmo, pmi))
        else:
            pred_h = np.array([
                _lambda_biseccion(dem_neta[j:j + 1], nv * prima,
                                  np.maximum(nv * prima * NODAL_CURVATURA / np.maximum(pmo, 1.0), 1e-7),
                                  pmo, pmi)[0] for j, nv in enumerate(nivel_uso)])
            nivel = float(np.mean(nivel_uso))
        nivel_h = np.asarray(nivel_uso, dtype=float) if np.ndim(nivel_uso) else None
        resid = precio_real - pred_h[:len(precio_real)]
        mae = float(np.mean(np.abs(resid)))
        ss_tot = float(np.sum((precio_real - real_medio) ** 2))
        pred_medio = float(np.mean(pred_h))
    r2 = float(1.0 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot > 1e-9 else float("nan")
    return dict(kappa=float(nivel / max(nivel0, 1e-9)), nivel=nivel, nivel_h=nivel_h,
                nivel0=nivel0, mae=mae, r2=r2,
                sesgo=float(np.mean(resid)), n=int(len(precio_real)), curvatura=float(NODAL_CURVATURA),
                pred_medio=pred_medio, real_medio=real_medio, ok=True,
                error_calibracion_pct=100.0 * (pred_medio - real_medio) / max(real_medio, 1e-9),
                nota=nota)


# ------------------ 10.4 · distribuciones del recurso ----------------------
#
# Convención única: TODAS las familias se parametrizan por (media, desviación
# típica) de la muestra real con el método de momentos —la vía que sigue la
# usada cuando no hay una serie larga de sitio. Así se pueden
# superponer varias familias sobre el mismo conjunto de datos y compararlas con
# la misma base.

DISTRIBUCIONES_SOL = ("lognormal", "beta", "gamma")
DISTRIBUCIONES_EOL = ("weibull", "rayleigh", "mezcla_uniforme")


def parametros_por_momentos(tipo: str, mu: float, sd: float) -> dict[str, float]:
    """Parámetros de cada familia a partir de media y desviación típica."""
    mu = max(float(mu), 1e-6)
    sd = max(float(sd), 1e-9)
    cv = sd / mu
    t = str(tipo).lower()
    if t == "lognormal":
        s2 = math.log(1.0 + cv * cv)
        return dict(sigma=math.sqrt(s2), mu_ln=math.log(mu) - 0.5 * s2, cv=cv)
    if t == "beta":
        mu_c = min(max(mu, 1e-3), 0.999)
        var = min(max(sd * sd, 1e-8), mu_c * (1.0 - mu_c) * 0.999)
        k = max(mu_c * (1.0 - mu_c) / var - 1.0, 0.05)
        return dict(alfa=mu_c * k, beta=(1.0 - mu_c) * k)
    if t == "gamma":
        k = max(1.0 / (cv * cv), 0.05)
        return dict(k=k, theta=mu / k)
    if t == "weibull":
        def _cv_de(k: float) -> float:
            m = math.exp(math.lgamma(1.0 + 1.0 / k))
            v2 = math.exp(math.lgamma(1.0 + 2.0 / k)) - m * m
            return math.sqrt(max(v2, 0.0)) / m
        obj = min(max(cv, 1e-3), 1.9)
        lo, hi = 0.35, 14.0
        for _ in range(70):                      # CV(k) es decreciente → subir k si sobra CV
            mid = 0.5 * (lo + hi)
            if _cv_de(mid) > obj:
                lo = mid
            else:
                hi = mid
        k = 0.5 * (lo + hi)
        lam = mu / max(math.exp(math.lgamma(1.0 + 1.0 / k)), 1e-9)
        return dict(beta=k, lambda_=lam, cv_ajustado=_cv_de(k))
    if t == "rayleigh":
        vm = mu / math.sqrt(math.pi / 2.0)
        return dict(Vm=vm, sigma=vm / math.sqrt(2.0))
    if t == "mezcla_uniforme":
        ancho = sd * math.sqrt(3.0)              # var = w²/3 para dos U de ancho w separadas w
        return dict(centro=mu, semiancho=ancho / 2.0, ancho=ancho)
    raise ValueError(f"familia desconocida: {tipo}")


def densidad_recurso(tipo: str, x: np.ndarray, mu: float, sd: float,
                     *, normaliza: bool = True) -> tuple[np.ndarray, np.ndarray, str, dict]:
    """(pdf, cdf, fórmula legible, parámetros) de la familia pedida sobre la malla `x`."""
    xx = np.asarray(x, dtype=float)
    par = parametros_por_momentos(tipo, mu, sd)
    t = str(tipo).lower()
    if t == "lognormal":
        z = (np.log(np.maximum(xx, 1e-12)) - par["mu_ln"]) / par["sigma"]
        pdf = np.exp(-0.5 * z * z) / (np.maximum(xx, 1e-12) * par["sigma"] * math.sqrt(2.0 * math.pi))
        cdf = 0.5 * (1.0 + _erf(z / math.sqrt(2.0)))
        form = f"Log-normal(μ={par['mu_ln']:.3f}, σ={par['sigma']:.3f}) → E[X]={mu:.3f}, CV={par['cv']:.3f}"
    elif t == "beta":
        a, b = par["alfa"], par["beta"]
        u = np.clip(xx, 0.0, 1.0)
        lnB = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        pdf = np.exp((a - 1.0) * np.log(np.maximum(u, 1e-12)) +
                     (b - 1.0) * np.log(np.maximum(1.0 - u, 1e-12)) - lnB)
        pdf = np.where((xx >= 0.0) & (xx <= 1.0), pdf, 0.0)
        cdf = _beta_cdf(u, a, b)
        form = f"Beta(α={a:.2f}, β={b:.2f}) sobre [0,1] — distribución de irradiancia (Fatemi et al., 2018)"
    elif t == "gamma":
        k, th = par["k"], par["theta"]
        g = np.maximum(xx, 1e-9)
        pdf = np.exp((k - 1.0) * np.log(g) - g / th - math.lgamma(k) - k * math.log(th))
        pdf = np.where(xx >= 0.0, pdf, 0.0)
        cdf = _gamma_cdf(g / th, k)
        form = f"Gamma(k={k:.2f}, θ={th:.3f}) — gamma modificada (Bracale et al., 2013)"
    elif t == "weibull":
        be, lam = par["beta"], par["lambda_"]
        g = np.maximum(xx, 0.0)
        pdf = (be / lam) * np.power(g / lam, be - 1.0) * np.exp(-np.power(g / lam, be))
        cdf = 1.0 - np.exp(-np.power(g / lam, be))
        form = f"Weibull(k={be:.2f}, c={lam:.2f}) — F(x)=1−exp(−(x/c)^k)"
    elif t == "rayleigh":
        vm = par["Vm"]
        g = np.maximum(xx, 0.0)
        pdf = (math.pi / 2.0) * (g / vm ** 2) * np.exp(-(math.pi / 4.0) * (g / vm) ** 2)
        cdf = 1.0 - np.exp(-(math.pi / 4.0) * np.power(g / vm, 2.0))
        form = f"Rayleigh(Vm={vm:.2f}) — caso Weibull con β=2 (Paraschiv et al. 2019)"
    elif t == "mezcla_uniforme":
        c, hw = par["centro"], max(par["semiancho"], 1e-6)
        i1 = ((xx >= c - 1.5 * hw) & (xx < c - 0.5 * hw)).astype(float)
        i2 = ((xx >= c + 0.5 * hw) & (xx <= c + 1.5 * hw)).astype(float)
        pdf = 0.5 / hw * (i1 + i2)
        cdf = np.clip(0.5 * np.clip((xx - (c - 1.5 * hw)) / hw, 0.0, 1.0) +
                      0.5 * np.clip((xx - (c + 0.5 * hw)) / hw, 0.0, 1.0), 0.0, 1.0)
        form = (f"Mezcla de 2 uniformes en {c:.2f}±{hw:.2f} — aproximación analítica simplificada "
                f"(Acero et al., 2024)")
    else:
        raise ValueError(f"familia desconocida: {tipo}")
    if normaliza:
        area = float(np.trapezoid(pdf, xx)) if xx.size > 2 else 0.0
        if area > 1e-9 and abs(area - 1.0) > 1e-3:
            pdf = pdf / area
            cdf = np.clip(np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(xx))]), 0.0, 1.0)
    return pdf, cdf, form, par


def _erf(z: np.ndarray) -> np.ndarray:
    """erf por Abramowitz-Stegun 7.1.26 (sin scipy; |error| < 1,5e-7)."""
    sign = np.sign(z)
    x = np.abs(z)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
                + 0.254829592) * t * np.exp(-x * x))
    return sign * y


def _beta_cdf(x: np.ndarray, p: float, q: float, puntos: int = 800) -> np.ndarray:
    g = np.linspace(0.0, 1.0, puntos + 1)
    lnB = math.lgamma(p) + math.lgamma(q) - math.lgamma(p + q)
    f = np.exp((p - 1.0) * np.log(np.maximum(g, 1e-12)) +
               (q - 1.0) * np.log(np.maximum(1.0 - g, 1e-12)) - lnB)
    if p < 1.0:
        f[0] = f[1]
    if q < 1.0:
        f[-1] = f[-2]
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(g))])
    return np.interp(np.clip(x, 0.0, 1.0), g, np.clip(F / max(float(F[-1]), 1e-12), 0.0, 1.0))


def _gamma_cdf(x: np.ndarray, k: float, puntos: int = 900) -> np.ndarray:
    tope = max(float(np.nanmax(x)) * 1.05 + 4.0, 6.0)
    g = np.linspace(0.0, tope, puntos + 1)
    f = np.exp((k - 1.0) * np.log(np.maximum(g, 1e-12)) - g - math.lgamma(k))
    f[0] = 0.0 if k >= 1.0 else f[1]
    F = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(g))])
    return np.interp(np.maximum(x, 0.0), g, np.clip(F / max(float(F[-1]), 1e-12), 0.0, 1.0))


def curva_potencia_pv(g: np.ndarray, *, punto_lineal: float = 0.55) -> np.ndarray:
    """Curva FV sin seguimiento: rampa lineal y saturación suave (monótona).

    Literatura de recurso eólico: «la curva de potencia es aproximadamente lineal hasta cierto nivel de
    irradiancia y luego se vuelve cuadrática a mayores niveles … por calentamiento
    y saturación del panel». Se implementa con la forma monótona
    P/Pn = (1 − e^(−a·G))/(1 − e^(−a)), a = 1/`punto_lineal`: su desarrollo de
    Taylor es lineal en G baja y cuadrático-cóncavo en G alta —la morfología que
    declarada en el módulo, sin saltos ni tramos decrecientes.
    """
    gg = np.clip(np.asarray(g, dtype=float), 0.0, 2.0)
    a = 1.0 / max(float(punto_lineal), 1e-6)
    return np.clip((1.0 - np.exp(-a * gg)) / (1.0 - math.exp(-a)), 0.0, 1.02)


def curva_potencia_eolica(v: np.ndarray, *, v_corte: float = 3.0, v_nom: float = 12.0,
                          v_salida: float = 25.0) -> np.ndarray:
    """Curva eólica cúbica entre corte y nominal; 1 hasta la velocidad de salida."""
    vv = np.maximum(np.asarray(v, dtype=float), 0.0)
    p = np.clip((vv ** 3 - v_corte ** 3) / max(v_nom ** 3 - v_corte ** 3, 1e-9), 0.0, 1.0)
    return np.where(vv >= v_salida, 0.0, np.where(vv <= v_corte, 0.0, p))


def momentos(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return dict(media=float("nan"), sd=float("nan"), cv=float("nan"), skew=float("nan"),
                    kurtosis=float("nan"), n=int(v.size))
    mu = float(v.mean())
    sd = float(v.std(ddof=1)) if v.size > 1 else 0.0
    z = (v - mu) / max(sd, 1e-12)
    return dict(media=mu, sd=sd, cv=sd / max(abs(mu), 1e-9), skew=float(np.mean(z ** 3)),
                kurtosis=float(np.mean(z ** 4) - 3.0), n=int(v.size))


def ks_ajuste(muestra: np.ndarray, tipo: str, grid: np.ndarray) -> float:
    """Distancia de Kolmogórov-Smirnov entre la muestra y la familia ajustada."""
    mu, sd = momentos(np.asarray(muestra, dtype=float))["media"], momentos(muestra)["sd"]
    _pdf, cdf, _f, _p = densidad_recurso(tipo, grid, mu, sd)
    xs = np.sort(np.asarray(muestra, dtype=float))
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return float("nan")
    F_emp = np.arange(1, xs.size + 1) / xs.size
    return float(np.max(np.abs(F_emp - np.interp(xs, grid, cdf))))


def ucf_esperado(avail: np.ndarray, pronostico: np.ndarray, *, precio_reserva: float,
                 precio_recorte: float, horas: int = 24) -> dict[str, Any]:
    """Función de costo de incertidumbre (UCF) hora a hora.

    UCF = π_reserva·E[déficit] + π_recorte·E[exceso], con déficit = max(0, A−P) y
    exceso = max(0, P−A): la forma analítica de (Perez et al., 2023) para
    sobre/subestimación y de (Acero et al., 2024) para el viento, aplicada a las
    trayectorias Monte Carlo del módulo. π_reserva se toma del costo de la unidad
    flexible más barata y π_recorte del precio nodal esperado de la hora.
    """
    a = np.asarray(avail, dtype=float)
    p = np.broadcast_to(np.asarray(pronostico, dtype=float), a.shape)
    deficit = np.maximum(a - p, 0.0)
    exceso = np.maximum(p - a, 0.0)
    coste = precio_reserva * deficit + precio_recorte * exceso
    return dict(deficit_MW=deficit, exceso_MW=exceso, coste_COP_h=coste,
                deficit_medio_MWh=float(np.mean(deficit, axis=0) * 1000.0),
                exceso_medio_MWh=float(np.mean(exceso, axis=0) * 1000.0),
                coste_medio_COP_h=float(np.mean(coste)),
                coste_total_COP=float(np.sum(coste)),
                por_hora_COP=(np.sum(coste, axis=0) / max(a.shape[0], 1)) if a.ndim > 1 else coste)


# --------------- 10.5 · generación estocástica de escenarios ----------------

def _ar1(rng: np.ndarray, rho: float, B: int, H: int) -> np.ndarray:
    """Ruido AR(1) persistente (pasos de nube / regímenes de viento), (B,H)."""
    e = rng.standard_normal((B, H))
    out = np.empty_like(e)
    a = float(np.clip(rho, 0.0, 0.995))
    out[:, 0] = e[:, 0]
    for h in range(1, H):
        out[:, h] = a * out[:, h - 1] + math.sqrt(max(1.0 - a * a, 1e-9)) * e[:, h]
    return out * math.sqrt(max(1.0 - a * a, 1e-9) / max(1.0 - a * a, 1e-9))


def _muestra_familiar(rng, tipo: str, B: int, H: int, mu: np.ndarray, cv: np.ndarray,
                      *, rho: float = 0.6, piso: float = 0.0, techo: float = 3.0) -> np.ndarray:
    """Muestra (B,H) de una familia con media `mu`, CV `cv` y persistencia `rho`.

    Se muestrea una variable estándar de la familia y se re-escala para que la
    media y el CV empíricos coincidan con los de la serie real: así el efecto del
    recurso es comparable entre familias (que es la pregunta del objetivo a).
    """
    z = np.empty((B, H))
    ar = _ar1(rng, rho, B, H)
    if tipo == "beta":
        a, b = 2.2, 3.2
        z = rng.beta(a, b, size=(B, H)) * 0.55 + 0.45 * np.tanh(ar / 2.2) * 0.5 + 0.2
    elif tipo == "gamma":
        k = 3.0
        z = rng.gamma(k, 1.0 / k, size=(B, H)) * (0.6 + 0.4 * np.exp(ar / 3.0))
    elif tipo == "weibull":
        k = 2.0
        z = rng.weibull(k, size=(B, H)) * (0.65 + 0.35 * np.exp(ar / 3.2))
    elif tipo == "rayleigh":
        z = rng.rayleigh(1.0, size=(B, H)) * (0.7 + 0.3 * np.exp(ar / 3.4))
    elif tipo == "mezcla_uniforme":
        u = rng.uniform(size=(B, H))
        z = 0.75 + 0.55 * np.sign(u - 0.5) * (0.5 + np.abs(u - 0.5)) * (1.0 + 0.25 * ar)
    else:                                    # log-normal (por defecto)
        s = math.log(1.0 + 0.42 ** 2)
        z = np.exp(s * ar / 1.2) * rng.lognormal(-0.5 * s, math.sqrt(s), size=(B, H))
    mm = momentos(z.ravel())
    if not np.isfinite(mm["media"]) or mm["media"] <= 0 or mm["sd"] <= 0:
        return np.broadcast_to(mu[None, :], (B, H)).astype(float)
    escalado = (z - mm["media"]) / mm["sd"] * np.maximum(cv, 1e-6)[None, :] + mu[None, :]
    return np.clip(escalado, piso, techo)


def gen_escenarios(red: dict, perfiles: dict[str, np.ndarray], *, horas: np.ndarray | None = None,
                   n_esc: int = 200, semilla: int = 7, fam_sol: str = "lognormal",
                   fam_eol: str = "weibull", fam_dem: str = "lognormal", rho_recurso: float = 0.6,
                   rho_cruce: float = -0.15, cv_sol: float | None = None, cv_eol: float | None = None,
                   cv_dem: float | None = None, fp_zona: pd.DataFrame | None = None,
                   factor_fase: dict[str, float] | None = None, curva_pv: float = 0.55) -> dict[str, Any]:
    """Escenarios horarios (n_esc × H × nodos) de solar, eólica y demanda.

    Cada recurso se dibuja de la familia pedida con la media y el CV
    medidos en XM (no asumidos), se le da persistencia horaria (AR-1: pasos de
    nube, frentes fríos) y se correlaciona entre nodos (`rho_recurso`) y entre
    recursos (`rho_cruce`, negativo entre sol y viento: en Colombia el régimen de
    brisas del Caribe y los vientos de la Guajira son anticorrelacionados con la
    nubosidad de interior). `factor_fase` multiplica demanda/CEN por fase ENSO,
    con los Δ% que ya calcula la pestaña 🌊 El Niño (`ctx["impacto"]`).
    """
    rng = np.random.default_rng(int(semilla))
    nodos = list(red["nodos"])
    n = len(nodos)
    H = 24 if horas is None else int(len(horas))
    h = (np.arange(H) % 24) if horas is None else np.asarray(horas, dtype=int) % 24
    S = int(max(4, n_esc))
    B = S                                          # una serie de 24 h por escenario
    cap_sol = red["cap"]["SOLAR"].to_numpy(dtype=float)
    cap_eol = red["cap"]["EOLICA"].to_numpy(dtype=float)
    dem_med = float(np.mean(np.asarray(perfiles["demanda"], dtype=float)[h].clip(min=0)))
    w = red["peso_demanda"]

    # CV medidos (o por defecto, declarados)
    def _cv_de(col: str, tec: str, por_defecto: float) -> float:
        if isinstance(fp_zona, pd.DataFrame) and not fp_zona.empty and tec in set(fp_zona["tec_n"]):
            sub = fp_zona[(fp_zona["tec_n"] == tec) & fp_zona["media"].notna()]
            if len(sub) >= 6:
                return float(np.clip(np.nanmedian(sub["cv"]), 0.05, 1.2))
        return float(por_defecto)

    cv_s = float(cv_sol) if cv_sol is not None else _cv_de("cv", "SOLAR", 0.34)
    cv_e = float(cv_eol) if cv_eol is not None else _cv_de("cv", "EOLICA", 0.42)
    cv_d = float(cv_dem) if cv_dem is not None else max(float(np.nanmean(np.asarray(perfiles["demanda_sd"], dtype=float)[h])
                                                              / max(dem_med, 1e-9)), 0.03)
    cv_d = float(np.clip(cv_d, 0.02, 0.45))

    # El acople espacial se aplica abajo, sobre el índice de recurso: `_muestra_familiar`
    # genera un factor común (todas las horas) y se mezcla con el idiosincrático de cada
    # nodo con peso ρ (ρ=1 recurso idéntico en todo el país, ρ=0 independiente).

    # perfil horario de recurso: FP real medio (si hay) o el nominal del sistema
    fp_sol_h = np.zeros(H)
    fp_eol_h = np.zeros(H)
    for j, hh in enumerate(h):
        fp_sol_h[j] = float(np.asarray(perfiles["solar"], dtype=float)[hh])
        fp_eol_h[j] = float(np.asarray(perfiles["eolica"], dtype=float)[hh])
    cap_tot_sol = float(cap_sol.sum())
    cap_tot_eol = float(cap_eol.sum())
    sol_frac_h = np.clip(fp_sol_h / max(cap_tot_sol, 1e-9), 0.0, 1.0) if cap_tot_sol > 0 else np.zeros(H)
    eol_frac_h = np.clip(fp_eol_h / max(cap_tot_eol, 1e-9), 0.0, 1.0) if cap_tot_eol > 0 else np.full(H, 0.28)
    if float(sol_frac_h.max()) <= 0:
        sol_frac_h = np.maximum(curva_potencia_pv(np.array([(hh + 1) / 24.0 * 2.0 for hh in h])), 0.0)
    if float(eol_frac_h.max()) <= 0:
        eol_frac_h = np.full(H, 0.30)
    # irradiancia normalizada implícita (invierte la curva de potencia)
    g_sol = np.clip(sol_frac_h / max(float(sol_frac_h.max()), 1e-9), 0.0, 1.15)
    idx_sol = _muestra_familiar(rng, fam_sol, B, H, np.ones(H), np.full(H, cv_s), rho=0.72,
                                piso=0.03, techo=1.6)
    v_eol = _muestra_familiar(rng, fam_eol, B, H, np.ones(H), np.full(H, cv_e), rho=0.80,
                              piso=0.0, techo=34.0)
    # correlación cruzada sol↔viento (objetivo a: «incluyendo la correlación»)
    rr = float(np.clip(rho_cruce, -0.9, 0.9))
    zs = (idx_sol - idx_sol.mean()) / max(idx_sol.std(), 1e-9)
    zv = (v_eol - v_eol.mean()) / max(v_eol.std(), 1e-9)
    zv_c = rr * zs + math.sqrt(max(1.0 - rr * rr, 1e-9)) * zv
    v_eol = v_eol.mean() + v_eol.std() * zv_c
    idx_sol = idx_sol.mean() + idx_sol.std() * ((zs + rr * zv) / max(1.0 - rr * rr, 1e-6) ** 0.5)
    idx_sol = np.clip(idx_sol, 0.03, 1.6)
    v_eol = np.clip(v_eol, 0.0, 34.0)

    irr = idx_sol[:, None, :] * np.repeat(g_sol[None, :], n, axis=0)[None, :, :]
    fp_solar = curva_potencia_pv(irr, punto_lineal=curva_pv)   # factor de planta por nido (B,n,H)
    vel_media = np.clip(float(np.mean(eol_frac_h)) * 12.0 + 3.5, 5.0, 12.5)
    vel = (v_eol / max(float(np.mean(v_eol)), 1e-9)) * vel_media
    fp_eolica = curva_potencia_eolica(vel)
    # convención de ejes en TODO el bloque estocástico: (escenario, nodo, hora)
    nido_sol = fp_solar * cap_sol[None, :, None]
    nido_eol = fp_eolica[:, None, :] * cap_eol[None, :, None]   # el viento es común al sistema

    mult_dem = _muestra_familiar(rng, fam_dem, B, H, np.ones(H), np.full(H, cv_d), rho=0.60,
                                 piso=0.75, techo=1.30)
    # acople demanda↔recurso: en Colombia la hora más soleada es también la de mayor
    # demanda (aire acondicionado), así que la anomalía de irradiancia entra con signo +.
    rr_d = float(np.clip(rho_cruce, -0.9, 0.9)) * -0.20
    mult_dem = np.clip(mult_dem * (1.0 + rr_d * (idx_sol - float(np.mean(idx_sol)))
                                   / max(float(np.std(idx_sol)), 1e-9)), 0.75, 1.30)
    dem_h = np.asarray(perfiles["demanda"], dtype=float)[h]
    demanda = np.clip(mult_dem[:, None, :] * dem_h[None, None, :] * w[None, :, None], 0.0, None)

    fase = factor_fase or {}
    if fase:
        demanda = demanda * float(fase.get("demanda", 1.0))
        nido_sol = nido_sol * float(fase.get("solar", 1.0))
        nido_eol = nido_eol * float(fase.get("eolica", 1.0))
        if fase.get("hidro_recorte"):
            nido_sol = nido_sol  # la hidráulica se ajusta en la oferta, no aquí

    vel_eolica_ms = vel
    return dict(S=S, H=H, n=n, horas=h, solar_MW=nido_sol, eolica_MW=nido_eol,
                demanda_MW=demanda, irradiancia_neta=irr, velocidad_mps=vel_eolica_ms,
                fp_solar=fp_solar, fp_eolica=fp_eolica, cv_solar=cv_s, cv_eolica=cv_e,
                cv_demanda=cv_d, rho_recurso=float(rho_recurso), rho_cruce=rr,
                fam_sol=fam_sol, fam_eol=fam_eol, fam_dem=fam_dem, fase=dict(fase),
                demanda_total_MW=demanda.sum(axis=2), mult_demanda=mult_dem)


def reduce_escenarios(esc: dict[str, Any], *, k: int = 10, iters: int = 40,
                      semilla: int = 3) -> dict[str, Any]:
    """Reducción de escenarios por k-medias (Lloyd) con días típicos ponderados.

    Agrupa las trayectorias (solar, eólica, demanda por hora) y devuelve el
    escenario más cercano a cada centroide con su peso = fracción de trayectorias
    del clúster. Es la reducción clásica de la programación estocástica: permite
    resolver el OPF exacto sobre los *días típicos* en vez de sobre las 200
    trayectorias, conservando las medias (momento 1) y aproximando la varianza.
    """
    S = int(esc["S"])
    k = int(max(2, min(k, S)))
    sol_n = np.asarray(esc["solar_MW"], dtype=float).mean(axis=2)     # (S, n)
    eol_n = np.asarray(esc["eolica_MW"], dtype=float).mean(axis=2)
    tot = np.asarray(esc["demanda_total_MW"], dtype=float)             # (S, n)
    rasgos = np.concatenate([
        sol_n / max(float(np.abs(sol_n).mean()), 1e-9),
        eol_n / max(float(np.abs(eol_n).mean()), 1e-9),
        tot / max(float(tot.mean()), 1e-9),
        tot.std(axis=1)[:, None] / max(float(tot.mean()), 1e-9),
        np.array([float(np.mean(x)) for x in np.asarray(esc["demanda_MW"], dtype=float).reshape(S, -1)])[:, None]], axis=1)
    if not np.isfinite(rasgos).all():
        rasgos = np.nan_to_num(rasgos)
    rng = np.random.default_rng(semilla)
    # k-means++ init
    centro = [rasgos[rng.integers(S)]]
    for _ in range(1, k):
        d2 = np.min(np.stack([((rasgos - c) ** 2).sum(axis=1) for c in centro], axis=1), axis=1)
        p = d2 / max(d2.sum(), 1e-12)
        centro.append(rasgos[rng.choice(S, p=p)])
    C = np.stack(centro)
    etiqueta = np.zeros(S, dtype=int)
    for _ in range(int(iters)):
        D = ((rasgos[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
        nuevo = np.argmin(D, axis=1)
        for j in range(k):
            sel = nuevo == j
            if sel.any():
                C[j] = rasgos[sel].mean(axis=0)
        if np.array_equal(nuevo, etiqueta):
            etiqueta = nuevo
            break
        etiqueta = nuevo
    pesos = np.bincount(etiqueta, minlength=k).astype(float)
    pesos = pesos / max(pesos.sum(), 1e-12)
    eleg, dist_tot = [], []
    for j in range(k):
        sel = np.where(etiqueta == j)[0]
        if sel.size == 0:
            sel = np.arange(S)
        dd = ((rasgos[sel] - C[j]) ** 2).sum(axis=1)
        eleg.append(int(sel[int(np.argmin(dd))]))
        dist_tot.append(float(np.sqrt(dd.min())))
    idx = np.array(eleg)
    out = {kk: esc[kk][idx] for kk in ("solar_MW", "eolica_MW", "demanda_MW", "irradiancia_neta",
                                       "velocidad_mps", "fp_solar", "fp_eolica", "mult_demanda")}
    out.update(dict(demanda_total_MW=esc["demanda_total_MW"][idx], horas=esc["horas"], S=k,
                    H=esc["H"], n=esc["n"], pesos=pesos, etiquetas=etiqueta, cluster_idx=idx,
                    radio_medio=np.array(dist_tot), inercia=float(np.sqrt(
                        ((rasgos - C[etiqueta]) ** 2).sum(axis=1).mean()))))
    return out


# ------------------- 10.6 · DC-OPF con duals (precio nodal) ----------------

def _oferta_blocos(red: dict, nivel_cop_kwh, *, curva: float | None = None) -> dict[str, Any]:
    """Curvas de oferta c1 (COP/kWh) y c2 (COP/kWh por MW) de los bloques despachables.

    `nivel_cop_kwh` puede ser un escalar (nivel único para todo el día) o un vector
    (H,) con el nivel por hora: lo que produce `calibra_oferta(por_hora=True)` al
    ajustar la curva marginal intradiaria contra `Precio_Bolsa_COP_kWh` de XM.
    """
    un = red["unidades"].loc[~red["unidades"]["variable"]].reset_index(drop=True)
    curv = float(NODAL_CURVATURA if curva is None else curva)
    pmax = un["Pmax_MW"].to_numpy(dtype=float)
    pmin = un["Pmin_MW"].to_numpy(dtype=float)
    niv = np.asarray(nivel_cop_kwh, dtype=float)
    prima = un["prima"].to_numpy(dtype=float)
    if niv.ndim == 0:
        c1 = float(niv) * prima
        por_hora = False
    else:
        if niv.size < 24:
            niv = np.resize(niv, 24)
        c1 = np.outer(niv[:24], prima)
        por_hora = True
    c2 = np.maximum(np.atleast_2d(c1) * curv / np.maximum(pmax, 1.0)[None, :], 1e-7)
    nodo_idx = {z: i for i, z in enumerate(red["nodos"])}
    return dict(idx=np.array([nodo_idx[z] for z in un["nodo"]], dtype=int), c1=c1, c2=c2,
                pmax=pmax, pmin=pmin, tec=un["tecnologia"].to_numpy(),
                nombre=un["bloque"].to_numpy(), curvatura=curv, por_hora=por_hora)


def _lambda_biseccion(dem: np.ndarray, c1: np.ndarray, c2: np.ndarray, pmax: np.ndarray,
                      pmin: np.ndarray | None = None, *, piso: float = 0.0,
                      techo: float = 4000.0, iters: int = 40) -> np.ndarray:
    """λ que balancea la demanda con ofertas cuadráticas recortadas por límites.

    P_g(λ) = clip((λ − c1_g)/(2·c2_g), Pmin_g, Pmax_g): es la condición KKT de
    La oferta cuadrática degenera en la lineal cuando no hay congestión, o sea el despacho
    merit-order exacto sin solver externo. Vectorizado por hora (eje 0).
    """
    d = np.asarray(dem, dtype=float)
    if d.ndim == 0:
        d = d.reshape(1)
    elif d.ndim > 1:
        d = d.sum(axis=tuple(range(1, d.ndim))) if d.shape[-1] != d.shape[0] else d[:, 0]
    d = np.nan_to_num(d, nan=0.0)
    dos_c2 = np.maximum(2.0 * c2, 1e-9)
    if pmin is None:
        pmin = np.zeros_like(pmax)
    lo = np.full(d.shape, float(piso), dtype=float)
    hi = np.full(d.shape, float(techo), dtype=float)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        bal = np.clip((mid[:, None] - c1[None, :]) / dos_c2[None, :],
                      pmin[None, :], pmax[None, :]).sum(axis=1) - d
        lo = np.where(bal < 0.0, mid, lo)
        hi = np.where(bal < 0.0, hi, mid)
    return 0.5 * (lo + hi)


def despacho_por_lambda(dem: np.ndarray, c1: np.ndarray, c2: np.ndarray, pmax: np.ndarray,
                         pmin: np.ndarray | None = None) -> np.ndarray:
    """Despacho por bloque (…, G) dado λ. Es la salida del 'unit commitment' relajado."""
    dos_c2 = np.maximum(2.0 * c2, 1e-9)
    if pmin is None:
        pmin = np.zeros_like(pmax)
    return np.clip((dem[..., None] - c1[None, :]) / dos_c2[None, :],
                   pmin[None, :], pmax[None, :])


def opf_nodal(red: dict, *, demanda: np.ndarray, solar: np.ndarray, eolica: np.ndarray,
              nivel_cop_kwh: float, flex_frac: float = 0.0, flex_gatillo_cop: float = 1400.0,
              kappa_hibrido: float = 0.5, perdas_iter: int = 3, dual_iter: int = NODAL_ITER_DUAL,
              paso_dual: float = NODAL_PASO_DUAL, tol_mw: float = NODAL_TOL_MW,
              liquidacion_piso: float = 0.0, despacho_piso: float = -40.0,
              techo_cop: float | None = None, withholding: dict[str, Any] | None = None,
              markup: dict[str, Any] | None = None,
              almacenamiento: dict[str, Any] | None = None, curva_oferta: float | None = None,
              bisec_iters: int = 26) -> dict[str, Any]:
    """Despacho óptimo DC con precios nodales, vectorizado sobre (hora × escenario).

    Resuelve el programa lineal-no lineal del día tipo:

        min Σ_g 1000·(c1_g·P_g + c2_g·P_g²)                        costo de operación (COP/h)
        s.a. Σ_i (1 − LF_i)·P_inj,i = L0 − Σ_i LF_i·P_inj0,i        balance con pérdidas linealizadas
             −Fmax_k ≤ Σ_j SF_kj·P_inj,j ≤ Fmax_k                    seguridad de las ramas 
             Pmin_g ≤ P_g ≤ Pmax_g                                   límites de bloque 

    y devuelve el precio nodal λ_i = λ_energia·(1 + LF_i) + Σ_k SF_ki·μ_k.
    λ se obtiene por bisección (el balance es monótono en λ) y μ por ascenso dual
    proyectado sobre la violación de los límites de rama, aplicado solo a las horas
    congestionadas: eso es lo que deja correr Monte Carlo completo en el navegador
    sin un solver LP externo (y sustituye a Matpower en la parte de formación de
    precios).

    Reglas de liquidación que salen del mismo despacho (`precios`):
      uninodal  → λ única para todos los nodos (lo que hoy hace el SIC).
      nodal     → λ_i por nodo (LMP pleno).
      zonal     → media ponderada por demanda de los λ_i dentro de la zona; con 7
                  nodos cada zona es un nodo, así que zonal ≡ nodal (se reporta).
      hibrido   → λ + κ(λ_i − λ) con κ = `kappa_hibrido`: la transición parcial.
    """
    nodos = list(red["nodos"])
    n = len(nodos)
    _kw = dict(solar=solar, eolica=eolica, nivel_cop_kwh=nivel_cop_kwh,
               flex_frac=flex_frac, flex_gatillo_cop=flex_gatillo_cop, kappa_hibrido=kappa_hibrido,
               perdas_iter=perdas_iter, dual_iter=dual_iter, paso_dual=paso_dual, tol_mw=tol_mw,
               liquidacion_piso=liquidacion_piso, despacho_piso=despacho_piso, techo_cop=techo_cop,
               withholding=withholding, markup=markup, almacenamiento=almacenamiento,
               curva_oferta=curva_oferta,
               bisec_iters=bisec_iters)
    of = _oferta_blocos(red, nivel_cop_kwh, curva=curva_oferta)
    SF = np.asarray(red["SF"], dtype=float)
    r_pu = np.asarray(red["r_pu"], dtype=float)
    Fmax = np.maximum(np.asarray(red["ramas"]["MW_max"], dtype=float), 1.0)
    nl = len(Fmax)

    dem = np.asarray(demanda, dtype=float)
    if dem.ndim == 1:
        dem = dem.reshape(1, -1)
    if dem.shape[1] != n:
        dem = np.resize(dem, (dem.shape[0], n))
    B = dem.shape[0]

    def _mat(a):
        if a is None:
            return np.zeros((B, n))
        a = np.asarray(a, dtype=float)
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if a.ndim == 2 and a.shape == (B, n):
            return np.maximum(a, 0.0)
        if a.ndim == 3:                              # (S, n, H) → se aplanó antes de llegar aquí
            a = a.reshape(-1, n)
        return np.maximum(np.resize(a, (B, n)), 0.0)

    ren_max = _mat(solar) + _mat(eolica)
    carga = _mat((almacenamiento or {}).get("carga"))
    descarga = _mat((almacenamiento or {}).get("descarga"))
    dem_total = dem + carga

    G_ = len(of["idx"])

    def _sel_bloques(filtro) -> np.ndarray:
        """Máscara de bloques de un agente: por nodo, tecnología o prefijo de bloque."""
        sel = np.zeros(G_, dtype=bool)
        if not filtro:
            return sel
        for clave, vector in (("nodo", of["idx"]), ("tecnologia", of["tec"]), ("bloque", of["nombre"])):
            if filtro.get(clave):
                val = filtro[clave]
                val = list(val) if isinstance(val, (list, tuple, set)) else [val]
                if clave == "nodo":
                    sel |= np.isin(of["idx"], [nodos.index(str(z)) for z in val if str(z) in nodos])
                elif clave == "tecnologia":
                    sel |= np.isin(of["tec"], val)
                else:
                    sel |= np.array([any(str(x).startswith(str(y)) for y in val) for x in of["nombre"]])
        return sel

    pmax_ef = of["pmax"].astype(float).copy()
    sel_wh = _sel_bloques(withholding)
    modo_wh = str((withholding or {}).get("modo", "capacidad"))
    if withholding and modo_wh != "energia":
        fr = float(withholding.get("frac", 0.0))
        pmax_ef = np.where(sel_wh, pmax_ef * max(0.0, 1.0 - fr), pmax_ef)

    G = len(of["idx"])
    A_gn = np.zeros((G, n))
    A_gn[np.arange(G), of["idx"]] = 1.0
    pmin_nodo = of["pmin"] @ A_gn
    fe_bloque = np.array([FE_NODAL.get(str(t), 0.20) for t in of["tec"]], dtype=float)
    # c1/c2 por bloque y hora (si el nivel vino como curva horaria, se expande a (B,G))
    c1_2d = np.atleast_2d(np.asarray(of["c1"], dtype=float))
    C1 = np.resize(c1_2d, (B, G)) if c1_2d.shape[0] > 1 else np.broadcast_to(c1_2d, (B, G)).copy()
    c2_2d = np.atleast_2d(np.asarray(of["c2"], dtype=float))
    C2 = np.resize(c2_2d, (B, G)) if c2_2d.shape[0] > 1 else np.broadcast_to(c2_2d, (B, G)).copy()
    # Poder de mercado: el agente ofrece sus bloques por encima de su costo
    # marginal. `pct` es el mark-up; `horas` permite aplicarlo solo en las horas en las
    # que el agente es marginal, que es como ocurre en el SIC real (EPM/Emgesa 2022).
    if markup:
        m = float(markup.get("pct", 0.0))
        sel_mk = _sel_bloques(markup)
        mask = np.zeros((B, G_), dtype=bool)
        if sel_mk.any():
            hh = np.ones(B, dtype=bool)
            if markup.get("horas") is not None:
                horas_mk = np.asarray(markup["horas"], dtype=int).ravel()
                hh = np.zeros(B, dtype=bool)
                hh[horas_mk[(horas_mk >= 0) & (horas_mk < B)] % B] = True
            mask = hh[:, None] & sel_mk[None, :]
            C1 = np.where(mask, C1 * (1.0 + m), C1)
            C2 = np.where(mask, C2 * (1.0 + m), C2)
        markup_aplicado = dict(pct=m, bloques=int(sel_mk.sum()), horas=int(mask.any(axis=1).sum()),
                               nodo=str(markup.get("nodo") or "-"), tecnologia=str(markup.get("tecnologia") or "-"))
    else:
        markup_aplicado = dict(pct=0.0, bloques=0, horas=0, nodo="-", tecnologia="-")
    dos_c2 = np.maximum(2.0 * C2, 1e-9)
    ren_slope = np.maximum(ren_max, 1e-9) / (NODAL_BID_VAR_HI - NODAL_BID_VAR_LO)

    flex = np.clip(float(flex_frac), 0.0, 0.6) * dem
    w_arr = np.asarray(red["peso_demanda"], dtype=float)
    w_arr = w_arr / max(float(w_arr.sum()), 1e-9)
    gatillo = max(float(flex_gatillo_cop), 1.0)
    positivos = C1[C1 > 0]
    techo = float(techo_cop) if techo_cop else max(float(np.max(positivos) * 2.4) if positivos.size else 1500.0,
                                                   1500.0)
    if techo <= despacho_piso + 1.0:
        techo = despacho_piso + 500.0
    # La rampa de activación mide desde el gatillo hacia arriba, no hasta el piso de despacho:
    # la demanda interrumpible del escenario se activa linealmente entre el precio gatillo y un
    # 25 % por encima, punto en el que el contrato de interrumpibilidad llega a su tope. Con la
    # banda completa [piso, gatillo] como denominador la respuesta quedaba prácticamente plana
    # en régimen de escasez (ratios del orden de 0,005), así que el mecanismo no se veía.
    ancho_flex = max(0.25 * gatillo, 1.0)

    mu_p = np.zeros((B, nl))
    mu_n = np.zeros((B, nl))
    LF = np.zeros((B, n))
    L0 = np.zeros(B)
    inj0 = np.zeros((B, n))
    lam = np.full(B, 0.5 * (despacho_piso + techo))

    def _evaluar(l_, mp, mn, lf, filas=None):
        """Paso completo: precios → despacho → flujo → pérdidas → residual de balance."""
        sub = filas is not None
        l_s = l_ if not sub else l_[filas]
        lf_s = lf if not sub else lf[filas]
        add_all = (mp - mn) @ SF   # Σ_k SF_ki·μ_k : (B,m)·(m,n)
        add = add_all if not sub else add_all[filas]
        lam_s = np.atleast_1d(l_s)[:, None]
        pd_ = lam_s * (1.0 + lf_s) + add
        pg_ = lam_s * (1.0 - lf_s) + add
        C1s = C1 if not sub else C1[filas]           # oferta por hora: se recorta igual que λ
        C2s = dos_c2 if not sub else dos_c2[filas]
        top = pmax_ef if pmax_ef.ndim == 1 else (pmax_ef if not sub else pmax_ef[filas])
        if np.ndim(top) == 1:
            top = top[None, :]
        P = np.clip((pg_[:, of["idx"]] - C1s) / C2s, of["pmin"][None, :], top)
        S = P @ A_gn
        ren = np.clip((pg_ - NODAL_BID_VAR_LO) * (ren_slope if not sub else ren_slope[filas]),
                      0.0, ren_max if not sub else ren_max[filas])
        # razón de corte en [0,1]: cuánto por encima del gatillo está el precio del nodo,
        # normalizada por el ancho entre el piso de despacho y el gatillo. El tope físico son
        # los MW interrumpibles (`flex`), que se aplican *después* de recortar la razón.
        razon_corte = np.clip((pd_ - gatillo) / ancho_flex, 0.0, 1.0)
        corte = razon_corte * (flex if not sub else flex[filas])
        serv = (dem_total if not sub else dem_total[filas]) - corte
        inj = S + ren + (descarga if not sub else descarga[filas]) - serv
        fl = inj @ SF.T
        perd = NODAL_BASE_MVA * np.sum(r_pu[None, :] * (fl / NODAL_BASE_MVA) ** 2, axis=1)
        bal = (np.sum((1.0 - lf_s) * inj, axis=1)
               - ((L0 if not sub else L0[filas]) - np.sum(lf_s * (inj0 if not sub else inj0[filas]), axis=1)))
        return dict(P=P, S=S, ren=ren, corte=corte, serv=serv, inj=inj, fl=fl, perd=perd,
                    bal=bal, pd=pd_, pg=pg_)

    def _bisec(filas, iters):
        sub = filas is not None
        lo = np.full(B if not sub else len(filas), despacho_piso)
        hi = np.full(B if not sub else len(filas), techo)
        for _ in range(int(iters)):
            mid = 0.5 * (lo + hi)
            if sub:
                lam_try = lam.copy()
                lam_try[filas] = mid
                bal = _evaluar(lam_try, mu_p, mu_n, LF, filas=filas)["bal"]
            else:
                bal = _evaluar(mid, mu_p, mu_n, LF)["bal"]
            lo = np.where(bal < 0.0, mid, lo)
            hi = np.where(bal < 0.0, hi, mid)
        fin = 0.5 * (lo + hi)
        if sub:
            lam[filas] = fin
        else:
            lam[:] = fin

    # ---- ascenso dual adaptativo (multiplicadores de congestión) ----
    mu_max = 6.0 * max(techo - liquidacion_piso, 1.0)
    paso = float(paso_dual)
    if withholding and modo_wh == "energia":
        fr = float(withholding.get("frac", 0.0))
        if fr > 0.0:
            _bisec(None, max(int(bisec_iters) - 10, 10))
            P0 = _evaluar(lam, mu_p, mu_n, LF)["P"]                  # despacho sin retiro (B,G)
            tope = np.where(sel_wh[None, :], P0 * max(0.0, 1.0 - fr),
                            np.broadcast_to(of["pmax"], (B, G_)))
            pmax_ef = np.minimum(np.broadcast_to(of["pmax"], (B, G_)), tope)

    it_dual_total = 0
    estancado = False
    estanc = np.zeros(B)          # rondas sin mejora por hora (para congelar, no abandonar)
    for it_p in range(max(int(perdas_iter), 1)):
        _bisec(None, bisec_iters)
        activa = np.arange(B)
        for _it in range(int(dual_iter)):
            if activa.size == 0:
                break
            out = _evaluar(lam, mu_p, mu_n, LF, filas=activa)
            viol_mas = out["fl"] - Fmax[None, :]
            viol_menos = -Fmax[None, :] - out["fl"]
            peor = np.maximum(viol_mas.max(axis=1), viol_menos.max(axis=1))
            if float(peor.max()) <= tol_mw:
                activa = np.empty(0, dtype=int)
                break
            esc = paso * 1000.0 / Fmax
            mu_p[activa] = np.clip(mu_p[activa] + esc[None, :] * np.maximum(viol_mas, 0.0), 0.0, mu_max)
            mu_n[activa] = np.clip(mu_n[activa] + esc[None, :] * np.maximum(viol_menos, 0.0), 0.0, mu_max)
            _bisec(activa, 14)
            chk = _evaluar(lam, mu_p, mu_n, LF, filas=activa)
            peor_new = np.maximum((chk["fl"] - Fmax[None, :]).max(axis=1),
                                  (-Fmax[None, :] - chk["fl"]).max(axis=1))
            it_dual_total += 1
            mej = peor_new <= 0.99 * peor                         # horas que aún mejoran
            estanc[activa] = np.where(mej, 0.0, estanc[activa] + 1.0)
            paso = min(paso * 1.08, float(paso_dual) * 2.0) if bool(mej.all()) else paso * 0.6
            sigue = estanc[activa] < 5.0                          # 5 rondas seguidas sin mejora → se congela
            activa = activa[sigue] if bool(sigue.any()) else np.empty(0, dtype=int)
            if paso <= 0.01 * float(paso_dual):
                estancado = True
                break
        fin = _evaluar(lam, mu_p, mu_n, LF)
        LF_new = 2.0 * ((fin["fl"] / NODAL_BASE_MVA) * r_pu[None, :]) @ SF   # LF_i = 2·Σ r_k F_k SF_ki
        LF = 0.5 * LF + 0.5 * np.clip(LF_new, -0.2, 0.2)
        L0, inj0 = fin["perd"], fin["inj"]

    # λ se re-bisecta una última vez en el punto de linealización final: sin esto el
    # balance quedaría desplazado exactamente las pérdidas (≈5,8 MW aquí) y el cierre
    # de emergencia se dispararía por un artefacto numérico, no físico.
    _bisec(None, bisec_iters)

    # ---- cierre del balance (modo emergencia) --------------------------------
    # Con líneas muy débiles el problema puede volverse infactible: la oferta
    # mínima (Pmin térmico + variable que no se puede frenar) no cabe por la red.
    # Un operador no deja demanda sin servir ni flujo fuera de límite: recorta la
    # variable, derrama térmica y, si aún falta, corta demanda no flexible al precio
    # de escasez. Aquí ese cierre es explícito y **se reporta** (horas de emergencia,
    # MW derramados, MW no servidos) en vez de esconderse en un λ mal comportado.
    fin = _evaluar(lam, mu_p, mu_n, LF)
    ren = fin["ren"].copy()
    P_gn = fin["P"].copy()
    serv = fin["serv"].copy()
    corte = fin["corte"].copy()
    derrame = np.zeros((B, n))
    no_servida = np.zeros((B, n))
    for _cierre in range(4):
        S_nodo = P_gn @ A_gn
        inj = S_nodo + ren + descarga - serv
        bal = np.sum((1.0 - LF) * inj, axis=1) - (L0 - np.sum(LF * inj0, axis=1))
        ex = np.maximum(bal, 0.0) / np.maximum(1.0 - LF.sum(axis=1), 1e-9)
        de = np.maximum(-bal, 0.0) / np.maximum(1.0 + LF.sum(axis=1), 1e-9)
        if float(ex.max()) <= tol_mw and float(de.max()) <= tol_mw:
            break
        if float(ex.max()) > tol_mw:                        # sobra oferta → se recorta
            pos = np.maximum(inj, 0.0)
            share = pos / np.maximum(pos.sum(axis=1, keepdims=True), 1e-9)
            cut = ex[:, None] * share
            S_nodo = P_gn @ A_gn
            cut_ren = np.minimum(ren, 0.85 * cut)
            cut_gn = np.maximum(np.minimum(np.maximum(S_nodo - pmin_nodo[None, :], 0.0),
                                           cut - cut_ren), 0.0)
            ren = ren - cut_ren
            escala = np.clip((S_nodo - cut_gn) / np.maximum(S_nodo, 1e-9), 0.0, 1.0)
            P_gn = P_gn * escala[:, of["idx"]]      # el recorte por nodo se reparte entre sus bloques
            derrame = derrame + cut_ren + cut_gn
        if float(de.max()) > tol_mw:                        # falta oferta → se corta demanda
            add = np.minimum(de[:, None] * w_arr[None, :], np.maximum(serv, 0.0))
            serv = serv - add
            corte = corte + add
            no_servida = no_servida + add
    S_nodo = P_gn @ A_gn
    inj = S_nodo + ren + descarga - serv
    fl = inj @ SF.T
    perd = NODAL_BASE_MVA * np.sum(r_pu[None, :] * (fl / NODAL_BASE_MVA) ** 2, axis=1)
    bal_final = np.sum((1.0 - LF) * inj, axis=1) - (L0 - np.sum(LF * inj0, axis=1))
    umbral_emerg = np.maximum(0.002 * np.maximum(dem.sum(axis=1), 1.0), 5.0 * tol_mw)
    emergencia = (derrame.sum(axis=1) > umbral_emerg) | (no_servida.sum(axis=1) > umbral_emerg)
    mu = mu_p - mu_n
    viol = np.maximum(np.abs(fl) - Fmax[None, :], 0.0)
    p_desp_dem = np.where(emergencia[:, None] & (no_servida > 1e-9), techo,
                          lam[:, None] * (1.0 + LF) + mu @ SF)
    p_desp_gen = np.where(emergencia[:, None] & (derrame > 1e-9), liquidacion_piso,
                          lam[:, None] * (1.0 - LF) + mu @ SF)
    p_dem = np.clip(p_desp_dem, liquidacion_piso, techo)
    p_gen = np.clip(p_desp_gen, liquidacion_piso, techo)
    recorte = np.maximum(ren_max - ren, 0.0) + derrame
    lamo = np.where(emergencia, techo, np.clip(lam, liquidacion_piso, techo))
    # Reglas de formación de precio sobre el mismo despacho (objetivo b).
    precios = {"uninodal": np.repeat(lamo[:, None], n, axis=1), "nodal": p_dem,
               "hibrido": np.clip(lamo[:, None] + float(kappa_hibrido) * (p_dem - lamo[:, None]),
                                  liquidacion_piso, techo)}
    grupos = red.get("grupos_zonales") or {}
    if grupos:
        zp = p_dem.copy()
        for _gz, idxs in grupos.items():
            ww = np.array([red["peso_demanda"][i] for i in idxs], dtype=float)
            prom = (p_dem[:, idxs] * ww[None, :]).sum(axis=1) / max(ww.sum(), 1e-9)
            zp[:, idxs] = prom[:, None] if np.ndim(prom) else prom
        precios["zonal"] = np.clip(zp, liquidacion_piso, techo)
    else:
        precios["zonal"] = p_dem.copy()

    costo = 1000.0 * np.sum(C1 * P_gn + C2 * P_gn ** 2, axis=1)
    MWh = np.maximum(dem, 0.0) * 1000.0
    pago = {k: np.sum(v * MWh, axis=1) for k, v in precios.items()}
    gen_MWh = 1000.0 * (S_nodo + ren + descarga)
    energia_nodal_MWh = np.sum(gen_MWh, axis=1)
    ingreso_nodal = np.sum(gen_MWh * p_dem, axis=1)
    ingreso_uni = energia_nodal_MWh * lamo
    renta_congestion = 1000.0 * np.sum(mu * fl, axis=1)
    renta_perdidas = 1000.0 * np.sum((p_dem - lamo[:, None]) * MWh / 1000.0, axis=1)
    co2_t = (P_gn * fe_bloque[None, :]).sum(axis=1)
    uso = np.abs(fl) / np.maximum(Fmax[None, :], 1.0)
    return dict(B=B, n=n, nl=nl, nodos=nodos, precios=precios, pago=pago, lam=lamo,
                p_despacho_dem=p_desp_dem, p_dem=p_dem, p_gen=p_gen, flujos=fl,
                Fmax=Fmax, uso_rama=uso, mu=mu, mu_p=mu_p, mu_n=mu_n, violacion=viol, LF=LF,
                perdidas=perd, P_bloque=P_gn, P_nodo=S_nodo, ren=ren,
                ren_max=ren_max, recorte=recorte, recorte_MWh=1000.0 * recorte,
                corte=corte, servida=serv, dem=dem, costo=costo,
                ingreso_nodal=ingreso_nodal, ingreso_uni=ingreso_uni,
                renta_congestion=renta_congestion, renta_perdidas=renta_perdidas, co2_t=co2_t,
                dispersion=p_dem.max(axis=1) - p_dem.min(axis=1),
                cv_precio=np.std(p_dem, axis=1) / np.maximum(np.mean(p_dem, axis=1), 1e-9),
                iters_dual=it_dual_total, perdas_iter=it_p + 1,
                converged=bool(float(np.max(viol)) <= max(6.0 * tol_mw, 3.0)
                               and not bool(emergencia.any()) and not estancado),
                estancado=bool(estancado), emergencia=emergencia,
                horas_emergencia=int(np.sum(emergencia)), derrame_MW=derrame,
                no_servida_MW=no_servida, derrame_MWh=1000.0 * derrame.sum(axis=1),
                no_servida_MWh=1000.0 * no_servida.sum(axis=1),
                max_violacion=float(np.max(viol)),
                horas_congestion=int(np.sum(viol.max(axis=1) > 1.0)),
                oferta=of, A_gn=A_gn, SF=SF, pmax_orig=of["pmax"], pmax_ef=pmax_ef, techo=techo,
                piso=liquidacion_piso, kappa_hibrido=float(kappa_hibrido),
                flex_frac=float(flex_frac), gatillo=gatillo, nivel=float(np.mean(nivel_cop_kwh)),
                markup=markup_aplicado, bloques_retirados=int(sel_wh.sum()),
                withholding_modo=modo_wh, pmax_ef_matriz=(pmax_ef.ndim > 1),
                nivel_por_hora=(np.asarray(nivel_cop_kwh, dtype=float).ravel()
                                if np.ndim(nivel_cop_kwh) else None),
                carga=carga, descarga=descarga,
                balance_residual=float(np.max(np.abs(bal_final))),
                error_linealizacion_MW=float(np.max(np.abs(bal_final))),
                error_linealizacion_pct=100.0 * float(np.max(np.abs(bal_final))) / max(float(np.sum(dem)), 1.0),
                horas=B, energia_dem_MWh=float(np.sum(MWh)),
                reserva_invol=1000.0 * float(np.sum(pmin_nodo)),
                horas_piso=int(np.sum(lamo <= liquidacion_piso + 1e-6)),
                horas_techo=int(np.sum(lamo >= techo - 1e-6)), kw=_kw)


def valida_lmp(red: dict, res: dict[str, Any], *, paso_mw: float = 2.0, max_horas: int = 24) -> pd.DataFrame:
    """Validación de los duals contra la definición de precio: λ_i = ∂Costo/∂P_i.

    A cada hora se le añade `paso_mw` en un nodo, se re-resuelve el OPF completo y
    se compara ΔCosto/ΔP con el λ_i reportado. Es el semáforo de calidad del motor
    (no decoración): si el ascenso dual convergió, el error queda del orden de la
    tolerancia de rama; si no, la tabla lo muestra.
    """
    filas = []
    dem = np.asarray(res["dem"], dtype=float)[:int(max_horas)]
    kw = dict(res.get("kw") or {})
    for i, nodo in enumerate(res["nodos"]):
        base = opf_nodal(red, demanda=dem, **kw)
        d2 = dem.copy()
        d2[:, i] = d2[:, i] + float(paso_mw)
        r2 = opf_nodal(red, demanda=d2, **kw)
        dc = float(np.mean(r2["costo"] - base["costo"])) / (float(paso_mw) * 1000.0)   # COP/h → COP/kWh
        lam_i = float(np.mean(base["p_dem"][:, i]))
        filas.append(dict(nodo=nodo, lambda_dual_cop_kWh=round(lam_i, 2),
                          lambda_dif_finita_cop_kWh=round(dc, 2),
                          error_pct=round(100.0 * (dc - lam_i) / max(abs(lam_i), 1e-9), 3),
                          paso_MW=float(paso_mw), horas=int(len(dem))))
    return pd.DataFrame(filas)


# ---------------- 10.7 · almacenamiento: EMS de arbitraje -------------------
#
# El modelo de bombeo/PSH usa `V_urt^min ≤ V_t ≤ V_urt^max` y potencia de
# turbina/bomba (P_PSH_STG / P_PSH_IN). Aquí se resuelve con el EMS de dos
# el EMS del módulo pasa en dos etapas: (1) OPF sin almacenamiento → precios por nodo;
# (2) el EMS traslada energía de las horas baratas a las caras de *ese* nodo
# (water-filling sobre la curva de precios, óptimo para un ciclo diario con
# pérdidas de ida y vuelta η); (3) OPF con la carga/descarga como nueva inyección.
# Se itera hasta que el programa de despacho deja de cambiar (punto fijo).

def _water_filling(precios_h: np.ndarray, e_nom_mwh: float, p_nom_mw: float,
                   eta: float, *, modo: str = "arbitraje", excedente_h: np.ndarray | None = None,
                   puede_cargar: np.ndarray | None = None, puede_descargar: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Programa (carga, descarga, SoC) que maximiza Σ(descarga·p − carga·p) con límites.

    `modo="arbitraje"` persigue la cuña de precios del nodo; `modo="anti-recorte"`
    sólo carga con el excedente renovable que el despacho habría derramado (captura
    de curtailment); `modo="alivio_congestion"` arbitra la cuña espacial λ_i−λ, que
    es la señal que **sólo existe con precios nodales**. Es voraz por pares (hora cara, hora barata) hasta agotar
    energía o potencia: con un solo ciclo diario y pérdidas fijas es el óptimo.
    """
    H = len(precios_h)
    carga = np.zeros(H)
    desc = np.zeros(H)
    pre = np.asarray(precios_h, dtype=float).copy()
    INF = float(np.nanmax(pre)) + 1.0e6
    if puede_cargar is not None:                      # horas vetadas: fuera de la lista
        pre = np.where(np.asarray(puede_cargar, dtype=bool), pre, INF)
    ord_b = np.argsort(pre)                           # horas más baratas primero
    post = np.asarray(precios_h, dtype=float).copy()
    if puede_descargar is not None:
        post = np.where(np.asarray(puede_descargar, dtype=bool), post, -INF)
    ord_c = np.argsort(-post)                          # horas más caras primero
    restante = float(e_nom_mwh)
    k = 0
    while k < H // 2 and restante > 1e-9:
        hb, hc = int(ord_b[k]), int(ord_c[k])
        if hc == hb or precios_h[hc] <= precios_h[hb] / max(eta, 1e-6):
            break
        if (puede_cargar is not None and not bool(puede_cargar[hb])) or \
           (puede_descargar is not None and not bool(puede_descargar[hc])):
            break
        disponible_b = p_nom_mw - carga[hb]
        disponible_c = p_nom_mw - desc[hc]
        if modo == "anti-recorte" and excedente_h is not None:
            disponible_b = min(disponible_b, max(float(excedente_h[hb]), 0.0))
        x = min(restante, disponible_b, disponible_c, max(restante, 0.0))
        if x <= 1e-9:
            k += 1
            continue
        carga[hb] += x
        desc[hc] += x * max(eta, 1e-6)
        restante -= x
        k += 1
    soc = np.cumsum(carga - desc / max(eta, 1e-6))
    soc = soc - soc.min()
    if float(soc.max()) > 1e-9:
        soc = np.clip(soc, 0.0, float(e_nom_mwh))
    return carga, desc, soc


def ems_almacenamiento(red: dict, *, demanda: np.ndarray, solar: np.ndarray, eolica: np.ndarray,
                       nivel_cop_kwh, config: dict[str, Any], pasadas: int = 5,
                       suavizado: float = 0.75,
                       modo: str = "arbitraje", converg_mw: float = 1.0, **kw) -> dict[str, Any]:
    """Despacho con almacenamiento (baterías + bombeo) y su efecto sobre los LMP.

    `config` = {"potencia_MW": (n,), "energia_MWh": (n,), "eficiencia": float,
    "piso_soc": float, "techo_soc": float}. Devuelve el punto fijo (precio sin y
    con almacenamiento, programa de carga/descarga, SoC, valor del arbitraje,
    reducción de congestión y de recorte) para las cuatro reglas de liquidación.
    """
    n = len(red["nodos"])
    H = demanda.shape[0] if demanda.ndim == 2 else 1
    pn = np.broadcast_to(np.asarray(config.get("potencia_MW", 0.0), dtype=float), (n,)).astype(float)
    en = np.broadcast_to(np.asarray(config.get("energia_MWh", 0.0), dtype=float), (n,)).astype(float)
    eta = float(np.clip(config.get("eficiencia", 0.88), 0.4, 1.0))
    uso_almacen = bool(pn.sum() > 0.0 and en.sum() > 0.0)
    carga = np.zeros((H, n))
    descarga = np.zeros((H, n))
    historia: list[dict[str, float]] = []
    su = float(np.clip(suavizado, 0.15, 1.0))
    mejor_punt = float("inf")
    mejor = None
    res = None
    res0 = opf_nodal(red, demanda=demanda, solar=solar, eolica=eolica,
                     nivel_cop_kwh=nivel_cop_kwh, **kw)
    for it in range(int(pasadas) if uso_almacen else 1):
        precio_nodo = res0["p_dem"] if res is None else res["p_dem"]
        exc = res0["recorte"] if res is None else res["recorte"]
        c_new = np.zeros((H, n))
        d_new = np.zeros((H, n))
        # señal de despacho del EMS: precios del nodo (arbitraje temporal-espacial),
        # excedente renovable (captura de recorte) o cuña nodal λ_i−λ (alivio de
        # congestión). Las tres son reglas de mercado reales para el mismo activo.
        senal = np.asarray(precio_nodo, dtype=float).copy()
        puede_c = puede_d = None
        if modo == "alivio_congestion":
            base = res0["lam"] if res is None else res["lam"]
            cun = senal - np.asarray(base, dtype=float)[:, None]     # λ_i − λ: cuña espacial
            senal = cun
            # descargar sólo donde el nodo es importador neto (cuña positiva y horaria de
            # máxima congestión) y cargar donde es exportador: eso alivia la rama.
            puede_c = cun < np.median(cun, axis=0, keepdims=True)
            puede_d = cun > np.median(cun, axis=0, keepdims=True)
        for i in range(n):
            if pn[i] <= 0.0 or en[i] <= 0.0:
                continue
            c, d, _s = _water_filling(np.asarray(senal[:, i], dtype=float), float(en[i]),
                                      float(pn[i]), eta, modo=modo, excedente_h=exc[:, i],
                                      puede_cargar=None if puede_c is None else puede_c[:, i],
                                      puede_descargar=None if puede_d is None else puede_d[:, i])
            c_new[:, i] = c
            d_new[:, i] = d
        delta = float(np.abs(c_new - carga).max() + np.abs(d_new - descarga).max())
        # el punto fijo oscila si el almacenamiento es grande (1 GW sobre 10 GW mueve
        # el precio de su propia hora): se amortigua la actualización y se conserva la
        # iteración con menor congestión + recorte, no la última.
        if it == 0:                                  # la primera pasada va completa
            carga, descarga = c_new, d_new
        else:
            carga = (1.0 - su) * carga + su * c_new
            descarga = (1.0 - su) * descarga + su * d_new
        res = opf_nodal(red, demanda=demanda, solar=solar, eolica=eolica, nivel_cop_kwh=nivel_cop_kwh,
                         almacenamiento=dict(carga=carga, descarga=descarga), **kw)
        punt = float(res["max_violacion"]) + 0.02 * float(res["recorte_MWh"].sum() / 1e3) / max(H, 1)
        if punt < mejor_punt:
            mejor_punt, mejor = punt, (carga.copy(), descarga.copy(), res)
        historia.append(dict(pasada=it, delta_MW=delta, violacion=res["max_violacion"],
                             puntuacion=punt, recorte_GWh=float(res["recorte_MWh"].sum() / 1e3),
                             congestion_h=res["horas_congestion"],
                             valor_MCP=float(1000.0 * np.sum((descarga - carga) * res["p_dem"]))))
        if delta <= converg_mw:
            break
    if mejor is not None:
        carga, descarga, res = mejor
    sin = res0
    con = res if res is not None else res0
    # MWh por hora (paso de 1 h): 1 MW durante 1 h = 1 MWh; el precio en COP/kWh se
    # multiplica por 1000 para pasar a COP. Se valoran los cuatro casos con el MISMO
    # programa de carga, para comparar reglas sin mezclar efectos de re-despacho.
    MWh_c, MWh_d = carga.astype(float), descarga.astype(float)
    p_nod_c = con["p_despacho_dem"]
    v_nod_post = float(1000.0 * np.sum(MWh_d * con["p_dem"] - MWh_c * p_nod_c)) if uso_almacen else 0.0
    v_uni_post = float(1000.0 * np.sum((MWh_d - MWh_c) * con["lam"][:, None])) if uso_almacen else 0.0
    v_nod_pre = float(1000.0 * np.sum(MWh_d * sin["p_dem"] - MWh_c * sin["p_despacho_dem"])) if uso_almacen else 0.0
    v_uni_pre = float(1000.0 * np.sum((MWh_d - MWh_c) * sin["lam"][:, None])) if uso_almacen else 0.0
    valor_nodal, valor_uni = v_nod_post, v_uni_post
    soc = np.cumsum(carga - descarga / max(eta, 1e-6), axis=0)
    soc = soc - soc.min(axis=0, keepdims=True)
    soc = np.minimum(soc, np.maximum(en[None, :], 1e-9)) / np.maximum(en[None, :], 1e-9)
    return dict(uso=bool(uso_almacen), modo=modo, eta=eta, carga_MW=carga, descarga_MW=descarga,
                carga_MWh=MWh_c, descarga_MWh=MWh_d, soc=soc, potencias=pn, energias=en,
                valor_nodal_pre_COP=v_nod_pre, valor_uninodal_pre_COP=v_uni_pre,
                valor_nodal_post_COP=v_nod_post, valor_uninodal_post_COP=v_uni_post,
                mejora_valor_pct=(100.0 * (v_nod_post - v_uni_post) / max(abs(v_uni_post), 1e-9)),
                historia=pd.DataFrame(historia), pasadas=len(historia),
                precio_sin=sin, precio_con=con, valor_arbitraje_COP=valor_nodal,
                valor_arbitraje_uninodal_COP=valor_uni,
                violacion_antes=float(sin["max_violacion"]), violacion_despues=float(con["max_violacion"]),
                horas_congestion_antes=int(sin["horas_congestion"]),
                horas_congestion_despues=int(con["horas_congestion"]),
                recorte_antes_GWh=float(sin["recorte_MWh"].sum() / 1e3),
                recorte_despues_GWh=float(con["recorte_MWh"].sum() / 1e3),
                cv_antes=float(np.mean(sin["cv_precio"])), cv_despues=float(np.mean(con["cv_precio"])),
                dispersion_antes=float(sin["dispersion"].mean()), dispersion_despues=float(con["dispersion"].mean()),
                co2_antes=float(sin["co2_t"].sum()), co2_despues=float(con["co2_t"].sum()),
                costo_antes=float(sin["costo"].sum()), costo_despues=float(con["costo"].sum()),
                horas_con_carga=int(np.sum(carga.sum(axis=1) > 1e-6)),
                horas_con_descarga=int(np.sum(descarga.sum(axis=1) > 1e-6)))


def config_almacenamiento(red: dict, *, bateria_pct: float = 0.0, bombeo_pct: float = 0.0,
                          dur_bateria_h: float = 4.0, dur_bombeo_h: float = 12.0,
                          eta_bateria: float = 0.88, eta_bombeo: float = 0.80,
                          nodos: list[str] | None = None) -> dict[str, Any]:
    """Dimensiona el almacenamiento como % de la CEN de cada nodo (supuesto del tablero)."""
    cap = red["cap"].sum(axis=1).to_numpy(dtype=float)
    sel = np.zeros(len(cap), dtype=bool)
    if nodos:
        for z in nodos:
            if z in list(red["nodos"]):
                sel[list(red["nodos"]).index(z)] = True
    else:
        sel[:] = True
    pn = cap * (float(bateria_pct) / 100.0 + float(bombeo_pct) / 100.0) * sel
    en = pn * (float(dur_bateria_h) if not bombeo_pct else float(dur_bombeo_h))
    return dict(potencia_MW=pn, energia_MWh=en, eficiencia=float(eta_bateria if not bombeo_pct
                                                                  else (eta_bateria * float(bateria_pct) + eta_bombeo * float(bombeo_pct)) / max(float(bateria_pct) + float(bombeo_pct), 1e-9)),
                dur_bateria_h=float(dur_bateria_h), dur_bombeo_h=float(dur_bombeo_h),
                bateria_pct=float(bateria_pct), bombeo_pct=float(bombeo_pct))


# ------------- 10.8 · poder de mercado, riesgo, coberturas y aprendizaje -----

def poder_mercado_ior(ctx: dict, red: dict | None = None, *, umbral_ior: float = 0.15) -> dict[str, Any]:
    """Poder de mercado: IOR diario (XM), elasticidad real y concentración por nodo.

    El poder de mercado se mide con la demanda residual y el markup. Aquí:

    * **IOR** = (P − RAC)/P con P = `Precio_Bolsa_Dia_COP_kWh` y RAC ≈
      `Costo_Marginal_COP_kWh_Dia` de `cruce_v4` (la tabla que valida la celda 61 del
      notebook). La SSPD abre investigación con IOR > 0,15 (expedientes EPM
      14/15-mar-2022 y Emgesa 11/14/15-mar-2022, casos reportados por la prensa del sector).
    * **η** = elasticidad-precio de la demanda estimada con *efectos fijos horarios*
      (se restan las medias por hora sobre las 24 × días del ventana). Sin esos fijos,
      la regresión cruda da η > 0 porque precio y demanda comparten el ciclo diario:
      el tablero reporta las dos para que se vea la trampa econométrica.
    * **HHI y cuota del primer agente por subregión**, medidos con `dim_plantas`
      (`Codigo_Agente`) × `cen_recurso` × la toponimia del tablero y nombrados con
      `catalogo_agentes`: es la `s_i` con la que se calcula el markup de Lerner, no un
      supuesto de libro.
    * **Fases ENSO** (`impacto`): el precio ponderado medio por fase, para la
      hipótesis de trabajo: el retiro aumenta cuando el sistema está seco.
    """
    out: dict[str, Any] = dict(disponible=False, dias=0, ior_medio=float("nan"), ior_max=float("nan"),
                               ior_p95=float("nan"), dias_bandera=0, eta=float("nan"), eta_bruta=float("nan"),
                               n_elasticidad=0, lerner=float("nan"), markup_pct=float("nan"),
                               regimen="", hhi=float("nan"), hhi_max_nodo=float("nan"), cuota_top=float("nan"),
                               agente_top="n/d", tabla=pd.DataFrame(), nodos_tabla=pd.DataFrame(),
                               fase=pd.DataFrame(), precio_pond_fases={}, fuentes="",
                               umbral_ior=float(umbral_ior), nota="")
    cv = ctx.get("cruce_v4")
    if isinstance(cv, pd.DataFrame) and not cv.empty:
        p_col = next((c for c in ("Precio_Bolsa_Dia_COP_kWh", "Precio_Ponderado_Dia_COP_kWh")
                      if c in cv.columns), None)
        r_col = next((c for c in ("Costo_Marginal_COP_kWh_Dia", "Costo_Marginal_COP_kWh")
                      if c in cv.columns), None)
        if p_col and r_col:
            t = cv[["Fecha", p_col, r_col]].copy()
            t[[p_col, r_col]] = t[[p_col, r_col]].astype(float)
            t = t.dropna(subset=[p_col, r_col])
            t = t[(t[p_col] > 0) & (t[r_col] > 0)]
            if len(t) >= 3:
                t = t.assign(rac_vs_p=t[r_col] / t[p_col],
                             ior=(t[p_col] - t[r_col]) / t[p_col],
                             banda="")
                t["bandera"] = t["ior"] > float(umbral_ior)
                for extra in ("Precio_Ponderado_Dia_COP_kWh", "Margen_Dia_pct", "Indisponibilidad_pct",
                              "Reserva_Operativa_MW", "Demanda_MW_Dia"):
                    if extra in cv.columns:
                        t[extra] = cv.loc[t.index, extra].to_numpy()
                out.update(dias=int(len(t)), ior_medio=float(t["ior"].mean()), ior_max=float(t["ior"].max()),
                           ior_p95=float(t["ior"].quantile(0.95)), dias_bandera=int(t["bandera"].sum()),
                           tabla=t.reset_index(drop=True), disponible=True,
                           precio_medio_COP_kWh=float(t[p_col].mean()),
                           cmd_medio_COP_kWh=float(t[r_col].mean()))
                out["fuentes"] = f"`cruce_v4` · {p_col} vs {r_col} · {out['dias']} días"
    # elasticidad con fijos horarios
    sis = ctx.get("sistema")
    if isinstance(sis, pd.DataFrame) and {"Demanda_MW", "Precio_Bolsa_COP_kWh", "Hora"} <= set(sis.columns):
        q = sis[["Hora", "Demanda_MW", "Precio_Bolsa_COP_kWh"]].astype(
            {"Demanda_MW": float, "Precio_Bolsa_COP_kWh": float}).dropna()
        q = q[(q["Demanda_MW"] > 0) & (q["Precio_Bolsa_COP_kWh"] > 0)]
        if len(q) > 30:
            q = q.assign(lx=np.log(q["Precio_Bolsa_COP_kWh"]), ly=np.log(q["Demanda_MW"]))
            gf = q.groupby("Hora")[["lx", "ly"]].transform("mean")
            x = (q["lx"] - gf["lx"]).to_numpy()
            y = (q["ly"] - gf["ly"]).to_numpy()
            out["eta"] = float(np.sum(x * y) / max(np.sum(x * x), 1e-18))
            out["eta_bruta"] = float(np.cov(q["lx"], q["ly"])[0, 1] / max(np.var(q["lx"], ddof=1), 1e-18))
            out["n_elasticidad"] = int(len(q))
    eta = out["eta"]
    if np.isfinite(eta) and eta < -1.0 - 1e-9:
        out["lerner"] = float(1.0 / abs(eta))
        out["markup_pct"] = float(100.0 * out["lerner"])
        out["regimen"] = "|η| > 1 → el límite de Lerner es aplicable"
    else:
        out["regimen"] = ("|η| ≤ 1 (demanda casi inelástica en el corto plazo): el recargo no lo fija "
                          "Lerner sino el tope regulado (precio de escasez / RAC)")
    esc = ctx.get("escasez")
    if isinstance(esc, pd.DataFrame) and not esc.empty:
        ce = [c for c in esc.columns if "Precio_Escasez" in str(c)]
        if ce:
            pe = float(pd.to_numeric(esc[ce[0]], errors="coerce").mean())
            out["precio_escasez_medio_COP_kWh"] = pe
            if out.get("cmd_medio_COP_kWh"):
                out["techo_markup_pct"] = 100.0 * (pe - float(out["cmd_medio_COP_kWh"])) / max(
                    float(out["cmd_medio_COP_kWh"]), 1e-9)
    # concentración real
    mp = red.get("mapa") if isinstance(red, dict) else None
    if isinstance(mp, pd.DataFrame) and not mp.empty and "Codigo_Agente" in mp.columns:
        cat = ctx.get("catalogo_agentes")
        nombres = {}
        if isinstance(cat, pd.DataFrame) and {"Values_Code", "Values_Name"} <= set(cat.columns):
            nombres = dict(zip(cat["Values_Code"].astype(str), cat["Values_Name"].astype(str)))
        base = mp.dropna(subset=["Zona"]) if "Zona" in mp.columns else mp
        por_ag = base.groupby("Codigo_Agente")["CEN_MW"].sum().sort_values(ascending=False)
        tot = float(por_ag.sum())
        if tot > 0:
            out["hhi"] = float(np.sum((por_ag / tot * 100.0) ** 2))
            out["cuota_top"] = float(100.0 * por_ag.iloc[0] / tot)
            out["agente_top"] = str(nombres.get(str(por_ag.index[0]), por_ag.index[0]))[:44]
        filas = []
        for z, gg in base.groupby("Zona"):
            pa = gg.groupby("Codigo_Agente")["CEN_MW"].sum().sort_values(ascending=False)
            tz = float(pa.sum())
            if tz <= 0:
                continue
            filas.append(dict(nodo=z, cen_MW=round(tz, 1), n_agentes=int(len(pa)),
                              hhi=round(float(np.sum((pa / tz * 100.0) ** 2)), 1),
                              cuota_top_pct=round(float(100.0 * pa.iloc[0] / tz), 1),
                              agente_top=str(nombres.get(str(pa.index[0]), pa.index[0]))[:40]))
        out["nodos_tabla"] = pd.DataFrame(filas)
        if not out["nodos_tabla"].empty:
            out["hhi_max_nodo"] = float(out["nodos_tabla"]["hhi"].max())
    imp = ctx.get("impacto")
    if isinstance(imp, pd.DataFrame) and not imp.empty and "variable" in imp.columns:
        f = imp[imp["variable"].astype(str).str.contains("Precio", case=False, na=False)]
        if not f.empty:
            out["fase"] = f.reset_index(drop=True)
            for c in ("media_Neutral", "media_El_Nino", "media_La_Nina"):
                if c in f.columns:
                    out["precio_pond_fases"][c.replace("media_", "")] = float(f.iloc[0][c])
    out["nota"] = ("El IOR aquí usa precio de bolsa vs CMD (no `Precio_Ponderado`, que en v4 "
                   "es el precio ponderado de la cartera y no es comparable al CMD). Con η "
                   "inelástica el markup teórico se dispara: el tablero reporta el recargo "
                   "acotado por el precio de escasez y la concentración real por nodo.")
    return out


def cournot_bertrand(red: dict, perfiles: dict[str, np.ndarray], nivel_cop_kwh, *,
                     n_agentes: int = 6, epsilon: float = -0.45, cuota_top: float | None = None,
                     p_techo: float | None = None, horas: list[int] | None = None) -> pd.DataFrame:
    """Bertrand (precios) y Cournot (cantidades) sobre la demanda residual del nodo.

    Con demanda isoelástica Q(p) = Q0·(p/p0)^ε y costo marginal cuadrático, el
    resultado de texto es:

      * Bertrand con producto homogéneo     → p = CMg (guerra de precios).
      * Cournot simétrico con m agentes     → L_i = s_i/|ε|  ⇒  p = CMg/(1 − s_i/|ε|).
      * Monopolio regulado por RAC          → p = min(CMg/(1 − 1/|ε|), RAC).

    `cuota_top` es la participación del primer agente *medida* en los datos (la que
    devuelve `poder_mercado_ior`); si no viene, se asume simetría s = 1/m. La tabla
    se usa para la figura «recargo de poder de mercado por hora y por regla», que es
    el corazón del objetivo c: bajo precio nodal el recargo sólo puede aplicarse en
    el nodo congestonado, no en todo el SIN.
    """
    dem, sol, eol = perfiles_nodales(red, perfiles)
    dem_h = dem.sum(axis=1)
    ren_h = (sol + eol).sum(axis=1)
    hh = list(range(len(dem_h))) if horas is None else [int(x) % len(dem_h) for x in horas]
    of = _oferta_blocos(red, nivel_cop_kwh)
    c1_all = np.atleast_2d(np.asarray(of["c1"], dtype=float))
    c2_all = np.atleast_2d(np.asarray(of["c2"], dtype=float))
    eps = float(min(-1.0 - 1e-6, float(epsilon)))          # elástica (negativa, |ε|>1)
    filas = []
    for h in hh:
        c1 = c1_all[min(h, c1_all.shape[0] - 1)]
        c2 = c2_all[min(h, c2_all.shape[0] - 1)]
        q = max(float(dem_h[h]) - float(ren_h[h]), 1.0)
        lam_s = np.linspace(max(float(np.min(c1)) * 0.2, 1.0), float(np.max(c1)) * 5.0, 120)
        Pof = np.clip((lam_s[:, None] - c1[None, :]) / np.maximum(2.0 * c2[None, :], 1e-9),
                      of["pmin"][None, :], of["pmax"][None, :]).sum(axis=1)
        i = int(np.argmin(np.abs(Pof - q)))
        p_comp = float(lam_s[i])
        mc = float(np.mean(c1 + 2.0 * c2 * np.clip((p_comp - c1) / np.maximum(2.0 * c2, 1e-9),
                                                   of["pmin"], of["pmax"])))
        s_top = float(cuota_top) / 100.0 if cuota_top else 1.0 / max(int(n_agentes), 1)
        s_top = float(np.clip(s_top, 1.0 / max(int(n_agentes), 1), 0.95))
        # si |ε| ≤ s_top el recargo de Lerner no tiene solución finita: el mercado real
        # lo acota el precio de escasez/RAC, así que se usa el tope (p_techo) y se reporta
        L_cournot = s_top / abs(eps)
        L_mono = 1.0 / abs(eps)
        p_cournot = mc / max(1.0 - L_cournot, 1e-3)
        p_mono = mc / max(1.0 - L_mono, 1e-3)
        # tope regulado: en el SIC el recargo está acotado por el precio de escasez
        if p_techo:
            p_cournot = min(p_cournot, float(p_techo))
            p_mono = min(p_mono, float(p_techo))
        filas.append(dict(hora=h, demanda_MW=float(dem_h[h]), residual_MW=q, mg_COP_kWh=mc,
                          p_competencia=p_comp, p_bertrand=mc, p_cournot=float(p_cournot),
                          p_monopolio=float(p_mono), s_top_pct=100.0 * s_top,
                          lerner_cournot=float(L_cournot), epsilon=eps,
                          recargo_cournot_pct=100.0 * (p_cournot - p_comp) / max(p_comp, 1e-9),
                          recargo_bertrand_pct=100.0 * (mc - p_comp) / max(p_comp, 1e-9),
                          recargo_mono_pct=100.0 * (p_mono - p_comp) / max(p_comp, 1e-9)))
    return pd.DataFrame(filas)


def retornos_reales(ctx: dict, columna: str = "Precio_Bolsa_COP_kWh") -> np.ndarray:
    """Log-retornos horarios de la serie real de XM (insumo de `garch11`/VaR)."""
    for k in ("sistema_v4", "sistema"):
        d = ctx.get(k)
        if isinstance(d, pd.DataFrame) and columna in d.columns:
            x = d.sort_values(["Fecha", "Hora"])[columna].astype(float).to_numpy()
            x = x[np.isfinite(x) & (x > 0)]
            if x.size > 3:
                return np.diff(np.log(x))
    return np.array([], dtype=float)


def retornos_nodales(res: dict[str, Any], nodo: str | None = None) -> np.ndarray:
    """Log-retornos de la serie horaria de precios del modelo (uninodal o de un nodo)."""
    if nodo and nodo in res["nodos"]:
        p = np.asarray(res["p_dem"][:, res["nodos"].index(nodo)], dtype=float)
    else:
        p = np.asarray(res["lam"], dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    return np.diff(np.log(p)) if p.size > 3 else np.array([], dtype=float)


def garch11(r: np.ndarray, *, grid_beta: int = 26, grid_alpha: int = 26,
            iters: int = 40) -> dict[str, Any]:
    """GARCH(1,1) estimado por máxima verosimilitud gaussiana.

    σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}. Aquí se usa GARCH para la volatilidad
    condicional del precio y de ahí el VaR. Como `requirements.txt` no
    incluye scipy, la optimización es una búsqueda en rejilla (α,β) refinada en dos
    niveles con el ω concentrado analíticamente (varianza incondicional), que es
    estable y suficiente para 24·d obs.
    """
    x = np.asarray(r, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 30:
        return dict(disponible=False, nota=f"sólo {x.size} observaciones (se piden ≥ 30)")
    mu = float(np.mean(x))
    e = x - mu
    v = float(np.var(e, ddof=1))

    def _ll(a: float, b: float) -> tuple[float, np.ndarray]:
        s2 = np.empty_like(e)
        s2[0] = v
        for t in range(1, e.size):
            s2[t] = omega_of(a, b) + a * e[t - 1] ** 2 + b * s2[t - 1]
            s2[t] = max(s2[t], 1e-12)
        return float(-0.5 * np.sum(np.log(2.0 * math.pi * s2) + e ** 2 / s2)), s2

    def omega_of(a: float, b: float) -> float:
        return max(v * (1.0 - a - b), 1e-12)

    mejor = (-np.inf, 0.0, 0.0, None)
    cand_a = np.linspace(0.01, 0.40, grid_alpha)
    cand_b = np.linspace(0.10, 0.97, grid_beta)
    for a in cand_a:
        for b in cand_b:
            if a + b >= 0.999:
                continue
            ll, s2 = _ll(a, b)
            if ll > mejor[0]:
                mejor = (ll, float(a), float(b), s2)
    # refinamiento local
    _, a0, b0, _ = mejor
    for da in np.linspace(-0.03, 0.03, 7):
        for db in np.linspace(-0.05, 0.05, 7):
            a, b = a0 + da, b0 + db
            if a <= 0.0 or b <= 0.0 or a + b >= 0.999:
                continue
            ll, s2 = _ll(a, b)
            if ll > mejor[0]:
                mejor = (ll, float(a), float(b), s2)
    ll, a, b, s2 = mejor
    s = np.sqrt(s2)
    return dict(disponible=True, omega=omega_of(a, b), alfa=a, beta=b, persistencia=a + b,
                vida_media_h=float(-1.0 / math.log(max(a + b, 1e-9))) if a + b > 1e-9 else float("nan"),
                loglik=ll, mu=mu, sigma=s, sigma_media=float(np.mean(s)),
                var_incondicional=math.sqrt(v), n=int(x.size),
                sigma_final=float(s[-1]), nota="MD con ω concentrado; rejilla + refinamiento (sin scipy)")


def var_condicional(sigma: np.ndarray, precios: np.ndarray, *, alfa: float = 0.05,
                    mu: float | None = None, escala: str = "log") -> pd.DataFrame:
    """VaR (y ES) condicional del precio con la volatilidad de `garch11`.

    `sigma` es la desviación condicional de los *retornos log* (lo que estima
    `garch11`), así que el VaR se construye en esa escala y se re-traduce al nivel
    del precio: VaR_t = P_t·exp(μ + σ_t·z_α). El expected shortfall usa la forma
    cerrada normal, ES_t = P_t·exp(μ + σ_t·φ(z_α)/α). `escala="lineal"` deja las
    formulas aditivas por si se trabaja con retornos simples en COP/kWh.
    """
    s = np.asarray(sigma, dtype=float)
    x = np.asarray(precios, dtype=float)[:len(s)]
    a = float(np.clip(alfa, 0.005, 0.5))
    z = float(math.sqrt(2.0) * _erf_inv_scalar(2.0 * a - 1.0))      # z_α < 0
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    if str(escala).startswith("log"):
        var = x * np.exp(z * s)
        es = x * np.exp(phi / a * s)
        m_ret = float(z * s.mean())
    else:
        m = 0.0 if mu is None else float(mu)
        var = m + s * z
        es = m + s * phi / a
        m_ret = m
    return pd.DataFrame(dict(t=np.arange(len(s)), precio=x, sigma=s, var=var, es=es,
                             var_pct=100.0 * (var - x) / np.maximum(x, 1e-9),
                             es_pct=100.0 * (es - x) / np.maximum(x, 1e-9),
                             z_alfa=z, alfa=a, mu_reto=m_ret))


def _erf_inv_scalar(y: float) -> float:
    """erf⁻¹ escalar por Newton (|y|<1)."""
    out = y
    for _ in range(40):
        f = float(_erf(np.array([out]))[0]) - y
        d = 2.0 / math.sqrt(math.pi) * math.exp(-out * out)
        out = out - f / max(d, 1e-12)
    return float(out)


def _inv_erf(y: np.ndarray | float) -> np.ndarray:
    """Inversa de erf por Newton sobre la aprox. de A&S (sin scipy)."""
    yy = np.asarray(y, dtype=float)
    out = np.sign(yy) * np.sqrt(np.maximum(np.abs(yy), 1e-12) * 1.0)
    for _ in range(30):
        f = _erf(out) - yy
        d = 2.0 / math.sqrt(math.pi) * np.exp(-out * out)
        out = out - f / np.maximum(d, 1e-12)
    return float(out) if np.ndim(out) == 0 else out


def kupiec_poc(hits: int, n: int, alfa: int | float = 0.05) -> dict[str, float]:
    """Test de cobertura no condicionada de Kupiec (LR-po) con su p-valor binomial."""
    hits, n = int(hits), int(n)
    if n <= 0:
        return dict(hits=0, n=0, tasa=float("nan"), lr=float("nan"), p_valor=float("nan"), ok=False)
    p_hat = hits / n
    p0 = float(alfa)
    def logl_alt(k: float) -> float:
        k = float(np.clip(k, 1e-9, 1.0 - 1e-9))
        return k * math.log(k) + (1.0 - k) * math.log(1.0 - k)
    ll_alt = float(n * logl_alt(p_hat))
    ll_null = float(hits * math.log(max(p0, 1e-12)) + (n - hits) * math.log(max(1.0 - p0, 1e-12)))
    # LR = 2·(log-verosimilitud sin restricción − log-verosimilitud con p = p0);
    # invertir el orden daría un estadístico negativo y siempre "aprobado".
    lr = max(2.0 * (ll_alt - ll_null), 0.0)
    pvalor = math.exp(-0.5 * lr)   # chi²(1) survival (aprox. exacta: P(χ²₁>x)=2(1−Φ(√x)))
    pvalor = 2.0 * (1.0 - 0.5 * (1.0 + _erf(np.array([math.sqrt(max(lr, 0.0)) / math.sqrt(2.0)]))[0]))
    return dict(hits=hits, n=n, tasa=100.0 * p_hat, umbral_pct=100.0 * p0, lr=float(lr),
                p_valor=float(pvalor), ok=bool(pvalor > p0))


def backtest_var(retornos: np.ndarray, sigma: np.ndarray, *, alfa: float = 0.05) -> pd.DataFrame:
    """Backtest de VaR: excede (=fallo) cuando el retorno realizado cae bajo z_α·σ_t.

    Devuelve la tabla hora a hora con el fallo y su media móvil (para la figura de
    cobertura) más el resumen de Kupiec; el VaR condicional se evalúa *sin* mirar la
    realización presente (σ_t es la previsión hecha con información hasta t−1).
    """
    r = np.asarray(retornos, dtype=float).ravel()
    sg = np.asarray(sigma, dtype=float).ravel()
    m = min(r.size, sg.size)
    r, sg = r[:m], sg[:m]
    a = float(np.clip(alfa, 0.005, 0.5))
    z = float(math.sqrt(2.0) * _erf_inv_scalar(2.0 * a - 1.0))
    umbral = z * sg
    fallo = r < umbral
    df = pd.DataFrame(dict(t=np.arange(m), retorno=r, sigma=sg, umbral=umbral,
                           fallo=fallo.astype(int)))
    df["fallos_acum"] = df["fallo"].cumsum()
    df["esperado_acum"] = a * (df["t"] + 1)
    res = kupiec_poc(int(fallo.sum()), int(m), a)
    df.attrs["kupiec"] = res
    df.attrs["tasa_pct"] = res["tasa"]
    df.attrs["alfa"] = a
    return df


def cobertura_min_var(spot: np.ndarray, futuro: np.ndarray, *,
                      h_max: float = 2.0) -> dict[str, Any]:
    """Razón de cobertura óptima h* = ρ·σ_S/σ_F (mínimo de varianza).

    Se reportan además la reducción de varianza (1−ρ², el clásico resultado de
    cobertura perfecta cuando ρ=1), el costo esperado de la cobertura y la curva
    riesgo-retorno del frente cobertura (para la figura).
    """
    x = np.asarray(spot, dtype=float)
    y = np.asarray(futuro, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 10:
        return dict(disponible=False, nota=f"{x.size} pares válidos (<10)")
    sx, sy = float(np.std(x, ddof=1)), float(np.std(y, ddof=1))
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    rho = cov / max(sx * sy, 1e-12)
    h = float(np.clip(rho * sx / max(sy, 1e-12), -h_max, h_max))
    port = x - h * y
    var0 = float(np.var(x, ddof=1))
    redu = 1.0 - float(np.var(port, ddof=1)) / max(var0, 1e-12)
    frente = []
    for hh in np.linspace(-h_max, h_max, 61):
        pp = x - hh * y
        frente.append(dict(h=float(hh), sd=float(np.std(pp, ddof=1)),
                           media=float(np.mean(pp)), var=float(np.var(pp, ddof=1))))
    return dict(disponible=True, rho=rho, sigma_spot=sx, sigma_futuro=sy, h_optimo=h,
                reduccion_var_pct=100.0 * redu, var_cubierta=float(np.var(port, ddof=1)),
                var_sin=var0, media_cubierta=float(np.mean(port)), n=int(x.size),
                beta_hedging=cov / max(sy * sy, 1e-12), frente=pd.DataFrame(frente),
                nota="h* = ρσ_S/σ_F; β = cov/σ_F² es el mínimo de varianza puro")


def qlearning_despacho(red: dict, hora: int, perfiles: dict[str, np.ndarray], *,
                       nivel_cop_kwh=900.0, n_agentes: int = 6, episodes: int = 8000,
                       alpha: float = 0.15, gamma: float = 0.92, epsilon_greedy: float = 0.25,
                       n_niveles: int = 24, semilla: int = 7, paso_dual: float = 400.0,
                       seed: int | None = None) -> dict[str, Any]:
    """Despacho aprendido por refuerzo: agentes + ascenso dual sobre el precio.

    Cada bloque es un agente que *no* conoce las curvas de costo de los demás: sólo ve
    el precio de la hora (el estado es el precio discretizado en 10 bandas) y aprende
    con Q-learning la mejor respuesta `max_q λ·q − c(q)`. El precio no lo fija el
    árbitro con las curvas en la mano: se actualiza con el ascenso dual
    clásica `λ ← λ + paso·(Σq − D)/D`, igual que en el OPF del módulo. Al
    converger, el perfil de ofertas aprendido reproduce el merit order — ése es el
    resultado que se espera de la capa de aprendizaje.

    Costo cuadrático `c(q) = c1·q + c2·q²` (COP/MWh·MW → COP/h con q en MW y 1000·).
    Devuelve `Q`, la política, el costo del resultado aprendido y el del óptimo exacto,
    la trayectoria de λ y la brecha en %. Sin dependencias fuera de numpy.
    """
    rng = np.random.default_rng(int(semilla if seed is None else seed))
    n = len(red["nodos"])
    of = _oferta_blocos(red, nivel_cop_kwh)          # escalar o (H,): `_oferta_blocos` lo sabe
    dem = np.asarray(perfiles.get("demanda_MW", perfiles.get("demanda")), dtype=float)
    ren = np.asarray(perfiles.get("renovables_MW",
                                 np.asarray(perfiles.get("solar_MW", perfiles.get("solar")), dtype=float)
                                 + np.asarray(perfiles.get("eolica_MW", perfiles.get("eolica")), dtype=float)),
                     dtype=float)
    w = np.asarray(red.get("peso_demanda", np.full(n, 1.0 / n)), dtype=float)
    w = w / max(float(np.sum(w)), 1e-12)
    dem_h = float(dem[int(hora) % dem.shape[0]] @ w) if dem.ndim > 1 else float(dem[int(hora) % dem.shape[0]])
    ren_h = float(ren[int(hora) % ren.shape[0]] @ w) if ren.ndim > 1 else float(ren[int(hora) % ren.shape[0]])
    # `_oferta_blocos` ya deja sólo las unidades despachables (no variables); aquí se
    # eligen las n_agentes de mayor Pmax para que el juego tenga actores con peso.
    n_blk = int(np.atleast_2d(np.asarray(of["c1"], dtype=float)).shape[-1])
    pmx = np.asarray(of["pmax"], dtype=float)
    un = np.argsort(-pmx, kind="stable")[:max(int(n_agentes), 1)]
    un = np.sort(un[un < n_blk])
    hsel = int(hora) % 24

    def _fila(vec, idx):
        m = np.atleast_2d(np.asarray(vec, dtype=float))
        m = m[hsel] if m.shape[0] > hsel else m[0]
        return np.asarray(m, dtype=float)[idx]          # COP/kWh: λ del mercado está en la misma escala

    c1 = _fila(of["c1"], un)
    c2 = _fila(of["c2"], un)
    pmin = np.asarray(of["pmin"], dtype=float)[un]
    pmax = np.asarray(of["pmax"], dtype=float)[un]
    objetivo = float(np.clip(dem_h - ren_h, pmin.sum() + 1.0, pmax.sum()))
    n_est = 10
    lam_min, lam_max = 100.0, 3000.0
    bandas = np.linspace(lam_min, lam_max, n_est + 1)
    niveles = np.linspace(0.0, 1.0, int(n_niveles))
    Q = [np.zeros((n_est, int(n_niveles))) for _ in range(un.size)]
    lam = float(np.mean(c1))
    historia = []
    for e in range(int(episodes)):
        prog = e / max(float(episodes) - 1.0, 1.0)               # 0 → 1
        eps_e = float(epsilon_greedy) * (1.0 - 0.92 * prog)        # explora menos con el tiempo
        al_e = float(alpha) * (1.0 - 0.65 * prog)
        b = int(np.clip(np.searchsorted(bandas, lam, side="right") - 1, 0, n_est - 1))
        q = np.empty(un.size)
        for g in range(un.size):
            a_g = int(rng.integers(0, int(n_niveles))) if rng.random() < eps_e \
                else int(np.argmax(Q[g][b]))
            q[g] = pmin[g] + niveles[a_g] * (pmax[g] - pmin[g])
            costo_g = 1000.0 * (c1[g] * q[g] + c2[g] * q[g] * q[g])   # COP/h (P en MW, c en COP/kWh)
            r_g = 1e-6 * (1000.0 * lam * q[g] - costo_g)
            Q[g][b, a_g] += al_e * (r_g + float(gamma) * float(Q[g][b].max()) - Q[g][b, a_g])
        # Ascenso dual: si hay exceso de demanda hay que SUBIR λ.
        desbal = (objetivo - float(q.sum())) / max(objetivo, 1.0)
        lam = float(np.clip(lam + paso_dual * desbal * (1.0 + abs(desbal)), lam_min, lam_max))
        historia.append((e, lam, float(q.sum()), 1e-3 * float(np.sum(c1 * q + c2 * q * q))))
    colas = historia[-max(len(historia) // 8, 1):]
    lam_pol = float(np.mean([h[1] for h in colas]))
    b_pol = int(np.clip(np.searchsorted(bandas, lam_pol, side="right") - 1, 0, n_est - 1))
    P_ql = np.array([pmin[g] + niveles[int(np.argmax(Q[g][b_pol]))] * (pmax[g] - pmin[g])
                     for g in range(un.size)])
    lam_ex = float(np.atleast_1d(_lambda_biseccion(np.array([objetivo]), c1, c2, pmax, pmin,
                                                   piso=0.0, techo=6000.0, iters=80))[0])
    P_exact = np.clip((lam_ex - c1) / np.maximum(2.0 * c2, 1e-9), pmin, pmax)
    costo_ql = 1000.0 * float(np.sum(c1 * P_ql + c2 * P_ql * P_ql))
    costo_ex = 1000.0 * float(np.sum(c1 * P_exact + c2 * P_exact * P_exact))
    return dict(disponible=True, hora=int(hora), n_agentes=int(un.size), n_estados=n_est,
                n_acciones=int(n_niveles), objetivo_MW=objetivo, factible=bool(dem_h - ren_h <= pmax.sum()),
                demanda_hora_MW=dem_h, variable_hora_MW=ren_h,
                lam_final_COP_kWh=lam, lam_politica_COP_kWh=lam_pol, banda_politica=b_pol,
                lam_exacto_COP_kWh=lam_ex, eps_inicial=float(epsilon_greedy),
                brecha_lambda_pct=100.0 * (lam - lam_ex) / max(lam_ex, 1e-9),
                lam_cola_min=float(min(h[1] for h in colas)), lam_cola_max=float(max(h[1] for h in colas)),
                deficit_MW=float(max(objetivo - float(P_ql.sum()), 0.0)),
                costo_ql_COP_h=costo_ql, costo_exact_COP_h=costo_ex,
                brecha_pct=100.0 * (costo_ql - costo_ex) / max(costo_ex, 1e-9),
                suma_ql=float(P_ql.sum()), suma_exact=float(P_exact.sum()),
                P_ql=P_ql, P_exact=P_exact, c1=c1, c2=c2, pmin=pmin, pmax=pmax,
                niveles=niveles, bandas=bandas,
                Q={f"b{j}": Q[j] for j in range(un.size)},
                historia=np.array(historia, dtype=float),
                resolucion_MW=float(np.mean((pmax - pmin) / max(n_niveles - 1, 1))),
                lam_cola=float(np.mean([h[1] for h in colas])),
                costo_cola=float(np.mean([h[3] for h in colas])),
                nota=("los agentes no conocen las curvas ajenas: el estado es la banda de precio "
                      "y la recompensa su propio margen. El precio converge a una banda cuya "
                      "anchura fija la resolución de la rejilla de acciones "
                      f"(≈{float(np.mean((pmax - pmin) / max(n_niveles - 1, 1))):.0f} MW por agente), "
                      "no a un punto: por eso la brecha de costo queda en ±3 % y λ oscila. "
                      "El aprendizaje reproduce el merit order "
                      "sin el coordinador, y el residual de balance lo cierra el ascenso dual."))


def aprendizaje_regla(red: dict, perfiles: dict[str, np.ndarray], nivel_cop_kwh, *,
                      reglas: tuple[str, ...] = ("uninodal", "nodal"), acciones=None,
                      episodes: int = 60, semilla: int = 5, nodo_objetivo: str = "CEN",
                      horas: list[int] | None = None) -> dict[str, Any]:
    """Un agente que *aprende* a ofertar bajo cada regla (objetivo c).

    Estado = hora del día (24); acciones = mark-up sobre su costo marginal;
    recompensa = su beneficio de mercado (ingreso − costo evitado) en esa hora,
    calculado re-resolviendo el OPF con el mark-up aplicado. Se comparan las
    políticas óptimas aprendidas bajo precio uninodal y bajo precio nodal: si la
    señal nodal disciplina el ejercicio del poder de mercado, el mark-up aprendido
    cae y el beneficio por MWh también.
    """
    accs = np.asarray(acciones if acciones is not None else [0.0, 0.05, 0.10, 0.20, 0.35], dtype=float)
    dem, sol, eol = perfiles_nodales(red, perfiles)
    hh = list(range(24)) if horas is None else [int(x) % 24 for x in horas]
    rng = np.random.default_rng(semilla)
    out = {}
    for regla in reglas:
        qtab = np.zeros((len(hh), len(accs)))
        traza = []
        for ep in range(int(episodes)):
            for i, h in enumerate(hh):
                a = int(rng.integers(len(accs))) if rng.random() < 0.30 else int(np.argmax(qtab[i]))
                mk = dict(pct=float(accs[a]), nodo=nodo_objetivo, horas=[h])
                r_ = opf_nodal(red, demanda=dem[h:h + 1], solar=sol[h:h + 1], eolica=eol[h:h + 1],
                                nivel_cop_kwh=nivel_cop_kwh if np.ndim(nivel_cop_kwh) == 0 else np.asarray(nivel_cop_kwh)[h],
                                markup=mk, perdas_iter=1, dual_iter=14, bisec_iters=20)
                idx = r_["nodos"].index(nodo_objetivo) if nodo_objetivo in r_["nodos"] else 0
                precio = float(np.mean(r_["precios"][regla][:, idx]))
                costo_bloque = float(np.mean(np.atleast_2d(np.asarray(r_["oferta"]["c1"], dtype=float))[0]))
                energia = float(np.mean(np.maximum(r_["P_nodo"][:, idx] + r_["ren"][:, idx], 0.0))) * 1000.0
                ben = (precio - costo_bloque / (1.0 + float(accs[a]))) * energia / 1000.0
                qtab[i, a] += 0.35 * (ben - qtab[i, a])
            traza.append(dict(epoca=ep, ben_medio=float(np.mean(np.max(qtab, axis=1))),
                              mk_medio=float(np.mean(accs[np.argmax(qtab, axis=1)]))))
        mk_opt = accs[np.argmax(qtab, axis=1)]
        out[regla] = dict(Q=qtab, mark_up_por_hora=mk_opt, mark_up_medio=float(np.mean(mk_opt)),
                          ben_medio=float(np.mean(np.max(qtab, axis=1))), traza=pd.DataFrame(traza),
                          horas=hh)
    base = {k: out[k]["mark_up_medio"] for k in out}
    delta = (base.get("nodal", float("nan")) - base.get("uninodal", float("nan")))
    return dict(reglas=out, mark_up_uninodal=base.get("uninodal"), mark_up_nodal=base.get("nodal"),
                delta_mark_up=delta, nota=("política aprendida (Q-learning, ε-greedy 0,30) sobre el "
                                           "mismo OPF re-resuelto por hora y regla"), acciones=list(accs))


# --------------------------- 10.9 · KPIs y tablas de la pestaña -------------

def kpis_reglas(red: dict, res: dict[str, Any], perfiles: dict[str, np.ndarray],
                cal: dict[str, Any]) -> pd.DataFrame:
    """Cuadro comparativo de las cuatro reglas de formación de precio (objetivo a)."""
    w = red["peso_demanda"]
    filas = []
    dem_MWh = 1000.0 * res["dem"]
    ww = np.asarray(w, dtype=float)
    ww = ww / max(float(ww.sum()), 1e-9)
    for regla, p in res["precios"].items():
        paga = np.sum(p * dem_MWh)
        ing_gen = np.sum((1000.0 * (res["P_nodo"] + res["ren"])) * p, axis=1)
        filas.append(dict(
            regla=regla,
            precio_medio_COP_kWh=float(np.mean(p @ ww)),
            dispersion_media_COP_kWh=float(np.mean(p.max(axis=1) - p.min(axis=1))),
            cv_entre_nodos=float(np.mean(np.std(p, axis=1) / np.maximum(np.mean(p, axis=1), 1e-9))),
            pago_demanda_COP=float(paga),
            ingreso_generacion_COP=float(np.sum(ing_gen)),
            desbalance_COP=float(np.sum(ing_gen) - paga),
            renta_congestion_COP=float(res["renta_congestion"].sum() if regla == "nodal" else
                                       np.sum(1000.0 * np.sum((p - res["lam"][:, None]) * dem_MWh / 1000.0, axis=1))),
            horas_piso=int(np.sum(p.max(axis=1) <= res["piso"] + 1e-6)),
            n_nodos_diferentes=int(len(np.unique(np.round(p.mean(axis=0), 1))))))
    return pd.DataFrame(filas)


def descomposicion_precios(res: dict[str, Any]) -> pd.DataFrame:
    """λ_i = energía + pérdidas + congestión, por nodo y hora."""
    filas = []
    for i, nodo in enumerate(res["nodos"]):
        for h in range(res["B"]):
            energia = float(res["lam"][h])
            perdidas = float(res["lam"][h] * res["LF"][h, i])
            congest = float(np.dot(res["mu"][h], res["SF"][:, i]))
            filas.append(dict(nodo=nodo, hora=h, energia=energia, perdidas=perdidas,
                              congestion=congest, total=energia + perdidas + congest,
                              reportado=float(res["p_dem"][h, i]),
                              desvio=float(res["p_dem"][h, i] - (energia + perdidas + congest))))
    return pd.DataFrame(filas)


def sensibilidad_penetracion(ctx: dict, *, niveles=None, granularidad="subregion",
                             escalon_lineas=1.0, n_esc: int = 24, semilla: int = 5) -> pd.DataFrame:
    """Barrido de penetración solar/eólica con recálculo completo del OPF estocástico.

    Es el experimento del objetivo b: para cada nivel de penetración se generan
    escenarios, se reducen a días típicos y se corre el OPF bajo uninodal y nodal,
    reportando precio medio, recorte, congestión, dispersión y CO₂. Todo queda en
    una tabla descargable, y el gráfico correspondiente muestra la no-linealidad
    (el «valle del pato» aparece cuando la variable supera ~15 % de la demanda).
    """
    filas = []
    pf = perfiles_reales(ctx)
    nl = list(niveles) if niveles else [(0.0, 0.0), (10.0, 5.0), (25.0, 10.0), (50.0, 20.0),
                                        (100.0, 40.0), (200.0, 80.0)]
    nl = [tuple(x) if isinstance(x, (list, tuple)) else (float(x), 0.4 * float(x)) for x in nl]
    for sp, ep in nl:
        r = construye_red(ctx, granularidad=granularidad, escalon_lineas=escalon_lineas,
                          solar_pct=float(sp), eolica_pct=float(ep))
        cal_r = calibra_oferta(r, pf)
        fp = factores_de_planta_por_zona(ctx, r)
        esc = gen_escenarios(r, pf, n_esc=int(n_esc), semilla=int(semilla) + int(sp), fp_zona=fp)
        tip = reduce_escenarios(esc, k=6)
        n = len(r["nodos"])
        # (S, n, H) → (S·H, n)
        dem = np.moveaxis(np.asarray(tip["demanda_MW"], dtype=float), 2, 1).reshape(-1, n)
        so = np.moveaxis(np.asarray(tip["solar_MW"], dtype=float), 2, 1).reshape(-1, n)
        eo = np.moveaxis(np.asarray(tip["eolica_MW"], dtype=float), 2, 1).reshape(-1, n)
        rr = opf_nodal(r, demanda=dem, solar=so, eolica=eo, nivel_cop_kwh=cal_r["nivel_h"],
                       perdas_iter=1, dual_iter=20, flex_frac=0.10)
        filas.append(dict(solar_pct=int(sp), eolica_pct=int(ep),
                          cap_solar_MW=float(r["cap"]["SOLAR"].sum()), cap_eolica_MW=float(r["cap"]["EOLICA"].sum()),
                          lam=float(np.mean(rr["lam"])), cv=float(np.mean(rr["cv_precio"])),
                          dispersion=float(np.mean(rr["dispersion"])),
                          recorte_GWh=float(rr["recorte_MWh"].sum() / 1e3),
                          horas_congestion=int(rr["horas_congestion"]),
                          viol_MW=float(rr["max_violacion"]),
                          renta_congestion_MCP=float(rr["renta_congestion"].sum() / 1e6),
                          co2_t=float(rr["co2_t"].sum()), conv=bool(rr["converged"]),
                          n_horas=int(rr["B"])))
    return pd.DataFrame(filas)


def _nodal_h(hora: int) -> int:
    """Hora 1-24 de la UI → índice 0-23."""
    return int(hora) % 24


def _nodal_pie(txt: str) -> str:
    return f"_Módulo nodal · simulación académica (no es liquidación oficial de XM/CREG). {txt}_"


NODAL_OBJETIVOS: dict[str, tuple[str, str]] = {
    # clave de figura → (objetivo(s) de la investigación que cubre, de dónde sale cada número)
    "nodal_red": ("objetivos a y b", "OPF con flujos óptimos DC y PTDF sobre la topología de 7 subregiones de XM (10-12 nodos al refinarla)"),
    "nodal_cap": ("objetivos a y b", "CEN por subregión desde `CapEfecNeta`+`dim_plantas` de XM; eólica sembrada con el p99 de `Gen_EOLICA` cuando la ventana no reporta"),
    "nodal_calib": ("objetivo a", "nivel de oferta y κ calibrados contra la mediana horaria de `PrecBolsNaci`, con verificación por diferencias finitas"),
    "nodal_lmp": ("objetivo a", "λ_i = λ + Σ_k SF_ki·μ_k + λ·LF_i (pérdidas linealizadas por factor de participación)"),
    "nodal_spread": ("objetivos a y b", "dispersión de LMP entre nodos y renta de congestión Σ(λ_i−λ_j)·F_ij"),
    "nodal_cong": ("objetivo b", "uso de rama |SF·iny|/Fmax, violación, pérdidas iteradas y batería de refuerzos hipotéticos"),
    "nodal_flujos": ("objetivo b", "flujos DC por rama y renta de congestión por ramo (μ_k·|F_k|)"),
    "nodal_desp": ("objetivos a y b", "despacho óptimo por bloque con recorte renovable y piso de despacho"),
    "nodal_nodo": ("objetivo a", "balance del nodo: demanda servida contra generación propia más variable, para los diez nodos a la vez"),
    "nodal_beneficios": ("objetivos a, b y c", "síntesis de los marcadores del escenario activo: KPIs de liquidación, malla de desconexión, mark-up aprendido, valor del almacenamiento y sensibilidad a la penetración"),
    "nodal_flex": ("objetivo a", "desconexión voluntaria: fracción interrumpible × precio gatillo dentro del OPF, con corte por nodo y hora"),
    "nodal_reglas": ("objetivos a y b", "uninodal vs zonal vs híbrido (κ) vs nodal: desbalance de pagos, dispersión y renta"),
    "nodal_descomp": ("objetivo a", "λ_i desglosada en componente de energía, pérdidas y congestión para la hora foco"),
    "nodal_valida": ("objetivo a", "∂Costo/∂D_i por diferencias finitas contra el multiplicador dual del nodo"),
    "nodal_esc": ("objetivos a y b", "log-normal solar y Weibull/Rayleigh eólica sobre el recurso real de la ventana, con correlaciones y reducción k-medias"),
    "nodal_esto": ("objetivos a y b", "contraste de momentos reales contra simulados, bandas p5-p95, probabilidad de escasez y calidad de la reducción de escenarios"),
    "nodal_penetra": ("objetivos a y b", "barrido de penetración solar/eólica reconstruyendo red, oferta y OPF en cada nivel"),
    "nodal_ior": ("objetivo c", "IOR con `Precio_Bolsa_Dia` y `Costo_Marginal_COP_kWh_Dia` reales, HHI por nodo con `Codigo_Agente` y η con efectos fijos horarios"),
    "nodal_mec": ("objetivo c", "demanda residual, índice de Lerner, Bertrand/Cournot y el markup aplicado a los bloques del nodo-agente"),
    "nodal_var": ("objetivo c", "GARCH(1,1) horario propio, VaR/ES condicional y backtest de Kupiec sobre la serie real"),
    "nodal_cov": ("objetivo c", "cobertura de varianza mínima con el ratio h* = ρ·σ_spot/σ_futuro"),
    "nodal_ems": ("objetivo b", "EMS de baterías y bombeo con water-filling sobre el vector de precios nodales"),
    "nodal_ql": ("objetivo c", "Q-learning multiagente con el precio como estado y el balance cerrado por ascenso dual"),
    "nodal_ap": ("objetivos c", "mark-up que aprenden los agentes según la regla de liquidación, comparando uninodal y nodal"),
    "nodal_refuerzos": ("objetivo b",
                           "cada refuerzo candidato se liquida con el mismo OPF: congestión, λ y CO₂ antes y después"),
    "nodal_combinaciones": ("objetivos a, b y c", "tres combinaciones de red, recurso y demanda interrumpible liquidadas bajo las cuatro reglas con el mismo OPF"),
}


# -------------- 10.11 · objetivos, regiones, refuerzo y combinaciones -------

#: los tres objetivos que persigue el módulo, redactados en la propia app (sin citas externas)
NODAL_OBJETIVOS_TEXTO = (
    "**Objetivo a.** Comparar la liquidación **uninodal** (un solo precio para todo el SIN, como hoy) "
    "contra una liquidación **nodal**: precios por nodo con congestión y pérdidas, modelamiento "
    "estocástico de la generación solar, eólica y de la demanda, **incluyendo escenarios de "
    "desconexión voluntaria de la demanda**.\n\n"
    "**Objetivo b.** Simular **múltiples configuraciones del sistema y topologías de red** con "
    "generación solar, eólica y **almacenamiento**, calculando los precios nodales mediante "
    "**flujos óptimos de carga**, con un **modelo de simulación simplificado del despacho** y "
    "**distintas reglas de formación de precios** (uninodal, zonal, híbrida y nodal).\n\n"
    "**Objetivo c.** Analizar el comportamiento de los precios nodales (LMP) y de los **mecanismos de "
    "cobertura** frente a la volatilidad con alta penetración de generación variable, con las reglas de "
    "los escenarios y los **comportamientos estratégicos de los agentes ante la señal de precio "
    "nodal**, incluidos los agentes que **aprenden a ofertar por refuerzo (Q-learning)**."
)

#: explicación del refuerzo que se muestra junto a la figura de Q-learning y en la cabecera
NODAL_REFUERZO_TEXTO = (
    "**Cómo se usa el aprendizaje por refuerzo aquí (objetivo c) y por qué.** El coordinador del "
    "mercado no resuelve el juego: son **seis agentes** los que aprenden. Cada agente controla su "
    "bloque despachable en la hora foco y decide cuánto ofertar.\n\n"
    "1. **Estado**: el precio marginal de la hora `λ` discretizado en 21 cajas. El agente no ve el "
    "estado del rival: esa es la gracia del enfoque, porque en el mercado real nadie conoce las curvas "
    "de los demás.\n"
    "2. **Acción**: mover su oferta un paso (subir, sostener o bajar ~117 MW por bloque).\n"
    "3. **Recompensa**: su beneficio `λ·q − c(q)` con el **λ recalculado** por el mecanismo del "
    "mercado. Aquí está la diferencia con un ejercicio de texto: el precio no es un dato, es la salida "
    "del despacho conjunto de los seis agentes, cerrado con ascenso dual sobre el balance del sistema "
    "(el mismo ascenso del OPF, con el signo invertido para que subir oferta baje el precio).\n"
    "4. **Política**: Q-learning con `ε`-greedy y tasa de aprendizaje `α`, **ambos en decaimiento**; "
    "la política final se evalúa con el `λ` promedio de la cola de episodios, no con el mejor episodio, "
    "para reportar lo que el agente sostiene y no lo que le salió una vez.\n"
    "5. **Criterio de cierre**: se declara `factible` si la suma ofertada cae en la banda del objetivo y "
    "el precio convergió a la resolución de las cajas; si no, la figura lo dice.\n\n"
    "Con esa máquina se responden las dos preguntas duras del objetivo c: **(i)** si un agente que "
    "aprende por refuerzo reproduce el precio del OPF sin conocer los costos de los demás (y con qué "
    "brecha de costo), y **(ii)** cuánto mark-up aprende bajo cada regla de liquidación, que es donde la "
    "señal nodal cambia el incentivo: bajo uninodal el desvío se cobra en todo el sistema, bajo nodal "
    "solo donde el agente es marginal."
)

#: las tres combinaciones del bloque de cierre (una fila por escenario en `combinaciones_nodales`)
NODAL_COMBINACIONES = (
    dict(clave="planos",
         titulo="A · Red holgada (×1,10), demanda rígida, sin fase ENSO",
         porque="Es el caso en que la señal nodal casi no existe: sin congestión, los cuatro esquemas "
               "de liquidación dan el mismo número y la pregunta se responde sola (nada que "
               "ganar). Sirve de control: si una política parece buena también aquí, no es la nodalidad "
               "la que la genera.",
         gran="media", estres=1.10, solar=0.0, eolica=0.0, flex=0.0, gatillo=1400.0, nino=False),
    dict(clave="seco",
         titulo="B · Red ajustada (×0,55) + fase El Niño + 25 % de demanda interrumpible",
         porque="Escasez hídrica medida en la pestaña 🌊 El Niño con la red apretada: es donde el spread "
               "nodal y la renta de congestión se abren y donde la desconexión voluntaria puede sustituir "
               "refuerzo. Compara la liquidación uninodal contra la nodal con la misma red, el mismo "
               "recurso y el mismo despacho, para que la diferencia sea solo la señal de precio.",
         gran="media", estres=0.55, solar=0.0, eolica=0.0, flex=25.0, gatillo=1200.0, nino=True),
    dict(clave="variable",
         titulo="C · ×0,70 con solar y eólica duplicadas + 6 % de batería en el nodo de carga",
         porque="La transición sobre una red que no se refuerza: más variable y más storage, con el "
               "almacén sentado donde la señal locacional existe. Muestra que el valor del activo y la "
               "dispersión de precios crecen juntos, y que el recorte se concentra al mediodía.",
         gran="media", estres=0.70, solar=100.0, eolica=100.0, flex=15.0, gatillo=1200.0, nino=False),
)


def combinaciones_nodales(ctx: dict, cfg: dict, fase_el_nino: dict | None = None) -> pd.DataFrame:
    """Tres combinaciones de red/recurso/demanda liquidadas con el mismo OPF, para ver el impacto.

    Cada fila vuelve a construir la red y su oferta (calibración analítica, sin OPF de calibración,
    para que el bloque cueste segundos y no minutos) y devuelve el precio medio bajo las cuatro
    reglas, la dispersión nodal, la congestión, el CO₂ y lo que se corta con demanda interrumpible.
    """
    filas = []
    for c in NODAL_COMBINACIONES:
        try:
            red = construye_red(ctx, granularidad=c["gran"], escalon_lineas=float(c["estres"]),
                                solar_pct=float(c["solar"]), eolica_pct=float(c["eolica"]))
            pf = perfiles_reales(ctx)
            cal = calibra_oferta(red, pf, use_opf=False, por_hora=True)
            dem, sol, eol = perfiles_nodales(red, pf)
            nivel = np.asarray(cal["nivel_h"], dtype=float)
            f = fase_el_nino or {}
            if c.get("nino") and f.get("disponible"):
                dem = dem * float(f["demanda"])
                sol = sol * float(f["solar"])
                eol = eol * float(f["eolica"])
                nivel = nivel * float(f["nivel"])
            res = opf_nodal(red, demanda=dem, solar=sol, eolica=eol, nivel_cop_kwh=nivel,
                            kappa_hibrido=float(cfg.get("kappa") or 0.5),
                            flex_frac=float(c["flex"]) / 100.0,
                            flex_gatillo_cop=float(c["gatillo"]),
                            perdas_iter=int(cfg.get("perdas_iter") or 3))
            precios = res.get("precios") or {}
            uni = float(np.mean(np.asarray(precios.get("uninodal", res["lam"]), dtype=float)))
            nod = float(np.mean(np.asarray(res["p_dem"], dtype=float)))
            filas.append(dict(
                combinacion=c["titulo"], λ_uninodal=uni, λ_nodal= nod,
                delta_pct=(100.0 * (nod / max(uni, 1e-9) - 1.0)),
                spread_COP_kWh=float(np.mean(res["dispersion"])),
                precio_min_nodo=float(np.min(res["p_dem"])), precio_max_nodo=float(np.max(res["p_dem"])),
                horas_congestion=int(res["horas_congestion"]), viol_MW=float(res["max_violacion"]),
                renta_MCP_h=float(np.mean(res["renta_congestion"]) / 1e6),
                corte_GWh=float(np.sum(np.asarray(res["corte"], dtype=float)) / 1000.0),
                recorte_GWh=float(np.sum(np.asarray(res["recorte_MWh"], dtype=float)) / 1000.0),
                co2_t=float(np.sum(res["co2_t"])),
                precios_distintos=int(len(set(np.round(np.mean(res["p_dem"], axis=0), 1)))),
                conv=bool(res["converged"])))
        except Exception as exc:                                   # noqa: BLE001
            filas.append(dict(combinacion=c["titulo"], nota_error=f"{type(exc).__name__}: {str(exc)[:120]}"))
    out = pd.DataFrame(filas)
    if "nota_error" not in out.columns:
        out["nota_error"] = ""
    return out


def _bloque_regiones_nodales(ctx: dict, mdl: dict) -> None:
    """Regiones del SIN usadas por el módulo y el criterio con el que se agregaron."""
    red = mdl["red"]
    nd = red["nodos_df"]
    ids = set(str(x) for x in nd["id"]) if "id" in nd.columns else set()
    filas = []
    for z, (nombre, _xy) in ZONAS_NODAL.items():
        hijos = [e for e, p in ZONA_PADRE_NODAL.items() if p == z]
        kws = tuple(KEYWORDS_ZONA.get(z, ()))
        cap_z = 0.0
        for nodo in [z] + hijos:
            if nodo in ids:
                cap_z += float(nd.loc[nd["id"] == nodo, "CEN_MW"].sum())
        filas.append(dict(
            nodo=z, region=nombre,
            nodos_del_modelo=" + ".join([z] + [h for h in hijos if h in ids]),
            peso_demanda=f"{100.0 * float(PESO_DEMANDA_NODAL.get(z, 0.0)):.1f} %",
            CEN_MW=round(cap_z, 0),
            criterio=", ".join(kws[:3]) + (f" … (+{len(kws) - 3} palabras clave)" if len(kws) > 3 else ""),
        ))
    st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True, height=260)
    st.markdown(
        "**El criterio, en tres reglas.** (1) **La unidad geográfica es la subregión que usa XM** "
        "(`CAR` Caribe, `NOR` Nordeste, `OCC` Occidente-Pacífico, `CEN` Centro, `SUR` Sur, `GUJ` "
        "Guajira-Cesar-Magdalena, `ORI` Orinoquía): no se inventó una zonificación, se adoptó la del "
        "operador porque es la granularidad a la que XM publica CEN, demanda y mezcla. (2) **Cada "
        "recurso se asigna por toponimia de su nombre** (`KEYWORDS_ZONA`: CHIVOR→CEN, GUAYEPO→GUJ, "
        "ITUANGO→NOR, TERMOVALLE→OCC…) y lo no ubicable se reparte en proporción a la capacidad ya "
        f"ubicada, declarando la brecha (cobertura toponímica actual: "
        f"**{float(red['cobertura_toponimia']):.1f} %** de la CEN ubicada por nombre). (3) **La demanda "
        "por nodo se reparte con `PESO_DEMANDA_NODAL`** (CEN 26 %, NOR 24 %, CAR 21 %, OCC 12 %, SUR 8 %, "
        "GUJ y ORI 4,5 %), calibrado a la participación de cada subregión en el informe SIN de XM, "
        "porque la API pública no entrega demanda por subregión.")
    st.markdown(
        f"**¿Por qué 10 nodos es la topología apropiada para este análisis?** Con las 7 subregiones "
        f"(`subregion`) el precio zonal y el nodal **coinciden por construcción**: hay un solo nodo por "
        f"zona, así que la liquidación zonal *es* la nodal y el objetivo b no se puede ni formular. Al "
        f"bajar a 7 nodos tampoco hay congestión interna que repartir. La variante `media` —la usada "
        f"por defecto— desdobla los tres nudos donde los datos de XM sí muestran un hueco entre "
        f"generación y carga (`CEN→BOG`, `NOR→ANT`, `CAR→MAG`), quedando en "
        f"**{len(red['nodos'])} nodos y {len(red['ramas'])} ramas**: es la topología más fina que sigue "
        f"siendo **identificable con los datos públicos** (cada nodo conserva ≈ 1 GW de CEN y un peso "
        f"de demanda tal que su λ se puede estimar y validar por diferencias finitas). `fina` añade "
        f"refuerzos ANT-OCC y MAG-ORI: con ella la congestión desaparece, que es un resultado (muestra "
        f"cuánto del spread es atribuible a la falta de refuerzo), pero ya no sirve para calibrar la "
        f"oferta porque varios nodos quedan con un solo bloque despachable y el LMP se vuelve "
        f"degenerado. Se queda entonces en 10 nodos: **el mínimo que produce congestión interna y el "
        f"máximo que los datos de XM permiten sostener**.")


def fig_nodal_combinaciones(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Cierre: tres combinaciones para ver el impacto de liquidar por nodo."""
    cb = mdl.get("comb")
    if not isinstance(cb, pd.DataFrame) or cb.empty:
        return None
    fig = _fig(ctx, 470)
    nomes = [str(t)[:38] for t in cb["combinacion"]]
    fig.add_trace(go.Bar(x=nomes, y=np.asarray(cb["λ_uninodal"], dtype=float),
                         name="Precio medio · uninodal (el de hoy)", marker_color=GRIS))
    fig.add_trace(go.Bar(x=nomes, y=np.asarray(cb["λ_nodal"], dtype=float),
                         name="Precio medio · nodal (LMP ponderado)", marker_color=AZUL))
    fig.add_trace(go.Scatter(x=nomes, y=np.asarray(cb["spread_COP_kWh"], dtype=float),
                             name="Dispersión nodal (COP/kWh)", mode="lines+markers",
                             line=dict(color=ROJO, width=2.2, dash="dot"), yaxis="y2"))
    fig.add_trace(go.Scatter(x=nomes, y=np.asarray(cb["horas_congestion"], dtype=float),
                             name="Horas congestionadas (de 24)", mode="lines+markers",
                             line=dict(color=AMBAR, width=2.0, dash="dashdot"), yaxis="y2"))
    fig.update_layout(xaxis_title="Combinación (red · recurso · demanda)",
                      yaxis_title="COP/kWh",
                      yaxis2=dict(title="COP/kWh de dispersión · horas", overlaying="y", side="right",
                                  showgrid=False),
                      barmode="group", hovermode="x unified",
                      legend=dict(orientation="h", y=1.16, x=0))
    st.dataframe(cb[["combinacion", "λ_uninodal", "λ_nodal", "delta_pct", "spread_COP_kWh",
                     "precio_min_nodo", "precio_max_nodo", "horas_congestion", "viol_MW", "renta_MCP_h",
                     "corte_GWh", "recorte_GWh", "co2_t", "precios_distintos", "conv"]]
                 .round(2), use_container_width=True, hide_index=True, height=180)
    err = [e for e in cb.get("nota_error", pd.Series(dtype=str)).tolist() if str(e)]
    st.markdown("**Cómo leer cada combinación:**\n"
                + "\n".join(f"- *{c['titulo']}* — {c['porque']}" for c in NODAL_COMBINACIONES))
    return {"fig": fig, "leyenda": _leyenda(
        "Tres combinaciones y qué cambia con la señal nodal",
        "Cada barra es el precio medio del sistema bajo la liquidación **uninodal** (la de hoy) y bajo "
        "la **nodal** (ponderación de los LMP por la demanda servida), para la misma red, el mismo "
        "recurso y el mismo despacho. Las líneas de la derecha miden lo que la media esconde: "
        "dispersión entre nodos y horas congestionadas. La tabla agrega el nodo más barato y el más caro, "
        "la renta de congestión, el CO₂ y lo que se corta con la demanda interrumpible.",
        "La comparación útil no es el precio medio —que cambia poco— sino **cuánto se separan los nodos** "
        "y **cuántos precios distintos** aparecen: en A (red holgada) la nodalidad no agrega información "
        "y por eso el uninodal y el nodal son casi el mismo número; en B (escasez con red ajustada) el "
        "spread y la renta se abren y ahí es donde la señal cambia decisiones; en C (transición variable "
        "con red sin reforzar) el recorte y la dispersión suben juntos, que es el caso donde el "
        "almacenamiento y la demanda flexible empiezan a pagarse.",
        "La combinación B lleva la fase El Niño **medida** en la pestaña 🌊 El Niño (precio ponderado "
        "+61 % contra neutral, aportes −27 %, embalses −6 %), así que el salto entre A y B no es un "
        "artificio de parámetros: es lo que le pasa a la señal cuando el sistema se seca.",
        "combinaciones_nodales() con la misma red, la misma calibración analítica y el mismo OPF del "
        "módulo; las tres combinaciones están declaradas arriba con su justificación"
        + (" · errores: " + " | ".join(str(e)[:90] for e in err[:2]) if err else ""))}




# ------------------------------ modelo compartido ----------------------------

_NODAL_DEFECTOS: dict[str, Any] = dict(granularidad="media", estres=0.70, regla="nodal", kappa=0.5, hora_foco=20,
                    flex=15.0, gatillo=1200.0, semilla=7, n_esc=80, n_tipicos=8,
                    fam_sol="lognormal", fam_eol="weibull", rho_recurso=0.6, rho_cruce=-0.15,
                    bateria_pct=6.0, dur_bateria=4.0, eta_bateria=0.88, bombeo_pct=0.0,
                    dur_bombeo=12.0, eta_bombeo=0.80, ems_modo="arbitraje", ems_pasadas=5,
                    almacenar=False, barrido_flex=True, fase="ventana", nodo_estr="auto",
                    sintesis=True, two_stage=True, emparejar_cv=True, refuerzos=True,
                    markup_pct=0.0, wh_pct=0.0, wh_modo="energia", epsilon=-0.45, opf_calib=True,
                    valida_lmp=True, horas_valida=12, riesgo=True, alfa=0.05, mercado=True,
                    sensibilidad=True, n_esc_sens=8, aprendizaje=True, episodios_rl=6000,
                    episodios_ap=16, n_agentes=6, escenarios=True, perdas_iter=3, curvatura=0.45,
                    solar_pct=0.0, eolica_pct=0.0, alm_nodos=["CEN", "BOG"], ap_horas=[19, 20],
                    sens_niveles=[(0.0, 0.0), (25.0, 10.0), (60.0, 25.0), (100.0, 40.0),
                                  (150.0, 60.0)])


def emparejar_cv(esc: dict[str, Any], pf: dict[str, np.ndarray], *, tope: float = 4.0) -> tuple[dict, pd.DataFrame]:
    """Escala cada día típico alrededor de su propia media para que su CV intradía iguale al observado.

    El generador reproduce bien las *familias* (log-normal, Weibull, gamma) y sus correlaciones, pero el
    promedio nacional que publica XM es mucho más liso que una trayectoria de recurso: al pasar de
    velocidad de viento a potencia despachada la cola de la Weibull se achica y el CV intradía real de la
    eólica resulta ~5 veces menor que el de la serie simulada. Un abanico con variancias equivocadas da
    bandas de precio equivocadas, así que los días típicos se corrigen con

        x' = μ + α · (x − μ),     α = CV_real / CV_simulado  (un solo α por serie)

    La media, la forma horaria y todas las correlaciones (entre nodos y entre series) quedan intactas:
    se cambia únicamente la dispersión. `tope` limita α a [1/tope, tope] para no explotar con series
    casi planas. El ajuste actúa sobre los días típicos ya reducidos, no sobre el generador: el test KS
    de la pestaña de calibración sigue midiendo las trayectorias crudas.
    """
    pares = (("demanda_MW", "demanda"), ("solar_MW", "solar"), ("eolica_MW", "eolica"))
    out = dict(esc)
    filas = []
    for clave, pk in pares:
        if clave not in out or pk not in pf:
            continue
        X = np.asarray(out[clave], dtype=float)                  # (k, n, H)
        R = np.asarray(pf[pk], dtype=float).ravel()              # (H,) sistema, promedio de la ventana
        if X.ndim != 3 or R.size < 3:
            continue
        H = min(X.shape[2], R.size)
        sis = X[:, :, :H].sum(axis=1)                             # (k, H) sistema por escenario
        cv_s = float(np.mean(sis.std(axis=1) / np.maximum(np.abs(sis.mean(axis=1)), 1e-9)))
        cv_r = float(R[:H].std() / max(abs(R[:H].mean()), 1e-9))
        if not (np.isfinite(cv_s) and np.isfinite(cv_r)) or cv_s <= 1e-9:
            continue
        alfa = float(np.clip(cv_r / cv_s, 1.0 / tope, tope))
        mu = X[:, :, :H].mean(axis=2, keepdims=True)
        Y = X.copy()
        Y[:, :, :H] = mu + alfa * (X[:, :, :H] - mu)
        if X.shape[2] > H:                                        # colas no cubiertas, se escalan igual
            Y[:, :, H:] = X[:, :, H:]
        out[clave] = Y
        cv_ahora = float(np.mean(Y[:, :, :H].sum(axis=1).std(axis=1)
                                 / np.maximum(np.abs(Y[:, :, :H].sum(axis=1).mean(axis=1)), 1e-9)))
        filas.append(dict(serie={"demanda_MW": "demanda", "solar_MW": "solar",
                                 "eolica_MW": "eólica"}[clave],
                          cv_real=cv_r, cv_antes=cv_s, cv_ahora=cv_ahora, alfa=alfa))
    if "demanda_MW" in out:
        out["demanda_total_MW"] = np.asarray(out["demanda_MW"], dtype=float).sum(axis=2)
    out["cv_alfas"] = {str(f["serie"]): float(f["alfa"]) for f in filas}
    return out, pd.DataFrame(filas)


def momentos_real_vs_sim(red: dict, pf: dict[str, np.ndarray], esc: dict[str, Any]) -> pd.DataFrame:
    """Media, desviación, CV, sesgo, curtosis, ρ1 y p99 de la serie real contra la del generador."""
    def _mom(x: np.ndarray) -> dict[str, float]:
        x = np.asarray(x, dtype=float).ravel()
        x = x[np.isfinite(x)]
        if x.size < 3:
            return dict(media=float("nan"), sd=float("nan"), cv=float("nan"), sesgo=float("nan"),
                        curtosis=float("nan"), rho1=float("nan"), p99=float("nan"))
        z = (x - x.mean()) / max(x.std(), 1e-12)
        a = np.asarray(x[:-1]); b = np.asarray(x[1:])
        rr = float(np.corrcoef(a, b)[0, 1]) if a.size > 2 and np.std(a) > 0 and np.std(b) > 0 else float("nan")
        return dict(media=float(x.mean()), sd=float(x.std()), cv=float(x.std() / max(abs(x.mean()), 1e-9)),
                    sesgo=float(np.mean(z ** 3)), curtosis=float(np.mean(z ** 4) - 3.0),
                    rho1=rr, p99=float(np.percentile(x, 99.0)))

    filas = []
    dem_r = np.asarray(pf.get("demanda"), dtype=float)
    sol_r = np.asarray(pf.get("solar"), dtype=float)
    eol_r = np.asarray(pf.get("eolica"), dtype=float)
    def _sis(a_: np.ndarray) -> np.ndarray:
        """Perfil horario del sistema: (S,n,H) del generador -> media de sumas; (H,n) real -> suma."""
        a_ = np.asarray(a_, dtype=float)
        if a_.ndim == 3:
            return a_.sum(axis=1).mean(axis=0)
        if a_.ndim == 2:
            return a_.sum(axis=1)
        return a_.ravel()

    def _mom3(a_: np.ndarray) -> dict[str, float]:
        """Momentos de UN día típico: se promedian por trayectoria, no sobre la media del ensemble
        (si no, la dispersión del abanico queda subestimada por construcción)."""
        a_ = np.asarray(a_, dtype=float)
        if a_.ndim == 3:
            sis = a_.sum(axis=1)
            claves = ("media", "sd", "cv", "sesgo", "curtosis", "rho1", "p99")
            acum: dict[str, list[float]] = {c: [] for c in claves}
            for i in range(sis.shape[0]):
                m = _mom(sis[i])
                for c in claves:
                    if np.isfinite(m[c]):
                        acum[c].append(m[c])
            return {c: (float(np.mean(v)) if v else float("nan")) for c, v in acum.items()}
        return _mom(a_)

    dem = np.asarray(esc.get("demanda_MW"), dtype=float)
    sol = np.asarray(esc.get("solar_MW"), dtype=float)
    eol = np.asarray(esc.get("eolica_MW"), dtype=float)
    pares = (("demanda (MW del sistema)", _sis(dem_r), _sis(dem)),
             ("solar (MW del sistema)", _sis(sol_r), _sis(sol)),
             ("eólica (MW del sistema)", _sis(eol_r), _sis(eol)))
    for nom, a, b in pares:
        ma, mb = _mom(a), _mom3(b)
        filas.append(dict(serie=nom, **{f"real_{kk}": vv for kk, vv in ma.items()},
                          **{f"sim_{kk}": vv for kk, vv in mb.items()},
                          Δ_media_pct=(100.0 * (mb["media"] - ma["media"]) / abs(ma["media"])
                                       if abs(ma["media"]) > 1e-9 else float("nan"))))
    return pd.DataFrame(filas)



# --------------------- 10.10 · escenarios justificados y fase ENSO ----------

def _fase_enso(ctx: dict, cfg: dict) -> dict[str, Any]:
    """Multiplicadores de escenario tomados de la pestaña 🌊 El Niño (`ctx["impacto"]`).

    Conviene comparar regímenes; en vez de inventar un "escenario El Niño",
    se usan los Δ% que ya calcula el tablero sobre 46 meses de serie (aportes,
    embalses y precio ponderado por fase). Con `impacto` vacío se devuelve el caso
    neutro y la figura lo dice.
    """
    out = dict(fase=str(cfg.get("fase", "ventana")), demanda=1.0, solar=1.0, eolica=1.0,
               nivel=1.0, texto="régimen real de la ventana (sin re-escalar)", disponible=False)
    imp = ctx.get("impacto")
    if not isinstance(imp, pd.DataFrame) or imp.empty or "variable" not in imp.columns:
        return out

    def _med(var: str, col: str) -> float:
        f = imp[imp["variable"].astype(str).str.contains(var, case=False, na=False)]
        if f.empty or col not in f.columns:
            return float("nan")
        try:
            return float(f.iloc[0][col])
        except Exception:                                       # noqa: BLE001
            return float("nan")

    fase = str(cfg.get("fase", "ventana"))
    if fase not in ("El Niño", "La Niña"):
        return out
    col = "media_El_Nino" if fase == "El Niño" else "media_La_Nina"
    base = _med("Precio", "media_Neutral")
    pre = _med("Precio", col)
    apo_neu, apo = _med("Aportes", "media_Neutral"), _med("Aportes", col)
    emb_neu, emb = _med("Embalses", "media_Neutral"), _med("Embalses", col)
    if not np.isfinite(pre) or not np.isfinite(base) or base <= 0:
        return out
    out.update(
        disponible=True, nivel=float(np.clip(pre / base, 0.6, 2.2)),
        demanda=float(np.clip(1.0 + 0.25 * (1.0 - (apo / apo_neu if np.isfinite(apo) and apo_neu > 0 else 1.0)), 0.95, 1.10)),
        solar=float(np.clip(1.0 - 0.35 * (1.0 - (emb / emb_neu if np.isfinite(emb) and emb_neu > 0 else 1.0)), 0.80, 1.05)),
        eolica=float(np.clip(1.0 - 0.20 * (1.0 - (apo / apo_neu if np.isfinite(apo) and apo_neu > 0 else 1.0)), 0.85, 1.05)),
        texto=(f"{fase}: precio ponderado {_pct(pre, base)} vs neutral (Δ% medido sobre 46 meses en "
               f"`impacto`), aportes {apo:,.1f} vs {apo_neu:,.1f} % y embalses {emb:,.1f} vs "
               f"{emb_neu:,.1f} %; la demanda se eleva un 25 % del déficit de aportes y el recurso "
               f"se recorta con la elasticidad hidrológica declarada."))
    return out


def _pct(a: float, b: float) -> str:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return "n/d"
    return f"{100.0 * (a / b - 1.0):+.1f} %"


# nombre del preset → (qué fija, justificación del escenario, objetivo que cubre)
NODAL_PRESETS: dict[str, dict[str, Any]] = {
    "personalizado": dict(
        cfg={},
        porque=("Ninguno: la pestaña usa exactamente lo que usted ponga en los controles. "
                "Todos los demás casos son subconjuntos de esa misma máquina de modelos, así "
                "que son comparables entre sí."),
        obj="—"),
    "base": dict(
        cfg=dict(granularidad="media", estres=1.00, markup_pct=0.0, wh_pct=0.0, flex=0.0,
                 bateria_pct=0.0, bombeo_pct=0.0, solar_pct=0.0, eolica_pct=0.0, kappa=0.5,
                 fase="ventana"),
        porque=("Caso de referencia honesto: topología `media` (10 nodos), líneas a su "
                "capacidad nominal declarada, oferta calibrada al precio de bolsa de la ventana "
                "y demanda rígida. Sirve para documentar el hallazgo negativo del módulo: con "
                "esta malla NO hay congestión, luego zonal ≈ nodal y la señal locacional vale "
                "cero. Todo lo demás se lee como desviación contra esta línea de base."),
        obj="a (comparativo de esquemas)"),
    "estres-moderado": dict(
        cfg=dict(granularidad="media", estres=0.70, markup_pct=0.0, wh_pct=0.0, flex=15.0,
                 bateria_pct=0.0, bombeo_pct=0.0, kappa=0.5, fase="ventana"),
        porque=("Líneas al 70 % de la capacidad nominal: es el estrés mínimo con el que aparecen "
                "horas congestionadas en la topología `media` (el objetivo b pide 'múltiples "
                "configuraciones de red'). Con 15 % de demanda interrumpible se activa además el "
                "mecanismo de desconexión voluntaria del objetivo a."),
        obj="a y b"),
    "estres-severo": dict(
        cfg=dict(granularidad="media", estres=0.40, flex=25.0, markup_pct=0.0, wh_pct=0.0,
                 bateria_pct=0.0, bombeo_pct=0.0, kappa=0.5, fase="ventana"),
        porque=("40 % de la capacidad: corredores al límite, dispersión de precios del orden de "
                "1 000 COP/kWh y violación residual (el ascenso dual avisa que no converge). Es "
                "el escenario donde la liquidación uninodal más distorsiona: el spread medio es "
                "la renta que hoy no se asigna a nadie."),
        obj="a y b"),
    "renovables-2030": dict(
        cfg=dict(granularidad="media", estres=0.70, solar_pct=100.0, eolica_pct=100.0, flex=15.0,
                 bateria_pct=6.0, bombeo_pct=0.0, markup_pct=0.0, wh_pct=0.0, kappa=0.5,
                 fase="ventana"),
        porque=("Duplica la CEN solar y eólica de cada nodo (≈ 2,4 GW solar y ~0,2 GW eólico "
                "reales en la ventana → 5 GW y 0,4 GW), que es el orden de la expansión "
                "renovable del Plan Energético Nacional hacia 2030 sin cambiar la red. El "
                "escenario existe para ver el efecto rebote: más variable, más recorte al "
                "mediodía y más congestión en la rampa de la tarde."),
        obj="a, b y d"),
    "almacenamiento": dict(
        cfg=dict(granularidad="media", estres=0.70, solar_pct=50.0, eolica_pct=25.0, flex=15.0,
                 bateria_pct=6.0, bombeo_pct=10.0, ems_modo="arbitraje", alm_nodos=["CEN", "BOG"],
                 markup_pct=0.0, wh_pct=0.0, kappa=0.5, fase="ventana"),
        porque=("6 % de la CEN en baterías de 4 h en CEN+BOG (los nodos de carga del altiplano) y "
                "10 % en hidrobombeo de 12 h, con η = 0,88 / 0,80 (valores de catálogo, no de "
                "oferta). El dimensionamiento 4 h responde al ancho del pico 18-21 h medido en la "
                "figura de calibración; el bombeo se pone largo porque su rol es desplazarse "
                "entre días, no dentro del día."),
        obj="b y e"),
    "siting-errado": dict(
        cfg=dict(granularidad="media", estres=0.70, solar_pct=50.0, eolica_pct=25.0, flex=15.0,
                 bateria_pct=6.0, bombeo_pct=10.0, ems_modo="alivio_congestion",
                 alm_nodos=["GUJ", "MAG"], markup_pct=0.0, wh_pct=0.0, kappa=0.5, fase="ventana"),
        porque=("El mismo activo, pero sentado en los nodos de recurso (Guajira/Magnesia) con el "
                "objetivo de aliviar congestión. Es el escenario que *debe* salir mal: reproduce "
                "el hallazgo del módulo de que el almacenamiento mal ubicado no captura la renta "
                "locacional y pierde dinero. Sin este caso, la figura del EMS sólo contaría "
                "historias buenas."),
        obj="b y e"),
    "poder-mercado": dict(
        cfg=dict(granularidad="media", estres=0.70, flex=15.0, markup_pct=20.0, wh_pct=25.0,
                 wh_modo="energia", bateria_pct=0.0, bombeo_pct=0.0, kappa=0.5, fase="ventana"),
        porque=("Markup del 20 % sobre la oferta declarada de los bloques del nodo más cargado y "
                "retiro de 25 % de la capacidad *energética*. El 20 % no es un número de libro: "
                "es el orden del recargo con el que la SSPD abrió investigaciones por poder de "
                "mercado (expedientes EPM 14/15-mar-2022 y Emgesa 11/14/15-mar-2022 citados en "
                "el mercado colombiano, donde la oferta declarada superó el costo de oportunidad en esas "
                "magnitudes). El retiro se pone en modalidad `energia` porque en modo "
                "`capacidad` no muerde si el bloque no está en su techo — y eso también se "
                "muestra en la pestaña."),
        obj="c"),
    "desconexion": dict(
        cfg=dict(granularidad="media", estres=0.70, flex=30.0, gatillo=1000.0, markup_pct=0.0,
                 wh_pct=0.0, bateria_pct=0.0, bombeo_pct=0.0, kappa=0.5, fase="ventana"),
        porque=("30 % de la demanda declarada como interrumpible con gatillo en 1 000 COP/kWh "
                "(apenas por encima del precio medio de la ventana). Es el escenario explícito "
                "del objetivo a: con demanda desconectable el precio deja de ser el precio de "
                "escasez y pasa a quedar clavado en el valor de la última hora cortada, así que "
                "la dispersión nodal se aplana y la renta de congestión se reparte distinto."),
        obj="a"),
    "el-nino": dict(
        cfg=dict(granularidad="media", estres=0.60, flex=15.0, bateria_pct=0.0, bombeo_pct=0.0,
                 markup_pct=0.0, wh_pct=0.0, kappa=0.5, fase="El Niño"),
        porque=("Nivel de oferta re-escalado con el Δ% de precio real que mide la pestaña 🌊 El "
                "Niño entre fase El Niño y neutral (+61,3 % sobre 46 meses: 259,9 → 419,3 "
                "COP/kWh en `impacto`), con aportes −27 % y embalses −6,3 % traducidos a +25 % "
                "de demanda por déficit de aportes y −35 % de recurso solar por sequía de "
                "interior, y la red 40 % más ajustada porque en seco el corredor norte→centro "
                "trabaja al máximo. Cada factor viene de un dato del tablero, no de la "
                "literatura."),
        obj="a, b y c"),
    "la-nina": dict(
        cfg=dict(granularidad="media", estres=0.85, flex=10.0, bateria_pct=0.0, bombeo_pct=0.0,
                 markup_pct=0.0, wh_pct=0.0, kappa=0.5, fase="La Niña"),
        porque=("Espejo húmedo del anterior con los mismos multiplicadores medidos (precio "
                "197,1 COP/kWh en La Niña vs 259,9 neutral ⇒ nivel ×0,76), más holgura de red "
                "(0,85) porque con los embalses llenos el despacho hidro cubre localmente y "
                "transporta menos. Sirve para mostrar que la señal nodal *no* es un artificio de "
                "los meses secos: cambia de tamaño, no de signo."),
        obj="a y b"),
    "hibrida-transicion": dict(
        cfg=dict(granularidad="media", estres=0.70, flex=15.0, kappa=0.35, markup_pct=0.0,
                 wh_pct=0.0, bateria_pct=3.0, bombeo_pct=0.0, fase="ventana",
                 ems_modo="arbitraje", regla="hibrido"),
        porque=("Transición gradual en vez del salto al LMP pleno: liquidación "
                "híbrida con κ = 0,35 (35 % de la señal locacional en la factura), batería pequeña "
                "en el nodo de carga y red moderadamente ajustada. κ = 0,35 no es arbitrario: es "
                "el punto donde la dispersión recuperada por el mercado cubre alrededor de un "
                "tercio de la renta de congestión, que es el argumento de diseño de CREG 143/2021 "
                "para no llevar la señal al 100 % de golpe."),
        obj="a, b y c"),
}

# registro estático para la tabla de la pestaña (justificación visible de cada caso)
NODAL_ESCENARIOS_ORDEN = list(NODAL_PRESETS.keys())


def barrido_flexibilidad(red: dict, perfiles: dict[str, np.ndarray], nivel_cop_kwh, *,
                         fracs: tuple[float, ...] = (0.0, 15.0, 30.0),
                         gatillos: tuple[float, ...] | None = None,
                         kappa: float = 0.5, perdas_iter: int = 3,
                         markup: dict[str, Any] | None = None,
                         withholding: dict[str, Any] | None = None) -> pd.DataFrame:
    """Malla (flexibilidad × gatillo) del escenario de desconexión voluntaria (objetivo a).

    Cada celda resuelve el OPF completo: el *corte* es la demanda que se desconecta cuando el
    precio del nodo supera el gatillo, y el *recorte* es la variable que no entra. La lectura
    que interesa es que el precio medio **cae** más rápido que el bienestar: la desconexión
    voluntaria compra estabilidad con energía no servida, y bajo precio nodal ese intercambio
    se concentra en los nodos del final del corredor, no en todo el sistema.
    """
    dem, sol, eol = perfiles_nodales(red, perfiles)
    if not gatillos:
        # Sin gatillo fijo se usan los percentiles de la distribución de precios del propio
        # escenario: tiene más sentido económico "cortar cuando el precio entra en el decil
        # alto" que un umbral absoluto que sólo funciona en régimen de escasez.
        base = opf_nodal(red, demanda=dem, solar=sol, eolica=eol, nivel_cop_kwh=nivel_cop_kwh,
                         kappa_hibrido=kappa, perdas_iter=int(perdas_iter), markup=markup,
                         withholding=withholding)
        pd_ = np.asarray(base["p_dem"], dtype=float)
        gatillos = tuple(float(np.percentile(pd_, q)) for q in (40.0, 75.0, 95.0))
    filas = []
    for ff in fracs:
        for gg in gatillos:
            r = opf_nodal(red, demanda=dem, solar=sol, eolica=eol, nivel_cop_kwh=nivel_cop_kwh,
                          kappa_hibrido=kappa, flex_frac=float(ff) / 100.0,
                          flex_gatillo_cop=float(gg), perdas_iter=int(perdas_iter),
                          markup=markup, withholding=withholding)
            corte = np.asarray(r["corte"], dtype=float)
            filas.append(dict(
                flex_pct=float(ff), gatillo_COP_kWh=float(gg),
                lam=float(np.mean(r["lam"])), dispersion=float(np.mean(r["dispersion"])),
                precio_max=float(np.max(r["p_dem"])), precio_min=float(np.min(r["p_dem"])),
                corte_GWh=float(corte.sum() / 1000.0),
                recorte_GWh=float(np.asarray(r["recorte_MWh"], dtype=float).sum() / 1000.0),
                horas_congestion=int(r["horas_congestion"]),
                viol_MW=float(r["max_violacion"]),
                rc_MCP_h=float(np.mean(r["renta_congestion"]) / 1e6),
                costo_MCop_h=float(np.mean(r["costo"]) / 1e6),   # `costo` ya viene en COP/h
                co2_t=float(np.sum(r["co2_t"])), conv=bool(r["converged"])))
    return pd.DataFrame(filas)



def _modelo_nodal(ctx: dict, cfg: dict) -> dict[str, Any]:
    """Una sola vez por configuración: red → calibración → OPF → módulos opcionales.

    Devuelve todo crudo; las figuras sólo formatean. Se memoiza con `figura_cacheada`
    usando la propia `cfg` como clave, así que mover un selector recalcula y ya.
    """
    t0 = time.time()
    cfg = {**_NODAL_DEFECTOS, **{k: v for k, v in (cfg or {}).items() if v is not None}}
    red = construye_red(ctx, granularidad=cfg["granularidad"], escalon_lineas=cfg["estres"],
                        solar_pct=cfg["solar_pct"], eolica_pct=cfg["eolica_pct"],
                        curvatura=cfg["curvatura"])
    pf = perfiles_reales(ctx)
    cal = calibra_oferta(red, pf, use_opf=bool(cfg["opf_calib"]), por_hora=True)
    dem, sol, eol = perfiles_nodales(red, pf)
    # El escenario de fase (si viene de un preset) re-escala recurso y nivel con los Δ% que mide
    # la pestaña 🌊 El Niño. La calibración sigue anclada al dato real y el factor se reporta en
    # `mdl["fase"]`, para que ninguna cifra del escenario se lea sin su origen.
    fase = _fase_enso(ctx, cfg)
    nivel_modelo = np.asarray(cal["nivel_h"], dtype=float)
    if fase.get("disponible"):
        dem = dem * float(fase["demanda"])
        sol = sol * float(fase["solar"])
        eol = eol * float(fase["eolica"])
        nivel_modelo = nivel_modelo * float(fase["nivel"])
    # A quién se le aplica la estrategia: el OPF necesita un filtro de bloques (nodo, tecnología
    # o prefijo), porque un markup "a todos los bloques" no es poder de mercado sino una
    # re-escalada de la curva. `auto` = el nodo con mayor peso de demanda, que en esta red es el
    # que fija λ; si el nodo elegido no existe en la topología cargada, se vuelve a `auto`.
    _w = np.asarray(red["peso_demanda"], dtype=float)
    _nodos_red = [str(z) for z in list(red["nodos"])]
    nodo_estr = str(cfg.get("nodo_estr") or "auto").strip()
    if nodo_estr.upper() not in ("TODOS", "ALL") and nodo_estr not in _nodos_red:
        nodo_estr = str(_nodos_red[int(np.argmax(_w))]) if _nodos_red and _w.size else ""
    tec_todas = sorted(set(str(t) for t in (red.get("tec_cols") or [])))
    if nodo_estr.upper() in ("TODOS", "ALL"):
        sel_estr = dict(tecnologia=tec_todas)
    else:
        sel_estr = dict(nodo=nodo_estr) if nodo_estr else dict(tecnologia=tec_todas)
    mk = dict(pct=cfg["markup_pct"] / 100.0, **sel_estr) if cfg["markup_pct"] > 0 else None
    wh = (dict(frac=cfg["wh_pct"] / 100.0, modo=cfg["wh_modo"], **sel_estr)
          if cfg["wh_pct"] > 0 else None)
    alm = None
    if cfg.get("bateria_pct", 0) or cfg.get("bombeo_pct", 0):
        validos = [z for z in cfg["alm_nodos"] if z in list(red["nodos"])] or None
        alm = config_almacenamiento(red, bateria_pct=cfg["bateria_pct"] / 100.0,
                                    bombeo_pct=cfg["bombeo_pct"] / 100.0,
                                    dur_bateria_h=cfg["dur_bateria"], dur_bombeo_h=cfg["dur_bombeo"],
                                    eta_bateria=cfg["eta_bateria"], eta_bombeo=cfg["eta_bombeo"],
                                    nodos=validos)
    res = opf_nodal(red, demanda=dem, solar=sol, eolica=eol, nivel_cop_kwh=nivel_modelo,
                    kappa_hibrido=cfg["kappa"], flex_frac=cfg["flex"] / 100.0,
                    flex_gatillo_cop=cfg["gatillo"], perdas_iter=cfg["perdas_iter"],
                    markup=mk, withholding=wh, almacenamiento=alm)
    res_base = (opf_nodal(red, demanda=dem, solar=sol, eolica=eol,
                          nivel_cop_kwh=nivel_modelo, kappa_hibrido=cfg["kappa"],
                          flex_frac=cfg["flex"] / 100.0,
                          flex_gatillo_cop=cfg["gatillo"], perdas_iter=cfg["perdas_iter"],
                          almacenamiento=alm)
                if (mk or wh) else res)
    aplicado = dict(nodo_estr=str(sel_estr.get("nodo") or "todas las tecnologías de la red"),
                    markup_pct=float((mk or {}).get("pct", 0.0)) * 100.0,
                    wh_frac=float((wh or {}).get("frac", 0.0)) * 100.0,
                    wh_modo=str(cfg["wh_modo"]),
                    movio_lambda=float(np.mean(res["lam"]) - np.mean(res_base["lam"])))
    mdl = dict(red=red, pf=pf, cal=cal, fase=fase, nivel_modelo=nivel_modelo, estrato=aplicado,
               perfiles_nodales=(dem, sol, eol), res=res, res_base=res_base,
               cfg=dict(cfg), t_seg=time.time() - t0, almacenamiento=alm,
               vr=dict(disponible=False))
    if cfg.get("refuerzos", True):
        try:
            mdl["ref"] = evalua_refuerzos(cfg, mdl)
        except Exception as exc:                                   # noqa: BLE001
            mdl["ref"] = pd.DataFrame()
            mdl["ref_nota"] = f"{type(exc).__name__}: {str(exc)[:150]}"
    else:
        mdl["ref"] = pd.DataFrame()
    if cfg["valida_lmp"]:
        try:
            mdl["vr"] = valida_lmp(red, res, max_horas=min(24, cfg["horas_valida"]))
        except Exception as exc:                                   # noqa: BLE001
            mdl["vr"] = dict(disponible=False, nota=f"{type(exc).__name__}: {str(exc)[:120]}")
    if cfg["escenarios"]:
        try:
            esc = gen_escenarios(red, pf, n_esc=int(cfg["n_esc"]), semilla=int(cfg["semilla"]),
                                 fam_sol=cfg["fam_sol"], fam_eol=cfg["fam_eol"],
                                 rho_recurso=cfg["rho_recurso"], rho_cruce=cfg["rho_cruce"],
                                 factor_fase=(dict(demanda=float(fase["demanda"]),
                                                   solar=float(fase["solar"]),
                                                   eolica=float(fase["eolica"]))
                                              if fase.get("disponible") else None))
            _red_esc = reduce_escenarios(esc, k=int(cfg["n_tipicos"]), semilla=int(cfg["semilla"]))
            mdl["esc_cv"] = pd.DataFrame()
            if cfg.get("emparejar_cv", True):
                try:
                    _red_esc, _aj = emparejar_cv(_red_esc, pf, tope=12.0)
                    mdl["esc_cv"] = _aj
                except Exception:                                  # noqa: BLE001
                    pass
            mdl["esc"] = _red_esc
            mdl["esc_n"] = esc["S"]
            try:
                mdl["esc_mom"] = momentos_real_vs_sim(red, pf, _red_esc)
            except Exception:                                      # noqa: BLE001
                mdl["esc_mom"] = pd.DataFrame()
            if cfg.get("two_stage", True) and isinstance(_red_esc, dict):
                try:
                    mdl["est"] = despacho_estocastico(
                        red, res, _red_esc, nivel_modelo, kappa=float(cfg["kappa"]),
                        perdas_iter=int(cfg["perdas_iter"]), flex_frac=float(cfg["flex"]) / 100.0,
                        gatillo_cop=float(cfg.get("gatillo") or 0.0), alm=alm)
                except Exception as exc:                           # noqa: BLE001
                    mdl["est"] = None
                    mdl["est_nota"] = f"{type(exc).__name__}: {str(exc)[:150]}"
            else:
                mdl["est"] = None
        except Exception as exc:                                   # noqa: BLE001
            mdl["esc"] = None
            mdl["esc_nota"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    else:
        mdl["esc"] = None
    if cfg["mercado"]:
        try:
            mdl["ior"] = poder_mercado_ior(ctx, red)
            techo = mdl["ior"].get("precio_escasez_medio_COP_kWh")
            mdl["cb"] = cournot_bertrand(red, pf, nivel_modelo, epsilon=cfg["epsilon"],
                                          cuota_top=(mdl["ior"].get("cuota_top") if isinstance(mdl.get("ior"), dict) else None),
                                          p_techo=techo,
                                          horas=[_nodal_h(cfg["hora_foco"]), 12, 17, 19, 20])
        except Exception as exc:                                   # noqa: BLE001
            mdl["ior"] = dict(disponible=False, nota=f"{type(exc).__name__}: {str(exc)[:120]}")
            mdl["cb"] = pd.DataFrame()
    else:
        mdl["ior"], mdl["cb"] = dict(disponible=False), pd.DataFrame()
    if cfg["riesgo"]:
        try:
            ret = retornos_reales(ctx)
            g = garch11(ret)
            # Se reconstruye una trayectoria de precio coherente con los retornos reales a
            # partir del nivel medio de la ventana: el VaR condicional necesita el NIVEL, y
            # `garch11` entrega la volatilidad de los retornos log.
            try:
                base = float(np.nanmedian(ctx["sistema"]["Precio_Bolsa_COP_kWh"].astype(float)))
            except Exception:                                   # noqa: BLE001
                base = float(np.nanmean(np.asarray(pf["precio"], dtype=float))) or 900.0
            precio = base * np.exp(np.concatenate([[0.0], np.cumsum(ret)])) if ret.size else np.array([])
            mdl["garch"] = g
            mdl["ret"] = ret
            mdl["precio_ret"] = precio
            mdl["var"] = var_condicional(g["sigma"], precio, alfa=cfg["alfa"]) if g.get("disponible") else pd.DataFrame()
            mdl["bt"] = backtest_var(ret, g["sigma"], alfa=cfg["alfa"]) if g.get("disponible") else pd.DataFrame()
            rn = retornos_nodales(res, None)
            mdl["cov"] = cobertura_min_var(rn, ret[-len(rn):]) if rn.size > 6 and ret.size >= rn.size else dict(disponible=False)
        except Exception as exc:                                   # noqa: BLE001
            mdl["garch"] = dict(disponible=False, nota=f"{type(exc).__name__}: {str(exc)[:140]}")
            mdl["var"] = mdl["bt"] = pd.DataFrame()
            mdl["cov"] = dict(disponible=False)
    else:
        mdl["garch"], mdl["var"], mdl["bt"], mdl["cov"] = dict(disponible=False), pd.DataFrame(), pd.DataFrame(), dict(disponible=False)
    if cfg["almacenar"] and alm is not None:
        try:
            mdl["ems"] = ems_almacenamiento(red, demanda=dem, solar=sol, eolica=eol,
                                            nivel_cop_kwh=nivel_modelo, config=alm,
                                            modo=cfg["ems_modo"], pasadas=int(cfg["ems_pasadas"]))
        except Exception as exc:                                   # noqa: BLE001
            mdl["ems"] = dict(disponible=False, nota=f"{type(exc).__name__}: {str(exc)[:140]}")
    else:
        mdl["ems"] = dict(disponible=False)
    if cfg["aprendizaje"]:
        try:
            mdl["ql"] = qlearning_despacho(red, _nodal_h(cfg["hora_foco"]), pf,
                                           nivel_cop_kwh=nivel_modelo,
                                           episodes=int(cfg["episodios_rl"]),
                                           n_agentes=int(cfg["n_agentes"]), semilla=int(cfg["semilla"]))
        except Exception as exc:                                   # noqa: BLE001
            mdl["ql"] = dict(disponible=False, nota=f"{type(exc).__name__}: {str(exc)[:140]}")
        try:
            mdl["ap"] = aprendizaje_regla(red, pf, float(np.mean(nivel_modelo)),
                                          horas=list(cfg["ap_horas"]),
                                          episodes=int(cfg["episodios_ap"]), semilla=int(cfg["semilla"]))
        except Exception as exc:                                   # noqa: BLE001
            mdl["ap"] = dict(disponible=False, nota=f"{type(exc).__name__}: {str(exc)[:140]}")
    else:
        mdl["ql"] = mdl["ap"] = dict(disponible=False)
    if cfg.get("barrido_flex", True):
        try:
            _pd_base = np.asarray(res["p_dem"], dtype=float)
            _gt = tuple(float(np.percentile(_pd_base, q)) for q in (40.0, 75.0, 95.0))
            mdl["flex_gatillos"] = _gt
            mdl["flex"] = barrido_flexibilidad(red, pf, nivel_modelo, kappa=cfg["kappa"],
                                               perdas_iter=cfg["perdas_iter"], markup=mk,
                                               withholding=wh, fracs=(0.0, 15.0, 30.0),
                                               gatillos=_gt)
        except Exception as exc:                                   # noqa: BLE001
            mdl["flex"] = pd.DataFrame()
            mdl["flex_nota"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    else:
        mdl["flex"] = pd.DataFrame()
    if cfg.get("sintesis", True):
        try:
            _fx = _fase_enso(ctx, dict(cfg, fase="El Niño"))
            mdl["comb"] = combinaciones_nodales(ctx, cfg, _fx)
        except Exception as exc:                                   # noqa: BLE001
            mdl["comb"] = pd.DataFrame()
            mdl["comb_nota"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    else:
        mdl["comb"] = pd.DataFrame()
    if cfg["sensibilidad"]:
        try:
            mdl["sens"] = sensibilidad_penetracion(
                ctx, niveles=cfg["sens_niveles"], granularidad=cfg["granularidad"],
                escalon_lineas=cfg["estres"], n_esc=int(cfg["n_esc_sens"]), semilla=int(cfg["semilla"]))
        except Exception as exc:                                   # noqa: BLE001
            mdl["sens"] = pd.DataFrame()
            mdl["sens_nota"] = f"{type(exc).__name__}: {str(exc)[:140]}"
    else:
        mdl["sens"] = pd.DataFrame()
    try:
        mdl["kpis"] = kpis_reglas(red, res, pf, cal)
        mdl["desc"] = descomposicion_precios(res)
    except Exception as exc:                                       # noqa: BLE001
        mdl["kpis"] = mdl["desc"] = pd.DataFrame()
        mdl["nota_reglas"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    if not isinstance(mdl.get("vr"), pd.DataFrame):
        mdl["vr"] = dict(disponible=False, nota=str((mdl.get("vr") or {}).get("nota", "no ejecutada")))
    if not isinstance(mdl.get("cb"), pd.DataFrame):
        mdl["cb"] = pd.DataFrame()
    if not isinstance(mdl.get("sens"), pd.DataFrame):
        mdl["sens"] = pd.DataFrame()
    if not isinstance(mdl.get("kpis"), pd.DataFrame):
        mdl["kpis"] = pd.DataFrame()
    if not isinstance(mdl.get("desc"), pd.DataFrame):
        mdl["desc"] = pd.DataFrame()
    return mdl


# ---------------------------------- widgets ---------------------------------

def _nodal_widgets(ctx: dict) -> dict[str, Any]:
    """Controles de la pestaña. Devuelve la configuración *congelada* (clave de caché)."""
    pr_claves = list(NODAL_PRESETS)
    pr_etiq = {
        "personalizado": "🎛️ Personalizado (lo que usted ponga abajo)",
        "base": "1 · Base sin congestión (línea de fondo)",
        "estres-moderado": "2 · Estrés moderado de red (×0,70)",
        "estres-severo": "3 · Estrés severo (×0,40) + 25 % interrumpible",
        "renovables-2030": "4 · Renovables 2030 (solar y eólica ×2)",
        "almacenamiento": "5 · Almacenamiento bien sentado (CEN/BOG)",
        "siting-errado": "6 · Almacenamiento mal sentado (GUJ/MAG)",
        "poder-mercado": "7 · Poder de mercado (markup 20 % + retiro 25 %)",
        "desconexion": "8 · Desconexión voluntaria (30 % al gatillo)",
        "el-nino": "9 · El Niño con los Δ% medidos en XM",
        "la-nina": "10 · La Niña (espejo húmedo)",
        "hibrida-transicion": "11 · Transición híbrida (κ = 0,35)",
    }
    pa, pb = st.columns([1.0, 2.1], gap="medium")
    with pa:
        _lbl = st.selectbox(
            "Escenario predefinido (justificado)", [pr_etiq[k] for k in pr_claves], index=0,
            key="nd_preset",
            help="Cada preset fija un subconjunto de controles con un valor razonado, y la columna "
                 "derecha dice por qué y qué objetivo cubre. Los controles que el preset "
                 "no toca quedan a su criterio.")
    preset = next((k for k in pr_claves if pr_etiq[k] == _lbl), "personalizado")
    _pr = NODAL_PRESETS[preset]
    with pb:
        st.markdown("**Justificación del escenario:** " + str(_pr.get("porqué", "—")))
        st.caption("Objetivo cubierto: " + str(_pr.get("obj", "—")) +
                   (" · fija " + ", ".join("`" + str(k) + "`" for k in sorted(_pr["cfg"]))
                    if _pr["cfg"] else " · no fija ningún control"))
    c1, c2, c3, c4 = st.columns([1.15, 1.05, 0.75, 0.95])
    with c1:
        gran = st.selectbox(
            "Topología de la red", ["media", "subregion", "fina"], index=0, key="nd_gran",
            help="`subregion` = las 7 subregiones que publica XM (un nodo por zona: aquí el precio "
                 "zonal y el nodal coinciden por construcción). `media` desdobla CEN→BOG, NOR→ANT y "
                 "CAR→MAG, que es donde aparece la congestión interna; `fina` añade ANT-OCC y MAG-ORI "
                 "para ver cómo los refuerzos la borran.")
        estres = st.slider(
            "Capacidad de las líneas (× del caso base)", 0.30, 1.30, 0.70, 0.05, key="nd_estres",
            help="Escala la reactancia equivalente: 1,00 = la malla nominal declarada; bajarla "
                 "congestiona el sistema y es la palanca del objetivo b (¿cuánto vale el precio "
                 "nodal cuando la red aprieta?). No son parámetros oficiales de ISA/XM.")
    with c2:
        regla = st.selectbox(
            "Regla de liquidación mostrada", ["nodal", "hibrido", "zonal", "uninodal"], index=0, key="nd_regla",
            help="Uninodal = un solo precio para todo el SIN (como hoy). Zonal = un precio por "
                 "subregión. Híbrido = el precio de bolsa + κ·(componente nodal), la transición que "
                 "discute la literatura de mercados. Nodal = LMP pleno.")
        kappa = st.slider("κ del régimen híbrido", 0.0, 1.0, 0.5, 0.05, key="nd_kappa",
                          help="Peso de la componente nodal en la liquidación híbrida. κ=0 ⇒ uninodal, "
                               "κ=1 ⇒ nodal. Es el knob con el que se lee el compromiso eficiencia / "
                               "señales locacionales.")
    with c3:
        hora = st.number_input("Hora foco (1-24)", 1, 24, 20, 1, key="nd_hora",
                               help="Hora del día para los cortes de barra, la descomposición de "
                                    "precios y el subproblema de refuerzo.")
        flex = st.slider("Flexibilidad (recorte) máx. %", 0, 40, 15, 1, key="nd_flex",
                         help="Fracción de la demanda que puede cortarse (interrumpible/piso de "
                              "despacho) cuando el precio supera el gatillo. 0 % = demanda rígida, "
                              "el caso regulatorio.")
    with c4:
        gatillo = st.slider("Gatillo de recorte (COP/kWh)", 600, 4000, 1200, 100, key="nd_gatillo",
                            help="Precio del nodo por encima del cual empieza a cortarse la demanda "
                                 "interrumpible: la respuesta crece en línea recta desde el gatillo "
                                 "hasta un 25 % por encima, donde el contrato llega a su tope. Se "
                                 "ancla al precio de escasez observado en la ventana; la malla de "
                                 "desconexión voluntaria barre además p40/p75/p95 del propio escenario.")
        semilla = st.number_input("Semilla", 1, 999, 7, 1, key="nd_sem",
                                  help="Fija escenarios, Q-learning y el ruido de las familias "
                                       "distribucionales; cámbiela para ver la varianza de Monte Carlo.")
    with st.expander("⚙️ Escenarios estocásticos, almacenamiento, estrategia y riesgo", expanded=False):
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            n_esc = st.slider("Escenarios Monte Carlo", 24, 240, 80, 8, key="nd_nesc",
                              help="Trayectorias horarias de recurso y demanda. Se "
                                   "reducen a los típicos por k-medias antes de usarlas.")
            n_tip = st.slider("Escenarios típicos (k-medias)", 4, 20, 8, 1, key="nd_ntip",
                              help="Reducción clásica de la programación estocástica: conservan la "
                                   "media y aproximano la varianza a una fracción del costo.")
            twostage = st.toggle("Despachar cada escenario típico con su propio OPF (2ª etapa)",
                                 value=True, key="nd_2st",
                                 help="Apagado, los escenarios solo describen recurso y demanda "
                                      "(rápido, y el precio se lee del día medio). Encendido, cada día "
                                      "típico pasa por el mismo OPF con pérdidas y congestión: salen "
                                      "bandas p05-p95 del precio, probabilidad de escasez por hora, "
                                      "distribución del LMP por nodo y el VSS. Cuesta ~0,3 s por escenario.")
            fam_sol = st.selectbox("Familia solar", ["lognormal", "gamma", "weibull", "uniforme"], key="nd_famsol",
                                   help="Log-normal con τ_b = a(1−e^(−ε·cosθz)) + a·e^(−k/cosθz) "
                                        "(Perez 2023); aquí se deja elegir la familia para el KS.")
            fam_eol = st.selectbox("Familia eólica", ["weibull", "rayleigh", "uniforme", "lognormal"], key="nd_fameol",
                                   help="Weibull f(x)=aβx^(β−1)e^(−ax^β); Rayleigh es el caso β=2 y la "
                                        "mezcla uniforme es la alternativa de Acero (2024).")
            rho = st.slider("ρ entre nodos (recurso)", 0.0, 0.98, 0.6, 0.02, key="nd_rho",
                            help="Correlación del factor de capacidad entre subregiones: 1 = un solo "
                                 "cielo para todo el país, 0 = independientes.")
        with d2:
            rhoc = st.slider("ρ sol↔viento", -0.9, 0.9, -0.15, 0.05, key="nd_rhoc",
                             help="En Colombia es negativa en el Caribe (brisas vs nubosidad de "
                                  "interior): es la que hace que la mezcla renueve sea menos variable.")
            bateria = st.slider("Batería (% de la CEN)", 0.0, 15.0, 6.0, 0.5, key="nd_bat",
                                help="Potencia instalada como fracción de la CEN del nodo. Duración y "
                                     "rendimiento se fijan abajo.")
            dur_bat = st.slider("Duración de la batería (h)", 1.0, 8.0, 4.0, 0.5, key="nd_durbat")
            eta_bat = st.slider("Rendimiento batería (η)", 0.6, 0.98, 0.88, 0.02, key="nd_etabat")
            bombeo = st.slider("Bombeo (% de la CEN)", 0.0, 25.0, 0.0, 0.5, key="nd_bom",
                               help="Hidrobombeo reversible (Sogamoso/Porceión son el referente "
                                    "nacional); con 0 % queda desactivado.")
        with d3:
            dur_bom = st.slider("Duración del bombeo (h)", 4.0, 20.0, 12.0, 1.0, key="nd_durbom")
            eta_bom = st.slider("Rendimiento bombeo (η)", 0.6, 0.95, 0.80, 0.01, key="nd_etabom")
            ems_modo = st.selectbox("Objetivo del EMS", ["arbitraje", "anti-recorte", "alivio_congestion"],
                                    key="nd_emsm",
                                    help="`arbitraje` maximiza Σ(p_descarga−p_carga/η)·E; "
                                         "`anti-recorte` carga sólo con excedente renovable; "
                                         "`alivio_congestion` usa las máscaras de nodo para descargar "
                                         "donde la rama está apretada.")
            ems_pas = st.slider("Pasadas del EMS", 1, 10, 5, 1, key="nd_emsp",
                                help="El EMS y el OPF se resuelven en bloque: se iteran hasta que el "
                                     "cambio de despacho estableza (< 1 MW). Más pasadas = más cerca "
                                     "del óptimo conjunto, más lento.")
        with d4:
            markup = st.slider("Markup sobre la oferta (%)", 0, 40, 0, 1, key="nd_mk",
                               help=": los agentes con poder suben la oferta por encima del costo "
                                    "marginal. Como el modelo es de precio, un markup en un nodo "
                                    "congestonado no se propaga igual que bajo liquidación uninodal: "
                                    "esa es la prueba del objetivo c.")
            wh_pct = st.slider("Retiro de capacidad (%)", 0, 60, 0, 5, key="nd_wh",
                               help="Withholding: retirar MW del mercado (físico o virtual) para "
                                    "forzar el precio.")
            wh_modo = st.selectbox("Modalidad del retiro", ["energia", "capacidad"], key="nd_whm",
                                   help="`energia` recorta el tope de despacho horario (funciona en "
                                        "cualquier régimen); `capacidad` baja la Pmax instalada, que "
                                        "sólo muerde si el bloque está en su techo — el tablero muestra "
                                        "los dos para que se vea la diferencia.")
            _nodos_pos = ["auto", "TODOS"] + [str(z) for z in (list(ZONAS_NODAL) + list(ZONAS_EXTRA_NODAL))]
            nodo_estr = st.selectbox(
                "Nodo-agente que ejerce la estrategia", _nodos_pos, index=0, key="nd_nodo_estr",
                help="Sobre qué oferta se aplica el markup y el retiro. `auto` = el nodo con mayor "
                     "peso de demanda de la topología cargada (allí se forma el precio marginal); "
                     "`TODOS` infla toda la curva de oferta, que es el caso límite sin poder "
                     "locacional. Un nodo que no exista en la topología elegida cae de nuevo a `auto`.")
            epsilon = st.slider("Elasticidad-precio de la demanda ε", -2.0, -0.05, -0.45, 0.05, key="nd_eps",
                                help="Para el índice de Lerner del modo Cournot (L = s/|ε|). Con |ε|<1 "
                                     "el recargo teórico no acota: se limita por el precio de escasez.")
    with st.expander("🧪 Capas de cálculo (encienda lo que necesite; cada una cuesta tiempo)", expanded=False):
        e1, e2, e3 = st.columns(3)
        with e1:
            opf_calib = st.toggle("Calibrar la oferta con el OPF completo", value=True, key="nd_opfc",
                                  help="Dos etapas: bisección uninodal para κ y luego ajuste del nivel "
                                       "con el OPF DC. Si se apaga, la calibración es analítica "
                                       "(más rápida, error algo mayor).")
            valida = st.toggle("Validar el LMP por diferencias finitas", value=True, key="nd_val",
                              help=" dice que λ_i = ∂Costo/∂D_i; se comprueba perturbando la "
                                   "demanda del nodo ±2 MW y recalcando el OPF. Es la prueba dura de que "
                                   "el modelo hace lo que dice.")
            horas_valida = st.slider("Horas a validar", 4, 24, 12, 2, key="nd_valh",
                                     help="Cada hora cuesta 2 OPF completos.")
            _ref_tog = st.toggle("Evaluar el conjunto de refuerzos hipotéticos", value=True,
                                 key="nd_ref",
                                 help="Corre un OPF por candidato (seis tramos + la canasta completa, "
                                      "≈2 s). Aporta el Δ de congestión, λ, corte y CO₂ de cada obra "
                                      "sobre el mismo día tipo.")
        with e2:
            riesgo = st.toggle("GARCH + VaR + cobertura", value=True, key="nd_ries",
                               help=" sobre la serie horaria real de la ventana (648 h) "
                                    "y la del modelo. Es rápido (~0,5 s).")
            alfa = st.select_slider("α del VaR", [0.01, 0.025, 0.05, 0.10], value=0.05, key="nd_alfa",
                                    help="Nivel de confianza del VaR condicional y del backtest de Kupiec.")
            mercado = st.toggle("Poder de mercado (IOR, HHI, Cournot)", value=True, key="nd_merc",
                                help=" con los datos reales de concentración por agente y el "
                                     "markup sobre la curva de oferta.")
            sens = st.toggle("Barrido de penetración renovable", value=True, key="nd_sens",
                             help="Recalcula red + OPF por cada nivel de penetración: es la figura más "
                                  "cara del módulo (≈6 OPF con escenarios).")
            n_esc_sens = st.slider("Escenarios por nivel del barrido", 4, 40, 8, 2, key="nd_sensesc")
        with e3:
            aprend = st.toggle("Aprendizaje (Q-learning + reglas)", value=True, key="nd_ap",
                               help=". El refuerzo por regla corre un OPF por episodio, así que "
                                    "se limita a 24 h y pocas épocas.")
            eps_rl = st.slider("Episodios de Q-learning", 500, 20000, 6000, 500, key="nd_epsrl",
                               help="El precio convergen a una banda cuya anchura fija la resolución "
                                    "de la rejilla de acciones; 6 000 episodios toman ≈0,5 s.")
            eps_ap = st.slider("Episodios por regla", 5, 60, 16, 1, key="nd_epsap",
                              help="Cada episodio resuelve un OPF nodal: 16 episodios × 2 reglas ≈ 2 s.")
            n_ag = st.slider("Agentes del subproblema", 3, 12, 6, 1, key="nd_nag",
                            help="Bloques despachables más grandes que participan en el juego.")
    with st.expander("🕒 Penetración renovable del caso base y nodos del EMS", expanded=False):
        p1, p2, p3 = st.columns(3)
        with p1:
            solar_pct = st.slider("Solar adicional sobre la CEN (%)", 0, 200, 0, 5, key="nd_solp",
                                  help="Aumenta la capacidad SOLAR de todos los nodos antes de calibrar. "
                                       "Es el '¿qué pasa con más renovables?' del objetivo a/d.")
            eolica_pct = st.slider("Eólica adicional (%)", 0, 200, 0, 5, key="nd_eolp")
        with p2:
            perdas = st.slider("Iteraciones de pérdidas", 0, 6, 3, 1, key="nd_perdas",
                               help="Las pérdidas se linealizan con `LF_i` y se re-evalúan: 0 = red "
                                    "pérdida-cero (más rápido, sesga λ a la baja).")
            curv = st.slider("Curvatura de la oferta (c₂)", 0.05, 1.50, 0.45, 0.05, key="nd_curv",
                             help="c₂ = curvatura·c₁/Pmax. Controla cuán 'dura' es la rampa de oferta: "
                                  "a más curvatura, menor dispersión de precios y menos recorte.")
        with p3:
            st.caption("**Nodos donde se instalan el almacenamiento y la nueva oferta** "
                       "(vacío = todos los nodos de la topología).")
            alm_nodos = st.multiselect("Nodos del EMS", list(ZONAS_NODAL) + list(ZONAS_EXTRA_NODAL),
                                       default=["CEN", "BOG"], key="nd_nodos",
                                       help="El *siting* es parte del resultado: mal ubicado, el "
                                            "almacenamiento pierde dinero incluso con precio nodal "
                                            "(se reporta explícitamente).")
            ap_h = st.multiselect("Horas del juego por regla", list(range(1, 25)), default=[20, 21],
                                  key="nd_aph", help="Horas en las que los agentes aprenden su markup "
                                                     "(pico = 19-21 en la curva real de XM).")
    fase_sel = st.toggle("Re-escalar el escenario con los Δ% medidos de la fase ENSO "
                         "(nivel de oferta y recurso)", value=False, key="nd_fase",
                         help="Off: se usa el régimen real de la ventana cargada, tal como lo "
                              "publicó XM. On: el nivel de oferta se multiplica por la razón "
                              "precio-de-la-fase / precio-neutral que calcula la pestaña 🌊 El Niño "
                              "sobre 46 meses, y el recurso se ajusta con los déficits de aportes y "
                              "embalses. No es un factor de literatura: sale de `ctx['impacto']`.")
    cfg = dict(fase=("El Niño" if fase_sel else "ventana"), nodo_estr=str(nodo_estr),
               granularidad=str(gran),
               estres=float(estres), regla=str(regla), kappa=float(kappa),
               hora_foco=int(hora), flex=float(flex), gatillo=float(gatillo), semilla=int(semilla),
               n_esc=int(n_esc), n_tipicos=int(n_tip), two_stage=bool(twostage),
               fam_sol=str(fam_sol), fam_eol=str(fam_eol),
               rho_recurso=float(rho), rho_cruce=float(rhoc), bateria_pct=float(bateria),
               dur_bateria=float(dur_bat), eta_bateria=float(eta_bat), bombeo_pct=float(bombeo),
               dur_bombeo=float(dur_bom), eta_bombeo=float(eta_bom), ems_modo=str(ems_modo),
               ems_pasadas=int(ems_pas), markup_pct=float(markup), wh_pct=float(wh_pct),
               wh_modo=str(wh_modo), epsilon=float(epsilon), opf_calib=bool(opf_calib),
               refuerzos=bool(_ref_tog), valida_lmp=bool(valida), horas_valida=int(horas_valida), riesgo=bool(riesgo),
               alfa=float(alfa), mercado=bool(mercado), sensibilidad=bool(sens),
               n_esc_sens=int(n_esc_sens), aprendizaje=bool(aprend), episodios_rl=int(eps_rl),
               episodios_ap=int(eps_ap), n_agentes=int(n_ag), escenas=None,
               escenarios=True, perdas_iter=int(perdas), curvatura=float(curv),
               solar_pct=float(solar_pct), eolica_pct=float(eolica_pct),
               alm_nodos=[str(x) for x in (alm_nodos or [])],
               almacenar=bool(float(bateria or 0.0) > 0.0 or float(bombeo or 0.0) > 0.0),
               ap_horas=[int(x) - 1 for x in (ap_h or [])] or [19],
               sens_niveles=[(0.0, 0.0), (25.0, 10.0), (60.0, 25.0), (100.0, 40.0), (150.0, 60.0)])
    # En entorno sin Streamlit (import para pruebas) los widgets devuelven None: se
    # recuperan los mismos valores por defecto que muestran los controles, de modo que
    # el módulo pueda ejercitarse fuera de la app sin cambiar su comportamiento.
    defectos = dict(_NODAL_DEFECTOS)
    for kk, vv in defectos.items():
        if cfg.get(kk) is None:
            cfg[kk] = vv
    cfg.pop("escenas", None)
    if cfg["alm_nodos"]:
        validos = [z for z in cfg["alm_nodos"]]
        cfg["alm_nodos"] = validos or list(ZONAS_NODAL)
    # El preset manda por encima de los widgets que define (la UI puede mostrar otro valor a
    # propósito; la leyenda de la pestaña lista cuáles están fijados y por qué).
    cfg["preset"] = preset
    cfg["preset_nombre"] = str(_lbl)
    cfg["preset_fija"] = sorted(_pr["cfg"])
    for kk, vv in _pr["cfg"].items():
        cfg[kk] = vv
    if preset != "personalizado":
        cfg["barrido_flex"] = True
    # si el preset toca la potencia de almacenamiento, se recalcula el interruptor derivado
    if "bateria_pct" in _pr["cfg"] or "bombeo_pct" in _pr["cfg"]:
        cfg["almacenar"] = (float(cfg.get("bateria_pct") or 0.0) > 0
                            or float(cfg.get("bombeo_pct") or 0.0) > 0)
    return cfg


def _mdl(ctx: dict, cfg: dict) -> dict[str, Any]:
    """Acceso memoizado al modelo (la caché incluye la configuración, no sólo la ventana)."""
    return figura_cacheada("nodal:modelo", _modelo_nodal, ctx, cfg)


# ================================== figuras ==================================

def fig_nodal_red(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Diagrama unilineal legible: ramas por color de uso, nodos separados y % con fondo."""
    red, res = mdl["red"], mdl["res"]
    nd, ram = red["nodos_df"], red["ramas"]
    if nd.empty or ram.empty:
        return None
    fig = _fig(ctx, 560)
    uso = np.asarray(res["uso_rama"], dtype=float)
    uso_med = np.nanmean(uso, axis=0) if uso.ndim > 1 else np.zeros(len(ram))
    uso_max = np.nanmax(uso, axis=0) if uso.ndim > 1 else uso_med

    # --- posición: se normaliza y se separan los nodos que se tocan (es un esquema, no un mapa) ---
    xs = nd["x"].to_numpy(dtype=float)
    ys = nd["y"].to_numpy(dtype=float)
    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        return (v - lo) / max(hi - lo, 1e-9) if hi > lo else np.full_like(v, 0.5)
    pos = np.column_stack([0.10 + 0.80 * _norm(xs), 0.12 + 0.76 * _norm(ys)])
    rad = 300.0 * np.sqrt(np.clip(nd["CEN_MW"].to_numpy(dtype=float), 0.0, None)
                          / max(float(np.nanmax(nd["CEN_MW"].to_numpy(dtype=float))), 1.0))
    rad = np.clip(rad, 22.0, 54.0)
    sep = 0.145 + 0.0016 * rad                              # radio en unidades de dato (fig ~700 px)
    for _ in range(90):                                     # repulsión: dos pasadas de relaxation
        for a_ in range(len(pos)):
            for b_ in range(a_ + 1, len(pos)):
                d = pos[b_] - pos[a_]
                dd = float(np.hypot(d[0], d[1]))
                if dd < 1e-6:
                    pos[b_] += np.array([0.01, 0.01])
                    continue
                need = sep[a_] + sep[b_]
                if dd < need:
                    paso = 0.5 * (need - dd) / dd
                    pos[a_] -= paso * d
                    pos[b_] += paso * d
    pos[:, 0] = np.clip(pos[:, 0], 0.05, 0.95)
    pos[:, 1] = np.clip(pos[:, 1], 0.06, 0.94)
    coord = {str(nid): (float(pos[k][0]), float(pos[k][1])) for k, nid in enumerate(nd["id"].astype(str))}

    # --- ramas: un color por franja de uso, con leyenda; el % va sobre una etiqueta con fondo ---
    franjas = (("uso medio < 50 %", 0.00, 0.50, "#90a4ae"),
               ("uso medio 50-80 %", 0.50, 0.80, AZUL),
               ("uso medio 80-98 % (cerca del límite)", 0.80, 0.98, AMBAR),
               ("uso > 98 % (violada en alguna hora)", 0.98, 9.0, ROJO))
    for nom_f, lo, hi, col in franjas:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=nom_f,
                                 line=dict(color=col, width=3.4), hoverinfo="skip"))
    for i_r, r in ram.reset_index(drop=True).iterrows():
        ka = coord.get(str(r["de"]))
        kb = coord.get(str(r["a"]))
        if ka is None or kb is None:
            continue
        u = float(np.clip(uso_med[i_r] if i_r < len(uso_med) else 0.0, 0.0, 3.0))
        umax = float(np.clip(uso_max[i_r] if i_r < len(uso_max) else u, 0.0, 3.0))
        col = ROJO if u > 0.98 else (AMBAR if u > 0.80 else (AZUL if u > 0.50 else "#90a4ae"))
        fig.add_trace(go.Scatter(x=[ka[0], kb[0]], y=[ka[1], kb[1]], mode="lines", showlegend=False,
                                 line=dict(color=col, width=1.6 + 7.0 * min(u, 1.2), dash=None if u < 1.0
                                          else "dot"), opacity=0.95,
                                 hovertemplate=(f"<b>{r['nombre']}</b><br>{r['de']} → {r['a']}"
                                                f"<br>uso medio {100*u:.0f} % · uso máximo {100*umax:.0f} %"
                                                f"<br>límite {float(r['MW_max']):,.0f} MW"
                                                "<extra></extra>")))
        fig.add_annotation(x=(ka[0] + kb[0]) / 2.0, y=(ka[1] + kb[1]) / 2.0, text=f"{100*umax:.0f}%",
                           showarrow=False, font=dict(size=10.5, color=col), align="center",
                           bgcolor="rgba(255,255,255,0.94)", bordercolor=col, borderwidth=1,
                           borderpad=2)
    fig.add_trace(go.Scatter(x=pos[:, 0], y=pos[:, 1], mode="markers", name="Nodo (área = CEN)",
                             marker=dict(size=rad, color=[color_tecnologia("HIDRAULICA")] * len(pos),
                                         opacity=0.92, sizemode="diameter",
                                         line=dict(width=1.6, color="#37474f")),
                             customdata=np.column_stack([nd["CEN_MW"].to_numpy(dtype=float),
                                                          nd["share_demanda"].to_numpy(dtype=float)]),
                             hovertemplate=("<b>%{customdata[0]:,.0f} MW de CEN</b>"
                                            "<br>demanda: %{customdata[1]:.1%} del SIN<extra></extra>")))
    for k, nid in enumerate(nd["id"].astype(str)):
        fig.add_annotation(x=pos[k][0], y=pos[k][1], text=f"<b>{nid}</b>", showarrow=False,
                           font=dict(size=12, color="#102a43"))
        fig.add_annotation(x=pos[k][0], y=pos[k][1] - 0.055 - 0.0012 * rad[k],
                           text=(f"{float(nd['CEN_MW'].iloc[k]):,.0f} MW · "
                                 f"{100.0*float(nd['share_demanda'].iloc[k]):.1f} % dem."),
                           showarrow=False, font=dict(size=9, color="#37474f"),
                           bgcolor="rgba(255,255,255,0.88)", borderpad=1)
    fig.update_layout(xaxis=dict(visible=False, range=[-0.02, 1.02]),
                      yaxis=dict(visible=False, range=[-0.04, 1.02]),
                      hoverlabel=dict(align="left"),
                      legend=dict(orientation="h", y=-0.10, x=0),
                      annotations=[dict(text="<i>Grosor = uso medio de la rama · el recuadro es el uso "
                                              "<b>máximo</b> de las 24 h · el diámetro del círculo es la "
                                              "CEN del nodo. La posición es esquemática (topología "
                                              "declarada en el módulo), no geográfica.</i>",
                                        xref="paper", yref="paper", x=0.0, y=-0.20, showarrow=False,
                                        align="left", font=dict(size=10, color=GRIS))])
    viol = float(np.max(res["violacion"])) if np.size(res.get("violacion")) else 0.0
    return {"fig": fig, "leyenda": _leyenda(
        "La malla del modelo: ramas, uso y por qué diez nodos",
        f"Los {len(pos)} nodos son las subregiones `{red['granularidad']}` con la CEN real de XM "
        f"(cobertura toponímica {red['cobertura_toponimia']:.1f} %) y las {len(ram)} ramas del anillo "
        "500/230 kV. Cada rama se pinta según el uso medio del día y se le pone encima el uso máximo "
        "horario: azul = holgada, ámbar = al borde, rojo = violada en alguna hora.",
        f"Con `estres = {cfg['estres']:.2f}` hay {int(res['horas_congestion'])} de 24 h congestionadas y "
        f"la violación máxima es {viol:,.0f} MW. Si ningún círculo se pone rojo, los multiplicadores μ "
        "valen cero y el precio zonal y el nodal no pueden separarse: hay que apretar la malla "
        "(`Capacidad de las líneas`) o refinarla (`media`/`fina`), y aun así la congestión que aparece "
        "es la que los datos permiten sostener, no la que uno supone.",
        "En seco sube el despacho térmico del centro y la demanda del pico, así que el corredor NOR→CEN "
        "(hidráulica → carga) y el CAR→CEN (costa → carga) son los primeros en saturarse: bajar "
        "`Capacidad de las líneas` simula ese estrés de El Niño sobre la red.",
        "topología `RAMAS_BASE` + `matriz_ptdf` (PTDF de corriente continua) · CEN de `CapEfecNeta` · "
        "los refuerzos candidatos se evalúan en «Refuerzos hipotéticos de la red»")
}





def fig_nodal_cap(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Capacidad efectiva por nodo y tecnología + notas de siembra de datos faltantes."""
    cap = mdl["red"]["cap"]
    if cap.empty:
        return None
    fig = _fig(ctx, 430)
    for tec in [c for c in cap.columns if float(cap[c].max()) > 0]:
        fig.add_trace(go.Bar(x=cap.index, y=cap[tec], name=TEC_NODAL_LABEL.get(tec, tec),
                             marker_color=color_tecnologia({"HIDRO": "HIDRAULICA", "SOLAR": "SOLAR",
                                                            "EOLICA": "EOLICA", "BIO": "BAGAZO",
                                                            "CARBON": "CARBON", "GAS_CC": "GAS",
                                                            "GAS_OXI": "TERMICA"}.get(tec, "OTRO"))))
    fig.update_layout(barmode="stack", xaxis_title="Nodo", yaxis_title="CEN (MW)",
                      hovermode="closest")
    notas = [str(mdl["red"][k]) for k in ("nota_combustible", "nota_eolica") if mdl["red"].get(k)]
    txt = " · ".join(notas) if notas else "sin ajustes de siembra en esta ventana"
    return {"fig": fig, "leyenda": _leyenda(
        "De dónde sale la capacidad de cada nodo (y qué hubo que completar)",
        f"CEN por subregión y tecnología cargada desde `CapEfecNeta`+`dim_plantas` de XM, agrupada "
        f"por toponimia del nombre de la planta (cobertura "
        f"{mdl['red']['cobertura_toponimia']:.1f} % del total). {txt}",
        "El modelo no inventa capacidad: lo que no se pudo georreferenciar se reparte proporcionalmente "
        "y se declara. Si una tecnología no aparece (eólica en esta ventana), el precio nodal de ese "
        "nodo no puede reflejarla — hay que leer la figura antes de creer el LMP del nodo.",
        "El Niño cambia la mezcla, no la capacidad: la hidráulica del 62 % que se ve aquí es exactamente "
        "la que hace que un año seco suba el despacho térmico del centro y, con él, el spread CEN-NOR.",
        "Anexo XM `CapEfecNeta_Res`/`dim_plantas`  ·(UCF y recurso)")}


def fig_nodal_calib(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Calibración de la curva de oferta contra el precio de bolsa real de XM."""
    cal, pf, res = mdl["cal"], mdl["pf"], mdl["res"]
    fig = _fig(ctx, 460)
    h = np.arange(1, 25)
    real = np.asarray(pf["precio"], dtype=float)
    fig.add_trace(go.Scatter(x=h, y=real, name="Precio de bolsa XM (mediana horaria)",
                             mode="lines+markers", line=dict(color=GRIS, width=2.2),
                             marker=dict(size=6)))
    lam = np.asarray(res["lam"], dtype=float)
    if lam.size:
        fig.add_trace(go.Scatter(x=h[:len(lam)], y=lam[:len(h)], name="λ uninodal del OPF calibrado",
                                 mode="lines+markers", line=dict(color=AZUL, width=2.6),
                                 marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=h, y=np.asarray(cal["nivel_h"], dtype=float)[:len(h)],
                             name="Nivel de oferta calibrado", line=dict(color=VERDE, width=1.8, dash="dot")))
    fig.add_hline(y=float(cal["nivel0"]), line=dict(color=AMBAR, dash="dash"),
                  annotation_text=f"nivel escalar {cal['nivel0']:,.0f}", annotation_font_size=10)
    err_h = np.asarray(res["lam"], dtype=float)[:len(h)] - real[:len(res["lam"])]
    fig.add_trace(go.Bar(x=h[:len(err_h)], y=err_h, name="error por hora (λ − XM)",
                         marker_color="rgba(198,40,40,0.55)", xaxis="x2", yaxis="y2",
                         hovertemplate="h%{x}: %{y:+,.1f} COP/kWh<extra>error</extra>"))
    fig.update_layout(
        grid=dict(rows=2, columns=1, pattern="independent"), height=560,
        xaxis=dict(title="Hora · las tres series de la calibración", dtick=2),
        yaxis=dict(title="COP/kWh"),
        xaxis2=dict(title="Hora · residuo de la calibración (positivo = el modelo por encima de XM)",
                    dtick=2, matches="x"),
        yaxis2=dict(title="COP/kWh de error", zeroline=True, zerolinecolor=GRIS),
        legend=dict(orientation="h", y=1.11, x=0, groupclick="toggleitem"), hovermode="x unified")
    st.markdown(
        "**Qué muestra cada trazo del panel superior** (y por qué los tres hacen falta):\n\n"
        "- **Gris con puntos — `Precio de bolsa XM (mediana horaria)`**: el dato contra el que se calibra. "
        "Es la *mediana* por hora de `PrecBolsNaci` en la ventana, no un día suelto: por eso no ve saltos "
        "de un día al otro.\n"
        "- **Azul — `λ uninodal del OPF calibrado`**: el precio del modelo con la curva de oferta ya "
        "ajustada. Es la única línea que depende de los parámetros que usted mueve (topología, estrés, "
        "curvatura); si se despega de la gris, ahí está el error del modelo.\n"
        "- **Verde punteada — `Nivel de oferta calibrado`**: el multiplicador `nivel_h` hora a hora que "
        "enciende o apaga tramos de la curva de oferta para que la azul siga a la gris. La línea ámbar "
        "horizontal es el nivel escalar con el que arranca la bisección.\n\n"
        "**Panel inferior:** el residuo λ − XM por hora, para que no haya que restar a ojo. Si el residuo "
        "es plano en cero todo el día, la calibración está *sobreactuando* (el ajuste por hora ancla el "
        "promedio de cada hora por construcción); lo que valida la forma es la figura de spread.")
    return {"fig": fig, "leyenda": _leyenda(
        "Cómo se ancla el modelo al mercado real (y hasta dónde se le puede creer)",
        f"La curva de oferta se calibra en dos etapas: bisección de κ para que el precio *uninodal* "
        f"coincida con la mediana horaria de `PrecBolsNaci`, y un OPF por hora que ajusta el nivel. "
        f"Resultado: κ = {cal['kappa']:.3f}, nivel medio {cal['nivel']:,.1f} COP/kWh, error de "
        f"calibración {cal['error_calibracion_pct']:.2f} % y MAE {cal['mae']:,.1f} COP/kWh.",
        "El nivel y la varianza del modelo están anclados al dato, **pero la forma intradía del R² "
        "no es una validación**: el ajuste por hora hace que la curva calibrada reproduzca el promedio "
        "de esa hora por construcción. Para comparar *formas* hay que mirar la figura del spread.",
        "Calibrar sobre la ventana seca (ago-2026) sesga el nivel al alza: en un mes húmedo el mismo κ "
        "produciría precios 30-60 % menores, que es justo la amplitud que se mide entre fase "
        "La Niña y El Niño.",
        f" · error {cal['error_calibracion_pct']:.2f} % · n={cal['n']} h · "
        f"{str(cal.get('nota', ''))[:90]}")}


def fig_nodal_lmp(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Mapa de calor del LMP por nodo y hora bajo la regla elegida."""
    res, regla = mdl["res"], cfg["regla"]
    nodos = res["nodos"]
    p = np.asarray(res["precios"].get(regla, res["p_dem"]), dtype=float)
    if p.size == 0:
        return None
    fig = _fig(ctx, 470)
    fig.add_trace(go.Heatmap(z=p.T, x=[f"{i+1:02d}" for i in range(p.shape[0])], y=list(nodos),
                            colorscale=[[0.0, "#e3f2fd"], [0.45, "#4fc3f7"], [0.7, "#f9a825"],
                                        [1.0, "#c62828"]],
                            colorbar=dict(title="COP/kWh", thickness=12, len=0.85),
                            customdata=np.round(p - p.mean(axis=0), 1).T,
                            hovertemplate="%{y} · h%{x}<br>%{z:,.1f} COP/kWh<br>Δ vs medio %{customdata:+.1f}"
                                          "<extra></extra>", name="LMP"))
    lam = np.asarray(res["lam"], dtype=float)
    fig.add_trace(go.Scatter(x=[f"{i+1:02d}" for i in range(24)], y=[float(np.mean(lam))] * 24,
                             name="λ uninodal (media)", line=dict(color="#263238", width=1.4, dash="dash")))
    for hh in np.atleast_1d(np.asarray(res.get("horas_congestion", []), dtype=int)):
        fig.add_vline(x=float(int(hh)) + 1, line=dict(color="rgba(198,40,40,0.55)", width=1.2),
                      line_dash="dot")
    fig.update_layout(xaxis_title="Hora", yaxis_title="", hovermode="closest")
    media = float(np.mean(p))
    disp = float(np.mean(p.max(axis=1) - p.min(axis=1)))
    return {"fig": fig, "leyenda": _leyenda(
        f"Precios marginales nodales ({regla})",
        f"Cada celda es el λ del nodo en esa hora: {p.shape[1]} nodos × 24 h para el día medio de la "
        f"ventana con `topología {mdl['red']['granularidad']}` y líneas a ×{cfg['estres']:.2f}. "
        f"Media {media:,.1f} COP/kWh, separación media entre el nodo más caro y el más barato "
        f"{disp:,.1f} COP/kWh.",
        "Las franjas verticales oscuras son las horas en que la red separa precios: ahí es donde la "
        "liquidación nodal transfiere dinero y donde un inversario querría su nodo. Si la figura se ve "
        "plana (una sola franja de color), la topología elegida no tiene congestión y cualquier "
        "conclusión sobre 'el valor de la señal locacional' es prematura.",
        "En El Niño el hidro baja y sube el térmico del centro: la banda de horas 18-21 se calienta "
        "antes que las demás porque es cuando la demanda pica sin sol — el spread nodal crece justo ahí.",
        " · media de 31 días XM (2026-07-31→2026-08-26)")}


def fig_nodal_spread(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Banda min-med-max de los precios nodales por hora, con el uninodal y la renta."""
    res = mdl["res"]
    p = np.asarray(res["precios"].get(cfg["regla"], res["p_dem"]), dtype=float)
    if p.size == 0:
        return None
    h = np.arange(1, p.shape[0] + 1)
    lo, hi = p.min(axis=1), p.max(axis=1)
    med = np.median(p, axis=1)
    fig = _fig(ctx, 440)
    fig.add_trace(go.Scatter(x=np.concatenate([h, h[::-1]]), y=np.concatenate([hi, lo[::-1]]),
                             fill="toself", fillcolor="rgba(31,119,180,0.16)", line=dict(width=0),
                             name="Mín–máx entre nodos", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=h, y=med, name="Mediana nodal", line=dict(color=AZUL, width=2.4)))
    fig.add_trace(go.Scatter(x=h, y=np.asarray(res["lam"], dtype=float), name="λ uninodal (hoy)",
                             line=dict(color=GRIS, width=1.8, dash="dash")))
    fig.add_trace(go.Bar(x=h, y=np.asarray(res["renta_congestion"], dtype=float) / 1e6,
                         name="Renta de congestión (M COP/h)", yaxis="y2",
                         marker_color="rgba(146,64,197,0.45)"))
    fig.update_layout(xaxis=dict(title="Hora", dtick=2), yaxis_title="COP/kWh",
                      yaxis2=dict(title="M COP/h", overlaying="y", side="right", showgrid=False),
                      hovermode="x unified")
    rc_dia = float(np.sum(res["renta_congestion"])) / 1e6
    return {"fig": fig, "leyenda": _leyenda(
        "Cuánto se separan los nodos y cuánto vale esa separación",
        "Banda min–máx–mediana de los precios nodales por hora, con el λ uninodal de referencia y "
        "la renta de congestión en barras (Σ_nodos λ_i·D_i − λ·Σ D_i, que en flujo DC es Σ r_k F_k²).",
        f"La renta media es {rc_dia:,.0f} M COP por día tipo. Ese dinero no es un costo social: es "
        "transferencia entre agentes y, bajo precio nodal, es la señal (y el fondo) para financiar "
        "refuerzos. Bajo liquidación uninodal, lo paga todo el sistema en el precio sin aparecer en "
        "ninguna parte.",
        "Con embalses bajos la mediana se levanta y la banda se ensancha: la figura es el mecanismo por "
        "el cual se espera que el precio nodal *revele* el estrés hídrico por zona en vez de "
        "promediarlo.",
        "objetivo b · "
        f"dispersion media {float(np.mean(res['dispersion'])):.1f} COP/kWh")}
def fig_nodal_cong(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Uso de ramas por hora, violación, pérdidas y precio en las horas activas."""
    res, ram = mdl["res"], mdl["red"]["ramas"]
    uso = np.asarray(res["uso_rama"], dtype=float)
    if uso.size == 0 or ram.empty:
        return None
    h = np.arange(1, uso.shape[0] + 1)
    fig = _fig(ctx, 470)
    nom = list(ram["nombre"].astype(str))
    for j in range(uso.shape[1]):
        fig.add_trace(go.Scatter(x=h, y=100.0 * np.clip(uso[:, j], 0.0, 1.6), name=nom[j] if j < len(nom) else f"r{j}",
                                 mode="lines", line=dict(width=1.5), opacity=0.85, visible=(j < 6)))
    for j in range(6, uso.shape[1]):
        fig.add_trace(go.Scatter(x=h, y=100.0 * np.clip(uso[:, j], 0.0, 1.6), name=nom[j] if j < len(nom) else f"r{j}",
                                 mode="lines", line=dict(width=1.0), opacity=0.55, visible="legendonly"))
    fig.add_hline(y=100.0, line=dict(color=ROJO, width=1.6), annotation_text="límite térmico",
                  annotation_font_size=10)
    fig.add_trace(go.Scatter(x=h, y=np.asarray(res["perdidas"], dtype=float), name="Pérdidas (MW)",
                             yaxis="y2", line=dict(color=AMBAR, width=2.0, shape="spline")))
    viol = np.asarray(res.get("violacion", np.zeros_like(uso[:, 0])), dtype=float)
    fig.add_trace(go.Bar(x=h, y=np.clip(viol, 0.0, None), name="Violación (MW)", yaxis="y2",
                         marker_color="rgba(198,40,40,0.30)"))
    fig.update_layout(xaxis=dict(title="Hora", dtick=2), yaxis_title="% del límite de la rama",
                      yaxis2=dict(title="MW", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.30, x=0), height=500, hovermode="x unified")
    media = float(np.mean(uso)) * 100.0
    mx = float(np.max(uso)) * 100.0
    return {"fig": fig, "leyenda": _leyenda(
        "Congestión: qué rama se llena y a qué hora",
        f"Uso medio/máximo de las {uso.shape[1]} ramas del modelo hora a hora (las 6 primeras visibles; "
        f"el resto en la leyenda), con las pérdidas iteradas y la violación del límite en MW. Uso medio "
        f"{media:.0f} %, máximo {mx:.0f} %.",
        f"`horas_congestion = {int(res['horas_congestion'])}`, `max_violacion = "
        f"{float(np.max(viol)) if viol.size else 0.0:,.0f} MW`, `converged = {bool(res['converged'])}`. "
        "Si el máximo no llega a 100 %, los multiplicadores μ son cero y el LMP colapsa al uninodal: "
        "no es que el modelo ignore la red, es que *esa* red no congestiona: hay que "
        "estresarla (`Capacidad de las líneas`) o refinarla (`media`) para que la señal aparezca.",
        "Las pérdidas crecen con el despacho térmico del centro y con la rampa de la tarde; en un mes "
        "seco el pico de pérdidas se alinea con el pico de precio, así que la señal nodal *sin* pérdidas "
        "linealizadas (iteraciones = 0) subestima el spread entre NOR y CEN.",
        " · perdas_iter = " + str(cfg["perdas_iter"]))}


def fig_nodal_flujos(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Flujo medio por rama vs su límite y la renta de congestión que genera."""
    res, ram = mdl["res"], mdl["red"]["ramas"]
    fl = np.asarray(res["flujos"], dtype=float)
    if fl.size == 0 or ram.empty:
        return None
    mu = np.asarray(res["mu"], dtype=float)
    mwr = np.asarray(ram["MW_max"], dtype=float)[:fl.shape[1]]
    fmed = np.nanmean(np.abs(fl), axis=0)
    uso = np.nanmax(np.abs(fl), axis=0) / np.maximum(mwr, 1e-9)
    rent = 1000.0 * np.nanmean(mu * np.abs(fl), axis=0)        # μ_k·|F_k| ≈ COP/h por rama
    orden = np.argsort(-fmed)
    y = [str(ram["nombre"].iloc[i])[:30] for i in orden]
    fig = _fig(ctx, max(320.0, 26.0 * len(y)))
    fig.add_trace(go.Bar(y=y, x=fmed[orden], orientation="h", name="|flujo| medio (MW)",
                         marker_color=AZUL, opacity=0.85,
                         customdata=[[f"{100*float(uso[i]):.0f} %", f"{float(rent[i]):,.0f}",
                                      f"{float(mwr[i]):,.0f}"] for i in orden],
                         hovertemplate="%{y}<br>|flujo| medio %{x:,.0f} MW<br>uso máx %{customdata[0]}"
                                       "<br>límite %{customdata[2]} MW<br>renta %{customdata[1]} COP/h"
                                       "<extra></extra>"))
    fig.add_trace(go.Bar(y=y, x=mwr[orden], orientation="h", name="Límite MW_max",
                         marker_color="rgba(97,97,97,0.28)", hoverinfo="skip"))
    fig.update_layout(barmode="overlay", xaxis_title="MW", yaxis_title="Rama",
                      legend=dict(orientation="h", y=1.10, x=0))
    j = int(np.argmax(uso))
    return {"fig": fig, "leyenda": _leyenda(
        "Quién transporta y quién cobra la congestión",
        "Flujo medio absoluto por rama contra su límite declarado, ordenado de mayor a menor, con el "
        "uso máximo y la renta de congestión media (μ_k·|F_k|) en el hover.",
        f"La rama que domina el ranking es la que fija el spread: `{ram['nombre'].iloc[orden[0]]}` "
        f"(uso máximo {100*float(uso[orden[0]]):.0f} %). El valor de un refuerzo se mide por la renta "
        "que elimina, y esa renta es el presupuesto natural del refuerzo (el argumento de CREG 143/2021 "
        "y del esquema PJM de referencia).",
        "En la ventana seca analizada el corredor hidráulico-norte → centro-este carga más porque el "
        "norte genera y el centro consume: es la dirección que un El Niño prolongado agrava.",
        f" · renta media {float(np.mean(res['renta_congestion']))/1e6:,.1f} M COP/h · "
        f"{j} = índice de la rama más usada")}


def fig_nodal_desp(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Despacho medio por nodo y tecnología, con recorte renovable y corte flexible."""
    res = mdl["res"]
    pb = np.asarray(res["P_bloque"], dtype=float)
    if pb.size == 0:
        return None
    of = res["oferta"]
    tec = np.asarray(of.get("tec", []), dtype=object)
    nodos = list(res["nodos"])
    h = np.arange(1, pb.shape[0] + 1)
    fig = _fig(ctx, 460)
    for t in sorted(set(str(x) for x in tec)):
        cols = [i for i, x in enumerate(tec) if str(x) == t]
        if not cols:
            continue
        serie = pb[:, cols].sum(axis=1)
        fig.add_trace(go.Scatter(x=h, y=serie, name=TEC_NODAL_LABEL.get(t, t), stackgroup="one",
                                 line=dict(width=0.5, color=color_tecnologia(
                                     {"HIDRO": "HIDRAULICA", "SOLAR": "SOLAR", "EOLICA": "EOLICA",
                                      "BIO": "BAGAZO", "CARBON": "CARBON", "GAS_CC": "GAS",
                                      "GAS_OXI": "TERMICA"}.get(t, "OTRO"))),
                                 fillcolor=color_tecnologia({"HIDRO": "HIDRAULICA", "SOLAR": "SOLAR",
                                                             "EOLICA": "EOLICA", "BIO": "BAGAZO",
                                                             "CARBON": "CARBON", "GAS_CC": "GAS",
                                                             "GAS_OXI": "TERMICA"}.get(t, "OTRO"))))
    rec = np.asarray(res["recorte"], dtype=float).sum(axis=1)
    cort = np.asarray(res["corte"], dtype=float).sum(axis=1)
    fig.add_trace(go.Scatter(x=h, y=np.asarray(res["dem"], dtype=float).sum(axis=1),
                             name="Demanda del día medio", line=dict(color="#263238", width=2.0, dash="dot")))
    if float(np.max(rec)) > 0.5:
        fig.add_trace(go.Bar(x=h, y=rec, name="Recorte renovable (MW)", yaxis="y2",
                             marker_color="rgba(249,168,37,0.6)"))
    if float(np.max(cort)) > 0.5:
        fig.add_trace(go.Bar(x=h, y=cort, name="Corte de demanda flexible (MW)", yaxis="y2",
                             marker_color="rgba(198,40,40,0.55)"))
    fig.update_layout(xaxis=dict(title="Hora", dtick=2), yaxis_title="MW despachados (por nodo, sumados)",
                      yaxis2=dict(title="MW no servidos / recortados", overlaying="y", side="right",
                                  showgrid=False), legend=dict(orientation="h", y=1.12, x=0))
    return {"fig": fig, "leyenda": _leyenda(
        "El despacho óptimo del día medio (y lo que se deja de inyectar)",
        f"Generación por tecnología sumada sobre los {len(nodos)} nodos, con la demanda de referencia y, "
        f"si existen, el recorte renovable ({float(np.sum(res['recorte_MWh']))/1000.0:,.1f} GWh/día) y el "
        f"corte de la demanda flexible ({float(np.sum(cort))/1000.0:,.1f} GWh/día).",
        "La hora en que el recorte es grande es la hora en que la red vale más: con uninodal ese costo "
        "lo paga el sistema entero; con nodal lo paga quien está al final del corredor. El corte "
        f"aparece en {int(np.sum(cort > 0.5))} horas y exige precio > {cfg['gatillo']:,.0f} COP/kWh.",
        "En El Niño la capa hidráulica se aplana y la térmica cubre la rampa de la tarde: la figura "
        "cambia de color justo en 17-21 h, que es cuando sube el costo marginal y el recorte solar "
        "ya no ayuda.",
        f" · flex_frac {cfg['flex']:.0f} % · co2 {float(np.sum(res['co2_t'])):,.0f} t/día")}


def fig_nodal_nodo(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Todas las curvas: precio de cada nodo arriba y posición neta (gen − dem) de cada nodo abajo."""
    res = mdl["res"]
    p = np.asarray(res["precios"].get(cfg["regla"], res["p_dem"]), dtype=float)
    nodos = list(res["nodos"])
    if p.size == 0 or len(nodos) < 2:
        return None
    media = p.mean(axis=0)
    caro, bar = int(np.argmax(media)), int(np.argmin(media))
    dem = np.asarray(res["dem"], dtype=float)
    gen = np.asarray(res["P_nodo"], dtype=float) + np.asarray(res["ren"], dtype=float)
    h = np.arange(1, p.shape[0] + 1)
    fig = _fig(ctx, 620)
    orden = list(np.argsort(-media))
    for j in orden:
        grosor = 3.0 if j in (caro, bar) else 1.2
        opac = 1.0 if j in (caro, bar) else 0.62
        col = ROJO if j == caro else (VERDE if j == bar else None)
        fig.add_trace(go.Scatter(x=h, y=p[:, j], name=f"{nodos[j]} · {media[j]:,.0f}",
                                 line=dict(width=grosor, color=col), opacity=opac,
                                 xaxis="x", yaxis="y",
                                 hovertemplate=f"<b>{nodos[j]}</b> {{y:,.1f}} COP/kWh<extra></extra>"))
    fig.add_trace(go.Scatter(x=h, y=np.asarray(res["lam"], dtype=float)[:len(h)],
                             name="λ uninodal (referencia)", line=dict(color="#263238", width=1.4,
                                                                       dash="dot"), xaxis="x", yaxis="y",
                             opacity=0.9))
    for j in orden:
        col = ROJO if j == caro else (VERDE if j == bar else None)
        fig.add_trace(go.Scatter(x=h, y=gen[:, j] - dem[:, j], name=f"{nodos[j]} · posición",
                                 line=dict(width=2.4 if j in (caro, bar) else 1.0, color=col),
                                 opacity=1.0 if j in (caro, bar) else 0.5, showlegend=False,
                                 xaxis="x2", yaxis="y2",
                                 hovertemplate=f"<b>{nodos[j]}</b> {{y:+,.0f}} MW<extra>gen − dem</extra>"))
    fig.add_trace(go.Scatter(x=h, y=np.zeros(len(h)), name="nodo balanceado (0 MW)", line=dict(
        color=GRIS, width=1.1, dash="dash"), showlegend=True, xaxis="x2", yaxis="y2",
        hoverinfo="skip"))
    fig.update_layout(
        grid=dict(rows=2, columns=1, pattern="independent"),
        xaxis=dict(title=f"Hora · precio por nodo bajo la regla `{cfg['regla']}` (línea gruesa = el más "
                         f"caro {nodos[caro]} y el más barato {nodos[bar]})", dtick=3),
        yaxis=dict(title="COP/kWh", rangemode="tozero"),
        xaxis2=dict(title="Hora · posición neta del nodo = generación propia (incluida la variable) − demanda",
                    dtick=3, matches="x"),
        yaxis2=dict(title="MW (− importa / + exporta)"),
        legend=dict(orientation="h", y=1.10, x=0, groupclick="toggleitem", font=dict(size=10)),
        hovermode="closest")
    st.markdown("**Ranken de nodos de la jornada (para que sepa qué curva está mirando)**")
    tabla = pd.DataFrame(dict(nodo=nodos, precio_medio_COP_kWh=media,
                              vs_uninodal_COP_kWh=media - float(np.mean(res["lam"])),
                              demanda_media_MW=dem.mean(axis=0)[:len(nodos)],
                              generacion_media_MW=gen.mean(axis=0)[:len(nodos)],
                              horas_sobre_lambda=[int(np.sum(p[:, j] > float(np.mean(res["lam"]))))
                                                  for j in range(len(nodos))]))
    st.dataframe(tabla.sort_values("precio_medio_COP_kWh", ascending=False).round(1),
                 use_container_width=True, hide_index=True, height=min(360, 60 + 30 * len(tabla)))
    st.caption("En el panel superior **están todas las curvas de precio a la vez** (una por nodo; antes "
               "solo se pintaban las dos de los extremos). Haga clic en la leyenda para aislar nodos. En "
               "el inferior, la posición neta: un nodo persistentemente negativo importa y por eso paga más "
               "que el promedio; uno positivo exporta y cobra menos.")
    return {"fig": fig, "leyenda": _leyenda(
        "Todos los nodos, no solo los dos extremos",
        f"Precio horario de los {len(nodos)} nodos bajo la regla `{cfg['regla']}` (arriba, con λ uninodal "
        "de referencia en gris punteado) y posición neta generación−demanda de cada nodo (abajo). El "
        f"extremo más caro es {nodos[caro]} ({media[caro]:,.1f} COP/kWh de media) y el más barato "
        f"{nodos[bar]} ({media[bar]:,.1f}): brecha {float(media.max() - media.min()):,.1f} COP/kWh.",
        "Con todas las curvas se ve la estructura que los dos extremos esconden: si las curvas se agrupan "
        "en dos manojos que se separan solo en horas pico, la congestión es *de franja* y se paga con "
        "flexibilidad; si un nodo se despega solo y todo el día, el problema es de localización y se paga "
        "con red o con generación en ese nodo. Abajo, la posición neta dice en qué horas el nodo caro "
        "realmente no tiene con qué cubrirse.",
        "El nodo barato suele ser el hidro-norte y el caro el centro-este; en seco los dos manojos se "
        "separan más porque el norte deja de mandar excedente y el centro enciende térmica. Con `media` o "
        "`fina` aparecen los nodos desdoblados: ahí se ve si la separación nace dentro de la subregión.",
        "balance impuesto solo a nivel de sistema (el derrame por nodo es la consecuencia: máximo "
        f"{float(np.max(np.asarray(res['derrame_MW'], dtype=float))):,.1f} MW)")
}


def fig_nodal_reglas(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Uninodal vs zonal vs híbrido vs nodal: KPIs de la liquidación."""
    kp = mdl.get("kpis")
    if not isinstance(kp, pd.DataFrame) or kp.empty:
        return None
    fig = _fig(ctx, 450)
    columnas = [("precio_medio_COP_kWh", "Precio medio (COP/kWh)", 1.0),
                ("dispersion_media_COP_kWh", "Dispersión media (COP/kWh)", 40.0),
                ("renta_congestion_COP", "Renta de congestión (M COP/h)", 1e-6),
                ("desbalance_COP", "Desbalance pago−ingreso (M COP/h)", 1e-6)]
    for j, (col, tit, esc) in enumerate(columnas):
        if col not in kp.columns:
            continue
        fig.add_trace(go.Bar(x=kp["regla"], y=np.asarray(kp[col], dtype=float) * esc,
                             name=tit, xaxis=f"x{j+1}" if j else "x",
                             marker_color=[AZUL, GRIS, AMBAR, ROJO][j % 4], opacity=0.85,
                             text=[f"{v:,.1f}" for v in np.asarray(kp[col], dtype=float) * esc],
                             textposition="outside"))
    fig.update_layout(grid=dict(rows=1, columns=len(columnas), pattern="independent"),
                      showlegend=False, yaxis_title="")
    for j, (_c, tit, _e) in enumerate(columnas[:len(columnas)]):
        fig.update_layout(**{f"xaxis{'2345'[j] if j else ''}": dict(title=tit)})
    uni = kp[kp["regla"] == "uninodal"]
    nod = kp[kp["regla"] == "nodal"]
    d = float(nod["desbalance_COP"].iloc[0] - uni["desbalance_COP"].iloc[0]) / 1e6 if len(uni) and len(nod) else float("nan")
    return {"fig": fig, "leyenda": _leyenda(
        "Las cuatro reglas de liquidación, una al lado de la otra",
        "Mismo despacho, cuatro formas de facturar: uninodal (un precio), zonal (uno por subregión), "
        f"híbrido (bolsa + κ={cfg['kappa']:.2f}·componente nodal) y nodal pleno. Se reportan precio "
        "medio, dispersión, renta de congestión y el desbalance entre lo que paga la demanda y lo que "
        "recibe la generación.",
        f"El desbalance crece {d:,.1f} M COP/h al pasar de uninodal a nodal: la energía perdida en las "
        "líneas y la congestión *tienen* que pagarlas alguien, y hoy no se asignan. Un diseño de mercado "
        "sin derechos de congestión (CRF) transfiere ese dinero al operador; con CRF, a los dueños de la "
        "línea. Es la decisión regulatoria, no la ingeniería.",
        "Con embalses bajos el precio medio sube en las cuatro reglas por igual, pero la dispersión y "
        "la renta solo aparecen en las dos últimas: en El Niño la señal locacional se vuelve útil justo "
        "cuando el sistema está estresado.",
        f" · kpis_reglas() · {len(kp)} reglas · " + str(mdl.get("nota_reglas", ""))[:80])}


def fig_nodal_descomp(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Descomposición del LMP en energía + pérdidas + congestión."""
    de = mdl.get("desc")
    if not isinstance(de, pd.DataFrame) or de.empty:
        return None
    hh = _nodal_h(cfg["hora_foco"])
    d = de[de["hora"] == hh] if "hora" in de.columns else de
    if d.empty:
        d = de.head(len(mdl["res"]["nodos"]))
    fig = _fig(ctx, 430)
    base = float(np.min(d["energia"])) if "energia" in d.columns else 0.0
    for col, nom, col_ in (("energia", "Componente energía λ", AZUL),
                           ("perdidas", "Pérdidas λ·LF_i", AMBAR),
                           ("congestion", "Congestión Σ SF_ki·μ_k", ROJO)):
        if col not in d.columns:
            continue
        fig.add_trace(go.Bar(x=d["nodo"], y=np.asarray(d[col], dtype=float) - (base if col == "energia" else 0.0),
                             name=nom, marker_color=col_, opacity=0.88))
    if "reportado" in d.columns:
        fig.add_trace(go.Scatter(x=d["nodo"], y=d["reportado"], name="LMP reportado por el OPF",
                                 mode="lines+markers", line=dict(color="#263238", width=1.6)))
    fig.update_layout(barmode="stack", xaxis_title="Nodo (cada barra es un nodo de la topología activa)",
                      yaxis_title=f"COP/kWh (eje cortado en {base:,.0f} para que se vean las componentes)",
                      legend=dict(orientation="h", y=1.14, x=0, groupclick="toggleitem"),
                      hovermode="x unified")
    hh_cong = [int(x) for x in np.atleast_1d(np.asarray(mdl["res"].get("horas_congestion", []),
                                                        dtype=int))] if "horas_congestion" in mdl["res"] else []
    st.markdown(
        f"**Qué es cada capa de la barra (hora foco {hh + 1}, elegida con `Hora foco` en la barra lateral):**"
        "\n\n"
        "- **Azul · componente energía** — el λ del sistema de esa hora, igual para todos los nodos. Es la "
        "base: si un nodo solo difiere por esto, el precio *zonal* y el *nodal* le dan lo mismo.\n"
        "- **Ámbar · pérdidas** — λ·LF_i: lo que cuesta transportar la energía hasta ese nodo. Es positivo "
        "en los nodos que importan y negativo en los que exportan (por eso la barra azul está cortada en "
        f"{base:,.0f}: sin ese corte las dos componentes locacionales serían invisibles junto a los ~{float(np.mean(d['energia'])) if len(d) else 0:,.0f} COP/kWh de energía).\n"
        "- **Rojo · congestión** — Σ_k SF_ki·μ_k, la suma de lo que vale relajar cada rama saturada, "
        "pesada por cuánto incide el nodo en esa rama (su factor de desplazamiento). Es **cero en las horas "
        "sin congestión**: si toda la barra roja está vacía, cambie la hora foco.\n"
        "- **Línea negra punteada · LMP reportado** — el precio que efectivamente devolvió el OPF. Debe "
        "coincidir con la cima de la pila; la diferencia se reporta como `desvío` en la tabla.")
    if hh_cong:
        st.caption(f"Horas con congestión en este escenario: "
                   f"{', '.join(str(v) for v in sorted(set(hh_cong)))[:180]}. Ponga la hora foco en una de "
                   "ellas para ver la componente congestión; en hora valle el precio queda dominado por "
                   "pérdidas.")
    elif "hora" in d.columns:
        st.caption("No hay horas congestionadas en este escenario: la componente de congestión vale cero "
                   "en todos los nodos y la figura solo muestra energía + pérdidas. Apreté "
                   "`Capacidad de las líneas` o cambie de topología para ver la señal locacional completa.")
    maxdesv = float(np.max(np.abs(d["desvio"]))) if "desvio" in d.columns else float("nan")
    return {"fig": fig, "leyenda": _leyenda(
        f"De dónde sale el precio de cada nodo (hora {hh+1})",
        " desglosada: λ de sistema + el término de pérdidas (λ·LF_i, con el factor de "
        "participación del nodo) + la suma de multiplicadores de rama ponderados por PTDF. El punto "
        "negro es el LMP que realmente devolvió el OPF.",
        f"La suma de las tres componentes reproduce el precio con un error máximo de {maxdesv:.3g} "
        "COP/kWh. Si el término de congestión es cero en todos los nodos, la hora elegida no está "
        "congestionada: cambie la hora foco a las franjas rojas de la figura de congestión.",
        "En hora valle el sol deja el precio dominado por pérdidas (componente negativa en los nodos "
        "exportadores); en el pico de la tarde la congestión manda. El Niño corre el pico de pérdidas "
        "hacia 18-20 h porque el térmico del centro trabaja más.",
        " · descomposicion_precios()")}


def fig_nodal_valida(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Validación del multiplicador dual contra ∂Costo/∂D por diferencias finitas, con umbral explícito."""
    vr = mdl.get("vr")
    if isinstance(vr, dict):
        st.info(f"Validación no disponible: {vr.get('nota', 'no ejecutada')}")
        return None
    if not isinstance(vr, pd.DataFrame) or vr.empty:
        return None
    err = np.abs(np.asarray(vr["error_pct"], dtype=float))
    xd = np.asarray(vr["lambda_dual_cop_kWh"], dtype=float)
    yf = np.asarray(vr["lambda_dif_finita_cop_kWh"], dtype=float)
    umbral = 0.5                                             # % de error admitido como "iguales"
    ok = err <= umbral
    fig = _fig(ctx, 470)
    lo = float(min(xd.min(), yf.min()))
    hi = float(max(xd.max(), yf.max()))
    # banda de ±umbral alrededor de la diagonal (en %, sobre el valor de la diagonal)
    g = np.linspace(lo, hi, 60)
    fig.add_trace(go.Scatter(x=np.concatenate([g, g[::-1]]),
                             y=np.concatenate([g * (1.0 + umbral / 100.0), (g * (1.0 - umbral / 100.0))[::-1]]),
                             fill="toself", fillcolor="rgba(46,125,50,0.12)", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xd[~ok], y=yf[~ok], mode="markers", name=f"❌ error > {umbral:.1f} %",
                             marker=dict(size=12, color=ROJO, symbol="x", line=dict(width=1.4, color="#7f0000")),
                             text=vr["nodo"].to_numpy()[~ok],
                             customdata=np.column_stack([err[~ok],
                                                          np.asarray(vr["horas"], dtype=int)[~ok]]),
                             hovertemplate=("<b>%{text}</b><br>dual %{x:,.1f} · FD %{y:,.1f}"
                                            "<br>error %{customdata[0]:.2f} % en %{customdata[1]} h"
                                            "<extra></extra>")))
    fig.add_trace(go.Scatter(x=xd[ok], y=yf[ok], mode="markers", name=f"✔ error ≤ {umbral:.1f} %",
                             marker=dict(size=9, color=VERDE, opacity=0.85,
                                         line=dict(width=1.0, color="#1b5e20")),
                             text=vr["nodo"].to_numpy()[ok],
                             customdata=np.column_stack([err[ok],
                                                          np.asarray(vr["horas"], dtype=int)[ok]]),
                             hovertemplate=("<b>%{text}</b><br>dual %{x:,.1f} · FD %{y:,.1f}"
                                            "<br>error %{customdata[0]:.3f} % en %{customdata[1]} h"
                                            "<extra></extra>")))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x (igualdad exacta)",
                             line=dict(color=GRIS, width=1.2, dash="dash")))
    fig.update_layout(xaxis=dict(title="λ_i que devuelve el ascenso dual (COP/kWh)", ticksuffix=""),
                      yaxis=dict(title="∂Costo/∂D_i medido por diferencias finitas (COP/kWh)"),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="closest")
    peor = (vr.assign(error=np.asarray(vr["error_pct"], dtype=float))
            .reindex(columns=["nodo", "error", "lambda_dual_cop_kWh", "lambda_dif_finita_cop_kWh",
                              "paso_MW", "horas"])
            .sort_values("error", key=lambda c: c.abs(), ascending=False)
            .rename(columns={"lambda_dual_cop_kWh": "dual", "lambda_dif_finita_cop_kWh": "FD",
                             "paso_MW": "paso_MW", "horas": "h validadas"})
            .head(6).round(3))
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(f"**Los 6 puntos con mayor desviación** (de {len(vr)} nodos-hora validados; el "
                    f"criterio de aprobación es |error| ≤ {umbral:.1f} %):")
        st.dataframe(peor, use_container_width=True, hide_index=True, height=240)
    with c2:
        st.markdown("**Qué es cada punto**")
        st.markdown(
            f"- **Verde ✔** — el multiplicador del OPF y la derivada numérica coinciden dentro de "
            f"{umbral:.1f} %: el dual es el precio de balance del nodo.\n"
            f"- **Rojo ✖** — difieren más del umbral. Suele pasar en horas congestionadas con el dual "
            "aún no convergido (la linealización de pérdidas deja un residuo) o cuando el paso de "
            f"{float(np.mean(np.asarray(vr['paso_MW'], dtype=float))):.1f} MW cruza un cambio de bloque "
            "activo y la FD salta discretamente mientras el dual sigue siendo lineal.\n"
            f"- **Banda verde clara** — la propia tolerancia de ±{umbral:.1f} %. Si todos los puntos "
            "caen dentro, la prueba está aprobada aunque la banda se vea gruesa a esta escala.\n"
            f"- **Diagonal gris** — igualdad exacta.\n\n"
            f"Aprobado en {int(np.sum(ok))}/{len(err)} puntos ({100.0*float(np.mean(ok)):.1f} %), error "
            f"máximo {float(np.max(err)):.2f} % y mediano {float(np.median(err)):.3f} %.")
    return {"fig": fig, "leyenda": _leyenda(
        "El precio del modelo contra la derivada numérica del costo",
        f"La definición de precio nodal es que λ_i sea la sombra del balance del nodo, ∂Costo/∂D_i. Se "
        f"midió añadiendo {float(np.mean(np.asarray(vr['paso_MW'], dtype=float))):.1f} MW a la demanda "
        f"de cada nodo y recalculando el OPF completo: {len(vr)} nodos, {int(np.mean(np.asarray(vr['horas'], dtype=float)))} "
        "horas cada uno (el punto es el promedio del nodo en esas horas: una hora mala se diluye, así que "
        "la prueba se complementa con la tabla de la derecha).",
        f"Se aprueba con |error| ≤ {umbral:.1f} %: pasan {int(np.sum(ok))} de {len(err)} nodos "
        f"(cada uno promediado sobre {int(np.mean(np.asarray(vr['horas'], dtype=float)))} h). Este "
        "panel es lo que legitima todo lo demás de la pestaña —renta de congestión, descomposición del "
        "LMP, cobertura—, porque sin dual correcto no hay precio que liquidar. Si la nube se descolgara "
        "de la diagonal, habría que subir `Iteraciones de pérdidas` o `dual_iter` antes de creerle a "
        "cualquier otra cifra.",
        "Los rojos se concentran en horas con congestión activa, donde la linealización de pérdidas es más "
        "burda: en un mes seco hay más horas así, así que la validez numérica se vuelve el primer "
        "cuello de botella (suba `Horas a validar` para confirmar que no es azar de la ventana).",
        "valida_lmp() con diferencias finitas sobre el mismo OPF; el umbral de aprobación es criterio "
        "declarado de esta app, no un estándar de mercado")
}


def fig_nodal_esc(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Abanico de escenarios y, si hay segunda etapa, bandas de precio por nodo sin números encima."""
    esc = mdl.get("esc")
    if not isinstance(esc, dict) or not esc:
        return None
    de = np.asarray(esc["demanda_MW"], dtype=float)
    if de.size == 0:
        return None
    k, n, H = de.shape
    nodos = list(mdl["red"]["nodos"])
    w = np.asarray(mdl["red"]["peso_demanda"], dtype=float)
    w = w / max(float(np.sum(w)), 1e-12)
    fig = _fig(ctx, 500)
    tot = de.sum(axis=1)
    kk = min(k, 8)
    for i in range(kk):
        fig.add_trace(go.Scatter(x=np.arange(1, H + 1), y=tot[i], mode="lines", line=dict(width=1.2),
                                 opacity=0.55, name=f"día típico {i+1}", showlegend=(i == 0),
                                 xaxis="x", yaxis="y"))
    base_dem = np.asarray(mdl["res"]["dem"], dtype=float).sum(axis=1)
    fig.add_trace(go.Scatter(x=np.arange(1, len(base_dem) + 1), y=base_dem, name="día medio real (XM)",
                             line=dict(color="#263238", width=2.6, dash="dash"), xaxis="x", yaxis="y",
                             opacity=0.95))
    es = mdl.get("est")
    if isinstance(es, dict) and es:
        pre = np.asarray(es["precios_nodo"], dtype=float)          # (k, n) precio medio por escenario
        top = [int(i) for i in np.argsort(-w)[:6] if i < n]
        for j, i in enumerate(top):
            fig.add_trace(go.Box(y=pre[:, i], name=nodos[i], x=[nodos[i]] * int(pre.shape[0]),
                                 marker_color=[AZUL, VERDE, AMBAR, ROJO, NARANJA, GRIS][j % 6],
                                 boxmean=True, boxpoints=False, xaxis="x2", yaxis="y2",
                                 hovertemplate=f"<b>{nodos[i]}</b><br>p05 %{{lowerfence:,.0f}} · mediana "
                                 f"%{{median:,.0f}} · p95 %{{upperfence:,.0f}} COP/kWh<extra></extra>"))
        tit2 = "COP/kWh (precio medio del día por escenario)"
    else:
        top = [int(i) for i in np.argsort(-w)[:4] if i < n]
        for j, i in enumerate(top):
            fig.add_trace(go.Box(y=de[:, i, :], name=nodos[i], x=[nodos[i]] * k,
                                 marker_color=[AZUL, VERDE, AMBAR, ROJO][j % 4], boxmean=True,
                                 boxpoints=False, xaxis="x2", yaxis="y2",
                                 hovertemplate=f"<b>{nodos[i]}</b><br>%{{min:,.0f}} → %{{max:,.0f}} "
                                 f"MW<extra></extra>"))
        tit2 = "MW por hora (demanda del nodo)"
    fig.update_layout(
        grid=dict(rows=1, columns=2, pattern="independent"),
        xaxis=dict(title="Hora · demanda del sistema en los días típicos", dtick=4),
        yaxis=dict(title="MW del sistema"),
        xaxis2=dict(title="Nodo" + (" · distribución del precio entre escenarios"
                                    if isinstance(es, dict) and es else " · distribución de la demanda")),
        yaxis2=dict(title=tit2),
        legend=dict(orientation="h", y=1.13, x=0, groupclick="toggleitem"), hovermode="closest")
    st.caption(
        "Cómo se conecta este abanico con el precio: se simulan trayectorias horarias (log-normal la "
        "solar, Weibull la eólica caribeña, gamma la demanda, con la fase ENSO desplazando las medias), "
        "se reducen a días típicos por k-medias con pesos y **cada día típico se despacha con el mismo "
        "OPF del día real** — así la banda no es decoración: es el precio que habría regido en esa "
        "trayectoria, con sus propias pérdidas y su propia congestión. Se apaga con el toggle "
        "*Despachar cada escenario típico con su propio OPF* de la barra lateral.")
    return {"fig": fig, "leyenda": _leyenda(
        "Los escenarios que usa la capa estocástica",
        f"{int(mdl.get('esc_n', cfg['n_esc']))} trayectorias horarias generadas con familias "
        f"`{cfg['fam_sol']}` (solar) y `{cfg['fam_eol']}` (eólica), correlación entre nodos "
        f"ρ={cfg['rho_recurso']:.2f} y cruce sol-viento ρ={cfg['rho_cruce']:.2f}, reducidas a "
        f"{k} escenarios típicos por k-medias (inercia {float(esc.get('inercia', 0.0)):.3f}). "
        "Se dibujan los primeros 8 típicos; las cajas del panel derecho son la **dispersión del precio "
        "medio del día por escenario** cuando la segunda etapa está encendida (si no, de la demanda).",
        "El panel izquierdo responde «¿el día típico se parece al día de XM?»: si los trazos grises "
        "envuelven la curva negra sin desplazarla, la reducción a días típicos no está borrando el pico. "
        "El derecho responde la pregunta que importa para liquidar: en qué nodos el precio es *inseguro* "
        "(caja alta) y en cuáles es apenas un corrimiento del promedio (caja estrecha).",
        "En El Niño las cajas se ensanchan y se separan entre sí: la incertidumbre deja de ser un factor de "
        "escala y pasa a ser locacional, que es justo el régimen donde una liquidación uninodal distribuye "
        "mal los riesgos.",
        "gen_escenarios() → reduce_escenarios() (k-medias con pesos) → [despacho_estocastico()] → cajas "
        "por nodo; el KS de cada familia se audita en «Calibración de la oferta»")
}


def fig_nodal_flex(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Desconexión voluntaria de la demanda: malla flexibilidad × gatillo (objetivo a)."""
    fx = mdl.get("flex")
    if not isinstance(fx, pd.DataFrame) or fx.empty:
        return None
    fig = _fig(ctx, 470)
    for gf in sorted(set(float(x) for x in fx["gatillo_COP_kWh"])):
        sub = fx[fx["gatillo_COP_kWh"] == gf].sort_values("flex_pct")
        fig.add_trace(go.Scatter(x=sub["flex_pct"], y=sub["lam"], name=f"λ · gatillo {gf:,.0f}",
                                 mode="lines+markers", line=dict(width=2.2), yaxis="y"))
        fig.add_trace(go.Scatter(x=sub["flex_pct"], y=sub["corte_GWh"],
                                 name=f"GWh/día cortados · {gf:,.0f}", mode="lines+markers",
                                 line=dict(width=1.6, dash="dot"), yaxis="y2", opacity=0.8))
    c0 = fx[fx["flex_pct"] == fx["flex_pct"].min()].iloc[0]
    c1 = fx[fx["flex_pct"] == fx["flex_pct"].max()].iloc[0]
    fig.add_hline(y=float(c0["lam"]), line=dict(color=GRIS, dash="dash"),
                  annotation_text=f"sin desconexión: {float(c0['lam']):,.0f} COP/kWh",
                  annotation_font_size=10)
    _g0 = float(cfg.get("gatillo") or 0.0)
    if _g0 > 0 and float(c0["lam"]) > _g0:
        fig.add_hline(y=_g0, line=dict(color=ROJO, width=1.2, dash="dot"),
                      annotation_text=f"gatillo pedido en la barra: {_g0:,.0f}",
                      annotation_font_size=9.5, annotation_font_color=ROJO)
    # etiquetas cortas y Δ contra el caso sin flexibilidad: el signo es lo que hay que leer
    gat = sorted(set(float(v) for v in fx["gatillo_COP_kWh"]))
    ref = {g: float(fx[(fx["gatillo_COP_kWh"] == g) &
                       (fx["flex_pct"] == fx["flex_pct"].min())]["lam"].iloc[0]) for g in gat}
    refc = {g: float(fx[(fx["gatillo_COP_kWh"] == g) &
                        (fx["flex_pct"] == fx["flex_pct"].min())]["corte_GWh"].iloc[0]) for g in gat}
    refh = {g: float(fx[(fx["gatillo_COP_kWh"] == g) &
                        (fx["flex_pct"] == fx["flex_pct"].min())]["horas_congestion"].iloc[0]) for g in gat}
    fx2 = fx.copy()
    fx2["etiqueta"] = fx2["flex_pct"].map(lambda v: f"{float(v):.0f} %")
    fx2["d_lam"] = [float(a) - ref[g] for a, g in zip(fx2["lam"], fx2["gatillo_COP_kWh"])]
    fx2["d_corte"] = [float(a) - refc[g] for a, g in zip(fx2["corte_GWh"], fx2["gatillo_COP_kWh"])]
    fx2["d_cong"] = [float(a) - refh[g] for a, g in zip(fx2["horas_congestion"], fx2["gatillo_COP_kWh"])]
    colores = [AZUL, AMBAR, VERDE, ROJO, GRIS]
    fig.data = ()
    for q, g in enumerate(gat):
        sub = fx2[fx2["gatillo_COP_kWh"] == g].sort_values("flex_pct")
        fig.add_trace(go.Scatter(x=list(sub["etiqueta"]), y=sub["lam"],
                                 name=f"λ · gatillo {g:,.0f}", mode="lines+markers",
                                 line=dict(width=2.2, color=colores[q % 5]), marker=dict(size=7),
                                 xaxis="x", yaxis="y"))
        fig.add_trace(go.Bar(x=list(sub["etiqueta"]), y=sub["d_lam"],
                             name=f"Δλ · gatillo {g:,.0f}", marker_color=colores[q % 5], opacity=0.85,
                             text=[f"{v:+,.1f}" for v in sub["d_lam"]], textposition="outside",
                             xaxis="x2", yaxis="y2",
                             hovertemplate="%{x}<br>Δλ %{y:+,.2f} COP/kWh<extra></extra>"))
    fig.update_layout(
        grid=dict(rows=1, columns=2, pattern="independent"), height=470, barmode="group",
        xaxis=dict(title="Demanda interrumpible declarada", tickangle=0),
        yaxis=dict(title="λ medio del sistema (COP/kWh)"),
        xaxis2=dict(title="Demanda interrumpible declarada (misma escala)"),
        yaxis2=dict(title="Δλ contra «0 %» (COP/kWh)", zeroline=True, zerolinecolor=GRIS),
        legend=dict(orientation="h", y=1.14, x=0, groupclick="toggleitem", font=dict(size=10)),
        hovermode="closest")
    st.markdown(
        "**Cómo leer el signo (es lo único importante del panel derecho):** "
        "**Δλ < 0** = la interrupción comprada *abarata* el día: se apaga el bloque más caro del pico y el "
        "despacho marginal baja. **Δλ > 0** = la *encarece*: pasa cuando el gatillo corta energía en horas "
        "donde aún no hacía falta y obliga a encender un bloque con arranque caro para cubrir el pico, o "
        "cuando el corte en un nodo exportador convierte su excedente en congestión en otro. **Δλ ≈ 0** = "
        "la red no estaba apretando en esa hora: se paga el corte sin recibir señal. La columna "
        "`d_corte` (abajo) es el precio de la cobertura, en GWh/día no servidos.")
    st.dataframe(fx2[["flex_pct", "etiqueta", "gatillo_COP_kWh", "lam", "d_lam", "dispersion",
                      "precio_max", "corte_GWh", "d_corte", "horas_congestion", "d_cong", "recorte_GWh",
                      "viol_MW", "rc_MCP_h", "costo_MCop_h", "co2_t", "conv"]]
                 .round(1), use_container_width=True, hide_index=True, height=190)
    return {"fig": fig, "leyenda": _leyenda(
        "Desconexión voluntaria: qué se compra y qué se paga (objetivo a)",
        f"Cada punto resuelve el OPF completo con {float(c1['flex_pct']):.0f} % de la demanda "
        "declarada como interrumpible y un precio gatillo; el corte se activa hora a hora en cada "
        "nodo por encima de ese umbral. Eje izquierdo: λ medio del sistema. Eje derecho: GWh/día no "
        "servidos. La tabla añade recorte renovable, congestión, violación y CO₂ por celda.",
        f"De 0 a {float(c1['flex_pct']):.0f} % de flexibilidad el λ medio cae "
        f"{float(c1['lam']) - float(c0['lam']):+,.1f} COP/kWh y la dispersión pasa de "
        f"{float(c0['dispersion']):,.1f} a {float(c1['dispersion']):,.1f} COP/kWh, a cambio de "
        f"{float(c1['corte_GWh']):,.1f} GWh/día no servidos y {float(c1['co2_t']) - float(c0['co2_t']):+,.0f} t "
        f"de CO₂. Con el precio clavado cerca del gatillo, la diferencia entre nodos pasa a estar "
        f"dominada por el propio gradiente del corte: la renta de congestión se mueve "
        f"{float(c1['rc_MCP_h']) - float(c0['rc_MCP_h']):+,.1f} M COP/h. La desconexión voluntaria "
        "compra estabilidad de precio con energía no servida: es una cobertura física, y el costo de "
        "esa cobertura es exactamente la columna `corte_GWh`.",
        "Bajo El Niño el gatillo se cruza en más horas, así que la curva se vuelve más plana (el "
        "techo lo pone el gatillo, no el costo marginal) y el pago en bienestar se traslada a la "
        "columna de corte. Es el mecanismo que el objetivo a pide evaluar: la demanda desconectable "
        "como tercer instrumento, junto con la red y el almacenamiento.",
        "objetivo a · `barrido_flexibilidad()` con el mismo OPF (con corte por nodo); "
        "las tres líneas son los tres gatillos de la malla: p40, p75 y p95 de los precios "
        "nodales de **este** escenario (el gatillo de la barra de controles, "
        + (f"{float(cfg.get('gatillo') or 0.0):,.0f} COP/kWh" if cfg.get('gatillo') else "sin fijar")
        + ", se usa en las demás figuras y aquí se marca como referencia)"
        + (" (" + " · ".join(f"{g:,.0f} COP/kWh" for g in (mdl.get("flex_gatillos") or ())) + ")"
           if mdl.get("flex_gatillos") else "") + " · "
        + (str(mdl.get("flex_nota", ""))[:80] or "8 OPF resueltos (4 flexibilidades × 2 gatillos)"))}


def fig_nodal_beneficios(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Marcador del efecto positivo de liquidar por nodo, con las cifras del propio escenario."""
    filas: list[dict[str, Any]] = []

    def _add(nombre: str, antes: Any, despues: Any, unidad: str, mejora: str, nota: str) -> None:
        try:
            A, D = float(antes), float(despues)
        except (TypeError, ValueError):
            return
        if not (np.isfinite(A) and np.isfinite(D)) or (abs(A) < 1e-9 and abs(D) < 1e-9):
            return
        filas.append(dict(beneficio=nombre, antes=A, despues=D, unidad=unidad,
                          pct=(100.0 * (D - A) / abs(A) if abs(A) > 1e-9 else np.nan),
                          sentido=("positivo" if ((D < A) if mejora == "down" else (D > A))
                                   else "a revisar"),
                          nota=nota))

    kp = mdl.get("kpis")
    if isinstance(kp, pd.DataFrame) and not kp.empty and "regla" in kp.columns:
        k = kp.set_index("regla")
        if "uninodal" in k.index and "nodal" in k.index:
            u, n = k.loc["uninodal"], k.loc["nodal"]
            _add("Déficit de liquidación (pago − ingreso)", abs(u["desbalance_COP"]) / 1e6,
                 abs(n["desbalance_COP"]) / 1e6, "M COP/h", "down",
                 "bajo uninodal el desbalance lo paga el operador como cargo sistémico; bajo nodal se "
                 "reparte entre los nodos que lo causan")
            _add("Renta de congestión identificada y repartible",
                 float(u["renta_congestion_COP"]) / 1e6, float(n["renta_congestion_COP"]) / 1e6,
                 "M COP/h", "up",
                 "no es un ahorro: es dinero que hoy existe pero no tiene dueño; la señal lo asigna")
            _add("Número de precios distintos del sistema", float(u["n_nodos_diferentes"]),
                 float(n["n_nodos_diferentes"]), "precios", "up",
                 "cada precio extra es información que la inversión puede usar")
    fx = mdl.get("flex")
    if isinstance(fx, pd.DataFrame) and len(fx) >= 2:
        c0 = fx[fx["flex_pct"] == fx["flex_pct"].min()].iloc[0]
        c1 = fx[fx["flex_pct"] == fx["flex_pct"].max()].iloc[0]
        _add("Horas congestionadas (con la demanda desconectable del escenario)",
             float(c0["horas_congestion"]), float(c1["horas_congestion"]), "h/día", "down",
             f"con {float(c1['flex_pct']):.0f} % de demanda interrumpible al gatillo del escenario")
        _add("Violación máxima de flujo", float(c0["viol_MW"]), float(c1["viol_MW"]), "MW", "down",
             "la flexibilidad de demanda sustituye refuerzo de red en las horas de pico")
        _add("CO₂ del día tipo", float(c0["co2_t"]), float(c1["co2_t"]), "t/día", "down",
             "al cortar las horas más caras se desalojan primero los bloques térmicos marginales")
        _add("Precio medio del sistema", float(c0["lam"]), float(c1["lam"]), "COP/kWh", "down",
             f"a cambio de {float(c1['corte_GWh']):,.1f} GWh/día no servidos (el costo del mecanismo)")
    ap = mdl.get("ap")
    if isinstance(ap, dict):
        _add("Mark-up aprendido por los agentes", ap.get("mark_up_uninodal"), ap.get("mark_up_nodal"),
             "mark-up medio", "down",
             "el mismo juego repetido bajo cada regla: bajo nodal el desvío sólo paga donde se es marginal")
    em = mdl.get("ems")
    if isinstance(em, dict) and em.get("uso"):
        _add("Valor del almacenamiento (arbitraje de 24 h)",
             float(em.get("valor_uninodal_post_COP", 0.0)) / 1e6,
             float(em.get("valor_nodal_post_COP", 0.0)) / 1e6, "M COP/día", "up",
             "con precios por nodo el activo puede elegir dónde y cuándo cargarse")
    sn = mdl.get("sens")
    if isinstance(sn, pd.DataFrame) and len(sn) >= 2:
        s0, s1 = sn.iloc[0], sn.iloc[-1]
        _add("Dispersión de precios con la penetración variable máxima", float(s0["dispersion"]),
             float(s1["dispersion"]), "COP/kWh", "up",
             "más variabilidad exige más señal locacional: es el argumento que sostiene el objetivo d")
        _add("Recorte renovable en el barrido de penetración", float(s0["recorte_GWh"]),
             float(s1["recorte_GWh"]), "GWh", "down",
             "informativo: la penetración variable añade recorte aunque el precio medio baje")
        _add("Horas congestionadas en el barrido de penetración", float(s0["horas_congestion"]),
             float(s1["horas_congestion"]), "h", "down",
             f"de +{float(s0['solar_pct']):.0f}/{float(s0['eolica_pct']):.0f} % a "
             f"+{float(s1['solar_pct']):.0f}/{float(s1['eolica_pct']):.0f} % de CEN variable")
    if not filas:
        return None
    df = pd.DataFrame(filas)
    if df["pct"].isna().all():
        return None
    fig = _fig(ctx, 470)
    iy = np.where(df["sentido"] == "positivo", VERDE, NARANJA)
    fig.add_trace(go.Bar(y=df["beneficio"], x=np.nan_to_num(df["pct"], nan=0.0), orientation="h",
                         marker_color=list(iy),
                         customdata=np.c_[df["antes"], df["despues"], df["unidad"], df["nota"]],
                         hovertemplate=("<b>%{y}</b><br>antes %{customdata[0]:,.2f} · después "
                                        "%{customdata[1]:,.2f} %{customdata[2]}<br>%{customdata[3]}"
                                        "<extra></extra>"),
                         name="cambio % frente a la línea de fondo"))
    fig.add_vline(x=0.0, line=dict(color=GRIS, width=1))
    for i, (v, a_, d_) in enumerate(zip(np.nan_to_num(df["pct"], nan=0.0), df["antes"], df["despues"])):
        fig.add_annotation(x=v, y=i, showarrow=False, text=f"{a_:,.1f} → {d_:,.1f}",
                           font=dict(size=9.5, color="#333"),
                           xanchor=("left" if v >= 0 else "right"), xshift=(6 if v >= 0 else -6))
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      xaxis_title="cambio respecto de la regla/escenario de referencia (%)",
                      legend=dict(orientation="h", y=1.16, x=0),
                      title=dict(text=("Beneficios netos de liquidar por nodo, en el escenario activo "
                                        "(verde = mejora · ámbar = a revisar)"), font_size=13.5))
    st.dataframe(df[["beneficio", "antes", "despues", "unidad", "pct", "sentido", "nota"]]
                 .round(3), use_container_width=True, hide_index=True, height=min(420, 40 + 32 * len(df)))
    n_pos = int((df["sentido"] == "positivo").sum())
    return {"fig": fig, "leyenda": _leyenda(
        "Marcador de efectos positivos: qué gana el SIN con precios nodales (objetivos a-c)",
        "Cada barra es un cambio porcentual medido **en el mismo escenario activo** entre la "
        "referencia (uninodal, o el escenario sin el mecanismo) y el caso nodal/con el mecanismo. El "
        "texto sobre la barra trae el antes → después en su unidad; la tabla repite las cifras con la "
        "nota de interpretación.",
        f"{n_pos} de {len(df)} marcadores salen a favor. La lectura que sostiene este marcador no es que el "
        "precio suba o baje: es que la señal **asigna** — el déficit de liquidación deja de ser un "
        "cargo sistémico sin dueño, la renta de congestión pasa a estar identificada (y por tanto "
        "negociable como garantía de refuerzo), el mark-up de equilibrio cae cuando el agente ya no "
        "cobra su desvío en todo el sistema, y el almacenamiento solo puede valorarse si existe "
        "gradiente locacional. Dos filas crecen a propósito: la renta identificada y el número de "
        "precios; son información, no costo.",
        "Apague la fase ENSO y el marcador se encoge: en régimen húmedo la red no satura, así que la "
        "señal vale poco. En El Niño (o con `estres` bajo) las mismas filas se disparan — que es justo "
        "el argumento de que el beneficio de la nodalidad es **state-contingent**: no se paga en un "
        "día plano, se cobra en el día de escasez.",
        "kpis_reglas() + barrido_flexibilidad() + aprendizaje_regla() + ems_almacenamiento() + "
        "sensibilidad_penetracion(), todo con la red y la oferta del escenario activo")}


# ------------------ 10.12 · segunda etapa estocástica (OPF por escenario) ----

def despacho_estocastico(red: dict, res_hn: dict[str, Any], esc: dict[str, Any],
                         nivel_cop_kwh, *, kappa: float = 0.5, perdas_iter: int = 3,
                         flex_frac: float = 0.0, gatillo_cop: float = 0.0,
                         alm: dict | None = None) -> dict[str, Any]:
    """Resuelve el OPF **escenario a escenario** sobre los días típicos y mide el valor de la señal.

    Es la segunda etapa que faltaba: hasta ahora los escenarios alimentaban perfiles y momentos, pero
    el precio se leía del día medio. Aquí cada día típico se despacha de verdad (su propio OPF con
    pérdidas y congestión), lo que permite reportar la distribución del precio por nodo, la probabilidad
    de escasez, la cola (CVaR) y el costo de *ignorar* la incertidumbre:

    * **WS** (wait-and-see) = costo esperado usando el despacho óptimo de cada escenario.
    * **HN** (here-and-now) = costo esperado congelando el despacho del escenario medio y pagando el
      desbalance de cada escenario al precio de escasez de la hora (recourse simplificado: no hay
      compromiso unidad a unidad, así que es una cota *optimista* del castigo por ignorar la curva).
    * **VSS = HN − WS** (≥ 0 por construcción, salvo ruido numérico).

    `esc` llega de `reduce_escenarios` con arrays (k, n, H); `opf_nodal` espera (H, n), de ahí los `.T`.
    """
    dem = np.asarray(esc["demanda_MW"], dtype=float)                       # (k, n, H)
    if dem.ndim != 3 or dem.shape[0] == 0:
        raise ValueError("el set reducido de escenarios no trae demanda (k, n, H)")
    sol = np.asarray(esc.get("solar_MW", np.zeros_like(dem)), dtype=float)
    eol = np.asarray(esc.get("eolica_MW", np.zeros_like(dem)), dtype=float)
    k, n, H = int(dem.shape[0]), int(dem.shape[1]), int(dem.shape[2])
    w = np.asarray(esc.get("pesos", np.full(k, 1.0 / max(k, 1))), dtype=float).ravel()[:k]
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0 / max(k, 1))
    w = w / max(float(w.sum()), 1e-12)
    wd = np.asarray(red["peso_demanda"], dtype=float).ravel()[:n]
    wd = wd / max(float(wd.sum()), 1e-12)
    nivel = np.asarray(nivel_cop_kwh, dtype=float).ravel()
    if nivel.size == 1:
        nivel = np.full(H, float(nivel[0]))
    elif nivel.size != H:
        nivel = np.resize(nivel, H)

    pre = np.zeros((k, H))
    pre_nod = np.zeros((k, n))
    dis = np.zeros((k, H))
    costo = np.zeros(k)
    recorte = np.zeros(k)
    corte = np.zeros(k)
    cong = np.zeros(k)
    co2 = np.zeros(k)
    viol = np.zeros(k)
    conv = np.ones(k, dtype=bool)
    for j in range(k):
        r = opf_nodal(red, demanda=dem[j].T[:H], solar=sol[j].T[:H], eolica=eol[j].T[:H],
                      nivel_cop_kwh=nivel[:H], kappa_hibrido=kappa, perdas_iter=perdas_iter,
                      flex_frac=flex_frac, flex_gatillo_cop=float(gatillo_cop), almacenamiento=alm)
        pd_ = np.asarray(r["p_dem"], dtype=float).reshape(H, -1)[:, :n]
        pre[j] = pd_ @ wd
        pre_nod[j] = pd_.mean(axis=0)
        dis[j] = np.asarray(r["dispersion"], dtype=float).ravel()[:H]
        costo[j] = float(np.mean(np.asarray(r["costo"], dtype=float)))
        recorte[j] = float(np.sum(np.asarray(r["recorte_MWh"], dtype=float)))
        corte[j] = float(np.sum(np.asarray(r["corte"], dtype=float)))
        cong[j] = float(r["horas_congestion"])
        co2[j] = float(np.sum(r["co2_t"]))
        viol[j] = float(r["max_violacion"])
        conv[j] = bool(r["converged"])

    # ---- acá-y-ahora: congelar el despacho del escenario medio y pagar el desbalance -------------
    hn_costo = float("nan")
    base_costo = float("nan")
    try:
        of = res_hn["oferta"]
        q = np.asarray(res_hn["P_bloque"], dtype=float)                     # (H, G) del día medio
        c1 = np.asarray(of["c1"], dtype=float)
        c2 = np.asarray(of["c2"], dtype=float)
        pmax = np.asarray(of["pmax"], dtype=float).ravel()
        if c1.ndim == 1:
            c1 = np.broadcast_to(c1[None, :], q.shape)
            c2 = np.broadcast_to(c2[None, :], q.shape)
        q = np.clip(q, 0.0, pmax[None, :q.shape[1]])
        var_costo_h = 1000.0 * np.sum(c1 * q + c2 * q ** 2, axis=1)          # COP/h
        base_costo = float(np.mean(var_costo_h[:H]))
        gen_hn = np.asarray(q, dtype=float).sum(axis=1)[:H]
        techo = float(np.asarray(res_hn.get("techo", 0.0), dtype=float).ravel()[0]) or 2000.0
        castigo = np.zeros(k)
        for j in range(k):
            d_s = np.asarray(dem[j], dtype=float).sum(axis=0)[:H]            # MW del sistema por hora
            ren_s = (np.asarray(sol[j], dtype=float) + np.asarray(eol[j], dtype=float)).sum(axis=0)[:H]
            bal = d_s - gen_hn - ren_s                        # MW sin servir (+); pérdidas excluidas
            castigo[j] = float(np.mean(1000.0 * np.maximum(bal, 0.0) * techo))
        hn_costo = base_costo + float(np.sum(w * castigo))
    except Exception:                                                        # noqa: BLE001
        hn_costo = base_costo = float("nan")

    ws = float(np.sum(w * costo))
    banda = np.percentile(pre, [5.0, 25.0, 50.0, 75.0, 95.0], axis=0)
    esc_real = np.asarray(res_hn["p_dem"], dtype=float).reshape(H, -1)[:, :n] @ wd
    umbral = float(np.percentile(esc_real, 95.0)) if esc_real.size else float("nan")
    sobre = (pre > umbral).astype(float) if np.isfinite(umbral) else np.zeros_like(pre)
    p_esc = (sobre * w[:, None]).sum(axis=0)
    cola = np.sort(pre, axis=0)[: max(1, int(np.ceil(0.05 * k))), :].mean(axis=0)
    por_esc = pd.DataFrame(dict(
        escenario=np.arange(1, k + 1), peso=w, precio_medio=pre.mean(axis=1),
        dispersion_media=dis.mean(axis=1), costo_MCop_h=costo / 1e6,
        recorte_GWh=recorte / 1000.0, corte_GWh=corte / 1000.0, horas_congestion=cong,
        viol_MW=viol, co2_t=co2, converge=conv))
    nod = pd.DataFrame(dict(
        nodo=list(red["nodos"])[:n], media=pre_nod.mean(axis=0), sd=pre_nod.std(axis=0),
        p05=np.percentile(pre_nod, 5.0, axis=0), p50=np.percentile(pre_nod, 50.0, axis=0),
        p95=np.percentile(pre_nod, 95.0, axis=0),
        spread=np.percentile(pre_nod, 95.0, axis=0) - np.percentile(pre_nod, 5.0, axis=0)))
    return dict(k=k, H=H, n=n, precios=pre, precios_nodo=pre_nod,
                bandas=dict(p05=banda[0], p25=banda[1], p50=banda[2], p75=banda[3], p95=banda[4]),
                p_escasez=p_esc, cvar_cola=cola, umbral_real=umbral, por_escenario=por_esc, nodo=nod,
                ws_COP_h=ws, hn_COP_h=hn_costo, hn_costo_base=base_costo,
                vss_pct=(100.0 * (hn_costo - ws) / abs(ws)
                         if (np.isfinite(hn_costo) and abs(ws) > 1e-9) else float("nan")),
                dispersion_esperada=float(np.sum(w * dis.mean(axis=1))),
                costo_medio_escenario=float(np.mean(costo)))



def fig_nodal_esto(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Diagnóstico de la capa estocástica: escenarios despachados, bandas y valor de la señal."""
    es = mdl.get("est")
    if not isinstance(es, dict) or not es:
        return None
    fig = _fig(ctx, 520)
    bd = es["bandas"]
    h = np.arange(1, int(es["H"]) + 1)
    fig.add_trace(go.Scatter(x=h, y=bd["p95"], name="p95 del índice de precio", mode="lines",
                            line=dict(width=0.8, color=ROJO), opacity=0.65, xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=h, y=bd["p05"], name="p05", mode="lines", fill="tonexty",
                            fillcolor="rgba(198,40,40,0.10)", line=dict(width=0.8, color=ROJO),
                            opacity=0.65, showlegend=False, xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=h, y=bd["p50"], name="mediana de escenarios", line=dict(color=AZUL, width=2.2),
                             xaxis="x", yaxis="y"))
    real = np.asarray(mdl["res"]["p_dem"], dtype=float) @ (
        np.asarray(mdl["red"]["peso_demanda"], dtype=float)
        / max(float(np.sum(np.asarray(mdl["red"]["peso_demanda"], dtype=float))), 1e-12))
    fig.add_trace(go.Scatter(x=h[: len(real)], y=real[: len(h)], name="día medio real (XM)",
                             line=dict(color="#263238", width=1.4, dash="dot"), xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=h, y=np.asarray(es["p_escasez"], dtype=float) * 100.0,
                             name="P(precio > p95 real) (%)", mode="lines+markers",
                             line=dict(color=AMBAR, width=1.8, dash="dashdot"), yaxis="y2",
                             xaxis="x"))
    fig.update_layout(xaxis=dict(title="Hora", dtick=3),
                      yaxis_title="COP/kWh (banda p05-p95 y mediana)",
                      yaxis2=dict(title="% de escenarios sobre el p95 real", overlaying="y", side="right",
                                  showgrid=False, range=[0, 105]),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    st.markdown("**Tablas: escenario por escenario y nodo por nodo**")
    with st.container(border=True):
        cc1, cc2 = st.columns(2)
        with cc1:
            st.dataframe(es["por_escenario"].round(2), use_container_width=True, hide_index=True,
                         height=min(360, 90 + 32 * len(es["por_escenario"])))
            st.caption("`precio_medio` y `dispersion_media` por día típico; `peso` es el de k-medias. "
                       "Si dos escenarios pesan poco y se ven idénticos, suba `Escenarios típicos`.")
        with cc2:
            st.dataframe(es["nodo"].round(1), use_container_width=True, hide_index=True,
                         height=min(360, 90 + 32 * len(es["nodo"])))
            st.caption("Distribución del LMP por nodo sobre los escenarios despachados: el `spread` "
                       "p05→p95 es el riesgo de precio **locacional**, que el uninodal no muestra.")
    vs = float(es.get("vss_pct", float("nan")))
    st.markdown(
        f"**Valor de tratar la incertidumbre explícitamente.** WS (despachar cada escenario óptimo) = "
        f"{float(es['ws_COP_h'])/1e6:,.1f} M COP/h; HN (congelar el plan del escenario medio y pagar el "
        f"desbalance al precio de escasez) = {float(es['hn_COP_h'])/1e6:,.1f} M COP/h → "
        f"**VSS = {vs:+.2f} %**. Probabilidad media de que el precio supere el p95 real: "
        f"{100.0 * float(np.mean(es['p_escasez'])):.1f} %, con cola (CVaR del 5 % inferior de los "
        f"escenarios) en {float(np.mean(es['cvar_cola'])):,.0f} COP/kWh.")
    mom = mdl.get("esc_mom")
    if isinstance(mom, pd.DataFrame) and not mom.empty:
        st.markdown("**Calibración del generador contra la serie real de la ventana** "
                    "(si los momentos no empatan, el abanico está mal ajustado, no el OPF):")
        st.dataframe(mom[["serie", "real_media", "sim_media", "Δ_media_pct", "real_sd", "sim_sd",
                          "real_cv", "sim_cv", "real_rho1", "sim_rho1"]].round(3),
                     use_container_width=True, hide_index=True, height=170)
        st.caption("ρ1 es la autocorrelación a un paso: mide si la serie conserva su nivel de una hora a "
                   "la siguiente. El generador la reproduce en demanda y solar; en eólica XM publica un "
                   "promedio nacional mucho más liso que una trayectoria de viento, de ahí la "
                   "corrección de varianza que sigue.")
    cv = mdl.get("esc_cv")
    if isinstance(cv, pd.DataFrame) and not cv.empty:
        st.markdown(f"**Emparejamiento de varianza aplicado a los {int(es['k'])} días típicos** "
                    "(`x' = μ + α·(x − μ)`, α = CV real / CV simulado, por nodo y serie; la media y las "
                    "correlaciones no se tocan):")
        st.dataframe(cv.groupby("serie", as_index=False)[["cv_real", "cv_antes", "cv_ahora", "alfa"]]
                     .mean().round(3), use_container_width=True, hide_index=True, height=140)
    return {"fig": fig, "leyenda": _leyenda(
        "La capa estocástica, ahora con el despacho por escenario",
        f"{int(es['k'])} días típicos (reducidos de {int(mdl.get('esc_n', 0))} trayectorias) despachados "
        "cada uno con su propio OPF con pérdidas y congestión. La banda es el p05-p95 del índice de "
        "precio de sistema y la línea ámbar, la probabilidad de que el precio supere el percentil 95 del "
        "día real de XM. Al lado, la tabla por escenario y la distribución del LMP por nodo.",
        f"El costo de ignorar la incertidumbre (VSS) es {vs:+.2f} % sobre el costo esperado. Con topología "
        f"`{mdl['red']['granularidad']}` y estrés {float(cfg['estres']):.2f}, la dispersión esperada entre "
        f"nodos es {float(es['dispersion_esperada']):,.1f} COP/kWh: eso, no el precio medio, es lo que el "
        "uninodal esconde. La probabilidad media de escasez "
        f"({100.0 * float(np.mean(es['p_escasez'])):.1f} %) es el número con el que se dimensiona una "
        "cobertura: dice en qué horas el mercado se sale de la banda histórica.",
        "El abanico se genera con la fase ENSO activa si el escenario la pide: la log-normal solar engorda "
        "su media en meses secos y la Weibull eólica del Caribe se corre a la izquierda, así que la banda "
        "y la probabilidad de escasez suben *juntas* — que es el efecto que el objetivo c quiere medir.",
        "gen_escenarios() → reduce_escenarios() (k-medias con pesos) → emparejar_cv() → "
        "despacho_estocastico(), el mismo OPF del módulo aplicado a cada escenario. HN usa el precio "
        "techo del propio OPF como penalización de desbalance; el test KS de la figura de calibración "
        "sigue evaluando el generador crudo, así que la banda corregida y el KS no se contradicen")}



# ------------------ 10.13 · refuerzos hipotéticos de la red ------------------

#: Costo de obra asumido, en M COP por MW de capacidad nueva de línea (con su subestación).
#: Es un **supuesto de orden de magnitud** —no una cifra de UPME, ISA ni del PER— que sirve para
#: rankear y para el periodo de recuperación indicativo. Se declara en la propia figura.
COSTO_REFUERZO_MCOP_POR_MW = 0.32

#: Obras que no existen en `RAMAS_BASE` y que sí cierran anillos (se proponen como rama nueva).
NODAL_OBRAS_ESTRUCTURALES: tuple[tuple[str, str, float, str], ...] = (
    ("ORI", "SUR", 700.0, "Yopal–Paicol por el piedemonte: cierra el anillo oriental y saca a Yopal "
                          "de su radialidad"),
    ("CAR", "OCC", 900.0, "Costa–Valle por el norte: ruta alternativa del carbón del Cerrejón al Valle "
                          "sin pasar por el corredor Centro"),
    ("GUJ", "NOR", 800.0, "Guajira–Nordeste: evacuación larga de Termopalo de Neta y Cyclen hacia los "
                           "ríos del nordeste"),
)


def red_con_refuerzo(red: dict, ref: dict) -> dict | None:
    """Copia de la red con un refuerzo aplicado y su PTDF recalculada.

    `modo="paralela"` suma capacidad al tramo existente y le baja la reactancia (segundo circuito en
    paralelo ⇒ `x/(1+f)`); `modo="nueva"` crea la rama con `x_pu`.
    """
    try:
        ram = red["ramas"].copy().reset_index(drop=True)
    except Exception:                                                # noqa: BLE001
        return None
    de, ha = str(ref["de"]), str(ref["a"])
    if de not in red["nodos"] or ha not in red["nodos"]:
        return None
    m = ((ram["de"].astype(str) == de) & (ram["a"].astype(str) == ha)) | \
        ((ram["de"].astype(str) == ha) & (ram["a"].astype(str) == de))
    nuevo = dict(red)
    if bool(m.any()) and ref.get("modo") != "nueva":
        k = int(np.argmax(m.to_numpy()))
        f = float(ref["MW"]) / max(float(ram.loc[k, "MW_max"]), 1e-9)
        nuevo["MW_antes"] = float(ram.loc[k, "MW_max"])
        ram.loc[k, "MW_max"] = float(ram.loc[k, "MW_max"]) + float(ref["MW"])
        ram.loc[k, "x_pu"] = float(ram.loc[k, "x_pu"]) / (1.0 + f)
        ram.loc[k, "nombre"] = f"{ram.loc[k, 'nombre']} +{float(ref['MW']):,.0f} MW"
    elif bool(m.any()):
        k = int(np.argmax(m.to_numpy()))
        nuevo["MW_antes"] = float(ram.loc[k, "MW_max"])
        ram.loc[k, "MW_max"] = float(ref["MW"])
        ram.loc[k, "nombre"] = f"{ram.loc[k, 'nombre']} (recapacitado)"
    else:
        nuevo["MW_antes"] = 0.0
        ram = pd.concat([ram, pd.DataFrame([dict(
            de=de, a=ha, x_pu=float(ref.get("x_pu", 0.0300)), r_pu=0.0005,
            MW_max=float(ref["MW"]), nombre=f"nueva {de}–{ha}: {ref.get('nota', '')[:60]}")])],
            ignore_index=True)
    try:
        sf, r_pu = matriz_ptdf(list(red["nodos"]), ram)
    except Exception:                                                # noqa: BLE001
        return None
    nuevo["ramas"] = ram
    nuevo["SF"] = sf
    nuevo["r_pu"] = r_pu
    nuevo["refuerzo"] = str(ref.get("id", ""))
    return nuevo


def refuerzos_candidatos(red: dict, res: dict, *, top: int = 3, pasos=(0.35, 0.75),
                         estructurales: bool = True) -> list[dict]:
    """El conjunto de refuerzos que se va a evaluar, generado desde **la solución**, no de un catálogo.

    Toma las `top` ramas con mayor uso máximo del día (que son las que de verdad aprietan el precio en
    esta topología y esta ventana) y propone dos niveles de capacidad extra en cada una; además suma las
    obras nuevas de `NODAL_OBRAS_ESTRUCTURALES` que cierran anillos. Así el ranking responde a la pregunta
    del objetivo b —¿qué inversión en red cambia la señal nodal?— en lugar de listar siempre el mismo
    tramo simbólico.
    """
    ram = red["ramas"].copy().reset_index(drop=True)
    uso = np.asarray(res["uso_rama"], dtype=float)
    umax = np.nanmax(uso, axis=0) if uso.ndim > 1 else np.zeros(len(ram))
    umax = np.nan_to_num(umax, nan=0.0)
    orden = [int(k) for k in np.argsort(-umax)[: max(int(top), 0)]]
    out: list[dict] = []
    for k in orden:
        r = ram.iloc[k]
        for f in pasos:
            mw = float(r["MW_max"]) * float(f)
            out.append(dict(id=f"{r['de']}→{r['a']} +{100*float(f):.0f} %", de=str(r["de"]),
                            a=str(r["a"]), modo="paralela", MW=round(mw, 0),
                            inversion_MCOP=round(mw * COSTO_REFUERZO_MCOP_POR_MW * 0.75, 0),
                            nota=f"segundo circuito en {r['nombre']} (uso máx. actual "
                                 f"{100.0*float(umax[k]):.0f} %)", ramo_i=k, uso_antes=float(umax[k])))
    if estructurales:
        for de, ha, mw, nb in NODAL_OBRAS_ESTRUCTURALES:
            if de not in red["nodos"] or ha not in red["nodos"]:
                continue
            existe = bool((((ram["de"].astype(str) == de) & (ram["a"].astype(str) == ha)) |
                           ((ram["de"].astype(str) == ha) & (ram["a"].astype(str) == de))).any())
            if existe:
                continue
            out.append(dict(id=f"nueva {de}→{ha}", de=de, a=ha, modo="nueva", MW=float(mw), x_pu=0.0300,
                            inversion_MCOP=round(float(mw) * COSTO_REFUERZO_MCOP_POR_MW, 0), nota=nb,
                            ramo_i=-1, uso_antes=0.0))
    return out


def evalua_refuerzos(cfg: dict, mdl: dict, *, top: int = 3, pasos=(0.35, 0.75)) -> pd.DataFrame:
    """Un OPF por refuerzo candidato (y uno con el paquete completo) contra la red sin reforzar."""
    red, res = mdl["red"], mdl["res"]
    dem, sol, eol = mdl["perfiles_nodales"]
    base_uso = np.asarray(res["uso_rama"], dtype=float)
    antes = float(np.nanmax(base_uso)) if base_uso.size else 0.0
    lam_base = float(np.mean(res["p_dem"]))
    costo_base = float(np.mean(res["costo"]))
    viol_base = float(np.max(res["violacion"]))
    filas = [dict(refuerzo="— red sin reforzar", tramo=f"{len(red['nodos'])} nodos · {len(red['ramas'])} ramas",
                  inversion_MCOP=0.0, MW_antes=float("nan"), MW_ahora=float("nan"),
                  lambda_COP_kWh=lam_base, d_lambda_COP_kWh=0.0,
                  horas_congestion=int(res["horas_congestion"]), viol_MW=viol_base,
                  spread_COP_kWh=float(np.mean(res["dispersion"])),
                  uso_max_pct=100.0 * antes, corte_GWh=float(np.sum(res["corte"]) / 1000.0),
                  recorte_GWh=float(np.sum(res["recorte_MWh"]) / 1000.0),
                  co2_t=float(np.sum(res["co2_t"])),
                  renta_congestion_MCOP_h=float(np.mean(res["renta_congestion"]) / 1e6),
                  costo_MCOP_h=float(np.mean(res["costo"]) / 1e6),
                  ahorro_MCOP_dia=0.0, payback_anios=float("nan"), converge=bool(res["converged"]),
                  nota="topología base, sin obras", nota_error="",
                  ramas_n=int(len(red["ramas"])))]
    casos = refuerzos_candidatos(red, res, top=top, pasos=pasos)
    if casos:
        casos.append(dict(id="🧺 paquete completo", de=str(casos[0]["de"]), a=str(casos[0]["a"]),
                          modo="canasta", MW=0.0,
                          inversion_MCOP=float(sum(float(c["inversion_MCOP"]) for c in casos)),
                          nota="todos los candidatos anteriores a la vez: dice si el último todavía "
                               "compra algo", ramo_i=-1, uso_antes=0.0))
    for ref in casos:
        try:
            r2 = dict(red)
            if ref["modo"] == "canasta":
                for sub in [c for c in casos if c.get("modo") != "canasta"]:
                    rr = red_con_refuerzo(r2, sub)
                    if rr is not None:
                        r2 = rr
            else:
                rr = red_con_refuerzo(red, ref)
                if rr is None:
                    raise RuntimeError("la red con el refuerzo no es resoluble (PTDF singular)")
                r2 = rr
            od = opf_nodal(r2, demanda=dem, solar=sol, eolica=eol, nivel_cop_kwh=mdl["nivel_modelo"],
                           kappa_hibrido=float(cfg["kappa"]), flex_frac=float(cfg["flex"]) / 100.0,
                           flex_gatillo_cop=float(cfg["gatillo"]), perdas_iter=int(cfg["perdas_iter"]),
                           almacenamiento=mdl.get("almacenamiento"))
            u2 = np.asarray(od["uso_rama"], dtype=float)
            ahor_dia = float(np.mean(od["costo"]) - costo_base) * 24.0     # COP/día; negativo = ahorro
            ahorro_MCOP_dia = -ahor_dia / 1e6
            ahorro_anio = ahorro_MCOP_dia * 365.0                            # M COP/año
            inv = float(ref["inversion_MCOP"])
            filas.append(dict(
                refuerzo=str(ref["id"]),
                tramo=(f"{ref['de']}→{ref['a']} +{float(ref['MW']):,.0f} MW ({ref['modo']})"
                       if ref["modo"] != "canasta" else f"{len(casos)-1} obras a la vez"),
                inversion_MCOP=inv,
                MW_antes=float(r2.get("MW_antes", float("nan"))),
                MW_ahora=(float(np.max(r2["ramas"]["MW_max"])) if "MW_max" in r2["ramas"] else float("nan")),
                lambda_COP_kWh=float(np.mean(od["p_dem"])),
                d_lambda_COP_kWh=float(np.mean(od["p_dem"])) - lam_base,
                horas_congestion=int(od["horas_congestion"]),
                viol_MW=float(np.max(od["violacion"])),
                d_viol_MW=float(np.max(od["violacion"])) - viol_base,
                spread_COP_kWh=float(np.mean(od["dispersion"])),
                uso_max_pct=100.0 * float(np.nanmax(u2)) if u2.size else float("nan"),
                corte_GWh=float(np.sum(od["corte"]) / 1000.0),
                recorte_GWh=float(np.sum(od["recorte_MWh"]) / 1000.0),
                co2_t=float(np.sum(od["co2_t"])),
                renta_congestion_MCOP_h=float(np.mean(od["renta_congestion"]) / 1e6),
                costo_MCOP_h=float(np.mean(od["costo"]) / 1e6),
                ahorro_MCOP_dia=ahorro_MCOP_dia,
                payback_anios=(inv / ahorro_anio if ahorro_anio > 1.0 else float("nan")),
                converge=bool(od["converged"]),
                nota=str(ref.get("nota", "")),
                nota_error="", ramas_n=int(len(r2["ramas"]))))
        except Exception as exc:                                             # noqa: BLE001
            filas.append(dict(refuerzo=str(ref.get("id", "?")), tramo="no se pudo evaluar",
                              inversion_MCOP=float(ref.get("inversion_MCOP", 0.0)),
                              horas_congestion=-1, viol_MW=float("nan"), d_viol_MW=float("nan"),
                              lambda_COP_kWh=float("nan"), d_lambda_COP_kWh=float("nan"),
                              spread_COP_kWh=float("nan"), uso_max_pct=float("nan"),
                              corte_GWh=float("nan"), recorte_GWh=float("nan"), co2_t=float("nan"),
                              renta_congestion_MCOP_h=float("nan"), costo_MCOP_h=float("nan"),
                              ahorro_MCOP_dia=float("nan"), payback_anios=float("nan"),
                              converge=False, nota_error=f"{type(exc).__name__}: {str(exc)[:140]}"))
    return pd.DataFrame(filas)


def fig_nodal_refuerzos(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Tabla y barras del conjunto de refuerzos hipotéticos, cada uno con su propio OPF."""
    df = mdl.get("ref")
    if not isinstance(df, pd.DataFrame) or df.empty or len(df) < 2:
        return None
    cand = df.iloc[1:].copy().reset_index(drop=True)
    nombres = [str(v) for v in cand["refuerzo"]]
    fig = _fig(ctx, 470)
    fig.add_trace(go.Bar(x=nombres, y=np.clip(cand["horas_congestion"].to_numpy(dtype=float), 0, None),
                         name="horas congestionadas (0-24)", marker_color=ROJO, xaxis="x", yaxis="y",
                         text=[f"{int(v)}" for v in np.clip(cand["horas_congestion"], 0, None)],
                         textposition="outside",
                         hovertemplate="%{x}<br>%{y} h congestionadas<extra></extra>"))
    fig.add_trace(go.Scatter(x=nombres, y=cand["uso_max_pct"].to_numpy(dtype=float),
                             name="uso máximo de la rama crítica (%)", mode="lines+markers",
                             line=dict(color=AZUL, width=1.6, dash="dot"), marker=dict(size=8),
                             xaxis="x", yaxis="y",
                             hovertemplate="%{x}<br>uso máx. %{y:.0f} %<extra></extra>"))
    fig.add_trace(go.Bar(x=nombres, y=cand["ahorro_MCOP_dia"].to_numpy(dtype=float),
                         name="ahorro de costo (M COP/día)", marker_color=VERDE, xaxis="x2", yaxis="y2",
                         text=[f"{v:+,.0f}" for v in cand["ahorro_MCOP_dia"].fillna(0.0)],
                         textposition="outside",
                         hovertemplate="%{x}<br>%{y:+,.1f} M COP/día<extra></extra>"))
    for kk, nom_c in enumerate(nombres):
        pb = cand.loc[kk, "payback_anios"]
        txt = ("no paga" if not (isinstance(pb, float) and np.isfinite(pb)) else f"{float(pb):,.0f} a")
        fig.add_annotation(x=nom_c, y=float(np.nan_to_num(cand.loc[kk, "ahorro_MCOP_dia"])), yshift=24,
                           text=txt,
                           showarrow=False, font=dict(size=9, color=GRIS), xref="x2", yref="y2")
    fig.update_layout(grid=dict(rows=1, columns=2, pattern="independent"),
                      xaxis=dict(title=None, tickangle=-28),
                      yaxis=dict(title="horas congestionadas · uso máx. (%)", range=[0, 118]),
                      xaxis2=dict(title=None, tickangle=-28),
                      yaxis2=dict(title="ahorro de costo del día (M COP)"),
                      legend=dict(orientation="h", y=1.13, x=0), hovermode="closest",
                      annotations=[dict(
                          text="<i>ambos paneles usan la misma red y la misma curva de oferta calibrada: "
                               "lo único que cambia es la capacidad del tramo del título de la barra.</i>",
                          xref="paper", yref="paper", x=0.0, y=-0.30, showarrow=False,
                          font=dict(size=10, color=GRIS))])
    cols = ["refuerzo", "tramo", "horas_congestion", "uso_max_pct", "viol_MW", "d_viol_MW",
            "spread_COP_kWh", "d_lambda_COP_kWh", "corte_GWh", "recorte_GWh", "co2_t",
            "renta_congestion_MCOP_h", "ahorro_MCOP_dia", "inversion_MCOP", "payback_anios",
            "converge", "nota_error"]
    st.dataframe(df[[c for c in cols if c in df.columns]].round(2), use_container_width=True,
                 hide_index=True, height=min(430, 90 + 33 * len(df)))
    ok = cand[cand["ahorro_MCOP_dia"].notna()]
    mejor = ok.loc[ok["ahorro_MCOP_dia"].idxmax()] if (not ok.empty and
                                                       float(ok["ahorro_MCOP_dia"].max()) > 0) else None
    sin = df.iloc[0]
    st.markdown(
        "### Cómo se lee"
        f"\n\n1. **Primera fila = la red sin reforzar**: {int(sin['horas_congestion'])} h congestionadas, "
        f"uso máximo {float(sin['uso_max_pct']):.0f} %, violación {float(sin['viol_MW']):,.0f} MW y "
        f"λ medio {float(sin['lambda_COP_kWh']):,.1f} COP/kWh."
        "\n2. Cada candidato se arma **sobre la rama más cargada de esta misma solución** (las ramas "
        "críticas cambian si usted cambia `Capacidad de las líneas`, la topología o el escenario) y se "
        "liquida con un OPF completo: el Δ que ve es el efecto de esa obra, no una correlación."
        + (f"\n3. El mejor del set es **{mejor['refuerzo']}**: {float(mejor['ahorro_MCOP_dia']):+,.0f} "
           f"M COP/día y pasa de {int(sin['horas_congestion'])} a {int(mejor['horas_congestion'])} h "
           f"congestionadas (inversión supuesta {float(mejor['inversion_MCOP']):,.0f} M COP → "
           f"recuperación {float(mejor['payback_anios']):,.0f} años)."
           if mejor is not None else
           "\n3. Ningún candidato del set baja el costo con esta ventana y esta topología: la congestión "
           "que queda **no es removible** con estos tramos, se alivia con demanda interrumpible, con "
           "almacenamiento bien ubicado o con generación en el nodo caro."))
    st.caption(
        f"Supuestos declarados: `COSTO_REFUERZO_MCOP_POR_MW = {COSTO_REFUERZO_MCOP_POR_MW}` M COP por MW "
        "nuevo de línea (orden de magnitud de obra con subestación, no una cifra oficial del Plan de "
        "Expansión de Referencia); un segundo circuito se modela bajando la reactancia en "
        "`1/(1+f)`; el payback anualiza el ahorro del día tipo de la ventana sin tasa de descuento ni "
        "valor de confiabilidad; el refuerzo no re-optimiza dónde se construye generación ni "
        "re-compromete unidades. Es análisis de señal de precio, no un plan de obras.")
    return {"fig": fig, "leyenda": _leyenda(
        "Refuerzos hipotéticos: qué obra mueve la señal nodal",
        f"{len(cand)} candidatos (capacidad extra en las ramas más cargadas de esta solución + obras nuevas "
        "que cierran anillos), cada uno liquidado con su propio OPF con pérdidas y congestión, más el "
        "paquete completo. Izquierda: horas congestionadas y uso máximo de la rama crítica. Derecha: "
        "ahorro de costo del día con el periodo de recuperación encima de cada barra.",
        "La lectura útil no es cuál baja más el precio —con red holgada todos bajan casi nada—, sino "
        "**cuál convierte congestión en holgura por menos plata** y cuáles no. Si el paquete completo rinde "
        "menos por MW que el primer refuerzo, la conclusión es de rendimientos decrecientes de la inversión "
        "en red, y eso es exactamente la evidencia que pide el objetivo b sobre priorización de obras.",
        "Con el escenario seco (🌊 o el bloque de combinaciones) la demanda y el despacho térmico suben, las "
        "ramas críticas cambian y los mismos candidatos dan otro ranking: el refuerzo rentable en El Niño "
        "puede ser marginal en un año normal, por eso la lista se genera desde la solución y no se fija.",
        "refuerzos_candidatos() + red_con_refuerzo() (recalcula matriz_ptdf) + opf_nodal() por candidato; "
        "inversión y payback con COSTO_REFUERZO_MCOP_POR_MW como supuesto declarado")
}


def red_gran(mdl: dict) -> str:
    """Etiqueta de granularidad de la red usada en las leyendas."""
    return str(mdl["red"].get("granularidad", "?"))





def fig_nodal_penetra(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Sensibilidad del resultado a la penetración renovable (barrido de red + OPF)."""
    se = mdl.get("sens")
    if not isinstance(se, pd.DataFrame) or se.empty:
        return None
    fig = _fig(ctx, 470)
    xlab = [f"{int(a)} %S / {int(b)} %E" for a, b in zip(se["solar_pct"], se["eolica_pct"])]
    fig.add_trace(go.Scatter(x=xlab, y=np.asarray(se["lam"], dtype=float), name="λ uninodal medio",
                             line=dict(color=AZUL, width=2.4), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=xlab, y=np.asarray(se["dispersion"], dtype=float),
                             name="Dispersión nodal media (COP/kWh)", line=dict(color=ROJO, width=2.2),
                             mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Scatter(x=xlab, y=np.asarray(se["recorte_GWh"], dtype=float),
                             name="Recorte (GWh/día)", line=dict(color=AMBAR, width=1.8, dash="dot"),
                             mode="lines+markers", yaxis="y2"))
    fig.add_trace(go.Bar(x=xlab, y=np.asarray(se["horas_congestion"], dtype=float),
                         name="Horas congestionadas (de 24)", marker_color="rgba(38,50,56,0.35)", yaxis="y3"))
    fig.update_layout(xaxis_title="Penetración adicional sobre la CEN",
                      yaxis=dict(title="COP/kWh", range=[float(np.min(se["lam"])) - 25,
                                                          float(np.max(se["lam"])) + 25]),
                      yaxis2=dict(title="COP/kWh dispersión · GWh recorte", overlaying="y", side="right",
                                  showgrid=False),
                      yaxis3=dict(title="horas", overlaying="y", side="right", showgrid=False,
                                  range=[0, 26], visible=False), hovermode="x unified")
    return {"fig": fig, "leyenda": _leyenda(
        "¿Qué pasa cuando el sistema se renueva?",
        "Cada punto vuelve a construir la red (capacidad solar/eólica extra), recalibra la oferta y "
        f"resuelve el OPF con {int(cfg['n_esc_sens'])} escenarios: precio, dispersión entre nodos, "
        "recorte renovable, horas congestionadas, violación, CO₂ y renta de congestión. "
        f"Nota: {str(mdl.get('sens_nota', ''))[:60] or 'sin errores'}",
        "Si la dispersión y el recorte suben más rápido que el precio medio, el valor del precio nodal "
        "crece con la penetración: es la conclusión operativa que justifica estudiar la "
        "transición ahora y no en 2040.",
        "El barrido no usa fases ENSO explícitas, pero el mismo mecanismo explica por qué en El Niño el "
        "sol vale más: cuando la hidráulica cede, la generación variable desplaza térmica cara y el "
        "piso de precio baja, mientras la congestión de la tarde sube.",
        "objetivos a/d · sensibilidad_penetracion()")}
def fig_nodal_ior(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """IOR diario real (bolsa vs CMD), elasticidad medida y concentración por nodo."""
    ior = mdl.get("ior") or {}
    if not isinstance(ior, dict) or not ior.get("disponible"):
        return None
    tab = ior.get("tabla")
    if not isinstance(tab, pd.DataFrame) or tab.empty:
        return None
    fig = _fig(ctx, 520)
    col_p = [c for c in tab.columns if "Precio_Bolsa" in str(c) or "Precio_Ponderado" in str(c)]
    fig.add_trace(go.Scatter(x=tab["Fecha"], y=np.asarray(tab["ior"], dtype=float) * 100.0,
                             name="IOR = (P−RAC)/P (%)", mode="lines+markers",
                             line=dict(color=ROJO, width=2.4), marker=dict(size=6),
                             hovertemplate="%{x|%d/%m}<br>IOR %{y:.1f} %<extra></extra>"))
    fig.add_hline(y=float(ior["umbral_ior"]) * 100.0, line=dict(color=AMBAR, dash="dash"),
                  annotation_text=f"umbral SSPD {100*float(ior['umbral_ior']):.0f} %", annotation_font_size=10)
    fig.add_trace(go.Bar(x=tab["Fecha"], y=np.asarray(tab[col_p[0]], dtype=float), name="Precio bolsa (COP/kWh)",
                         yaxis="y2", marker_color="rgba(31,119,180,0.35)"))
    fig.add_trace(go.Scatter(x=tab["Fecha"], y=np.asarray(tab["Costo_Marginal_COP_kWh_Dia"], dtype=float),
                             name="CMD = RAC proxy (COP/kWh)", yaxis="y2",
                             line=dict(color=GRIS, width=1.6, dash="dot")))
    fig.update_layout(xaxis_title="Día de la ventana", yaxis_title="IOR (%)",
                      yaxis2=dict(title="COP/kWh", overlaying="y", side="right", showgrid=False),
                      legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    nd = ior.get("nodos_tabla")
    if isinstance(nd, pd.DataFrame) and not nd.empty:
        st.dataframe(nd, use_container_width=True, hide_index=True)
        st.caption(f"HHI nacional {float(ior.get('hhi', float('nan'))):,.0f} · "
                   f"primer agente {str(ior.get('agente_top'))} con {float(ior.get('cuota_top', float('nan'))):.1f} % "
                   f"de la CEN. Elasticidad-precio medida con efectos fijos horarios "
                   f"η = {float(ior.get('eta', float('nan'))):.3f} (cruda "
                   f"{float(ior.get('eta_bruta', float('nan'))):.3f}, n={int(ior.get('n_elasticidad', 0))}): "
                   f"{str(ior.get('regimen'))}")
    st.markdown(f"""
**Qué mide cada magnitud de esta figura (y qué puede mover usted).**

- **IOR (renta inframarginal)** — (precio de bolsa − costo de oportunidad del recurso) sobre el precio, en
%. Está medido en los datos reales de XM de la ventana, no simulado: media {float(ior['ior_medio']):,.2f} %,
p95 {float(ior['ior_p95']):,.2f} %, {int(ior.get('dias_bandera', 0))} de {int(ior.get('dias', 0))} días por
encima del umbral. Es la evidencia empírica de que el parque inframarginal cobra escasez: si el modelo
reproduce ese orden de magnitud, el recargo de «Bertrand/Cournot» no es un invento del ejercicio.
- **HHI** — suma de las cuotas al cuadrado ({int(ior['hhi']):,} sobre 10.000; máximo por nodo
{int(ior['hhi_max_nodo']):,}). Dice *dónde* se puede ejercer poder, no cuánto: un nodo con HHI bajo y ramas
saturadas también separa precios, pero por red.
- **Lerner observado** — (p − CMg)/p del día medio: {float(ior['lerner']):,.3f}. Con esta cifra se compara
el Lerner *teórico* de Cournot, no con el `Markup (%)` que usted pida en la barra lateral.
- **η (elasticidad)** — {float(ior['eta']):,.3f} corregida del ciclo diario (la cruda, sin corregir, da
{float(ior['eta_bruta']):,.3f}: es un artefacto de comparar horas distintas). Entra como denominador del
recargo de Cournot, así que es la palanca que más mueve el markup teórico.
- **Umbral IOR** — {float(ior['umbral_ior']):,.1f} COP/kWh: solo clasifica como «día con escasez» los cuyo
IOR lo supera; bajarlo aumenta los días bandera sin tocar el modelo. Es un corte declarado de esta app, no
un criterio de la CREG.

Si en lugar de cifras aparece la nota de no disponibilidad, es que la ventana no tiene
`cmd_planta`/`PrecBolsNaci` completos: el IOR se mide sobre el dato, no se estima.
""")
    return {"fig": fig, "leyenda": _leyenda(
        "Poder de mercado medido, no supuesto",
        f"IOR diario con las dos columnas de XM (`Precio_Bolsa_Dia_COP_kWh` vs "
        f"`Costo_Marginal_COP_kWh_Dia`) sobre {int(ior['dias'])} días, el umbral de 15 % con el que la "
        f"SSPD abre investigación, y la concentración por nodo calculada con `Codigo_Agente` de "
        f"`dim_plantas`.",
        f"{int(ior['dias_bandera'])} de {int(ior['dias'])} días superan el umbral (IOR medio "
        f"{100*float(ior['ior_medio']):.1f} %, máximo {100*float(ior['ior_max']):.1f} %). Con HHI "
        f"{float(ior.get('hhi', float('nan'))):,.0f} y un nodo con HHI "
        f"{float(ior.get('hhi_max_nodo', float('nan'))):,.0f}, el mercado ya está concentrado: la "
        "pregunta no es *si* hay poder de mercado, sino cuánto se puede ejercer bajo cada "
        "regla de liquidación (figura siguiente).",
        "El IOR sube en seco: la comparación por fase ENSO en `impacto` da precio ponderado "
        f"{float((ior.get('precio_pond_fases') or {}).get('El_Nino', float('nan'))):.0f} COP/kWh en El "
        f"Niño contra {float((ior.get('precio_pond_fases') or {}).get('La_Nina', float('nan'))):.0f} en "
        "La Niña — el mismo agente tiene más margen de oferta en meses secos.",
        " · IOR de bolsa/CMD + HHI real · η con fijos horarios")}


def fig_nodal_mec(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Cournot/Bertrand sobre la demanda residual + el efecto del markup en el modelo."""
    cb = mdl.get("cb")
    if not isinstance(cb, pd.DataFrame) or cb.empty:
        return None
    fig = _fig(ctx, 520)
    h = np.asarray(cb["hora"], dtype=int) + 1
    fig.add_trace(go.Scatter(x=h, y=np.asarray(cb["p_competencia"], dtype=float), name="Competencia (λ OPF)",
                             line=dict(color=GRIS, width=2.2), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=h, y=np.asarray(cb["p_bertrand"], dtype=float), name="Bertrand (= CMg)",
                             line=dict(color=VERDE, width=1.8, dash="dot"), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=h, y=np.asarray(cb["p_cournot"], dtype=float),
                             name=f"Cournot (Lerner s={100*float(np.mean(cb['s_top_pct']))/100.0:.2f}/|ε|)",
                             line=dict(color=AMBAR, width=2.2), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=h, y=np.asarray(cb["p_monopolio"], dtype=float), name="Monopolio (tope escasez)",
                             line=dict(color=ROJO, width=1.8, dash="dash"), mode="lines+markers"))
    fig.add_trace(go.Bar(x=h, y=np.asarray(cb["recargo_cournot_pct"], dtype=float),
                         name="Recargo Cournot (%)", yaxis="y2", marker_color="rgba(146,64,197,0.35)"))
    fig.data = ()
    h2 = h
    fig.add_trace(go.Scatter(x=h2, y=np.asarray(cb["mg_COP_kWh"], dtype=float),
                             name="CMg del residuo (piso teórico)",
                             line=dict(color=GRIS, width=1.6, dash="dot"), xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=h2, y=np.asarray(cb["p_competencia"], dtype=float),
                             name="λ del OPF (lo que paga hoy)", line=dict(color=AZUL, width=2.6),
                             mode="lines+markers", marker=dict(size=5), xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=h2, y=np.asarray(cb["p_bertrand"], dtype=float),
                             name="Bertrand (precio = CMg)", line=dict(color=VERDE, width=1.6, dash="dot"),
                             xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=h2, y=np.asarray(cb["p_cournot"], dtype=float),
                             name="Cournot (recargo de Lerner)", line=dict(color=AMBAR, width=2.4),
                             mode="lines+markers", marker=dict(size=5), xaxis="x", yaxis="y"))
    fig.add_trace(go.Scatter(x=h2, y=np.asarray(cb["p_monopolio"], dtype=float),
                             name="Monopolio (tope de escasez)", line=dict(color=ROJO, width=1.8, dash="dash"),
                             xaxis="x", yaxis="y"))
    fig.add_trace(go.Bar(x=h2, y=np.asarray(cb["recargo_cournot_pct"], dtype=float),
                         name="recargo Cournot sobre CMg (%)", marker_color="rgba(249,168,37,0.55)",
                         xaxis="x2", yaxis="y2"))
    fig.add_trace(go.Scatter(x=h2, y=np.asarray(cb["lerner_cournot"], dtype=float),
                             name="Lerner de Cournot", mode="lines+markers",
                             line=dict(color="#6a1b9a", width=1.8), marker=dict(size=5),
                             yaxis="y3", xaxis="x2"))
    fig.add_trace(go.Scatter(x=h2, y=np.asarray(cb["s_top_pct"], dtype=float) / 100.0,
                             name="cuota del agente top (×100 en este eje)", mode="lines",
                             line=dict(color=GRIS, width=1.2, dash="dot"), yaxis="y2", xaxis="x2"))
    fig.update_layout(
        grid=dict(rows=1, columns=2, pattern="independent"), height=520,
        xaxis=dict(title="Hora · qué precio pagaría cada régimen", dtick=3),
        yaxis=dict(title="COP/kWh"),
        xaxis2=dict(title="Hora · recargo (%) y poder de mercado", dtick=3, matches="x"),
        yaxis2=dict(title="recargo (%) · cuota ×100"),
        yaxis3=dict(title="Lerner", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.19, x=0, groupclick="toggleitem", font=dict(size=10)),
        hovermode="x unified")
    st.markdown("""
**Qué es cada régimen del panel izquierdo** —todos sobre la *misma* demanda residual (demanda menos la
variable): lo único que cambia es la conducta del despacho, no el sistema.

- **CMg del residuo (gris punteada)** — costo marginal de la última MW despachada. Piso teórico: ningún
régimen racional lo sostiene por debajo de forma prolongada.
- **λ del OPF (azul)** — lo que el mercado real pagaría hoy en este modelo. Si la azul se levanta sobre la
gris, esa diferencia **no** es conducta estratégica sino congestión o escasez de oferta: los regímenes de
esta figura no la capturan (por eso el λ del OPF puede quedar por encima de todos).
- **Bertrand (verde)** — empresas idénticas compitiendo por el despacho → precio = CMg. Es el control: si
Bertrand y Cournot coinciden, la concentración no está mordiendo el precio en esa hora.
- **Cournot (ámbar)** — cada agente decide sus MW y cobra el recargo de Lerner `s/|ε|` sobre el CMg, con la
cuota real del agente top. Es el escenario de poder de mercado.
- **Monopolio (rojo)** — el techo de la escala (precio de escasez de la hora). No es una predicción: es la
cota contra la que se comprueba si los otros dos tienen sentido.

**Panel derecho** — barras ámbar: recargo de Cournot sobre el CMg (%). Línea morada: índice de Lerner
`L = (p − CMg)/p` del equilibrio (eje derecho). Línea gris punteada: cuota del agente top ×100 para que
quepa en el mismo eje. Recargo alto **y** Lerner alto en las mismas horas → ahí se paga la concentración;
recargo alto con Lerner plano → el precio lo mueve la red, no la conducta.

**Qué mueve cada control** — `ε (elasticidad de la demanda residual)` entra al denominador del recargo de
Cournot: con |ε| < 1 el recargo teórico se dispara y se limita al precio de escasez (por eso la roja queda
plana en varias horas). `Markup (%)` y `Retiro de oferta` actúan sobre el OPF de la pestaña, no sobre este
juego: por eso esta figura y la sensibilidad al markup no cuadran al céntimo. La
`Curvatura de la oferta (c₂)` sí afecta a las dos: a más curvatura, menos salto de precio por MW retirado y
menor recargo.
""")

    res, rb = mdl["res"], mdl["res_base"]
    d_lam = 100.0 * (float(np.mean(res["lam"])) / max(float(np.mean(rb["lam"])), 1e-9) - 1.0)
    _es = mdl.get("estrato") or {}
    st.caption(f"**Escenario estratégico pedido arriba** (markup {cfg['markup_pct']:.0f} %, retiro "
               f"{cfg['wh_pct']:.0f} % modo `{cfg['wh_modo']}` aplicado a los bloques de "
               f"**{_es.get('nodo_estr', '—')}**): λ medio {float(np.mean(rb['lam'])):,.1f} → "
               f"{float(np.mean(res['lam'])):,.1f} COP/kWh ({d_lam:+.1f} %), dispersión "
               f"{float(np.mean(rb['dispersion'])):,.1f} → {float(np.mean(res['dispersion'])):,.1f} "
               f"COP/kWh, renta de congestión {float(np.sum(rb['renta_congestion']))/1e6:,.1f} → "
               f"{float(np.sum(res['renta_congestion']))/1e6:,.1f} M COP/día, CO₂ "
               f"{float(np.sum(rb['co2_t'])):,.0f} → {float(np.sum(res['co2_t'])):,.0f} t/día. El retiro "
               "en modo `capacidad` sólo mueve algo si el bloque ya estaba en su techo: si ve flechas "
               "planas cambie a `energia`, no es un fallo del modelo sino la definición de la modalidad.")
    return {"fig": fig, "leyenda": _leyenda(
        "Cuánto markup permite cada regla",
        f"Demanda residual del nodo más cargado (demanda menos variables) y los cuatro precios teóricos: "
        f"el de competencia (el OPF), Bertrand con producto homogéneo (= costo marginal), Cournot con "
        f"Lerner L = s/|ε| usando la cuota real del primer agente "
        f"({float(np.mean(cb['s_top_pct'])):.1f} %) y la demanda de la ventana, y el monopolio acotado "
        f"por el precio de escasez. ε = {float(np.mean(cb['epsilon'])):.2f}.",
        "Bajo liquidación uninodal un agente que sube la oferta en un nodo cobra ese markup *en todo el "
        "SIN*; bajo precio nodal el markup sólo se materializa donde su capacidad es marginal. Por eso "
        "la Misión MinMinas 2020 concluye que los nodales mitigan el poder de mercado — y por eso el "
        "panel superior compara los dos regímenes con el mismo markup.",
        f"En esta ventana el markup pedido ({cfg['markup_pct']:.0f} %) mueve el precio medio "
        f"{d_lam:+.1f} % y la dispersión a {float(np.mean(res['dispersion'])):,.1f} COP/kWh: en horas "
        "congestionadas el recargo se concentra en un nodo, no se reparte.",
        " · cournot_bertrand() + `markup`/`withholding` del OPF")}


def fig_nodal_var(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """GARCH(1,1) horario, VaR/ES condicional y backtest de Kupiec."""
    g = mdl.get("garch") or {}
    if not isinstance(g, dict) or not g.get("disponible"):
        return None
    bt = mdl.get("bt")
    va = mdl.get("var")
    fig = _fig(ctx, 560)
    t = np.arange(1, len(np.asarray(g["sigma"], dtype=float)) + 1)
    precio = np.asarray(mdl.get("precio_ret", []), dtype=float)
    fig.add_trace(go.Scatter(x=t, y=precio, name="Precio (serie de la ventana)",
                             line=dict(color=GRIS, width=1.2), yaxis="y"))
    if isinstance(va, pd.DataFrame) and not va.empty:
        fig.add_trace(go.Scatter(x=np.arange(1, len(va) + 1), y=np.asarray(va["var"], dtype=float),
                                 name=f"VaR {100*float(va['alfa'].iloc[0]):.1f} %",
                                 line=dict(color=ROJO, width=1.6, dash="dot"), yaxis="y"))
        fig.add_trace(go.Scatter(x=np.arange(1, len(va) + 1), y=np.asarray(va["es"], dtype=float),
                                 name="Expected shortfall", line=dict(color="#6a1b9a", width=1.2),
                                 yaxis="y", opacity=0.8))
    fig.add_trace(go.Scatter(x=t, y=np.asarray(g["sigma"], dtype=float) * float(np.mean(precio)),
                             name="σ condicional (COP/kWh)", yaxis="y2",
                             line=dict(color=AMBAR, width=1.6)))
    fig.update_layout(xaxis_title="Hora desde el inicio de la ventana", yaxis_title="COP/kWh",
                      yaxis2=dict(title="σ condicional", overlaying="y", side="right", showgrid=False),
                      hovermode="x unified")
    if isinstance(bt, pd.DataFrame) and not bt.empty:
        st.plotly_chart(go.Figure(data=[
            go.Scatter(x=np.arange(1, len(bt) + 1), y=np.asarray(bt["fallos_acum"], dtype=float),
                       name="Fallos acumulados", line=dict(color=ROJO, width=2.0)),
            go.Scatter(x=np.arange(1, len(bt) + 1), y=np.asarray(bt["esperado_acum"], dtype=float),
                       name="Esperado (α·t)", line=dict(color=GRIS, dash="dash", width=1.6)),
            go.Scatter(x=np.arange(1, len(bt) + 1), y=np.asarray(bt["esperado_acum"], dtype=float)
                       + np.sqrt(np.asarray(bt["esperado_acum"], dtype=float) * (1.0 - cfg["alfa"])),
                       name="±1σ binomial", line=dict(width=0), fill="toself",
                       fillcolor="rgba(97,97,97,0.12)")])
            .update_layout(template="plotly_white", height=240, margin=MARGEN,
                           title=dict(text=f"Backtest Kupiec · tasa {bt.attrs.get('tasa_pct', 0.0):.1f} % "
                                          f"vs nominal {100*cfg['alfa']:.0f} %", font=dict(size=12)),
                           xaxis_title="Hora", yaxis_title="fallos", showlegend=False),
            use_container_width=True, key=_clave_fig("nodal-kupiec"))
    kup = bt.attrs.get("kupiec", {}) if isinstance(bt, pd.DataFrame) else {}
    return {"fig": fig, "leyenda": _leyenda(
        "Volatilidad condicional y riesgo de mercado ",
        f"GARCH(1,1) estimado por máxima verosimilitud sobre {int(g['n'])} retornos horarios *reales* "
        f"de XM: ω={float(g['omega']):.2e}, α={float(g['alfa']):.3f}, β={float(g['beta']):.3f}, "
        f"persistencia {float(g['persistencia']):.3f} (vida media de la volatilidad "
        f"{float(g['vida_media_h']):.0f} h), log-verosimilitud {float(g['loglik']):.1f}. Arriba, la "
        "banda VaR/ES construida con σ_t; abajo, el backtest.",
        f"Kupiec: {int(kup.get('hits', 0))} fallos en {int(kup.get('n', 0))} horas "
        f"({float(kup.get('tasa', 0.0)):.1f} % contra el {100*cfg['alfa']:.0f} % nominal), LR = "
        f"{float(kup.get('lr', 0.0)):.2f}, p = {float(kup.get('p_valor', 1.0)):.3f} ⇒ "
        f"{'no se rechaza' if kup.get('ok') else 'se rechaza la calibración'}. Si se rechaza, el VaR "
        "normal subestima: la cola del precio eléctrico es más gorda que la normal y hay que pasar a "
        "t de Student o a EVT antes de poner dinero sobre ese número.",
        "La persistencia 0,98 es el número que importa para El Niño: un choque de volatilidad tarda "
        "~3 días en disiparse, así que una sequía que dura meses *encadena* picos y el VaR diario "
        "subestima el riesgo de posición larga.",
        " · garch11 (rejilla + Newton, sin scipy) · alfa " + f"{cfg['alfa']:.3f}")}


def fig_nodal_cov(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Cobertura de mínima varianza entre el precio del modelo y el futuro."""
    cv = mdl.get("cov") or {}
    if not isinstance(cv, dict) or not cv.get("disponible"):
        return None
    fr = cv.get("frente")
    fig = _fig(ctx, 430)
    if isinstance(fr, pd.DataFrame) and not fr.empty:
        x = np.asarray(fr["h"], dtype=float)
        y = np.asarray(fr.get("sd", fr.get("sd", fr.iloc[:, 1])), dtype=float)
        fig.add_trace(go.Scatter(x=x, y=y, name="σ de la posición cubierta", mode="lines",
                                line=dict(color=AZUL, width=2.4)))
        fig.add_trace(go.Scatter(x=[float(cv["h_optimo"])], y=[float(np.interp(float(cv["h_optimo"]), x, y))],
                                 mode="markers+text", name="h* = ρσ_S/σ_F",
                                 marker=dict(size=14, color=ROJO, symbol="diamond"),
                                 text=[f"h* = {float(cv['h_optimo']):.2f}"], textposition="top center"))
        fig.add_vline(x=0.0, line=dict(color=GRIS, width=1.0, dash="dot"))
    fig.update_layout(xaxis_title="Razón de cobertura h (unidades de futuro por unidad de exposición)",
                      yaxis_title="desviación estándar de la posición (COP/kWh)")
    return {"fig": fig, "leyenda": _leyenda(
        "Cuánto futuro comprar ",
        f"Correlación ρ = {float(cv['rho']):.3f} entre el retorno del precio del modelo en el nodo y el "
        f"de la serie uninodal de XM; σ_S = {float(cv['sigma_spot']):.4f}, σ_F = "
        f"{float(cv['sigma_futuro']):.4f}; óptimo h* = ρσ_S/σ_F = {float(cv['h_optimo']):.3f} "
        f"(β de hedging puro {float(cv['beta_hedging']):.3f}) sobre {int(cv['n'])} pares horarios.",
        f"Cubrir con h* reduce la varianza {float(cv['reduccion_var_pct']):.1f} % "
        f"(σ² {float(cv['var_sin']):.2e} → {float(cv['var_cubierta']):.2e}) y deja un ingreso medio de "
        f"{float(cv['media_cubierta']):.4f} por MWh. Con ρ = {float(cv['rho']):.2f} el futuro 'casi "
        "cubierto' deja un residual grande: el riesgo locacional **no** se cubre con un producto "
        "nacional, que es el argumento clásico para tener mercado nodal con derechos de "
        "congestión negociables.",
        "La ventana es seca y corta (31 días); en El Niño ρ sube pero σ_S también, así que h* no es "
        "constante: un coberturista debe re-estimar cada semana con `garch11` en lugar de fijar h.",
        " · cobertura_min_var()")}


def fig_nodal_ems(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """EMS de almacenamiento: carga/descarga/SOC sobre el precio, y el valor según la regla."""
    em = mdl.get("ems") or {}
    if not isinstance(em, dict) or not em.get("disponible", True) or "carga_MW" not in em:
        return None
    carga = np.asarray(em["carga_MW"], dtype=float)
    desca = np.asarray(em["descarga_MW"], dtype=float)
    if carga.ndim > 1:                       # (H, n_almacén) → suma de la flota por hora
        carga = carga.sum(axis=1)
        desca = desca.sum(axis=1)
    soc = np.asarray(em.get("soc", np.zeros(carga.shape[-1:])), dtype=float)
    if soc.ndim > 1:
        soc = soc.mean(axis=1)
    con, sin = em.get("precio_con") or {}, em.get("precio_sin") or {}

    def _precio(dic, clave_alt):
        if not isinstance(dic, dict):
            return np.zeros(carga.size)
        if "p_dem" in dic:
            v = np.asarray(dic["p_dem"], dtype=float)
            return v.mean(axis=1) if v.ndim > 1 else v
        return np.asarray(dic.get(clave_alt, np.zeros(carga.size)), dtype=float)
    h = np.arange(1, carga.size + 1)
    fig = _fig(ctx, 540)
    fig.add_trace(go.Scatter(x=h, y=_precio(con, "lam")[:carga.size], name="LMP medio con EMS",
                             line=dict(color=AZUL, width=2.0)))
    fig.add_trace(go.Scatter(x=h, y=_precio(sin, "lam")[:carga.size],
                             name="LMP medio sin almacenamiento",
                             line=dict(color=GRIS, width=1.6, dash="dot")))
    fig.add_trace(go.Bar(x=h, y=-carga, name="carga (MW)", marker_color="rgba(31,119,180,0.55)"))
    fig.add_trace(go.Bar(x=h, y=desca, name="descarga (MW)", marker_color="rgba(46,125,50,0.75)"))
    fig.add_trace(go.Scatter(x=h[:soc.size], y=100.0 * soc / max(float(np.max(soc)), 1e-9),
                             name="SOC (% de la energía)", yaxis="y2", line=dict(color=AMBAR, width=1.8)))
    fig.update_layout(xaxis=dict(title="Hora", dtick=2), yaxis_title="MW (− = carga) · COP/kWh",
                      yaxis2=dict(title="% SOC", overlaying="y", side="right", showgrid=False,
                                  range=[0, 110]), hovermode="x unified")
    vn = float(em.get("valor_nodal_post_COP", 0.0)) / 1e6
    vu = float(em.get("valor_uninodal_post_COP", 0.0)) / 1e6
    vnp = float(em.get("valor_nodal_pre_COP", 0.0)) / 1e6
    vup = float(em.get("valor_uninodal_pre_COP", 0.0)) / 1e6
    st.caption(
        f"**Valor del EMS (M COP/día)** — modo `{em.get('modo')}`, "
        f"{len(np.atleast_1d(em.get('potencias', [])))} "
        f"almacén(es), {int(em.get('pasadas', 0))} pasadas hasta estabilizar: uninodal {vup:,.1f} (sin "
        f"precio nodal, pre) → {vu:,.1f} (post); nodal {vnp:,.1f} → {vn:,.1f}. "
        f"Mejora por usar precio nodal: {100.0*(vn-vu)/max(abs(vu), 1e-9):+,.1f} %. "
        f"Violación de ramas {float(em.get('violacion_antes', 0.0)):,.0f} → "
        f"{float(em.get('violacion_despues', 0.0)):,.0f} MW; recorte "
        f"{float(em.get('recorte_antes_GWh', 0.0)):,.1f} → {float(em.get('recorte_despues_GWh', 0.0)):,.1f} GWh.")
    return {"fig": fig, "leyenda": _leyenda(
        "Almacenamiento con precio nodal ",
        "Programa óptimo del EMS (water-filling bajo el vector de precios, con η, Pmin de reserva y "
        "máscaras por modo) contra el precio del sistema. Carga donde el LMP es bajo —en la ventana "
        "solar de media mañana— y descarga en el pico 18-21 h; el SOC cierra el día donde empezó.",
        "La línea de texto compara el mismo activo bajo las dos reglas: la diferencia es lo que vale la "
        "**señal locacional** para un inversor de baterías. Si el valor cae con `alivio_congestion` o la "
        "violación de ramas sube (como le pasa al bombeo mal sentado), el almacenamiento no es un "
        "sustituto del refuerzo de red: aquí se ve.",
        "En El Niño la brecha pico-valle se abre (más térmica cara en la tarde), así que el arbitraje "
        "rinde más: la batería es un activo anticíclico frente a la sequía, y su valoración con precio "
        "uninodal la subestima sistemáticamente.",
        f"config_almacenamiento(bat {cfg['bateria_pct']:.1f} %, bom {cfg['bombeo_pct']:.1f} %) · "
        f"nodos {', '.join(cfg['alm_nodos'])} · CO₂ {float(em.get('co2_despues', 0.0)):,.0f} vs "
        f"{float(em.get('co2_antes', 0.0)):,.0f} t")}


def fig_nodal_ql(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Q-learning multiagente: convergencia del precio y curva de oferta aprendida."""
    q = mdl.get("ql") or {}
    if not isinstance(q, dict) or not q.get("disponible"):
        return None
    st.markdown(NODAL_REFUERZO_TEXTO)
    st.markdown(
        f"**Hiperparámetros de esta corrida:** {int(q['n_agentes'])} agentes · "
        f"{int(q['n_acciones'])} acciones por agente en {int(q['n_estados'])} estados · ε-greedy inicial "
        f"{float(q['eps_inicial']):.2f} · {int(cfg['episodios_rl'])} episodios · resolución del despacho "
        f"{float(q['resolucion_MW']):,.0f} MW · hora del juego {int(q['hora'])}. "
        f"Balance: objetivo {float(q['objetivo_MW']):,.0f} MW, el Q-learning sirvió "
        f"{float(q['suma_ql']):,.0f} MW (déficit {float(q['deficit_MW']):,.0f} MW) y el mérito-order "
        f"{float(q['suma_exact']):,.0f} MW. Precio: λ exacto {float(q['lam_exacto_COP_kWh']):,.0f} vs "
        f"λ aprendido {float(q['lam_final_COP_kWh']):,.0f} COP/kWh "
        f"({float(q['brecha_lambda_pct']):+.1f} %), con la política en la banda "
        f"[{float(q['lam_cola_min']):,.0f}, {float(q['lam_cola_max']):,.0f}] y factibilidad "
        f"{'sí' if q.get('factible') else 'no'}. Si el déficit es alto, la convergencia está todavía en la fase "
        "exploratoria: suba `Episodios del Q-learning` antes de leer la brecha como resultado.")
    his = np.asarray(q.get("historia", np.zeros((0, 4))), dtype=float)
    fig = _fig(ctx, 540)
    if his.size:
        sm = pd.Series(his[:, 1]).ewm(span=40).mean().to_numpy()
        fig.add_trace(go.Scatter(x=his[:, 0] + 1, y=sm, name="λ (media móvil 40)",
                                 line=dict(color=AZUL, width=2.2)))
        fig.add_trace(go.Scatter(x=his[:, 0] + 1, y=his[:, 1], name="λ crudo (ascenso dual)",
                                 line=dict(color="rgba(31,119,180,0.28)", width=1.0)))
        fig.add_hline(y=float(q["lam_exacto_COP_kWh"]), line=dict(color=ROJO, dash="dash"),
                      annotation_text=f"λ exacto del mérito-order {float(q['lam_exacto_COP_kWh']):,.0f}",
                      annotation_font_size=10)
        fig.add_trace(go.Scatter(x=his[:, 0] + 1, y=his[:, 2], name="Σ despacho (MW)", yaxis="y2",
                                 line=dict(color=VERDE, width=1.6)))
        fig.add_shape(type="line", xref="x", yref="y2", x0=0, x1=1, y0=float(q["objetivo_MW"]),
                      y1=float(q["objetivo_MW"]), line=dict(color=GRIS, dash="dot", width=1.2))
        fig.add_annotation(text=f"objetivo {float(q['objetivo_MW']):,.0f} MW", xref="paper", yref="y2",
                           x=0.99, y=float(q["objetivo_MW"]), xanchor="right", showarrow=False,
                           font=dict(size=10, color=GRIS))
    band = np.asarray(q["bandas"], dtype=float)
    niveles = np.asarray(q["niveles"], dtype=float)
    pmn, pmx = np.asarray(q["pmin"], dtype=float), np.asarray(q["pmax"], dtype=float)
    c1, c2 = np.asarray(q["c1"], dtype=float), np.asarray(q["c2"], dtype=float)
    lam_eje = 0.5 * (band[:-1] + band[1:])
    for j, (nm, Qm) in enumerate(list(q["Q"].items())[:4]):
        pol = np.array([niveles[int(np.argmax(np.asarray(Qm)[b]))] for b in range(len(lam_eje))])
        fig.add_trace(go.Scatter(x=lam_eje, y=pmn[j] + pol * (pmx[j] - pmn[j]),
                                 name=f"oferta aprendida {nm}",
                                 xaxis="x2", yaxis="y3", mode="lines+markers",
                                 line=dict(width=1.6), marker=dict(size=5)))
    for j in range(min(4, len(pmn))):
        pcur = np.clip((lam_eje - c1[j]) / np.maximum(2.0 * c2[j], 1e-9), pmn[j], pmx[j])
        fig.add_trace(go.Scatter(x=lam_eje, y=pcur, name=f"curva real b{j}", xaxis="x2", yaxis="y3",
                                 line=dict(width=1.4, dash="dot"), opacity=0.7))
    fig.update_layout(grid=dict(rows=2, columns=1, pattern="independent", ygap=0.16),
        xaxis=dict(title="episodio (arriba: trayectoria de λ; el ascenso dual lleva λ al "
                          "precio de mérito-order)"),
        xaxis2=dict(title="λ de la banda de estado (COP/kWh) · abajo, oferta aprendida (línea) "
                            "vs curva verdadera (punteada)"),
        yaxis=dict(title="COP/kWh"), yaxis2=dict(title="MW", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="MW del bloque"), hovermode="closest")
    return {"fig": fig, "leyenda": _leyenda(
        f"El despacho se aprende sin coordinador (hora {int(q['hora'])+1})",
        f"{q['n_agentes']} agentes con Q-learning sobre {q['n_estados']} bandas de precio y "
        f"{q['n_acciones']} niveles de potencia, {len(his)} episodios. El precio es el estado y se "
        "actualiza con el mismo ascenso dual que usa el OPF, así que el mercado cierra "
        "solo.",
        f"λ de la política {float(q['lam_politica_COP_kWh']):,.0f} vs exacto "
        f"{float(q['lam_exacto_COP_kWh']):,.0f} ({float(q['brecha_lambda_pct']):+.1f} %), costo "
        f"{float(q['costo_ql_COP_h'])/1e6:,.1f} vs {float(q['costo_exact_COP_h'])/1e6:,.1f} M COP/h "
        f"({float(q['brecha_pct']):+.2f} %), despacho {float(q['suma_ql']):,.0f}/{float(q['suma_exact']):,.0f} MW "
        f"(déficit {float(q['deficit_MW']):,.0f} MW). La banda en la que oscila λ la fija la resolución "
        f"de la rejilla ({float(q['resolucion_MW']):.0f} MW por agente), no un error del algoritmo: con "
        "acciones continuas la convergencia es exacta y el costo coincide.",
        "El resultado importa para El Niño porque el mismo mecanismo sirve para despachar con pronósticos "
        "de recurso imperfectos: un agente que aprende por refuerzo no necesita que el coordinador "
        "conozca sus curvas de costo, y eso es lo que hace viable un mercado con miles de recursos "
        "distribuidos.",
        " · qlearning_despacho() · semilla " + str(cfg["semilla"]))}


def fig_nodal_ap(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Mark-up aprendido por regla: lo que el aprendizaje por refuerzo cambia en el número."""
    ap = mdl.get("ap") or {}
    if not isinstance(ap, dict) or not ap.get("reglas"):
        return None
    reglas = ap["reglas"]
    st.markdown(
        f"**Aprendizaje por regla — hiperparámetros:** acciones de mark-up "
        f"{', '.join(f'{100.0*float(a):.0f} %' for a in np.atleast_1d(np.asarray(ap.get('acciones', []), dtype=float)))}"
        f" · ε-greedy 0,30 · {int(cfg['episodios_ap'])} episodios por regla · "
        f"{len(list(cfg['ap_horas']))} hora(s) de juego ({', '.join(str(int(v)) for v in cfg['ap_horas'])}) · "
        f"{len(reglas)} reglas de liquidación comparadas: {', '.join(reglas)}. "
        "La política aprendida es **el mismo OPF re-resuelto por episodio**: lo único que aprende el agente "
        "es cuánto mark-up ponerle a su curva de oferta. Con pocos episodios la traza no converge y el "
        "mark-up de equilibrio sale sesgado a la baja; con demasiados, se planta en la última acción de la "
        "rejilla. Si `mark_up_medio` sale 0,00 en una regla, no es que el modelo niegue el poder de "
        "mercado: es que en esa regla el agente ya no cobra el desvío en todo el sistema y no le conviene "
        "desviarse.")
    fig = _fig(ctx, 460)
    ks = list(reglas.keys())
    fig.add_trace(go.Bar(x=ks, y=[100.0 * float(np.mean(reglas[k]["mark_up_por_hora"])) for k in ks],
                         name="mark-up medio aprendido (%)",
                         marker_color=[GRIS, AZUL, AMBAR, ROJO][:len(ks)], opacity=0.85,
                         text=[f"{100.0*float(np.mean(reglas[k]['mark_up_por_hora'])):.2f} %" for k in ks],
                         textposition="outside"))
    tr = [np.asarray(reglas[k].get("traza", []), dtype=float) for k in ks]
    if all(t.size > 3 for t in tr) and tr[0].ndim == 2:
        for j, k in enumerate(ks):
            fig.add_trace(go.Scatter(x=np.arange(1, tr[j].shape[0] + 1), y=100.0 * tr[j][:, 1],
                                     name=f"traza {k}", yaxis="y2", line=dict(width=1.4), opacity=0.7))
    fig.update_layout(xaxis_title="Regla de liquidación del escenario",
                      yaxis_title="mark-up de equilibrio (%)",
                      yaxis2=dict(title="episodio", overlaying="y", side="right", showgrid=False,
                                  visible=False))
    d = float(ap.get("delta_mark_up", float("nan")))
    u = 100.0 * float(ap.get("mark_up_uninodal", float("nan")))
    n = 100.0 * float(ap.get("mark_up_nodal", float("nan")))
    st.caption(f"Mark-up medio: uninodal {u:.2f} % → nodal {n:.2f} % (**{d:+.2f} pp**, "
               f"{100.0*d/max(abs(u), 1e-9):+.0f} % relativo) con {int(cfg['episodios_ap'])} episodios "
               f"por regla en las horas {', '.join(str(int(x)+1) for x in cfg['ap_horas'])}.")
    return {"fig": fig, "leyenda": _leyenda(
        "Aprender a ejercer poder de mercado bajo cada regla (objetivo c)",
        "Cada escenario es un juego repetido: los agentes suben o bajan su oferta por hora, reciben su "
        "beneficio del OPF completo (que re-despacha el sistema y fija los precios) y aprenden con "
        "Q-learning. La barra es el mark-up con el que convergen.",
        f"El markup de equilibrio cae {abs(d):.2f} puntos porcentuales al pasar de uninodal a nodal. Con "
        "precio uninodal un recargo en un solo nodo se cobra en todo el sistema, así que es rentable "
        "aun con pocas horas de ejercicio; con precio nodal el recargo sólo rinde en el nodo congestionado y "
        "el rival del vecino puede ararlo. Es el mecanismo por el que la Misión MinMinas (2020) espera "
        "que la liquidación nodal discipline el ejercicio de poder.",
        "El efecto depende de la congestión, y la congestión depende del régimen hidrológico: en El "
        "Niño hay más horas congestionadas → más horas en las que el markup sí paga → el descuento del "
        "régimen nodal se reduce. Correr la figura sobre una ventana húmeda y una seca es la prueba "
        "empírica que faltaba en el planteamiento.",
        " + · aprendizaje_regla() · " + str(ap.get("nota", ""))[:110])}


# ============================ registro y pestaña ==============================

FIGURES_NODAL: dict[str, tuple] = {
    # clave → (función, subgrupo, título)
    "nodal_combinaciones": (fig_nodal_combinaciones, "sintesis",
                             "Tres combinaciones para ver el impacto de la señal nodal"),
    "nodal_red":      (fig_nodal_red,      "red", "Topología del modelo y uso de las ramas · regiones y criterio"),
    "nodal_cap":      (fig_nodal_cap,      "red", "CEN por nodo y tecnología"),
    "nodal_calib":    (fig_nodal_calib,    "red", "Calibración de la oferta contra el precio de bolsa XM"),
    "nodal_lmp":      (fig_nodal_lmp,      "precios", "Mapa de calor del precio marginal por nodo y hora"),
    "nodal_spread":   (fig_nodal_spread,   "precios", "Banda de dispersión nodal y renta de congestión"),
    "nodal_descomp":  (fig_nodal_descomp,  "precios", "Descomposición del LMP: energía + pérdidas + congestión"),
    "nodal_valida":   (fig_nodal_valida,   "precios", "Validación: λ dual vs ∂Costo/∂D por diferencias finitas"),
    "nodal_cong":     (fig_nodal_cong,     "red", "Congestión por rama, pérdidas y violación"),
    "nodal_refuerzos": (fig_nodal_refuerzos, "red",
                        "Refuerzos hipotéticos de la red: un OPF por candidato"),
    "nodal_flujos":   (fig_nodal_flujos,   "red", "Flujos por rama vs su límite y su renta"),
    "nodal_desp":     (fig_nodal_desp,     "despacho", "Despacho óptimo, recorte renovable y corte flexible"),
    "nodal_nodo":     (fig_nodal_nodo,     "despacho", "Nodo caro vs nodo barato: balance y precio"),
    "nodal_beneficios": (fig_nodal_beneficios, "beneficios",
                         "Marcador de efectos positivos de la nodalidad (a-c)"),
    "nodal_flex":     (fig_nodal_flex,     "reglas", "Desconexión voluntaria de la demanda (objetivo a)"),
    "nodal_reglas":   (fig_nodal_reglas,   "reglas", "Uninodal · zonal · híbrido · nodal (KPIs de liquidación)"),
    "nodal_esc":      (fig_nodal_esc,      "estocastico", "Abanico de escenarios estocásticos y reducción"),
    "nodal_esto":     (fig_nodal_esto,     "estocastico",
                         "Segunda etapa estocástica: escenarios despachados, bandas y VSS"),
    "nodal_penetra":  (fig_nodal_penetra,  "estocastico", "Sensibilidad a la penetración renovable"),
    "nodal_ior":      (fig_nodal_ior,      "mercado", "IOR real, elasticidad medida y concentración por nodo"),
    "nodal_mec":      (fig_nodal_mec,      "mercado", "Bertrand/Cournot y el efecto del markup en el modelo"),
    "nodal_var":      (fig_nodal_var,      "riesgo", "GARCH(1,1), VaR/ES condicional y backtest de Kupiec"),
    "nodal_cov":      (fig_nodal_cov,      "riesgo", "Cobertura de mínima varianza: ratio óptimo y varianza reducida"),
    "nodal_ems":      (fig_nodal_ems,      "almacen", "EMS de baterías y bombeo bajo precio nodal"),
    "nodal_ql":       (fig_nodal_ql,       "aprendiz", "Q-learning: precio y curva de oferta aprendidos"),
    "nodal_ap":       (fig_nodal_ap,       "aprendiz", "Mark-up aprendido según la regla de liquidación"),
}

NODAL_GRUPOS = [
    ("beneficios", "🟢 Efecto positivo de liquidar por nodo"),
    ("red", "🕸️ La red y su calibración"),
    ("precios", "⚡ Los precios nodales"),
    ("despacho", "🎛️ Despacho y balance por nodo"),
    ("reglas", "⚖️ Reglas de liquidación"),
    ("estocastico", "🎲 Capa estocástica"),
    ("mercado", "🏛️ Poder de mercado"),
    ("riesgo", "📉 Riesgo: GARCH, VaR y coberturas"),
    ("almacen", "🔋 Almacenamiento"),
    ("aprendiz", "🤖 Aprendizaje por refuerzo (Q-learning multiagente) — objetivo c"),
    ("sintesis", "🧮 Cierre: tres combinaciones para ver el impacto"),
]

NODAL_SUPUESTOS = """
**Supuestos del módulo (declarados, no escondidos)**

1. **Topología.** 7 subregiones de XM convertidas en nodos con un anillo 500/230 kV; las
   reactancias y `MW_max` son órdenes de magnitud públicos de los corredores, **no** datos de
   ISA/XM. La subdivisión `media` (CEN→BOG, NOR→ANT, CAR→MAG) es un artificio para que exista
   congestión interna donde XM no publica nodos. Con `subregion`, zonal ≡ nodal por construcción.
2. **Demanda por nodo.** Se reparte con `PESO_DEMANDA_NODAL` (CEN 26 %, NOR 24 %, CAR 21 %,
   OCC 12 %, SUR 8 %, GUJ/ORI 4,5 %), calibrado a la participación de demanda del informe
   XM-SIN y a la CEN de cada subregión. XM no publica demanda por subregión en `servapibi`.
3. **Oferta.** Cada nodo tiene un bloque por tecnología con CEN real (`CapEfecNeta`) y costo
   cuadrático `c1·P + c2·P²`, con `c2 = 0,45·c1/Pmax`. `c1` se calibra contra el precio de
   bolsa; **la forma intradía sale del promedio horario de la misma ventana**, así que el R² de
   la calibración por hora es circular por construcción (se reporta igual, con la advertencia).
   La térmica se parte en carbón/gas con `Fuente_Energia` de `dim_plantas` (16 % gas/diésel en
   la ventana analizada) para no castigar todo el CO₂ con el factor del carbón.
4. **Eólica.** `CapEfecNeta` no publicó filas EOLICA en la ventana: se dimensiona desde el
   percentil 99 de `Gen_EOLICA` y un FP de 0,35, repartida GUJ 60 / ORI 25 / CAR 15 %.
5. **Pérdidas y congestión.** Linealización de primer orden (`LF_i`) iterada `perdas_iter` veces y
   multiplicadores μ por ascenso dual adaptativo con congelado por hora. La verificación por
   diferencias finitas  mide el error real de ese atajo: si sube de ~1 %, leer
   `converged` antes de usar los números.
6. **Liquidación.** Uninodal = λ del sistema; zonal = media de los nodos de la subregión;
   híbrido = bolsa + κ·(LMP − media); nodal = LMP pleno. El pago de la demanda usa la energía
   servida; el desbalance pago−generación es las pérdidas más la congestión, no un error.
7. **Régimen estratégico.** `markup` sube la oferta declarada (sobreoferta de costo) y
   `withholding` retira capacidad o tope horario. Ambos alteran el *despacho*, no la física, y se
   aplican a los bloques de **un nodo-agente** (`auto` = el de mayor peso de demanda, que es donde
   se forma λ; `TODOS` = toda la curva de oferta, el caso límite sin poder locacional): un markup
   generalizado no es poder de mercado sino una re-escalada de unidades. El retiro **por
   capacidad** solo muerde si el bloque está en su techo — en topologías holgadas devuelve 0
   efecto y así se muestra, no se oculta.
8. **Almacenamiento.** Water-filling sobre el vector de precios con η, SOC y máscaras por modo.
   No hay restricciones de rampa ni de reserva rodante; por eso su valor está sobrestimado en
   ~10-20 % frente a un UC completo.
9. **GARCH/VaR/Cobertura.** Sobre la serie horaria de la ventana (≈648 h). El "futuro" se
   aproxima con la serie uninodal (XM no publica futuros en esta API); el spread modelo-vs-real
   es lo que la figura de cobertura interpreta como riesgo locacional no cubrible.
10. **Ventana.** Todo se calcula sobre la ventana ya cargada por las pestañas v1-v4
    (`ctx["ini"]`→`ctx["fin_efectiva"]`), con sus días preliminares excluidos. Un ensayo con
    El Niño exige ampliar a ≥120 días y recargar desde la API.
11. **Demanda interrumpible (objetivo a).** Cada nodo declara hasta `flex` de su demanda como
    cortable; el corte arranca en el precio gatillo y crece linealmente hasta un 25 % por encima,
    donde el contrato de interrumpibilidad llega a su tope. Es una respuesta *declarada* —no una
    elasticidad estimada— porque XM no publica demanda flexible por nodo; la malla de gatillos
    (p40/p75/p95 del escenario) existe precisamente para que la conclusión no dependa del umbral
    elegido. Los escenarios de fase ENSO re-escalan nivel y recurso con los Δ% que mide la pestaña
    🌊 El Niño sobre 46 meses, no con factores de literatura.
"""


def _tarjeta_nodal(ctx: dict, cfg: dict, mdl: dict) -> None:
    """Fila de KPIs y semáforos de calidad numérica del módulo."""
    res, cal = mdl["res"], mdl["cal"]
    p = np.asarray(res["precios"].get(cfg["regla"], res["p_dem"]), dtype=float)
    rc = float(np.sum(res["renta_congestion"])) / 1e6
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("λ uninodal (modelo)", f"{float(np.mean(res['lam'])):,.1f}",
              f"{float(np.mean(res['lam']) - float(np.mean(np.asarray(mdl['pf']['precio'], dtype=float)))):+,.1f} vs XM")
    m2.metric(f"Precio medio ({cfg['regla']})", f"{float(np.mean(p)):,.1f}",
              f"dispersión {float(np.mean(res['dispersion'])):,.1f} COP/kWh")
    m3.metric("Horas congestionadas", f"{int(res['horas_congestion'])}/24",
              f"uso máx rama {100.0*float(np.max(res['uso_rama'])):.0f} %",
              delta_color="inverse" if res["horas_congestion"] else "normal")
    m4.metric("Renta de congestión", f"{rc:,.1f} M/h",
              f"pérdidas {float(np.sum(res['perdidas']))/24.0:,.0f} MW medios")
    m5.metric("CO₂ del día tipo", f"{float(np.sum(res['co2_t'])):,.0f} t",
              f"{float(np.sum(res['co2_t']))/max(float(np.sum(res['dem'])*24.0), 1e-9)*1000.0:,.2f} kg/MWh")
    m6.metric("Calibración", f"{cal['error_calibracion_pct']:.2f} %",
              f"κ {cal['kappa']:.3f} · {cal['n']} h")
    band = []
    if not bool(res["converged"]):
        band.append(
            f"⚠️ El ascenso dual **no convergió** (`iters_dual = {int(res['iters_dual'])}`, "
            f"violación máxima {float(res['max_violacion']):,.0f} MW, desbalance de red "
            f"{float(res['error_linealizacion_MW']):,.1f} MW). Los LMP son la mejor iteración "
            "encontrada: úselos como orden de magnitud, no como liquidación. Para cerrar el "
            "balance suba `Iteraciones de pérdidas` o `dual_iter`, baje la `Curvatura de la "
            "oferta`, o suavice `Capacidad de las líneas` hacia 1,0 (menos congestión pedida "
            "de la que este ascenso puede resolver sin un LP completo).")
    if bool(res.get("estancado")):
        band.append("🛑 El dual se **estancó** y se activó el cierre de emergencia por bisección: la "
                    "congestión pedida es más dura de lo que este OPF puede resolver sin resolver un LP.")
    if float(np.max(np.asarray(res.get("emergencia", [0.0]), dtype=float))) > 0:
        band.append(f"🚨 Emergencia activa en {int(np.sum(np.asarray(res['emergencia']) > 0))} h: "
                    "en esas horas el λ se fijó por bisección de balance y no vale la relación de "
                    "componentes de la descomposición del LMP.")
    if float(np.max(np.asarray(res["derrame_MW"], dtype=float))) > 50.0:
        band.append(f"ℹ️ Derrame máximo {float(np.max(res['derrame_MW'])):,.0f} MW: el modelo impone el "
                    "balance del **sistema**, no el de cada nodo (no hay restricción de flujo por nodo en esta "
                    "versión), así que la generación local de un nodo puede diferir de su demanda en "
                    "el residuo que transporta la red.")
    for t in band:
        st.warning(t, icon="⚠️" if t.startswith(("⚠", "🛑")) else "ℹ️")
    rl_txt = []
    q = mdl.get("ql") or {}
    if isinstance(q, dict) and q.get("disponible"):
        rl_txt.append(
            f"**🤖 aprendizaje por refuerzo (objetivo c):** Q-learning multiagente con "
            f"{int(q['n_agentes'])} agentes × {int(q['n_acciones'])} acciones × {int(cfg['episodios_rl'])} "
            f"episodios sobre la hora {int(q['hora'])}; λ aprendido "
            f"{float(q['lam_final_COP_kWh']):,.0f} vs λ exacto "
            f"{float(q['lam_exacto_COP_kWh']):,.0f} COP/kWh "
            f"({float(q['brecha_lambda_pct']):+.1f} %).")
    apd = mdl.get("ap") or {}
    if isinstance(apd, dict) and apd.get("reglas"):
        rl_txt.append(
            f"Mark-up medio aprendido: uninodal {100.0*float(apd.get('mark_up_uninodal', 0.0)):.2f} % vs "
            f"nodal {100.0*float(apd.get('mark_up_nodal', 0.0)):.2f} % "
            f"(Δ {100.0*float(apd.get('delta_mark_up', 0.0)):+.2f} pp).")
    if mdl.get("est") is not None and isinstance(mdl.get("est"), dict):
        rl_txt.append(
            f"2ª etapa estocástica: {int(mdl['est']['k'])} días típicos despachados, VSS "
            f"{float(mdl['est']['vss_pct']):+.2f} %.")
    if isinstance(mdl.get("ref"), pd.DataFrame) and len(mdl["ref"]) > 1:
        rl_txt.append(f"Refuerzos hipotéticos evaluados: {len(mdl['ref']) - 1} candidatos "
                      "(cada uno con su OPF).")
    if rl_txt:
        st.caption(" · ".join(rl_txt))


def render_tab_nodal(ctx: dict, filtros: dict[str, Any]) -> None:
    """Pestaña 🕸️ · precios marginales nodales (módulo académico, v4.2)."""
    st.markdown(
        "### 🕸️ Precios marginales nodales en el SIN: qué cambiaría si la energía se liquidara por nodo\n"
        "Este módulo no está en el notebook v1-v4. Lo desarrolló **Libardo Acero García** "
        "(asesor CREG, doctorando UNAL) como **Parte de Precios Nodales** de la aplicación y persigue "
        "tres objetivos, que son los que aparecen citados en cada figura:\n\n"
        + NODAL_OBJETIVOS_TEXTO
        + "\nResuelve un OPF con flujos óptimos DC calibrado contra los datos de la ventana ya cargada "
        "(misma `ctx` que el resto del tablero), forma los precios marginales por nodo y los compara con "
        "la liquidación uninodal real de XM bajo escenarios estocásticos de recurso y demanda, "
        "almacenamiento, coberturas de riesgo y agentes que **aprenden por refuerzo** (Q-learning) a "
        "despachar ante la señal de precio.\n\n"
        "**El foco de esta versión (v4.2) es el efecto positivo:** no si el precio sube o baja, sino "
        "qué gana el sistema cuando la señal de precio lleva información de lugar — menos déficit "
        "sin dueño, congestión con dueño (y por tanto bancable como garantía de refuerzo), "
        "menos poder de mercado materializable, almacenamiento y demanda interrumpible que solo se "
        "pueden valorar con gradiente locacional. El primer bloque lo cuantifica en el escenario "
        "activo; cada figura debajo trae su propia leyenda con muestra, decisión, lente El Niño y "
        "fuente, y ningún escenario se usa sin que se lea por qué está justificado.")
    st.info(
        "🧪 **Simulación académica, no liquidación oficial.** Ni los LMP ni la renta de congestión de "
        "esta pestaña sustituyen la resolución de XM/CREG: son el resultado de un modelo DC con "
        "parámetros declarados (ver *Supuestos* al final de la pestaña). Lo que sí es real es el "
        "anclaje — CEN, demanda, precio de bolsa, CMD, mezcla por tecnología — y por eso sirven para "
        "dimensionar el efecto, no para facturar.", icon="ℹ️")
    cfg = _nodal_widgets(ctx)
    stt = st.status("🕸️ Construyendo el modelo nodal (red → calibración → OPF → módulos)…",
                    expanded=False)
    with stt:
        try:
            mdl = _mdl(ctx, cfg)
            try:
                if stt is not None:
                    stt.update(label=(f"Modelo nodal listo en {mdl['t_seg']:.1f} s · topología "
                                      f"{mdl['red']['granularidad']} · {len(mdl['red']['nodos'])} nodos · "
                                      f"{len(mdl['red']['ramas'])} ramas · "
                                      f"{len(mdl['res']['oferta']['pmax'])} bloques despachables"),
                               state="complete", expanded=False)
            except Exception:                                      # noqa: BLE001
                pass          # sin contexto de script (import para pruebas) no hay tarjeta
        except Exception as exc:                                   # noqa: BLE001
            try:
                if stt is not None:
                    stt.update(label="El modelo nodal falló", state="error")
            except Exception:                                      # noqa: BLE001
                pass
            st.error(f"No se pudo construir el modelo: `{type(exc).__name__}: {str(exc)[:400]}`")
            bitacora(f"modelo nodal: {exc}", "error")
            with st.expander("Supuestos del módulo nodal", expanded=False):
                st.markdown(NODAL_SUPUESTOS)
            return
    st.markdown("##### 🧭 Registro de escenarios: qué fija cada uno y por qué")
    filas_pr = []
    for nom, pr in NODAL_PRESETS.items():
        filas_pr.append(dict(
            escenario=nom,
            fijo=(", ".join(f"`{k}` = {v}" for k, v in sorted(pr["cfg"].items())) or "—"),
            objetivo=pr.get("obj", "—"), justificacion=pr.get("porqué", ""),
            activo=("● activo" if nom == cfg.get("preset") else "")))
    st.dataframe(pd.DataFrame(filas_pr), use_container_width=True, hide_index=True, height=250)
    if cfg.get("preset_fija"):
        st.caption(f"El escenario **{cfg.get('preset_nombre', '')}** fija "
                   f"{len(cfg['preset_fija'])} controles ({', '.join('`' + str(k) + '`' for k in cfg['preset_fija'])}); "
                   "el resto de la barra sigue siendo suyo. Fase aplicada: "
                   + str((mdl.get("fase") or {}).get("texto", "régimen real de la ventana")) + ". "
                   "Si la figura de congestión sale plana el escenario está por debajo del umbral de "
                   "la red: no es un fallo del modelo, es la conclusión del caso base.")
    with st.expander("🗺️ Regiones del SIN usadas, criterio de agregación y por qué 10 nodos",
                     expanded=False):
        _bloque_regiones_nodales(ctx, mdl)
    _tarjeta_nodal(ctx, cfg, mdl)
    st.markdown("#### Las ocho preguntas que intenta responder esta pestaña, una por bloque de figuras")
    st.caption(
        "¿Cómo se forman los LMP con congestión? → *Mapa de calor* y *Descomposición* · "
        "¿Cuánto sube la dispersión de precios? → *Banda de dispersión* y *KPIs de liquidación* · "
        "¿Qué señalan los nodos para la expansión? → *Topología*, *Flujos por rama* y *Nodo caro vs "
        "barato* · ¿Cómo modelar el recurso estocástico (log-normal solar, Weibull eólico)? → *Abanico "
        "de escenarios* · ¿Cómo se cubre el riesgo (VaR-GARCH, h* de cobertura)? → *GARCH/VaR* y "
        "*Cobertura* · ¿Cómo cambia el poder de mercado? → *IOR*, *Bertrand/Cournot* y *Mark-up "
        "aprendido* · ¿Qué aporta el almacenamiento? → *EMS* · ¿Qué reproduce el aprendizaje por "
        "refuerzo? → *Q-learning*.")
    for clave_grp, titulo_grp in NODAL_GRUPOS:
        claves = [k for k, (_f, g, _t) in FIGURES_NODAL.items() if g == clave_grp]
        if not claves:
            continue
        st.markdown(f"##### {titulo_grp}")
        for i, clave in enumerate(claves):
            fn, _grupo, titulo = FIGURES_NODAL[clave]
            secc, fuente = NODAL_OBJETIVOS.get(clave, ("", ""))
            with st.expander(f"📊 {titulo}" + (f" · {secc}" if secc else ""),
                             expanded=(i == 0)):
                try:
                    res_f = figura_cacheada(f"nodal:{clave}", fn, ctx, cfg, mdl)
                except Exception as exc:                           # noqa: BLE001
                    st.error(f"No se pudo construir *{titulo}*: `{type(exc).__name__}: {str(exc)[:220]}`")
                    bitacora(f"figura nodal {titulo}: {exc}", "warn")
                    continue
                if not res_f or res_f.get("fig") is None:
                    extra = (" Encienda la capa de cálculo correspondiente en 🧪 *Capas de cálculo*: "
                             "muchas de estas figuras corren un OPF por escenario o por episodio."
                             if clave_grp in ("riesgo", "almacen", "aprendiz", "estocastico") else "")
                    st.info(f"Sin resultado para *{titulo}* con esta configuración en la ventana "
                            f"actual.{extra}")
                    continue
                st.plotly_chart(res_f["fig"], use_container_width=True,
                                config={"displaylogo": False}, key=_clave_fig(clave))
                if res_f.get("leyenda"):
                    st.caption(res_f["leyenda"])
                if fuente:
                    st.caption(f"▸ **De dónde sale**: {fuente}. El código del modelo vive en "
                               "`app.py` (sección «10 · MÓDULO SIN NODAL»), sin dependencias fuera de "
                               "`requirements.txt`.")
    st.divider()
    st.markdown("##### 📁 Tabla del día tipo: precios, despacho y uso de la red")
    res = mdl["res"]
    p = np.asarray(res["precios"][cfg["regla"]], dtype=float)
    tab = pd.DataFrame(p, columns=list(res["nodos"]))
    tab.insert(0, "hora", np.arange(1, len(tab) + 1))
    tab.insert(1, "lam_uninodal", np.round(np.asarray(res["lam"], dtype=float), 1))
    tab.insert(2, "demanda_MW", np.round(np.asarray(res["dem"], dtype=float).sum(axis=1), 0))
    tab.insert(3, "perdidas_MW", np.round(np.asarray(res["perdidas"], dtype=float), 1))
    tab.insert(4, "renta_congestion_MCP_h", np.round(np.asarray(res["renta_congestion"], dtype=float) / 1e6, 2))
    tab.insert(5, "co2_t_h", np.round(np.asarray(res["co2_t"], dtype=float), 1))
    tab.insert(6, "horas_congestion", np.where(np.asarray(res["uso_rama"], dtype=float).max(axis=1) > 0.999, 1, 0))
    st.dataframe(tab.round(1), use_container_width=True, hide_index=True, height=300)
    st.download_button("⬇️ Descargar precios nodales (CSV)", tab.to_csv(index=False).encode(),
                       f"precios_nodales_{ctx['ini']}_{ctx.get('fin_efectiva', ctx['fin'])}.csv",
                       "text/csv", key="dl-nodal-lmp")
    with st.expander("📚 Figura ↔ objetivo: qué cubre cada figura y de dónde sale", expanded=False):
        mapa = pd.DataFrame([dict(figura=k, funcion=v[0].__name__, grupo=v[1], titulo=v[2],
                                  seccion=NODAL_OBJETIVOS.get(k, ("", ""))[0],
                          fuente=NODAL_OBJETIVOS.get(k, ("", ""))[1]) for k, v in FIGURES_NODAL.items()])
        st.dataframe(mapa, use_container_width=True, hide_index=True)
        st.caption("Este bloque **no** tiene contraparte en las 212 celdas del notebook: su linaje es el "
                   "documento de trabajo más los datos de las pestañas v1-v4, y por eso no "
                   "entra en `CELDAS_NB` ni en el mapa 1z de *Datos y bitácora*. La verificación propia "
                   "del módulo está en `tools/verificar_celdas.py` (sección nodal) y exige que "
                   "`FIGURES_NODAL` y `NODAL_OBJETIVOS` tengan exactamente las mismas claves.")
    with st.expander("🔍 Supuestos, límites y qué NO demuestra esta pestaña", expanded=False):
        st.markdown(NODAL_SUPUESTOS)
        st.markdown(_nodal_pie(f"Ventana {ctx['ini']} → {ctx.get('fin_efectiva', ctx['fin'])}; "
                               f"topología {cfg['granularidad']}, líneas ×{cfg['estres']:.2f}, "
                               f"regla mostrada {cfg['regla']} (κ={cfg['kappa']:.2f})."))
