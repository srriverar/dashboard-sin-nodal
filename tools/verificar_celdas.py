#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verificación celda por celda del notebook contra `app.py` (pedida por el autor).

Convención de numeración: **índice absoluto de celda dentro de `cells` del JSON, 0-based,
contando markdown y código** (es el índice que devuelve `jupyter nbconvert --to script` y el
que usa `analysis/cells.txt`). El número que muestra Jupyter en pantalla (`In [n]`) cuenta
solo celdas de código, así que no coincide: para convertir, `celda_j = j + 1 - (#markdown ≤ j)`.

Tres tablas curadas viven aquí (y se verifican contra el notebook real, no contra la memoria):

  * `CELDAS_NB`  : figura de la app → (celda donde se define `graficar_*`, celda donde se ejecuta)
  * `CALC_NB`    : celda de cálculo del notebook → función de la app que la reproduce
  * `ALIAS_NB`   : símbolo del notebook → símbolo equivalente en la app (cuando fue renombrado)

Clasifica las 212 celdas en: figura · cálculo portado · export de archivos · salida en
notebook (print/.head()/display) · config/imports · documentación (markdown). Cualquier celda
que no encaje queda en `REVISAR` y el script sale con código 1.

    python tools/verificar_celdas.py                # resumen + REVISAR
    python tools/verificar_celdas.py --literal      # literales CELDAS_NB / COBERTURA_NB
    python tools/verificar_celdas.py --json         # máquina-legible
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "app.py"
NB = [RAIZ.parent / "uploads" / "01_ETL_exploracion_v1.json", RAIZ / "01_ETL_exploracion_v1.json"]
RE_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(", re.M)

# ── figura de la app → (celda_def, celda_ejecucion) en el JSON del notebook ────────────────────
CELDAS_NB: dict[str, tuple[int, int]] = {
    "balance": (70, 71), "perfil": (72, 73), "precio": (74, 75), "escasez": (76, 77),
    "corr": (81, 83), "corr_pares": (76, 80), "precio_saeb": (84, 85), "capacidad": (131, 131), "embalses": (86, 89), "zonas": (86, 87),
    "aportes_hist": (88, 90), "tec_hora": (91, 92), "mat_tec_hora": (91, 92),
    "perfiles": (93, 94), "mat_perfiles": (93, 94), "heatmap_precio": (110, 111),
    "mat_precio": (110, 111), "solares": (112, 114), "cen": (129, 129),
    "margen_bandas": (129, 129), "fp_tec": (135, 135), "fp_plantas": (137, 137),
    "fp_dia": (139, 139), "emisiones": (141, 143), "co2_emis": (143, 143),
    "co2_perfil": (145, 145), "pareto": (147, 147), "co2_pareto": (147, 147),
    "co2_corr": (149, 149), "nino_onda": (155, 155), "nino_oni": (157, 157),
    "nino_evo": (159, 161), "mat_enso": (161, 161), "nino_scatter_apo": (163, 163),
    "nino_scatter_pre": (163, 163), "mat_fase": (167, 205), "v4_matriz": (180, 190),
    "v4_01": (182, 184), "costo_dist": (186, 186), "mat_costo": (186, 186),
    "mat_costo_dh": (188, 188), "v4_heat": (143, 145), "v4_enso": (195, 199),
    "en_hallazgos": (207, 207),
}
# tablas que el notebook dibuja como figura aparte y que el autor pidió retirar: siguen
# calculándose, pero se muestran como tabla en la app (no FIGURES, no desplegable).
TABLAS_NB: dict[str, tuple[int, str]] = {
    "impacto_eventos_v4": (203, "tabla_impacto_eventos_v4"),
}
# ── celdas de cálculo sin figura: la app las reproduce en esta función ─────────────────────────
CALC_NB: dict[int, str] = {
    2: "requirements.txt / pip install pydataxm", 4: "cabecera de imports de app.py",
    5: "cabecera de imports de app.py", 6: "cabecera de imports de app.py",
    8: "descargar_ventana", 9: "descargar_ventana", 10: "descargar_ventana",
    12: "_detectar_base_dir", 13: "items_a_dataframe", 14: "preparar_dim_plantas",
    15: "preparar_dim_plantas", 16: "buscar_cache", 17: "preparar_dim_plantas",
    18: "preparar_dim_plantas", 19: "preparar_dim_plantas", 20: "preparar_dim_plantas",
    22: "transformar_generacion", 23: "preparar_dim_plantas", 25: "verificar_y_corregir_integridad",
    26: "preparar_dim_plantas", 27: "verificar_y_corregir_integridad",
    30: "enriquecer_generacion", 31: "enriquecer_generacion", 33: "perfil_tecnologia_hora",
    34: "perfil_tecnologia_hora", 35: "enriquecer_generacion", 36: "ctx_agentes_lookup",
    37: "ctx_agentes_lookup", 38: "ctx_agentes_lookup", 39: "enriquecer_generacion",
    40: "enriquecer_generacion", 41: "enriquecer_generacion", 43: "enriquecer_generacion",
    45: "enriquecer_generacion", 46: "enriquecer_generacion", 47: "enriquecer_generacion",
    48: "preparar_dim_plantas", 49: "preparar_dim_plantas", 50: "comparar_catalogos_historicos",
    51: "cabecera de app.py", 52: "METRICAS_V2 / constantes de app.py",
    53: "_detectar_base_dir", 54: "post_xm / descargar_metrica", 55: "mode_demo y MODO",
    56: "METRICAS_V2", 57: "rango_fechas / filtrar_por_periodo", 58: "guardar_cache",
    59: "wide_to_long / daily_to_long", 60: "procesar_todas_las_metricas",
    61: "validar_tablas_analiticas", 62: "render_datos (bloque 3)",
    63: "_detectar_base_dir", 64: "descargar_ventana", 65: "render_datos (linaje)",
    66: "render_datos (linaje)", 67: "sistema_hora_a_dia", 68: "sistema_hora_a_dia",
    69: "resumen_diario", 95: "cargar_mapa_operativo", 96: "gen_con_ambito",
    97: "niveles_disponibles", 101: "explorar_rio_zona / _selector_ambito",
    102: "cargar_mapa_operativo", 103: "niveles_disponibles", 104: "gen_con_ambito",
    105: "filtrar_ambito", 109: "explorar_rio_zona", 111: "fig_09_heatmap_precio_hora",
    115: "fig_10_perfiles_solares", 116: "construir_hallazgos", 117: "render_resultado",
    121: "METRICAS_V3 / constante", 122: "descargar_ventana", 123: "sondea_metrica / ventana_efectiva",
    124: "descargar_ventana", 125: "wide_to_long_combustible / daily_to_long_recurso",
    126: "enriquecer_generacion", 127: "normalizar_tecnologia", 128: "resumen_capacidad",
    129: "resumen_capacidad", 130: "resumen_capacidad", 132: "fig_v3_01_cen_margen",
    133: "factores_planta", 134: "factores_planta", 136: "fig_v3_02_fp_tecnologia",
    138: "fig_v3_03_fp_plantas", 140: "fig_v3_04b_heatmap_fp_diario", 141: "agregar_emisiones_diarias",
    142: "agregar_emisiones_diarias", 144: "fig_v3_05_emisiones_diarias",
    146: "fig_v3_06_perfil_horario_emisiones", 148: "fig_v3_07_top_emisores_pareto",
    149: "fig_v3_08_correlacion_extendida", 150: "descargar_oni_noaa", 153: "descargar_oni_noaa",
    154: "descargar_nino34", 155: "descargar_nino34", 156: "parse_oni_noaa / clasificar_fase",
    157: "detectar_eventos_enso", 158: "construir_nino_mensual", 159: "construir_nino_mensual",
    160: "fig_v3_09_dashboard_enso", 162: "fig_v3_09_dashboard_enso", 163: "fig_enso_05_scatter",
    164: "tabla_impacto_enso", 165: "tabla_impacto_enso", 166: "fig_v3_11_climatologia_fase",
    167: "fig_v3_11_climatologia_fase", 168: "bloque_hallazgos", 169: "construir_hallazgos_v3",
    170: "render_datos (descargables)", 171: "render_datos (descargables)",
    176: "METRICAS_V4", 177: "descargar_ventana", 178: "descargar_ventana",
    179: "sistema_v4_h", 180: "cruzar_diario_v4", 181: "calcular_indisponibilidad",
    182: "calcular_indisponibilidad", 183: "fig_v4_02_disponibilidad", 185: "fig_v4_02_costo_marginal",
    187: "fig_v4_02_costo_marginal", 189: "fig_v4_03_heatmap_costo_dia_hora",
    190: "fig_v3_08_correlacion_extendida", 194: "v4_mensual_para_enso", 195: "v4_mensual_para_enso",
    196: "v4_mensual_para_enso", 197: "tabla_impacto_eventos_v4", 198: "fig_v4_05_enso_economico",
    200: "fig_v4_05_enso_economico", 202: "tabla_impacto_eventos_v4", 203: "tabla_impacto_eventos_v4",
    204: "fig_v3_11_climatologia_fase", 205: "fig_v3_11_climatologia_fase",
    206: "fig_v4_17_hallazgos_v4", 207: "fig_v4_17_hallazgos_v4", 208: "render_datos (descargables)",
    209: "render_datos (descargables)",
}
# ── símbolos renombrados: def del notebook → def real de la app ─────────────────────────────────
ALIAS_NB: dict[str, str] = {
    "obtener_generacion": "descargar_metrica", "_encontrar_catalogo_mas_reciente": "buscar_cache",
    "cargar_catalogos": "buscar_cache", "actualizar_catalogos_maestros": "preparar_dim_plantas",
    "comparar_catalogos": "comparar_catalogos_historicos",
    "comparar_todos_los_catalogos": "comparar_catalogos_historicos",
    "detectar_base_dir": "_detectar_base_dir", "clasificar_fechas": "rango_fechas",
    "_configuracion_endpoint": "post_xm", "cargar_csv_mas_reciente": "buscar_cache",
    "rango_fechas_raw": "rango_fechas", "filtrar_raw_por_periodo": "filtrar_por_periodo",
    "validar_raw_xm": "verificar_y_corregir_integridad", "to_date": "agregar_timestamp",
    "_tendencia_valida": "fig_matriz_correlaciones", "_configurar_scatter_correlacion": "fig_matriz_correlaciones",
    "preparar_matriz_correlacion": "_corr", "color_hora": "fig_v2_05b_precio_horario_saeb",
    "_preparar_aportes": "fig_08_aportes_historico", "agregar": "construir_hallazgos",
    "_silenciar_descarga": "sondea_metrica", "_normalizar_tecnologia": "normalizar_tecnologia",
    "wide_to_long_recurso_combustible": "wide_to_long_combustible", "color_tec": "color_tecnologia",
    "_parsear_oni_ascii": "parse_oni_noaa", "_parsear_oni_psl": "parse_oni_noaa",
    "obtener_oni": "descargar_oni_noaa", "_sombrear_eventos": "fig_v3_09_dashboard_enso",
    "_scatter_con_tendencia": "fig_enso_05_scatter", "_envolver_texto": "bloque_hallazgos",
    "mostrar_hallazgos_v3": "bloque_hallazgos", "_resumen_tec": "calcular_indisponibilidad",
    "_r": "_corr", "_mes_wide": "v4_mensual_para_enso", "_melt_horario": "v4_mensual_para_enso",
    "_procesar_mes": "v4_mensual_para_enso", "construir_serie_mensual_v4": "v4_mensual_para_enso",
    "construir_hallazgos_v4": "fig_v4_17_hallazgos_v4", "construir_hallazgos": "construir_hallazgos",
    "construir_hallazgos_v3": "construir_hallazgos_v3", "_localizar_script_operativo": "cargar_mapa_operativo",
    "_cargar_modulo_operativo": "cargar_mapa_operativo", "_localizar_script_catalogo": "cargar_mapa_operativo",
    "_cargar_modulo_catalogo": "cargar_mapa_operativo", "enriquecer_generacion_operativa": "gen_con_ambito",
    "resumir_cobertura_operativa": "niveles_disponibles", "auditar_y_exportar_cobertura": "niveles_disponibles",
    "enriquecer_generacion_geografica": "gen_con_ambito", "filtrar_generacion_geografica": "filtrar_ambito",
    "filtrar_generacion_operativa": "filtrar_ambito", "crear_explorador_operativo": "explorar_rio_zona",
    "graficar_resumen_areas": "_fig_scope_perfil", "graficar_heatmap_operativo": "_fig_scope_perfil",
    "graficar_perfiles_operativos": "_fig_scope_perfil", "graficar_resumen_regiones": "_fig_scope_perfil",
    "graficar_heatmap_geografico": "_fig_scope_perfil", "graficar_perfiles_geograficos": "_fig_scope_perfil",
    "crear_explorador_geografico": "explorar_rio_zona", "actualizar_ambitos": "_selector_ambito",
    "actualizar_figura": "_fig_scope_perfil", "to_date_": "agregar_timestamp",
}
# helpers de un solo uso dentro de la misma celda (no requieren espejo propio)
INLINE = {"_cero", "col_def", "validar_tablas_analiticas", "_num"}


def figura_de(celda: int) -> list[str]:
    out = []
    for k, (a, b) in CELDAS_NB.items():
        if a <= celda <= b:
            out.append(k)
    return out


def clasificar(cells, tipos, app_src: str):
    filas = []
    for i, (txt, tp) in enumerate(zip(cells, tipos)):
        if tp != "code":
            filas.append({"celda": i, "estado": "documentación", "detalle": "", "n": 0})
            continue
        cuerpo = "\n".join(l for l in txt.splitlines() if not l.strip().startswith("#"))
        defs = list(dict.fromkeys(RE_DEF.findall(cuerpo)))
        figs = figura_de(i)
        if figs:
            filas.append({"celda": i, "estado": "figura", "detalle": ", ".join(sorted(set(figs))),
                          "n": len(cuerpo.splitlines())})
            continue
        no_port = [d for d in defs if d.startswith("graficar_") and not _existe(d, app_src)]
        if no_port:
            filas.append({"celda": i, "estado": "REVISAR", "detalle": f"figura sin mapa: {no_port}",
                          "n": len(cuerpo.splitlines())})
            continue
        malas = [d for d in defs if d not in INLINE and not _existe(d, app_src)]
        if defs and not malas:
            filas.append({"celda": i, "estado": "cálculo portado",
                          "detalle": ", ".join(f"{d} → {_equiv(d)}" for d in defs[:3]),
                          "n": len(cuerpo.splitlines())})
            continue
        if malas:
            filas.append({"celda": i, "estado": "REVISAR", "detalle": f"defs sin espejo: {malas}",
                          "n": len(cuerpo.splitlines())})
            continue
        if i in CALC_NB:
            filas.append({"celda": i, "estado": "cálculo portado", "detalle": CALC_NB[i],
                          "n": len(cuerpo.splitlines())})
            continue
        if not cuerpo.strip():
            filas.append({"celda": i, "estado": "celda vacía", "detalle": "", "n": 0})
            continue
        if re.search(r"^\s*(?:%\w+|import |from \w+)", cuerpo) and len(cuerpo.splitlines()) <= 30:
            filas.append({"celda": i, "estado": "config/imports", "detalle": "requirements.txt y cabecera de app.py",
                          "n": len(cuerpo.splitlines())})
            continue
        if re.search(r"\b(?:to_csv|savefig|FIG_DIR\w*|mkdir)\b", cuerpo):
            filas.append({"celda": i, "estado": "export de archivos", "detalle": "botones ⬇️ en 📚 Datos",
                          "n": len(cuerpo.splitlines())})
            continue
        if re.search(r"ipywidgets|interactive\(|plt\.show\(\)|display\(|\.head\(\)|print\(|st\.", cuerpo) \
                and len(cuerpo.splitlines()) <= 20:
            filas.append({"celda": i, "estado": "salida en notebook", "detalle": "render de la app (caption/tablas)",
                          "n": len(cuerpo.splitlines())})
            continue
        filas.append({"celda": i, "estado": "REVISAR", "detalle": (cuerpo.strip().splitlines() or [""])[0][:70],
                      "n": len(cuerpo.splitlines())})
    return filas


def _equiv(sym: str) -> str:
    """Símbolo de la app que reproduce el del notebook (o el mismo si no fue renombrado)."""
    return ALIAS_NB.get(sym, sym)


def _existe(sym: str, app_src: str) -> bool:
    for cand in (sym, ALIAS_NB.get(sym, "")):
        if cand and re.search(r"\b" + re.escape(cand) + r"\b", app_src):
            return True
    return False


def intervalos(idx):
    out = []
    for i in sorted(idx):
        if out and i <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], i)
        else:
            out.append([i, i])
    return tuple((a, b) for a, b in out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--literal", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--nb", default=None)
    a = ap.parse_args()
    rutas = [Path(a.nb)] if a.nb else NB
    for r in rutas:
        if r.exists():
            nb = json.loads(r.read_text(encoding="utf-8"))
            uso = r
            break
    else:
        sys.exit(f"notebook no encontrado en {rutas}")
    cells = ["".join(c.get("source") or []) for c in nb["cells"]]
    tipos = [c.get("cell_type") for c in nb["cells"]]
    app_src = APP.read_text(encoding="utf-8")
    filas = clasificar(cells, tipos, app_src)
    cnt = Counter(f["estado"] for f in filas)
    ncode = sum(1 for t in tipos if t == "code")
    # autochequeos duros
    problemas = []
    bloque = re.search(r"^FIGURES[^{]*\{(.*?)^\}", app_src, re.M | re.S)
    claves_fig = set(re.findall(r'^\s{4}"(\w+)":', bloque.group(1), re.M)) if bloque else set()
    sin_celda = sorted({m.group(1) for m in re.finditer(r'^\s{4}"(\w+)":\s*\(\s*(?:fig_|lambda)',
                       bloque.group(1), re.M)} - set(CELDAS_NB)) if bloque else []
    if sin_celda:
        problemas.append(f"figuras de FIGURES sin celda mapeada: {sin_celda}")
    huerfanas = sorted(set(CELDAS_NB) - claves_fig) if claves_fig else []
    if huerfanas:
        problemas.append(f"CELDAS_NB con claves que no están en FIGURES: {huerfanas}")
    for k, (d, r) in CELDAS_NB.items():
        if not (0 <= d < len(cells) and 0 <= r < len(cells)):
            problemas.append(f"{k}: celdas fuera de rango ({d},{r})")
            continue
        for x in (d, r):
            if tipos[x] != "code":
                problemas.append(f"{k}: la celda {x} no es de código (es {tipos[x]})")
    # unicidad del ancla: ninguna celda-figura del notebook puede quedar sin dueño
    celdas_fig = set()
    for ini_f, fin_f in CELDAS_NB.values():
        celdas_fig |= set(range(ini_f, fin_f + 1))
    huerf = []
    for i, t in enumerate(cells):
        if tipos[i] != "code":
            continue
        for d in RE_DEF.findall("\n".join(l for l in t.splitlines() if not l.strip().startswith("#"))):
            if d.startswith("graficar_") and i not in celdas_fig and not _existe(d, app_src):
                huerf.append(f"{d}@c{i}")
    if huerf:
        problemas.append(f"figuras del notebook sin espejo en la app: {huerf}")
    # ── módulo nodal (tesis, no notebook): coherencia FIGURES_NODAL ↔ NODAL_TESIS ──
    n_nodal = 0
    bn = re.search(r"^FIGURES_NODAL[^{]*\{(.*?)^\}", app_src, re.M | re.S)
    bt = re.search(r"^NODAL_TESIS[^{]*\{(.*?)^\}", app_src, re.M | re.S)
    if bn and bt:
        kn = set(re.findall(r'^\s{4}"(\w+)":', bn.group(1), re.M))
        kt = set(re.findall(r'^\s{4}"(\w+)":', bt.group(1), re.M))
        n_nodal = len(kn)
        if kn - kt:
            problemas.append(f"figuras nodales sin ancla en la tesis: {sorted(kn - kt)}")
        if kt - kn:
            problemas.append(f"anclas nodales sin figura en FIGURES_NODAL: {sorted(kt - kn)}")
        for m in re.finditer(r'^\s{4}"(\w+)":\s*\((fig_\w+),\s*"(\w+)"', bn.group(1), re.M):
            fun, grp = m.group(2), m.group(3)
            if f"def {fun}(" not in app_src:
                problemas.append(f"figura nodal {m.group(1)} apunta a {fun}(), inexistente")
            if f'("{grp}",' not in app_src:
                problemas.append(f"figura nodal {m.group(1)}: el subgrupo {grp!r} no está en NODAL_GRUPOS")
        for m in re.finditer(r"""^\s{4}"(\w+)": \("([^"]*)", "([^"]*)" """, bt.group(1), re.M):
            if not (m.group(2).startswith("§") or "objetivo" in m.group(2)):
                problemas.append(f"NODAL_TESIS[{m.group(1)}]: la sección no empieza por § ({m.group(2)!r})")
            if len(m.group(3)) < 24:
                problemas.append(f"NODAL_TESIS[{m.group(1)}]: fuente demasiado breve ({m.group(3)!r})")
        if "def render_tab_nodal(" not in app_src or '("🕸️ Precios nodales", "nodal")' not in app_src:
            problemas.append("el módulo nodal no está conectado a una pestaña (TITULOS_TABS/render_tab_nodal)")
    elif bn or bt:
        problemas.append("existe FIGURES_NODAL o NODAL_TESIS pero no el otro (deben ser pares)")
    revisar = [f for f in filas if f["estado"] == "REVISAR"]
    print(f"# notebook: {uso.name} · {len(cells)} celdas ({ncode} de código)")
    for k, v in cnt.most_common():
        print(f"#   {k:22s} {v:3d}   ({100*v/len(filas):.0f} %)")
    cub = cnt["figura"] + cnt["cálculo portado"]
    print(f"#   → celdas de código con lógica en la app: {cub}/{ncode} "
          f"({100*cub/max(1,ncode):.0f} %) · sin clasificar: {len(revisar)}")
    if n_nodal:
        print(f"#   módulo nodal (tesis doctoral, fuera del notebook): {n_nodal} figuras con ancla "
              f"§/Eq. y sin contraparte en CELDAS_NB por diseño")
    if problemas:
        print("\n# PROBLEMAS DE COHERENCIA (app ↔ tablas curadas):")
        for x in problemas:
            print(f"#   ✗ {x}")
    if revisar:
        print("\n# REVISAR:")
        for f in revisar:
            print(f"#   c{f['celda']:3d} n={f['n']:3d} {f['detalle'][:88]}")
    if a.json:
        print(json.dumps(filas, ensure_ascii=False))
    if a.literal:
        etiquetas = {"figura": "la celda construye una figura que la app muestra",
                     "cálculo portado": "cálculo del notebook reproducido en app.py (función indicada)",
                     "export de archivos": "to_csv/savefig → botones de descarga de la pestaña 📚 Datos",
                     "salida en notebook": "print/.head()/display → caption y tablas de la app",
                     "config/imports": "imports y %pip → requirements.txt y cabecera de app.py",
                     "celda vacía": "celda vacía en el notebook",
                     "documentación": "markdown: secciones, portadas y notas del cuaderno"}
        print("\n# COBERTURA_NB generado por `python tools/verificar_celdas.py --literal` "
              "(celdas = índice absoluto 0-based en `cells` del JSON)")
        print("COBERTURA_NB: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {")
        for e in etiquetas:
            iv = intervalos([f["celda"] for f in filas if f["estado"] == e])
            if not iv:
                continue
            print(f'    "{e}": ({json.dumps(etiquetas[e], ensure_ascii=False)}, {iv!r}),   '
                  f"# {cnt[e]} celdas")
        print("}")
        print("\n# celda → función de la app que la reproduce (para la tabla de auditoría)")
        print("CALC_APP_NB: dict[int, str] = {")
        for f in filas:
            if f["estado"] == "cálculo portado" and f.get("detalle"):
                print(f"    {f['celda']}: {json.dumps(f['detalle'], ensure_ascii=False)},")
        print("}")
        print("\nCELDAS_NB: dict[str, tuple[int, int]] = {")
        for k in sorted(CELDAS_NB):
            print(f"    {k!r}: {CELDAS_NB[k]!r},")
        print("}")
    sys.exit(1 if (revisar or problemas) else 0)


if __name__ == "__main__":
    main()
