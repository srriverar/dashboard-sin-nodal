#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resolutor **heurístico** figura↔celda (por similitud de nombres y títulos).

Herramienta de trabajo: el emparejamiento automático se revisó a mano y la relación curada y
verificada quedó en `tools/verificar_celdas.py` (esa es la autoritativa, y la que se ejecuta en CI).
Este script sirve para detectar derivas: si el notebook cambia y alguna `graficar_*` ya no se
parece a ninguna figura de la app, aquí aparece en `SIN MAPA`/`ALERTAS`.


Por qué existe: los números que aparecen en los docstrings de `app.py` (o en los nombres
`fig_v3_NN`) **no** son fiables como índice de celda: el notebook numeró sus figuras v3/v4
por sección («CELDA 13 — fig_v3_05»), y esos números difieren del orden absoluto del JSON.
Aquí la relación se resuelve con dos anclas duras y se verifica:

  A. el notebook declara en markdown, para v3/v4, la pareja `(fig_vN_MM · título)` junto a
     cada celda que define el `graficar_*`;
  B. para v1/v2 (celdas 70-116) no hay declaración: se empareja por **slug** del nombre de
     función (`graficar_balance_diario` ↔ `fig_balance_diario`) con similitud de tokens.

Comandos:
    python tools/mapa_celdas.py                 # tabla figura → celda (def/llamada) + alertas
    python tools/mapa_celdas.py --escribir      # además, literal CELDAS_NB para pegar en app.py
    python tools/mapa_celdas.py --json          # salida máquina-legible
Salida sana: 0 filas en `SIN MAPA`, 0 `ALERTA` de título no coincidente.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
APP = RAIZ / "app.py"
NB_DEFAULT = [RAIZ.parent / "uploads" / "01_ETL_exploracion_v1.json", RAIZ / "01_ETL_exploracion_v1.json"]
RE_DEFGRAF = re.compile(r"^[ \t]*def[ \t]+(graficar_\w+|[a-z_]\w*dashboard\w*)[ \t]*\(", re.M)
RE_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(", re.M)
RE_DECL = re.compile(r"\((fig_v\d_\d\d)\s*·\s*([^)]{6,90})\)")
RE_ASIG = re.compile(r"^([a-z_][a-z0-9_]*)[ \t]*=")
PARADA = {"horario", "serie", "vs", "por", "del", "las", "los", "con", "dia", "hora", "fig", "v3", "v4"}


def leer_nb(rutas):
    for r in rutas:
        if Path(r).exists():
            nb = json.loads(Path(r).read_text(encoding="utf-8"))
            return (["".join(c.get("source") or []) for c in nb.get("cells", [])],
                    [c.get("cell_type") for c in nb.get("cells", [])], Path(r))
    sys.exit(f"Notebook no encontrado: {[str(x) for x in rutas]}")


def slug(txt: str) -> set[str]:
    return {t for t in re.split(r"[_\s\-·,.()]+", str(txt).lower()) if len(t) > 3 and t not in PARADA}


def similitud(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return 2.0 * inter / (len(a) + len(b)) + (0.25 if inter else 0.0)


def indice_notebook(cells, tipos):
    """→ {graficar_x: {'def':i,'llamada':j,'titulos':[...]}}, {fig_vN_MM: i}, filas por celda."""
    figs: dict[str, dict] = {}
    por_clave: dict[str, int] = {}
    cod = []
    for i, (t, tp) in enumerate(zip(cells, tipos)):
        if tp != "code":
            continue
        cuerpo = "\n".join(l for l in t.splitlines() if not l.strip().startswith("#"))
        defs = RE_DEFGRAF.findall(cuerpo)
        titulos = re.findall(r"title[_\w]*\s*=\s*[\"']([^\"']{8,120})[\"']", cuerpo)
        for d in defs:
            e = figs.setdefault(d, {"def": i, "llamada": None, "titulos": []})
            e["titulos"] += titulos[:2]
        cod.append({"i": i, "defs": RE_DEF.findall(cuerpo), "graf": defs, "nlineas": len(cuerpo.splitlines())})
    # llamada: la primera celda posterior que invoca el nombre
    for d, e in figs.items():
        for j in range(e["def"] + 1, min(len(cells), e["def"] + 5)):
            if tipos[j] == "code" and re.search(re.escape(d) + r"\s*\(", cells[j]):
                e["llamada"] = j
                break
    # declaraciones en markdown «(fig_v3_05 · título)» → ancla absoluta
    decl_tit: dict[str, str] = {}
    for i, (t, tp) in enumerate(zip(cells, tipos)):
        for m in RE_DECL.finditer(t):
            clave, tit = m.group(1), m.group(2).strip()
            decl_tit[clave] = tit
            tgt = clave.replace("fig_", "graficar_", 1)
            # celda del graficar correspondiente: la siguiente celda de código que lo defina
            for j in range(i, min(len(cells), i + 4)):
                if tipos[j] == "code":
                    gs = RE_DEFGRAF.findall("\n".join(l for l in cells[j].splitlines()
                                                      if not l.strip().startswith("#")))
                    if gs:
                        por_clave[clave] = j
                        break
            if clave not in por_clave:                    # buscar hacia adelante por nombre
                for d, e in figs.items():
                    if similitud(slug(tit), slug(d)) >= 0.3:
                        por_clave[clave] = e["def"]
                        break
    return figs, por_clave, decl_tit, cod


def figuras_app():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    asign = None
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "FIGURES" for t in n.targets):
            asign = n.value
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "FIGURES":
            asign = n.value
    if not isinstance(asign, ast.Dict):
        sys.exit("FIGURES no es un dict literal")
    docs, fuentes = {}, {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            docs[n.name] = " ".join(x.strip() for x in (ast.get_docstring(n) or "").splitlines() if x.strip())
            fuentes[n.name] = "".join(ast.unparse(x) for x in n.body[:6])
    out = []
    for k, val in zip(asign.keys, asign.values):
        if not isinstance(val, ast.Tuple) or not val.elts:
            continue
        fn = None
        if isinstance(val.elts[0], ast.Name):
            fn = val.elts[0].id
        if len(val.elts) > 2 and isinstance(val.elts[2], ast.Constant):
            titulo = val.elts[2].value
        else:
            titulo = ""
        sub = val.elts[1].value if len(val.elts) > 1 and isinstance(val.elts[1], ast.Constant) else ""
        if fn:
            out.append({"clave": k.value, "func": fn, "sub": sub, "titulo_app": titulo,
                        "doc": docs.get(fn, ""), "cuerpo": fuentes.get(fn, "")})
    return out


def emparejar(fa, figs, por_clave, decl_tit):
    """→ dict con celda resuelta y cómo se resolvió."""
    m = re.search(r"\b(fig_v\d_\d\d)\b", fa["func"] + " " + fa["doc"])
    if m and m.group(1) in por_clave:
        return por_clave[m.group(1)], f"declaración {m.group(1)}"
    tokens_app = slug(re.sub(r"^fig_((v\d_\d\d)|\d*_?\d*)_?", "", fa["func"])) | slug(fa["titulo_app"])
    # 1) nombres de graficar_* que aparecen escritos en el docstring o en el cuerpo
    citados = [d for d in figs if re.search(rf"\b{re.escape(d)}\b", fa["doc"] + " " + fa["cuerpo"][:900])]
    candidatos = [(similitud(tokens_app, slug(d) | slug(" ".join(figs[d]["titulos"]))), d, "nombre/título")
                   for d in (citados or list(figs))]
    candidatos = [c for c in candidatos if c[0] > 0]
    if not candidatos:
        return None, "sin candidato"
    candidatos.sort(reverse=True)
    mejor, d, via = candidatos[0]
    if mejor < 0.30:
        return None, f"similitud baja ({mejor:.2f} → {d})"
    return figs[d]["def"], f"{via}: {d} ({mejor:.2f})"



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--escribir", action="store_true", help="imprime el literal CELDAS_NB")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--nb", default=None)
    a = ap.parse_args()
    rutas = [Path(a.nb)] if a.nb else NB_DEFAULT
    cells, tipos, ruta = leer_nb(rutas)
    figs, por_clave, decl_tit, _cod = indice_notebook(cells, tipos)
    filas, sin_mapa, alerta = [], [], []
    for fa in figuras_app():
        cdef, como = emparejar(fa, figs, por_clave, decl_tit)
        if cdef is None:
            sin_mapa.append((fa["clave"], fa["func"], como))
            continue
        filas.append({"clave": fa["clave"], "func": fa["func"], "sub": fa.get("sub", ""),
                      "titulo": fa.get("titulo", ""), "celda_def": cdef,
                      "celda_llamada": next((e["llamada"] for e in figs.values() if e["def"] == cdef), None),
                      "resuelto_por": como})
    n_fig_nb = len({e["def"] for e in figs.values()})
    print(f"# notebook {ruta.name}: {len(cells)} celdas ({sum(1 for t in tipos if t == 'code')} de código) · "
          f"{len(figs)} graficar_* en {n_fig_nb} celdas · heurística resuelta para {len(filas)} de "
          f"{len(filas) + len(sin_mapa)} entradas de FIGURES")
    print("# (la relación curada y verificada vive en tools/verificar_celdas.py; este script solo "
          "detecta derivas)")
    print(f"# SIN MAPA: {len(sin_mapa)} · ALERTAS de título: {len(alerta)}")
    if a.json:
        print(json.dumps({"figuras": filas, "sin_mapa": sin_mapa, "alertas": alerta},
                         ensure_ascii=False, indent=1))
    else:
        for f in sorted(filas, key=lambda x: x["celda_def"]):
            lla = f["celda_llamada"] if f["celda_llamada"] is not None else "—"
            print(f"  {str(f['clave']):16s} {str(f['sub']):11s} celda {str(f['celda_def']):>4s} "
                  f"(run {str(lla):>4s})  {f['func']:38s} {f['resuelto_por']}")
        if sin_mapa:
            print("\n# SIN MAPA (revisar a mano en tools/verificar_celdas.py → CELDAS_NB):")
            for k, fn, porque in sin_mapa:
                print(f"   {k:16s} {fn:38s} {porque}")
        if alerta:
            print("\n# ALERTAS de título:")
            for k, t in alerta:
                print(f"   {k:16s} {t}")
    if a.escribir:
        print("\nCELDAS_NB = {")
        for f in sorted(filas, key=lambda x: x["clave"]):
            fin = f["celda_llamada"] if f["celda_llamada"] is not None else f["celda_def"]
            print(f"    {f['clave']!r}: ({f['celda_def']}, {fin}),")
        print("}")


if __name__ == "__main__":
    main()
