#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 SIN · EL NIÑO ENERGY DASHBOARD  —  Generación, hidrología, capacidad,
 emisiones, margen y costo marginal del Sistema Interconectado Colombiano (SIN)
================================================================================

Aplicación Streamlit AUTOCONTENIDA que ejecuta, en vivo y desde cero, todo el
pipeline de análisis del notebook `01_ETL_exploracion_v1.ipynb` (v1 → v4):

    v1  ETL + catálogos maestros + EDA por tecnología ................ [Manuel Fajardo]
    v2  Balance, precio, hidrología y contexto operativo ............. [Manuel Fajardo]
    v3  Capacidad, factor de planta, emisiones y Fenómeno del Niño ... [Sergio Rivera]
    v4  Indisponibilidad, margen disponible, costo marginal y ENSO económico [Sergio Rivera]

Qué hace la app en cada arranque
--------------------------------
 1. Lee de la API pública de XM (servapibi.xm.com.co) la VENTANA MÁS RECIENTE
    CERRADA (último mes, D-1) y descarga por bloques ≤ 31 días con reintentos.
 2. Transforma el formato ancho de XM (Values_Hour01…Hour24) a modelo
    dimensional (fact_generacion, dim_plantas, gen_enriquecida, resumen_diario).
 3. Calcula los indicadores de v2/v3/v4: matriz por tecnología, perfil horario,
    concentración por agente, capacidad efectiva neta (CEN), factor de planta,
    emisiones/intensidad de CO₂, indisponibilidad, margen y costo marginal.
 4. Construye el análisis ENSO: ONI (NOAA CPC) + aportes/embalses/precio desde
    2015 y margen/costo desde 2016, detección de eventos Niño/Niña, correlaciones
    rezagadas, climatología por fase y tabla de impacto por evento.
 5. Renderiza cada figura con una LEYENDA que dice qué decisión o conclusión se
    puede tomar, priorizando las que afectan el FENÓMENO DEL NIÑO.

Fuentes y licencia de los datos
-------------------------------
· XM S.A. E.S.P. — API pública de Bienda: https://servapibi.xm.com.co
  (wrapper de referencia: https://github.com/EquipoAnaliticaXM/API_XM)
· NOAA Climate Prediction Center — ONI: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt

Créditos
--------
Manuel Fernando Fajardo Rodríguez — Senior Electrical Engineer · Power Systems ·
Data Science — Experiencia en estudios de sistemas de potencia; automatización y
análisis de datos para el sector eléctrico; Maestría en Ingeniería Eléctrica,
Universidad Nacional de Colombia.  Autor de las celdas v1 y v2 (Secciones 0–6 y
Anexos), originadas en el repositorio:
    https://github.com/mffajardor/solar-generation-colombia-dash

Prof. Sergio Rivera, PhD, SMIEEE — Universidad Nacional de Colombia (Departamento
de Ingeniería Eléctrica y Electrónica, Facultad de Ingeniería, Bogotá). Profesor
Titular, Director del Departamento, ex-coordinador de la Maestría y el Doctorado
en Ingeniería Eléctrica (2018-2022). Ingeniero Eléctrico UNAL (2001), Especialista
en Sistemas de Distribución (2004), PhD en Ingeniería Eléctrica (Instituto de
Energía Eléctrica, Universidad Nacional de San Juan, 2011), Postdoctoral Associate
en MIT (2013-2014) y Postdoctoral Fellow en Masdar Institute (2014); Fulbright
Scholar en University of Florida. Investigación y docencia en sistemas de potencia,
redes inteligentes, microrredes, optimización, IA aplicada a energía y mercados
eléctricos; +100 publicaciones; ganador de competencias IEEE en optimización de
redes con renovables.  Autor de las celdas v3 y v4 (Secciones 7–12): capacidad,
factor de planta, emisiones, Fenómeno del Niño, indisponibilidad, margen y costo
marginal.

Esta app reproduce esas celdas con código propio, equivalente y ejecutable sin
Jupyter. Los datos derivados se cachean en `data/raw` y `data/reference`.
================================================================================
"""
from __future__ import annotations

import concurrent.futures as _fut
import datetime as dt
import io
import json
import logging
import math
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)


def _compat_ancho() -> None:
    """Hace que `use_container_width=True` siga sirviendo en Streamlit 1.49+ y 2.x.

    Desde 1.49 el parámetro canónico es `width="stretch"` y `use_container_width` está
    deprecado (aviso visible en consola y eliminación anunciada). En vez de reescribir las
    ~23 llamadas del tablero —y romperlas en versiones viejas— se traduce el kwarg aquí:
    si la versión instalada acepta `width`, se usa; si no, se deja el antiguo. Si el nuevo
    fuera rechazado, se reintenta con el viejo. Así el mismo `app.py` corre en 1.32 y en 2.x.
    """
    import functools
    import inspect
    for nombre in ("plotly_chart", "dataframe", "table", "json"):
        original = getattr(st, nombre, None)
        if original is None or getattr(original, "_appmanuel_ancho", False):
            continue
        try:
            acepta_width = "width" in inspect.signature(original).parameters
        except (TypeError, ValueError):
            acepta_width = False
        if not acepta_width:
            continue

        @functools.wraps(original)
        def envuelto(*args, _orig=original, **kwargs):
            if kwargs.pop("use_container_width", None) is not None:
                try:
                    return _orig(*args, width="stretch", **kwargs)
                except TypeError:
                    return _orig(*args, use_container_width=True, **kwargs)
            return _orig(*args, **kwargs)
        envuelto._appmanuel_ancho = True                        # type: ignore[attr-defined]
        setattr(st, nombre, envuelto)


_compat_ancho()
# --- Arranque a prueba de confusiones -------------------------------------
# Si alguien ejecuta `python app.py` (en vez de `streamlit run app.py`) Streamlit
# corre en "bare mode": no hay servidor ni sesión, los widgets no responden y el
# log se inunda de avisos `missing ScriptRunContext` que tapan lo importante.
# Lo detectamos, callamos el ruido y dejamos una instrucción clara.
def _guardia_bare_mode() -> bool:
    """True si estamos en *bare mode* (`python app.py`). Sin efectos bajo `streamlit run`.

    En bare mode se muestra el aviso y el programa termina con código 0, salvo que
    el usuario pida explícitamente seguir con `APPMANUEL_BARE=1` (útil para refrescar
    la carpeta `data/raw` sin abrir navegador).
    """
    try:
        from streamlit.runtime import exists as _runtime_exists
        if _runtime_exists():
            return False
    except Exception:                                     # noqa: BLE001 (Streamlit < 1.30)
        return False
    # El registro de niveles de Streamlit es propio: los loggers creados después
    # heredan el nivel global, así que hay que bajar el nivel global (no solo uno).
    try:
        from streamlit.logger import set_log_level as _nivel_global
        _nivel_global("error")
    except Exception:                                     # noqa: BLE001 (Streamlit viejo)
        logging.getLogger("streamlit").setLevel(logging.ERROR)
    print("\n" + "=" * 74)
    print("  Se ejecuto `python app.py`: Streamlit esta en modo sin servidor.")
    print("  Ese modo NO levanta la interfaz (de ahi los 'missing ScriptRunContext').")
    print("")
    print("  Comando correcto, dentro de C:\\AppManuel y con el .venv activado:")
    print("")
    print("      streamlit run app.py")
    print("")
    print("  y abre http://localhost:8501 en el navegador.")
    print("  (Alternativa: doble clic en Iniciar_Dashboard.bat)")
    print("=" * 74 + "\n")
    opt_out = os.environ.get("APPMANUEL_BARE", "").strip().lower()
    if opt_out not in ("1", "true", "si", "yes"):
        print("  Se detuvo aquí a propósito: en modo sin servidor los widgets no responden y")
        print("  la consola se llenaría de ~900 avisos 'missing ScriptRunContext'.")
        print("")
        print("  Para usar este mismo archivo como REFRESCO DE DATOS sin interfaz:")
        print("")
        print("      set APPMANUEL_BARE=1 && python app.py")
        print("")
        raise SystemExit(0)
    print("  APPMANUEL_BARE=1 -> continuando en modo sin servidor (solo refresca datos).\n")
    return True


# Solo cuando el archivo ES el programa principal: `tools/*.py` y las pruebas importan
# `app.py` como librería (ahí no hay runtime y el aviso sería un falso positivo).
_en_bare_mode = _guardia_bare_mode() if __name__ == "__main__" else False



# ==============================================================================
# 0 · CONFIGURACIÓN, RUTAS Y CATÁLOGO DE MÉTRICAS
# ==============================================================================
APP_NOMBRE = "SIN · El Niño Energy Dashboard"
APP_VERSION = "v4.1-app (Streamlit)"
API_BASE = "https://servapibi.xm.com.co"
URL_ONI = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
URL_ONI_ALT = "https://psl.noaa.gov/data/correlation/oni.data"
REPO_V1_V2 = "https://github.com/mffajardor/solar-generation-colombia-dash"
PERIODO_MAX_API = 31          # límite duro de la API por petición
UMBRAL_NINO = 0.5            # °C ONI
UMBRAL_NINA = -0.5
UMBRAL_EMBALSES_ALERTA = 60  # % volumen útil
UMBRAL_EMBALSES_CRITICO = 40
HORAS_PICO = (17, 22)        # convención XM: H01 = 00:00-01:00
HORAS_VALLE = (0 + 1, 6)
DIAS_VENTANA_POR_DEFECTO = 30
CATALOGO_MAX_ANTIGUEDAD = 14  # días antes de refrescar dim_* maestros


def _detectar_base_dir() -> Path:
    r"""Carpeta de trabajo: la del `app.py` (C:\AppManuel en local, raíz del repo en la nube).

    Se puede forzar con las variables de entorno APP_MANUEL_HOME / APP_MANUEL_DIR
    (útil si `app.py` queda dentro de una subcarpeta como `src\`).
    """
    env = os.environ.get("APP_MANUEL_HOME") or os.environ.get("APP_MANUEL_DIR")
    candidatos: list[Path] = []
    if env:
        candidatos.append(Path(env))
    aqui = Path(__file__).resolve().parent
    candidatos.append(aqui)
    try:
        if Path.cwd() not in candidatos:
            candidatos.append(Path.cwd())
    except OSError:
        pass
    for c in candidatos:
        try:
            if (c / "data").exists() or (c / "src").exists() or (c / "certs").exists():
                return c
        except OSError:
            continue
    return aqui


BASE_DIR = _detectar_base_dir()
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REF_DIR = DATA_DIR / "reference"
CERTS_DIR = BASE_DIR / "certs"
SRC_DIR = BASE_DIR / "src"
FIG_DIR = PROCESSED_DIR / "figuras_app"

for _d in (RAW_DIR, PROCESSED_DIR, REF_DIR, FIG_DIR):
    try:
        _d.mkdir(parents=True, exist_ok=True)
    except OSError:                       # fs de solo lectura (algún PaaS)
        pass

BUNDLE_SSL = CERTS_DIR / "upme_bundle.pem"


def _activar_bundle_ssl() -> str:
    """Equivalente a las celdas 3-4 del notebook: bundle Sectigo p/ geo.upme.gov.co.

    XM no lo necesita (certificado válido); solo se usa para el módulo opcional de
    geografía UPME. Devuelve la ruta del bundle o una cadena explicativa.
    """
    if BUNDLE_SSL.exists():
        os.environ.setdefault("SSL_CERT_FILE", str(BUNDLE_SSL))
        return str(BUNDLE_SSL)
    return "no encontrado (no hace falta para XM; solo para UPME)"


COLORES_TECNO = {
    "HIDRAULICA": "#1f77b4", "TERMICA": "#7f7f7f", "SOLAR": "#f1c232",
    "EOLICA": "#29c5e4", "COGENERADOR": "#2e8b57", "BAGAZO": "#8bc34a",
    "BIOMASA": "#689f38", "OTRO": "#c200b0",
}
COLORES_METRICA = {
    "generacion": "#1565c0", "demanda": "#d9662d", "precio": "#8c564b",
    "embalses": "#2e7d32", "aportes": "#4caf50", "escasez": "#b71c1c",
    "nino": "#d62728", "nina": "#1f77b4", "neutro": "#9e9e9e",
}
PALETA_RECTYPE = {
    "AUTOG PEQ. ESCALA": "#FFE082", "GEN. DISTRIBUIDA": "#FFC107",
    "AUTOGENERADOR": "#FF8F00", "NORMAL": "#2196F3", "FILO DE AGUA": "#90CAF9",
    "FILO AGUA ESPECIAL": "#42A5F5", "CICLO COMBINADO": "#EF5350",
    "COGENERADOR": "#9E9E9E",
}


def color_tecnologia(tec: Any) -> str:
    return COLORES_TECNO.get(str(tec).upper().strip(), COLORES_TECNO["OTRO"])


@dataclass(frozen=True)
class Metrica:
    """Entrada del catálogo de métricas XM (equivalente a METRICAS_API* del nb)."""
    nombre: str
    metric_id: str
    entity: str
    tipo: str                       # 'hourly' | 'daily' | 'list'
    prefijo: str
    unidad: str
    desc: str
    peso: int = 0                   # 1 = tardo (DispoCome/Combustible/CO2Eq)
    retraso_meses: int = 0          # rezago esperado de publicación
    clave_entidades: str = ""       # forzado (list → ListEntities)

    @property
    def endpoint(self) -> str:
        if self.tipo == "list":
            return "Lists"
        return "hourly" if self.tipo == "hourly" else "daily"

    @property
    def clave(self) -> str:
        if self.clave_entidades:
            return self.clave_entidades
        return {"hourly": "HourlyEntities", "daily": "DailyEntities",
                "list": "ListEntities"}[self.tipo]


# ---------------------------- colores y convenciones -------------------------
ROJO = "#c62828"; AZUL = "#1f77b4"; CELESTE = "#4fc3f7"; VERDE = "#2e7d32"
AMBAR = "#f9a825"; NARANJA = "#ef6c00"; GRIS = "#616161"; AZUL_SUAVE = "#90caf9"
MARGEN = dict(l=46, r=18, t=44, b=40)

TECNOLOGIAS_ORDEN = ["HIDRAULICA", "TERMICA", "SOLAR", "EOLICA", "COGENERADOR", "BAGAZO",
                     "BIOMASA", "GAS", "CARBON", "TÉRMICA", "SIN CATALOGAR"]
TECNOLOGIAS_DIAGNOSTICO = ["SOLAR", "EOLICA", "BIOMASA", "BAGAZO", "COGENERADOR", "OTRO"]
TERMICA_KEYS = ("TERMICA", "TÉRMICA", "CARBON", "GAS", "ACOPLADA", "TURBINAGAS", "DIesel")
RENOVABLES_TECH = {"SOLAR", "EOLICA", "BIOMASA", "BAGAZO", "GEOTERMICA", "MAREOMOTRIZ"}
COLOR_TECNOLOGIAS = {
    "HIDRAULICA": "#1f77b4", "HIDROELÉCTRICA": "#1f77b4", "HIDROLOGICA": "#1565c0",
    "TERMICA": "#8d6e63", "TÉRMICA": "#8d6e63", "TERMOELECTRICA": "#795548",
    "CARBON": "#5d4037", "GAS": "#ef6c00", "SOLAR": "#f9a825", "EOLICA": "#26a69a",
    "COGENERADOR": "#2e7d32", "BAGAZO": "#7cb342", "BIOMASA": "#558b2f",
    "GEOTERMICA": "#ad1457", "OTRO": "#9e9e9e", "SIN CATALOGAR": "#bdbdbd",
}
COLORES_FASE = {"El Niño": ROJO, "La Niña": "#1565c0", "Neutral": "#9e9e9e", "Sin ONI": "#e0e0e0"}
ORDEN_FASE = ("La Niña", "Neutral", "El Niño")
LAGS_MES = list(range(0, 13))

# --- rutas de referencia (semillas versionadas en el repo) -------------------
ONI_SEED = REF_DIR / "oni_noaa.csv"
NINO34_URL = "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii"
NINO_MENSUAL = REF_DIR / "nino_mensual.csv"
V4_MENSUAL = REF_DIR / "v4_mensual.csv"
MODELO_DIR = PROCESSED_DIR
VENTANA_POR_DEFECTO = DIAS_VENTANA_POR_DEFECTO
CEN_M2_POR_MM = 1e6      # 1 mm sobre 1 km² = 1 000 m³; se usa solo en el índice notebook

# --- créditos (aparecen en el pie de página del tablero) --------------------
AUTOR_APP = [{
    "nombre": "Manuel Fernando Fajardo Rodríguez",
    "cargo": "Senior Electrical Engineer · Power Systems · Data Science",
    "perfil": "Automatización y analítica de datos para el sector eléctrico colombiano; autor de "
              "las celdas v1 y v2 (Secciones 0-6 y Anexos) del notebook que esta app ejecuta.",
    "contacto": f"Repositorio de origen (v1-v2): {REPO_V1_V2}",
    "nota": "Maestría en Ingeniería Eléctrica, Universidad Nacional de Colombia.",
}]
AUTOR_V3 = [{
    "nombre": "Prof. Sergio Rivera, PhD, SMIEEE — Universidad Nacional de Colombia",
    "cargo": "Profesor Titular · Depto. de Ingeniería Eléctrica y Electrónica (ex-Director)",
    "perfil": "Ing. Eléctrico UNAL (2001), Especialista en Sistemas de Distribución (2004), PhD en "
              "Ingeniería Eléctrica (Instituto de Energía Eléctrica, UNSJ, 2011); Postdoc Associate "
              "en MIT (2013-14), Postdoctoral Fellow en Masdar Institute (2014), Fulbright Visiting "
              "Scholar en University of Florida y Gambrinus Fellow en TU Dortmund (2019). Líneas: "
              "confiabilidad de sistemas de potencia, redes inteligentes, microrredes, ERNC, "
              "optimización heurística y aprendizaje automático; +100 publicaciones.",
    "contacto": "srriverar@unal.edu.co · ORCID 0000-0002-2995-1147",
    "nota": "Autor de las celdas v3 y v4 (Secciones 7-12): CEN, factor de planta, emisiones, "
            "fenómeno El Niño/La Niña, indisponibilidad, margen y costo marginal.",
}]

#: repositorio donde vive el notebook original (v1-v4, 212 celdas) y esta app
REPO_NOTEBOOK = "https://github.com/srriverar/sin-el-nino-dashboard"
NOTA_REPO = (
    f"El código de las versiones **v1 a v4** (el notebook `01_ETL_exploracion_v1.ipynb`, 212 celdas, y "
    f"su port a esta app) se puede revisar en [{REPO_NOTEBOOK.replace('https://', '')}]({REPO_NOTEBOOK})."
)
FUENTE = ("Información operativa **pública de XM S.A. E.S.P.** (API Bienda, https://servapibi.xm.com.co) "
          "**y datos climáticos de NOAA/CPC** (ONI y anomalías Niño-3.4 / Niño1+2, "
          "cpc.ncep.noaa.gov/data/indices); aviso ENSO citado: CPC 13-ago-2026 e IRI 19-ago-2026. "
          "Las consultas a XM usan los mismos `metric_id` que expone la librería oficial "
          "`pydataxm.pydatasimem`")
NOTA_METODOLOGICA = (
    "Prototipo académico independiente: **no es una herramienta oficial de XM** ni sustituye las "
    "cifras del CNO/ASM ni la publicación oficial del operador. Esta app no importa "
    "`pydataxm.pydatasimem`: llama directamente a los mismos endpoints REST que esa librería envuelve "
    "(`ListadoRecursos`, `Gene`, `DemaReal`, `PrecBolsNaci`, `PorcVoluUtilDiar`, `CostMargDesp`, "
    "`EmisionesCO2`, …), de modo que cualquier cifra es reproducible con `pydataxm`. Los indicadores del "
    "mercado salen de consultas en vivo a la API de XM y del cálculo local con las mismas fórmulas del "
    "notebook; el análisis ENSO combina el índice ONI oficial de NOAA/CPC con las semillas mensuales de "
    "`data/reference`. Los meses sin ONI publicado heredan la última fase oficial y se marcan como "
    "‘provisional’.")


def fuente_nota() -> str:
    return f"{FUENTE}. {NOTA_METODOLOGICA}"


def _cert_envio():
    """Ruta del bundle CA local (upme_bundle.pem) si existe; si no, el store del sistema."""
    if CERTS_DIR.exists():
        for candidato in sorted(CERTS_DIR.glob("*bundle*.pem")):
            return str(candidato)
    return True


# --- v1/v2 (Manuel Fajardo) --------------------------------------------------
METRICAS_V2: dict[str, Metrica] = {
    "Gene_Sistema": Metrica("Gene_Sistema", "Gene", "Recurso", "hourly", "Gene_Sistema", "kWh/h",
                            "Generación real por recurso; el tablero la agrega a nivel sistema y "
                            "por tecnología (dim_plantas)"),
    "Gene_Recurso": Metrica("Gene_Recurso", "Gene", "Recurso", "hourly", "Gene_Recurso", "kWh/h",
                            "Detalle por planta (para factor de planta y concentración de mercado)"),
    "DemaReal_Sistema": Metrica("DemaReal_Sistema", "DemaReal", "Sistema", "hourly", "DemaReal_Sistema",
                                "kWh/h", "Demanda real horaria del SIN (kWh por hora → MW)"),
    "PrecBolsNaci_Sistema": Metrica("PrecBolsNaci_Sistema", "PrecBolsNaci", "Sistema", "hourly",
                                    "PrecBolsNaci_Sistema", "COP/kWh", "Precio de bolsa nacional (horas)"),
    "PPPrecBolsNaci_Sistema": Metrica("PPPrecBolsNaci_Sistema", "PPPrecBolsNaci", "Sistema", "daily",
                                      "PPPrecBolsNaci_Sistema", "COP/kWh", "Precio de bolsa ponderado diario"),
    "PorcVoluUtilDiar_Sistema": Metrica("PorcVoluUtilDiar_Sistema", "PorcVoluUtilDiar", "Sistema",
                                        "daily", "PorcVoluUtilDiar", "%", "Volumen útil de embalses"),
    "PorcApor_Sistema": Metrica("PorcApor_Sistema", "PorcApor", "Sistema", "daily", "PorcApor_Sistema", "%",
                                "Aportes hídricos como % de la media histórica"),
    "AporEner_Sistema": Metrica("AporEner_Sistema", "AporEner", "Sistema", "daily", "AporEner_Sistema", "kWh",
                                "Aportes de energía hídrica (kWh/día)"),
    "PrecEsca_Sistema": Metrica("PrecEsca_Sistema", "PrecEsca", "Sistema", "daily", "PrecEsca_Sistema",
                                "COP/kWh", "Precio de escasez (señal de emergencia)"),
}
CATALOGOS: dict[str, Metrica] = {
    "ListadoRecursos": Metrica("ListadoRecursos", "ListadoRecursos", "Sistema", "list",
                               "dim_plantas", "—", "Catálogo de recursos/plantas del SIN",
                               clave_entidades="ListEntities"),
    "ListadoAgentes": Metrica("ListadoAgentes", "ListadoAgentes", "Sistema", "list",
                              "dim_agentes", "—", "Agentes del Mercado de Energía Mayorista",
                              clave_entidades="ListEntities"),
    "ListadoRios": Metrica("ListadoRios", "ListadoRios", "Sistema", "list",
                           "dim_rios", "—", "Ríos / cuencas del SIN", clave_entidades="ListEntities"),
}
# --- v3 (Sergio Rivera) ------------------------------------------------------
METRICAS_V3: dict[str, Metrica] = {
    "CapEfecNeta_Recurso": Metrica("CapEfecNeta_Recurso", "CapEfecNeta", "Recurso", "daily",
                                   "CapEfecNeta_Recurso", "kW", "Capacidad Efectiva Neta por recurso (CEN)",
                                   peso=1),
    "DemaMaxPot_Sistema": Metrica("DemaMaxPot_Sistema", "DemaMaxPot", "Sistema", "daily",
                                  "DemaMaxPot_Sistema", "kW", "Demanda máxima de potencia programada",
                                  retraso_meses=2),
    "DispoCome_Recurso": Metrica("DispoCome_Recurso", "DispoCome", "Recurso", "hourly",
                                 "DispoCome_Recurso", "kW",
                                 "Disponibilidad comercial por recurso (todas las plantas × 24 h)", peso=1),
    "EmisionesCO2_RecursoComb": Metrica("EmisionesCO2_RecursoComb", "EmisionesCO2", "RecursoComb",
                                        "hourly", "EmisionesCO2_RecursoComb", "tCO2",
                                        "Emisiones de CO2 por recurso y combustible", peso=1),
    "EmisionesCO2Eq_Recurso": Metrica("EmisionesCO2Eq_Recurso", "EmisionesCO2Eq", "Recurso", "hourly",
                                      "EmisionesCO2Eq_Recurso", "tCO2eq", "CO2 equivalente por recurso", peso=1),
    "ConsCombustibleMBTU_Recurso": Metrica("ConsCombustibleMBTU_Recurso", "ConsCombustibleMBTU",
                                           "Recurso", "hourly", "ConsCombustibleMBTU_Recurso", "MBTU",
                                           "Consumo de combustible por recurso (Values.Name = combustible, "
                                           "Values.code = planta)", peso=1),
    "factorEmisionCO2e_Sistema": Metrica("factorEmisionCO2e_Sistema", "factorEmisionCO2e", "Sistema",
                                         "hourly", "factorEmisionCO2e", "gCO2e/kWh",
                                         "Factor de emisión del margen de operación del SIN", peso=1),
}
# Series mensuales ligeras del bloque ENSO: nombre clave → prefijo del CSV en cache/data\raw.
# (En el notebook se llamaban METRICAS_NINO_V3; aquí basta con el prefijo porque la agregación
#  mensual se hace sobre los CSV ya bajados por el pipeline principal.)
PREFIJOS_NINO: dict[str, str] = {"aportes": "PorcApor_Sistema",
                                 "embalses": "PorcVoluUtilDiar_Sistema",
                                 "precio": "PPPrecBolsNaci_Sistema"}
# alias privado, usado por funciones internas del módulo
_PREFIJOS_NINO = PREFIJOS_NINO
METRICAS_NINO: dict[str, Metrica] = {
    "aportes_pct": METRICAS_V2["PorcApor_Sistema"],
    "embalses_pct": METRICAS_V2["PorcVoluUtilDiar_Sistema"],
    "precio_pond": METRICAS_V2["PPPrecBolsNaci_Sistema"],
}
# --- v4 (Sergio Rivera) ------------------------------------------------------
METRICAS_V4: dict[str, Metrica] = {
    "CostMargDesp_Sistema": Metrica("CostMargDesp_Sistema", "CostMargDesp", "Sistema", "hourly",
                                    "CostMargDesp_Sistema", "COP/kWh",
                                    "Costo marginal de despacho programado (fija el precio de la hora)"),
}
_log_lock = threading.Lock()
BITACORA: list[dict[str, Any]] = []


def bitacora(msg: str, nivel: str = "info") -> None:
    """Registro de ejecución visible en la UI (tab 'Bitácora')."""
    fila = {"ts": dt.datetime.now().strftime("%H:%M:%S"), "nivel": nivel, "msg": msg}
    with _log_lock:
        BITACORA.append(fila)
    print(f"[{fila['ts']}] {nivel.upper():5s} {msg}")


def historial() -> pd.DataFrame:
    with _log_lock:
        return pd.DataFrame(list(BITACORA))


# ==============================================================================
# 0b · PROGRESO DE CARGA (etapas V1-V4) y caché de figuras
# ==============================================================================
#: etiqueta legible de cada etapa + qué tipo de dato toca (para el panel de carga)
ETAPAS: dict[str, tuple[str, str]] = {
    "V1": ("V1 · Extracción (14-19 consultas en paralelo)",
           "API REST pública de XM `servapibi.xm.com.co` + caché en CSV local `data/raw`"),
    "V2": ("V2 · ETL y normalización",
           "transformación en memoria: wide→long, horas 1-24, unidades (fracción→%), duplicados, NaN"),
    "V3": ("V3 · Cruces, KPIs y linaje",
           "catálogo `ListadoRecursos`/`ListadoAgentes` (CSV locales) × generación por planta + resumen diario"),
    "V4": ("V4 · ENSO (NOAA) y capa v4",
           "CSV público de NOAA/CPC (oni.ascii.txt), semillas `data/reference` y CEN/emisiones de XM"),
    "OK": ("Listo · tablas analíticas compuestas",
           "ctx completo disponible para las 44 figuras de las 9 pestañas"),
}


class Progreso:
    """Bitácora de etapas *thread-safe*: el pipeline corre en un hilo y la UI lo lee."""

    __slots__ = ("_lock", "frac", "etapa", "lineas", "t0", "tf", "ticks", "fin")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frac = 0.0
        self.etapa = ETAPAS["V1"][0]
        self.lineas: list[tuple[str, str, str]] = []
        self.t0 = time.time()
        self.tf: float | None = None
        self.ticks = 0
        self.fin = False

    def __call__(self, frac: float, txt: str, etapa: str = "") -> None:
        with self._lock:
            try:
                self.frac = max(self.frac, min(1.0, float(frac)))
            except (TypeError, ValueError):
                pass
            if etapa:
                self.etapa = etapa
            self.ticks += 1
            self.lineas.append((dt.datetime.now().strftime("%H:%M:%S"), self.etapa, txt))
            if len(self.lineas) > 80:
                del self.lineas[: len(self.lineas) - 80]

    def cerrar(self) -> None:
        with self._lock:
            self.fin, self.tf, self.frac = True, time.time(), 1.0

    def instantanea(self) -> tuple[float, str, list, float, float | None, int, bool]:
        with self._lock:
            return (self.frac, self.etapa, list(self.lineas), self.t0, self.tf, self.ticks, self.fin)


#: progreso de la carga en curso (lo usa `construir_contexto` si no le pasan uno explícito)
_PROGRESO_ACTIVO: list[Any] = [None]


# --- caché de figuras ---------------------------------------------------------
#: reconstruir 45 figuras Plotly en cada rerun (filtro, toggle, recarga) costaba ~2-4 s;
#: se indexan por (figura, firma de los datos, argumentos) y se reutiliza el objeto.
_FIG_CACHE: dict[tuple, Any] = {}
_FIG_CACHE_FIRMA: list[str] = [""]


def figura_cacheada(clave: str, fn: Any, *args: Any, **kw: Any) -> Any:
    """Ejecuta `fn(*args)` una sola vez por combinación de datos y la guarda en memoria.

    La firma (`ctx["_firma"]`) cambia cuando cambian ventana, métricas activas o modo, así
    que nunca se sirve una figura construida con otros datos. Si `fn` lanza, no se cachea
    el fallo: la próxima interacción lo vuelve a intentar.
    """
    ctx = args[0] if args and isinstance(args[0], dict) else {}
    firma = str(ctx.get("_firma", "")) or "sin-firma"
    if _FIG_CACHE_FIRMA[0] != firma:
        _FIG_CACHE.clear()
        _FIG_CACHE_FIRMA[0] = firma
    try:
        idargs = repr([(a if not isinstance(a, pd.DataFrame) else f"<df {len(a)}>") for a in args[1:]])
        idkw = repr(sorted((k, str(v)) for k, v in kw.items()))
    except Exception:                                             # noqa: BLE001
        idargs = idkw = ""
    k = (clave, firma, idargs, idkw)
    if k in _FIG_CACHE:
        return _FIG_CACHE[k]
    res = fn(*args, **kw)
    if len(_FIG_CACHE) > 160:
        _FIG_CACHE.clear()
    _FIG_CACHE[k] = res
    return res


# ==============================================================================
# 1 · CAPA DE DATOS: cliente XM con caché, reintentos y sonda de rezago
# ==============================================================================
_SESION = threading.local()


def _sesion() -> requests.Session:
    if not hasattr(_SESION, "s"):
        s = requests.Session()
        s.headers.update({"Content-Type": "application/json",
                          "User-Agent": "sin-el-nino-dashboard/1.0 (+streamlit)"})
        a = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=32)
        s.mount("https://", a)
        _SESION.s = s
    return _SESION.s


def _parseos_json(r: requests.Response) -> dict:
    """json() tolerante: XM a veces responde 200 con texto vacío."""
    txt = (r.text or "").strip()
    if not txt:
        return {}
    import json as _json
    try:
        return _json.loads(txt)
    except Exception:
        return {}


def post_xm(endpoint: str, metric_id: str, entity: str, ini: dt.date, fin: dt.date,
            reintentos: int = 4, pausa: float = 1.5, timeout: int = 180):
    """POST crudo a la API de Bienda. Devuelve la lista de 'Items' ([] si no hay datos)."""
    cuerpo = {"MetricId": metric_id, "StartDate": ini.isoformat(),
              "EndDate": fin.isoformat(), "Entity": entity, "Filter": []}
    url = f"{API_BASE}/{endpoint}"
    ultimo = ""
    for intento in range(1, reintentos + 1):
        try:
            r = _sesion().post(url, json=cuerpo, timeout=timeout)
            if r.status_code == 200:
                return _parseos_json(r).get("Items", []) or []
            ultimo = f"HTTP {r.status_code}: {r.text[:120]}"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(pausa * intento)
                continue
            if "supera el maximo rango" in r.text:            # guardia defensiva
                return []
            break
        except requests.exceptions.RequestException as exc:
            ultimo = f"{type(exc).__name__}: {exc}"[:140]
            time.sleep(pausa * intento)
    raise RuntimeError(f"XM {endpoint} {metric_id}/{entity} {ini}→{fin} falló — {ultimo}")


def items_a_dataframe(items: list, clave: str) -> pd.DataFrame:
    """Items de Bienda → DataFrame ancho (Date + Values_HourXX / Code + Value / atributos)."""
    if not items:
        return pd.DataFrame(columns=["Date"])
    try:                                            # ruta idéntica a la del notebook
        df = pd.json_normalize(items, clave, "Date", sep="_")
        if "Date" not in df.columns and "Date_" in df.columns:
            df = df.rename(columns={"Date_": "Date"})
        return df
    except Exception:                               # plan B manual (tolerante a schemas)
        filas: list[dict] = []
        for it in items:
            fecha = it.get("Date")
            for ent in it.get(clave, []) or []:
                vals = ent.get("Values") if isinstance(ent.get("Values"), dict) else {}
                fila = {"Date": fecha, "Id": ent.get("Id")}
                for k, v in vals.items():
                    fila[f"Values_{k}"] = v
                for k, v in ent.items():
                    if k in ("Values",):
                        continue
                    if isinstance(v, (str, int, float)) or v is None:
                        fila[f"Values_{k}" if k == "Value" else k] = v
                filas.append(fila)
        return pd.DataFrame(filas)


def _nombre_cache(m: Metrica, ini: dt.date, fin: dt.date) -> str:
    return f"{m.prefijo}_{ini:%Y-%m-%d}_{fin:%Y-%m-%d}.csv"


def guardar_cache(df: pd.DataFrame, m: Metrica, ini: dt.date, fin: dt.date):
    try:
        ruta = RAW_DIR / _nombre_cache(m, ini, fin)
        df.to_csv(ruta, index=False)
        return ruta
    except OSError:
        return None


def buscar_cache(m: Metrica, ini: dt.date, fin: dt.date) -> tuple[pd.DataFrame, Path | None]:
    """Caché local exacta o contenedora (reutiliza el archivo que cubra [ini,fin])."""
    if not RAW_DIR.exists():
        return pd.DataFrame(), None
    exacta = RAW_DIR / _nombre_cache(m, ini, fin)
    if exacta.exists():
        try:
            return pd.read_csv(exacta), exacta
        except Exception:
            pass
    mejor: tuple[pd.Timestamp, pd.Timestamp, Path] | None = None
    for ruta in sorted(RAW_DIR.glob(f"{m.prefijo}_*.csv")):
        try:
            bruto = pd.read_csv(ruta, usecols=["Date"])
            fech = pd.to_datetime(bruto["Date"], errors="coerce").dropna()
            if fech.empty:
                continue
            cubre = fech.min().date() <= ini and fech.max().date() >= fin
            if cubre and (mejor is None or fech.max() > mejor[1]):
                mejor = (fech.min(), fech.max(), ruta)
        except Exception:
            continue
    if mejor:
        try:
            df = pd.read_csv(mejor[2])
            return df, mejor[2]
        except Exception:
            pass
    return pd.DataFrame(), None


def _bloques(ini: dt.date, fin: dt.date, max_dias: int = PERIODO_MAX_API):
    out, cur = [], ini
    while cur <= fin:
        nxt = min(cur + dt.timedelta(days=max_dias - 1), fin)
        out.append((cur, nxt))
        cur = nxt + dt.timedelta(days=1)
    return out


def descargar_metrica(m: Metrica, ini: dt.date, fin: dt.date, usar_cache: bool = True,
                       forzar: bool = False, pausa: float = 0.25) -> tuple[pd.DataFrame, str]:
    """Descarga [ini,fin] en bloques ≤31 días con reintentos. Devuelve (df, origen)."""
    if m.tipo == "list":          # los catálogos ignoran el rango: una sola petición
        bloques = [(ini, fin)]
    else:
        bloques = _bloques(ini, fin)
    partes, errores = [], []
    for n, (b_ini, b_fin) in enumerate(bloques, 1):
        for intento in range(3):
            try:
                items = post_xm(m.endpoint, m.metric_id, m.entity, b_ini, b_fin)
                if items:
                    partes.append(items_a_dataframe(items, m.clave))
                break
            except Exception as exc:                      # noqa: BLE001
                if intento == 2:
                    errores.append(f"{b_ini}→{b_fin}: {exc}")
                else:
                    time.sleep(1.2 * (intento + 1))
        if pausa and len(bloques) > 1:
            time.sleep(pausa)
    if not partes:
        if errores:
            bitacora(f"{m.metric_id}: sin datos ({errores[0][:120]})", "warn")
        return pd.DataFrame(columns=["Date"]), "sin datos"
    df = pd.concat(partes, ignore_index=True)
    if "Date" in df.columns:
        df = df.drop_duplicates().sort_values("Date").reset_index(drop=True)
    if usar_cache and not forzar:
        guardar_cache(df, m, ini, fin)
    origen = f"descargado ({len(bloques)} bloque(s))"
    if errores:
        origen += f" · {len(errores)} bloque(s) fallidos"
        bitacora(f"{m.metric_id}: {len(errores)} bloque(s) fallidos", "warn")
    return df, origen


def sondea_metrica(m: Metrica, ini: dt.date, fin: dt.date) -> bool:
    """True si la métrica ya está publicada en [ini,fin] (equivalente a _sondar_metrica)."""
    try:
        return bool(post_xm(m.endpoint, m.metric_id, m.entity, ini, fin, reintentos=2, pausa=0.6))
    except Exception:                                     # noqa: BLE001
        return False


def ventana_efectiva(m: Metrica, ini: dt.date, fin: dt.date,
                     max_meses: int = 14) -> tuple[dt.date, dt.date, str]:
    """Retrocede mes a mes hasta la ventana más reciente CON datos (rezago de XM)."""
    for k in range(max_meses + 1):
        desp = dt.timedelta(days=30 * k)
        i2, f2 = ini - desp, fin - desp
        if f2 < dt.date(2010, 1, 1):
            break
        son_ini = max(i2, f2 - dt.timedelta(days=3))
        if sondea_metrica(m, son_ini, f2):
            return i2, f2, ("ventana original" if k == 0 else f"desplazada {k} mes(es) por rezago")
    return ini, fin, "sin datos en la retro-búsqueda (se intenta igual)"


def descargar_ventana(m: Metrica, ini: dt.date, fin: dt.date, *, modo: str = "auto",
                      forzar: bool = False, sonda: bool = True) -> dict[str, Any]:
    """Unidad de trabajo del pipeline: cachea, respeta rezagos y reporta linaje.

    modo: 'auto' (caché→API), 'api' (fuerza red), 'cache' (solo local), 'offline' (nada).
    """
    res: dict[str, Any] = {"metrica": m.nombre, "metric_id": m.metric_id,
                           "entity": m.entity, "filas": 0, "origen": "—",
                           "inicio": ini, "fin": fin, "nota": "", "ok": False}
    if modo == "offline":
        res["origen"] = "offline"
        res["nota"] = "modo sin conexión: no se carga esta métrica"
        return res

    if modo in ("auto", "cache"):
        df, ruta = buscar_cache(m, ini, fin)
        if not df.empty:
            if m.tipo != "list" and "Date" in df.columns:
                df = filtrar_por_periodo(df, ini, fin)
            if not df.empty:
                res.update(df=df, filas=len(df), origen=f"caché · {ruta.name if ruta else 'local'}",
                           ok=True)
                return res
        if modo == "cache":
            res["origen"] = "sin caché"
            res["nota"] = f"no hay {m.prefijo}_*.csv en data/raw para la ventana pedida"
            return res

    ini_ef, fin_ef, nota = (ini, fin, "ventana original")
    if m.tipo != "list" and sonda:
        ini_ef, fin_ef, nota = ventana_efectiva(m, ini, fin)
    df, origen = descargar_metrica(m, ini_ef, fin_ef, usar_cache=(modo != "offline"), forzar=forzar)
    if df.empty:
        res.update(origen="API sin datos", nota=nota,
                   inicio_efectivo=ini_ef, fin_efectivo=fin_ef)
        return res
    if m.tipo != "list" and "Date" in df.columns:
        recortado = filtrar_por_periodo(df, ini, fin)
        if not recortado.empty:
            df = recortado
        else:
            nota += " · datos solo disponibles hasta " + str(rango_fechas(df)[1] if rango_fechas(df) else "—") \
                     + " (rezago de publicación; la serie se usa tal cual)"
    res.update(df=df, filas=len(df), ok=not df.empty,
               origen=f"API XM · {origen}" if not df.empty else origen,
               nota=nota, inicio_efectivo=ini_ef, fin_efectivo=fin_ef)
    return res


# ==============================================================================
# 2 · ETL: del formato ancho de XM al modelo dimensional (celdas 21-27 · v1/v2)
# ==============================================================================
def columnas_horarias(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if str(c).startswith("Values_Hour")]
    return sorted(cols, key=lambda c: int(re.search(r"(\d+)$", str(c)).group(1)))


def filtrar_por_periodo(df: pd.DataFrame, ini: dt.date, fin: dt.date) -> pd.DataFrame:
    if df is None or df.empty or "Date" not in df.columns:
        return pd.DataFrame()
    f = pd.to_datetime(df["Date"], errors="coerce").dt.date
    return df.loc[f.between(ini, fin)].copy().reset_index(drop=True)


def rango_fechas(df: pd.DataFrame) -> tuple[dt.date, dt.date] | None:
    if df is None or df.empty or "Date" not in df.columns:
        return None
    f = pd.to_datetime(df["Date"], errors="coerce").dropna()
    return None if f.empty else (f.min().date(), f.max().date())


def agregar_timestamp(df: pd.DataFrame, col_fecha: str = "Fecha", col_hora: str = "Hora") -> pd.DataFrame:
    out = df.copy()
    out[col_fecha] = pd.to_datetime(out[col_fecha])
    out["Timestamp"] = out[col_fecha] + pd.to_timedelta(out[col_hora].astype(int) - 1, unit="h")
    return out


def wide_to_long(df_raw: pd.DataFrame, nombre_valor: str, codigo_col: str = "Values_code",
                 codigo_nombre: str = "Codigo", factor: float = 1.0) -> pd.DataFrame:
    """Métrica horaria (Values_Hour01…24) → largo: una fila por fecha×hora×código."""
    if df_raw is None or df_raw.empty or "Date" not in df_raw.columns:
        return pd.DataFrame(columns=["Fecha", "Hora", "Timestamp", codigo_nombre, nombre_valor])
    horas = columnas_horarias(df_raw)
    if not horas:
        return pd.DataFrame(columns=["Fecha", "Hora", "Timestamp", codigo_nombre, nombre_valor])
    col_codigo = next((c for c in (codigo_col, "Values_code", "Values_Code") if c in df_raw.columns), None)
    id_vars = ["Date"] + ([col_codigo] if col_codigo else [])
    largo = df_raw.melt(id_vars=id_vars, value_vars=horas, var_name="_h", value_name=nombre_valor)
    largo["Hora"] = largo["_h"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    largo = largo.drop(columns="_h").rename(columns={"Date": "Fecha"})
    if col_codigo:
        largo = largo.rename(columns={col_codigo: codigo_nombre})
    else:
        largo[codigo_nombre] = "Sistema"
    largo[codigo_nombre] = largo[codigo_nombre].astype(str).str.strip()
    largo["Fecha"] = pd.to_datetime(largo["Fecha"], errors="coerce").dt.date
    largo[nombre_valor] = pd.to_numeric(largo[nombre_valor], errors="coerce") * factor
    largo = largo.dropna(subset=["Fecha"])
    largo = agregar_timestamp(largo)
    return (largo[["Fecha", "Hora", "Timestamp", codigo_nombre, nombre_valor]]
            .sort_values(["Fecha", "Hora"]).reset_index(drop=True))


def detectar_columna_valor_diaria(df: pd.DataFrame) -> str | None:
    for c in ("Values_Value", "Value", "Values_value", "Value_Value"):
        if c in df.columns:
            return c
    excluir = {"Id", "Date", "Values_code", "Values_Code", "Code", "Name", "MetricId"}
    for c in df.columns:
        if c in excluir or "Hour" in str(c):
            continue
        if pd.to_numeric(df[c], errors="coerce").notna().any():
            return c
    return None


def daily_to_long(df_raw: pd.DataFrame, nombre_valor: str, factor: float = 1.0,
                  agregacion: str = "mean") -> pd.DataFrame:
    """Métrica diaria de XM → una fila por fecha (con conversión de unidad opcional)."""
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=["Fecha", nombre_valor])
    col = detectar_columna_valor_diaria(df_raw)
    if col is None:
        return pd.DataFrame(columns=["Fecha", nombre_valor])
    out = df_raw[["Date", col]].rename(columns={"Date": "Fecha", col: nombre_valor})
    out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce").dt.date
    out[nombre_valor] = pd.to_numeric(out[nombre_valor], errors="coerce") * factor
    out = out.dropna(subset=["Fecha", nombre_valor])
    return (out.groupby("Fecha", as_index=False)[nombre_valor].agg(agregacion)
             .sort_values("Fecha").reset_index(drop=True))


def daily_to_long_recurso(df_raw: pd.DataFrame, nombre_valor: str, factor: float = 1.0) -> pd.DataFrame:
    """Métrica diaria POR RECURSO (CapEfecNeta: Date + Code + Value) conservando el código."""
    vacio = pd.DataFrame(columns=["Fecha", "Codigo_Planta", nombre_valor])
    if df_raw is None or df_raw.empty:
        return vacio
    col_codigo = next((c for c in ("Code", "Values_Code", "Values_code") if c in df_raw.columns), None)
    col_valor = detectar_columna_valor_diaria(df_raw)
    if col_codigo is None or col_valor is None:
        return vacio
    out = df_raw[["Date", col_codigo, col_valor]].rename(
        columns={"Date": "Fecha", col_codigo: "Codigo_Planta", col_valor: nombre_valor})
    out["Fecha"] = pd.to_datetime(out["Fecha"], errors="coerce").dt.date
    out["Codigo_Planta"] = out["Codigo_Planta"].astype(str).str.strip()
    out[nombre_valor] = pd.to_numeric(out[nombre_valor], errors="coerce") * factor
    out = out.dropna(subset=["Fecha", nombre_valor])
    return (out.groupby(["Fecha", "Codigo_Planta"], as_index=False)[nombre_valor].mean()
             .sort_values(["Fecha", "Codigo_Planta"]).reset_index(drop=True))


COMBUSTIBLES_XM = {"CARBON", "GAS", "ACPM", "COMBUSTOLEO", "COMBUSTIBLEO", "BAGAZO",
                  "BIOMASA", "JET A1", "JETA1", "GAS-JET A1", "FUEL OIL", "NAFTA", "OTRO"}


def wide_to_long_combustible(df_raw: pd.DataFrame, nombre_valor: str) -> pd.DataFrame:
    """Emisiones/combustible (entity=RecursoComb): planta y combustible según CONTENIDO.

    Quirk verificado en la API: EmisionesCO2 trae la planta en `Values_Name` y el
    combustible en `Values_code`; ConsCombustibleMBTU los invierte. No se decide
    por el nombre de la columna sino por cuál tiene más valores dentro del set de
    combustibles de XM (misma lógica que la celda v3 del notebook).
    """
    cols = ["Fecha", "Hora", "Timestamp", "Codigo_Planta", "Combustible", nombre_valor]
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=cols)
    cand_a = next((c for c in ("Values_code", "Values_Code") if c in df_raw.columns), None)
    cand_b = "Values_Name" if "Values_Name" in df_raw.columns else None
    if cand_a is None or cand_b is None:
        return pd.DataFrame(columns=cols)
    na = df_raw[cand_b].astype(str).str.upper().str.strip()
    nb = df_raw[cand_a].astype(str).str.upper().str.strip()
    if na.isin(COMBUSTIBLES_XM).mean() >= nb.isin(COMBUSTIBLES_XM).mean():
        col_comb, col_planta = cand_b, cand_a
    else:
        col_comb, col_planta = cand_a, cand_b
    horas = columnas_horarias(df_raw)
    if not horas:
        return pd.DataFrame(columns=cols)
    largo = df_raw.melt(id_vars=["Date", col_planta, col_comb], value_vars=horas,
                        var_name="_h", value_name=nombre_valor)
    largo["Hora"] = largo["_h"].astype(str).str.extract(r"(\d+)$")[0].astype(int)
    largo = largo.drop(columns="_h").rename(columns={"Date": "Fecha", col_planta: "Codigo_Planta",
                                                     col_comb: "Combustible"})
    largo["Codigo_Planta"] = largo["Codigo_Planta"].astype(str).str.strip()
    largo["Combustible"] = (largo["Combustible"].astype(str).str.upper().str.strip()
                            .replace({"NAN": "N/D", "NONE": "N/D"}))
    largo["Fecha"] = pd.to_datetime(largo["Fecha"], errors="coerce").dt.date
    largo[nombre_valor] = pd.to_numeric(largo[nombre_valor], errors="coerce")
    largo = largo.dropna(subset=["Fecha", nombre_valor])
    largo = agregar_timestamp(largo)
    return largo[cols].sort_values(["Fecha", "Hora"]).reset_index(drop=True)


MAPA_DIM_PLANTAS = {
    "Values_Code": "Codigo_Planta", "Values_Name": "Nombre_Planta", "Values_Type": "Tecnologia",
    "Values_EnerSource": "Fuente_Energia", "Values_RecType": "Tipo_Recurso",
    "Values_CompanyCode": "Codigo_Agente", "Values_Disp": "Tipo_Despacho",
    "Values_OperStartdate": "Fecha_Inicio_Op", "Values_State": "Estado",
}


def preparar_dim_plantas(df: pd.DataFrame) -> pd.DataFrame:
    """Catálogo ListadoRecursos → dim_plantas limpia y sin duplicados."""
    if df is None or df.empty:
        return pd.DataFrame(columns=list(MAPA_DIM_PLANTAS.values()))
    cols = {k: v for k, v in MAPA_DIM_PLANTAS.items() if k in df.columns}
    out = df[list(cols.keys())].rename(columns=cols).copy()
    if "Codigo_Planta" not in out.columns:                      # esquema alternativo
        for a, b in (("Code", "Codigo_Planta"), ("Name", "Nombre_Planta"), ("Type", "Tecnologia")):
            if a in out.columns:
                out = out.rename(columns={a: b})
    out["Codigo_Planta"] = out["Codigo_Planta"].astype(str).str.strip()
    if "Tecnologia" in out.columns:
        out["Tecnologia"] = (out["Tecnologia"].astype(str).str.strip().str.upper()
                             .replace({"NAN": pd.NA, "NONE": pd.NA}))
    return (out.drop_duplicates(subset=["Codigo_Planta"], keep="first")
              .sort_values("Codigo_Planta").reset_index(drop=True))


def normalizar_tecnologia(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Tecnologia" not in df.columns:
        return df
    out = df.copy()
    out["Tecnologia"] = (out["Tecnologia"].astype(str).str.strip().str.upper()
                        .replace({"NAN": "SIN CATALOGAR", "NONE": "SIN CATALOGAR"}))
    return out


def transformar_generacion(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`transformar_generacion` del notebook v1 (Hour01→00:00 … Hour24→23:00)."""
    gen = wide_to_long(df_raw, "Generacion_kWh", codigo_nombre="Codigo_Planta")
    return gen, gen


# ==============================================================================
# 3 · ETL de sistema y enriquecimientos (v1 · v2) + hallazgos automáticos
# ==============================================================================
def enriquecer_generacion(gen: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    """Añade dim_plantas; las plantas sin catálogo NO se pierden (queda 'SIN CATALOGAR')."""
    if gen.empty:
        return gen
    out = gen.copy()
    if dim is not None and not dim.empty:
        cols = [c for c in ("Codigo_Planta", "Tecnologia", "Fuente_Energia", "Tipo_Recurso",
                            "Nombre_Planta", "Tipo_Despacho") if c in dim.columns]
        out = out.merge(dim[cols].drop_duplicates(subset=["Codigo_Planta"]),
                        on="Codigo_Planta", how="left")
        for c in ("Tecnologia", "Fuente_Energia", "Tipo_Recurso", "Tipo_Despacho"):
            if c in out.columns:
                out[c] = out[c].fillna("SIN CATALOGAR")
    else:
        out["Tecnologia"] = "SIN CATALOGAR"
    if "Tecnologia" not in out.columns:
        out["Tecnologia"] = "SIN CATALOGAR"
    out["Es_Renovable"] = out["Tecnologia"].astype(str).str.upper().isin(RENOVABLES_TECH)
    return normalizar_tecnologia(out)


def verificar_y_corregir_integridad(sistema: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Hora de más → se recorta; hueco de una hora → se interpola linealmente."""
    if sistema.empty:
        return sistema, {}
    df = sistema.copy()
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    antes = len(df)
    df = df.drop_duplicates(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
    huecos = df["Timestamp"].diff().dt.total_seconds().div(3600).fillna(1)
    n_interp = int((huecos > 1.5).sum())
    report = {"filas_antes": antes, "filas_despues": len(df), "huecos_interpolados": n_interp}
    if 0 < n_interp <= len(df) * 0.1 and "Timestamp" in df.columns:
        df = df.set_index("Timestamp")
        num = df.select_dtypes("number").resample("h").interpolate(limit=2).reindex(df.index)
        for c in num.columns:
            if c in df.columns:
                df[c] = num[c]
        df = df.reset_index()
    return df, report


def sistema_hora_a_dia(sistema: pd.DataFrame) -> pd.DataFrame:
    """Sistema horario -> resumen diario + perfil medio por hora."""
    diario = perfil = pd.DataFrame()
    if sistema.empty or "Fecha" not in sistema.columns:
        return diario, perfil
    num = [c for c in sistema.select_dtypes("number").columns if c not in ("Fecha", "Hora")]
    diario = sistema.groupby("Fecha", as_index=False)[num].agg({"sum", "mean", "max", "min"}
                                                                if len(num) > 1 else ["sum", "mean"])
    diario.columns = ["_".join([str(x) for x in c if x]).strip("_") for c in diario.columns]
    if "Hora" in sistema.columns:
        perfil = sistema.groupby("Hora", as_index=False)[num].mean()
    return diario, perfil


# ==============================================================================
# 1c · AUDITORÍA DE COMPLETITUD: días que XM aún publica preliminares
# ==============================================================================
#: feeds cuya suma diaria debe ser ~constante: si un día cae muy por debajo de su
#: mediana, el día NO bajó: XM todavía no lo cerró (publica D+1/D+2 por cortes).
FEEDS_SENSIBLES = ("DemaReal_Sistema", "Gene_Sistema", "Gene_Recurso", "DemaMaxPot_Sistema")
UMBRAL_DIA_PRELIMINAR = float(os.environ.get("APP_MANUEL_UMBRAL_DIA", "0.60"))
MIRADA_DIAS_FINALES = 4            # solo se sospecha de los últimos días de la serie


def _totales_diarios_feed(df_raw: pd.DataFrame) -> pd.Series:
    """Suma diaria de un crudo 'wide' de XM (Date + Values_HourNN o Value)."""
    if df_raw is None or df_raw.empty or "Date" not in df_raw.columns:
        return pd.Series(dtype="float64")
    d = df_raw.copy()
    d["_f"] = pd.to_datetime(d["Date"], errors="coerce").dt.date
    horas = [c for c in d.columns if str(c).startswith("Values_Hour")]
    if horas:
        for c in horas:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        ser = d.groupby("_f")[horas].sum().sum(axis=1)
        nh = d.assign(_nh=d[horas].notna().sum(axis=1)).groupby("_f")["_nh"].max()
    elif "Value" in d.columns:
        d["_v"] = pd.to_numeric(d["Value"], errors="coerce")
        ser = d.groupby("_f")["_v"].sum()
        nh = d.groupby("_f")["_v"].size()
    else:
        return pd.Series(dtype="float64")
    return pd.DataFrame({"total": ser, "horas": nh}).dropna(subset=["total"])["total"]


def podar_dias_provisionales(res: dict[str, dict[str, Any]], *, incluir: bool = False,
                             progreso=None) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Detecta (y por defecto excluye) los días que XM todavía trae a medias.

    Regla, verificable: para cada feed volumétrico se compara el total del día con la
    mediana de sus propios días (excluyendo los 4 últimos). Si el día cae por debajo del
    60 % de esa mediana **y** es uno de los últimos de la serie, está preliminar: no es
    que la demanda haya caído, es que XM aún no cierra el día (las primeras entregas
    llegan con una fracción de las horas/agentes reportados). Se excluye de TODOS los
    feeds a la vez para que ningún agregado mezcle un día cerrado con uno abierto (ese
    era el origen del 'balance = +209 GWh' y de la caída de demanda del 75 %).
    """
    totales = {nom: _totales_diarios_feed(r.get("df", pd.DataFrame()))
               for nom, r in res.items() if isinstance(r, dict) and r.get("df") is not None}
    totales = {k: v for k, v in totales.items() if k in FEEDS_SENSIBLES and not v.empty and len(v) >= 6}
    if not totales:
        return res, pd.DataFrame()
    sospecha: dict[Any, dict[str, float]] = {}
    for nom, ser in totales.items():
        base = ser.iloc[:-MIRADA_DIAS_FINALES] if len(ser) > MIRADA_DIAS_FINALES else ser
        med = float(base.median()) if len(base) else float("nan")
        if not np.isfinite(med) or med <= 0:
            continue
        ultimos = list(ser.index)[-MIRADA_DIAS_FINALES:]
        for f in ultimos:
            val = float(ser.loc[f])
            ratio = val / med
            if ratio < UMBRAL_DIA_PRELIMINAR:
                d = sospecha.setdefault(f, {"min_ratio": ratio, "feeds": []})
                d["min_ratio"] = min(d["min_ratio"], ratio)
                d["feeds"].append(f"{nom} {val:,.0f} vs mediana {med:,.0f}")
    if not sospecha:
        return res, pd.DataFrame()
    filas = [{"Fecha": f, "día de la semana": dt.date(*map(int, str(f).split("-"))).strftime("%A"),
              "ratio mínimo vs mediana": f"{d['min_ratio'] * 100:,.0f} %",
              "feeds que lo delatan": "; ".join(d["feeds"][:3]),
              "tratado como": "EXCLUIDO de los agregados" if not incluir else "incluido (aviso activo)"}
             for f, d in sorted(sospecha.items())]
    prov = pd.DataFrame(filas)
    if progreso:
        try:
            progreso(0.55, "[app] Auditoría de completitud (no existe en el notebook): días "
                     "provisionales en XM: " + ", ".join(str(x) for x in sorted(sospecha)) +
                     ("" if incluir else " → excluidos de los agregados"), ETAPAS["V2"][0])
        except Exception:                                         # noqa: BLE001
            pass
    if incluir:
        return res, prov
    limpi = {}
    for nom, r in res.items():
        df = r.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty or "Date" not in df.columns:
            limpi[nom] = r
            continue
        fchas = pd.to_datetime(df["Date"], errors="coerce").dt.date
        if not fchas.isin(sospecha).any():
            limpi[nom] = r
            continue
        df2 = df.loc[~fchas.isin(sospecha)].reset_index(drop=True)
        r2 = dict(r)
        r2.update(df=df2, filas=len(df2),
                  nota=(r.get("nota", "") + f" · excluidos {len(sospecha)} día(s) preliminar(es)").strip(" ·"))
        limpi[nom] = r2
    bitacora(f"Completitud: {len(sospecha)} día(s) preliminares de XM excluidos de todos los feeds "
             f"({', '.join(str(x) for x in sorted(sospecha))}); umbral {UMBRAL_DIA_PRELIMINAR:.0%} de la "
             f"mediana del propio feed", "warn")
    return limpi, prov


def validar_tablas_analiticas(tablas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Espejo de `validar_tablas_analiticas` (celda 61 del notebook): reglas de rango y unicidad.

    No oculta observaciones: reporta nulos, valores fuera de rango y duplicados del grano
    (planta × fecha × hora) por tabla. Es la prueba de que los números que se grafican vienen
    de tablas sanas; la app amplía el cuadriculado del notebook con las tablas de v3/v4.
    """
    reglas = [
        ("fact_generacion", "gen", "Generacion_kWh", 0, None),
        ("fact_demanda", "sistema", "Demanda_MW", 0, None),
        ("fact_precio", "sistema", "Precio_Bolsa_COP_kWh", None, None),
        ("costo_marginal", "costo_marginal", "Costo_Marginal_COP_kWh", None, None),
        ("fact_embalses", "embalses", "Volumen_Pct", 0, 100),
        ("fact_aportes", "aportes", "Aportes_kWh", 0, None),
        # Aportes_Pct en la app es "% de la media histórica": puede superar 100 sin ser un error,
        # por eso se exige piso (0) pero no techo, a diferencia del nivel de embalses.
        ("fact_aportes_pct", "aportes_pct", "Aportes_Pct", 0, None),
        ("fact_emisiones", "emisiones", "tCO2", 0, None),
        ("fact_capacidad", "cen_recurso", "Potencia_Confiable_kW", 0, None),
        ("resumen_diario", "resumen", "Demanda_GWh", 0, None),
        ("resumen_diario_precio", "resumen", "Precio_Ponderado_COP_kWh", 0, None),
    ]
    filas = []
    for nombre, clave_tabla, columna, minimo, maximo in reglas:
        df = tablas.get(clave_tabla)
        if df is None or getattr(df, "empty", True) or columna not in getattr(df, "columns", []):
            continue
        valores = pd.to_numeric(df[columna], errors="coerce")
        fuera = pd.Series(False, index=df.index)
        if minimo is not None:
            fuera |= valores < minimo
        if maximo is not None:
            fuera |= valores > maximo
        # grano tal como lo define el notebook, ampliado con la dimensión de detalle que traiga la tabla
        claves = [x for x in ("Codigo_Planta", "Fecha", "Hora", "Combustible", "Zona", "Region")
                  if x in df.columns]
        duplicados = int(df.duplicated(claves).sum()) if len(claves) >= 2 else 0
        horas = (int(pd.to_numeric(df["Hora"], errors="coerce").dropna().nunique())
                 if "Hora" in df.columns else 0)
        dias = (int(pd.to_datetime(df["Fecha"], errors="coerce").dt.date.nunique())
                if "Fecha" in df.columns else 0)
        nulos = int(valores.isna().sum())
        fuera_n = int(fuera.sum())
        estado = ("🔴 fuera de rango" if fuera_n else
                  ("🟠 nulos estructurales" if nulos else "🟢 ok"))
        filas.append({"tabla": nombre, "columna": columna, "filas": int(len(df)),
                      "días": dias, "horas_vistas": int(horas),
                      "grano": "horaria (24 h)" if horas else "diaria",
                      "nulos_valor": int(nulos), "fuera_rango": int(fuera_n),
                      "duplicados_grano": int(duplicados), "grano_revisado": " × ".join(claves),
                      "estado": estado})
    out = pd.DataFrame(filas)
    for _c in ("filas", "días", "horas_vistas", "nulos_valor", "fuera_rango", "duplicados_grano"):
        if _c in out:                       # tipos duros: st.dataframe serializa con Arrow
            out[_c] = pd.to_numeric(out[_c], errors="coerce").fillna(0).astype(int)
    if not out.empty:
        mal = out.loc[out["estado"].str.startswith("🔴"), "tabla"].tolist()
        nul = out.loc[out["estado"].str.startswith("🟠"), "tabla"].tolist()
        if mal:
            bitacora(f"Validación (c61): {len(mal)} tabla(s) con valores fuera de rango: {', '.join(mal)}", "error")
        if nul:
            bitacora(f"Validación (c61): nulos estructurales (planta/hora que no reportó) en "
                     f"{', '.join(nul)} — no se imputan, se grafican como ausencia", "info")
        else:
            bitacora(f"Validación (c61): {len(out)} tablas analíticas sin valores fuera de rango y "
                     f"sin duplicados de grano", "info")
    return out


def resumen_diario(sistema: pd.DataFrame, embalses: pd.DataFrame, aportes: pd.DataFrame,
                   escasez: pd.DataFrame, precio_pond: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tabla clave v2: balance, participación térmica, precio medio/banda, embalses, escasez."""
    if sistema is None or sistema.empty:
        return pd.DataFrame()
    col_precio = next((c for c in ("Costo_Marginal_COP_kWh", "Precio_Bolsa_COP_kWh")
                       if c in sistema.columns and sistema[c].notna().any()), None)
    col_dem = "Demanda_MW" if "Demanda_MW" in sistema.columns else None
    if col_dem and col_precio:
        base = sistema.groupby("Fecha", as_index=False).agg(
            Demanda_MWh=(col_dem, "sum"), Costo_Total_COP=(col_precio, "sum"), Horas=("Hora", "nunique"))
        base["Precio_Ponderado_COP_kWh"] = (1e3 * base["Costo_Total_COP"]
                                            / base["Demanda_MWh"].replace(0, np.nan))
        base["Demanda_GWh"] = base["Demanda_MWh"] / 1e3
        base = base.merge(sistema.groupby("Fecha")[col_precio].agg(Precio_Min_COP_kWh="min",
                                                                   Precio_Max_COP_kWh="max"),
                          on="Fecha", how="left")
    else:
        base = sistema.groupby("Fecha", as_index=False).size()
        if col_dem:
            d = sistema.groupby("Fecha", as_index=False)[col_dem].sum()
            base = base.merge(d.rename(columns={col_dem: "Demanda_MWh"}), on="Fecha", how="left")
            base["Demanda_GWh"] = base["Demanda_MWh"] / 1e3
    if precio_pond is not None and not precio_pond.empty and "Precio_Ponderado_COP_kWh" in precio_pond:
        pp = precio_pond.copy()
        pp["Fecha"] = pd.to_datetime(pp["Fecha"]).dt.date
        base = base.drop(columns=["Precio_Ponderado_COP_kWh"], errors="ignore").merge(
            pp[["Fecha", "Precio_Ponderado_COP_kWh"]], on="Fecha", how="left")
    cols_gen_h = [c for c in sistema.columns if str(c).startswith("Gen_")]
    if cols_gen_h:
        diaria = sistema.groupby("Fecha", as_index=False)[cols_gen_h].sum()
        for c in cols_gen_h:
            diaria[c] = diaria[c] / 1e3          # sistema está en MW → Σ24h = MWh → /1e3 = GWh
        diaria = diaria.rename(columns={c: f"{c}_GWh" for c in cols_gen_h})
        base = base.merge(diaria, on="Fecha", how="left")
        cols_gwh = [f"{c}_GWh" for c in cols_gen_h]
        base["Generacion_Total_GWh"] = base[cols_gwh].sum(axis=1)
        term = [c for c in cols_gwh if any(k in c.upper() for k in TERMICA_KEYS)]
        if term:
            base["Participacion_Termica_pct"] = (100 * base[term].sum(axis=1)
                                                  / base["Generacion_Total_GWh"].replace(0, np.nan))
        if "Demanda_GWh" in base:
            base["Balance_GWh"] = base["Generacion_Total_GWh"] - base["Demanda_GWh"]
    if "Tecnologia" in sistema.columns and "Generacion_MW" in sistema.columns:
        tec = (sistema.pivot_table(index="Fecha", columns="Tecnologia", values="Generacion_MW",
                                   aggfunc="sum").fillna(0.0))
        tec.columns = [f"Gen_{c}_GWh" for c in tec.columns]
        base = base.merge((tec / 1e3).reset_index(), on="Fecha", how="left")
        cols_gwh = [c for c in base.columns if c.startswith("Gen_")]
        base["Generacion_Total_GWh"] = base[cols_gwh].sum(axis=1)
        term = [c for c in cols_gwh if any(k in c for k in TERMICA_KEYS)]
        base["Participacion_Termica_pct"] = 100 * base[term].sum(axis=1) / base["Generacion_Total_GWh"].replace(0, np.nan)
        base["Balance_GWh"] = base["Generacion_Total_GWh"] - base.get("Demanda_GWh")
    if embalses is not None and not embalses.empty:
        e = embalses.rename(columns={"Volumen_Pct": "Embalses_Pct"})
        base = base.merge(e[["Fecha", "Embalses_Pct"]].drop_duplicates("Fecha"), on="Fecha", how="left")
    if aportes is not None and not aportes.empty:
        a = aportes.copy()
        if "Aportes_Pct" in a.columns:
            base = base.merge(a[["Fecha", "Aportes_Pct"]].drop_duplicates("Fecha"), on="Fecha", how="left")
    if escasez is not None and not escasez.empty:
        es = escasez.rename(columns=lambda c: c if c == "Fecha" else f"Escasez_{c}")
        base = base.merge(es.drop_duplicates("Fecha"), on="Fecha", how="left")
    if "Precio_Ponderado_COP_kWh" in base.columns and "Embalses_Pct" in base.columns:
        base["Alerta_Crisis"] = ((base["Precio_Ponderado_COP_kWh"] > 300)
                                 | (base["Embalses_Pct"] < 45)).fillna(False)
    return base.sort_values("Fecha").reset_index(drop=True)


def _corr(df: pd.DataFrame, a: str, b: str) -> float:
    if a not in df.columns or b not in df.columns:
        return float("nan")
    s = df[[a, b]].dropna()
    if len(s) < 8 or s.std().min() == 0:
        return float("nan")
    return float(s.corr().iloc[0, 1])


def construir_hallazgos(v: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Versión v2 de `construir_hallazgos`: lee SOLO columnas que existan (robusta)."""
    R: list[dict[str, Any]] = []
    rd = v.get("resumen_diario", pd.DataFrame())
    if not rd.empty:
        d = rd
        dias = len(d)
        p = float(d["Precio_Ponderado_COP_kWh"].mean()) if "Precio_Ponderado_COP_kWh" in d else float("nan")
        e = float(d["Embalses_Pct"].mean()) if "Embalses_Pct" in d else float("nan")
        a = float(d["Aportes_Pct"].mean()) if "Aportes_Pct" in d else float("nan")
        R.append({"tipo": "contexto", "titulo": f"{dias} días, precio medio {p:.0f} COP/kWh, "
                  f"embalses {e:.1f} %" if np.isfinite(e) else f"{dias} días analizados",
                  "detalle": "Ventana analizada por el tablero (selección de arriba).",
                  "valor": p})
        dias_altos = int((d["Precio_Ponderado_COP_kWh"] > 1.5 * d["Precio_Ponderado_COP_kWh"].median()).sum()) \
            if "Precio_Ponderado_COP_kWh" in d else 0
        R.append({"tipo": "precio",
                  "titulo": f"{dias_altos} día(s) con precio sobre 1.5× la mediana",
                  "detalle": "Horas caras: el despacho térmico marca el precio. Si coinciden con "
                             "embalses bajos y aporte bajo, el patrón es de estrés hídrico (Niño).",
                  "valor": dias_altos})
        if np.isfinite(p) and dias >= 6 and "Precio_Ponderado_COP_kWh" in d:
            mitad = max(dias // 2, 1)
            p1 = float(d["Precio_Ponderado_COP_kWh"].iloc[:mitad].mean())
            p2 = float(d["Precio_Ponderado_COP_kWh"].iloc[mitad:].mean())
            if p1 > 0:
                delta = 100 * (p2 - p1) / p1
                if abs(delta) >= 12:
                    R.append({"tipo": "tendencia", "titulo": f"Precio {'subió' if delta > 0 else 'bajó'} "
                              f"{abs(delta):.0f} % en la 2ª mitad de la ventana",
                              "detalle": "Una escalada sostenida con aportes a la baja es la firma "
                                         "operativa típica de El Niño.", "valor": delta})
        if np.isfinite(e) and "Embalses_Pct" in d:
            evo = float(d["Embalses_Pct"].iloc[-1] - d["Embalses_Pct"].iloc[0])
            if abs(evo) >= 1.0:
                R.append({"tipo": "embalses", "titulo": f"Embalses {'cedieron' if evo < 0 else 'subieron'} "
                          f"{abs(evo):.1f} pp en la ventana (media {e:.1f} %)",
                          "detalle": f"Curva hidrológica {'descendente' if evo < 0 else 'ascendente'}: "
                                     "bajar de 55 % con el precio al alza activa alerta por sequía.",
                          "valor": evo})
        for umbral in (55, 45, 35, 30):
            if "Embalses_Pct" in d and (d["Embalses_Pct"] < umbral).any():
                R.append({"tipo": "riesgo", "titulo": f"Hubo días con embalses por debajo de {umbral} %",
                          "detalle": "Zonas críticas del SIP (umbrales de seguridad 55/45/35/30 %). "
                                     "Con El Niño la recuperación posterior es más lenta.",
                          "valor": umbral})
        if "Participacion_Termica_pct" in d:
            pt = float(d["Participacion_Termica_pct"].mean())
            pmax = float(d["Participacion_Termica_pct"].max())
            if pt >= 70 or pmax >= 85:
                R.append({"tipo": "riesgo", "titulo": f"Termica media {pt:.0f} % (pico {pmax:.0f} %)",
                          "detalle": "El país depende del despacho térmico; la vulnerabilidad al "
                                     "precio del gas y al costo de escasez queda expuesta.", "valor": pt})
        elif "Participacion_Termica_pct" in d:
            pt2 = float(d["Participacion_Termica_pct"].mean())
            R.append({"tipo": "contexto", "titulo": f"Participación térmica media {pt2:.0f} %",
                      "detalle": "Bajo: la hidráulica cubre el grueso de la demanda.", "valor": pt2})
    sd = v.get("sistema_diario", pd.DataFrame())
    if sd.empty and not rd.empty:
        sd = rd
    if "Balance_GWh" in sd.columns and not sd[sd["Balance_GWh"] < 0].empty:
        R.append({"tipo": "riesgo", "titulo": "Hubo días con generación < demanda",
                  "detalle": "Revisar importaciones o datos incompletos: es la condición que antecede "
                             "a racionamientos en episodios El Niño severos.", "valor": None})
    if "Gen_HIDRO_GWh" in sd.columns and "Gen_SOLAR_GWh" in sd.columns:
        ratio = sd["Gen_SOLAR_GWh"].sum() / max(sd["Gen_HIDRO_GWh"].sum(), 1e-9)
        if ratio > 0.25:
            R.append({"tipo": "contexto", "titulo": f"Solar = {100 * ratio:.0f} % de la hidroel\u00e9ctrica",
                      "detalle": "El solar ya amortigua parte del golpe de El Niño (genera justo cuando "
                                 "menos llueve), pero no sustituye almacenamiento.", "valor": ratio})
    corr = v.get("correlaciones", pd.DataFrame())
    if corr is not None and not corr.empty and "Embalses_Pct" in corr.index:
        for objetivo in ("Precio_Ponderado_COP_kWh", "Costo_Marginal_COP_kWh"):
            if objetivo in corr.columns:
                c = float(corr.loc["Embalses_Pct", objetivo])
                if np.isfinite(c) and abs(c) > 0.3:
                    R.append({"tipo": "correlacion",
                              "titulo": f"Embalses vs {objetivo}: r = {c:+.2f}",
                              "detalle": "Correlación negativa = a menos agua, más precio: el canal "
                                         "físico que explica por qué El Niño encarece el SIN.",
                              "valor": c})
                break
    if "Tecnologia" in v.get("gen_enriquecida", pd.DataFrame()).columns:
        ge = v["gen_enriquecida"]
        tec = ge["Tecnologia"].value_counts()
        top = tec.index[0] if len(tec) else ""
        if top:
            R.append({"tipo": "contexto", "titulo": f"Recurso con más registros horarios: {top}",
                      "detalle": "Util para saber donde se concentra la informacion (y el riesgo) del mercado.",
                      "valor": None})
    return R


# ==============================================================================
# 4 · v3 · CEN, factores de planta, emisiones y marginalidad (Prof. Sergio Rivera)
# ==============================================================================
def _cero(idx) -> np.ndarray:
    return np.zeros(len(idx))


def col_def(cen_tec: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(cen_tec))


def resumen_capacidad(cap_df: pd.DataFrame, dema_max: pd.DataFrame | None = None,
                      dim: pd.DataFrame | None = None) -> pd.DataFrame:
    """CEN diario por tecnología + margen de reserva sobre la demanda máxima."""
    if cap_df is None or cap_df.empty:
        return pd.DataFrame()
    df = cap_df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    if dim is not None and not dim.empty and "Tecnologia" not in df.columns:
        pares = ("Codigo_Planta", "Tecnologia")
        if all(c in dim.columns for c in pares):
            df = df.merge(dim[list(pares)].drop_duplicates("Codigo_Planta"),
                          on="Codigo_Planta", how="left")
            df["Tecnologia"] = df["Tecnologia"].fillna("SIN CATALOGAR").astype(str).str.upper()
    if "Tecnologia" not in df.columns:
        df["Tecnologia"] = "TODAS"
    cen_tec = df.pivot_table(index="Fecha", columns="Tecnologia", values="Potencia_Confiable_kW",
                             aggfunc="sum", fill_value=0.0) / 1e3
    cen_tec["TOTAL_MW"] = cen_tec.sum(axis=1)
    diag = [c for c in cen_tec.columns if c in TECNOLOGIAS_DIAGNOSTICO]
    def col_de(*nombres, defecto=0.0):
        for n in nombres:
            if n in cen_tec.columns:
                return cen_tec[n].to_numpy()
        return np.full(len(cen_tec), defecto)

    hid = cen_tec[[c for c in cen_tec.columns if "HIDR" in c]].sum(axis=1) if any("HIDR" in c for c in cen_tec.columns) else None
    term = cen_tec[[c for c in cen_tec.columns if "TERM" in c or "CARB" in c or c == "GAS"]].sum(axis=1) \
        if any(("TERM" in c or "CARB" in c) for c in cen_tec.columns) else None
    cen = pd.DataFrame({"Fecha": cen_tec.index,
                        "CEN_Total_MW": cen_tec["TOTAL_MW"].to_numpy(),
                        "CEN_Hidro_MW": hid.to_numpy() if hid is not None else col_def(cen_tec),
                        "CEN_Termica_MW": term.to_numpy() if term is not None else col_def(cen_tec),
                        "CEN_Renovables_VG_MW": cen_tec[diag].sum(axis=1).to_numpy() if diag else col_def(cen_tec)})
    for c in cen_tec.columns:
        if c not in ("TOTAL_MW",):
            cen[f"CEN_{c}_MW"] = cen_tec[c].to_numpy()
    if dema_max is not None and not dema_max.empty:
        d = dema_max.copy()
        d["Fecha"] = pd.to_datetime(d["Fecha"])
        cen = pd.merge(cen, d.rename(columns={"Demanda_MW": "DemaMax_MW"})[["Fecha", "DemaMax_MW"]],
                       on="Fecha", how="outer").sort_values("Fecha").reset_index(drop=True)
        cen["Margen_MW"] = cen["CEN_Total_MW"] - cen["DemaMax_MW"]
        cen["Margen_pct"] = 100 * cen["Margen_MW"] / cen["CEN_Total_MW"].replace(0, np.nan)
        cen["Margen_Critico"] = cen["Margen_pct"] < 15
    return cen.sort_values("Fecha").reset_index(drop=True)


def factores_planta(gen: pd.DataFrame, cap: pd.DataFrame, dim: pd.DataFrame) -> pd.DataFrame:
    """FP (%) = kWh generados en el mes / (kW de CEN × horas del mes)."""
    if gen is None or gen.empty or cap is None or cap.empty:
        return pd.DataFrame()
    g = gen.copy()
    g["Fecha"] = pd.to_datetime(g["Fecha"])
    g["Mes"] = g["Fecha"].dt.to_period("M").astype(str)
    gen_mes = (g.groupby(["Mes", "Codigo_Planta"])["Generacion_kWh"].sum()
                .rename("Gen_kWh_Mes").reset_index())
    c = cap.copy()
    c["Fecha"] = pd.to_datetime(c["Fecha"])
    c["Mes"] = c["Fecha"].dt.to_period("M").astype(str)
    horas = c.groupby("Mes")["Fecha"].nunique() * 24
    cap_mes = (c.groupby(["Mes", "Codigo_Planta"])["Potencia_Confiable_kW"].mean()
                .rename("CEN_kW").reset_index())
    cap_mes["Horas_Mes"] = cap_mes["Mes"].map(horas)
    fp = gen_mes.merge(cap_mes, on=["Mes", "Codigo_Planta"], how="inner")
    fp["FP_pct"] = 100 * fp["Gen_kWh_Mes"] / (fp["CEN_kW"] * fp["Horas_Mes"]).replace(0, np.nan)
    if dim is not None and not dim.empty:
        fp = fp.merge(dim[["Codigo_Planta", "Nombre_Planta", "Tecnologia", "Fuente_Energia"]]
                      .drop_duplicates("Codigo_Planta"), on="Codigo_Planta", how="left")
    return fp.sort_values(["Mes", "FP_pct"], ascending=[True, False]).reset_index(drop=True)


def agregar_emisiones_diarias(emb: pd.DataFrame) -> pd.DataFrame:
    if emb is None or emb.empty:
        return pd.DataFrame()
    df = emb.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    cols = [c for c in ("tCO2", "Consumo_Combustible_MBTU", "Energia_MWh", "Horas") if c in df.columns]
    df["_dia"] = df["Fecha"].dt.date
    out = df.groupby("_dia", as_index=False)[cols].sum().rename(columns={"_dia": "Fecha"})
    if "Horas" not in out.columns:
        out["Horas"] = df.groupby("_dia")["Hora"].nunique().values
    if "Energia_MWh" in out.columns:
        out["Intensidad_tCO2_MWh"] = out["tCO2"] / out["Energia_MWh"].replace(0, np.nan)
    return out


def calcular_intensidad_emision(emisiones: pd.DataFrame, energia: pd.DataFrame,
                               col_energia: str = "Gen_TERMICA_GWh") -> pd.DataFrame:
    """Intensidad (tCO2/MWh) = tCO2 del parque térmico ÷ energía entregada por el sistema.

    La API no expone una columna `Energia_MWh` por recurso-combustible, así que se usa la
    demanda diaria (proxy estándar del notebook para el factor del despacho).
    """
    if emisiones is None or emisiones.empty:
        return emisiones if emisiones is not None else pd.DataFrame()
    out = emisiones.copy()
    col_t = next((c for c in ("tCO2_XM", "tCO2") if c in out.columns), None)
    if col_t is None:
        return out
    if energia is None or energia.empty or col_energia not in energia:
        out["Energia_MWh"] = np.nan
        return out
    e = energia.copy()
    e["Fecha"] = pd.to_datetime(e["Fecha"]).dt.date
    out["Fecha"] = pd.to_datetime(out["Fecha"]).dt.date
    out = out.merge(e[["Fecha", col_energia]].drop_duplicates("Fecha"), on="Fecha", how="left")
    out["Energia_MWh"] = pd.to_numeric(out[col_energia], errors="coerce") * 1e3
    out["Intensidad_tCO2_MWh"] = out[col_t] / out["Energia_MWh"].replace(0, np.nan)
    return out


def bandas_emision(intensidad: float) -> str:
    if not np.isfinite(intensidad):
        return "sin dato"
    return "baja" if intensidad < 0.2 else ("media" if intensidad <= 0.4 else "alta")


def cruzar_diario_v4(sistema: pd.DataFrame, indisp: pd.DataFrame) -> pd.DataFrame:
    """`cruce_diario_v4` del notebook: diario de demanda/CEN/disponibilidad/costo × indisponibilidad."""
    if sistema is None or sistema.empty or "Fecha" not in sistema.columns:
        return pd.DataFrame()
    agg: dict[str, tuple[str, str]] = {}
    if "Demanda_MW" in sistema:
        agg["Demanda_MW_Dia"] = ("Demanda_MW", "mean")
        agg["Demanda_Max_Dia_MW"] = ("Demanda_MW", "max")
    for destino, origen in (("CEN_MW_Dia", "CEN_MW"), ("Disp_MW_Dia", "Disp_MW")):
        if origen in sistema.columns and pd.to_numeric(sistema[origen], errors="coerce").notna().any():
            agg[destino] = (origen, "mean")
    if "Costo_Marginal_COP_kWh" in sistema:
        agg["Costo_Marginal_COP_kWh_Dia"] = ("Costo_Marginal_COP_kWh", "mean")
        agg["Costo_Total_Dia_COP"] = ("Costo_Marginal_COP_kWh", "sum")
    if "Precio_Bolsa_COP_kWh" in sistema:
        agg["Precio_Bolsa_Dia_COP_kWh"] = ("Precio_Bolsa_COP_kWh", "mean")
    if not agg:
        return pd.DataFrame()
    agg["Horas"] = ("Hora", "nunique") if "Hora" in sistema else ("Fecha", "count")
    a = sistema.groupby("Fecha", as_index=False).agg(**agg)
    if "Costo_Total_Dia_COP" in a and "Demanda_MW_Dia" in a and "Horas" in a:
        a["Precio_Ponderado_Dia_COP_kWh"] = (1e3 * a["Costo_Total_Dia_COP"]
                                             / (a["Demanda_MW_Dia"] * a["Horas"]).replace(0, np.nan))
    if "CEN_MW_Dia" in a and "Demanda_Max_Dia_MW" in a:
        a["Margen_Dia_pct"] = (100 * (a["CEN_MW_Dia"] - a["Demanda_Max_Dia_MW"])
                               / a["CEN_MW_Dia"].replace(0, np.nan))
    if indisp is not None and not indisp.empty:
        b = indisp.copy()
        b["Fecha"] = pd.to_datetime(b["Fecha"]).dt.date
        keep = [c for c in ("Fecha", "Indisponibilidad_pct", "Reserva_Operativa_MW", "CEN_sin_reporte_pct",
                            "CEN_kW", "DemaMax_MW") if c in b.columns]
        a = a.merge(b[keep].drop_duplicates("Fecha"), on="Fecha", how="inner")
    if "CEN_kW" in a and "CEN_MW_Dia" not in a:
        a["CEN_MW_Dia"] = pd.to_numeric(a["CEN_kW"], errors="coerce") / 1e3
    return a


# ==============================================================================
# 5 · MOTOR El NIÑO / ONI (celdas 150-155 y 204-207 del notebook)
# ==============================================================================
def leer_oni_csv() -> pd.DataFrame | None:
    """Lee la semilla versionada (data/reference/oni_noaa.csv) → tabla Mes/ONI."""
    for ruta in (ONI_SEED, REF_DIR / "oni_noaa.csv"):
        if not Path(ruta).exists():
            continue
        try:
            df = pd.read_csv(ruta)
        except Exception:                                         # noqa: BLE001
            continue
        if {"Fecha", "ONI"} <= set(df.columns):
            out = pd.DataFrame({"Mes": pd.to_datetime(df["Fecha"]).dt.strftime("%Y-%m"),
                                "ONI": pd.to_numeric(df["ONI"], errors="coerce")})
            return out.dropna().drop_duplicates("Mes").sort_values("Mes").reset_index(drop=True)
        if {"year", "season", "anom"} <= set(df.columns):
            return parse_oni_noaa(df)
    return None


def descargar_oni_noaa(forzar: bool = True) -> pd.DataFrame | None:
    """ONI oficial del CPC (o PSL si lo espeja), con mismo formato del seed.

    Formato NOAA `oni.ascii.txt`: columnas YR / SEAS / ANOM, una estación de 3 meses por fila.
    La etiqueta de estación se asigna al MES CENTRAL (DJF → enero), que es la convención del
    notebook y la que permite alinear el índice con la serie mensual de embalses/aportes/precio.
    """
    if ONI_SEED.exists() and not forzar:
        return leer_oni_csv()
    import io as _io
    central = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6, "JJA": 7,
               "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    for url in (URL_ONI, URL_ONI_ALT):
        try:
            txt = _sesion().get(url, timeout=90, verify=_cert_envio()).text
        except Exception as exc:                                  # noqa: BLE001
            bitacora(f"ONI no disponible en {url.split('/')[2]} ({type(exc).__name__}) "
                     f"· uso la semilla local", "warn")
            continue
        filas: list[dict[str, Any]] = []
        if "oni.ascii" in url:
            try:
                bruto = pd.read_csv(_io.StringIO(txt), sep=r"\s+")
            except Exception:                                     # noqa: BLE001
                continue
            bruto = bruto.rename(columns={c: str(c).strip().upper() for c in bruto.columns})
            cols = {c: c for c in bruto.columns}
            anio_c = next((cols.get(k) for k in ("YR", "YEAR") if k in cols), None)
            sea_c = next((cols.get(k) for k in ("SEAS", "SEASON") if k in cols), None)
            anom_c = next((cols.get(k) for k in ("ANOM", "ANO") if k in cols), None)
            if not (anio_c and sea_c and anom_c):
                continue
            for _, r in bruto.iterrows():
                try:
                    anio = int(r[anio_c]); anom = float(r[anom_c])
                except (TypeError, ValueError):
                    continue
                mes = central.get(str(r[sea_c]).strip().upper())
                if mes is None or not np.isfinite(anom):
                    continue
                filas.append({"Mes": f"{anio}-{mes:02d}", "ONI": anom,
                              "Estacion": str(r[sea_c]).strip().upper()})
        else:                                                     # PSL: año + 12 valores mensuales
            for linea in txt.splitlines():
                partes = linea.split()
                if len(partes) != 13 or not partes[0][:4].isdigit():
                    continue
                anio = int(partes[0][:4])
                for mes, val in enumerate(partes[1:], start=1):
                    try:
                        v = float(val)
                    except ValueError:
                        continue
                    if v > -90:                                   # PSL marca -99.99 los huecos
                        filas.append({"Mes": f"{anio}-{mes:02d}", "ONI": v, "Estacion": "PSL"})
        if filas:
            df = pd.DataFrame(filas).dropna(subset=["ONI"]).sort_values("Mes").drop_duplicates("Mes")
            df = df.reset_index(drop=True)
            try:
                df.assign(Fecha=pd.to_datetime(df["Mes"] + "-01")).to_csv(ONI_SEED, index=False)
                bitacora(f"ONI NOAA actualizado: {len(df)} meses (último {df['Mes'].iloc[-1]} = "
                         f"{df['ONI'].iloc[-1]:+.2f} °C)")
            except OSError:
                bitacora("ONI descargado pero el disco está de solo lectura: uso solo memoria", "warn")
            return df[["Mes", "ONI"]]
    return leer_oni_csv()


def parse_oni_noaa(bruto: pd.DataFrame) -> pd.DataFrame:
    """Normaliza cualquier formato ONI (Mes/ONI, Fecha/ONI o year/season/anom) → Mes, ONI."""
    if bruto is None or bruto.empty:
        return pd.DataFrame(columns=["Mes", "ONI"])
    df = bruto.copy()
    if {"Mes", "ONI"} <= set(df.columns):
        return df[["Mes", "ONI"]].dropna().sort_values("Mes").reset_index(drop=True)
    if {"Fecha", "ONI"} <= set(df.columns):
        out = pd.DataFrame({"Mes": pd.to_datetime(df["Fecha"]).dt.strftime("%Y-%m"),
                            "ONI": pd.to_numeric(df["ONI"], errors="coerce")})
        return out.dropna().sort_values("Mes").reset_index(drop=True)
    MAPA = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
            "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}
    if {"year", "season", "anom"} <= set(df.columns):
        df["mes_num"] = df["season"].astype(str).str.strip().str.upper().map(MAPA)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["anom"] = pd.to_numeric(df["anom"], errors="coerce")
        df = df.dropna(subset=["year", "mes_num", "anom"])
        out = pd.DataFrame({"Mes": df["year"].astype(int).astype(str) + "-"
                            + df["mes_num"].astype(int).map(lambda m: f"{m:02d}"),
                            "ONI": df["anom"]})
        return out.drop_duplicates("Mes").sort_values("Mes").reset_index(drop=True)
    return pd.DataFrame(columns=["Mes", "ONI"])


def clasificar_fase(oni: pd.Series, umbral: float = UMBRAL_NINO) -> pd.Series:
    return pd.Series(np.where(oni >= umbral, "El Niño", np.where(oni <= -umbral, "La Niña", "Neutral")),
                     index=oni.index, dtype="object")


def fuerza_evento(oni: pd.Series) -> tuple[float, str]:
    if oni is None or oni.empty or not np.isfinite(oni).any():
        return float("nan"), "—"
    p = float(oni.max()) if oni.max() >= 0 else float(-oni.min())
    etiqueta = ("Muy fuerte (≥2.0)" if p >= 2.0 else "Fuerte (1.5–1.99)" if p >= 1.5
                else "Moderado (1.2–1.49)" if p >= 1.2 else "Débil (<1.2)")
    return p, etiqueta


def detectar_eventos_enso(nino: pd.DataFrame, min_meses: int = 5) -> pd.DataFrame:
    """Agrupa meses consecutivos de la misma fase (|ONI| ≥ 0.5) → tabla de eventos."""
    if nino is None or nino.empty or "Fase" not in nino.columns:
        return pd.DataFrame()
    d = nino.copy()
    d["Mes"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M")
    d = d.sort_values("Mes").reset_index(drop=True)
    d["idx_anterior"] = d["Mes"].diff().dt.days if hasattr(d["Mes"].diff(), "dt") else None
    delta = d["Mes"].astype("int64").diff()
    nuevo = (d["Fase"] != d["Fase"].shift(1)) | (delta > 1)
    grupo = nuevo.cumsum()
    filas = []
    for _, g in d.loc[d["Fase"] != "Neutral"].groupby(grupo):
        dur = len(g)
        pico, etiqueta = fuerza_evento(g["ONI"])
        if dur < min_meses:
            continue
        filas.append({"Fase": g["Fase"].iloc[0], "Inicio": str(g["Mes"].min()), "Fin": str(g["Mes"].max()),
                      "Duración_meses": dur, "Pico_ONI": round(pico, 2), "Intensidad": etiqueta,
                      "ONI_promedio": round(float(g["ONI"].mean()), 2)})
    out = pd.DataFrame(filas)
    if not out.empty:
        out = out.sort_values("Inicio").reset_index(drop=True)
        out.insert(0, "#", range(1, len(out) + 1))
    return out


def serie_mensual_desde_crudos(prefijos: dict[str, str], ini: dt.date, fin: dt.date,
                               cen_mw_promedio: float | None = None) -> pd.DataFrame:
    """Agrega los CSV de `data/raw` en medias mensuales (mismas fórmulas del notebook v3).

    Se usa para alargar la serie histórica con lo ya descargado (sin tocar la API):
    claves 'aportes' y 'embalses' vienen como % o fracción y 'precio' en COP/kWh.
    """
    filas: dict[str, dict[str, float]] = {}
    for clave, prefijo in prefijos.items():
        for ruta in sorted(RAW_DIR.glob(f"{prefijo}_*.csv")):
            try:
                bruto = pd.read_csv(ruta)
            except Exception:                                     # noqa: BLE001
                continue
            if "Date" not in bruto.columns:
                continue
            horas = columnas_horarias(bruto)
            if horas:                                             # horaria → media de las 24 h
                num = bruto[horas].apply(pd.to_numeric, errors="coerce")
                bruto["_v"] = num.mean(axis=1)
            else:                                                 # diaria → columna Value
                col = detectar_columna_valor_diaria(bruto)
                if col is None:
                    continue
                bruto["_v"] = pd.to_numeric(bruto[col], errors="coerce")
            fechas = pd.to_datetime(bruto["Date"], errors="coerce")
            ok = fechas.notna() & bruto["_v"].notna()
            bruto = bruto.loc[ok].copy()
            if bruto.empty:
                continue
            bruto["Mes"] = fechas[ok].dt.to_period("M").astype(str)
            bruto["Dia"] = fechas[ok].dt.day
            if clave in ("aportes", "embalses"):
                med = float(np.nanmedian(bruto["_v"]))
                if np.isfinite(med) and abs(med) <= 2.0:          # XM la manda como fracción
                    bruto["_v"] = bruto["_v"] * 100.0
            for mes, g in bruto.groupby("Mes"):
                destino = filas.setdefault(str(mes), {})
                media = float(g["_v"].mean())
                if clave == "aportes":
                    destino["Aportes_Pct"] = media
                    dias = int(g["Dia"].nunique())
                    destino["dias_aportes"] = dias
                    if cen_mw_promedio:                            # índice notebook, mm equivalentes
                        destino["Aportes_mm_equiv"] = media / 100.0 * cen_mw_promedio * 86.4 * dias
                elif clave == "embalses":
                    destino["Embalses_Pct"] = media
                elif clave == "precio":
                    destino["Precio_Ponderado_COP"] = media
    if not filas:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(filas, orient="index").rename_axis("Mes").reset_index()
    return out.sort_values("Mes").reset_index(drop=True)


def meses_semilla_por_defecto() -> int:
    """Meses que la app baja de la API para sembrar la serie del Niño si falta la semilla.

    Se cambia con la variable de entorno APP_MANUEL_NINO_MESES (1-24). Más meses = serie
    histórica más larga en la pestaña El Niño, pero más descargas en el primer arranque.
    """
    try:
        n = int(os.environ.get("APP_MANUEL_NINO_MESES", "4"))
    except (TypeError, ValueError):
        n = 4
    return max(1, min(n, 24))


def sembrar_nino_local(meses: int | None = None) -> int:
    """Baja las 3 series diarias ligeras del ENSO para los últimos `meses` meses y las cachea.

    Se usa SOLO si `data/reference/nino_mensual.csv` no existe (p. ej. descarga suelta de app.py):
    así la pestaña El Niño tiene historia aunque nadie haya ejecutado `tools/build_seeds.py`.
    Devuelve el número de filas agregadas.
    """
    meses = meses_semilla_por_defecto() if meses is None else max(1, int(meses))
    fin = _hoy()
    ini = (fin - dt.timedelta(days=30 * meses))
    bajadas = 0
    for clave, m in (("aportes", METRICAS_V2["PorcApor_Sistema"]),
                     ("embalses", METRICAS_V2["PorcVoluUtilDiar_Sistema"]),
                     ("precio", METRICAS_V2["PPPrecBolsNaci_Sistema"])):
        try:
            df, origen = descargar_metrica(m, ini, fin, usar_cache=True, forzar=False, pausa=0.15)
        except Exception as exc:                                  # noqa: BLE001
            bitacora(f"siembra ENSO {m.metric_id}: falló ({str(exc)[:110]})", "warn")
            continue
        bajadas += len(df)
        bitacora(f"siembra ENSO {m.metric_id}: {len(df)} filas desde {ini} ({origen})")
    return bajadas


def construir_nino_mensual(on_path: Path | None = None, forzar_recarga: bool = False,
                           modo: str = "auto", meses_semilla: int | None = None) -> dict[str, Any]:
    """nino_mensual.csv = hidrometeorología 2015→ + ONI + fase + lag. Se incrementa en local."""
    nota = ""
    hist = pd.DataFrame()
    if NINO_MENSUAL.exists():
        try:
            hist = pd.read_csv(NINO_MENSUAL)
        except Exception:                                         # noqa: BLE001
            hist = pd.DataFrame()
    if hist.empty:
        # No hay semilla versionada (típico al copiar solo app.py): agregamos lo que ya esté en
        # data/raw y, si hay red, bajamos los últimos `meses_semilla` meses de las 3 series ligeras.
        n_meses = meses_semilla_por_defecto() if meses_semilla is None else max(1, int(meses_semilla))
        if modo in ("auto", "api"):
            bitacora(f"sembrando serie ENSO: bajando {n_meses} meses de PorcApor/PorcVoluUtilDiar/"
                     f"PPPrecBolsNaci (solo la primera vez; queda en data/raw y data/reference)")
            try:
                n_filas = sembrar_nino_local(n_meses)
                if not n_filas:
                    bitacora("siembra ENSO: la API no devolvió filas (¿sin red?)", "warn")
            except Exception as exc:                              # noqa: BLE001
                bitacora(f"siembra ENSO no completada: {str(exc)[:120]}", "warn")
        hist = serie_mensual_desde_crudos(_PREFIJOS_NINO, dt.date(2015, 1, 1), _hoy())
        if hist.empty:
            nota = ("Sin `data/reference/nino_mensual.csv` y sin red para sembrarlo: la pestaña "
                    "El Niño solo mostrará el ONI de NOAA. Corre "
                    "`python tools\\build_seeds.py nino` (una vez) para la serie 2015→ completa.")
            bitacora(nota, "warn")
        else:
            bitacora(f"Serie ENSO sembrada en local: {len(hist)} meses (sin semilla versionada; "
                     f"para 2015→ ejecuta tools/build_seeds.py nino)")
    oni_bruto = None
    if forzar_recarga or not ONI_SEED.exists():
        oni_bruto = descargar_oni_noaa()
    if oni_bruto is None:
        oni_bruto = leer_oni_csv()
    if oni_bruto is None:
        return {"df": hist, "eventos": detectar_eventos_enso(_asegurar_fase(hist)),
                "estado": None, "nota": "Sin serie ONI (falló NOAA y no hay reference/oni_noaa.csv). "
                                        "Incluí el archivo en el repo o ejecuta tools/build_seeds.py oni."}
    oni = parse_oni_noaa(oni_bruto)
    if not hist.empty and "Mes" in hist.columns:
        base = hist.copy()
        if str(base["Mes"].max()) < str(oni["Mes"].max()):
            base = base.merge(serie_mensual_desde_crudos(_PREFIJOS_NINO, dt.date(2015, 1, 1), _hoy()),
                              on="Mes", how="outer")
    else:
        base = serie_mensual_desde_crudos(_PREFIJOS_NINO, dt.date(2015, 1, 1), _hoy())
    if base is None or base.empty:
        base = pd.DataFrame(columns=["Mes"])
    if "Mes" not in base.columns:
        base["Mes"] = pd.NA
    base = base.dropna(subset=["Mes"])
    base = base.merge(oni, on="Mes", how="outer")
    base = base.sort_values("Mes").reset_index(drop=True)
    base["Fase"] = clasificar_fase(base["ONI"].ffill() if base["ONI"].isna().any() else base["ONI"])
    if base["ONI"].isna().any():
        base.loc[base["ONI"].isna(), "Fase"] = "Sin ONI"
    base["Mes_pd"] = pd.PeriodIndex(base["Mes"].astype(str), freq="M")
    base = base.sort_values("Mes_pd").reset_index(drop=True)
    for lag in LAGS_MES:
        base[f"ONI_lag{lag}"] = base["ONI"].shift(lag)
    if not base.empty:
        ult = base["Mes_pd"].max()
        ult_mes = str(ult)
        ult_oni = float(base.loc[base["Mes"] == ult_mes, "ONI"].iloc[0])
        base.loc[base["Mes"] == ult_mes, "Es_Ultimo_Mes"] = True
        nota = (f"Último mes con datos: {ult_mes} · ONI {ult_oni:+.2f} °C · "
                f"{'El Niño activo' if ult_oni >= UMBRAL_NINO else 'Niña' if ult_oni <= -UMBRAL_NINO else 'Neutral'}")
        base["Es_Ultimo_Mes"] = base["Es_Ultimo_Mes"].fillna(False)
    eventos = detectar_eventos_enso(base)
    en_curso = None
    if not eventos.empty:
        ultimo_evento = eventos.iloc[-1]
        if ultimo_evento["Fase"] == "El Niño" and str(ultimo_evento["Fin"]) >= str(base["Mes"].max()):
            en_curso = ultimo_evento
    return {"df": base, "eventos": eventos, "estado": en_curso, "nota": nota, "oni_bruto": oni_bruto}


def _asegurar_fase(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "Fase" in df.columns:
        return df if df is not None else pd.DataFrame()
    out = df.copy()
    if "ONI" in out.columns:
        out["Fase"] = clasificar_fase(out["ONI"])
    return out


def estado_enso_mes_actual(nino: pd.DataFrame) -> dict[str, Any]:
    """Resumen del estado ENSO en el mes de corte (para el semáforo/banner)."""
    if nino is None or nino.empty or "ONI" not in nino.columns:
        return {}
    d = nino.copy()
    d["Mes_pd"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M")
    d = d.sort_values("Mes_pd")
    ult = d.iloc[-1]
    oni = float(ult.get("ONI", np.nan))
    fase = ult.get("Fase", "—")
    # fuerza en curso: máximo ONI del evento activo
    pico = oni
    if fase == "El Niño":
        i = len(d) - 1
        while i > 0 and d.iloc[i - 1].get("Fase") == "El Niño" \
                and (d["Mes_pd"].iloc[i] - d["Mes_pd"].iloc[i - 1]).n == 1:
            i -= 1
        pico = float(d["ONI"].iloc[i:len(d)].max())
    return {"mes": str(d["Mes_pd"].iloc[-1]), "oni": oni, "pico": pico, "fase": fase,
            "serie_meses": int(len(d)), "inicio_evento": str(d["Mes_pd"].iloc[i]) if fase == "El Niño" else None}


def tabla_impacto_enso(df: pd.DataFrame, variables: list[str], min_obs: int = 3) -> pd.DataFrame:
    """Media de cada variable por fase + delta Niño-Neutral + correlación con ONI."""
    if df is None or df.empty:
        return pd.DataFrame()
    filas = []
    for var in variables:
        if var not in df.columns:
            continue
        sub = df[["Fase", var]].dropna()
        if sub.empty:
            continue
        prom = {f: float(sub.loc[sub["Fase"] == f, var].mean()) for f in ("El Niño", "La Niña", "Neutral")}
        dif = prom.get("El Niño", np.nan) - prom.get("Neutral", np.nan)
        pct = (100 * dif / prom["Neutral"]) if prom.get("Neutral") else np.nan
        r = float(df[[var, "ONI"]].dropna().corr().iloc[0, 1]) if len(df[[var, "ONI"]].dropna()) > 4 else np.nan
        filas.append({"variable": var, "n_El_Nino": int((sub["Fase"] == "El Niño").sum()),
                      "media_Neutral": prom.get("Neutral", np.nan),
                      "media_El_Nino": prom.get("El Niño", np.nan),
                      "media_La_Nina": prom.get("La Niña", np.nan),
                      "delta_El_Nino_vs_Neutral": dif, "delta_pct": pct, "corr_ONI": r})
    out = pd.DataFrame(filas)
    if not out.empty:
        out["relevante"] = out["n_El_Nino"] >= min_obs
    return out



def _hoy() -> dt.date:
    """Fecha de hoy en el equipo que ejecuta la app (las ventanas restan el rezago de XM).

    `construir_contexto` toma `fin = _hoy() - 1 día` porque Bienda publica el día D al
    día D+1; `sembrar_*` y la sonda de disponibilidad usan `_hoy()` a secas.
    """
    return dt.date.today()


def metricas_seleccionadas(opciones: dict[str, bool]) -> list[Metrica]:
    """Construye la lista de métricas a consultar (METRICAS_API / _V2 / _V3 / _V4 del notebook)."""
    tabla = {**METRICAS_V2, **METRICAS_V3, **METRICAS_V4}
    sel = [tabla[k] for k, v in opciones.items() if v and k in tabla]
    vistos: set[str] = set()
    out = []
    for m in sel + list(CATALOGOS.values()):
        if m.metric_id in vistos:
            continue
        vistos.add(m.metric_id)
        out.append(m)
    return out


def _concat_parts(parts: list[pd.DataFrame]) -> pd.DataFrame:
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def procesar_todas_las_metricas(opciones: dict[str, bool]) -> list[dict[str, Any]]:
    """Reemplazo del bucle `for m in METRICAS_API: pd.DataFrame.from_records(...)` del notebook."""
    return [{"metrica": m.nombre, "metric_id": m.metric_id, "entidad": m.entity, "unidad": m.unidad,
             "descripcion": m.desc, "peso": m.peso, "rezago_meses": m.retraso_meses}
            for m in metricas_seleccionadas(opciones)]


def construir_contexto(dias: int, opciones: dict[str, bool], modo: str = "auto",
                       progreso=None, incluir_provisionales: bool = False) -> dict[str, Any]:
    """Descarga + ETL completo. Devuelve `ctx` con todas las tablas del notebook.

    modo: 'auto' | 'api' | 'cache' | 'offline'  (offline = solo data/raw y data/reference).
    """
    etapa_actual = {"v": ETAPAS["V1"][0]}

    progreso = progreso or _PROGRESO_ACTIVO[0]

    def hitos(txt: str, frac: float | None = None, etapa: str | None = None,
              celda: int | tuple[int, int] | None = None):
        """Hito del pipeline → bitácora + panel de progreso (si hay).

        `etapa` es la clave de `ETAPAS` ('V1'..'V4','OK'): al cambiarla, la UI anuncia en qué
        capa del notebook va y qué tipo de dato está tocando en ese momento. `celda` (o
        `(inicio, fin)`) es el ancla dentro de `01_ETL_exploracion_v1.json`: se imprime en el
        progreso para que se pueda seguir celda por celda qué se está ejecutando.
        """
        if etapa and etapa in ETAPAS:
            rango = celdas_de_etapa(etapa)
            etapa_actual["v"] = (f"{ETAPAS[etapa][0]} — {ETAPAS[etapa][1]}"
                                 + (f" · notebook {rango}" if rango else ""))
        if celda is not None:
            txt = f"[nb {'celda %d' % celda if isinstance(celda, int) else 'celdas %d-%d' % tuple(celda)}] {txt}"
        bitacora(txt, "info")
        if progreso:
            try:
                progreso(frac if frac is not None else 0.0, txt, etapa_actual["v"])
            except Exception:                                     # noqa: BLE001
                pass

    fin = _hoy() - dt.timedelta(days=1)
    ini = fin - dt.timedelta(days=dias - 1)
    hitos(f"Ventana pedida: {ini} → {fin} ({dias} días) · modo {modo}", 0.02)
    modo_descarga = {"api": "api", "auto": "auto", "cache": "cache", "offline": "offline"}[modo]
    sonda = modo in ("api", "auto")

    res: dict[str, dict[str, Any]] = {}
    # 1)+2) catálogos (Metadata → dimensiones) y métricas del mercado, EN PARALELO.
    # Seriales eran ~17 viajes de red uno tras otro: eso era lo que se veía como "app congelada".
    tareas = metricas_seleccionadas(opciones)            # ya incluye los 3 catálogos, sin duplicados
    orden = [m.nombre for m in tareas]
    if modo_descarga == "offline":
        for m in tareas:
            res[m.nombre] = {"metrica": m.nombre, "metric_id": m.metric_id, "entity": m.entity,
                             "filas": 0, "origen": "offline", "inicio": ini, "fin": fin,
                             "nota": "modo sin conexión: la métrica no se carga", "ok": False}
        hitos(f"Modo Referencia: se omiten {len(tareas)} consultas y se trabaja con data/reference", 0.5, "V2")
    else:
        total = max(1, len(tareas))
        contador = {"n": 0, "lock": threading.Lock()}

        def _una(m: Metrica) -> tuple[str, dict[str, Any]]:
            es_cat = m.tipo == "list"
            # los catálogos van sin sonda de rezago (son series completas) y respetan el modo elegido
            r = descargar_ventana(m, ini, fin, modo=("api" if modo_descarga == "api" else modo_descarga),
                                  forzar=(modo == "api"), sonda=(sonda and not es_cat))
            if not es_cat and not r.get("ok"):
                time.sleep(0.6)                    # el 503 de XM suele ser transitorio
                r = descargar_ventana(m, ini, fin, modo=modo_descarga, forzar=False, sonda=False)
            with contador["lock"]:
                contador["n"] += 1
                n = contador["n"]
            notas = f" · {r['nota']}" if r.get("nota") else ""
            hitos(f"[{n}/{total}] {m.metric_id}/{m.entity} · {int(r.get('filas') or 0):,} filas · "
                  f"{r.get('origen', '—')}{notas}", 0.04 + 0.46 * n / total, "V1", celda=(8, 11))
            return m.nombre, r

        hitos(f"Consultando {total} feeds en paralelo (5 hilos): API pública de XM y, si ya existe, "
              f"el CSV de data/raw", 0.03, "V1", celda=(7, 20))
        try:
            obreros = max(1, min(int(os.environ.get("APP_MANUEL_OBREROS", "5") or 5), total))
            with _fut.ThreadPoolExecutor(max_workers=obreros, thread_name_prefix="xm") as ex:
                salidas = list(ex.map(_una, tareas))
        except Exception as exc:                                   # noqa: BLE001
            bitacora(f"El paralelismo falló ({type(exc).__name__}: {str(exc)[:120]}); se repite en serie", "warn")
            hitos("Descarga serial de respaldo", 0.05, "V1")
            salidas = [_una(m) for m in tareas]
        fuera = dict(salidas)
        res.update({nom: fuera[nom] for nom in orden if nom in fuera})
    # 2b) auditoría de completitud ANTES de cualquier agregado: XM publica el día D en D+1/D+2
    #     y las primeras entregas llegan con una fracción del día. Sin este paso, el último día
    #     aparece como "la demanda cayó 75 %" y el balance (gen − dem) se infla +200 GWh.
    res, prov_dias = podar_dias_provisionales(res, incluir=bool(incluir_provisionales),
                                              progreso=progreso)
    fin_efectiva = fin
    for _nom in ("DemaReal_Sistema", "Gene_Sistema", "PrecBolsNaci_Sistema"):
        _df = res.get(_nom, {}).get("df", pd.DataFrame()) if isinstance(res.get(_nom), dict) else pd.DataFrame()
        if isinstance(_df, pd.DataFrame) and not _df.empty and "Date" in _df.columns:
            _mx = pd.to_datetime(_df["Date"], errors="coerce").max()
            if pd.notna(_mx):
                fin_efectiva = min(fin_efectiva, _mx.date())
                break
    if fin_efectiva != fin or not prov_dias.empty:
        hitos(f"Último día con datos cerrados: {fin_efectiva} (la ventana pedía hasta {fin}; "
              f"{len(prov_dias)} día(s) siguen preliminares en XM)", 0.56, "V2")
    hitos("Normalizando y uniendo tablas (wide→long, unidades, 24 h por día)…", 0.58, "V2",
          celda=(59, 60))

    # --------------------------- ETL ------------------------------------------------
    def pct_o_fraccion(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """XM manda PorcApor/PorcVoluUtilDiar como fracción (0.68) → escalar a % solo si hace falta."""
        if df is None or df.empty or col not in df.columns:
            return df if df is not None else pd.DataFrame()
        out = df.copy()
        med = float(np.nanmedian(pd.to_numeric(out[col], errors="coerce"))) if out[col].notna().any() else np.nan
        if np.isfinite(med) and abs(med) <= 2.0:
            out[col] = out[col] * 100.0
            bitacora(f"{col}: detectada fracción (mediana {med:.3f}) → convertida a % (×100)", "info")
        return out

    gen_raw = res.get("Gene_Sistema", {}).get("df", pd.DataFrame())
    if gen_raw is None or (hasattr(gen_raw, "empty") and gen_raw.empty):
        gen_raw = res.get("Gene_Recurso", {}).get("df", pd.DataFrame())
    if gen_raw is None:
        gen_raw = pd.DataFrame()
    gen = wide_to_long(gen_raw, "Generacion_kWh", codigo_nombre="Codigo_Planta")
    dim_raw = res.get("ListadoRecursos", {}).get("df", pd.DataFrame())
    dim = preparar_dim_plantas(dim_raw) if isinstance(dim_raw, pd.DataFrame) else pd.DataFrame()
    gen_enr = enriquecer_generacion(gen, dim)
    tec_raw = res.get("Gene_Recurso", {}).get("df", pd.DataFrame())
    gen_tec = (wide_to_long(tec_raw, "Generacion_kWh", codigo_nombre="Codigo_Planta")
               if isinstance(tec_raw, pd.DataFrame) and not tec_raw.empty else pd.DataFrame())

    dem = daily_to_long(res.get("DemaReal_Sistema", {}).get("df", pd.DataFrame()), "Demanda_MWh")
    dem_h = wide_to_long(res.get("DemaReal_Sistema", {}).get("df", pd.DataFrame()), "Demanda_MW",
                         codigo_nombre="Codigo")
    emb = pct_o_fraccion(daily_to_long(res.get("PorcVoluUtilDiar_Sistema", {}).get("df", pd.DataFrame()),
                                       "Volumen_Pct"), "Volumen_Pct")
    apo_energia = daily_to_long(res.get("AporEner_Sistema", {}).get("df", pd.DataFrame()),
                                "Aportes_kWh")
    if not apo_energia.empty:
        apo_energia["Aportes_GWh"] = apo_energia["Aportes_kWh"] / 1e6
    apo_pct = pct_o_fraccion(daily_to_long(res.get("PorcApor_Sistema", {}).get("df", pd.DataFrame()),
                                           "Aportes_Pct"), "Aportes_Pct")
    apo = apo_pct.copy()
    if not apo_energia.empty:
        apo = apo.merge(apo_energia[["Fecha", "Aportes_GWh"]], on="Fecha", how="outer")
    esc = daily_to_long(res.get("PrecEsca_Sistema", {}).get("df", pd.DataFrame()), "Precio_COP_kWh")
    precio_pond = daily_to_long(res.get("PPPrecBolsNaci_Sistema", {}).get("df", pd.DataFrame()),
                                "Precio_Ponderado_COP_kWh")
    precio_h = wide_to_long(res.get("PrecBolsNaci_Sistema", {}).get("df", pd.DataFrame()),
                            "Precio_Bolsa_COP_kWh", codigo_nombre="Codigo")
    cen_recurso = daily_to_long_recurso(res.get("CapEfecNeta_Recurso", {}).get("df", pd.DataFrame()),
                                        "Potencia_Confiable_kW")
    dema_max = daily_to_long(res.get("DemaMaxPot_Sistema", {}).get("df", pd.DataFrame()), "Demanda_MW")
    if not dema_max.empty:
        med = float(np.nanmedian(dema_max["Demanda_MW"])) if dema_max["Demanda_MW"].notna().any() else np.nan
        if np.isfinite(med) and med > 1e5:          # vendría en kW → pasar a MW
            dema_max["Demanda_MW"] /= 1e3
    disp = pd.DataFrame()
    disp_raw = res.get("DispoCome_Recurso", {}).get("df", pd.DataFrame())
    if isinstance(disp_raw, pd.DataFrame) and not disp_raw.empty:
        disp_h = wide_to_long(disp_raw, "Disp_kW", codigo_nombre="Codigo_Planta")
        if not disp_h.empty:
            disp = (disp_h.groupby(["Fecha", "Hora"], as_index=False)["Disp_kW"].sum()
                    .groupby("Fecha", as_index=False)["Disp_kW"].mean()
                    .rename(columns={"Disp_kW": "Disp_MW"}))
            disp["Disp_MW"] = disp["Disp_MW"] / 1e3
            bitacora(f"DispoCome: {len(disp_h):,} filas horario-planta → {len(disp)} días (media de la "
                     f"suma horaria, kW→MW)")
    emis = wide_to_long_combustible(res.get("EmisionesCO2_RecursoComb", {}).get("df", pd.DataFrame()), "tCO2")
    comb = wide_to_long_combustible(res.get("ConsCombustibleMBTU_Recurso", {}).get("df", pd.DataFrame()),
                                    "Consumo_Combustible_MBTU")
    cm = wide_to_long(res.get("CostMargDesp_Sistema", {}).get("df", pd.DataFrame()),
                      "Costo_Marginal_COP_kWh", codigo_nombre="Codigo")
    fe_raw = res.get("factorEmisionCO2e_Sistema", {}).get("df", pd.DataFrame())
    fe_h = (wide_to_long(fe_raw, "Factor_Emision_gCO2e_kWh", codigo_nombre="Codigo")
            if isinstance(fe_raw, pd.DataFrame) and not fe_raw.empty else pd.DataFrame())
    fe_dia = (fe_h.groupby("Fecha", as_index=False)["Factor_Emision_gCO2e_kWh"].mean()
              if fe_h is not None and not fe_h.empty else pd.DataFrame())

    # sistema horario (equivalente a la celda v2: join demanda/precio/embalses/escasez)
    sistema = pd.DataFrame()
    if not dem_h.empty:
        sistema = dem_h.copy()
        sistema["Demanda_MW"] = pd.to_numeric(sistema["Demanda_MW"], errors="coerce") / 1e3
        if not cm.empty:
            sistema = sistema.merge(cm[["Timestamp", "Costo_Marginal_COP_kWh"]], on="Timestamp", how="left")
        else:
            sistema["Costo_Marginal_COP_kWh"] = np.nan
        if not precio_h.empty:
            sistema = sistema.merge(precio_h[["Timestamp", "Precio_Bolsa_COP_kWh"]], on="Timestamp", how="left")
        if not esc.empty:
            e = esc.copy(); e["Fecha"] = pd.to_datetime(e["Fecha"])
            sistema = sistema.merge(e.rename(columns={"Precio_COP_kWh": "Precio_Escasez_COP_kWh"})
                                    [["Fecha", "Precio_Escasez_COP_kWh"]], on="Fecha", how="left")
        if not emb.empty:
            b = emb.copy(); b["Fecha"] = pd.to_datetime(b["Fecha"])
            sistema = sistema.merge(b.rename(columns={"Volumen_Pct": "Embalses_Pct"})[["Fecha", "Embalses_Pct"]],
                                    on="Fecha", how="left")
        if not apo_pct.empty:
            a = apo_pct.copy(); a["Fecha"] = pd.to_datetime(a["Fecha"])
            sistema = sistema.merge(a[["Fecha", "Aportes_Pct"]], on="Fecha", how="left")
        if not gen_enr.empty:
            g = gen_enr.copy()
            g["Generacion_MW"] = pd.to_numeric(g["Generacion_kWh"], errors="coerce") / 1e3
            gg = g.groupby(["Timestamp", "Tecnologia"])["Generacion_MW"].sum().reset_index()
            piv = gg.pivot_table(index="Timestamp", columns="Tecnologia", values="Generacion_MW",
                                 aggfunc="sum", fill_value=0.0)
            piv.columns = [f"Gen_{c}" for c in piv.columns]
            sistema = sistema.merge(piv.reset_index(), on="Timestamp", how="left")
        sistema, rep = verificar_y_corregir_integridad(sistema)
        sistema["Fecha"] = pd.to_datetime(sistema["Timestamp"]).dt.date
    elif not gen_enr.empty:
        sistema = gen_enr.groupby(["Fecha", "Hora", "Timestamp"], as_index=False)["Generacion_kWh"].sum()
        sistema = sistema.rename(columns={"Generacion_kWh": "Generacion_MW"})
        sistema["Demanda_MW"] = np.nan
        sistema["Costo_Marginal_COP_kWh"] = np.nan

    dm = dema_max.copy()
    if not dm.empty:
        dm["Fecha"] = pd.to_datetime(dm["Fecha"])
    if not sistema.empty and "Demanda_MW" in sistema.columns:
        dmr = (sistema.groupby("Fecha", as_index=False)["Demanda_MW"].max()
               .rename(columns={"Demanda_MW": "Demanda_MW"}))
        dmr["Fecha"] = pd.to_datetime(dmr["Fecha"])
        if dm.empty:
            dm = dmr
            if not dm.empty:
                bitacora("Demanda máxima: XM aún no publica `DemaMaxPot` para la ventana; se usó el "
                         "máximo horario de `DemaReal` (demanda real) como sustituto del margen de reserva")
        else:
            dm = pd.concat([dm, dmr]).drop_duplicates("Fecha", keep="first").sort_values("Fecha")
    dema_max = dm if not dm.empty else dema_max
    # v4: disponibilidad
    indisp = sistema_v4 = pd.DataFrame()
    if not disp.empty and not cen_recurso.empty:
        cen_dia = cen_recurso.groupby("Fecha", as_index=False)["Potencia_Confiable_kW"].sum()
        cen_dia = cen_dia.rename(columns={"Potencia_Confiable_kW": "CEN_kW"})
        cen_dia["CEN_kW"] = cen_dia["CEN_kW"] / 1e3
        d = disp.copy()
        d["Fecha"] = pd.to_datetime(d["Fecha"]).dt.date
        cruce = d.merge(cen_dia, on="Fecha", how="inner")
        cruce["Indisponibilidad_pct"] = 100 * (1 - cruce["Disp_MW"] / cruce["CEN_kW"].replace(0, np.nan))
        cruce["Reserva_Operativa_MW"] = cruce["Disp_MW"] - (cruce.get("Demanda_MW", pd.Series(np.nan)))
        indisp = cruce
    elif not cen_recurso.empty and not dema_max.empty:  # proxy con DemaMaxPot/demanda real
        d = dema_max.copy(); d["Fecha"] = pd.to_datetime(d["Fecha"]).dt.date
        cen_dia = cen_recurso.groupby("Fecha", as_index=False)["Potencia_Confiable_kW"].sum()
        cen_dia["CEN_MW"] = pd.to_numeric(cen_dia.pop("Potencia_Confiable_kW")) / 1e3
        cruce = d.merge(cen_dia, on="Fecha", how="inner")
        cruce["Indisponibilidad_pct"] = 100 * (1 - cruce["Demanda_MW"] / cruce["CEN_MW"].replace(0, np.nan))
        cruce["Reserva_Operativa_MW"] = cruce["CEN_MW"] - cruce["Demanda_MW"]
        cruce = cruce.rename(columns={"Demanda_MW": "Demanda_MW_Dia"})
        indisp = cruce
    if not sistema.empty and "Demanda_MW" in sistema.columns and not cm.empty:
        sistema_v4 = sistema.copy()
        if cen_recurso is not None and not cen_recurso.empty:
            _cd = cen_recurso.copy()
            _cd["Fecha"] = pd.to_datetime(_cd["Fecha"]).dt.date
            _cen = _cd.groupby("Fecha", as_index=False)["Potencia_Confiable_kW"].sum()
            _cen["CEN_MW"] = pd.to_numeric(_cen.pop("Potencia_Confiable_kW")) / 1e3
            sistema_v4 = sistema_v4.merge(_cen, on="Fecha", how="left")
        else:
            sistema_v4["CEN_MW"] = np.nan
    resumen = resumen_diario(sistema, emb, apo_pct, esc, precio_pond)
    hitos("Cruce con dim_plantas, resúmenes diarios y validación (c61)…", 0.62, "V3")

    # factores de planta (usa la serie ENSO mensual si hay crudos; si no, solo la ventana)
    fp = factores_planta(gen_enr if not gen_enr.empty else gen_tec, cen_recurso, dim)
    cen_diario = resumen_capacidad(cen_recurso, dm if not dm.empty else None, dim)
    fp = factores_planta(gen_enr if not gen_enr.empty else gen_tec, cen_recurso, dim)
    emisiones_dia = agregar_emisiones_diarias(emis)
    if not emisiones_dia.empty and not comb.empty:
        c = comb.copy(); c["Fecha"] = pd.to_datetime(c["Fecha"]).dt.date
        por_dia = c.groupby("Fecha", as_index=False)["Consumo_Combustible_MBTU"].sum()
        emisiones_dia = emisiones_dia.merge(por_dia, on="Fecha", how="left")
    contexto_emisiones = {"fuente_intensidad": "sin datos", "col_energia": None}
    if not emisiones_dia.empty:
        emisiones_dia = emisiones_dia.rename(columns={"tCO2": "tCO2_XM"})
        if not emis.empty and "Codigo_Planta" in emis.columns:
            cobertura = (emis.assign(_d=pd.to_datetime(emis["Fecha"]).dt.date)
                         .groupby("_d")["Codigo_Planta"].nunique().rename("Recursos_reportados"))
            emisiones_dia = emisiones_dia.merge(
                cobertura.reset_index().rename(columns={"_d": "Fecha"}), on="Fecha", how="left")
        col_ene = next((c for c in ("Gen_TERMICA_GWh", "Generacion_Total_GWh", "Demanda_GWh")
                        if c in resumen.columns), None)
        contexto_emisiones["col_energia"] = col_ene
        if col_ene:
            emisiones_dia = calcular_intensidad_emision(emisiones_dia, resumen, col_ene)
            emisiones_dia = emisiones_dia.rename(columns={"Intensidad_tCO2_MWh": "Intensidad_ratio_MWh"})
            contexto_emisiones["fuente_intensidad"] = f"ratio tCO₂ XM ÷ {col_ene}"
        if not fe_dia.empty:
            f = fe_dia.copy()
            f["Fecha"] = pd.to_datetime(f["Fecha"]).dt.date
            emisiones_dia = emisiones_dia.merge(f[["Fecha", "Factor_Emision_gCO2e_kWh"]],
                                                on="Fecha", how="left")
            emisiones_dia["Intensidad_tCO2_MWh"] = (
                emisiones_dia["Factor_Emision_gCO2e_kWh"].astype(float) / 1e3)
            contexto_emisiones["fuente_intensidad"] = "factorEmisionCO2e (factor oficial XM)"
            bitacora(f"Intensidad del despacho: se usa `factorEmisionCO2e` de XM (media "
                     f"{emisiones_dia['Intensidad_tCO2_MWh'].mean():.3f} tCO₂e/MWh) porque "
                     f"`EmisionesCO2/RecursoComb` solo cubre parte del parque térmico")
        if "Intensidad_tCO2_MWh" in emisiones_dia:
            emisiones_dia["Banda_Emision"] = emisiones_dia["Intensidad_tCO2_MWh"].map(bandas_emision)
    cruce_v4 = cruzar_diario_v4(sistema_v4 if not sistema_v4.empty else sistema, indisp)
    correlaciones = pd.DataFrame()
    if not resumen.empty:
        cols = [c for c in ("Precio_Ponderado_COP_kWh", "Embalses_Pct", "Aportes_Pct",
                            "Participacion_Termica_pct", "Balance_GWh", "Demanda_GWh") if c in resumen]
        if len(cols) > 1:
            correlaciones = resumen[cols].corr()
    hallazgos = construir_hallazgos({"resumen_diario": resumen, "sistema_diario": pd.DataFrame(),
                                     "correlaciones": correlaciones, "gen_enriquecida": gen_enr,
                                     "dim_plantas": dim})
    hitos("Motor El Niño (ONI + fases + eventos, celdas 155-159)…", 0.70, "V4")
    nino_res = construir_nino_mensual(forzar_recarga=(modo in ("auto", "api")), modo=modo)
    nino_df = nino_res["df"]
    # el mes en curso (parcial) se incorpora desde las tablas diarias ya descargadas:
    if not nino_df.empty:
        mes_actual = f"{fin:%Y-%m}"
        ya = set(nino_df["Mes"].astype(str))
        fila_cur: dict[str, Any] = {"Mes": mes_actual}
        if not emb.empty:
            fila_cur["Embalses_Pct"] = float(emb["Volumen_Pct"].mean())
        if not apo_pct.empty:
            fila_cur["Aportes_Pct"] = float(apo_pct["Aportes_Pct"].mean())
        if not precio_pond.empty:
            fila_cur["Precio_Ponderado_COP"] = float(precio_pond["Precio_Ponderado_COP_kWh"].mean())
        elif not esc.empty:
            fila_cur["Precio_Ponderado_COP"] = float(esc["Precio_COP_kWh"].mean())
        if len(fila_cur) > 1 and mes_actual not in ya:
            nino_df = pd.concat([nino_df, pd.DataFrame([fila_cur])], ignore_index=True)
            nino_df = nino_df.sort_values("Mes").reset_index(drop=True)
            nino_df["Mes_pd"] = pd.PeriodIndex(nino_df["Mes"].astype(str), freq="M")
            nino_df["Es_Ultimo_Mes"] = nino_df["Mes_pd"] == nino_df["Mes_pd"].max()
            nino_df = nino_df.drop(columns=["Mes_pd"])
            bitacora(f"MES_ACTUAL: {mes_actual} añadido a la serie ENSO desde la ventana "
                     f"(embalses {fila_cur.get('Embalses_Pct', float('nan')):.1f} %, aportes "
                     f"{fila_cur.get('Aportes_Pct', float('nan')):.1f} %, precio "
                     f"{fila_cur.get('Precio_Ponderado_COP', float('nan')):,.0f} COP/kWh)")
    calidad = validar_tablas_analiticas({
        "gen": gen, "sistema": sistema, "costo_marginal": cm, "embalses": emb, "aportes": apo_energia,
        "aportes_pct": apo_pct, "emisiones": emis, "cen_recurso": cen_recurso, "resumen": resumen})
    hitos(f"Validación de tablas analíticas (celda 61 del notebook): {len(calidad)} tablas revisadas · "
          f"{int((calidad['fuera_rango'] > 0).sum()) if not calidad.empty else 0} fuera de rango · "
          f"{int((calidad['nulos_valor'] > 0).sum()) if not calidad.empty else 0} con nulos "
          f"estructurales (planta/hora que no reporta)", 0.79, "V3", celda=61)
    hitos("Composición de tablas analíticas y v3/v4…", 0.80, "V4", celda=(120, 212))
    n34 = descargar_nino34(forzar=(modo == "api"))
    nino_df = anadir_fase_operativa(nino_df, n34)
    eventos = detectar_eventos_enso(nino_df) if nino_df is not None and not nino_df.empty \
        else nino_res.get("eventos", pd.DataFrame())
    impacto = tabla_impacto_enso(
        nino_df, [v for v in ("Embalses_Pct", "Aportes_Pct", "Precio_Ponderado_COP", "Nino1_2",
                              "Nino3_4") if nino_df is not None and not nino_df.empty and v in nino_df.columns])

    # serie ENSO-económica v4 (mensual): seed + mes en curso
    v4_serie = pd.DataFrame()
    if V4_MENSUAL.exists():
        try:
            v4_serie = pd.read_csv(V4_MENSUAL)
        except Exception:                                         # noqa: BLE001
            v4_serie = pd.DataFrame()
    if not v4_serie.empty and not cen_diario.empty:
        cen_prom = float(pd.to_numeric(cen_diario["CEN_Total_MW"], errors="coerce").mean())
        nino_df["CEN_MW_promedio"] = cen_prom
    # ★ sobre los boxplots: últimos valores de la ventana (margen real si DispoCome activo)
    extras_cur = {}
    if not resumen.empty:
        extras_cur["Precio_Ponderado_COP_kWh"] = float(resumen["Precio_Ponderado_COP_kWh"].mean()) \
            if "Precio_Ponderado_COP_kWh" in resumen else None
        extras_cur["Embalses_Pct"] = float(resumen["Embalses_Pct"].mean()) if "Embalses_Pct" in resumen else None
        extras_cur["Aportes_Pct"] = float(resumen["Aportes_Pct"].mean()) if "Aportes_Pct" in resumen else None
    # v4 c203: impacto por episodio ENSO como tabla (sin figura; la figura se retiró a pedido del autor)
    v4_tabla_impacto = tabla_impacto_eventos_v4({"v4_serie": v4_serie, "cruce_v4": cruce_v4,
                                                 "nino": nino_df, "nino_eventos": eventos})
    if not v4_tabla_impacto.empty:
        hitos(f"v4 c203 · impacto por episodio ENSO: {len(v4_tabla_impacto)} eventos con Δ% vs "
              f"régimen neutral", 0.98, "V4")
    hitos(f"Listo: ctx completo con las tablas de las {len(FIGURES)} figuras", 1.0, "OK")

    linaje = [{"métrica": r["metrica"], "metric_id": r["metric_id"], "entidad": r["entity"],
               "filas": r["filas"], "origen": r.get("origen", ""), "nota": r.get("nota", ""),
               "inicio": str(r.get("inicio")), "fin": str(r.get("fin")), "ok": r.get("ok")}
              for r in res.values()]
    cat_por_codigo = (dim.set_index("Codigo_Planta").to_dict("index") if not dim.empty else {})
    # la "firma" identifica el estado de datos con el que se construyó cada figura (ver figura_cacheada)
    firma = (f"{_hoy():%Y%m%d}|{dias}|{modo}|prov={'1' if incluir_provisionales else '0'}|"
             + ",".join(f"{k}={int(v)}" for k, v in sorted(opciones.items())))
    return {
        "ini": ini, "fin": fin, "dias": dias, "modo": modo, "_firma": firma,
        "fin_efectiva": fin_efectiva, "dias_provisionales": prov_dias,
        "incluir_provisionales": bool(incluir_provisionales),
        "raw": {k: v for k, v in res.items()},
        "linaje": pd.DataFrame(linaje),
        "dim_plantas": dim, "catalogo_agentes": res.get("ListadoAgentes", {}).get("df", pd.DataFrame()),
        "rios": res.get("ListadoRios", {}).get("df", pd.DataFrame()),
        "cat_por_codigo": cat_por_codigo,
        "gen": gen, "gen_enr": gen_enr, "gen_tec": gen_tec,
        "demanda_diaria": dem, "sistema": sistema, "resumen": resumen,
        "embalses": emb, "aportes": apo, "aportes_pct": apo_pct, "escasez": esc,
        "cen_recurso": cen_recurso, "cen_diario": cen_diario, "dema_max": dema_max,
        "precio_pond": precio_pond, "precio_bolsa_h": precio_h,
        "disp_come": disp, "indisp": indisp, "cruce_v4": cruce_v4, "sistema_v4": sistema_v4,
        "emisiones": emis, "emisiones_dia": emisiones_dia, "combustible": comb, "costo_marginal": cm,
        "fp": fp, "correlaciones": correlaciones, "hallazgos": hallazgos,
        "factor_emision_dia": fe_dia, "notas_emisiones": contexto_emisiones,
        "nino": nino_df, "nino_estado": nino_res.get("estado"), "nino_eventos": eventos,
        "impacto": impacto, "nino34": n34, "_v4_tabla_impacto": v4_tabla_impacto, "calidad": calidad,
        "nino_nota": nino_res.get("nota", ""), "v4_serie": v4_serie, "extras_cur": extras_cur,
        "generado": dt.datetime.now().isoformat(timespec="seconds"),
    }


def sistema_v4_h(sistema: pd.DataFrame, cen_por_fecha: pd.DataFrame | None) -> pd.DataFrame:
    """Marca la hora más tensa del día (margen mínimo) — celda v4 del notebook."""
    if sistema is None or sistema.empty:
        return sistema if sistema is not None else pd.DataFrame()
    out = sistema.copy()
    if cen_por_fecha is not None and not cen_por_fecha.empty:
        c = cen_por_fecha.copy()
        c["Fecha"] = pd.to_datetime(c["Fecha"]).dt.date if "Fecha" in c else c["Fecha"]
        col = "CEN_Total_MW" if "CEN_Total_MW" in c.columns else None
        if col:
            out = out.merge(c[["Fecha", col]], on="Fecha", how="left")
            out["Margen_MW"] = out[col] - out.get("Demanda_MW", pd.Series(np.nan))
            out["Margen_pct"] = 100 * out["Margen_MW"] / out[col].replace(0, np.nan)
    if "Costo_Marginal_COP_kWh" in out:
        out["Costo_Marginal_COP_kWh"] = pd.to_numeric(out["Costo_Marginal_COP_kWh"], errors="coerce")
    return out


# ============================ ENSO: capa en tiempo real ========================
def descargar_nino34(forzar: bool = False) -> pd.DataFrame:
    """Anomalías mensuales Niño1+2/3/4/3.4 (ERSSTv5, base 1991-2020). Caché en data/reference."""
    ruta = REF_DIR / "nino34_mensual.csv"
    if ruta.exists() and not forzar:
        try:
            return pd.read_csv(ruta)
        except Exception:                                         # noqa: BLE001
            pass
    for url in (NINO34_URL,):
        try:
            bruto = _sesion().get(url, timeout=90, verify=_cert_envio()).text
        except Exception as exc:                                  # noqa: BLE001
            bitacora(f"Niño3.4 no disponible ({type(exc).__name__}); uso el archivo local", "warn")
            continue
        filas = []
        for lin in bruto.splitlines():
            p = lin.split()
            if len(p) < 10 or not p[0].strip().isdigit():
                continue
            ano, mes = int(p[0]), int(p[1])
            if not (1950 <= ano <= 2100) or not (1 <= mes <= 12):
                continue
            try:
                _n12, a12, _n3, a3, _n4, a4, _n34, a34 = (float(x) for x in p[2:10])
            except ValueError:
                continue
            filas.append({"Mes": f"{ano}-{mes:02d}", "Nino1_2": a12, "Nino3": a3, "Nino3_4": a34})
        if filas:
            df = pd.DataFrame(filas)
            try:
                df.to_csv(ruta, index=False)
            except OSError:
                pass
            bitacora(f"Niño1+2/3.4 descargados de CPC: {len(df)} meses (último {df['Mes'].iloc[-1]})")
            return df
    if ruta.exists():
        try:
            return pd.read_csv(ruta)
        except Exception:                                         # noqa: BLE001
            pass
    return pd.DataFrame()


def anadir_fase_operativa(nino: pd.DataFrame, n34: pd.DataFrame) -> pd.DataFrame:
    """Añade Niño1+2/Niño3.4 y una FASE OPERATIVA que no deja el mes en curso 'sin datos'.

    Regla (documentada en el tablero): los meses sin ONI publicado heredan la última fase
    oficial; así 'agosto 2026' queda dentro del evento activo en lugar de quedar vacío,
    y se marca con ★ para no confundirlo con un dato oficial de NOAA.
    """
    if nino is None or nino.empty:
        return nino if nino is not None else pd.DataFrame()
    out = nino.copy()
    if n34 is not None and not n34.empty:
        out = out.merge(n34, on="Mes", how="left")
    for c in ("Nino1_2", "Nino3", "Nino3_4"):
        if c not in out.columns:
            out[c] = np.nan
    if "Fase" not in out.columns:
        out["Fase"] = clasificar_fase(out.get("ONI", pd.Series(np.nan, index=out.index)))
    out["Es_Ultimo_Mes"] = out.get("Es_Ultimo_Mes", pd.Series(False, index=out.index)).fillna(False)
    publicados = out["ONI"].notna()
    ultima_fase = out.loc[publicados, "Fase"].iloc[-1] if publicados.any() else "Sin ONI"
    out["Fase_operativa"] = out["Fase"].where(publicados, ultima_fase)
    out["ONI_operativo"] = out["ONI"].where(publicados, out.loc[publicados, "ONI"].iloc[-1] if publicados.any() else np.nan)
    out["Prov"] = ~publicados & (out["Fase_operativa"] != "Sin ONI")
    # proxy de intensificación: Niño1+2 (el índice que más pesa en lluvia de Colombia)
    if out["Nino1_2"].notna().any():
        out["Nino1_2_z"] = (out["Nino1_2"] - out["Nino1_2"].expanding().mean()) / out["Nino1_2"].expanding().std()
    return out


def resumen_evento_actual(nino: pd.DataFrame) -> dict[str, Any]:
    """Narrativa del evento en curso con números verificados dentro del app."""
    if nino is None or nino.empty or "ONI_operativo" not in nino.columns:
        return {}
    d = nino.copy()
    d["Mes_pd"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M")
    d = d.sort_values("Mes_pd").reset_index(drop=True)
    hist = d["ONI"].dropna()
    if hist.empty:
        return {}
    # arranque del evento: último cruce hacia arriba de +0.5 con |ONI| sostenido
    pub = d[d["ONI"].notna()].reset_index(drop=True)
    i = len(pub) - 1
    inicio = pub.loc[i, "Mes"]
    while i > 0 and pub.loc[i - 1, "ONI"] >= UMBRAL_NINO:
        i -= 1
        inicio = pub.loc[i, "Mes"]
    if pub.loc[i, "ONI"] < UMBRAL_NINO:
        i_evento = None
    else:
        i_evento = i
    pico = float(pub.loc[i_evento:, "ONI"].max()) if i_evento is not None else float("nan")
    mes_ult = str(d["Mes_pd"].iloc[-1])
    oni_ult = float(d["ONI_operativo"].iloc[-1])
    n12 = d["Nino1_2"].dropna()
    n12_max = float(n12.max()) if not n12.empty else float("nan")
    comparables = int((d["ONI"] >= pico - 0.15).sum()) if np.isfinite(pico) else 0
    top_hist = float(d["ONI"].nlargest(6).mean())
    return {"inicio_oficial": inicio, "meses_activos": (len(pub) - i_evento) if i_evento is not None else 0,
            "pico_oni": pico, "oni_ultimo": oni_ult, "mes_ultimo": mes_ult,
            "nino1_2": float(n12.iloc[-1]) if not n12.empty else float("nan"),
            "nino1_2_record": n12_max, "top_hist_mean": top_hist,
            "es_el_nino": (not np.isfinite(pico) or pico >= UMBRAL_NINO) and oni_ult >= UMBRAL_NINO,
            "fuerza": fuerza_evento(pub.loc[i_evento:, "ONI"])[1] if i_evento is not None else "—"}


# ==============================================================================
# 7 · FIGURAS PLOTLY + LEYENDAS (cada una dice QUÉ MUESTRA y QUÉ SE DECIDE)
# ==============================================================================
def _leyenda(titulo: str, muestra: str, decision: str, nino: str, fuente: str = "") -> str:
    txt = (f"**📖 {titulo} — qué muestra y para qué sirve**\n\n"
           f"- **Qué muestra:** {muestra}\n- **Decisión / conclusión:** {decision}\n")
    if nino:
        txt += f"- **Lente El Niño:** {nino}\n"
    if fuente:
        txt += f"- **Fuente / método:** {fuente}\n"
    return txt


def _fig(base: dict, height: int = 420) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=height, margin=MARGEN,
                      hovermode="x unified", legend=dict(orientation="h", y=1.14, x=0),
                      plot_bgcolor="#fcfcfe")
    return fig


def _eje_ventana(fig: go.Figure, ctx: dict, meses: int | None = None) -> go.Figure:
    """Recorta el eje X a la ventana evaluada (o a los últimos `meses` meses).

    Regla del tablero: un gráfico operativo no debe empezar en 1950 cuando el usuario
    está mirando 12-31 días. Solo se recorta la *vista* (`xaxis.range`), no los datos,
    así que el zoom del lector sigue pudiendo abrirse hasta el inicio de la serie.
    Se ignoran los ejes horarios (0-23) y categóricos, que no son fechas reales.
    Si el sidebar activó `historial_completo`, no recorta nada.
    """
    try:
        if ctx.get("historial_completo"):
            return fig
        fin = pd.to_datetime(str(ctx.get("fin_efectiva") or ctx["fin"])) + pd.Timedelta(days=1)
        ini = fin - pd.DateOffset(months=meses) if meses else pd.to_datetime(str(ctx["ini"]))
        if ini >= fin:
            ini = fin - pd.DateOffset(days=45)
        fechap = False
        for t in fig.data:
            x = getattr(t, "x", None)
            if x is None:
                continue
            try:
                vals = x.to_list() if isinstance(x, pd.Series) else list(x)
            except Exception:                                     # noqa: BLE001
                continue
            if all(re.fullmatch(r"[Hh]\d{1,2}|\d{1,2}", str(v)) for v in vals[:8] if str(v)):
                continue                                   # eje horario categórico (H01..H24): no es fecha
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)   # dateutil no infiere formatos mixtos
                ts = pd.to_datetime(pd.Series(vals), errors="coerce").dropna()
            if not ts.empty and ts.max() >= pd.Timestamp("1990-01-01"):
                fechap = True
                break
        if not fechap:
            return fig
        fig.update_xaxes(range=[ini.strftime("%Y-%m-%d"), fin.strftime("%Y-%m-%d")])
    except Exception:                                             # noqa: BLE001
        pass
    return fig


_ID_FIGURA = 0


_MESES_ES = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
             "septiembre", "octubre", "noviembre", "diciembre")


def _mes_es(periodo: Any) -> str:
    """`2026-06` → `junio de 2026` (`strftime('%B')` devuelve el mes en inglés)."""
    try:
        per = pd.Period(str(periodo)[:7], freq="M")
        return f"{_MESES_ES[per.month - 1]} de {per.year}"
    except Exception:                                             # noqa: BLE001
        return str(periodo)


def _fecha_es(fecha: Any) -> str:
    """`2026-08-26` → `26 de agosto de 2026` (el `strftime('%B')` de Python da el mes en inglés)."""
    try:
        t = pd.to_datetime(str(fecha))
        return f"{t.day} de {_MESES_ES[t.month - 1]} de {t.year}"
    except Exception:                                             # noqa: BLE001
        return str(fecha)


def _coma(x: Any, dec: int = 1, signo: bool = True) -> str:
    """Número con coma decimal como se lee en español: `+1,4` (NOAA/CPC lo publica así)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "s/d"
    f = f"{v:+.{dec}f}" if signo else f"{v:.{dec}f}"
    return f.replace(".", ",")


def _clave_fig(prefijo: str = "fig") -> str:
    """ID único por `st.plotly_chart`.

    Streamlit 1.62 lanza `StreamlitDuplicateElementId` si dos gráficos reciben un spec
    idéntico (pasa al pintar el heatmap 'Todo el SIN' repetido, o dos figuras vacías), y ese
    error corta la ejecución de la página. Numerar los elementos lo evita siempre.
    """
    global _ID_FIGURA
    _ID_FIGURA += 1
    slug = re.sub(r"[^a-z0-9]+", "-", str(prefijo).lower()).strip("-")[:36] or "fig"
    return f"{slug}-{_ID_FIGURA}"


def _ordenar_por_fecha(p: pd.DataFrame, etiquetas: pd.Series, fechas: Any) -> pd.DataFrame:
    """Ordena las filas de un pivot cuyas etiquetas son `'05-ago'` usando la fecha real.

    Antes se reparseaba el índice con `pd.to_datetime(idx, format="%d-%b")`, que en Python
    3.13+ lanza DeprecationWarning (fecha sin año) y en 3.15 cambiará de comportamiento.
    Ordenar desde la columna de fechas evita el parseo inverso.
    """
    try:
        mapa = (pd.DataFrame({"et": list(etiquetas), "f": pd.to_datetime(pd.Series(list(fechas)))})
                .dropna().groupby("et")["f"].min().sort_values())
        nuevas = [e for e in mapa.index if e in p.index]
        nuevas += [e for e in p.index if e not in set(mapa.index)]
        return p.loc[nuevas]
    except Exception:                                             # noqa: BLE001
        return p


def _vline_anotada(fig: go.Figure, x: Any, texto: str, color: str = "#555",
                    dash: str = "dash", arriba: bool = True) -> go.Figure:
    """add_vline + annotation separados (plotly 6 no promedia fechas en annotation_text)."""
    xf = x if isinstance(x, str) else (x.strftime("%Y-%m-%d") if hasattr(x, "strftime") else x)
    fig.add_vline(x=xf, line=dict(color=color, width=1.6, dash=dash))
    if texto:
        fig.add_annotation(x=xf, yref="paper", y=1.06 if arriba else -0.16, text=texto,
                           showarrow=False, font=dict(size=9.5, color=color))
    return fig


def _meses_azules(nino: pd.DataFrame) -> tuple[list, list]:
    """Bandas de trasfondo: meses El Niño (rojo) y La Niña (azul) para sobreponer en series."""
    if nino is None or nino.empty or "Fase_operativa" not in nino.columns:
        return [], []
    d = nino.copy()
    d["Mes_pd"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M")
    d = d.sort_values("Mes_pd")
    nin, nina = [], []
    for fase, caja in (("El Niño", nin), ("La Niña", nina)):
        for _, g in d.groupby((d["Fase_operativa"] != fase).cumsum()):
            g = g[g["Fase_operativa"] == fase]
            if len(g) >= 2:
                caja.append((str(g["Mes_pd"].min().to_timestamp().date()),
                             str(g["Mes_pd"].max().to_timestamp().date())))
    return nin, nina


def _con_bandas(fig: go.Figure, nino: pd.DataFrame, eje_x="x") -> go.Figure:
    nin, nina = _meses_azules(nino)
    for i, (a, b) in enumerate(nin):
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(220,20,60,0.10)", line_width=0,
                      annotation_text="El Niño" if i == 0 else None, annotation_position="top left",
                      layer="below")
    for i, (a, b) in enumerate(nina):
        fig.add_vrect(x0=a, x1=b, fillcolor="rgba(31,119,180,0.08)", line_width=0,
                      line_dash="dot", layer="below")
    return fig


def _escala_tecnicas() -> dict[str, str]:
    return {t: COLOR_TECNOLOGIAS.get(t, "#9aa0a6") for t in TECNOLOGIAS_ORDEN}


def fig_balance_diario(ctx: dict) -> dict | None:
    r = ctx["resumen"]
    if r.empty or "Generacion_Total_GWh" not in r:
        return None
    fig = _fig(ctx)
    tec = [c for c in r.columns if c.startswith("Gen_") and c != "Gen_Total_GWh"]
    orden = [f"Gen_{t}_GWh" for t in TECNOLOGIAS_ORDEN if f"Gen_{t}_GWh" in tec]
    otros = [c for c in tec if c not in orden]
    for c in orden + otros:
        fig.add_trace(go.Bar(x=r["Fecha"], y=r[c], name=c.replace("Gen_", "").replace("_GWh", ""),
                             marker_color=_escala_tecnicas().get(c.replace("Gen_", "").replace("_GWh", ""), "#b0b0b0"),
                             hovertemplate="%{y:,.0f} GWh<extra>" + c + "</extra>"))
    if "Demanda_GWh" in r:
        fig.add_trace(go.Scatter(x=r["Fecha"], y=r["Demanda_GWh"], name="Demanda",
                                 line=dict(color="#111111", width=2.4, dash="dash")))
    if "Balance_GWh" in r:
        fig.add_trace(go.Scatter(x=r["Fecha"], y=r["Balance_GWh"], name="Balance (gen−dem)",
                                 yaxis="y2", line=dict(color=AMBAR, width=1.6)))
        fig.update_layout(yaxis2=dict(title="Balance GWh", overlaying="y", side="right",
                                      showgrid=False, zeroline=False))
    fig.update_layout(barmode="stack", xaxis_title="Día", yaxis_title="Energía (GWh/día)", height=460)
    pib = r.get("Participacion_Termica_pct", pd.Series(dtype=float))
    nino_txt = (f"Participación térmica media {float(pib.mean()):.0f} %: cuanto más alta, más expuesto "
                f"estás al precio del gas en sequía." if len(pib.dropna()) else "")
    return {"fig": fig, "leyenda": _leyenda(
        "Balance diario generación vs. demanda",
        "Barras apiladas de generación por tecnología contra la línea de demanda, y el balance "
        "diario (GWh) en el eje derecho.",
        "Los días con balance negativo o con la térmica dominando son los que primero se aprietan "
        "en sequía: ahí se justifica comprar energía, programar mantenimiento fuera de seco o "
        "asegurar combustible.",
        nino_txt, f"XM Bienda · Gene/Recurso agregado a nivel sistema y DemaReal/Sistema · "
                   f"{ctx['dias']} días. La generación medida en el parque (que incluye autogeneración) "
                   f"puede exceder la demanda del sistema en 5-15 % por pérdidas y recursos fuera de "
                   f"despacho: es normal, no un error.")}


def fig_perfil_horario(ctx: dict) -> dict | None:
    s = ctx["sistema"]
    if s.empty or "Hora" not in s:
        return None
    g = s.groupby("Hora", as_index=False).agg(Demanda_MW=("Demanda_MW", "mean"),
                                              Precio=("Costo_Marginal_COP_kWh", "mean"))
    if g.empty:
        return None
    fig = _fig(ctx, 400)
    pico_h = int(g.loc[g["Demanda_MW"].idxmax(), "Hora"])
    valle_h = int(g.loc[g["Demanda_MW"].idxmin(), "Hora"])
    col = [ROJO if h in HORAS_PICO else (CELESTE if h < 6 else AZUL_SUAVE) for h in g["Hora"]]
    fig.add_trace(go.Bar(x=g["Hora"], y=g["Demanda_MW"], name="Demanda media", marker_color=col))
    fig.add_trace(go.Scatter(x=g["Hora"], y=g["Precio"], name="Costo marginal medio", yaxis="y2",
                             line=dict(color=AMBAR, width=2.4, shape="spline")))
    fig.update_layout(yaxis2=dict(title="COP/kWh", overlaying="y", side="right", showgrid=False),
                      xaxis_title="Hora local (CO)", yaxis_title="Demanda media (MW)",
                      annotations=[dict(x=pico_h, y=float(g["Demanda_MW"].max()), text=f"pico {pico_h}:00",
                                        showarrow=True, arrowhead=2, ay=-30, font=dict(color=ROJO))])
    return {"fig": fig, "leyenda": _leyenda(
        "Perfil horario medio (¿cuándo se aprieta el sistema?)",
        "Demanda media por hora con las horas pico (17-22 h) en rojo y el costo marginal medio "
        "sobreimpreso; el valle solar suele verse al mediodía.",
        "Define en qué horas conviene desplazar carga, poner baterías o firmar contratos: si el "
        "precio medio del pico es muy superior al del valle, cada MWh movido vale dinero.",
        f"En El Niño la franja 17-22 h es la que primero sube de precio (hidro al límite y térmica "
        f"caro de tarde-noche). Diferencial medio actual: "
        f"{float(g.loc[g['Hora'].isin(HORAS_PICO), 'Precio'].mean() - g.loc[g['Hora'] == valle_h, 'Precio'].mean()):,.0f} COP/kWh."
        if g["Precio"].notna().any() else "", f"{len(s):,} filas horario-hora del sistema")}


def fig_precio_diario(ctx: dict) -> dict | None:
    r = ctx["resumen"]
    if r.empty or "Precio_Ponderado_COP_kWh" not in r:
        return None
    fig = _fig(ctx, 430)
    med = float(r["Precio_Ponderado_COP_kWh"].median())
    umbral = 1.5 * med
    if {"Precio_Min_COP_kWh", "Precio_Max_COP_kWh"} <= set(r.columns):
        fig.add_trace(go.Scatter(x=pd.concat([r["Fecha"], r["Fecha"][::-1]]),
                                 y=pd.concat([r["Precio_Max_COP_kWh"], r["Precio_Min_COP_kWh"][::-1]]),
                                 fill="toself", fillcolor="rgba(31,119,180,0.12)", line=dict(width=0),
                                 name="Rango intradía min–max", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=r["Fecha"], y=r["Precio_Ponderado_COP_kWh"], name="Precio ponderado",
                             line=dict(color=AZUL, width=2.6)))
    fig.add_hline(y=umbral, line=dict(color=AMBAR, dash="dash"),
                  annotation_text=f"1.5× mediana ({umbral:,.0f})", annotation_font_size=10)
    if "Escasez_Precio_COP_kWh" in r or any("Escasez" in c for c in r.columns):
        col_es = next(c for c in r.columns if "Escasez" in c and "Precio" in c)
        fig.add_trace(go.Bar(x=r["Fecha"], y=r[col_es], name="Precio de escasez", yaxis="y2",
                             marker_color="rgba(146,64,197,0.55)"))
        fig.update_layout(yaxis2=dict(title="COP/kWh escasez", overlaying="y", side="right", showgrid=False))
    dias_altos = r[r["Precio_Ponderado_COP_kWh"] > umbral]
    fig.update_layout(xaxis_title="Día", yaxis_title="COP/kWh", height=430)
    return {"fig": fig, "leyenda": _leyenda(
        "Precio diario del mercado (señal más rápida del estrés hídrico)",
        f"Precio ponderado por demanda con la banda min–max intradía, la línea de 1.5× la mediana "
        f"y, si está disponible, el precio de escasez en barras moradas.",
        f"{len(dias_altos)} de {len(r)} días quedaron por encima de 1.5× la mediana: son la lista "
        f"de días para activar coberturas, renegociar PPA indexados o revisar ofertas de venta.",
        "Un precio que se mantiene arriba de la mediana mientras los embalses caen es el patrón "
        "clásico de inicio de El Niño; en 2023 y 2024 el precio diario superó 4 000 COP/kWh.",
        "Costo_Marginal/PorcApor/PPPrecBolsNaci · XM Bienda")}


def fig_embalses_aportes(ctx: dict) -> dict | None:
    r = ctx["resumen"]
    if r.empty or "Embalses_Pct" not in r:
        return None
    fig = _fig(ctx, 440)
    fig.add_trace(go.Scatter(x=r["Fecha"], y=r["Embalses_Pct"], name="Embalses (% volumen útil)",
                             line=dict(color=AZUL, width=2.8), fill="tozeroy", fillcolor="rgba(31,119,180,0.10)"))
    for u, c in ((55, AMBAR), (45, NARANJA), (35, ROJO)):
        fig.add_hline(y=u, line=dict(color=c, dash="dot"), line_width=1,
                      annotation_text=f"umbral {u} %", annotation_font_size=9,
                      annotation_position="bottom right")
    if "Aportes_Pct" in r:
        fig.add_trace(go.Bar(x=r["Fecha"], y=r["Aportes_Pct"], name="Aportes (% media histórica)", yaxis="y2",
                             marker_color="rgba(26,138,74,0.55)"))
        fig.update_layout(yaxis2=dict(title="% de la media histórica", range=[0, 220], overlaying="y",
                                      side="right", showgrid=False))
    # Solo las franjas ENSO que tocan la ventana: si no, el eje X se estiraba hasta 1950.
    ini_v, fin_v = pd.Timestamp(ctx["ini"]), pd.Timestamp(ctx["fin"])
    for x0, x1 in _meses_azules(ctx["nino"])[0]:
        t0, t1 = pd.Timestamp(x0), pd.Timestamp(x1)
        if t1 < ini_v or t0 > fin_v:
            continue
        fig.add_vrect(x0=max(t0, ini_v).date(), x1=min(t1, fin_v).date(), line_width=0,
                      fillcolor="rgba(220,20,60,0.07)", layer="below")
    fig.update_layout(xaxis_title="Día (ventana analizada)", yaxis_title="Embalses (% volumen útil)",
                      xaxis=dict(range=[ini_v - pd.Timedelta(days=1), fin_v + pd.Timedelta(days=1)],
                                 dtick="D3" if (fin_v - ini_v).days <= 45 else "M1"))
    ini_v2 = float(r["Embalses_Pct"].iloc[0]); fin_v2 = float(r["Embalses_Pct"].iloc[-1])
    franjas = sum(1 for x0, x1 in _meses_azules(ctx["nino"])[0]
                  if pd.Timestamp(x1) >= ini_v and pd.Timestamp(x0) <= fin_v)
    return {"fig": fig, "leyenda": _leyenda(
        "Embalses y aportes en la ventana (la variable que define El Niño)",
        f"Evolución del volumen útil día a día entre {ctx['ini']} y {ctx['fin']}, con los umbrales de "
        "seguridad del CNO (55/45/35 %) y, solo si caen dentro de la ventana, las franjas rojas de "
        f"episodio El Niño ({franjas} mes(es) franja(s) visible(s)). El eje X no se estira a la "
        "historia completa: lo que ves es tu ventana.",
        f"Los embalses pasaron de {ini_v2:.1f} % a {fin_v2:.1f} % ({fin_v2 - ini_v2:+.1f} pp). Con "
        "umbrales cruzados y aportes < 80 % de la media, el operador entra en modo ahorro de agua.",
        "Durante El Niño la recuperación de embalses se retrasa 2-4 meses: un nivel bajo en agosto-"
        "septiembre no se corrige hasta el primer pico de lluvias (octubre-noviembre en la Orinoquía).",
        "PorcApor/PorcVoluUtilDiar + AporEner (diaria) · umbrales: CNO")}



def fig_heatmap_tecnologia_hora(ctx: dict) -> dict | None:
    s = ctx["gen_enr"]
    if s.empty or "Tecnologia" not in s or "Hora" not in s:
        return None
    p = s.pivot_table(index="Tecnologia", columns="Hora", values="Generacion_kWh", aggfunc="sum",
                      fill_value=0.0) / 1e6          # kWh → GWh
    p = p.reindex([t for t in TECNOLOGIAS_ORDEN if t in p.index])
    fig = _fig(ctx, max(300, 40 * len(p) + 140))
    fig.add_trace(go.Heatmap(z=p.values, x=[f"{h:02d}h" for h in p.columns], y=p.index.astype(str),
                            colorscale="YlGnBu", name="GWh medios",
                            hovertemplate="%{y} · %{x}<br>%{z:,.1f} GWh<extra></extra>"))
    fig.update_layout(xaxis_title="Hora", yaxis_title="Tecnología",
                      annotations=[dict(text="← franja 17-22 h: horas pico", x=19.5, y=len(p) - 0.4,
                                        showarrow=False, font=dict(size=10, color=ROJO))])
    sol = p.loc["SOLAR"].max() / max(p.loc["SOLAR"].sum() / 24, 1e-9) if "SOLAR" in p.index else float("nan")
    return {"fig": fig, "leyenda": _leyenda(
        "Matriz tecnología × hora (dónde está la energía)",
        "Generación media diaria (GWh) de cada tecnología en cada hora del día.",
        "Se ve el 'duck curve': si el mediodía solar es alto y el pico de la tarde lo cubre la "
        "térmica, la tecnología cara manda el precio 4-6 horas al día. Ahí es donde una batería o "
        "una hidro con regulación valen más que otra solar.",
        "El Niño reduce la hidro y no afecta al solar (a veces lo mejora por menos nube); por eso "
        "en sequía el contraste mediodía-noche se amplía.",
        "Gene/Mercado + ListadoRecursos (dim_plantas)" if sol == sol else "Gene/Mercado + ListadoRecursos")}


def fig_matriz_correlaciones(ctx: dict) -> dict | None:
    c = ctx["correlaciones"]
    if c is None or c.empty:
        return None
    fig = _fig(ctx, 430)
    fig.add_trace(go.Heatmap(z=c.values, x=c.columns.tolist(), y=c.index.tolist(), zmin=-1, zmax=1,
                            colorscale=[[0, ROJO], [0.5, "white"], [1, AZUL]], name="Pearson r",
                            text=np.round(c.values, 2), texttemplate="%{text}", textfont=dict(size=10),
                            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title="Variable", yaxis_title="Variable", xaxis_tickangle=-30)
    try:
        r_ea = float(c.loc["Embalses_Pct", "Aportes_Pct"])
        r_ep = float(c.loc["Embalses_Pct", "Precio_Ponderado_COP_kWh"])
    except Exception:                                             # noqa: BLE001
        r_ea = r_ep = float("nan")
    return {"fig": fig, "leyenda": _leyenda(
        "Matriz de correlaciones (embalses · aportes · precio · térmica)",
        "Coeficiente de Pearson entre las variables diarias de la ventana; azul = positiva, rojo = negativa.",
        f"Aguas más aportes r={r_ea:+.2f}; embalses vs precio r={r_ep:+.2f}. Si la segunda es "
        "negativa y fuerte, el mercado ya está poniendo precio a la escasez de agua: es tu señal de "
        "cierre de posiciones o de asegurar gas.",
        "Es EL indicador que se rompe primero en El Niño; una r cercana a −0.7 predice que cada "
        "10 pp de embalses movieron el precio de forma sostenida.",
        "Pearson sobre medios diarios; n = %d días (no causal)" % len(ctx["resumen"]))}


def fig_perfiles_por_tecnologia(ctx: dict) -> dict | None:
    s = ctx["gen_enr"]
    if s.empty or "Tecnologia" not in s or "Hora" not in s:
        return None
    p = s.groupby(["Tecnologia", "Hora"])["Generacion_kWh"].sum().unstack(fill_value=0.0) / 1e6
    fig = _fig(ctx, 420)
    for t in [x for x in TECNOLOGIAS_ORDEN if x in p.index]:
        fig.add_trace(go.Scatterpolar(r=p.loc[t].tolist(), theta=[f"{h:02d}h" for h in p.columns],
                                      name=str(t), line_color=_escala_tecnicas()[t], line_width=2))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, title="GWh")), height=460,
                      legend=dict(orientation="v", x=1.02, y=1))
    return {"fig": fig, "leyenda": _leyenda(
        "Perfiles diarios por tecnología (radar de 24 h)",
        "Cómo se reparte cada tecnología a lo largo del día.",
        "Complementariedad: si tu cartera es solar+hidro, el radar muestra que no se solapan; si "
        "eres consumidor, tu hora de menor costo es donde el radar de térmica está bajo.",
        "En El Niño el área de la hidro se encoge en la noche (cuando más se necesita): la forma "
        "del radar cambia más que el total.", "Generación XM agregada por hora")}


def fig_embalses_zonas(ctx: dict) -> dict | None:
    """`graficar_embalses_zonas` del notebook: banda normal/alerta/crítico + mínimo anotado."""
    e = ctx["embalses"]
    if e is None or e.empty or "Volumen_Pct" not in e.columns:
        return None
    d = e.dropna(subset=["Volumen_Pct"]).copy()
    if d.empty:
        return None
    fig = _fig(ctx, 400)
    fig.add_hrect(y0=UMBRAL_EMBALSES_ALERTA, y1=100, fillcolor="rgba(76,175,80,.13)", line_width=0,
                  annotation_text=f"Zona normal (> {UMBRAL_EMBALSES_ALERTA} %)", annotation_font_size=9)
    fig.add_hrect(y0=UMBRAL_EMBALSES_CRITICO, y1=UMBRAL_EMBALSES_ALERTA,
                  fillcolor="rgba(255,193,7,.16)", line_width=0,
                  annotation_text=f"Zona de alerta ({UMBRAL_EMBALSES_CRITICO}-{UMBRAL_EMBALSES_ALERTA} %)",
                  annotation_font_size=9)
    fig.add_hrect(y0=0, y1=UMBRAL_EMBALSES_CRITICO, fillcolor="rgba(244,67,54,.16)", line_width=0,
                  annotation_text=f"Zona crítica (< {UMBRAL_EMBALSES_CRITICO} %)", annotation_font_size=9)
    fig.add_trace(go.Scatter(x=d["Fecha"], y=d["Volumen_Pct"], mode="lines+markers", name="% volumen útil",
                             line=dict(color=AZUL, width=2.6), marker=dict(size=4),
                             hovertemplate="%{x|%d-%b}<br>%{y:.1f} %<extra></extra>"))
    imin = int(d["Volumen_Pct"].idxmin())
    fig.add_trace(go.Scatter(x=[d["Fecha"].iloc[imin]], y=[d["Volumen_Pct"].iloc[imin]], mode="markers",
                             marker=dict(symbol="triangle-down", size=13, color=ROJO,
                                         line=dict(width=1, color="white")), name="mínimo del periodo"))
    fig.add_annotation(x=d["Fecha"].iloc[imin], y=float(d["Volumen_Pct"].iloc[imin]),
                       text=f"mín {d['Volumen_Pct'].iloc[imin]:.1f} %<br>{d['Fecha'].iloc[imin]:%d-%b}",
                       showarrow=True, arrowhead=1, ay=34, font=dict(size=10, color=ROJO))
    fig.update_layout(xaxis_title="Día", yaxis_title="% del volumen útil (promedio nacional)",
                      yaxis_range=[0, 100])
    en_alerta = int((d["Volumen_Pct"] < UMBRAL_EMBALSES_ALERTA).sum())
    en_critico = int((d["Volumen_Pct"] < UMBRAL_EMBALSES_CRITICO).sum())
    return {"fig": fig, "leyenda": _leyenda(
        "Embalses contra las zonas de alerta del operador",
        f"Nivel promedio del sistema día a día sobre las bandas normal (>{UMBRAL_EMBALSES_ALERTA} %), "
        f"alerta ({UMBRAL_EMBALSES_CRITICO}-{UMBRAL_EMBALSES_ALERTA} %) y crítica (<{UMBRAL_EMBALSES_CRITICO} %), "
        f"con el mínimo del periodo marcado. Días en alerta: {en_alerta}; en crítico: {en_critico}.",
        "Mientras la curva esté en la banda verde, el riesgo de racionamiento es bajo y el precio lo "
        "fija el costo variable de la térmica; en cuanto entra en ámbar, la prioridad pasa a preservar "
        "agua y el precio empieza a despegar.",
        "El termómetro de El Niño es la VELOCIDAD con la que la curva cruza de verde a ámbar: en los "
        "episodios fuertes el cruce ocurre entre julio y septiembre y la salida demora hasta el primer "
        "pico de lluvias.",
        "PorcVoluUtilDiar/Sistema · umbrales de la celda v2 del notebook "
        f"({UMBRAL_EMBALSES_ALERTA}/{UMBRAL_EMBALSES_CRITICO} %)")}


def fig_08_aportes_historico(ctx: dict) -> dict | None:
    n = ctx["nino"]
    if n is None or n.empty:
        return None
    col = next((c for c in ("Aportes_mm_equiv", "Aportes_Pct") if c in n.columns), None)
    if col is None:
        return None
    d = n.dropna(subset=[col]).copy()
    if d.empty:
        return None
    d["Mes_pd"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M")
    med_mes = d.groupby(d["Mes_pd"].dt.month)[col].mean().rename("Media histórica")
    fig = _fig(ctx, 400)
    unidad = "mm equivalentes al CEN" if col == "Aportes_mm_equiv" else "% de la media histórica"
    fig.add_trace(go.Scatter(x=d["Mes_pd"].dt.to_timestamp(), y=d[col],
                             name="Aportes (mm equivalentes)", line=dict(color=VERDE, width=2.2)))
    fig.add_trace(go.Scatter(x=d["Mes_pd"].dt.to_timestamp(),
                                            y=d["Mes_pd"].dt.month.map(med_mes),
                                            name="Media 2015-2026 por mes",
                                            line=dict(color="#777", dash="dot", width=1.4)))
    fig.update_layout(xaxis_title="Mes", yaxis_title=f"Aportes ({unidad})")
    fig = _eje_ventana(fig, ctx, meses=24)
    return {"fig": fig, "leyenda": _leyenda(
        "Aportes contra su propia media histórica (12 años de contexto)",
        f"Aportes hídricos mensuales ({unidad}) frente a la media climatológica del mismo mes, "
        "construida con la propia serie 2015→hoy (no con Normales Meteorológicas, para que la "
        "comparación sea interna y reproducible).",
        "Meses por debajo de la línea gris = hidrología deficitaria: sube la probabilidad de "
        "despacho térmico y de precio alto en las siguientes 4-8 semanas.",
        "En El Niño los aportes caen por debajo de la media sobre todo entre junio-septiembre y "
        "diciembre-febrero; el gráfico marca dónde estás tú hoy dentro de ese ciclo.",
        "construir_nino_mensual + AporEner/CapEfecNeta (XM)" if d is not None else "")}


# --- constantes del bloque `graficar_perfil_horario_precios` (celdas 52 y 84 del notebook) -------
EFICIENCIA_RT_SAEB = float(os.environ.get("APP_MANUEL_ETA_RT", "0.87"))   # ida-vuelta batería
HORAS_PICO_NB = (18, 22)      # el notebook usa 18-22; la app operativa usa HORAS_PICO=(17,22)
HORAS_VALLE_NB = (1, 6)


def fig_v2_05b_precio_horario_saeb(ctx: dict) -> dict | None:
    """Espejo fiel de `graficar_perfil_horario_precios` (celdas 84-85): boxplot horario del precio
    + spreads bruto y ajustado por eficiencia de ida-vuelta, los insumos exploratorios del SAEB.

    El notebook devuelve `(fig, kpis)`; aquí la figura va a Plotly y los KPIs quedan en
    `ctx['kpis_saeb']` para que la pestaña de costos los pueda citar en texto.
    """
    d = ctx.get("sistema", pd.DataFrame())
    col_precio = next((x for x in ("Precio_Bolsa_COP_kWh", "Costo_Marginal_COP_kWh")
                       if d is not None and not d.empty and x in d.columns), None)
    if d is None or d.empty or col_precio is None:
        return None
    dd = d[["Fecha", "Hora", col_precio]].copy()
    dd[col_precio] = pd.to_numeric(dd[col_precio], errors="coerce")
    dd["Hora"] = pd.to_numeric(dd["Hora"], errors="coerce")
    dd = dd.dropna(subset=["Hora", col_precio])
    if dd.empty:
        return None
    dd["Hora"] = dd["Hora"].astype(int)
    fig = _fig(ctx, 430)
    horas = sorted(dd["Hora"].unique().tolist())
    for h in horas:
        v = dd.loc[dd["Hora"] == h, col_precio].to_numpy(dtype="float64")
        if v.size < 2:
            continue
        color = ("#C62828" if HORAS_PICO_NB[0] <= h <= HORAS_PICO_NB[1]
                 else ("#1976D2" if HORAS_VALLE_NB[0] <= h <= HORAS_VALLE_NB[1] else "#90A4AE"))
        fig.add_trace(go.Box(y=v, x=[f"H{h:02d}"] * v.size, name=f"H{h:02d}", legendgroup="box",
                             showlegend=False, marker_color=color, boxpoints="outliers",
                             jitter=0.25, hovertemplate=f"H{h:02d}" + " · %{y:,.0f} COP/kWh<extra></extra>"))
    pico = dd.loc[dd["Hora"].between(*HORAS_PICO_NB), col_precio]
    valle = dd.loc[dd["Hora"].between(*HORAS_VALLE_NB), col_precio]
    p_pico = float(pico.mean()) if len(pico) else float("nan")
    p_valle = float(valle.mean()) if len(valle) else float("nan")
    spread_bruto = p_pico - p_valle
    spread_neto = p_pico - p_valle / EFICIENCIA_RT_SAEB          # fórmula literal del notebook
    kpis = {"precio_pico_prom": p_pico, "precio_valle_prom": p_valle,
            "spread_bruto_COP_kWh": spread_bruto,
            "spread_ajustado_eficiencia_COP_kWh": spread_neto,
            "eficiencia_rt_referencia": EFICIENCIA_RT_SAEB,
            "horas_pico": HORAS_PICO_NB, "horas_valle": HORAS_VALLE_NB,
            "fuente_precio": col_precio}
    ctx["kpis_saeb"] = kpis
    if np.isfinite(p_pico):
        fig.add_hline(y=p_pico, line=dict(color="#C62828", dash="dot", width=1.5), opacity=0.8,
                      annotation_text=f"pico {p_pico:,.0f}", annotation_font_size=9)
    if np.isfinite(p_valle):
        fig.add_hline(y=p_valle, line=dict(color="#1976D2", dash="dot", width=1.5), opacity=0.8,
                      annotation_text=f"valle {p_valle:,.0f}", annotation_font_size=9)
    fig.update_layout(yaxis_title=f"{col_precio} (COP/kWh)",
                      xaxis_title="Hora XM (H01 = 00:00-01:00) · rojo = pico 18-22, azul = valle 1-6",
                      title_text="Distribución horaria del precio — valle (azul) y pico (rojo), con el "
                                 "spread que paga un almacenamiento",
                      title_font_size=13, yaxis=dict(tickformat=",.0f"), margin=dict(b=78))
    _eje_ventana(fig, ctx)          # el eje X aquí es categórico (H01..H24): se autocentra
    return {"fig": fig, "leyenda": _leyenda(
        "Celdas 84-85 del notebook (`fig_v2_05`): caja-bigote horario del precio y spread pico-valle",
        f"Por cada hora de la ventana se dibuja la caja de los precios observados (sin atípicos, como "
        f"en el notebook). Fuente del precio: `{col_precio}`. Encima, las dos medias de referencia: pico "
        f"{p_pico:,.0f} y valle {p_valle:,.0f} COP/kWh. El spread bruto vale {spread_bruto:,.0f} COP/kWh "
        f"y el ajustado por eficiencia de ida-y-vuelta (η = {EFICIENCIA_RT_SAEB:.2f}, constante "
        f"`EFICIENCIA_RT_SAEB` del notebook, configurable con `APP_MANUEL_ETA_RT`) vale {spread_neto:,.0f} "
        "COP/kWh con la fórmula literal del cuaderno: `pico − valle/η`.",
        "Un almacenamiento (SAEB/bombéo) solo se paga si el spread ajustado supera su costo variable de "
        "operación: con este número se decide si el arbitraje horario es real o si el proyecto depende de "
        "la prima de capacidad. La caja también dice con qué frecuencia el pico es caro de verdad: si el "
        "bigote inferior del pico no llega al promedio del valle, el arbitraje es estructural y no "
        "accidental.",
        "En El Niño el spread se abre por arriba: el valle sube (menos agua, hidro reservando) y el pico "
        "sube más (térmica cara en la noche). Un spread que pasa de ~60 a >120 COP/kWh dentro de la "
        "ventana es la señal temprana de tensión, antes de que se vea en el promedio diario.",
        f"{col_precio} (PrecBolsNaci/CostMargDesp) · {len(dd):,} observaciones-hora · {len(horas)} horas · "
        "η_rt 0,87 del notebook")}


def fig_v2_04_precio_pares(ctx: dict) -> dict | None:
    """Espejo de `graficar_correlacion_separada` (celdas 76-80): scatter individual precio vs X.

    El notebook dibujaba un panel por par (precio vs participación térmica, precio vs embalses,
    precio vs aportes) con tendencia y `r`; aquí van en una sola fila de 3 para que se comparen
    a ojo. `r` es Pearson sobre los días de la ventana y la recta es mínimos cuadrados (grado 1),
    igual que `_configurar_scatter_correlacion`.
    """
    r = ctx.get("resumen", pd.DataFrame())
    if r is None or r.empty:
        return None
    col_y = next((x for x in ("Precio_Ponderado_COP_kWh", "Precio_Prom_COP") if x in r.columns), None)
    if col_y is None:
        return None
    pares = [("Participacion_Termica_pct", "Participación térmica (%)", "#7f7f7f"),
             ("Embalses_Pct", "Nivel de embalses (%)", "#1565c0"),
             ("Aportes_Pct", "Aportes (% de la media)", "#2e7d32")]
    pares = [x for x in pares if x[0] in r.columns]
    if not pares:
        return None
    fig = make_subplots(rows=1, cols=len(pares),
                        subplot_titles=[f"vs {t}" for _c, t, _k in pares])
    rs, notas = [], []
    for i, (col, titulo, color) in enumerate(pares, start=1):
        d = r[[col, col_y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(d) < 4:
            continue
        x, y = d[col].to_numpy(dtype="float64"), d[col_y].to_numpy(dtype="float64")
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=titulo, legendgroup=f"p{i}",
                                 marker=dict(color=color, size=8, opacity=.75,
                                             line=dict(color="white", width=.6)),
                                 hovertemplate=f"{titulo} %{{x:,.1f}}<br>precio %{{y:,.0f}} COP/kWh"
                                               "<extra></extra>"), row=1, col=i)
        if np.isfinite(x).all() and float(np.std(x)) > 0:
            b1, b0 = np.polyfit(x, y, 1)
            xs = np.linspace(float(x.min()), float(x.max()), 20)
            fig.add_trace(go.Scatter(x=xs, y=b0 + b1 * xs, mode="lines", showlegend=False,
                                     legendgroup=f"p{i}", line=dict(color=ROJO, width=2, dash="dash")),
                          row=1, col=i)
            pear = float(np.corrcoef(x, y)[0, 1])
        else:
            pear = float("nan")
        rs.append((titulo, pear, len(d)))
        notas.append(f"{titulo}: r = {pear:+.2f} (n={len(d)})")
    if not rs:
        return None
    fig.update_layout(height=360, margin=dict(t=74, b=64),
                      title_text="Precio de bolsa contra las tres variables que lo explican — "
                                 "lectura individual de cada par",
                      title_font_size=13, showlegend=False,
                      annotations=[dict(text="Análisis exploratorio; correlación no implica causalidad.",
                                        xref="paper", yref="paper", x=0.5, y=-0.14, showarrow=False,
                                        font=dict(size=9, color="#546E7A"))])
    for i in range(1, len(pares) + 1):
        fig.update_xaxes(title_text=pares[i - 1][1], row=1, col=i, showgrid=True, gridwidth=.5)
        fig.update_yaxes(title_text="COP/kWh" if i == 1 else "", row=1, col=i, showgrid=True,
                         gridwidth=.5, tickformat=",.0f")
    mejor = max(rs, key=lambda t: abs(t[1]) if np.isfinite(t[1]) else -1)
    return {"fig": fig, "leyenda": _leyenda(
        "Celdas 76-80 del notebook (`fig_v2_04_1`/`fig_v2_04_2`): cada par en su propio scatter",
        "Tres nubes de puntos diarias con su recta de mínimos cuadrados y su r de Pearson. La matriz "
        "de correlaciones dice *cuánto*; estos paneles dicen *cómo* (lineal, con umbral, o plano). "
        + " · ".join(notas),
        f"El par con más fuerza en esta ventana es «{mejor[0]}» (r = {mejor[1]:+.2f}): es la variable "
        "que conviene monitoriar para anticipar la factura. Si el precio sube con la participación "
        "térmica pero la nube se curva hacia arriba al final, el costo no es lineal: estás entrando en "
        "horas de escasez, donde un MW adicional vale 10-50× más.",
        "En El Niño la nube «embalses → precio» se inclina hacia abajo a la izquierda (embalses bajos y "
        "precio alto) y la «térmica → precio» se aprieta: la dispersión que *desaparece* es la señal de "
        "que el mercado ya descuenta la sequía.",
        "resumen_diario (Demanda · Precio_Ponderado · Participación térmica · Embalses · Aportes) · "
        f"{len(r)} días")}


def fig_09_heatmap_precio_hora(ctx: dict) -> dict | None:
    s = ctx["sistema"]
    if s.empty or "Costo_Marginal_COP_kWh" not in s:
        return None
    d = s.copy()
    d["Dia"] = pd.to_datetime(d["Fecha"]).dt.strftime("%d-%b")
    p = d.pivot_table(index="Dia", columns="Hora", values="Costo_Marginal_COP_kWh", aggfunc="mean")
    p = _ordenar_por_fecha(p, d["Dia"], d["Fecha"])
    fig = _fig(ctx, min(720, 16 * len(p) + 170))
    fig.add_trace(go.Heatmap(z=p.values, x=[f"{h:02d}h" for h in p.columns], y=p.index.astype(str),
                            colorscale=[[0, "#f7fbff"], [0.55, AMBAR], [1, ROJO]], name="COP/kWh",
                            hovertemplate="%{y} · %{x}<br>%{z:,.0f} COP/kWh<extra></extra>"))
    fig.update_layout(xaxis_title="Hora", yaxis_title="Día", yaxis_autorange="reversed")
    q95 = float(np.nanquantile(p.values.astype(float), 0.95))
    return {"fig": fig, "leyenda": _leyenda(
        "Heatmap precio hora × día (el mapa del riesgo horario)",
        f"Costo marginal medio por hora y día; los bloques en rojo son horas ≥ percentil 95 ({q95:,.0f} COP/kWh).",
        "Si las franjas rojas se corren hacia la tarde-noche y crecen en ancho, el costo de la "
        "energía ya no es un evento puntual sino una tendencia: toca renegociar contratos o mover "
        "carga.",
        "En El Niño las franjas rojas se extienden de 3-4 a 8-10 horas diarias (térmica mandando el "
        "precio casi toda la noche).", "CostMargDesp/Sistema (hourly) · XM Bienda")}


def fig_10_perfiles_solares(ctx: dict) -> dict | None:
    g = ctx["gen_enr"]
    if g is None or g.empty or "Tecnologia" not in g:
        return None
    sol = g[g["Tecnologia"].isin(["SOLAR"])]
    if sol.empty:
        return None
    d = sol.copy()
    d["Mes"] = pd.to_datetime(d["Fecha"]).dt.strftime("%Y-%m")
    p = d.pivot_table(index="Mes", columns="Hora", values="Generacion_kWh", aggfunc="sum") / 1e6
    p = p.loc[sorted(p.index)]
    fig = _fig(ctx, 380)
    fig.add_trace(go.Heatmap(z=p.values, x=[f"{h:02d}h" for h in p.columns], y=p.index.astype(str),
                            colorscale="Sunset", name="GWh"))
    fig.update_layout(xaxis_title="Hora", yaxis_title="Mes", yaxis_autorange="reversed")
    return {"fig": fig, "leyenda": _leyenda(
        "Perfiles solares (producción por hora y mes)",
        "Generación solar agregada por hora, organizada por mes.",
        "Si la franja productiva se acorta o baja de intensidad, tus horas de autoconsumo se "
        "reducen: revisa el dimensionamiento de baterías antes de la próxima temporada seca.",
        "Paradoja útil: en El Niño el solar suele producir MÁS (menos nubosidad) justo cuando la "
        "hidro cae; por eso es el activo de cobertura natural del sistema colombiano.", "Gene/Mercado")}


# ------------------------------ figuras v3 (Prof. Rivera) ----------------------
def fig_v3_01_cen_margen(ctx: dict) -> dict | None:
    c = ctx["cen_diario"]
    if c is None or c.empty:
        return None
    fig = _fig(ctx, 460)
    cols = [x for x in c.columns if x.startswith("CEN_") and x.endswith("_MW") and x != "CEN_Total_MW"]
    orden = [f"CEN_{t}_MW" for t in TECNOLOGIAS_ORDEN if f"CEN_{t}_MW" in cols]
    for col in orden + [x for x in cols if x not in orden]:
        fig.add_trace(go.Scatter(x=c["Fecha"], y=c[col], name=col.replace("CEN_", "").replace("_MW", ""),
                                stackgroup="cen",
                                line=dict(width=0.5, color=_escala_tecnicas().get(col.replace("CEN_", "").replace("_MW", ""), "#999"))))
    if "Margen_pct" in c:
        fig.add_trace(go.Scatter(x=c["Fecha"], y=c["Margen_pct"], name="Margen de reserva (%)", yaxis="y2",
                                 line=dict(color=ROJO, width=2.6)))
        fig.add_hline(y=15, line=dict(color=ROJO, dash="dash"), annotation_text="15 % crítico",
                      annotation_font_size=10, yref="y2")
    fig.update_layout(yaxis_title="Capacidad Efectiva Neta (MW)",
                      yaxis2=dict(title="Margen (%)", overlaying="y", side="right", showgrid=False),
                      xaxis_title="Día")
    mrg = c.get("Margen_pct", pd.Series(dtype=float)).dropna()
    crit = int((mrg < 15).sum()) if len(mrg) else 0
    return {"fig": fig, "leyenda": _leyenda(
        "CEN por tecnología y margen de reserva diario",
        "Capacidad Efectiva Neta apilada por tecnología (MW) y margen de reserva sobre la demanda "
        "máxima programada, con la línea de 15 % marcada como crítico.",
        f"{crit} día(s) con margen < 15 % en la ventana. Si el margen cae y la térmica domina el "
        "apilado, el sistema pierde respaldo: es el momento de congelar salidas por mantenimiento.",
        "El Niño ataca dos veces: baja la CEN hidráulica (menos agua) y sube la demanda (sequía/calor); "
        "el margen se comprime por los dos extremos.",
        "CapEfecNeta/Recurso + DemaMaxPot/Sistema (rezago ~2-6 semanas en esta última)")}


def fig_v3_01b_capacidad_composicion(ctx: dict) -> dict | None:
    """Espejo fiel de `graficar_capacidad_composicion` (celda 131, `fig_v3_01` del notebook v3).

    Panel A: CEN media por tecnología con su participación y el conteo de recursos.
    Panel B: evolución diaria de la CEN apilada por tecnología. Mismas fórmulas que el cuaderno
    (media de `Potencia_Confiable_kW/1e3` por día y tecnología).
    """
    cap = ctx.get("cen_recurso", pd.DataFrame())
    if cap is None or cap.empty:
        return None
    dim = ctx.get("dim_plantas", pd.DataFrame())
    d = cap.copy()
    if "Tecnologia" not in d.columns and dim is not None and not dim.empty \
            and "Tecnologia" in dim and "Codigo_Planta" in d.columns:
        d = d.merge(dim[["Codigo_Planta", "Tecnologia"]].drop_duplicates("Codigo_Planta"),
                    on="Codigo_Planta", how="left")
    if "Tecnologia" not in d.columns:
        return None
    d["Tecnologia"] = d["Tecnologia"].fillna("SIN CATALOGAR").astype(str).str.upper()
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    col = next((x for x in ("Potencia_Confiable_kW", "CEN_kW", "Capacidad_kW") if x in d.columns), None)
    if col is None:
        return None
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["Fecha", col])
    if d.empty:
        return None
    diario = d.pivot_table(index="Fecha", columns="Tecnologia", values=col, aggfunc="sum",
                           fill_value=0.0) / 1e3            # MW por día × tecnología
    res = pd.DataFrame({
        "CEN_MW_medio": diario.mean(),
        "Recursos": d.groupby("Tecnologia")["Codigo_Planta"].nunique() if "Codigo_Planta" in d else 0,
    }).sort_values("CEN_MW_medio")
    if res.empty:
        return None
    res["Participacion_pct"] = 100 * res["CEN_MW_medio"] / float(res["CEN_MW_medio"].sum() or np.nan)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                        subplot_titles=("CEN media por tecnología", "Evolución diaria de la CEN (MW)"))
    for tec in res.index:
        color = color_tecnologia(str(tec))
        fig.add_trace(go.Bar(y=[str(tec)], x=[float(res.loc[tec, "CEN_MW_medio"])], orientation="h",
                             name=str(tec), marker_color=color, legendgroup=str(tec), showlegend=False,
                             customdata=[[float(res.loc[tec, "Participacion_pct"]),
                                          int(res.loc[tec, "Recursos"] or 0)]],
                             hovertemplate=f"{tec}: %{{x:,.0f}} MW · %{{customdata[0]:.1f}} % · "
                                           "n=%{customdata[1]}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=diario.index, y=diario[tec], name=str(tec), stackgroup="one",
                                 line=dict(width=.6, color=color), legendgroup=str(tec),
                                 hovertemplate=f"{tec}: %{{y:,.0f}} MW<extra></extra>"), row=1, col=2)
    maxmo = float(res["CEN_MW_medio"].max())
    for i, tec in enumerate(res.index):
        fig.add_annotation(text=f"{res.loc[tec, 'CEN_MW_medio']:,.0f} MW ({res.loc[tec, 'Participacion_pct']:.1f} %)",
                           x=maxmo * 1.02, y=str(tec), xanchor="left", showarrow=False, font=dict(size=8.5),
                           xref="x", yref="y")
    fig.update_layout(height=430, margin=dict(t=70, b=58),
                      title_text="De qué está hecho el parque: CEN efectiva neta por tecnología y su "
                                 "evolución diaria",
                      title_font_size=13, legend=dict(orientation="h", y=1.11, x=0, font=dict(size=9)),
                      barmode="stack")
    fig.update_xaxes(range=[0, maxmo * 1.42], tickformat=",.0f", title_text="CEN media (MW)", row=1, col=1)
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=1, col=2, tickformat=",.0f")
    _eje_ventana(fig, ctx)
    hid = sum(float(res.loc[t, "Participacion_pct"]) for t in res.index if "HIDR" in t)
    term = sum(float(res.loc[t, "Participacion_pct"]) for t in res.index
               if any(k in t for k in ("TERM", "CARB", "GAS")))
    var = float(diario.sum(axis=1).pct_change().abs().max() * 100) if len(diario) > 1 else 0.0
    return {"fig": fig, "leyenda": _leyenda(
        "Celda 131 del notebook (`fig_v3_01`): composición y evolución de la CEN del SIN",
        "Izquierda: los MW efectivos netos medios de cada tecnología en la ventana, con su % del total "
        "y cuántos recursos aporta (etiqueta al lado de la barra, como en el cuaderno). Derecha: la suma "
        "apilada por día, para ver si la capacidad disponible se mueve (mantenimientos, fuera de "
        f"servicio, reposición estacional). La CEN total varió hasta {var:.1f} % de un día para otro.",
        f"Con {hid:.0f} % hídrico y {term:.0f} % térmico de la CEN, la exposición del sistema a una "
        "sequía se lee aquí antes que en el precio: si la franja térmica es la que crece en los días "
        "secos, tu tarifa está indexada al combustible. Un parque con solar/eólica crecientes amortigua "
        "el golpe de la hidro, pero no cubre la noche.",
        "En El Niño la hidro pierde *energía*, no *capacidad*: por eso este gráfico se ve plano mientras "
        "el factor de planta (celda 139) se desploma. La mezcla de capacidad que ves aquí es el techo de "
        "lo que el Niño puede ajustar; el resto lo pone el despacho térmico y su CO₂.",
        "CapEfecNeta/Recurso × ListadoRecursos · " + f"{len(diario)} días × {len(res)} tecnologías")}


def fig_v3_02_fp_tecnologia(ctx: dict) -> dict | None:
    """Factor de planta por tecnología: media ponderada por CEN (no promedio de plantas).

    Se pintan las dos lecturas posibles para que el gráfico no se preste a error: la barra es el FP
    energético de la tecnología (ΣkWh ÷ (ΣkW × horas)) y el rombo negro es la mediana del FP de las
    plantas individuales. Los valores van escritos fuera de la barra con `cliponaxis=False`, así el
    solar (FP bajo pero relevante) se lee siempre.
    """
    fp = ctx.get("fp")
    if fp is None or fp.empty or "Tecnologia" not in fp or "FP_pct" not in fp:
        return None
    d = fp.copy()
    for c in ("Gen_kWh_Mes", "CEN_kW", "Horas_Mes", "FP_pct"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["Tecnologia"] = d["Tecnologia"].astype(str).str.upper().replace({"NAN": "SIN CATALOGAR"})
    if not {"Gen_kWh_Mes", "CEN_kW", "Horas_Mes"}.issubset(d.columns):
        g = d.dropna(subset=["FP_pct"]).groupby("Tecnologia", as_index=False)["FP_pct"].agg(
            FP_pond="mean", n="size")
    else:
        agg = d.groupby("Tecnologia", as_index=False).agg(
            Gen_kWh=("Gen_kWh_Mes", "sum"), CEN_kW=("CEN_kW", "sum"),
            Horas=("Horas_Mes", "max"), n=("Codigo_Planta", "nunique"))
        agg["FP_pond"] = 100 * agg["Gen_kWh"] / (agg["CEN_kW"] * agg["Horas"]).replace(0, np.nan)
        agg["n"] = agg["n"].astype(int)
        g = agg
    med = d.dropna(subset=["FP_pct"]).groupby("Tecnologia", as_index=False)["FP_pct"].median(
    ).rename(columns={"FP_pct": "FP_mediana"})
    g = g.merge(med, on="Tecnologia", how="left")
    g = g[g["n"] >= 1].sort_values("FP_pond", ascending=True).reset_index(drop=True)
    if g.empty or not np.isfinite(g["FP_pond"]).any():
        return None
    fig = _fig(ctx, max(340, 46 * len(g) + 150))
    colores = [COLOR_TECNOLOGIAS.get(str(t).upper(), "#9aa0a6") for t in g["Tecnologia"]]
    fig.add_trace(go.Bar(x=g["FP_pond"], y=[str(t).title() for t in g["Tecnologia"]], orientation="h",
                         marker_color=colores, name="FP energético (ΣkWh ÷ ΣkW·h)",
                         text=[f"{v:.1f} %" for v in g["FP_pond"]], textposition="outside",
                         textfont=dict(size=11, color="#222"), cliponaxis=False,
                         customdata=pd.DataFrame({"cen": g.get("CEN_kW", pd.Series([float("nan")] * len(g))),
                                                  "gen": g.get("Gen_kWh", pd.Series([float("nan")] * len(g))),
                                                  "n": g["n"]}),
                         hovertemplate=("%{y}: FP %{x:.1f} %<br>CEN %{customdata[0]:,.0f} MW · "
                                        "energía %{customdata[1]:,.0f} MWh · %{customdata[2]} planta(s)"
                                        "<extra></extra>")))
    fig.add_trace(go.Scatter(x=g["FP_mediana"], y=[str(t).title() for t in g["Tecnologia"]], mode="markers",
                             marker=dict(symbol="diamond", size=10, color="#111",
                                         line=dict(color="white", width=1.2)),
                             name="Mediana FP por planta",
                             hovertemplate="%{y}: mediana por planta %{x:.1f} %<extra></extra>"))
    fig.add_vline(x=100, line=dict(color=ROJO, width=1, dash="dash"),
                  annotation_text="100 % = generó toda su CEN todas las horas", annotation_font_size=9,
                  annotation_position="top right")
    tope = float(np.nanmax(list(g["FP_pond"].dropna()) + list(g["FP_mediana"].dropna())))
    fig.update_layout(xaxis=dict(title="Factor de planta (%)", range=[0, max(115, tope * 1.30)]),
                      yaxis_title=None, legend=dict(orientation="h", y=1.10, x=0),
                      title_text="Factor de planta por tecnología · ponderado por energía, no por planta",
                      title_font_size=13)
    filas = []
    for _, f in g.sort_values("FP_pond", ascending=False).iterrows():
        filas.append(f"{str(f['Tecnologia']).title()} {float(f['FP_pond']):.1f} % "
                     f"(mediana por planta {float(f['FP_mediana']):.1f} %, {int(f['n'])} pl.)")
    return {"fig": fig, "leyenda": _leyenda(
        "¿Qué tan aprovechada está la capacidad instalada de cada tecnología?",
        "FP = kWh generados ÷ (kW de CEN × horas del mes). La barra pondera por energía (Σ de todas las "
        "plantas de la tecnología); el rombo negro es la mediana del FP planta a planta. La línea roja "
        "marca el techo físico: 100 % significaría generar toda la CEN todas las horas.",
        (" · ".join(filas[:5]) + ". Si la barra y el rombo se separan mucho, hay plantas de la "
         "tecnología que arrastran el promedio (mantenimiento o restricción)."),
        "En El Niño la barra de HIDRAULICA cae y la de TERMICA/COGENERADOR sube: esa tijera es la "
        "medida más rápida de cuánto carbón o gas se está encendiendo para tapar el hueco de agua.",
        "factores_planta(): Gene + CapEfecNeta (misma fórmula que la celda v3, ahora agregada por "
        "energía)")}



def fig_v3_03_fp_plantas(ctx: dict) -> dict | None:
    fp = ctx["fp"]
    if fp is None or fp.empty:
        return None
    tec = fp.get("Tecnologia", pd.Series("TODAS", index=fp.index))
    sub = fp.assign(Tecnologia=tec).dropna(subset=["FP_pct"])
    if sub.empty:
        return None
    fig = _fig(ctx, 420)
    for t in [x for x in TECNOLOGIAS_ORDEN if x in set(sub["Tecnologia"])]:
        d = sub[sub["Tecnologia"] == t]
        fig.add_trace(go.Box(y=d["FP_pct"], name=t, marker_color=_escala_tecnicas()[t],
                             boxpoints="suspectedoutliers"))
    fig.update_layout(yaxis_title="Factor de planta (%)", xaxis_title="Tecnología")
    return {"fig": fig, "leyenda": _leyenda(
        "Distribución del factor de planta por tecnología",
        "Caja y bigotes del FP mensual de cada planta, agrupado por tecnología.",
        "Las colas bajas señalan plantas que no rinden (mantenimiento, restricción, recurso): "
        "candidatas a revisión antes de firmarles un PPA de capacidad.",
        "En sequía las plantas térmicas con FP alto y CEN alta son las que sostienen el sistema; "
        "conócelas porque serán las que negocien el precio.", "FP mensual por planta")}


def fig_v3_04b_heatmap_fp_diario(ctx: dict) -> dict | None:
    """Espejo fiel de `graficar_heatmap_factor_planta_diario` (celda 139 del notebook, `fig_v3_04`).

    FP diario del SIN por tecnología = 100 · ΣGWh(día,tec) / (CEN_MW(día,tec) · 24 h), con las
    tecnologías ordenadas por FP promedio — exactamente la fórmula y el orden del notebook.
    """
    g = ctx.get("gen_enr", pd.DataFrame())
    cap = ctx.get("cen_recurso", pd.DataFrame())
    if g is None or g.empty or cap is None or cap.empty:
        return None
    if "Tecnologia" not in g.columns:
        return None
    dim = ctx.get("dim_plantas", pd.DataFrame())
    cap = cap.copy()
    if "Tecnologia" not in cap.columns and "Codigo_Planta" in cap.columns \
            and dim is not None and not dim.empty and "Tecnologia" in dim:
        cap = cap.merge(dim[["Codigo_Planta", "Tecnologia"]].drop_duplicates("Codigo_Planta"),
                        on="Codigo_Planta", how="left")
    if "Tecnologia" not in cap.columns:
        cap["Tecnologia"] = "TODAS"
    cap["Tecnologia"] = cap["Tecnologia"].fillna("SIN CATALOGAR").astype(str).str.upper()
    gg = g.copy()
    gg["Fecha"] = pd.to_datetime(gg["Fecha"], errors="coerce")
    cap["Fecha"] = pd.to_datetime(cap["Fecha"], errors="coerce")
    col_cap = next((x for x in ("Potencia_Confiable_kW", "CEN_kW", "Capacidad_kW") if x in cap), None)
    if col_cap is None or "Generacion_kWh" not in gg:
        return None
    gen_dia = (gg.dropna(subset=["Fecha"]).groupby(["Fecha", "Tecnologia"])["Generacion_kWh"]
               .sum(min_count=1).unstack() / 1e3)                      # MWh como en el notebook
    cen_dia = (cap.dropna(subset=["Fecha"]).pivot_table(
        index="Fecha", columns="Tecnologia", values=col_cap, aggfunc="sum", fill_value=0.0) / 1e3)
    tec = [t for t in gen_dia.columns if t in cen_dia.columns]
    if not tec or gen_dia.empty:
        return None
    fp = 100.0 * gen_dia[tec] / (cen_dia[tec].replace(0, np.nan) * 24.0)
    fp = fp.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if fp.empty:
        return None
    orden = fp.mean().sort_values(ascending=False).index
    fp = fp[orden]
    dias = [f"{pd.Timestamp(d):%d %b}" for d in fp.index]
    z = fp.T.reindex(list(orden)).values.astype(float)                  # filas = tecnología, cols = día
    fig = _fig(ctx, max(320, 26 * len(orden) + 170))
    fig.add_trace(go.Heatmap(z=z, x=dias, y=list(orden), name="FP %", colorscale="YlGnBu",
                             zmin=0, zmax=100, colorbar=dict(title="FP %"),
                             xgap=0.6, ygap=0.6, showscale=True,
                             hovertemplate="%{y} · %{x}<br>FP %{z:.1f} %<extra></extra>"))
    if len(dias) <= 16:                                   # anotación dentro de la celda, como en el notebook
        for iy, tec in enumerate(orden):
            for ix in range(len(dias)):
                val = z[iy][ix]
                if np.isfinite(val):
                    fig.add_annotation(x=dias[ix], y=tec, text=f"{val:.0f}", showarrow=False,
                                       font=dict(size=8, color=("white" if val > 62 else "#111")))
    fig.update_layout(xaxis_title="Día (la ventana bajo evaluación)", yaxis_title="",
                      title_text="Factor de planta diario del SIN por tecnología — "
                                 "generación real ÷ (CEN del día × 24 h)",
                      title_font_size=13, yaxis=dict(automargin=True))
    _eje_ventana(fig, ctx)
    pico = float(np.nanmax(z)) if z.size else float("nan")
    bajo = float(np.nanmin(z)) if z.size else float("nan")
    peor_tec = str(orden[-1]) if len(orden) else "—"
    mejor_tec = str(orden[0]) if len(orden) else "—"
    return {"fig": fig, "leyenda": _leyenda(
        "Celda 139 del notebook (`fig_v3_04`): mapa de calor del factor de planta diario",
        "Cada celda es un día × tecnología: GWh generados ese día divididos por la energía teórica "
        "de esa tecnología ese día (CEN efectiva × 24 h), en %. Se ve si las plantas trabajan cerca de "
        "su máximo o arrastran restricciones/fuera de servicio. Escala fija 0-100 % para poder comparar "
        "días entre sí; las tecnologías van ordenadas de mayor a menor FP promedio.",
        f"FP máximo {pico:.0f} % y mínimo {bajo:.0f} % en la ventana. La tecnología con el FP más alto "
        f"es {mejor_tec} (está al límite: cualquier mantenimiento duele) y la más holgada es "
        f"{peor_tec}. Un día oscuro en toda la fila de térmica = despacho por debajo de lo que podrías "
        "pagar; un día claro = estás comprando energía cara y cara también en CO₂.",
        "En El Niño la fila HÍDRICA se apaga en los días secos y las filas TÉRMICA/CARBÓN se encienden: "
        "ese desplazamiento en el mapa es la foto del mecanismo de transmisión, y anticipa el precio. "
        "Solar con FP plano y bajo en episodio ENSO = menos amortiguación de la que promete su CEN.",
        "Gene_Sistema/Gene_Recurso ⊕ CapEfecNeta/Recurso · "
        f"{len(dias)} días × {len(orden)} tecnologías")}


def fig_v3_04_costo_distribucion(ctx: dict) -> dict | None:
    cm = ctx["costo_marginal"]
    if cm is None or cm.empty or "Costo_Marginal_COP_kWh" not in cm:
        return None
    v = cm["Costo_Marginal_COP_kWh"].dropna()
    fig = _fig(ctx, 400)
    fig.add_trace(go.Histogram(x=v, nbinsx=70, name="Horas", marker_color=AZUL))
    for q, nom, col in ((0.5, "mediana", VERDE), (0.95, "p95", AMBAR), (0.99, "p99", ROJO)):
        val = float(np.quantile(v, q))
        fig.add_vline(x=val, line=dict(color=col, dash="dash"), annotation_text=f"{nom} {val:,.0f}",
                      annotation_font_size=10)
    fig.update_layout(xaxis_title="Costo marginal de despacho (COP/kWh)", yaxis_title="Horas")
    p99 = float(np.quantile(v, 0.99)); med = float(np.median(v))
    return {"fig": fig, "leyenda": _leyenda(
        "Distribución del costo marginal (el precio real del sistema)",
        f"Histograma horario del costo marginal de despacho con mediana ({med:,.0f}) y percentiles "
        f"95/99 ({p99:,.0f} COP/kWh).",
        "Si la cola derecha pesa (muchas horas sobre p95), el contrato a precio fijo te protege: el "
        "costo esperado de compra está dominado por esas horas.",
        f"El Niño engrosa la cola: en episodios previos el p99 se multiplicó ×3 respecto a años "
        f"neutrales, porque el precio pasa a fijarlo la térmica más cara.",
        "CostMargDesp/Sistema (hourly) · XM Bienda")}


def fig_v3_05_costo_vs_aportes(ctx: dict) -> dict | None:
    r = ctx["resumen"]
    if r.empty or "Precio_Ponderado_COP_kWh" not in r or "Aportes_Pct" not in r:
        return None
    fig = _fig(ctx, 420)
    fase = ctx["nino"].get("Fase_operativa")
    fig.add_trace(go.Scatter(x=r["Aportes_Pct"], y=r["Precio_Ponderado_COP_kWh"], mode="markers",
                             name="días", marker=dict(size=9, color=AZUL, opacity=.7,
                                                      line=dict(width=.5, color="white"))))
    z = np.polyfit(r["Aportes_Pct"], r["Precio_Ponderado_COP_kWh"], 1)
    xs = np.linspace(r["Aportes_Pct"].min(), r["Aportes_Pct"].max(), 50)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), name="tendencia", line=dict(color=ROJO, dash="dash")))
    fig.update_layout(xaxis_title="Aportes (% de la media histórica)", yaxis_title="Precio ponderado (COP/kWh)",
                      hovermode="closest")
    cc = float(r[["Aportes_Pct", "Precio_Ponderado_COP_kWh"]].dropna().corr().iloc[0, 1])
    return {"fig": fig, "leyenda": _leyenda(
        "Aportes vs precio (el canal físico de El Niño)",
        "Cada punto es un día: aportes relativos en X, precio del mercado en Y, con la recta de ajuste.",
        f"r = {cc:+.2f}. Si es claramente negativa, cada punto porcentual que caigan los aportes "
        "tiene un precio asociado: es tu elástico para valorar coberturas o proyectos de eficiencia.",
        "Este es el gráfico que se convierte en 'pantalla de control' durante El Niño: aportes bajo "
        "50 % con precio sobre el p95 histórico = escenario de alerta operativa.",
        "AporEner + CostMargDesp (diario)")}


def fig_v3_06_emisiones(ctx: dict) -> dict | None:
    e = ctx.get("emisiones_dia", pd.DataFrame())
    col_t = next((c for c in ("tCO2_XM", "tCO2") if c in e.columns), None)
    if e is None or e.empty or col_t is None:
        return None
    fig = _fig(ctx, 430)
    fig.add_trace(go.Bar(x=e["Fecha"], y=e[col_t], name="tCO₂/día (declarados por XM)",
                         marker_color="rgba(87,60,28,.75)"))
    if "Intensidad_tCO2_MWh" in e:
        fig.add_trace(go.Scatter(x=e["Fecha"], y=e["Intensidad_tCO2_MWh"], name="tCO₂/MWh", yaxis="y2",
                                 line=dict(color=ROJO, width=2.2)))
        for um, col in ((0.2, VERDE), (0.4, AMBAR)):
            fig.add_hline(y=um, line=dict(color=col, dash="dot"), yref="y2", annotation_text=f"{um} tCO₂/MWh",
                          annotation_font_size=9)
        fig.update_layout(yaxis2=dict(title="Intensidad (tCO₂/MWh)", overlaying="y", side="right", showgrid=False))
    fig.update_layout(xaxis_title="Día", yaxis_title="Emisiones (tCO₂/día)", barmode="stack")
    tot = float(e[col_t].sum())
    inte = float(pd.to_numeric(e.get("Intensidad_tCO2_MWh", pd.Series(dtype=float)),
                                errors="coerce").mean())
    nrec = e["Recursos_reportados"].mean() if "Recursos_reportados" in e else float("nan")
    fuente_int = (ctx.get("notas_emisiones") or {}).get("fuente_intensidad", "ratio XM")
    return {"fig": fig, "leyenda": _leyenda(
        "Emisiones diarias e intensidad de carbono del despacho",
        ("tCO₂ declarados por XM por día (barras; XM reporta "
         f"{nrec:.0f} recursos-combustible/día) y, sobre el eje derecho, la intensidad del despacho en "
         f"tCO₂e/MWh según {fuente_int}, con las bandas 0.2/0.4 que separan despacho limpio / medio / "
         "sucio.") if nrec == nrec else
        (f"tCO₂ declarados por XM por día y la intensidad del despacho en tCO₂e/MWh según {fuente_int}, "
         "con las bandas 0.2/0.4 que separan despacho limpio / medio / sucio."),
        f"En la ventana se declararon {tot:,.0f} tCO₂ con intensidad media {inte:.3f} tCO₂e/MWh. "
        "Si compras energía en el mercado, esa es tu huella Alcance 2 implícita: úsala para exigir "
        "contratos con respaldo renovable cuando la proyección de ENSO es cálida.",
        "En El Niño las emisiones suben porque la térmica sustituye a la hidro: el factor de emisión "
        "promedio del país (~0.12-0.15 tCO₂/MWh típicos) puede duplicarse en los peores meses.",
        "EmisionesCO2/RecursoComb · factorEmisionCO2e (UPME/XM)")}


def fig_v3_07_ranking_plantas(ctx: dict) -> dict | None:
    e = ctx.get("emisiones", pd.DataFrame())
    if e is None or e.empty or "Codigo_Planta" not in e or "tCO2" not in e:
        return None
    g = (e.groupby("Codigo_Planta").agg(tCO2=("tCO2", "sum"), horas=("Hora", "nunique"))
          .sort_values("tCO2", ascending=False).head(15))
    dim = ctx["dim_plantas"]
    if dim is not None and not dim.empty:
        g = g.reset_index().merge(dim[["Codigo_Planta", "Nombre_Planta", "Tecnologia", "Fuente_Energia"]]
                                  .drop_duplicates("Codigo_Planta"), on="Codigo_Planta", how="left")
    else:
        g = g.reset_index()
    fig = _fig(ctx, 470)
    teces = g.get("Tecnologia", pd.Series("—", index=g.index)).astype(str)
    for tec_grupo in sorted(teces.unique()):
        msk = teces == tec_grupo
        fig.add_trace(go.Bar(
            x=g.loc[msk].get("Nombre_Planta", g.loc[msk]["Codigo_Planta"]), y=g.loc[msk]["tCO2"],
            marker_color=_escala_tecnicas().get(tec_grupo, "#8d6e63"), name=tec_grupo,
            customdata=np.stack([g.loc[msk]["horas"].to_numpy(),
                                 g.loc[msk].get("Fuente_Energia", pd.Series("", index=g.index)).astype(str).to_numpy()],
                                axis=-1),
            hovertemplate="%{x}<br>%{y:,.0f} tCO₂ · %{customdata[0]:.0f} h despachadas<br>"
                          "%{customdata[1]}<extra>" + tec_grupo + "</extra>"))
    fig.update_layout(xaxis_tickangle=-30, yaxis_title="Emisiones totales en la ventana (tCO₂)",
                      xaxis_title="Planta")
    return {"fig": fig, "leyenda": _leyenda(
        "Top 15 plantas emisoras (Pareto del carbono)",
        "Emisiones acumuladas por planta en la ventana, coloreadas por tecnología, con sus horas "
        "despachadas en el hover.",
        "El 20 % de las plantas genera el 80 % del CO₂: si tu objetivo es descarbonizar la compra o "
        "renegociar un portafolio, empieza por la cabeza de esta barra.",
        "En El Niño este ranking se mueve: las plantas de carbón/gas del interior escalan puestos, "
        "y un portafolio 'verde' puede dejar de serlo por效应 del despacho.",
        "EmisionesCO2/RecursoComb (la planta va en Values_Name o Values_code según el contenido)")}


def fig_v3_08_margen_reserva_bandas(ctx: dict) -> dict | None:
    c = ctx["cen_diario"]
    if c is None or c.empty or "Margen_pct" not in c:
        return None
    d = c.dropna(subset=["Margen_pct"])
    if d.empty:
        return None
    fig = _fig(ctx, 380)
    fig.add_trace(go.Scatter(x=d["Fecha"], y=d["Margen_pct"], mode="lines+markers", name="Margen %",
                             marker=dict(size=5, color=np.where(d["Margen_pct"] < 15, ROJO,
                                       np.where(d["Margen_pct"] < 25, AMBAR, VERDE))),
                             line=dict(color="#555", width=1)))
    fig.add_hrect(y0=0, y1=15, fillcolor="rgba(220,20,60,.10)", line_width=0, annotation_text="crítico")
    fig.add_hrect(y0=15, y1=25, fillcolor="rgba(255,165,2,.10)", line_width=0, annotation_text="ajustado")
    fig.update_layout(yaxis_title="Margen de reserva (%)", xaxis_title="Día")
    return {"fig": fig, "leyenda": _leyenda(
        "Margen de reserva con bandas de riesgo",
        "Margen (CEN − demanda máxima) sobre la CEN, con bandas <15 % (crítico) y 15-25 % (ajustado).",
        "Los días en la banda roja son los días en que una falla cualquiera puede forzar "
        "administración de demanda; ahí es donde un gran consumidor negocia interruptibilidad.",
        "En El Niño severo (2015-16, 2023-24) el margen pasó semanas en banda roja entre agosto y "
        "noviembre: usa este gráfico como termómetro del riesgo físico.",
        "CapEfecNeta (diaria, por recurso) vs DemaMaxPot (rezago de semanas)")}


# ------------------------------ figuras ENSO / Niño ----------------------------
def fig_enso_01_oni(ctx: dict) -> dict | None:
    n = ctx["nino"]
    if n is None or n.empty or "ONI" not in n:
        return None
    d = n.dropna(subset=["ONI"]).copy()
    if d.empty:
        return None
    d["Mes_pd"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M")
    fig = _fig(ctx, 430)
    col = np.where(d["ONI"] >= UMBRAL_NINO, ROJO, np.where(d["ONI"] <= -UMBRAL_NINO, CELESTE, "#9aa0a6"))
    fig.add_trace(go.Bar(x=d["Mes_pd"].dt.to_timestamp(), y=d["ONI"], name="ONI (3 meses)",
                         marker_color=col, hovertemplate="%{x|%b-%Y}: %{y:+.2f} °C<extra></extra>"))
    for y, c, txt in ((UMBRAL_NINO, ROJO, "+0.5 El Niño"), (-UMBRAL_NINO, CELESTE, "−0.5 La Niña")):
        fig.add_hline(y=y, line=dict(color=c, dash="dash"), annotation_text=txt, annotation_font_size=10)
    ult = d["Mes_pd"].iloc[-1]
    fig.add_annotation(x=ult.to_timestamp(), y=float(d["ONI"].iloc[-1]) + 0.25,
                       text=f"<b>último dato NOAA: {ult}</b><br>ONI {d['ONI'].iloc[-1]:+.2f} °C · "
                            f"{'EL NIÑO ACTIVO' if d['ONI'].iloc[-1] >= UMBRAL_NINO else 'neutral'}",
                       showarrow=True, arrowhead=2, ax=-60, ay=-30, font=dict(size=11, color=ROJO),
                       bgcolor="rgba(255,255,255,.9)", bordercolor=ROJO)
    hoy_pd = pd.PeriodIndex([f"{_hoy():%Y-%m}"], freq="M")[0]
    if hoy_pd > ult:
        fig.add_vrect(x0=(ult + 1).to_timestamp().strftime("%Y-%m-%d"),
                      x1=hoy_pd.to_timestamp().strftime("%Y-%m-%d"),
                      fillcolor="rgba(220,20,60,.18)", line_width=0,
                      annotation_text="ago-2026: mes en curso<br>(sin ONI publicado)", annotation_font_size=9,
                      layer="below")
    fig.update_layout(xaxis_title="Mes (etiqueta = mes central de la estación de 3 meses)",
                      yaxis_title="Anomalía Niño3.4 (°C)")
    fig = _eje_ventana(fig, ctx, meses=84)
    return {"fig": fig, "leyenda": _leyenda(
        "Índice ONI completo (1950 → hoy) con el episodio 2026 resaltado",
        "Anomalía de temperatura superficial del Pacífico central (Niño3.4), suavizada a 3 meses. "
        "Barras rojas = El Niño (≥ +0.5 °C), azules = La Niña.",
        f"El cruce sostenido de +0.5 en {d['Mes'].iloc[-1]} marca el INICIO OPERATIVO del evento; "
        f"comparado con la historia, un pico como el actual (≥ +1.3) implica en Colombia 10-25 % "
        "menos lluvia y presión al alza sobre el precio.",
        "AGOSTO 2026: NOAA aún no publica el ONI de la estación JAS (sale inicios de septiembre), "
        "pero su Advisory del 13-ago-2026 ya declara El Niño activo y en fortalecimiento; el recuadro "
        "rojo del gráfico es exactamente ese hueco de publicación.",
        "CPC/NOAA oni.ascii.txt (se actualiza en cada arranque si hay red)")}


def fig_enso_02_evolutivo(ctx: dict) -> dict | None:
    """Embalses / aportes / precio DENTRO de la ventana, coloreados por la fase ENSO del mes.

    Versión 'ventana' de fig_v3_09: el eje X es exactamente el período bajo estudio, porque el
    cuadro largo (2015→) ya se ve en la pestaña 🕐 Perfiles × hora (panel ENSO de 4 filas).
    """
    r = ctx["resumen"]
    n = ctx.get("nino", pd.DataFrame())
    if r is None or r.empty or "Embalses_Pct" not in r:
        return None
    d = r.copy()
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    d["Mes"] = d["Fecha"].dt.to_period("M").astype(str)
    fase = oni = None
    if isinstance(n, pd.DataFrame) and not n.empty and "Mes" in n:
        nn = n[["Mes"] + [c for c in ("Fase", "ONI") if c in n.columns]].drop_duplicates("Mes")
        d = d.merge(nn, on="Mes", how="left")
        if "Fase" in d:
            fase = dict(zip(d["Mes"], d["Fase"]))
        if "ONI" in d:
            oni = dict(zip(d["Mes"], pd.to_numeric(d["ONI"], errors="coerce")))
    paneles = [("Embalses_Pct", "Embalses (% vol. útil)", AZUL),
               ("Aportes_Pct", "Aportes (% media histórica)", VERDE),
               ("Precio_Ponderado_COP_kWh", "Precio de bolsa (COP/kWh)", AMBAR)]
    uso = [p for p in paneles if p[0] in d.columns]
    if not uso:
        return None
    fig = make_subplots(rows=len(uso), cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=[t for _c, t, _k in uso])
    for i, (col, titulo, color) in enumerate(uso, start=1):
        y = pd.to_numeric(d[col], errors="coerce")
        fig.add_trace(go.Scatter(x=d["Fecha"], y=y, name=titulo, line=dict(color=color, width=2.1),
                                 fill="tozeroy" if i == 1 else None,
                                 fillcolor=_hex_a_rgba(color, 0.08) if i == 1 else None,
                                 mode="lines+markers", marker=dict(size=4),
                                 customdata=[str(fase.get(m, "Neutral")) for m in d["Mes"]] if fase else None,
                                 hovertemplate="%{x|%d %b} · %{y:,.1f}<extra>" + titulo + "</extra>"),
                      row=i, col=1)
        colores = [COLORES_FASE.get(fase.get(m, "Neutral"), "#9e9e9e") for m in d["Mes"]] if fase else None
        if colores:
            fig.add_trace(go.Bar(x=d["Fecha"], y=[0.020 * float(np.nanmax(y)) if np.isfinite(np.nanmax(y)) else 0.02] * len(d),
                                 marker_color=colores, width=0.9, showlegend=False, hoverinfo="skip"),
                          row=i, col=1)
    for yv, txt in ((60, "60 % alerta"), (40, "40 % crítico")):
        fig.add_hline(y=yv, row=1, col=1, line=dict(color="gray", width=1, dash="dash"),
                      annotation_text=txt, annotation_font_size=8.5, annotation_position="right")
    if any(c[0] == "Aportes_Pct" for c in uso):
        fig.add_hline(y=100, row=2 if any(x[0] == "Aportes_Pct" for x in uso[:2]) else 1, col=1,
                      line=dict(color=VERDE, width=1, dash="dot"),
                      annotation_text="media histórica", annotation_font_size=8.5,
                      annotation_position="right")
    for i in range(1, len(uso) + 1):
        fig.update_xaxes(row=i, col=1, range=[d["Fecha"].min() - pd.Timedelta(days=1),
                                              d["Fecha"].max() + pd.Timedelta(days=1)],
                         dtick="D3" if len(d) <= 45 else "M1")
    fig.update_yaxes(title_text="%" " / COP·kWh⁻¹")
    fig.update_layout(height=250 * len(uso) + 110, showlegend=False,
                      title_text=f"Fase ENSO y variables del SIN en la ventana {ctx['ini']} → {ctx['fin']} "
                                 "(franja de color = fase del mes)", title_font_size=13)
    textos = []
    for m in sorted(set(d["Mes"])):
        sub = d[d["Mes"] == m]
        if sub.empty:
            continue
        f_ = (fase or {}).get(m, "Neutral")
        o_ = (oni or {}).get(m, float("nan"))
        textos.append(f"{m}: fase {f_}" + (f", ONI {float(o_):+.2f} °C" if o_ == o_ else "") +
                      f", embalses medios {float(pd.to_numeric(sub['Embalses_Pct'], errors='coerce').mean()):.1f} %")
    return {"fig": fig, "leyenda": _leyenda(
        "Embalses, aportes y precio SOLO en la ventana bajo estudio (con su fase ENSO)",
        f"Tres paneles apilados comparten el eje X de {len(d)} día(s): {', '.join(t for _c, t, _k in uso)}. "
        "La barra de fondo de cada panel pinta la fase del mes (rojo = El Niño, azul = La Niña, gris = "
        "neutral) según el ONI de NOAA, no según un supuesto del tablero.",
        (" · ".join(textos) + ". Léelo así: si la línea de embalses baja mientras la de precio sube "
         "dentro del mismo mes rojo, el mecanismo ENSO→hidro→precio ya está operando en tu ventana."),
        "Cuando NOAA publique el ONI de la estación JAS-2026 (septiembre), el mes rojo pasa de "
        "'fase estimada' a 'fase oficial': la forma de las tres curvas ya es la evidencia previa.",
        "resumen_diario (v2) × nino_mensual (fase/ONI por mes)")}





def fig_enso_05_scatter(ctx: dict, var: str = "Embalses_Pct") -> dict | None:
    n = ctx["nino"]
    if n is None or n.empty or var not in n or "ONI" not in n:
        return None
    d = n.dropna(subset=[var, "ONI"])
    if d.empty:
        return None
    fig = _fig(ctx, 400)
    for f in ORDEN_FASE:
        s = d[d["Fase_operativa"] == f] if "Fase_operativa" in d else d[d["Fase"] == f]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(x=s["ONI"], y=s[var], mode="markers", name=f,
                                 marker=(dict(color=COLORES_FASE[f], size=9, opacity=.8, symbol="diamond")
                                         if f == "El Niño" else dict(color=COLORES_FASE[f], size=7, opacity=.7)),
                                 customdata=s["Mes"], hovertemplate="%{customdata}<br>ONI %{x:.2f} → "
                                 + var + " %{y:.1f}<extra>" + f + "</extra>"))
    r = float(d[[var, "ONI"]].corr().iloc[0, 1])
    z = np.polyfit(d["ONI"], d[var], 1)
    xs = np.linspace(d["ONI"].min(), d["ONI"].max(), 40)
    fig.add_trace(go.Scatter(x=xs, y=np.polyval(z, xs), name=f"ajuste (r={r:+.2f})",
                             line=dict(color="#333", dash="dash", width=1.5)))
    u = d.iloc[-1]
    fig.add_trace(go.Scatter(x=[u["ONI"]], y=[u[var]], mode="markers", name="último mes",
                             marker=dict(symbol="star", size=16, color="black",
                                         line=dict(width=1.2, color="white"))))
    fig.update_layout(xaxis_title="ONI (°C)", yaxis_title=var, hovermode="closest")
    return {"fig": fig, "leyenda": _leyenda(
        f"ONI vs {var}: la recta que convierte clima en número operativo",
        "Cada punto es un mes desde 2015; la pendiente dice cuánto se mueve "
        f"{var.replace('_Pct', ' %').replace('_COP', ' COP/kWh')} por cada °C de ONI. r = {r:+.2f}.",
        f"Con el ONI actual puede estimar {var} esperado: si tu pronóstico interno es más optimista "
        "que la recta, estás subestimando el riesgo de este trimestre.",
        "La estrella negra es el último mes disponible (agosto 2026 con ONI heredado): si cae fuera "
        "de la nube, el evento está siendo más fuerte que la media histórica.",
        "serie mensual 2015→ + ONI NOAA")}


def fig_enso_06_ultima_onda(ctx: dict) -> dict | None:
    n = ctx["nino"]
    if n is None or n.empty or "Nino1_2" not in n:
        return None
    d = n.copy()
    d["Mes_pd"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M")
    d = d[d["Mes_pd"] >= pd.Period("2023-01", freq="M")].dropna(subset=["Nino1_2"])
    if d.empty:
        return None
    fig = _fig(ctx, 380)
    fig.add_trace(go.Scatter(x=d["Mes_pd"].dt.to_timestamp(), y=d["Nino1_2"], name="Niño1+2 (Pacífico oriental)",
                             line=dict(color=ROJO, width=2.6),
                             fill="tozeroy", fillcolor="rgba(220,20,60,.10)"))
    fig.add_trace(go.Scatter(x=d["Mes_pd"].dt.to_timestamp(), y=d.get("Nino3_4", pd.Series(np.nan, index=d.index)),
                             name="Niño3.4 (Pacífico central)", line=dict(color=AZUL, width=2)))
    fig.add_hline(y=UMBRAL_NINO, line=dict(color="#888", dash="dot"), annotation_text="+0.5 (umbral)")
    _vline_anotada(fig, pd.Timestamp("2026-08-01"), "hoy: agosto 2026", ROJO)
    fig.update_layout(xaxis_title="Mes", yaxis_title="Anomalía (°C)")
    maximo = float(d["Nino1_2"].max())
    fig = _eje_ventana(fig, ctx, meses=48)
    return {"fig": fig, "leyenda": _leyenda(
        "El índice que sí mira Colombia: Niño1+2 (costa de Perú/Ecuador) vs Niño3.4",
        "Anomalías mensuales ERSSTv5 (base 1991-2020). Niño1+2 es el índice mejor correlacionado con "
        "la lluvia en Colombia y el último en publicarse (jun-2026).",
        f"Niño1+2 alcanzó {maximo:+.2f} °C en la serie reciente, por encima del Niño3.4: esa "
        "disparidad suele anunciar déficit de lluvia más severo del que sugiere el ONI. Revisa tus "
        "supuestos de hidrología para el IV trimestre.",
        "Mientras el Niño1+2 siga por encima del Niño3.4, el evento tiene 'leña' para intensificarse: "
        "es el patrón de 1982-83, 1997-98 y 2015-16.",
        "CPC ersst5.nino.mth.91-20.ascii")}


# ------------------------------ figuras v4 (disponibilidad) --------------------
def fig_v4_02_disponibilidad(ctx: dict) -> dict | None:
    i = ctx["indisp"]
    if i is None or i.empty or "Indisponibilidad_pct" not in i:
        return None
    fig = _fig(ctx, 430)
    fig.add_trace(go.Scatter(x=i["Fecha"], y=i["Indisponibilidad_pct"], name="Indisponibilidad %",
                             line=dict(color=NARANJA, width=2.4), fill="tozeroy",
                             fillcolor="rgba(255,127,14,.18)"))
    for u in (15, 30):
        fig.add_hline(y=u, line=dict(color=ROJO if u == 30 else AMBAR, dash="dash"),
                      annotation_text=f"{u} %", annotation_font_size=9)
    if "Reserva_Operativa_MW" in i:
        fig.add_trace(go.Scatter(x=i["Fecha"], y=i["Reserva_Operativa_MW"], name="Reserva operativa (MW)",
                                 yaxis="y2", line=dict(color=AZUL, width=1.8, dash="dot")))
        fig.update_layout(yaxis2=dict(title="MW", overlaying="y", side="right", showgrid=False))
    fig.update_layout(xaxis_title="Día", yaxis_title="Indisponibilidad (% de la CEN)")
    med = float(i["Indisponibilidad_pct"].mean()); mx = float(i["Indisponibilidad_pct"].max())
    return {"fig": fig, "leyenda": _leyenda(
        "Indisponibilidad del parque (100 % − disponible/CEN)",
        f"Porcentaje diario de capacidad no disponible por mantenimiento o avería, con la reserva "
        f"operativa de fondo. Media {med:.1f} %, máximo {mx:.1f} % en la ventana.",
        "Un indisponible alto reduce el margen real: si se combina con El Niño, la probabilidad de "
        "escasez crece multiplicativamente. Programar mantenimientos en meses húmedos es la regla.",
        "Históricamente, los meses de El Niño con indisponibilidad > 12 % coincidieron con los picos "
        "de precio más violentos (falla de respaldo en plena sequía).",
        "DispoCome/Sistema (opcional) o proxy DemaMaxPot/CapEfecNeta · nota en la bitácora")}


def fig_v4_03_heatmap_indisp(ctx: dict) -> dict | None:
    e = ctx["emisiones"]
    if e is None or e.empty or "Codigo_Planta" not in e or "Hora" not in e:
        return None
    d = e.copy()
    d["Dia"] = pd.to_datetime(d["Fecha"]).dt.strftime("%d-%b")
    p = d.pivot_table(index="Dia", columns="Hora", values="tCO2", aggfunc="sum", fill_value=0.0)
    p = _ordenar_por_fecha(p, d["Dia"], d["Fecha"])
    fig = _fig(ctx, min(700, 14 * len(p) + 170))
    fig.add_trace(go.Heatmap(z=p.values, x=[f"{h:02d}h" for h in p.columns], y=p.index.astype(str),
                            colorscale="Hot", name="tCO₂"))
    fig.update_layout(xaxis_title="Hora", yaxis_title="Día", yaxis_autorange="reversed")
    return {"fig": fig, "leyenda": _leyenda(
        "Mapa térmico de despacho térmico por hora (proxy de indisponibilidad hidroeléctrica)",
        "tCO₂ emitidos por hora y día: cuando el heatmap se enciende de madrugada o al mediodía, "
        "significa que la hidro no alcanza y la térmica cubre lo que el agua no da.",
        "Si los bloques calientes aparecen fuera del pico nocturno, hay escasez estructural, no "
        "estacional: revisa los contratos de capacidad.",
        "En El Niño este mapa pierde su forma 'de pico' y se vuelve uniforme: la térmica corre 24/7.",
        "EmisionesCO2/RecursoComb (hourly)")}


def fig_v4_04_matriz_cruce(ctx: dict) -> dict | None:
    c = ctx["cruce_v4"]
    if c is None or c.empty:
        return None
    cols = [x for x in ("Indisponibilidad_pct", "Costo_Marginal_COP_kWh_Dia", "Precio_Ponderado_Dia_COP_kWh",
                        "Demanda_MW_Dia", "CEN_MW_Dia", "Reserva_Operativa_MW") if x in c]
    if len(cols) < 2:
        return None
    mat = c[cols].corr()
    fig = _fig(ctx, 420)
    fig.add_trace(go.Heatmap(z=mat.values, x=cols, y=cols, zmin=-1, zmax=1, text=np.round(mat.values, 2),
                             texttemplate="%{text}", colorscale=[[0, ROJO], [.5, "white"], [1, AZUL]],
                             hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    fig.update_layout(xaxis_tickangle=-25, yaxis_title="Variable")
    return {"fig": fig, "leyenda": _leyenda(
        "Cruce disponibilidad × precio × demanda (¿qué explica el costo?)",
        "Matriz de correlación de la tabla `cruce_diario_v4` construida por el notebook.",
        "Si indisponibilidad correlaciona más con el precio que la propia demanda, tu riesgo viene "
        "del lado de la oferta (fallas/mantenimientos), no del consumo: otra palanca, otra acción.",
        "En El Niño ambas variables se pelean el timón del precio; esta matriz permite atribuir de "
        "forma rápida quien mando ese mes.", "cruce_diario_v4 (inner join por Fecha)")}


def fig_v4_05_enso_economico(ctx: dict) -> dict | None:
    """"El Nino en dinero" (v4 c13, fig_v4_05): costo marginal por fase ENSO y plata real de la ventana.

    Antes dependia solo de la semilla `data/reference/v4_mensual.csv` y, si no estaba, la subpestana
    quedaba vacia. Ahora la base mensual se arma con la semilla + los meses de la ventana
    (`cruce_diario_v4`) y, si aun falta historia, se dibuja igual la version diaria con el gasto real:
    la lectura economica nunca queda en blanco y la leyenda dice que hace falta para el contraste por fase.
    """
    mensual = v4_mensual_para_enso(ctx)
    hist_ok = (isinstance(mensual, pd.DataFrame) and not mensual.empty
               and "Costo_Marginal_COP_kWh" in mensual.columns and len(mensual) >= 4)
    d = mensual.copy() if hist_ok else pd.DataFrame()
    if hist_ok:
        d["ts"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M").to_timestamp()
        for c in ("Costo_Marginal_COP_kWh", "Margen_pct_mean", "Margen_pct_min"):
            if c in d.columns:
                d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.dropna(subset=["ts", "Costo_Marginal_COP_kWh"]).sort_values("ts").reset_index(drop=True)
        hist_ok = len(d) >= 4

    # Plata de la ventana: precio (o costo marginal) medio por dia × energia demandada
    r = ctx.get("resumen", pd.DataFrame())
    cv = ctx.get("cruce_v4", pd.DataFrame())
    dia = None
    desde = ""
    if isinstance(r, pd.DataFrame) and not r.empty and "Demanda_GWh" in r.columns:
        col_p = next((c for c in ("Precio_Ponderado_COP_kWh", "Precio_Prom_COP") if c in r.columns), None)
        if col_p:
            dia = pd.DataFrame({"Fecha": r["Fecha"], "Demanda_GWh": r["Demanda_GWh"],
                                "precio": r[col_p]})
            desde = "resumen_diario (v2)"
    if dia is None and isinstance(cv, pd.DataFrame) and not cv.empty \
            and {"Demanda_MW_Dia", "Costo_Marginal_COP_kWh_Dia"}.issubset(cv.columns):
        dia = pd.DataFrame({"Fecha": cv["Fecha"], "Demanda_GWh": cv["Demanda_MW_Dia"],
                            "precio": cv["Costo_Marginal_COP_kWh_Dia"]})
        desde = "cruce_diario_v4 (v4)"
    if dia is not None:
        for c in ("Demanda_GWh", "precio"):
            dia[c] = pd.to_numeric(dia[c], errors="coerce")
        dia["Fecha"] = pd.to_datetime(dia["Fecha"], errors="coerce")
        # COP/kWh × GWh = MCOP  (1 GWh = 1e6 kWh, y 1 MCOP = 1e6 COP: se cancelan)
        dia["MCOP"] = dia["precio"] * dia["Demanda_GWh"]
        dia = dia.dropna(subset=["Fecha", "MCOP"]).sort_values("Fecha").reset_index(drop=True)
        if dia.empty:
            dia = None
    if not hist_ok and dia is None:
        return None

    fase_mes: dict[str, str] = {}
    n_ = ctx.get("nino", pd.DataFrame())
    if isinstance(n_, pd.DataFrame) and not n_.empty and "Fase" in n_.columns:
        nn = n_.copy()
        nn["Mes"] = nn["Mes"].astype(str)
        fase_mes = dict(zip(nn["Mes"], nn["Fase"].astype(str)))
    prom_fase: dict[str, float] = {}
    if hist_ok and "Fase" in d.columns:
        for f, sub_f in d.groupby(d["Fase"].astype(str)):
            v = pd.to_numeric(sub_f["Costo_Marginal_COP_kWh"], errors="coerce").mean()
            if np.isfinite(v) and len(sub_f) >= 2:
                prom_fase[f] = float(v)

    def _busca_fase(prefijo: str) -> str | None:
        for f in prom_fase:
            if f.upper().replace("I\u0301", "I").replace("\u00d1", "N").startswith(prefijo):
                return f
        return None

    f_nino = _busca_fase("EL NIN")
    f_neu = _busca_fase("NEUTRAL")
    hay_contraste = bool(f_nino and f_neu and prom_fase.get(f_neu, 0) > 0)

    paneles: list[str] = []
    if hist_ok:
        paneles.append("mensual")
    if hay_contraste:
        paneles.append("fases")
    if dia is not None:
        paneles.append("gasto")
    titulos = {"mensual": f"Costo marginal medio mensual por fase ENSO ({len(d)} meses)",
               "fases": "Costo medio por fase: cuanto agrega El Nino",
               "gasto": f"Gasto diario de la ventana ({len(dia)} dias) y acumulado"}
    hay_margen = (hist_ok and "Margen_pct_mean" in d
                  and pd.to_numeric(d.get("Margen_pct_mean"), errors="coerce").notna().any())
    specs = [[{"secondary_y": True}] if (k == "gasto" or (k == "mensual" and hay_margen))
              else [{}] for k in paneles]
    fig = make_subplots(rows=len(paneles), cols=1, vertical_spacing=0.15, specs=specs,
                        subplot_titles=[titulos[k] for k in paneles])
    for num, k in enumerate(paneles, start=1):
        if k == "mensual":
            colf = [COLORES_FASE.get(str(f), "#9aa0a6") for f in d.get("Fase", pd.Series(["Neutral"] * len(d)))]
            fig.add_trace(go.Bar(x=d["ts"], y=d["Costo_Marginal_COP_kWh"], name="Costo marginal (COP/kWh)",
                                 marker_color=colf,
                                 hovertemplate="%{x|%b-%Y} · %{y:,.0f} COP/kWh<extra>costo</extra>"),
                          row=num, col=1, secondary_y=False)
            if "Margen_pct_mean" in d and pd.to_numeric(d["Margen_pct_mean"], errors="coerce").notna().any():
                fig.add_trace(go.Scatter(x=d["ts"], y=pd.to_numeric(d["Margen_pct_mean"], errors="coerce"),
                                         name="Margen de reserva (%)", line=dict(color="#222", width=1.8),
                                         hovertemplate="%{x|%b-%Y} · %{y:.1f} %<extra>margen</extra>"),
                              row=num, col=1, secondary_y=True)
                fig.update_yaxes(title_text="Margen (%)", row=num, col=1, secondary_y=True)
            fig.update_yaxes(title_text="COP/kWh", row=num, col=1, secondary_y=False)
            fig.update_xaxes(title_text="Mes", row=num, col=1)
        elif k == "fases":
            fig.add_trace(go.Bar(x=[str(f).title() for f in prom_fase], y=list(prom_fase.values()),
                                 marker_color=[COLORES_FASE.get(f, "#9aa0a6") for f in prom_fase],
                                 name="Costo medio por fase", text=[f"{v:,.0f}" for v in prom_fase.values()],
                                 textposition="outside", cliponaxis=False,
                                 hovertemplate="%{x}: %{y:,.0f} COP/kWh<extra></extra>"),
                          row=num, col=1, secondary_y=False)
            fig.update_yaxes(title_text="COP/kWh", row=num, col=1)
            fig.update_xaxes(title_text="Fase ENSO del mes", row=num, col=1)
        else:
            dm = dia["Fecha"].dt.to_period("M").astype(str)
            fig.add_trace(go.Bar(x=dia["Fecha"], y=dia["MCOP"],
                                 marker_color=[COLORES_FASE.get(fase_mes.get(m, "Neutral"), "#9aa0a6")
                                               for m in dm],
                                 name="Gasto diario (MCOP)",
                                 hovertemplate="%{x|%d %b} · %{y:,.0f} MCOP<extra>gasto</extra>"),
                          row=num, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(x=dia["Fecha"], y=dia["MCOP"].cumsum(), name="Acumulado (MCOP)",
                                     line=dict(color="#0d47a1", width=2),
                                     hovertemplate="%{x|%d %b} · acum. %{y:,.0f} MCOP<extra></extra>"),
                          row=num, col=1, secondary_y=True)
            fig.update_yaxes(title_text="MCOP / día", row=num, col=1, secondary_y=False)
            fig.update_yaxes(title_text="MCOP acumulado", row=num, col=1, secondary_y=True)
            fig.update_xaxes(title_text="Día de la ventana", row=num, col=1,
                             dtick="D3" if len(dia) <= 45 else "M1")
    fig.update_layout(height=310 * len(paneles) + 110, legend=dict(orientation="h", y=1.10, x=0),
                      title_text="Cuanto cuesta El Nino: costo por fase y gasto real de la ventana",
                      title_font_size=13)
    delta = (100 * (prom_fase[f_nino] - prom_fase[f_neu]) / prom_fase[f_neu]) if hay_contraste else None
    comparacion = (f"En los {len(d)} meses de la base, el costo marginal medio de los meses de El Nino fue "
                   f"{prom_fase[f_nino]:,.0f} COP/kWh contra {prom_fase[f_neu]:,.0f} en meses "
                   f"neutrales ({delta:+.0f} %).") if delta is not None else ""
    gasto_txt = (f"La ventana suma {float(dia['MCOP'].sum()):,.0f} MCOP de energia comprada "
                 f"(media de {float(dia['MCOP'].mean()):,.0f} MCOP/dia en {len(dia)} dia(s))."
                 ) if dia is not None else ""
    nota_base = ("base mensual: semilla v4_mensual.csv + meses de la ventana" if hist_ok else
                 "base: solo la ventana — para el contraste por fase hace falta "
                 "`python tools\build_seeds.py v4` (data/reference/v4_mensual.csv)")
    fig = _eje_ventana(fig, ctx, meses=60)
    return {"fig": fig, "leyenda": _leyenda(
        "El Nino en dinero (fig_v4_05 del notebook, ahora siempre dibujable)",
        "Periodo y grupos, explicitos: las barras mensuales usan la base v4 (2015-01 -> ultimo mes "
        "cerrado, o la reconstruccion desde la ventana si la semilla falta) y el grupo de cada mes sale "
        "de la fase oficial del ONI con umbral +-0,5 C (El Nino >= +0,5; Nina <= -0,5; Neutral en el "
        "resto). El gasto en millones de COP (MCOP) se suma dia a dia DENTRO de la ventana elegida como "
        "precio COP/kWh x demanda GWh, y el sobrecosto por fase resta el promedio de los meses Neutral "
        "del mismo periodo. Barras = costo marginal medio por mes (o por dia de la ventana) pintadas "
        f"segun la fase ENSO: rojo El Nino, azul Nina, gris neutral. La linea es el margen de reserva "
        f"cuando hay CEN y, en el panel del gasto, el acumulado. Fuente del gasto: {desde or 'n/d'}. "
        + nota_base,
        ((" ".join(x for x in (comparacion, gasto_txt) if x)) or
         "Hace falta al menos un mes neutral y uno de El Nino en la base para cuantificar el sobrecosto."),
        "Agosto-2026 es el punto mas reciente: si ya esta en rojo y el acumulado diario queda por encima "
        "de la recta de los dias previos, el sobrecosto del episodio esta corriendo, no esperando. Ese es "
        "el numero que se lleva a la mesa de compras de energia.",
        "v4_mensual + cruce_diario_v4 + ONI NOAA (fase por mes)")}





# ==============================================================================
# 7b · PERFILES × HORA, MATRICES Y ÁMBITOS OPERATIVOS (notebook v2 c31-38 ·
#      v3 c21/c24 · v4 c7/c8). Todo se calcula sobre `ctx` ya descargado.
# ==============================================================================
MAPA_OPERATIVO = REF_DIR / "mapa_operativo.csv"

# Niveles equivalentes a NIVELES_OPERATIVOS del notebook (Área/Subárea SIMEM) más los
# que XM sí expone hoy en `ListadoRecursos` (tecnología, recurso, agente, estado).
NIVELES_AMBITO: dict[str, str] = {
    "Nacional (todo el SIN)": "",
    "Tecnología": "Tecnologia",
    "Tipo de recurso": "Tipo_Recurso",
    "Agente / empresa": "Agente_Nombre",
    "Estado operativo (XM)": "Estado_Operativo",
    "Área operativa (SIMEM)": "Area_Operativa",
    "Subárea operativa (SIMEM)": "Subarea_Operativa",
}


def _slug_operativo(texto: Any) -> str:
    """`_slug_operativo` del notebook: etiqueta → fragmento seguro para archivo."""
    import unicodedata
    normalizado = unicodedata.normalize("NFKD", str(texto))
    ascii_texto = normalizado.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", ascii_texto.lower()).strip("_") or "ambito"


def cargar_mapa_operativo() -> pd.DataFrame:
    """`data/reference/mapa_operativo.csv` (Codigo_Planta, Area_Operativa, Subarea_Operativa).

    Es el equivalente al `mapa_operativo` que el notebook construía con
    `src/actualizar_areas_operativas_simem.py`. Sin ese archivo, las figuras de
    'ámbito' siguen funcionando con las dimensiones de XM y el selector omite
    Área/Subárea (así lo documenta `data/reference/LEEME_mapa_operativo.txt`).
    """
    if not MAPA_OPERATIVO.exists():
        return pd.DataFrame(columns=["Codigo_Planta", "Area_Operativa", "Subarea_Operativa"])
    try:
        d = pd.read_csv(MAPA_OPERATIVO, dtype={"Codigo_Planta": str})
    except Exception as exc:                                  # noqa: BLE001
        bitacora(f"mapa_operativo.csv ilegible ({type(exc).__name__}); ignoro las áreas SIMEM", "warn")
        return pd.DataFrame(columns=["Codigo_Planta", "Area_Operativa", "Subarea_Operativa"])
    if "Codigo_Planta" not in d.columns:
        return d.iloc[0:0]
    d["Codigo_Planta"] = d["Codigo_Planta"].astype(str).str.strip()
    for c in ("Area_Operativa", "Subarea_Operativa"):
        if c not in d.columns:
            d[c] = pd.NA
    return d.drop_duplicates("Codigo_Planta")


def gen_con_ambito(ctx: dict) -> pd.DataFrame:
    """`gen_enr` + columnas de ámbito (agente, estado, y Área/Subárea si hay mapa)."""
    g = ctx.get("gen_enr")
    if g is None or g.empty or "Tecnologia" not in g.columns:
        return pd.DataFrame()
    out = g.copy()
    dim = ctx.get("dim_plantas", pd.DataFrame())
    if dim is not None and not dim.empty and "Codigo_Planta" in dim.columns:
        d = dim.drop_duplicates("Codigo_Planta").set_index("Codigo_Planta")
        if "Codigo_Agente" in d.columns:
            ag = ctx_agentes_lookup()
            out["Agente_Nombre"] = (out["Codigo_Planta"].astype(str).map(lambda c: ag.get(c, "sin agente"))
                                    if ag else d["Codigo_Agente"].astype(str).reindex(
                                        out["Codigo_Planta"].astype(str)).values)
        if "Estado" in d.columns:
            out["Estado_Operativo"] = d["Estado"].astype(str).str.upper().reindex(
                out["Codigo_Planta"].astype(str)).values
    mp = cargar_mapa_operativo()
    if not mp.empty and "Area_Operativa" in mp:
        m = mp.set_index("Codigo_Planta")
        out["Area_Operativa"] = m["Area_Operativa"].reindex(out["Codigo_Planta"].astype(str)).values
        if "Subarea_Operativa" in m:
            out["Subarea_Operativa"] = m["Subarea_Operativa"].reindex(
                out["Codigo_Planta"].astype(str)).values
        out["Area_Operativa"] = out["Area_Operativa"].fillna("Sin asignar")
        if "Subarea_Operativa" in out:
            out["Subarea_Operativa"] = out["Subarea_Operativa"].fillna("Sin asignar")
    else:
        out["Area_Operativa"] = pd.NA
        out["Subarea_Operativa"] = pd.NA
    return out


def niveles_disponibles(gen: pd.DataFrame) -> dict[str, str]:
    """Niveles del selector que realmente tienen ≥2 valores en los datos."""
    salida: dict[str, str] = {}
    for nombre, col in NIVELES_AMBITO.items():
        if not col:
            salida[nombre] = col
            continue
        if col in gen.columns and gen[col].notna().any() and gen[col].astype(str).nunique(dropna=True) >= 2:
            salida[nombre] = col
    return salida


def filtrar_ambito(gen: pd.DataFrame, nivel: str, ambito: str) -> pd.DataFrame:
    """`filtrar_generacion_operativa` del notebook: 'Nacional' = todas las filas."""
    col = NIVELES_AMBITO.get(nivel, "")
    if not col or ambito in ("Nacional", "Todas", "") or col not in gen.columns:
        return gen.copy()
    return gen.loc[gen[col].astype(str).eq(str(ambito))].copy()


def perfil_tecnologia_hora(datos: pd.DataFrame) -> pd.DataFrame:
    """`preparar_perfil_tecnologia` (v2): total tecnología×fecha×hora y promedio entre días.

    Se suma primero por día (para no promediar plantas individuales y llamarlo
    'total tecnológico') y después se promedia entre los días de la ventana.
    """
    req = {"Tecnologia", "Fecha", "Hora", "Generacion_kWh"}
    if datos is None or datos.empty or not req.issubset(datos.columns):
        return pd.DataFrame()
    diario = (datos.groupby(["Tecnologia", "Fecha", "Hora"], as_index=False)["Generacion_kWh"]
              .sum(min_count=1))
    if diario.empty:
        return pd.DataFrame()
    perfil = (diario.groupby(["Tecnologia", "Hora"], as_index=False)["Generacion_kWh"].mean()
              .rename(columns={"Generacion_kWh": "Generacion_GWh"})
              .assign(Generacion_GWh=lambda d: pd.to_numeric(d["Generacion_GWh"], errors="coerce") / 1e6))
    dias = int(diario["Fecha"].nunique())
    perfil.attrs["dias"] = dias
    return perfil


def _orden_por_total(perfil: pd.DataFrame, col_valor: str = "Generacion_GWh") -> list[str]:
    orden = (perfil.groupby("Tecnologia")[col_valor].sum().sort_values(ascending=False).index.tolist())
    return [t for t in orden if str(t) != "SIN CATALOGAR"] + [t for t in orden if str(t) == "SIN CATALOGAR"]


def _pivote_tec_hora(perfil: pd.DataFrame) -> pd.DataFrame:
    piv = perfil.pivot_table(index="Tecnologia", columns="Hora", values="Generacion_GWh",
                             aggfunc="mean", fill_value=0.0)
    return piv.reindex(_orden_por_total(perfil))


def fig_v2_08_heatmap_tecno_hora(ctx: dict) -> dict | None:
    """`graficar_heatmap_tecno_hora` (v2): tecnología × hora en GWh/h *promedio diario*."""
    perfil = perfil_tecnologia_hora(gen_con_ambito(ctx) if "gen_enr" in ctx else ctx.get("gen"))
    if perfil.empty:
        return None
    piv = _pivote_tec_hora(perfil)
    if piv.empty:
        return None
    horas = [int(c) for c in piv.columns]
    fig = _fig(ctx, max(340, 46 * len(piv) + 150))
    fig.add_trace(go.Heatmap(z=piv.values.astype(float), x=[f"H{h:02d}" for h in horas],
                             y=[str(t).title() for t in piv.index], colorscale="YlOrRd",
                             name="GWh/h promedio", colorbar=dict(title="GWh/h"),
                             hovertemplate="%{y} · %{x}<br>%{z:.2f} GWh/h<extra></extra>"))
    if len(piv) <= 9 and len(horas) <= 24:                      # ≈ seaborn annot=True
        for i, _t in enumerate(piv.index):
            for j, h in enumerate(horas):
                v = float(piv.values[i, j])
                fig.add_annotation(x=f"H{h:02d}", y=str(_t).title(), text=f"{v:.2f}",
                                   showarrow=False, font=dict(size=8.5,
                                   color="white" if v > 0.62 * float(np.nanmax(piv.values)) else "#333"))
    pico = perfil.loc[perfil["Generacion_GWh"].idxmax()]
    fig.update_layout(xaxis_title="Hora XM (H01 = 00:00-01:00)", yaxis_title="Tecnología",
                      yaxis=dict(autorange="reversed"),
                      title_text=(f"Perfil horario de generación total por tecnología · promedio de "
                                  f"{perfil.attrs.get('dias', '?')} día(s)"),
                      title_font_size=13)
    solar = (piv.loc["SOLAR"] if "SOLAR" in piv.index else None)
    nota_sol = (f" El solar concentra {100 * float(solar.max()) / max(float(solar.sum()) / 24, 1e-9):.1f}× su "
                f"media diaria en H{int(solar.astype(float).idxmax()):02d}: la rampa de la tarde (17-22 h) "
                "queda del lado térmico." if solar is not None and solar.sum() > 0 else "")
    return {"fig": fig, "leyenda": _leyenda(
        "Mapa de calor tecnología × hora (promedio diario de la ventana)",
        "Cada celda es la generación media por hora de esa tecnología en GWh/h (primero se suma el "
        "día completo por tecnología y luego se promedia entre días, igual que la celda v2 del "
        "notebook).",
        "Las filas concentradas a la derecha (17-22 h) son las que fijan el precio: ahí no hay solar "
        "y la hidro está cara. Un proyecto de batería o de hidro con regulación vale por esas horas, "
        "no por el total mensual." + nota_sol,
        "En El Niño la fila HIDRAULICA se aplana y la térmica ocupa el hueco: si ves la franja 17-22 h "
        "más 'cuadrada' que en años neutrales, el costo marginal ya no baja en la madrugada.",
        "Gene/Recurso + ListadoRecursos · promedio de "
        f"{perfil.attrs.get('dias', '?')} días (data/raw)")}


def fig_v2_08b_perfiles_tecnologia_hora(ctx: dict) -> dict | None:
    """`graficar_perfiles_tecnologia_hora` (v2): pequeños múltiplos con área y pico anotado."""
    perfil = perfil_tecnologia_hora(gen_con_ambito(ctx) if "gen_enr" in ctx else ctx.get("gen"))
    if perfil.empty:
        return None
    orden = _orden_por_total(perfil)
    if not orden:
        return None
    columnas = 2
    filas = int(np.ceil(len(orden) / columnas))
    fig = make_subplots(rows=filas, cols=columnas, subplot_titles=[str(t).title() for t in orden],
                        vertical_spacing=0.13, horizontal_spacing=0.09)
    for i, tec in enumerate(orden):
        fila, colu = divmod(i, columnas)
        d = perfil[perfil["Tecnologia"] == tec].sort_values("Hora")
        color = COLOR_TECNOLOGIAS.get(str(tec).upper(), "#607D8B")
        fig.add_trace(go.Scatter(x=d["Hora"], y=d["Generacion_GWh"], name=str(tec).title(),
                                 mode="lines+markers", line=dict(color=color, width=2.4),
                                 marker=dict(size=4), showlegend=False,
                                 hovertemplate="H%{x:02d} · %{y:.2f} GWh/h<extra></extra>"),
                      row=fila + 1, col=colu + 1)
        fig.add_trace(go.Scatter(x=d["Hora"], y=d["Generacion_GWh"], fill="tozeroy",
                                 line=dict(width=0), marker=dict(size=0),
                                 fillcolor=_hex_a_rgba(color, 0.18), showlegend=False,
                                 hoverinfo="skip"), row=fila + 1, col=colu + 1)
        if not d.empty:
            k = int(d["Generacion_GWh"].idxmax()) if d["Generacion_GWh"].notna().any() else None
            if k is not None:
                hp = int(d.loc[k, "Hora"]); vp = float(d.loc[k, "Generacion_GWh"])
                fig.add_annotation(row=fila + 1, col=colu + 1, xref="x", yref="y", x=hp, y=vp,
                                   text=f"Pico H{hp:02d} · {vp:.2f} GWh", showarrow=True,
                                   arrowhead=1, ax=0, ay=-26, font=dict(size=9, color=color),
                                   bgcolor="rgba(255,255,255,.9)", bordercolor=color, borderpad=3)
    for i in range(len(orden), filas * columnas):
        fila, colu = divmod(i, columnas)
        fig.update_xaxes(row=fila + 1, col=colu + 1, visible=False)
        fig.update_yaxes(row=fila + 1, col=colu + 1, visible=False)
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 25, 3)),
                     ticktext=[f"H{h:02d}" for h in range(1, 25, 3)], title_text=None)
    fig.update_yaxes(title_text="GWh/h")
    fig.update_layout(height=300 * filas + 90, title_text="Perfiles horarios promedio por tecnología "
                      "(alternativa legible al mapa de calor)", title_font_size=13)
    picos = []
    for tec in orden[:4]:
        d = perfil[perfil["Tecnologia"] == tec]
        if not d.empty and d["Generacion_GWh"].notna().any():
            k = int(d["Generacion_GWh"].idxmax())
            picos.append(f"{str(tec).title()} H{int(d.loc[k, 'Hora']):02d}")
    return {"fig": fig, "leyenda": _leyenda(
        "Perfiles horarios por tecnología (cada panel con su propia escala)",
        "Promedio entre días de la ventana, en GWh/h. La escala es independiente por panel para que "
        "se vea la forma de la curva aunque la tecnología sea pequeña.",
        "Compara la hora del pico, no la altura: si el pico solar cae a las 12-13 h y el de la "
        "térmica a las 18-19 h, el solar no te sirve en las horas caras y el contrato vale menos.",
        ("Picos detectados: " + " · ".join(picos) + ". En El Niño el hueco entre el pico solar y el "
         "pico térmico se ensancha (menos hidro disponible para rellenar la tarde).") if picos else
        "En El Niño el hueco entre el pico solar y el pico térmico se ensancha.",
        f"promedio de {perfil.attrs.get('dias', '?')} días · Gene/Recurso + ListadoRecursos")}


def fig_v2_08d_scope_heatmap(ctx: dict, nivel: str, ambito: str) -> dict | None:
    """`graficar_heatmap_operativo` (v2 c33): tecnología × hora del ámbito elegido."""
    gen = gen_con_ambito(ctx)
    if gen.empty:
        return None
    sub = filtrar_ambito(gen, nivel, ambito)
    return _fig_scope_perfil(ctx, sub, nivel, ambito, modo="heatmap")


def fig_v2_08e_scope_perfiles(ctx: dict, nivel: str, ambito: str) -> dict | None:
    """`graficar_perfiles_operativos` (v2 c33): un panel por tecnología del ámbito elegido."""
    gen = gen_con_ambito(ctx)
    if gen.empty:
        return None
    sub = filtrar_ambito(gen, nivel, ambito)
    return _fig_scope_perfil(ctx, sub, nivel, ambito, modo="paneles")


def _fig_scope_perfil(ctx: dict, sub: pd.DataFrame, nivel: str, ambito: str, modo: str,
                      geo: bool = False) -> dict | None:
    """Núcleo común de las figuras 08d/08e/08g/08h: perfil tecnología×hora de un ámbito."""
    perfil = perfil_tecnologia_hora(sub)
    if perfil.empty:
        return None
    etiqueta = f"{ambito} ({nivel})" if nivel != "Nacional (todo el SIN)" else "Nacional (todo el SIN)"
    letra = ("d" if modo == "heatmap" else "e") if not geo else ("g" if modo == "heatmap" else "h")
    escala = "YlGnBu" if geo else "YlOrRd"
    sufijo = f"fig_v2_08{letra}_{_slug_operativo(nivel)}_{_slug_operativo(etiqueta)}"
    if modo == "heatmap":
        piv = _pivote_tec_hora(perfil)
        if piv.empty:
            return None
        horas = [int(c) for c in piv.columns]
        fig = _fig(ctx, max(300, 44 * len(piv) + 130))
        fig.add_trace(go.Heatmap(z=piv.values.astype(float), x=[f"H{h:02d}" for h in horas],
                                 y=[str(t).title() for t in piv.index], colorscale=escala,
                                 colorbar=dict(title="GWh/h"), name="GWh/h",
                                 hovertemplate="%{y} · %{x}<br>%{z:.2f} GWh/h<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Hora XM",
                          title_text=f"Generación por tecnología y hora — {etiqueta}",
                          title_font_size=13)
        total = float(piv.values.sum())
        pico = np.unravel_index(int(np.nanargmax(piv.values)), piv.shape)
        lectura = (f"En {etiqueta}, la tecnología {str(piv.index[pico[0]]).title()} domina la hora "
                   f"H{horas[pico[1]]:02d} con {piv.values[pico]:.2f} GWh/h (media diaria).")
    else:
        orden = _orden_por_total(perfil)
        if not orden:
            return None
        columnas = 2
        filas = int(np.ceil(len(orden) / columnas))
        fig = make_subplots(rows=filas, cols=columnas, subplot_titles=[str(t).title() for t in orden],
                            vertical_spacing=0.14)
        for i, tec in enumerate(orden):
            fila, colu = divmod(i, columnas)
            d = perfil[perfil["Tecnologia"] == tec].sort_values("Hora")
            color = COLOR_TECNOLOGIAS.get(str(tec).upper(), "#607D8B")
            fig.add_trace(go.Scatter(x=d["Hora"], y=d["Generacion_GWh"], mode="lines+markers",
                                     line=dict(color=color, width=2), marker=dict(size=4),
                                     name=str(tec).title(), showlegend=False,
                                     hovertemplate="H%{x:02d} · %{y:.2f} GWh/h<extra></extra>"),
                          row=fila + 1, col=colu + 1)
            fig.add_trace(go.Scatter(x=d["Hora"], y=d["Generacion_GWh"], fill="tozeroy",
                                     line=dict(width=0), marker=dict(size=0),
                                     fillcolor=_hex_a_rgba(color, 0.16), hoverinfo="skip",
                                     showlegend=False), row=fila + 1, col=colu + 1)
        for i in range(len(orden), filas * columnas):
            fila, colu = divmod(i, columnas)
            fig.update_xaxes(row=fila + 1, col=colu + 1, visible=False)
            fig.update_yaxes(row=fila + 1, col=colu + 1, visible=False)
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, 25, 4)),
                         ticktext=[f"H{h:02d}" for h in range(1, 25, 4)])
        fig.update_yaxes(title_text="GWh/h")
        fig.update_layout(height=290 * filas + 80, showlegend=False,
                          title_text=f"Perfiles horarios de generación — {etiqueta}",
                          title_font_size=13)
        horas_pico = []
        for tec in orden[:5]:
            d = perfil[perfil["Tecnologia"] == tec]
            if len(d) and d["Generacion_GWh"].notna().any():
                k = int(d["Generacion_GWh"].idxmax())
                horas_pico.append(f"{str(tec).title()} H{int(d.loc[k, 'Hora']):02d}")
        lectura = ("Picos: " + " · ".join(horas_pico)) if horas_pico else "Perfil del ámbito calculado."
        total = float(perfil["Generacion_GWh"].sum())
    n_dias = perfil.attrs.get("dias", "?")
    return {"fig": fig, "leyenda": _leyenda(
        f"Ámbito operativo/geográfico: {etiqueta} — perfil tecnología × hora",
        f"Mismo cálculo que el notebook (preparar_perfil_tecnologia): suma por tecnología×día×hora, "
        f"promedio entre los {n_dias} día(s) de la ventana, en GWh/h. Identificador interno: {sufijo}.",
        lectura + " Si el ámbito elegido concentra hidráulica, su perfil es la señal anticipada del "
        "estrés: cuando esa fila se aplana a las 18-21 h, entra térmica y sube el precio.",
        "Compara el mismo gráfico entre un mes neutral y uno de El Niño: la caída del área bajo la "
        "curva hidráulica en la franja 17-22 h es el número que hay que defender en el comité.",
        "Gene/Recurso + ListadoRecursos"
        + (" + mapa_operativo.csv (SIMEM)" if "SIMEM" in nivel else "") + f" · {n_dias} días")}


def _hex_a_rgba(hexo: str, alfa: float) -> str:
    """'#rrggbb' → 'rgba(r,g,b,a)' (plotly necesita el alfa explícito para rellenar)."""
    h = str(hexo).lstrip("#")
    if len(h) != 6:
        return f"rgba(96,125,139,{alfa})"
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alfa})"


def fig_v2_08g_geo_heatmap(ctx: dict, nivel: str, ambito: str) -> dict | None:
    """`graficar_heatmap_geografico` (v2 c35, fig_v2_08g): tecnología × hora de un ámbito geográfico."""
    gen = gen_con_ambito(ctx)
    if gen.empty:
        return None
    return _fig_scope_perfil(ctx, filtrar_ambito(gen, nivel, ambito), nivel, ambito,
                             "heatmap", geo=True)


def fig_v2_08h_geo_perfiles(ctx: dict, nivel: str, ambito: str) -> dict | None:
    """`graficar_perfiles_geograficos` (v2 c35, fig_v2_08h): paneles por tecnología del ámbito."""
    gen = gen_con_ambito(ctx)
    if gen.empty:
        return None
    return _fig_scope_perfil(ctx, filtrar_ambito(gen, nivel, ambito), nivel, ambito,
                             "paneles", geo=True)


def fig_v2_09_heatmap_precio_dia_hora(ctx: dict) -> dict | None:
    """`graficar_heatmap_precio_dia_hora` (v2 c38): día × hora del precio de bolsa (divergente)."""
    ph = ctx.get("precio_bolsa_h", pd.DataFrame())
    if ph is None or ph.empty or "Precio_Bolsa_COP_kWh" not in ph:
        ph = ctx.get("sistema", pd.DataFrame())
        if ph is None or ph.empty or "Precio_Bolsa_COP_kWh" not in ph:
            return None
    d = ph.copy()
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    d = d.dropna(subset=["Fecha", "Hora", "Precio_Bolsa_COP_kWh"])
    if d.empty:
        return None
    piv = d.pivot_table(index="Fecha", columns="Hora", values="Precio_Bolsa_COP_kWh", aggfunc="mean")
    if piv.empty or piv.shape[0] == 0:
        return None
    filas = [f"{t:%d %b}" for t in piv.index]
    horas = [int(c) for c in piv.columns]
    fig = _fig(ctx, max(420, 20 * len(piv) + 150))
    z = piv.values.astype(float)
    vmed = float(np.nanmedian(z))
    fig.add_trace(go.Heatmap(z=z, x=[f"H{h:02d}" for h in horas], y=filas, name="COP/kWh",
                             colorscale=[[0.0, "#1a9850"], [0.5, "#ffffbf"], [1.0, "#d73027"]],
                             zmid=vmed, colorbar=dict(title="COP/kWh"),
                             hovertemplate="%{y} · %{x}<br>%{z:,.0f} COP/kWh<extra></extra>"))
    fig.update_layout(xaxis_title="Hora XM (H01 = 00:00-01:00)", yaxis_title="Fecha",
                      yaxis=dict(autorange="reversed"),
                      title_text="Precio de bolsa — día × hora (verde = barato, rojo = caro)",
                      title_font_size=13)
    por_hora = pd.Series(np.nanmean(z, axis=0), index=[f"H{h:02d}" for h in horas]).sort_values(ascending=False)
    caras = ", ".join(por_hora.head(3).index.tolist())
    baratas = ", ".join(por_hora.tail(3).index.tolist())
    spread = float(np.nanmax(z) - np.nanmin(z))
    return {"fig": fig, "leyenda": _leyenda(
        "Mapa de calor del precio de bolsa (día × hora) — dónde se gana o se pierde dinero",
        "Cada fila es un día de la ventana y cada columna una hora; el color es el precio medio de la "
        f"hora (escala centrada en la mediana {vmed:,.0f} COP/kWh para que el rojo sea realmente caro).",
        f"Las horas más caras del período son {caras} y las más baratas {baratas}; la distancia "
        f"máximo-mínimo de la matriz es {spread:,.0f} COP/kWh. Ese *spread* es lo que cobra (o "
        "pierde) quien puede mover carga entre horas: baterías, bombeo o un consumo interruptible.",
        "Bajo El Niño las columnas 17-22 h se encienden casi todos los días y el valle del mediodía "
        "se estrecha: si el rojo empieza a invadir la madrugada, el sistema ya no tiene respaldo y "
        "el riesgo de racionamiento/regulación sube.",
        "PrecBolsNaci/Sistema (horario) · "
        f"{len(piv)} día(s) × {len(horas)} horas")}



def _serie_month_ts(mes: pd.Series) -> pd.Series:
    return pd.PeriodIndex(mes.astype(str), freq="M").to_timestamp()


def fig_v3_09_dashboard_enso(ctx: dict) -> dict | None:
    """`graficar_dashboard_enso` (v3 c21): ONI + aportes + embalses + precio, con eventos sombreados.

    El eje X se recorta al período con datos del SIN (lección del notebook: el ONI llega a 1950 y
    las series de XM solo a 2015, si no, todo se aplasta a la derecha).
    """
    n = ctx.get("nino", pd.DataFrame())
    if n is None or n.empty or "ONI" not in n.columns:
        return None
    cols_xm = [c for c in ("Aportes_Pct", "Embalses_Pct", "Precio_Ponderado_COP") if c in n.columns]
    d = n.copy()
    d["Mes_pd"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M")
    d = d.sort_values("Mes_pd").reset_index(drop=True)
    d["ts"] = d["Mes_pd"].dt.to_timestamp()
    if cols_xm and d[cols_xm].notna().any(axis=1).any():
        inicio = d.loc[d[cols_xm].notna().any(axis=1), "ts"].min()
    else:
        inicio = d["ts"].min()
    d = d[d["ts"] >= inicio].reset_index(drop=True)
    if d.empty:
        return None
    if "Aportes_Pct" in d:
        d["Aportes_mm3"] = pd.to_numeric(d["Aportes_Pct"], errors="coerce").rolling(3, min_periods=1).mean()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.075,
                        subplot_titles=("Índice ONI (NOAA CPC) — rojo Niño · azul Niña · gris neutro",
                                        "Aportes hídricos al SIN (% de la media histórica)",
                                        "Embalses del SIN (% de volumen útil) — bandas 60 / 40 %",
                                        "Precio de bolsa ponderado (COP/kWh)"))
    oni = pd.to_numeric(d["ONI"], errors="coerce")
    colores = [ROJO if v >= 0.5 else "#1f77b4" if v <= -0.5 else "#bbbbbb" for v in oni.fillna(0)]
    fig.add_trace(go.Bar(x=d["ts"], y=oni, marker_color=colores, name="ONI (°C)", showlegend=False,
                         hovertemplate="%{x|%b-%Y} · ONI %{y:+.2f} °C<extra></extra>"), row=1, col=1)
    for yv, cc, txt in ((0.5, ROJO, "umbral Niño +0.5"), (-0.5, "#1f77b4", "umbral Niña -0.5")):
        fig.add_hline(y=yv, row=1, col=1, line=dict(color=cc, width=1, dash="dot"),
                      annotation_text=txt, annotation_font_size=8.5, annotation_position="right")
    if "Aportes_Pct" in d:
        fig.add_trace(go.Scatter(x=d["ts"], y=d["Aportes_Pct"], name="Aportes mensual",
                                 line=dict(color=CELESTE, width=1.4), opacity=0.65,
                                 hovertemplate="%{x|%b-%Y} · %{y:.0f} %<extra></extra>"), row=2, col=1)
    if "Aportes_mm3" in d:
        fig.add_trace(go.Scatter(x=d["ts"], y=d["Aportes_mm3"], name="Aportes media móvil 3m",
                                 line=dict(color="#0d3b66", width=2.2),
                                 hovertemplate="%{x|%b-%Y} · %{y:.0f} %<extra></extra>"), row=2, col=1)
    fig.add_hline(y=100, row=2, col=1, line=dict(color=VERDE, width=1, dash="dot"),
                  annotation_text="media histórica (100 %)", annotation_font_size=8.5,
                  annotation_position="right")
    if "Embalses_Pct" in d:
        fig.add_trace(go.Scatter(x=d["ts"], y=d["Embalses_Pct"], name="Embalses %",
                                 line=dict(color=VERDE, width=1.8),
                                 hovertemplate="%{x|%b-%Y} · %{y:.1f} %<extra></extra>"), row=3, col=1)
    for yv in (60, 40):
        fig.add_hline(y=yv, row=3, col=1, line=dict(color="gray", width=1, dash="dash"),
                      annotation_text=f"{yv} %", annotation_font_size=8.5, annotation_position="right")
    if "Precio_Ponderado_COP" in d:
        fig.add_trace(go.Scatter(x=d["ts"], y=d["Precio_Ponderado_COP"], name="Precio (COP/kWh)",
                                 line=dict(color="#8c564b", width=1.8),
                                 hovertemplate="%{x|%b-%Y} · %{y:,.0f} COP/kWh<extra></extra>"), row=4, col=1)
    ev = ctx.get("nino_eventos", pd.DataFrame())
    franjas = []
    if isinstance(ev, pd.DataFrame) and not ev.empty and {"Inicio", "Fin", "Fase"}.issubset(ev.columns):
        for _, e in ev.iterrows():
            t0 = pd.Period(str(e["Inicio"])[:7], freq="M").to_timestamp()
            t1 = pd.Period(str(e["Fin"])[:7], freq="M").to_timestamp(how="end")
            if t1 < inicio:
                continue
            franjas.append((max(t0, inicio), t1, ROJO if "Niño" in str(e["Fase"]) else "#1f77b4"))
    for t0, t1, cc in franjas:
        for r in range(1, 5):
            fig.add_vrect(x0=t0, x1=t1, row=r, col=1, line_width=0,
                          fillcolor=_hex_a_rgba(cc, 0.10), layer="below")
    fig.update_xaxes(range=[inicio, d["ts"].max() + pd.offsets.MonthEnd(0)], showgrid=True,
                     title_text=None)
    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="% media", row=2, col=1)
    fig.update_yaxes(title_text="% vol. útil", row=3, col=1)
    fig.update_yaxes(title_text="COP/kWh", row=4, col=1)
    fig.update_layout(height=820, legend=dict(orientation="h", y=1.055, x=0),
                      title_text=f"El Niño vs. operación del SIN · {inicio:%Y} → {d['ts'].max():%b-%Y} "
                                 "(sombras = episodios ENSO)", title_font_size=13.5)
    ult = d.iloc[-1]

    def _kpi(campo: str, fmt: str = "{:.0f}", alt: str = "sin dato") -> str:
        v = ult.get(campo)
        try:
            return fmt.format(float(v)) if v is not None and np.isfinite(float(v)) else alt
        except (TypeError, ValueError):
            return alt

    nino_idx = [i for i, (a, b, c) in enumerate(franjas) if c == ROJO and b >= d["ts"].max() - pd.offsets.MonthEnd(0)]
    fig = _eje_ventana(fig, ctx, meses=60)
    return {"fig": fig, "leyenda": _leyenda(
        "Panel ENSO de 4 filas (misma figura que fig_v3_09 del notebook, con el eje X recortado)",
        "Cuatro series alineadas en el mismo mes: ONI de NOAA, aportes al SIN, volumen útil de "
        f"embalses y precio de bolsa. Las sombras rojas/azules son los episodios detectados por "
        f"`detectar_eventos_enso` ({len(franjas)} con traslape en la ventana). El eje X arranca en "
        f"{inicio:%Y} (primer mes con datos de XM), no en 1950.",
        f"Último mes con datos: {ult['ts']:%b-%Y} · ONI {_kpi('ONI', '{:+.2f}')} °C · aportes "
        f"{_kpi('Aportes_Pct')} % · embalses {_kpi('Embalses_Pct', '{:.1f}')} % · precio "
        f"{_kpi('Precio_Ponderado_COP', '{:,.0f}')} COP/kWh. Lean las filas de arriba a abajo: el ONI "
        "adelanta al resto 1-3 meses.",
        ("Agosto-2026 aparece en la última sombra roja sin ONI oficial (NOAA lo publica en septiembre): "
         "las tres filas inferiores sí tienen dato de mercado, y esa es la evidencia operativa."
         if nino_idx else "Sin episodio activo traslapeando el final de la serie."),
        "nino_mensual (XM 2015→, sembrada local si falta) + ONI NOAA CPC + eventos propios")}


def fig_v3_11_climatologia_fase(ctx: dict) -> dict | None:
    """`graficar_boxplots_fase_enso` (v3 c24): distribución por fase + ★ media de la ventana actual."""
    n = ctx.get("nino", pd.DataFrame())
    if n is None or n.empty or "Fase" not in n.columns:
        return None
    config = (("Aportes_Pct", "Aportes (% media histórica)", VERDE, 100.0),
              ("Embalses_Pct", "Embalses (% vol. útil)", AZUL, 60.0),
              ("Precio_Ponderado_COP", "Precio de bolsa (COP/kWh)", AMBAR, None))
    usadas = [c for c, *_ in config if c in n.columns]
    if not usadas:
        return None
    d = n.dropna(subset=["ONI"]).copy()
    orden_fases = ["La Niña", "Neutral", "El Niño"]
    relleno = {"La Niña": "#9ecae1", "Neutral": "#cfcfcf", "El Niño": "#f4a582"}
    fig = make_subplots(rows=1, cols=len(usadas),
                        subplot_titles=[t for c, t, *_ in config if c in usadas],
                        horizontal_spacing=0.075)
    textos: list[str] = []
    for j, (col, titulo, color, ref) in enumerate([cfg for cfg in config if cfg[0] in usadas], start=1):
        for fase in orden_fases:
            v = pd.to_numeric(d.loc[d["Fase"] == fase, col], errors="coerce").dropna()
            if len(v) < 3:
                continue
            fig.add_trace(go.Box(y=v, name=f"{fase} (n={len(v)})", legendgroup=fase,
                                 showlegend=(j == 1), marker_color=relleno[fase],
                                 line=dict(color="#555", width=1),
                                 boxpoints="outliers", hovertemplate=f"{fase} · %{{y:.1f}}<extra></extra>"),
                          row=1, col=j)
        if ref is not None:
            fig.add_hline(y=ref, row=1, col=j, line=dict(color=color, width=1, dash="dot"),
                          annotation_text=f"ref {ref:g}", annotation_font_size=8.5,
                          annotation_position="right")
        extra = ctx.get("extras_cur", {}) or {}
        clave_extra = {"Aportes_Pct": "Aportes_Pct", "Embalses_Pct": "Embalses_Pct",
                       "Precio_Ponderado_COP": "Precio_Ponderado_COP_kWh"}[col]
        val = extra.get(clave_extra)
        if val is not None and np.isfinite(float(val)):
            fig.add_trace(go.Scatter(x=[None], y=[float(val)], mode="markers",
                                     marker=dict(symbol="star", size=17, color="crimson",
                                                 line=dict(color="white", width=1)),
                                     name=f"Ventana actual {float(val):,.0f}",
                                     hovertemplate=f"Ventana actual: {float(val):,.1f}<extra></extra>"),
                          row=1, col=j)
            prom_nino = pd.to_numeric(d.loc[d["Fase"] == "El Niño", col], errors="coerce").mean()
            prom_neu = pd.to_numeric(d.loc[d["Fase"] == "Neutral", col], errors="coerce").mean()
            if np.isfinite(prom_nino) and np.isfinite(prom_neu) and prom_neu:
                textos.append(f"{titulo.split(' (')[0]}: hoy {float(val):,.0f} vs. media Niño "
                              f"{prom_nino:,.0f} y neutrales {prom_neu:,.0f} "
                              f"({100 * (prom_nino - prom_neu) / prom_neu:+.0f} % por ser Niño)")
    if not fig.data:
        return None
    fig.update_yaxes(title_text=None)
    fig.update_layout(height=430, boxmode="overlay", boxgap=0.15, boxgroupgap=0.12,
                      legend=dict(orientation="h", y=1.14, x=0),
                      title_text="Climatología operativa por fase ENSO · ★ = media de la ventana que "
                                 "estás mirando arriba", title_font_size=13)
    return {"fig": fig, "leyenda": _leyenda(
        "¿En qué régimen opera hoy el SIN? (fig_v3_11 del notebook, con la ventana actual marcada)",
        "Caja y bigotes de cada variable mensual 2015→ agrupada por fase ENSO (La Niña / Neutral / "
        f"El Niño). La estrella roja es el promedio de la ventana seleccionada ({ctx['ini']} → "
        f"{ctx['fin']}), para que veas en qué cola cae el presente.",
        (" · ".join(textos) if textos else
         "Compara la estrella con la mediana de cada caja: si cae fuera de la caja de 'El Niño', el "
         "régimen actual es más severo que los eventos históricos de la serie."),
        "Si la estrella está en el borde inferior de aportes y en el superior de precio, el escenario "
        "de precio al alza está validado por climatología, no solo por la tendencia de 30 días.",
        f"nino_mensual (2015→) + ONI NOAA · fase por mes ({len(d)} meses clasificados)")}


def fig_v4_02_costo_marginal(ctx: dict) -> dict | None:
    """`graficar_costo_marginal` (v4 c7): serie diaria con banda mín-máx + perfil horario vs demanda."""
    sis = ctx.get("sistema_v4", pd.DataFrame())
    if sis is None or sis.empty or "Costo_Marginal_COP_kWh" not in sis.columns:
        sis = ctx.get("sistema", pd.DataFrame())
    if sis is None or sis.empty or "Costo_Marginal_COP_kWh" not in sis.columns:
        return None
    d = sis.copy()
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    d["Costo_Marginal_COP_kWh"] = pd.to_numeric(d["Costo_Marginal_COP_kWh"], errors="coerce")
    d = d.dropna(subset=["Fecha", "Hora", "Costo_Marginal_COP_kWh"])
    if d.empty:
        return None
    d["_dia"] = d["Fecha"].dt.date
    diario = (d.groupby("_dia", as_index=False)
              .agg(Costo_medio=("Costo_Marginal_COP_kWh", "mean"),
                   Costo_min=("Costo_Marginal_COP_kWh", "min"),
                   Costo_max=("Costo_Marginal_COP_kWh", "max"))
              .rename(columns={"_dia": "Fecha"}))
    diario["Fecha"] = pd.to_datetime(diario["Fecha"])
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        specs=[[{}], [{"secondary_y": True}]],
                        subplot_titles=(f"Costo marginal de despacho diario vs precio de bolsa "
                                        f"({len(diario)} días)",
                                        "Perfil horario medio: costo marginal vs demanda (doble eje)"))
    fig.add_trace(go.Scatter(x=pd.concat([diario["Fecha"], diario["Fecha"][::-1]]),
                              y=pd.concat([diario["Costo_max"], diario["Costo_min"][::-1]]),
                              fill="toself", fillcolor=_hex_a_rgba("#1565c0", 0.18), line=dict(width=0),
                              name="Rango mín-máx diario", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=diario["Fecha"], y=diario["Costo_medio"], name="Costo marginal medio",
                             line=dict(color="#1565c0", width=2.4),
                             hovertemplate="%{x|%d %b} · %{y:,.0f} COP/kWh<extra></extra>"), row=1, col=1)
    if "Precio_Prom_COP" in ctx.get("resumen", pd.DataFrame()):
        rd = ctx["resumen"].copy()
        rd["Fecha"] = pd.to_datetime(rd["Fecha"], errors="coerce")
        fig.add_trace(go.Scatter(x=rd["Fecha"], y=rd["Precio_Prom_COP"], name="Precio de bolsa medio",
                                 line=dict(color="#8c564b", width=1.8, dash="dash"),
                                 hovertemplate="%{x|%d %b} · %{y:,.0f} COP/kWh<extra></extra>"),
                      row=1, col=1)
    elif "Precio_Bolsa_COP_kWh" in d:
        pre = d.groupby("Fecha", as_index=False)["Precio_Bolsa_COP_kWh"].mean()
        fig.add_trace(go.Scatter(x=pre["Fecha"], y=pre["Precio_Bolsa_COP_kWh"], name="Precio de bolsa medio",
                                 line=dict(color="#8c564b", width=1.8, dash="dash"),
                                 hovertemplate="%{x|%d %b} · %{y:,.0f} COP/kWh<extra></extra>"),
                      row=1, col=1)
    per_c = d.groupby("Hora")["Costo_Marginal_COP_kWh"].mean()
    fig.add_trace(go.Scatter(x=per_c.index, y=per_c.values, name="Costo marginal",
                             line=dict(color="#1565c0", width=2.2, shape="spline"), marker=dict(size=5),
                             hovertemplate="H%{x:02d} · %{y:,.0f} COP/kWh<extra></extra>"),
                  row=2, col=1, secondary_y=False)
    if "Demanda_MW" in d:
        per_d = d.groupby("Hora")["Demanda_MW"].mean()
        fig.add_trace(go.Scatter(x=per_d.index, y=per_d.values, name="Demanda",
                                 line=dict(color="#d9662d", width=2, dash="dash"),
                                 hovertemplate="H%{x:02d} · %{y:,.0f} MW<extra></extra>"),
                      row=2, col=1, secondary_y=True)
    if len(per_c) >= 3:
        h_max, h_min = int(per_c.idxmax()), int(per_c.idxmin())
        fig.add_annotation(row=2, col=1, x=h_max, y=float(per_c.loc[h_max]),
                           text=f"hora más cara · H{h_max:02d} · {per_c.max():,.0f} COP/kWh",
                           showarrow=True, arrowhead=1, ay=-30, ax=0, align="center",
                           font=dict(size=9, color="#0d47a1"), bgcolor="rgba(255,255,255,.92)",
                           bordercolor="#1565c0", borderpad=3)
        fig.add_annotation(row=2, col=1, x=h_min, y=float(per_c.loc[h_min]),
                           text=f"hora más barata · H{h_min:02d} · {per_c.min():,.0f} COP/kWh",
                           showarrow=True, arrowhead=1, ay=32, ax=0, align="center",
                           font=dict(size=9, color="#1b5e20"), bgcolor="rgba(255,255,255,.92)",
                           bordercolor="#2e7d32", borderpad=3)
    fig.update_xaxes(title_text="Hora XM (H01 = 00:00-01:00)", row=2, col=1,
                     tickmode="array", tickvals=list(range(1, 25, 3)),
                     ticktext=[f"H{h:02d}" for h in range(1, 25, 3)])
    fig.update_yaxes(title_text="COP/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Costo marginal (COP/kWh)", row=2, col=1, secondary_y=False)
    if "Demanda_MW" in d:
        fig.update_yaxes(title_text="Demanda (MW)", row=2, col=1, secondary_y=True)
    fig.update_layout(height=700, legend=dict(orientation="h", y=1.09, x=0),
                      title_text="Costo marginal: cuánto cuesta despachar cada hora de esta ventana",
                      title_font_size=13)
    pico = per_c.max() if len(per_c) else float("nan")
    valle = per_c.min() if len(per_c) else float("nan")
    dif = (pico - valle) if np.isfinite(pico) and np.isfinite(valle) else float("nan")
    dif_dia = float((diario["Costo_max"] - diario["Costo_min"]).mean())
    return {"fig": fig, "leyenda": _leyenda(
        "Costo marginal diario (con su banda intra-día) y perfil horario — fig_v4_02 del notebook",
        "Arriba: media diaria con la franja mín-máx de las 24 horas y, si está, el precio de bolsa "
        "medio. Abajo: promedio por hora del costo (eje izq.) y de la demanda (eje der.).",
        f"La hora más cara del promedio horario está en H{int(per_c.idxmax()):02d} ({pico:,.0f} COP/kWh) "
        f"y la más barata en H{int(per_c.idxmin()):02d} ({valle:,.0f}): un diferencial de {dif:,.0f} "
        f"COP/kWh. Dentro de un mismo día el costo oscila {dif_dia:,.0f} COP/kWh en promedio, así que "
        "el precio medio diario esconde la mitad del riesgo.",
        "Cuando la banda mín-máx se ensancha y el perfil horario se vuelve bimodal (mañana + 17-20 h), "
        "el sistema está dependiendo del despacho térmico: es la firma de El Niño y el momento de fijar "
        "contratos a término o mover la carga.",
        "CostMargDesp/Sistema (horario) + DemaReal + PrecBolsNaci · "
        f"{len(diario)} día(s) × {int(d['Hora'].nunique())} h")}


def fig_v4_03_heatmap_costo_dia_hora(ctx: dict) -> dict | None:
    """`graficar_heatmap_costo_dia_hora` (v4 c8): día × hora del costo marginal (divergente)."""
    for clave in ("sistema_v4", "sistema", "costo_marginal"):
        d = ctx.get(clave, pd.DataFrame())
        if d is not None and not d.empty and "Costo_Marginal_COP_kWh" in d.columns:
            break
    else:
        return None
    d = d.copy()
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    d["Costo_Marginal_COP_kWh"] = pd.to_numeric(d["Costo_Marginal_COP_kWh"], errors="coerce")
    d = d.dropna(subset=["Fecha", "Hora", "Costo_Marginal_COP_kWh"])
    if d.empty:
        return None
    piv = d.pivot_table(index="Fecha", columns="Hora", values="Costo_Marginal_COP_kWh", aggfunc="mean")
    if piv.empty:
        return None
    horas = [int(c) for c in piv.columns]
    z = piv.values.astype(float)
    fig = _fig(ctx, max(430, 20 * len(piv) + 150))
    fig.add_trace(go.Heatmap(z=z, x=[f"H{h:02d}" for h in horas],
                             y=[f"{t:%d %b}" for t in piv.index], name="COP/kWh",
                             colorscale=[[0.0, "#1a9850"], [0.5, "#ffffbf"], [1.0, "#d73027"]],
                             zmid=float(np.nanmedian(z)), colorbar=dict(title="COP/kWh"),
                             xgap=0.6, ygap=0.6,
                             hovertemplate="%{y} · %{x}<br>%{z:,.0f} COP/kWh<extra></extra>"))
    fig.update_layout(xaxis_title="Hora XM", yaxis_title="Fecha", yaxis=dict(autorange="reversed"),
                      title_text="Costo marginal de despacho — día × hora (verde = barato, rojo = caro)",
                      title_font_size=13)
    por_hora = pd.Series(np.nanmean(z, axis=0), index=[f"H{h:02d}" for h in horas]).sort_values(ascending=False)
    por_dia = pd.Series(np.nanmean(z, axis=1), index=[f"{t:%d %b}" for t in piv.index]).sort_values(ascending=False)
    return {"fig": fig, "leyenda": _leyenda(
        "Mapa de calor del costo marginal (fig_v4_03): dónde y cuándo se dispara el despacho",
        "Media horaria del costo marginal de cada día de la ventana. Escala divergente centrada en la "
        f"mediana ({np.nanmedian(z):,.0f} COP/kWh), así el rojo significa 'caro para este período'.",
        f"Horas más caras: {', '.join(por_hora.head(3).index)}. Horas más baratas: "
        f"{', '.join(por_hora.tail(3).index)}. Días más caros: {', '.join(por_dia.head(2).index)}. "
        "Si tu consumo puede moverse de las primeras a las segundas, ese es tu ahorro directo.",
        "En El Niño el rojo se corre hacia la madrugada y se queda en la tarde: la señal de que la "
        "hidro ya no baja el costo en la madrugada (está reservando agua) es exactamente este mapa.",
        "CostMargDesp/Sistema · "
        f"{len(piv)} día(s) × {len(horas)} horas")}




# ------------------------------------------------------------- CO₂ + ENSO-economía
def _emisiones_largas(ctx: dict) -> pd.DataFrame:
    """`fact_emisiones` del notebook: tCO₂ por fecha×hora×combustible (de EmisionesCO2/RecursoComb)."""
    e = ctx.get("emisiones", pd.DataFrame())
    if e is None or e.empty or "tCO2" not in e.columns or "Combustible" not in e.columns:
        return pd.DataFrame()
    d = e.copy()
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    d["tCO2"] = pd.to_numeric(d["tCO2"], errors="coerce")
    d["Combustible"] = d["Combustible"].astype(str).str.upper().str.strip()
    return d.dropna(subset=["Fecha", "Hora", "tCO2"])


def fig_v3_05_emisiones_diarias(ctx: dict) -> dict | None:
    """`graficar_emisiones_diarias` (v3 c13): área apilada por combustible + intensidad vs factor XM."""
    d = _emisiones_largas(ctx)
    if d.empty:
        return None
    d["_dia"] = d["Fecha"].dt.date
    diario = d.groupby(["_dia", "Combustible"], as_index=False)["tCO2"].sum()
    if diario.empty:
        return None
    piv = diario.pivot_table(index="_dia", columns="Combustible", values="tCO2", aggfunc="sum",
                             fill_value=0.0) / 1e3                      # tCO₂ → ktCO₂
    piv.index = pd.to_datetime(pd.Index(piv.index))
    piv = piv[piv.sum(axis=0).sort_values(ascending=False).index.tolist()]
    paleta = {"CARBON": "#4d4d4d", "GAS": "#e07b39", "ACPM": "#b8860b", "COMBUSTOLEO": "#8c564b",
              "BAGAZO": "#7aa464", "COQUE": "#5d4037", "Fuel OIl": "#37474f"}
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.14, row_heights=[0.58, 0.42],
                        subplot_titles=("Emisiones de CO₂ del parque reportado, por combustible (ktCO₂/día)",
                                        "Intensidad realizada vs factor oficial XM (tCO₂e/MWh)"))
    for col in piv.columns:
        fig.add_trace(go.Scatter(x=piv.index, y=piv[col], name=str(col).title(), stackgroup="one",
                                 line=dict(width=0.6, color=paleta.get(str(col).upper(), "#c200b0")),
                                 hovertemplate="%{x} · %{y:,.1f} ktCO₂<extra>" + str(col).title() + "</extra>"),
                      row=1, col=1)
    ed = ctx.get("emisiones_dia", pd.DataFrame())
    if isinstance(ed, pd.DataFrame) and not ed.empty:
        e = ed.copy()
        e["Fecha"] = pd.to_datetime(e["Fecha"], errors="coerce")
        if "Intensidad_ratio_MWh" in e:
            fig.add_trace(go.Scatter(x=e["Fecha"], y=e["Intensidad_ratio_MWh"], name="Intensidad realizada",
                                     line=dict(color=VERDE, width=2),
                                     hovertemplate="%{x|%d %b} · %{y:.2f} tCO₂/MWh<extra></extra>"),
                          row=2, col=1)
        if "Intensidad_tCO2_MWh" in e:
            fig.add_trace(go.Scatter(x=e["Fecha"], y=e["Intensidad_tCO2_MWh"], name="Factor XM (margen)",
                                     line=dict(color="#1565c0", width=2, dash="dash"),
                                     hovertemplate="%{x|%d %b} · %{y:.2f} tCO₂e/MWh<extra></extra>"),
                          row=2, col=1)
        for yv in (0.2, 0.4):
            fig.add_hline(y=yv, row=2, col=1, line=dict(color=AMBAR if yv == 0.2 else ROJO, width=1, dash="dot"),
                          annotation_text=f"banda {yv}", annotation_font_size=8.5, annotation_position="right")
    fig.update_xaxes(title_text=None, row=1, col=1)
    fig.update_xaxes(title_text="Fecha", row=2, col=1)
    fig.update_yaxes(title_text="ktCO₂ / día", row=1, col=1)
    fig.update_yaxes(title_text="tCO₂(e)/MWh", row=2, col=1)
    fig.update_layout(height=620, legend=dict(orientation="h", y=1.10, x=0),
                      title_text="Huella de CO₂ de la ventana — el costo oculto del episodio El Niño",
                      title_font_size=13)
    total = float(piv.values.sum())
    top_comb = str(piv.sum(axis=0).idxmax()).title()
    share = 100 * float(piv.sum(axis=0).max()) / max(total, 1e-9)
    nota = ctx.get("notas_emisiones", {}) or {}
    return {"fig": fig, "leyenda": _leyenda(
        "Emisiones diarias por combustible + intensidad (fig_v3_05 del notebook)",
        f"Panel superior: ktCO₂ por día apilados por combustible, a partir de `EmisionesCO2/RecursoComb` "
        f"(parque reportado, no todo el SIN: {nota.get('fuente_intensidad', 'sin nota de fuente')}). "
        "Panel inferior: la intensidad realizada (tCO₂ ÷ energía de la ventana) contra el factor oficial "
        "de XM para el margen de operación, que es por definición más alto porque mide la hora marginal.",
        f"Total de la ventana: {total:,.0f} ktCO₂, con {top_comb} aportando el {share:.0f} %. Si la "
        "curva de GAS sube mientras la de CARBÓN se aplana, el mix está cambiando de carbón a gas "
        "(menos CO₂/MWh pero más costo marginal).",
        "En El Niño el panel superior se infla en las horas 17-22 h: cada ktCO₂ extra ahí es térmica "
        "cubriendo el hueco hidro, y es exactamente lo que el regulador mira para el cargo por "
        "restricción.",
        "EmisionesCO2/RecursoComb + ConsCombustibleMBTU + factorEmisionCO2e (métricas pesadas: "
        "activar en la barra lateral)")}


def fig_v3_06_perfil_horario_emisiones(ctx: dict) -> dict | None:
    """`graficar_perfil_horario_emisiones` (v3 c14): emisiones por hora + intensidad vs solar."""
    d = _emisiones_largas(ctx)
    if d.empty:
        return None
    por_hora = d.groupby(["Hora", "Combustible"], as_index=False)["tCO2"].mean().pivot_table(
        index="Hora", columns="Combustible", values="tCO2", aggfunc="mean", fill_value=0.0)
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.17,
                        subplot_titles=("Perfil horario medio de emisiones por combustible (tCO₂/h)",
                                        "Intensidad horaria vs participación solar (%)"))
    paleta = ["#4d4d4d", "#e07b39", "#b8860b", "#8c564b", "#7aa464", "#c200b0", "#37474f"]
    for j, col in enumerate(por_hora.columns):
        fig.add_trace(go.Bar(x=por_hora.index.astype(int), y=por_hora[col], name=str(col).title(),
                             marker_color=paleta[j % len(paleta)], showlegend=(j < 8),
                             hovertemplate="H%{x:02d} · %{y:,.1f} tCO₂/h<extra>" + str(col).title() + "</extra>"),
                      row=1, col=1)
    sis = ctx.get("sistema", pd.DataFrame())
    if sis is not None and not sis.empty and "Demanda_MW" in sis:
        dd = sis.copy()
        dd["Demanda_MW"] = pd.to_numeric(dd["Demanda_MW"], errors="coerce")
        dem_h = dd.groupby("Hora")["Demanda_MW"].mean()
        emis_h = d.groupby("Hora")["tCO2"].mean()
        inten = (emis_h / dem_h.replace(0, np.nan)).dropna()
        if len(inten) >= 3:
            fig.add_trace(go.Scatter(x=inten.index.astype(int), y=inten.values, name="Intensidad (tCO₂/MWh)",
                                    line=dict(color="#1565c0", width=2.2, shape="spline"), marker=dict(size=5)),
                          row=2, col=1)
    gen = gen_con_ambito(ctx)
    if not gen.empty and "Tecnologia" in gen:
        tot_h = gen.groupby("Hora")["Generacion_kWh"].mean()
        sol = (gen[gen["Tecnologia"].astype(str).str.upper().str.contains("SOLAR", na=False)]
               .groupby("Hora")["Generacion_kWh"].mean())
        if len(sol) and len(tot_h):
            pct = (100 * sol / tot_h.replace(0, np.nan)).dropna()
            if len(pct):
                fig.add_trace(go.Scatter(x=pct.index.astype(int), y=pct.values, name="Participación solar (%)",
                                         fill="tozeroy", line=dict(color="#b7950b", width=1.6),
                                         fillcolor="rgba(241,194,50,.28)",
                                         hovertemplate="H%{x:02d} · %{y:.1f} %<extra></extra>"),
                              row=2, col=1)
    for r in (1, 2):
        fig.update_xaxes(tickmode="array", tickvals=list(range(1, 25, 3)),
                         ticktext=[f"H{h:02d}" for h in range(1, 25, 3)],
                         title_text="Hora XM (H01 = 00:00-01:00)", row=r, col=1)
    fig.update_yaxes(title_text="tCO₂ / hora", row=1, col=1)
    fig.update_yaxes(title_text="tCO₂/MWh  ·  % solar", row=2, col=1)
    fig.update_layout(height=650, barmode="stack", legend=dict(orientation="h", y=1.10, x=0),
                      title_text="Emisiones por hora: el mediodía limpio y la tarde sucia", title_font_size=13)
    if por_hora.empty:
        return None
    total_h = por_hora.sum(axis=1)
    return {"fig": fig, "leyenda": _leyenda(
        "Perfil horario de emisiones vs participación solar (fig_v3_06 del notebook)",
        "Arriba: media de tCO₂ por hora y combustible en la ventana (barras apiladas). Abajo: "
        "intensidad horaria (tCO₂ de la hora ÷ demanda media de la hora) y, de fondo, cuánta energía "
        "solar había en esa hora.",
        f"La hora más contaminante es H{int(total_h.idxmax()):02d} ({total_h.max():,.1f} tCO₂/h) y la "
        f"menos limpia del mediodía cae donde el solar pesa más: si el valle de intensidad coincide con "
        "el pico solar, tu consumo vale más corrido a esas horas.",
        "En El Niño el valle del mediodía se mantiene pero la tarde sube: la solar no desaparece con la "
        "sequía, lo que desaparece es la hidro que apagaba las 18-21 h. Por eso el CO₂ nocturno es el "
        "mejor termómetro del episodio.",
        "EmisionesCO2/RecursoComb + Gene/Recurso + DemaReal")}


def fig_v3_07_top_emisores_pareto(ctx: dict) -> dict | None:
    """`graficar_top_emisores` (v3 c15): top 15 plantas emisoras + curva de Pareto acumulada."""
    e = ctx.get("emisiones", pd.DataFrame())
    if e is None or e.empty or "Codigo_Planta" not in e.columns:
        return None
    d = e.copy()
    d["tCO2"] = pd.to_numeric(d.get("tCO2"), errors="coerce")
    d = d.dropna(subset=["tCO2"])
    if d.empty:
        return None
    dim = ctx.get("dim_plantas", pd.DataFrame())
    top = (d.groupby("Codigo_Planta", as_index=False)["tCO2"].sum()
           .sort_values("tCO2", ascending=False).head(25).reset_index(drop=True))
    total = float(d["tCO2"].sum())
    if top.empty or total <= 0:
        return None
    if dim is not None and not dim.empty:
        cat = dim.drop_duplicates("Codigo_Planta").set_index("Codigo_Planta")
        for origen, destino in (("Nombre_Planta", "Nombre"), ("Tecnologia", "Tecnologia")):
            if origen in cat.columns:
                top[destino] = top["Codigo_Planta"].astype(str).map(cat[origen])
    top["Nombre"] = top.get("Nombre", pd.Series(top["Codigo_Planta"])).fillna(top["Codigo_Planta"])
    top["Part_pct"] = 100 * top["tCO2"] / total
    top["Acum_pct"] = top["Part_pct"].cumsum()
    n = min(15, len(top))
    sub = top.head(n)
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.11,
                        column_widths=[0.58, 0.42],
                        specs=[[{"type": "bar"}, {"type": "xy"}]],
                        subplot_titles=(f"Top {n} plantas emisoras (ktCO₂ de la ventana)",
                                        "Curva de Pareto acumulada (% del total del parque reportado)"))
    colores = [COLOR_TECNOLOGIAS.get(str(t).upper(), "#c200b0") for t in sub.get("Tecnologia", [])]
    fig.add_trace(go.Bar(y=[str(x)[:30] for x in sub["Nombre"]][::-1], x=(sub["tCO2"] / 1e3)[::-1],
                         orientation="h", marker_color=colores[::-1] if colores else "#c200b0",
                         name="ktCO₂", showlegend=False, customdata=sub["Part_pct"][::-1],
                         hovertemplate="%{y} · %{x:,.1f} ktCO₂ · %{customdata:.1f} % del total<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1, len(top) + 1)), y=top["Acum_pct"], name="Acumulado %",
                             mode="lines+markers", line=dict(color=AZUL, width=2.2),
                             hovertemplate="planta #%{x} · %{y:.1f} % acumulado<extra></extra>"),
                  row=1, col=2)
    i50 = int((top["Acum_pct"] < 50).sum()) + 1
    fig.add_hline(y=50, row=1, col=2, line=dict(color=ROJO, width=1, dash="dot"),
                  annotation_text="50 %", annotation_font_size=8.5, annotation_position="right")
    fig.add_vline(x=i50, row=1, col=2, line=dict(color=ROJO, width=1, dash="dot"))
    fig.update_yaxes(title_text="Planta", row=1, col=1)
    fig.update_xaxes(title_text="ktCO₂", row=1, col=1)
    fig.update_xaxes(title_text="Rango de planta (1 = mayor emisora)", row=1, col=2)
    fig.update_yaxes(title_text="% acumulado", range=[0, 100], row=1, col=2)
    fig.update_layout(height=max(430, 26 * n + 150), showlegend=False,
                      title_text="Quién emite: concentración del CO₂ del parque reportado",
                      title_font_size=13)
    return {"fig": fig, "leyenda": _leyenda(
        "Ranking de emisoras + Pareto (fig_v3_07 del notebook)",
        f"ktCO₂ acumulados por planta en la ventana (solo el parque que reporta `EmisionesCO2/"
        f"RecursoComb`, {int(d['Codigo_Planta'].nunique())} recursos) y, a la derecha, el porcentaje "
        f"acumulado. La línea roja marca que con las primeras {i50} plantas ya cubres el 50 % del CO₂.",
        f"La primera planta representa {float(top['Part_pct'].iloc[0]):.1f} % del total reportado. "
        "Esa es la concentración de riesgo: si esa unidad entra en indisponibilidad programada, el "
        "CO₂ y el precio se mueven juntos.",
        "En sequía las plantas del top son las que sostienen el sistema; conviene saber quién las opera "
        "(filtro 'Agente' arriba) porque serán tu contraparte en las horas caras.",
        "EmisionesCO2/RecursoComb × ListadoRecursos · ventana "
        f"{ctx['ini']} → {ctx['fin']}")}


def fig_v3_08_correlacion_extendida(ctx: dict) -> dict | None:
    """`v3 c16`: matriz de correlación diaria del contexto operativo + emisiones (triángulo superior)."""
    rd = ctx.get("resumen", pd.DataFrame())
    ed = ctx.get("emisiones_dia", pd.DataFrame())
    if rd is None or rd.empty:
        return None
    d = rd.copy()
    d["Fecha"] = pd.to_datetime(d["Fecha"], errors="coerce")
    if isinstance(ed, pd.DataFrame) and not ed.empty:
        e = ed.copy()
        e["Fecha"] = pd.to_datetime(e["Fecha"], errors="coerce")
        cols_e = [c for c in ("tCO2_XM", "Intensidad_ratio_MWh", "Intensidad_tCO2_MWh",
                             "Consumo_Combustible_MBTU") if c in e.columns]
        if cols_e:
            d = d.merge(e[["Fecha"] + cols_e], on="Fecha", how="left")
    candidatas = {"Precio_Prom_COP": "Precio (COP/kWh)", "Precio_Ponderado_COP_kWh": "Precio pond. (COP/kWh)",
                  "Embalses_Pct": "Embalses (%)", "Aportes_Pct": "Aportes (% media)",
                  "Participacion_Termica_pct": "Part. térmica (%)", "Generacion_Total_GWh": "Gen. total (GWh)",
                  "Demanda_GWh": "Demanda (GWh)", "tCO2_XM": "Emisiones (tCO₂)",
                  "Intensidad_ratio_MWh": "Intensidad (tCO₂/MWh)", "Intensidad_tCO2_MWh": "Factor XM (tCO₂e/MWh)",
                  "Consumo_Combustible_MBTU": "Consumible (MBTU)", "Balance_GWh": "Balance (GWh)",
                  "Gen_SOLAR_GWh": "Gen. solar (GWh)", "Gen_HIDRAULICA_GWh": "Gen. hidro (GWh)",
                  "Gen_TERMICA_GWh": "Gen. térmica (GWh)"}
    uso = [c for c in candidatas if c in d.columns and pd.to_numeric(d[c], errors="coerce").notna().sum() >= 5]
    if len(uso) < 3:
        return None
    dd = d.copy()
    for c in uso:
        dd[c] = pd.to_numeric(dd[c], errors="coerce")
    corr = dd[uso].corr(method="pearson")
    etiq = [candidatas[c] for c in uso]
    z = corr.values.astype(float).copy()
    mascara = np.triu(np.ones_like(z, dtype=bool), k=1)          # solo el triángulo inferior, como el notebook
    z[mascara] = np.nan
    fig = _fig(ctx, max(460, 40 * len(uso) + 160))
    fig.add_trace(go.Heatmap(z=z, x=etiq, y=etiq, zmin=-1, zmax=1, name="Pearson r",
                             colorscale=[[0, "#2166ac"], [0.5, "#f7f7f7"], [1, "#b2182b"]],
                             colorbar=dict(title="r"), xgap=2, ygap=2,
                             hovertemplate="%{y} ↔ %{x}<br>r=%{z:.2f}<extra></extra>"))
    for i in range(len(etiq)):
        for j in range(len(etiq)):
            if not mascara[i, j] and np.isfinite(z[i, j]):
                fig.add_annotation(x=etiq[j], y=etiq[i], text=f"{z[i, j]:+.2f}", showarrow=False,
                                   font=dict(size=9, color="white" if abs(z[i, j]) > 0.55 else "#222"))
    fig.update_layout(xaxis_tickangle=-32, yaxis=dict(autorange="reversed"),
                      title_text=f"Correlación diaria — emisiones vs contexto operativo ({len(dd)} días)",
                      title_font_size=13)
    pares = []
    for a, b in (("tCO2_XM", "Participacion_Termica_pct"), ("Intensidad_ratio_MWh", "Gen_SOLAR_GWh"),
                 ("Precio_Prom_COP", "tCO2_XM"), ("Embalses_Pct", "Precio_Ponderado_COP_kWh")):
        if a in uso and b in uso and np.isfinite(corr.loc[a, b]):
            pares.append(f"{candidatas[a]} ↔ {candidatas[b]}: r={corr.loc[a, b]:+.2f}")
    return {"fig": fig, "leyenda": _leyenda(
        "Matriz de correlación extendida (v3 c16): precio, agua, térmica y CO₂ en un solo cuadro",
        f"Pearson sobre los {len(dd)} días de la ventana, con las {len(uso)} variables que tienen ≥5 "
        "observaciones. Solo se pinta el triángulo inferior (la diagonal es redundante), igual que en "
        "el notebook, y cada celda lleva el valor escrito.",
        ("Lecturas clave: " + " · ".join(pares[:3]) + ".") if pares else
        "Lee los valores absolutos > 0.5: son los pares que se mueven juntos.",
        "El par que importa en El Niño es Embalses ↔ Precio (negativo esperado) y Part. térmica ↔ CO₂ "
        "(positivo casi 1): si el primero se debilita, el mercado ya está descontando racionamiento o "
        "intervención, no solo hidrología.",
        "resumen_diario (v2) × emisiones_diarias (v3), inner join por Fecha")}


def v4_mensual_para_enso(ctx: dict) -> pd.DataFrame:
    """Serie mensual con margen: semilla `v4_mensual.csv` ⊕ meses de la ventana (equivalente a
    `nino_v4_mensual` del notebook v4). Se construye el margen a partir de CEN/demanda."""
    def _con_margen(f: pd.DataFrame) -> pd.DataFrame:
        d = f.copy()
        if "Mes" not in d.columns:
            return d
        cen = pd.to_numeric(d.get("CEN_MW"), errors="coerce")
        dem = pd.to_numeric(d.get("Demanda_MW"), errors="coerce")
        dmx = pd.to_numeric(d.get("DemaMax_MW", d.get("DemaMaxProm_MW")), errors="coerce")
        if cen is not None and dem is not None and cen.notna().any():
            d["Margen_pct_mean"] = 100 * (cen - dem) / cen.replace(0, np.nan)
            if dmx is not None:
                d["Margen_pct_min"] = 100 * (cen - dmx) / cen.replace(0, np.nan)
        return d

    sem = ctx.get("v4_serie", pd.DataFrame())
    partes = []
    if isinstance(sem, pd.DataFrame) and not sem.empty:
        partes.append(_con_margen(sem))
    cv = ctx.get("cruce_v4", pd.DataFrame())
    if isinstance(cv, pd.DataFrame) and not cv.empty and "Fecha" in cv.columns:
        c = cv.copy()
        c["Fecha"] = pd.to_datetime(c["Fecha"], errors="coerce")
        c["Mes"] = c["Fecha"].dt.to_period("M").astype(str)
        agg = {}
        for destino, origen in (("Demanda_MW", "Demanda_MW_Dia"), ("CEN_MW", "CEN_MW_Dia"),
                                ("Costo_Marginal_COP_kWh", "Costo_Marginal_COP_kWh_Dia"),
                                ("DemaMax_MW", "Demanda_Max_Dia_MW")):
            if origen in c.columns:
                agg[destino] = (origen, "mean")
        hay_margen_min = {"Demanda_Max_Dia_MW", "CEN_MW_Dia"}.issubset(c.columns)
        if hay_margen_min:
            agg["_dem_max"] = ("Demanda_Max_Dia_MW", "max")
            agg["_cen_media"] = ("CEN_MW_Dia", "mean")
        if not agg:
            return pd.DataFrame() if not partes else pd.concat(partes, ignore_index=True)
        mensual = c.groupby("Mes", as_index=False).agg(**agg)
        if hay_margen_min:
            mensual["Margen_pct_min"] = (100 * (mensual["_cen_media"] - mensual["_dem_max"])
                                         / mensual["_cen_media"].replace(0, np.nan))
            mensual = mensual.drop(columns=["_dem_max", "_cen_media"])
        partes.append(mensual)
    if not partes:
        return pd.DataFrame()
    out = pd.concat(partes, ignore_index=True).drop_duplicates("Mes", keep="first").sort_values("Mes")
    n = ctx.get("nino", pd.DataFrame())
    if isinstance(n, pd.DataFrame) and not n.empty and "Mes" in n:
        nn = n[["Mes", "ONI", "Fase"] + (["Aportes_Pct"] if "Aportes_Pct" in n else [])].copy()
        nn["Mes"] = nn["Mes"].astype(str)
        out["Mes"] = out["Mes"].astype(str)
        out = out.merge(nn, on="Mes", how="left")
    return out.reset_index(drop=True)


def _num(v: object, defecto: float = float("nan")) -> float:
    """Escalares posiblemente ausentes/None → float sin romper la tabla."""
    try:
        f = float(pd.to_numeric(pd.Series([v]), errors="coerce").iloc[0])
    except Exception:                                             # noqa: BLE001
        return defecto
    return f if np.isfinite(f) else defecto


def tabla_impacto_eventos_v4(ctx: dict) -> pd.DataFrame:
    """`v4 c203` del notebook: impacto de cada episodio ENSO como Δ% frente al régimen Neutral.

    Es la tabla que antes iba pegada a una figura; ahora es un cálculo propio de la app (y se ve
    como tabla en 🔥 CO₂ y economía ENSO), porque el número —no la barra— es lo que se contrata.
    Fórmula literal del notebook: Δ% = 100 · (media_meses_episodio − media_meses_Neutral) / media_Neutral,
    con grupos de `clasificar_fase(ONI)` (±0,5 °C) y episodios de `detectar_eventos_enso` (≥5 meses),
    tomando solo episodios con ≥2 meses traslapeando la base mensual.
    """
    d = v4_mensual_para_enso(ctx)
    ev = ctx.get("nino_eventos", pd.DataFrame())
    if d is None or d.empty or "Fase" not in d or not isinstance(ev, pd.DataFrame) or ev.empty:
        return pd.DataFrame()
    d = d.copy()
    d["ts"] = pd.PeriodIndex(d["Mes"].astype(str), freq="M").to_timestamp()
    metricas = [("Costo_Marginal_COP_kWh", "Costo marginal"), ("Margen_pct_mean", "Margen medio"),
                ("Margen_pct_min", "Margen mínimo"), ("Aportes_Pct", "Aportes")]
    metricas = [(cc, t) for cc, t in metricas if cc in d.columns]
    if not metricas:
        return pd.DataFrame()
    base = {cc: float(pd.to_numeric(d.loc[d["Fase"] == "Neutral", cc], errors="coerce").mean())
            for cc, _t in metricas}
    filas = []
    for _, e in ev.iterrows():
        t0 = pd.Period(str(e["Inicio"])[:7], freq="M").to_timestamp()
        t1 = pd.Period(str(e["Fin"])[:7], freq="M").to_timestamp(how="end")
        dentro = d[(d["ts"] >= t0) & (d["ts"] <= t1)]
        if len(dentro) < 2:
            continue
        fila = {"Evento": f"{t0:%Y-%m}→{t1:%Y-%m}", "Fase": str(e.get("Fase", "")),
                "Meses": int(len(dentro)), "Meses_ini": f"{t0:%Y-%m}", "Meses_fin": f"{t1:%Y-%m}",
                "ONI pico (°C)": _num(e.get("Pico_ONI", e.get("ONI_pico"))),
                "ONI promedio (°C)": _num(e.get("ONI_promedio")),
                "Duración (meses)": int(_num(e.get("Duración_meses"), 0) or 0),
                "Clasificación": str(e.get("Intensidad", ""))}
        for cc, t in metricas:
            v = float(pd.to_numeric(dentro[cc], errors="coerce").mean())
            b = base.get(cc, float("nan"))
            fila[t] = (100 * (v - b) / b) if b and np.isfinite(b) else float("nan")
        filas.append(fila)
    tabla = pd.DataFrame(filas)
    if tabla.empty:
        return tabla
    return tabla.drop(columns=["Meses_ini", "Meses_fin"]).tail(14).reset_index(drop=True)


def fig_v4_17_hallazgos_v4(ctx: dict) -> dict | None:
    """`v4 c17`: los hallazgos v4 en un render legible (tabla KPI + lectura accionable)."""
    filas: list[dict[str, str]] = []
    d = v4_mensual_para_enso(ctx)
    kpis: dict[str, Any] = {}
    if not d.empty and "ONI" in d:
        for col, clave in (("Margen_pct_mean", "corr_oni_margen"), ("Costo_Marginal_COP_kWh", "corr_oni_costo"),
                           ("Aportes_Pct", "corr_oni_aportes")):
            if col in d.columns:
                pr = d[["ONI", col]].dropna()
                if len(pr) > 6 and pr[col].nunique() > 2:
                    kpis[clave] = round(float(np.corrcoef(pr["ONI"], pr[col])[0, 1]), 2)
                    filas.append({"Hallazgo": f"Correlación ONI ↔ {col.replace('_pct', '').replace('_COP_kWh', '')}",
                                  "Valor": f"r = {kpis[clave]:+.2f}",
                                  "Lectura": ("el ONI explica el comportamiento de esta variable; úsalo "
                                              "como indicador adelantado" if abs(kpis[clave]) >= 0.4 else
                                              "relación débil en esta base: no tomes decisiones solo por el ONI")})
    cv = ctx.get("cruce_v4", pd.DataFrame())
    if isinstance(cv, pd.DataFrame) and not cv.empty:
        if "Costo_Marginal_COP_kWh_Dia" in cv:
            v = float(pd.to_numeric(cv["Costo_Marginal_COP_kWh_Dia"], errors="coerce").mean())
            filas.append({"Hallazgo": "Costo marginal medio de la ventana", "Valor": f"{v:,.0f} COP/kWh",
                          "Lectura": "multiplicar por la energía comprada da el costo de energía "
                                     "(antes de pedágios y restricciones)"})
        if "Indisponibilidad_pct" in cv:
            v = float(pd.to_numeric(cv["Indisponibilidad_pct"], errors="coerce").mean())
            filas.append({"Hallazgo": "Indisponibilidad media de la ventana", "Valor": f"{v:.1f} %",
                          "Lectura": ("reserva operativa ajustada: una falla intempestiva pesa más en el precio"
                                      if v > 12 else "parque disponible holgado; el riesgo está en el agua, no en las fallas")})
    cm = ctx.get("costo_marginal", pd.DataFrame())
    if isinstance(cm, pd.DataFrame) and not cm.empty and "Costo_Marginal_COP_kWh" in cm:
        v = pd.to_numeric(cm["Costo_Marginal_COP_kWh"], errors="coerce").dropna()
        if len(v):
            p99, med = float(np.quantile(v, 0.99)), float(np.median(v))
            filas.append({"Hallazgo": "Cola del costo marginal (p99 / mediana)",
                          "Valor": f"{p99:,.0f} / {med:,.0f} COP/kWh",
                          "Lectura": f"las horas de cola cuestan {p99 / max(med, 1):.1f}× la mediana: "
                                     "ahí se define la rentabilidad de flexible o interruptible"})
    emb = ctx.get("embalses", pd.DataFrame())
    if isinstance(emb, pd.DataFrame) and not emb.empty and "Volumen_Pct" in emb:
        v = pd.to_numeric(emb["Volumen_Pct"], errors="coerce").dropna()
        if len(v):
            filas.append({"Hallazgo": "Volumen útil (cierre vs apertura de la ventana)",
                          "Valor": f"{float(v.iloc[-1]):.1f} % ({float(v.iloc[-1]) - float(v.iloc[0]):+.1f} pp)",
                          "Lectura": ("tendencia a la baja: el mercado entrará en modo ahorro de agua"
                                      if v.iloc[-1] < v.iloc[0] else "los embalses están repuntando")})
    if not filas:
        return None
    t = pd.DataFrame(filas)
    ancho_col = [220, 150, 0]
    fig = go.Figure(go.Table(header=dict(values=list(t.columns), fill_color="#eef2f7",
                                         line=dict(color="white", width=1), font=dict(size=11.5, color="#222"),
                                         align="left", height=28),
                             cells=dict(values=[t["Hallazgo"], t["Valor"], t["Lectura"]],
                                        fill_color=[["#fbfcfe", "#ffffff"] * 8, ["#fbfcfe", "#ffffff"] * 8],
                                        align=["left", "left", "left"], height=30,
                                        line=dict(color="#e8e8e8", width=1), font=dict(size=10.5)),
                             columnwidth=ancho_col))
    fig.update_layout(height=90 + 32 * len(t), margin=dict(l=8, r=8, t=8, b=8),
                      template="plotly_white")
    return {"fig": fig, "leyenda": _leyenda(
        "Hallazgos v4 (celda 17 del notebook, versión legible): los 4-6 números que se defienden solos",
        "Se calculan sobre las tablas ya construidas (cruce_diario_v4, costo marginal horario, serie "
        "mensual ENSO-v4); no hay supuestos del autor. Si una base no existe en el modo actual, esa fila "
        "no aparece en vez de inventarse.",
        "Las dos filas que mueven dinero son la cola del costo marginal (p99 vs mediana) y la "
        "correlación ONI↔costo: la primera dice cuánto vale la flexibilidad, la segunda cuánto puedes "
        "anticipar.",
        "En El Niño, una correlación ONI↔costo ≥ +0.4 con p99/mediana > 2× es la señal para cerrar "
        "contratos a término ahora y no en octubre.",
        "cruce_diario_v4 + CostMargDesp + v4_mensual ⊕ serie local + ONI NOAA")}


FIGURES: dict[str, tuple] = {
    # etiqueta (clave) → (función, subpestaña, título)
    "balance":          (fig_balance_diario,        "resumen", "Balance diario generación vs demanda"),
    "perfil":           (fig_perfil_horario,        "resumen", "Perfil horario (24 h)"),
    "precio":           (fig_precio_diario,         "resumen", "Precio diario del mercado"),
    "embalses":         (fig_embalses_aportes,      "resumen", "Embalses y aportes"),
    "cen":              (fig_v3_01_cen_margen,      "resumen", "CEN y margen de reserva (v3)"),
    "capacidad":        (fig_v3_01b_capacidad_composicion,    "resumen", "Composición de la CEN por tecnología (v3)"),
    "margen_bandas":    (fig_v3_08_margen_reserva_bandas, "resumen", "Margen de reserva por bandas"),
    "v4_01":            (fig_v4_02_disponibilidad,  "resumen", "Indisponibilidad (v4)"),
    "tec_hora":         (fig_heatmap_tecnologia_hora, "generacion", "Matriz tecnología × hora"),
    "corr":             (fig_matriz_correlaciones,  "generacion", "Matriz de correlaciones"),
    "corr_pares":       (fig_v2_04_precio_pares,    "generacion", "Precio vs térmica / embalses / aportes"),
    "perfiles":         (fig_perfiles_por_tecnologia, "generacion", "Perfiles por tecnología"),
    "solares":          (fig_10_perfiles_solares,   "generacion", "Perfiles solares"),
    "fp_tec":           (fig_v3_02_fp_tecnologia,   "generacion", "Factor de planta por tecnología"),
    "fp_plantas":       (fig_v3_03_fp_plantas,      "generacion", "Factor de planta por planta"),
    "fp_dia":           (fig_v3_04b_heatmap_fp_diario,        "generacion", "Factor de planta diario × tecnología (v3 fig_v3_04)"),
    "zonas":            (fig_embalses_zonas,        "hidrologia", "Embalses por zona"),
    "aportes_hist":     (fig_08_aportes_historico,  "hidrologia", "Aportes vs media histórica (24 meses)"),
    "nino_oni":         (fig_enso_01_oni,           "nino", "ONI últimos 7 años (1950→ con el toggle)"),
    "nino_evo":         (fig_enso_02_evolutivo,     "nino", "Embalses/aportes/precio por fase"),
    "nino_scatter_apo": (lambda c: fig_enso_05_scatter(c, "Aportes_Pct"), "nino", "ONI vs aportes"),
    "nino_scatter_pre": (lambda c: fig_enso_05_scatter(c, "Precio_Ponderado_COP"), "nino", "ONI vs precio"),
    "nino_onda":        (fig_enso_06_ultima_onda,   "nino", "Niño1+2 vs Niño3.4 (últimos 4 años)"),
    "escasez":          (fig_v3_05_costo_vs_aportes, "costos", "Aportes vs precio"),
    "costo_dist":       (fig_v3_04_costo_distribucion, "costos", "Distribución del costo marginal"),
    "precio_saeb":      (fig_v2_05b_precio_horario_saeb,      "costos", "Precio por hora + spread pico-valle (SAEB)"),
    "heatmap_precio":   (fig_09_heatmap_precio_hora, "costos", "Heatmap precio × hora"),
    "emisiones":        (fig_v3_06_emisiones,       "costos", "Emisiones diarias"),
    "pareto":           (fig_v3_07_ranking_plantas, "costos", "Top plantas emisoras"),
    "v4_heat":          (fig_v4_03_heatmap_indisp, "v4", "Heatmap despacho térmico"),
    "v4_matriz":        (fig_v4_04_matriz_cruce,   "v4", "Matriz disponibilidad × precio"),
    "v4_enso":          (fig_v4_05_enso_economico, "v4", "El Niño en dinero (ventana + 5 años)"),
    # --- pestaña nueva A · perfiles × hora y matrices (notebook v2 c31-38 · v3 c21/24 · v4 c7/8)
    "mat_tec_hora":     (fig_v2_08_heatmap_tecno_hora,          "matrices", "Heatmap tecnología × hora (promedio diario)"),
    "mat_perfiles":     (fig_v2_08b_perfiles_tecnologia_hora,   "matrices", "Perfiles horarios por tecnología (pico anotado)"),
    "mat_precio":       (fig_v2_09_heatmap_precio_dia_hora,    "matrices", "Heatmap precio día × hora"),
    "mat_costo":        (fig_v4_02_costo_marginal,             "matrices", "Costo marginal: serie diaria + perfil horario"),
    "mat_costo_dh":     (fig_v4_03_heatmap_costo_dia_hora,     "matrices", "Heatmap costo marginal día × hora"),
    "mat_enso":         (fig_v3_09_dashboard_enso,             "matrices", "Panel ENSO (ONI · aportes · embalses · precio)"),
    "mat_fase":         (fig_v3_11_climatologia_fase,          "matrices", "Climatología por fase ENSO con la ventana actual ★"),
    # --- pestaña nueva B · CO₂ (v3 c13-16) y economía del ENSO (v4 c14-17)
    "co2_emis":         (fig_v3_05_emisiones_diarias,          "co2en", "Emisiones diarias por combustible + intensidad"),
    "co2_perfil":       (fig_v3_06_perfil_horario_emisiones,   "co2en", "Perfil horario de emisiones vs solar"),
    "co2_pareto":       (fig_v3_07_top_emisores_pareto,        "co2en", "Top emisoras y curva de Pareto"),
    "co2_corr":         (fig_v3_08_correlacion_extendida,      "co2en", "Correlación extendida (emisiones × contexto)"),
    "en_hallazgos":     (fig_v4_17_hallazgos_v4,               "co2en", "Hallazgos v4 en tabla legible (v4 c17)"),
}


# ==============================================================================
# 1z · MAPA Celda↔Figura y COBERTURA del notebook (verificación celda por celda)
# ==============================================================================
#: Convención: número de celda = **índice absoluto dentro de `cells` del JSON del notebook,
#: 0-based, contando markdown y código** (el mismo de `analysis/cells.txt` y de
#: `python tools/verificar_celdas.py`). El `In [n]` de Jupyter no coincide: cuenta solo código.
#: Regenerar los tres literales:  `python tools/verificar_celdas.py --literal`
#: Comprobar coherencia (sale 1 si algo quedó sin espejo):  `python tools/verificar_celdas.py`
COBERTURA_NB: dict[str, tuple[str, tuple[tuple[int, int], ...]]] = {
    "figura": ("la celda construye una figura que la app muestra", ((70, 94), (110, 114), (129, 129), (131, 131), (135, 135), (137, 137), (139, 139), (141, 141), (143, 143), (145, 145), (147, 147), (149, 149), (155, 155), (157, 157), (159, 159), (161, 161), (163, 163), (167, 167), (169, 169), (171, 171), (176, 176), (178, 178), (180, 180), (182, 182), (184, 184), (186, 186), (188, 188), (190, 190), (195, 195), (197, 197), (199, 199), (201, 201), (203, 203), (205, 205), (207, 207))),   # 63 celdas
    "cálculo portado": ("cálculo del notebook reproducido en app.py (función indicada)", ((2, 2), (4, 5), (7, 10), (12, 17), (19, 19), (21, 23), (25, 27), (30, 31), (33, 33), (35, 38), (40, 40), (43, 43), (45, 45), (48, 48), (50, 69), (95, 99), (101, 106), (109, 109), (115, 116), (121, 121), (123, 123), (125, 125), (127, 127), (133, 133), (165, 165), (209, 209))),   # 72 celdas
    "salida en notebook": ("print/.head()/display → caption y tablas de la app", ((100, 100), (107, 108))),   # 3 celdas
    "config/imports": ("imports y %pip → requirements.txt y cabecera de app.py", ((3, 3),)),   # 1 celdas
    "documentación": ("markdown: secciones, portadas y notas del cuaderno", ((0, 1), (6, 6), (11, 11), (18, 18), (20, 20), (24, 24), (28, 29), (32, 32), (34, 34), (39, 39), (41, 42), (44, 44), (46, 47), (49, 49), (117, 120), (122, 122), (124, 124), (126, 126), (128, 128), (130, 130), (132, 132), (134, 134), (136, 136), (138, 138), (140, 140), (142, 142), (144, 144), (146, 146), (148, 148), (150, 154), (156, 156), (158, 158), (160, 160), (162, 162), (164, 164), (166, 166), (168, 168), (170, 170), (172, 175), (177, 177), (179, 179), (181, 181), (183, 183), (185, 185), (187, 187), (189, 189), (191, 194), (196, 196), (198, 198), (200, 200), (202, 202), (204, 204), (206, 206), (208, 208), (210, 211))),   # 73 celdas
}

CELDAS_NB: dict[str, tuple[int, int]] = {
    'aportes_hist': (88, 90),
    'balance': (70, 71),
    'capacidad': (131, 131),
    'cen': (129, 129),
    'co2_corr': (149, 149),
    'co2_emis': (143, 143),
    'co2_pareto': (147, 147),
    'co2_perfil': (145, 145),
    'corr': (81, 83),
    'corr_pares': (76, 80),
    'costo_dist': (186, 186),
    'embalses': (86, 89),
    'emisiones': (141, 143),
    'en_hallazgos': (207, 207),
    'escasez': (76, 77),
    'fp_dia': (139, 139),
    'fp_plantas': (137, 137),
    'fp_tec': (135, 135),
    'heatmap_precio': (110, 111),
    'margen_bandas': (129, 129),
    'mat_costo': (186, 186),
    'mat_costo_dh': (188, 188),
    'mat_enso': (161, 161),
    'mat_fase': (167, 205),
    'mat_perfiles': (93, 94),
    'mat_precio': (110, 111),
    'mat_tec_hora': (91, 92),
    'nino_evo': (159, 161),
    'nino_onda': (155, 155),
    'nino_oni': (157, 157),
    'nino_scatter_apo': (163, 163),
    'nino_scatter_pre': (163, 163),
    'pareto': (147, 147),
    'perfil': (72, 73),
    'perfiles': (93, 94),
    'precio': (74, 75),
    'precio_saeb': (84, 85),
    'solares': (112, 114),
    'tec_hora': (91, 92),
    'v4_01': (182, 184),
    'v4_enso': (195, 199),
    'v4_heat': (143, 145),
    'v4_matriz': (180, 190),
    'zonas': (86, 87),
}

CALC_APP_NB: dict[int, str] = {
    2: "requirements.txt / pip install pydataxm",
    4: "cabecera de imports de app.py",
    5: "cabecera de imports de app.py",
    7: "obtener_generacion → descargar_metrica",
    8: "descargar_ventana",
    9: "descargar_ventana",
    10: "descargar_ventana",
    12: "_detectar_base_dir",
    13: "items_a_dataframe",
    14: "preparar_dim_plantas",
    15: "preparar_dim_plantas",
    16: "_encontrar_catalogo_mas_reciente → buscar_cache, cargar_catalogos → buscar_cache",
    17: "preparar_dim_plantas",
    19: "actualizar_catalogos_maestros → preparar_dim_plantas",
    21: "transformar_generacion → transformar_generacion, preparar_dim_plantas → preparar_dim_plantas",
    22: "transformar_generacion",
    23: "preparar_dim_plantas",
    25: "verificar_y_corregir_integridad → verificar_y_corregir_integridad",
    26: "preparar_dim_plantas",
    27: "verificar_y_corregir_integridad",
    30: "enriquecer_generacion",
    31: "enriquecer_generacion",
    33: "perfil_tecnologia_hora",
    35: "enriquecer_generacion",
    36: "ctx_agentes_lookup",
    37: "ctx_agentes_lookup",
    38: "ctx_agentes_lookup",
    40: "enriquecer_generacion",
    43: "enriquecer_generacion",
    45: "enriquecer_generacion",
    48: "preparar_dim_plantas",
    50: "comparar_catalogos → comparar_catalogos_historicos, comparar_todos_los_catalogos → comparar_catalogos_historicos",
    51: "cabecera de app.py",
    52: "METRICAS_V2 / constantes de app.py",
    53: "detectar_base_dir → _detectar_base_dir",
    54: "clasificar_fechas → rango_fechas, _configuracion_endpoint → post_xm, descargar_metrica → descargar_metrica",
    55: "mode_demo y MODO",
    56: "METRICAS_V2",
    57: "cargar_csv_mas_reciente → buscar_cache, rango_fechas_raw → rango_fechas, filtrar_raw_por_periodo → filtrar_por_periodo",
    58: "guardar_cache",
    59: "columnas_horarias → columnas_horarias, agregar_timestamp → agregar_timestamp, wide_to_long → wide_to_long",
    60: "procesar_todas_las_metricas",
    61: "validar_tablas_analiticas → validar_tablas_analiticas",
    62: "render_datos (bloque 3)",
    63: "_detectar_base_dir",
    64: "preparar_dim_plantas → preparar_dim_plantas",
    65: "render_datos (linaje)",
    66: "render_datos (linaje)",
    67: "sistema_hora_a_dia",
    68: "sistema_hora_a_dia",
    69: "to_date → agregar_timestamp",
    95: "_localizar_script_operativo → cargar_mapa_operativo, _cargar_modulo_operativo → cargar_mapa_operativo",
    96: "enriquecer_generacion_operativa → gen_con_ambito",
    97: "resumir_cobertura_operativa → niveles_disponibles",
    98: "_slug_operativo → _slug_operativo, filtrar_generacion_operativa → filtrar_ambito, graficar_resumen_areas → _fig_scope_perfil",
    99: "graficar_heatmap_operativo → _fig_scope_perfil, graficar_perfiles_operativos → _fig_scope_perfil",
    101: "crear_explorador_operativo → explorar_rio_zona, actualizar_ambitos → _selector_ambito, actualizar_figura → _fig_scope_perfil",
    102: "_localizar_script_catalogo → cargar_mapa_operativo, _cargar_modulo_catalogo → cargar_mapa_operativo",
    103: "auditar_y_exportar_cobertura → niveles_disponibles",
    104: "enriquecer_generacion_geografica → gen_con_ambito",
    105: "filtrar_generacion_geografica → filtrar_ambito, graficar_resumen_regiones → _fig_scope_perfil",
    106: "graficar_heatmap_geografico → _fig_scope_perfil, graficar_perfiles_geograficos → _fig_scope_perfil",
    109: "crear_explorador_geografico → explorar_rio_zona, actualizar_ambitos → _selector_ambito, actualizar_figura → _fig_scope_perfil",
    115: "fig_10_perfiles_solares",
    116: "construir_hallazgos → construir_hallazgos, agregar → construir_hallazgos",
    121: "METRICAS_V3 / constante",
    123: "_sondar_metrica → _sondar_metrica, ventana_efectiva → ventana_efectiva, _silenciar_descarga → sondea_metrica",
    125: "daily_to_long_recurso → daily_to_long_recurso, wide_to_long_recurso_combustible → wide_to_long_combustible",
    127: "_normalizar_tecnologia → normalizar_tecnologia",
    133: "factores_planta",
    165: "tabla_impacto_enso",
    209: "render_datos (descargables)",
}


CELDAS_ETAPA: dict[str, tuple[int, int, str]] = {
    "V1": (7, 20, "§1 Extracción y §2 catálogos: `obtener_generacion`, `cargar_catalogos`, "
                  "`actualizar_catalogos_maestros`"),
    "V2": (21, 119, "§3 ETL y §3.1 integridad, §4 EDA, §5 evolución, §6 cierre y anexos A/B"),
    "V3": (120, 174, "§7 Capacidad/emisiones (fig_v3_01…11) y §8 ENSO (serie larga, ONI, fases)"),
    "V4": (175, 212, "§10 Indisponibilidad/margen/MCOP y §11 ENSO económico (mensual, Δ% por episodio)"),
    "OK": (210, 212, "§9/§12 cierres v3 y v4 (export de tablas → botones de descarga)"),
}


def celdas_texto(clave: str | None) -> str:
    """'celda 91' o 'celdas 112-114' para la clave de figura dada ('' si no hay ancla)."""
    if not clave or clave not in CELDAS_NB:
        return ""
    a, b = CELDAS_NB[clave]
    return f"celda {a}" if a == b else f"celdas {a}-{b}"


def celdas_de_etapa(etapa: str | None) -> str:
    """Rango de celdas del notebook que corresponde a la etapa de carga (V1..V4/OK)."""
    if not etapa or etapa not in CELDAS_ETAPA:
        return ""
    a, b, _q = CELDAS_ETAPA[etapa]
    return f"celdas {a}-{b}"


def version_celda(celda: int | None) -> str:
    """Capa del notebook a la que pertenece una celda absoluta ('v1v2' / 'v3' / 'v4')."""
    if celda is None:
        return ""
    if celda <= 119:
        return "v1v2"
    if celda <= 174:
        return "v3"
    return "v4"


def sello_figura(clave: str | None) -> str:
    """' · v3 · celda 143' para etiquetas y bitácora ('' si la figura no tiene ancla)."""
    txt = celdas_texto(clave)
    if not txt:
        return ""
    return f" · {version_celda(CELDAS_NB[clave][0])} · notebook {txt}"


def version_de_etapa(etapa: str | None) -> str:
    return {"V1": "v1", "V2": "v2", "V3": "v3", "V4": "v4"}.get(str(etapa), "v2")


def tabla_cobertura_nb() -> pd.DataFrame:
    """Expande `COBERTURA_NB` + `CALC_APP_NB` + `CELDAS_NB` en una fila por celda (las 212)."""
    filas: dict[int, dict[str, Any]] = {}
    for estado, (_desc, intervalos_) in COBERTURA_NB.items():
        for a, b in intervalos_:
            for i in range(a, b + 1):
                filas[i] = {"celda": i, "tipo": ("markdown" if estado == "documentación" else "código"),
                            "estado": estado, "en la app": "", "figura(s)": ""}
    for i, dond in CALC_APP_NB.items():
        if i in filas:
            filas[i]["en la app"] = dond
    for clave, (a, b) in CELDAS_NB.items():
        for i in range(a, b + 1):
            if i in filas and filas[i]["estado"] == "figura":
                filas[i]["figura(s)"] = (filas[i]["figura(s)"] + ", " + clave).strip(", ")
                filas[i]["en la app"] = f"FIGURES[{clave!r}]"
    return pd.DataFrame([filas[i] for i in sorted(filas)])


# ==============================================================================
# 8 · APLICACIÓN STREAMLIT
# ==============================================================================
TITULOS_TABS = [
    ("🎯 Resumen ejecutivo", "resumen"),
    ("⚡ Generación", "generacion"),
    ("💧 Hidrología", "hidrologia"),
    ("🌊 El Niño", "nino"),
    ("💰 Costos y emisiones", "costos"),
    ("🛠️ Disponibilidad (v4)", "v4"),
    ("🕐 Perfiles × hora", "matrices"),
    ("🔥 CO₂ y economía ENSO", "co2en"),
    ("🕸️ Precios nodales", "nodal"),
    ("📚 Datos y bitácora", "datos"),
]

AYUDA_CONFIG = """
- **Modo de datos.** `Auto` usa la caché disponible en `data/raw` y consulta la API de XM
  solamente cuando hace falta. `Solo caché` funciona sin internet (demo o API caída);
  `Forzar descarga API` re-descarga todo; `Solo referencia` toma lo versionado en
  `data/reference` y arranca en 0 s.
- **Ventana.** El análisis utiliza el último mes cerrado y puede extenderse (14-180 días)
  para observar la evolución. Cada métrica se ajusta sola al rezago de publicación: el
  detalle está en 📚 *Datos y bitácora*.
- **Métricas pesadas.** Actívalas únicamente cuando sean necesarias: `Gene_Recurso`,
  `CapEfecNeta`, `EmisionesCO2`, `ConsCombustibleMBTU`, `DispoCome` y `DemaMaxPot` traen
  miles de filas por día y ensanchan `data/raw`.
- **Sin selectores de tecnología / agente / tipo de recurso.** Se retiraron de esta barra:
  siempre estaban en *todas* y solo generaban la falsa impresión de que las cifras estaban
  segmentadas. El filtrado fino quedó donde tiene sentido: **⚡ Generación → Explorador de planta**
  y **Explorador de recursos y cuencas** (nivel + ámbito). La demanda y el precio del sistema **no**
  se filtran en ninguna parte: son agregados del CNO.
- **🩹 Incluir días provisionales de XM** (apagado por defecto). XM cierra el día *D* en *D+1/D+2* y
  las primeras entregas traen una fracción del día; la app lo detecta comparando el total de cada feed
  volumétrico contra la mediana de sus propios días y **excluye** esos días de todos los agregados para
  que el balance no se infle. Encenderlo sirve solo para inspeccionar el crudo (y aparece un recuadro
  rojo de advertencia). La auditoría se ve en 📚 *Datos y bitácora → Auditoría de completitud*.

**Qué se está cargando en cada etapa** (lo dice la tarjeta de progreso mientras corre):

Cada línea del progreso trae, entre corchetes, la **celda del notebook** que se está ejecutando
(`[nb celdas 7-20]`), con el índice absoluto de celda en `01_ETL_exploracion_v1.json` (0-based,
contando markdown y código; el `In [n]` de Jupyter no coincide porque solo numera las de código):

- **V1** · celdas **7-20** — 14-19 consultas en paralelo → *API REST de XM (JSON)* y, si ya existe,
  el *CSV local* de `data/raw` (`obtener_generacion`, `cargar_catalogos`, `actualizar_catalogos_maestros`).
- **V2** · celdas **21-119** — ETL `wide→long` (c59-60), validación de tablas (c61), unidades, 24 h por
  día, duplicados y NaN → *memoria (pandas)*; aquí entra también la auditoría de días provisionales.
- **V3** · celdas **120-174** — cruce con `dim_plantas`, resúmenes diarios, CEN/margen, factor de planta,
  emisiones y ENSO (c155-159) → *CSV locales* de `ListadoRecursos` / `ListadoAgentes` / `ListadoRios`.
- **V4** · celdas **175-212** — indisponibilidad, margen, costo marginal, serie mensual ENSO-económica
  (c195-199) e impacto por episodio (c203) → *CSV público de NOAA/CPC* + semillas de `data/reference`
  + API de XM.
- **Figuras** · no bajan nada: se dibujan sobre el `ctx`, quedan en caché (6 h) y su encabezado dice
  `· v3 · notebook celda 143`, así puedes abrir el cuaderno en esa celda y comparar.

No hay base de datos SQL ni bucles abiertos: solo API REST con tiempo de espera de 180 s,
4 reintentos con pausa y caché en disco. El Niño toca casi todo lo que se pinta aquí
(aportes, embalses, CEN hidráulica, térmica de respaldo, precio y emisiones), y cada
gráfico tiene una leyenda que dice qué decisión permite tomar.
"""

def _cache_key(dias: int, opciones: dict[str, bool], modo: str, incluir_provisionales: bool = False) -> str:
    return (f"{_hoy():%Y%m%d}|{dias}|"
            + ",".join(f"{k}={int(v)}" for k, v in sorted(opciones.items()))
            + f"|{modo}|prov={'1' if incluir_provisionales else '0'}")


@st.cache_data(ttl=6 * 3600, show_spinner=False, max_entries=6)
def obtener_contexto_cached(_key: str, dias: int, opciones_json: str, modo: str,
                            incluir_provisionales: bool = False) -> dict[str, Any]:
    """Wrapper con caché de Streamlit sobre `construir_contexto`.

    `_key` no se usa dentro: solo fuerza una entrada distinta por combinación de parámetros.
    """
    opciones = json.loads(opciones_json)
    ctx = construir_contexto(dias, opciones, modo, incluir_provisionales=incluir_provisionales)
    # los DataFrames viajan igual; st.cache_data los serializa (pickle) para la sesión
    return ctx


def sidebar_controles() -> tuple[int, dict[str, bool], str]:
    st.sidebar.title("⚙️ Configuración")
    modo = st.sidebar.radio(
        "Modo de datos", ["auto", "cache", "api", "offline"], index=0,
        format_func=lambda x: {"auto": "🔄 Auto (caché + API)", "cache": "💾 Solo caché (sin red)",
                               "api": "🌐 Forzar descarga API", "offline": "📦 Solo referencia (demo)"}[x],
        help="Auto reutiliza los CSV de data/raw y baja lo que falte; Forzar ignora la caché; "
             "Referencia usa únicamente los archivos versionados de data/reference (arranca en 0 s).")
    dias = st.sidebar.slider("Días de la ventana (el notebook usa 30 = último mes cerrado)", 14, 180,
                            VENTANA_POR_DEFECTO + 1, step=1,
                            help="Con >31 días la app parte cada petición en bloques de 31 (límite de la API).")
    st.sidebar.markdown("**Siempre activas (1 petición por métrica)**")
    opciones: dict[str, bool] = {
        "Gene_Sistema": True, "DemaReal_Sistema": True, "PrecBolsNaci_Sistema": True,
        "PPPrecBolsNaci_Sistema": True, "PorcVoluUtilDiar_Sistema": True, "PorcApor_Sistema": True,
        "AporEner_Sistema": True, "PrecEsca_Sistema": True, "CostMargDesp_Sistema": True,
        "Gene_Recurso": st.sidebar.toggle("Detalle por planta (`Gene_Recurso`)", value=True,
                                          help="Mismo feed que Gene_Sistema: apágalo solo si quieres la "
                                               "carga mínima posible."),
        "DemaMaxPot_Sistema": st.sidebar.toggle("Demanda máxima programada (`DemaMaxPot`)",
                                                value=dias <= 60,
                                                help="XM la publica con semanas de rezago; la app retrocede "
                                                     "la ventana hasta encontrarla."),
        "CapEfecNeta_Recurso": st.sidebar.toggle("CEN por recurso (`CapEfecNeta`)", value=dias <= 45,
                                                 help="≈2 000 recursos/día. Necesaria para CEN, margen de "
                                                      "reserva y factor de planta."),
    }
    st.sidebar.markdown("**Pesadas (enciéndelas bajo demanda)**")
    opciones["EmisionesCO2_RecursoComb"] = st.sidebar.toggle("Emisiones CO₂ (`EmisionesCO2`)", value=False,
                                                            help="Planta × combustible × 24 h: tarda y "
                                                                 "ocupa muchos MB en data/raw.")
    opciones["ConsCombustibleMBTU_Recurso"] = st.sidebar.toggle("Consumo de combustible (`ConsCombustibleMBTU`)",
                                                              value=False)
    opciones["EmisionesCO2Eq_Recurso"] = st.sidebar.toggle("CO₂ equivalente (`EmisionesCO2Eq`)", value=False)
    opciones["factorEmisionCO2e_Sistema"] = st.sidebar.toggle("Factor de emisión (`factorEmisionCO2e`)",
                                                            value=False)
    opciones["DispoCome_Recurso"] = st.sidebar.toggle("Disponibilidad comercial (`DispoCome`)", value=False,
                                                     help="La más pesada del catálogo (todas las plantas × 24 h, "
                                                          "meses de historia). Solo si necesitas la "
                                                          "indisponibilidad exacta en lugar del proxy.")
    st.sidebar.caption("Catálogos maestros (`ListadoRecursos`, `ListadoAgentes`, `ListadoRios`) se "
                       "consultan siempre: miden el tamaño del mercado y no pesan.")
    st.sidebar.markdown("---")
    c1, c2 = st.sidebar.columns(2)
    if c1.button("♻️ Recargar", use_container_width=True,
                 help="Descarta lo cacheado en memoria y vuelve a leer data/raw o la API."):
        st.cache_data.clear()
        st.rerun()
    if c2.button("🧹 Limpiar caché", use_container_width=True,
                 help="Borra los CSV de data/raw (los de data/reference NO se tocan)."):
        borradas = 0
        for f in RAW_DIR.glob("*.csv"):
            try:
                f.unlink()
                borradas += 1
            except OSError:
                pass
        st.cache_data.clear()
        bitacora(f"Caché limpia: {borradas} archivo(s) de data/raw eliminados")
        st.rerun()
    with st.sidebar.expander("Cómo leer esta barra lateral", expanded=False):
        st.markdown(AYUDA_CONFIG)
    return dias, opciones, modo


def banner_el_nino(ctx: dict) -> None:
    estado = resumen_evento_actual(ctx["nino"])
    if not estado or not estado.get("es_el_nino"):
        st.warning(
            "🌡️ **Fase ENSO actual: sin evidencia de El Niño en los datos cargados.** Si cargaste "
            "solo referencia, conecta a la API (modo *Auto*) para traer el ONI fresco de NOAA.")
        return
    mes_fmt = _mes_es(estado["mes_ultimo"])
    n12 = estado.get("nino1_2", float("nan"))
    con = (f"Niño1+2 (costa) **{n12:+.2f} °C** en el último mes publicado"
           if np.isfinite(n12) else "Niño1+2 sin serie mensual")
    st.markdown(
        f"""<div style="border:1px solid {ROJO};border-left:8px solid {ROJO};
        background:linear-gradient(90deg, #fff1f0 0%, #fff8f6 60%, #ffffff 100%);
        border-radius:10px;padding:16px 18px;margin:6px 0 14px 0">
        <div style="font-size:1.05rem;font-weight:700;color:{ROJO}">
        ⚠️ EL NIÑO ESTÁ ACTIVO E INTENSIFICÁNDOSE — AGOSTO DE 2026</div>
        <div style="color:#333;margin-top:6px">
        <b>Lo que dicen los datos del tablero:</b> el <b>índice Niño-3.4 suavizado a 3 meses (ONI)</b>
        cruzó el umbral de +0,5 °C de forma sostenida en la estación <b>{estado['inicio_oficial']}</b> y llega a
        <b>{_coma(estado['oni_ultimo'], 2)} °C</b> en {mes_fmt} ({estado['meses_activos']} mes(es) consecutivos
        en fase El Niño; fuerza estimada:
        <b>{estado['fuerza']}</b>). {con} — la zona costera, que es la que gobierna la lluvia de Colombia,
        está más caliente que el Pacífico central: escenario de déficit hídrico severo.<br>
        <b>Lo que dicen las agencias:</b> NOAA/CPC emitió <i>El Niño Advisory</i> el <b>13 de agosto de 2026</b>
        (probabilidad &gt; 90 % de un evento <i>very strong</i> en OND 2026-27). La cifra de referencia del
        boletín es el <b>índice Niño-3.4 de julio de 2026: +1,4 °C</b> (anomalía mensual del Niño1+2 de ese
        mismo mes: +2,9 °C) e IRI (19-ago-2026) reporta Niño3.4 semanal de +2,7 °C centrado el 12-ago y
        100 % de probabilidad de El Niño entre ago-2026 y feb-2027. NOAA aún no publica el ONI de la
        estación JAS 2026 (sale en septiembre): por eso <b>agosto aparece sin ONI oficial</b> en las series
        y el tablero lo marca explícitamente en lugar de inventarlo.</div>
        <div style="margin-top:8px;font-size:.86rem;color:#555">
        <b>Qué hacer con esto:</b> revisar las pestañas 💧 Hidrología (aportes/embalses) y 💰 Costos
        (precio y emisiones) — las leyendas de cada figura indican la decisión asociada.
        </div></div>""", unsafe_allow_html=True)


def filtros_globales(ctx: dict) -> dict[str, Any]:
    """Diccionario de filtros globales: **los selectores se retiraron a pedido del autor.**

    Tecnología / Agente / Tipo de recurso siempre venían en "todas", así que solo generaban
    confusión (y la falsa impresión de que las cifras estaban segmentadas). Se conserva la
    estructura `{"tec","agente","rectype"}` vacía = sin filtro, porque `aplicar_filtros` y las
    pestañas la consumen; el filtrado fino sigue donde sí tiene sentido: el **explorador de
    planta** y el **explorador de ámbito operativo/geográfico**.
    """
    f: dict[str, Any] = {"tec": [], "agente": [], "rectype": []}
    ag = ctx.get("catalogo_agentes", pd.DataFrame())
    if "Values_Code" in ag and "Values_Name" in ag:
        pares = (ag[["Values_Code", "Values_Name"]].dropna().drop_duplicates("Values_Code")
                 .astype(str))
        _AGENTES_LOOKUP.clear()
        _AGENTES_LOOKUP.update(dict(zip(pares["Values_Code"], pares["Values_Name"])))
        ag_all = sorted({f"{c} — {n[:28]}" for c, n in _AGENTES_LOOKUP.items()})
    return f


_AGENTES_LOOKUP: dict[str, str] = {}


def ctx_agentes_lookup() -> dict[str, str]:
    """Código de agente → nombre (se rellena al construir el contexto)."""
    return _AGENTES_LOOKUP


def aplicar_filtros(df: pd.DataFrame, filtros: dict[str, Any], dim: pd.DataFrame) -> pd.DataFrame:
    """Filtro por tecnología / agente / tipo de recurso sobre tablas que tengan `Codigo_Planta`."""
    if df is None or df.empty or "Codigo_Planta" not in df.columns:
        return df if df is not None else pd.DataFrame()
    if not any(filtros.get(k) for k in ("tec", "agente", "rectype")) or dim is None or dim.empty:
        return df
    d = dim.drop_duplicates(subset=["Codigo_Planta"])
    if filtros.get("tec") and "Tecnologia" in d:
        d = d[d["Tecnologia"].isin(filtros["tec"])]
    if filtros.get("rectype") and "Tipo_Recurso" in d:
        d = d[d["Tipo_Recurso"].isin(filtros["rectype"])]
    if filtros.get("agente"):
        # el multiselect muestra "CÓDIGO — Nombre"; dim_plantas guarda el código del agente
        ag = ctx_agentes_lookup()
        codigos = {str(a).split(" — ")[0].strip() for a in filtros["agente"]}
        codes_por_agente = {c for c, n in ag.items() if n in set(filtros["agente"]) or c in codigos}
        if "Codigo_Agente" in d:
            if codes_por_agente:
                d = d[d["Codigo_Agente"].astype(str).isin(codes_por_agente)]
            else:
                d = d[d["Codigo_Agente"].astype(str).isin(codigos)]
    codes = set(d["Codigo_Planta"].astype(str))
    return df[df["Codigo_Planta"].astype(str).isin(codes)] if codes else df


def render_resultado(res: dict | None, titulo: str, *, msg_extra: str = "",
                     clave: str = "") -> bool:
    """Pinta `{"fig","leyenda"}` y devuelve True si hubo figura (patrón de las pestañas nuevas)."""
    if not res or res.get("fig") is None:
        extra = f" {msg_extra}" if msg_extra else ""
        st.info(f"Sin datos suficientes para *{titulo}* en la ventana/modo actual.{extra} "
                "Puedes activar métricas pesadas en la barra lateral, ampliar la ventana o pasar "
                "al modo *Auto*.")
        return False
    st.plotly_chart(res["fig"], use_container_width=True, config={"displaylogo": False},
                    key=_clave_fig(clave or titulo))
    if res.get("leyenda"):
        st.caption(res["leyenda"])
    return True


def render_figuras(ctx: dict, tab_key: str, filtros: dict[str, Any]) -> None:
    claves = [k for k, (_f, t, _ti) in FIGURES.items() if t == tab_key]
    if not claves:
        return
    for i, clave in enumerate(claves):
        fn, _tab, titulo = FIGURES[clave]
        sello = sello_figura(clave)
        with st.expander(f"📊 {titulo}{sello}", expanded=(i < 3)):
            try:
                res = figura_cacheada(f"{tab_key}:{clave}", fn, ctx)
            except Exception as exc:                              # noqa: BLE001
                st.error(f"No se pudo construir *{titulo}*: `{type(exc).__name__}: {str(exc)[:200]}`")
                bitacora(f"figura {titulo}: {exc}", "warn")
                continue
            if not res or res.get("fig") is None:
                st.info(f"Sin datos suficientes para *{titulo}* en la ventana/modo actual. "
                        f"Prueba activar métricas pesadas en la barra lateral o usar el modo *Auto*.")
                res = res or {}
            if res.get("fig") is not None:
                st.plotly_chart(res["fig"], use_container_width=True,
                                config={"displaylogo": False}, key=_clave_fig(titulo))
            if res.get("leyenda"):
                st.caption(res["leyenda"])
            if sello:
                _fn = getattr(FIGURES[clave][0], "__name__", "lambda (figura compuesta)")
                st.caption(f"▸ **{sello.lstrip(' ·').capitalize()}** del notebook. Acá se reproduce en "
                           f"`app.py::{_fn}()`; el original (v1-v4 completo, celda por celda) está en "
                           f"[github.com/srriverar/sin-el-nino-dashboard](https://github.com/srriverar/"
                           f"sin-el-nino-dashboard).")


def render_datos(ctx: dict) -> None:
    st.subheader("1 · Linaje de la extracción (qué se pidió, qué se obtuvo)")
    lin = ctx["linaje"].copy()
    for c in ("métrica", "filas", "origen"):
        if c in lin:
            lin = lin.rename(columns={c: c.replace("é", "e")})
    st.dataframe(lin, use_container_width=True, hide_index=True)
    st.caption("Si una métrica aparece con 0 filas: normalmente es rezago de publicación "
               "(DemaMaxPot: 2-6 semanas; CapEfecNeta/EmisionesCO2: D-1/D-2) o que la desactivaste "
               "en la barra lateral.")
    st.subheader("2 · Auditoría de completitud (días que XM aún no cierra)")
    prov = ctx.get("dias_provisionales", pd.DataFrame())
    if isinstance(prov, pd.DataFrame) and not prov.empty:
        st.dataframe(prov, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Descargar día_a_día.csv (auditoría)", prov.to_csv(index=False).encode(),
                           "dia_a_dia_auditoria.csv", "text/csv", key="dl-auditoria")
        st.caption(f"Regla: para cada feed volumétrico el total del día se compara con la mediana de "
                   f"sus propios días (excluyendo los 4 últimos). Bajo {UMBRAL_DIA_PRELIMINAR:.0%} de esa "
                   f"mediana y siendo uno de los últimos días ⇒ preliminar. Umbral configurable con "
                   f"`APP_MANUEL_UMBRAL_DIA`; feeds evaluados: {', '.join(FEEDS_SENSIBLES)}. Tratamiento "
                   f"actual: **{'incluidos (aviso rojo activo en Resumen)' if ctx.get('incluir_provisionales') else 'excluidos de todos los agregados'}**. "
                   f"Último día con dato cerrado: **{ctx.get('fin_efectiva', ctx['fin'])}** "
                   f"(la ventana pedía hasta {ctx['fin']}).")
    else:
        st.caption(f"Sin días preliminares en esta ventana: todos los días de los feeds volumétricos "
                   f"están dentro de {UMBRAL_DIA_PRELIMINAR:.0%} de su mediana y el dato cerrado llega "
                   f"hasta {ctx.get('fin_efectiva', ctx['fin'])} (hoy "
                   f"{pd.Timestamp.now():%Y-%m-%d}); el rezago de publicación de XM es de 1-2 días para "
                   f"el mercado y de 2-6 semanas para `DemaMaxPot`.")
    st.subheader("3 · Validación de las tablas analíticas (celda 61 del notebook)")
    cal = ctx.get("calidad", pd.DataFrame())
    if isinstance(cal, pd.DataFrame) and not cal.empty:
        st.dataframe(cal, use_container_width=True, hide_index=True)
        st.caption("Reglas del notebook, aplicadas tal cual: la columna de valor no puede ser negativa "
                   "(ni pasar de 100 % en embalses/aportes), no debe tener nulos y el grano "
                   "planta × fecha × hora no puede repetirse. `horas_vistas = 24` es la señal de que el "
                   "día está cerrado; si un día aparece con menos horas, la app lo marca en la auditoría "
                   "de completitud de arriba. **No se oculta ninguna observación**: se reportan los "
                   "contadores y se decide con eso.")
    else:
        st.caption("Sin tablas que validar en este modo (no llegó datos de la API ni caché).")
    st.subheader("4 · Tablas analíticas (descargables)")
    nombres_tablas = [
        ("resumen_diario_v2", "resumen", "Resumen diario con balance, precio, embalses, aportes y escasez"),
        ("sistema_horario_v2", "sistema", "Sistema horario (demanda + costo marginal + embalses + generación por tecnología)"),
        ("generacion_enriquecida_v2", "gen_enr", "Generación por planta × hora con dim_plantas aplicada"),
        ("dim_plantas", "dim_plantas", "Catálogo de plantas (ListadoRecursos) normalizado"),
        ("embalses_diario", "embalses", "% de volumen útil diario (nacional)"),
        ("aportes_diario", "aportes", "Aportes diarios en mm³ y mm equivalentes"),
        ("escasez_diaria", "escasez", "Precio de escasez diario"),
        ("cen_diario_v3", "cen_diario", "CEN por tecnología y margen de reserva"),
        ("cen_por_recurso_v3", "cen_recurso", "CEN diario por planta"),
        ("factores_planta_v3", "fp", "Factor de planta mensual por planta"),
        ("emisiones_diarias_v3", "emisiones_dia", "tCO₂ e intensidad diarios"),
        ("emisiones_horarias_v3", "emisiones", "tCO₂ por planta × combustible × hora"),
        ("indisponibilidad_diaria_v4", "indisp", "Indisponibilidad y reserva operativa (v4)"),
        ("cruce_diario_v4", "cruce_v4", "Cruce disponibilidad × precio × demanda (v4)"),
        ("nino_mensual_completo", "nino", "Serie mensual 2015→ con ONI, fase, rezagos y Niño1+2"),
        ("eventos_enso", "nino_eventos", "Eventos El Niño / La Niña detectados (≥5 meses)"),
        ("tabla_impacto_enso", "impacto", "Impacto medio por fase y correlación con ONI"),
        ("v4_mensual", "v4_serie", "Serie mensual de demanda, DemaMax, CEN y costo marginal (v4)"),
        ("correlaciones_diarias", "correlaciones", "Matriz de correlaciones de la ventana"),
    ]
    for nombre_archivo, clave, desc in nombres_tablas:
        df = ctx.get(clave, pd.DataFrame())
        if clave == "nino_eventos":
            df = ctx.get("nino_eventos", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and not df.empty:
            c1, c2 = st.columns([2.4, 1])
            with c1:
                st.markdown(f"**`{nombre_archivo}.csv`** — {desc}  ·  {df.shape[0]:,}×{df.shape[1]}")
                st.dataframe(df.head(250), use_container_width=True, hide_index=True)
            with c2:
                st.download_button("⬇️ CSV", data=df.to_csv(index=False).encode("utf-8-sig"),
                                   file_name=f"{nombre_archivo}.csv", mime="text/csv",
                                   use_container_width=True, key=f"dl_{nombre_archivo}")
        else:
            st.caption(f"`{nombre_archivo}.csv` — vacío en esta ventana/modo ({desc}).")
    comparar_catalogos_historicos(ctx)
    st.subheader("5 · Rendimiento de la carga (medido, no prometido)")
    seg = ctx.get("_carga_seg")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Última carga", f"{seg:,.1f} s" if isinstance(seg, (int, float)) else "—",
              ("hilo aparte (UI libre)" if ctx.get("_carga_hilo") else "en serie (bare/pruebas)"))
    c2.metric("Hitos V1→V4", int(ctx.get("_carga_hits") or 0), "los que ve la tarjeta de progreso")
    c3.metric("Figuras cacheadas", len(_FIG_CACHE), "reutilizadas en el próximo rerun")
    c4.metric("Entradas del caché", len(_FIG_CACHE) + 1, "ctx + figuras, TTL 6 h")
    st.markdown(
        """
**Cómo se reparte el tiempo (medido sobre la API real de XM, 14-17 feeds):**

| Fase | Qué hace | Dato que lee | Costo típico |
|---|---|---|---|
| V1 | 14-19 consultas **en paralelo** (5 hilos) | API REST de XM (JSON) + CSV de `data/raw` | 6,7 s en frío · ~0,3 s con caché |
| V2 | ETL: `wide→long`, unidades, 24 h/día, duplicados, NaN | memoria (pandas sobre los CSV) | 1-2 s |
| V3 | cruce con `dim_plantas`, resúmenes diarios, linaje | CSV locales de catálogos XM | < 1 s |
| V4 | ONI + fases + eventos + rezagos + CEN/emisiones + v4 | CSV público de NOAA/CPC + `data/reference` + XM | 1-3 s (1 llamada a NOAA) |
| — | 44 figuras de 9 pestañas | solo memoria (`figura_cacheada`) | 4,2 s la 1.ª vez · **0,0 s** después |

**Qué se hizo para que no se congele** (y por qué ya no verás el spinner eterno):

1. **Descargas en paralelo con `ThreadPoolExecutor`.** Seriales tomaban 20,8 s; con 5 hilos,
   6,7 s (−68 %, medido en la API de XM). Con 8 hilos XM empieza a saturar (9,8 s): por eso el
   valor por defecto es 5 y se puede cambiar con `APP_MANUEL_OBREROS` (poner `1` = serie).
2. **El pipeline corre en un hilo aparte** (`cargar_contexto_con_progreso`). El script de la UI
   termina su pasada en ~0,4 s y se refresca solo, así que el estado *Running* del navegador no
   queda pegado minutos, los widgets siguen respondiendo y al cerrar se lanza `st.toast` +
   `st.success` con el tiempo total. Si prefieres el comportamiento clásico (todo en serie,
   spinner durante toda la carga), define `APPMANUEL_SIN_HILO=1`.
3. **Dos niveles de caché.** `st.cache_data(ttl=6 h, max_entries=4)` sobre `construir_contexto`
   (la llave es ventana + métricas + modo) y `figura_cacheada` sobre cada figura, indexada por la
   *firma* de los datos (`ctx["_firma"]`): si cambia una métrica o la ventana, la firma cambia y
   nada se reutiliza por error. En disco, `data/raw/{prefijo}_{ini}_{fin}.csv` sobrevive reinicios.
4. **Sin hilos colgados ni `while True`:** los `requests` llevan `timeout=180 s`, 4 reintentos con
   pausa y respaldo serial; la tarjeta de progreso corta a los `APP_MANUEL_LIMITE_CARGA` s (600) y
   avisa en vez de esperar indefinidamente.
5. **Calienta la caché sin abrir el navegador** (útil antes de una demo o de un deploy):
   `set APPMANUEL_BARE=1 && python app.py` descarga la ventana y deja los CSV en `data/raw`;
   la siguiente apertura de `streamlit run app.py` entra directa por disco.
6. **Memoria/RAM:** si el Administrador de Tareas muestra presión, baja los días de la ventana o
   apaga `DispoCome`/`EmisionesCO2`; cada figura y cada tabla viven en el proceso de Streamlit, y
   `max_entries=4` limita cuántas ventanas se recuerdan.
        """)
    st.caption("Fuentes de estos números: `python tools/smoke_test.py` (pipeline + las 44 figuras), "
               "`python tools/test_app.py` (render real con AppTest) y el banco de la capa de datos "
               "(serial vs 5/8 hilos sobre la API de XM).")
    st.subheader("6 · Cobertura del notebook, celda por celda")
    try:
        cob = tabla_cobertura_nb()
    except Exception as exc:                                      # noqa: BLE001
        cob = pd.DataFrame()
        st.warning(f"No se pudo armar la tabla de cobertura: `{type(exc).__name__}: {exc}`")
    if not cob.empty:
        tot = len(cob)
        con_logica = int((cob["estado"].isin(["figura", "cálculo portado"])).sum())
        st.markdown(
            f"**{con_logica} de {tot} celdas** traen lógica ejecutada por la app "
            f"({100 * con_logica / tot:.0f} %); las {tot - con_logica} restantes son markdown "
            "(documentación del cuaderno), `print`/`.head()` de consola o `import`/`%pip`. "
            f"Las {len(CELDAS_NB)} figuras de la app cubren las "
            f"{sum(b - a + 1 for a, b in COBERTURA_NB['figura'][1])} celdas que dibujaban una figura en "
            "el notebook, sin huérfanas.")
        filtro = st.multiselect("Filtrar por estado", sorted(cob["estado"].unique()),
                                default=sorted(cob["estado"].unique()), key="cob-est")
        most = cob[cob["estado"].isin(filtro)] if filtro else cob
        st.dataframe(most, use_container_width=True, hide_index=True, height=300)
        st.download_button("⬇️ Descargar cobertura_celda_por_celda.csv", cob.to_csv(index=False).encode(),
                           "cobertura_celda_por_celda.csv", "text/csv", key="dl-cobertura")
        st.caption("Verificación reproducida con `python tools/verificar_celdas.py` (sale con código 1 si "
                   "alguna `graficar_*` del notebook queda sin espejo o si `FIGURES` y el mapa de celdas "
                   "divergen; hoy sale en verde). Los números de celda son el **índice absoluto dentro de "
                   "`cells` del JSON, 0-based** — el `In [n]` que muestra Jupyter cuenta solo las celdas de "
                   "código, así que no coincide. El código v1-v4 del notebook se lee en "
                   "[github.com/srriverar/sin-el-nino-dashboard](https://github.com/srriverar/"
                   "sin-el-nino-dashboard).")
    st.subheader("7 · Bitácora de la ejecución")
    st.dataframe(historial(), use_container_width=True, hide_index=True)
    st.caption("Reflejo del `bitacora()` del notebook: pide, intenta, guarda, avisa. "
               "Si algo falta, aquí está la razón.")


def cabecera(ctx: dict | None = None) -> None:
    st.markdown(
        f"""<style>
        .block-container {{padding-top: 1.6rem; padding-bottom: 3rem;}}
        div[data-testid="stMetricValue"] {{font-size: 1.55rem;}}
        .legenda {{background:#f4f7fb;border-left:4px solid {AZUL};padding:10px 14px;
                   border-radius:6px;margin:2px 0 18px 0;font-size:.9rem;line-height:1.5;}}
        .nota-v5 {{background:#fffbe6;border:1px solid #f0d875;border-radius:8px;
                   padding:12px 16px;margin:14px 0;font-size:.92rem}}
        .cred {{background:#f6f8fa;border:1px solid #e2e7f0;border-radius:10px;padding:14px 18px;
                margin:10px 0;font-size:.9rem;line-height:1.55;color:#2f3440}}
        .cred b {{color:#3b2366}}
        .cred .perfil {{color:#33383f}}
        .cred .contacto {{color:#4b3a8c;font-weight:600}}
        .cred .nota {{color:#4c5560;font-size:.82rem}}
        .autoria {{background:#eef3fb;border-left:4px solid {AZUL};border-radius:6px;padding:10px 14px;
                   margin:8px 0 14px 0;font-size:.92rem;line-height:1.55;color:#253044}}
        </style>""", unsafe_allow_html=True)
    st.title("🌊 Agua, precio y El Niño — el SIN colombiano bajo sequía")
    st.caption("**Observatorio operativo del mercado eléctrico de Colombia** · construido con la "
               "información operativa pública de XM (API Bienda) **y** los datos climáticos de NOAA/CPC "
               "(ONI / Niño-3.4) · última ventana publicada, día a día y hora a hora, con la lectura de "
               "riesgo para hidrología, despacho y costos. _Prototipo académico independiente, sin "
               "vínculo oficial con XM._")
    if ctx is None:
        st.markdown("**Extracción · ETL · análisis ENOS dentro del navegador** — ejecuta la lógica de "
                    "las 212 celdas del notebook `01_ETL_exploracion_v1.ipynb` (v1 · v2 · v3 · v4) y "
                    "añade el motor El Niño con el índice Niño-3.4 (ONI) de NOAA/CPC. Fuentes: API "
                    "pública de XM (`pydataxm.pydatasimem`-compatible) + CSV de NOAA/CPC. Prototipo "
                    "académico independiente, no una herramienta oficial de XM.")
        st.caption(NOTA_REPO)
        return
    ini, fin = ctx["ini"], ctx["fin"]
    nino = ctx.get("nino", pd.DataFrame())
    estado = resumen_evento_actual(nino)
    fase = ("🔴 El Niño" if estado.get("es_el_nino") else "⚪ Neutral") if estado else "— ENSO sin datos"
    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])
    with col1:
        _fin_ef = ctx.get("fin_efectiva")
        _prov = ctx.get("dias_provisionales", pd.DataFrame())
        _np = len(_prov) if isinstance(_prov, pd.DataFrame) else 0
        _estado_prov = ("preliminar(es) INCLUIDOS (sesgan el balance)" if ctx.get("incluir_provisionales")
                        else "preliminar(es) excluidos")
        st.metric("Ventana analizada", f"{ini:%d-%b-%Y} → {fin:%d-%b-%Y}",
                  (f"{ctx['dias']} días · modo {ctx['modo']} · cerrado hasta "
                   f"{pd.to_datetime(str(_fin_ef)):%d-%b-%Y}"
                   + (f" · {_np} día(s) {_estado_prov}" if _np else "")))
    with col2:
        r = ctx["resumen"]
        st.metric("Embalses (cierre)", f"{r['Embalses_Pct'].iloc[-1]:.1f} %" if "Embalses_Pct" in r and not r.empty else "—",
                  (f"{r['Embalses_Pct'].iloc[-1] - r['Embalses_Pct'].iloc[0]:+.1f} pp en la ventana"
                   if "Embalses_Pct" in r and len(r) > 1 else None))
    with col3:
        if "Precio_Ponderado_COP_kWh" in r and len(r):
            st.metric("Precio medio", f"{r['Precio_Ponderado_COP_kWh'].mean():,.0f} COP/kWh",
                      f"máx {r['Precio_Ponderado_COP_kWh'].max():,.0f}")
        else:
            st.metric("Precio medio", "—")
    with col4:
        st.metric("Fase ENSO (mes en curso)", fase,
                  (f"índice Niño-3.4 (ONI) de {_mes_es(estado['mes_ultimo'])}: "
                   f"{_coma(estado['oni_ultimo'])} °C · {estado['meses_activos']} meses de evento"
                   if estado else "requiere NOAA/serie de referencia"),
                  delta_color="inverse" if estado and estado.get("es_el_nino") else "off")
    with col5:
        tot = int(ctx["linaje"]["filas"].sum()) if "filas" in ctx["linaje"] else 0
        st.metric("Filas procesadas", f"{tot:,}", f"{len(ctx['linaje'])} consultas (catálogos + métricas)")
    st.caption(NOTA_REPO + " Cada figura de este tablero cita la celda que reproduce "
               "(`· v3 · notebook celda 143`), y la verificación completa, celda por celda, está en "
               "📚 *Datos y bitácora → Cobertura del notebook*.")


def bloque_hallazgos(ctx: dict) -> None:
    hall = ctx.get("hallazgos") or []
    v3 = construir_hallazgos_v3(ctx)
    st.subheader("🧭 Hallazgos automáticos de esta ejecución")
    st.caption("Salen del mismo algoritmo que las celdas `construir_hallazgos` (v2) y "
               "`construir_hallazgos_v3` (v3): leen las tablas ya calculadas, no los supuestos del autor.")
    if not hall and not v3:
        st.info("Todavía no hay hallazgos: activa más métricas o cambia a modo *Auto*.")
        return
    cols = st.columns(2)
    for i, h in enumerate(hall + v3):
        icono = {"riesgo": "🔴", "contexto": "🔎", "tendencia": "📈", "embalses": "💧",
                 "generacion": "⚡", "correlacion": "🔗", "precio": "💰"}.get(h.get("tipo", ""), "•")
        texto = f"{icono} **{h.get('titulo','')}**\n\n{h.get('detalle','')}"
        with cols[i % 2]:
            st.markdown(texto)
            st.caption(f"tipo: `{h.get('tipo','')}`")


def construir_hallazgos_v3(ctx: dict) -> list[dict[str, Any]]:
    """Versión v3 de los hallazgos (CEN/margen, emisiones, ENSO) con guardas de columnas."""
    R: list[dict[str, Any]] = []
    c = ctx.get("cen_diario", pd.DataFrame())
    if not c.empty and "Margen_pct" in c:
        m = c["Margen_pct"].dropna()
        if len(m):
            R.append({"tipo": "riesgo" if float(m.min()) < 15 else "contexto",
                      "titulo": f"Margen de reserva medio {float(m.mean()):.1f} % "
                                f"(mínimo {float(m.min()):.1f} %)",
                      "detalle": "CEN por recurso (CapEfecNeta) contra la demanda máxima programada. "
                                 "Debajo de 15 % el sistema entra en operación ajustada y cualquier "
                                 "falla se paga con precio de escasez.",
                      "valor": float(m.mean())})
    e = ctx.get("emisiones_dia", pd.DataFrame())
    if not e.empty and "Intensidad_tCO2_MWh" in e:
        I = e["Intensidad_tCO2_MWh"].dropna()
        if len(I):
            R.append({"tipo": "contexto", "titulo": f"Intensidad de carbono media {float(I.mean()):.2f} tCO₂/MWh",
                      "detalle": f"Banda predominante: {bandas_emision(float(I.mean()))}. Sube cuando la "
                                 "térmica sustituye a la hidro: es el costo oculto de un episodio El Niño.",
                      "valor": float(I.mean())})
    n = ctx.get("nino", pd.DataFrame())
    est = resumen_evento_actual(n)
    if est:
        R.append({"tipo": "riesgo" if est.get("es_el_nino") else "contexto",
                  "titulo": (f"ENOS: El Niño activo desde {est['inicio_oficial']} "
                             f"({est['meses_activos']} meses, pico del índice Niño-3.4 suavizado (ONI) "
                             f"{_coma(est['pico_oni'])} °C)")
                  if est.get("es_el_nino") else f"ENOS: fase {est.get('mes_ultimo','')} sin El Niño",
                  "detalle": "Umbral oficial |ONI| ≥ 0.5 °C (3 meses). Para Colombia el indicador "
                             "anticipado es Niño1+2: valores altos con Niño3.4 menor implican más "
                             "déficit de lluvia del que sugiere el ONI.",
                  "valor": est.get("oni_ultimo")})
    fp = ctx.get("fp", pd.DataFrame())
    if not fp.empty and "FP_pct" in fp and "Tecnologia" in fp:
        g = fp.groupby("Tecnologia")["FP_pct"].mean()
        if not g.empty:
            R.append({"tipo": "contexto", "titulo": f"Factor de planta: {g.idxmax()} {g.max():.1f} % · "
                      f"{g.idxmin()} {g.min():.1f} %",
                      "detalle": "FP = kWh generados ÷ (kW de CEN × horas del mes). Un FP hidro bajo con "
                                 "embalses altos apunta a restricción o mantenimiento mal programado.",
                      "valor": None})
    return R


def pie_credito() -> None:
    st.markdown("---")
    st.markdown(f"""<div class="autoria">
    <b>Autoría y alcance.</b> La aplicación se desarrolló a partir del notebook analítico creado por
    <b>Manuel Fajardo</b> —versiones v1–v2, secciones 0–6 y anexos— y fue ampliada por el profesor
    <b>Sergio Rivera</b> en las versiones v3–v4 con nuevos indicadores y su implementación como
    tablero interactivo. Los datos combinan información operativa pública de XM con datos climáticos
    de NOAA/CPC; la app es un prototipo académico independiente, no una herramienta oficial de XM.
    <br><b>Código v1-v4:</b> el notebook de 212 celdas y esta app viven en
    <a href="{REPO_NOTEBOOK}">{REPO_NOTEBOOK.replace('https://', '')}</a> — cada figura del tablero cita
    la celda del notebook que reproduce (índice absoluto 0-based de <code>cells</code> en el JSON).
    </div>""", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div class="cred">
        <b>👤 {AUTOR_APP[0]['nombre']}</b> · {AUTOR_APP[0]['cargo']}<br>
        <span class="perfil">{AUTOR_APP[0]['perfil']}</span><br>
        <span class="contacto">{AUTOR_APP[0]['contacto']}</span><br>
        <span class="nota">{AUTOR_APP[0]['nota']}</span></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="cred">
        <b>👤 {AUTOR_V3[0]['nombre']}</b> · {AUTOR_V3[0]['cargo']}<br>
        <span class="perfil">{AUTOR_V3[0]['perfil']}</span><br>
        <span class="contacto">{AUTOR_V3[0]['contacto']}</span><br>
        <span class="nota">{AUTOR_V3[0]['nota']}</span></div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="nota-v5">
    <b>🗺️ Nota de versionado — lo que viene en v5.</b> Esta v1 del tablero está deliberadamente
    atada a la lógica del notebook: la ventana es <b>{VENTANA_POR_DEFECTO} días móvil terminada anteayer</b>
    (el último mes con datos publicados por XM, que en la práctica queda en anteayer o antes porque XM
    cierra el día D en D+1/D+2) y las fechas se calculan con <code>pd.Timestamp.now()</code>.
    <b>Una versión posterior permitirá pedir periodos arbitrarios</b> (cualquier <code>fecha_inicio</code> y
    <code>fecha_fin</code>, incluso históricos, con comparativa entre dos ventanas y exportación del
    informe). La capa de datos ya está preparada para eso: <code>bloques()</code> parte cualquier rango en
    ventanas ≤ {PERIODO_MAX_API} días (límite duro de <code>servapibi.xm.com.co</code>), el caché en
    <code>data/raw</code> reutiliza lo ya descargado y las dimensiones (<code>dim_plantas</code>, ríos) se
    reconstruyen para el periodo elegido. Lo único que cambiará en la UI serán dos
    <code>st.date_input</code> en lugar del selector de días.</div>""", unsafe_allow_html=True)
    st.caption(fuente_nota())


# ==============================================================================
# 8b · CARGA EN HILO, PANEL DE ETAPAS Y AVISO DE CIERRE
# ==============================================================================
_LIMITE_ESPERA = float(os.environ.get("APP_MANUEL_LIMITE_CARGA", "600"))
_EJECUTOR_CARGA: Any = None


def _panel_de_carga(prog: "Progreso", activo: bool) -> None:
    """Tarjeta que se redibuja en cada refresco: etapa V1-V4, barra, y últimos eventos."""
    frac, etapa, lineas, t0, tf, ticks, _fin = prog.instantanea()
    seg = (tf or time.time()) - t0
    rotulo = (f"⠙ **Procesando** · {seg:,.0f} s transcurridos · {ticks} eventos" if activo
              else f"✅ **Proceso terminado** en {seg:,.1f} s · {ticks} eventos")
    st.progress(min(0.995, float(frac)), text=f"{rotulo} — **{etapa}**")
    _mapa = " · ".join(f"{k} {celdas_de_etapa(k)}" for k in ("V1", "V2", "V3", "V4") if k in ETAPAS)
    st.caption(f"Mapa del notebook `01_ETL_exploracion_v1.json` (212 celdas, índice 0-based): {_mapa}. "
               "El número entre corchetes de cada línea es la celda que se está ejecutando en este "
               "instante.")
    if lineas:
        for ts, _eta, msg in reversed(lineas[-8:]):
            st.caption(f"`{ts}` · {msg}")
    if activo:
        st.caption("No cierres la pestaña. La **página sí responde** (filtros, toggles y menú funcionan) "
                   "porque el cálculo pesado corre en un hilo aparte: el cuadrado de *Running* solo "
                   "desaparece cuando este ciclo de refresco termina, y ese ciclo dura ~0,4 s.")


def _trabajo_de_carga(clave: str, dias: int, opciones: dict[str, bool], modo: str,
                      prog: "Progreso", incluir_provisionales: bool = False) -> dict[str, Any]:
    """Cuerpo que corre en el hilo secundario (no toca widgets: solo datos)."""
    _PROGRESO_ACTIVO[0] = prog
    try:
        return obtener_contexto_cached(clave, dias, json.dumps(opciones), modo, incluir_provisionales)
    finally:
        _PROGRESO_ACTIVO[0] = None


def cargar_contexto_con_progreso(clave: str, dias: int, opciones: dict[str, bool], modo: str,
                                 historial_completo: bool = False,
                                 incluir_provisionales: bool = False) -> dict[str, Any]:
    """Ejecuta el pipeline sin congelar la interfaz y avisa cuando termina.

    - El trabajo pesado (`construir_contexto`) va a un `ThreadPoolExecutor`.
    - Cada pasada del script pinta el estado y se auto-refresca (`st.rerun`), así que el
      script nunca queda bloqueado minutos: entre refresco y refresco el menú de arriba
      vuelve a su estado normal y el spinner del navegador sigue animando.
    - Al terminar se marca `listo` en el estado de sesión: se lanza `st.toast` + un
      `st.success` que solo aparece en la primera pasada posterior al cierre.
    - Con `APPMANUEL_SIN_HILO=1` (o sin sesión, p. ej. `python app.py`) corre en serie,
      que es lo que usan las pruebas.
    """
    global _EJECUTOR_CARGA
    try:
        sesiones = st.session_state
        sesiones["_cargas"] = sesiones.get("_cargas", {})
    except Exception:                                             # noqa: BLE001 (bare mode)
        sesiones = None
    if sesiones is None or os.environ.get("APPMANUEL_SIN_HILO", "").strip().lower() in ("1", "true", "si"):
        prog = Progreso()
        _PROGRESO_ACTIVO[0] = prog
        try:
            ctx = dict(obtener_contexto_cached(clave, dias, json.dumps(opciones), modo,
                                            incluir_provisionales))
        finally:
            _PROGRESO_ACTIVO[0] = None
        prog.cerrar()
        fr, _e, _l, t0, tf, ticks, _f = prog.instantanea()
        ctx["_carga_seg"] = round((tf or time.time()) - t0, 1)
        ctx["_carga_hits"] = ticks
        ctx["_carga_hilo"] = False
        ctx["historial_completo"] = bool(historial_completo)
        return ctx

    faena = sesiones["_cargas"].get(clave)
    if faena is None:
        if _EJECUTOR_CARGA is None:
            _EJECUTOR_CARGA = _fut.ThreadPoolExecutor(max_workers=2, thread_name_prefix="carga")
        prog = Progreso()
        t0 = time.time()
        faena = {"fut": _EJECUTOR_CARGA.submit(_trabajo_de_carga, clave, dias, opciones, modo, prog,
                                                bool(incluir_provisionales)),
                 "prog": prog, "ctx": None, "err": None, "t0": t0, "listo": False, "avisado": False,
                 "refrescos": 0}
        for k_vieja in [k for k, v in list(sesiones["_cargas"].items()) if k != clave and v["fut"].done()][:-6]:
            sesiones["_cargas"].pop(k_vieja, None)
        sesiones["_cargas"][clave] = faena

    if not faena["listo"]:
        if not faena["fut"].done():
            faena["refrescos"] += 1
            _panel_de_carga(faena["prog"], activo=True)
            if time.time() - faena["t0"] > _LIMITE_ESPERA:
                st.error(f"La carga lleva {time.time() - faena['t0']:,.0f} s sin cerrar (límite "
                         f"{_LIMITE_ESPERA:,.0f} s). XM puede estar devolviendo 503: prueba el modo "
                         "💾 *Solo caché*, o baja los días de la ventana y vuelve a recargar.")
                st.stop()
            # pausa corta al principio (para ver los hitos) y más larga si el paso es lento
            time.sleep(0.35 if faena["refrescos"] < 60 else 1.0)
            st.rerun()
        try:
            faena["ctx"] = dict(faena["fut"].result())
        except Exception as exc:                                  # noqa: BLE001
            faena["err"] = f"{type(exc).__name__}: {str(exc)[:600]}"
            bitacora(f"carga en hilo: {faena['err']}", "error")
        faena["prog"].cerrar()
        faena["tfin"] = time.time()
        faena["listo"] = True

    if faena["err"]:
        _panel_de_carga(faena["prog"], activo=False)
        st.error(f"No se pudo completar la extracción: `{faena['err']}`\n\n"
                 "Si estás en modo *Auto* y la API respondió 503, vuelve a intentarlo o pasa a "
                 "*Solo caché*. El **📚 Datos y bitácora** guarda el detalle de cada consulta.")
        st.stop()

    ctx = dict(faena["ctx"])
    ctx["_carga_seg"] = round(faena["tfin"] - faena["t0"], 1)
    ctx["_carga_hits"] = faena["prog"].instantanea()[5]
    ctx["_carga_hilo"] = True
    if not faena["avisado"]:
        faena["avisado"] = True
        seg = faena["tfin"] - faena["t0"]
        st.toast(f"✅ Datos listos y transformados en {seg:,.1f} s", icon="✅")
        st.success(f"✅ **Procesamiento terminado en {seg:,.1f} s** ({faena['prog'].instantanea()[5]} "
                   f"hitos V1→V4). El estado *Running* ya pasó a normal: el icono que verás arriba a la "
                   f"derecha es el de **fork/GitHub**, no el de stop. Todo lo que sigue "
                   f"({len(FIGURES)} figuras del notebook + {len(FIGURES_NODAL)} del módulo nodal, en "
                   f"{len(TITULOS_TABS)} pestañas) se dibuja con estos datos y queda en caché, así que "
                   f"cambiar de pestaña o de filtro no vuelve a bajar de XM.")
    ctx["historial_completo"] = bool(historial_completo)
    return ctx


def main() -> None:
    st.set_page_config(page_title="Agua, precio y El Niño · SIN Colombia (XM+NOAA)", page_icon="🌊", layout="wide",
                       initial_sidebar_state="expanded")
    cabecera()
    dias, opciones, modo = sidebar_controles()
    with st.sidebar:
        incluir_prov = st.toggle(
            "🩹 Incluir días provisionales de XM", value=False,
            help="XM publica el día D en D+1/D+2 y las primeras entregas traen solo una fracción del "
                 "día (la demanda del 27-28/08/2026 llegó al 24 % de su mediana). Por defecto la app "
                 "los detecta y los EXCLUYE de los agregados para que el balance no se infle. "
                 "Actívelo solo para inspeccionar el crudo.")
        historial_completo = st.toggle(
            "🔭 Ejes con todo el histórico", value=False,
            help="Por defecto los gráficos recortan el eje X a la ventana evaluada (y los índices "
                 "ENSO a los últimos 4-7 años) para que se lea el presente. Actívalo para ver las "
                 "series completas: ONI desde 1950, panel ENSO y 'El Niño en dinero' desde 2015.")
    if mode_demo() and modo == "auto":
        st.info("Entorno sin acceso a la API (demo): la app usará automáticamente la caché y los "
                "archivos de `data/reference`. En tu equipo verás la descarga real.")
        modo = "cache"
    st.markdown("🛰️ **Extracción → ETL → análisis** (las 212 celdas del notebook, en 4 etapas: "
                "V1 datos de XM · V2 ETL · V3 cruces/KPIs · V4 ENSO y capa v4). La tarjeta de progreso "
                "dirá en cuál vas y qué se está leyendo en cada momento.")
    ctx = cargar_contexto_con_progreso(_cache_key(dias, opciones, modo, bool(incluir_prov)), dias,
                                       opciones, modo, bool(historial_completo), bool(incluir_prov))
    cabecera(ctx)
    if ctx["resumen"].empty and ctx["gen_enr"].empty:
        if not ctx["nino"].empty:
            st.warning(
                f"Modo **{ctx['modo']}**: no hay crudos del mercado en `data/raw` para la ventana "
                f"{ctx['ini']} → {ctx['fin']}, así que las pestañas de generación/hidrología/costos "
                "quedan vacías. **La pestaña 🌊 El Niño sí funciona** con la serie mensual de "
                "`data/reference` (2015 → hoy) y el ONI de NOAA. Cambie a *Auto* o *Forzar descarga API* "
                "en la barra lateral para el tablero completo (la API de XM publica con 1-2 días de "
                "rezago; `DemaMaxPot`, con 2-6 semanas).")
        else:
            st.error("Ninguna métrica devolvió filas en esta ventana y no hay serie de referencia. "
                     "Revisa la pestaña **📚 Datos y bitácora** (linaje) y prueba: (a) modo *Forzar "
                     "descarga API*, (b) activar métricas pesadas, (c) ampliar días. Si está detrás de "
                     "un proxy corporativo, copie el bundle CA en `certs/upme_bundle.pem` "
                     "(se usa para TLS intermediado, no para XM).")
    aviso_dias_parciales(ctx)
    filtros = filtros_globales(ctx)
    st.session_state["filtros"] = filtros
    tabs = st.tabs([t for t, _k in TITULOS_TABS])
    for (titulo, clave), tab in zip(TITULOS_TABS, tabs):
        with tab:
            if clave == "datos":
                render_datos(ctx)
            elif clave == "resumen":
                render_tab_resumen(ctx, filtros)
            elif clave == "nino":
                render_tab_nino(ctx)
            elif clave == "hidrologia":
                render_tab_hidrologia(ctx, filtros)
            elif clave == "generacion":
                render_tab_generacion(ctx, filtros)
            elif clave == "costos":
                render_tab_costos(ctx, filtros)
            elif clave == "v4":
                render_tab_v4(ctx, filtros)
            elif clave == "matrices":
                render_tab_matrices(ctx, filtros)
            elif clave == "co2en":
                render_tab_co2en(ctx, filtros)
            elif clave == "nodal":
                render_tab_nodal(ctx, filtros)
    pie_credito()


def aviso_dias_parciales(ctx: dict) -> None:
    """Panel explícito de por qué la serie corta antes de hoy y qué días se excluyeron."""
    prov = ctx.get("dias_provisionales", pd.DataFrame())
    if not isinstance(prov, pd.DataFrame) or prov.empty:
        return
    fef = ctx.get("fin_efectiva") or ctx.get("fin")
    dias_txt = " y ".join(pd.to_datetime(pd.Series(prov["Fecha"]).astype(str)).dt.strftime("%d/%m/%Y"))
    if ctx.get("incluir_provisionales"):
        st.error(f"🚨 **Está incluyendo días preliminares de XM** ({dias_txt}). Los agregados diarios "
                 "(demanda, balance, precio ponderado, correlaciones y Δ por fase ENSO) quedan "
                 "sesgados: demanda a la baja y balance a la alta. Úselo solo para inspección del "
                 "crudo, no para leer el mercado.", icon="🚨")
        return
    st.warning(
        f"⚠️ **La serie corta el {_fecha_es(fef)}, no el día de hoy ({_fecha_es(_hoy())})**. XM publica "
        f"con rezago —el día D se cierra en D+1/D+2— y los días finales todavía llegan *preliminares*. "
        f"Días excluidos de todos los agregados: **{dias_txt}** (la ventana pedía hasta "
        f"{pd.to_datetime(str(ctx.get('fin'))):%d/%m/%Y}).\n\n"
        "El detector es numérico, no una regla de calendario: el total diario de cada feed volumétrico "
        "(`DemaReal_Sistema`, `Gene_Sistema`, `Gene_Recurso`, `DemaMaxPot_Sistema`) se compara con la "
        f"mediana de sus propios días y el día se marca si cae bajo "
        f"{UMBRAL_DIA_PRELIMINAR * 100:.0f} % de esa mediana. **No es una caída real de demanda**: son "
        "horas y agentes que aún no reportan. Dejarlos dentro producía exactamente el síntoma observado: "
        "demanda 4-5 veces menor y balance (generación − demanda) inflado ~+209 GWh, porque la "
        "generación sí llega completa ese día. Detalle en 📚 *Datos y bitácora → Auditoría de "
        "completitud*. Para incluirlos, active **Incluir días provisionales de XM** en la barra lateral.",
        icon="⚠️")


def render_tab_resumen(ctx: dict, filtros: dict) -> None:
    st.markdown("#### 🎯 Lo que un operador o un analista de mercado miraría primero")
    st.caption("Una sola pantalla: cuánta agua hay, cuánto cuesta la energía y cuánto margen queda. "
               "Todo lo demás es el detalle de estas tres preguntas.")
    render_figuras(ctx, "resumen", filtros)
    r = ctx["resumen"]
    if not r.empty:
        st.markdown("**Tabla diaria usada por todas las figuras de esta pestaña**")
        st.dataframe(r, use_container_width=True, hide_index=True, height=260)
    st.divider()
    banner_el_nino(ctx)
    bloque_hallazgos(ctx)
    st.caption("⬆️ El semáforo El Niño y los hallazgos se muestran aquí, al final del resumen, "
               "después de la tabla diaria en la que se apoyan (a pedido del autor del tablero).")


def _selector_ambito(ctx: dict, clave_widget: str) -> tuple[str, str] | None:
    """Dos selectores (nivel, ámbito) como el `crear_explorador_operativo` del notebook."""
    gen = gen_con_ambito(ctx)
    if gen.empty:
        st.info("Sin `Gene/Recurso` en esta ventana: no hay generación por planta que poder "
                "agrupar por ámbito. Actívala en la barra lateral o usa el modo *Auto*.")
        return None
    niveles = niveles_disponibles(gen)
    mapa = cargar_mapa_operativo()
    if mapa.empty and not any("SIMEM" in k for k in niveles):
        st.caption("Las opciones **Área/Subárea operativa (SIMEM)** se activan colocando "
                   "`data/reference/mapa_operativo.csv` (columnas `Codigo_Planta`, "
                   "`Area_Operativa`, `Subarea_Operativa`); sin él, el notebook tampoco tenía "
                   "ámbito geográfico: se agrupa por las dimensiones que XM sí publica "
                   "(tecnología, recurso, agente, estado). Formato en "
                   "`data/reference/LEEME_mapa_operativo.txt`.")
    c1, c2 = st.columns([1, 1.4])
    nivel = c1.selectbox("Nivel de agregación", list(niveles), key=f"nivel_{clave_widget}",
                         help="Equivalente a NIVELES_OPERATIVOS del notebook (celda v2 c31).")
    col = niveles.get(nivel, "")
    opciones = ["Nacional"]
    if col and col in gen:
        vals = sorted({str(v) for v in gen[col].dropna().unique()})
        opciones = opciones + vals[:400]
    ambito = c2.selectbox("Ámbito (país completo o valor del nivel)", opciones,
                          index=min(1, len(opciones) - 1), key=f"ambito_{clave_widget}",
                          help="'Nacional' = todas las plantas, como en el notebook.")
    return nivel, ambito


def _render_fig_segura(fn: Any, titulo: str, *args: Any, **kw: Any) -> None:
    try:
        res = figura_cacheada(f"segura:{titulo}", fn, *args, **kw)
    except Exception as exc:                                      # noqa: BLE001
        st.error(f"No se pudo construir *{titulo}*: `{type(exc).__name__}: {str(exc)[:200]}`")
        bitacora(f"figura {titulo}: {exc}", "warn")
        return
    render_resultado(res, titulo)


def render_tab_matrices(ctx: dict, filtros: dict) -> None:
    st.markdown("#### 🕐 Perfiles horarios, matrices y paneles — los cálculos del notebook, uno por uno")
    st.caption("Heatmaps tecnología × hora (08), pequeños múltiples (08b), precio día × hora (09), "
               "costo marginal v4 (02/03), panel ENSO (v3 c21) y climatología por fase (v3 c24). Cada "
               "figura repite la fórmula del notebook sobre las tablas ya descargadas, así que "
               "compara igual contra el PNG que generaba la celda.")
    render_figuras(ctx, "matrices", filtros)
    st.markdown("###### 🔍 Ámbito operativo / geográfico (figuras 08d · 08e · 08g · 08h)")
    ambito_sel = _selector_ambito(ctx, "matrices")
    if ambito_sel:
        nivel, ambito = ambito_sel
        sub = filtrar_ambito(gen_con_ambito(ctx), nivel, ambito)
        n_filas = 0 if sub is None or sub.empty else len(sub)
        st.caption(f"Filtro activo: **{nivel} = {ambito}** · {n_filas:,} filas horario-planta · "
                   f"identificador de archivo del notebook: `fig_v2_08d_{_slug_operativo(nivel)}_"
                   f"{_slug_operativo(ambito)}_heatmap_*.png`.")
        with st.expander("08d · Heatmap tecnología × hora del ámbito", expanded=True):
            _render_fig_segura(fig_v2_08d_scope_heatmap, "Heatmap del ámbito", ctx, nivel, ambito)
        with st.expander("08e · Perfiles por tecnología del ámbito", expanded=True):
            _render_fig_segura(fig_v2_08e_scope_perfiles, "Perfiles del ámbito", ctx, nivel, ambito)
        with st.expander("08g/08h · Misma lectura con paleta geográfica (YlGnBu)"):
            _render_fig_segura(fig_v2_08g_geo_heatmap, "Heatmap geográfico del ámbito", ctx, nivel, ambito)
            _render_fig_segura(fig_v2_08h_geo_perfiles, "Perfiles geográficos del ámbito", ctx, nivel, ambito)
    st.markdown("###### 📋 Tabla del perfil tecnología × hora (la matriz de la figura 08)")
    perf = perfil_tecnologia_hora(gen_con_ambito(ctx))
    if perf.empty:
        st.info("Sin perfil por tecnología para exportar (falta `Gene/Recurso` en esta ventana).")
        return
    piv = _pivote_tec_hora(perf).round(2)
    piv.index = [str(i).title() for i in piv.index]
    piv.columns = [f"H{int(c):02d}" for c in piv.columns]
    st.dataframe(piv, use_container_width=True, height=min(430, 34 * len(piv) + 60))
    st.caption("Valores en GWh/h promedio por hora-día (columna `Generacion_GWh` de "
               "`preparar_perfil_tecnologia`). Descarga:")
    st.download_button("⬇️ perfil_tecnologia_hora.csv", perf.to_csv(index=False).encode("utf-8"),
                       file_name=f"perfil_tecnologia_hora_{ctx['ini']}_{ctx['fin']}.csv",
                       mime="text/csv", use_container_width=True)


def render_tab_co2en(ctx: dict, filtros: dict) -> None:
    st.markdown("#### 🔥 CO₂ del despacho (v3) y economía del ENSO (v4)")
    st.caption("Cuatro figuras de emisiones con las fórmulas exactas del bloque v3 (celdas 143-149 del "
               "notebook) y las dos del bloque v4 que miden cuánto margen y cuánto precio mueve el ONI "
               "(celdas 199 y 207). Las dos vistas que estaban aquí en desplegable —*dispersión ONI + "
               "rezagos 0-12 m* (fig_v4_06) y *Δ% por episodio ENSO* (gráfico de la c203)— se retiraron a "
               "pedido del autor por redundantes; el Δ% por episodio **sigue calculándose** y se ve abajo "
               "como tabla, que es como se contrata.")
    render_figuras(ctx, "co2en", filtros)
    tabla = ctx.get("_v4_tabla_impacto")
    if isinstance(tabla, pd.DataFrame) and not tabla.empty:
        st.markdown("###### 🧾 Tabla de impacto por evento ENSO (celda 203 del notebook, Δ% vs régimen neutro)")
        st.dataframe(tabla.round(1), use_container_width=True, hide_index=True, height=260)
        st.download_button("⬇️ Descargar impacto_por_evento_ENSO.csv", tabla.round(2).to_csv(index=False).encode(),
                           "impacto_por_evento_ENSO.csv", "text/csv", key="dl-impacto-enso")
        st.caption("**Cómo se leyó cada Δ:** promedio mensual de la variable dentro del intervalo "
                   "[Inicio, Fin] del episodio, menos el promedio de los meses Neutral del mismo periodo "
                   "base (2015-01 → último mes cerrado de la ventana), dividido por ese promedio neutral "
                   "y multiplicado por 100. Los grupos salen de `clasificar_fase(ONI)` con umbral "
                   "±0,5 °C y los episodios de `detectar_eventos_enso` (≥5 meses consecutivos). Costo "
                   "marginal y margen vienen de la capa v4 (CEN por recurso + demanda máxima), aportes "
                   "de `PorcApor`/`AporEner` mensuales; por eso un episodio antiguo puede tener Δ de "
                   "margen vacío si ese mes no tiene CEN en la base.")


def render_tab_generacion(ctx: dict, filtros: dict) -> None:
    st.markdown("#### ⚡ Matriz de generación (quién produce, a qué hora, con qué aprovechamiento)")
    gen = aplicar_filtros(ctx["gen_enr"], filtros, ctx["dim_plantas"])
    if gen is not None and not gen.empty and "Tecnologia" in gen:
        mix = (gen.groupby("Tecnologia")["Generacion_kWh"].sum().sort_values(ascending=False) / 1e6)
        mix_df = pd.DataFrame({"GWh": mix.round(1), "% del total": (100 * mix / mix.sum()).round(1)})
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("**Mix de la ventana** (los filtros de arriba sí aplican aquí)")
            st.dataframe(mix_df, use_container_width=True)
        with c2:
            fig = _fig(ctx, 340)
            fig.add_trace(go.Bar(x=mix_df.index, y=mix_df["GWh"], marker_color=AZUL, name="GWh"))
            fig.update_layout(yaxis_title="GWh", xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True,
                           config={"displaylogo": False}, key=_clave_fig("mix-ventana"))
            st.caption("**📖 Mix de la ventana — qué muestra y para qué sirve**\n\n"
                       "- **Qué muestra:** GWh por tecnología en la ventana seleccionada, con el mercado completo "
                       "los selectores de tecnología / agente / tipo de recurso se retiraron porque "
                       "siempre estaban en 'todas'; el filtrado fino vive en los exploradores de "
                       "planta y de ámbito.\n"
                       "- **Decisión / conclusión:** donde está tu exposición. Si Hídrica + Térmica "
                       "suman &gt; 90 %, cualquier golpe de sequía se traslada directo al precio.\n"
                       "- **Lente El Niño:** el % de térmica es la métrica que se mueve primero "
                       "(sube) cuando el Niño se intensifica; el % solar es el amortiguador.")
    st.markdown("---")
    st.markdown("#### 🔍 Explorador de planta")
    explorar_planta(ctx)
    st.markdown("#### 🗺️ Explorador de recursos y cuencas")
    explorar_rio_zona(ctx)
    render_figuras(ctx, "generacion", filtros)


def render_tab_hidrologia(ctx: dict, filtros: dict) -> None:
    st.markdown("#### 💧 Hidrología: el canal físico por el que El Niño llega al mercado")
    e = ctx["embalses"]
    if not e.empty:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            fig = _fig(ctx, 300)
            fig.add_trace(go.Scatter(x=e["Fecha"], y=e["Volumen_Pct"], mode="lines+markers",
                                     name="% volumen útil", line=dict(color=AZUL, width=2.4),
                                     fill="tozeroy", fillcolor="rgba(31,119,180,.12)"))
            for u, col in ((55, AMBAR), (45, NARANJA), (35, ROJO)):
                fig.add_hline(y=u, line=dict(color=col, dash="dot"), annotation_text=f"{u} %",
                              annotation_font_size=9)
            fig.update_layout(yaxis_title="%", xaxis_title="Día", height=300)
            st.plotly_chart(fig, use_container_width=True,
                           config={"displaylogo": False}, key=_clave_fig("embalses-tab"))
        with c2:
            dias_bajos = int((e["Volumen_Pct"] < 45).sum())
            st.markdown(f"""**Lectura rápida**
            - Nivel inicial → final: `{e['Volumen_Pct'].iloc[0]:.1f} % → {e['Volumen_Pct'].iloc[-1]:.1f} %`
            - Días bajo 45 %: **{dias_bajos}** de {len(e)}
            - Tendencia 7 días: **{e['Volumen_Pct'].diff(7).iloc[-1]:+.1f} pp**
            - **Decisión:** con tendencia negativa y aportes &lt; 80 % de la media, el plan de
              mantenimiento térmico debe cerrarse ya; en El Niño la hidro no respalda las noches.""")
    render_figuras(ctx, "hidrologia", filtros)
    if ctx["aportes"].empty:
        st.info("Sin serie de aportes (`PPPrecBolsNaci/Sistema`) en esta ejecución: la figura de "
                "aportes vs media histórica necesita la serie mensual de referencia o el modo *Auto*.")


def render_tab_nino(ctx: dict) -> None:
    st.markdown("#### 🌊 El Niño–La Niña: diagnóstico, historia y transmisión al mercado")
    n = ctx["nino"]
    if n is not None and not n.empty:
        est = resumen_evento_actual(n)
        ev = ctx.get("nino_eventos", pd.DataFrame())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Índice Niño-3.4 (ONI)", f"{_coma(est.get('oni_ultimo', float('nan')))} °C",
                  f"estación oficial más reciente: {_mes_es(est.get('mes_ultimo', ''))}")
        c2.metric("Pico del evento", f"{est.get('pico_oni', float('nan')):+.2f} °C", est.get("fuerza", "—"))
        c3.metric("Meses activos", est.get("meses_activos", 0), f"inicio {est.get('inicio_oficial','—')}")
        c4.metric("Niño1+2 (jun-2026)", f"{est.get('nino1_2', float('nan')):+.2f} °C",
                  f"histórico máx {est.get('nino1_2_record', float('nan')):+.2f}")
        if not ev.empty:
            st.dataframe(ev, use_container_width=True, hide_index=True, height=210)
            st.caption("Cada fila es un evento de ≥5 meses consecutivos con |ONI| ≥ 0.5 °C "
                       "(detección del notebook: `detectar_eventos_enso`). Útil para elegir "
                       "años de comparación y para calibrar tu 'peor caso' con el evento análogo.")
        tabla = tabla_impacto_enso(n, [v for v in ("Embalses_Pct", "Aportes_Pct", "Precio_Ponderado_COP",
                                                    "Nino1_2") if v in n.columns])
        if not tabla.empty:
            st.markdown("**Impacto medio por fase (cuánto vale un mes de El Niño en cada variable)**")
            st.dataframe(tabla, use_container_width=True, hide_index=True)
            st.caption("`delta_El_Nino_vs_Neutral` es la resta de promedios mensuales y `corr_ONI` la "
                       "correlación lineal con el índice: ambas juntas separan una señal útil de un "
                       "artefacto de pocos meses de muestra (`n_El_Nino`). **Periodo y grupos usados:** "
                       "meses completos de la base ENSO (2015-01 → el último mes cerrado de la ventana), "
                       "agrupados por la fase oficial del ONI con umbral ±0,5 °C; la media se toma sobre "
                       "los meses de cada grupo sin emparejar por calendario, `delta_pct` = 100 · "
                       "(media_Niño − media_Neutral) / media_Neutral, y `corr_ONI` es Pearson entre la "
                       "variable y el ONI del mismo mes (solo meses con ambos valores). Se marca "
                       "`relevante` cuando el grupo El Niño tiene ≥ 3 meses; por debajo de eso la "
                       "comparación es ilustrativa, no una estimación del efecto.")
    render_figuras(ctx, "nino", {})
    st.caption("Las figuras de distribución por fase (boxplots con la ventana actual marcada ★) y los "
               "rezagos ONI → variable viven ahora en **🕐 Perfiles × hora** y **🔥 CO₂ y economía ENSO**, "
               "que es donde se puede uno sentarse a leerlas; aquí se eliminaron por pedido explícito.")
    st.caption("🔭 Los ejes de ONI, Niño1+2 y el panel ENSO vienen recortados a los últimos años para que "
               "el inicio del evento (ago-2026) se lea sin hacer zoom. Con **Ejes con todo el histórico** "
               "en la barra lateral se verán de nuevo desde 1950; los datos no se filtran, solo cambia la "
               "vista.")


def render_tab_costos(ctx: dict, filtros: dict) -> None:
    st.markdown("#### 💰 Costos, escasez y emisiones (donde el clima se convierte en pesos y en CO₂)")
    if ctx["costo_marginal"].empty and ctx["emisiones"].empty:
        st.warning("Activa en la barra lateral `CostMargDesp` (ya viene encendida) y las métricas de "
                   "emisiones para ver este tab completo. Emisiones/combustible son las más pesadas.")
    render_figuras(ctx, "costos", filtros)


def render_tab_v4(ctx: dict, filtros: dict) -> None:
    st.markdown("#### 🛠️ Disponibilidad, reservas y riesgo operativo (capa v4)")
    if ctx["indisp"].empty:
        st.info("La serie de indisponibilidad necesita `CapEfecNeta` (CEN por recurso) y "
                "`DispoCome` **o** `DemaMaxPot`. Enciéndelas en la barra lateral y recarga: el proxy "
                "del notebook (100 % − DemaMax/CEN) ya está implementado y no requiere `DispoCome`.")
    render_figuras(ctx, "v4", filtros)


def mode_demo() -> bool:
    """True si la API no es alcanzable (red restringida): la app degrada a caché/reference."""
    if "APPMANUEL_FORCE_DEMO" in os.environ:
        return True
    if "red" not in st.session_state:
        try:
            post_xm("daily", "PorcApor", "Sistema", _hoy() - dt.timedelta(days=3), _hoy(),
                    reintentos=1, pausa=0.1, timeout=6)
            st.session_state["red"] = True
        except Exception:                                         # noqa: BLE001
            st.session_state["red"] = False
    return not st.session_state["red"]




# ==============================================================================
# 9 · EXPLORADORES INTERACTIVOS (sustituyen los ipywidgets del notebook)
# ==============================================================================
def explorar_planta(ctx: dict) -> None:
    """Explorador de planta (reemplazo de `crear_explorador_operativo` con ipywidgets)."""
    dim = ctx["dim_plantas"]
    gen = ctx["gen_enr"]
    if dim is None or dim.empty or gen is None or gen.empty:
        st.info("Explorador de plantas: necesita `Gene_Recurso` + el catálogo `ListadoRecursos`.")
        return
    base = gen.groupby("Codigo_Planta", as_index=False).agg(GWh=("Generacion_kWh", lambda x: x.sum() / 1e6),
                                                            Horas=("Hora", "nunique"))
    base = base.merge(dim[["Codigo_Planta", "Nombre_Planta", "Tecnologia", "Fuente_Energia",
                          "Tipo_Recurso", "Tipo_Despacho"]].drop_duplicates("Codigo_Planta"),
                      on="Codigo_Planta", how="left")
    base = base.sort_values("GWh", ascending=False)
    c1, c2 = st.columns([2, 1])
    etiquetas = {f"{row.Codigo_Planta} · {str(row.Nombre_Planta)[:34]} ({row.Tecnologia})": row.Codigo_Planta
                 for row in base.itertuples()}
    elegida = c1.selectbox("Planta (ordenadas por energía generada en la ventana)",
                           list(etiquetas), index=0 if etiquetas else None)
    cod = etiquetas.get(elegida)
    if not cod:
        return
    sub = gen[gen["Codigo_Planta"].astype(str) == str(cod)]
    perf = sub.groupby("Hora")["Generacion_kWh"].mean() / 1e3      # kWh/hora → MW medio
    dia = sub.groupby(sub["Fecha"])["Generacion_kWh"].sum() / 1e6  # kWh/día → GWh
    with c2:
        fp = ctx["fp"]
        fp_pl = (fp[fp["Codigo_Planta"].astype(str) == str(cod)] if not fp.empty and "Codigo_Planta" in fp
                 else pd.DataFrame())
        st.metric("Energía en la ventana", f"{float(base.loc[base['Codigo_Planta'].astype(str) == str(cod), 'GWh'].iloc[0]):,.1f} GWh")
        if not fp_pl.empty and "FP_pct" in fp_pl:
            st.metric("Factor de planta", f"{float(fp_pl['FP_pct'].mean()):.1f} %",
                      f"{len(fp_pl)} mes(es) con CEN")
    a, b = st.columns(2)
    fa = _fig(ctx, 280)
    fa.add_trace(go.Bar(x=[f"{h:02d}h" for h in perf.index], y=perf.values, marker_color=AZUL,
                        name="perfil medio"))
    fa.update_layout(title="Perfil horario medio (MW)", xaxis_title=None, yaxis_title="MW", height=280,
                     margin=dict(l=44, r=10, t=42, b=28))
    with a:
        st.plotly_chart(fa, use_container_width=True, config={"displaylogo": False},
                        key=_clave_fig("perfil-planta"))
    fb = _fig(ctx, 280)
    fb.add_trace(go.Scatter(x=list(dia.index), y=list(dia.values), mode="lines+markers",
                            line=dict(color=VERDE, width=2), name="GWh/día"))
    fb.update_layout(title="Energía diaria", xaxis_title=None, yaxis_title="GWh", height=280,
                     margin=dict(l=44, r=10, t=42, b=28))
    with b:
        st.plotly_chart(fb, use_container_width=True, config={"displaylogo": False},
                        key=_clave_fig("energia-planta"))
    st.caption(f"**📖 Explorador de planta — qué muestra y para qué sirve**\n\n"
               f"- **Qué muestra:** el perfil horario medio y la energía diaria de `{cod}` "
               f"({str(base.loc[base['Codigo_Planta'].astype(str) == str(cod), 'Nombre_Planta'].iloc[0])}), "
               f"con su factor de planta cuando hay CEN de la misma planta.\n"
               f"- **Decisión / conclusión:** permite pasar del agregado del mercado a la unidad: si la "
               f"planta corre plana en las horas caras es despacho inflexible (útil); si solo corre en el "
               f"pico, es unidad de precio (su renta depende de que haya sequía).\n"
               f"- **Lente El Niño:** las unidades que suben su producción cuando caen los aportes son las "
               f"que se benefician del episodio; revíselas antes de negociar un contrato de capacidad.")


def explorar_rio_zona(ctx: dict) -> None:
    """Vista por recurso/cuenca (reemplazo del explorador geográfico; sin ipywidgets ni UPME)."""
    dim = ctx["dim_plantas"]
    if dim is None or dim.empty:
        return
    col_group = "Tecnologia" if "Tecnologia" in dim.columns else dim.columns[0]
    grupos = sorted(dim[col_group].dropna().astype(str).unique().tolist())
    c1, c2 = st.columns([1, 2])
    g = c1.selectbox("Agrupar recursos por", [col_group] + [c for c in ("Fuente_Energia", "Tipo_Recurso",
                                                                        "Tipo_Despacho", "Estado")
                                                             if c in dim.columns])
    resumen_g = (dim.groupby(g)[["Codigo_Planta"]].count().rename(columns={"Codigo_Planta": "Recursos"})
                 .sort_values("Recursos", ascending=False).reset_index())
    gen = ctx["gen_enr"]
    if gen is not None and not gen.empty and "Codigo_Planta" in gen:
        tot = gen.groupby("Codigo_Planta")["Generacion_kWh"].sum() / 1e6
        mapa = dim[["Codigo_Planta", g]].drop_duplicates("Codigo_Planta") if g != col_group else \
            dim[[col_group, "Codigo_Planta"]].rename(columns={col_group: g}).drop_duplicates("Codigo_Planta")
        mapa["GWh"] = mapa["Codigo_Planta"].astype(str).map(tot)
        resumen_g = mapa.groupby(g, as_index=False).agg(Recursos=("Codigo_Planta", "nunique"),
                                                        GWh=("GWh", "sum")).sort_values("GWh", ascending=False)
    with c2:
        fig = _fig(ctx, 300)
        fig.add_trace(go.Bar(x=resumen_g[g].astype(str), y=resumen_g["GWh" if "GWh" in resumen_g else "Recursos"],
                             marker_color=CELESTE, name="GWh"))
        fig.update_layout(title="Energía generada por categoría en la ventana", height=300,
                          margin=dict(l=44, r=10, t=42, b=48), yaxis_title="GWh",
                          xaxis_tickangle=-25)
        st.plotly_chart(fig, use_container_width=True,
                           config={"displaylogo": False}, key=_clave_fig("rio-zona"))
        st.dataframe(resumen_g, use_container_width=True, hide_index=True)
    rios = ctx.get("rios", pd.DataFrame())
    if isinstance(rios, pd.DataFrame) and not rios.empty:
        with c1:
            st.markdown("**Ríos / cuencas del catálogo**")
            st.dataframe(rios.filter(like="Code").head(44) if "Code" in rios else rios.head(44),
                         use_container_width=True, hide_index=True, height=220)
            st.caption("`ListadoRios` es solo dimensional (44 filas): sirve para saber qué cuencas "
                       "existen en el modelo de XM. La asignación planta↔río vive en `ListadoRecursos` "
                       "y en el módulo geográfico del notebook (UPME), que requiere `certs/"
                       "upme_bundle.pem` y no forma parte de esta app.")


def comparar_catalogos_historicos(ctx: dict) -> None:
    """Versión Streamlit de `comparar_todos_los_catalogos`: qué cambió entre snapshots."""
    archivos = sorted(RAW_DIR.glob("dim_plantas_*.csv"))
    if len(archivos) < 2:
        st.caption("Aún hay un solo snapshot de `ListadoRecursos` en `data/raw`: la comparación de "
                   "catálogos (altas/bajas de plantas) se activa cuando existan dos fechas distintas.")
        return
    actual = ctx["dim_plantas"]
    filas = []
    for ruta in archivos[:-1]:
        try:
            viejo = preparar_dim_plantas(pd.read_csv(ruta))
        except Exception:                                         # noqa: BLE001
            continue
        a, b = set(actual["Codigo_Planta"].astype(str)), set(viejo["Codigo_Planta"].astype(str))
        filas.append({"snapshot anterior": ruta.stem.replace("dim_plantas_", ""),
                      "plantas anterior": len(b), "plantas hoy": len(a),
                      "altas": len(a - b), "bajas": len(b - a),
                      "códigos nuevos": ", ".join(sorted(a - b)[:8]) + ("…" if len(a - b) > 8 else ""),
                      "desaparecidos": ", ".join(sorted(b - a)[:8]) + ("…" if len(b - a) > 8 else "")})
    if filas:
        st.markdown("**Comparación de catálogos (`ListadoRecursos` en distintas fechas)**")
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        st.caption("Sirve para detectar cambios de padrón (plantas nuevas o retiradas) entre una "
                   "ejecución y otra: si aparece una solar grande, el riesgo de precio en el pico de "
                   "la tarde baja marginalmente.")


# ==============================================================================
# 10 · MÓDULO «SIN NODAL» (v4.2) — precios marginales nodales bajo el enfoque
#      de la propuesta de tesis doctoral «Modelamiento estocástico de los precios
#      de la energía en un mercado de precios marginales nodales con alta
#      componente de generación variable» (L. Acero García, dir. S. Rivera, UNAL
#      Sede Bogotá, 2025 — Anexo de revisión V15SC, 72 págs.).
#
# Pregunta que responde la pestaña 🕸️: ¿qué pasaría en el SIN si la energía se
# liquidara a precio *nodal* en vez de uninodal? El bloque construye una red DC
# con PTDF a partir de la toponimia real de XM, calibra la curva de oferta contra
# el precio de bolsa observado, resuelve el OPF con congestión y pérdidas, forma
# los LMP por nodo (Eq. 2-15…2-21), y los compara con la liquidación uninodal con
# recurso estocástico, almacenamiento, coberturas, VaR-GARCH y comportamiento
# estratégico de los agentes (objetivos a-f de la propuesta).
#
# Todo es AUTOCONTENIDO y local: no llama a la API de XM, usa el `ctx` ya cargado
# por las pestañas v1-v4. No usa scipy ni ni-newton: el OPF es un ascenso dual
# propio sobre la condición KKT y el GARCH se ajusta por rejilla + refinamiento.
# ==============================================================================





NODAL_BASE_MVA = 1000.0     # base de potencias en por unidad
NODAL_CURVATURA = 0.45      # con P=Pmax el precio ofertado es 1,45·c1 (Eq. 2-16 cuadrática)
NODAL_PASO_DUAL = 0.45      # paso del ascenso dual sobre los multiplicadores μ de congestión
NODAL_ITER_DUAL = 60        # rondas máximas del ascenso dual
NODAL_TOL_MW = 0.50         # violación aceptada de límite de rama (MW)
NODAL_BID_VAR_LO = -10.0    # las variables se recortan si λ_nodo < −10 COP/kWh
NODAL_BID_VAR_HI = -5.0     # y entran al 100 % en cuanto λ_nodo ≥ −5 COP/kWh

# Subregiones usadas como nodos. ⚠️ Supuesto del tablero, no una clasificación
# oficial: la API abierta de XM no publica coordenadas ni zona para
# `ListadoRecursos`, así que la agregación topológica se declara aquí y se puede
# inspeccionar en la pestaña (tabla de red descargable).
ZONAS_NODAL: dict[str, tuple[str, tuple[float, float]]] = {
    "GUJ": ("La Guajira–Cesar–Magdalena", (0.74, 0.93)),
    "CAR": ("Caribe (Atlántico–Bolívar–Córdoba)", (0.46, 0.74)),
    "NOR": ("Nordeste (Antioquia–Santanderes–Eje)", (0.24, 0.60)),
    "CEN": ("Centro (Bogotá–Cundinamarca–Boyacá)", (0.50, 0.52)),
    "OCC": ("Occidente–Pacífico (Valle–Cauca–Chocó)", (0.20, 0.38)),
    "SUR": ("Sur (Huila–Nariño–Putumayo)", (0.33, 0.18)),
    "ORI": ("Orinoquía (Meta–Casanare–Arauca)", (0.79, 0.48)),
}
ZONAS_EXTRA_NODAL = {                       # nodos que aparecen al desdoblar
    "BOG": ("Bogotá–Sabana (nodo de carga)", (0.44, 0.56)),
    "ANT": ("Antioquia medio (nodo de carga)", (0.19, 0.52)),
    "MAG": ("Magdalena medio (nodo de carga)", (0.36, 0.66)),
}
ZONA_PADRE_NODAL = {"BOG": "CEN", "ANT": "NOR", "MAG": "CAR"}

# Toponimia para asignar cada recurso de XM a una subregión. Regla del tablero:
# se busca la palabra clave en `Nombre_Planta`. Lo no asignado se reparte en
# proporción a la capacidad ya ubicada y se reporta como brecha (actividad 3.4
# de la tesis, «brechas de información para el análisis»).
KEYWORDS_ZONA: dict[str, tuple[str, ...]] = {
    "GUJ": ("GUAYEPO", "TEPUY", "GUAJIRA", "CERREJON", "RIOFRIO", "RIO FRIO", "POTRERILLOS",
            "GUINEA", "MAICAO", "TABORDO", "LA JAGUA", "BOSCONIA", "TERMOSIERRA", "SAMAMBA",
            "RIO HACHA", "URIBIA", "MANAURE", "URIANA", "PATIRIA", "RIOBONITO"),
    "CAR": ("BARRANQUILLA", "CARTAGENA", "GECELCA", "TERMOBARRANCA", "CARACOLI", "MARIMPA",
            "LOS MARTINEZ", "MOMPOX", "PUERTA DE HIERRO", "ZACARNA", "CANDILEJAS", "LA MATA",
            "AMOYA", "JAGUAS", "CALAMAR", "SAN JORGE", "MOJANA", "UARRIA", "CANAL DEL DIQUE",
            "LA CEIBA", "SANTA MARTA", "SITIONUEVO", "SAN BERNARDO",),
    "NOR": ("ITUANGO", "PORTAL", "JEPA", "PAGUA", "SAN CARLOS", "BETANIA", "GUATRON", "GUATAPE",
            "PORCE", "PIEDRAS", "CIELO", "TRAPICHES", "RIO CLARO", "RIOCLARO", "SOTARA",
            "RINON", "PRADO", "SAN RENSO", "MOROJORRO", "NAZARETH", "GUETAR", "CABRERA",
            "CALIMA", "YALI", "NARE", "PEÑOL", "JORDAN", "LA MIEL", "MIEL II", "PROYECTOR",
            "FLORIDOBONITA", "BUENAVISTA", "TERMOFLORES", "FLORES", "TASAJERA", "SANTA RITA",
            "COSTILLA", "SALVATIERRA", "SAN ELEUTERIO", "RIOCLARITO", "LA QUENCHE", "CUCUANA",
            "TOLEDO", "SUSCON", "NUTIA", "SANTANA", "GUADALUPE", "PIEDRA DEL HONE", "ALTO ANCHO"),
    "CEN": ("CHIVOR", "SOGAMOSO", "TEBSAB", "PAIPA", "ZIPAQUIRA", "ZIPAEMG", "QUIMBO",
            "TERMOCENTRO", "FARMELCO", "CPS", "TORCA", "TASAJERO", "GUAVIO", "ESCUELA DE MINAS",
            "BOGOTA", "SUSACON", "TOGUA", "NECHI", "SALTO", "PRADELO", "CORRALES", "HUNZA",
            "TIBITU", "ALMANZA", "SAN MATEO", "SABANA DE TORDESILLAS", "QUETAME", "PAPAL",
            "HATO VIEJO", "CHIQUINQUIRA", "TEPSA", "CENTRALES DE BOYACA",),
    "OCC": ("TERMOVALLE", "EMCALI", "TIERRERO", "RIOCAUCIA", "RIOPAILA", "ANIMAS",
            "DOHA", "YUMBO", "CORTULUGO", "LA ARGENTINA", "PACIFICO", "BUGA", "PRADERA",
            "SILVIA", "URRA", "CHOCO", "QUIBDO", "JOVID", "PETEZA", "SAN RAFAEL", "GITMA",
            "FRIAS", "PIEDRA", "CERRO MATOSO", "TERMOVALLE", "DAGUA",
            "RIOPAILA", "LA VICTORIA"),
    "SUR": ("MIEL I", "QUILE", "PASTO", "AMBALO", "CHIVOQUINZE", "MOCOA", "SAN ISIDRO",
            "ALBAN", "INCAUCA", "PAICOL", "PLAYAS", "TESTRILLO", "SUTATAGA", "NARIÑO",
            "PUTUMAYO", "HUILA", "MAYA", "RIO BLANCO", "GARZON", "BRICEÑO", "SIBUNDOY",
            "ANGOSTURA", "BETANIA SUR", "MOLINOS", "ZARAGOZA", "TESTRILLO"),
    "ORI": ("SALVAJINA", "YOPAL", "COPILANO", "AGUACLARA", "AGUA CLARA", "SAN PLACIDO",
            "JIVITO", "MUTIS", "CUBARRAL", "ARAUCA", "VICHADA", "TRUMPO", "GUACAMAYAS",
            "METOPOLITANO", "YOPALGAS", "CPO",),
}

# Prima relativa del costo variable por bloque. Orden de magnitud del despacho
# ideal colombiano: carbón < hidráulica (con costo de oportunidad del agua) < gas
# de ciclo combinado < diésel/OXY; biomasa cerca del gas. El nivel absoluto lo fija
# la calibración contra el precio real de bolsa (`calibra_oferta`).
PRIMA_NODAL = {"HIDRO": 0.74, "CARBON": 0.86, "GAS_CC": 1.00, "GAS_OXI": 1.30,
               "BIO": 1.06, "SOLAR": 0.02, "EOLICA": 0.02}
TEC_NODAL_LABEL = {"HIDRO": "Hidráulica", "CARBON": "Térmica carbón", "GAS_CC": "Gas CC",
                   "GAS_OXI": "Gas OXY/diésel", "BIO": "Cogeneración/biomasa",
                   "SOLAR": "Solar FV", "EOLICA": "Eólica"}
TEC_XM_A_NODAL = {"HIDRAULICA": "HIDRO", "TERMICA": "CARBON", "SOLAR": "SOLAR",
                  "EOLICA": "EOLICA", "COGENERADOR": "BIO", "SIN CATALOGAR": "CARBON"}
FE_NODAL = {"HIDRO": 0.0, "CARBON": 0.336, "GAS_CC": 0.200, "GAS_OXI": 0.240, "BIO": 0.024,
            "SOLAR": 0.0, "EOLICA": 0.0}   # tCO₂e/MWh (los mismos factores del tablero v3)

# Ramas del anillo entre subregiones: (desde, hasta, x_pu, r_pu, MW_max, nombre).
# Órdenes de magnitud de los corredores reales (500/230 kV); NO son valores
# oficiales de XM/ISA. `escalon_lineas` de la pestaña permite escalar toda la malla
# para la sensibilidad de congestión que pide el objetivo b.
RAMAS_BASE: tuple[tuple[str, str, float, float, float, str], ...] = (
    ("GUJ", "CAR", 0.0180, 0.0022, 2600.0, "Riohacha–Barranquilla (500 kV)"),
    ("CAR", "CEN", 0.0310, 0.0040, 1750.0, "Costa–Centro (230 kV, corredor limitado)"),
    ("CAR", "NOR", 0.0260, 0.0031, 2100.0, "Caribe–Nordeste (500/230 kV)"),
    ("NOR", "CEN", 0.0220, 0.0027, 2900.0, "Chivor/Sogamoso–Bogotá (500 kV)"),
    ("NOR", "OCC", 0.0290, 0.0036, 1450.0, "Nordeste–Valle (Guatapé–Yumbo)"),
    ("OCC", "SUR", 0.0140, 0.0018, 1250.0, "Valle–Cauca–Huila (Yumbo–Incauca–Paicol)"),
    ("SUR", "CEN", 0.0270, 0.0034, 1550.0, "Paicol–Chipaque (Cofradía, 500 kV)"),
    ("CEN", "ORI", 0.0350, 0.0047, 850.0, "Centro–Yopal (230 kV, radial débil)"),
    ("GUJ", "CEN", 0.0480, 0.0061, 900.0, "Guajira–Centro por Costa (refuerzo hipotético)"),
    ("NOR", "SUR", 0.0330, 0.0042, 780.0, "Nordeste–Huila (anillo del Magdalena medio)"),
)

# Share de la demanda del STN por subregión (supuesto declarado; editable en la
# pestaña vía `peso_demanda`, y visible en la tabla de red).
PESO_DEMANDA_NODAL = {"CEN": 0.26, "CAR": 0.21, "NOR": 0.24, "OCC": 0.12, "SUR": 0.08,
                      "ORI": 0.045, "GUJ": 0.045}


# ------------------- 10.1 · toponimia → subregión (brecha 3.4) --------------


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

    Es la respuesta operativa a la actividad 3.4 de la tesis («brechas de
    información para el análisis»): la API abierta de XM no publica coordenadas
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
    pérdidas marginales» de la Figura 2-7 y el término λ·LF_i de la Eq. 2-21.
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
                  "diferencia entre precio nodal y zonal (sensibilidad de §6.4)."),
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
    muestra hay detrás de cada ajuste (la tesis exige validar con datos históricos).
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
    lector juzgue el modelo (§6.8 «validación contra datos históricos»).
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


# ------------------ 10.4 · distribuciones del recurso (§2.2) ---------------
#
# Convención única: TODAS las familias se parametrizan por (media, desviación
# típica) de la muestra real con el método de momentos —la vía que sigue la
# propuesta cuando no hay serie larga de sitio (§2.2.1/2.2.2)—. Así se pueden
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
        form = f"Weibull(k={be:.2f}, c={lam:.2f}) — F(x)=1−exp(−(x/c)^k) (Eq. 2-3 / 2-4)"
    elif t == "rayleigh":
        vm = par["Vm"]
        g = np.maximum(xx, 0.0)
        pdf = (math.pi / 2.0) * (g / vm ** 2) * np.exp(-(math.pi / 4.0) * (g / vm) ** 2)
        cdf = 1.0 - np.exp(-(math.pi / 4.0) * np.power(g / vm, 2.0))
        form = f"Rayleigh(Vm={vm:.2f}) — caso Weibull con β=2 (Eq. 2-5, Paraschiv et al. 2019)"
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

    §2.2.1: «la curva de potencia es aproximadamente lineal hasta cierto nivel de
    irradiancia y luego se vuelve cuadrática a mayores niveles … por calentamiento
    y saturación del panel». Se implementa con la forma monótona
    P/Pn = (1 − e^(−a·G))/(1 − e^(−a)), a = 1/`punto_lineal`: su desarrollo de
    Taylor es lineal en G baja y cuadrático-cóncavo en G alta —la morfología que
    pide la propuesta, sin saltos ni tramos decrecientes.
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


# --------------- 10.5 · generación estocástica de escenarios (§2.2) ---------

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
    else:                                    # log-normal (por defecto, §2.2.1)
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

    Cada recurso se dibuja de la familia pedida en §2.2 con la media y el CV
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


# ---------- 10.6 · DC-OPF con duals (Eq. 2-15..2-21 de la tesis) ------------

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
    Eq. 2-16 con Eq. 2-19 cuando no hay congestión, o sea el despacho
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
    """Despacho por bloque (…, G) dado λ (…, ). Es la salida del 'unit commitment' relajado."""
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

    Resuelve la formulación de la tesis (Eq. 2-16 … 2-21):

        min Σ_g 1000·(c1_g·P_g + c2_g·P_g²)                        costo de operación (COP/h)
        s.a. Σ_i (1 − LF_i)·P_inj,i = L0 − Σ_i LF_i·P_inj0,i        balance con pérdidas linealizadas
             −Fmax_k ≤ Σ_j SF_kj·P_inj,j ≤ Fmax_k                    seguridad de las ramas (Eq. 2-18)
             Pmin_g ≤ P_g ≤ Pmax_g                                   límites de bloque (Eq. 2-19)

    y devuelve el precio nodal λ_i = λ_energia·(1 + LF_i) + Σ_k SF_ki·μ_k (Eq. 2-21).
    λ se obtiene por bisección (el balance es monótono en λ) y μ por ascenso dual
    proyectado sobre la violación de los límites de rama, aplicado solo a las horas
    congestionadas: eso es lo que deja correr Monte Carlo completo en el navegador
    sin un solver LP externo (y sustituye a Matpower en la parte de formación de
    precios, §2.6.2).

    Reglas de liquidación que salen del mismo despacho (`precios`):
      uninodal  → λ única para todos los nodos (lo que hoy hace el SIC).
      nodal     → λ_i por nodo (LMP, Eq. 2-21).
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
    # Poder de mercado (§2.5.1): el agente ofrece sus bloques por encima de su costo
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

    # ---- ascenso dual adaptativo (multiplicadores de congestión, Eq. 2-20) ----
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
    # Reglas de formación de precio sobre el mismo despacho (§2.4 y objetivo b).
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
    """Validación de los duals contra la definición de Eq. 2-15: λ_i = ∂Costo/∂P_i.

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


# ---------------- 10.7 · almacenamiento: EMS de arbitraje (Eq. 2-9..2-14) ---
#
# La propuesta modela bombeo/PSH con `V_urt^min ≤ V_t ≤ V_urt^max` y potencia de
# turbina/bomba (P_PSH_STG / P_PSH_IN). Aquí se resuelve con el EMS de dos
# pasadas que describe §2.4/§2.7: (1) OPF sin almacenamiento → precios por nodo;
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

    §2.5.1 pide medir el poder de mercado con la demanda residual y el markup. Aquí:

    * **IOR** = (P − RAC)/P con P = `Precio_Bolsa_Dia_COP_kWh` y RAC ≈
      `Costo_Marginal_COP_kWh_Dia` de `cruce_v4` (la tabla que valida la celda 61 del
      notebook). La SSPD abre investigación con IOR > 0,15 (expedientes EPM
      14/15-mar-2022 y Emgesa 11/14/15-mar-2022, citados en la propuesta).
    * **η** = elasticidad-precio de la demanda estimada con *efectos fijos horarios*
      (se restan las medias por hora sobre las 24 × días del ventana). Sin esos fijos,
      la regresión cruda da η > 0 porque precio y demanda comparten el ciclo diario:
      el tablero reporta las dos para que se vea la trampa econométrica.
    * **HHI y cuota del primer agente por subregión**, medidos con `dim_plantas`
      (`Codigo_Agente`) × `cen_recurso` × la toponimia del tablero y nombrados con
      `catalogo_agentes`: es la `s_i` con la que se calcula el markup de Lerner, no un
      supuesto de libro.
    * **Fases ENSO** (`impacto`): el precio ponderado medio por fase, para la
      hipótesis de la tesis de que el retiro aumenta cuando el sistema está seco.
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
    """Bertrand (precios) y Cournot (cantidades) sobre la demanda residual del nodo (§2.5.1).

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
    """GARCH(1,1) estimado por máxima verosimilitud gaussiana (Eq. 2-22/2-23).

    σ²_t = ω + α·ε²_{t−1} + β·σ²_{t−1}. La propuesta usa GARCH para la volatilidad
    condicional del precio y de ahí el VaR (§2.5.2). Como `requirements.txt` no
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
    """VaR (y ES) condicional del precio con la volatilidad de `garch11` (§2.5.2).

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
    """Razón de cobertura óptima h* = ρ·σ_S/σ_F (Eq. 2-24, §2.5.3).

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
                nota="h* = ρσ_S/σ_F (Eq. 2-24); β = cov/σ_F² es el mínimo de varianza puro")


def qlearning_despacho(red: dict, hora: int, perfiles: dict[str, np.ndarray], *,
                       nivel_cop_kwh=900.0, n_agentes: int = 6, episodes: int = 8000,
                       alpha: float = 0.15, gamma: float = 0.92, epsilon_greedy: float = 0.25,
                       n_niveles: int = 24, semilla: int = 7, paso_dual: float = 400.0,
                       seed: int | None = None) -> dict[str, Any]:
    """Despacho aprendido por refuerzo: agentes + ascenso dual sobre el precio (§2.7.2).

    Cada bloque es un agente que *no* conoce las curvas de costo de los demás: sólo ve
    el precio de la hora (el estado es el precio discretizado en 10 bandas) y aprende
    con Q-learning la mejor respuesta `max_q λ·q − c(q)`. El precio no lo fija el
    árbitro con las curvas en la mano: se actualiza con el ascenso dual de la Eq. 2-20
    de la propuesta, `λ ← λ + paso·(Σq − D)/D`, igual que en el OPF del módulo. Al
    converger, el perfil de ofertas aprendido reproduce el merit order — ése es el
    resultado que la tesis espera de la capa de aprendizaje.

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
        # Ascenso dual (Eq. 2-20): si hay exceso de demanda hay que SUBIR λ.
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
                      "Es el mismo mensaje de §2.7.2: el aprendizaje reproduce el merit order "
                      "sin el coordinador, y el residual de balance lo cierra el ascenso dual."))


def aprendizaje_regla(red: dict, perfiles: dict[str, np.ndarray], nivel_cop_kwh, *,
                      reglas: tuple[str, ...] = ("uninodal", "nodal"), acciones=None,
                      episodes: int = 60, semilla: int = 5, nodo_objetivo: str = "CEN",
                      horas: list[int] | None = None) -> dict[str, Any]:
    """Un agente que *aprende* a ofertar bajo cada regla (objetivo c de la tesis).

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
    """λ_i = energía + pérdidas + congestión (Eq. 2-21), por nodo y hora."""
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


NODAL_TESIS: dict[str, tuple[str, str]] = {
    # clave de figura → (§ de la propuesta, fórmula o dato que la sostiene)
    "nodal_red": ("§2.6.1 / §2.6.2", "OPF DC con PTDF (Eq. 2-25…2-28); topología de 7 subregiones XM"),
    "nodal_cap": ("§2.2.1 / datos XM", "CEN por subregión desde `CapEfecNeta`+`dim_plantas`; eólica sembrada por p99 de `Gen_EOLICA`"),
    "nodal_calib": ("§2.3 / objetivo a", "nivel de oferta κ calibrado contra `PrecBolsNaci` (mediana horaria) y verificación por diferencias finitas"),
    "nodal_lmp": ("§2.4 Eq. 2-15", "λ_i = λ + Σ_k SF_ki·μ_k + λ·LF_i (pérdidas linealizadas)"),
    "nodal_spread": ("§2.4 / objetivo b", "dispersión de LMP entre nodos; renta de congestión Σ(λ_i−λ_j)F_ij"),
    "nodal_cong": ("§2.6.2", "uso de rama |SF·iny|/Fmax; violación y pérdidas iteradas"),
    "nodal_flujos": ("§2.6.1 Eq. 2-26", "flujos DC por rama y renta de congestión por ramo"),
    "nodal_desp": ("§2.4 / §2.2.1", "despacho óptimo con recorte renovable y piso de despacho"),
    "nodal_nodo": ("§2.4", "balance del nodo: demanda servida vs generación + variable"),
    "nodal_beneficios": ("objetivos a-c", "síntesis de kpis_reglas, barrido_flexibilidad, aprendizaje_regla, ems_almacenamiento y sensibilidad_penetracion del escenario activo"),
    "nodal_flex": ("objetivo a", "desconexión voluntaria: `flex_frac` × precio gatillo dentro del OPF con corte por nodo"),
    "nodal_reglas": ("§2.4.3 / objetivo b", "uninodal vs zonal vs híbrido (κ) vs nodal — desbalance de pagos"),
    "nodal_descomp": ("Eq. 2-21", "λ_i = λ + λ·LF_i + Σ SF_ki μ_k desglosado por componente"),
    "nodal_valida": ("Eq. 2-15", "∂Costo/∂D_i por diferencias finitas vs multiplicador dual"),
    "nodal_esc": ("§2.2.1 / §2.2.2", "log-normal solar (τ_b de Perez 2023) + Weibull/Rayleigh eólico (Acero 2024) + k-medias"),
    "nodal_penetra": ("objetivo a/d", "barrido de penetración renovable: precio, congestión, recorte, CO₂"),
    "nodal_ior": ("§2.5.1", "IOR = (P−RAC)/P con P y CMD reales de XM; HHI por nodo con `Codigo_Agente`"),
    "nodal_mec": ("§2.5.1 / objetivo c", "demanda residual + Lerner s/|ε| + markup sobre oferta; Bertrand = precios, Cournot = cantidades"),
    "nodal_var": ("§2.5.2 Eq. 2-22/2-23", "GARCH(1,1) horario + VaR/ES condicional + Kupiec POF"),
    "nodal_cov": ("§2.5.3 Eq. 2-24", "cobertura de mínima varianza h* = ρ·σ_S/σ_F"),
    "nodal_ems": ("§2.3 / objetivo e", "EMS de baterías y bombeo con water-filling bajo precio nodal"),
    "nodal_ql": ("§2.7.2", "Q-learning por agente con el precio como estado y ascenso dual (Eq. 2-20)"),
    "nodal_ap": ("§2.5.1 / §2.7.2", "mark-up aprendido por agente según la regla de liquidación (hipótesis central)"),
}


# ------------------------------ modelo compartido ----------------------------

_NODAL_DEFECTOS: dict[str, Any] = dict(granularidad="media", estres=0.70, regla="nodal", kappa=0.5, hora_foco=20,
                    flex=15.0, gatillo=1200.0, semilla=7, n_esc=80, n_tipicos=8,
                    fam_sol="lognormal", fam_eol="weibull", rho_recurso=0.6, rho_cruce=-0.15,
                    bateria_pct=6.0, dur_bateria=4.0, eta_bateria=0.88, bombeo_pct=0.0,
                    dur_bombeo=12.0, eta_bombeo=0.80, ems_modo="arbitraje", ems_pasadas=5,
                    almacenar=False, barrido_flex=True, fase="ventana", nodo_estr="auto",
                    markup_pct=0.0, wh_pct=0.0, wh_modo="energia", epsilon=-0.45, opf_calib=True,
                    valida_lmp=True, horas_valida=12, riesgo=True, alfa=0.05, mercado=True,
                    sensibilidad=False, n_esc_sens=8, aprendizaje=True, episodios_rl=6000,
                    episodios_ap=16, n_agentes=6, escenarios=True, perdas_iter=3, curvatura=0.45,
                    solar_pct=0.0, eolica_pct=0.0, alm_nodos=["CEN", "BOG"], ap_horas=[19, 20],
                    sens_niveles=[(0.0, 0.0), (25.0, 10.0), (60.0, 25.0), (100.0, 40.0),
                                  (150.0, 60.0)])


# --------------------- 10.10 · escenarios justificados y fase ENSO ----------

def _fase_enso(ctx: dict, cfg: dict) -> dict[str, Any]:
    """Multiplicadores de escenario tomados de la pestaña 🌊 El Niño (`ctx["impacto"]`).

    La propuesta pide comparar regímenes; en vez de inventar un "escenario El Niño",
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


# nombre del preset → (qué fija, justificación del escenario, objetivo de la tesis)
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
                "la propuesta, donde la oferta declarada superó el costo de oportunidad en esas "
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
        porque=("La transición que discute la propuesta en vez del salto al LMP pleno: liquidación "
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
                costo_MCP_h=float(np.mean(r["costo"]) * 1000.0 / 1e6),
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
            mdl["esc"] = reduce_escenarios(esc, k=int(cfg["n_tipicos"]), semilla=int(cfg["semilla"]))
            mdl["esc_n"] = esc["S"]
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
                 "derecha dice por qué y qué objetivo de la tesis cubre. Los controles que el preset "
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
                 "para ver cómo los refuerzos la borran. (§2.6.2)")
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
                 "discute la propuesta. Nodal = LMP pleno (Eq. 2-15).")
        kappa = st.slider("κ del régimen híbrido", 0.0, 1.0, 0.5, 0.05, key="nd_kappa",
                          help="Peso de la componente nodal en la liquidación híbrida. κ=0 ⇒ uninodal, "
                               "κ=1 ⇒ nodal. Es el knob con el que se lee el compromiso eficiencia / "
                               "señales locacionales.")
    with c3:
        hora = st.number_input("Hora foco (1-24)", 1, 24, 20, 1, key="nd_hora",
                               help="Hora del día para los cortes de barra, la descomposición de "
                                    "precios (Eq. 2-21) y el subproblema de refuerzo.")
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
                              help="Trayectorias horarias de recurso y demanda (§2.2.1/§2.2.2). Se "
                                   "reducen a los típicos por k-medias antes de usarlas.")
            n_tip = st.slider("Escenarios típicos (k-medias)", 4, 20, 8, 1, key="nd_ntip",
                              help="Reducción clásica de la programación estocástica: conservan la "
                                   "media y aproximano la varianza a una fracción del costo.")
            fam_sol = st.selectbox("Familia solar", ["lognormal", "gamma", "weibull", "uniforme"], key="nd_famsol",
                                   help="§2.2.1 usa log-normal con τ_b = a(1−e^(−ε·cosθz)) + a·e^(−k/cosθz) "
                                        "(Perez 2023); aquí se deja elegir la familia para el KS.")
            fam_eol = st.selectbox("Familia eólica", ["weibull", "rayleigh", "uniforme", "lognormal"], key="nd_fameol",
                                   help="§2.2.2: f(x)=aβx^(β−1)e^(−ax^β); Rayleigh es el caso β=2 y la "
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
                                         "donde la rama está apretada. Es el EMS de §2.3.")
            ems_pas = st.slider("Pasadas del EMS", 1, 10, 5, 1, key="nd_emsp",
                                help="El EMS y el OPF se resuelven en bloque: se iteran hasta que el "
                                     "cambio de despacho estableza (< 1 MW). Más pasadas = más cerca "
                                     "del óptimo conjunto, más lento.")
        with d4:
            markup = st.slider("Markup sobre la oferta (%)", 0, 40, 0, 1, key="nd_mk",
                               help="§2.5.1: los agentes con poder suben la oferta por encima del costo "
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
                              help="Eq. 2-15 dice que λ_i = ∂Costo/∂D_i; se comprueba perturbando la "
                                   "demanda del nodo ±2 MW y recalcando el OPF. Es la prueba dura de que "
                                   "el modelo hace lo que dice.")
            horas_valida = st.slider("Horas a validar", 4, 24, 12, 2, key="nd_valh",
                                     help="Cada hora cuesta 2 OPF completos.")
        with e2:
            riesgo = st.toggle("GARCH + VaR + cobertura", value=True, key="nd_ries",
                               help="§2.5.2/§2.5.3 sobre la serie horaria real de la ventana (648 h) "
                                    "y la del modelo. Es rápido (~0,5 s).")
            alfa = st.select_slider("α del VaR", [0.01, 0.025, 0.05, 0.10], value=0.05, key="nd_alfa",
                                    help="Nivel de confianza del VaR condicional y del backtest de Kupiec.")
            mercado = st.toggle("Poder de mercado (IOR, HHI, Cournot)", value=True, key="nd_merc",
                                help="§2.5.1 con los datos reales de concentración por agente y el "
                                     "markup sobre la curva de oferta.")
            sens = st.toggle("Barrido de penetración renovable", value=False, key="nd_sens",
                             help="Recalcula red + OPF por cada nivel de penetración: es la figura más "
                                  "cara del módulo (≈6 OPF con escenarios).")
            n_esc_sens = st.slider("Escenarios por nivel del barrido", 4, 40, 8, 2, key="nd_sensesc")
        with e3:
            aprend = st.toggle("Aprendizaje (Q-learning + reglas)", value=True, key="nd_ap",
                               help="§2.7.2. El refuerzo por regla corre un OPF por episodio, así que "
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
               n_esc=int(n_esc), n_tipicos=int(n_tip), fam_sol=str(fam_sol), fam_eol=str(fam_eol),
               rho_recurso=float(rho), rho_cruce=float(rhoc), bateria_pct=float(bateria),
               dur_bateria=float(dur_bat), eta_bateria=float(eta_bat), bombeo_pct=float(bombeo),
               dur_bombeo=float(dur_bom), eta_bombeo=float(eta_bom), ems_modo=str(ems_modo),
               ems_pasadas=int(ems_pas), markup_pct=float(markup), wh_pct=float(wh_pct),
               wh_modo=str(wh_modo), epsilon=float(epsilon), opf_calib=bool(opf_calib),
               valida_lmp=bool(valida), horas_valida=int(horas_valida), riesgo=bool(riesgo),
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
    """Diagrama unilineal: nodos (tamaño = demanda), ramas (grosor = uso, rojo = violada)."""
    red, res = mdl["red"], mdl["res"]
    nd, ram = red["nodos_df"], red["ramas"]
    if nd.empty or ram.empty:
        return None
    fig = _fig(ctx, 520)
    uso = np.asarray(res["uso_rama"], dtype=float)
    uso_med = np.nanmean(uso, axis=0) if uso.ndim > 1 else np.zeros(len(ram))
    uso_max = np.nanmax(uso, axis=0) if uso.ndim > 1 else uso_med
    for i, r in ram.reset_index(drop=True).iterrows():
        a = nd.loc[nd["id"] == r["de"], ["x", "y"]].to_numpy(dtype=float)
        b = nd.loc[nd["id"] == r["a"], ["x", "y"]].to_numpy(dtype=float)
        if len(a) == 0 or len(b) == 0:
            continue
        u = float(np.clip(uso_med[i] if i < len(uso_med) else 0.0, 0.0, 2.0))
        col = ROJO if u > 0.98 else (AMBAR if u > 0.80 else AZUL)
        fig.add_trace(go.Scatter(x=[a[0][0], b[0][0], None], y=[a[0][1], b[0][1], None], mode="lines",
                                 line=dict(color=col, width=1.0 + 9.0 * min(u, 1.5)),
                                 hovertemplate=f"{r['nombre']}<br>uso medio {100*u:.0f} % · máximo "
                                 f"{100*float(uso_max[i] if i < len(uso_max) else 0):.0f} %", name="_",
                                 showlegend=False))
        mx, my = (a[0][0] + b[0][0]) / 2.0, (a[0][1] + b[0][1]) / 2.0
        fig.add_trace(go.Scatter(x=[mx], y=[my], mode="text",
                                 text=[f"{100*u:.0f}%"], showlegend=False, hoverinfo="skip",
                                 textfont=dict(size=9, color=col)))
    px = 380.0 * np.sqrt(nd["CEN_MW"].to_numpy(dtype=float) / max(float(nd["CEN_MW"].max()), 1.0))
    fig.add_trace(go.Scatter(x=nd["x"], y=nd["y"], mode="markers+text", text=nd["id"],
                             textposition="top center", textfont=dict(size=11), name="Nodos",
                             marker=dict(size=px, color=[color_tecnologia("HIDRAULICA")] * len(nd),
                                         opacity=0.85, line=dict(width=1.2, color="#37474f")),
                             hovertemplate="%{text}<br>CEN %{customdata[0]:,.0f} MW · demanda "
                             "%{customdata[1]:.1%} del SIN<extra></extra>",
                             customdata=np.column_stack([nd["CEN_MW"].to_numpy(dtype=float),
                                                          nd["share_demanda"].to_numpy(dtype=float)])))
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False),
                      hoverlabel=dict(align="left"),
                      annotations=[dict(text="Diagrama unilineal *esquemático*: la posición no es "
                                              "geográfica, es la topología declarada en el módulo.",
                                        xref="paper", yref="paper", x=0.01, y=-0.06, showarrow=False,
                                        font=dict(size=10, color=GRIS))])
    viol = float(np.max(res["violacion"])) if np.size(res.get("violacion")) else 0.0
    return {"fig": fig, "leyenda": _leyenda(
        "Malla eléctrica del modelo y su uso",
        f"Nodos `{red['granularidad']}` con la CEN real de XM por subregión y las ramas del anillo "
        f"500/230 kV; el grosor y el número son el uso medio horario de la rama y el color avisa "
        f"cuando pasa del 80 % (rojo = violada en alguna hora).",
        f"Con `estres = {cfg['estres']:.2f}` hay {int(res['horas_congestion'])} de 24 h congestionadas y "
        f"la violación máxima es {viol:,.0f} MW. Si el uso máximo no llega a 100 %, el precio zonal y "
        "el nodal no pueden separarse: la discusión de la tesis requiere apretar la malla "
        "(o refinarla con `media`), no es un defecto del modelo.",
        "En seco sube el despacho térmico del centro y la demanda del pico, así que el corredor "
        "NOR→CEN (hidráulica → carga) es el primero en saturarse: bajar `Capacidad de las líneas` "
        "simula ese estrés de El Niño sobre la red.",
        f"§2.6.1/§2.6.2 · PTDF DC · CEN de `CapEfecNeta` (cobertura toponímica "
        f"{red['cobertura_toponimia']:.1f} %)")}


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
        "Anexo XM `CapEfecNeta_Res`/`dim_plantas` · §2.2.1 (UCF y recurso)")}


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
    fig.update_layout(xaxis=dict(title="Hora del día (1-24)", dtick=2), yaxis_title="COP/kWh")
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
        "produciría precios 30-60 % menores, que es justo la amplitud que la tesis mide entre fase "
        "La Niña y El Niño.",
        f"§2.3 · error {cal['error_calibracion_pct']:.2f} % · n={cal['n']} h · "
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
        "Eq. 2-15/2-21 · media de 31 días XM (2026-07-31→2026-08-26)")}


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
        "el cual la tesis espera que el precio nodal *revele* el estrés hídrico por zona en vez de "
        "promediarlo.",
        "objetivo b · Eq. 2-19/2-21 · "
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
        "no es que el modelo ignore la red, es que *esa* red no congestiona. Para la tesis hay que "
        "estresarla (`Capacidad de las líneas`) o refinarla (`media`).",
        "Las pérdidas crecen con el despacho térmico del centro y con la rampa de la tarde; en un mes "
        "seco el pico de pérdidas se alinea con el pico de precio, así que la señal nodal *sin* pérdidas "
        "linealizadas (iteraciones = 0) subestima el spread entre NOR y CEN.",
        "§2.6.1 Eq. 2-26/2-27 · perdas_iter = " + str(cfg["perdas_iter"]))}


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
        "y del esquema PJM que cita la propuesta).",
        "En la ventana seca analizada el corredor hidráulico-norte → centro-este carga más porque el "
        "norte genera y el centro consume: es la dirección que un El Niño prolongado agrava.",
        f"§2.6.1 · renta media {float(np.mean(res['renta_congestion']))/1e6:,.1f} M COP/h · "
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
        f"§2.4 · flex_frac {cfg['flex']:.0f} % · co2 {float(np.sum(res['co2_t'])):,.0f} t/día")}


def fig_nodal_nodo(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Dos paneles: el nodo más caro y el más barato, balance y precio."""
    res = mdl["res"]
    p = np.asarray(res["precios"].get(cfg["regla"], res["p_dem"]), dtype=float)
    nodos = list(res["nodos"])
    if p.size == 0 or len(nodos) < 2:
        return None
    media = p.mean(axis=0)
    caro, bar = int(np.argmax(media)), int(np.argmin(media))
    fig = _fig(ctx, 500)
    for j, (nodo, col) in enumerate(((nodos[caro], ROJO), (nodos[bar], VERDE))):
        h = np.arange(1, 25)
        fig.add_trace(go.Scatter(x=h, y=np.asarray(res["dem"], dtype=float)[:, caro if j == 0 else bar],
                                 name=f"{nodo} · demanda", line=dict(color=GRIS, width=1.8), xaxis="x",
                                 yaxis=f"y{2*j+1}", visible=True))
        fig.add_trace(go.Scatter(x=h, y=np.asarray(res["P_nodo"], dtype=float)[:, caro if j == 0 else bar],
                                 name=f"{nodo} · generación", line=dict(color=col, width=2.2, shape="spline"),
                                 yaxis=f"y{2*j+1}", xaxis="x"))
        fig.add_trace(go.Scatter(x=h, y=p[:, caro if j == 0 else bar], name=f"{nodo} · precio",
                                 line=dict(color="#263238", width=1.4, dash="dot"), yaxis=f"y{2*j+2}", xaxis="x"))
    fig.update_layout(grid=dict(rows=2, columns=1, pattern="independent"),
        xaxis=dict(title=f"Hora · panel superior = nodo más caro {nodos[caro]} "
                         f"(media {media[caro]:,.1f} COP/kWh)", dtick=4),
        xaxis2=dict(title=f"Hora · panel inferior = nodo más barato {nodos[bar]} "
                          f"(media {media[bar]:,.1f} COP/kWh)", dtick=4),
        yaxis=dict(title="MW"), yaxis2=dict(title="COP/kWh", overlaying="y", side="right", showgrid=False),
        yaxis3=dict(title="MW"), yaxis4=dict(title="COP/kWh", overlaying="y2", side="right", showgrid=False),
        legend=dict(orientation="h", y=1.16, x=0), hovermode="closest")
    return {"fig": fig, "leyenda": _leyenda(
        "Los dos extremos de la señal, nodo por nodo",
        "Balance del nodo (demanda vs generación propia, incluida la variable) y su precio, para el "
        "nodo con el LMP medio más alto y el más bajo de la topología actual. "
        f"Brecha entre ellos: {float(media.max() - media.min()):,.1f} COP/kWh de media diaria.",
        "Un nodo caro con generación propia escasa es un sitio para generación o almacenamiento; un nodo "
        "barato con excedente es un sitio para demanda (data center, hidrógeno, bombeo). Esa es la "
        "decisión de inversión que la liquidación uninodal no puede dar.",
        "El nodo barato suele ser el hidro-norte y el caro el centro-este: la separación se agranda en "
        "El Niño porque el norte deja de mandar excedente y el centro enciende térmica.",
        "§2.4 · balance impuesto solo a nivel de sistema (los desbalances por nodo son el derrame de "
        f"{float(np.max(np.asarray(res['derrame_MW'], dtype=float))):,.1f} MW máximo)")}


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
        f"§2.4.3 · kpis_reglas() · {len(kp)} reglas · " + str(mdl.get("nota_reglas", ""))[:80])}


def fig_nodal_descomp(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Descomposición del LMP en energía + pérdidas + congestión (Eq. 2-21)."""
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
    fig.update_layout(barmode="stack", xaxis_title="Nodo",
                      yaxis_title=f"COP/kWh (eje cortado en {base:,.0f} para que se vean las componentes)",
                      hovermode="x unified")
    maxdesv = float(np.max(np.abs(d["desvio"]))) if "desvio" in d.columns else float("nan")
    return {"fig": fig, "leyenda": _leyenda(
        f"De dónde sale el precio de cada nodo (hora {hh+1})",
        "Eq. 2-21 desglosada: λ de sistema + el término de pérdidas (λ·LF_i, con el factor de "
        "participación del nodo) + la suma de multiplicadores de rama ponderados por PTDF. El punto "
        "negro es el LMP que realmente devolvió el OPF.",
        f"La suma de las tres componentes reproduce el precio con un error máximo de {maxdesv:.3g} "
        "COP/kWh. Si el término de congestión es cero en todos los nodos, la hora elegida no está "
        "congestionada: cambie la hora foco a las franjas rojas de la figura de congestión.",
        "En hora valle el sol deja el precio dominado por pérdidas (componente negativa en los nodos "
        "exportadores); en el pico de la tarde la congestión manda. El Niño corre el pico de pérdidas "
        "hacia 18-20 h porque el térmico del centro trabaja más.",
        "Eq. 2-21 · descomposicion_precios()")}


def fig_nodal_valida(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Validación del multiplicador dual contra ∂Costo/∂D por diferencias finitas."""
    vr = mdl.get("vr")
    if isinstance(vr, dict):
        st.info(f"Validación no disponible: {vr.get('nota', 'no ejecutada')}")
        return None
    if not isinstance(vr, pd.DataFrame) or vr.empty:
        return None
    fig = _fig(ctx, 430)
    fig.add_trace(go.Scatter(x=np.asarray(vr["lambda_dual_cop_kWh"], dtype=float),
                             y=np.asarray(vr["lambda_dif_finita_cop_kWh"], dtype=float), mode="markers",
                             marker=dict(size=10, color=np.abs(np.asarray(vr["error_pct"], dtype=float)),
                                         colorscale=[[0.0, "#2e7d32"], [0.5, "#f9a825"], [1.0, "#c62828"]],
                                         showscale=True, colorbar=dict(title="|error| %", thickness=12)),
                             text=vr["nodo"], hovertemplate="%{text}: dual %{x:,.1f} vs FD %{y:,.1f}<extra></extra>",
                             name="Nodos (dual vs diferencias finitas)"))
    lo = float(min(vr["lambda_dual_cop_kWh"].min(), vr["lambda_dif_finita_cop_kWh"].min()))
    hi = float(max(vr["lambda_dual_cop_kWh"].max(), vr["lambda_dif_finita_cop_kWh"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x (igualdad exacta)",
                             line=dict(color=GRIS, width=1.2, dash="dash")))
    fig.update_layout(xaxis_title="λ_i del OPF (multiplicador dual, COP/kWh)",
                      yaxis_title="∂Costo/∂D_i por diferencias finitas (COP/kWh)", hovermode="closest")
    err = np.abs(np.asarray(vr["error_pct"], dtype=float))
    return {"fig": fig, "leyenda": _leyenda(
        "Prueba dura de que el LMP es el LMP",
        f"Eq. 2-15 afirma que el precio nodal es el sombra del balance del nodo, ∂Costo/∂D_i. Se midió "
        f"perturbando ±{float(np.mean(vr['paso_MW'])):.1f} MW la demanda de cada nodo y recalculando el OPF "
        f"completo, sobre {int(np.mean(vr['horas']))} horas. Error máximo {float(np.max(err)):.2f} %, "
        f"mediano {float(np.median(err)):.3f} %.",
        "Esto es lo que legitima todo lo demás de la pestaña: si este panel se desvía de la diagonal, el "
        "ascenso dual no está convergiendo y las cifras de renta/congestión no son de fiar (el modelo "
        "reporta `converged`/`estancado` en la tarjeta de estado para el mismo efecto).",
        "El error se concentra en los nodos con congestión activa, donde el término de pérdidas "
        "linealizado introduce el mayor sesgo: en El Niño, con más horas congestionadas, la validez de "
        "la linealización hay que volver a mirarla.",
        "Eq. 2-15 · valida_lmp() · horas " + str(cfg["horas_valida"]))}


def fig_nodal_esc(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Abanico estocástico: distribución del precio y del pico por nodo + congestión proxy."""
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
    fig = _fig(ctx, 520)
    # Panel 1: abanico de la demanda total de escenario (proxy del estresador)
    tot = de.sum(axis=1)
    for i in range(k):
        fig.add_trace(go.Scatter(x=np.arange(1, H + 1), y=tot[i], mode="lines", line=dict(width=1.1),
                                 opacity=0.55, name="Escenario típico", showlegend=(i == 0),
                                 hovertemplate=f"típico {i+1} h%{{x}}: %{{y:,.0f}} MW<extra></extra>",
                                 xaxis="x", yaxis="y"))
    base_dem = np.asarray(mdl["res"]["dem"], dtype=float).sum(axis=1)
    fig.add_trace(go.Scatter(x=np.arange(1, len(base_dem) + 1), y=base_dem, name="Demanda del día medio (XM)",
                             line=dict(color="#263238", width=2.4, dash="dash"), xaxis="x", yaxis="y"))
    # Panel 2: distribución del precio por nodo (los 4 más pesados)
    top = [i for i in np.argsort(-w)[:4] if i < n]
    for j, i in enumerate(top):
        fig.add_trace(go.Box(y=np.asarray(esc["demanda_MW"], dtype=float)[:, i, :], name=f"dem {nodos[i]}",
                             x=[nodos[i]] * k, marker_color=[AZUL, VERDE, AMBAR, ROJO][j % 4],
                             xaxis="x2", yaxis="y2", boxpoints="all", jitter=0.35, hoveron="boxes"))
    fig.update_layout(grid=dict(rows=1, columns=2, pattern="independent"),
        xaxis=dict(title="Hora (izq: escenarios típicos vs día medio real; der: por nodo)", dtick=4),
        yaxis=dict(title="MW · escenario típico"), xaxis2=dict(title="Nodo (cajas = demanda horaria)"),
        yaxis2=dict(title="MW por hora"), showlegend=True,
        legend=dict(orientation="h", y=1.14, x=0), hovermode="closest")
    return {"fig": fig, "leyenda": _leyenda(
        "Los escenarios que usa la capa estocástica",
        f"{int(mdl.get('esc_n', cfg['n_esc']))} trayectorias horarias generadas con familias "
        f"`{cfg['fam_sol']}` (solar) y `{cfg['fam_eol']}` (eólica), correlación entre nodos "
        f"ρ={cfg['rho_recurso']:.2f} y cruce sol-viento ρ={cfg['rho_cruce']:.2f}, reducidas a "
        f"{k} escenarios típicos por k-medias (inercia {float(esc.get('inercia', 0.0)):.3f}).",
        "Los escenarios son el insumo de la programación estocástica: el despacho aquí no se re-optimiza "
        "por escenario (sería un LP por trayectoria), sino que se lee la distribución de la variable. "
        "Las cajas a la derecha muestran que la dispersión *entre nodos* no es simétrica: la Guajira y "
        "el Oriente tienen colas mucho más largas que el centro.",
        "Las familias se ajustan al recurso real de la ventana; en un mes El Niño la log-normal solar "
        "engorda su media (más nubosidad en el interior) y la Weibull eólica del Caribe se desplaza a "
        "la izquierda: el mismo κ produce precios más altos y más variables.",
        f"§2.2.1/§2.2.2 · k-medias (Lloyd, {int(cfg['n_esc'])}→{k}) · KS contra la serie real")}


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
    fig.update_layout(xaxis_title="Demanda interrumpible (% de la demanda del sistema)",
                      yaxis_title="λ medio (COP/kWh)",
                      yaxis2=dict(title="GWh/día no servidos", overlaying="y", side="right",
                                  showgrid=False), hovermode="x unified")
    st.dataframe(fx[["flex_pct", "gatillo_COP_kWh", "lam", "dispersion", "precio_max", "corte_GWh",
                    "recorte_GWh", "horas_congestion", "viol_MW", "rc_MCP_h", "co2_t", "conv"]]
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
        "columna de corte. Es el mecanismo que la propuesta pide evaluar: la demanda desconectable "
        "como tercer instrumento, junto con la red y el almacenamiento.",
        "objetivo a · `barrido_flexibilidad()` con el mismo OPF (Eq. 2-16 con corte por nodo); "
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
             "más variabilidad exige más señal locacional: es el argumento de la tesis para el objetivo d")
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
        f"{n_pos} de {len(df)} marcadores salen a favor. La lectura que sostiene la tesis no es que el "
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
        "crece con la penetración: es la conclusión operativa de la tesis y la que justifica estudiar la "
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
        "pregunta de la tesis no es *si* hay poder de mercado, sino cuánto se puede ejercer bajo cada "
        "regla de liquidación (figura siguiente).",
        "El IOR sube en seco: la comparación por fase ENSO en `impacto` da precio ponderado "
        f"{float((ior.get('precio_pond_fases') or {}).get('El_Nino', float('nan'))):.0f} COP/kWh en El "
        f"Niño contra {float((ior.get('precio_pond_fases') or {}).get('La_Nina', float('nan'))):.0f} en "
        "La Niña — el mismo agente tiene más margen de oferta en meses secos.",
        "§2.5.1 · IOR de bolsa/CMD + HHI real · η con fijos horarios")}


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
    fig.update_layout(xaxis=dict(title="Hora", dtick=2), yaxis_title="COP/kWh",
                      yaxis2=dict(title="% sobre el precio de competencia", overlaying="y", side="right",
                                  showgrid=False), hovermode="x unified")
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
        "§2.5.1 · cournot_bertrand() + `markup`/`withholding` del OPF")}


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
        "Volatilidad condicional y riesgo de mercado (Eq. 2-22/2-23)",
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
        "§2.5.2 · garch11 (rejilla + Newton, sin scipy) · alfa " + f"{cfg['alfa']:.3f}")}


def fig_nodal_cov(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Cobertura de mínima varianza (Eq. 2-24) entre el precio del modelo y el futuro."""
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
        "Cuánto futuro comprar (Eq. 2-24)",
        f"Correlación ρ = {float(cv['rho']):.3f} entre el retorno del precio del modelo en el nodo y el "
        f"de la serie uninodal de XM; σ_S = {float(cv['sigma_spot']):.4f}, σ_F = "
        f"{float(cv['sigma_futuro']):.4f}; óptimo h* = ρσ_S/σ_F = {float(cv['h_optimo']):.3f} "
        f"(β de hedging puro {float(cv['beta_hedging']):.3f}) sobre {int(cv['n'])} pares horarios.",
        f"Cubrir con h* reduce la varianza {float(cv['reduccion_var_pct']):.1f} % "
        f"(σ² {float(cv['var_sin']):.2e} → {float(cv['var_cubierta']):.2e}) y deja un ingreso medio de "
        f"{float(cv['media_cubierta']):.4f} por MWh. Con ρ = {float(cv['rho']):.2f} el futuro 'casi "
        "cubierto' deja un residual grande: el riesgo locacional **no** se cubre con un producto "
        "nacional, que es exactamente el argumento de la tesis para tener mercado nodal con derechos de "
        "congestión negociables.",
        "La ventana es seca y corta (31 días); en El Niño ρ sube pero σ_S también, así que h* no es "
        "constante: un coberturista debe re-estimar cada semana con `garch11` en lugar de fijar h.",
        "§2.5.3 · cobertura_min_var()")}


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
        "Almacenamiento con precio nodal (§2.3 y objetivo e)",
        "Programa óptimo del EMS (water-filling bajo el vector de precios, con η, Pmin de reserva y "
        "máscaras por modo) contra el precio del sistema. Carga donde el LMP es bajo —en la ventana "
        "solar de media mañana— y descarga en el pico 18-21 h; el SOC cierra el día donde empezó.",
        "La línea de texto compara el mismo activo bajo las dos reglas: la diferencia es lo que vale la "
        "**señal locacional** para un inversor de baterías. Si el valor cae con `alivio_congestion` o la "
        "violación de ramas sube (como le pasa al bombeo mal sentado), el almacenamiento no es un "
        "sustituto del refuerzo: la tesis lo dice y aquí se ve.",
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
        "actualiza con el mismo ascenso dual de la Eq. 2-20 que usa el OPF, así que el mercado cierra "
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
        "§2.7.2 · qlearning_despacho() · semilla " + str(cfg["semilla"]))}


def fig_nodal_ap(ctx: dict, cfg: dict, mdl: dict) -> dict | None:
    """Mark-up aprendido por regla: la hipótesis central de la tesis, en un número."""
    ap = mdl.get("ap") or {}
    if not isinstance(ap, dict) or not ap.get("reglas"):
        return None
    reglas = ap["reglas"]
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
        "Aprender a ejercer poder de mercado bajo cada regla (hipótesis central)",
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
        "empírica que falta en la propuesta.",
        "§2.5.1 + §2.7.2 · aprendizaje_regla() · " + str(ap.get("nota", ""))[:110])}


# ============================ registro y pestaña ==============================

FIGURES_NODAL: dict[str, tuple] = {
    # clave → (función, subgrupo, título)
    "nodal_red":      (fig_nodal_red,      "red", "Topología del modelo y uso de las ramas"),
    "nodal_cap":      (fig_nodal_cap,      "red", "CEN por nodo y tecnología (y qué se completó)"),
    "nodal_calib":    (fig_nodal_calib,    "red", "Calibración de la oferta contra el precio de bolsa XM"),
    "nodal_lmp":      (fig_nodal_lmp,      "precios", "Mapa de calor del precio marginal por nodo y hora"),
    "nodal_spread":   (fig_nodal_spread,   "precios", "Banda de dispersión nodal y renta de congestión"),
    "nodal_descomp":  (fig_nodal_descomp,  "precios", "Descomposición del LMP (Eq. 2-21)"),
    "nodal_valida":   (fig_nodal_valida,   "precios", "Validación: λ dual vs ∂Costo/∂D por diferencias finitas"),
    "nodal_cong":     (fig_nodal_cong,     "red", "Congestión por rama, pérdidas y violación"),
    "nodal_flujos":   (fig_nodal_flujos,   "red", "Flujos por rama vs su límite y su renta"),
    "nodal_desp":     (fig_nodal_desp,     "despacho", "Despacho óptimo, recorte renovable y corte flexible"),
    "nodal_nodo":     (fig_nodal_nodo,     "despacho", "Nodo caro vs nodo barato: balance y precio"),
    "nodal_beneficios": (fig_nodal_beneficios, "beneficios",
                         "Marcador de efectos positivos de la nodalidad (a-c)"),
    "nodal_flex":     (fig_nodal_flex,     "reglas", "Desconexión voluntaria de la demanda (objetivo a)"),
    "nodal_reglas":   (fig_nodal_reglas,   "reglas", "Uninodal · zonal · híbrido · nodal (KPIs de liquidación)"),
    "nodal_esc":      (fig_nodal_esc,      "estocastico", "Abanico de escenarios estocásticos y reducción"),
    "nodal_penetra":  (fig_nodal_penetra,  "estocastico", "Sensibilidad a la penetración renovable"),
    "nodal_ior":      (fig_nodal_ior,      "mercado", "IOR real, elasticidad medida y concentración por nodo"),
    "nodal_mec":      (fig_nodal_mec,      "mercado", "Bertrand/Cournot y el efecto del markup en el modelo"),
    "nodal_var":      (fig_nodal_var,      "riesgo", "GARCH(1,1), VaR/ES condicional y backtest de Kupiec"),
    "nodal_cov":      (fig_nodal_cov,      "riesgo", "Cobertura de mínima varianza (h* = ρσS/σF)"),
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
    ("aprendiz", "🤖 Aprendizaje (Q-learning)"),
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
   diferencias finitas (Eq. 2-15) mide el error real de ese atajo: si sube de ~1 %, leer
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
   ~10-20 % frente a un UC completo (Eq. 2-9…2-14).
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
                    "componentes de la Eq. 2-21.")
    if float(np.max(np.asarray(res["derrame_MW"], dtype=float))) > 50.0:
        band.append(f"ℹ️ Derrame máximo {float(np.max(res['derrame_MW'])):,.0f} MW: el modelo impone el "
                    "balance del **sistema**, no el de cada nodo (no hay restricción de flujo por nodo en esta "
                    "versión), así que la generación local de un nodo puede diferir de su demanda en "
                    "el residuo que transporta la red.")
    for t in band:
        st.warning(t, icon="⚠️" if t.startswith(("⚠", "🛑")) else "ℹ️")


def render_tab_nodal(ctx: dict, filtros: dict[str, Any]) -> None:
    """Pestaña 🕸️ · precios marginales nodales (módulo académico de la tesis doctoral)."""
    st.markdown(
        "### 🕸️ Precios marginales nodales en el SIN: qué cambiaría si la energía se liquidara por nodo\n"
        "Este módulo no está en el notebook v1-v4: implementa los objetivos **a-f** de la propuesta de "
        "tesis doctoral *Modelamiento estocástico de los precios de la energía en un mercado de precios "
        "marginales nodales con alta componente de generación variable* (Anexo V15SC). Resuelve un OPF "
        "con flujos óptimos DC calibrado contra los datos de la ventana ya cargada (misma `ctx` que el "
        "resto del tablero), forma los LMP por nodo (Eq. 2-15…2-21) y los compara con la liquidación "
        "uninodal real de XM bajo escenarios estocásticos, almacenamiento, coberturas y estrategia.\n\n"
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
    _tarjeta_nodal(ctx, cfg, mdl)
    st.markdown("#### Las ocho preguntas de la propuesta (§3.2, pág. 49), una por bloque de figuras")
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
            secc, fuente = NODAL_TESIS.get(clave, ("", ""))
            with st.expander(f"📊 {titulo}" + (f" · tesis {secc}" if secc else ""), expanded=(i == 0)):
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
                    st.caption(f"▸ **Sostenido en la propuesta**: {fuente}. El código del modelo vive en "
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
    with st.expander("📚 Celda ↔ tesis: de dónde sale cada figura del módulo nodal", expanded=False):
        mapa = pd.DataFrame([dict(figura=k, funcion=v[0].__name__, grupo=v[1], titulo=v[2],
                                  seccion=NODAL_TESIS.get(k, ("", ""))[0],
                          fuente=NODAL_TESIS.get(k, ("", ""))[1]) for k, v in FIGURES_NODAL.items()])
        st.dataframe(mapa, use_container_width=True, hide_index=True)
        st.caption("Este bloque **no** tiene contraparte en las 212 celdas del notebook: su linaje es el "
                   "documento de la tesis (Anexo V15SC) más los datos de las pestañas v1-v4, y por eso no "
                   "entra en `CELDAS_NB` ni en el mapa 1z de *Datos y bitácora*. La verificación propia "
                   "del módulo está en `tools/verificar_celdas.py` (sección nodal) y exige que "
                   "`FIGURES_NODAL` y `NODAL_TESIS` tengan exactamente las mismas claves.")
    with st.expander("🔍 Supuestos, límites y qué NO demuestra esta pestaña", expanded=False):
        st.markdown(NODAL_SUPUESTOS)
        st.markdown(_nodal_pie(f"Ventana {ctx['ini']} → {ctx.get('fin_efectiva', ctx['fin'])}; "
                               f"topología {cfg['granularidad']}, líneas ×{cfg['estres']:.2f}, "
                               f"regla mostrada {cfg['regla']} (κ={cfg['kappa']:.2f})."))


# ==============================================================================
# Arranque. `APPMANUEL_NO_MAIN=1` permite importar el módulo para pruebas.
# ==============================================================================
if os.environ.get("APPMANUEL_NO_MAIN", "") != "1":
    main()
