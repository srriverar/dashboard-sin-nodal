"""Vista histórica de GitHub Pages; reutiliza las figuras y el OPF de app.py."""
import os

import pandas as pd
import streamlit as st

# app.py se importa como biblioteca. Esta vista nunca arranca su modo operativo.
os.environ["APPMANUEL_NO_MAIN"] = "1"
import app as sin  # noqa: E402


ESCENARIOS = {
    "base": "Base · capacidad nominal",
    "estres-moderado": "Estrés moderado · 70 % de la capacidad",
    "estres-severo": "Estrés severo · 40 % de la capacidad",
    "el-nino": "El Niño · ajustes por fase histórica",
}


def render_nodal(ctx: dict) -> None:
    st.subheader("Simulación académica de precios nodales")
    st.caption("Se usan perfiles horarios y capacidades nominales, no mediciones actuales de XM. "
               "Los resultados son simulados; no sirven para liquidación ni pronóstico.")
    with st.form("escenario_nodal"):
        escenario = st.selectbox("Escenario", ESCENARIOS, format_func=ESCENARIOS.__getitem__)
        regla = st.selectbox("Regla de liquidación", ["nodal", "zonal", "hibrido", "uninodal"])
        if st.form_submit_button("Simular escenario"):
            st.session_state["escenario_simulado"] = (escenario, regla)

    if "escenario_simulado" not in st.session_state:
        return
    escenario, regla = st.session_state["escenario_simulado"]
    # Solo el despacho determinista: los módulos adicionales quedan en la versión con servidor.
    cfg = {**sin.NODAL_PRESETS[escenario]["cfg"], "regla": regla,
           "opf_calib": False, "refuerzos": False, "valida_lmp": False,
           "riesgo": False, "mercado": False, "sensibilidad": False,
           "aprendizaje": False, "escenarios": False, "two_stage": False,
           "almacenar": False, "barrido_flex": False}
    with st.spinner("Calculando el despacho nodal…"):
        modelo = sin._mdl(ctx, cfg)
    st.caption(f"Resultado: **{ESCENARIOS[escenario]}** · regla **{regla}**")
    if not modelo["res"]["converged"]:
        st.warning("El OPF no convergió; interpreta los valores como aproximaciones.")
    for clave in ("nodal_lmp", "nodal_desp"):
        figura, _, titulo = sin.FIGURES_NODAL[clave]
        st.markdown(f"#### {titulo}")
        # Algunas leyendas de la app con servidor citan ventanas horarias de XM;
        # en Pages la fuente de ambas figuras es exclusivamente el escenario nominal.
        resultado = figura(ctx, modelo["cfg"], modelo)
        if resultado:
            st.plotly_chart(resultado["fig"], width="stretch", key=clave)


def main() -> None:
    st.set_page_config(page_title="Agua, precio y El Niño · SIN Colombia", page_icon="🌊", layout="wide")
    st.title("🌊 Agua, precio y El Niño — el SIN colombiano bajo sequía")
    st.info("**GitHub Pages · modo de referencia.** El tablero corre en este navegador con los CSV "
            "históricos incluidos en el repositorio. No descarga datos en vivo de XM ni de NOAA. "
            "Para consultar las series horarias y el módulo nodal completo, usa la "
            "[versión Streamlit con servidor](https://github.com/srriverar/dashboard-sin-nodal#2--arranque-local-windows-cmd).")

    ctx = sin.obtener_contexto_cached("pages", sin.VENTANA_POR_DEFECTO, "{}", "offline")
    nino = ctx["nino"]
    oni = nino.dropna(subset=["ONI"])
    st.caption(f"Último ONI disponible: **{oni['Mes'].max()}** · "
               f"Último mes en la serie económica: **{ctx['v4_serie']['Mes'].max()}**.")
    # En una edición histórica los gráficos no se desplazan a medida que cambia la fecha de visita.
    ctx["fin_efectiva"] = pd.Period(nino["Mes"].max(), freq="M").end_time.date()
    ctx["historial_completo"] = st.sidebar.toggle("Ejes con todo el histórico")

    clima, economia, nodal, datos = st.tabs(["🌊 El Niño", "💰 Economía ENSO", "🕸️ Precios nodales", "📚 Datos"])
    with clima:
        st.subheader("Historia de El Niño y La Niña")
        sin.render_resultado(sin.fig_v3_09_dashboard_enso(ctx), "Panel histórico ENSO")
        st.markdown("#### Episodios detectados en la serie ONI")
        st.dataframe(ctx["nino_eventos"], width="stretch", hide_index=True)
    with economia:
        st.subheader("El Niño en dinero")
        sin.render_resultado(sin.fig_v4_05_enso_economico(ctx), "Costo marginal mensual por fase ENSO")
        st.dataframe(ctx["_v4_tabla_impacto"], width="stretch", hide_index=True)
    with nodal:
        render_nodal(ctx)
    with datos:
        st.subheader("Datos de referencia")
        for archivo in sorted(sin.REF_DIR.glob("*.csv")):
            st.download_button(f"Descargar {archivo.name}", archivo.read_bytes(),
                               file_name=archivo.name, mime="text/csv")
        st.caption("Fuentes: XM (Bienda) y NOAA/CPC. Prototipo académico independiente.")
    st.caption("Autores: Manuel Fernando Fajardo Rodríguez, Sergio Rivera y Libardo Acero García.")


main()
