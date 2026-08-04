"""Router Streamlit del predictor, sus datos revisados y las apuestas reales."""

import subprocess
import sys

import streamlit as st

from ufc import rutas
from ufc.ui import (comunes, tab_apuestas, tab_cartelera, tab_historial, tab_matchup,
                    tab_intel, tab_predictores, tab_resumen)

# El pipeline del README, en orden. fetch baja los CSVs, wiki y sherdog los completan,
# features construye la tabla y train re-entrena el modelo que la app carga.
PIPELINE = ["ufc.datos.fetch", "ufc.datos.wiki", "ufc.datos.sherdog",
            "ufc.modelo.features", "ufc.modelo.train"]

st.set_page_config(page_title="Predictor UFC", page_icon=":material/sports_mma:",
                   layout="wide")
# La marca vive en la cabecera y cada pagina pone su propio titulo con
# `comunes.encabezado`. Antes el titulo global se repetia arriba de cada uno.
st.logo(str(rutas.RAIZ / "assets" / "marca.svg"), size="large")

with st.sidebar:
    st.subheader("Datos")
    st.caption("El pipeline completo, en orden. Tarda unos minutos: wiki y sherdog "
               "están cacheados y solo piden lo nuevo, features y train no.")
    if st.button("Actualizar y re-entrenar", icon=":material/refresh:", type="primary",
                 width="stretch"):
        with st.status("Actualizando…", expanded=True) as estado_run:
            hubo_avisos = False
            for script in PIPELINE:
                st.write(f"`{script}`")
                # -u: sin esto Python bufferea la salida y el script parece colgado.
                p = subprocess.Popen([sys.executable, "-u", "-m", script],
                                     cwd=rutas.RAIZ, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
                hueco, lineas = st.empty(), []
                for linea in p.stdout:
                    lineas.append(linea.rstrip())
                    hubo_avisos |= linea.startswith("AVISO:")
                    hueco.code("\n".join(lineas[-12:]))
                if p.wait():
                    estado_run.update(label=f"Falló {script}", state="error")
                    detalle = next((linea for linea in reversed(lineas)
                                    if linea.strip()), "Sin detalle del proceso")
                    st.error(f"No se pudo completar la actualización. {detalle}")
                    st.stop()
            etiqueta = ("Listo — modelo re-entrenado con copias locales"
                        if hubo_avisos else "Listo — datos actualizados y modelo re-entrenado")
            estado_run.update(label=etiqueta, state="complete")
        st.cache_data.clear()
        st.cache_resource.clear()

(modelo, estado), nombres = comunes.cargar()
manifest = modelo.get("manifest", {})
with st.sidebar:
    if manifest:
        st.subheader("Modelo")
        with st.container(border=True, gap=None):
            st.caption("Entrenado")
            st.markdown(f"**{manifest['trained_at_utc'][:10]}**")
            st.caption(f"Datos hasta {manifest['data']['max_date']}")
            st.caption(f"Commit `{(manifest.get('code_commit') or 'sin commit')[:7]}`")
        st.badge("Sin apuestas automáticas", color="gray",
                 icon=":material/do_not_disturb_on:")
        st.caption("La app no apuesta ni recomienda apostar por su cuenta. Todo lo que "
                   "queda registrado lo confirmás vos.")


def pagina_resumen():
    tab_resumen.render(modelo, estado, PAGINA_CARTELERA)


def pagina_cartelera():
    tab_cartelera.render(modelo, estado, PAGINA_PREDICTORES)


def pagina_predictores():
    tab_predictores.render(modelo, estado)


def pagina_apuestas():
    tab_apuestas.render(modelo, estado)


def pagina_matchup():
    tab_matchup.render(modelo, estado, nombres)


def pagina_seguimiento():
    tab_historial.render()


def pagina_inteligencia():
    tab_intel.render()


PAGINA_PREDICTORES = st.Page(
    pagina_predictores, title="Predictores", icon=":material/groups:",
    url_path="predictores")
PAGINA_CARTELERA = st.Page(pagina_cartelera, title="Cartelera", icon=":material/event:",
                           url_path="cartelera")

pagina = st.navigation([
    st.Page(pagina_resumen, title="Resumen", icon=":material/dashboard:", default=True),
    PAGINA_CARTELERA,
    st.Page(pagina_inteligencia, title="Inteligencia", icon=":material/manage_search:",
            url_path="inteligencia"),
    PAGINA_PREDICTORES,
    st.Page(pagina_apuestas, title="Apuestas", icon=":material/receipt_long:",
            url_path="apuestas"),
    st.Page(pagina_matchup, title="Matchup", icon=":material/compare_arrows:",
            url_path="matchup"),
    st.Page(pagina_seguimiento, title="Seguimiento", icon=":material/query_stats:",
            url_path="seguimiento"),
], position="top")
pagina.run()
