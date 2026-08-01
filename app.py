"""App Streamlit: el modelo, el mercado, y cuánto confiar en la diferencia.

Este archivo es solo el armazón: encabezado, el sidebar del pipeline y las cuatro
pestañas. Cada pestaña vive en `ufc/ui/tab_*.py` y lo que comparten en `ufc/ui/comunes.py`.
"""

import subprocess
import sys

import streamlit as st

from ufc import rutas
from ufc.ui import comunes, tab_cartelera, tab_historial, tab_matchup, tab_predictores

# El pipeline del README, en orden. fetch baja los CSVs, wiki y sherdog los completan,
# features construye la tabla y train re-entrena el modelo que la app carga.
PIPELINE = ["ufc.datos.fetch", "ufc.datos.wiki", "ufc.datos.sherdog",
            "ufc.modelo.features", "ufc.modelo.train"]

st.set_page_config(page_title="Predictor UFC", page_icon=":material/sports_mma:",
                   layout="wide")
st.title("Predictor de peleas UFC")
st.caption("Una segunda opinión sobre la cuota de la casa, no un detector de valor.")
# El parrafo entero es la advertencia mas importante de la app, pero leerlo una vez
# alcanza: arriba de todo tapaba la pantalla en cada visita.
with st.expander("Cómo funciona esto", icon=":material/help:"):
    st.markdown(
        "El modelo aprende del historial de ufcstats.com. Está medido que la cuota de la "
        "casa le gana (log loss 0.611 vs 0.661 sobre 5679 peleas), así que esto es una "
        "segunda opinión independiente, no un detector de valor: apostar todo su EV "
        "positivo rindió −5.2% de ROI. Lo único que no perdió en el backtest es apostar "
        "cuando el modelo coincide con la casa y aun así encuentra EV — eso es lo que se "
        "marca como candidata.")

with st.sidebar:
    st.subheader("Datos")
    st.caption("El pipeline completo, en orden. Tarda unos minutos: wiki y sherdog "
               "están cacheados y solo piden lo nuevo, features y train no.")
    if st.button("Actualizar y re-entrenar", icon=":material/refresh:"):
        with st.status("Actualizando…", expanded=True) as estado_run:
            for script in PIPELINE:
                st.write(f"`{script}`")
                # -u: sin esto Python bufferea la salida y el script parece colgado.
                p = subprocess.Popen([sys.executable, "-u", "-m", script],
                                     cwd=rutas.RAIZ, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
                hueco, lineas = st.empty(), []
                for linea in p.stdout:
                    lineas.append(linea.rstrip())
                    hueco.code("\n".join(lineas[-12:]))
                if p.wait():
                    estado_run.update(label=f"Falló {script}", state="error")
                    st.stop()
            estado_run.update(label="Listo — modelo re-entrenado", state="complete")
        st.cache_data.clear()
        st.cache_resource.clear()

(modelo, estado), nombres = comunes.cargar()
tab_m, tab_c, tab_p, tab_h = st.tabs(
    ["Matchup", "Cartelera", "Predictores", "Historial"])

with tab_m:
    tab_matchup.render(modelo, estado, nombres)

with tab_c:
    tab_cartelera.render(modelo, estado)

with tab_p:
    tab_predictores.render(modelo, estado)

with tab_h:
    tab_historial.render()
