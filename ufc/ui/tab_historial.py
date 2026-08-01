"""Pestania Historial: el forward test — cada prediccion congelada y su CLV."""

import pandas as pd
import streamlit as st

from ufc.registro import ledger
from ufc.ui import comunes


def render():
    st.caption("Cada pelea que apareció con cuota queda congelada acá con la predicción "
               "y la cuota de ese momento. Cuando la pelea ocurre se cruza con el "
               "resultado y con el último tick de Betano (el cierre): el CLV positivo "
               "sostenido es el único predictor confiable de rentabilidad, y converge "
               "en decenas de apuestas, no en miles.")
    df_ledger, resumen = ledger.evaluar()
    if df_ledger.empty:
        st.info("Todavía no hay predicciones registradas. Se van guardando solas al "
                "mirar carteleras con cuota publicada.")
    else:
        with st.container(horizontal=True):
            st.metric("Predicciones", resumen["predicciones"], border=True,
                      help="Peleas congeladas con cuota. Las que ya ocurrieron: "
                           f"{resumen['con_resultado']}.")
            if resumen["con_resultado"]:
                st.metric("Acierto modelo", f"{resumen['acierto_modelo']:.0%}", border=True,
                          delta=f"{resumen['acierto_modelo'] - resumen['acierto_mercado']:+.0%}"
                                " vs mercado",
                          help="Sobre las peleas ya resueltas: cuántas veces el favorito "
                               "del modelo terminó ganando.")
            st.metric("Candidatas", resumen["candidatas"], border=True,
                      help="Peleas donde el modelo coincide con la casa y aun así "
                           "encuentra EV. Es el único tramo que no perdió medido.")
            if pd.notna(resumen["roi_candidatas"]):
                st.metric("ROI candidatas", f"{resumen['roi_candidatas']:+.1%}",
                          border=True, help="Flat-bet 1u en cada candidata ya resuelta.")
            if pd.notna(resumen["clv_medio"]):
                st.metric("CLV medio", f"{resumen['clv_medio']:+.1%}", border=True,
                          help="Cuota congelada vs cuota de cierre del lado apostado. "
                               "Positivo = le ganaste al cierre.")

        tabla = comunes.historial(df_ledger)
        hechas = df_ledger["gano"].notna().to_numpy()
        st.caption("Cada fila muestra **un solo lado** de la pelea: la candidata a valor "
                   "si la hubo, si no el lado de mayor EV. Las cuotas y el retorno son "
                   "siempre de ese lado.")
        if not hechas.all():
            proximas = tabla[~hechas].sort_values("fecha")
            st.subheader(f"Sin resolver ({len(proximas)})", divider="gray")
            # el CLV todavia no existe: el "cierre" de un evento futuro es la cuota de hoy
            st.dataframe(proximas.drop(columns=["ganador", "acerto", "clv", "retorno"]),
                         hide_index=True, column_config=comunes.COLUMNAS)
        if hechas.any():
            resueltas = tabla[hechas].sort_values("fecha", ascending=False)
            st.subheader(f"Resueltas ({len(resueltas)})", divider="gray")
            st.dataframe(resueltas, hide_index=True, column_config=comunes.COLUMNAS)
