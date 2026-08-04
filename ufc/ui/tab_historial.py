"""Seguimiento hipotetico de cada prediccion congelada."""

import pandas as pd
import streamlit as st

from ufc.registro import ledger
from ufc.ui import comunes


def render():
    df_ledger, resumen = ledger.evaluar()
    chips = []
    if not df_ledger.empty:
        chips.append((f"{resumen['predicciones']} congeladas", "blue",
                      ":material/ac_unit:"))
        if resumen["con_resultado"]:
            chips.append((f"{resumen['con_resultado']} resueltas", "green",
                          ":material/task_alt:"))
    comunes.encabezado(
        "Seguimiento del modelo",
        "Cada pelea que apareció con cuota queda congelada acá con la predicción y la "
        "cuota de ese momento. Cuando la pelea ocurre se cruza con el resultado y con la "
        "cuota final. Conseguir mejores precios de forma sostenida es una señal útil, "
        "pero esta sección no representa apuestas reales.",
        seccion="Auditoría", chips=chips)

    if df_ledger.empty:
        st.info("Todavía no hay predicciones registradas. Se van guardando solas al "
                "mirar carteleras con cuota publicada.", icon=":material/ac_unit:")
        return

    with st.container(horizontal=True):
        st.metric("Predicciones", resumen["predicciones"], border=True,
                  help="Peleas congeladas con cuota. Las que ya ocurrieron: "
                       f"{resumen['con_resultado']}.")
        if resumen["con_resultado"]:
            st.metric("Acierto modelo", f"{resumen['acierto_modelo']:.0%}", border=True,
                      delta=f"{resumen['acierto_modelo'] - resumen['acierto_mercado']:+.0%}",
                      delta_description="vs mercado",
                      help="Sobre las peleas ya resueltas: cuántas veces el favorito "
                           "del modelo terminó ganando.")
        st.metric("Candidatas históricas", resumen["candidatas"], border=True,
                  help="Filas creadas por la regla anterior. La regla fue retirada y "
                       "las versiones nuevas no generan candidatas automáticas.")
        if pd.notna(resumen["roi_candidatas"]):
            st.metric("ROI regla retirada", f"{resumen['roi_candidatas']:+.1%}",
                      border=True, help="Resultado histórico de la regla anterior.")
        if pd.notna(resumen["clv_medio"]):
            st.metric("Mejora frente al cierre", f"{resumen['clv_medio']:+.1%}", border=True,
                      help="Cuota registrada vs cuota de cierre del lado seguido. "
                           "Positivo = le ganaste al cierre.")

    tabla = comunes.historial(df_ledger)
    hechas = df_ledger["gano"].notna().to_numpy()
    st.caption("Cada fila muestra **un solo lado** de la pelea: la marca histórica si "
               "existió; si no, el lado con mayor EV puntual para seguimiento. Ninguna "
               "fila nueva es una recomendación de apuesta.")

    # Las dos tablas en pestañas: son la misma tabla en dos estados y antes venían una
    # abajo de la otra, así que la de resueltas quedaba fuera de pantalla.
    etiquetas, vistas = [], []
    if not hechas.all():
        etiquetas.append(f"Sin resolver ({(~hechas).sum()})")
        vistas.append(("pendiente", tabla[~hechas].sort_values("fecha")))
    if hechas.any():
        etiquetas.append(f"Resueltas ({hechas.sum()})")
        vistas.append(("resuelta", tabla[hechas].sort_values("fecha", ascending=False)))
    for pestania, (tipo, datos) in zip(st.tabs(etiquetas), vistas):
        with pestania:
            if tipo == "pendiente":
                # el CLV todavia no existe: el "cierre" de un evento futuro es la cuota de hoy
                datos = datos.drop(columns=["ganador", "acerto", "clv", "retorno"])
            st.dataframe(datos, hide_index=True, column_config=comunes.COLUMNAS)
