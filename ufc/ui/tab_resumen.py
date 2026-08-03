"""Resumen operativo: proximo evento, calidad de datos y dinero real."""

import pandas as pd
import streamlit as st

from ufc.datos import betano, cartelera
from ufc.registro import apuestas, predictores
from ufc.ui import comunes


def _clp(valor):
    return f"${valor:,.0f}".replace(",", ".")


def render(modelo, estado):
    st.header("Resumen")
    st.caption("La próxima cartelera, las señales revisadas y tu rendimiento real en un lugar.")
    eventos = comunes.carteleras()
    picks = predictores.leer()
    pendientes = int((~picks["revisado"]).sum()) if len(picks) else 0
    df_apuestas, _, resumen = apuestas.evaluar()

    with st.container(horizontal=True):
        st.metric("Picks revisadas", int(picks["revisado"].sum()) if len(picks) else 0,
                  border=True)
        st.metric("Pendientes de revisar", pendientes, border=True,
                  help="No afectan la precisión ni el ranking hasta que las confirmes.")
        st.metric("Beneficio real", _clp(resumen.get("beneficio", 0)), border=True,
                  delta=(f"{resumen['roi']:+.1%} ROI" if resumen and
                         pd.notna(resumen["roi"]) else None))
        st.metric("Dinero pendiente", _clp(resumen.get("pendiente", 0)), border=True)

    if eventos:
        evento = eventos[0]
        st.subheader(f"Próximo evento · {evento['evento']}")
        st.caption(evento["fecha"])
        tabla = comunes.cuotas_betano()
        cuotas = [betano.buscar(tabla, p["a"], p["b"]) for p in evento["peleas"]]
        preds = [cartelera.predecir(p, modelo, estado, c)
                 for p, c in zip(evento["peleas"], cuotas)]
        rank = predictores.ranking(evento["peleas"], predictores.leer(evento["evento"]),
                                   preds, cuotas)
        fuertes = rank[rank["fuerte"]] if len(rank) else rank
        if len(fuertes):
            st.dataframe(fuertes[["pelea", "seleccion", "apoyo", "votos", "predictores",
                                  "modelo_confirma", "mercado_confirma", "cuota"]],
                         hide_index=True, column_config={
                             "pelea": st.column_config.TextColumn("Pelea", pinned=True),
                             "seleccion": st.column_config.TextColumn("Selección"),
                             "apoyo": st.column_config.ProgressColumn("Apoyo", format="percent",
                                                                        min_value=0, max_value=1),
                             "votos": st.column_config.NumberColumn("Votos"),
                             "predictores": st.column_config.NumberColumn("Cargados"),
                             "modelo_confirma": st.column_config.CheckboxColumn("Modelo"),
                             "mercado_confirma": st.column_config.CheckboxColumn("Mercado"),
                             "cuota": st.column_config.NumberColumn("Cuota", format="%.2f"),
                         })
        else:
            st.info("No hay señales fuertes revisadas para el próximo evento.",
                    icon=":material/pending_actions:")
    else:
        st.warning("No se pudo cargar la próxima cartelera.")

    confianza = predictores.confiabilidad()
    if len(confianza):
        st.subheader("Predictores")
        st.dataframe(confianza, hide_index=True, column_config={
            "predictor": st.column_config.TextColumn("Predictor", pinned=True),
            "aciertos": st.column_config.NumberColumn("Aciertos"),
            "total": st.column_config.NumberColumn("Resultados"),
            "carteleras": st.column_config.NumberColumn("Carteleras"),
            "acierto": st.column_config.ProgressColumn("Precisión", format="percent",
                                                         min_value=0, max_value=1),
            "peso": st.column_config.NumberColumn("Peso conservador", format="percent"),
        })

    evolucion = apuestas.evolucion(df_apuestas)
    if len(evolucion):
        st.subheader("Beneficio acumulado")
        st.line_chart(evolucion, x="fecha", y="beneficio acumulado", y_label="CLP")
