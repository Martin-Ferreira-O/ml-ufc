"""Resumen operativo: proximo evento, calidad de datos y dinero real."""

import pandas as pd
import streamlit as st

from ufc.datos import betano, cartelera
from ufc.registro import apuestas, predictores
from ufc.ui import comunes


def _clp(valor):
    return f"${valor:,.0f}".replace(",", ".")


def render(modelo, estado, pagina_cartelera=None):
    eventos = comunes.carteleras()
    picks = predictores.leer()
    revisadas = int(picks["revisado"].sum()) if len(picks) else 0
    pendientes = int((~picks["revisado"]).sum()) if len(picks) else 0
    df_apuestas, _, resumen = apuestas.evaluar()
    evolucion = apuestas.evolucion(df_apuestas)

    chips = [(f"{len(eventos)} carteleras anunciadas", "blue", ":material/event:")]
    if pendientes:
        chips.append((f"{pendientes} picks sin revisar", "orange",
                      ":material/rate_review:"))
    if resumen.get("pendiente"):
        chips.append((f"{_clp(resumen['pendiente'])} en juego", "violet",
                      ":material/hourglass_top:"))
    comunes.encabezado(
        "Resumen", "La próxima cartelera, las señales revisadas y tu rendimiento real "
                   "en un solo lugar.", seccion="Panel", chips=chips)

    with st.container(horizontal=True):
        st.metric("Picks revisadas", revisadas, border=True,
                  help="Las que ya confirmaste. Son las únicas que pesan en el ranking.")
        st.metric("Pendientes de revisar", pendientes, border=True,
                  help="No afectan la precisión ni el ranking hasta que las confirmes.")
        st.metric("Beneficio real", _clp(resumen.get("beneficio", 0)), border=True,
                  delta=(f"{resumen['roi']:+.1%} ROI" if resumen and
                         pd.notna(resumen["roi"]) else None),
                  delta_description="sobre lo apostado",
                  # La curva dentro de la tarjeta: el numero solo no dice si viene
                  # subiendo o si fue un golpe de suerte hace tres meses.
                  chart_data=(evolucion["beneficio acumulado"]
                              if len(evolucion) > 1 else None), chart_type="area")
        st.metric("Dinero pendiente", _clp(resumen.get("pendiente", 0)), border=True,
                  help="Apostado en peleas que todavía no se liquidaron.")

    if not eventos:
        st.warning("No se pudo cargar la próxima cartelera. Suele ser la API de ESPN "
                   "sin responder; probá de nuevo en un rato.",
                   icon=":material/cloud_off:")
    else:
        evento = eventos[0]
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute",
                              vertical_alignment="center"):
                st.badge("Próximo evento", icon=":material/local_fire_department:",
                         color="orange")
                if falta := comunes.cuenta_regresiva(evento["fecha"]):
                    st.badge(falta, icon=":material/schedule:", color="blue")
            st.header(evento["evento"])
            n = len(evento["peleas"])
            st.caption(f"{evento['fecha']} · {n} {'pelea' if n == 1 else 'peleas'}")

            tabla = comunes.cuotas_betano()
            cuotas = [betano.buscar(tabla, p["a"], p["b"]) for p in evento["peleas"]]
            preds = [cartelera.predecir(p, modelo, estado, c)
                     for p, c in zip(evento["peleas"], cuotas)]
            rank = predictores.ranking(evento["peleas"],
                                       predictores.leer(evento["evento"]), preds, cuotas)
            fuertes = rank[rank["fuerte"]] if len(rank) else rank
            if len(fuertes):
                st.caption(f"**{len(fuertes)} señales fuertes** — apoyo humano de 66.7% "
                           "o más, confirmado por el modelo o por el mercado.")
                st.dataframe(fuertes[["pelea", "seleccion", "apoyo", "votos", "predictores",
                                      "modelo_confirma", "mercado_confirma", "cuota"]],
                             hide_index=True, column_config={
                                 "pelea": st.column_config.TextColumn("Pelea", pinned=True),
                                 "seleccion": st.column_config.TextColumn("Selección"),
                                 "apoyo": st.column_config.ProgressColumn(
                                     "Apoyo humano", format="percent",
                                     min_value=0, max_value=1),
                                 "votos": st.column_config.NumberColumn("Votos"),
                                 "predictores": st.column_config.NumberColumn("Cargados"),
                                 "modelo_confirma": st.column_config.CheckboxColumn("Modelo"),
                                 "mercado_confirma": st.column_config.CheckboxColumn("Mercado"),
                                 "cuota": st.column_config.NumberColumn("Cuota", format="%.2f"),
                             })
            else:
                st.info("Todavía no hay señales fuertes revisadas para este evento.",
                        icon=":material/pending_actions:")
            if pagina_cartelera is not None:
                st.page_link(pagina_cartelera, label="Ver la cartelera completa",
                             icon=":material/arrow_forward:")

    confianza = predictores.confiabilidad()
    if len(confianza):
        st.subheader("Predictores")
        st.caption("Precisión medida sobre resultados ya confirmados. El peso conservador "
                   "achica el efecto de las muestras cortas.")
        st.dataframe(confianza, hide_index=True, column_config={
            "predictor": st.column_config.TextColumn("Predictor", pinned=True),
            "aciertos": st.column_config.NumberColumn("Aciertos"),
            "total": st.column_config.NumberColumn("Resultados"),
            "carteleras": st.column_config.NumberColumn("Carteleras"),
            "acierto": st.column_config.ProgressColumn("Precisión", format="percent",
                                                       min_value=0, max_value=1),
            "peso": st.column_config.NumberColumn("Peso conservador", format="percent"),
        })

    if len(evolucion):
        st.subheader("Beneficio acumulado")
        st.caption("Solo apuestas reales confirmadas por vos, liquidadas en CLP.")
        st.area_chart(evolucion, x="fecha", y="beneficio acumulado", y_label="CLP",
                      color="#60A5FA")
