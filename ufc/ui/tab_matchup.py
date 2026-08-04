"""Pestania Matchup: dos peleadores a mano, con sus cuotas y circunstancias."""

import datetime

import numpy as np
import pandas as pd
import streamlit as st

from ufc.modelo import predict
from ufc.ui import comunes


ESTADOS_CIRCUNSTANCIA = ["Desconocido", "No", "Sí"]


def _circunstancia(valor):
    return {"Desconocido": np.nan, "No": 0.0, "Sí": 1.0}[valor]


def render(modelo, estado, nombres):
    comunes.encabezado(
        "Matchup", "Cualquier par de peleadores, con la cuota y las circunstancias que "
                   "vos le pongas. No tiene que estar anunciado.",
        seccion="Simulador",
        chips=[(f"{len(nombres)} peleadores en el modelo", "blue",
                ":material/person_search:")])

    with st.form("matchup", border=False):
        # Tres bloques en vez de siete controles seguidos: quiénes pelean, a qué precio,
        # y en qué circunstancias. Las circunstancias van plegadas porque casi siempre
        # se dejan en "Desconocido".
        with st.container(border=True):
            st.subheader("Los peleadores")
            with st.container(horizontal=True):
                a = st.selectbox("Peleador A", nombres, index=None,
                                 placeholder="Buscá un nombre…")
                b = st.selectbox("Peleador B", nombres, index=None,
                                 placeholder="Buscá un nombre…")
            fecha_evento = st.date_input("Fecha programada del evento",
                                         value=datetime.date.today())
            st.caption("La edad y el descanso se calculan en esta fecha, no con el día "
                       "en que abrís la app.")

        with st.container(border=True):
            st.subheader("Las cuotas")
            with st.container(horizontal=True):
                cuota_a = st.number_input("Cuota decimal de A", min_value=1.01, value=None,
                                          step=0.05, placeholder="ej. 1.50")
                cuota_b = st.number_input("Cuota decimal de B", min_value=1.01, value=None,
                                          step=0.05, placeholder="ej. 2.60")
            st.caption("Sin las dos cuotas no hay nivel de coincidencia: compararlas con "
                       "el mercado es lo único que resultó predecir cuándo el modelo se "
                       "equivoca.")

        with st.expander("Circunstancias de esta pelea", icon=":material/tune:"):
            with st.container(horizontal=True):
                ree_a = st.selectbox("A entró de reemplazo", ESTADOS_CIRCUNSTANCIA)
                peso_a = st.selectbox("A no dio el peso", ESTADOS_CIRCUNSTANCIA)
                ree_b = st.selectbox("B entró de reemplazo", ESTADOS_CIRCUNSTANCIA)
                peso_b = st.selectbox("B no dio el peso", ESTADOS_CIRCUNSTANCIA)
            st.caption("Circunstancias de esta pelea, no del historial — el modelo no "
                       "puede deducirlas y son noticia pública. Medido sobre 8639 peleas: "
                       "el que entra de reemplazo gana el 39% y el que no da el peso el "
                       "41%, contra 50% de base.")

        enviado = st.form_submit_button("Predecir", type="primary", width="stretch",
                                        icon=":material/query_stats:")

    if not enviado:
        return
    if not (a and b):
        st.warning("Elegí los dos peleadores.", icon=":material/person_off:")
        return
    if a == b:
        st.warning("Elegí dos peleadores distintos.", icon=":material/person_off:")
        return

    cuotas = (cuota_a, cuota_b) if cuota_a and cuota_b else None
    r = predict.predict(a, b, modelo, estado, cuotas=cuotas, event_date=fecha_evento,
                        circ_a=(_circunstancia(ree_a), _circunstancia(peso_a)),
                        circ_b=(_circunstancia(ree_b), _circunstancia(peso_b)))

    with st.container(border=True):
        st.subheader(f"{a} vs {b}", divider="gray")
        comunes.resultado(a, b, r, cuotas)

    st.subheader("Qué mueve la predicción del modelo")
    st.caption(f"Impacto de cada variable para **{a}**. A la derecha lo favorece; "
               f"a la izquierda favorece a {b}.")
    factores = pd.DataFrame(r["factores"], columns=["factor", "aporte"])
    factores["factor"] = factores["factor"].map(lambda k: comunes.ETIQUETAS.get(k, k))
    # Un solo color: colorear por lado obligaria a pasar el nombre del peleador como
    # categoria, y ahi Streamlit asigna el color por orden alfabetico, no por esquina.
    # El signo de la barra ya dice a quien favorece.
    st.bar_chart(factores, x="factor", y="aporte", horizontal=True,
                 x_label="impacto", y_label="", color=comunes.COLOR_A)
