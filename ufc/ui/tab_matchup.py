"""Pestania Matchup: dos peleadores a mano, con sus cuotas y circunstancias."""

import pandas as pd
import streamlit as st

from ufc.modelo import predict
from ufc.ui import comunes


def render(modelo, estado, nombres):
    st.header("Matchup")
    with st.form("matchup", border=False):
        with st.container(horizontal=True):
            a = st.selectbox("Peleador A", nombres, index=None,
                             placeholder="Buscá un nombre…")
            b = st.selectbox("Peleador B", nombres, index=None,
                             placeholder="Buscá un nombre…")
        with st.container(horizontal=True):
            cuota_a = st.number_input("Cuota decimal de A", min_value=1.01, value=None,
                                      step=0.05, placeholder="ej. 1.50")
            cuota_b = st.number_input("Cuota decimal de B", min_value=1.01, value=None,
                                      step=0.05, placeholder="ej. 2.60")
        st.caption("Sin las dos cuotas no hay nivel de confianza: la coincidencia con el "
                   "mercado es lo único que resultó predecir cuándo el modelo se equivoca.")
        with st.container(horizontal=True):
            ree_a = st.checkbox("A entró de reemplazo")
            peso_a = st.checkbox("A no dio el peso")
            ree_b = st.checkbox("B entró de reemplazo")
            peso_b = st.checkbox("B no dio el peso")
        st.caption("Circunstancias de esta pelea, no del historial — el modelo no puede "
                   "deducirlas y son noticia pública. Medido sobre 8639 peleas: el que "
                   "entra de reemplazo gana el 39% y el que no da el peso el 41%, contra "
                   "50% de base.")
        enviado = st.form_submit_button("Predecir", type="primary",
                                        icon=":material/query_stats:")

    if enviado:
        if not (a and b):
            st.warning("Elegí los dos peleadores.")
        elif a == b:
            st.warning("Elegí dos peleadores distintos.")
        else:
            cuotas = (cuota_a, cuota_b) if cuota_a and cuota_b else None
            r = predict.predict(a, b, modelo, estado, cuotas=cuotas,
                                circ_a=(ree_a, peso_a), circ_b=(ree_b, peso_b))

            with st.container(border=True):
                st.subheader(f"{a} vs {b}", divider="gray")
                comunes.resultado(a, b, r, cuotas)

            st.subheader("Qué mueve la predicción del modelo")
            st.caption(f"Impacto de cada variable para {a}. Positivo lo favorece; "
                       "negativo favorece a su rival.")
            factores = pd.DataFrame(r["factores"], columns=["factor", "aporte"])
            factores["factor"] = factores["factor"].map(lambda k: comunes.ETIQUETAS.get(k, k))
            st.bar_chart(factores, x="factor", y="aporte", horizontal=True,
                         x_label="impacto", y_label="")
