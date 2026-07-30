"""App Streamlit: el modelo, el mercado, y cuánto confiar en la diferencia."""

import pandas as pd
import streamlit as st

import predict

# Los nombres internos de las features no le dicen nada a nadie, y el punto de mostrar
# los factores es que se entiendan.
ETIQUETAS = {
    "elo": "Elo", "n_fights": "peleas en UFC", "win_rate": "% de victorias",
    "streak": "racha", "slpm": "golpes conectados por minuto",
    "sapm": "golpes recibidos por minuto", "str_acc": "precisión de golpeo",
    "td_per15": "derribos por 15 min", "td_acc": "precisión de derribo",
    "sub_per15": "intentos de sumisión por 15 min", "ctrl_per_min": "control en el suelo",
    "finish_rate": "% de finalizaciones", "days_since_last": "días desde la última pelea",
    "age": "edad", "height_in": "altura", "reach_in": "alcance",
    "kd_per15": "knockdowns por 15 min", "kd_against_per15": "knockdowns recibidos",
    "finished_against_rate": "% de veces finalizado", "td_def": "defensa de derribo",
    "str_def": "defensa de golpeo", "avg_opp_elo": "calidad de los rivales",
    "mkt_logit": "cuota del mercado",
}
ESTILO = {"alta": st.success, "media": st.warning, "baja": st.error}

st.set_page_config(page_title="Predictor UFC", page_icon=":material/sports_mma:")
st.title("Predictor de peleas UFC")
st.caption(
    "El modelo aprende del historial de ufcstats.com. Está medido que la cuota de la casa "
    "le gana (log loss 0.612 vs 0.661 sobre 5773 peleas), así que esto no sirve para "
    "encontrar valor: sirve como segunda opinión independiente. Cuando el modelo y la "
    "casa coinciden, la estimación es sólida.")


@st.cache_resource
def _cargar():
    return predict.cargar(), predict.peleadores()


(modelo, estado), nombres = _cargar()

with st.form("matchup", border=False):
    with st.container(horizontal=True):
        a = st.selectbox("Peleador A", nombres, index=None, placeholder="Buscá un nombre…")
        b = st.selectbox("Peleador B", nombres, index=None, placeholder="Buscá un nombre…")
    with st.container(horizontal=True):
        cuota_a = st.number_input("Cuota decimal de A", min_value=1.01, value=None,
                                  step=0.05, placeholder="ej. 1.50")
        cuota_b = st.number_input("Cuota decimal de B", min_value=1.01, value=None,
                                  step=0.05, placeholder="ej. 2.60")
    st.caption("Sin las dos cuotas no hay nivel de confianza: la coincidencia con el "
               "mercado es lo único que resultó predecir cuándo el modelo se equivoca.")
    enviado = st.form_submit_button("Predecir", type="primary",
                                    icon=":material/query_stats:")

if enviado:
    if not (a and b):
        st.warning("Elegí los dos peleadores.")
    elif a == b:
        st.warning("Elegí dos peleadores distintos.")
    else:
        cuotas = (cuota_a, cuota_b) if cuota_a and cuota_b else None
        r = predict.predict(a, b, modelo, estado, cuotas=cuotas)

        with st.container(border=True):
            st.subheader(f"{a} vs {b}", divider="gray")
            with st.container(horizontal=True):
                st.metric(f"Modelo — {a}", f"{r['p_a']:.1%}",
                          help="No usa la cuota. Es la opinión independiente.")
                if cuotas:
                    st.metric(f"Mercado — {a}", f"{r['p_a_mercado']:.1%}",
                              help="Probabilidad implícita sin el margen de la casa.")
                    st.metric(f"Modelo con cuota — {a}", f"{r['p_a_con_odds']:.1%}",
                              delta=f"{r['p_a_con_odds'] - r['p_a_mercado']:+.1%} vs mercado",
                              help="Medido: no le gana al mercado solo. Está para ver "
                                   "cuánto lo mueve el historial, no para apostarlo.")
            st.progress(r["p_a"], text=f"{a} — {r['p_a']:.1%}")
            st.progress(r["p_b"], text=f"{b} — {r['p_b']:.1%}")

        if cuotas:
            ESTILO[r["confianza"]](f"**Confianza {r['confianza']}.** {r['motivo']}")

        st.subheader("Qué mueve la predicción del modelo")
        st.caption(f"Aporte de cada feature al logit de {a}. Positivo lo favorece a él, "
                   "negativo a su rival.")
        factores = pd.DataFrame(r["factores"], columns=["factor", "aporte"])
        factores["factor"] = factores["factor"].map(lambda k: ETIQUETAS.get(k, k))
        st.bar_chart(factores, x="factor", y="aporte", horizontal=True,
                     x_label="aporte al logit", y_label="")
