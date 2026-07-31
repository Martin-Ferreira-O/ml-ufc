"""App Streamlit: el modelo, el mercado, y cuánto confiar en la diferencia."""

import pandas as pd
import streamlit as st

import betano
import cartelera
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
    "le gana (log loss 0.612 vs 0.661 sobre 5773 peleas), así que esto es una segunda "
    "opinión independiente, no un detector de valor: apostar todo su EV positivo rindió "
    "−5.2% de ROI. Lo único que no perdió en el backtest es apostar cuando el modelo "
    "coincide con la casa y aun así encuentra EV — eso es lo que se marca como candidata.")


@st.cache_resource
def _cargar():
    return predict.cargar(), predict.peleadores()


@st.cache_data(ttl=3600, show_spinner="Buscando las carteleras…")
def _carteleras():
    return cartelera.proximas()


# El TTL es tambien el rate limit contra Betano: una request cada media hora.
@st.cache_data(ttl=1800, show_spinner="Buscando las cuotas en Betano…")
def _cuotas_betano():
    return betano.cuotas()


def _resultado(a, b, r, cuotas):
    """Las probabilidades de una pelea. Igual en el matchup suelto y en la cartelera."""
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
        st.caption(f"EV del modelo — {a} {r['ev_a']:+.0%} · {b} {r['ev_b']:+.0%} "
                   "(probabilidad × cuota − 1, con el margen de la casa adentro).")
        if r["apuesta"]:
            quien, ev = ((a, r["ev_a"]) if r["apuesta"] == "a" else (b, r["ev_b"]))
            st.success(
                f"**Candidata a valor: {quien} ({ev:+.0%} de EV).** Único tramo que no "
                f"pierde medido: ROI {predict.ROI_APOSTABLE}. El IC95% cruza el cero, "
                "así que es break-even con esperanza, no una ventaja probada — "
                "stake chico y plano.")
        elif max(r["ev_a"], r["ev_b"]) > 0:
            st.caption("Hay EV positivo en el papel, pero en este tramo de discrepancia "
                       "el flat-bet rindió −7% medido: el EV sale de que el modelo se "
                       "aparta del mercado, y ahí el que se equivoca es el modelo.")


(modelo, estado), nombres = _cargar()
tab_matchup, tab_cartelera = st.tabs(["Matchup", "Cartelera"])

with tab_matchup:
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
                _resultado(a, b, r, cuotas)

            st.subheader("Qué mueve la predicción del modelo")
            st.caption(f"Aporte de cada feature al logit de {a}. Positivo lo favorece a "
                       "él, negativo a su rival.")
            factores = pd.DataFrame(r["factores"], columns=["factor", "aporte"])
            factores["factor"] = factores["factor"].map(lambda k: ETIQUETAS.get(k, k))
            st.bar_chart(factores, x="factor", y="aporte", horizontal=True,
                         x_label="aporte al logit", y_label="")

with tab_cartelera:
    st.caption("Las peleas anunciadas, de la API pública de ESPN, con la cuota de Betano "
               "cuando ya está publicada. Betano abre mercado unos días antes del evento, "
               "así que las carteleras lejanas van a aparecer sin cuota.")
    eventos = _carteleras()
    if not eventos:
        st.warning("No hay carteleras anunciadas (o la API de ESPN no respondió).")
    else:
        tabla = _cuotas_betano()
        if not tabla:
            st.warning("Betano no respondió: la cartelera va sin cuotas ni confianza.")
        evento = st.selectbox("Evento", eventos,
                              format_func=lambda e: f"{e['fecha']} — {e['evento']}")
        cuotas_de = [betano.buscar(tabla, p["a"], p["b"]) for p in evento["peleas"]]
        # Si no hay ninguna, el motivo es el evento entero, no cada pelea: decirlo una
        # vez evita que parezca que el matcheo falló pelea por pelea.
        if tabla and not any(cuotas_de):
            st.info("Betano todavía no abrió mercado para este evento — lo hace unos días "
                    "antes. Va sin cuotas ni nivel de confianza.")
        for i, (pelea, cuotas) in enumerate(zip(evento["peleas"], cuotas_de)):
            with st.container(border=True):
                st.subheader(f"{pelea['a']} vs {pelea['b']}", divider="gray")
                st.caption(pelea["peso"] + (" · main event" if i == 0 else ""))
                # La cuota va antes de predecir: en un debut el modelo no tiene nada que
                # decir, pero el precio del mercado sirve igual y es lo único que hay.
                if cuotas:
                    st.caption(f"Betano — {pelea['a']} {cuotas[0]:.2f} · "
                               f"{pelea['b']} {cuotas[1]:.2f}")
                elif any(cuotas_de):
                    st.caption("Betano no publicó la cuota de esta pelea todavía.")
                r = cartelera.predecir(pelea, modelo, estado, cuotas)
                if "error" in r:
                    st.info(f"{r['error']} — el modelo no puede predecir un debut.")
                    continue
                _resultado(pelea["a"], pelea["b"], r, cuotas)
