"""Lo que comparten las cuatro pestanias: constantes de presentacion, los
wrappers cacheados contra las fuentes, y los dos bloques que se dibujan igual en mas
de una pestania.

Los wrappers viven aca y no en cada pestania porque el cache de Streamlit es por
objeto funcion: `carteleras()` la llaman Cartelera y Predictores, y si cada una
tuviera la suya serian dos requests a ESPN.
"""

import pandas as pd
import requests
import streamlit as st

from ufc.datos import betano, cartelera, oddsapi
from ufc.modelo import predict
from ufc.registro import predictores

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
# Emoji y no un badge de markdown porque en la celda de un dataframe el markdown sale
# crudo: solo se renderiza en el overlay que aparece al clickearla.
BADGE = {"alta": "🟢 alta", "media": "🟡 media", "baja": "🔴 baja"}
# Las columnas del ledger son internas ("p_a", "gano" = "a"/"b"): sin esto la tabla
# hay que decodificarla mirando otras dos columnas.
COLUMNAS = {
    "fecha": st.column_config.DateColumn("Fecha", format="DD MMM YYYY"),
    "pelea": st.column_config.TextColumn("Pelea", pinned=True),
    "seguido": st.column_config.TextColumn(
        "Lado seguido", help="La candidata a valor si la hubo; si no, el lado de mayor "
                             "EV — hipotético, para poder medirlo igual."),
    "p_modelo": st.column_config.ProgressColumn(
        "Modelo", format="percent", min_value=0, max_value=1,
        help="Probabilidad que le da el modelo a ese lado, sin mirar la cuota."),
    "p_mercado": st.column_config.ProgressColumn(
        "Mercado", format="percent", min_value=0, max_value=1,
        help="Probabilidad implícita en la cuota, sin el margen de la casa."),
    "cuota": st.column_config.NumberColumn("Cuota", format="%.2f",
                                           help="La cuota congelada al registrar."),
    "confianza": st.column_config.TextColumn(
        "Confianza", help="Qué tanto coincide el modelo con el mercado. Alta es la "
                          "única que habilita una candidata."),
    "candidata": st.column_config.CheckboxColumn(
        "Candidata", disabled=True, help="El único tramo que no perdió en el backtest."),
    "ganador": st.column_config.TextColumn("Ganador"),
    "acerto": st.column_config.CheckboxColumn("Acertó", disabled=True),
    "clv": st.column_config.NumberColumn(
        "CLV", format="percent",
        help="Cuota congelada vs cuota de cierre. Positivo = le ganaste al cierre."),
    "retorno": st.column_config.NumberColumn(
        "Retorno", format="%+.2f u", help="Flat-bet de 1 unidad en ese lado."),
}
# La comparativa: las columnas fijas. Las de cada predictor se agregan al vuelo, porque
# dependen de quien haya subido picks para ese evento.
COLUMNAS_COMP = {
    "pelea": st.column_config.TextColumn("Pelea", pinned=True),
    "modelo": st.column_config.TextColumn(
        "Modelo", help="El favorito del modelo. Vacío en un debut: no tiene historial."),
    "p_modelo": st.column_config.ProgressColumn(
        "Prob.", format="percent", min_value=0, max_value=1,
        help="Cuánta probabilidad le da el modelo a su propio favorito."),
    "consenso": st.column_config.TextColumn(
        "Consenso", help="El peleador que eligieron todos los predictores. Vacío si "
                         "hay desacuerdo o si a alguno le falta la pick."),
    "cuota": st.column_config.NumberColumn(
        "Cuota", format="%.2f", help="La cuota de Betano del lado del consenso."),
    "ev": st.column_config.NumberColumn(
        "EV modelo", format="percent",
        help="EV que le da el modelo al lado del consenso (probabilidad × cuota − 1)."),
    "senal": st.column_config.TextColumn("Señal"),
}


@st.cache_resource
def cargar():
    return predict.cargar(), predict.peleadores()


@st.cache_data(ttl=3600, show_spinner="Buscando las carteleras…")
def carteleras():
    try:
        return cartelera.proximas()
    except requests.RequestException:
        return []  # el warning de la pestania ya explica que ESPN no respondio


# El TTL es tambien el rate limit contra Betano: una request cada media hora.
@st.cache_data(ttl=1800, show_spinner="Buscando las cuotas en Betano…")
def cuotas_betano():
    return betano.cuotas()


# 30 min de cache tambien alcanza para el tier gratis de The Odds API (500/mes).
@st.cache_data(ttl=1800, show_spinner="Buscando el consenso multi-casa…")
def consenso():
    return oddsapi.cuotas()



def resultado(a, b, r, cuotas):
    """Las probabilidades de una pelea. Igual en el matchup suelto y en la cartelera."""
    if "aviso" in r:
        st.warning(f"**Historial mezclado.** {r['aviso']}")
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
    st.caption(f"Cuota mínima para que haya valor según el modelo — {a} "
               f"**{1 / r['p_a']:.2f}** · {b} **{1 / r['p_b']:.2f}**. Sirve para "
               "comparar contra cualquier casa, no solo Betano.")
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


def historial(df):
    """El ledger con nombres de peleador en vez de "a"/"b". Una fila = un solo lado."""
    es_a = df["lado"] == "a"
    return pd.DataFrame({
        "pelea": df["a"] + " vs " + df["b"],
        "fecha": df["fecha_evento"],
        "seguido": df["a"].where(es_a, df["b"]),
        "p_modelo": df["p_a"].where(es_a, 1 - df["p_a"]),
        "p_mercado": df["p_mercado"].where(es_a, 1 - df["p_mercado"]),
        "cuota": df["cuota_lado"],
        "confianza": df["confianza"].map(BADGE),
        "candidata": df["apuesta"].fillna("") != "",
        "ganador": df["a"].where(df["gano"] == "a", df["b"]).where(df["gano"].notna()),
        "acerto": df["gano"] == df["lado"],
        "clv": df["clv"],
        "retorno": df["retorno"],
    })


@st.cache_data(show_spinner="Leyendo la imagen…")
def leer_imagen(datos, mime, pares):
    """La imagen del tipster -> sus picks. Cacheada porque si no, cada tecla que tocás
    en la tabla de abajo es un rerun, y cada rerun sería otro llamado a la API."""
    return predictores.extraer(datos, mime, [{"a": a, "b": b} for a, b in pares])

