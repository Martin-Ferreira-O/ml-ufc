"""App Streamlit: el modelo, el mercado, y cuánto confiar en la diferencia."""

import pathlib
import subprocess
import sys

import pandas as pd
import requests
import streamlit as st

import betano
import cartelera
import ledger
import oddsapi
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

# El pipeline del README, en orden. fetch baja los CSVs, wiki y sherdog los completan,
# features construye la tabla y train re-entrena el modelo que la app carga.
PIPELINE = ["fetch_data.py", "wiki.py", "sherdog.py", "features.py", "train.py"]
RAIZ = pathlib.Path(__file__).parent

st.set_page_config(page_title="Predictor UFC", page_icon=":material/sports_mma:")
st.title("Predictor de peleas UFC")
st.caption(
    "El modelo aprende del historial de ufcstats.com. Está medido que la cuota de la casa "
    "le gana (log loss 0.611 vs 0.661 sobre 5679 peleas), así que esto es una segunda "
    "opinión independiente, no un detector de valor: apostar todo su EV positivo rindió "
    "−5.2% de ROI. Lo único que no perdió en el backtest es apostar cuando el modelo "
    "coincide con la casa y aun así encuentra EV — eso es lo que se marca como candidata.")


@st.cache_resource
def _cargar():
    return predict.cargar(), predict.peleadores()


@st.cache_data(ttl=3600, show_spinner="Buscando las carteleras…")
def _carteleras():
    try:
        return cartelera.proximas()
    except requests.RequestException:
        return []  # el warning de la pestania ya explica que ESPN no respondio


# El TTL es tambien el rate limit contra Betano: una request cada media hora.
@st.cache_data(ttl=1800, show_spinner="Buscando las cuotas en Betano…")
def _cuotas_betano():
    return betano.cuotas()


# 30 min de cache tambien alcanza para el tier gratis de The Odds API (500/mes).
@st.cache_data(ttl=1800, show_spinner="Buscando el consenso multi-casa…")
def _consenso():
    return oddsapi.cuotas()


def _resultado(a, b, r, cuotas):
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


def _historial(df):
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


with st.sidebar:
    st.subheader("Datos")
    st.caption("El pipeline completo, en orden. Tarda unos minutos: wiki y sherdog "
               "están cacheados y solo piden lo nuevo, features y train no.")
    if st.button("Actualizar y re-entrenar", icon=":material/refresh:"):
        with st.status("Actualizando…", expanded=True) as estado_run:
            for script in PIPELINE:
                st.write(f"`{script}`")
                # -u: sin esto Python bufferea la salida y el script parece colgado.
                p = subprocess.Popen([sys.executable, "-u", script], cwd=RAIZ,
                                     stdout=subprocess.PIPE,
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

(modelo, estado), nombres = _cargar()
tab_matchup, tab_cartelera, tab_historial = st.tabs(["Matchup", "Cartelera",
                                                     "Historial"])

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
        consenso = _consenso()
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
                info = oddsapi.buscar(consenso, pelea["a"], pelea["b"]) \
                    if consenso else None
                if info:
                    ma, mb = info["mejor"]
                    st.caption(f"Consenso de {info['casas']} casas — {pelea['a']} "
                               f"{info['p_a']:.1%} · mejor cuota {ma:.2f} / {mb:.2f}")
                    # top-down: la mejor cuota le gana al consenso del mercado. No usa
                    # el modelo, asi que vale incluso para los debuts.
                    ev_td = (info["p_a"] * ma - 1, (1 - info["p_a"]) * mb - 1)
                    if max(ev_td) > 0:
                        lado = 0 if ev_td[0] >= ev_td[1] else 1
                        quien = pelea["a"] if lado == 0 else pelea["b"]
                        st.info(f"**Línea desalineada: {quien} paga "
                                f"{info['mejor'][lado]:.2f}, más que el consenso "
                                f"({ev_td[lado]:+.1%} de EV).** Señal de line-shopping "
                                "contra el mercado, independiente del modelo.")
                m = predict.metodo(modelo, *cartelera.contexto(pelea["peso"], i == 0))
                if m:
                    st.caption(f"Cómo suele terminar: KO/TKO {m['ko']:.0%} · sumisión "
                               f"{m['sub']:.0%} · decisión {m['dec']:.0%} — está medido "
                               "que depende de la división, no del historial del par.")
                r = cartelera.predecir(pelea, modelo, estado, cuotas)
                if "error" in r:
                    st.info(f"{r['error']} — el modelo no puede predecir un debut.")
                    continue
                if cuotas:
                    # la primera cuota vista queda congelada en el ledger: es la
                    # "apuesta al abrir el mercado" que el Historial evalua despues
                    ledger.registrar(evento["evento"], evento["fecha"],
                                     pelea["a"], pelea["b"], r, cuotas)
                _resultado(pelea["a"], pelea["b"], r, cuotas)

with tab_historial:
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

        tabla = _historial(df_ledger)
        hechas = df_ledger["gano"].notna().to_numpy()
        st.caption("Cada fila muestra **un solo lado** de la pelea: la candidata a valor "
                   "si la hubo, si no el lado de mayor EV. Las cuotas y el retorno son "
                   "siempre de ese lado.")
        if not hechas.all():
            proximas = tabla[~hechas].sort_values("fecha")
            st.subheader(f"Sin resolver ({len(proximas)})", divider="gray")
            # el CLV todavia no existe: el "cierre" de un evento futuro es la cuota de hoy
            st.dataframe(proximas.drop(columns=["ganador", "acerto", "clv", "retorno"]),
                         hide_index=True, column_config=COLUMNAS)
        if hechas.any():
            resueltas = tabla[hechas].sort_values("fecha", ascending=False)
            st.subheader(f"Resueltas ({len(resueltas)})", divider="gray")
            st.dataframe(resueltas, hide_index=True, column_config=COLUMNAS)
