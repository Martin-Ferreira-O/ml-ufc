"""Lo que rindio la regla de apuesta sobre veinte anios, y lo que no se puede medir.

Lee `data/backtest.json`, que produce `python -m ufc.modelo.backtest`. No recalcula nada:
la grilla sale de un rolling-origin de 20 folds y no puede correr en un rerun de Streamlit.
"""

import json

import pandas as pd
import streamlit as st

from ufc.modelo import backtest
from ufc.ui import comunes


@st.cache_data(show_spinner=False)
def _datos(mtime):
    """`mtime` no se usa adentro: esta en la firma para invalidar el cache al regenerar."""
    return json.loads(backtest.SALIDA.read_text())


def _grilla(d):
    g = pd.DataFrame([f for f in d["grilla"] if f["n"]])
    return g.assign(banda=g["hi"] - g["lo"])


def render():
    if not backtest.SALIDA.exists():
        comunes.encabezado("Backtest", "Todavía no se corrió.", seccion="Auditoría")
        st.info("Corré `python -m ufc.modelo.backtest` para generar `data/backtest.json`. "
                "Tarda unos minutos: entrena 20 folds de rolling-origin.",
                icon=":material/terminal:")
        return

    d = _datos(backtest.SALIDA.stat().st_mtime)
    g = _grilla(d)
    hay_ventaja = any(f["n"] >= d["min_n"] and f["lo"] > 0 for f in d["grilla"])
    comunes.encabezado(
        "Backtest",
        f"Flat-bet de 1 unidad sobre {d['lados']} lados de {d['peleas']} peleas "
        f"({d['desde']} a {d['hasta']}), con las probabilidades fuera de muestra del "
        "rolling-origin y la cuota real de cada pelea. Todos los intervalos son IC95% "
        "bootstrap clusterizados por evento: dos peleas de la misma cartelera no son dos "
        "observaciones independientes.",
        seccion="Auditoría",
        chips=[(f"{d['peleas']} peleas", "blue", ":material/history:"),
               (f"calibrador: {d['calibrador_elegido']}", "gray", ":material/tune:"),
               (("hay umbral con ventaja" if hay_ventaja else "sin ventaja demostrada"),
                "green" if hay_ventaja else "red",
                ":material/check:" if hay_ventaja else ":material/do_not_disturb_on:")])

    if not hay_ventaja:
        st.error("**Ningún umbral deja el IC95% del ROI por encima de cero.** Con esta "
                 "evidencia no hay regla de apuesta que sostener. Los números de abajo "
                 "sirven para entender por qué, no para elegir un umbral.",
                 icon=":material/do_not_disturb_on:")

    mejor = max((f for f in d["grilla"] if f["n"]), key=lambda f: f["roi"])
    with st.container(horizontal=True):
        st.metric("ROI a umbral 0", f"{d['grilla'][0]['roi']:+.2%}", border=True,
                  delta=f"IC95% [{d['grilla'][0]['lo']:+.1%}, {d['grilla'][0]['hi']:+.1%}]",
                  delta_color="off",
                  help="Apostando todo lado donde el modelo le gana al mercado.")
        st.metric("Mejor ROI de la grilla", f"{mejor['roi']:+.2%}", border=True,
                  delta=f"umbral {mejor['umbral']:.0%}, n={mejor['n']}",
                  delta_color="off",
                  help="El mejor de 21 umbrales elegido mirando estos mismos datos. Es "
                       "el número que hay que desconfiar más de toda la pantalla.")
        st.metric("EV medio prometido", f"{d['grilla'][0]['ev_medio']:+.1%}", border=True,
                  help="Lo que el modelo creía que iba a ganar por apuesta. La distancia "
                       "contra el ROI real es el tamaño del autoengaño.")
        st.metric("Vig histórico", f"{d['dominio']['vig_mediano_historico']:.2%}",
                  border=True,
                  delta=f"Betano hoy {d['dominio'].get('vig_mediano_betano', float('nan')):.2%}",
                  delta_color="off",
                  help="Margen de la casa. El mercado se desviguea antes de calcular "
                       "ventaja: con la implícita cruda, ese margen se contaría como valor.")

    st.warning(
        f"**{d['grilla'][0]['ev_medio']:+.0%} de EV esperado contra "
        f"{d['grilla'][0]['roi']:+.2%} de ROI real.** El modelo no está midiendo mal la "
        "suerte: está midiendo mal su propia ventaja. Un EV que no aparece en el "
        "resultado es una probabilidad sesgada contra el mercado, no una racha.",
        icon=":material/warning:")

    tabs = st.tabs(["ROI por umbral", "Bankroll", "Segmentos", "Calibración"])

    with tabs[0]:
        st.caption("Cada punto es un umbral de ventaja distinto sobre las MISMAS peleas. "
                   "La banda es el IC95%; donde toca cero, el umbral no dice nada.")
        st.line_chart(g.set_index("umbral")[["roi", "lo", "hi"]],
                      x_label="umbral de ventaja", y_label="ROI",
                      color=["#60A5FA", "#94A3B8", "#94A3B8"])
        st.line_chart(g.set_index("umbral")[["n"]], x_label="umbral de ventaja",
                      y_label="apuestas", color="#F59E0B")
        st.caption("Menos apuestas al subir el umbral es lo esperable. Lo que hay que "
                   "mirar es si el ROI sube más rápido de lo que se ensancha la banda.")
        st.dataframe(g[["umbral", "n", "roi", "yield", "neto", "acierto", "ev_medio",
                        "cuota_media", "lo", "hi"]],
                     column_config=comunes.COLUMNAS_BT, hide_index=True)
        ev0 = d["ev_positivo"]
        if ev0["n"]:
            st.caption(f"Filtro clásico `EV > 0`: {ev0['n']} apuestas, ROI "
                       f"{ev0['roi']:+.2%} IC95% [{ev0['lo']:+.2%}, {ev0['hi']:+.2%}].")

    with tabs[1]:
        curva = pd.DataFrame(d["curva"])
        if curva.empty:
            st.info("Sin apuestas al umbral reportado.")
        else:
            curva["date"] = pd.to_datetime(curva["date"])
            st.caption(f"Banca de 100 unidades, 1 unidad por apuesta, umbral "
                       f"{d['umbral_reportado']:.0%}, en orden cronológico real.")
            st.line_chart(curva.set_index("date")[["banca"]], y_label="unidades",
                          color="#60A5FA")
            st.caption("Drawdown en unidades contra el pico del acumulado. Va en unidades "
                       "y no en porcentaje porque a stake flat la banca puede llegar a "
                       "cero, y ahí el porcentaje se indefine justo donde más importa.")
            st.area_chart(curva.set_index("date")[["drawdown"]], y_label="unidades",
                          color="#F59E0B")

    with tabs[2]:
        niveles = sum(len(v) for v in d["segmentos"].values())
        st.caption(f"Al umbral {d['umbral_reportado']:.0%}. Son **{niveles} niveles** "
                   f"mirados sobre los mismos datos: con ~5% de falsos positivos por "
                   f"nivel se esperan ~{round(niveles * 0.05)} “hallazgos” de puro azar. "
                   "Un ROI positivo acá es una hipótesis para testear hacia adelante, no "
                   "un segmento rentable.")
        for corte, filas in d["segmentos"].items():
            with st.expander(corte.capitalize(), expanded=corte == "división"):
                st.dataframe(pd.DataFrame(filas)[
                    ["nivel", "peleas", "log_loss", "brier", "accuracy", "n", "roi",
                     "lo", "hi", "concluyente"]],
                    column_config=comunes.COLUMNAS_BT, hide_index=True)

    with tabs[3]:
        st.caption("Cada fold se calibra con los folds anteriores, nunca con los suyos. "
                   "Un calibrador ajustado sobre las predicciones que corrige da un ECE "
                   "de casi cero por construcción y no significa nada.")
        st.dataframe(pd.DataFrame(d["calibradores"])[
            ["calibrador", "log_loss", "brier", "ece", "veredicto", "queda"]],
            column_config=comunes.COLUMNAS_BT, hide_index=True)
        st.caption(f"Entra **{d['calibrador_elegido']}**. Calibrar corrige el sesgo de la "
                   "probabilidad contra la realidad; no corrige que el modelo esté "
                   "comprimido contra el mercado, que es lo que fabrica EV en el no "
                   "favorito. Son dos problemas distintos.")

    st.info(f"**CLV:** {d['clv']}", icon=":material/info:")
    st.caption("El backtest cobra con las cuotas de `ufc_odds.csv`, que son de cierre o "
               "consenso; la app congela en apertura y en una sola casa. Son dos juegos "
               "distintos: estos números acotan la expectativa, no la reproducen.")
