"""Pestania Cartelera: los eventos anunciados de ESPN, pelea por pelea."""

import streamlit as st

from ufc.datos import betano, cartelera, oddsapi
from ufc.modelo import predict
from ufc.registro import ledger
from ufc.ui import comunes


def render(modelo, estado):
    st.caption("Las peleas anunciadas, de la API pública de ESPN, con la cuota de Betano "
               "cuando ya está publicada. Betano abre mercado unos días antes del evento, "
               "así que las carteleras lejanas van a aparecer sin cuota.")
    eventos = comunes.carteleras()
    if not eventos:
        st.warning("No hay carteleras anunciadas (o la API de ESPN no respondió).")
    else:
        tabla = comunes.cuotas_betano()
        consenso = comunes.consenso()
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
                comunes.resultado(pelea["a"], pelea["b"], r, cuotas)
