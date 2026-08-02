"""Pestania Cartelera: agenda de eventos ESPN y detalle pelea por pelea."""

import datetime

import streamlit as st

from ufc.datos import betano, cartelera, oddsapi
from ufc.modelo import predict
from ufc.registro import ledger, predictores
from ufc.ui import comunes


_EVENTO_ACTIVO = "cartelera_evento_activo"
_VER_ANTERIORES = "cartelera_ver_anteriores"
_MESES = ("ene", "feb", "mar", "abr", "may", "jun",
          "jul", "ago", "sept", "oct", "nov", "dic")


def _clave(evento):
    """Identidad estable entre reruns y refrescos de la lista de ESPN."""
    return f"{evento['fecha']}|{evento['evento']}"


def _fecha(fecha):
    """Fecha ISO de ESPN en una etiqueta corta e independiente del locale del host."""
    try:
        d = datetime.date.fromisoformat(str(fecha)[:10])
    except ValueError:
        return str(fecha)
    return f"{d.day} {_MESES[d.month - 1]} {d.year}"


def _meta(evento):
    n = len(evento["peleas"])
    return f"{_fecha(evento['fecha'])} · {n} {'pelea' if n == 1 else 'peleas'}"


def _activar(clave):
    """Callback: actualiza la seleccion antes de que empiece a dibujarse el rerun."""
    st.session_state[_EVENTO_ACTIVO] = clave


def _cambiar_vista(anteriores):
    st.session_state[_VER_ANTERIORES] = anteriores


def _anteriores(actuales):
    """Une resultados UFCStats con eventos ya cargados que aun no llegaron al dataset."""
    hoy = datetime.date.today().isoformat()
    vistos = {e["evento"] for e in actuales}
    salida = []
    for evento in [*cartelera.anteriores(), *predictores.eventos_guardados()]:
        if (evento["evento"] not in vistos and evento.get("fecha", "") <= hoy):
            salida.append(evento)
            vistos.add(evento["evento"])
    return sorted(salida, key=lambda e: e.get("fecha", ""), reverse=True)


def _agenda_anteriores(eventos, pagina_predictores):
    st.subheader("Carteleras anteriores")
    st.caption("Abrí una cartelera para cargar las picks de cada predictor que seguís. "
               "Cuando confirmes también sus resultados, su rendimiento pasa a formar "
               "parte del peso que usa la app.")
    st.button("Volver a próximos eventos", icon=":material/arrow_back:",
              on_click=_cambiar_vista, args=(False,))
    if not eventos:
        st.info("Todavía no hay carteleras anteriores en los datos locales.",
                icon=":material/history:")
        return
    for evento in eventos:
        with st.container(border=True, horizontal=True,
                          horizontal_alignment="distribute",
                          vertical_alignment="center"):
            with st.container(gap=None):
                st.markdown(f"**{evento['evento']}**")
                st.caption(_meta(evento))
            if pagina_predictores is not None:
                st.page_link(pagina_predictores, label="Cargar picks",
                             icon=":material/edit_note:",
                             query_params={"evento": evento["evento"]})


def _agenda(eventos):
    """Dibuja la portada y agenda; devuelve solo el evento que debe calcularse."""
    eventos = sorted(eventos, key=lambda e: e.get("fecha") or "9999-12-31")
    por_clave = {_clave(e): e for e in eventos}
    if st.session_state.get(_EVENTO_ACTIVO) not in por_clave:
        st.session_state[_EVENTO_ACTIVO] = _clave(eventos[0])

    st.subheader("Próximos eventos")
    proximo = eventos[0]
    clave_proximo = _clave(proximo)
    with st.container(border=True):
        st.badge("Próximo", icon=":material/event:", color="blue")
        st.subheader(proximo["evento"])
        st.caption(_meta(proximo))
        seleccionado = st.session_state[_EVENTO_ACTIVO] == clave_proximo
        st.button("Seleccionado" if seleccionado else "Ver cartelera",
                  key="cartelera_evento_proximo", icon=":material/check:" if seleccionado
                  else ":material/arrow_forward:", type="primary",
                  disabled=seleccionado, width="stretch", on_click=_activar,
                  args=(clave_proximo,))

    if len(eventos) > 1:
        st.caption("Más adelante")
        for i, evento in enumerate(eventos[1:], 1):
            clave = _clave(evento)
            seleccionado = st.session_state[_EVENTO_ACTIVO] == clave
            with st.container(border=True, horizontal=True,
                              horizontal_alignment="distribute",
                              vertical_alignment="center"):
                with st.container(gap=None):
                    st.markdown(f"**{evento['evento']}**")
                    st.caption(_meta(evento))
                st.button("Seleccionado" if seleccionado else "Ver cartelera",
                          key=f"cartelera_evento_{i}",
                          icon=":material/check:" if seleccionado
                          else ":material/arrow_forward:", disabled=seleccionado,
                          on_click=_activar, args=(clave,))

    return por_clave[st.session_state[_EVENTO_ACTIVO]]


def render(modelo, estado, pagina_predictores=None):
    st.header("Cartelera")
    st.caption("Las peleas anunciadas, de la API pública de ESPN, con la cuota de Betano "
               "cuando ya está publicada. Betano abre mercado unos días antes del evento, "
               "así que las carteleras lejanas van a aparecer sin cuota.")
    eventos = comunes.carteleras()
    st.session_state.setdefault(_VER_ANTERIORES, False)
    if st.session_state[_VER_ANTERIORES]:
        _agenda_anteriores(_anteriores(eventos), pagina_predictores)
        return
    if not eventos:
        st.warning("No hay carteleras anunciadas (o la API de ESPN no respondió).")
        st.button("Ver carteleras anteriores", icon=":material/history:",
                  on_click=_cambiar_vista, args=(True,))
    else:
        evento = _agenda(eventos)
        st.button("Ver carteleras anteriores", icon=":material/history:",
                  on_click=_cambiar_vista, args=(True,))
        st.caption("Cartelera seleccionada")
        st.subheader(evento["evento"])
        st.caption(_meta(evento))

        # La agenda aparece antes que estas consultas: el usuario ve y elige los eventos
        # sin esperar los precios, y solo se calcula el detalle de la seleccion activa.
        tabla = comunes.cuotas_betano()
        consenso = comunes.consenso()
        if not tabla:
            st.warning("Betano no respondió: la cartelera va sin cuotas ni confianza.")
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
                        st.info(f"**Mejor cuota que el promedio: {quien} paga "
                                f"{info['mejor'][lado]:.2f}.** Esa casa ofrece "
                                f"{ev_td[lado]:+.1%} más valor que el precio medio del "
                                "mercado. Compará precios antes de apostar.")
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
