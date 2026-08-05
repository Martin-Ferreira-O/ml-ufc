"""Pestania Cartelera: agenda de eventos ESPN y detalle pelea por pelea."""

import datetime

import streamlit as st

from ufc.datos import betano, cartelera, oddsapi
from ufc.modelo import predict
from ufc.registro import ledger, predictores
from ufc.ui import comunes


_EVENTO_ACTIVO = "cartelera_evento_activo"
_VER_ANTERIORES = "cartelera_ver_anteriores"
_BUSCAR = "cartelera_buscar"
_SOLO_CUOTA = "cartelera_solo_cuota"
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

    proximo = eventos[0]
    clave_proximo = _clave(proximo)
    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute",
                          vertical_alignment="center"):
            st.badge("Próximo evento", icon=":material/local_fire_department:",
                     color="orange")
            if falta := comunes.cuenta_regresiva(proximo["fecha"]):
                st.badge(falta, icon=":material/schedule:", color="blue")
        st.header(proximo["evento"])
        st.caption(_meta(proximo))
        seleccionado = st.session_state[_EVENTO_ACTIVO] == clave_proximo
        st.button("Cartelera en pantalla" if seleccionado else "Ver esta cartelera",
                  key="cartelera_evento_proximo",
                  icon=":material/check:" if seleccionado else ":material/arrow_forward:",
                  type="primary", disabled=seleccionado, width="stretch",
                  on_click=_activar, args=(clave_proximo,))

    if len(eventos) > 1:
        st.caption("MÁS ADELANTE")
        for i, evento in enumerate(eventos[1:], 1):
            clave = _clave(evento)
            seleccionado = st.session_state[_EVENTO_ACTIVO] == clave
            with st.container(border=True, horizontal=True,
                              horizontal_alignment="distribute",
                              vertical_alignment="center"):
                with st.container(gap=None):
                    st.markdown(f"**{evento['evento']}**")
                    st.caption(_meta(evento))
                with st.container(horizontal=True, vertical_alignment="center",
                                  width="content"):
                    if falta := comunes.cuenta_regresiva(evento["fecha"]):
                        st.badge(falta, color="gray", icon=":material/schedule:")
                    st.button("En pantalla" if seleccionado else "Ver cartelera",
                              key=f"cartelera_evento_{i}",
                              icon=":material/check:" if seleccionado
                              else ":material/arrow_forward:", disabled=seleccionado,
                              on_click=_activar, args=(clave,))

    return por_clave[st.session_state[_EVENTO_ACTIVO]]


def _metodos(m):
    """Como suele terminar una pelea de esa division, en tres badges en vez de un texto."""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(f"KO/TKO {m['ko']:.0%}", color="red", icon=":material/bolt:")
        st.badge(f"Sumisión {m['sub']:.0%}", color="violet", icon=":material/lock:")
        st.badge(f"Decisión {m['dec']:.0%}", color="gray", icon=":material/gavel:")
    st.caption("Está medido que depende de la división y del lugar en la cartelera, no "
               "del historial del par.")


def _visible(pelea, busqueda, solo_cuota, cuotas):
    if solo_cuota and not cuotas:
        return False
    if not busqueda:
        return True
    texto = f"{pelea['a']} {pelea['b']} {pelea.get('peso', '')}".lower()
    return busqueda.lower().strip() in texto


def render(modelo, estado, pagina_predictores=None):
    eventos = comunes.carteleras()
    st.session_state.setdefault(_VER_ANTERIORES, False)
    if st.session_state[_VER_ANTERIORES]:
        comunes.encabezado("Cartelera", "Las carteleras que ya ocurrieron, para cargar "
                                        "picks y confirmar resultados.",
                           seccion="Eventos")
        _agenda_anteriores(_anteriores(eventos), pagina_predictores)
        return

    comunes.encabezado(
        "Cartelera",
        "Las peleas anunciadas, de la API pública de ESPN, con la cuota de Betano cuando "
        "ya está publicada. Betano abre mercado unos días antes del evento, así que las "
        "carteleras lejanas van a aparecer sin cuota.",
        seccion="Eventos",
        chips=[(f"{len(eventos)} eventos anunciados", "blue", ":material/event:")]
        if eventos else ())

    if not eventos:
        st.warning("No hay carteleras anunciadas (o la API de ESPN no respondió).",
                   icon=":material/cloud_off:")
        st.button("Ver carteleras anteriores", icon=":material/history:",
                  on_click=_cambiar_vista, args=(True,))
        return

    evento = _agenda(eventos)
    st.button("Ver carteleras anteriores", icon=":material/history:",
              on_click=_cambiar_vista, args=(True,))

    st.header(evento["evento"], divider="gray")
    st.caption(_meta(evento))

    # La agenda aparece antes que estas consultas: el usuario ve y elige los eventos
    # sin esperar los precios, y solo se calcula el detalle de la seleccion activa.
    tabla = comunes.cuotas_betano()
    consenso = comunes.consenso()
    if not tabla:
        st.warning("Betano no respondió: la cartelera va sin cuotas ni coincidencia.",
                   icon=":material/cloud_off:")
    cuotas_de = [betano.buscar(tabla, p["a"], p["b"]) for p in evento["peleas"]]
    # Si no hay ninguna, el motivo es el evento entero, no cada pelea: decirlo una
    # vez evita que parezca que el matcheo falló pelea por pelea.
    if tabla and not any(cuotas_de):
        st.info("Betano todavía no abrió mercado para este evento — lo hace unos días "
                "antes. Va sin cuotas ni nivel de coincidencia.",
                icon=":material/storefront:")

    # Filtro solo de pantalla: una cartelera numerada tiene catorce peleas y casi siempre
    # se entra buscando una sola. No toca ningun calculo ni el registro del ledger.
    with st.container(horizontal=True, vertical_alignment="bottom"):
        busqueda = st.text_input("Buscar peleador o división", key=_BUSCAR,
                                 placeholder="ej. Pereira, peso pesado…",
                                 icon=":material/search:")
        solo_cuota = st.toggle("Solo con cuota", key=_SOLO_CUOTA,
                               help="Esconde las peleas para las que Betano todavía no "
                                    "publicó precio.")

    mostradas = 0
    for i, (pelea, cuotas) in enumerate(zip(evento["peleas"], cuotas_de)):
        # Predecir y congelar va antes de filtrar: el ledger tiene que registrar toda la
        # cartelera, no solo lo que quedó a la vista.
        r = cartelera.predecir(pelea, modelo, estado, cuotas, evento["fecha"])
        info = oddsapi.buscar(consenso, pelea["a"], pelea["b"]) if consenso else None
        if cuotas and "error" not in r:
            # la primera cuota vista queda congelada en el ledger: es la "apuesta al
            # abrir el mercado" que el Historial evalua despues. La mejor cuota del
            # mercado va con ella: sin eso el Seguimiento solo puede medir Betano.
            ledger.registrar(evento["evento"], evento["fecha"],
                             pelea["a"], pelea["b"], r, cuotas,
                             mejor=info["mejor"] if info else None)
        if not _visible(pelea, busqueda, solo_cuota, cuotas):
            continue
        mostradas += 1
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute",
                              vertical_alignment="center"):
                with st.container(horizontal=True, width="content",
                                  vertical_alignment="center"):
                    if i == 0:
                        st.badge("Main event", color="orange",
                                 icon=":material/workspace_premium:")
                    else:
                        st.badge(f"Pelea {len(evento['peleas']) - i}", color="gray")
                    if pelea["peso"]:
                        st.badge(pelea["peso"], color="gray",
                                 icon=":material/monitor_weight:")
                if cuotas:
                    # Un badge por esquina, con el nombre adentro: el par "3.10 / 1.30"
                    # solo se entiende si ya sabes de memoria en que orden van.
                    with st.container(horizontal=True, width="content",
                                      vertical_alignment="center"):
                        st.badge(f"Betano — {pelea['a']} {cuotas[0]:.2f}",
                                 color="blue", icon=":material/sell:")
                        st.badge(f"{pelea['b']} {cuotas[1]:.2f}", color="orange")
                elif any(cuotas_de):
                    st.badge("Sin cuota publicada", color="gray", icon=":material/sell:")
            st.subheader(f"{pelea['a']} vs {pelea['b']}")

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
                            "mercado. Compará precios antes de apostar.",
                            icon=":material/price_check:")
            if m := predict.metodo(modelo, *cartelera.contexto(pelea["peso"], i == 0)):
                _metodos(m)
            if "error" in r:
                st.info(f"{r['error']} — el modelo no puede predecir un debut.",
                        icon=":material/person_search:")
                continue
            comunes.resultado(pelea["a"], pelea["b"], r, cuotas)

    if not mostradas:
        st.info("Ninguna pelea de esta cartelera coincide con el filtro.",
                icon=":material/filter_alt_off:")
