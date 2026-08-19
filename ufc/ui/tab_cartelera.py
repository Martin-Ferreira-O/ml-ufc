"""Pestania Cartelera: agenda de eventos ESPN y detalle pelea por pelea."""

import datetime

import streamlit as st

from ufc import nombres
from ufc.datos import betano, cartelera, oddsapi
from ufc.ia import analista, consenso as ia_consenso, historial
from ufc.modelo import predict
from ufc.registro import ledger, predictores
from ufc.ui import cartel, comunes


_VER_ANTERIORES = "cartelera_ver_anteriores"
# El indice de la pelea a abrir. Lo escribe el clic en el cartel; queda como estado para
# que los tests, que no pueden clickear un componente, abran el mismo modal.
_ABRIR_PELEA = "cartelera_abrir_pelea"


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
                st.caption(comunes.meta_evento(evento))
            if pagina_predictores is not None:
                st.page_link(pagina_predictores, label="Cargar picks",
                             icon=":material/edit_note:",
                             query_params={"evento": evento["evento"]})


def _agenda(eventos):
    """Selector de cartelera; devuelve solo el evento que debe calcularse.

    ponytail: un selectbox en vez de una tarjeta por evento. La UFC anuncia diez o
    doce carteleras a la vez y la agenda empujaba la cartelera que se venia a ver
    debajo de dos pantallas de eventos de dentro de tres meses.
    """
    with st.container(horizontal=True, vertical_alignment="bottom"):
        evento = comunes.selector_cartelera(eventos)
        st.button("Anteriores", icon=":material/history:",
                  help="Carteleras que ya ocurrieron, para cargar picks y confirmar "
                       "resultados.",
                  on_click=_cambiar_vista, args=(True,))
    return evento


def _metodos(m):
    """Como suele terminar una pelea de esa division, en tres badges en vez de un texto."""
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(f"KO/TKO {m['ko']:.0%}", color="red", icon=":material/bolt:")
        st.badge(f"Sumisión {m['sub']:.0%}", color="violet", icon=":material/lock:")
        st.badge(f"Decisión {m['dec']:.0%}", color="gray", icon=":material/gavel:")
    st.caption("Está medido que depende de la división y del lugar en la cartelera, no "
               "del historial del par.")


def _intel(evento):
    """Los informes del bot indexados por peleador, solo si son de este evento.

    El bot revisa la cartelera a menos de ocho dias y Cartelera lista noventa: para
    cualquier otro evento no hay nada que mostrar todavia.
    """
    informe = comunes.intel_evento()
    if not informe or (nombres.normalizar(informe["event_name"]) !=
                       nombres.normalizar(evento["evento"])):
        return {}, {}
    checks = {nombres.normalizar(c["fighter"]): c for c in informe["checks"]}
    perfiles = comunes.intel_perfiles(tuple(c["fighter"] for c in informe["checks"]))
    return checks, perfiles


_RIVAL = "cartelera_rival"
_ULTIMAS = 5


def _rivales(pelea, evento, estado, retratos):
    """Contra quien viene peleando cada uno, y la ficha del rival que se toque.

    Los widgets de aca adentro no cierran el modal: `st.dialog` es un fragment (envuelve
    su cuerpo en `_fragment`), asi que tocar las pills re-corre solo la ventana y no el
    script entero — que ademas volveria a pedirle precios a Betano y a The Odds API.
    """
    largo, stances = historial.cargar()   # lru_cache: una lectura de disco por proceso
    resumenes = {pelea[lado]: historial.resumen(pelea[lado], evento["fecha"], largo=largo,
                                                stances=stances, estado=estado,
                                                n=_ULTIMAS)
                 for lado in ("a", "b")}
    with st.container(horizontal=True):
        for nombre, resumen in resumenes.items():
            with st.container(border=True):
                st.markdown(f"**{nombre}**")
                if resumen:
                    comunes.desglose_metodos(resumen)
                    comunes.ultimas_peleas(resumen, n=_ULTIMAS)
                else:
                    st.caption("Sin peleas en UFC antes de este evento.")
    st.caption("El Elo es el de **hoy**, no el que tenía ese rival el día de esa pelea: es "
               "lo único que guarda el dataset. O sea que un rival que se hundió después "
               "aparece más bajo de lo que estaba cuando se pelearon.")

    opciones = [u["rival"] for r in resumenes.values() if r for u in r["ultimas"]]
    if not opciones:
        return
    # `dict.fromkeys` y no `set`: dos peleadores pueden compartir rival y el orden tiene
    # que seguir siendo el de la lista de arriba, no uno aleatorio.
    elegido = st.pills("Ver la ficha de un rival", list(dict.fromkeys(opciones)),
                       key=f"{_RIVAL}_{evento['evento']}_{pelea['a']}")
    if elegido:
        # ponytail: la foto sale del lote que ya bajo la cartelera, y un rival no esta ahi
        # -> monograma. Bajarla al vuelo seria un request y un spinner adentro del modal en
        # cada clic; si algun dia se quiere la cara, va `comunes.retratos((elegido,))`.
        comunes.ficha_peleador(
            elegido, estado, evento["fecha"],
            historial.resumen(elegido, evento["fecha"], largo=largo, stances=stances,
                              estado=estado, n=_ULTIMAS),
            foto=retratos.get(elegido))


def _detalle(pelea, fila, suyos, perfiles, retratos):
    """El voto humano y el contexto de la pelea, plegados: en una cartelera numerada son
    catorce peleas y dos informes por pelea."""
    with st.expander("Picks e inteligencia", icon=":material/how_to_vote:"):
        if fila.detalle:
            comunes.picks_pelea(fila)
        else:
            st.caption("Ningún predictor cargó esta pelea todavía.")
        for lado, check in suyos:
            comunes.intel_tarjeta(check, perfiles[check["fighter"]],
                                  titulo=pelea[lado], evidencias=False,
                                  foto=retratos.get(pelea[lado]))


@st.fragment
def _boton_ia(evento):
    """Dispara el analisis de la cartelera activa, aislado del resto de la pagina.

    Va en un fragment a proposito: sin el, cada tick de progreso volveria a correr el
    script entero — o sea, otra consulta a Betano y otra a The Odds API por cada pelea
    analizada. Con el, solo se redibuja este bloque.
    """
    hay_ia = analista.hay_api()
    disparar = st.button(
        "Analizar con IA", icon=":material/smart_toy:", disabled=not hay_ia,
        help=("Un prompt por pelea con el historial deportivo de los dos —récord, cómo "
              "gana y cómo pierde cada uno, sus últimas peleas— y sin cuotas ni mercado. "
              "Cada pelea se analiza una vez por día: volver a apretar no vuelve a "
              "gastar." if hay_ia else
              "Falta GEMINI_API_KEY en el entorno o en el archivo .env."))
    if not disparar:
        return
    with st.status("Analizando la cartelera…", expanded=True) as estado:
        try:
            ctx = ia_consenso.contexto(evento)
            proveedor = analista.Analista()
            barra = st.progress(0.0)

            def avance(hechas, total, pelea):
                barra.progress(hechas / max(total, 1),
                               text=f"{pelea['a']} vs {pelea['b']}")

            r = ia_consenso.ejecutar(evento, ctx=ctx, provider=proveedor,
                                     progreso=avance)
            n = ia_consenso.sincronizar_picks(evento)
        except (ValueError, OSError) as exc:
            estado.update(label="No se pudo analizar", state="error")
            st.error(str(exc), icon=":material/error:")
            return
        estado.update(
            label=(f"Listo — {r['analizadas']} peleas analizadas, {r['omitidas']} ya "
                   f"estaban de hoy, {r['errores']} con error"),
            state="error" if r["errores"] else "complete")
        if r["fallos"]:
            # Las que salieron bien ya quedaron guardadas por `run_day`, asi que volver a
            # apretar solo reintenta estas y no vuelve a pagar las otras.
            st.warning("No se pudieron analizar estas peleas. Volvé a apretar el botón "
                       "para reintentar solo esas:", icon=":material/warning:")
            for fallo in r["fallos"]:
                st.caption(f"· {fallo}")
        st.caption(f"{n} picks registradas como «{ia_consenso.IA_PREDICTOR}» · "
                   f"{r['prompt_tokens']} tokens de entrada, {r['output_tokens']} de salida")
    comunes.ia_veredictos.clear()
    st.rerun()


def _pelea(pelea, i, evento, modelo, estado, cuotas, r, info, hay_mercado,
           fila, suyos, perfiles, retratos, ia):
    """Todo lo que se sabe de una pelea. Es el contenido del modal que abre el cartel."""
    with st.container(horizontal=True, vertical_alignment="center"):
        if i == 0:
            st.badge("Main event", color="orange",
                     icon=":material/workspace_premium:")
        else:
            st.badge(f"Pelea {len(evento['peleas']) - i}", color="gray")
        if pelea["peso"]:
            st.badge(pelea["peso"], color="gray", icon=":material/monitor_weight:")
        if cuotas:
            # Un badge por esquina, con el nombre adentro: el par "3.10 / 1.30" solo se
            # entiende si ya sabes de memoria en que orden van.
            st.badge(f"Betano — {pelea['a']} {cuotas[0]:.2f}", color="blue",
                     icon=":material/sell:")
            st.badge(f"{pelea['b']} {cuotas[1]:.2f}", color="orange")
        elif hay_mercado:
            st.badge("Sin cuota publicada", color="gray", icon=":material/sell:")

    if info:
        ma, mb = info["mejor"]
        st.caption(f"Consenso de {info['casas']} casas — {pelea['a']} "
                   f"{info['p_a']:.1%} · mejor cuota {ma:.2f} / {mb:.2f}")
        # top-down: la mejor cuota le gana al consenso del mercado. No usa el modelo, asi
        # que vale incluso para los debuts. El EV sale de `oddsapi.valor`, que para cada
        # lado usa el consenso SIN la casa que ofrece ese precio y descuenta la dispersion
        # entre casas.
        valores = oddsapi.valor(info)
        mejor_td = max(valores, key=lambda v: v["ev_low"]) if valores else None
        if mejor_td and mejor_td["ev_low"] > 0:
            quien = pelea["a"] if mejor_td["lado"] == "a" else pelea["b"]
            st.info(f"**Mejor cuota que el consenso: {quien} paga "
                    f"{mejor_td['cuota']:.2f} en {mejor_td['casa']}.** Contra el "
                    f"consenso de las otras casas eso vale "
                    f"{mejor_td['ev_low']:+.1%} ya descontada la dispersión "
                    f"({mejor_td['ev']:+.1%} sin descontarla). Tomar el precio "
                    "más alto de N casas infla la ventaja aparente aunque no "
                    "haya ninguna: compará antes de apostar.",
                    icon=":material/price_check:")
    if m := predict.metodo(modelo, *cartelera.contexto(pelea["peso"], i == 0)):
        _metodos(m)
    if "error" in r:
        # Un debut se queda sin modelo, pero las picks y el contexto siguen siendo lo
        # unico que hay para leerlo: van igual.
        st.info(f"{r['error']} — el modelo no puede predecir un debut.",
                icon=":material/person_search:")
    else:
        comunes.resultado(pelea["a"], pelea["b"], r, cuotas,
                          fotos=(retratos.get(pelea["a"]), retratos.get(pelea["b"])))

    # Hermanos de `_detalle` y no adentro: los expanders no se anidan.
    with st.expander("Estadísticas comparadas", icon=":material/bar_chart:"):
        comunes.comparativa(pelea["a"], pelea["b"], estado, evento["fecha"])
    with st.expander("Últimos rivales", icon=":material/history:"):
        _rivales(pelea, evento, estado, retratos)

    if ia is not None:
        comunes.ia_veredicto(ia, pelea)
        with st.expander("Análisis de la IA", icon=":material/smart_toy:"):
            comunes.ia_tarjeta(ia, pelea)

    if fila.predictores or suyos:
        with st.container(horizontal=True, vertical_alignment="center"):
            if fila.predictores:
                st.badge(f"{fila.senal} · {fila.seleccion or 'sin mayoría'} · "
                         f"{fila.predictores} "
                         f"{'predictor' if fila.predictores == 1 else 'predictores'}",
                         color="green" if fila.fuerte else "gray",
                         icon=":material/verified:" if fila.fuerte
                         else ":material/how_to_vote:")
            for lado, check in suyos:
                st.badge(f"{pelea[lado]} {comunes.intel_etiqueta(check['score'])}",
                         color=comunes.intel_color(check["score"]),
                         icon=":material/manage_search:")
    _detalle(pelea, fila, suyos, perfiles, retratos)


def _modal(pelea, *args, **kwargs):
    """El mismo bloque, en una ventana. El titulo va en el encabezado del modal, asi que
    el nombre de la pelea no se repite adentro."""

    @st.dialog(f"{pelea['a']} vs {pelea['b']}", width="large")
    def ventana():
        _pelea(pelea, *args, **kwargs)

    ventana()


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

    st.header(evento["evento"], divider="gray")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption(comunes.meta_evento(evento))
        if falta := comunes.cuenta_regresiva(evento["fecha"]):
            st.badge(falta, icon=":material/schedule:", color="blue")

    # La agenda aparece antes que estas consultas: el usuario ve y elige los eventos
    # sin esperar los precios, y solo se calcula el detalle de la seleccion activa.
    tabla = comunes.cuotas_betano()
    consenso = comunes.consenso()
    if not tabla:
        st.warning("Betano no respondió: la cartelera va sin cuotas ni coincidencia.",
                   icon=":material/cloud_off:")
    cuotas_de = [betano.buscar(tabla, p["a"], p["b"]) for p in evento["peleas"]]
    preds = [cartelera.predecir(p, modelo, estado, c, evento["fecha"])
             for p, c in zip(evento["peleas"], cuotas_de)]
    # `ranking` necesita la cartelera entera y reordena sus filas: `orden` es el indice
    # original de la pelea, que es por donde se vuelve a unir con el loop de abajo.
    rank = predictores.ranking(evento["peleas"], predictores.leer(evento["evento"]),
                               preds, cuotas_de)
    por_orden = {int(f.orden): f for f in rank.itertuples()}
    checks, perfiles = _intel(evento)
    veredictos = comunes.ia_veredictos(evento["evento"])
    # Una sola pasada para toda la cartelera: bajar por pelea serian catorce tandas de
    # requests y catorce spinners.
    retratos = comunes.retratos(tuple(p[lado] for p in evento["peleas"]
                                      for lado in ("a", "b")))
    # Si no hay ninguna, el motivo es el evento entero, no cada pelea: decirlo una
    # vez evita que parezca que el matcheo falló pelea por pelea.
    if tabla and not any(cuotas_de):
        st.info("Betano todavía no abrió mercado para este evento — lo hace unos días "
                "antes. Va sin cuotas ni nivel de coincidencia.",
                icon=":material/storefront:")

    # El registro va sobre la cartelera entera y no sobre la pelea que se abre: el ledger
    # tiene que congelar todos los precios del evento, se haya mirado el detalle o no.
    infos = [oddsapi.buscar(consenso, p["a"], p["b"]) if consenso else None
             for p in evento["peleas"]]
    for pelea, cuotas, r, info in zip(evento["peleas"], cuotas_de, preds, infos):
        if cuotas and "error" not in r:
            # la primera cuota vista queda congelada en el ledger: es la "apuesta al
            # abrir el mercado" que el Historial evalua despues. La mejor cuota del
            # mercado va con ella: sin eso el Seguimiento solo puede medir Betano.
            ledger.registrar(evento["evento"], evento["fecha"],
                             pelea["a"], pelea["b"], r, cuotas,
                             mejor=info["mejor"] if info else None,
                             inicio_utc=evento.get("inicio_utc"))

    elegida = cartel.mostrar(evento, preds, key=f"cartel_{evento['evento']}")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.caption("Tocá una pelea del cartel para ver el modelo, el mercado, la IA y "
                   "las picks de esa pelea.")
        _boton_ia(evento)

    # El clic del cartel dura un solo rerun, igual que un boton. `pop` le da la misma
    # semantica al gancho de session_state, que es por donde entra la suite.
    if elegida is None:
        elegida = st.session_state.pop(_ABRIR_PELEA, None)
    if elegida is None or not 0 <= elegida < len(evento["peleas"]):
        return

    pelea = evento["peleas"][elegida]
    ia = veredictos.get((pelea["a"], pelea["b"]))
    suyos = [(lado, checks[clave]) for lado in ("a", "b")
             if (clave := nombres.normalizar(pelea[lado])) in checks]
    _modal(pelea, elegida, evento, modelo, estado, cuotas_de[elegida], preds[elegida],
           infos[elegida], any(cuotas_de), por_orden[elegida], suyos, perfiles,
           retratos, ia)
