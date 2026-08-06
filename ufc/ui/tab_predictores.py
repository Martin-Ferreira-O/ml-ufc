"""Carga, correccion y comparacion de predictores humanos."""

import hashlib

import pandas as pd
import streamlit as st

from ufc import nombres
from ufc.datos import betano, cartelera
from ufc.registro import predictores
from ufc.ui import boleta, comunes


def _eventos():
    actuales = comunes.carteleras()
    salida = [e | {"historico": False} for e in actuales]
    natural = lambda e: (e.get("fecha", ""), nombres.normalizar(e["evento"]))  # noqa: E731
    vistos = {natural(e) for e in actuales}
    # Lo guardado conserva el nombre exacto que enlaza con picks.csv. UFCStats lo
    # enriquece con peso/ganador aunque el nombre del evento difiera solo por un acento.
    guardados = predictores.eventos_guardados()
    locales = {natural(e): e for e in cartelera.anteriores()}
    historicos = []
    for evento in guardados:
        local = locales.pop(natural(evento), None)
        if local:
            por_par = {tuple(sorted((nombres.normalizar(p["a"]),
                                     nombres.normalizar(p["b"])))): p
                       for p in local["peleas"]}
            peleas = []
            for pelea in evento["peleas"]:
                clave = tuple(sorted((nombres.normalizar(pelea["a"]),
                                      nombres.normalizar(pelea["b"]))))
                fuente = por_par.get(clave, {})
                enriquecida = pelea | {"peso": fuente.get("peso", pelea.get("peso", ""))}
                if fuente.get("ganador") in {"a", "b"}:
                    ganador_nombre = fuente[fuente["ganador"]]
                    enriquecida["ganador"] = (
                        "a" if nombres.normalizar(ganador_nombre) ==
                        nombres.normalizar(pelea["a"]) else "b")
                peleas.append(enriquecida)
            evento = evento | {"peleas": peleas}
        historicos.append(evento)
    historicos.extend(locales.values())
    for evento in historicos:
        if natural(evento) not in vistos:
            salida.append(evento | {"historico": True})
            vistos.add(natural(evento))
    return salida


def _clave_evento(evento):
    """Valor estable para el widget; no guarda dicts mutables en session_state."""
    return f"{evento.get('fecha', '')}|{evento['evento']}"


def _preseleccionar(eventos):
    """Aplica una tarjeta histórica antes de crear los widgets de esta página."""
    solicitado = st.query_params.get("evento")
    evento = next((e for e in eventos
                   if nombres.normalizar(e["evento"]) == nombres.normalizar(solicitado)), None)
    if evento is not None:
        st.session_state["periodo_predictores"] = (
            "Pasados" if evento.get("historico") else "Próximos")
        st.session_state["evento_predictores_id"] = _clave_evento(evento)
        st.session_state["vista_predictores"] = "Picks"
        st.query_params.clear()


def _valor(tabla, pelea, columna):
    if not len(tabla):
        return None
    fila = tabla[(tabla["a"] == pelea["a"]) & (tabla["b"] == pelea["b"])]
    if not len(fila):
        return None
    valor = fila.iloc[0][columna]
    return None if pd.isna(valor) else valor


def _selector_pelea(pelea, valor, prefijo, detalles=False, valores_detalle=None,
                    disabled=False, resultado=None):
    base = f"{prefijo}_{pelea['a']}_{pelea['b']}"
    if disabled:
        # Si antes hubo una correccion manual, el resultado oficial debe reemplazar
        # tambien ese estado transitorio antes de crear el widget bloqueado.
        st.session_state[base] = valor
    else:
        st.session_state.setdefault(base, valor)
    with st.container(border=True):
        if pelea.get("peso"):
            st.badge(pelea["peso"], color="gray", icon=":material/monitor_weight:")
        ganador = st.segmented_control(
            f"{pelea['a']} vs {pelea['b']}", [pelea["a"], pelea["b"]], key=base,
            width="stretch", disabled=disabled,
            help=("Resultado oficial de UFCStats; no se puede editar."
                  if disabled else
                  "Elegí al ganador; volvé a tocarlo para dejar la pelea sin pick."))
        if ganador and resultado in {"a", "b"}:
            acerto = nombres.normalizar(ganador) == nombres.normalizar(pelea[resultado])
            st.badge(
                f"{ganador} · {'Acertó' if acerto else 'Falló'}",
                color="green" if acerto else "red",
                icon=":material/check_circle:" if acerto else ":material/cancel:",
                help=f"Ganador oficial: {pelea[resultado]}",
            )
        extras = {}
        if detalles:
            valores_detalle = valores_detalle or {}
            st.session_state.setdefault(f"{base}_metodo", valores_detalle.get("metodo"))
            ronda = valores_detalle.get("round")
            st.session_state.setdefault(f"{base}_round", int(ronda) if ronda is not None else None)
            confianza = valores_detalle.get("confianza")
            st.session_state.setdefault(f"{base}_confianza",
                                        round(confianza * 100) if confianza is not None else None)
            with st.popover("Detalles opcionales", icon=":material/tune:"):
                extras["metodo"] = st.selectbox(
                    "Método", [None, *predictores.METODOS], key=f"{base}_metodo",
                    format_func=lambda x: "Sin especificar" if x is None else x.upper())
                extras["round"] = st.number_input(
                    "Round", min_value=1, max_value=5, value=None, step=1,
                    key=f"{base}_round")
                extras["confianza"] = st.number_input(
                    "Confianza publicada", min_value=0, max_value=100, value=None,
                    step=1, key=f"{base}_confianza",
                    help="Ingresá 78 para una confianza publicada de 78%.")
    return ganador, extras


def _cargar_picks(evento, peleas, picks_evento):
    conocidos = sorted(predictores.leer()["predictor"].dropna().unique())
    confirmacion = st.session_state.pop("confirmacion_picks", None)
    if confirmacion:
        st.success(confirmacion, icon=":material/check_circle:")
    quien = st.selectbox(
        "Predictor", conocidos, index=None, accept_new_options=True,
        key="predictor_seleccionado", persist_state="session",
        placeholder="Elegí o escribí un predictor nuevo…",
        help="Presioná Enter después de escribir un nombre nuevo.")
    if not quien:
        st.info("Elegí un predictor para cargar o corregir sus picks.",
                icon=":material/person:")
        return
    guardadas = picks_evento[picks_evento["predictor"] == quien]
    pendientes = len(guardadas) and not guardadas["revisado"].all()
    if pendientes:
        st.warning("Estas picks vienen del archivo anterior y todavía no cuentan para el "
                   "ranking. Revisalas y guardalas para confirmarlas.",
                   icon=":material/rate_review:")

    leidas, imagen = [], None
    if predictores.hay_api():
        imagen = st.file_uploader(
            "Importar imagen", type=["png", "jpg", "jpeg", "webp"],
            help="Gemini prepara un borrador. Nada se guarda hasta que lo revises.")
        if imagen:
            datos = imagen.getvalue()
            leidas = comunes.leer_imagen(datos, imagen.type,
                                          tuple((p["a"], p["b"]) for p in peleas))
            marca = f"{evento['evento']}|{quien}|{hashlib.sha1(datos).hexdigest()[:12]}"
            if st.session_state.get("imagen_picks_aplicada") != marca:
                for item in leidas:
                    p = peleas[item["i"]]
                    base = f"pick_{evento['evento']}_{quien}_{p['a']}_{p['b']}"
                    st.session_state[base] = p[item["ganador"]]
                    st.session_state[f"{base}_metodo"] = item.get("metodo")
                    st.session_state[f"{base}_round"] = item.get("round")
                    st.session_state[f"{base}_confianza"] = \
                        round(item["confianza"] * 100) if item.get("confianza") is not None else None
                st.session_state["imagen_picks_aplicada"] = marca
            st.caption(f"Borrador leído: {len(leidas)} peleas. Confirmá cada selección.")

    resultados = (predictores.resolver_resultados(evento["evento"], peleas)
                  if evento.get("historico") else None)
    filas = []
    for i, pelea in enumerate(peleas):
        lado = _valor(guardadas, pelea, "pick")
        valor = pelea[lado] if lado in {"a", "b"} else None
        resultado = resultados.iloc[i]["ganador"] if resultados is not None else None
        valores_detalle = {c: _valor(guardadas, pelea, c)
                            for c in ("metodo", "round", "confianza")}
        ganador, extras = _selector_pelea(
            pelea, valor, f"pick_{evento['evento']}_{quien}", detalles=True,
            valores_detalle=valores_detalle, resultado=resultado)
        filas.append({"pelea": f"{pelea['a']} vs {pelea['b']}", "ganador": ganador,
                      "metodo": extras.get("metodo"), "round": extras.get("round"),
                      "confianza": (extras["confianza"] / 100
                                    if extras.get("confianza") is not None else None)})
    if st.button(f"Guardar picks de {quien}", type="primary", icon=":material/save:"):
        tabla = pd.DataFrame(filas)
        predictores.guardar(
            predictores.filas_picks(tabla, peleas, evento["evento"], evento["fecha"], quien),
            evento["evento"], quien, origen="gemini" if imagen else "manual", revisado=True)
        st.session_state["confirmacion_picks"] = f"Picks de {quien} guardadas correctamente."
        st.rerun()


def _resultados(evento, peleas):
    resueltos = predictores.resolver_resultados(evento["evento"], peleas)
    oficiales = int((resueltos["origen"] == "ufcstats").sum())
    if oficiales:
        st.caption("Los ganadores de UFCStats son oficiales y no se pueden editar. "
                   "Solo podés completar resultados todavía ausentes de esa fuente.")
    else:
        st.caption("Marcá cada ganador. Podés volver a este evento y corregirlo después.")
    filas_editables, peleas_editables = [], []
    for pelea, resultado in zip(peleas, resueltos.itertuples()):
        lado = resultado.ganador
        valor = pelea[lado] if lado in {"a", "b"} else None
        oficial = resultado.origen == "ufcstats"
        ganador, _ = _selector_pelea(
            pelea, valor, f"resultado_{evento['evento']}", disabled=oficial)
        if not oficial:
            peleas_editables.append(pelea)
            filas_editables.append({"ganador": ganador})
    if peleas_editables and st.button(
            "Guardar resultados", type="primary", icon=":material/save:"):
        tabla = pd.DataFrame(filas_editables)
        predictores.guardar_resultados(
            predictores.filas_resultados(
                tabla, peleas_editables, evento["evento"]), evento["evento"])
        st.toast("Resultados guardados", icon=":material/check_circle:")
        st.rerun()
    elif not peleas_editables:
        st.info("Esta cartelera ya tiene todos sus resultados oficiales.",
                icon=":material/verified:")


def _comparativa(evento, peleas, picks_evento, modelo, estado):
    if evento.get("historico"):
        # El estado servido contiene toda la carrera hasta hoy. Ejecutarlo sobre una
        # pelea pasada filtraría el futuro y falsearía la confirmación del modelo.
        cuotas = [None] * len(peleas)
        preds = [{"error": "sin forecast preevento congelado"}] * len(peleas)
        st.info("En eventos pasados se evalúan las picks humanas congeladas. No se "
                "recalcula hoy el modelo para esa fecha porque usaría historial futuro.",
                icon=":material/history:")
    else:
        tabla = comunes.cuotas_betano()
        cuotas = [betano.buscar(tabla, p["a"], p["b"]) for p in peleas]
        preds = [cartelera.predecir(p, modelo, estado, c, evento["fecha"])
                 for p, c in zip(peleas, cuotas)]
    rank = predictores.ranking(peleas, picks_evento, preds, cuotas)
    if not len(rank) or not (rank["predictores"] > 0).any():
        st.info("Todavía no hay picks revisadas para este evento.",
                icon=":material/how_to_vote:")
        return
    fuertes = int(rank["fuerte"].sum())
    with st.container(horizontal=True):
        st.metric("Peleas con picks", int((rank["predictores"] > 0).sum()), border=True)
        st.metric("Señales fuertes", fuertes, border=True,
                  help="Apoyo humano de 66.7% o más y confirmación del modelo o mercado.")
        st.metric("Picks revisadas", int(picks_evento["revisado"].sum()), border=True)
    for fila in rank.itertuples():
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute",
                              vertical_alignment="center"):
                st.subheader(fila.pelea)
                if fila.seleccion:
                    st.badge(fila.senal, color="green" if fila.fuerte else "gray",
                             icon=":material/verified:" if fila.fuerte
                             else ":material/how_to_vote:")
            comunes.picks_pelea(fila)
            if fila.seleccion and pd.notna(fila.cuota):
                if st.button("Agregar a la boleta", icon=":material/add_shopping_cart:",
                             key=f"rank_bet_{evento['evento']}_{fila.orden}"):
                    boleta.agregar(evento["evento"], evento["fecha"], fila.a, fila.b,
                                   fila.lado, fila.cuota)
                    st.toast("Selección agregada", icon=":material/check:")

    confianza = predictores.confiabilidad()
    if len(confianza):
        st.subheader("Rendimiento de predictores")
        st.dataframe(confianza, hide_index=True, column_config={
            "predictor": st.column_config.TextColumn("Predictor", pinned=True),
            "aciertos": st.column_config.NumberColumn("Aciertos"),
            "total": st.column_config.NumberColumn("Resultados"),
            "carteleras": st.column_config.NumberColumn("Carteleras"),
            "acierto": st.column_config.ProgressColumn("Precisión", format="percent",
                                                         min_value=0, max_value=1),
            "peso": st.column_config.NumberColumn("Peso conservador", format="percent",
                                                    help="Reduce el efecto de muestras cortas."),
        })


def render(modelo, estado):
    eventos = _eventos()
    if not eventos:
        comunes.encabezado("Predictores", "Cargá, corregí y compará picks humanas.",
                           seccion="Consenso")
        st.warning("No hay eventos disponibles.", icon=":material/event_busy:")
        return
    _preseleccionar(eventos)
    proximos = [e for e in eventos if not e.get("historico")]
    pasados = [e for e in eventos if e.get("historico")]
    conocidos = predictores.leer()
    comunes.encabezado(
        "Predictores",
        "Cargá, corregí y compará picks humanas sin editar nombres dentro de tablas.",
        seccion="Consenso",
        chips=[(f"{len(proximos)} próximos", "blue", ":material/event_upcoming:"),
               (f"{len(pasados)} pasados", "gray", ":material/history:"),
               (f"{conocidos['predictor'].nunique()} predictores", "violet",
                ":material/groups:")] if len(conocidos) else
              [(f"{len(proximos)} próximos", "blue", ":material/event_upcoming:"),
               (f"{len(pasados)} pasados", "gray", ":material/history:")])

    periodo_inicial = "Próximos" if proximos else "Pasados"
    st.session_state.setdefault("periodo_predictores", periodo_inicial)
    if st.session_state["periodo_predictores"] == "Próximos" and not proximos:
        st.session_state["periodo_predictores"] = "Pasados"
    if st.session_state["periodo_predictores"] == "Pasados" and not pasados:
        st.session_state["periodo_predictores"] = "Próximos"

    # Los tres controles juntos en una barra: son un solo gesto ("qué evento miro y para
    # qué"), antes estaban sueltos entre medio del contenido.
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            periodo = st.segmented_control(
                "Período", ["Próximos", "Pasados"], key="periodo_predictores")
            st.session_state.setdefault("vista_predictores", "Comparar")
            st.segmented_control("Vista", ["Comparar", "Picks", "Ganadores"],
                                 key="vista_predictores")
        disponibles = proximos if periodo == "Próximos" else pasados
        if disponibles:
            por_clave = {_clave_evento(e): e for e in disponibles}
            if st.session_state.get("evento_predictores_id") not in por_clave:
                st.session_state["evento_predictores_id"] = next(iter(por_clave))
            clave = st.selectbox(
                "Evento", list(por_clave), key="evento_predictores_id",
                format_func=lambda k: f"{por_clave[k].get('fecha') or 'Sin fecha'} · "
                                      f"{por_clave[k]['evento']}", width="stretch")
    if not disponibles:
        st.info(f"No hay eventos {periodo.lower()} en los datos locales.",
                icon=":material/event_busy:")
        return
    evento = por_clave[clave]
    peleas = evento["peleas"]
    picks_evento = predictores.leer(evento["evento"])
    vista = st.session_state["vista_predictores"]
    st.header(evento["evento"], divider="gray")
    if vista == "Picks":
        _cargar_picks(evento, peleas, picks_evento)
    elif vista == "Ganadores":
        _resultados(evento, peleas)
    else:
        _comparativa(evento, peleas, picks_evento, modelo, estado)
