"""Carga, correccion y comparacion de predictores humanos."""

import hashlib

import pandas as pd
import streamlit as st

from ufc.datos import betano, cartelera
from ufc.registro import predictores
from ufc.ui import boleta, comunes


def _eventos():
    actuales = comunes.carteleras()
    salida = [e | {"historico": False} for e in actuales]
    vistos = {e["evento"] for e in actuales}
    historicos = [*cartelera.anteriores(), *predictores.eventos_guardados()]
    for evento in historicos:
        if evento["evento"] not in vistos:
            salida.append(evento | {"historico": True})
            vistos.add(evento["evento"])
    return salida


def _preseleccionar(eventos):
    """Aplica una tarjeta histórica antes de crear los widgets de esta página."""
    solicitado = st.query_params.get("evento")
    evento = next((e for e in eventos if e["evento"] == solicitado), None)
    if evento is not None:
        st.session_state["evento_predictores"] = evento
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


def _selector_pelea(pelea, valor, prefijo, detalles=False, valores_detalle=None):
    base = f"{prefijo}_{pelea['a']}_{pelea['b']}"
    st.session_state.setdefault(base, valor)
    with st.container(border=True):
        ganador = st.segmented_control(
            f"{pelea['a']} vs {pelea['b']}", [pelea["a"], pelea["b"]], key=base,
            width="stretch", help="Elegí al ganador; volvé a tocarlo para dejar la pelea sin pick.")
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
    quien = st.selectbox(
        "Predictor", conocidos, index=None, accept_new_options=True,
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

    filas = []
    for pelea in peleas:
        lado = _valor(guardadas, pelea, "pick")
        valor = pelea[lado] if lado in {"a", "b"} else None
        valores_detalle = {c: _valor(guardadas, pelea, c)
                            for c in ("metodo", "round", "confianza")}
        ganador, extras = _selector_pelea(
            pelea, valor, f"pick_{evento['evento']}_{quien}", detalles=True,
            valores_detalle=valores_detalle)
        filas.append({"pelea": f"{pelea['a']} vs {pelea['b']}", "ganador": ganador,
                      "metodo": extras.get("metodo"), "round": extras.get("round"),
                      "confianza": (extras["confianza"] / 100
                                    if extras.get("confianza") is not None else None)})
    if st.button(f"Guardar picks de {quien}", type="primary", icon=":material/save:"):
        tabla = pd.DataFrame(filas)
        predictores.guardar(
            predictores.filas_picks(tabla, peleas, evento["evento"], evento["fecha"], quien),
            evento["evento"], quien, origen="gemini" if imagen else "manual", revisado=True)
        st.toast("Picks revisadas y guardadas", icon=":material/check_circle:")
        st.rerun()


def _resultados(evento, peleas):
    guardados = predictores.leer_resultados(evento["evento"])
    st.caption("Marcá cada ganador. Podés volver a este evento y corregirlo después.")
    filas = []
    for pelea in peleas:
        lado = _valor(guardados, pelea, "ganador")
        if lado not in {"a", "b"}:
            lado = pelea.get("ganador")
        valor = pelea[lado] if lado in {"a", "b"} else None
        ganador, _ = _selector_pelea(pelea, valor, f"resultado_{evento['evento']}")
        filas.append({"ganador": ganador})
    if st.button("Guardar resultados", type="primary", icon=":material/save:"):
        tabla = pd.DataFrame(filas)
        predictores.guardar_resultados(
            predictores.filas_resultados(tabla, peleas, evento["evento"]), evento["evento"])
        st.toast("Resultados guardados", icon=":material/check_circle:")
        st.rerun()


def _comparativa(evento, peleas, picks_evento, modelo, estado):
    tabla = comunes.cuotas_betano()
    cuotas = [betano.buscar(tabla, p["a"], p["b"]) for p in peleas]
    preds = [cartelera.predecir(p, modelo, estado, c) for p, c in zip(peleas, cuotas)]
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
            st.subheader(fila.pelea)
            if fila.seleccion:
                st.markdown(f"**Selección humana: {fila.seleccion}** · apoyo {fila.apoyo:.0%} "
                            f"({fila.votos} de {fila.predictores})")
                with st.container(horizontal=True):
                    st.badge(fila.senal, color="green" if fila.fuerte else "gray")
                    if fila.modelo_confirma:
                        st.badge("Modelo coincide", color="blue")
                    if fila.mercado_confirma:
                        st.badge("Mercado coincide", color="orange")
                if pd.notna(fila.cuota):
                    if st.button("Agregar a la boleta", icon=":material/add_shopping_cart:",
                                 key=f"rank_bet_{evento['evento']}_{fila.orden}"):
                        boleta.agregar(evento["evento"], evento["fecha"], fila.a, fila.b,
                                       fila.lado, fila.cuota)
                        st.toast("Selección agregada", icon=":material/check:")
            else:
                st.caption("No hay una mayoría humana para esta pelea.")

    confianza = predictores.confiabilidad()
    if len(confianza):
        st.subheader("Rendimiento de predictores")
        st.dataframe(confianza, hide_index=True, column_config={
            "predictor": st.column_config.TextColumn("Predictor", pinned=True),
            "aciertos": st.column_config.NumberColumn("Aciertos"),
            "total": st.column_config.NumberColumn("Resultados"),
            "acierto": st.column_config.ProgressColumn("Precisión", format="percent",
                                                         min_value=0, max_value=1),
            "peso": st.column_config.NumberColumn("Peso conservador", format="percent",
                                                    help="Reduce el efecto de muestras cortas."),
        })


def render(modelo, estado):
    st.header("Predictores")
    st.caption("Cargá, corregí y compará picks humanas sin editar nombres dentro de tablas.")
    eventos = _eventos()
    if not eventos:
        st.warning("No hay eventos disponibles.")
        return
    _preseleccionar(eventos)
    if st.session_state.get("evento_predictores") not in eventos:
        st.session_state["evento_predictores"] = eventos[0]
    evento = st.selectbox("Evento", eventos, index=None, key="evento_predictores",
                          format_func=lambda e: f"{e.get('fecha') or 'Sin fecha'} · {e['evento']}",
                          width="stretch")
    peleas = evento["peleas"]
    picks_evento = predictores.leer(evento["evento"])
    st.session_state.setdefault("vista_predictores", "Comparar")
    vista = st.segmented_control(
        "Vista", ["Comparar", "Picks", "Ganadores"],
        key="vista_predictores", width="stretch")
    if vista == "Picks":
        _cargar_picks(evento, peleas, picks_evento)
    elif vista == "Ganadores":
        _resultados(evento, peleas)
    else:
        _comparativa(evento, peleas, picks_evento, modelo, estado)
