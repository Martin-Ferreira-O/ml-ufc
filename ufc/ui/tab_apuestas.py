"""Boleta confirmable e historial de apuestas realmente hechas."""

import numpy as np
import pandas as pd
import streamlit as st

from ufc.datos import betano, cartelera
from ufc.registro import apuestas, predictores
from ufc.ui import boleta, comunes


def _clp(valor):
    return f"${valor:,.0f}".replace(",", ".")


def _selecciones_evento(modelo, estado):
    eventos = comunes.carteleras()
    if not eventos:
        st.warning("No hay próximos eventos disponibles.")
        return
    evento = st.selectbox("Evento", eventos, key="evento_apuestas",
                          format_func=lambda e: f"{e['fecha']} · {e['evento']}")
    tabla = comunes.cuotas_betano()
    cuotas = [betano.buscar(tabla, p["a"], p["b"]) for p in evento["peleas"]]
    preds = [cartelera.predecir(p, modelo, estado, c)
             for p, c in zip(evento["peleas"], cuotas)]
    rank = predictores.ranking(evento["peleas"], predictores.leer(evento["evento"]),
                               preds, cuotas)
    st.subheader("Selecciones del evento", divider="gray")
    st.caption("El ranking prioriza picks revisadas. Vos decidís el lado y la cuota real.")
    por_orden = rank.set_index("orden") if len(rank) else pd.DataFrame()
    for i, (pelea, precios) in enumerate(zip(evento["peleas"], cuotas)):
        sugerida = por_orden.loc[i]["lado"] if len(por_orden) and i in por_orden.index else None
        valor = pelea[sugerida] if sugerida in {"a", "b"} else None
        clave = f"bet_pick_{evento['evento']}_{i}"
        st.session_state.setdefault(clave, valor)
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute",
                              vertical_alignment="center"):
                if len(por_orden) and i in por_orden.index and por_orden.loc[i]["seleccion"]:
                    fila = por_orden.loc[i]
                    st.badge(f"Ranking: {fila['seleccion']} · apoyo {fila['apoyo']:.0%}",
                             color="green" if fila["fuerte"] else "gray",
                             icon=":material/how_to_vote:",
                             help=str(fila["senal"]))
                else:
                    st.badge("Sin consenso humano", color="gray",
                             icon=":material/help:")
                if precios:
                    st.badge(f"Betano {precios[0]:.2f} / {precios[1]:.2f}", color="blue",
                             icon=":material/sell:")
            eleccion = st.segmented_control(
                f"{pelea['a']} vs {pelea['b']}", [pelea["a"], pelea["b"]], key=clave,
                width="stretch")
            lado = "a" if eleccion == pelea["a"] else "b" if eleccion == pelea["b"] else None
            precio_base = (precios[0] if lado == "a" else precios[1]) if precios and lado else 2.0
            with st.container(horizontal=True, vertical_alignment="bottom"):
                cuota = st.number_input("Cuota decimal", min_value=1.01,
                                        value=float(precio_base), step=0.01,
                                        key=f"bet_quote_{evento['evento']}_{i}_{lado}",
                                        disabled=lado is None)
                if st.button("Agregar a la boleta", icon=":material/add_shopping_cart:",
                             key=f"bet_add_{evento['evento']}_{i}", disabled=lado is None,
                             type="primary" if lado else "secondary"):
                    if boleta.agregar(evento["evento"], evento["fecha"], pelea["a"],
                                      pelea["b"], lado, cuota):
                        st.toast("Selección agregada", icon=":material/check:")
                    else:
                        st.warning("La boleta ya contiene otro evento. Vaciala antes "
                                   "de cambiar.", icon=":material/shopping_cart_off:")
            if lado is None:
                st.caption("Elegí un lado para habilitar la cuota.")


def _boleta():
    items = boleta.leer()
    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute",
                          vertical_alignment="center"):
            st.subheader("Boleta")
            if items:
                st.badge(f"{len(items)} selecciones" if len(items) > 1
                         else "1 selección", color="blue",
                         icon=":material/shopping_cart:")
        if not items:
            st.caption("Agregá una o más selecciones para registrar una apuesta. "
                       "Nada se guarda hasta que confirmes.")
            return
        st.caption(f"{items[0]['evento']} · {items[0]['fecha']}")
        for i, item in enumerate(list(items)):
            with st.container(border=True, horizontal=True,
                              horizontal_alignment="distribute",
                              vertical_alignment="center"):
                with st.container(gap=None):
                    st.markdown(f"**{item[item['pick']]}**")
                    st.caption(f"{item['a']} vs {item['b']}")
                with st.container(horizontal=True, width="content",
                                  vertical_alignment="center"):
                    st.badge(f"{item['cuota']:.2f}", color="orange",
                             icon=":material/sell:")
                    if st.button("", icon=":material/delete:",
                                 help="Quitar de la boleta", key=f"quitar_boleta_{i}"):
                        boleta.quitar(i)
                        st.rerun()
        if st.button("Vaciar boleta", icon=":material/remove_shopping_cart:",
                     key="vaciar_boleta"):
            boleta.vaciar()
            st.rerun()

        tipo = st.segmented_control("Tipo", ["Simples", "Combinada"], default="Simples",
                                    width="stretch", key="tipo_boleta")
        if tipo == "Simples":
            importes = []
            for i, item in enumerate(items):
                importes.append(st.number_input(
                    f"Importe para {item[item['pick']]}", min_value=100, value=1000,
                    step=500, key=f"importe_simple_{i}", format="%d"))
            total = sum(importes)
            # El cobro potencial estaba solo en la combinada; en simples habia que
            # multiplicar de cabeza cuota por importe, apuesta por apuesta.
            potencial = sum(importe * x["cuota"] for importe, x in zip(importes, items))
            with st.container(horizontal=True):
                st.metric("Total a registrar", f"{_clp(total)}", border=True)
                st.metric("Cobro potencial", f"{_clp(potencial)}", border=True,
                          delta=f"{_clp(potencial - total)} de ganancia",
                          help="Si acertaras todas. No es una proyección de resultado.")
            if st.button("Confirmar apuestas simples", type="primary", width="stretch",
                         icon=":material/check_circle:"):
                for item, importe in zip(items, importes):
                    apuestas.crear(item["evento"], item["fecha"], [item], "simple", importe)
                boleta.vaciar()
                st.toast("Apuestas registradas", icon=":material/check_circle:")
                st.rerun()
        else:
            calculada = float(np.prod([x["cuota"] for x in items]))
            with st.container(horizontal=True):
                cuota = st.number_input("Cuota total tomada", min_value=1.01,
                                        value=calculada, step=0.01, key="cuota_combinada",
                                        help=f"Multiplicada da {calculada:.2f}; poné la "
                                             "que realmente te tomó la casa.")
                importe = st.number_input("Importe", min_value=100, value=1000, step=500,
                                          key="importe_combinada", format="%d")
            with st.container(horizontal=True):
                st.metric("Total a registrar", f"{_clp(importe)}", border=True)
                st.metric("Cobro potencial", f"{_clp(importe * cuota)}", border=True,
                          delta=f"{_clp(importe * cuota - importe)} de ganancia",
                          help="Una combinada cae entera si falla una sola selección.")
            if st.button("Confirmar combinada", type="primary", width="stretch",
                         icon=":material/check_circle:"):
                primero = items[0]
                apuestas.crear(primero["evento"], primero["fecha"], items, "combinada",
                               importe, cuota=cuota)
                boleta.vaciar()
                st.toast("Combinada registrada", icon=":material/check_circle:")
                st.rerun()


def _historial(df, detalle, resumen):
    if not len(df):
        st.info("Todavía no registraste apuestas reales.", icon=":material/receipt_long:")
        return
    with st.container(horizontal=True):
        st.metric("Apostado", _clp(resumen["apostado"]), border=True,
                  help="Suma de todo lo que registraste, liquidado o no.")
        st.metric("Beneficio neto", _clp(resumen["beneficio"]), border=True,
                  delta=f"{resumen['roi']:+.1%} ROI" if pd.notna(resumen["roi"]) else None,
                  delta_description="sobre lo apostado")
        st.metric("Pendiente", _clp(resumen["pendiente"]), border=True,
                  help="En peleas que todavía no se liquidaron.")
        st.metric("Apuestas", len(df), border=True)
    st.caption("Tocá una fila para ver sus selecciones, liquidarla o eliminarla.")
    vista = df[["id", "fecha_evento", "evento", "tipo", "importe_clp", "cuota", "estado",
                "cobro_clp", "beneficio_clp"]].sort_values("fecha_evento", ascending=False)
    seleccion = st.dataframe(
        vista, hide_index=True, on_select="rerun", selection_mode="single-row",
        key="seleccionar_apuesta", column_config={
            "id": st.column_config.TextColumn("ID"),
            "fecha_evento": st.column_config.DateColumn("Fecha"),
            "evento": st.column_config.TextColumn("Evento"),
            "tipo": st.column_config.TextColumn("Tipo"),
            "importe_clp": st.column_config.NumberColumn("Apostado", format="$ %d"),
            "cuota": st.column_config.NumberColumn("Cuota", format="%.2f"),
            "estado": st.column_config.TextColumn("Estado"),
            "cobro_clp": st.column_config.NumberColumn("Cobrado", format="$ %.0f"),
            "beneficio_clp": st.column_config.NumberColumn("Beneficio", format="$ %.0f"),
        })
    if seleccion.selection.rows:
        fila = vista.iloc[seleccion.selection.rows[0]]
        legs = detalle[detalle["apuesta_id"] == fila["id"]]
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute",
                              vertical_alignment="center"):
                st.subheader(f"{fila['evento']} · {fila['tipo']}")
                st.badge(fila["estado"], icon=":material/flag:",
                         color={"ganada": "green", "perdida": "red"}.get(
                             str(fila["estado"]).lower(), "gray"))
            for leg in legs.itertuples():
                elegido = leg.a if leg.pick == "a" else leg.b
                with st.container(border=True, horizontal=True,
                                  horizontal_alignment="distribute",
                                  vertical_alignment="center"):
                    with st.container(gap=None):
                        st.markdown(f"**{elegido}**")
                        st.caption(f"{leg.a} vs {leg.b}")
                    st.badge(str(leg.estado), color="gray", icon=":material/flag:")
            with st.form("liquidar_apuesta", border=False):
                with st.container(horizontal=True, vertical_alignment="bottom"):
                    estado = st.selectbox("Liquidación", apuestas.ESTADOS_MANUALES)
                    cobro = st.number_input("Cobro real", min_value=0, value=0, step=500,
                                            format="%d",
                                            disabled=estado in {"Automático", "Anulada"})
                nota = st.text_input("Nota", placeholder="Cash-out, ajuste o motivo…")
                if st.form_submit_button("Guardar liquidación", type="primary",
                                         icon=":material/save:"):
                    apuestas.liquidar(fila["id"], estado,
                                      None if estado in {"Automático", "Anulada"} else cobro,
                                      nota)
                    st.rerun()
            with st.popover("Eliminar apuesta", icon=":material/delete:"):
                st.caption("Se borra el registro completo y no se puede deshacer.")
                confirmar = st.checkbox("Confirmo que quiero eliminar este registro",
                                        key=f"confirmar_eliminar_{fila['id']}")
                if st.button("Eliminar", type="primary", icon=":material/delete_forever:",
                             disabled=not confirmar, key=f"eliminar_{fila['id']}"):
                    apuestas.eliminar(fila["id"])
                    st.rerun()


def render(modelo, estado):
    items = boleta.leer()
    df, detalle, resumen = apuestas.evaluar()
    chips = []
    if items:
        chips.append((f"{len(items)} en la boleta", "blue", ":material/shopping_cart:"))
    if resumen.get("pendiente"):
        chips.append((f"{_clp(resumen['pendiente'])} sin liquidar", "violet",
                      ":material/hourglass_top:"))
    comunes.encabezado(
        "Apuestas",
        "Registrá únicamente lo que realmente apostaste; las sugerencias no cuentan solas.",
        seccion="Dinero real", chips=chips)
    vista = st.segmented_control("Vista", ["Nueva apuesta", "Rendimiento"],
                                default="Nueva apuesta", width="stretch", key="vista_apuestas")
    if vista == "Rendimiento":
        _historial(df, detalle, resumen)
    else:
        _selecciones_evento(modelo, estado)
        _boleta()
