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
    st.subheader("Selecciones del evento")
    st.caption("El ranking prioriza picks revisadas. Vos decidís el lado y la cuota real.")
    por_orden = rank.set_index("orden") if len(rank) else pd.DataFrame()
    for i, (pelea, precios) in enumerate(zip(evento["peleas"], cuotas)):
        sugerida = por_orden.loc[i]["lado"] if len(por_orden) and i in por_orden.index else None
        valor = pelea[sugerida] if sugerida in {"a", "b"} else None
        clave = f"bet_pick_{evento['evento']}_{i}"
        st.session_state.setdefault(clave, valor)
        with st.container(border=True):
            eleccion = st.segmented_control(
                f"{pelea['a']} vs {pelea['b']}", [pelea["a"], pelea["b"]], key=clave,
                width="stretch")
            if len(por_orden) and i in por_orden.index:
                fila = por_orden.loc[i]
                if fila["seleccion"]:
                    st.caption(f"Ranking: {fila['seleccion']} · apoyo {fila['apoyo']:.0%} · "
                               f"{fila['senal'].lower()}.")
            lado = "a" if eleccion == pelea["a"] else "b" if eleccion == pelea["b"] else None
            precio_base = (precios[0] if lado == "a" else precios[1]) if precios and lado else 2.0
            cuota = st.number_input("Cuota decimal", min_value=1.01, value=float(precio_base),
                                    step=0.01, key=f"bet_quote_{evento['evento']}_{i}_{lado}",
                                    disabled=lado is None)
            if st.button("Agregar", icon=":material/add_shopping_cart:",
                         key=f"bet_add_{evento['evento']}_{i}", disabled=lado is None):
                if boleta.agregar(evento["evento"], evento["fecha"], pelea["a"], pelea["b"],
                                  lado, cuota):
                    st.toast("Selección agregada", icon=":material/check:")
                else:
                    st.warning("La boleta ya contiene otro evento. Vaciala antes de cambiar.")


def _boleta():
    items = boleta.leer()
    st.subheader(f"Boleta ({len(items)})")
    if not items:
        st.caption("Agregá una o más selecciones para registrar una apuesta.")
        return
    for i, item in enumerate(list(items)):
        with st.container(horizontal=True, vertical_alignment="center"):
            nombre = item[item["pick"]]
            st.write(f"**{nombre}** · {item['a']} vs {item['b']} · {item['cuota']:.2f}")
            if st.button("", icon=":material/delete:", help="Quitar de la boleta",
                         key=f"quitar_boleta_{i}"):
                boleta.quitar(i)
                st.rerun()
    tipo = st.segmented_control("Tipo", ["Simples", "Combinada"], default="Simples",
                                width="stretch", key="tipo_boleta")
    if tipo == "Simples":
        importes = []
        for i, item in enumerate(items):
            importes.append(st.number_input(
                f"Importe para {item[item['pick']]}", min_value=100, value=1000, step=500,
                key=f"importe_simple_{i}", format="%d"))
        total = sum(importes)
        st.caption(f"Total a registrar: {_clp(total)} CLP")
        if st.button("Confirmar apuestas simples", type="primary",
                     icon=":material/check_circle:"):
            for item, importe in zip(items, importes):
                apuestas.crear(item["evento"], item["fecha"], [item], "simple", importe)
            boleta.vaciar()
            st.toast("Apuestas registradas", icon=":material/check_circle:")
            st.rerun()
    else:
        calculada = float(np.prod([x["cuota"] for x in items]))
        cuota = st.number_input("Cuota total tomada", min_value=1.01, value=calculada,
                                step=0.01, key="cuota_combinada")
        importe = st.number_input("Importe", min_value=100, value=1000, step=500,
                                  key="importe_combinada", format="%d")
        st.caption(f"Cobro potencial: {_clp(importe * cuota)} CLP")
        if st.button("Confirmar combinada", type="primary", icon=":material/check_circle:"):
            primero = items[0]
            apuestas.crear(primero["evento"], primero["fecha"], items, "combinada",
                            importe, cuota=cuota)
            boleta.vaciar()
            st.toast("Combinada registrada", icon=":material/check_circle:")
            st.rerun()


def _historial():
    df, detalle, resumen = apuestas.evaluar()
    if not len(df):
        st.info("Todavía no registraste apuestas reales.", icon=":material/receipt_long:")
        return
    with st.container(horizontal=True):
        st.metric("Apostado", _clp(resumen["apostado"]), border=True)
        st.metric("Beneficio neto", _clp(resumen["beneficio"]), border=True,
                  delta=f"{resumen['roi']:+.1%} ROI" if pd.notna(resumen["roi"]) else None)
        st.metric("Pendiente", _clp(resumen["pendiente"]), border=True)
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
        st.markdown("**Selecciones**")
        for leg in legs.itertuples():
            elegido = leg.a if leg.pick == "a" else leg.b
            st.caption(f"{elegido} · {leg.a} vs {leg.b} · {leg.estado}")
        with st.form("liquidar_apuesta"):
            estado = st.selectbox("Liquidación", apuestas.ESTADOS_MANUALES)
            cobro = st.number_input("Cobro real", min_value=0, value=0, step=500,
                                    format="%d", disabled=estado in {"Automático", "Anulada"})
            nota = st.text_input("Nota", placeholder="Cash-out, ajuste o motivo…")
            if st.form_submit_button("Guardar liquidación", icon=":material/save:"):
                apuestas.liquidar(fila["id"], estado,
                                   None if estado in {"Automático", "Anulada"} else cobro,
                                   nota)
                st.rerun()
        with st.popover("Eliminar apuesta", icon=":material/delete:"):
            confirmar = st.checkbox("Confirmo que quiero eliminar este registro",
                                    key=f"confirmar_eliminar_{fila['id']}")
            if st.button("Eliminar", type="primary", icon=":material/delete_forever:",
                         disabled=not confirmar, key=f"eliminar_{fila['id']}"):
                apuestas.eliminar(fila["id"])
                st.rerun()


def render(modelo, estado):
    st.header("Apuestas")
    st.caption("Registrá únicamente lo que realmente apostaste; las sugerencias no cuentan solas.")
    vista = st.segmented_control("Vista", ["Nueva apuesta", "Rendimiento"],
                                default="Nueva apuesta", width="stretch", key="vista_apuestas")
    if vista == "Rendimiento":
        _historial()
    else:
        _selecciones_evento(modelo, estado)
        _boleta()
