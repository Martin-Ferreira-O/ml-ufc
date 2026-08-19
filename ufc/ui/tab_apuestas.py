"""Boleta confirmable e historial de apuestas realmente hechas."""

import numpy as np
import pandas as pd
import streamlit as st

from ufc.datos import betano, cartelera, oddsapi
from ufc.modelo import gate, predict
from ufc.modelo import apuesta as decidir
from ufc.registro import apuestas, banca, predictores
from ufc.ui import boleta, comunes


def _clp(valor):
    return f"${valor:,.0f}".replace(",", ".")


def _gate():
    """El banner de arriba de todo: si el proyecto esta autorizado a apostar, y por que.

    Va antes que cualquier numero de la pantalla. Un stake sugerido debajo de un gate
    cerrado es informacion; arriba, seria una recomendacion.
    """
    estado = gate.estado()
    if estado.get("autorizado"):
        st.success(f"**Gate ABIERTO** (regla {estado.get('version')}). "
                   f"{estado['motivo']}", icon=":material/lock_open:")
    else:
        st.error(f"**Sin apuesta autorizada.** {estado['motivo']}",
                 icon=":material/do_not_disturb_on:")
        if estado.get("n"):
            st.caption(f"Progreso del forward test: {estado['n']} de "
                       f"{estado['n_minimo']} apuestas con CLV medido. La regla está "
                       "congelada en `config/gate.json` y no se toca hasta completarla.")
    return estado


def _banca_actual(resumen):
    """El control de banca. Devuelve la banca vigente, que es el denominador del stake."""
    config = banca.leer()
    actual = banca.actual(resumen)
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="bottom"):
            inicial = st.number_input(
                "Banca declarada", min_value=0, step=10000, format="%d",
                value=int(config["inicial_clp"]),
                help="El denominador de todo. Sin banca no hay stake: un importe suelto "
                     "no es una decisión de riesgo.")
            if st.button("Guardar banca", icon=":material/savings:"):
                banca.guardar(inicial)
                st.rerun()
        if actual:
            st.caption(f"Banca vigente: **{_clp(actual)}** "
                       f"(declarada {_clp(config['inicial_clp'])} "
                       f"{'+' if actual >= config['inicial_clp'] else '−'} "
                       f"{_clp(abs(actual - config['inicial_clp']))} ya liquidado). "
                       "El stake se calcula sobre la vigente, no sobre la inicial.")
        else:
            st.caption("Declará una banca para ver el tamaño de apuesta que "
                       "correspondería. Con banca en 0 el cálculo queda apagado.")
    return actual


def _sugerido(p, cuota, banca_actual, sigma=None):
    """El stake que correspondería, con los parametros congelados de la regla."""
    if not banca_actual or p is None or not np.isfinite(p):
        return None
    par = gate.stake_params()
    return decidir.stake(p, cuota, banca_actual, fraccion=par["fraccion"],
                         tope=par["tope"], sigma=sigma, potencia=par["potencia"])


def _mejor_precio(info, lado):
    """El badge de la casa que más paga ese lado, con el EV contra el consenso limpio.

    Es la única señal de la pantalla que no depende de que el modelo tenga razón: compara
    un precio contra la opinión del resto del mercado. `ev_low` descuenta la dispersión
    entre casas, porque cuando no se ponen de acuerdo el consenso vale menos.
    """
    if not info:
        return
    fila = next((v for v in oddsapi.valor(info) if v["lado"] == lado), None)
    if fila is None:
        return
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(f"Mejor precio {fila['cuota']:.2f} · {fila['casa']}", color="violet",
                 icon=":material/storefront:")
        st.badge(f"vs consenso {fila['ev_low']:+.1%}",
                 color="green" if fila["ev_low"] > 0 else "gray",
                 icon=":material/balance:",
                 help=f"Consenso de {info['casas']} casas desvigueado, EXCLUYENDO a la "
                      f"que ofrece este precio (si no, el precio se compara contra un "
                      f"promedio que lo contiene). Puntual {fila['ev']:+.1%}; el número "
                      f"grande ya descuenta la dispersión entre casas (±{fila['sigma']:.1%}).")
    return fila


def _selecciones_evento(modelo, estado, banca_actual):
    eventos = comunes.carteleras()
    if not eventos:
        st.warning("No hay próximos eventos disponibles.")
        return
    evento = st.selectbox("Evento", eventos, key="evento_apuestas",
                          format_func=lambda e: f"{e['fecha']} · {e['evento']}")
    tabla = comunes.cuotas_betano()
    consenso = comunes.consenso()
    cuotas = [betano.buscar(tabla, p["a"], p["b"]) for p in evento["peleas"]]
    preds = [cartelera.predecir(p, modelo, estado, c)
             for p, c in zip(evento["peleas"], cuotas)]
    rank = predictores.ranking(evento["peleas"], predictores.leer(evento["evento"]),
                               preds, cuotas)
    st.subheader("Selecciones del evento", divider="gray")
    st.caption("El ranking prioriza picks revisadas. Vos decidís el lado y la cuota real.")
    por_orden = rank.set_index("orden") if len(rank) else pd.DataFrame()
    for i, (pelea, precios) in enumerate(zip(evento["peleas"], cuotas)):
        info = oddsapi.buscar(consenso, pelea["a"], pelea["b"]) if consenso else None
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
            fila = _mejor_precio(info, lado) if lado else None
            # El precio por defecto es el mejor del mercado, no el de Betano: tomar el
            # precio mas alto es la unica ventaja que no depende de que el modelo acierte.
            precio_base = 2.0
            if fila:
                precio_base = fila["cuota"]
            elif precios and lado:
                precio_base = precios[0] if lado == "a" else precios[1]
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
            elif fila and banca_actual:
                s = _sugerido(fila["p"], cuota, banca_actual, fila["sigma"])
                st.caption(
                    f"Stake que correspondería: **{_clp(s['monto'])}** "
                    f"({s['fraccion']:.2%} de la banca · {s['limita']}). "
                    f"Kelly pediría {s['kelly']:.1%} sobre una probabilidad de "
                    f"{s['p_decide']:.1%} (cota inferior, no la puntual). "
                    "No es una recomendación: el gate está cerrado.")


def _exposicion(importes, banca_actual):
    """El medidor de exposición del evento. Kelly por apuesta no ve la cartelera entera."""
    if not banca_actual or not importes:
        return
    par = gate.stake_params()
    e = decidir.exposicion(importes, banca_actual, par["tope_evento"])
    texto = (f"Exposición del evento: **{_clp(e['comprometido'])}** = "
             f"{e['fraccion']:.2%} de la banca (tope {par['tope_evento']:.0%} = "
             f"{_clp(e['limite'])}).")
    if e["excede"]:
        st.warning(texto + " **Por encima del tope.** Dos peleas de la misma cartelera "
                   "no son dos apuestas independientes: comparten condiciones y a veces "
                   "el mismo peleador, así que el riesgo conjunto es mayor que la suma "
                   "de los riesgos individuales.", icon=":material/warning:")
    else:
        st.caption(texto)


def _combinada(items, cuota_tomada):
    """La matemática de la combinada, dicha entera.

    La app permite registrarlas porque el usuario las hace; lo que no hace es dejar que
    se vean mejor de lo que son. El vig se compone en cada leg, y sin una distribución
    conjunta el EV no se puede calcular multiplicando marginales.
    """
    tabla = comunes.cuotas_betano()
    legs = []
    for x in items:
        precios = betano.buscar(tabla, x["a"], x["b"])
        if not precios:
            continue
        justa = predict._desvig(1 / precios[0], 1 / precios[1])
        legs.append((justa if x["pick"] == "a" else 1 - justa, float(x["cuota"])))
    if len(legs) < 2:
        return
    ev = decidir.ev_combinada(legs)
    dep = decidir.dependencia_necesaria(legs)
    st.warning(
        f"**EV de esta combinada: {ev:+.1%}** con las probabilidades implícitas del "
        f"mercado (sin vig) y las cuotas que pusiste. Cada leg paga el margen de la casa "
        f"y los márgenes se multiplican: por eso {len(legs)} selecciones que por "
        "separado están cerca de cero terminan bastante abajo.",
        icon=":material/functions:")
    st.caption(
        f"Para que fuera neutra, las selecciones tendrían que salir juntas un "
        f"**{dep['lift'] - 1:+.0%}** más seguido de lo que dice el producto de sus "
        f"probabilidades ({dep['p_necesaria']:.1%} contra {dep['p_independiente']:.1%} "
        f"que implica la independencia)"
        + (f", o sea una correlación de {dep['phi']:+.2f}." if "phi" in dep else ".")
        + " Una combinada solo tiene sentido si los resultados están correlacionados y "
        "la casa los precia como si no lo estuvieran. `config/gate.json` las prohíbe "
        "dentro de la regla preregistrada; registrarlas acá es para llevar la cuenta.")


def _boleta(banca_actual=0.0):
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
                # El default sale del tope por apuesta de la regla congelada, no de un
                # $1.000 escrito a mano: el importe por defecto es una decision de riesgo
                # y tiene que salir de la banca.
                sugerido = int(banca_actual * gate.stake_params()["tope"]) or 1000
                importes.append(st.number_input(
                    f"Importe para {item[item['pick']]}", min_value=100,
                    value=max(sugerido, 100), step=500, key=f"importe_simple_{i}",
                    format="%d",
                    help=f"Por defecto el tope por apuesta de `config/gate.json` "
                         f"({gate.stake_params()['tope']:.0%} de la banca)."
                         if banca_actual else "Declará una banca para que el importe "
                                              "por defecto salga de ella."))
            _exposicion(importes, banca_actual)
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
            _combinada(items, cuota)
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
        _gate()
        banca_actual = _banca_actual(resumen)
        _selecciones_evento(modelo, estado, banca_actual)
        _boleta(banca_actual)
