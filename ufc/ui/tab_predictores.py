"""Pestania Predictores: las picks de los tipsters contra el modelo y la cuota."""

import streamlit as st

from ufc.datos import betano, cartelera
from ufc.registro import predictores
from ufc.ui import comunes


def render(modelo, estado):
    st.caption("Dónde coinciden los tipsters, el modelo y la cuota.")
    with st.expander("Cómo leer esto", icon=":material/help:"):
        st.markdown("Las picks de los tipsters que seguís, cruzadas con el modelo y con "
                    "la cuota. El modelo pierde contra la línea de cierre y ellos vienen "
                    "acertando 10-14 de 14, así que acá el modelo es el que desempata, no "
                    "el que decide: lo que se marca es dónde coinciden todos.")
    eventos_p = comunes.carteleras()
    if not eventos_p:
        st.warning("No hay carteleras anunciadas (o la API de ESPN no respondió).")
    else:
        evento_p = st.selectbox("Evento", eventos_p, key="evento_predictores",
                                format_func=lambda e: f"{e['fecha']} — {e['evento']}")
        peleas_p = evento_p["peleas"]
        pares_p = tuple((p["a"], p["b"]) for p in peleas_p)
        nombres_p = [n for par in pares_p for n in par]   # opciones de los dos editores
        tabla_p = comunes.cuotas_betano()
        cuotas_p = [betano.buscar(tabla_p, p["a"], p["b"]) for p in peleas_p]
        preds_p = [cartelera.predecir(p, modelo, estado, c)
                   for p, c in zip(peleas_p, cuotas_p)]
        picks_ev = predictores.leer(evento_p["evento"])

        # La comparativa se arma antes de elegir vista porque el resumen de arriba sale
        # de ella: la vista solo decide que se muestra abajo, no que se calcula.
        comp = predictores.comparar(peleas_p, picks_ev, preds_p, cuotas_p) \
            if len(picks_ev) else None
        lados, cuota_parlay = predictores.parlay(comp) if comp is not None else ([], 0.0)

        with st.container(horizontal=True):
            st.metric("Peleas", len(peleas_p), border=True,
                      help="Peleas anunciadas para este evento.")
            st.metric("Con consenso", int((comp["consenso"] != "").sum())
                      if comp is not None else "—", border=True,
                      help="Peleas donde todos los predictores cargados eligieron el "
                           "mismo lado.")
            st.metric("Locks", int((comp["senal"] == predictores.LOCK).sum())
                      if comp is not None else "—", border=True,
                      help="Consenso total y además el modelo del mismo lado.")
            st.metric("Parlay paga", f"{cuota_parlay:.2f}" if cuota_parlay else "—",
                      border=True,
                      help="Cuota combinada de los locks que ya tienen cuota publicada.")

        # Tres vistas en vez de tres secciones apiladas: cargar picks se usa una vez por
        # tipster por evento y no tiene por que estar arriba de lo que se mira siempre.
        vista = st.segmented_control(
            "Vista", ["Comparativa", "Cargar picks", "Resultados"],
            default="Comparativa", key="vista_predictores",
            label_visibility="collapsed") or "Comparativa"

        if vista == "Cargar picks":
            with st.container(border=True):
                conocidos = sorted(predictores.leer()["predictor"].unique())
                # un desplegable vacio no se ve como algo donde se pueda escribir: hasta
                # que haya alguno guardado, el input suelto es mas claro
                quien = st.selectbox(
                    "Predictor", conocidos, index=None, accept_new_options=True,
                    placeholder="Elegí uno o escribí un nombre nuevo…",
                    help="Escribí el nombre y presioná Enter para agregar uno que "
                         "todavía no exista.") if conocidos else \
                    st.text_input("Predictor", placeholder="Nombre del tipster…",
                                  help="El primero: escribilo y cargá sus picks abajo.")
                leidas = []
                if not quien:
                    st.caption("Elegí de quién son las picks que vas a cargar.")
                elif predictores.hay_api():
                    imagen = st.file_uploader(
                        f"Imagen con las picks de {quien}",
                        type=["png", "jpg", "jpeg", "webp"],
                        help="La infografía o la tabla tal cual la publica. Se lee con "
                             "Gemini y podés corregir lo que salga mal antes de guardar.")
                    if imagen:
                        leidas = comunes.leer_imagen(imagen.getvalue(), imagen.type, pares_p)
                        if not leidas:
                            st.warning("No se reconoció ninguna pelea de esta cartelera "
                                       "en la imagen. Cargalas a mano abajo.")
                else:
                    st.caption("Sin `GEMINI_API_KEY` en el entorno no se pueden leer "
                               "imágenes: cargá las picks a mano.")
            if quien:
                guardadas = picks_ev[picks_ev["predictor"] == quien] if len(picks_ev) \
                    else picks_ev
                if len(guardadas):
                    st.badge(f"{quien} ya tiene {len(guardadas)} picks en este evento — "
                             "editá y volvé a guardar para corregirlas",
                             icon=":material/history:", color="blue")
                editado = st.data_editor(
                    predictores.tabla_picks(peleas_p, guardadas, leidas), hide_index=True,
                    key=f"picks_{evento_p['evento']}_{quien}",
                    column_config={
                        "pelea": st.column_config.TextColumn("Pelea", disabled=True,
                                                             pinned=True),
                        "ganador": st.column_config.SelectboxColumn(
                            "Gana", options=nombres_p,
                            help="Dejalo vacío si no dio pick para esa pelea."),
                        "metodo": st.column_config.SelectboxColumn(
                            "Método", options=predictores.METODOS,
                            help="ko, sub o dec. Vacío si no lo dijo."),
                        "round": st.column_config.NumberColumn("Round", min_value=1,
                                                               max_value=5, step=1),
                        "confianza": st.column_config.NumberColumn(
                            "Confianza", format="percent", min_value=0.0, max_value=1.0,
                            step=0.01,
                            help="Si publica un porcentaje. 78% se carga 0.78."),
                    })
                if st.button(f"Guardar picks de {quien}", type="primary",
                             icon=":material/save:"):
                    predictores.guardar(
                        predictores.filas_picks(editado, peleas_p, evento_p["evento"],
                                                evento_p["fecha"], quien),
                        evento_p["evento"], quien)
                    st.rerun()

        elif vista == "Resultados":
            st.caption("Cargá quién ganó después del evento y se calcula el acierto de "
                       "cada uno. Es a mano a propósito: los resultados de ufcstats "
                       "recién aparecen cuando corrés el pipeline.")
            res_ev = predictores.leer_resultados(evento_p["evento"])
            editado_res = st.data_editor(
                predictores.tabla_resultados(peleas_p, res_ev), hide_index=True,
                key=f"res_{evento_p['evento']}",
                column_config={
                    "pelea": st.column_config.TextColumn("Pelea", disabled=True,
                                                         pinned=True),
                    "ganador": st.column_config.SelectboxColumn("Ganó", options=nombres_p),
                })
            if st.button("Guardar resultados", icon=":material/save:"):
                predictores.guardar_resultados(
                    predictores.filas_resultados(editado_res, peleas_p,
                                                 evento_p["evento"]),
                    evento_p["evento"])
                st.rerun()
            por_evento = predictores.aciertos(por_evento=True)
            if not len(por_evento):
                st.caption("El acierto aparece cuando haya picks y resultados cargados "
                           "del mismo evento.")
            else:
                st.dataframe(por_evento, hide_index=True,
                             column_config={
                                 "predictor": st.column_config.TextColumn("Predictor",
                                                                          pinned=True),
                                 "evento": st.column_config.TextColumn("Evento"),
                                 "aciertos": st.column_config.NumberColumn("Aciertos"),
                                 "total": st.column_config.NumberColumn("Peleas"),
                                 "acierto": st.column_config.ProgressColumn(
                                     "%", format="percent", min_value=0, max_value=1),
                             })

        elif comp is None:
            st.caption("Todavía no hay picks cargadas para este evento: cargalas en la "
                       "vista «Cargar picks».")
        else:
            # Lo que no esta en comunes.COLUMNAS_COMP es una columna de tipster: comparar() las
            # arma segun quien haya subido picks para este evento.
            cols_pred = [c for c in comp.columns if c not in comunes.COLUMNAS_COMP]
            historico = predictores.aciertos().set_index("predictor")
            ayuda = {}
            with st.container(horizontal=True):
                for c in cols_pred:
                    if c in historico.index:
                        f = historico.loc[c]
                        ayuda[c] = (f"Pick de {c}. Acierto histórico {f['acierto']:.0%} "
                                    f"({f['aciertos']} de {f['total']} peleas).")
                        st.metric(c, f"{f['acierto']:.0%}", border=True, help=ayuda[c])
                    else:
                        ayuda[c] = f"Pick de {c}. Todavía sin resultados cargados."
                        st.metric(c, "—", border=True, help=ayuda[c])
            filtro = st.pills(
                "Señal", [predictores.LOCK, predictores.CONTRA, predictores.SPLIT,
                          predictores.SIN_MODELO],
                selection_mode="multi", key="filtro_senal", label_visibility="collapsed")
            st.dataframe(
                comp[comp["senal"].isin(filtro)] if filtro else comp, hide_index=True,
                column_config=comunes.COLUMNAS_COMP | {
                    c: st.column_config.TextColumn(c, help=ayuda[c]) for c in cols_pred})
            if lados:
                with st.container(border=True):
                    st.metric("Parlay de consenso", f"{cuota_parlay:.2f}",
                              help="Las peleas donde coinciden todos los predictores y "
                                   "el modelo, con la cuota de Betano multiplicada.")
                    with st.container(horizontal=True):
                        for quien_l in lados:
                            st.badge(quien_l, icon=":material/check:", color="green")
                    st.caption("Una combinada de N peleas necesita acertar las N, y el "
                               "margen de la casa se multiplica igual que la cuota.")
            else:
                st.caption("No hay ninguna pelea con consenso total más modelo a favor "
                           "y cuota publicada, así que no hay parlay que sugerir.")
