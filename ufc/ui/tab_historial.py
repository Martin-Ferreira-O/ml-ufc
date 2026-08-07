"""Seguimiento hipotetico de cada prediccion congelada."""

import pandas as pd
import streamlit as st

from ufc.modelo import gate
from ufc.registro import ledger
from ufc.ui import comunes


_UMBRAL = "hist_umbral"


def _clv(resumen):
    """El CLV con su intervalo, que es lo unico que puede cerrar o abrir el gate.

    Sin el IC, un CLV medio siempre se puede leer como buena noticia. Con muestra chica
    casi cualquier media es ruido, y el punto de este bloque es que eso se vea en pantalla
    en vez de tener que saberlo de memoria.
    """
    n = resumen.get("clv_n") or 0
    estado = gate.estado()
    with st.container(border=True):
        st.markdown("**CLV económico — el criterio que decide**")
        if not n:
            st.caption("Todavía no hay lados seguidos con cierre registrado. El CLV "
                       "económico compara la apuesta contra la probabilidad justa del "
                       "cierre, que es lo que se puede comparar contra cero.")
        else:
            with st.container(horizontal=True):
                st.metric("CLV medio", f"{resumen['clv_ev']:+.2%}", border=True,
                          delta=f"IC95% [{resumen['clv_lo']:+.1%}, "
                                f"{resumen['clv_hi']:+.1%}]", delta_color="off",
                          help="Retorno esperado de la apuesta valuada a la probabilidad "
                               "justa del cierre. Bootstrap clusterizado por evento: dos "
                               "peleas de la misma cartelera no son independientes.")
                st.metric("Batió al cierre", f"{resumen['clv_batio']:.0%}", border=True,
                          help="Proporción de apuestas que consiguieron mejor precio que "
                               "el de cierre.")
                st.metric("Muestra", f"{n} de {estado.get('n_minimo', '—')}", border=True,
                          help="Apuestas con CLV medido contra el mínimo preregistrado "
                               "en config/gate.json.")
        if estado.get("autorizado"):
            st.success(estado["motivo"], icon=":material/lock_open:")
        else:
            st.error(f"**Gate cerrado.** {estado['motivo']}",
                     icon=":material/do_not_disturb_on:")
        st.caption("El criterio es CLV y no ROI por aritmética, no por preferencia: "
                   "distinguir una ventaja del 2% de cero pide ~20.000 apuestas por ROI "
                   "y decenas por CLV. A ~500 apuestas por año, el ROI como criterio son "
                   "cuarenta años de espera.")


def render():
    # el slider va antes de evaluar: mueve el lado seguido de todas las filas, no filtra
    umbral = st.session_state.get(_UMBRAL, ledger.UMBRAL_VENTAJA * 100) / 100
    df_ledger, resumen = ledger.evaluar(umbral)
    chips = []
    if not df_ledger.empty:
        chips.append((f"{resumen['predicciones']} congeladas", "blue",
                      ":material/ac_unit:"))
        if resumen["con_resultado"]:
            chips.append((f"{resumen['con_resultado']} resueltas", "green",
                          ":material/task_alt:"))
    comunes.encabezado(
        "Seguimiento del modelo",
        "Cada pelea que apareció con cuota queda congelada acá con la predicción y la "
        "cuota de ese momento. Cuando la pelea ocurre se cruza con el resultado y con la "
        "cuota final. Conseguir mejores precios de forma sostenida es una señal útil, "
        "pero esta sección no representa apuestas reales.",
        seccion="Auditoría", chips=chips)

    if df_ledger.empty:
        st.info("Todavía no hay predicciones registradas. Se van guardando solas al "
                "mirar carteleras con cuota publicada.", icon=":material/ac_unit:")
        return

    with st.container(horizontal=True):
        st.metric("Predicciones", resumen["predicciones"], border=True,
                  help="Peleas congeladas con cuota. Las que ya ocurrieron: "
                       f"{resumen['con_resultado']}.")
        if resumen["con_resultado"]:
            st.metric("Acierto modelo", f"{resumen['acierto_modelo']:.0%}", border=True,
                      delta=f"{resumen['acierto_modelo'] - resumen['acierto_mercado']:+.0%}",
                      delta_description="vs mercado",
                      help="Sobre las peleas ya resueltas: cuántas veces el favorito "
                           "del modelo terminó ganando.")
        st.metric("Candidatas históricas", resumen["candidatas"], border=True,
                  help="Filas creadas por la regla anterior. La regla fue retirada y "
                       "las versiones nuevas no generan candidatas automáticas.")
        if pd.notna(resumen["roi_candidatas"]):
            st.metric("ROI regla retirada", f"{resumen['roi_candidatas']:+.1%}",
                      border=True, help="Resultado histórico de la regla anterior.")
        if pd.notna(resumen["extra_medio"]):
            st.metric("Extra por line-shopping", f"{resumen['extra_medio']:+.1%}",
                      border=True,
                      help="Cuánto paga de más la mejor casa sobre Betano, promediando "
                           "los lados seguidos.")
        if pd.notna(resumen["clv_medio"]):
            st.metric("Mejora frente al cierre", f"{resumen['clv_medio']:+.1%}", border=True,
                      help="Cuota registrada vs cuota de cierre del lado seguido. "
                           "Positivo = le ganaste al cierre.")

    _clv(resumen)

    st.slider("Piso de ventaja para seguir un lado", 0, 20,
              int(ledger.UMBRAL_VENTAJA * 100), key=_UMBRAL, format="%d pts",
              help="Puntos de probabilidad que el modelo le tiene que sacar al precio "
                   "tomado. Debajo de este piso ninguna fila sigue un lado. Va en puntos "
                   "y no en EV porque el mismo piso de EV era cuatro veces más fácil de "
                   "pasar del lado del no favorito. Mové y mirá cómo cambian el ROI y el "
                   "CLV de arriba.")

    tabla = comunes.historial(df_ledger)
    hechas = df_ledger["gano"].notna().to_numpy()
    st.caption(f"Cada fila muestra **un solo lado** de la pelea: la marca histórica si "
               f"existió; si no, el lado donde el modelo le saca ventaja a la **mejor "
               f"cuota del mercado**, y solo si esa ventaja supera {umbral:.0%} y el EV "
               f"a ese precio da positivo ({resumen['seguidos']} de "
               f"{resumen['predicciones']} lo pasan). El resto queda sin lado seguido. "
               f"Ninguna fila es una recomendación de apuesta.")

    # Las dos tablas en pestañas: son la misma tabla en dos estados y antes venían una
    # abajo de la otra, así que la de resueltas quedaba fuera de pantalla.
    etiquetas, vistas = [], []
    if not hechas.all():
        etiquetas.append(f"Sin resolver ({(~hechas).sum()})")
        vistas.append(("pendiente", tabla[~hechas].sort_values("fecha")))
    if hechas.any():
        etiquetas.append(f"Resueltas ({hechas.sum()})")
        vistas.append(("resuelta", tabla[hechas].sort_values("fecha", ascending=False)))
    for pestania, (tipo, datos) in zip(st.tabs(etiquetas), vistas):
        with pestania:
            if tipo == "pendiente":
                # el CLV todavia no existe: el "cierre" de un evento futuro es la cuota de hoy
                datos = datos.drop(columns=["ganador", "acerto", "clv", "retorno"])
            st.dataframe(datos, hide_index=True, column_config=comunes.COLUMNAS)
