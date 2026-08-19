"""Lo que comparten las cuatro pestanias: constantes de presentacion, los
wrappers cacheados contra las fuentes, y los dos bloques que se dibujan igual en mas
de una pestania.

Los wrappers viven aca y no en cada pestania porque el cache de Streamlit es por
objeto funcion: `carteleras()` la llaman Cartelera y Predictores, y si cada una
tuviera la suya serian dos requests a ESPN.
"""

import datetime
import html

import pandas as pd
import requests
import streamlit as st

from ufc import nombres
from ufc.datos import betano, cartelera, fotos, oddsapi
from ufc.ia import analista, dossier, store as ia_store
from ufc.intel import identities, store
from ufc.modelo import features, predict
from ufc.registro import predictores

# Los nombres internos de las features no le dicen nada a nadie, y el punto de mostrar
# los factores es que se entiendan.
ETIQUETAS = {
    "elo": "Elo", "n_fights": "peleas en UFC", "win_rate": "% de victorias",
    "streak": "racha", "slpm": "golpes conectados por minuto",
    "sapm": "golpes recibidos por minuto", "str_acc": "precisión de golpeo",
    "td_per15": "derribos por 15 min", "td_acc": "precisión de derribo",
    "sub_per15": "intentos de sumisión por 15 min", "ctrl_per_min": "control en el suelo",
    "finish_rate": "% de finalizaciones", "days_since_last": "días desde la última pelea",
    "age": "edad", "height_in": "altura", "reach_in": "alcance",
    "kd_per15": "knockdowns por 15 min", "kd_against_per15": "knockdowns recibidos",
    "finished_against_rate": "% de veces finalizado", "td_def": "defensa de derribo",
    "str_def": "defensa de golpeo", "avg_opp_elo": "calidad de los rivales",
    "mkt_logit": "cuota del mercado",
}
ESTILO = {"alta": st.success, "media": st.warning, "baja": st.error}
# Las dos esquinas. Azul y ambar en vez del rojo/azul de la jaula real: el rojo ya
# significa error en toda la app y un peleador no es un error. Van siempre con el
# porcentaje escrito al lado, el color no es el unico indicador.
COLOR_A, COLOR_B = "#60A5FA", "#F59E0B"
# Emoji y no un badge de markdown porque en la celda de un dataframe el markdown sale
# crudo: solo se renderiza en el overlay que aparece al clickearla.
BADGE = {"alta": "Alta", "media": "Media", "baja": "Baja"}
# Las columnas del ledger son internas ("p_a", "gano" = "a"/"b"): sin esto la tabla
# hay que decodificarla mirando otras dos columnas.
COLUMNAS = {
    "fecha": st.column_config.DateColumn("Fecha", format="DD MMM YYYY"),
    "pelea": st.column_config.TextColumn("Pelea", pinned=True),
    "seguido": st.column_config.TextColumn(
        "Lado seguido", help="La candidata si la hubo; si no, el lado donde el modelo le "
                             "saca ventaja al precio tomado, y solo si supera el piso de "
                             "arriba. Vacío = ningún lado lo pasó. Es seguimiento "
                             "hipotético, no una apuesta real."),
    "p_modelo": st.column_config.ProgressColumn(
        "Modelo", format="percent", min_value=0, max_value=1,
        help="Probabilidad que le da a ese lado el modelo alimentado con la cuota, que "
             "es la que decide el lado seguido. El modelo ciego está comprimido hacia "
             "50% contra el mercado y por sí solo elegía al no favorito siempre."),
    "p_mercado": st.column_config.ProgressColumn(
        "Mercado", format="percent", min_value=0, max_value=1,
        help="Probabilidad implícita en la cuota, sin el margen de la casa."),
    "ventaja": st.column_config.NumberColumn(
        "Ventaja", format="percent",
        help="Puntos de probabilidad que el modelo le saca al precio tomado. Es lo que "
             "tiene que superar el piso de arriba para que la fila siga un lado."),
    "cuota": st.column_config.NumberColumn(
        "Cuota", format="%.2f",
        help="La mejor cuota congelada al registrar: la de Betano o la de la casa que "
             "más pagaba, la que fuera más alta."),
    "extra": st.column_config.NumberColumn(
        "Extra vs Betano", format="percent",
        help="Cuánto suma la mejor casa sobre Betano en ese mismo lado. 0% = Betano "
             "pagaba igual o no había consenso multi-casa al registrar."),
    "confianza": st.column_config.TextColumn(
        "Coincidencia", help="Qué tanto coincide el modelo con el mercado. No es una "
                             "validación económica ni habilita apuestas."),
    "candidata": st.column_config.CheckboxColumn(
        "Candidata histórica", disabled=True,
        help="Marca heredada de una regla retirada; las versiones nuevas no la generan."),
    "ganador": st.column_config.TextColumn("Ganador"),
    "acerto": st.column_config.CheckboxColumn("Acertó", disabled=True),
    "clv": st.column_config.NumberColumn(
        "Mejora vs cierre", format="percent",
        help="Cuota registrada frente a la cuota final. Positivo = conseguiste mejor precio."),
    "retorno": st.column_config.NumberColumn(
        "Retorno", format="%+.2f u", help="Flat-bet de 1 unidad en ese lado."),
}
# Backtest. Una sola tabla de configs para las cuatro vistas: comparten casi todas las
# columnas y tener cuatro dicts casi iguales es como se desincronizan las ayudas.
COLUMNAS_BT = {
    "umbral": st.column_config.NumberColumn("Umbral", format="percent"),
    "nivel": st.column_config.TextColumn("Segmento", pinned=True),
    "calibrador": st.column_config.TextColumn("Calibrador", pinned=True),
    "peleas": st.column_config.NumberColumn("Peleas"),
    "n": st.column_config.NumberColumn("Apuestas", help="Lados que pasan el umbral."),
    "roi": st.column_config.NumberColumn(
        "ROI", format="percent", help="Beneficio sobre lo apostado, stake flat de 1 unidad."),
    "yield": st.column_config.NumberColumn(
        "Yield", format="percent",
        help="Beneficio por apuesta. Con stake flat es el mismo número que el ROI por "
             "definición; se separan solo si el stake varía."),
    "neto": st.column_config.NumberColumn("Neto", format="%+.1f u"),
    "acierto": st.column_config.NumberColumn(
        "Acierto", format="percent",
        help="Secundaria: sube seleccionando favoritos grandes, que es lo que el mercado "
             "ya hace gratis. No mide ventaja."),
    "accuracy": st.column_config.NumberColumn(
        "Acierto", format="percent", help="Secundaria. No mide ventaja."),
    "ev_medio": st.column_config.NumberColumn(
        "EV prometido", format="percent",
        help="Lo que el modelo creía que ganaba por apuesta. Comparalo con el ROI."),
    "n_para_concluir": st.column_config.NumberColumn(
        "Necesita", format="%.0f",
        help="Apuestas que pediría una prueba de potencia (80%, α=0.05) para distinguir "
             "ese ROI de cero. Si es mayor que la columna Apuestas, la fila no concluye "
             "nada por sí sola: no dice 'gana poco', dice 'no se sabe'."),
    "cuota_media": st.column_config.NumberColumn("Cuota media", format="%.2f"),
    "log_loss": st.column_config.NumberColumn("Log loss", format="%.4f"),
    "brier": st.column_config.NumberColumn("Brier", format="%.4f"),
    "ece": st.column_config.NumberColumn(
        "ECE", format="%.4f", help="Expected Calibration Error: |predicho - real| por "
                                   "bin, pesado. 0 = perfectamente calibrado."),
    "lo": st.column_config.NumberColumn("IC95% ↓", format="percent"),
    "hi": st.column_config.NumberColumn("IC95% ↑", format="percent"),
    "veredicto": st.column_config.TextColumn("Veredicto"),
    "queda": st.column_config.CheckboxColumn("Pasa", disabled=True),
    "concluyente": st.column_config.CheckboxColumn(
        "Concluyente", disabled=True,
        help="El IC95% no cruza cero y hay muestra suficiente."),
}

# La comparativa: las columnas fijas. Las de cada predictor se agregan al vuelo, porque
# dependen de quien haya subido picks para ese evento.
COLUMNAS_COMP = {
    "pelea": st.column_config.TextColumn("Pelea", pinned=True),
    "modelo": st.column_config.TextColumn(
        "Modelo", help="El favorito del modelo. Vacío en un debut: no tiene historial."),
    "p_modelo": st.column_config.ProgressColumn(
        "Prob.", format="percent", min_value=0, max_value=1,
        help="Cuánta probabilidad le da el modelo a su propio favorito."),
    "consenso": st.column_config.TextColumn(
        "Consenso", help="El peleador que eligieron todos los predictores. Vacío si "
                         "hay desacuerdo o si a alguno le falta la pick."),
    "cuota": st.column_config.NumberColumn(
        "Cuota", format="%.2f", help="La cuota de Betano del lado del consenso."),
    "ev": st.column_config.NumberColumn(
        "Ventaja estimada", format="percent",
        help="Probabilidad del modelo por cuota, menos uno. No garantiza rentabilidad."),
    "senal": st.column_config.TextColumn("Señal"),
}


@st.cache_resource
def cargar():
    return predict.cargar(), predict.peleadores()


@st.cache_data(ttl=3600, show_spinner="Buscando las carteleras…")
def carteleras():
    try:
        return cartelera.proximas()
    except requests.RequestException:
        return []  # el warning de la pestania ya explica que ESPN no respondio


EVENTO_ACTIVO = "cartelera_evento_activo"
_MESES = ("ene", "feb", "mar", "abr", "may", "jun",
          "jul", "ago", "sept", "oct", "nov", "dic")


def clave_evento(evento):
    """Identidad estable entre reruns y refrescos de la lista de ESPN."""
    return f"{evento['fecha']}|{evento['evento']}"


def fecha_evento(fecha):
    """Fecha ISO de ESPN en una etiqueta corta e independiente del locale del host."""
    try:
        d = datetime.date.fromisoformat(str(fecha)[:10])
    except ValueError:
        return str(fecha)
    return f"{d.day} {_MESES[d.month - 1]} {d.year}"


def meta_evento(evento):
    n = len(evento["peleas"])
    return f"{fecha_evento(evento['fecha'])} · {n} {'pelea' if n == 1 else 'peleas'}"


def selector_cartelera(eventos):
    """Selector de cartelera compartido; devuelve el evento elegido.

    Vive aca porque Resumen y Cartelera tienen que coincidir: la clave de estado es la
    misma, asi que elegir la cartelera del sabado en una la deja elegida en la otra. Sin
    esto, Resumen mostraba `eventos[0]` y un Contender Series tapaba el evento que
    interesa.

    ponytail: si la pagina que dibuja este selector deja de renderizarlo un run entero
    (Cartelera en vista "anteriores"), Streamlit puede descartar el estado del widget y
    la seleccion cae al evento mas cercano — el mismo fallback que ya existia. Si molesta,
    el arreglo es un keep-alive `st.session_state[k] = st.session_state[k]`.
    """
    eventos = sorted(eventos, key=lambda e: e.get("fecha") or "9999-12-31")
    por_clave = {clave_evento(e): e for e in eventos}
    if st.session_state.get(EVENTO_ACTIVO) not in por_clave:
        st.session_state[EVENTO_ACTIVO] = clave_evento(eventos[0])
    proximo = clave_evento(eventos[0])

    def etiqueta(clave):
        e = por_clave[clave]
        prefijo = "Próximo · " if clave == proximo else ""
        return f"{prefijo}{e['evento']} · {meta_evento(e)}"

    st.selectbox("Cartelera", list(por_clave), key=EVENTO_ACTIVO, format_func=etiqueta,
                 help="Las carteleras anunciadas, de la más cercana a la más lejana.")
    return por_clave[st.session_state[EVENTO_ACTIVO]]


# El TTL es tambien el rate limit contra Betano: una request cada media hora.
@st.cache_data(ttl=1800, show_spinner="Buscando las cuotas en Betano…")
def cuotas_betano():
    return betano.cuotas()


# 30 min de cache tambien alcanza para el tier gratis de The Odds API (500/mes).
@st.cache_data(ttl=1800, show_spinner="Buscando el consenso multi-casa…")
def consenso():
    return oddsapi.cuotas()


# Inteligencia y Cartelera leen el mismo informe: cacheado aca, es una sola consulta.
@st.cache_data(ttl=60, show_spinner=False)
def intel_evento():
    return store.ultimo_evento()


@st.cache_data(ttl=60, show_spinner=False)
def intel_perfiles(peleadores):
    return {peleador: identities.profiles_for(peleador) for peleador in peleadores}


# Lee dos archivos locales, no la API: el TTL corto es para que el resultado del boton
# "Analizar con IA" aparezca sin recargar la pagina, no para limitar ningun costo.
@st.cache_data(ttl=60, show_spinner=False)
def ia_veredictos(evento):
    return ia_store.veredictos(evento)


@st.cache_data(ttl=86400, show_spinner="Buscando las fotos…")
def retratos(peleadores):
    """peleador -> URL de su foto servida por Streamlit, o None si no tiene.

    La tupla de nombres es la clave del cache, asi que una pagina entera se resuelve en
    una sola pasada. Despues no vuelve a costar nada: la foto ya esta en disco y el
    browser la cachea por URL. El TTL de un dia es para el debutante que la UFC firma y
    recien ahi tiene ficha.
    """
    return {n: f"app/static/fotos/{f.name}" if f else None
            for n, f in fotos.sincronizar(peleadores).items()}



def cuenta_regresiva(fecha):
    """Cuantos dias faltan, en palabras. Es lo primero que uno mira de una cartelera."""
    try:
        dias = (datetime.date.fromisoformat(str(fecha)[:10]) -
                datetime.date.today()).days
    except ValueError:
        return None
    if dias < 0:
        return "Ya ocurrió"
    return {0: "Hoy", 1: "Mañana"}.get(dias, f"En {dias} días")


def encabezado(titulo, bajada, seccion=None, chips=()):
    """El mismo bloque de titulo en las siete paginas.

    El titulo global vive en la barra lateral, asi que cada pagina se presenta sola:
    linea de seccion, titulo grande y una sola bajada. Los chips son la fila de estado
    de esa pagina (cuantos eventos, si hay cuotas, etc.).
    """
    if seccion:
        st.caption(seccion.upper())
    st.title(titulo)
    st.caption(bajada)
    if chips:
        with st.container(horizontal=True):
            for texto, color, icono in chips:
                st.badge(texto, color=color, icon=icono)


def _monograma(nombre, color, lado):
    """Las iniciales, para el que no tiene foto en ninguna de las dos fuentes: el
    debutante del Contender Series no es roster todavia ni tiene ficha en Sherdog."""
    iniciales = "".join(p[0] for p in nombre.split()[:2]).upper() or "?"
    # `role`/`aria-label` porque dos iniciales sueltas no le dicen nada a un lector de
    # pantalla, mientras que la foto de al lado si tiene su `alt` con el nombre entero.
    return (f'<div role="img" aria-label="{html.escape(nombre)}" '
            f'style="width:{lado}px;height:{lado}px;border-radius:50%;flex:none;'
            f'display:flex;align-items:center;justify-content:center;'
            f'background:{color}1F;border:1px solid {color}55;color:{color};'
            f'font:600 {max(lado // 3, 11)}px \'Barlow Condensed\',sans-serif">'
            f'{html.escape(iniciales)}</div>')


def retrato(url, nombre, color, alto=150):
    """El recorte de cuerpo entero de ufc.com, con su fondo transparente.

    Devuelve HTML y no dibuja: quien lo llama lo compone adentro de un st.html mas
    grande, igual que ya hace `enfrentamiento` con la barra.
    """
    if not url:
        return _monograma(nombre, color, alto * 3 // 5)
    # El recorte de la UFC termina a media pierna con un corte recto. La mascara lo
    # desvanece, que es lo que lo hace leer como cartel y no como una foto pegada.
    mascara = "linear-gradient(#000 76%,transparent)"
    return (f'<img src="{url}" alt="{html.escape(nombre)}" loading="lazy" '
            f'style="height:{alto}px;width:auto;max-width:100%;object-fit:contain;'
            f'-webkit-mask-image:{mascara};mask-image:{mascara};'
            f'filter:drop-shadow(0 8px 20px {color}4D)">')


def avatar(url, nombre, color, lado=52):
    """La cabeza en circulo, recortada por CSS del mismo archivo de cuerpo entero.

    Un solo archivo por peleador sirve para el cartel y para el avatar: `object-position:
    center top` deja la cabeza encuadrada en los recortes de ufc.com, que son 138 de las
    145 fotos. Las 7 de Sherdog son fotos de pesaje y quedan mas lejos, pero se entienden.
    """
    if not url:
        return _monograma(nombre, color, lado)
    return (f'<img src="{url}" alt="{html.escape(nombre)}" loading="lazy" '
            f'style="width:{lado}px;height:{lado}px;flex:none;border-radius:50%;'
            f'object-fit:cover;object-position:center top;background:{color}1F;'
            f'border:1px solid {color}55">')


def enfrentamiento(a, p_a, b, p_b, pie=None, fotos=None):
    """Una sola barra partida con las dos esquinas, en vez de dos barras apiladas.

    Es lo unico del rediseño que no existe como elemento nativo: `st.progress` dibuja
    una barra por valor y dos barras seguidas se leen como dos peleas distintas, no como
    los dos lados de la misma. Va todo con estilos inline para no inyectar CSS global.

    `fotos` es opcional y va al final a proposito: sin el, esto sale igual que siempre,
    que es lo que necesita Matchup — ahi se comparan dos peleadores cualesquiera y la
    pagina no baja fotos.
    """
    izq = min(max(round(p_a * 100), 3), 97)  # los extremos igual tienen que verse
    nom = "font:600 15px Inter,sans-serif;letter-spacing:.01em"
    pct = "font:700 22px 'Barlow Condensed',sans-serif;line-height:1"
    fila = "display:flex;align-items:baseline;justify-content:space-between;gap:12px"
    # Sin "VS" entre las fotos: la fila de nombres que va justo abajo ya lo trae.
    caras = (f'<div style="display:flex;align-items:flex-end;gap:12px;margin-bottom:6px">'
             f'<div style="flex:1;display:flex;justify-content:flex-start">'
             f'{retrato(fotos[0], a, COLOR_A)}</div>'
             f'<div style="flex:1;display:flex;justify-content:flex-end">'
             f'{retrato(fotos[1], b, COLOR_B)}</div></div>') if fotos else ""
    st.html(
        f'<div style="margin:2px 0 6px" role="img" aria-label="'
        f'{html.escape(a)} {p_a:.0%}, {html.escape(b)} {p_b:.0%}">'
        f'{caras}'
        f'<div style="{fila}">'
        f'<span style="{nom};color:{COLOR_A}">{html.escape(a)}</span>'
        f'<span style="font:600 11px Inter,sans-serif;color:#8A94A6;'
        f'letter-spacing:.18em">VS</span>'
        f'<span style="{nom};color:{COLOR_B};text-align:right">{html.escape(b)}</span>'
        f'</div>'
        f'<div style="display:flex;gap:3px;margin:7px 0 5px;height:9px;'
        f'border-radius:999px;overflow:hidden">'
        f'<div style="width:{izq}%;background:{COLOR_A};border-radius:999px"></div>'
        f'<div style="width:{100 - izq}%;background:{COLOR_B};border-radius:999px"></div>'
        f'</div>'
        f'<div style="{fila}">'
        f'<span style="{pct};color:{COLOR_A}">{p_a:.0%}</span>'
        f'<span style="font:400 12px Inter,sans-serif;color:#8A94A6;text-align:center">'
        f'{html.escape(pie or "")}</span>'
        f'<span style="{pct};color:{COLOR_B}">{p_b:.0%}</span>'
        f'</div></div>')


def resultado(a, b, r, cuotas, fotos=None):
    """Las probabilidades de una pelea. Igual en el matchup suelto y en la cartelera."""
    if "aviso" in r:
        st.warning(f"**Historial mezclado.** {r['aviso']}", icon=":material/merge:")
    enfrentamiento(a, r["p_a"], b, r["p_b"], pie="probabilidad del modelo", fotos=fotos)
    with st.container(horizontal=True):
        st.metric("Modelo", f"{r['p_a']:.1%}", border=True,
                  help="No usa la cuota. Es la opinión independiente.")
        if cuotas:
            st.metric("Mercado", f"{r['p_a_mercado']:.1%}", border=True,
                      help="Probabilidad implícita sin el margen de la casa.")
            st.metric("Modelo + mercado", f"{r['p_a_con_odds']:.1%}", border=True,
                      delta=f"{r['p_a_con_odds'] - r['p_a_mercado']:+.1%}",
                      delta_description="vs mercado",
                      help="Medido: no le gana al mercado solo. Está para ver "
                           "cuánto lo mueve el historial, no para apostarlo.")
        st.metric("Cuota mínima", f"{1 / r['p_a']:.2f}", border=True,
                  delta=f"{b} {1 / r['p_b']:.2f}", delta_arrow="off", delta_color="off",
                  help="Debajo de esta cuota el modelo no ve valor. Sirve para comparar "
                       "contra cualquier casa, no solo Betano.")
    st.caption(f"Las cuatro cifras son del lado de **{a}**, salvo donde diga {b}.")
    if cuotas:
        ESTILO[r["confianza"]](
            f"**Coincidencia modelo-mercado {r['confianza']}.** {r['motivo']} "
            "Esta etiqueta se midió con líneas históricas tardías y no valida la cuota "
            "actual de Betano.", icon=":material/compare_arrows:")
        with st.container(horizontal=True):
            st.badge(f"Ventaja {a} {r['ev_a']:+.0%}", icon=":material/trending_up:",
                     color="orange" if r["ev_a"] > 0 else "gray")
            st.badge(f"Ventaja {b} {r['ev_b']:+.0%}", icon=":material/trending_up:",
                     color="orange" if r["ev_b"] > 0 else "gray")
        st.caption("Ventaja estimada = probabilidad × cuota − 1, con el margen de la "
                   "casa adentro.")
        if max(r["ev_a"], r["ev_b"]) > 0:
            st.warning("**Sin apuesta automática.** El EV mostrado usa una probabilidad "
                       "puntual y no supera un límite conservador validado. Queda solo como "
                       "seguimiento experimental.", icon=":material/do_not_disturb_on:")


@st.cache_resource
def percentiles(_estado):
    """-> DataFrame de percentiles 0-1 por stat, sobre el roster entero.

    La barra comparativa necesita una escala comun. El reparto crudo `a/(a+b)` sirve para
    los golpes por minuto (0 a 18) pero no para el Elo, donde los 2726 peleadores caen
    entre 1415 y 1775 y la barra quedaria clavada al medio en todas las peleas. El
    percentil pone cada rubro en la misma escala y ademas aguanta los outliers, que aca
    son reales: `sub_per15` llega a 58 por una sola pelea corta con muchos intentos.

    `age` y `days_since_last` no son columnas del CSV —se recalculan a la fecha del
    evento—, pero su ORDEN no depende de la fecha: todos envejecen y descansan lo mismo.
    Asi que rankear `dob` y `last_date` al reves da el percentil correcto para cualquier
    fecha, y esto se puede cachear una sola vez.
    """
    cols = [c for c in features.FEATURES if c in _estado.columns]
    pct = _estado[cols].rank(pct=True)
    pct["age"] = _estado["dob"].rank(pct=True, ascending=False)
    pct["days_since_last"] = _estado["last_date"].rank(pct=True, ascending=False)
    return pct


_VALOR = "font:700 16px 'Barlow Condensed',sans-serif;line-height:1;white-space:nowrap"
_RUBRO = "font:400 11px Inter,sans-serif;color:#8A94A6;text-align:center;margin-bottom:5px"
_PISTA = "display:flex;gap:3px;height:8px;border-radius:999px;overflow:hidden"


def barras(a, b, filas):
    """Un rubro por fila: el valor de cada esquina y una barra partida entre las dos.

    `filas` = [(etiqueta, texto_a, texto_b, izq, mejor)] con `izq` en 0-100 —o None
    cuando falta el dato— y `mejor` en "a"/"b"/None. Es presentacion pura: no sabe que es
    un percentil ni en que rubros el numero chico es la buena noticia, eso lo resuelve
    `comparativa`.

    Va todo en un solo `st.html`, igual que `enfrentamiento` y por el mismo motivo: veinte
    rubros serian sesenta elementos de Streamlit para dibujar lo que es una tabla.
    """
    trozos = []
    for etiqueta, texto_a, texto_b, izq, mejor in filas:
        # El perdedor va apagado, no gris: el color de su esquina es lo que lo identifica
        # en toda la app, y cambiarselo lo desconecta de su foto y de su cuota.
        opaco = {"a": (1, .4), "b": (.4, 1)}.get(mejor, (1, 1))
        if izq is None:
            pista = f'<div style="{_PISTA};background:#222A38"></div>'
        else:
            pista = (f'<div style="{_PISTA}">'
                     f'<div style="width:{izq}%;background:{COLOR_A};opacity:{opaco[0]};'
                     f'border-radius:999px"></div>'
                     f'<div style="width:{100 - izq}%;background:{COLOR_B};'
                     f'opacity:{opaco[1]};border-radius:999px"></div></div>')
        gana = {"a": f", mejor {a}", "b": f", mejor {b}"}.get(mejor, "")
        trozos.append(
            f'<div role="img" aria-label="{html.escape(etiqueta)}: '
            f'{html.escape(a)} {html.escape(texto_a)}, {html.escape(b)} '
            f'{html.escape(texto_b)}{html.escape(gana)}" '
            f'style="display:grid;grid-template-columns:64px 1fr 64px;gap:12px;'
            f'align-items:center;margin:9px 0">'
            f'<span style="{_VALOR};color:{COLOR_A};opacity:{opaco[0]}">'
            f'{html.escape(texto_a)}</span>'
            f'<div><div style="{_RUBRO}">{html.escape(etiqueta)}</div>{pista}</div>'
            f'<span style="{_VALOR};color:{COLOR_B};opacity:{opaco[1]};text-align:right">'
            f'{html.escape(texto_b)}</span></div>')
    st.html("".join(trozos))


# `dossier.GRUPOS` escribe sus etiquetas sin tildes porque van al prompt de la IA, y
# cambiarlas ahi cambiaria la huella de todos los dossiers ya analizados (o sea, otra
# corrida completa contra la API). Se acentuan al mostrarlas y nada mas. Si alguien
# reescribe una etiqueta esto deja de aplicar solo, que es lo peor que puede pasar.
_TILDES = {"Record": "Récord", "Fisico": "Físico", "Precision": "Precisión",
           "sumision": "sumisión", "Dias": "Días", "ultima": "última"}


def _acentuar(texto):
    for pelado, con_tilde in _TILDES.items():
        texto = texto.replace(pelado, con_tilde)
    return texto


def _rango(pct, clave, col):
    """-> percentil 0-1 de ese peleador en ese rubro, ya dado vuelta si menos es mejor."""
    if clave not in pct.index or col not in pct.columns:
        return None
    valor = pct.loc[clave, col]
    if pd.isna(valor):
        return None
    return 1 - float(valor) if col in dossier.MENOR_MEJOR else float(valor)


def _fila_comparativa(col, etiqueta, formato, perfiles, rangos):
    (perfil_a, perfil_b), (ra, rb) = perfiles, rangos
    texto_a = dossier.fmt((perfil_a or {}).get(col), formato)
    texto_b = dossier.fmt((perfil_b or {}).get(col), formato)
    etiqueta = _acentuar(etiqueta)
    if ra is None or rb is None or not (ra + rb):
        return etiqueta, texto_a, texto_b, None, None
    # El mismo clamp que `enfrentamiento`: una ventaja aplastante igual tiene que dejar
    # ver que del otro lado hay alguien.
    izq = min(max(round(ra / (ra + rb) * 100), 3), 97)
    mejor = None if col in dossier.NEUTRO or ra == rb else ("a" if ra > rb else "b")
    return etiqueta, texto_a, texto_b, izq, mejor


def comparativa(a, b, estado, fecha):
    """Las 26 stats del modelo, rubro por rubro, con la barra inclinada hacia el mejor.

    Reutiliza `dossier.GRUPOS` en vez de una lista propia: es la misma tabla que ve la IA,
    tiene un assert que la mantiene sincronizada con `features.FEATURES`, y una segunda
    lista aca se desincronizaria en silencio la primera vez que entre una feature nueva.
    """
    perfiles = (dossier.perfil(a, estado, fecha), dossier.perfil(b, estado, fecha))
    if not any(perfiles):
        st.caption("Ninguno de los dos tiene historial en UFC: no hay nada que comparar.")
        return
    pct = percentiles(estado)
    claves = (nombres.normalizar(a), nombres.normalizar(b))
    for grupo, filas in dossier.GRUPOS:
        if grupo in dossier.SIN_COMPARAR:
            continue
        st.markdown(f"**{_acentuar(grupo)}**")
        barras(a, b, [_fila_comparativa(col, etiqueta, formato, perfiles,
                                        [_rango(pct, c, col) for c in claves])
                      for col, etiqueta, formato in filas])
    st.caption("La barra reparte el **percentil** de cada uno contra los "
               f"{len(pct)} peleadores del dataset, no los valores crudos: sin eso el Elo "
               "—que va de 1415 a 1775— daría siempre mitad y mitad. Donde menos es mejor "
               "(golpes recibidos, knockdowns recibidos, veces que lo terminaron) la barra "
               "se inclina igual hacia el que está mejor.")


_RESULTADO = {True: ("V", "#34D399"), False: ("D", "#F43F5E")}
# Los mismos metodos, con acento. `dossier.METODO_LARGO` los escribe pelados porque
# alimenta el prompt de la IA, y tocarlo cambiaria la huella de los dossiers ya guardados.
_VIA = {"ko": "KO/TKO", "sub": "sumisión", "dec": "decisión"}


def ultimas_peleas(resumen, n=5):
    """Las ultimas n peleas: rival, Elo actual del rival, resultado, via y fecha."""
    trozos = []
    for u in resumen["ultimas"][:n]:
        marca, color = _RESULTADO[u["gano"]]
        via = _VIA.get(u["metodo"], u["metodo"] or "s/d")
        if u["round"]:
            via += f" · R{u['round']}"
        elo = ("Elo s/d" if u["elo_rival"] is None else f"Elo {u['elo_rival']:.0f}")
        trozos.append(
            f'<div style="display:grid;grid-template-columns:20px 1fr auto;gap:10px;'
            f'align-items:baseline;margin:7px 0">'
            f'<span style="font:700 15px \'Barlow Condensed\',sans-serif;color:{color}" '
            f'aria-label="{"ganó" if u["gano"] else "perdió"}">{marca}</span>'
            f'<div><div style="font:600 14px Inter,sans-serif">'
            f'{html.escape(u["rival"])}</div>'
            f'<div style="font:400 11px Inter,sans-serif;color:#8A94A6">'
            f'{html.escape(u["fecha"])} · {html.escape(via)}</div></div>'
            f'<span style="font:600 13px \'Barlow Condensed\',sans-serif;color:#8A94A6">'
            f'{elo}</span></div>')
    st.html("".join(trozos))


def _desglose(cuenta):
    return ", ".join(f"{n} por {_VIA[m]}" for m, n in cuenta.items() if n) or "ninguna"


def desglose_metodos(resumen):
    """Como gana y como pierde. `fighter_state` tiene el % de finalizaciones, pero no si
    lo noquearon o lo sometieron, que es la primera pregunta que se hace mirando un par."""
    st.caption(f"**Gana:** {_desglose(resumen['gana_por'])} · "
               f"**Pierde:** {_desglose(resumen['pierde_por'])}")


def ficha_peleador(nombre, estado, fecha, resumen, foto=None):
    """La tarjeta de un peleador suelto: quien es, como gana, como pierde y contra quien.

    Es lo que se abre al tocar un rival reciente. No dibuja la comparativa de barras: ahi
    no hay dos esquinas que comparar, hay un peleador solo.
    """
    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.html(avatar(foto, nombre, COLOR_A, lado=44))
            st.markdown(f"### {nombre}")
        clave = nombres.normalizar(nombre)
        if clave in estado.index:
            fila = estado.loc[clave]
            with st.container(horizontal=True, vertical_alignment="center"):
                st.badge(f"Elo {fila['elo']:.0f}", color="blue",
                         icon=":material/military_tech:")
                st.badge(f"Elo medio de sus rivales {fila['avg_opp_elo']:.0f}",
                         color="gray")
                if resumen:
                    st.badge(f"{resumen['record']['w']}-{resumen['record']['l']} en UFC",
                             color="gray", icon=":material/scoreboard:")
                if resumen and resumen["stance"]:
                    st.badge(resumen["stance"], color="gray")
        else:
            st.caption("No está en el dataset del modelo: peleó antes del corte de datos "
                       "o su ficha quedó con otro nombre.")
        if not resumen:
            st.caption("Sin peleas registradas en UFC antes de esta fecha.")
            return
        desglose_metodos(resumen)
        ultimas_peleas(resumen)


def historial(df):
    """El ledger con nombres de peleador en vez de "a"/"b". Una fila = un solo lado."""
    sin_lado = df["lado"] == ""
    # sin lado seguido la fila igual muestra probabilidades: orientadas al favorito del
    # modelo, que es lo unico que queda para leerla
    es_a = (df["p_dec"] > 0.5).where(sin_lado, df["lado"] == "a")
    return pd.DataFrame({
        "pelea": df["a"] + " vs " + df["b"],
        "fecha": df["fecha_evento"],
        "seguido": df["a"].where(es_a, df["b"]).where(~sin_lado),
        "p_modelo": df["p_dec"].where(es_a, 1 - df["p_dec"]),
        "p_mercado": df["p_mercado"].where(es_a, 1 - df["p_mercado"]),
        "ventaja": df["ventaja"].where(es_a, -df["ventaja"]),
        "cuota": df["cuota_lado"],
        "extra": df["extra"],
        "confianza": df["confianza"].map(BADGE),
        "candidata": df["apuesta"].fillna("") != "",
        "ganador": df["a"].where(df["gano"] == "a", df["b"]).where(df["gano"].notna()),
        "acerto": (df["gano"] == df["lado"]).where(~sin_lado),
        "clv": df["clv"],
        "retorno": df["retorno"],
    })


def picks_pelea(fila, detalle=True):
    """El voto humano de una pelea: seleccion, apoyo y quien eligio que.

    `fila` es una fila de `predictores.ranking()`. La dibujan Predictores y Cartelera.

    Con `detalle=False` se corta la lista de quien voto que. Es lo que pide el Resumen:
    once peleas por tres predictores son treinta y tres lineas, y ahi el panel deja de
    ser un panel. El nombre de cada predictor sigue estando en Predictores.
    """
    if not fila.seleccion:
        st.caption("No hay una mayoría humana para esta pelea.")
        return
    st.markdown(f"**Selección humana: {fila.seleccion}**")
    st.progress(float(fila.apoyo),
                text=f"apoyo {fila.apoyo:.0%} · {fila.votos} de "
                     f"{fila.predictores} predictores")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge("Modelo coincide" if fila.modelo_confirma else "Modelo no coincide",
                 color="blue" if fila.modelo_confirma else "gray",
                 icon=":material/psychology:")
        st.badge("Mercado coincide" if fila.mercado_confirma else "Mercado no coincide",
                 color="orange" if fila.mercado_confirma else "gray",
                 icon=":material/storefront:")
        # La IA es el tercer confirmador, al lado del modelo y del mercado. No vota el
        # apoyo humano de arriba: si votara, "consenso humano" dejaria de ser humano.
        if getattr(fila, "ia_eligio", ""):
            st.badge("IA coincide" if fila.ia_confirma else "IA no coincide",
                     color="green" if fila.ia_confirma else "gray",
                     icon=":material/smart_toy:")
        if pd.notna(fila.cuota):
            st.badge(f"Cuota {fila.cuota:.2f}", color="violet", icon=":material/sell:")
    for pick in (fila.detalle if detalle else ()):
        # La precision va al lado de la pick: sin eso, cuatro nombres pesan igual y el
        # que acierta el 62% se lee como el que acierta el 40%.
        historial = (f"{pick['acierto']:.0%} · {pick['total']} resultados"
                     if pick["acierto"] is not None and pick["total"]
                     else "sin historial")
        st.caption(f"**{pick['predictor']}** → {pick['eligio']} · {historial}")


IA_ICONO = {"definido": ":material/check_circle:", "parejo": ":material/balance:"}
IA_FUENTE = {"record": "Récord", "estadistica": "Estadística", "estilo": "Estilos",
             "historial": "Historial", "inteligencia": "Inteligencia",
             "contexto": "Contexto"}


def ia_disponible():
    """La IA necesita key. Sin ella el boton se muestra apagado y con el motivo."""
    return analista.hay_api()


def _ia_lado(fila, pelea):
    quien = pelea["a"] if fila["pick"] == "a" else pelea["b"]
    p = float(fila["p_a_ia"])
    return quien, (p if fila["pick"] == "a" else 1 - p)


def ia_veredicto(fila, pelea):
    """El titular: a quien elige la IA, o que no elige.

    El EV va abajo y aclarado. Lo calculo `analista.ev_contra_mercado` en Python DESPUES
    del veredicto: la IA no vio ninguna cuota, y una linea que no lo diga se lee como si
    hubiera opinado del precio.
    """
    quien, p = _ia_lado(fila, pelea)
    if (fila.get("veredicto") or "definido") == "parejo":
        st.info(f"**La IA no ve un favorito claro.** Se inclina apenas por {quien} "
                f"({p:.0%}), pero considera la pelea pareja.", icon=IA_ICONO["parejo"])
    else:
        st.success(f"**La IA elige a {quien} ({p:.0%}).** Confianza "
                   f"{fila['confianza']}.", icon=IA_ICONO["definido"])

    cuota, ev = fila.get("cuota_tomada"), fila.get("ev_ia")
    if pd.notna(cuota) and pd.notna(ev):
        st.caption(f"Contra el mejor precio publicado ({float(cuota):.2f} en "
                   f"{fila['casa']}) esa probabilidad da un EV de {float(ev):+.1%}. "
                   "Lo calculó el código después; la IA nunca vio esta cuota.")


def ia_apuesta(fila, pelea):
    """Una linea: si la IA da esta pelea para apostar, y a quien.

    No decide nada. `cumple_regla` ya lo resolvio `analista.ev_contra_mercado` cuando
    corrio el analisis, con el piso de EV preregistrado en `config/gate.json` y contra el
    mejor precio de ese momento. Aca solo se muestra lo que quedo escrito en la fila.

    `fila` puede ser None: una cartelera sin analizar tiene que dibujarse igual y decir
    que le falta, no desaparecer la linea.
    """
    if fila is None:
        st.caption(":material/smart_toy: Sin análisis de IA para esta pelea.")
        return

    quien = pelea["a"] if fila["pick"] == "a" else pelea["b"]
    if fila.get("cumple_regla"):
        st.success(f"**Apostar a {quien}** — EV {float(fila['ev_ia']):+.1%} a cuota "
                   f"{float(fila['cuota_tomada']):.2f} en {fila['casa']}.",
                   icon=":material/savings:")
    elif (fila.get("veredicto") or "definido") == "parejo":
        st.caption(f":material/balance: La IA ve la pelea pareja — se inclina apenas por "
                   f"{quien}, pero no la da para apostar.")
    elif pd.notna(fila.get("ev_ia")):
        st.caption(f":material/smart_toy: La IA elige a {quien} (confianza "
                   f"{fila['confianza']}), pero al precio de hoy da un EV de "
                   f"{float(fila['ev_ia']):+.1%}: no llega al piso para apostar.")
    else:
        st.caption(f":material/smart_toy: La IA elige a {quien} (confianza "
                   f"{fila['confianza']}). Sin cuota publicada no se puede medir si "
                   "el precio la paga.")


def ia_tarjeta(fila, pelea):
    """El razonamiento detras del veredicto. No abre expander: la dibuja el de la pelea.

    El titular no se repite aca: lo dibuja `ia_veredicto` afuera, porque tiene que verse
    sin desplegar nada.
    """
    informe = fila.get("informe")
    if not informe:
        st.caption("El informe con el razonamiento no está en esta máquina "
                   "(`data/ia_informes/` no se versiona). El veredicto de arriba sí.")
        return

    if informe.get("razones"):
        st.markdown("**Por qué**")
        for r in informe["razones"]:
            fuente = IA_FUENTE.get(r["fuente"], r["fuente"])
            st.caption(f"**{fuente}** · peso {r['peso']} — {r['texto']}")

    if informe.get("factores_no_modelables"):
        # Este bloque es el motivo de existir de toda la capa: lo que ninguna columna ve.
        st.markdown("**Lo que ninguna estadística captura**")
        for f in informe["factores_no_modelables"]:
            quien = pelea["a"] if f["favorece"] == "a" else pelea["b"]
            st.caption(f"**{f['titulo']}** · favorece a {quien} · {f['certeza']} — "
                       f"{f['explicacion']}")

    if informe.get("contra"):
        st.markdown("**El mejor argumento en contra de su propio pick**")
        st.caption(informe["contra"])

    for bandera in informe.get("banderas", []):
        st.caption(f":material/flag: {bandera}")


def intel_color(score):
    if score is None:
        return "gray"
    return "red" if score <= -2 else "green" if score >= 2 else "gray"


def intel_etiqueta(score):
    if score is None:
        return "Sin analizar"
    if score <= -2:
        return f"{score:+d} · posible impacto negativo"
    if score >= 2:
        return f"{score:+d} · posible impacto positivo"
    return f"{score:+d} · sin impacto material"


def intel_tarjeta(check, perfiles, titulo=None, evidencias=True, foto=None):
    """El informe del bot para un peleador. Igual en Inteligencia y en Cartelera.

    `evidencias=False` omite el desplegable de fuentes: Streamlit no anida expanders y
    en Cartelera la tarjeta ya va dentro de uno.
    """
    report = check["report"]
    score = check["score"]
    color = intel_color(score)
    with st.container(border=True):
        with st.container(horizontal=True, horizontal_alignment="distribute",
                          vertical_alignment="center"):
            with st.container(horizontal=True, width="content",
                              vertical_alignment="center"):
                # La foto va pegada al nombre del encabezado y no adentro del resumen:
                # ahi el nombre es `check["fighter"]`, exacto y estructurado, mientras que
                # el resumen es prosa del LLM donde el nombre puede venir de cualquier
                # forma (apodo, solo apellido, mal escrito).
                # COLOR_A fijo y no el color del score: `intel_color` devuelve nombres de
                # badge de Streamlit ("red"), no hex, y `avatar` los usa como CSS.
                st.html(avatar(foto, check["fighter"], COLOR_A, 44))
                st.subheader(titulo or f"{check['fighter']} vs {check['opponent']}")
            st.badge(intel_etiqueta(score), color=color,
                     icon=":material/report:" if color == "red"
                     else ":material/fact_check:")
        st.write(report.get("resumen") or "Sin resumen.")
        for warning in report.get("advertencias", []):
            st.warning(warning, icon=":material/warning:")

        social = [x for x in perfiles if x.get("confidence") == "official"]
        with st.container(horizontal=True, vertical_alignment="center"):
            if check["confidence"]:
                st.badge(f"Confianza {check['confidence']}", color="blue",
                         icon=":material/verified:")
            st.badge(f"{check['evidence_count']} evidencias", color="gray",
                     icon=":material/inventory_2:")
            for perfil in social:
                st.badge(perfil["platform"].capitalize(), color="violet",
                         icon=":material/link:")
        if social:
            st.caption("Perfiles identificados: " + " · ".join(
                f"[{x['platform'].capitalize()}]({x['url']})" for x in social))
        elif any(x.get("confidence") == "candidate" for x in perfiles):
            st.caption("Wikidata propuso perfiles, pero quedan pendientes de "
                       "validación antes de recolectarlos.")

        evidencias_por_ref = {f"E{x['position']}": x for x in check["evidencias"]}
        for hallazgo in report.get("hallazgos", []):
            refs = [r for r in hallazgo.get("evidencias", [])
                    if r in evidencias_por_ref]
            # Cada hallazgo en su propia caja: antes eran tres parrafos seguidos y no se
            # veia donde terminaba uno y empezaba el siguiente.
            with st.container(border=True):
                with st.container(horizontal=True, horizontal_alignment="distribute",
                                  vertical_alignment="center"):
                    st.markdown(f"**{hallazgo['titulo']}**")
                    st.badge(f"impacto {hallazgo['impacto']:+d}",
                             color=intel_color(hallazgo["impacto"]),
                             icon=":material/adjust:")
                st.caption(f"Certeza: {hallazgo['certeza']}")
                st.write(hallazgo["explicacion"])
                if refs:
                    st.caption("Fuentes: " + " · ".join(
                        f"[{r} — {evidencias_por_ref[r]['source']}]"
                        f"({evidencias_por_ref[r]['url']})" for r in refs))
        if evidencias and check["evidencias"]:
            with st.expander(f"Ver las {len(check['evidencias'])} evidencias",
                             icon=":material/library_books:"):
                for e in check["evidencias"]:
                    st.markdown(f"**E{e['position']} · [{e['title']}]({e['url']})**")
                    st.caption(f"{e['source']} · {e['published_at'] or 'fecha desconocida'}")
                    if e["snippet"]:
                        st.write(e["snippet"])


@st.cache_data(show_spinner="Leyendo la imagen…")
def leer_imagen(datos, mime, pares):
    """La imagen del tipster -> sus picks. Cacheada porque si no, cada tecla que tocás
    en la tabla de abajo es un rerun, y cada rerun sería otro llamado a la API."""
    return predictores.extraer(datos, mime, [{"a": a, "b": b} for a, b in pares])
