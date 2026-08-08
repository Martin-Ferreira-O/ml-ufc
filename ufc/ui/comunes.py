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

from ufc.datos import betano, cartelera, fotos, oddsapi
from ufc.ia import analista, store as ia_store
from ufc.intel import identities, store
from ufc.modelo import predict
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


def picks_pelea(fila):
    """El voto humano de una pelea: seleccion, apoyo y quien eligio que.

    `fila` es una fila de `predictores.ranking()`. La dibujan Predictores y Cartelera.
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
    for pick in fila.detalle:
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
