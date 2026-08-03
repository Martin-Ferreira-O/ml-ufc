"""Features point-in-time por pelea, sin leakage temporal.

Recorre las peleas en orden cronologico manteniendo el estado acumulado de cada
peleador. Para cada pelea, las features salen del estado *previo*; recien despues
se actualiza con el resultado. Las peleas del mismo dia se calculan todas antes de
aplicar sus updates (torneos de los UFC 1-8: un peleador pelea dos veces el mismo dia).

Cada pelea produce dos filas: diffs A-B y la espejada B-A con el target invertido.
"""

import os
import pathlib
import re
import tempfile

import numpy as np
import pandas as pd

from ufc import rutas
from ufc.modelo import predict, settlement

RAW = rutas.RAW
OUT = rutas.DATOS / "features.csv"
AVISOS = rutas.DATOS / "wiki_avisos.csv"
WIKI_EVENTOS = rutas.DATOS / "wiki_eventos.csv"
PREVIO = rutas.DATOS / "sherdog_previo.csv"
K_ELO = 32

# nombre -> como se calcula desde el estado previo del peleador
FEATURES = [
    "elo", "n_fights", "win_rate", "streak", "slpm", "sapm", "str_acc",
    "td_per15", "td_acc", "sub_per15", "ctrl_per_min", "finish_rate",
    "days_since_last", "age", "height_in", "reach_in",
    "kd_per15", "kd_against_per15", "finished_against_rate",
    "td_def", "str_def", "avg_opp_elo",
    # Circunstancia de ESTA pelea, no historial: sale de Wikipedia via wiki.py. Medido
    # -0.0020 IC95% [-0.0033, -0.0006] en el rolling-origin de 20 folds — el unico bloque
    # que paso el protocolo. Crudo: el que entra de reemplazo gana 39.1% (n=860) y el que
    # no da el peso 41.1% (n=253), contra 50% de base. Ninguna feature acumulada lo ve.
    "reemplazo", "peso_no_dado",
    # Como ganaba y como perdia ANTES de UFC, del circuito regional (sherdog.py). Medido
    # -0.0022 IC95% [-0.0039, -0.0006]. Tapa el agujero mas grande que quedaba: en el 25%
    # de las peleas hay un debutante, que sin esto entra con n_fights=0 y elo default.
    # Crudo, y esto es lo que hay que entender de la feature: el que llega con MAS peleas
    # regionales gana el 47.4%, no mas — el veterano de circuito no es un prospecto. Las
    # victorias previas no informan (48.9%, sesgo de seleccion: a UFC no llega nadie con
    # record perdedor); informan las derrotas (44.7%) y sobre todo los KO recibidos
    # (43.8%), que es lo unico que el filtro de entrada no borra.
    "prev_ko_w", "prev_sub_w", "prev_ko_l", "prev_sub_l",
]
# Contexto simetrico de la pelea (vale igual en las dos filas espejadas). Para el
# target de ganador esta medido no concluyente (rolling-origin 2026-07-31), pero es
# el insumo principal del modelo de metodo: las tasas de finish cambian por division.
CONTEXTO = ["wc_lbs", "mujer", "cinco_r"]

# Candidatas EN MEDICION. Se calculan y viajan en features.csv, pero NO entran al modelo:
# `python -m ufc.modelo.train probar` las mide de a bloques contra el baseline y solo
# pasan a FEATURES las que dejan el IC95% del delta pareado sin tocar cero.
#
# La ronda anterior probo TODO lo que quedaba sin usar en los CSVs de ufcstats — mezcla de
# golpeo por objetivo y posicion, cardio por round, calidad de las decisiones, peleas de
# titulo, stance — y salieron las seis no concluyentes, con deltas dentro de +-0.0005.
# El diagnostico quedo claro al mirar AUC contra correlacion: lo que tenia senial ya estaba
# capturado (title_win_rate correlaciona 0.59 con win_rate, ground_pct 0.64 con
# ctrl_per_min) y lo genuinamente ortogonal es ruido (fade AUC 0.496, es_southpaw 0.518).
# Por eso las candidatas de aca ya no salen del historial deportivo: salen de afuera.
# Ya promovido a FEATURES: "circunstancia" (reemplazo + peso_no_dado), -0.0020.
# Ya descartado: "historial_peso" (peso_no_dado_rate, la tasa acumulada de no dar el peso),
# +0.0000 IC95% [-0.0003, +0.0004] — el habito no informa, la circunstancia si.
# Ya promovido a FEATURES: "previo_metodo" (prev_ko_w/sub_w/ko_l/sub_l), -0.0022.
# Ya descartado: "previo" (prev_n, prev_w, prev_l, el record pre-UFC crudo). Pasa solo
# (-0.0014 [-0.0028, -0.0001]) pero no aporta NADA arriba del metodo: los siete juntos
# rinden -0.0021, menos que los cuatro solos. Lo que informa no es cuanto peleo el tipo
# antes, es como terminaban esas peleas.
BLOQUES = {}
CANDIDATAS = [c for cols in BLOQUES.values() for c in cols]
_TODAS = FEATURES + CANDIDATAS
# las de sherdog, esten promovidas o en medicion: se leen del CSV igual
_PREV = [c for c in _TODAS if c.startswith("prev_")]
_LBS = [("Strawweight", 115), ("Flyweight", 125), ("Bantamweight", 135),
        ("Featherweight", 145), ("Lightweight", 155), ("Welterweight", 170),
        ("Middleweight", 185), ("Light Heavyweight", 205), ("Heavyweight", 265)]
# El mercado no sale del historial del peleador: es una fuente aparte, y por eso vive en
# su propio conjunto. Se entrena un modelo con cada uno para poder mostrar los dos.
MERCADO = "mkt_logit"
FEATURES_ODDS = FEATURES + [MERCADO]
# Props de metodo del mercado: simetricas ante el espejado (suman las dos esquinas), asi
# que son legales en el modelo de metodo, cuyo target tambien es simetrico.
PROPS = ["p_ko_mkt", "p_sub_mkt", "p_dec_mkt"]
CONTEXTO_ODDS = CONTEXTO + PROPS


def _num(x):
    """'19 of 39' -> (19, 39); '--' o vacio -> (nan, nan)."""
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)", str(x))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def _ctrl_seconds(x):
    """'m:ss' -> segundos; '--' (no se registro control) -> nan, no cero."""
    m = re.match(r"\s*(\d+):(\d+)", str(x))
    return float(m.group(1)) * 60 + float(m.group(2)) if m else np.nan


def _inches(x):
    """'5\\' 11\"' -> 71; '66\"' -> 66; '--' -> nan."""
    s = str(x)
    m = re.match(r"(\d+)'\s*(\d+)", s)
    if m:
        return float(m.group(1)) * 12 + float(m.group(2))
    m = re.match(r"(\d+)\"", s)
    return float(m.group(1)) if m else np.nan


def _fight_minutes(row):
    """Duracion total usando la secuencia real declarada en ``TIME FORMAT``."""
    m = re.match(r"\s*(\d+):(\d+)", str(row["TIME"]))
    if not m:
        return np.nan
    last = float(m.group(1)) + float(m.group(2)) / 60
    try:
        round_number = int(row["ROUND"])
    except (TypeError, ValueError):
        return np.nan
    declared = re.search(r"\((\d+(?:-\d+)*)\)", str(row.get("TIME FORMAT", "")))
    if round_number <= 1 or not declared:
        # No Time Limit solo tiene un round; para un formato desconocido con rounds
        # previos no se inventan bloques de cinco minutos.
        return last if round_number == 1 else np.nan
    durations = [float(x) for x in declared.group(1).split("-")]
    if round_number - 1 > len(durations):
        return np.nan
    return sum(durations[:round_number - 1]) + last


# stats por pelea que se acumulan tal cual: propias y, espejadas, las del rival
_MIAS = ("sig_l", "sig_a", "td_l", "td_a", "sub", "ctrl", "kd")
# contador propio -> columna del rival de la que sale (defensas y golpeo recibido)
_SUYAS = {"sig_abs": "sig_l", "sig_abs_a": "sig_a", "kd_abs": "kd",
          "td_abs_l": "td_l", "td_abs_a": "td_a"}


def _new_state():
    st = dict(n=0, w=0, streak=0, finishes=0, last_date=None, elo=1500.0,
              finished_against=0, opp_elo=0.0)
    exposiciones = ("sig_for_minutes", "sig_against_minutes", "td_minutes",
                    "sub_minutes", "ctrl_minutes", "kd_minutes",
                    "kd_against_minutes")
    return {**st, **{k: 0.0 for k in (*_MIAS, *_SUYAS, *exposiciones)}}


def _snapshot(st, date, p):
    """Features de un peleador a la fecha `date`, desde su estado previo.

    `p` es su fila de fisicos (o None si ufcstats no lo tiene).
    """
    n = st["n"]
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    dob = p["dob"] if p is not None else pd.NaT
    return {
        "elo": st["elo"],
        "n_fights": float(n),
        "win_rate": div(st["w"], n),
        "streak": float(st["streak"]) if n else np.nan,
        "slpm": div(st["sig_l"], st["sig_for_minutes"]),
        "sapm": div(st["sig_abs"], st["sig_against_minutes"]),
        "str_acc": div(st["sig_l"], st["sig_a"]),
        "td_per15": div(st["td_l"], st["td_minutes"]) * 15
                    if st["td_minutes"] else np.nan,
        "td_acc": div(st["td_l"], st["td_a"]),
        "sub_per15": div(st["sub"], st["sub_minutes"]) * 15
                     if st["sub_minutes"] else np.nan,
        "ctrl_per_min": div(st["ctrl"] / 60, st["ctrl_minutes"]),
        "finish_rate": div(st["finishes"], n),
        "kd_per15": div(st["kd"], st["kd_minutes"]) * 15
                    if st["kd_minutes"] else np.nan,
        "kd_against_per15": div(st["kd_abs"], st["kd_against_minutes"]) * 15
                            if st["kd_against_minutes"] else np.nan,
        "finished_against_rate": div(st["finished_against"], n),
        "td_def": 1 - div(st["td_abs_l"], st["td_abs_a"]),
        "str_def": 1 - div(st["sig_abs"], st["sig_abs_a"]),
        "avg_opp_elo": div(st["opp_elo"], n),
        "days_since_last": (date - st["last_date"]).days if st["last_date"] else np.nan,
        "age": (date - dob).days / 365.25 if pd.notna(dob) else np.nan,
        "height_in": p["height_in"] if p is not None else np.nan,
        "reach_in": p["reach_in"] if p is not None else np.nan,
    }


def _norm(s):
    return s.astype(str).str.lower().str.split().str.join(" ")


def _mercado(results):
    """logit de la probabilidad implicita de fighter_a, sin vig.

    Ya es antisimetrico: al espejar la pelea se niega solo, igual que los diffs. Las dos
    implicitas suman >1 (ese exceso es el margen de la casa), asi que se normalizan.
    Sin el CSV de odds, o para peleas sin match, queda NaN y el modelo base no se entera.
    """
    vacio = pd.Series(np.nan, index=results.index)
    archivo = RAW / "ufc_odds.csv"
    if not archivo.exists():
        return vacio

    o = pd.read_csv(archivo, low_memory=False)
    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    # cuota americana -> probabilidad implicita. El denominador comun evita el inf de
    # 100/(x+100) cuando x == -100 exacto (np.where evalua las dos ramas igual).
    implicita = lambda x: np.where(x < 0, np.abs(x), 100.0) / (np.abs(x) + 100)  # noqa: E731
    pr, pb = implicita(o["R_odds"].to_numpy(float)), implicita(o["B_odds"].to_numpy(float))
    p_r = predict._desvig(pr, pb)  # power, no proporcional: medido mejor (ver predict)

    # clave simetrica (fecha + par ordenado): el orden R/B de la fuente no tiene por que
    # coincidir con nuestro A/B, asi que se guarda la probabilidad del nombre menor
    r, b = _norm(o["R_fighter"]), _norm(o["B_fighter"])
    menor = np.minimum(r, b)
    tabla = dict(zip(zip(o["date"].dt.date, menor, np.maximum(r, b)),
                     np.where(r == menor, p_r, 1 - p_r)))

    a, b2 = _norm(results["fighter_a"]), _norm(results["fighter_b"])
    p = np.array([tabla.get(k, np.nan) for k in
                  zip(results["DATE"].dt.date, np.minimum(a, b2), np.maximum(a, b2))])
    p = np.where(a.to_numpy() <= b2.to_numpy(), p, 1 - p)  # orientar hacia fighter_a
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return pd.Series(np.log(p / (1 - p)), index=results.index)


_SIN_AVISO = {"reemplazo": 0.0, "peso_no_dado": 0.0}


def _avisos():
    """-> (dict (evento, peleador) -> aviso, set de eventos con articulo validado).

    Sale de `wiki.py`. Sin esos CSVs queda todo NaN y el modelo base no se entera.
    """
    if not (AVISOS.exists() and WIKI_EVENTOS.exists()):
        return {}, set()
    a = pd.read_csv(AVISOS)
    tabla = {(r.evento, r.peleador): {"reemplazo": r.reemplazo,
                                      "peso_no_dado": r.peso_no_dado}
             for r in a.itertuples()}
    return tabla, set(pd.read_csv(WIKI_EVENTOS)["evento"])


def _aviso(tabla, cubiertos, evento, nombre):
    """En un evento CON articulo, no ser mencionado significa 0: no hubo aviso.

    En uno sin articulo significa NaN. Confundir las dos cosas le pondria "todo normal"
    a media base y volveria la feature ruido con cara de dato.
    """
    if evento not in cubiertos:
        return {k: np.nan for k in _SIN_AVISO}
    return dict(tabla.get((evento, nombre), _SIN_AVISO))


def _previos():
    """-> dict peleador -> record pre-UFC, de `sherdog.py`. Sin el CSV queda todo NaN.

    No es estado acumulado: es una condicion inicial fija por peleador, anterior a su
    debut, asi que no se actualiza nunca y no puede filtrar futuro. Los 18 peleadores sin
    ficha quedan en NaN — un 0 ahi seria decir "no peleo antes", que es otra cosa.
    """
    if not PREVIO.exists():
        return {}
    p = pd.read_csv(PREVIO)
    return {r.peleador: {c: float(getattr(r, c)) for c in _PREV}
            for r in p.itertuples()}


def _clave(fechas, x, y):
    """Clave simetrica (fecha + par ordenado) para cruzar con la fuente de cuotas.

    El orden R/B de la fuente no tiene por que coincidir con nuestro A/B.
    """
    return list(zip(fechas, np.minimum(x, y), np.maximum(x, y)))


def _props(results):
    """P(ko), P(sub), P(dec) del mercado. Simetricas: valen igual en las dos filas.

    Las props de metodo vienen por esquina (r_ko_odds / b_ko_odds); sumar las dos da la
    probabilidad de que la pelea termine asi, sin importar quien gane — que es exactamente
    el target del modelo de metodo. Normalizar las tres a 1 les saca el vig (proporcional:
    el power de `_desvig` es para mercados de dos vias).
    """
    vacio = pd.DataFrame(np.nan, index=results.index, columns=PROPS)
    archivo = RAW / "ufc_odds.csv"
    if not archivo.exists():
        return vacio

    o = pd.read_csv(archivo, low_memory=False)
    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    implicita = lambda x: np.where(x < 0, np.abs(x), 100.0) / (np.abs(x) + 100)  # noqa: E731
    cruda = np.column_stack([
        implicita(o[f"r_{v}_odds"].to_numpy(float)) + implicita(o[f"b_{v}_odds"].to_numpy(float))
        for v in ("ko", "sub", "dec")])
    with np.errstate(invalid="ignore"):
        p = cruda / cruda.sum(axis=1, keepdims=True)

    r, b = _norm(o["R_fighter"]), _norm(o["B_fighter"])
    tabla = dict(zip(_clave(o["date"].dt.date, r, b), p))
    a, b2 = _norm(results["fighter_a"]), _norm(results["fighter_b"])
    fila = np.array([tabla.get(k, (np.nan,) * 3)
                     for k in _clave(results["DATE"].dt.date, a, b2)], dtype=float)
    return pd.DataFrame(fila, index=results.index, columns=PROPS)


def _load():
    results = pd.read_csv(RAW / "ufc_fight_results.csv")
    events = pd.read_csv(RAW / "ufc_event_details.csv")
    stats = pd.read_csv(RAW / "ufc_fight_stats.csv")
    tott = pd.read_csv(RAW / "ufc_fighter_tott.csv")

    # los CSVs traen espacios colgantes en casi todas las columnas de texto
    for df in (results, events, stats, tott):
        for c in df.columns:
            if pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].str.strip()

    events["DATE"] = pd.to_datetime(events["DATE"], format="%B %d, %Y")
    results = results.merge(events[["EVENT", "DATE"]], on="EVENT", how="inner")

    # solo peleas con ganador claro (fuera NC y draws)
    results = results[results["OUTCOME"].isin(["W/L", "L/W"])].copy()
    bout = results["BOUT"].str.split(" vs. ", n=1, expand=True)
    results["fighter_a"], results["fighter_b"] = bout[0], bout[1]
    results = results[results["fighter_b"].notna()]
    results["target"] = (results["OUTCOME"] == "W/L").astype(int)
    results["minutes"] = results.apply(_fight_minutes, axis=1)
    canonical = results["METHOD"].map(settlement.canonical_method)
    results["finish"] = canonical.isin({settlement.KO, settlement.SUB}).astype(int)
    # DQ/CNC/other no se fuerzan a decision: quedan fuera del target de props hasta
    # contar con la regla exacta del sportsbook.
    results["metodo"] = results["METHOD"].map(settlement.method_market_class)
    # contexto de la pelea ("Light Heavyweight" antes que "Heavyweight": el orden del
    # scan importa). Catch/Open Weight quedan NaN.
    lbs = pd.Series(np.nan, index=results.index)
    for nombre, peso in _LBS:
        lbs = lbs.mask(lbs.isna() & results["WEIGHTCLASS"].str.contains(nombre, na=False),
                       float(peso))
    results["wc_lbs"] = lbs
    results["mujer"] = results["WEIGHTCLASS"].str.contains("Women", na=False).astype(float)
    results["cinco_r"] = results["TIME FORMAT"].str.startswith("5 Rnd").astype(float)
    results = results.sort_values(["DATE", "EVENT", "BOUT"]).reset_index(drop=True)
    results[MERCADO] = _mercado(results)
    results[PROPS] = _props(results)

    # stats por round -> agregado por (evento, pelea, peleador)
    sl = stats["SIG.STR."].map(_num)
    td = stats["TD"].map(_num)
    agg = pd.DataFrame({
        "EVENT": stats["EVENT"], "BOUT": stats["BOUT"], "FIGHTER": stats["FIGHTER"],
        "sig_l": [x[0] for x in sl], "sig_a": [x[1] for x in sl],
        "td_l": [x[0] for x in td], "td_a": [x[1] for x in td],
        "sub": pd.to_numeric(stats["SUB.ATT"], errors="coerce"),
        "ctrl": stats["CTRL"].map(_ctrl_seconds),
        "kd": pd.to_numeric(stats["KD"], errors="coerce"),
    }).groupby(["EVENT", "BOUT", "FIGHTER"]).sum(min_count=1)

    phys = tott.drop_duplicates("FIGHTER").set_index("FIGHTER")
    phys = pd.DataFrame({
        "height_in": phys["HEIGHT"].map(_inches),
        "reach_in": phys["REACH"].map(_inches),
        "dob": pd.to_datetime(phys["DOB"], format="%b %d, %Y", errors="coerce"),
    })
    # ufcstats tiene 8 nombres repetidos (dos "Bruno Silva" distintos, etc.) y las
    # peleas solo traen el nombre: sus historiales quedan mezclados sin arreglo posible.
    # El flag deja avisarlo en vez de servir una prediccion contaminada en silencio.
    phys["homonimo"] = phys.index.map(tott["FIGHTER"].value_counts()) > 1
    return results, agg, phys


def build():
    """-> (features_df, state_df). Una sola pasada cronologica."""
    results, agg, phys = _load()
    empty = pd.Series({c: np.nan for c in agg.columns})
    tabla, cubiertos = _avisos()
    previos = _previos()
    sin_previo = {c: np.nan for c in _PREV}
    states, rows = {}, []

    for date, dia in results.groupby("DATE", sort=True):
        pendientes = []
        for _, f in dia.iterrows():
            a, b = f["fighter_a"], f["fighter_b"]
            snaps = []
            for name in (a, b):
                st = states.setdefault(name, _new_state())
                # anti-leakage: el estado nunca puede incluir una pelea de hoy o del futuro
                assert st["last_date"] is None or st["last_date"] < date, (
                    f"leakage: {name} tiene estado con fecha {st['last_date']} "
                    f"al predecir la pelea del {date}")
                p = phys.loc[name] if name in phys.index else None
                # la circunstancia de ESTA pelea no es estado acumulado, se pega aparte
                snaps.append({**_snapshot(st, date, p),
                              **_aviso(tabla, cubiertos, f["EVENT"], name),
                              **previos.get(name, sin_previo)})

            # n_fights_min y n_nan son simetricos ante el intercambio, asi que valen igual
            # en las dos filas espejadas. No son features: miden cuanto sabemos de la
            # pelea, y de ahi sale el nivel de confianza que se muestra en la app.
            base = {"date": date, "event": f["EVENT"], "bout": f["BOUT"],
                    "fight_id": f"{f['EVENT']}|{f['BOUT']}", "metodo": f["metodo"],
                    "n_fights_min": min(snaps[0]["n_fights"], snaps[1]["n_fights"]),
                    "n_nan": sum(pd.isna(snaps[0][k]) or pd.isna(snaps[1][k])
                                 for k in FEATURES)}
            diffs = {k: snaps[0][k] - snaps[1][k] for k in _TODAS}
            ctx = {k: f[k] for k in (*CONTEXTO, *PROPS)}
            rows.append({**base, "fighter_a": a, "fighter_b": b, **diffs, **ctx,
                         MERCADO: f[MERCADO], "target": f["target"]})
            rows.append({**base, "fighter_a": b, "fighter_b": a,
                         **{k: -v for k, v in diffs.items()}, **ctx,
                         MERCADO: -f[MERCADO], "target": 1 - f["target"]})
            pendientes.append(f)

        for f in pendientes:  # updates recien al cerrar el dia
            _update(states, agg, empty, f, date)

    feats = pd.DataFrame(rows)
    return feats, _state_df(states, phys, previos)


def _update(states, agg, empty, f, date):
    a, b, ganador = f["fighter_a"], f["fighter_b"], f["target"]
    sa = agg.loc[(f["EVENT"], f["BOUT"], a)] if (f["EVENT"], f["BOUT"], a) in agg.index else empty
    sb = agg.loc[(f["EVENT"], f["BOUT"], b)] if (f["EVENT"], f["BOUT"], b) in agg.index else empty

    ea, eb = states[a]["elo"], states[b]["elo"]
    esperado_a = 1 / (1 + 10 ** ((eb - ea) / 400))
    states[a]["elo"] = ea + K_ELO * (ganador - esperado_a)
    states[b]["elo"] = eb + K_ELO * ((1 - ganador) - (1 - esperado_a))
    # calidad de rivales: siempre el Elo PRE-pelea del rival (ea/eb ya capturados)
    states[a]["opp_elo"] += eb
    states[b]["opp_elo"] += ea

    for st, mio, suyo, gano in ((states[a], sa, sb, ganador),
                                (states[b], sb, sa, 1 - ganador)):
        st["n"] += 1
        st["w"] += gano
        st["streak"] = max(st["streak"], 0) + 1 if gano else min(st["streak"], 0) - 1
        st["finishes"] += f["finish"] * gano
        st["finished_against"] += f["finish"] * (1 - gano)
        st["last_date"] = date
        for k in _MIAS:
            if pd.notna(mio[k]):
                st[k] += mio[k]
        # lo que hizo el rival: absorbido / intentos en contra (para defensas)
        for k, col in _SUYAS.items():
            if pd.notna(suyo[col]):
                st[k] += suyo[col]
        mins = f["minutes"]
        if pd.notna(mins):
            for col, exposure in (("sig_l", "sig_for_minutes"),
                                  ("td_l", "td_minutes"),
                                  ("sub", "sub_minutes"),
                                  ("ctrl", "ctrl_minutes"),
                                  ("kd", "kd_minutes")):
                if pd.notna(mio[col]):
                    st[exposure] += mins
            if pd.notna(suyo["sig_l"]):
                st["sig_against_minutes"] += mins
            if pd.notna(suyo["kd"]):
                st["kd_against_minutes"] += mins


def _state_df(states, phys, previos):
    """Estado final de cada peleador (para predecir matchups futuros sin skew).

    El record pre-UFC viaja aca aunque no sea estado: es fijo por peleador y `predict`
    arma el snapshot desde esta tabla, asi que sin esto la app serviria NaN en cuatro
    features que el modelo si usa.
    """
    filas = []
    sin_previo = {c: np.nan for c in _PREV}
    for name, st in states.items():
        p = phys.loc[name] if name in phys.index else None
        snap = {**_snapshot(st, st["last_date"], p), **previos.get(name, sin_previo)}
        snap.pop("age"), snap.pop("days_since_last")
        filas.append({"fighter": name, **snap,
                      "dob": p["dob"] if p is not None else pd.NaT,
                      "last_date": st["last_date"],
                      "homonimo": bool(p["homonimo"]) if p is not None else False})
    return pd.DataFrame(filas).sort_values("fighter")


def main():
    feats, estado = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", dir=OUT.parent,
                                     delete=False) as tmp:
        temporal = pathlib.Path(tmp.name)
        feats.to_csv(tmp, index=False)
    os.replace(temporal, OUT)
    assert feats["target"].notna().all(), "target con NaN"
    print(f"{OUT}: {len(feats)} filas ({len(feats) // 2} peleas), "
          f"{len(FEATURES)} features + {len(CANDIDATAS)} candidatas, "
          f"{len(estado)} peleadores, "
          f"odds en el {feats[MERCADO].notna().mean():.1%} de las filas")
    # cobertura de cada candidata: un veredicto "no concluyente" con 80% de NaN es falta
    # de datos, no falta de senial, y hay que poder distinguirlos antes de descartar
    if CANDIDATAS:
        print("cobertura de candidatas: " + ", ".join(
            f"{c} {feats[c].notna().mean():.0%}" for c in CANDIDATAS))
    print(f"circunstancia (wiki): reemplazo/peso con dato en el "
          f"{feats['reemplazo'].notna().mean():.0%} de las filas")


if __name__ == "__main__":
    main()
