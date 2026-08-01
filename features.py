"""Features point-in-time por pelea, sin leakage temporal.

Recorre las peleas en orden cronologico manteniendo el estado acumulado de cada
peleador. Para cada pelea, las features salen del estado *previo*; recien despues
se actualiza con el resultado. Las peleas del mismo dia se calculan todas antes de
aplicar sus updates (torneos de los UFC 1-8: un peleador pelea dos veces el mismo dia).

Cada pelea produce dos filas: diffs A-B y la espejada B-A con el target invertido.
"""

import pathlib
import re

import numpy as np
import pandas as pd

import predict

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/features.csv")
K_ELO = 32

# nombre -> como se calcula desde el estado previo del peleador
FEATURES = [
    "elo", "n_fights", "win_rate", "streak", "slpm", "sapm", "str_acc",
    "td_per15", "td_acc", "sub_per15", "ctrl_per_min", "finish_rate",
    "days_since_last", "age", "height_in", "reach_in",
    "kd_per15", "kd_against_per15", "finished_against_rate",
    "td_def", "str_def", "avg_opp_elo",
]
# Contexto simetrico de la pelea (vale igual en las dos filas espejadas). Para el
# target de ganador esta medido no concluyente (rolling-origin 2026-07-31), pero es
# el insumo principal del modelo de metodo: las tasas de finish cambian por division.
CONTEXTO = ["wc_lbs", "mujer", "cinco_r"]
_LBS = [("Strawweight", 115), ("Flyweight", 125), ("Bantamweight", 135),
        ("Featherweight", 145), ("Lightweight", 155), ("Welterweight", 170),
        ("Middleweight", 185), ("Light Heavyweight", 205), ("Heavyweight", 265)]
# El mercado no sale del historial del peleador: es una fuente aparte, y por eso vive en
# su propio conjunto. Se entrena un modelo con cada uno para poder mostrar los dos.
MERCADO = "mkt_logit"
FEATURES_ODDS = FEATURES + [MERCADO]


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
    """Duracion total: rounds completos de 5' + el tiempo del ultimo."""
    m = re.match(r"\s*(\d+):(\d+)", str(row["TIME"]))
    last = (float(m.group(1)) + float(m.group(2)) / 60) if m else 0.0
    # ponytail: asume rounds de 5'; los pocos formatos viejos (1 Rnd de 12'/15')
    # subestiman la duracion, lo que solo diluye las tasas por minuto de esas peleas.
    return (float(row["ROUND"]) - 1) * 5 + last


def _new_state():
    return dict(n=0, w=0, streak=0, sig_l=0.0, sig_a=0.0, sig_abs=0.0, td_l=0.0,
                td_a=0.0, sub=0.0, ctrl=0.0, minutes=0.0, finishes=0,
                last_date=None, elo=1500.0, kd=0.0, kd_abs=0.0, finished_against=0,
                td_abs_l=0.0, td_abs_a=0.0, sig_abs_a=0.0, opp_elo=0.0)


def _snapshot(st, date, dob, height, reach):
    """Features de un peleador a la fecha `date`, desde su estado previo."""
    n, mins = st["n"], st["minutes"]
    div = lambda a, b: a / b if b else np.nan  # noqa: E731
    return {
        "elo": st["elo"],
        "n_fights": float(n),
        "win_rate": div(st["w"], n),
        "streak": float(st["streak"]) if n else np.nan,
        "slpm": div(st["sig_l"], mins),
        "sapm": div(st["sig_abs"], mins),
        "str_acc": div(st["sig_l"], st["sig_a"]),
        "td_per15": div(st["td_l"], mins) * 15 if mins else np.nan,
        "td_acc": div(st["td_l"], st["td_a"]),
        "sub_per15": div(st["sub"], mins) * 15 if mins else np.nan,
        "ctrl_per_min": div(st["ctrl"] / 60, mins),
        "finish_rate": div(st["finishes"], n),
        "kd_per15": div(st["kd"], mins) * 15 if mins else np.nan,
        "kd_against_per15": div(st["kd_abs"], mins) * 15 if mins else np.nan,
        "finished_against_rate": div(st["finished_against"], n),
        "td_def": 1 - div(st["td_abs_l"], st["td_abs_a"]),
        "str_def": 1 - div(st["sig_abs"], st["sig_abs_a"]),
        "avg_opp_elo": div(st["opp_elo"], n),
        "days_since_last": (date - st["last_date"]).days if st["last_date"] else np.nan,
        "age": (date - dob).days / 365.25 if pd.notna(dob) else np.nan,
        "height_in": height,
        "reach_in": reach,
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
    results["finish"] = results["METHOD"].str.contains(
        "KO/TKO|Submission", na=False, regex=True).astype(int)
    # como termino: el target del modelo de metodo (simetrico ante el espejado)
    results["metodo"] = np.where(
        results["METHOD"].str.contains("KO/TKO", na=False), "ko",
        np.where(results["METHOD"].str.contains("Submission", na=False), "sub", "dec"))
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
    empty = pd.Series({"sig_l": np.nan, "sig_a": np.nan, "td_l": np.nan,
                       "td_a": np.nan, "sub": np.nan, "ctrl": np.nan, "kd": np.nan})
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
                snaps.append(_snapshot(
                    st, date,
                    p["dob"] if p is not None else pd.NaT,
                    p["height_in"] if p is not None else np.nan,
                    p["reach_in"] if p is not None else np.nan))

            # n_fights_min y n_nan son simetricos ante el intercambio, asi que valen igual
            # en las dos filas espejadas. No son features: miden cuanto sabemos de la
            # pelea, y de ahi sale el nivel de confianza que se muestra en la app.
            base = {"date": date, "event": f["EVENT"], "bout": f["BOUT"],
                    "fight_id": f"{f['EVENT']}|{f['BOUT']}", "metodo": f["metodo"],
                    "n_fights_min": min(snaps[0]["n_fights"], snaps[1]["n_fights"]),
                    "n_nan": sum(pd.isna(snaps[0][k]) or pd.isna(snaps[1][k])
                                 for k in FEATURES)}
            diffs = {k: snaps[0][k] - snaps[1][k] for k in FEATURES}
            ctx = {k: f[k] for k in CONTEXTO}
            rows.append({**base, "fighter_a": a, "fighter_b": b, **diffs, **ctx,
                         MERCADO: f[MERCADO], "target": f["target"]})
            rows.append({**base, "fighter_a": b, "fighter_b": a,
                         **{k: -v for k, v in diffs.items()}, **ctx,
                         MERCADO: -f[MERCADO], "target": 1 - f["target"]})
            pendientes.append(f)

        for f in pendientes:  # updates recien al cerrar el dia
            _update(states, agg, empty, f, date)

    feats = pd.DataFrame(rows)
    return feats, _state_df(states, phys)


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

    for st, mio, suyo, gano in ((states[a], sa, sb, ganador), (states[b], sb, sa, 1 - ganador)):
        st["n"] += 1
        st["w"] += gano
        st["streak"] = max(st["streak"], 0) + 1 if gano else min(st["streak"], 0) - 1
        st["minutes"] += f["minutes"]
        st["finishes"] += f["finish"] * gano
        st["finished_against"] += f["finish"] * (1 - gano)
        st["last_date"] = date
        for k, col in (("sig_l", "sig_l"), ("sig_a", "sig_a"), ("td_l", "td_l"),
                       ("td_a", "td_a"), ("sub", "sub"), ("ctrl", "ctrl"),
                       ("kd", "kd")):
            v = mio[col]
            if pd.notna(v):
                st[k] += v
        # lo que hizo el rival: absorbido / intentos en contra (para defensas)
        for k, col in (("sig_abs", "sig_l"), ("sig_abs_a", "sig_a"), ("kd_abs", "kd"),
                       ("td_abs_l", "td_l"), ("td_abs_a", "td_a")):
            v = suyo[col]
            if pd.notna(v):
                st[k] += v


def _state_df(states, phys):
    """Estado final de cada peleador (para predecir matchups futuros sin skew)."""
    filas = []
    for name, st in states.items():
        p = phys.loc[name] if name in phys.index else None
        snap = _snapshot(st, st["last_date"], pd.NaT,
                         p["height_in"] if p is not None else np.nan,
                         p["reach_in"] if p is not None else np.nan)
        snap.pop("age"), snap.pop("days_since_last")
        filas.append({"fighter": name, **snap,
                      "dob": p["dob"] if p is not None else pd.NaT,
                      "last_date": st["last_date"],
                      "homonimo": bool(p["homonimo"]) if p is not None else False})
    return pd.DataFrame(filas).sort_values("fighter")


def main():
    feats, estado = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT, index=False)
    assert feats["target"].notna().all(), "target con NaN"
    print(f"{OUT}: {len(feats)} filas ({len(feats) // 2} peleas), "
          f"{len(FEATURES)} features, {len(estado)} peleadores, "
          f"odds en el {feats[MERCADO].notna().mean():.1%} de las filas")


if __name__ == "__main__":
    main()
