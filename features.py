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

RAW = pathlib.Path("data/raw")
OUT = pathlib.Path("data/features.csv")
K_ELO = 32

# nombre -> como se calcula desde el estado previo del peleador
FEATURES = [
    "elo", "n_fights", "win_rate", "streak", "slpm", "sapm", "str_acc",
    "td_per15", "td_acc", "sub_per15", "ctrl_per_min", "finish_rate",
    "days_since_last", "age", "height_in", "reach_in",
]


def _num(x):
    """'19 of 39' -> (19, 39); '--' o vacio -> (nan, nan)."""
    m = re.match(r"\s*(\d+)\s+of\s+(\d+)", str(x))
    return (float(m.group(1)), float(m.group(2))) if m else (np.nan, np.nan)


def _ctrl_seconds(x):
    m = re.match(r"\s*(\d+):(\d+)", str(x))
    return float(m.group(1)) * 60 + float(m.group(2)) if m else 0.0


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
                last_date=None, elo=1500.0)


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
        "days_since_last": (date - st["last_date"]).days if st["last_date"] else np.nan,
        "age": (date - dob).days / 365.25 if pd.notna(dob) else np.nan,
        "height_in": height,
        "reach_in": reach,
    }


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
    results = results.sort_values(["DATE", "EVENT", "BOUT"]).reset_index(drop=True)

    # stats por round -> agregado por (evento, pelea, peleador)
    sl = stats["SIG.STR."].map(_num)
    td = stats["TD"].map(_num)
    agg = pd.DataFrame({
        "EVENT": stats["EVENT"], "BOUT": stats["BOUT"], "FIGHTER": stats["FIGHTER"],
        "sig_l": [x[0] for x in sl], "sig_a": [x[1] for x in sl],
        "td_l": [x[0] for x in td], "td_a": [x[1] for x in td],
        "sub": pd.to_numeric(stats["SUB.ATT"], errors="coerce"),
        "ctrl": stats["CTRL"].map(_ctrl_seconds),
    }).groupby(["EVENT", "BOUT", "FIGHTER"]).sum(min_count=1)

    phys = tott.drop_duplicates("FIGHTER").set_index("FIGHTER")
    phys = pd.DataFrame({
        "height_in": phys["HEIGHT"].map(_inches),
        "reach_in": phys["REACH"].map(_inches),
        "dob": pd.to_datetime(phys["DOB"], format="%b %d, %Y", errors="coerce"),
    })
    return results, agg, phys


def build():
    """-> (features_df, state_df). Una sola pasada cronologica."""
    results, agg, phys = _load()
    empty = pd.Series({"sig_l": np.nan, "sig_a": np.nan, "td_l": np.nan,
                       "td_a": np.nan, "sub": np.nan, "ctrl": np.nan})
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

            base = {"date": date, "event": f["EVENT"], "bout": f["BOUT"],
                    "fight_id": f"{f['EVENT']}|{f['BOUT']}"}
            diffs = {k: snaps[0][k] - snaps[1][k] for k in FEATURES}
            rows.append({**base, "fighter_a": a, "fighter_b": b,
                         **diffs, "target": f["target"]})
            rows.append({**base, "fighter_a": b, "fighter_b": a,
                         **{k: -v for k, v in diffs.items()},
                         "target": 1 - f["target"]})
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

    for st, mio, suyo, gano in ((states[a], sa, sb, ganador), (states[b], sb, sa, 1 - ganador)):
        st["n"] += 1
        st["w"] += gano
        st["streak"] = max(st["streak"], 0) + 1 if gano else min(st["streak"], 0) - 1
        st["minutes"] += f["minutes"]
        st["finishes"] += f["finish"] * gano
        st["last_date"] = date
        for k, col in (("sig_l", "sig_l"), ("sig_a", "sig_a"), ("td_l", "td_l"),
                       ("td_a", "td_a"), ("sub", "sub"), ("ctrl", "ctrl")):
            v = mio[col]
            if pd.notna(v):
                st[k] += v
        if pd.notna(suyo["sig_l"]):
            st["sig_abs"] += suyo["sig_l"]


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
                      "last_date": st["last_date"]})
    return pd.DataFrame(filas).sort_values("fighter")


def main():
    feats, estado = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT, index=False)
    assert feats["target"].notna().all(), "target con NaN"
    print(f"{OUT}: {len(feats)} filas ({len(feats) // 2} peleas), "
          f"{len(FEATURES)} features, {len(estado)} peleadores")


if __name__ == "__main__":
    main()
