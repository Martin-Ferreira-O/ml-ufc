"""Registro de cada prediccion hecha con cuota, y su evaluacion posterior.

La primera vez que una pelea aparece con cuota, la prediccion queda congelada: eso
simula "apostar apenas abre el mercado", que es la unica ventana donde un modelo que
pierde contra la linea de cierre todavia puede tener valor. `betano_hist.csv` guarda
el resto del camino de la cuota, asi que el CLV (closing line value) sale de comparar
la cuota congelada contra el ultimo tick antes del evento. CLV positivo sostenido es
el unico predictor confiable de rentabilidad: converge en decenas de apuestas, no en
las miles que necesita el ROI.
"""

import csv

import numpy as np
import pandas as pd

from ufc import nombres, rutas
from ufc.datos import betano

LEDGER = rutas.DATOS / "ledger.csv"
COLS = ["visto", "evento", "fecha_evento", "a", "b", "p_a", "p_mercado",
        "cuota_a", "cuota_b", "confianza", "apuesta", "ev"]


def registrar(evento, fecha_evento, a, b, r, cuotas, hoy=None):
    """Congela la prediccion la primera vez que la pelea aparece con cuota."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    vistas = set()
    if LEDGER.exists():
        with LEDGER.open() as f:
            vistas = {(fila[1], fila[3], fila[4]) for fila in csv.reader(f)}
    if (evento, a, b) in vistas:
        return
    lado = r.get("apuesta")
    ev = r[f"ev_{lado}"] if lado else max(r["ev_a"], r["ev_b"])
    with LEDGER.open("a", newline="") as f:
        w = csv.writer(f)
        if not vistas:
            w.writerow(COLS)
        w.writerow([(hoy or pd.Timestamp.today()).date().isoformat(), evento,
                    fecha_evento, a, b, round(r["p_a"], 4),
                    round(r["p_a_mercado"], 4), cuotas[0], cuotas[1],
                    r["confianza"], lado or "", round(ev, 4)])


def _resultados():
    """Peleas ya ocurridas: (par normalizado, fecha, clave del ganador)."""
    r = pd.read_csv(rutas.RAW / "ufc_fight_results.csv")
    e = pd.read_csv(rutas.RAW / "ufc_event_details.csv")
    for d in (r, e):
        for c in d.columns:
            if pd.api.types.is_string_dtype(d[c]):
                d[c] = d[c].str.strip()
    e["DATE"] = pd.to_datetime(e["DATE"], format="%B %d, %Y")
    r = r.merge(e[["EVENT", "DATE"]], on="EVENT")
    r = r[r["OUTCOME"].isin(["W/L", "L/W"])].copy()
    bout = r["BOUT"].str.split(" vs. ", n=1, expand=True)
    r["fa"], r["fb"] = bout[0], bout[1]
    r = r[r["fb"].notna()]
    ganador = np.where(r["OUTCOME"] == "W/L", r["fa"], r["fb"])
    return pd.DataFrame({
        "par": [tuple(sorted((nombres.normalizar(x), nombres.normalizar(y))))
                for x, y in zip(r["fa"], r["fb"])],
        "fecha": r["DATE"].to_numpy(),
        "ganador": [nombres.normalizar(g) for g in ganador]})


def _cierres(fecha_por_par):
    """{par: (cuota_x, cuota_y)} con el ultimo tick de betano_hist antes del evento."""
    if not betano.HIST.exists():
        return {}
    h = pd.read_csv(betano.HIST, parse_dates=["ts"])
    tabla = {}
    for (a, b), g in h.groupby(["a", "b"]):
        limite = fecha_por_par.get((a, b))
        if limite is not None:
            g = g[g["ts"] < limite + pd.Timedelta(days=1)]
        if len(g):
            fila = g.sort_values("ts").iloc[-1]
            tabla[(a, b)] = (fila["cuota_a"], fila["cuota_b"])
    return tabla


def evaluar():
    """-> (df, resumen). Cada prediccion congelada con resultado, cierre y CLV."""
    if not LEDGER.exists():
        return pd.DataFrame(), {}
    df = pd.read_csv(LEDGER, parse_dates=["fecha_evento"])
    if df.empty:
        return df, {}
    df["par"] = [tuple(sorted((nombres.normalizar(a), nombres.normalizar(b))))
                 for a, b in zip(df["a"], df["b"])]

    res = _resultados()
    por_par = {p: g for p, g in res.groupby("par")}
    ganador, cierre_a, cierre_b = [], [], []
    cierres = _cierres(dict(zip(df["par"], df["fecha_evento"])))
    for _, fila in df.iterrows():
        g = por_par.get(fila["par"])
        if g is not None:
            cerca = (g["fecha"] - fila["fecha_evento"].to_datetime64())
            g = g[np.abs(cerca) <= np.timedelta64(2, "D")]
        ganador.append(g["ganador"].iloc[0] if g is not None and len(g) else None)
        par = betano.buscar(cierres, fila["a"], fila["b"])
        cierre_a.append(par[0] if par else np.nan)
        cierre_b.append(par[1] if par else np.nan)

    df["gano"] = [None if g is None
                  else "a" if g == nombres.normalizar(a) else "b"
                  for g, a in zip(ganador, df["a"])]
    df["cierre_a"], df["cierre_b"] = cierre_a, cierre_b

    # el lado "apostado": la candidata si la hubo, si no el de mayor EV (hipotetico)
    lado = np.where(df["apuesta"].fillna("") != "", df["apuesta"],
                    np.where(df["cuota_a"] * df["p_a"] > df["cuota_b"] * (1 - df["p_a"]),
                             "a", "b"))
    df["lado"] = lado
    es_a = lado == "a"
    df["cuota_lado"] = np.where(es_a, df["cuota_a"], df["cuota_b"])
    df["clv"] = df["cuota_lado"] / np.where(es_a, df["cierre_a"], df["cierre_b"]) - 1
    con_res = df["gano"].notna()
    df["retorno"] = np.where(~con_res, np.nan,
                             np.where(df["gano"] == lado, df["cuota_lado"] - 1, -1.0))

    hecho = df[con_res]
    candidatas = hecho[hecho["apuesta"].fillna("") != ""]
    resumen = {
        "predicciones": len(df),
        "con_resultado": len(hecho),
        "acierto_modelo": ((hecho["p_a"] > 0.5) == (hecho["gano"] == "a")).mean()
                          if len(hecho) else np.nan,
        "acierto_mercado": ((hecho["p_mercado"] > 0.5) == (hecho["gano"] == "a")).mean()
                           if len(hecho) else np.nan,
        "candidatas": int((df["apuesta"].fillna("") != "").sum()),
        "roi_candidatas": candidatas["retorno"].mean() if len(candidatas) else np.nan,
        "clv_medio": df["clv"].mean(),
    }
    return df, resumen
