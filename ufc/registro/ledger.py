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
from ufc.modelo import predict

LEDGER = rutas.DATOS / "ledger.csv"
# `p_con_odds` va al final a proposito: la migracion de abajo rellena las columnas que
# falten agregando vacias al final, asi que una columna nueva en el medio corre todo.
COLS = ["visto", "evento", "fecha_evento", "a", "b", "p_a", "p_mercado",
        "cuota_a", "cuota_b", "confianza", "apuesta", "ev", "mejor_a", "mejor_b",
        "p_con_odds"]
# Piso para seguir un lado, en PUNTOS DE PROBABILIDAD contra el precio que se toma.
# Antes el piso era sobre el EV, y eso pide una ventaja de umbral*p_mercado: 0.9 puntos
# en una cuota de 5.80 contra 3.9 en una de 1.30. El mismo piso era cuatro veces mas
# facil de pasar del lado del underdog. En puntos los dos lados piden lo mismo.
# ponytail: constante y no un slider; si alguna vez hay que calibrarlo, se calibra aca.
UMBRAL_VENTAJA = 0.05


def registrar(evento, fecha_evento, a, b, r, cuotas, hoy=None, mejor=None):
    """Congela la prediccion la primera vez que la pelea aparece con cuota."""
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    filas = []
    if LEDGER.exists():
        with LEDGER.open() as f:
            filas = [fila for fila in csv.reader(f) if fila]
    # el ledger viejo no tenia las columnas de mejor cuota: se rellenan vacias una vez,
    # si no las filas nuevas salen mas anchas que el header y el csv queda ilegible
    if filas and len(filas[0]) < len(COLS):
        with LEDGER.open("w", newline="") as f:
            csv.writer(f).writerows(
                [COLS] + [fila + [""] * (len(COLS) - len(fila)) for fila in filas[1:]])
    vistas = {(fila[1], fila[3], fila[4]) for fila in filas}
    if (evento, a, b) in vistas:
        return
    lado = r.get("apuesta")
    ev = r[f"ev_{lado}"] if lado else max(r["ev_a"], r["ev_b"])
    with LEDGER.open("a", newline="") as f:
        w = csv.writer(f)
        if not filas:
            w.writerow(COLS)
        w.writerow([(hoy or pd.Timestamp.today()).date().isoformat(), evento,
                    fecha_evento, a, b, round(r["p_a"], 4),
                    round(r["p_a_mercado"], 4), cuotas[0], cuotas[1],
                    r["confianza"], lado or "", round(ev, 4),
                    *(mejor or ("", "")),
                    round(r["p_a_con_odds"], 4) if "p_a_con_odds" in r else ""])


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


def evaluar(umbral=UMBRAL_VENTAJA):
    """-> (df, resumen). Cada prediccion congelada con resultado, cierre y CLV.

    `umbral` es el piso de ventaja (en puntos de probabilidad sobre el precio tomado)
    para seguir un lado: se recalcula entero en cada llamada, asi la pestania puede
    moverlo y ver como cambia el historial completo.
    """
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

    # El EV se mide contra la mejor cuota disponible, no contra la de Betano: apostar al
    # precio mas alto del mercado es la unica ventaja que no depende de que el modelo
    # acierte. Las filas viejas no tienen la mejor cuota guardada y caen en la de Betano.
    for c in ("mejor_a", "mejor_b"):
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    df["tope_a"] = df[["cuota_a", "mejor_a"]].max(axis=1)
    df["tope_b"] = df[["cuota_b", "mejor_b"]].max(axis=1)

    # La probabilidad que decide el lado es la del modelo alimentado con la cuota. El
    # modelo ciego esta comprimido hacia 0.5 contra el mercado, y como EV = cuota*p - 1
    # ~= p_modelo/p_mercado - 1, esa compresion sola alcanza para que el underdog gane
    # la comparacion en TODAS las peleas: no era una senal, era el sesgo del modelo.
    # Las filas viejas no la tienen guardada y caen en el modelo ciego.
    p = (pd.to_numeric(df["p_con_odds"], errors="coerce").fillna(df["p_a"])
         if "p_con_odds" in df else df["p_a"])
    ev_a = df["tope_a"] * p - 1
    ev_b = df["tope_b"] * (1 - p) - 1

    # La ventaja se mide contra el precio que se toma (el tope, no Betano): asi el
    # line-shopping cuenta como lo que es, mejor precio = mas ventaja sobre el mercado.
    ventaja = p - predict._desvig(1 / df["tope_a"], 1 / df["tope_b"])
    # el lado "apostado": la candidata si la hubo, si no el lado donde el modelo le gana
    # al precio, siempre que pase el piso y el EV a ese precio sea positivo. El segundo
    # filtro no es redundante: una ventaja chica no alcanza para cubrir el vig.
    df["p_dec"], df["ventaja"] = p, ventaja
    ev_lado = np.where(ventaja > 0, ev_a, ev_b)
    lado = np.where(df["apuesta"].fillna("") != "", df["apuesta"],
                    np.where((np.abs(ventaja) < umbral) | (ev_lado <= 0), "",
                             np.where(ventaja > 0, "a", "b")))
    df["lado"] = lado
    sin_lado = lado == ""
    es_a = lado == "a"
    df["cuota_lado"] = np.where(sin_lado, np.nan,
                                np.where(es_a, df["tope_a"], df["tope_b"]))
    # cuanto suma el line-shopping sobre Betano en ese mismo lado
    df["extra"] = df["cuota_lado"] / np.where(es_a, df["cuota_a"], df["cuota_b"]) - 1
    df["clv"] = df["cuota_lado"] / np.where(es_a, df["cierre_a"], df["cierre_b"]) - 1
    con_res = df["gano"].notna()
    df["retorno"] = np.where(~con_res | sin_lado, np.nan,
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
        "seguidos": int((~sin_lado).sum()),
        "extra_medio": df["extra"].mean(),
    }
    return df, resumen
