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
# `inicio_utc` sigue la misma regla y por eso va ultima.
COLS = ["visto", "evento", "fecha_evento", "a", "b", "p_a", "p_mercado",
        "cuota_a", "cuota_b", "confianza", "apuesta", "ev", "mejor_a", "mejor_b",
        "p_con_odds", "inicio_utc"]
# Piso para seguir un lado, en PUNTOS DE PROBABILIDAD contra el precio que se toma.
# Antes el piso era sobre el EV, y eso pide una ventaja de umbral*p_mercado: 0.9 puntos
# en una cuota de 5.80 contra 3.9 en una de 1.30. El mismo piso era cuatro veces mas
# facil de pasar del lado del underdog. En puntos los dos lados piden lo mismo.
# ponytail: constante y no un slider; si alguna vez hay que calibrarlo, se calibra aca.
UMBRAL_VENTAJA = 0.05


def registrar(evento, fecha_evento, a, b, r, cuotas, hoy=None, mejor=None,
              inicio_utc=None):
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
                    round(r["p_a_con_odds"], 4) if "p_a_con_odds" in r else "",
                    inicio_utc or ""])


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


def _cierres(limite_por_par):
    """{par: (cuota_x, cuota_y)} con el ultimo tick ANTERIOR al inicio de la pelea.

    `limite_por_par` trae el corte ya resuelto por pelea. Cuando el ledger guardo el
    `inicio_utc` del evento, el corte es esa hora exacta y el cierre es cierre de verdad.
    Sin esa columna (filas viejas) se cae a `fecha_evento + 1 dia`, que es lo que hacia
    antes y puede colar cuotas EN VIVO o posteriores al combate: una cuota live de un
    peleador que ya esta ganando no es la linea de cierre, y usarla infla o destruye el
    CLV segun como venia la pelea. Es la unica metrica que decide si este proyecto puede
    apostar, asi que el corte tiene que ser estricto.
    """
    if not betano.HIST.exists():
        return {}
    h = pd.read_csv(betano.HIST, parse_dates=["ts"])
    tabla = {}
    for (a, b), g in h.groupby(["a", "b"]):
        limite = limite_por_par.get((a, b))
        if limite is not None:
            g = g[g["ts"] < limite]
        if len(g):
            fila = g.sort_values("ts").iloc[-1]
            tabla[(a, b)] = (fila["cuota_a"], fila["cuota_b"])
    return tabla


def _limite(fila):
    """El instante a partir del cual una cuota ya no es "antes de la pelea".

    Con `inicio_utc` es la hora real de inicio; sin el, el fallback historico de
    `fecha_evento + 1 dia`. Se devuelve naive (sin tz) porque los `ts` de `betano_hist`
    son hora local sin tz y comparar aware contra naive levanta.
    """
    crudo = fila.get("inicio_utc")
    if crudo not in (None, "") and pd.notna(crudo):
        ts = pd.to_datetime(crudo, errors="coerce", utc=True)
        if pd.notna(ts):
            return ts.tz_localize(None)
    return fila["fecha_evento"] + pd.Timedelta(days=1)


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
    cierres = _cierres({p: _limite(f) for p, (_, f) in zip(df["par"], df.iterrows())})
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

    # CLV en unidades economicas. `clv` compara dos cuotas y contesta "¿consegui mejor
    # precio?"; esto contesta "¿cuanto vale esa apuesta segun lo que el mercado terminó
    # creyendo?", que es lo unico comparable contra cero y lo que evalua el gate:
    #     ev_al_cierre = p_justa_del_cierre(lado) * cuota_tomada - 1
    # Positivo = el mercado se movio hacia el lado apostado despues de tomarlo. Es el
    # mismo numero que un ROI esperado, pero medido contra el precio de cierre en vez de
    # contra el resultado, y por eso converge en decenas de apuestas y no en miles.
    p_cierre_a = predict._desvig(1 / df["cierre_a"], 1 / df["cierre_b"])
    p_cierre = np.where(es_a, p_cierre_a, 1 - p_cierre_a)
    df["p_cierre"] = np.where(sin_lado, np.nan, p_cierre)
    df["ev_al_cierre"] = np.where(sin_lado, np.nan, p_cierre * df["cuota_lado"] - 1)
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
    resumen.update(_clv(df))
    return df, resumen


def _clv(df):
    """El bloque de CLV con incertidumbre: media, IC95% clusterizado y cuanto falta.

    Sin el IC, un CLV medio es un numero que siempre se puede leer como buena noticia.
    Con n=16 apuestas, cualquier media entre -10% y +10% es ruido, y el objeto de esta
    funcion es que eso se vea en la pantalla en vez de tener que saberlo.

    Se clusteriza por evento porque las peleas de una misma cartelera comparten el
    movimiento del mercado de esa cartelera: contarlas como observaciones independientes
    angosta el intervalo y adelanta la conclusion.
    """
    from ufc.modelo import apuesta, train

    d = df[(df["lado"] != "") & df["ev_al_cierre"].notna()]
    n = len(d)
    if not n:
        return {"clv_n": 0, "clv_ev": np.nan, "clv_lo": np.nan, "clv_hi": np.nan,
                "clv_batio": np.nan, "clv_n_para_concluir": np.nan}
    v = d["ev_al_cierre"].to_numpy(float)
    media, lo, hi = train.bootstrap(v, clusters=d["evento"].to_numpy())
    sigma = float(np.std(v, ddof=1)) if n > 1 else np.nan
    return {
        "clv_n": int(n), "clv_ev": float(media), "clv_lo": float(lo), "clv_hi": float(hi),
        "clv_sigma": sigma,
        "clv_batio": float((v > 0).mean()),
        # Con la dispersion observada, cuantas apuestas pediria una prueba de potencia
        # para separar ESTE CLV de cero. Es orientativo y NO reemplaza el `n_minimo`
        # preregistrado del gate: con 4 apuestas la dispersion misma es una estimacion
        # ruidosa, asi que este numero puede salir absurdamente chico.
        "clv_n_para_concluir": (float(apuesta.n_para_detectar(media, sigma))
                                if n > 1 and np.isfinite(sigma) and media else np.inf),
    }
