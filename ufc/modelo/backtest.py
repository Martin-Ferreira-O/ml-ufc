"""Backtest de la regla de apuesta: grilla de umbrales de ventaja, curvas y segmentos.

Generaliza `train.apostabilidad`, que medía flat-bet por tramo de confianza, a una grilla
de ventaja (`p_modelo - p_mercado`, en puntos de probabilidad) con IC95% clusterizado por
evento. La pregunta que responde es una sola: **¿existe algún umbral con ventaja real?**
La respuesta puede ser que no, y ese resultado también sirve.

Tres cosas que hay que tener presentes al leer cualquier número de acá:

1. **El mercado se desviguea.** El vig mediano de `ufc_odds.csv` es 3.7%: las dos
   implícitas crudas suman ~1.037. Usar `1/cuota` inflaría cada ventaja ~3.7 puntos y
   fabricaría EV positivo que es el margen de la casa, no señal. Se usa
   `predict._desvig` (power), igual que `features._mercado`.
2. **El CLV histórico NO es computable.** `ufc_odds.csv` trae UNA sola foto de cuota por
   pelea (7177 filas, un `R_odds`/`B_odds`), no apertura y cierre. El CLV solo existe
   hacia adelante, en `ufc/registro/ledger.py`, contra los ticks de `betano_hist.csv`.
   Este módulo no lo reporta porque no puede.
3. **El backtest juega un juego distinto al de la app.** Esas cuotas son de cierre o
   consenso; la app congela en apertura. `caveat_dominio()` mide el desfase de vig contra
   los ticks reales de Betano para dimensionarlo, pero no lo corrige: no hay con qué.

El calibrador de cada fold se ajusta con las predicciones OOF de los folds ANTERIORES.
Son fuera de muestra y solo miran el pasado, así que no hay leakage y no cuesta un
entrenamiento extra por fold.
"""

import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ufc import rutas
from ufc.modelo import apuesta, calibra, features, predict, train

OOF = rutas.DATOS / "oof.csv"
SALIDA = rutas.DATOS / "backtest.json"
# Ventaja en puntos de probabilidad, no en EV: el piso de EV pide una ventaja de
# umbral*p_mercado, o sea 0.9 puntos en una cuota de 5.80 contra 3.9 en una de 1.30.
GRID = [round(x, 2) for x in np.arange(0, 0.21, 0.01)]
# Abajo de esto se reporta el n y no se concluye: un ROI +30% con 20 apuestas es una racha.
MIN_N = 50
# Atributos de la pelea que viajan hasta el frame de apuestas para poder segmentar.
ATRIBUTOS = ["wc_lbs", "mujer", "cinco_r", "reemplazo", "peso_no_dado",
             "n_fights_min", "age"]
DIVISION = {float(lbs): nombre for nombre, lbs in features._LBS}


# --------------------------------------------------------------------- OOF

def _extras(u):
    """Columnas de `ufc_odds.csv` que solo sirven para segmentar: edad, ranking, título.

    Mismo join simétrico que `train._cuotas_con_vig` (fecha + par ordenado), pero para
    columnas que el modelo no usa. Se segmenta CON ellas los resultados, no se entrena:
    el ranking es endógeno y como feature traería el problema que documenta
    posibles_mejoras.md, como etiqueta post-hoc no decide nada.
    """
    vacio = {c: np.full(len(u), np.nan) for c in
             ("edad_a", "edad_b", "rank_a", "rank_b", "title_bout")}
    archivo = features.RAW / "ufc_odds.csv"
    if not archivo.exists():
        return pd.DataFrame(vacio, index=u.index)

    o = pd.read_csv(archivo, low_memory=False)
    o["date"] = pd.to_datetime(o["date"], errors="coerce")
    r, b = features._norm(o["R_fighter"]), features._norm(o["B_fighter"])
    menor, mayor = np.minimum(r, b), np.maximum(r, b)
    es_r = (r == menor).to_numpy()
    pares = {"edad": ("R_age", "B_age"),
             "rank": ("R_match_weightclass_rank", "B_match_weightclass_rank")}
    tabla = {}
    for k, (cr, cb) in pares.items():
        vr, vb = o[cr].to_numpy(float), o[cb].to_numpy(float)
        tabla[k] = dict(zip(zip(o["date"].dt.date, menor, mayor),
                            zip(np.where(es_r, vr, vb), np.where(es_r, vb, vr))))
    titulo = dict(zip(zip(o["date"].dt.date, menor, mayor),
                      o["title_bout"].astype(float)))

    a, b2 = features._norm(u["fighter_a"]), features._norm(u["fighter_b"])
    claves = list(zip(u["date"].dt.date, np.minimum(a, b2), np.maximum(a, b2)))
    izq = (a.to_numpy() <= b2.to_numpy())
    out = {}
    for k, t in tabla.items():
        v = np.array([t.get(c, (np.nan, np.nan)) for c in claves])
        out[f"{k}_a"] = np.where(izq, v[:, 0], v[:, 1])
        out[f"{k}_b"] = np.where(izq, v[:, 1], v[:, 0])
    out["title_bout"] = np.array([titulo.get(c, np.nan) for c in claves])
    return pd.DataFrame(out, index=u.index)


def calcular_oof(df=None, corte=None, n_folds=20):
    """-> DataFrame, una fila por pelea: prob OOF (cruda y calibrada), cuota real y resultado.

    El calibrador de cada fold se ajusta con el OOF acumulado de los folds anteriores.
    El primer fold no tiene historia y va crudo: queda marcado en `fold` para poder
    excluirlo al comparar calibradores.
    """
    if df is None:
        df = pd.read_csv(train.FEATS, parse_dates=["date"])
    corte = df["date"].max() if corte is None else corte

    partes = []
    for i, (tr, te) in enumerate(train.rolling_origin(df, corte, n_folds)):
        p = train.probas(train.entrenar(tr, features.FEATURES),
                         te, features.FEATURES)["blend"]
        u = te.iloc[0::2].reset_index(drop=True)
        fila = pd.DataFrame({"fold": i, "date": u["date"], "event": u["event"],
                             "fighter_a": u["fighter_a"], "fighter_b": u["fighter_b"],
                             "p": p, "y": train.objetivo(te)})
        # el mercado ya viene desvigueado y orientado a fighter_a desde features._mercado
        fila["p_mkt"] = 1 / (1 + np.exp(-u[features.MERCADO].to_numpy(float)))
        fila["q_a"], fila["q_b"] = train._cuotas_con_vig(u)
        for c in ATRIBUTOS:
            fila[c] = u[c].to_numpy()
        partes.append(fila)
        print(f"  fold {i + 1}/{n_folds}  {len(u):5d} peleas hasta {u['date'].max().date()}",
              flush=True)

    oof = pd.concat(partes, ignore_index=True)
    for nombre, p in calibra.prequencial(oof["p"], oof["y"], oof["fold"]).items():
        oof[f"p_{nombre}"] = p
    oof = oof.sort_values("date").reset_index(drop=True)
    oof = pd.concat([oof, _extras(oof)], axis=1)
    # revancha: el mismo par ya peleó antes. Va después del sort por fecha — con las
    # filas en orden de fold, la "primera vez" podría caer después de la revancha.
    par = pd.Series(["|".join(sorted(x)) for x in
                     zip(oof["fighter_a"], oof["fighter_b"])], index=oof.index)
    oof["rematch"] = par.groupby(par).cumcount().gt(0).astype(float)
    return oof


def cargar_oof(recalcular=False):
    if OOF.exists() and not recalcular:
        return pd.read_csv(OOF, parse_dates=["date"])
    oof = calcular_oof()
    oof.to_csv(OOF, index=False)
    return oof


# ------------------------------------------------------------------ apuestas

def ganador(cal):
    """-> nombre de columna del calibrador que entra: el de menor log loss entre los que
    pasaron el IC95%. Si ninguno pasa, gana `identidad`: menos codigo y menos supuestos."""
    pasan = [f for f in cal if f["queda"]]
    return "p_" + (min(pasan, key=lambda f: f["log_loss"])["calibrador"]
                   if pasan else "identidad")


def apuestas(oof, col="p_platt"):
    """-> un lado por fila. Los dos lados de cada pelea son dos apuestas candidatas.

    `ventaja` (edge) y `ev` salen de la MISMA cuota con la que se paga: medir la ventaja
    contra un precio y cobrar con otro es como se fabrica un backtest optimista.
    """
    p = oof[col].to_numpy(float)
    lados = []
    for lado, q, pl, pm, gana in (("a", oof["q_a"], p, oof["p_mkt"], oof["y"]),
                                  ("b", oof["q_b"], 1 - p, 1 - oof["p_mkt"], 1 - oof["y"])):
        d = oof.drop(columns=["q_a", "q_b", "p_mkt", "y"]).copy()
        d["lado"], d["q"], d["p"] = lado, q.to_numpy(float), pl
        d["p_mkt"], d["gana"] = np.asarray(pm, float), np.asarray(gana, float)
        lados.append(d)
    ap = pd.concat(lados, ignore_index=True)
    ap = ap[np.isfinite(ap["q"]) & np.isfinite(ap["p_mkt"])].copy()
    ap["ventaja"] = ap["p"] - ap["p_mkt"]
    ap["ev"] = ap["p"] * ap["q"] - 1
    ap["pago"] = np.where(ap["gana"] == 1, ap["q"] - 1, -1.0)
    return ap.sort_values(["date", "event"]).reset_index(drop=True)


def _fila(s, seed=0):
    """Estadisticas de un conjunto de apuestas flat de 1 unidad, con IC clusterizado."""
    if not len(s):
        return {"n": 0}
    pago = s["pago"].to_numpy(float)
    roi, lo, hi = train.bootstrap(pago, seed=seed, clusters=s["event"].to_numpy())
    sigma = float(np.std(pago, ddof=1)) if len(pago) > 1 else np.nan
    # Cuantas apuestas harian falta para distinguir ESTE ROI de cero. Va al lado del ROI
    # a proposito: una fila con ROI +2% y n=800 que necesita 20.000 no dice "gana poco",
    # dice "no dice nada", y sin este numero las dos se leen igual.
    necesarias = (apuesta.n_para_detectar(roi, sigma)
                  if np.isfinite(sigma) and roi else np.inf)
    return {
        "n": int(len(s)),
        # con stake flat ROI y yield son el mismo numero por definicion (turnover = n).
        # Divergen con Kelly, donde el stake cambia por apuesta: ver `curva`.
        "roi": float(roi), "yield": float(s["pago"].sum() / len(s)),
        "neto": float(s["pago"].sum()),
        "acierto": float(s["gana"].mean()),
        "ev_medio": float(s["ev"].mean()),
        "cuota_media": float(s["q"].mean()),
        "lo": float(lo), "hi": float(hi),
        "sigma": sigma, "n_para_concluir": float(necesarias),
        "concluyente": bool(len(s) >= MIN_N and (lo > 0 or hi < 0)),
    }


def grilla(ap, umbrales=GRID):
    """Una fila por umbral de ventaja. `ev>0` va aparte: es el filtro clasico, no un umbral."""
    filas = [{"umbral": u, **_fila(ap[ap["ventaja"] >= u])} for u in umbrales]
    return filas, {"filtro": "ev>0", **_fila(ap[ap["ev"] > 0])}


def curva(ap, umbral, banca0=100.0):
    """-> DataFrame cronologico: beneficio acumulado, bankroll y drawdown a stake flat.

    El drawdown va EN UNIDADES contra el pico del acumulado, no como fraccion de la banca:
    a stake flat la banca puede llegar a cero (y este backtest lo hace), y ahi el
    porcentaje se indefine justo donde mas importa.
    """
    s = ap[ap["ventaja"] >= umbral]
    if not len(s):
        return pd.DataFrame(columns=["date", "acumulado", "banca", "drawdown"])
    acum = s["pago"].cumsum()
    return pd.DataFrame({"date": s["date"].to_numpy(), "acumulado": acum.to_numpy(),
                         "banca": (banca0 + acum).to_numpy(),
                         "drawdown": (acum - acum.cummax()).to_numpy()})


def curva_kelly(ap, umbral, banca0=100.0, fraccion=None, tope=None, tope_evento=None):
    """-> DataFrame por EVENTO con banca compuesta, drawdown en % y crecimiento.

    Tres decisiones que cambian el numero y hay que tener a la vista:

    1. **Se dimensiona por evento, no por apuesta.** Todas las apuestas de una cartelera
       se calculan contra la banca al empezar esa cartelera. Apostar la segunda pelea de
       la noche con la banca ya actualizada por la primera supone que se puede esperar el
       resultado, y no se puede: las lineas se cierran juntas. Ademas saca el artefacto de
       que el orden de las filas dentro de un evento cambie el resultado.
    2. **Hay tope por evento.** Si la suma de los stakes de una cartelera pasa el tope,
       se escalan todos proporcionalmente. Kelly por apuesta trata cada pelea como
       independiente y dos peleas de la misma noche no lo son.
    3. **El drawdown va en % del pico**, que con Kelly fraccional si esta bien definido:
       la banca es multiplicativa y no puede tocar cero. Es la unica version comparable
       entre estrategias — en unidades, apostar mas siempre parece peor.

    Con `fraccion=1.0` es Kelly completo, que esta aca para poder mostrar por que no se usa.
    """
    from ufc.modelo import gate

    par = gate.stake_params()
    fraccion = par["fraccion"] if fraccion is None else fraccion
    tope = par["tope"] if tope is None else tope
    tope_evento = par["tope_evento"] if tope_evento is None else tope_evento

    cols = ["date", "banca", "drawdown", "apostado", "n"]
    s = ap[ap["ventaja"] >= umbral]
    if not len(s):
        return pd.DataFrame(columns=cols)

    banca, pico, filas = float(banca0), float(banca0), []
    for evento, g in s.groupby("event", sort=False):
        f = np.minimum(apuesta.kelly(g["p"].to_numpy(float), g["q"].to_numpy(float))
                       * fraccion, tope)
        montos = banca * f
        total = montos.sum()
        if total > banca * tope_evento and total > 0:
            montos = montos * (banca * tope_evento / total)
        banca += float((montos * g["pago"].to_numpy(float)).sum())
        banca = max(banca, 1e-9)      # con Kelly fraccional no llega a cero, pero no se asume
        pico = max(pico, banca)
        filas.append({"date": g["date"].iloc[0], "banca": banca,
                      "drawdown": banca / pico - 1, "apostado": float(montos.sum()),
                      "n": int(len(g))})
    return pd.DataFrame(filas, columns=cols)


def _tope_manda(ap, umbral, fraccion, tope):
    """Que proporcion de las apuestas quedan pegadas al tope en vez de a Kelly.

    Es el numero que hace legible la tabla de staking. Este modelo cree tener +38% de EV,
    asi que su Kelly pide fracciones enormes y el tope las corta TODAS: las columnas de
    1/4, 1/2 y Kelly completo terminan casi identicas y eso no significa "la fraccion no
    importa", significa "aca no esta decidiendo Kelly, esta decidiendo el tope".
    """
    s = ap[ap["ventaja"] >= umbral]
    if not len(s):
        return 0.0
    f = apuesta.kelly(s["p"].to_numpy(float), s["q"].to_numpy(float)) * fraccion
    return float((f > tope).mean())


def comparar_staking(ap, umbral, banca0=100.0, fracciones=(0.25, 0.5, 1.0)):
    """Flat contra Kelly fraccional sobre las MISMAS apuestas: banca final y peor caida.

    El punto no es elegir la que termina mas arriba — con ROI negativo, la que menos
    apuesta gana siempre, y con ROI positivo gana la que mas, hasta que quiebra. El punto
    es ver la forma: cuanto crecimiento se compra con cuanto drawdown, y que el flat de
    1 unidad sobre una banca de 100 no es "conservador", es apostar el 1% de la banca
    INICIAL para siempre — o sea subir la apuesta relativa a medida que se pierde.
    """
    from ufc.modelo import gate

    tope = gate.stake_params()["tope"]
    plana = curva(ap, umbral, banca0)
    out = [{"staking": "flat 1u", "banca_final": float(plana["banca"].iloc[-1])
            if len(plana) else banca0,
            "drawdown_max": float(plana["drawdown"].min()) if len(plana) else 0.0,
            "unidad": "unidades", "n": int(len(plana))}]
    for c in fracciones:
        k = curva_kelly(ap, umbral, banca0, fraccion=c)
        nombre = "kelly completo" if c == 1.0 else f"{c:g} kelly"
        out.append({
            "staking": nombre, "fraccion": float(c),
            "banca_final": float(k["banca"].iloc[-1]) if len(k) else banca0,
            "drawdown_max": float(k["drawdown"].min()) if len(k) else 0.0,
            "unidad": "fraccion", "n": int(k["n"].sum()) if len(k) else 0,
            "eventos": int(len(k)),
            "tope_manda": _tope_manda(ap, umbral, c, tope),
            "ruina_50": float(apuesta.riesgo_de_ruina(c, 0.5))})
    return out


# ----------------------------------------------------------------- segmentos

def etiquetas(d):
    """-> {nombre del corte: etiquetas por fila}. Sirve igual para peleas y para apuestas."""
    edad = (d["edad_a"] + d["edad_b"]) / 2
    return {
        "división": d["wc_lbs"].map(DIVISION),
        "género": np.where(d["mujer"] == 1, "femenina", "masculina"),
        "debutante": np.where(d["n_fights_min"] == 0, "con debutante", "ambos con UFC"),
        "short notice": np.where(d["reemplazo"] != 0, "un reemplazo", "sin reemplazo"),
        "peso": np.where(d["peso_no_dado"] != 0, "alguien no dio el peso", "peso ok"),
        "rondas": np.where(d["cinco_r"] == 1, "5 rondas", "3 rondas"),
        # NaN = la pelea no matcheó contra ufc_odds.csv. Va a su propio nivel: meterla en
        # "sin título" mezcla "sabemos que no lo era" con "no tenemos el dato".
        "título": np.where(d["title_bout"].isna(), "sin dato",
                           np.where(d["title_bout"] == 1, "título", "sin título")),
        "revancha": np.where(d["rematch"] == 1, "revancha", "primera vez"),
        "dif. de edad": pd.cut(d["age"].abs(), [-0.01, 2, 5, 8, 99],
                               labels=["0-2 años", "2-5", "5-8", "8+"]),
        "edad media": pd.cut(edad, [0, 28, 32, 36, 99],
                             labels=["<28", "28-32", "32-36", "36+"]),
        "ranking": np.where(d[["rank_a", "rank_b"]].notna().all(axis=1),
                            "ambos rankeados", "alguno sin ranking"),
    }


def segmentos(oof, ap, umbral, col="p_platt"):
    """Por segmento: calidad probabilistica sobre peleas + ROI sobre las apuestas del umbral.

    Cada corte se reporta con su n. Los que quedan abajo de MIN_N salen igual, marcados
    `concluyente: false`: esconderlos seria peor que mostrarlos con la advertencia.
    """
    s = ap[ap["ventaja"] >= umbral]
    et_oof, et_ap = etiquetas(oof), etiquetas(s)
    out = {}
    for corte in et_oof:
        filas = []
        for nivel, g in oof.groupby(et_oof[corte], observed=True):
            y = g["y"].to_numpy(float)
            hay = len(np.unique(y)) == 2
            sub = s[np.asarray(et_ap[corte], dtype=object) == nivel] if len(s) else s
            filas.append({
                "nivel": str(nivel), "peleas": int(len(g)),
                "log_loss": float(log_loss(y, g[col], labels=[0, 1])) if hay else None,
                "brier": float(brier_score_loss(y, g[col])) if hay else None,
                # secundaria a proposito: sube seleccionando favoritos grandes
                "accuracy": float(((g[col] > 0.5) == (y == 1)).mean()),
                **_fila(sub),
            })
        out[corte] = sorted(filas, key=lambda f: -f["peleas"])
    return out


# ------------------------------------------------------------------- reporte

def comparar_calibradores(oof):
    """-> filas con log loss, Brier y ECE de cada calibrador, y el veredicto contra el crudo.

    Excluye el fold 0, que no tuvo historia para calibrar y salio crudo: incluirlo le
    regala al baseline filas donde los tres calibradores son identicos.
    """
    d = oof[oof["fold"] > 0]
    y, clusters = d["y"].to_numpy(float), d["event"].to_numpy()
    base = d["p_identidad"].to_numpy(float)
    filas = []
    for nombre in calibra.CALIBRADORES:
        p = d[f"p_{nombre}"].to_numpy(float)
        delta = train.comparar(p, base, y, clusters=clusters)
        filas.append({"calibrador": nombre,
                      "log_loss": float(log_loss(y, p)),
                      "brier": float(brier_score_loss(y, p)),
                      "ece": calibra.ece(p, y),
                      "veredicto": train.veredicto(*delta),
                      "queda": bool(delta[2] < 0)})
    return filas


def caveat_dominio(oof):
    """Cuanto se parece la cuota historica a la que la app puede tomar hoy.

    No corrige nada: mide el desfase. `ufc_odds.csv` es una foto (probablemente de cierre
    o consenso) y la app congela en apertura en una sola casa, asi que el ROI de este
    backtest describe un juego distinto al que juega el Historial.
    """
    from ufc.datos import betano
    q = oof[["q_a", "q_b"]].dropna()
    out = {"peleas_con_cuota": int(len(q)),
           "vig_mediano_historico": float(np.median(1 / q["q_a"] + 1 / q["q_b"] - 1))}
    if betano.HIST.exists():
        h = pd.read_csv(betano.HIST)
        v = 1 / h["cuota_a"] + 1 / h["cuota_b"] - 1
        out["vig_mediano_betano"] = float(np.median(v))
        out["ticks_betano"] = int(len(h))
    return out


def main(recalcular="--recalcular" in sys.argv):
    from ufc.modelo import gate

    oof = cargar_oof(recalcular)
    print(f"\n{len(oof)} peleas OOF ({oof['date'].min().date()} a "
          f"{oof['date'].max().date()}), {oof['q_a'].notna().sum()} con cuota real")

    print("\ncalibracion (fuera de muestra, folds 2+):")
    cal = comparar_calibradores(oof)
    for f in cal:
        print(f"  {f['calibrador']:10s} log loss {f['log_loss']:.4f}  Brier {f['brier']:.4f}"
              f"  ECE {f['ece']:.4f}   {f['veredicto']}")
    col = ganador(cal)
    print(f"  entra {col[2:]}: accuracy queda como secundaria (sube seleccionando "
          "favoritos grandes, no mide ventaja).")

    ap = apuestas(oof, col)
    dom = caveat_dominio(oof)
    print(f"\nvig mediano historico {dom['vig_mediano_historico']:.2%}"
          + (f" | Betano en vivo {dom['vig_mediano_betano']:.2%}"
             if "vig_mediano_betano" in dom else "")
          + "  (el backtest cobra con cuotas de cierre; la app congela en apertura)")

    g, ev0 = grilla(ap)
    print(f"\nflat-bet de 1 unidad sobre {len(ap)} lados, IC95% clusterizado por evento:")
    print(f"  {'umbral':>7s} {'n':>6s} {'ROI':>9s} {'IC95%':>20s} {'EV medio':>9s} {'neto':>9s}")
    for f in g:
        if f["n"]:
            print(f"  {f['umbral']:>6.0%} {f['n']:>6d} {f['roi']:>+9.2%} "
                  f"[{f['lo']:>+7.2%},{f['hi']:>+7.2%}] {f['ev_medio']:>+9.2%} "
                  f"{f['neto']:>+9.1f}" + ("   <-- concluyente" if f["concluyente"] else ""))
        else:
            print(f"  {f['umbral']:>6.0%} {0:>6d}   sin apuestas")
    if ev0["n"]:
        print(f"  {'ev>0':>6s} {ev0['n']:>6d} {ev0['roi']:>+9.2%} "
              f"[{ev0['lo']:>+7.2%},{ev0['hi']:>+7.2%}]")

    mejor = max((f for f in g if f["n"] >= MIN_N), key=lambda f: f["lo"], default=None)
    umbral = mejor["umbral"] if mejor else 0.0

    # Cuanta muestra pide cada umbral para que su propio ROI se distinga de cero. Es la
    # columna que desarma la tabla de arriba: los ROI mas lindos son los de menos n.
    print("\ncuantas apuestas harian falta para concluir cada fila:")
    for f in g:
        if f["n"] >= MIN_N:
            print(f"  {f['umbral']:>6.0%} n={f['n']:>5d}  ROI {f['roi']:+7.2%}  "
                  f"necesita ~{f['n_para_concluir']:>10,.0f}"
                  + ("  <-- alcanza" if f["n"] >= f["n_para_concluir"] else ""))

    stk = comparar_staking(ap, umbral)
    print(f"\nstaking sobre las mismas apuestas (umbral {umbral:.0%}, banca 100):")
    for f in stk:
        caida = (f"{f['drawdown_max']:>8.1%}" if f["unidad"] == "fraccion"
                 else f"{f['drawdown_max']:>7.1f}u")
        extra = (f"   tope manda en {f['tope_manda']:.0%}   "
                 f"P(perder la mitad) {f['ruina_50']:.1%}" if "ruina_50" in f else "")
        print(f"  {f['staking']:16s} banca final {f['banca_final']:>9.2f}  "
              f"peor caida {caida}{extra}")
    print("  Las tres fracciones dan casi lo mismo porque el TOPE las corta a todas: este "
          "modelo\n  cree tener +38% de EV y su Kelly pide fracciones absurdas. Cuando el "
          "tope manda en\n  casi todas las apuestas, quien dimensiona no es Kelly — y esa "
          "es la defensa del tope.")

    seg = segmentos(oof, ap, umbral, col)
    niveles = sum(len(v) for v in seg.values())
    print(f"\nsegmentos al umbral {umbral:.0%}. Son {niveles} niveles mirados sobre los "
          f"MISMOS datos:\ncon ~5% de falsos positivos por nivel se esperan "
          f"{niveles * 0.05:.0f} 'hallazgos' de puro azar. Un ROI positivo aca es una "
          "hipotesis para testear\nhacia adelante, no un segmento rentable.")
    for corte, filas in seg.items():
        print(f"\n  {corte}")
        for f in filas:
            ll = f"log loss {f['log_loss']:.4f}" if f["log_loss"] is not None else " " * 18
            if not f["n"]:
                print(f"    {f['nivel']:<22s} peleas {f['peleas']:>5d}  {ll}  sin apuestas")
                continue
            print(f"    {f['nivel']:<22s} peleas {f['peleas']:>5d}  {ll}  "
                  f"ROI {f['roi']:+7.2%} [{f['lo']:+7.2%},{f['hi']:+7.2%}] n={f['n']:<5d}"
                  + ("  <-- IC no cruza cero" if f["concluyente"] else ""))

    payload = {"peleas": int(len(oof)), "lados": int(len(ap)),
               "desde": str(oof["date"].min().date()), "hasta": str(oof["date"].max().date()),
               "calibradores": cal, "calibrador_elegido": col[2:],
               "grilla": g, "ev_positivo": ev0,
               "umbral_reportado": umbral, "min_n": MIN_N, "dominio": dom,
               "curva": curva(ap, umbral).assign(
                   date=lambda d: d["date"].astype(str)).to_dict("records"),
               "curva_kelly": curva_kelly(ap, umbral).assign(
                   date=lambda d: d["date"].astype(str)).to_dict("records"),
               "staking": stk, "stake_params": gate.stake_params(),
               "segmentos": seg,
               "clv": "no computable historicamente: ufc_odds.csv trae una sola foto de "
                      "cuota por pelea, no apertura y cierre. Solo existe prospectivamente "
                      "en el Historial (ledger.py) contra los ticks de betano_hist.csv."}
    SALIDA.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    print(f"\n{SALIDA} escrito")
    if not any(f["n"] >= MIN_N and f["lo"] > 0 for f in g):
        print("\nNINGUN umbral deja el IC95% del ROI por encima de cero con n suficiente.\n"
              "Ese es el resultado: con esta evidencia no hay regla de apuesta que sostener.")


if __name__ == "__main__":
    main()
