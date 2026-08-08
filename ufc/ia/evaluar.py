"""Si la capa de IA sirvio de algo, medido. Cohorte propia, sin tocar el ledger.

Este modulo existe porque sin el la IA es una opinion que nadie puede desmentir. El repo
mide todo con IC95% y esto no puede ser la excepcion: compara la probabilidad de la IA
contra la del modelo y la del mercado **sobre las mismas peleas**, y evalua sus apuestas
contra la linea de cierre.

Deliberadamente NO escribe en `data/ledger.csv`. Ese archivo es la cohorte preregistrada
que lee `gate.estado()`, y mezclarle las apuestas de la IA arruinaria la unica serie de
CLV que el proyecto viene acumulando. Son dos cohortes separadas, y cada una se puede
concluir por su cuenta.

    python -m ufc.ia.evaluar
"""

import sys

import numpy as np
import pandas as pd

from ufc import nombres
from ufc.ia import dossier, store
from ufc.modelo import apuesta, devig, train
from ufc.registro import ledger, predictores

EPS = 1e-6


def _resultados():
    """{par normalizado: ganador normalizado}. UFCStats primero, manual de respaldo."""
    indice = {}
    for clave, datos in predictores._indice_resultados().items():
        indice[clave[1]] = datos["ganador"]
    try:
        for fila in ledger._resultados().itertuples():
            indice[fila.par] = fila.ganador
    except (OSError, KeyError, ValueError):
        pass  # sin data/raw no hay resultados de ufcstats; queda lo cargado a mano
    return indice


def _log_loss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluar(prompt_v=dossier.VERSION):
    """-> DataFrame con una fila por pelea analizada que ya tiene resultado.

    Columnas clave: `y` (gano A), `acierto`, `ll_ia` / `ll_modelo` / `ll_mercado` y
    `ev_al_cierre`.

    Solo la cohorte de `prompt_v`. Las filas de la v1 salieron de un prompt que veia el
    modelo, el mercado y las picks humanas: promediarlas con estas daria el rendimiento de
    un predictor que no existe. `prompt_v=None` las trae todas, para poder compararlas.
    """
    df = store.leer()
    if not len(df):
        return pd.DataFrame()
    if prompt_v is not None:
        df = df[df["prompt_v"].astype(str) == str(prompt_v)]
        if not len(df):
            return pd.DataFrame()

    ganadores = _resultados()
    df = df.copy()
    df["par"] = [tuple(sorted((nombres.normalizar(a), nombres.normalizar(b))))
                 for a, b in zip(df["a"], df["b"])]
    df["ganador"] = df["par"].map(ganadores)
    df = df[df["ganador"].notna()].copy()
    if not len(df):
        return pd.DataFrame()

    df["y"] = [float(nombres.normalizar(a) == g) for a, g in zip(df["a"], df["ganador"])]
    for col in ("p_a_ia", "p_a_modelo", "p_a_con_odds", "p_a_mercado",
                "cuota_tomada", "ev_ia"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["acierto"] = ((df["p_a_ia"] > 0.5) == (df["y"] > 0.5)).astype(float)
    df["ll_ia"] = _log_loss(df["p_a_ia"], df["y"])
    df["ll_modelo"] = np.where(df["p_a_modelo"].notna(),
                               _log_loss(df["p_a_modelo"].fillna(0.5), df["y"]), np.nan)
    df["ll_mercado"] = np.where(df["p_a_mercado"].notna(),
                                _log_loss(df["p_a_mercado"].fillna(0.5), df["y"]), np.nan)
    df["ev_al_cierre"] = _clv(df)
    return df


def _clv(df):
    """EV de la pick de la IA contra la linea de cierre. NaN si no hay precio.

    Se mide sobre TODA pick con precio registrado, no sobre un subconjunto que la IA haya
    elegido apostar — no elige: es ciega al mercado. Y por eso mismo esto es la prueba
    mas limpia que tiene el proyecto: si una lectura puramente deportiva, hecha sin ver el
    precio, le gana sistematicamente a la linea de cierre, esa ventaja es real. Si no, la
    capa no aporta sobre el mercado y el numero lo va a decir.

    Reusa el corte estricto de `ledger._limite` (la hora real de inicio del evento, con
    fallback a fecha+1 dia para las filas viejas): un tick posterior al primer campanazo
    es una cuota EN VIVO y no la linea de cierre.
    """
    salida = pd.Series(np.nan, index=df.index)
    apostadas = df[df["cuota_tomada"].notna() & df["lado"].isin(["a", "b"])]
    if not len(apostadas):
        return salida

    limites = {}
    for fila in apostadas.itertuples():
        limites[(fila.a, fila.b)] = ledger._limite({
            "inicio_utc": fila.inicio_utc,
            "fecha_evento": pd.to_datetime(fila.fecha_evento, errors="coerce")})
    try:
        cierres = ledger._cierres(limites)
    except (OSError, KeyError, ValueError):
        return salida

    for fila in apostadas.itertuples():
        par = cierres.get((fila.a, fila.b))
        if not par or not all(par):
            continue
        p_cierre_a = float(devig.desvig(1 / par[0], 1 / par[1]))
        p_cierre = p_cierre_a if fila.lado == "a" else 1 - p_cierre_a
        salida.at[fila.Index] = p_cierre * fila.cuota_tomada - 1
    return salida


def _linea(nombre, valores, clusters):
    v = np.asarray(valores, float)
    ok = np.isfinite(v)
    if ok.sum() < 2:
        return f"  {nombre:22s} n={int(ok.sum())} — sin muestra suficiente"
    media, lo, hi = train.bootstrap(v[ok], clusters=np.asarray(clusters)[ok])
    return f"  {nombre:22s} {media:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]  n={int(ok.sum())}"


def resumen(df=None):
    """-> texto con acierto, log loss comparado y CLV. El formato de `devig.main`."""
    df = evaluar() if df is None else df
    if not len(df):
        return "Todavia no hay ninguna pelea analizada por IA con resultado cargado."

    clusters = df["evento"].to_numpy()
    lineas = [f"Cohorte IA (prompt v{dossier.VERSION}, ciego al mercado): {len(df)} "
              f"peleas resueltas de {df['evento'].nunique()} carteleras.", ""]
    lineas.append(f"  acierto de la pick     {df['acierto'].mean():.1%} "
                  f"({int(df['acierto'].sum())}/{len(df)})")

    lineas += ["", "log loss (mas bajo es mejor):"]
    for nombre, col in (("IA", "ll_ia"), ("modelo", "ll_modelo"),
                        ("mercado", "ll_mercado")):
        v = df[col].to_numpy(float)
        ok = np.isfinite(v)
        lineas.append(f"  {nombre:22s} {v[ok].mean():.4f}  n={int(ok.sum())}"
                      if ok.any() else f"  {nombre:22s} sin muestra")

    lineas += ["", "delta pareado contra la IA (negativo = el otro es mejor que la IA):"]
    for nombre, col in (("modelo - IA", "ll_modelo"), ("mercado - IA", "ll_mercado")):
        pareado = df[col] - df["ll_ia"]
        lineas.append(_linea(nombre, pareado, clusters))

    parejas = int((df["veredicto"] == "parejo").sum())
    lineas += ["", f"peleas que la IA declaro parejas: {parejas} de {len(df)} "
                   "(no se registran como pick)"]

    con_precio = df[df["ev_al_cierre"].notna()]
    lineas += ["", "CLV de la pick contra la linea de cierre — la IA nunca vio el precio,",
               "asi que esto mide si su lectura deportiva le gana al mercado:"]
    if len(con_precio):
        clv = con_precio["ev_al_cierre"]
        lineas.append(_linea("EV al cierre", clv, con_precio["evento"].to_numpy()))
        finito = clv[np.isfinite(clv)]
        if len(finito) >= 2:
            falta = apuesta.n_para_detectar(0.02, float(np.std(finito, ddof=1)) or 0.04)
            lineas.append(f"  para concluir un CLV de +2% harian falta ~{falta} peleas "
                          f"(hay {len(finito)} con cierre)")
    else:
        lineas.append("  sin ninguna pelea con precio de apertura y de cierre todavia")

    lineas += ["",
               "La cohorte del gate (`data/ledger.csv`) no se toca: esto se mide aparte."]
    return "\n".join(lineas)


def main(argv=None):
    try:
        print(resumen())
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
