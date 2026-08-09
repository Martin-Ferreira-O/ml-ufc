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
    # Direccional sobre TODA la cohorte, parejas incluidas: el log loss las necesita y el
    # lado hacia el que se inclino existe aunque no sea una pick. Quien reporta un record
    # tiene que partir por `es_pick`; ver `metricas`.
    df["acierto"] = ((df["p_a_ia"] > 0.5) == (df["y"] > 0.5)).astype(float)
    # NaN cuenta como pick, igual que `tipster._abstuvo`: solo el "parejo" explicito se
    # abstiene. Una fila sin veredicto es una fila vieja, no una abstencion.
    df["es_pick"] = df["veredicto"] != "parejo"
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


def _delta(valores, clusters):
    """-> (media, lo, hi, n). `lo`/`hi` en None cuando no hay IC95% posible.

    `train.bootstrap` remuestrea CLUSTERS enteros —una cartelera es un cluster, porque las
    peleas de una misma noche comparten arbitros, altitud y hasta el humor del juez— y con
    una sola cartelera todos los resampleos salen identicos: devuelve un intervalo de ancho
    cero. Eso no es un IC estrecho, es la ausencia de IC disfrazada de certeza absoluta, que
    es la peor forma posible de reportarlo. Se prefiere decir que falta muestra.
    """
    v = np.asarray(valores, float)
    ok = np.isfinite(v)
    c = np.asarray(clusters)[ok]
    media = float(v[ok].mean()) if ok.any() else float("nan")
    if ok.sum() < 2 or len(np.unique(c)) < 2:
        return media, None, None, int(ok.sum())
    media, lo, hi = train.bootstrap(v[ok], clusters=c)
    return float(media), float(lo), float(hi), int(ok.sum())


def _record_total():
    """-> (aciertos, total) de la IA en `picks.csv`, todas las versiones del prompt.

    Es otro numero que el de la cohorte, y por eso se muestra aparte en vez de reemplazarlo:
    incluye las peleas que analizo la v1, que veia el modelo, el mercado y las picks humanas.
    Como historial vale —son picks que se hicieron y se ganaron— pero como medida de ESTA
    version del prompt no, y mezclarlos da el rendimiento de un predictor que no existe.

    Filtra por `origen` y no por el nombre del predictor: el nombre lleva el proveedor
    adentro ("IA (Gemini)") y cambia el dia que se cambie de modelo.
    """
    try:
        picks = predictores.leer()
        if not len(picks):
            return 0, 0
        suyos = set(picks[picks["origen"] == predictores.ORIGEN_IA]["predictor"])
        g = predictores.aciertos()
        g = g[g["predictor"].isin(suyos)]
        return int(g["aciertos"].sum()), int(g["total"].sum())
    except (OSError, KeyError, ValueError):
        return 0, 0


def metricas(df=None):
    """-> dict con todo lo que hay que decir de la cohorte, calculado una sola vez.

    Existe porque el record se calculaba en dos lados con criterios distintos: `resumen()`
    contaba las peleas que la IA declaro parejas como si fueran picks (de ahi el "7 de 8"
    cuando habia hecho 3 picks) y `tipster.marcador()` las excluia. Un dict, dos
    formateadores —terminal y Telegram— y los dos numeros no se pueden volver a separar.

    Vacio si no hay cohorte. Los formateadores tratan `{}` como "todavia no hay nada".
    """
    df = evaluar() if df is None else df
    if not len(df):
        return {}
    picks, parejas = df[df["es_pick"]], df[~df["es_pick"]]
    clusters = df["evento"].to_numpy()

    ll = {}
    for nombre, col in (("IA", "ll_ia"), ("modelo", "ll_modelo"),
                        ("mercado", "ll_mercado")):
        v = df[col].to_numpy(float)
        ok = np.isfinite(v)
        ll[nombre] = (float(v[ok].mean()), int(ok.sum())) if ok.any() else (None, 0)

    deltas = {nombre: _delta(df[col] - df["ll_ia"], clusters)
              for nombre, col in (("modelo", "ll_modelo"), ("mercado", "ll_mercado"))}

    clv = None
    con_precio = df[df["ev_al_cierre"].notna()]
    if len(con_precio):
        media, lo, hi, n = _delta(con_precio["ev_al_cierre"],
                                  con_precio["evento"].to_numpy())
        finito = con_precio["ev_al_cierre"].to_numpy(float)
        finito = finito[np.isfinite(finito)]
        clv = {"media": media, "lo": lo, "hi": hi, "n": n,
               "faltan": (apuesta.n_para_detectar(
                   0.02, float(np.std(finito, ddof=1)) or 0.04)
                   if len(finito) >= 2 else None)}

    return {"prompt_v": dossier.VERSION,
            "n": len(df), "carteleras": int(df["evento"].nunique()),
            "picks_n": len(picks), "picks_ok": int(picks["acierto"].sum()),
            "parejas_n": len(parejas), "parejas_ok": int(parejas["acierto"].sum()),
            "ll": ll, "deltas": deltas, "clv": clv, "total_picks": _record_total()}


def marca(ok, n):
    """-> "3 de 4 (75%)", o "sin muestra" cuando no hay nada que promediar."""
    return f"{ok} de {n} ({ok / n:.0%})" if n else "sin muestra"


def plural(n, singular, plural_=None):
    return f"{n} {singular if n == 1 else (plural_ or singular + 's')}"


def _linea(nombre, delta):
    media, lo, hi, n = delta
    if lo is None:
        return (f"  {nombre:22s} {media:+.4f}  n={n} — sin IC95%: hace falta mas de una "
                f"cartelera")
    return f"  {nombre:22s} {media:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]  n={n}"


def resumen(df=None):
    """-> texto con acierto, log loss comparado y CLV. El formato de `devig.main`."""
    m = metricas(df)
    if not m:
        return "Todavia no hay ninguna pelea analizada por IA con resultado cargado."

    lineas = [f"Cohorte IA (prompt {m['prompt_v']}, ciego al mercado): {m['n']} peleas "
              f"resueltas de {plural(m['carteleras'], 'cartelera')}.", ""]
    lineas.append(f"  acierto de sus picks   {marca(m['picks_ok'], m['picks_n'])}")
    lineas.append(f"  parejas declaradas     {m['parejas_n']} de {m['n']} — no son pick; "
                  f"su inclinacion acerto {marca(m['parejas_ok'], m['parejas_n'])}")

    lineas += ["", "log loss (mas bajo es mejor), sobre toda la cohorte:"]
    for nombre, (media, n) in m["ll"].items():
        lineas.append(f"  {nombre:22s} {media:.4f}  n={n}" if media is not None
                      else f"  {nombre:22s} sin muestra")

    lineas += ["", "delta pareado contra la IA (negativo = el otro es mejor que la IA):"]
    lineas += [_linea(f"{nombre} - IA", d) for nombre, d in m["deltas"].items()]

    lineas += ["", "CLV de la pick contra la linea de cierre — la IA nunca vio el precio,",
               "asi que esto mide si su lectura deportiva le gana al mercado:"]
    if m["clv"]:
        clv = m["clv"]
        lineas.append(_linea("EV al cierre",
                             (clv["media"], clv["lo"], clv["hi"], clv["n"])))
        if clv["faltan"]:
            lineas.append(f"  para concluir un CLV de +2% harian falta ~{clv['faltan']} "
                          f"peleas (hay {clv['n']} con cierre)")
    else:
        lineas.append("  sin ninguna pelea con precio de apertura y de cierre todavia")

    ok, total = m["total_picks"]
    if total:
        lineas += ["", f"Historial completo en picks.csv: {marca(ok, total)}.",
                   "Incluye las peleas de versiones anteriores del prompt, que veian el",
                   "modelo y el mercado: son otro predictor, y por eso no entran arriba."]

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
