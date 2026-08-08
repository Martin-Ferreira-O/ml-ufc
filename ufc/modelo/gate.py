"""El interruptor: si el proyecto esta autorizado a apostar, y por que no.

Existe para que la respuesta "no apostar" deje de ser una constante hardcodeada y pase a
ser una CONCLUSION auditable. Antes `predict.APUESTAS_AUTOMATICAS = False` era correcto
pero no explicaba nada: no decia que haria falta para que fuera True, ni cuanto falta, ni
quien podria cambiarlo. Una constante no se puede refutar, y un sistema que no se puede
refutar termina cambiandose despues de una racha buena.

La regla vive en `config/gate.json` y se escribe ANTES de mirar los resultados. Este
modulo solo la lee y la evalua contra el forward test del ledger. El criterio es CLV y no
ROI por una razon aritmetica, no filosofica: distinguir una ventaja del 2% de cero pide
~20.000 apuestas por ROI y ~decenas por CLV (ver `apuesta.n_para_detectar`). Con ~500
apuestas al anio, el ROI como criterio significa cuarenta anios de espera; o sea que no
es un criterio, es una excusa para no tener ninguno.

Que el gate este cerrado NO apaga la app: se sigue prediciendo, registrando, comparando
precios y midiendo CLV. Lo unico que bloquea es que algo se presente como recomendacion.
"""

import json

import numpy as np

from ufc import rutas

CONFIG = rutas.RAIZ / "config" / "gate.json"


def _falta_config():
    return {"autorizado": False, "motivo": f"Falta {CONFIG.name}: sin regla preregistrada "
            "no hay nada que autorizar.", "n": 0, "faltan": None}


def leer():
    """La regla preregistrada. Sin el archivo devuelve None y todo queda cerrado."""
    if not CONFIG.exists():
        return None
    return json.loads(CONFIG.read_text())


def stake_params(config=None):
    """Los parametros de staking de la regla, con los defaults de `apuesta` como respaldo.

    Van juntos a proposito: fraccion y topes son parte de la regla congelada, no una
    preferencia de la UI. Si se mueven, cambia la version y se reinicia la cohorte.
    """
    from ufc.modelo import apuesta

    s = (config or leer() or {}).get("stake", {})
    return {"fraccion": float(s.get("fraccion_kelly", apuesta.FRACCION)),
            "tope": float(s.get("tope_por_apuesta", apuesta.TOPE_APUESTA)),
            "tope_evento": float(s.get("tope_por_evento", apuesta.TOPE_EVENTO)),
            "potencia": float(s.get("potencia_cota", 0.95))}


def evaluar(clv, eventos=None, config=None, n_boot=2000, seed=0):
    """-> dict con el veredicto, a partir de la serie de CLV economico ya realizada.

    `clv` son los `ev_al_cierre` de las apuestas seguidas (retorno esperado de la apuesta
    valuado a la probabilidad justa del cierre). `eventos` clusteriza el bootstrap: dos
    peleas de la misma cartelera no son dos observaciones independientes.

    Devuelve tambien `faltan`: cuantas apuestas mas hacen falta para poder concluir con la
    dispersion observada. Es lo unico que convierte "todavia no" en un plan.
    """
    from ufc.modelo import apuesta
    from ufc.modelo import train

    config = config if config is not None else leer()
    if config is None:
        return _falta_config()
    criterio = config.get("criterio", {})
    n_minimo = int(criterio.get("n_minimo", 100))
    piso = float(criterio.get("ic_inferior_mayor_que", 0.0))

    clv = np.asarray(clv, float)
    hay = np.isfinite(clv)
    clv = clv[hay]
    eventos = np.asarray(eventos)[hay] if eventos is not None else None
    n = int(len(clv))

    out = {"version": config.get("version"), "n": n, "n_minimo": n_minimo,
           "metrica": criterio.get("metrica", "ev_al_cierre"), "autorizado": False}
    if not n:
        out["motivo"] = (config.get("estado_inicial", {}).get("motivo")
                         or "Sin apuestas registradas: no hay nada que evaluar.")
        out["faltan"] = n_minimo
        return out

    media, lo, hi = train.bootstrap(clv, n_boot=n_boot, seed=seed, clusters=eventos)
    sigma = float(np.std(clv, ddof=1)) if n > 1 else np.nan
    necesarias = (float(apuesta.n_para_detectar(media, sigma))
                  if n > 1 and np.isfinite(sigma) and media != 0 else np.inf)
    out.update({"clv": float(media), "lo": float(lo), "hi": float(hi),
                "sigma": sigma, "batio_cierre": float((clv > 0).mean()),
                "n_para_concluir": necesarias,
                "faltan": max(n_minimo - n, 0)})

    if n < n_minimo:
        # El n_minimo preregistrado manda, y no la prueba de potencia: con muestra chica
        # la dispersion es ella misma una estimacion ruidosa y puede pedir cuatro
        # apuestas. Bajar la vara con el numero que sale de los propios datos es
        # exactamente lo que el preregistro existe para impedir.
        out["motivo"] = (f"Muestra insuficiente: {n} de {n_minimo} apuestas "
                         f"preregistradas. CLV medio {media:+.2%} "
                         f"[{lo:+.2%}, {hi:+.2%}], todavia sin valor probatorio.")
    elif lo > piso:
        out["autorizado"] = True
        out["motivo"] = (f"CLV medio {media:+.2%} con IC95% [{lo:+.2%}, {hi:+.2%}] sobre "
                         f"{n} apuestas: el limite inferior supera {piso:.0%}.")
    else:
        out["motivo"] = (f"CLV medio {media:+.2%} pero el IC95% [{lo:+.2%}, {hi:+.2%}] "
                         f"toca {piso:.0%}: no se puede distinguir de conseguir el mismo "
                         "precio que el cierre.")
    return out


def estado(config=None):
    """El veredicto contra el ledger real. Es lo que mira la app.

    Import perezoso de `ledger` porque `ledger` importa `predict` y `predict` pregunta por
    el gate: al nivel del modulo eso es un ciclo.
    """
    from ufc.registro import ledger

    try:
        df, _ = ledger.evaluar()
    except (FileNotFoundError, OSError, ValueError):
        df = None
    if df is None or not len(df) or "ev_al_cierre" not in df:
        return evaluar([], config=config)
    seguidas = df[(df["lado"] != "") & df["ev_al_cierre"].notna()]
    return evaluar(seguidas["ev_al_cierre"], seguidas["evento"], config=config)


def autorizado():
    """True solo si la regla preregistrada se cumple contra los datos reales."""
    return bool(estado().get("autorizado"))
