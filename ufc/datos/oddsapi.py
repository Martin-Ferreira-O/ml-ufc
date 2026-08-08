"""Consenso multi-casa via The Odds API, si hay ODDS_API_KEY en el entorno.

Este modulo es la unica via de EV positivo del proyecto que NO necesita ganarle a nadie
con un modelo, y por eso vale la pena hacerlo bien. La idea es vieja y aritmetica: el
consenso desvigueado de muchas casas es el mejor estimador publico de la probabilidad
real, asi que si UNA casa paga mas que ese consenso, la diferencia es valor sin que haya
hecho falta predecir nada. El modelo de este repo no le gana al mercado (medido); el
mercado se le puede ganar a una casa suelta.

Hay una trampa central y hay que nombrarla antes que nada: **tomar el maximo de N precios
y compararlo contra el consenso de esos mismos N da EV positivo con frecuencia aunque los
precios sean puro ruido alrededor de la misma probabilidad.** El maximo de una muestra
esta arriba de su centro por construccion. Contra eso no alcanza con ser cuidadoso al
promediar: hace falta descontar la dispersion (`sigma` -> `apuesta.p_conservadora`) y
exigir un EV minimo, que es lo que hace `config/gate.json`.

Con eso dicho, tres detalles de implementacion que cambian el numero:

  1. **La casa del mejor precio NO entra al consenso de ese lado.** El motivo es
     independencia, no prudencia: si la referencia contiene el precio que se esta
     juzgando, es en parte una funcion de el, y lo que mide deja de ser "cuanto se aparta
     este precio del mercado". Ojo con la direccion — excluirla SUBE el EV medido
     (sobre el fixture de los tests: +1.8% -> +6.1%), no lo baja. Incluirla comprime
     artificialmente la ventaja hacia cero y esconderia una ventaja real igual que
     inventaria una falsa; el que protege del ruido es el punto de arriba, no este.
  2. **Precios stale afuera.** Una casa que no actualiza hace horas no esta ofreciendo
     ese precio, esta mostrando uno viejo. Entra al consenso como si fuera opinion actual
     y encima suele quedar como la que "mejor paga".
  3. **La dispersion entre casas es incertidumbre, no ventaja.** Cuando las casas no se
     ponen de acuerdo, el consenso es menos confiable, no mas explotable. Sale como
     `sigma` para que `apuesta.p_conservadora` la descuente.

Sin key (gratis en the-odds-api.com, 500 requests/mes) devuelve {} y la app sigue igual.
El tier gratis alcanza: la app cachea 30 minutos.
"""

import csv
import datetime
import os

import numpy as np
import requests

from ufc import nombres, rutas
from ufc.datos import betano
from ufc.modelo import devig

API = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
ROTO = (requests.RequestException, ValueError, KeyError, TypeError)

# Una casa que no toca su precio en este tiempo no esta cotizando: esta mostrando una
# foto vieja. El limite es generoso a proposito — en MMA las lineas se mueven poco entre
# carteleras y un limite corto dejaria el consenso con dos casas.
STALE = datetime.timedelta(hours=12)
# Abajo de esto no hay consenso que valga: la mediana de dos casas es el promedio de dos
# casas, y excluir una para comparar deja una sola.
MIN_CASAS = 3
HIST = rutas.DATOS / "odds_hist.csv"
COLS_HIST = ["ts_utc", "bookmaker", "mercado", "a", "b", "cuota_a", "cuota_b",
             "last_update"]


def _fecha(x):
    try:
        return datetime.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _precios(evento, x, y, ahora=None):
    """-> [{casa, cx, cy, last_update}] de un evento, ya filtrado de stale y roto."""
    ahora = ahora or datetime.datetime.now(datetime.timezone.utc)
    out = []
    for casa in evento.get("bookmakers", []):
        visto = _fecha(casa.get("last_update"))
        if visto is not None and ahora - visto > STALE:
            continue
        for m in casa.get("markets", []):
            if m.get("key") != "h2h":
                continue
            precios = {nombres.normalizar(o.get("name", "")): o.get("price")
                       for o in m.get("outcomes", [])}
            try:
                cx, cy = float(precios[x]), float(precios[y])
            except ROTO:
                continue
            # Una cuota <= 1 no paga nada y una implicita que suma < 1 seria arbitraje
            # contra si misma: en los dos casos el dato esta roto, no es una oportunidad.
            if cx <= 1 or cy <= 1 or 1 / cx + 1 / cy < 1:
                continue
            out.append({"casa": casa.get("title") or casa.get("key", "?"),
                        "cx": cx, "cy": cy,
                        "last_update": casa.get("last_update", "")})
    return out


def _consenso_sin(precios, excluida):
    """P(gana X) por mediana desvigueada, ignorando a `excluida`.

    Cada casa se desviguea POR SEPARADO y despues se toma la mediana de las
    probabilidades. Al reves (mediana de cuotas y despues desviguear) mezcla margenes
    distintos y el resultado no es la probabilidad de ninguna casa.

    Mediana y no media: una sola casa con un precio roto o stale mueve la media y no
    mueve la mediana, y el consenso tiene que sobrevivir a que una fuente se rompa.
    """
    ps = [devig.de_cuotas(p["cx"], p["cy"]) for p in precios if p["casa"] != excluida]
    if len(ps) < MIN_CASAS:
        return None, np.nan, 0
    return float(np.median(ps)), float(np.std(ps, ddof=1)), len(ps)


def _parsear(eventos, ahora=None):
    """-> {(x, y) ordenadas: fila de consenso} por pelea.

    Cada lado trae su propio consenso, calculado SIN la casa que ofrece su mejor precio.
    Por eso `p_x` y `p_y` no suman 1: son dos estimaciones distintas de la misma pelea,
    cada una limpia respecto del precio contra el que se la va a comparar.
    """
    tabla = {}
    for e in eventos:
        ka = nombres.normalizar(e.get("home_team", ""))
        kb = nombres.normalizar(e.get("away_team", ""))
        if not ka or not kb or ka == kb:
            continue
        x, y = sorted((ka, kb))
        precios = _precios(e, x, y, ahora)
        if not precios:
            continue
        mejor_x = max(precios, key=lambda p: p["cx"])
        mejor_y = max(precios, key=lambda p: p["cy"])
        p_x, sigma_x, n_x = _consenso_sin(precios, mejor_x["casa"])
        p_y, sigma_y, n_y = _consenso_sin(precios, mejor_y["casa"])
        if p_x is None or p_y is None:
            continue
        todas = [devig.de_cuotas(p["cx"], p["cy"]) for p in precios]
        tabla[(x, y)] = {
            "p_x": p_x, "p_y": 1 - p_y,          # los dos orientados a X
            "sigma_x": sigma_x, "sigma_y": sigma_y,
            "mejor": (mejor_x["cx"], mejor_y["cy"]),
            "casa": (mejor_x["casa"], mejor_y["casa"]),
            "casas": len(precios), "casas_consenso": (n_x, n_y),
            # el consenso CON todas, solo para mostrar: no decide nada
            "p_todas": float(np.median(todas)),
            "vig_mediano": float(np.median([devig.vig(p["cx"], p["cy"])
                                            for p in precios])),
        }
    return tabla


def _archivar(tabla_cruda, ahora=None):
    """Un tick por casa y pelea, append-only. Es la fuente historica que hoy no existe.

    `ufc_odds.csv` trae UNA foto por pelea y por eso el CLV historico no es computable.
    Esto empieza a acumular apertura->cierre por casa desde el dia uno: no se puede
    recuperar hacia atras, asi que se guarda aunque todavia no se use.
    """
    HIST.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not HIST.exists()
    ts = (ahora or datetime.datetime.now(datetime.timezone.utc)).isoformat(
        timespec="seconds")
    with HIST.open("a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(COLS_HIST)
        for (x, y), precios in sorted(tabla_cruda.items()):
            for p in precios:
                w.writerow([ts, p["casa"], "h2h", x, y, p["cx"], p["cy"],
                            p["last_update"]])


def cuotas(archivar=True):
    """La tabla de consenso, o {} sin key o si la API no responde. No levanta."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}
    try:
        r = requests.get(API, timeout=20, params={
            "apiKey": key, "regions": "eu,us,uk", "markets": "h2h",
            "oddsFormat": "decimal"})
        r.raise_for_status()
        eventos = r.json()
    except ROTO:
        return {}
    if archivar:
        crudo = {}
        for e in eventos:
            ka, kb = (nombres.normalizar(e.get("home_team", "")),
                      nombres.normalizar(e.get("away_team", "")))
            if ka and kb and ka != kb:
                x, y = sorted((ka, kb))
                precios = _precios(e, x, y)
                if precios:
                    crudo[(x, y)] = precios
        if crudo:
            try:
                _archivar(crudo)
            except OSError:
                pass          # no poder archivar no puede tumbar la consulta
    return _parsear(eventos)


def buscar(tabla, a, b):
    """El consenso orientado al orden pedido, o None. Mismo matcheo que Betano.

    `p_a` es el consenso limpio para juzgar el mejor precio de A, y `p_b_propio` el de B.
    No suman 1 y no tienen que sumar: cada uno excluye una casa distinta.
    """
    ka, kb = nombres.normalizar(a), nombres.normalizar(b)
    ordenadas = tuple(sorted((ka, kb)))
    fila = tabla.get(ordenadas) or betano._aproximado(tabla, ordenadas)
    if fila is None:
        return None
    cx, cy = fila["mejor"]
    casa_x, casa_y = fila["casa"]
    derecho = ka <= kb
    return {
        "p_a": fila["p_x"] if derecho else 1 - fila["p_y"],
        "p_b_propio": (1 - fila["p_y"]) if derecho else fila["p_x"],
        "sigma_a": fila["sigma_x"] if derecho else fila["sigma_y"],
        "sigma_b": fila["sigma_y"] if derecho else fila["sigma_x"],
        "mejor": (cx, cy) if derecho else (cy, cx),
        "casa": (casa_x, casa_y) if derecho else (casa_y, casa_x),
        "casas": fila["casas"], "vig_mediano": fila["vig_mediano"],
    }


def valor(info):
    """-> [{lado, cuota, casa, p, sigma, ev, ev_low}] con el EV contra el consenso limpio.

    Es la senal central de la estrategia top-down: no dice quien gana la pelea, dice que
    una casa esta pagando por encima de lo que opina el resto del mercado. `ev_low` usa
    la cota inferior por dispersion entre casas — cuando no se ponen de acuerdo, el
    consenso vale menos y la ventaja tiene que ser mas grande para contar.
    """
    from ufc.modelo import apuesta, gate

    if not info:
        return []
    potencia = gate.stake_params()["potencia"]
    out = []
    for lado, p, sigma, cuota, casa in (
            ("a", info["p_a"], info["sigma_a"], info["mejor"][0], info["casa"][0]),
            ("b", info["p_b_propio"], info["sigma_b"], info["mejor"][1], info["casa"][1])):
        sigma = 0.0 if not np.isfinite(sigma) else float(sigma)
        p_low = float(apuesta.p_conservadora(p, sigma, potencia))
        out.append({"lado": lado, "cuota": float(cuota), "casa": casa,
                    "p": float(p), "sigma": sigma, "p_low": p_low,
                    "ev": float(apuesta.ev(p, cuota)),
                    "ev_low": float(apuesta.ev(p_low, cuota))})
    return out
