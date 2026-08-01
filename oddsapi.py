"""Consenso multi-casa via The Odds API, si hay ODDS_API_KEY en el entorno.

Una sola casa no permite ni line-shopping ni la estrategia que si es rentable sin
ganarle a nadie con un modelo: tomar el consenso del mercado como probabilidad
"verdadera" y apostar donde una casa paga mas que eso (top-down). Este modulo trae
las cuotas de varias casas de una vez: la mediana desvigueada es el consenso, y la
mejor cuota por lado es donde conviene apostar.

Sin key (gratis en the-odds-api.com, 500 requests/mes) devuelve {} y la app sigue
igual. El tier gratis alcanza: la app cachea 30 minutos.
"""

import os

import numpy as np
import requests

import betano
import predict

API = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
ROTO = (requests.RequestException, ValueError, KeyError, TypeError)


def _parsear(eventos):
    """-> {(x, y) ordenadas: {"p_x", "mejor": (cx, cy), "casas"}} por pelea."""
    tabla = {}
    for e in eventos:
        ka = predict._normalizar(e.get("home_team", ""))
        kb = predict._normalizar(e.get("away_team", ""))
        if not ka or not kb or ka == kb:
            continue
        x, y = sorted((ka, kb))
        ps, cxs, cys = [], [], []
        for casa in e.get("bookmakers", []):
            for m in casa.get("markets", []):
                if m.get("key") != "h2h":
                    continue
                precios = {predict._normalizar(o.get("name", "")): o.get("price")
                           for o in m.get("outcomes", [])}
                try:
                    cx, cy = float(precios[x]), float(precios[y])
                except ROTO:
                    continue
                ps.append(float(predict._desvig(1 / cx, 1 / cy)))
                cxs.append(cx)
                cys.append(cy)
        if ps:
            tabla[(x, y)] = {"p_x": float(np.median(ps)),
                             "mejor": (max(cxs), max(cys)), "casas": len(ps)}
    return tabla


def cuotas():
    """La tabla de consenso, o {} sin key o si la API no responde. No levanta."""
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}
    try:
        r = requests.get(API, timeout=20, params={
            "apiKey": key, "regions": "eu", "markets": "h2h",
            "oddsFormat": "decimal"})
        r.raise_for_status()
        return _parsear(r.json())
    except ROTO:
        return {}


def buscar(tabla, a, b):
    """El consenso orientado al orden pedido, o None. Mismo matcheo que Betano."""
    ka, kb = predict._normalizar(a), predict._normalizar(b)
    ordenadas = tuple(sorted((ka, kb)))
    fila = tabla.get(ordenadas) or betano._aproximado(tabla, ordenadas)
    if fila is None:
        return None
    cx, cy = fila["mejor"]
    if ka <= kb:
        return {"p_a": fila["p_x"], "mejor": (cx, cy), "casas": fila["casas"]}
    return {"p_a": 1 - fila["p_x"], "mejor": (cy, cx), "casas": fila["casas"]}
