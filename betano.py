"""Cuotas decimales de lat.betano.com para las peleas de la cartelera.

Betano no tiene API publica, pero tampoco hace falta: el sitio deja todo el estado de la
pagina embebido en el HTML como `window["initial_state"]`, con los nombres y las cuotas
adentro. Eso se baja con un GET y se parsea como JSON — nada de headless browser.

Cada evento de UFC es una "liga" aparte para Betano, asi que hay que pasar por la pagina
de la region UFC para sacar la lista y despues por cada liga. Son dos o tres requests.

Es scraping, o sea que se va a romper: cuando pase, `cuotas()` devuelve {} y la app
sigue andando sin cuotas. Para arreglarlo:

    python betano.py           # baja el estado crudo a data/betano_raw.json y lo resume
"""

import csv
import datetime
import difflib
import json
import pathlib

import requests

import predict

BASE = "https://lat.betano.com"
REGION = "188692"                                  # UFC dentro del deporte MMA
UFC = f"/sport/mma/campeonatos/ufc/{REGION}/"
ANCLA = 'window["initial_state"]='

# Sin headers de browser la respuesta es el splash de Cloudflare, no la pagina.
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-419,es;q=0.9",
    "Referer": f"{BASE}/sport/mma/",
}
CRUDO = pathlib.Path("data/betano_raw.json")
HIST = pathlib.Path("data/betano_hist.csv")

# Errores esperables de una fuente que no controlamos: red caida, HTML sin el ancla,
# JSON cortado, o que le hayan cambiado la forma.
ROTO = (requests.RequestException, ValueError, KeyError, TypeError)


def _estado(ruta):
    """El `window["initial_state"]` de una pagina de Betano, ya parseado."""
    r = requests.get(BASE + ruta, headers=HEADERS, timeout=20)
    r.raise_for_status()
    i = r.text.index(ANCLA) + len(ANCLA)
    return json.loads(r.text[i:r.text.index("</script>", i)].strip().rstrip(";"))


def _ligas(estado):
    """Las rutas de las carteleras UFC, menos la que ya vino en la pagina que las lista."""
    for region in estado["data"].get("dropdownList", []):
        if region.get("id") == REGION:
            return [l["url"] for l in region.get("leagues", []) if not l.get("selected")]
    return []


def _mercados(nodo):
    """Todo dict con 'selections' del arbol, sin depender de como este anidado.

    Hoy viven en data.blocks[].events[].markets[], pero Betano mueve la forma seguido y
    buscarlos recursivamente sobrevive a que cambien el nivel. Son 8 lineas.
    """
    if isinstance(nodo, dict):
        if isinstance(nodo.get("selections"), list):
            yield nodo
        for v in nodo.values():
            yield from _mercados(v)
    elif isinstance(nodo, list):
        for v in nodo:
            yield from _mercados(v)


def _precio(sel):
    try:
        return float(sel["price"])
    except ROTO:
        return None


def _parsear(estado):
    """-> {(clave_x, clave_y) ordenadas alfabeticamente: (cuota_x, cuota_y)}.

    Se queda con los mercados de dos selecciones, que para MMA es el ganador. Los otros
    mercados de dos vias (over/under de rounds) entran igual pero con claves tipo
    'mas de 1.5' que ninguna pelea va a buscar nunca, asi que no molestan.
    """
    tabla = {}
    for m in _mercados(estado):
        sels = m["selections"]
        if len(sels) != 2:
            continue
        nombres = [predict._normalizar(s.get("name", "")) for s in sels]
        precios = [_precio(s) for s in sels]
        if not all(nombres) or not all(precios) or nombres[0] == nombres[1]:
            continue
        par = dict(zip(nombres, precios))
        clave = tuple(sorted(nombres))
        # El primero gana: el moneyline viene antes que los mercados derivados.
        tabla.setdefault(clave, (par[clave[0]], par[clave[1]]))
    return tabla


def _archivar(tabla, ahora=None):
    """Un tick por par y corrida: el historico apertura -> cierre que hace medible el
    CLV y el vig real de Betano. Los mercados de rounds entran igual: son las odds de
    props que ninguna otra fuente guarda."""
    HIST.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not HIST.exists()
    ts = (ahora or datetime.datetime.now()).isoformat(timespec="seconds")
    with HIST.open("a", newline="") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["ts", "a", "b", "cuota_a", "cuota_b"])
        for (a, b), (ca, cb) in sorted(tabla.items()):
            w.writerow([ts, a, b, ca, cb])


def cuotas():
    """Las cuotas de todas las carteleras UFC, o {} si Betano no responde. No levanta."""
    try:
        raiz = _estado(UFC)
    except ROTO:
        return {}
    tabla = _parsear(raiz)
    for ruta in _ligas(raiz):
        # Una cartelera que falle no se lleva puestas a las demas.
        try:
            tabla.update(_parsear(_estado(ruta)))
        except ROTO:
            continue
    if tabla:
        _archivar(tabla)
    return tabla


def _aproximado(tabla, ordenadas):
    """La pelea mas parecida, para cuando Betano escribe el nombre distinto que ESPN.

    `ordenadas` viene ya alfabetizada, igual que las claves de la tabla: comparar un par
    con el otro al reves da una similitud baja y no matchea nada.

    ponytail: match difuso ciego sobre el string de la pelea entera. Alcanza para 'Jr.',
    apodos y tildes raras. Si empieza a fallar seguido, matchear por apellido.
    """
    claves = {f"{x}|{y}": (x, y) for x, y in tabla}
    cerca = difflib.get_close_matches("|".join(ordenadas), claves, n=1, cutoff=0.85)
    return tabla[claves[cerca[0]]] if cerca else None


def buscar(tabla, a, b):
    """(cuota_a, cuota_b) en el orden pedido, o None si la pelea no esta en Betano."""
    ka, kb = predict._normalizar(a), predict._normalizar(b)
    ordenadas = tuple(sorted((ka, kb)))
    par = tabla.get(ordenadas) or _aproximado(tabla, ordenadas)
    if par is None:
        return None
    return par if ka <= kb else par[::-1]


def main():
    """Modo descubrimiento: para cuando hay que arreglar el parser."""
    raiz = _estado(UFC)
    CRUDO.parent.mkdir(parents=True, exist_ok=True)
    CRUDO.write_text(json.dumps(raiz, indent=1, ensure_ascii=False))
    print(f"crudo -> {CRUDO}  ({CRUDO.stat().st_size / 1e3:.0f} KB)")
    print(f"carteleras: {[UFC] + _ligas(raiz)}")

    mercados = list(_mercados(raiz))
    print(f"mercados en la pagina raiz: {len(mercados)}")
    if mercados:
        print("\nun mercado de ejemplo:")
        print(json.dumps(mercados[0], indent=1, ensure_ascii=False)[:600])

    tabla = cuotas()
    print(f"\npeleas con cuota: {len(tabla)}")
    for (x, y), (cx, cy) in sorted(tabla.items()):
        print(f"  {x:28s} {cx:6.2f}   vs   {y:28s} {cy:6.2f}")


if __name__ == "__main__":
    main()
