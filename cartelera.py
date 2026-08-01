"""Carteleras que vienen, con la prediccion del modelo pelea por pelea.

La fuente es la API publica de ESPN: es la unica que quedo. ufcstats.com metio un
challenge JS anti-bot en su pagina de eventos futuros, y el dataset de odds
(shortlikeafox/ultimate_ufc_dataset) esta congelado desde abril de 2026.

ESPN no da cuotas, asi que aca no hay nivel de confianza: para eso hay que ingresar la
cuota a mano en la app o en predict.py.

    python cartelera.py        # lista los eventos proximos
    python cartelera.py 3      # recorre el evento 3, una pelea a la vez
"""

import csv
import datetime
import pathlib
import sys

import requests

import features
import predict

API = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
SIN_RIVAL = {"tba", "opponent tba"}
HIST = pathlib.Path("data/cartelera_hist.csv")


def _parsear(payload):
    """-> [{'evento', 'fecha', 'peleas': [{'peso', 'a', 'b'}]}], main event primero."""
    eventos = []
    for e in payload.get("events", []):
        peleas = []
        for c in e.get("competitions", []):
            nombres = [x["athlete"]["displayName"] for x in c.get("competitors", [])]
            if len(nombres) != 2 or any(n.lower() in SIN_RIVAL for n in nombres):
                continue
            peleas.append({"peso": c.get("type", {}).get("abbreviation", ""),
                           "a": nombres[0], "b": nombres[1]})
        if peleas:
            eventos.append({"evento": e.get("name", "?"),
                            "fecha": str(e.get("date", ""))[:10],
                            # ESPN lista los preliminares primero y el main event ultimo
                            "peleas": peleas[::-1]})
    return eventos


def _archivar(eventos, hoy=None):
    """Primer avistaje de cada pelea anunciada. La fecha en que aparece una pelea es
    la unica forma de detectar reemplazos tardios (anunciada <14 dias antes), que es
    justo la informacion que el mercado tiene y el modelo no."""
    HIST.parent.mkdir(parents=True, exist_ok=True)
    vistas = set()
    if HIST.exists():
        with HIST.open() as f:
            vistas = {tuple(fila[1:4]) for fila in csv.reader(f)}
    with HIST.open("a", newline="") as f:
        w = csv.writer(f)
        if not vistas:
            w.writerow(["visto", "evento", "a", "b", "fecha_evento"])
        hoy = (hoy or datetime.date.today()).isoformat()
        for e in eventos:
            for p in e["peleas"]:
                clave = (e["evento"], p["a"], p["b"])
                if clave not in vistas:
                    w.writerow([hoy, *clave, e["fecha"]])
                    vistas.add(clave)


def proximas(dias=90):
    hoy = datetime.date.today()
    r = requests.get(API, timeout=30, params={
        "dates": f"{hoy:%Y%m%d}-{hoy + datetime.timedelta(days=dias):%Y%m%d}"})
    r.raise_for_status()
    eventos = _parsear(r.json())
    _archivar(eventos)
    return eventos


def contexto(peso, es_main):
    """(wc_lbs, mujer, cinco_r) desde el texto de peso de ESPN; NaN si no matchea.

    El orden de _LBS pone "Light Heavyweight" antes que "Heavyweight", asi que el
    primer match es el correcto. cinco_r se aproxima con "es el main event".
    """
    p = str(peso).lower()
    wc = next((float(lbs) for nombre, lbs in features._LBS if nombre.lower() in p),
              float("nan"))
    return wc, float("women" in p), float(es_main)


def predecir(pelea, modelo, estado, cuotas=None):
    """La pelea + su prediccion. Los debutantes no tienen historial: van con `error`."""
    try:
        return pelea | predict.predict(pelea["a"], pelea["b"], modelo, estado,
                                       cuotas=cuotas)
    except ValueError:
        return pelea | {"error": "sin historial en UFC"}


def main():
    eventos = proximas()
    if len(sys.argv) < 2:
        print("Eventos proximos:\n")
        for i, e in enumerate(eventos, 1):
            print(f"  {i:2d}  {e['fecha']}  {e['evento']}  ({len(e['peleas'])} peleas)")
        print(f"\n  python {sys.argv[0]} <n>   para recorrer uno")
        return

    i = int(sys.argv[1])
    if not 1 <= i <= len(eventos):
        print(f"Elegi un evento entre 1 y {len(eventos)}.")
        sys.exit(1)
    e = eventos[i - 1]
    modelo, estado = predict.cargar()
    peleas = [predecir(p, modelo, estado) for p in e["peleas"]]

    print(f"\n{e['evento']} — {e['fecha']}\n")
    for n, p in enumerate(peleas, 1):
        print(f"[{n}/{len(peleas)}] {p['peso']}")
        print(f"  {p['a']}  vs  {p['b']}")
        if "error" in p:
            print(f"  {p['error']} — no se puede predecir")
        else:
            print(f"  modelo   {p['p_a']:.1%}  |  {p['p_b']:.1%}")
            print("  factores " + ", ".join(f"{k} {v:+.2f}" for k, v in p["factores"]))
        m = predict.metodo(modelo, *contexto(p["peso"], n == 1))
        if m:
            print(f"  metodo   KO/TKO {m['ko']:.0%} | sumision {m['sub']:.0%} | "
                  f"decision {m['dec']:.0%}  (por division, no por el par)")
        # ponytail: pausa solo en terminal, para no colgarse cuando lo pipeas a un archivo
        if n < len(peleas) and sys.stdin.isatty():
            if input("  [Enter] siguiente · [q] salir  ").strip().lower() == "q":
                return
        print()


if __name__ == "__main__":
    main()
