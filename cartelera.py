"""Carteleras que vienen, con la prediccion del modelo pelea por pelea.

La fuente es la API publica de ESPN: es la unica que quedo. ufcstats.com metio un
challenge JS anti-bot en su pagina de eventos futuros, y el dataset de odds
(shortlikeafox/ultimate_ufc_dataset) esta congelado desde abril de 2026.

ESPN no da cuotas, asi que aca no hay nivel de confianza: para eso hay que ingresar la
cuota a mano en la app o en predict.py.

    python cartelera.py        # lista los eventos proximos
    python cartelera.py 3      # recorre el evento 3, una pelea a la vez
"""

import datetime
import sys

import requests

import predict

API = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
SIN_RIVAL = {"tba", "opponent tba"}


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


def proximas(dias=90):
    hoy = datetime.date.today()
    r = requests.get(API, timeout=30, params={
        "dates": f"{hoy:%Y%m%d}-{hoy + datetime.timedelta(days=dias):%Y%m%d}"})
    r.raise_for_status()
    return _parsear(r.json())


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
        # ponytail: pausa solo en terminal, para no colgarse cuando lo pipeas a un archivo
        if n < len(peleas) and sys.stdin.isatty():
            if input("  [Enter] siguiente · [q] salir  ").strip().lower() == "q":
                return
        print()


if __name__ == "__main__":
    main()
