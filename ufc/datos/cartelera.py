"""Carteleras que vienen, con la prediccion del modelo pelea por pelea.

La fuente es la API publica de ESPN: es la unica que quedo. ufcstats.com metio un
challenge JS anti-bot en su pagina de eventos futuros, y el dataset de odds
(shortlikeafox/ultimate_ufc_dataset) esta congelado desde abril de 2026.

ESPN no da cuotas, asi que aca no hay nivel de confianza: para eso hay que ingresar la
cuota a mano en la app o en predict.py.

    python -m ufc.datos.cartelera     # lista los eventos proximos
    python -m ufc.datos.cartelera 3   # recorre el evento 3, una pelea a la vez
"""

import csv
import datetime
import sys

import requests

from ufc import rutas
from ufc.modelo import features, predict

API = "https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard"
SIN_RIVAL = {"tba", "opponent tba"}
HIST = rutas.DATOS / "cartelera_hist.csv"
EVENTOS_HIST = rutas.RAW / "ufc_event_details.csv"
PELEAS_HIST = rutas.RAW / "ufc_fight_results.csv"


def _parsear(payload):
    """-> [{'evento', 'fecha', 'peleas': [{'peso', 'a', 'b'}]}], main event primero.

    Cada pelea tambien se lleva la bandera y el pais de los dos lados, que ESPN manda en
    la misma respuesta (`athlete.flag`) y el cartel de la app dibuja en cada esquina. Van
    en "" cuando no vienen: el peleador recien firmado a veces llega sin ficha completa.
    """
    eventos = []
    for e in payload.get("events", []):
        peleas = []
        for c in e.get("competitions", []):
            # ESPN sigue devolviendo el evento el dia entero despues de que termino, y una
            # pelea peleada no se puede apostar. "pre" por defecto: si ESPN no manda
            # status la damos por pendiente antes que hacerla desaparecer.
            if c.get("status", {}).get("type", {}).get("state", "pre") != "pre":
                continue
            atletas = [x.get("athlete", {}) for x in c.get("competitors", [])]
            nombres = [x["displayName"] for x in atletas]
            if len(nombres) != 2 or any(n.lower() in SIN_RIVAL for n in nombres):
                continue
            banderas = [x.get("flag") or {} for x in atletas]
            peleas.append({"peso": c.get("type", {}).get("abbreviation", ""),
                           "a": nombres[0], "b": nombres[1],
                           "bandera_a": banderas[0].get("href", ""),
                           "bandera_b": banderas[1].get("href", ""),
                           "pais_a": banderas[0].get("alt", ""),
                           "pais_b": banderas[1].get("alt", "")})
        if peleas:
            eventos.append({"evento": e.get("name", "?"),
                            "fecha": str(e.get("date", ""))[:10],
                            # ESPN manda "2026-08-15T21:00Z" y antes se tiraba la hora.
                            # Es el unico corte estricto que hace computable el CLV: sin
                            # ella, el "cierre" se elige con fecha+1 dia y puede tomar
                            # cuotas EN VIVO o posteriores al combate.
                            "inicio_utc": str(e.get("date", "")) or None,
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
            # `inicio_utc` va ultima: las filas viejas tienen una columna menos y
            # agregarla al final las deja leibles sin migrar el archivo.
            w.writerow(["visto", "evento", "a", "b", "fecha_evento", "inicio_utc"])
        hoy = (hoy or datetime.date.today()).isoformat()
        for e in eventos:
            for p in e["peleas"]:
                clave = (e["evento"], p["a"], p["b"])
                if clave not in vistas:
                    w.writerow([hoy, *clave, e["fecha"], e.get("inicio_utc") or ""])
                    vistas.add(clave)


def scoreboard(desde, hasta):
    """El payload crudo de ESPN para un rango de fechas.

    Vive aca y no en cada llamador porque el mismo endpoint sirve las dos cosas: las
    carteleras que vienen (`proximas`) y los resultados de la que se esta peleando
    (`ufc.datos.resultados`). Una sola URL, un solo timeout, un solo lugar donde mirar
    cuando ESPN cambie algo.
    """
    r = requests.get(API, timeout=30, params={
        "dates": f"{desde:%Y%m%d}-{hasta:%Y%m%d}"})
    r.raise_for_status()
    return r.json()


def proximas(dias=90):
    hoy = datetime.date.today()
    eventos = _parsear(scoreboard(hoy, hoy + datetime.timedelta(days=dias)))
    _archivar(eventos)
    return eventos


def anteriores(limite=20):
    """Carteleras disputadas, reconstruidas desde los CSV locales de UFCStats.

    A diferencia de ``HIST``, estas son las peleas que efectivamente ocurrieron: el
    primer avistaje ESPN tambien conserva combates cancelados y rivales reemplazados.
    """
    if not (EVENTOS_HIST.exists() and PELEAS_HIST.exists()):
        return []
    fechas = {}
    with EVENTOS_HIST.open() as f:
        for fila in csv.DictReader(f):
            evento = fila.get("EVENT", "").strip()
            if not evento:
                continue
            try:
                fecha = datetime.datetime.strptime(
                    fila.get("DATE", "").strip(), "%B %d, %Y").date().isoformat()
            except ValueError:
                fecha = fila.get("DATE", "").strip()
            fechas[evento] = fecha

    peleas = {}
    with PELEAS_HIST.open() as f:
        for fila in csv.DictReader(f):
            evento = fila.get("EVENT", "").strip()
            nombres = fila.get("BOUT", "").split(" vs. ", 1)
            if evento not in fechas or len(nombres) != 2:
                continue
            peso = fila.get("WEIGHTCLASS", "").strip().removesuffix(" Bout")
            pelea = {"peso": peso, "a": nombres[0].strip(), "b": nombres[1].strip()}
            resultado = fila.get("OUTCOME", "").strip()
            if resultado in {"W/L", "L/W"}:
                pelea["ganador"] = "a" if resultado == "W/L" else "b"
            if pelea not in peleas.setdefault(evento, []):
                peleas[evento].append(pelea)

    eventos = [{"evento": evento, "fecha": fechas[evento], "peleas": cartelera,
                "historico": True}
               for evento, cartelera in peleas.items() if cartelera]
    eventos.sort(key=lambda e: e["fecha"], reverse=True)
    return eventos if limite is None else eventos[:limite]


def contexto(peso, es_main):
    """(wc_lbs, mujer, cinco_r) desde el texto de peso de ESPN; NaN si no matchea.

    El orden de _LBS pone "Light Heavyweight" antes que "Heavyweight", asi que el
    primer match es el correcto. cinco_r se aproxima con "es el main event".
    """
    p = str(peso).lower()
    wc = next((float(lbs) for nombre, lbs in features._LBS if nombre.lower() in p),
              float("nan"))
    return wc, float("women" in p), float(es_main)


def predecir(pelea, modelo, estado, cuotas=None, event_date=None,
             circ_a=(float("nan"), float("nan")),
             circ_b=(float("nan"), float("nan"))):
    """La pelea + su prediccion. Los debutantes no tienen historial: van con `error`."""
    try:
        return pelea | predict.predict(pelea["a"], pelea["b"], modelo, estado,
                                       event_date=event_date, cuotas=cuotas,
                                       circ_a=circ_a, circ_b=circ_b)
    except ValueError:
        return pelea | {"error": "sin historial en UFC"}


def main():
    eventos = proximas()
    if len(sys.argv) < 2:
        print("Eventos proximos:\n")
        for i, e in enumerate(eventos, 1):
            print(f"  {i:2d}  {e['fecha']}  {e['evento']}  ({len(e['peleas'])} peleas)")
        print("\n  python -m ufc.datos.cartelera <n>   para recorrer uno")
        return

    i = int(sys.argv[1])
    if not 1 <= i <= len(eventos):
        print(f"Elegi un evento entre 1 y {len(eventos)}.")
        sys.exit(1)
    e = eventos[i - 1]
    modelo, estado = predict.cargar()
    peleas = [predecir(p, modelo, estado, event_date=e["fecha"]) for p in e["peleas"]]

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
