"""Resultados de la cartelera que se esta peleando, minutos despues de cada combate.

El pipeline del repo baja los resultados del mirror de UFCStats de Greco1899
(`ufc.datos.fetch`), que refresca una vez al dia. Sirve para entrenar y no sirve para
narrar: un sabado a las once de la noche, la pelea que termino hace veinte minutos
todavia no existe en ningun CSV. Y sin resultado no se puede decir si la IA acerto.

La fuente de aca es la MISMA API de ESPN que ya usa `ufc.datos.cartelera` para las
carteleras futuras — el endpoint devuelve las dos cosas, `cartelera._parsear` simplemente
descarta todo lo que no este en `state == "pre"`. O sea: cero dependencias nuevas y cero
scraping de ufcstats.com, que ademas metio un challenge anti-bot (ver `cartelera`).

Lo que ESPN da por pelea, verificado contra carteleras de abril a agosto de 2026:

    competitors[].winner                -> quien gano
    status.type.state                   -> pre | in | post
    status.period / status.displayClock -> asalto y minuto
    details[].type.text                 -> "Unofficial Winner Kotko" | "... Submission"
                                           | "... Decision"

El metodo se traduce al vocabulario que ya existe en `ufc.modelo.settlement` en vez de
inventar un tercero, y como esos textos son los NO oficiales (ESPN los publica antes de
que el resultado quede firme), lo que se escribe en `data/resultados.csv` es solo el
ganador. Un `overturned` posterior lo corrige el mirror de UFCStats al dia siguiente, que
en `predictores._indice_resultados` tiene prioridad sobre lo que escribimos aca.

    python -m ufc.datos.resultados              # la cartelera de hoy
    python -m ufc.datos.resultados 2026-07-25   # una fecha puntual
"""

import datetime
import sys

from ufc.datos import cartelera
from ufc.modelo import settlement
from ufc.registro import predictores

# ESPN escribe el metodo como "Unofficial Winner Kotko". Es play-by-play, no la ficha
# oficial: los tres valores de abajo son los unicos que aparecieron en 315 peleas de
# 2026. Lo que no matchee queda en None y el mensaje lo dice en vez de inventar.
METODOS = {"kotko": settlement.KO, "submission": settlement.SUB,
           "decision": settlement.DEC}
PREFIJO = "Unofficial Winner "


def _metodo(competencia):
    for detalle in competencia.get("details", []):
        texto = str(detalle.get("type", {}).get("text", ""))
        if texto.startswith(PREFIJO):
            return METODOS.get(texto[len(PREFIJO):].strip().lower())
    return None


def _pelea(competencia):
    """-> dict de una pelea, o None si ESPN no mando los dos lados."""
    atletas = [x.get("athlete", {}) for x in competencia.get("competitors", [])]
    nombres = [x.get("displayName") for x in atletas]
    if len(nombres) != 2 or not all(nombres):
        return None
    estado = competencia.get("status", {})
    # Mismo criterio que `cartelera._parsear`: si ESPN no manda estado, la damos por
    # pendiente. Es preferible no anunciar nada a anunciar un ganador que no existe.
    fase = estado.get("type", {}).get("state", "pre")
    ganador = next(("ab"[i] for i, x in enumerate(competencia["competitors"])
                    if x.get("winner")), None)
    # Un empate o un no-contest terminan en `post` sin ganador: la pelea queda resuelta
    # pero sin lado, y quien la lea tiene que poder distinguir eso de "todavia no paso".
    return {
        "a": nombres[0], "b": nombres[1],
        "peso": competencia.get("type", {}).get("abbreviation", ""),
        "estado": fase,
        "ganador": ganador if fase == "post" else None,
        "metodo": _metodo(competencia) if fase == "post" else None,
        "asalto": estado.get("period") if fase == "post" else None,
        "reloj": estado.get("displayClock") or "" if fase == "post" else "",
    }


def parsear(payload, fecha=None):
    """-> el evento pedido con sus peleas, o None.

    Sin `fecha` devuelve el ultimo evento de la ventana, que es lo que quiere un bot
    narrando en vivo: la cartelera de anoche sigue siendo "la de hoy" a las 2 AM UTC,
    cuando la fecha del calendario ya cambio y la del evento no.
    """
    eventos = []
    for e in payload.get("events", []):
        peleas = [p for p in map(_pelea, e.get("competitions", [])) if p]
        if not peleas:
            continue
        eventos.append({
            "evento": e.get("name", "?"),
            "fecha": str(e.get("date", ""))[:10],
            "inicio_utc": str(e.get("date", "")) or None,
            # ESPN lista los preliminares primero; `cartelera._parsear` invierte para que
            # el main event quede arriba y aca se hace igual, si no la "pelea 1" de un
            # mensaje y la del cartel de la app serian dos peleas distintas.
            "peleas": peleas[::-1]})

    if fecha is not None:
        fecha = str(fecha)[:10]
        return next((e for e in eventos if e["fecha"] == fecha), None)
    return eventos[-1] if eventos else None


def en_vivo(fecha=None, payload=None):
    """-> {'evento', 'fecha', 'inicio_utc', 'peleas': [...]} o None si no hay nada.

    La ventana es de un dia para atras y uno para adelante: una cartelera que arranca
    21:00Z del sabado termina el domingo por la madrugada, y ESPN la indexa por su fecha
    de inicio.
    """
    if payload is None:
        ancla = (datetime.date.fromisoformat(str(fecha)[:10]) if fecha
                 else datetime.datetime.now(datetime.UTC).date())
        dia = datetime.timedelta(days=1)
        payload = cartelera.scoreboard(ancla - dia, ancla + dia)
    return parsear(payload, fecha)


def resueltas(evento):
    """-> las peleas de `evento` que ya terminaron con un ganador."""
    return [p for p in (evento or {}).get("peleas", []) if p["ganador"]]


def volcar(evento):
    """Escribe los ganadores en `data/resultados.csv`. -> cuantas filas quedaron.

    Reusa `predictores.guardar_resultados`, que reemplaza TODAS las filas del evento de
    una vez: hay que pasarle la cartelera resuelta completa, no pelea por pelea, o cada
    llamada borraria la anterior. Es el mismo patron que `consenso.sincronizar_picks`.

    Con esto la app entera —`ia.evaluar`, `ledger`, `predictores.aciertos`— ve el
    resultado el mismo dia en vez de esperar el refresh diario del mirror.
    """
    filas = [[evento["evento"], p["a"], p["b"], p["ganador"]]
             for p in resueltas(evento)]
    if filas:
        predictores.guardar_resultados(filas, evento["evento"])
    return len(filas)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        evento = en_vivo(argv[0] if argv else None)
    except OSError as exc:
        print(f"ERROR: no se pudo consultar ESPN: {exc}", file=sys.stderr)
        return 2
    if not evento:
        print("No hay ninguna cartelera en esa ventana.")
        return 1

    print(f"{evento['evento']} — {evento['fecha']}\n")
    for i, p in enumerate(evento["peleas"], 1):
        if p["estado"] != "post":
            print(f"  {i:2d}  {p['a']} vs {p['b']} — {p['estado']}")
            continue
        if not p["ganador"]:
            print(f"  {i:2d}  {p['a']} vs {p['b']} — sin ganador (empate o no contest)")
            continue
        gana, pierde = p[p["ganador"]], p["b" if p["ganador"] == "a" else "a"]
        print(f"  {i:2d}  {gana} def {pierde} por {p['metodo'] or '?'} "
              f"R{p['asalto']} {p['reloj']}")
    print(f"\n{volcar(evento)} resultados en data/resultados.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
