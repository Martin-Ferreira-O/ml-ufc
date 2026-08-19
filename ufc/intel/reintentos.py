"""El ritmo al que se le habla a Gemini: el grifo que regula, y el reintento que perdona.

Vivia dentro de `bot.py`, pero el bot importa media docena de modulos (cartelera,
identities, sources, store) y cualquier otro consumidor de la API terminaba arrastrando
todo eso solo para reintentar un 429. Aca queda sin dependencias: `ufc/ia/` lo usa igual.

`bot.py` sigue re-exportando los tres nombres, asi que nada de lo que ya llamaba a
`bot.con_reintentos` se entera del cambio.

Las dos mitades son distintas y hacen falta las dos. `con_reintentos` es reactivo: el 429
ya paso, se lee el `RetryInfo` y se espera. `espaciar` es lo que evita que pase, y es la
mitad que faltaba — `ufc/ia/consenso` mandaba catorce peleas con cuatro hilos y sin ningun
freno, asi que el free tier contestaba 429 y los cuatro hilos volvian juntos a insistir.
Es la decision que ya estaba escrita en `docs/handoff/fighter-intel/DECISIONS.md`: el free
tier se regula, no se paraleliza.
"""

import os
import re
import threading
import time

# 429 es cuota, el resto es la nube teniendo un mal dia. Ninguno significa que el
# pedido este mal armado: los 4xx que si lo significan tienen que explotar de una.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}

# Requests por minuto del free tier. UFC_IA_MODELO puede apuntar a un modelo con otra
# cuota, y una cuenta paga aguanta mucho mas: por eso el numero se puede subir con
# UFC_IA_RPM en vez de estar clavado.
RPM = 10

# ponytail: un lock global y un timestamp, que es exactamente el alcance del problema —
# un proceso, una API key. Dos procesos a la vez (el cron y el boton de Streamlit) pueden
# sumarse y pasarse igual; para eso queda el backoff de abajo.
_grifo = threading.Lock()
_ultima = 0.0


def espaciar(*, sleep=time.sleep, reloj=time.monotonic):
    """Bloquea hasta que se pueda llamar sin pasarse del RPM.

    Duerme con el lock tomado a proposito: eso serializa a todos los hilos contra el mismo
    reloj sin necesidad de un token bucket. `sleep` y `reloj` son inyectables por la misma
    razon que en `con_reintentos` — el test no tiene por que esperar de verdad.
    """
    global _ultima
    try:
        rpm = float(os.getenv("UFC_IA_RPM") or RPM)
    except ValueError:
        rpm = RPM
    gap = 60.0 / max(rpm, 0.1)
    with _grifo:
        falta = _ultima + gap - reloj()
        if falta > 0:
            sleep(falta)
        _ultima = reloj()


def _retry_after(exc):
    """El `retryDelay` que manda Gemini en el error, en segundos. None si no viene.

    Google sabe cuanto falta para que se libere la cuota mejor que cualquier backoff
    que inventemos, asi que cuando lo dice se le hace caso. El +1 es margen: el reloj
    del servidor y el nuestro no son el mismo.
    """
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    for detail in details.get("error", {}).get("details", []):
        if not isinstance(detail, dict):
            continue
        value = detail.get("retryDelay")
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", str(value or ""))
        if match:
            return float(match.group(1)) + 1
    return None


def con_reintentos(call, *, sleep=time.sleep, attempts=4, label="IA"):
    """Respeta RetryInfo de Gemini y reintenta errores transitorios.

    `sleep` es inyectable para que los tests no esperen de verdad.
    """
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            response = getattr(exc, "response", None)
            code = (getattr(exc, "code", None) or
                    getattr(response, "status_code", None))
            if code not in TRANSIENT_STATUS or attempt == attempts - 1:
                raise
            delay = _retry_after(exc) or (5 * (3 ** attempt))
            print(f"REINTENTO {label}: HTTP {code}, espera {delay:.0f}s "
                  f"({attempt + 1}/{attempts - 1})", flush=True)
            sleep(delay)
