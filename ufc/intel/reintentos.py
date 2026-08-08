"""Reintentos contra la API de Gemini, respetando el RetryInfo que manda Google.

Vivia dentro de `bot.py`, pero el bot importa media docena de modulos (cartelera,
identities, sources, store) y cualquier otro consumidor de la API terminaba arrastrando
todo eso solo para reintentar un 429. Aca queda sin dependencias: `ufc/ia/` lo usa igual.

`bot.py` sigue re-exportando los tres nombres, asi que nada de lo que ya llamaba a
`bot.con_reintentos` se entera del cambio.
"""

import re
import time

# 429 es cuota, el resto es la nube teniendo un mal dia. Ninguno significa que el
# pedido este mal armado: los 4xx que si lo significan tienen que explotar de una.
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


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
