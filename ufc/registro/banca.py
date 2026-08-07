"""La banca: el denominador de todo lo demas.

Sin una banca declarada no hay staking posible. "Aposte $1.000" no es una decision de
riesgo, es un numero; "aposte el 1% de la banca" si lo es, y son la misma plata solo si
la banca son $100.000. Todo `ufc/modelo/apuesta.py` habla en fracciones y necesita este
archivo para traducirlas a pesos.

Vive aparte de `apuestas.csv` a proposito: la banca es una sola y cambia cuando el
usuario ingresa o retira plata, no cuando se registra una apuesta. Mezclarlas obligaria a
reescribir el historial para corregir un deposito.

El archivo no se versiona (`data/*` esta en .gitignore): es plata real del usuario.
"""

import json

from ufc import rutas

BANCA = rutas.DATOS / "banca.json"
DEFAULT = {"inicial_clp": 0, "moneda": "CLP"}


def leer():
    """-> dict con la banca declarada. Sin archivo devuelve 0, que apaga el staking."""
    if not BANCA.exists():
        return dict(DEFAULT)
    try:
        return {**DEFAULT, **json.loads(BANCA.read_text())}
    except (ValueError, OSError):
        return dict(DEFAULT)


def guardar(inicial_clp, moneda="CLP"):
    if float(inicial_clp) < 0:
        raise ValueError("La banca no puede ser negativa")
    BANCA.parent.mkdir(parents=True, exist_ok=True)
    BANCA.write_text(json.dumps({"inicial_clp": int(inicial_clp), "moneda": moneda},
                                indent=1))


def actual(resumen=None):
    """Banca declarada mas el beneficio ya liquidado. 0 si no hay banca declarada.

    Es sobre la banca ACTUAL que se dimensiona la proxima apuesta, no sobre la inicial:
    stakear un porcentaje fijo de la banca inicial es apostar cada vez mas fuerte en
    relacion a lo que queda mientras se pierde, que es la forma mas comun de quebrar
    creyendo que se esta siendo prudente.
    """
    inicial = float(leer()["inicial_clp"])
    if not inicial:
        return 0.0
    return max(inicial + float((resumen or {}).get("beneficio") or 0.0), 0.0)
