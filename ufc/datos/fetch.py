"""Descarga los CSVs de ufcstats.com ya scrapeados por Greco1899/scrape_ufc_stats.

Ese repo refresca los datos a diario, asi que re-correr este script = datos al dia.

Despues de esto van las dos fuentes externas, las dos cacheadas (solo piden lo nuevo):
`python -m ufc.datos.wiki` trae los reemplazos y los pesos no dados de los eventos
nuevos, y `python -m ufc.datos.sherdog` el record pre-UFC de los peleadores nuevos. Sin
la primera, las peleas nuevas quedan con `reemplazo`/`peso_no_dado` en NaN; sin la
segunda, los debutantes quedan sin su historial regional, que es justo donde el modelo
no tiene nada mas.
"""

import time

import requests

from ufc import rutas

BASE = "https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/"
CSVS = [
    "ufc_event_details",
    "ufc_fight_details",
    "ufc_fight_results",
    "ufc_fight_stats",
    "ufc_fighter_details",
    "ufc_fighter_tott",
]
# Odds de mercado desde 2010, de otro repo. Se usan como feature del segundo modelo y
# como benchmark. Matchea el 95% de las peleas del periodo que cubre (78.9% del total);
# el resto queda NaN. Si el repo muere, el pipeline sigue andando sin este archivo.
ODDS = ("https://raw.githubusercontent.com/shortlikeafox/ultimate_ufc_dataset/main/"
        "ufc-master.csv", "ufc_odds")
RAW = rutas.RAW
REINTENTOS = 3


def _descargar(url, name, obligatorio=True):
    """Actualiza un CSV sin destruir la copia buena si la red falla.

    GitHub puede tener cortes breves de DNS/conexion. Si ya hay una descarga anterior,
    el resto del pipeline todavia puede reconstruir y entrenar el modelo con ella. La
    fuente de odds, ademas, es opcional por contrato.
    """
    destino = RAW / f"{name}.csv"
    error = None
    for intento in range(1, REINTENTOS + 1):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            if not r.content:
                raise requests.RequestException("respuesta vacia")
            # Escribir al costado y reemplazar al final evita dejar un CSV truncado si
            # el proceso se interrumpe durante la descarga.
            temporal = destino.with_suffix(".csv.part")
            temporal.write_bytes(r.content)
            temporal.replace(destino)
            print(f"{destino} — {len(r.content) / 1e6:.1f} MB", flush=True)
            return True
        except requests.RequestException as exc:
            error = exc
            if intento < REINTENTOS:
                print(f"  reintento {intento}/{REINTENTOS - 1} para {name}…", flush=True)
                time.sleep(intento)

    detalle = f"{type(error).__name__}: {error}"
    if destino.exists() and destino.stat().st_size:
        print(f"AVISO: no se pudo actualizar {name}; se conserva la copia local "
              f"({detalle})", flush=True)
        return False
    if not obligatorio:
        print(f"AVISO: no se pudo bajar la fuente opcional {name}; se omite "
              f"({detalle})", flush=True)
        return False
    raise RuntimeError(
        f"No se pudo descargar {name} y no existe una copia local. "
        "Revisa la conexion a internet e intenta nuevamente. "
        f"Detalle: {detalle}"
    ) from error


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    actualizados = sum(_descargar(f"{BASE}{name}.csv", name) for name in CSVS)
    actualizados += _descargar(*ODDS, obligatorio=False)
    if actualizados < len(CSVS) + 1:
        print(f"AVISO: {actualizados}/{len(CSVS) + 1} fuentes actualizadas; "
              "el pipeline continua con las copias locales disponibles.", flush=True)


if __name__ == "__main__":
    main()
