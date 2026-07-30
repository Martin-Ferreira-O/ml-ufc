"""Descarga los CSVs de ufcstats.com ya scrapeados por Greco1899/scrape_ufc_stats.

Ese repo refresca los datos a diario, asi que re-correr este script = datos al dia.
"""

import pathlib

import requests

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
RAW = pathlib.Path("data/raw")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    fuentes = [(f"{BASE}{n}.csv", n) for n in CSVS] + [ODDS]
    for url, name in fuentes:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        destino = RAW / f"{name}.csv"
        destino.write_bytes(r.content)
        print(f"{destino} — {len(r.content) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
