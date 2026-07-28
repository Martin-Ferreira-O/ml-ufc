"""Probabilidad de victoria para cualquier matchup, desde el estado guardado.

Usa `data/fighter_state.csv` (generado por train.py con el mismo codigo de features)
para no tener skew contra el entrenamiento. Predice en las dos orientaciones y
promedia, asi el resultado es simetrico ante el intercambio de peleadores.
"""

import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

import features

MODEL = pathlib.Path("model.pkl")
STATE = pathlib.Path("data/fighter_state.csv")


def _normalizar(s):
    return " ".join(str(s).lower().split())


def cargar():
    estado = pd.read_csv(STATE, parse_dates=["dob", "last_date"])
    estado["clave"] = estado["fighter"].map(_normalizar)
    return pickle.loads(MODEL.read_bytes()), estado.set_index("clave")


def peleadores():
    return sorted(pd.read_csv(STATE)["fighter"])


def _snapshot(fila, hoy):
    """Features del peleador a fecha `hoy` (edad y descanso se recalculan)."""
    s = {k: fila[k] for k in features.FEATURES if k in fila.index}
    s["age"] = (hoy - fila["dob"]).days / 365.25 if pd.notna(fila["dob"]) else np.nan
    s["days_since_last"] = ((hoy - fila["last_date"]).days
                            if pd.notna(fila["last_date"]) else np.nan)
    return s


def predict(nombre_a, nombre_b, modelo=None, estado=None, hoy=None):
    """-> (p_a, p_b), suman 1 y son simetricas ante intercambio."""
    if modelo is None or estado is None:
        modelo, estado = cargar()
    hoy = pd.Timestamp.today().normalize() if hoy is None else pd.Timestamp(hoy)

    filas = []
    for nombre in (nombre_a, nombre_b):
        clave = _normalizar(nombre)
        if clave not in estado.index:
            raise ValueError(f"No encuentro a '{nombre}' en el dataset.")
        filas.append(_snapshot(estado.loc[clave], hoy))

    diffs = [filas[0][k] - filas[1][k] for k in features.FEATURES]
    X = pd.DataFrame([diffs, [-d for d in diffs]], columns=features.FEATURES)
    p = modelo.predict_proba(X)[:, 1]
    p_a = (p[0] + (1 - p[1])) / 2
    return float(p_a), float(1 - p_a)


def main():
    if len(sys.argv) != 3:
        print('Uso: python predict.py "Peleador A" "Peleador B"')
        sys.exit(1)
    a, b = sys.argv[1], sys.argv[2]
    try:
        p_a, p_b = predict(a, b)
    except ValueError as e:
        print(e)
        sys.exit(1)
    print(f"{a}: {p_a:.1%}")
    print(f"{b}: {p_b:.1%}")


if __name__ == "__main__":
    main()
