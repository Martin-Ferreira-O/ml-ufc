"""Entrena el clasificador con split temporal y calibracion isotonica.

Split por fecha: test = ultimos 2 anios, validacion = los 2 anteriores, train = el
resto. Las dos filas espejadas de una pelea comparten fecha, asi que siempre caen
del mismo lado del split.
"""

import pathlib
import pickle

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

import features

FEATS = pathlib.Path("data/features.csv")
MODEL = pathlib.Path("model.pkl")
STATE = pathlib.Path("data/fighter_state.csv")


def main():
    df = pd.read_csv(FEATS, parse_dates=["date"])
    fin = df["date"].max()
    corte_test = fin - pd.DateOffset(years=2)
    corte_val = fin - pd.DateOffset(years=4)

    train = df[df["date"] < corte_val]
    val = df[(df["date"] >= corte_val) & (df["date"] < corte_test)]
    test = df[df["date"] >= corte_test]
    print(f"train {len(train)} | val {len(val)} | test {len(test)} "
          f"(test desde {corte_test.date()})")

    X = lambda d: d[features.FEATURES]  # noqa: E731
    modelo = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        l2_regularization=1.0, random_state=0)
    modelo.fit(X(train), train["target"])

    calibrado = CalibratedClassifierCV(FrozenEstimator(modelo), method="isotonic")
    calibrado.fit(X(val), val["target"])

    p = calibrado.predict_proba(X(test))[:, 1]
    y = test["target"]
    print(f"\ntest  log loss {log_loss(y, p):.4f} | Brier {brier_score_loss(y, p):.4f} "
          f"| accuracy {accuracy_score(y, p > 0.5):.4f}")
    print(f"coin  log loss {log_loss(y, [0.5] * len(y)):.4f} | Brier "
          f"{brier_score_loss(y, [0.5] * len(y)):.4f} | accuracy 0.5000")
    elo = accuracy_score(y, test["elo"] > 0)
    print(f"elo   accuracy {elo:.4f}  (gana el de mayor Elo)")

    MODEL.write_bytes(pickle.dumps(calibrado))
    _, estado = features.build()
    estado.to_csv(STATE, index=False)
    print(f"\n{MODEL} y {STATE} ({len(estado)} peleadores) guardados")


if __name__ == "__main__":
    main()
