"""Entrena el clasificador con split temporal.

Split por fecha: test = ultimos 2 anios, validacion = los 2 anteriores, train = el
resto. Las dos filas espejadas de una pelea comparten fecha, asi que siempre caen
del mismo lado del split y quedan adyacentes ([0::2] / [1::2]).

Las metricas se calculan sobre peleas unicas con la prediccion desplegada
(promedio de ambas orientaciones, igual que predict.py). Sin calibracion: el
modelo crudo ya dio mejor log loss que isotonica y sigmoid (ver DECISIONS.md).
El model.pkl final se re-entrena con todo el historial; las metricas reportadas
vienen del modelo de split.
"""

import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

import features

FEATS = pathlib.Path("data/features.csv")
MODEL = pathlib.Path("model.pkl")
STATE = pathlib.Path("data/fighter_state.csv")

PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=1.0, random_state=0)


def deployed_probs(modelo, d):
    """P(gana A) por pelea unica, promediando ambas orientaciones como predict.py."""
    p = modelo.predict_proba(d[features.FEATURES])[:, 1]
    return (p[0::2] + 1 - p[1::2]) / 2


def report(nombre, p, y):
    print(f"{nombre:7s} log loss {log_loss(y, p):.4f} | Brier "
          f"{brier_score_loss(y, p):.4f} | accuracy {accuracy_score(y, p > 0.5):.4f}")


def main():
    df = pd.read_csv(FEATS, parse_dates=["date"])
    fin = df["date"].max()
    corte_test = fin - pd.DateOffset(years=2)
    corte_val = fin - pd.DateOffset(years=4)

    train = df[df["date"] < corte_val]
    val = df[(df["date"] >= corte_val) & (df["date"] < corte_test)]
    test = df[df["date"] >= corte_test]
    print(f"train {len(train) // 2} | val {len(val) // 2} | test {len(test) // 2} "
          f"peleas (test desde {corte_test.date()})")

    modelo = HistGradientBoostingClassifier(**PARAMS)
    modelo.fit(train[features.FEATURES], train["target"])

    # val decide (features/hiperparametros); test solo reporta
    report("val", deployed_probs(modelo, val), val["target"].to_numpy()[0::2])

    p, y = deployed_probs(modelo, test), test["target"].to_numpy()[0::2]
    report("test", p, y)
    print(f"moneda  log loss {log_loss(y, np.full(len(y), 0.5)):.4f} | Brier "
          f"{brier_score_loss(y, np.full(len(y), 0.5)):.4f} | accuracy 0.5000")
    elo = accuracy_score(y, test["elo"].to_numpy()[0::2] > 0)
    print(f"elo     accuracy {elo:.4f}  (gana el de mayor Elo)")

    # confiabilidad: probabilidad predicha vs frecuencia real por decil
    tabla = (pd.DataFrame({"pred": p, "real": y})
             .groupby(pd.cut(p, np.arange(0, 1.01, 0.1)), observed=True)
             .agg(pred=("pred", "mean"), real=("real", "mean"), n=("real", "size")))
    print(f"\n{tabla.round(3)}")

    # el modelo desplegado aprende de TODO el historial, no solo la ventana train
    deploy = HistGradientBoostingClassifier(**PARAMS)
    deploy.fit(df[features.FEATURES], df["target"])
    MODEL.write_bytes(pickle.dumps(deploy))
    _, estado = features.build()
    estado.to_csv(STATE, index=False)
    print(f"\n{MODEL} y {STATE} ({len(estado)} peleadores) guardados")


if __name__ == "__main__":
    main()
