"""Calibracion de probabilidades: que un 70% del modelo sea un 70% real.

El modelo sirve para decidir apuestas solo si sus probabilidades estan calibradas: el
EV es `p * cuota - 1`, asi que un p sesgado se traduce directo en EV inventado. El
modelo ciego de este proyecto esta comprimido hacia 0.5 contra el mercado, y como
`EV = cuota*p - 1 ~= p_modelo/p_mercado - 1`, esa compresion sola pone al underdog
arriba en TODAS las peleas. Calibrar es lo que separa "el modelo ve valor" de "el
modelo es timido".

Dos reglas que no se pueden romper:

  1. El calibrador se ajusta SOLO con el train de cada fold de `train.rolling_origin`
     y se aplica al test. Ajustarlo sobre las predicciones que va a corregir da un ECE
     falso ~0: la curva de fiabilidad de un isotonico visto sobre sus propios datos es
     perfecta por construccion.
  2. La prediccion del proyecto es antisimetrica (`p(A,B) + p(B,A) = 1`, con assert en
     train.py). La isotonica lo rompe, asi que se ajusta sobre las dos orientaciones
     concatenadas y se aplica simetrizando. Sin eso, invertir el orden de los
     peleadores cambia la apuesta.
"""

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _espejar(p, y):
    """(p, y) -> las dos orientaciones concatenadas: [p, 1-p] y [y, 1-y]."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y])


EPS = 1e-6


class Calibrador:
    """Aplica el ajuste simetrizando y recortando. Escalar entra, escalar sale.

    Es una clase de modulo y no un closure porque el ganador se guarda en `model.pkl`, y
    un lambda no se puede picklear.

    Simetriza porque la prediccion del proyecto es antisimetrica (`p(A,B) + p(B,A) = 1`,
    con assert en train). La isotonica lo rompe, y sin esto invertir el orden de los
    peleadores cambiaria la apuesta. Recorta a [EPS, 1-EPS] porque la isotonica devuelve
    0 y 1 exactos en los extremos: un 0 exacto es log loss infinito y, peor, un "esta
    pelea es imposible" que ninguna evidencia sostiene.
    """

    def __init__(self, tipo, modelo=None):
        self.tipo, self.modelo = tipo, modelo

    def _crudo(self, q):
        if self.tipo == "identidad":
            return q
        if self.tipo == "platt":
            return self.modelo.predict_proba(_logit(q).reshape(-1, 1))[:, 1]
        return self.modelo.predict(q)

    def __call__(self, q):
        arr = np.atleast_1d(np.asarray(q, float))
        out = np.clip((self._crudo(arr) + 1 - self._crudo(1 - arr)) / 2, EPS, 1 - EPS)
        return out if np.ndim(q) else float(out[0])


def identidad(p, y):
    """Baseline. Existe para poder concluir 'calibrar no mejoro' con un numero."""
    return Calibrador("identidad")


def platt(p, y):
    """Logistica sobre el logit: corrige compresion/expansion global, un parametro util.

    C alto = casi sin regularizacion: con una sola feature y miles de filas no hace falta,
    y encoger el coeficiente hacia 0 seria encoger hacia la moneda.
    """
    pe, ye = _espejar(p, y)
    return Calibrador("platt",
                      LogisticRegression(C=1e6).fit(_logit(pe).reshape(-1, 1), ye))


def isotonica(p, y):
    """Monotona no parametrica: corrige cualquier forma, a costa de mas varianza.

    `out_of_bounds="clip"` porque el test puede traer probabilidades fuera del rango
    visto en train; sin eso devuelve NaN y el EV de esa pelea desaparece en silencio.
    """
    pe, ye = _espejar(p, y)
    return Calibrador("isotonica",
                      IsotonicRegression(out_of_bounds="clip").fit(pe, ye))


CALIBRADORES = {"identidad": identidad, "platt": platt, "isotonica": isotonica}


def prequencial(p, y, folds):
    """-> {nombre: array calibrado}. Cada fold se calibra con los folds ANTERIORES.

    Es la unica forma honesta de calibrar una serie temporal sin pagar un entrenamiento
    extra por fold: las predicciones de los folds previos ya son fuera de muestra y solo
    miran el pasado. El primer fold no tiene historial y sale crudo — al comparar
    calibradores hay que excluirlo, si no le regala al baseline filas identicas.
    """
    p, y, folds = np.asarray(p, float), np.asarray(y, float), np.asarray(folds)
    out = {n: np.empty(len(p)) for n in CALIBRADORES}
    for f in np.unique(folds):          # np.unique ordena: los folds son cronologicos
        actual, previo = folds == f, folds < f
        for nombre, fabrica in CALIBRADORES.items():
            g = fabrica(p[previo], y[previo]) if previo.any() else identidad(None, None)
            out[nombre][actual] = g(p[actual])
    return out


def fiabilidad(p, y, bins=10, espejar=True):
    """-> DataFrame (pred, real, n) por bin. Es el reliability diagram.

    Por defecto puntua las dos orientaciones: en el CSV el ganador va primero el 56% de
    las veces, asi que mirar solo el orden del archivo le suma ~0.09 a cada bin y simula
    un sesgo hacia arriba que no existe (mismo motivo que la tabla de deciles de train).
    """
    if espejar:
        p, y = _espejar(p, y)
    p, y = np.asarray(p, float), np.asarray(y, float)
    corte = pd.cut(p, np.linspace(0, 1, bins + 1), include_lowest=True)
    return (pd.DataFrame({"p": p, "y": y})
            .groupby(corte, observed=True)
            .agg(pred=("p", "mean"), real=("y", "mean"), n=("y", "size"))
            .reset_index(drop=True))


def ece(p, y, bins=10, espejar=True):
    """Expected Calibration Error: |pred - real| por bin, pesado por cuantos caen ahi."""
    t = fiabilidad(p, y, bins, espejar)
    return float((t["n"] * (t["pred"] - t["real"]).abs()).sum() / t["n"].sum())
