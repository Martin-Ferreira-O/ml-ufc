"""Combinar modelo y mercado en logit, con el peso ajustado y no supuesto.

El problema que resuelve esta medido y documentado en el README: el modelo ciego esta
COMPRIMIDO hacia 0.5 contra el mercado. Y como

    EV = cuota * p_modelo - 1 ~= p_modelo / p_mercado - 1

esa compresion sola pone al underdog arriba de cero en TODAS las peleas. No es una senal
de valor, es la forma de la distribucion del modelo. El backtest lo muestra crudo: el
modelo promete +37% de EV y entrega -5.29% de ROI.

Calibrar no lo arregla — y esta medido que no: ni Platt ni la isotonica le ganan al crudo.
Calibrar corrige el sesgo contra la REALIDAD (que el modelo no tiene, ECE 0.0139); esto
es sesgo contra el MERCADO, que es otro problema y necesita otra herramienta.

La herramienta es el pool logaritmico de opiniones:

    logit(p) = w_mkt * logit(p_mercado) + w_mod * logit(p_modelo)

Tres propiedades que lo hacen la respuesta correcta y no una mas:

  1. **No puede quedar comprimido contra el mercado por accidente.** Los pesos se ajustan
     sobre el resultado real; si el modelo aporta cero, `w_mod -> 0` y `w_mkt -> 1`, y el
     pool ES el mercado. El piso es el mercado, no una moneda.
  2. **Conserva la antisimetria exacta.** Sin intercepto, `logit(B,A) = -logit(A,B)`, o sea
     `p(A,B) + p(B,A) = 1` sin promediar orientaciones (mismo argumento que `train._lineal`).
     Con intercepto se rompe, y ahi invertir el orden de los peleadores cambia la apuesta.
  3. **Es de dos parametros.** Ya esta medido que meterle `mkt_logit` al HistGB como
     feature numero 29 no le gana al mercado solo (+0.0014, IC95% que cruza cero). Esa
     medicion no cierra la pregunta: un GBM con 29 features tiene varianza de sobra para
     tapar una senal chica. Dos parametros es el estimador de minima varianza para la
     MISMA pregunta, asi que es el que puede contestarla.

Si `w_mod` sale ~0 con el IC cruzando cero, ese es el resultado y hay que escribirlo:
el modelo no agrega nada sobre el precio, y entonces el EV de este proyecto tiene que
salir del precio (mejor cuota, consenso multi-casa) y no del modelo. Es una conclusion
util, no un fracaso — cierra una via y deja de gastarse plata en ella.

    python -m ufc.modelo.pool     # w_mod, log loss del pool y veredicto contra mercado
"""

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression

from ufc.modelo import calibra

# Sin regularizacion efectiva: con dos features y miles de filas no hace falta, y encoger
# los coeficientes hacia 0 seria encoger hacia la moneda, que es lo contrario de lo que
# se busca (el piso deseable es el mercado).
C = 1e6


class Pool:
    """Los dos pesos ya ajustados. Escalar entra, escalar sale.

    Clase de modulo y no un closure porque termina picklada adentro de `model.pkl`, igual
    que `calibra.Calibrador`, y un lambda no se puede picklear.
    """

    def __init__(self, w_mercado, w_modelo):
        self.w_mercado, self.w_modelo = float(w_mercado), float(w_modelo)

    def __call__(self, p_modelo, p_mercado):
        z = (self.w_mercado * calibra._logit(p_mercado)
             + self.w_modelo * calibra._logit(p_modelo))
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, calibra.EPS, 1 - calibra.EPS)
        return float(p) if np.ndim(p_modelo) == 0 and np.ndim(p_mercado) == 0 else p

    def __repr__(self):
        return f"Pool(w_mercado={self.w_mercado:.3f}, w_modelo={self.w_modelo:.3f})"


IDENTIDAD_MERCADO = Pool(1.0, 0.0)   # el baseline: servir el mercado tal cual


def ajustar(p_modelo, p_mercado, y):
    """-> Pool. Logistica sin intercepto sobre las dos orientaciones espejadas.

    Espejar es obligatorio y no cosmetico: en el CSV el ganador va primero el 56% de las
    veces, asi que ajustar solo el orden del archivo mete ese desbalance en los pesos.
    Espejado, el problema es simetrico por construccion y el intercepto optimo es 0 exacto
    — por eso se puede sacar sin perder nada.
    """
    p_modelo, p_mercado = np.asarray(p_modelo, float), np.asarray(p_mercado, float)
    y = np.asarray(y, float)
    hay = np.isfinite(p_modelo) & np.isfinite(p_mercado) & np.isfinite(y)
    if hay.sum() < 50 or len(np.unique(y[hay])) < 2:
        return IDENTIDAD_MERCADO
    mod, ye = calibra._espejar(p_modelo[hay], y[hay])
    mkt, _ = calibra._espejar(p_mercado[hay], y[hay])
    X = np.column_stack([calibra._logit(mkt), calibra._logit(mod)])
    ajuste = LogisticRegression(C=C, fit_intercept=False, max_iter=2000).fit(X, ye)
    return Pool(*ajuste.coef_[0])


def prequencial(p_modelo, p_mercado, y, folds):
    """-> (p pooleada fuera de muestra, pesos por fold). Cada fold usa los ANTERIORES.

    Misma regla que `calibra.prequencial`, y por el mismo motivo: ajustar los pesos sobre
    las predicciones que van a corregir da una mejora que no existe. El primer fold no
    tiene historia y sale como el mercado, que es el baseline honesto — no crudo del
    modelo, porque el modelo sin pesos no es un competidor de nada.
    """
    p_modelo = np.asarray(p_modelo, float)
    p_mercado = np.asarray(p_mercado, float)
    y, folds = np.asarray(y, float), np.asarray(folds)
    out, pesos = np.full(len(y), np.nan), []
    for f in np.unique(folds):            # np.unique ordena: los folds son cronologicos
        actual, previo = folds == f, folds < f
        g = (ajustar(p_modelo[previo], p_mercado[previo], y[previo])
             if previo.any() else IDENTIDAD_MERCADO)
        out[actual] = g(p_modelo[actual], p_mercado[actual])
        pesos.append({"fold": int(f), "w_mercado": g.w_mercado, "w_modelo": g.w_modelo,
                      "n_train": int(previo.sum())})
    return out, pesos


def main():
    import pandas as pd
    from sklearn.metrics import log_loss

    from ufc.modelo import features, train

    if not train.FEATS.exists():
        print("Falta data/features.csv — corre `python -m ufc.modelo.features`")
        sys.exit(1)
    df = pd.read_csv(train.FEATS, parse_dates=["date"])
    corte = df["date"].max() - pd.DateOffset(years=2)

    acum, mkt, ys, eventos, folds = [], [], [], [], []
    for i, (tr, te) in enumerate(train.rolling_origin(df, corte)):
        u = te.iloc[0::2]
        acum.append(train.probas(train.entrenar(tr, features.FEATURES),
                                 te, features.FEATURES)["blend"])
        mkt.append(1 / (1 + np.exp(-u[features.MERCADO].to_numpy(float))))
        ys.append(train.objetivo(te))
        eventos.append(u["event"].to_numpy())
        folds.append(np.full(len(u), i))
        print(f"  fold {i + 1}/20  {len(u):5d} peleas hasta {u['date'].max().date()}",
              flush=True)

    p_mod = np.concatenate(acum)
    p_mkt = np.concatenate(mkt)
    y, eventos, folds = (np.concatenate(ys), np.concatenate(eventos),
                         np.concatenate(folds))

    # Solo donde HAY mercado: comparar contra el mercado en peleas sin cuota mediria
    # cobertura, no senal.
    hay = np.isfinite(p_mkt)
    p_mod, p_mkt, y = p_mod[hay], p_mkt[hay], y[hay]
    eventos, folds = eventos[hay], folds[hay]
    p_pool, pesos = prequencial(p_mod, p_mkt, y, folds)
    # el primer fold con historia manda: antes de eso el pool ES el mercado y comparar
    # ahi le regala al baseline filas identicas
    vivo = folds > folds.min()

    print(f"\n{hay.sum()} peleas con cuota, {vivo.sum()} evaluables "
          f"(el primer fold sale como mercado por falta de historia)\n")
    print("peso del modelo por fold:")
    for p in pesos[1:]:
        print(f"    [{p['fold']:2d}]  w_mercado {p['w_mercado']:+.3f}  "
              f"w_modelo {p['w_modelo']:+.3f}   (ajustado con {p['n_train']} peleas)")

    final = ajustar(p_mod, p_mkt, y)
    print(f"\najustado con todo: {final}")

    yv, cl = y[vivo], eventos[vivo]
    ll = {"modelo solo": p_mod[vivo], "mercado solo": p_mkt[vivo], "pool": p_pool[vivo]}
    print()
    for nombre, p in ll.items():
        print(f"  {nombre:14s} log loss {log_loss(yv, p):.4f}")
    print("\n  pool vs mercado : "
          f"{train.veredicto(*train.comparar(p_pool[vivo], p_mkt[vivo], yv, clusters=cl))}")
    print("  pool vs modelo  : "
          f"{train.veredicto(*train.comparar(p_pool[vivo], p_mod[vivo], yv, clusters=cl))}")

    # Lo que decide si esto entra al serving: ¿el pool le gana al mercado solo?
    delta = train.comparar(p_pool[vivo], p_mkt[vivo], yv, clusters=cl)
    if delta[2] < 0:
        print(f"\nEl pool le gana al mercado. Entra al bundle con w_modelo="
              f"{final.w_modelo:.3f}.")
    else:
        print("\nEl pool NO le gana al mercado con IC95% limpio. El modelo no agrega "
              "informacion\nsobre el precio: el EV de este proyecto tiene que salir del "
              "PRECIO (mejor cuota,\nconsenso multi-casa), no del modelo. Se sirve el "
              "mercado y se deja de buscar aca.")


if __name__ == "__main__":
    main()
