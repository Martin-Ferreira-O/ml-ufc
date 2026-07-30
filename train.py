"""Entrena el clasificador con split temporal y protocolo de medicion pareado.

Split por fecha: test = ultimos 2 anios, validacion = los 2 anteriores, train = el
resto. Las dos filas espejadas de una pelea comparten fecha, asi que siempre caen
del mismo lado del split y quedan adyacentes ([0::2] / [1::2]).

Se sirve el promedio de dos modelos sobre las mismas features:
  - HistGradientBoosting, que captura no-linealidades y maneja NaN nativo;
  - una logistica sin intercepto, que sobre features antisimetricas cumple
    p(A,B) + p(B,A) = 1 exacto (el HistGB crudo lo viola hasta en 0.205) y ademas
    explica cada prediccion con coef*x, sin SHAP.
Medido: la logistica sola le gana al HistGB tuneado y el promedio le gana a las dos.
El problema tiene techo de informacion, no de capacidad — la curva train/val es
plana de 50 a 800 arboles mientras el train loss cae 0.17. No tocar los
hiperparametros del HistGB: ese pozo ya se seco (ver DECISIONS.md).

Las decisiones NO se toman con el delta crudo de una ventana: el SE del delta
pareado sobre las 999 peleas de val es 0.0028, o sea que el umbral viejo de 0.003
era ruido. `rolling_origin` + `comparar` dan el delta sobre ~4500 peleas con IC95%.

El model.pkl final se re-entrena con todo el historial; las metricas reportadas
vienen del modelo de split.
"""

import pathlib
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features

FEATS = pathlib.Path("data/features.csv")
MODEL = pathlib.Path("model.pkl")
STATE = pathlib.Path("data/fighter_state.csv")

# early_stopping=False explicito: con 'auto' sklearn lo activa (train > 10000 filas) y
# separa un 10% ALEATORIO. Como cada pelea esta dos veces (espejada), el gemelo de cada
# fila de la validacion interna queda en train y el criterio de parada se evalua sobre
# filas ya vistas. Con max_iter fijo, 100 es el mejor medido con rolling_origin: el
# HistGB solo se degrada 0.6632 -> 0.6716 entre 100 y 300 iteraciones, pero el blend
# apenas se mueve (0.6617 -> 0.6630) porque la logistica lo estabiliza.
PARAMS = dict(max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
              l2_regularization=1.0, random_state=0, early_stopping=False)
C_LINEAL = 0.01  # la curva de C es plana entre 0.003 y 1.0; cualquiera sirve
MODELOS = ("hgb", "lineal", "blend")


def _lineal():
    """Logistica antisimetrica por construccion.

    NaN -> 0 es el relleno natural de un diff (empate), y sobrevive al espejado.
    Sin intercepto y sin centrar, logit(B,A) = -logit(A,B) exacto, o sea
    p(A,B) + p(B,A) = 1 sin necesidad de promediar orientaciones.
    Sin indicador de faltante: sobre un dataset espejado su peso optimo es 0 exacto
    (la loss es invariante ante cambiarle el signo), asi que solo agregaria columnas.
    """
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0.0),
        StandardScaler(with_mean=False),
        LogisticRegression(C=C_LINEAL, fit_intercept=False, max_iter=2000))


def entrenar(d, cols):
    """-> (hgb, lineal) ajustados sobre las filas espejadas de `d`."""
    hgb = HistGradientBoostingClassifier(**PARAMS).fit(d[cols], d["target"])
    return hgb, _lineal().fit(d[cols], d["target"])


def probas(modelos, d, cols):
    """-> {nombre: P(gana A)} por pelea unica, promediando ambas orientaciones."""
    hgb, lineal = modelos
    desplegar = lambda p: (p[0::2] + 1 - p[1::2]) / 2  # noqa: E731
    p_h = desplegar(hgb.predict_proba(d[cols])[:, 1])
    p_l = desplegar(lineal.predict_proba(d[cols])[:, 1])
    return {"hgb": p_h, "lineal": p_l, "blend": (p_h + p_l) / 2}


def objetivo(d):
    """Target por pelea unica (las filas espejadas son adyacentes)."""
    return d["target"].to_numpy()[0::2]


def rolling_origin(df, corte, n_folds=20):
    """(train, test) por anio: cada fold entrena con TODO lo anterior a su ventana.

    Evaluar sobre ~7200 peleas en vez de las 999 de val baja el SE del delta pareado
    de 0.0028 a ~0.0011, que es lo unico que hace medibles las features nuevas: con
    10 folds el blend salia "no concluyente" y con 20 el IC95% ya no toca cero.
    """
    for i in range(n_folds, 0, -1):
        ini, fin = corte - pd.DateOffset(years=i), corte - pd.DateOffset(years=i - 1)
        tr = df[df["date"] < ini]
        te = df[(df["date"] >= ini) & (df["date"] < fin)]
        assert tr["date"].max() < te["date"].min(), (
            f"leakage: el fold que evalua desde {ini.date()} entrena con el futuro")
        yield tr, te


def _perdida(y, p):
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def comparar(p_nuevo, p_viejo, y, n_boot=2000, seed=0):
    """-> (delta, lo, hi) del log loss pareado. Negativo = el nuevo es mejor.

    Pareado sobre las mismas peleas: los errores de dos modelos parecidos estan muy
    correlacionados, asi que la diferencia tiene mucho menos ruido que cada log loss
    por separado. Regla de corte: el cambio queda solo si el IC95% no toca 0.
    """
    d = _perdida(y, p_nuevo) - _perdida(y, p_viejo)
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def _veredicto(p_nuevo, p_viejo, y):
    delta, lo, hi = comparar(p_nuevo, p_viejo, y)
    corte = "queda" if hi < 0 else "se descarta" if lo > 0 else "no concluyente"
    return f"{delta:+.4f}  IC95% [{lo:+.4f}, {hi:+.4f}]  {corte}"


def main():
    df = pd.read_csv(FEATS, parse_dates=["date"])
    cols = features.FEATURES
    fin = df["date"].max()
    corte_test = fin - pd.DateOffset(years=2)
    corte_val = fin - pd.DateOffset(years=4)
    partes = {"train": df[df["date"] < corte_val],
              "val": df[(df["date"] >= corte_val) & (df["date"] < corte_test)],
              "test": df[df["date"] >= corte_test]}
    print(" | ".join(f"{k} {len(d) // 2}" for k, d in partes.items())
          + f" peleas (test desde {corte_test.date()})")

    modelos = entrenar(partes["train"], cols)
    p = {k: probas(modelos, d, cols) for k, d in partes.items()}

    # el train loss es lo unico que distingue overfitting de techo de informacion
    print(f"\nlog loss  {'train':>7s} {'val':>7s} {'test':>7s}")
    for m in MODELOS:
        print(f"{m:9s} " + " ".join(
            f"{log_loss(objetivo(partes[k]), p[k][m]):7.4f}" for k in partes))

    y = objetivo(partes["test"])
    print(f"\ntest (blend)  Brier {brier_score_loss(y, p['test']['blend']):.4f} | "
          f"accuracy {accuracy_score(y, p['test']['blend'] > 0.5):.4f}")
    print(f"moneda        Brier {brier_score_loss(y, np.full(len(y), 0.5)):.4f} | "
          f"accuracy 0.5000 | log loss {log_loss(y, np.full(len(y), 0.5)):.4f}")
    print("mayor Elo     accuracy "
          f"{accuracy_score(y, partes['test']['elo'].to_numpy()[0::2] > 0):.4f}")

    # Confiabilidad por decil sobre AMBAS orientaciones. La prediccion desplegada es
    # simetrica, asi que el log loss no cambia, pero puntuar la tabla solo en el orden
    # del CSV (donde el ganador va primero el 56% de las veces) le suma ~0.09 a cada
    # bucket y simula un sesgo hacia arriba que no existe.
    pb = np.concatenate([p["test"]["blend"], 1 - p["test"]["blend"]])
    yb = np.concatenate([y, 1 - y])
    tabla = (pd.DataFrame({"pred": pb, "real": yb})
             .groupby(pd.cut(pb, np.arange(0, 1.01, 0.1)), observed=True)
             .agg(pred=("pred", "mean"), real=("real", "mean"), n=("real", "size")))
    print(f"\n{tabla.round(3)}")

    # --- decision: rolling-origin sobre 20 anios, no una sola ventana
    con_odds = features.FEATURES_ODDS
    acum = {m: [] for m in (*MODELOS, "con odds", "mercado")}
    ys = []
    for tr, te in rolling_origin(df, corte_test):
        ps = probas(entrenar(tr, cols), te, cols)
        for m in MODELOS:
            acum[m].append(ps[m])
        # el modelo con odds solo existe desde que hay odds (2010). Antes, la columna
        # entera es NaN y el binner de HistGB no puede ni construir los bins; esos folds
        # tampoco tienen cuota para evaluar, asi que quedan fuera de la comparacion.
        acum["con odds"].append(
            probas(entrenar(tr, con_odds), te, con_odds)["blend"]
            if tr[features.MERCADO].notna().any() else np.full(len(te) // 2, np.nan))
        acum["mercado"].append(1 / (1 + np.exp(-te[features.MERCADO].to_numpy()[0::2])))
        ys.append(objetivo(te))
    acum = {m: np.concatenate(v) for m, v in acum.items()}
    ys = np.concatenate(ys)

    print(f"\nrolling-origin: 20 folds de 1 anio, {len(ys)} peleas")
    for m in MODELOS:
        print(f"  {m:8s} log loss {log_loss(ys, acum[m]):.4f}")
    for m in ("lineal", "blend"):
        print(f"  {m:8s} vs hgb: {_veredicto(acum[m], acum['hgb'], ys)}")

    # --- cuanto agrega el mercado, y cuanto agrega el modelo por encima del mercado
    hay = np.isfinite(acum["mercado"]) & np.isfinite(acum["con odds"])
    print(f"\nsolo las {hay.sum()} peleas con cuota ({hay.mean():.0%} del rolling-origin):")
    for m in ("blend", "mercado", "con odds"):
        etiqueta = "sin odds" if m == "blend" else m
        print(f"  {etiqueta:8s} log loss {log_loss(ys[hay], acum[m][hay]):.4f}")
    for m in ("mercado", "con odds"):
        print(f"  {m:8s} vs sin odds: "
              f"{_veredicto(acum[m][hay], acum['blend'][hay], ys[hay])}")
    # la comparacion que decide si el modelo aporta algo que el mercado no tenga ya
    print("  con odds vs mercado : "
          f"{_veredicto(acum['con odds'][hay], acum['mercado'][hay], ys[hay])}")

    # --- deploy: aprenden de TODO el historial, no solo de la ventana train
    bundle = {}
    for nombre, c in (("sin_odds", cols), ("con_odds", con_odds)):
        hgb, lineal = entrenar(df, c)
        p_ida = lineal.predict_proba(df[c].head(2))[:, 1]
        assert abs(p_ida[0] + p_ida[1] - 1) < 1e-9, f"{nombre}: logistica no antisimetrica"
        bundle[nombre] = {"hgb": hgb, "lineal": lineal, "cols": list(c)}
    MODEL.write_bytes(pickle.dumps(bundle))
    _, estado = features.build()
    estado.to_csv(STATE, index=False)
    print(f"\n{MODEL} y {STATE} ({len(estado)} peleadores) guardados")


if __name__ == "__main__":
    main()
