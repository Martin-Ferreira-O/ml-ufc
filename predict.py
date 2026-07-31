"""Probabilidad de victoria para cualquier matchup, desde el estado guardado.

Usa `data/fighter_state.csv` (generado por train.py con el mismo codigo de features)
para no tener skew contra el entrenamiento. Sirve el promedio de los dos modelos que
train.py picklea (HistGB + logistica) y predice en las dos orientaciones promediando,
asi el resultado es simetrico ante el intercambio de peleadores.

Si se le pasan las cuotas, ademas devuelve la probabilidad del mercado, la del modelo
alimentado con ella, y un nivel de confianza. La confianza sale de cuanto coinciden
modelo y mercado, que es lo unico que resulto predecir el error (ver CONFIANZA).
"""

import pathlib
import pickle
import sys
import unicodedata

import numpy as np
import pandas as pd

MODEL = pathlib.Path("model.pkl")
STATE = pathlib.Path("data/fighter_state.csv")

# Umbrales sobre |p_modelo - p_mercado|, medidos con rolling-origin sobre 5773 peleas.
# Log loss del modelo por tramo: <0.05 -> 0.6592 (igual que el mercado, 0.6606);
# 0.15-0.25 -> 0.6620 vs 0.5790; >0.25 -> 0.7416 vs 0.5402, peor que una moneda.
# Cuando eligen ganadores distintos, el mercado acierta 58.8% y el modelo 41.8%.
CONFIANZA = (
    (0.05, "alta", "El modelo y el mercado coinciden. En este tramo el modelo acierta "
                   "tanto como la casa (log loss 0.659 vs 0.661)."),
    (0.15, "media", "Hay una discrepancia moderada. Historicamente el mercado empieza a "
                    "ganarle al modelo desde aca (0.599 vs 0.639)."),
    (1.01, "baja", "Discrepan fuerte, y eso NO es valor: donde el modelo mas se aparta "
                   "rinde peor que una moneda (0.742) y la casa acierta 58.8% contra "
                   "41.8% del modelo. Es senal de que el mercado sabe algo que el "
                   "modelo no ve."),
)


# ROI de flat-bet out-of-sample (train.apostabilidad, 5773 peleas con cuota REAL, vig
# mediano 3.6%), apostando todo lado con EV = p_modelo*cuota - 1 > 0, por tramo:
#   alta      +3.66%  IC95% [-3.93%, +11.46%]  n=959   <- lo unico que no pierde medido
#   media     -7.35%  IC95% [-12.27%, -2.35%]  n=2413
#   baja      -7.34%  IC95% [-15.17%, +0.72%]  n=1300
#   muy baja  -6.48%  IC95% [-19.69%, +7.35%]  n=573
# Todo junto: -5.24%. Control siempre-favorito: -3.35%; siempre-underdog: -6.73%.
# O sea: el unico tramo apostable es donde el modelo YA coincide con la casa, y ni ahi
# el IC95% se despega de cero. Es break-even con esperanza, no una ventaja probada.
APOSTABLE = "alta"
ROI_APOSTABLE = "+3.7% IC95% [-3.9%, +11.5%] sobre 959 apuestas"


def _normalizar(s):
    """Minusculas, sin acentos: ESPN escribe 'Spasić' donde ufcstats escribe 'Spasic'.

    Verificado sobre las 2716 filas del estado: ningun par de peleadores distintos
    colapsa a la misma clave, asi que el indice sigue siendo unico.
    """
    s = unicodedata.normalize("NFKD", str(s))
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).lower().split())


def cargar():
    estado = pd.read_csv(STATE, parse_dates=["dob", "last_date"])
    estado["clave"] = estado["fighter"].map(_normalizar)
    return pickle.loads(MODEL.read_bytes()), estado.set_index("clave")


def peleadores():
    return sorted(pd.read_csv(STATE)["fighter"])


def _snapshot(fila, hoy, cols):
    """Features del peleador a fecha `hoy` (edad y descanso se recalculan)."""
    s = {k: fila[k] for k in cols if k in fila.index}
    s["age"] = (hoy - fila["dob"]).days / 365.25 if pd.notna(fila["dob"]) else np.nan
    s["days_since_last"] = ((hoy - fila["last_date"]).days
                            if pd.notna(fila["last_date"]) else np.nan)
    return s


def _prob(sub, X):
    """P(gana A) de un submodelo, promediando las dos orientaciones de X."""
    p = (sub["hgb"].predict_proba(X)[:, 1] + sub["lineal"].predict_proba(X)[:, 1]) / 2
    return float((p[0] + (1 - p[1])) / 2)


def _factores(sub, X, n=4):
    """Las n features que mas mueven el logit, con signo. Exacto: coef * valor escalado.

    Sale de la logistica del par, que es lineal: no hace falta SHAP ni aproximar nada.
    """
    lineal = sub["lineal"]
    aporte = lineal[-1].coef_[0] * lineal[:-1].transform(X)[0]
    return [(sub["cols"][i], float(aporte[i]))
            for i in np.argsort(-np.abs(aporte))[:n]]


def _mercado(cuotas):
    """(cuota_a, cuota_b) decimales -> P(gana A) sin vig."""
    pa, pb = 1 / cuotas[0], 1 / cuotas[1]
    return pa / (pa + pb)


def predict(nombre_a, nombre_b, modelo=None, estado=None, hoy=None, cuotas=None):
    """-> dict. `p_a`/`p_b` son del modelo que NO usa odds (la opinion independiente).

    Con `cuotas` = (decimal_a, decimal_b) agrega `p_a_mercado`, `p_a_con_odds`,
    `confianza` y `motivo`. Sin cuotas no hay nivel de confianza: es lo unico medido
    que predice el error, y sin cuota no se puede calcular.
    """
    if modelo is None or estado is None:
        modelo, estado = cargar()
    hoy = pd.Timestamp.today().normalize() if hoy is None else pd.Timestamp(hoy)

    base = modelo["sin_odds"]
    filas = []
    for nombre in (nombre_a, nombre_b):
        clave = _normalizar(nombre)
        if clave not in estado.index:
            raise ValueError(f"No encuentro a '{nombre}' en el dataset.")
        filas.append(_snapshot(estado.loc[clave], hoy, base["cols"]))

    diffs = [filas[0][k] - filas[1][k] for k in base["cols"]]
    X = pd.DataFrame([diffs, [-d for d in diffs]], columns=base["cols"])
    p_a = _prob(base, X)
    out = {"p_a": p_a, "p_b": 1 - p_a, "factores": _factores(base, X)}
    if cuotas is None:
        return out

    p_mkt = _mercado(cuotas)
    odds = modelo["con_odds"]
    logit = np.log(p_mkt / (1 - p_mkt))
    Xo = X.copy()
    Xo[odds["cols"][-1]] = [logit, -logit]
    out["p_a_mercado"] = p_mkt
    out["p_a_con_odds"] = _prob(odds, Xo[odds["cols"]])
    brecha = abs(p_a - p_mkt)
    out["confianza"], out["motivo"] = next(
        (nivel, texto) for corte, nivel, texto in CONFIANZA if brecha < corte)
    # Los textos de brecha grande citan el stat de cuando eligen ganadores distintos.
    # Si coinciden en el ganador eso no aplica: la discrepancia es solo de magnitud.
    if out["confianza"] != "alta" and (p_a > 0.5) == (p_mkt > 0.5):
        out["motivo"] += (" Ojo: aca coinciden en el ganador, lo que discrepa es la "
                          "magnitud (el modelo lo ve mucho mas parejo que la casa).")

    # EV con la cuota que realmente paga la casa. `apuesta` solo se marca en el tramo
    # que resulto no perder (ver ROI arriba): fuera de ahi un EV alto es ruido caro.
    out["ev_a"] = p_a * cuotas[0] - 1
    out["ev_b"] = (1 - p_a) * cuotas[1] - 1
    lado = "a" if out["ev_a"] >= out["ev_b"] else "b"
    out["apuesta"] = (lado if out["confianza"] == APOSTABLE and out[f"ev_{lado}"] > 0
                      else None)
    return out


def main():
    if len(sys.argv) not in (3, 5):
        print('Uso: python predict.py "Peleador A" "Peleador B" [cuota_a cuota_b]')
        sys.exit(1)
    a, b = sys.argv[1], sys.argv[2]
    cuotas = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) == 5 else None
    try:
        r = predict(a, b, cuotas=cuotas)
    except ValueError as e:
        print(e)
        sys.exit(1)

    print(f"modelo (sin odds)   {a}: {r['p_a']:.1%} | {b}: {r['p_b']:.1%}")
    if cuotas:
        print(f"mercado             {a}: {r['p_a_mercado']:.1%}")
        print(f"modelo (con odds)   {a}: {r['p_a_con_odds']:.1%}")
        print(f"\nconfianza: {r['confianza'].upper()} — {r['motivo']}")
    print("\nque mueve la prediccion (aporte al logit de A):")
    for nombre, aporte in r["factores"]:
        print(f"  {nombre:24s} {aporte:+.3f}")


if __name__ == "__main__":
    main()
