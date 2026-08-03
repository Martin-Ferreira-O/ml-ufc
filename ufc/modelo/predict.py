"""Probabilidad de victoria para cualquier matchup, desde el estado guardado.

Usa `data/fighter_state.csv` (generado por train.py con el mismo codigo de features)
para no tener skew contra el entrenamiento. Sirve el promedio de los dos modelos que
train.py picklea (HistGB + logistica) y predice en las dos orientaciones promediando,
asi el resultado es simetrico ante el intercambio de peleadores.

Si se le pasan las cuotas, ademas devuelve la probabilidad del mercado, la del modelo
alimentado con ella, y un nivel de confianza. La confianza sale de cuanto coinciden
modelo y mercado, que es lo unico que resulto predecir el error (ver CONFIANZA).
"""

import pickle
import sys

import numpy as np
import pandas as pd

from ufc import nombres, rutas

MODEL = rutas.MODELO
STATE = rutas.DATOS / "fighter_state.csv"

# Umbrales sobre |p_modelo - p_mercado|, medidos con rolling-origin sobre 5773 peleas
# (mercado desvigueado con power, 2026-07-31). Log loss del modelo por tramo:
# <0.05 -> 0.6487 (igual que el mercado, 0.6503); 0.15-0.25 -> 0.6540 vs 0.5695;
# >0.25 -> 0.7328 vs 0.5674, peor que una moneda. Cuando eligen ganadores distintos,
# el mercado acierta 57.0% y el modelo 43.0%.
# Sale de `train.calidad_por_tramo`, que lo imprime en cada corrida: son numeros que
# cambian con el modelo, y una etiqueta de confianza vieja miente con autoridad.
CONFIANZA = ((0.05, "alta"), (0.15, "media"), (1.01, "baja"))


# ROI de flat-bet out-of-sample (train.apostabilidad, 5773 peleas con cuota REAL, vig
# mediano 3.6%), apostando todo lado con EV = p_modelo*cuota - 1 > 0, por tramo:
#   alta      +1.87%  IC95% [-5.56%, +9.40%]   n=863   <- lo unico que no pierde medido
#   media     -5.22%  IC95% [-10.50%, -0.19%]  n=2417
#   baja     -10.83%  IC95% [-18.19%, -3.17%]  n=1358
#   muy baja  +2.74%  IC95% [-11.07%, +16.33%] n=605
# Todo junto: -4.59%. Control siempre-favorito: -3.35%; siempre-underdog: -6.73%.
# O sea: el unico tramo apostable es donde el modelo YA coincide con la casa, y ni ahi
# el IC95% se despega de cero. Es break-even con esperanza, no una ventaja probada —
# y encima es el mejor de 4 tramos elegido a posteriori, o sea que la evidencia real
# es mas debil todavia. El Historial de la app (CLV) es el forward test que decide.
# El mejor tramo fue elegido a posteriori y su IC95% cruza cero. Se conserva el EV como
# diagnostico, pero ninguna version actual esta autorizada a recomendar una apuesta.
APUESTAS_AUTOMATICAS = False


# Circunstancias de la pelea que el modelo usa pero el estado del peleador no tiene.
# El orden es el de `circ_a`/`circ_b` en `predict`.
CIRCUNSTANCIA = ("reemplazo", "peso_no_dado")


def cargar():
    estado = pd.read_csv(STATE, parse_dates=["dob", "last_date"])
    estado["clave"] = estado["fighter"].map(nombres.normalizar)
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


def _desvig(pa, pb, iters=60):
    """Probabilidades implicitas con vig -> justas, metodo power (k: pa^k + pb^k = 1).

    El proporcional reparte el vig parejo y sobreestima al underdog — el sesgo
    favorito-longshot que el README documenta. Medido sobre 6916 peleas con cuota:
    power gana -0.0009 de log loss (IC95% [-0.0015, -0.0003]), y -0.0048 en los
    favoritos de >75%, que es donde se juega el EV.
    """
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    lo, hi = np.ones_like(pa), np.full_like(pa, 20.0)
    for _ in range(iters):
        k = (lo + hi) / 2
        arriba = pa ** k + pb ** k > 1
        lo, hi = np.where(arriba, k, lo), np.where(arriba, hi, k)
    k = (lo + hi) / 2
    return pa ** k / (pa ** k + pb ** k)


def _mercado(cuotas):
    """(cuota_a, cuota_b) decimales -> P(gana A) sin vig."""
    return float(_desvig(1 / cuotas[0], 1 / cuotas[1]))


def _motivo(modelo, brecha, nivel):
    buckets = modelo.get("manifest", {}).get("metrics", {}).get(
        "coincidence_buckets", [])
    fila = next((x for x in buckets
                 if float(x["from"]) <= brecha < float(x["to"])), None)
    prefijo = {
        "alta": "Modelo y mercado están muy cerca.",
        "media": "Hay una discrepancia moderada.",
        "baja": "La discrepancia es fuerte y no equivale a valor.",
    }[nivel]
    if fila is None:
        return prefijo + " No hay métricas compatibles serializadas para esta versión."
    return (f"{prefijo} En la validación histórica tardía de esta versión, el log loss "
            f"fue {fila['model_log_loss']:.3f} para el modelo y "
            f"{fila['market_log_loss']:.3f} para el mercado (n={fila['n']}).")


def metodo(modelo, wc_lbs=np.nan, mujer=np.nan, cinco_r=np.nan):
    """-> {"ko", "sub", "dec"}: como suele terminar una pelea de ese contexto.

    No pide peleadores a proposito: esta medido (ver train.entrenar_metodo) que el
    historial de los dos no agrega nada sobre la division. Con NaN predice el base
    rate global, asi que se puede llamar sin saber el contexto.
    """
    sub = modelo.get("metodo")
    if sub is None:
        return None  # model.pkl anterior a este modelo
    X = pd.DataFrame([[wc_lbs, mujer, cinco_r]], columns=sub["cols"])
    p = sub["hgb"].predict_proba(X)[0]
    return dict(zip(sub["hgb"].classes_, (float(x) for x in p)))


def predict(nombre_a, nombre_b, modelo=None, estado=None, event_date=None, cuotas=None,
            circ_a=(np.nan, np.nan), circ_b=(np.nan, np.nan), hoy=None):
    """-> dict. `p_a`/`p_b` son del modelo que NO usa odds (la opinion independiente).

    Con `cuotas` = (decimal_a, decimal_b) agrega `p_a_mercado`, `p_a_con_odds`,
    `confianza` y `motivo`. Sin cuotas no hay nivel de confianza: es lo unico medido
    que predice el error, y sin cuota no se puede calcular.

    `event_date` fija la fecha programada usada para edad y descanso. `hoy` queda como
    alias retrocompatible. `circ_a`/`circ_b` = (reemplazo, peso_no_dado), con 1=si,
    0=no y NaN=desconocido. Desconocido es el default: ausencia de verificacion no se
    convierte en un "no" inventado.
    """
    if modelo is None or estado is None:
        modelo, estado = cargar()
    fecha = event_date if event_date is not None else hoy
    fecha = pd.Timestamp.today().normalize() if fecha is None else pd.Timestamp(fecha)

    base = modelo["sin_odds"]
    filas, mezclados = [], []
    for nombre, circ in ((nombre_a, circ_a), (nombre_b, circ_b)):
        clave = nombres.normalizar(nombre)
        if clave not in estado.index:
            raise ValueError(f"No encuentro a '{nombre}' en el dataset.")
        fila = estado.loc[clave]
        if fila.get("homonimo"):
            mezclados.append(nombre)
        filas.append({**_snapshot(fila, fecha, base["cols"]),
                      **dict(zip(CIRCUNSTANCIA, (float(x) for x in circ)))})

    diffs = [filas[0][k] - filas[1][k] for k in base["cols"]]
    X = pd.DataFrame([diffs, [-d for d in diffs]], columns=base["cols"])
    p_a = _prob(base, X)
    out = {"p_a": p_a, "p_b": 1 - p_a, "factores": _factores(base, X)}
    if mezclados:
        out["aviso"] = (
            "Hubo más de un peleador llamado " + " y ".join(mezclados) + " en UFC, y "
            "el dataset no los distingue: este historial mezcla a los dos, así que la "
            "predicción no es confiable.")
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
    out["confianza"] = next(nivel for corte, nivel in CONFIANZA if brecha < corte)
    out["motivo"] = _motivo(modelo, brecha, out["confianza"])
    # Los textos de brecha grande citan el stat de cuando eligen ganadores distintos.
    # Si coinciden en el ganador eso no aplica: la discrepancia es solo de magnitud.
    if out["confianza"] != "alta" and (p_a > 0.5) == (p_mkt > 0.5):
        out["motivo"] += (" Ojo: aca coinciden en el ganador, lo que discrepa es la "
                          "magnitud (el modelo lo ve mucho mas parejo que la casa).")

    # EV puntual para diagnostico. No habilita una recomendacion: el mejor tramo fue
    # elegido a posteriori, su intervalo cruza cero y las cuotas vivas no son el mismo
    # dominio que las lineas historicas usadas para medirlo.
    out["ev_a"] = p_a * cuotas[0] - 1
    out["ev_b"] = (1 - p_a) * cuotas[1] - 1
    lado = "a" if out["ev_a"] >= out["ev_b"] else "b"
    out["apuesta"] = None
    out["seguimiento"] = lado if out[f"ev_{lado}"] > 0 else None
    out["estado_apuesta"] = "sin apuesta — señal no validada prospectivamente"
    return out


def main():
    if len(sys.argv) not in (3, 5):
        print('Uso: python -m ufc.modelo.predict "Peleador A" "Peleador B" [cuota_a cuota_b]')
        sys.exit(1)
    a, b = sys.argv[1], sys.argv[2]
    cuotas = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) == 5 else None
    try:
        r = predict(a, b, cuotas=cuotas)
    except ValueError as e:
        print(e)
        sys.exit(1)

    if "aviso" in r:
        print(f"OJO: {r['aviso']}\n")
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
