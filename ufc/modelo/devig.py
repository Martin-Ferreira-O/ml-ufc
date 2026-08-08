"""Cuota -> probabilidad justa: los cuatro metodos, y cual gana medido.

Toda decision de apuesta se apoya en un solo numero, `p_mercado`, y ese numero no sale
de los datos: sale de repartir el margen de la casa entre los dos lados. `1/cuota` suma
mas que 1 (ese exceso es el vig) y como se reparta ese exceso mueve la probabilidad uno
o dos puntos justo en el rango de favoritos, que es donde se decide el EV.

    EV = p * cuota - 1

Un punto de probabilidad en un favorito de 1.30 es 1.3 puntos de EV. O sea: el metodo
de desvigueo no es un detalle de implementacion, es una de las tres o cuatro cosas que
de verdad mueven la rentabilidad de este proyecto.

Los cuatro metodos difieren en QUE suponen sobre de donde viene el margen:

  proporcional  el margen es un impuesto parejo sobre las dos implicitas. Es el default
                de todo el mundo y esta medido que sobreestima al underdog: es el sesgo
                favorito-longshot que el README documenta.
  power         la casa eleva las probabilidades justas a una potencia. Campeon actual
                del repo (-0.0009 de log loss contra proporcional, IC95% sin cruzar cero).
  shin          supone que una fraccion `z` del volumen viene de apostadores informados
                y la casa se cubre de ellos. Ataca el mismo sesgo pero con un modelo de
                por que existe, no con una forma funcional.
  odds_ratio    (Wheeler) el margen es constante en odds-ratio, no en probabilidad.

Los tres no triviales se resuelven por biseccion sobre su parametro exigiendo que las
probabilidades sumen 1. Es dos lineas mas que la formula cerrada, no tiene casos
degenerados y es lo que ya hacia `predict._desvig`.

    python -m ufc.modelo.devig     # cual gana, global y por tramo de favorito

Ojo con lo que mide ese CLI: log loss contra el resultado real. Un metodo gana si su
probabilidad describe mejor lo que pasa, no si deja mas EV. Un devig que "encuentra"
mas valor es sospechoso por definicion — el valor tiene que venir del precio.
"""

import sys

import numpy as np

# Campeon actual. Lo cambia la medicion de `main()`, no una preferencia: entra otro solo
# si el IC95% del delta de log loss no toca cero, que es la regla de todo el repo.
CAMPEON = "power"

ITERS = 60          # biseccion: 60 pasos dejan el parametro con ~1e-18 de ancho
EPS = 1e-9


def _biseccion(f, lo, hi, iters=ITERS):
    """Ultimo parametro donde `f` sigue siendo positiva. `f` tiene que ser decreciente.

    Vectorizada: `lo`/`hi` son arrays y cada pelea converge a su propio parametro.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    for _ in range(iters):
        medio = (lo + hi) / 2
        arriba = f(medio) > 0
        lo, hi = np.where(arriba, medio, lo), np.where(arriba, hi, medio)
    return (lo + hi) / 2


def _implicitas(pa, pb):
    return np.asarray(pa, float), np.asarray(pb, float)


def proporcional(pa, pb):
    """El margen como impuesto parejo: p = pa / (pa + pb).

    Es el metodo que usa todo el mundo y el que peor le va en este dataset: reparte el
    vig en proporcion a la implicita, asi que le saca mas puntos al favorito que al
    underdog, justo al reves del sesgo que muestran los datos.
    """
    pa, pb = _implicitas(pa, pb)
    return pa / (pa + pb)


def power(pa, pb):
    """pa^k + pb^k = 1. Campeon actual del repo.

    k > 1 siempre que haya vig (las implicitas suman > 1), y elevar a k > 1 castiga mas
    a la probabilidad chica: le saca mas al underdog que al favorito.
    """
    pa, pb = _implicitas(pa, pb)
    k = _biseccion(lambda k: pa ** k + pb ** k - 1, np.ones_like(pa),
                   np.full_like(pa, 20.0))
    return pa ** k / (pa ** k + pb ** k)


def _shin_p(pi, suma, z):
    """La probabilidad de Shin para una implicita, dado z y la suma de implicitas."""
    return (np.sqrt(z ** 2 + 4 * (1 - z) * pi ** 2 / suma) - z) / (2 * (1 - z))


def shin(pa, pb):
    """Shin (1993): una fraccion `z` del dinero es de informados y la casa se cubre.

    A diferencia de power y odds_ratio, `z` tiene una lectura: es la proporcion de
    volumen informado que la casa esta asumiendo. Por eso es el unico de los cuatro que
    ataca el sesgo favorito-longshot con un modelo de POR QUE existe (la casa se protege
    de los que saben, y saber vale mas en el underdog) y no con una forma funcional.

    Se resuelve por biseccion sobre z en [0, 1). La suma de las dos probabilidades baja
    con z: en z=0 vale sqrt(suma) > 1 y en z->1 tiende a (pa^2 + pb^2)/suma < 1 para
    cualquier par de cuotas reales, asi que la raiz existe y es unica.

    El `where` del final es por las cuotas absurdas (un lado a 50.0) donde ese limite se
    acerca a 1 y la raiz se escapa del intervalo: ahi cae al proporcional en vez de
    devolver un numero sin sentido. Con las cuotas de MMA no llega a pasar, pero un
    devig que devuelve NaN en silencio borra el EV de esa pelea sin avisar.
    """
    pa, pb = _implicitas(pa, pb)
    suma = pa + pb
    sumar = lambda z: _shin_p(pa, suma, z) + _shin_p(pb, suma, z) - 1  # noqa: E731
    z = _biseccion(sumar, np.zeros_like(pa), np.full_like(pa, 1 - 1e-6))
    p = _shin_p(pa, suma, z)
    sirve = np.isfinite(p) & (np.abs(sumar(z)) < 1e-6)
    return np.where(sirve, p, proporcional(pa, pb))


def _or_p(pi, c):
    """Wheeler: p tal que odds(p) = odds(pi) / c, o sea el margen es constante en OR."""
    return pi / (c + pi - c * pi)


def odds_ratio(pa, pb):
    """Wheeler: el margen es un odds-ratio constante entre implicita y probabilidad justa.

    `c > 1` con vig. Es la version multiplicativa en odds de lo que power hace en
    probabilidad; cae entre proporcional y power en cuanto castiga al underdog.
    """
    pa, pb = _implicitas(pa, pb)
    c = _biseccion(lambda c: _or_p(pa, c) + _or_p(pb, c) - 1, np.ones_like(pa),
                   np.full_like(pa, 100.0))
    return _or_p(pa, c)


METODOS = {"proporcional": proporcional, "power": power, "shin": shin,
           "odds_ratio": odds_ratio}


def desvig(pa, pb, metodo=None):
    """(implicita_a, implicita_b) -> P(gana A) sin vig, con el metodo pedido.

    Acepta escalares o arrays. Con implicitas que ya suman 1 los cuatro metodos
    devuelven lo mismo, que es lo correcto: sin vig no hay nada que repartir.
    """
    p = METODOS[metodo or CAMPEON](pa, pb)
    return float(p) if np.ndim(pa) == 0 and np.ndim(pb) == 0 else p


def de_cuotas(cuota_a, cuota_b, metodo=None):
    """Igual que `desvig` pero desde cuotas decimales, que es como llegan de la casa."""
    return desvig(1 / np.asarray(cuota_a, float), 1 / np.asarray(cuota_b, float), metodo)


def vig(cuota_a, cuota_b):
    """El margen de la casa: cuanto suman de mas las dos implicitas."""
    return 1 / np.asarray(cuota_a, float) + 1 / np.asarray(cuota_b, float) - 1


# --------------------------------------------------------------------- medicion

def _datos():
    """-> (pa, pb, y, eventos) de `ufc_odds.csv`: implicitas crudas y quien gano.

    Se lee directo de la fuente y no de `features.csv` porque ahi la probabilidad ya
    viene desvigueada con el campeon actual, y comparar metodos contra un dato que ya
    aplico uno de ellos no compara nada.
    """
    import pandas as pd

    from ufc.modelo import features

    archivo = features.RAW / "ufc_odds.csv"
    if not archivo.exists():
        return None
    o = pd.read_csv(archivo, low_memory=False)
    # americana -> implicita. Denominador comun para no dividir por cero en -100 exacto.
    implicita = lambda x: np.where(x < 0, np.abs(x), 100.0) / (np.abs(x) + 100)  # noqa: E731
    pa = implicita(o["R_odds"].to_numpy(float))
    pb = implicita(o["B_odds"].to_numpy(float))
    # `Winner` viene como "Red"/"Blue" en esta fuente
    gano = o["Winner"].astype(str).str.strip().str.lower()
    y = np.where(gano == "red", 1.0, np.where(gano == "blue", 0.0, np.nan))
    hay = np.isfinite(pa) & np.isfinite(pb) & np.isfinite(y) & (pa + pb > 1)
    evento = (o["date"].astype(str) + "|" + o.get("location", "").astype(str)).to_numpy()
    return pa[hay], pb[hay], y[hay], evento[hay]


def main():
    from sklearn.metrics import log_loss

    from ufc.modelo import train

    datos = _datos()
    if datos is None:
        print("Falta data/raw/ufc_odds.csv — corre `python -m ufc.datos.fetch`")
        sys.exit(1)
    pa, pb, y, eventos = datos
    v = vig(1 / pa, 1 / pb)
    print(f"{len(y)} peleas con cuota y resultado, vig mediano {np.median(v):.2%}")
    print(f"campeon actual: {CAMPEON}\n")

    ps = {n: np.clip(f(pa, pb), 1e-6, 1 - 1e-6) for n, f in METODOS.items()}
    base = ps[CAMPEON]

    print(f"{'metodo':14s} {'log loss':>9s}  vs {CAMPEON}")
    for nombre, p in ps.items():
        marca = "  (campeon)" if nombre == CAMPEON else ""
        veredicto = "" if nombre == CAMPEON else \
            train.veredicto(*train.comparar(p, base, y, clusters=eventos))
        print(f"{nombre:14s} {log_loss(y, p):9.4f}  {veredicto}{marca}")

    # Por tramo de favorito: una mejora global chica puede ser grande donde se apuesta.
    # El tramo se define con el campeon para que los cuatro miren las MISMAS peleas.
    fav = np.maximum(base, 1 - base)
    print("\npor tramo de favorito (el tramo lo define el campeon, "
          "asi los cuatro miran las mismas peleas):")
    desde = 0.5
    for corte in (0.6, 0.75, 0.9, 1.01):
        t = (fav >= desde) & (fav < corte)
        if t.sum() > 50:
            print(f"  {desde:.0%}-{corte:.0%}  n={t.sum():5d}  " + "  ".join(
                f"{n} {log_loss(y[t], p[t], labels=[0, 1]):.4f}" for n, p in ps.items()))
        desde = corte

    mejor = min(ps, key=lambda n: log_loss(y, ps[n]))
    if mejor != CAMPEON:
        delta = train.comparar(ps[mejor], base, y, clusters=eventos)
        print(f"\n{mejor} tiene el log loss mas bajo: {train.veredicto(*delta)}")
        print(f"Cambiar CAMPEON a '{mejor}'" if delta[2] < 0 else
              f"El IC95% toca cero: se queda {CAMPEON}. Un log loss mas bajo sin IC "
              "limpio es ruido.")
    else:
        print(f"\nSigue ganando {CAMPEON}: ningun challenger le saca ventaja medible.")


if __name__ == "__main__":
    main()
