"""De una probabilidad a un monto: EV, Kelly, ruina y cuanta muestra hace falta.

Este modulo no estima nada y no lee nada. Recibe una probabilidad y una cuota y contesta
las cuatro preguntas que separan "tener razon" de "ganar plata":

  1. ¿hay ventaja?           `ev`, `break_even`, `p_conservadora`
  2. ¿cuanto se apuesta?     `kelly`, `stake`, `crecimiento`
  3. ¿cuanto se puede perder? `riesgo_de_ruina`, `exposicion`
  4. ¿cuando se sabe?        `sigma_apuesta`, `n_para_detectar`

La cuarta es la que cambia el proyecto. Con cuota 2.00 la desviacion de una apuesta es
~1.0 unidad, asi que detectar una ventaja del 2% con 80% de potencia pide

    n = ((1.96 + 0.8416) * 1.0 / 0.02)^2 ~= 19.600 apuestas

que a ~500 apuestas por anio son cuarenta anios. **El ROI no puede ser el criterio de
decision de este proyecto, nunca.** Por eso el gate se juega en el CLV, que tiene una
decima parte de la varianza y converge en decenas de apuestas: no porque sea mas lindo,
sino porque es el unico estadistico que alcanza a converger en una vida humana.

La otra cuenta que hay que tener a la vista: Kelly no crea ventaja, la convierte en
crecimiento. Apostar flat una ventaja real deja crecimiento sobre la mesa; apostar de
mas la destruye. Y como `p` es estimada y no conocida, el optimo esta ESTRICTAMENTE
abajo de `f*(p_estimada)` (Baker y McHale, 2013), de ahi que todo lo de aca decida sobre
`p_conservadora` y con una fraccion <= 1/2.
"""

import numpy as np

# Defaults de staking. `config/gate.json` los pisa: aca viven para que el modulo se pueda
# usar suelto (y testear) sin leer un archivo.
FRACCION = 0.25          # cuarto de Kelly
TOPE_APUESTA = 0.01      # 1% de la banca por apuesta, tope duro
TOPE_EVENTO = 0.03       # 3% de la banca comprometida en una misma cartelera

# Cuantiles normales, para no arrastrar scipy por dos numeros. Son los unicos que se usan.
Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263,
     0.995: 2.5758}


def _z(p):
    """Cuantil normal para una cola. Solo los valores tabulados: pedir otro es un error."""
    if p not in Z:
        raise ValueError(f"Cuantil no tabulado: {p}. Disponibles: {sorted(Z)}")
    return Z[p]


# ------------------------------------------------------------------- ¿hay ventaja?

def break_even(q):
    """La probabilidad que hay que superar para que la apuesta no pierda: 1/cuota."""
    return 1 / np.asarray(q, float)


def ev(p, q):
    """Retorno esperado por unidad apostada: p*q - 1.

    Con la cuota REAL de la casa adentro, o sea con el vig ya pagado. Un EV de 0 no es
    una apuesta neutra: es una apuesta que ya perdio el margen y lo recupero exacto.
    """
    return np.asarray(p, float) * np.asarray(q, float) - 1


def p_conservadora(p, sigma, potencia=0.95):
    """Cota inferior de la probabilidad: `p - z*sigma`, recortada a (0, 1).

    Es la correccion mas importante de todo el modulo y la mas barata. `EV = p*q - 1` es
    lineal en `p`, asi que un error de estimacion se traduce uno a uno en EV inventado —
    y el error no es simetrico en sus consecuencias: sobreestimar `p` hace apostar de mas
    en algo que no tiene ventaja, subestimarla solo hace no apostar.

    Va en puntos de probabilidad y no en logit a proposito: `sigma` sale de la tabla de
    fiabilidad fuera de muestra (|predicho - real| por bin) y de la dispersion entre
    casas, que son dos cosas que se miden en puntos, y el numero tiene que poder
    compararse de frente contra `1/cuota`.
    """
    p, sigma = np.asarray(p, float), np.asarray(sigma, float)
    return np.clip(p - _z(potencia) * sigma, 1e-6, 1 - 1e-6)


# ------------------------------------------------------------------ ¿cuanto se apuesta?

def kelly(p, q):
    """Fraccion de banca que maximiza el crecimiento logaritmico: (p*q - 1)/(q - 1).

    Es EV dividido por lo que se gana neto si acierta. Recortada en 0: una fraccion
    negativa querria decir "apostar al otro lado", y el otro lado ya es su propia
    apuesta con su propia cuota, no la inversa de esta.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (p * q - 1) / (q - 1)
    return np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def crecimiento(p, q, f):
    """Crecimiento logaritmico esperado por apuesta: lo que Kelly maximiza de verdad.

    G(f) = p*ln(1 + f*(q-1)) + (1-p)*ln(1 - f)

    No es el EV. El EV es lineal en `f` y por lo tanto siempre recomienda apostar todo;
    el crecimiento es concavo y tiene maximo interior. Esa diferencia es toda la teoria
    del bankroll: maximizar EV por apuesta quiebra con probabilidad 1.
    """
    p, q, f = np.asarray(p, float), np.asarray(q, float), np.asarray(f, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = p * np.log1p(f * (q - 1)) + (1 - p) * np.log1p(-f)
    return np.where(f >= 1, -np.inf, g)


def crecimiento_relativo(fraccion):
    """Que proporcion del crecimiento optimo conserva Kelly fraccional: c*(2-c).

    Medio Kelly conserva el 75% del crecimiento con la mitad de la volatilidad, y un
    cuarto conserva el 44% con un cuarto de la volatilidad. Es el argumento entero a
    favor de apostar fraccionado: la curva es plana cerca del optimo y el castigo por
    quedarse corto es de segundo orden, mientras que el castigo por pasarse no lo es.

    Es la aproximacion cuadratica (ventajas chicas). Contra `crecimiento` exacto con
    p=0.60 y cuota 2.00 da 0.438 contra 0.435 real: sirve para razonar, no para reportar.
    """
    c = np.asarray(fraccion, float)
    return c * (2 - c)


def stake(p, q, banca, fraccion=FRACCION, tope=TOPE_APUESTA, sigma=None,
          potencia=0.95):
    """-> dict con la fraccion, el monto y QUE restriccion mando.

    Decide sobre `p_conservadora(p, sigma)` si le pasan `sigma`; sin `sigma` decide sobre
    `p` crudo y lo deja anotado en `sobre`, porque no es lo mismo y quien lea el dict
    tiene que poder distinguirlo.

    `limita` es el punto: sin eso, una apuesta topeada y una apuesta chica se ven igual
    en la UI, y son cosas distintas — una dice "el modelo ve poco", la otra dice "el
    modelo ve mucho y no le creemos".
    """
    p_decide = p if sigma is None else float(p_conservadora(p, sigma, potencia))
    f_kelly = float(kelly(p_decide, q))
    f_frac = f_kelly * fraccion
    f = min(f_frac, tope)
    limita = ("sin ventaja" if f_kelly <= 0 else
              "tope por apuesta" if f_frac > tope else "kelly fraccional")
    return {
        "p": float(p), "p_decide": float(p_decide),
        "sobre": "cota inferior" if sigma is not None else "probabilidad puntual",
        "cuota": float(q), "ev": float(ev(p_decide, q)),
        "kelly": f_kelly, "fraccion": f, "monto": float(banca) * f,
        "limita": limita,
        "crecimiento": float(crecimiento(p_decide, q, f)),
    }


# --------------------------------------------------------------- ¿cuanto se puede perder?

def riesgo_de_ruina(fraccion, caida=0.5):
    """P(la banca toque alguna vez la fraccion `caida` de su valor inicial) ~= caida^(2/c - 1).

    Aproximacion de difusion, valida para apuestas chicas y muchas: con Kelly completo
    (c=1) da `caida`, o sea 50% de probabilidad de perder alguna vez la mitad de la banca
    APOSTANDO CON VENTAJA REAL Y LA PROBABILIDAD EXACTA. Con un cuarto de Kelly esa misma
    caida baja a 3.1%.

    Es el numero que hay que mirar antes que el ROI: un sistema con ventaja real y stake
    mal calibrado quiebra igual, solo que mas despacio.
    """
    c = np.asarray(fraccion, float)
    caida = np.asarray(caida, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.clip(caida ** (2 / c - 1), 0.0, 1.0)


def exposicion(montos, banca, tope_evento=TOPE_EVENTO):
    """-> dict con lo comprometido, el tope y si se paso.

    Dos peleas de la misma cartelera no son dos apuestas independientes (comparten
    condiciones, arbitro, jueces, y muchas veces al mismo peleador en una combinada), asi
    que Kelly por apuesta subestima el riesgo conjunto. El tope por evento es la version
    barata de tratar esa correlacion: no la modela, la acota.
    """
    total = float(np.sum(montos))
    limite = float(banca) * tope_evento
    return {"comprometido": total, "fraccion": total / float(banca) if banca else np.nan,
            "limite": limite, "excede": total > limite,
            "disponible": max(limite - total, 0.0)}


# -------------------------------------------------------------------- combinadas

def ev_combinada(legs):
    """EV de una combinada de legs independientes: prod(1 + ev_i) - 1.

    El margen de la casa se compone en cada leg. Tres selecciones a -5% de EV no dan
    -5%: dan -14.3%. Por eso una combinada de legs sin ventaja no es "mas riesgo por mas
    premio", es la misma apuesta con el vig cobrado tres veces.
    """
    legs = [(float(p), float(q)) for p, q in legs]
    return float(np.prod([p * q for p, q in legs]) - 1)


def dependencia_necesaria(legs):
    """Cuanta correlacion hace falta para que la combinada sea neutra.

    -> dict con la probabilidad conjunta que se necesita, la que implica la independencia,
    y el `lift` entre las dos. Para dos legs agrega el phi (correlacion de dos binarias)
    que haria falta.

    Sirve para lo unico que justifica una combinada: legs POSITIVAMENTE correlacionados
    que la casa precia como independientes. Si el lift necesario es 1.15, la pregunta
    concreta es "¿estos dos resultados pasan juntos un 15% mas seguido de lo que dice el
    producto?" — que se puede contestar, a diferencia de "¿me siento confiado?".
    """
    legs = [(float(p), float(q)) for p, q in legs]
    ps = np.array([p for p, _ in legs])
    cuota = float(np.prod([q for _, q in legs]))
    independiente = float(np.prod(ps))
    necesaria = 1 / cuota
    out = {"p_independiente": independiente, "p_necesaria": necesaria,
           "cuota": cuota, "lift": necesaria / independiente if independiente else np.inf}
    if len(legs) == 2:
        denominador = float(np.sqrt(np.prod(ps * (1 - ps))))
        out["phi"] = (necesaria - independiente) / denominador if denominador else np.nan
    return out


# ------------------------------------------------------------------- ¿cuando se sabe?

def sigma_apuesta(p, q):
    """Desviacion del retorno de una apuesta flat de 1 unidad: q*sqrt(p*(1-p)).

    Gana `q-1` con probabilidad `p` y pierde 1 con probabilidad `1-p`. Cerca de cuota
    2.00 vale ~1.0: la apuesta tiene cincuenta veces mas ruido que la senal que se busca.
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    return q * np.sqrt(p * (1 - p))


def n_para_detectar(media, sigma, potencia=0.80, alfa=0.05):
    """Apuestas necesarias para distinguir `media` de cero con esa potencia.

    n = ((z_{1-alfa/2} + z_{potencia}) * sigma / media)^2

    Es el numero que decide que se puede verificar y que no. Con `media` = ROI esperado y
    `sigma` = `sigma_apuesta` da decenas de miles; con `media` = CLV medio y `sigma` = la
    dispersion del CLV da decenas. Misma formula, dos ordenes de magnitud de diferencia,
    y ahi esta la razon de que el gate de este proyecto mire el CLV.
    """
    media, sigma = np.abs(np.asarray(media, float)), np.asarray(sigma, float)
    z = _z(1 - alfa / 2) + _z(potencia)
    with np.errstate(divide="ignore", invalid="ignore"):
        n = (z * sigma / media) ** 2
    return np.where(media > 0, np.ceil(n), np.inf)
