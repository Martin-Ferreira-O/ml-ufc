"""Un nombre de peleador -> su clave canonica.

Vive aca y no en `modelo/predict.py` porque lo necesitan los tres adaptadores de
`datos/` y el `registro/` para matchear contra el historial, y ninguno tendria por
que importar el modelo para normalizar un string.
"""

import unicodedata


def normalizar(s):
    """Minusculas, sin acentos: ESPN escribe 'Spasić' donde ufcstats escribe 'Spasic'.

    Verificado sobre las 2716 filas del estado: ningun par de peleadores distintos
    colapsa a la misma clave, asi que el indice sigue siendo unico.
    """
    s = unicodedata.normalize("NFKD", str(s))
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).lower().split())
