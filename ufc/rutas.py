"""Todas las rutas del proyecto, ancladas al repo y no al CWD.

Sin esto, un script corrido desde otra carpeta escribe los CSVs donde no va y no
se queja: `pathlib.Path("data/x.csv")` es relativo al directorio actual.
"""

import pathlib

RAIZ = pathlib.Path(__file__).resolve().parent.parent
DATOS = RAIZ / "data"
RAW = DATOS / "raw"
MODELO = DATOS / "model.pkl"
