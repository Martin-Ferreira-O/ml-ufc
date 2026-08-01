"""Las picks de los tipsters humanos, y cuanto coinciden con el modelo y la cuota.

Tres personas publican sus picks antes de cada cartelera. Dos las suben como imagen
(una infografia con un tilde verde sobre la cara del elegido, una tabla con nivel de
confianza) y la tercera las explica en video. Las imagenes las lee Gemini con vision,
no un OCR: el pick es un tilde sobre una foto, no texto.

Para no tener que matchear nombres mal escritos o sin acentos contra el historial, al
modelo se le pasa la cartelera real numerada y solo devuelve el indice de la pelea y
de que lado va. El matching de nombres, que es donde esto se rompia, no existe.

Los resultados se cargan a mano en la app: `ledger._resultados()` los saca de ufcstats
pero recien despues de correr el pipeline, y la gracia es medir el acierto la misma
noche del evento.
"""

import json
import os

import numpy as np
import pandas as pd

from ufc import rutas

PICKS = rutas.DATOS / "picks.csv"
RESULTADOS = rutas.DATOS / "resultados.csv"
COLS = ["predictor", "evento", "fecha_evento", "a", "b", "pick", "metodo", "round",
        "confianza"]
COLS_RES = ["evento", "a", "b", "ganador"]

# ponytail: el modelo barato alcanza para leer una infografia. Si llega a confundir de
# que lado esta el tilde, subir a "gemini-3.5-flash" es cambiar esta linea.
MODELO = "gemini-3.5-flash"
METODOS = ["ko", "sub", "dec"]

LOCK = "✅ consenso + modelo"
CONTRA = "⚠️ modelo en contra"
SPLIT = "🔀 split"
SIN_MODELO = "✅ consenso (sin modelo)"

ESQUEMA = {
    "type": "OBJECT",
    "required": ["picks"],
    "properties": {"picks": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "required": ["i", "ganador"],
        "properties": {
            "i": {"type": "INTEGER"},
            "ganador": {"type": "STRING", "enum": ["a", "b"]},
            # sin "" en el enum: la API lo rechaza. metodo no es required, se omite
            "metodo": {"type": "STRING", "enum": METODOS},
            "round": {"type": "INTEGER"},
            "confianza": {"type": "NUMBER"},
        }}}},
}

INSTRUCCION = """Esta imagen son las predicciones de un tipster para una cartelera de UFC.

Peleas del evento:
{cartelera}

Devolve una entrada por cada pelea de esa lista que aparezca en la imagen; las que no
aparezcan, omitilas.

- i: el indice de la pelea en la lista de arriba. Matchea por apellido: la imagen puede
  escribirlo distinto, sin acentos o solo con el apellido.
- ganador: "a" si el tipster elige al primero de la lista, "b" si elige al segundo. El
  orden de la imagen puede estar invertido respecto de la lista, asi que guiate por el
  nombre y nunca por la posicion. El elegido suele estar marcado con un tilde verde
  sobre su foto, o escrito en una columna aparte.
- metodo: "ko", "sub" o "dec" si la imagen lo dice. Un cartel de dos como "KO/Dec" o
  "Sub/KO" lista el mas probable primero: quedate con ese. "Ends by Dec" es "dec" y
  "Ends ITD" (inside the distance) no es ninguno: omitilo.
- round: el numero de round si lo dice ("KO rd 1" -> 1). Omitilo si no.
- confianza: el porcentaje como fraccion entre 0 y 1 si la imagen lo muestra
  (HIGH (78%) -> 0.78). Omitilo si no.
"""


def cargar_env(archivo=rutas.RAIZ / ".env"):
    """La key vive en .env: leer una linea `CLAVE=valor` no justifica python-dotenv.

    ponytail: sin comillas multilinea ni `export`. Si el .env crece, python-dotenv.
    """
    if not archivo.exists():
        return
    for linea in archivo.read_text().splitlines():
        clave, _, valor = linea.partition("=")
        if valor and not linea.lstrip().startswith("#"):
            os.environ.setdefault(clave.strip(), valor.strip().strip("\"'"))


cargar_env()


def hay_api():
    """El uploader solo tiene sentido con la key puesta; el editor a mano anda igual."""
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def extraer(imagen, mime, peleas):
    """-> [{'i', 'ganador', 'metodo'?, 'round'?, 'confianza'?}] leidas de la imagen."""
    from google import genai
    from google.genai import types

    cartelera = "\n".join(f"{i}: {p['a']} vs {p['b']}" for i, p in enumerate(peleas))
    # el Client tiene que vivir en una variable: si se llama encadenado, el garbage
    # collector lo cierra a mitad del request ("client has been closed")
    cliente = genai.Client()
    r = cliente.models.generate_content(
        model=MODELO,
        contents=[INSTRUCCION.format(cartelera=cartelera),
                  types.Part.from_bytes(data=imagen, mime_type=mime)],
        config=types.GenerateContentConfig(response_mime_type="application/json",
                                           response_schema=ESQUEMA))
    picks = json.loads(r.text).get("picks", [])
    # el indice lo inventa el modelo: uno fuera de rango romperia la cartelera entera
    return [p for p in picks if isinstance(p.get("i"), int) and 0 <= p["i"] < len(peleas)]


def _leer(archivo, cols, evento=None):
    if not archivo.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(archivo)
    return df[df["evento"] == evento].copy() if evento is not None else df


def _reemplazar(archivo, cols, filas, clave):
    """Reescribe el archivo cambiando solo las filas que matchean `clave`.

    Son ~40 filas por evento: reescribir todo sale mas barato en codigo que el
    append con dedup de ledger.registrar, y ademas permite corregir una pick.
    """
    viejo = _leer(archivo, cols)
    if len(viejo):
        m = pd.Series(True, index=viejo.index)
        for c, v in clave.items():
            m &= viejo[c] == v
        viejo = viejo[~m]
    archivo.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([viejo, pd.DataFrame(filas, columns=cols)])[cols].to_csv(archivo,
                                                                      index=False)


def leer(evento=None):
    return _leer(PICKS, COLS, evento)


def guardar(filas, evento, predictor):
    _reemplazar(PICKS, COLS, filas, {"evento": evento, "predictor": predictor})


def leer_resultados(evento=None):
    return _leer(RESULTADOS, COLS_RES, evento)


def guardar_resultados(filas, evento):
    _reemplazar(RESULTADOS, COLS_RES, filas, {"evento": evento})


def _nada_si_falta(v):
    """NaN -> None. Un csv sin metodo vuelve como float NaN, y NaN es truthy: sin esto
    el `or ""` de mas abajo no lo atrapa y se termina guardando el string "nan"."""
    return None if v is None or pd.isna(v) else v


def _lado(pelea, nombre):
    """El nombre elegido en la tabla -> "a"/"b". None si la fila quedo vacia."""
    return next((k for k in "ab" if pelea[k] == nombre), None)


def _fila_de(tabla, pelea):
    """La fila de esa pelea dentro de una tabla guardada, o None."""
    if not len(tabla):
        return None
    m = tabla[(tabla["a"] == pelea["a"]) & (tabla["b"] == pelea["b"])]
    return m.iloc[0] if len(m) else None


def tabla_picks(peleas, guardadas, leidas):
    """La tabla que se edita en la app: lo ya guardado si existe, si no lo de la imagen."""
    por_indice = {p["i"]: p for p in leidas}
    filas = []
    for i, pelea in enumerate(peleas):
        fila = {"pelea": f"{pelea['a']} vs {pelea['b']}", "ganador": None,
                "metodo": None, "round": None, "confianza": None}
        vieja = _fila_de(guardadas, pelea)
        if vieja is not None:
            fila |= {"ganador": pelea[vieja["pick"]],
                     "metodo": _nada_si_falta(vieja["metodo"]),
                     "round": _nada_si_falta(vieja["round"]),
                     "confianza": _nada_si_falta(vieja["confianza"])}
        elif i in por_indice:
            leida = por_indice[i]
            fila |= {"ganador": pelea[leida["ganador"]],
                     "metodo": leida.get("metodo") or None,
                     "round": leida.get("round"), "confianza": leida.get("confianza")}
        filas.append(fila)
    # sin el astype una columna toda vacia queda object y NumberColumn no la deja editar
    return pd.DataFrame(filas).astype({"round": "Float64", "confianza": "Float64"})


def filas_picks(editado, peleas, evento, fecha, quien):
    """La tabla editada -> filas del csv. Las peleas sin ganador elegido no se guardan."""
    filas = []
    for (_, fila), pelea in zip(editado.iterrows(), peleas):
        lado = _lado(pelea, fila["ganador"])
        if lado:
            filas.append([quien, evento, fecha, pelea["a"], pelea["b"], lado,
                          _nada_si_falta(fila["metodo"]) or "",
                          _nada_si_falta(fila["round"]),
                          _nada_si_falta(fila["confianza"])])
    return filas


def tabla_resultados(peleas, guardados):
    """La tabla para marcar quien gano, con lo ya cargado si lo hay."""
    filas = []
    for pelea in peleas:
        vieja = _fila_de(guardados, pelea)
        filas.append({"pelea": f"{pelea['a']} vs {pelea['b']}",
                      "ganador": pelea[vieja["ganador"]] if vieja is not None else None})
    return pd.DataFrame(filas)


def filas_resultados(editado, peleas, evento):
    return [[evento, pelea["a"], pelea["b"], _lado(pelea, fila["ganador"])]
            for (_, fila), pelea in zip(editado.iterrows(), peleas)
            if _lado(pelea, fila["ganador"])]


def _quien(pelea, lado):
    """El nombre del peleador de ese lado. "" si no hay pick."""
    return pelea["a"] if lado == "a" else pelea["b"] if lado == "b" else ""


def comparar(peleas, picks, preds, cuotas_de):
    """Una fila por pelea: la pick de cada predictor, la del modelo, la cuota y la senal.

    `preds` y `cuotas_de` van alineados con `peleas` (lo que ya arma la pestania
    Cartelera). Un debut viene sin p_a: ahi el consenso vale igual, es lo unico que hay.
    """
    predictores = sorted(picks["predictor"].unique()) if len(picks) else []
    filas = []
    for pelea, r, cuotas in zip(peleas, preds, cuotas_de):
        fila = {"pelea": f"{pelea['a']} vs {pelea['b']}"}
        lados = []
        for quien in predictores:
            suya = picks[(picks["predictor"] == quien) & (picks["a"] == pelea["a"])
                         & (picks["b"] == pelea["b"])]
            lado = suya["pick"].iloc[0] if len(suya) else None
            lados.append(lado)
            fila[quien] = _quien(pelea, lado)

        modelo = ("a" if r["p_a"] >= 0.5 else "b") if "p_a" in r else None
        fila["modelo"] = _quien(pelea, modelo)
        fila["p_modelo"] = max(r["p_a"], 1 - r["p_a"]) if "p_a" in r else np.nan

        consenso = lados[0] if lados and all(x and x == lados[0] for x in lados) else None
        fila["consenso"] = _quien(pelea, consenso)
        fila["cuota"] = (cuotas[0] if consenso == "a" else cuotas[1]) \
            if cuotas and consenso else np.nan
        fila["ev"] = r.get(f"ev_{consenso}", np.nan) if consenso else np.nan
        fila["senal"] = (SPLIT if not consenso else
                         SIN_MODELO if not modelo else
                         LOCK if consenso == modelo else CONTRA)
        filas.append(fila)
    return pd.DataFrame(filas)


def parlay(df):
    """-> (peleadores, cuota combinada) con las peleas donde todos coinciden."""
    lock = df[(df["senal"] == LOCK) & df["cuota"].notna()] if len(df) else df
    if not len(lock):
        return [], 0.0
    return list(lock["consenso"]), float(lock["cuota"].prod())


def aciertos(por_evento=False):
    """-> DataFrame con aciertos/total/% por predictor sobre las peleas ya cargadas."""
    claves = ["predictor", "evento"] if por_evento else ["predictor"]
    picks, res = leer(), leer_resultados()
    if not len(picks) or not len(res):
        return pd.DataFrame(columns=[*claves, "aciertos", "total", "acierto"])
    df = picks.merge(res, on=["evento", "a", "b"])
    if not len(df):
        return pd.DataFrame(columns=[*claves, "aciertos", "total", "acierto"])
    df["ok"] = df["pick"] == df["ganador"]
    g = df.groupby(claves)["ok"].agg(aciertos="sum", total="count").reset_index()
    g["acierto"] = g["aciertos"] / g["total"]
    return g.sort_values("acierto", ascending=False)
