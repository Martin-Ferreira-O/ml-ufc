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

import csv
import json
import os

import numpy as np
import pandas as pd

from ufc import nombres, rutas
from ufc.datos import cartelera

PICKS = rutas.DATOS / "picks.csv"
PICKS_AUDIT = rutas.DATOS / "picks_audit.csv"
RESULTADOS = rutas.DATOS / "resultados.csv"
COLS_BASE = ["predictor", "evento", "fecha_evento", "a", "b", "pick", "metodo",
             "round", "confianza"]
COLS = [*COLS_BASE, "origen", "revisado"]
COLS_RES = ["evento", "a", "b", "ganador"]
COLS_AUDIT = ["ts_utc", "accion", "revisor", *COLS]

# ponytail: el modelo barato alcanza para leer una infografia. Si llega a confundir de
# que lado esta el tilde, subir a "gemini-3.5-flash" es cambiar esta linea.
MODELO = "gemini-3.5-flash"
METODOS = ["ko", "sub", "dec"]

LOCK = "Consenso + modelo"
CONTRA = "Modelo en contra"
SPLIT = "Opiniones divididas"
SIN_MODELO = "Consenso sin modelo"

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
    if cols == COLS:
        # Los CSV anteriores no declaraban de donde salia la pick ni si una persona la
        # habia confirmado. Se conservan, pero no deben influir en el ranking hasta que
        # se abran y guarden desde el editor historico.
        if "origen" not in df:
            df["origen"] = "legacy"
        if "revisado" not in df:
            df["revisado"] = False
        df["origen"] = df["origen"].fillna("legacy")
        df["revisado"] = df["revisado"].fillna(False).map(
            lambda v: v if isinstance(v, bool) else str(v).lower() == "true")
    for c in cols:
        if c not in df:
            df[c] = None
    df = df[cols]
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
    nuevo = pd.DataFrame(filas, columns=cols) if filas else pd.DataFrame(columns=cols)
    pd.concat([viejo, nuevo], ignore_index=True)[cols].to_csv(archivo, index=False)


def leer(evento=None):
    return _leer(PICKS, COLS, evento)


def _auditar(filas, accion, revisor="usuario_app", ts=None):
    """Append-only: una revision nunca borra la seleccion que existia antes."""
    if not filas:
        return
    PICKS_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not PICKS_AUDIT.exists()
    timestamp = ts or pd.Timestamp.now(tz="UTC").isoformat()
    with PICKS_AUDIT.open("a", newline="") as f:
        writer = csv.writer(f)
        if nuevo:
            writer.writerow(COLS_AUDIT)
        for fila in filas:
            writer.writerow([timestamp, accion, revisor, *fila])


def leer_auditoria(evento=None):
    if not PICKS_AUDIT.exists():
        return pd.DataFrame(columns=COLS_AUDIT)
    df = pd.read_csv(PICKS_AUDIT)
    return df[df["evento"] == evento].copy() if evento is not None else df


def guardar(filas, evento, predictor, origen="manual", revisado=True,
            revisor="usuario_app", ts=None):
    """Guarda todas las picks de una persona para un evento.

    Acepta las filas historicas de nueve columnas para mantener compatible la API. La
    UI nueva explicita el origen, y cualquier guardado humano confirma la revision.
    """
    anteriores = leer(evento)
    anteriores = anteriores[anteriores["predictor"] == predictor]
    completas = []
    for fila in filas:
        valores = list(fila)
        if len(valores) == len(COLS_BASE):
            valores += [origen, revisado]
        elif len(valores) != len(COLS):
            raise ValueError(f"Pick con {len(valores)} campos; se esperaban 9 u 11")
        completas.append(valores)
    previas = {(r[3], r[4]): list(r) for r in anteriores[COLS].itertuples(index=False,
                                                                          name=None)}
    actuales = {(f[3], f[4]): f for f in completas}
    creadas, revisadas = [], []
    for clave, fila in actuales.items():
        (revisadas if clave in previas else creadas).append(fila)
    anuladas = []
    for clave, fila in previas.items():
        if clave not in actuales:
            anuladas.append(fila)
    _auditar(creadas, "pick_created", revisor, ts)
    _auditar(revisadas, "pick_revised", revisor, ts)
    _auditar(anuladas, "pick_voided", revisor, ts)
    _reemplazar(PICKS, COLS, completas, {"evento": evento, "predictor": predictor})


def leer_resultados(evento=None):
    return _leer(RESULTADOS, COLS_RES, evento)


def guardar_resultados(filas, evento):
    _reemplazar(RESULTADOS, COLS_RES, filas, {"evento": evento})


def _clave_resultado(evento, a, b):
    """Identidad estable aunque una fuente cambie acentos u orden de peleadores."""
    par = tuple(sorted((nombres.normalizar(a), nombres.normalizar(b))))
    return nombres.normalizar(evento), par


def _indice_resultados():
    """Ganador normalizado por evento/pelea; UFCStats prevalece sobre lo manual."""
    indice = {}
    for fila in leer_resultados().itertuples():
        if fila.ganador not in {"a", "b"}:
            continue
        indice[_clave_resultado(fila.evento, fila.a, fila.b)] = {
            "ganador": nombres.normalizar(fila.a if fila.ganador == "a" else fila.b),
            "origen": "manual",
        }
    # Se recorren todos los eventos locales, no solo los 20 que se muestran en la UI.
    for evento in cartelera.anteriores(limite=None):
        for pelea in evento["peleas"]:
            lado = pelea.get("ganador")
            if lado not in {"a", "b"}:
                continue
            indice[_clave_resultado(evento["evento"], pelea["a"], pelea["b"])] = {
                "ganador": nombres.normalizar(pelea[lado]),
                "origen": "ufcstats",
            }
    return indice


def resolver_resultados(evento, peleas):
    """Resultados alineados con ``peleas`` y expresados en el orden recibido.

    Devuelve una fila por pelea, incluso si todavia no tiene resultado. ``origen`` vale
    ``ufcstats``, ``manual`` o ``None`` y permite a la UI bloquear solo lo autoritativo.
    """
    indice = _indice_resultados()
    filas = []
    for pelea in peleas:
        encontrado = indice.get(_clave_resultado(evento, pelea["a"], pelea["b"]))
        lado = None
        if encontrado:
            ganador = encontrado["ganador"]
            lado = next((x for x in "ab"
                         if nombres.normalizar(pelea[x]) == ganador), None)
        filas.append({"evento": evento, "a": pelea["a"], "b": pelea["b"],
                      "ganador": lado,
                      "origen": encontrado["origen"] if encontrado and lado else None})
    return pd.DataFrame(filas, columns=[*COLS_RES, "origen"])


def eventos_guardados():
    """Carteleras reconstruidas desde picks/resultados para poder corregir el pasado."""
    picks, resultados = leer(), leer_resultados()
    nombres_evento = list(dict.fromkeys([
        *picks.get("evento", pd.Series(dtype=str)).dropna().tolist(),
        *resultados.get("evento", pd.Series(dtype=str)).dropna().tolist(),
    ]))
    eventos = []
    for evento in nombres_evento:
        p = picks[picks["evento"] == evento]
        r = resultados[resultados["evento"] == evento]
        pares = pd.concat([p[["a", "b"]], r[["a", "b"]]], ignore_index=True) \
            .drop_duplicates()
        fecha = p["fecha_evento"].dropna().iloc[0] if len(p) and p["fecha_evento"].notna().any() else ""
        eventos.append({"evento": evento, "fecha": str(fecha),
                        "peleas": [{"a": x.a, "b": x.b, "peso": ""}
                                    for x in pares.itertuples()], "historico": True})
    return eventos


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


def aciertos(por_evento=False, solo_revisadas=True):
    """-> DataFrame con aciertos/total/% por predictor sobre las peleas ya cargadas."""
    columnas = (["predictor", "evento", "aciertos", "total", "acierto", "carteleras"]
                if por_evento else
                ["predictor", "aciertos", "total", "acierto", "carteleras"])
    picks = leer()
    if solo_revisadas and len(picks):
        picks = picks[picks["revisado"]]
    if not len(picks):
        return pd.DataFrame(columns=columnas)

    indice = _indice_resultados()
    filas = []
    for pick in picks.itertuples():
        encontrado = indice.get(_clave_resultado(pick.evento, pick.a, pick.b))
        if not encontrado or pick.pick not in {"a", "b"}:
            continue
        elegido = nombres.normalizar(pick.a if pick.pick == "a" else pick.b)
        filas.append({"predictor": pick.predictor, "evento": pick.evento,
                      "evento_clave": nombres.normalizar(pick.evento),
                      "ok": elegido == encontrado["ganador"]})
    if not filas:
        return pd.DataFrame(columns=columnas)
    df = pd.DataFrame(filas)
    if por_evento:
        g = df.groupby(["predictor", "evento_clave"], sort=False).agg(
            evento=("evento", "first"), aciertos=("ok", "sum"),
            total=("ok", "count")).reset_index(drop=False)
        g["carteleras"] = 1
    else:
        g = df.groupby("predictor").agg(
            aciertos=("ok", "sum"), total=("ok", "count"),
            carteleras=("evento_clave", "nunique")).reset_index()
    g["acierto"] = g["aciertos"] / g["total"]
    return g[columnas].sort_values("acierto", ascending=False)


def confiabilidad():
    """Precision observada y peso conservador para cada predictor revisado.

    El prior Beta(5, 5) evita que una racha corta de 14-0 se trate como 100% real.
    """
    g = aciertos()
    if not len(g):
        return pd.DataFrame(columns=["predictor", "aciertos", "total", "acierto",
                                     "carteleras", "peso"])
    g["peso"] = (g["aciertos"] + 5) / (g["total"] + 10)
    return g


def ranking(peleas, picks, preds, cuotas_de):
    """Ordena las peleas por apoyo humano y confirmaciones independientes.

    Solo usa picks revisadas. El modelo y el favorito de mercado no cambian el voto
    humano: confirman la direccion y resuelven el orden entre apoyos parecidos.
    """
    picks = picks[picks["revisado"]].copy() if len(picks) else picks
    pesos_df = confiabilidad()
    pesos = dict(zip(pesos_df["predictor"], pesos_df["peso"]))
    precision = dict(zip(pesos_df["predictor"], pesos_df["acierto"]))
    totales = dict(zip(pesos_df["predictor"], pesos_df["total"]))
    filas = []
    for orden, (pelea, r, cuotas) in enumerate(zip(peleas, preds, cuotas_de)):
        pp = picks[(picks["a"] == pelea["a"]) & (picks["b"] == pelea["b"])] \
            if len(picks) else picks
        votos = {"a": 0.0, "b": 0.0}
        conteo = {"a": 0, "b": 0}
        detalle = []
        for pick in pp.itertuples():
            peso = pesos.get(pick.predictor, 0.5)
            if pick.pick in votos:
                votos[pick.pick] += peso
                conteo[pick.pick] += 1
            # Un predictor sin resultados cargados todavia no tiene precision: va en None
            # y no en 0%, que se leeria como "nunca acerto".
            detalle.append({"predictor": pick.predictor,
                            "eligio": _quien(pelea, pick.pick),
                            "acierto": precision.get(pick.predictor),
                            "total": int(totales.get(pick.predictor, 0))})
        total_peso = sum(votos.values())
        if not total_peso or votos["a"] == votos["b"]:
            lado, apoyo = None, (0.5 if total_peso else np.nan)
        else:
            lado = "a" if votos["a"] > votos["b"] else "b"
            apoyo = votos[lado] / total_peso

        lado_modelo = ("a" if r["p_a"] >= 0.5 else "b") if "p_a" in r else None
        lado_mercado = None if not cuotas else ("a" if cuotas[0] <= cuotas[1] else "b")
        conf_modelo = bool(lado and lado == lado_modelo)
        conf_mercado = bool(lado and lado == lado_mercado)
        confirmaciones = int(conf_modelo) + int(conf_mercado)
        fuerte = bool(lado and len(pp) >= 2 and apoyo >= 2 / 3 and confirmaciones)
        if fuerte:
            senal = "Señal fuerte"
        elif lado and conteo[lado] == len(pp) and len(pp):
            senal = "Consenso humano"
        elif lado:
            senal = "Mayoría humana"
        elif len(pp):
            senal = "Opiniones divididas"
        else:
            senal = "Sin picks revisadas"
        filas.append({
            "orden": orden,
            "pelea": f"{pelea['a']} vs {pelea['b']}",
            "a": pelea["a"], "b": pelea["b"], "lado": lado,
            "seleccion": _quien(pelea, lado), "apoyo": apoyo,
            "votos": conteo.get(lado, 0) if lado else 0,
            "predictores": len(pp), "modelo_confirma": conf_modelo,
            "mercado_confirma": conf_mercado, "confirmaciones": confirmaciones,
            "senal": senal, "fuerte": fuerte, "detalle": detalle,
            "cuota": (cuotas[0] if lado == "a" else cuotas[1]) if cuotas and lado else np.nan,
        })
    df = pd.DataFrame(filas)
    if not len(df):
        return df
    return df.sort_values(["fuerte", "confirmaciones", "apoyo", "predictores", "orden"],
                          ascending=[False, False, False, False, True]).reset_index(drop=True)
