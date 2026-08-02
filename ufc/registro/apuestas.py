"""Apuestas realmente hechas, separadas del seguimiento hipotetico del modelo."""

import uuid

import numpy as np
import pandas as pd

from ufc import nombres, rutas
from ufc.registro import ledger, predictores


APUESTAS = rutas.DATOS / "apuestas.csv"
DETALLE = rutas.DATOS / "apuestas_detalle.csv"
COLS = ["id", "creada", "evento", "fecha_evento", "tipo", "importe_clp", "cuota",
        "estado_manual", "cobro_manual_clp", "liquidada", "nota"]
COLS_DET = ["apuesta_id", "orden", "a", "b", "pick", "cuota"]
ESTADOS_MANUALES = ["Automático", "Ganada", "Perdida", "Anulada", "Cash-out"]


def _leer(archivo, columnas):
    if not archivo.exists():
        return pd.DataFrame(columns=columnas)
    df = pd.read_csv(archivo)
    for c in columnas:
        if c not in df:
            df[c] = None
    if columnas == COLS:
        for c in ("id", "creada", "evento", "fecha_evento", "tipo", "estado_manual",
                  "liquidada", "nota"):
            df[c] = df[c].fillna("").astype(object)
    return df[columnas]


def leer():
    return _leer(APUESTAS, COLS)


def leer_detalle():
    return _leer(DETALLE, COLS_DET)


def _escribir(archivo, df, columnas):
    archivo.parent.mkdir(parents=True, exist_ok=True)
    df[columnas].to_csv(archivo, index=False)


def crear(evento, fecha_evento, selecciones, tipo, importe_clp, cuota=None, hoy=None):
    """Crea simples (una por seleccion) o una combinada y devuelve sus IDs."""
    if tipo not in {"simple", "combinada"}:
        raise ValueError("El tipo debe ser simple o combinada")
    if not selecciones:
        raise ValueError("La boleta no tiene selecciones")
    if float(importe_clp) <= 0:
        raise ValueError("El importe debe ser mayor que cero")
    for s in selecciones:
        if s.get("pick") not in {"a", "b"} or float(s.get("cuota", 0)) <= 1:
            raise ValueError("Cada seleccion necesita lado y cuota decimal mayor que 1")

    existentes, detalles = leer(), leer_detalle()
    creadas = (hoy or pd.Timestamp.now()).isoformat(timespec="seconds")
    grupos = [[s] for s in selecciones] if tipo == "simple" else [selecciones]
    filas, legs, ids = [], [], []
    for grupo in grupos:
        apuesta_id = uuid.uuid4().hex[:12]
        ids.append(apuesta_id)
        precio = float(grupo[0]["cuota"]) if tipo == "simple" else \
            float(cuota or np.prod([float(s["cuota"]) for s in grupo]))
        filas.append([apuesta_id, creadas, evento, fecha_evento, tipo,
                      int(importe_clp), round(precio, 4), "", np.nan, "", ""])
        for orden, s in enumerate(grupo):
            legs.append([apuesta_id, orden, s["a"], s["b"], s["pick"], s["cuota"]])
    _escribir(APUESTAS, pd.concat([existentes, pd.DataFrame(filas, columns=COLS)],
                                  ignore_index=True), COLS)
    _escribir(DETALLE, pd.concat([detalles, pd.DataFrame(legs, columns=COLS_DET)],
                                 ignore_index=True), COLS_DET)
    return ids


def liquidar(apuesta_id, estado="Automático", cobro_clp=None, nota="", hoy=None):
    """Guarda o elimina una liquidacion manual de una apuesta."""
    if estado not in ESTADOS_MANUALES:
        raise ValueError("Estado de liquidacion desconocido")
    df = leer()
    m = df["id"] == apuesta_id
    if not m.any():
        raise ValueError("Apuesta inexistente")
    if estado in {"Ganada", "Perdida", "Cash-out"} and cobro_clp is None:
        raise ValueError("La liquidacion manual necesita el cobro real")
    df.loc[m, "estado_manual"] = "" if estado == "Automático" else estado.lower()
    df.loc[m, "cobro_manual_clp"] = np.nan if estado == "Automático" else \
        (float(cobro_clp) if cobro_clp is not None else float(df.loc[m, "importe_clp"].iloc[0]))
    df.loc[m, "liquidada"] = "" if estado == "Automático" else \
        (hoy or pd.Timestamp.now()).isoformat(timespec="seconds")
    df.loc[m, "nota"] = nota
    _escribir(APUESTAS, df, COLS)


def eliminar(apuesta_id):
    """Elimina una apuesta y sus selecciones; la UI exige confirmacion explicita."""
    df, detalle = leer(), leer_detalle()
    if apuesta_id not in set(df["id"]):
        raise ValueError("Apuesta inexistente")
    _escribir(APUESTAS, df[df["id"] != apuesta_id], COLS)
    _escribir(DETALLE, detalle[detalle["apuesta_id"] != apuesta_id], COLS_DET)


def _resultados_manuales():
    res = predictores.leer_resultados()
    return {(r.evento, nombres.normalizar(r.a), nombres.normalizar(r.b)): r.ganador
            for r in res.itertuples()}


def _resultado_leg(apuesta, leg, manuales, ufc):
    clave = (apuesta.evento, nombres.normalizar(leg.a), nombres.normalizar(leg.b))
    lado = manuales.get(clave)
    if lado in {"a", "b"}:
        return lado
    par = tuple(sorted((nombres.normalizar(leg.a), nombres.normalizar(leg.b))))
    candidatos = ufc[ufc["par"] == par]
    if len(candidatos):
        fecha = pd.to_datetime(apuesta.fecha_evento)
        candidatos = candidatos[np.abs(candidatos["fecha"] - fecha) <= pd.Timedelta(days=2)]
    if not len(candidatos):
        return None
    ganador = candidatos.iloc[0]["ganador"]
    return "a" if ganador == nombres.normalizar(leg.a) else "b"


def evaluar():
    """Devuelve apuestas liquidadas/calculadas, detalle de legs y KPIs economicos."""
    df, detalle = leer(), leer_detalle()
    if not len(df):
        return df, detalle, {}
    manuales = _resultados_manuales()
    try:
        ufc = ledger._resultados()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        ufc = pd.DataFrame(columns=["par", "fecha", "ganador"])

    estados, cobros = [], []
    estados_leg = {}
    for apuesta in df.itertuples():
        legs = detalle[detalle["apuesta_id"] == apuesta.id]
        resultados = []
        for leg in legs.itertuples():
            gano = _resultado_leg(apuesta, leg, manuales, ufc)
            estado_leg = "pendiente" if gano is None else "ganada" if gano == leg.pick else "perdida"
            estados_leg[(leg.apuesta_id, leg.orden)] = estado_leg
            resultados.append(estado_leg)
        manual = str(apuesta.estado_manual) if pd.notna(apuesta.estado_manual) else ""
        if manual and manual != "nan":
            estado = manual
            cobro = float(apuesta.cobro_manual_clp)
        elif "perdida" in resultados:
            estado, cobro = "perdida", 0.0
        elif resultados and all(x == "ganada" for x in resultados):
            estado, cobro = "ganada", float(apuesta.importe_clp) * float(apuesta.cuota)
        else:
            estado, cobro = "pendiente", np.nan
        estados.append(estado)
        cobros.append(cobro)

    df = df.copy()
    df["estado"] = estados
    df["cobro_clp"] = cobros
    df["beneficio_clp"] = df["cobro_clp"] - df["importe_clp"]
    detalle = detalle.copy()
    detalle["estado"] = [estados_leg.get((r.apuesta_id, r.orden), "pendiente")
                         for r in detalle.itertuples()]
    cerradas = df[df["estado"] != "pendiente"]
    decididas = cerradas[cerradas["estado"].isin(["ganada", "perdida"])]
    resumen = {
        "apuestas": len(df),
        "apostado": float(df["importe_clp"].sum()),
        "cobrado": float(cerradas["cobro_clp"].sum()),
        "beneficio": float(cerradas["beneficio_clp"].sum()),
        "roi": (float(cerradas["beneficio_clp"].sum()) /
                float(cerradas["importe_clp"].sum())) if len(cerradas) else np.nan,
        "acierto": float((decididas["estado"] == "ganada").mean()) if len(decididas) else np.nan,
        "pendiente": float(df.loc[df["estado"] == "pendiente", "importe_clp"].sum()),
    }
    return df, detalle, resumen


def evolucion(df):
    """Serie de beneficio acumulado para el dashboard."""
    if not len(df):
        return pd.DataFrame(columns=["fecha", "beneficio acumulado"])
    d = df[df["estado"] != "pendiente"].copy()
    if not len(d):
        return pd.DataFrame(columns=["fecha", "beneficio acumulado"])
    fecha_manual = pd.to_datetime(d["liquidada"], errors="coerce")
    d["fecha"] = fecha_manual.fillna(pd.to_datetime(d["fecha_evento"], errors="coerce"))
    d = d.sort_values(["fecha", "creada"])
    d["beneficio acumulado"] = d["beneficio_clp"].cumsum()
    return d[["fecha", "beneficio acumulado"]]
