"""El historial deportivo de un peleador, en la forma en que lo lee un analista.

`fighter_state.csv` tiene promedios: golpes por minuto, defensa de derribo, finish rate.
Eso responde "cuanto pega" pero no responde las dos preguntas que un tipster hace primero:
**como gana** y **como pierde**. Un 0.15 de `finished_against_rate` no dice si lo
noquearon o si lo sometieron, y no dice contra quien.

Este modulo saca eso de `data/raw/ufc_fight_results.csv`, que ya esta en el repo y trae
metodo y round de cada pelea: el record en UFC, el desglose de victorias y derrotas por
via, y las ultimas peleas con nombre de rival, resultado, metodo y el Elo de ese rival.
Es la diferencia entre "gana el 70% de sus peleas" y "esta invicto por KO en 13 peleas y
sus tres derrotas fueron por decision contra top-5".

Lee disco, asi que vive aca y no en `dossier.py`, que es puro a proposito.
"""

import functools

import pandas as pd

from ufc import nombres, rutas
from ufc.modelo import settlement

METODOS = ("ko", "sub", "dec")
ULTIMAS = 6


@functools.lru_cache(maxsize=1)
def cargar():
    """-> (DataFrame en forma larga, {clave: stance}). Una vez por proceso.

    Forma larga = una fila por (peleador, pelea), o sea cada pelea aparece dos veces, una
    por lado. Es lo que permite filtrar por peleador sin mirar dos columnas.
    """
    res = pd.read_csv(rutas.RAW / "ufc_fight_results.csv")
    eventos = pd.read_csv(rutas.RAW / "ufc_event_details.csv")
    tott = pd.read_csv(rutas.RAW / "ufc_fighter_tott.csv")
    # los CSVs de ufcstats traen espacios colgantes en casi todas las columnas de texto
    for df in (res, eventos, tott):
        for c in df.columns:
            if pd.api.types.is_string_dtype(df[c]):
                df[c] = df[c].str.strip()

    eventos["DATE"] = pd.to_datetime(eventos["DATE"], format="%B %d, %Y")
    res = res.merge(eventos[["EVENT", "DATE"]], on="EVENT", how="inner")
    # solo peleas con ganador claro: un NC o un draw no dice nada de como gana nadie
    res = res[res["OUTCOME"].isin(["W/L", "L/W"])].copy()
    bout = res["BOUT"].str.split(" vs. ", n=1, expand=True)
    res["a"], res["b"] = bout[0], bout[1]
    res = res[res["b"].notna()]
    res["metodo"] = res["METHOD"].map(settlement.canonical_method)
    gano_a = res["OUTCOME"] == "W/L"

    largo = pd.concat([
        pd.DataFrame({"peleador": res[lado], "rival": res[otro], "fecha": res["DATE"],
                      "gano": gano_a if lado == "a" else ~gano_a,
                      "metodo": res["metodo"], "round": res["ROUND"],
                      "evento": res["EVENT"]})
        for lado, otro in (("a", "b"), ("b", "a"))])
    largo["clave"] = largo["peleador"].map(nombres.normalizar)
    largo = largo.sort_values("fecha").reset_index(drop=True)

    stances = (tott.drop_duplicates("FIGHTER")
               .assign(clave=lambda t: t["FIGHTER"].map(nombres.normalizar))
               .set_index("clave")["STANCE"].dropna().to_dict())
    return largo, stances


def _elo(estado, rival):
    if estado is None:
        return None
    clave = nombres.normalizar(rival)
    if clave not in estado.index:
        return None
    valor = estado.loc[clave, "elo"]
    return None if pd.isna(valor) else float(valor)


def resumen(nombre, fecha, *, largo=None, stances=None, estado=None, n=ULTIMAS):
    """-> dict con record, desglose por metodo y ultimas peleas. None si no peleo en UFC.

    Corta en `fecha`: solo peleas ANTERIORES al evento. Es la misma regla anti-leakage que
    respeta `predict._snapshot`, y sin ella el dossier diria una cosa y el modelo habria
    visto otra sobre la misma pelea.
    """
    if largo is None or stances is None:
        cargados = cargar()
        largo = largo if largo is not None else cargados[0]
        stances = stances if stances is not None else cargados[1]

    clave = nombres.normalizar(nombre)
    suyas = largo[(largo["clave"] == clave) & (largo["fecha"] < pd.Timestamp(fecha))]
    if not len(suyas):
        return None

    gano = suyas["gano"].to_numpy(bool)
    metodo = suyas["metodo"].to_numpy()
    cuenta = lambda mask: {m: int(((metodo == m) & mask).sum()) for m in METODOS}  # noqa: E731
    return {
        "peleas": len(suyas),
        "record": {"w": int(gano.sum()), "l": int((~gano).sum())},
        "gana_por": cuenta(gano),
        "pierde_por": cuenta(~gano),
        "stance": stances.get(clave),
        "ultimas": [{"fecha": str(f.fecha.date()), "rival": f.rival, "gano": bool(f.gano),
                     "metodo": f.metodo, "round": None if pd.isna(f.round) else int(f.round),
                     "elo_rival": _elo(estado, f.rival)}
                    for f in suyas.tail(n).iloc[::-1].itertuples()],
    }


def para(pelea, fecha, estado=None):
    """-> {'a': resumen|None, 'b': resumen|None}. Lo que consume el dossier."""
    largo, stances = cargar()
    return {lado: resumen(pelea[lado], fecha, largo=largo, stances=stances, estado=estado)
            for lado in ("a", "b")}
