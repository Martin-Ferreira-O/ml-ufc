"""Donde queda lo que dijo la IA, partido en dos por una razon.

`data/ia_consenso.csv` son numeros y enums, nada de prosa: es el forward test, se
versiona en git y es lo que despues mide `ufc/ia/evaluar.py`. Un veredicto con las cuotas
y la inteligencia del dia en que se emitio no se puede reconstruir despues, asi que entra
en la misma categoria que `ledger.csv` y `picks.csv`.

`data/ia_informes/` es el texto: razones, factores, contraargumento — y, bajo `_prompt` y
`_dossier`, el input exacto con el que se produjo. Eso queda fuera de git por el mismo
motivo que `data/intel.db`, que el `.gitignore` deja escrito: son dossiers sobre personas
reales generados por un LLM. La app los lee de disco; si no estan, muestra la fila
numerica y lo dice.

La clave de idempotencia es (evento, a, b, run_day), igual que `intel.store.ya_revisado`:
una pelea se analiza una vez por dia, y abrir la app veinte veces no cuesta nada.

`prompt_v` separa cohortes. La v1 le mostraba a la IA la probabilidad del modelo, el
precio del mercado y las picks humanas; la v2 es ciega a todo eso. Son dos predictores
distintos con el mismo nombre, y promediarlos daria un numero que no describe a ninguno.
Las filas viejas quedan legibles con la columna vacia, que es exactamente lo que son.
"""

import json

import pandas as pd

from ufc import nombres, rutas
from ufc.ia import dossier as dossier_mod

ARCHIVO = rutas.DATOS / "ia_consenso.csv"
INFORMES = rutas.DATOS / "ia_informes"

COLS = [
    "ts_utc", "run_day", "evento", "fecha_evento", "inicio_utc", "a", "b", "peso",
    "modelo_ia", "prompt_v", "huella",
    "veredicto", "pick", "p_a_ia", "confianza", "metodo_probable",
    "lado", "cuota_tomada", "casa", "ev_ia", "cumple_regla",
    "p_a_modelo", "p_a_con_odds", "p_a_mercado", "cuota_a", "cuota_b",
    "n_razones", "n_factores", "prompt_tokens", "output_tokens",
]
CLAVE = ["evento", "a", "b", "run_day"]


def _slug(texto):
    """Nombre de archivo estable: sin acentos, sin espacios, sin sorpresas del FS."""
    limpio = "".join(c if c.isalnum() or c in "-_" else "-"
                     for c in nombres.normalizar(texto))
    return "-".join(x for x in limpio.split("-") if x)[:80] or "sin-nombre"


def ruta_informe(evento, a, b):
    return INFORMES / _slug(evento) / f"{_slug(a)}__{_slug(b)}.json"


def leer(evento=None):
    """-> DataFrame de `ia_consenso.csv`, filtrado por evento si se pide."""
    if not ARCHIVO.exists():
        return pd.DataFrame(columns=COLS)
    df = pd.read_csv(ARCHIVO, dtype={"prompt_v": str})
    for c in COLS:
        if c not in df:
            df[c] = None  # columnas nuevas van al final y las filas viejas siguen leibles
    df = df[COLS]
    return df[df["evento"] == evento].copy() if evento is not None else df


def ya_analizada(evento, a, b, run_day):
    df = leer(evento)
    if not len(df):
        return False
    m = ((df["a"] == a) & (df["b"] == b) & (df["run_day"].astype(str) == str(run_day)))
    return bool(m.any())


def fila_desde(veredicto, dossier, *, run_day, modelo_ia, usage=None, ts=None):
    """-> dict con las columnas del CSV. Solo numeros y enums: el texto va al JSON."""
    from ufc.ia import analista

    p, mercado, modelo = dossier["pelea"], dossier["mercado"], dossier["modelo"]
    apuesta = analista.ev_contra_mercado(veredicto, dossier)
    consenso = mercado.get("consenso") or {}
    usage = usage or {}
    return {
        "ts_utc": ts or pd.Timestamp.now(tz="UTC").isoformat(),
        "run_day": str(run_day),
        "evento": p["evento"], "fecha_evento": p["fecha"],
        "inicio_utc": p.get("inicio_utc") or "",
        "a": p["a"], "b": p["b"], "peso": p["peso"],
        "modelo_ia": modelo_ia, "prompt_v": dossier_mod.VERSION,
        "huella": dossier_mod.huella(dossier),
        "veredicto": veredicto.get("veredicto", "definido"),
        "pick": veredicto["pick"], "p_a_ia": veredicto["p_a"],
        "confianza": veredicto["confianza"],
        "metodo_probable": veredicto["metodo_probable"] or "",
        # Lo calculo Python despues del veredicto, con la cuota real. La IA no vio nada
        # de esta fila: por eso su EV y su CLV se pueden medir contra el mercado.
        "lado": apuesta["lado"] or "", "cuota_tomada": apuesta["cuota"],
        "casa": apuesta["casa"] or "", "ev_ia": apuesta["ev"],
        "cumple_regla": bool(apuesta["cumple_regla"]),
        # El snapshot de las tres probabilidades del momento en que opino. Sin esto no se
        # puede comparar log loss contra modelo y mercado sobre las mismas peleas: las
        # cuotas de manana ya no son las que habia cuando la IA opino.
        "p_a_modelo": modelo.get("p_a"),
        "p_a_con_odds": modelo.get("p_a_con_odds"),
        "p_a_mercado": mercado.get("p_a_mercado") or consenso.get("p_a"),
        "cuota_a": mercado.get("cuota_a"), "cuota_b": mercado.get("cuota_b"),
        "n_razones": len(veredicto["razones"]),
        "n_factores": len(veredicto["factores_no_modelables"]),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
    }


def guardar(fila, veredicto, dossier=None):
    """Escribe la fila numerica y el informe. Reemplaza si ya existe esa clave.

    Reescribir el CSV entero cuesta menos codigo que un append con dedup y ademas permite
    que `--force` corrija un veredicto en vez de duplicarlo. Son ~14 filas por cartelera.

    Con `dossier`, el informe se lleva ademas el texto exacto que se le envio al modelo.
    La columna `huella` es un sha256 del prompt: sirve para saber si dos veredictos vieron
    lo mismo, pero no para auditar por que opino lo que opino. Sin el prompt guardado, una
    pick vieja solo se puede explicar reconstruyendo el dossier con el codigo de hoy, que
    ya no es el que corrio. Son ~15 KB por pelea en una carpeta que igual esta fuera de
    git.
    """
    viejo = leer()
    if len(viejo):
        m = pd.Series(True, index=viejo.index)
        for c in CLAVE:
            m &= viejo[c].astype(str) == str(fila[c])
        viejo = viejo[~m]
    ARCHIVO.parent.mkdir(parents=True, exist_ok=True)
    nuevo = pd.concat([viejo, pd.DataFrame([fila], columns=COLS)], ignore_index=True)
    nuevo[COLS].to_csv(ARCHIVO, index=False)

    # El input va con `_` adelante y adentro del mismo dict: `informe()` devuelve esto tal
    # cual y sus lectores (`comunes.ia_tarjeta`) indexan las claves del veredicto
    # directamente. Anidarlo en {"veredicto": ...} los rompe a todos; el prefijo alcanza
    # para distinguir lo que respondio el modelo de lo que se le mando.
    cuerpo = dict(veredicto)
    if dossier is not None:
        cuerpo["_prompt"] = dossier_mod.render(dossier)
        cuerpo["_dossier"] = dossier

    destino = ruta_informe(fila["evento"], fila["a"], fila["b"])
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(cuerpo, ensure_ascii=False, indent=1, default=str))


def informe(evento, a, b):
    """-> el veredicto completo, o None si el JSON no esta en esta maquina."""
    destino = ruta_informe(evento, a, b)
    if not destino.exists():
        return None
    try:
        return json.loads(destino.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def veredictos(evento):
    """-> {(a, b): {..fila.., 'informe': dict|None}} con el analisis mas reciente.

    Se queda con el `run_day` mas nuevo de cada pelea: la cartelera se puede analizar
    varios dias seguidos y lo que la app tiene que mostrar es la ultima lectura.
    """
    df = leer(evento)
    if not len(df):
        return {}
    df = df.sort_values(["run_day", "ts_utc"]).drop_duplicates(["a", "b"], keep="last")
    salida = {}
    for fila in df.to_dict("records"):
        fila["informe"] = informe(evento, fila["a"], fila["b"])
        salida[(fila["a"], fila["b"])] = fila
    return salida


def resumen():
    """-> dict con la ultima corrida: evento, cobertura y tokens. Para `--status`.

    Filtra por `prompt_v` ademas de por evento y run_day. Sin eso, reanalizar media
    cartelera con el prompt nuevo daria un status que suma peras con manzanas: las filas
    de la v1 no tienen ni `veredicto` ni EV comparable.
    """
    df = leer()
    if not len(df):
        return None
    ultimo = df.sort_values(["run_day", "ts_utc"]).iloc[-1]
    mismo = df[(df["evento"] == ultimo["evento"]) &
               (df["run_day"].astype(str) == str(ultimo["run_day"])) &
               (df["prompt_v"].astype(str) == str(ultimo["prompt_v"]))]
    return {
        "evento": ultimo["evento"], "fecha_evento": ultimo["fecha_evento"],
        "run_day": str(ultimo["run_day"]), "modelo_ia": ultimo["modelo_ia"],
        "peleas": len(mismo),
        "parejas": int((mismo["veredicto"] == "parejo").sum()),
        "con_ev": int((pd.to_numeric(mismo["ev_ia"], errors="coerce") > 0).sum()),
        "prompt_tokens": int(pd.to_numeric(mismo["prompt_tokens"],
                                           errors="coerce").fillna(0).sum()),
        "output_tokens": int(pd.to_numeric(mismo["output_tokens"],
                                           errors="coerce").fillna(0).sum()),
    }
