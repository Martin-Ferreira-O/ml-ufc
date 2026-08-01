"""Reemplazos de corto aviso y peso no dado, desde los articulos de evento de Wikipedia.

Es la unica informacion del pipeline que NO sale del historial deportivo. Un peleador que
entro con tres dias de aviso, o que llego dos kilos pasado, esta en una pelea distinta a la
que dice su record — y ninguna feature acumulada puede verlo. Esta medido que extraerle mas
detalle a ufcstats no mueve la aguja (ver DECISIONS.md, ronda de candidatas): lo que falta
esta afuera.

Wikipedia y no otra fuente, verificado: es la unica que tiene esto CON HISTORIA (desde
UFC 1), via API publica y con licencia CC BY-SA. Sherdog solo lo narra en prosa dentro de
notas de pesaje de anios recientes. Tapology esta detras de un challenge activo de
Cloudflare y ademas declara `Content-Signal: ai-train=no` — no se scrapea.

Dos trampas, las dos resueltas aca:
  - el buscador de la API resuelve a articulos EQUIVOCADOS ("Jung vs. Ige" devuelve la
    ficha de Dan Ige; "Belfort vs Henderson 2" devuelve el 3). Por eso todo articulo se
    valida contra la FECHA del evento antes de creerle: si no coincide, se descarta.
  - los nombres en la prosa no se extraen, se BUSCAN: ya sabemos quien peleo en cada
    evento, asi que alcanza con ver que se dice cerca de cada apellido conocido.
"""

import pathlib
import re
import sys
import time

import numpy as np
import pandas as pd
import requests

RAW = pathlib.Path("data/raw")
CACHE = RAW / "wiki"
OUT = pathlib.Path("data/wiki_avisos.csv")
# Que eventos tienen articulo validado. Es imprescindible aparte de los avisos: en un
# evento con articulo, "el peleador no aparece" significa 0 (no hubo aviso); en uno sin
# articulo significa NaN (no sabemos). Sin esta lista las dos cosas se confunden y la
# feature le pone "todo normal" a media base.
EVENTOS = pathlib.Path("data/wiki_eventos.csv")
API = "https://en.wikipedia.org/w/api.php"
UA = "ml-ufc/0.1 (fight-outcome research; https://github.com/)"
PAUSA = 0.5   # cortesia con una API gratis y sin key. Con 0.15 devuelve 429.
REINTENTOS = 4

# La seccion que importa. Wikipedia la titula casi siempre "Background"; el resto del
# articulo (Results, Bonus awards, See also) solo mete ruido y nombres sin contexto.
_SECCION = re.compile(r"^==\s*(Background|Preparation)\s*==$(.*?)(?=^==\s[^=]|\Z)",
                      re.M | re.S)

# Gatillos POSICIONALES, no "el apellido esta en la misma frase". La frase tipica es
#   "A bout between X and Y was scheduled, but Y withdrew and was replaced by Z."
# donde X sigue en la pelea y no es reemplazo de nadie: pedir solo co-ocurrencia lo marca
# igual. El reemplazo es el nombre que va DESPUES de "replaced by"; el que no da el peso es
# el que va ANTES de "weighed in". Cuesta recall ("Z stepped in", con el nombre delante,
# se pierde) y lo vale: un aviso atribuido al peleador equivocado es peor que un NaN.
_REEMPLAZO = re.compile(r"replaced by|in place of|as a replacement,?\s", re.I)
_PESO = re.compile(r"missed weight|failed to make weight|weighed in at|came in (?:at )?"
                   r"\d|pounds? over", re.I)
_VENTANA = 70

# "nine days before the event", "less than two weeks before", "on 10 days' notice"
_DIAS = re.compile(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
                   r"[\s-]+(day|days|week|weeks)\b[’'s]*\s*(?:before|notice|prior)", re.I)
_HORAS = re.compile(r"hours before|day of the (?:event|weigh)|during fight week", re.I)
_LIBRAS = re.compile(r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)"
                     r"[\s-]+pounds?\s+over", re.I)
_PALABRA = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def _numero(s):
    return float(_PALABRA.get(s.lower(), s))


def _sesion():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _titulo(evento):
    """Nombre de ufcstats -> titulo probable en Wikipedia.

    Los numerados viven en "UFC 329" a secas. El resto usa el mismo formato que ufcstats
    salvo el punto de "vs." — que ufcstats a veces se come.
    """
    m = re.match(r"(UFC \d+)", evento)
    return m.group(1) if m else re.sub(r"\bvs\b(?!\.)", "vs.", evento)


def _get(sesion, params):
    """GET con backoff. Wikipedia tira 429 si se la apura; no es un error, es un semaforo."""
    for intento in range(REINTENTOS):
        r = sesion.get(API, params=params, timeout=30)
        if r.status_code == 429:
            time.sleep(PAUSA * 4 * (intento + 1))
            continue
        r.raise_for_status()
        time.sleep(PAUSA)
        return r.json()
    r.raise_for_status()


def _bajar(sesion, titulos):
    """-> {titulo_pedido: texto}. Hasta 20 por request, siguiendo redirects."""
    d = _get(sesion, {
        "action": "query", "prop": "extracts", "explaintext": 1, "format": "json",
        "redirects": 1, "titles": "|".join(titulos)}).get("query", {})
    # los redirects cambian el titulo, hay que poder volver al que pedimos
    vuelta = {x["to"]: x["from"] for x in d.get("redirects", [])}
    vuelta.update({x["to"]: x["from"] for x in d.get("normalized", [])})
    out = {}
    for p in d.get("pages", {}).values():
        if "missing" not in p and p.get("extract"):
            t = p["title"]
            out[vuelta.get(t, t)] = p["extract"]
    return out


def _buscar(sesion, evento):
    """Fallback: el buscador. Devuelve candidato SIN validar — lo valida `_es_el_evento`."""
    try:
        d = _get(sesion, {"action": "query", "list": "search", "srsearch": evento,
                          "srlimit": 3, "format": "json"})
        return [x["title"] for x in d["query"]["search"]]
    except (requests.RequestException, ValueError, KeyError):
        return []


def _es_el_evento(texto, fecha):
    """La unica validacion que sirve: que el articulo hable de la fecha del evento.

    El titulo no alcanza — el buscador devuelve el evento vecino o la ficha de un peleador
    con total naturalidad, y un reemplazo atribuido al evento equivocado es peor que un NaN.
    """
    if not texto or pd.isna(fecha):
        return False
    # Wikipedia escribe "June 29, 2024" y a veces "29 June 2024"
    cab = texto[:4000]
    return (fecha.strftime("%B %-d, %Y") in cab or fecha.strftime("%-d %B %Y") in cab)


def texto_evento(sesion, evento, fecha, cache=True):
    """-> texto del articulo validado por fecha, o None. Cachea en disco por evento."""
    f = CACHE / (re.sub(r"[^\w -]", "_", evento)[:120] + ".txt")
    if cache and f.exists():
        t = f.read_text(encoding="utf-8")
        return t or None

    t = None
    for cand in [_titulo(evento), *(_buscar(sesion, evento))]:
        got = _bajar(sesion, [cand]).get(cand)
        if _es_el_evento(got, fecha):
            t = got
            break
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(t or "", encoding="utf-8")  # cachea tambien el fracaso, para no reintentar
    return t


def _background(texto):
    m = _SECCION.search(texto or "")
    return m.group(2) if m else ""


def _oraciones(t):
    return re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", t))


def _dias_aviso(frase):
    m = _DIAS.search(frase)
    if m:
        return _numero(m.group(1)) * (7 if m.group(2).lower().startswith("week") else 1)
    return 1.0 if _HORAS.search(frase) else np.nan


def _ventanas(frase):
    """-> (texto despues del gatillo de reemplazo, texto antes del gatillo de peso).

    Ahi y solo ahi se busca el apellido. Ver el comentario de los gatillos.
    """
    r = _REEMPLAZO.search(frase)
    p = _PESO.search(frase)
    return (frase[r.end():r.end() + _VENTANA] if r else "",
            frase[max(0, p.start() - _VENTANA):p.start()] if p else "")


def avisos(eventos, peleadores_por_evento, sesion=None, cache=True, textos=None):
    """-> DataFrame(evento, peleador, reemplazo, dias_aviso, peso_no_dado, libras_pasado).

    Una fila por peleador mencionado con algun gatillo. Los que no aparecen no generan
    fila: ausencia de noticia es la normalidad, no un dato faltante.

    `textos` corta la red y usa articulos dados a mano — es lo que consume `autocheck`.
    """
    sesion = sesion or (None if textos else _sesion())
    filas, cubiertos = [], []
    for evento, fecha in eventos.items():
        t = (textos or {}).get(evento) if textos else \
            texto_evento(sesion, evento, fecha, cache=cache)
        bg = _background(t)
        if not bg:
            continue
        cubiertos.append(evento)
        frases = [(f, *_ventanas(f)) for f in _oraciones(bg)]
        for nombre in peleadores_por_evento.get(evento, ()):
            apellido = nombre.split()[-1]
            if len(apellido) < 3:
                continue
            pat = re.compile(rf"\b{re.escape(apellido)}\b")
            ree = [f for f, vr, _ in frases if pat.search(vr)]
            pes = [f for f, _, vp in frases if pat.search(vp)]
            if not ree and not pes:
                continue
            lb = [_numero(m.group(1)) for f in pes for m in [_LIBRAS.search(f)] if m]
            dias = [d for f in ree for d in [_dias_aviso(f)] if not pd.isna(d)]
            filas.append({
                "evento": evento, "peleador": nombre,
                "reemplazo": float(bool(ree)),
                "dias_aviso": min(dias) if dias else np.nan,
                "peso_no_dado": float(bool(pes)),
                "libras_pasado": max(lb) if lb else np.nan,
            })
    return pd.DataFrame(filas), cubiertos


def _catalogo():
    """-> (fecha por evento, peleadores por evento) desde los CSVs crudos."""
    ev = pd.read_csv(RAW / "ufc_event_details.csv")
    re_ = pd.read_csv(RAW / "ufc_fight_results.csv")
    for d in (ev, re_):
        for c in d.columns:
            if pd.api.types.is_string_dtype(d[c]):
                d[c] = d[c].str.strip()
    ev["DATE"] = pd.to_datetime(ev["DATE"], format="%B %d, %Y", errors="coerce")
    bout = re_["BOUT"].str.split(" vs. ", n=1, expand=True)
    re_ = re_.assign(a=bout[0], b=bout[1]).dropna(subset=["b"])
    porev = (pd.concat([re_[["EVENT", "a"]].rename(columns={"a": "f"}),
                        re_[["EVENT", "b"]].rename(columns={"b": "f"})])
             .groupby("EVENT")["f"].apply(list).to_dict())
    return dict(zip(ev["EVENT"], ev["DATE"])), porev


def main():
    fechas, porev = _catalogo()
    if "--muestra" in sys.argv:  # prueba barata antes de bajar 781 articulos
        n = int(sys.argv[sys.argv.index("--muestra") + 1])
        fechas = dict(list(fechas.items())[:n])
    df, cubiertos = avisos(fechas, porev)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    pd.DataFrame({"evento": cubiertos}).to_csv(EVENTOS, index=False)
    print(f"{OUT}: {len(df)} avisos sobre {len(cubiertos)}/{len(fechas)} eventos con "
          f"articulo validado por fecha")
    if len(df):
        print(f"  reemplazos {int(df.reemplazo.sum())} "
              f"(con dias de aviso: {df.dias_aviso.notna().sum()}) | "
              f"peso no dado {int(df.peso_no_dado.sum())} "
              f"(con libras: {df.libras_pasado.notna().sum()})")


def autocheck():
    """Los casos que la extraccion tiene que acertar y los que NO tiene que marcar.

    Sin red: son frases reales congeladas. Si alguien afloja los gatillos para ganar
    recall, el caso de Zhu Kangjie (el que se queda en la pelea) empieza a fallar.
    """
    def uno(frase, peleadores):
        df, _ = avisos({"E": None}, {"E": peleadores},
                       textos={"E": f"== Background ==\n{frase}\n== Results ==\n"})
        return {r.peleador: (r.reemplazo, r.dias_aviso, r.peso_no_dado, r.libras_pasado)
                for r in df.itertuples()}

    f1 = ("A featherweight bout between Zhu Kangjie and Ramon Taveras was scheduled for "
          "the event, but Taveras withdrew and was replaced by Rodrigo Vera.")
    r = uno(f1, ["Zhu Kangjie", "Rodrigo Vera"])
    assert "Zhu Kangjie" not in r, f"el que se queda no es reemplazo: {r}"
    assert r["Rodrigo Vera"][0] == 1.0, r

    f2 = ("However, less than two weeks before the event, Rountree withdrew due to an "
          "injury and was replaced by Bogdan Guskov.")
    assert uno(f2, ["Bogdan Guskov"])["Bogdan Guskov"][:2] == (1.0, 14.0)

    f3 = ("At the weigh-ins, Kevin Borjas weighed in at 129 pounds, three pounds over "
          "the flyweight non-title fight limit.")
    r = uno(f3, ["Kevin Borjas"])["Kevin Borjas"]
    assert (r[2], r[3]) == (1.0, 3.0), r

    # el que OFRECIERON y no acepto no es un reemplazo
    f4 = ("Jan Blachowicz was originally asked to step in on short notice but said he "
          "was not able to make the weight limit.")
    assert not uno(f4, ["Jan Blachowicz"]), "oferta rechazada no es reemplazo"

    assert _es_el_evento("held on June 29, 2024 at", pd.Timestamp("2024-06-29"))
    assert not _es_el_evento("held on June 29, 2024 at", pd.Timestamp("2024-07-06")), \
        "un articulo de otra fecha no puede pasar la validacion"
    print("autocheck ok")


if __name__ == "__main__":
    if "--autocheck" in sys.argv:
        autocheck()
    else:
        main()
