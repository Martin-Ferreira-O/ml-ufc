"""Record pre-UFC (circuito regional) desde Sherdog.

El agujero que tapa: un debutante entra al modelo con `n_fights=0` y elo default. Para
features.py, Ilia Topuria el dia de su debut y un tipo con 2-3 en regionales son el mismo
peleador. Sherdog tiene el historial completo, con FECHA por combate — la fecha es lo que
permite contar solo lo anterior al debut en UFC y no filtrar futuro.

Sherdog y no otra fuente, verificado: `robots.txt` es `User-agent: * / Allow: /`, sin
challenge ni content-signals, y la ficha trae el historial en una tabla HTML plana.
Tapology esta descartada por dos razones independientes (Cloudflare activo y
`ai-train=no`), ver DECISIONS.md.

Dos trampas, las dos resueltas aca:
  - las URLs por nombre solo dan 404: el ID es obligatorio. Se resuelve por el buscador,
    pero ANTES se mira el mapa de IDs que se va llenando solo — cada ficha bajada linkea a
    todos sus rivales, asi que el crawl se alimenta a si mismo y ahorra media descarga.
  - el buscador devuelve HOMONIMOS con cara de nada: "Alex Pereira" da cuatro fichas. Por
    eso toda ficha se valida contra peleas de UFC que ya conocemos (fecha + apellido del
    rival). Si no coincide ninguna, no es el peleador y se descarta.
"""

import json
import re
import sys
import time
import unicodedata

import pandas as pd
import requests

from ufc import rutas

RAW = rutas.RAW
CACHE = RAW / "sherdog"
IDS = CACHE / "ids.json"          # clave de nombre -> /fighter/Slug-123456
BUSQUEDAS = CACHE / "busquedas"   # HTML del buscador, ver `_busqueda`
OUT = rutas.DATOS / "sherdog_previo.csv"
BASE = "https://www.sherdog.com"
# UA de navegador: con el UA por defecto de requests la ficha responde igual, pero el
# buscador devuelve paginas vacias.
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
PAUSA = 0.5
REINTENTOS = 4

_FILA = re.compile(r"<tr>(.*?)</tr>", re.S)
_RESULTADO = re.compile(r'final_result (\w+)">')
_RIVAL = re.compile(r'href="/fighter/[^"]*">([^<]+)</a>')
_FECHA = re.compile(r'sub_line">(\w{3}) / (\d{2}) / (\d{4})')
_METODO = re.compile(r'winby"><b>([^<]*)')
_LINK = re.compile(r'href="(/fighter/[^"]+)"[^>]*>([^<]+)</a>')
_RIVAL_LINK = re.compile(r'href="(/fighter/[^"]+)"')
_HISTORIAL = "FIGHT HISTORY - "
_KO = re.compile(r"\bko\b|\btko\b|knockout", re.I)
_SUB = re.compile(r"submission|\bsub\b", re.I)


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).lower().split())


def _clave(s):
    """Solo alfanumerico: ufcstats escribe "AJ Dobson" y "Abdul Rakhman Yakhyaev" donde
    Sherdog escribe "A.J. Dobson" y "Abdulrakhman Yakhyaev". Los puntos y los espacios no
    distinguen a nadie."""
    return re.sub(r"[^a-z0-9]", "", _norm(s))


def _sesion():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _get(sesion, url, params=None):
    """GET con backoff. El 429 no es un error, es un semaforo."""
    for intento in range(REINTENTOS):
        r = sesion.get(url, params=params, timeout=30)
        if r.status_code in (429, 500, 503):  # el 500 del buscador es intermitente
            time.sleep(PAUSA * 4 * (intento + 1))
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        time.sleep(PAUSA)
        return r.text
    return None


def _ficha(sesion, ruta):
    """HTML de la ficha, cacheado en disco por ID. El cache es lo que hace reanudable
    un crawl de 2716 peleadores: si se corta, lo bajado no se vuelve a pedir."""
    f = CACHE / (ruta.rsplit("/", 1)[-1] + ".html")
    if f.exists():
        return f.read_text(encoding="utf-8") or None
    html = _get(sesion, BASE + ruta)
    CACHE.mkdir(parents=True, exist_ok=True)
    f.write_text(html or "", encoding="utf-8")
    return html


def _busqueda(sesion, q):
    """HTML del buscador, cacheado en disco igual que la ficha.

    Sin esto los ~200 peleadores que Sherdog no tiene se re-buscaban en CADA corrida —
    dos requests con pausa y backoff por cabeza, y siempre para volver con las manos
    vacias. Es lo que hacia que una corrida incremental tardara quince minutos.

    ponytail: cachea tambien el fallo, asi que si Sherdog agrega una ficha nueva el cache
    la tapa. `rm -r data/raw/sherdog/busquedas` fuerza la re-busqueda de todos.
    """
    f = BUSQUEDAS / (re.sub(r"[^\w-]", "_", q.lower()) + ".html")
    if f.exists():
        return f.read_text(encoding="utf-8") or None
    html = _get(sesion, BASE + "/stats/fightfinder", {"SearchTxt": q})
    BUSQUEDAS.mkdir(parents=True, exist_ok=True)
    f.write_text(html or "", encoding="utf-8")
    return html


def _buscar(sesion, nombre):
    """-> rutas candidatas con el mismo nombre. SIN validar: eso lo hace `_es_el`.

    El filtro por nombre es obligatorio: la pagina de resultados trae ademas un widget de
    peleadores destacados (Ankalaev, McGregor…) que no tiene nada que ver con la busqueda.

    Segunda vuelta por apellido cuando la primera no da nada: el buscador no tolera la
    diferencia de puntuacion. "AJ Dobson" devuelve CERO resultados; "Dobson" devuelve
    "A.J. Dobson", que es el mismo tipo.
    """
    obj = _clave(nombre)
    consultas = [nombre] + ([nombre.split()[-1]] if len(nombre.split()) > 1 else [])
    for q in consultas:
        html = _busqueda(sesion, q) or ""
        rutas = [r for r, txt in _LINK.findall(html) if _clave(txt) == obj][:3]
        if rutas:
            return rutas
    return []


def _pro(html):
    """Solo la seccion PRO de la ficha, que son tres tablas identicas apiladas.

    Debajo vienen "PRO EXHIBITION" (las peleas de la casa del TUF) y "AMATEUR", que no
    cuentan para el record. Sin este corte quedaba inflado: O'Malley salia 13-1 cuando
    debuto 8-0, y 451 de 2760 fichas no cuadraban con el total que declara la propia
    pagina. Verificado con ese cruce: la seccion PRO sola cuadra.
    """
    partes = (html or "").split(_HISTORIAL)
    for parte in partes[1:]:
        if parte.split("<", 1)[0].strip() == "PRO":
            return parte
    # Hay encabezados pero ninguno es PRO: el tipo solo tiene historial amateur, o sea
    # cero peleas profesionales. Devolver la ficha entera aca las contaria como pro.
    return "" if len(partes) > 1 else (html or "")


def historial(html):
    """-> [(fecha, resultado, rival, metodo)] de la tabla de combates.

    Las filas sin resultado (peleas anunciadas) se caen solas: no tienen `final_result`.
    """
    out = []
    for fila in _FILA.findall(_pro(html)):
        res, riv, fec = _RESULTADO.search(fila), _RIVAL.search(fila), _FECHA.search(fila)
        # El rival puede faltar ("Unknown Fighter" del circuito viejo no lleva link). La
        # pelea cuenta igual: el nombre solo se usa para validar la ficha, y sin nombre
        # simplemente no valida por esa fila.
        if not (res and fec):
            continue
        try:
            fecha = pd.Timestamp(f"{fec.group(1)} {fec.group(2)} {fec.group(3)}")
        except ValueError:
            continue
        met = _METODO.search(fila)
        out.append((fecha, res.group(1).lower(), riv.group(1).strip() if riv else "",
                    met.group(1).strip() if met else ""))
    return out


def _es_el(hist, ufc):
    """La unica validacion que sirve: que la ficha contenga peleas de UFC que ya sabemos.

    El nombre no alcanza — el buscador da cuatro "Alex Pereira" y tres son otra persona.
    Se pide fecha exacta + apellido del rival: una coincidencia basta, cero descarta.
    """
    tengo = {(f.date(), _norm(r).split()[-1]) for f, _, r, _ in hist if r.split()}
    return any((f.date(), _norm(r).split()[-1]) in tengo for f, r in ufc if r.split())


def _links(html):
    """nombre normalizado -> ruta, de todos los rivales linkeados en la ficha.

    Gratis: la ficha ya esta bajada. Es lo que evita 2716 busquedas.
    """
    return {_clave(txt): ruta for ruta, txt in _LINK.findall(html or "")
            if txt.strip() and not txt.strip().isdigit()}


def resolver(sesion, nombre, ufc, ids):
    """-> (ruta, html) de la ficha validada, o (None, None)."""
    def candidatos():
        """Perezoso a proposito: `cands + _buscar(...)` evaluaba el buscador SIEMPRE, aun
        cuando el id cacheado validaba en la primera vuelta. Era una request por peleador
        del catalogo con el cache entero en disco — el crawl "incremental" de dos horas."""
        if _clave(nombre) in ids:
            yield ids[_clave(nombre)]
        yield from _buscar(sesion, nombre)

    for ruta in candidatos():
        html = _ficha(sesion, ruta)
        hist = historial(html)
        if hist and _es_el(hist, ufc):
            ids.update(_links(html))
            return ruta, html
    return None, None


def _es_ko(metodo):
    return bool(_KO.search(metodo))


def _es_sub(metodo):
    """El KO manda: "TKO (Submission to Punches)" es un nocaut, no una sumision — y sin
    esta precedencia la misma pelea se cuenta en las dos columnas."""
    return bool(_SUB.search(metodo)) and not _es_ko(metodo)


def _por_rival(ufc, resueltos):
    """-> ruta del peleador, sacada de la ficha de un rival ya bajado. Cero requests.

    Es lo que rescata las transliteraciones: ufcstats escribe "Aleksei Oleinik" donde
    Sherdog escribe "Alexey Oleynik", y ningun buscador une esos dos strings. Pero la
    ficha del rival tiene la fila de ESA pelea, con la fecha exacta y el link. El nombre
    deja de importar: la pelea ya la conocemos, solo faltaba la URL.
    """
    for fecha, rival in ufc:
        ruta = resueltos.get(rival)
        if not ruta:
            continue
        html = (CACHE / (ruta.rsplit("/", 1)[-1] + ".html"))
        if not html.exists():
            continue
        for fila in _FILA.findall(html.read_text(encoding="utf-8")):
            f = _FECHA.search(fila)
            link = _RIVAL_LINK.search(fila)
            if not (f and link):
                continue
            if pd.Timestamp(f"{f.group(1)} {f.group(2)} {f.group(3)}") == fecha:
                return link.group(1)
    return None


def _previo(hist, debut):
    """Agregados de lo anterior al debut en UFC. Todo lo posterior se ignora: no es
    'lo que no sabiamos', es futuro."""
    ant = [(res, met) for f, res, _, met in hist if f < debut]
    g = [m for r, m in ant if r == "win"]
    p = [m for r, m in ant if r == "loss"]
    return {
        "prev_n": len(ant), "prev_w": len(g), "prev_l": len(p),
        "prev_ko_w": sum(_es_ko(m) for m in g),
        "prev_sub_w": sum(_es_sub(m) for m in g),
        "prev_ko_l": sum(_es_ko(m) for m in p),
        "prev_sub_l": sum(_es_sub(m) for m in p),
    }


def _catalogo():
    """-> {peleador: (debut_ufc, [(fecha, rival)])} desde los CSVs crudos.

    Los rivales conocidos son el material de validacion; el debut, el corte anti-leakage.
    """
    ev = pd.read_csv(RAW / "ufc_event_details.csv")
    res = pd.read_csv(RAW / "ufc_fight_results.csv")
    for d in (ev, res):
        for c in d.columns:
            if pd.api.types.is_string_dtype(d[c]):
                d[c] = d[c].str.strip()
    fechas = dict(zip(ev["EVENT"], pd.to_datetime(ev["DATE"], format="%B %d, %Y",
                                                  errors="coerce")))
    bout = res["BOUT"].str.split(" vs. ", n=1, expand=True)
    res = res.assign(a=bout[0], b=bout[1]).dropna(subset=["b"])
    res["fecha"] = res["EVENT"].map(fechas)
    cat = {}
    for r in res.dropna(subset=["fecha"]).itertuples():
        cat.setdefault(r.a, []).append((r.fecha, r.b))
        cat.setdefault(r.b, []).append((r.fecha, r.a))
    return {n: (min(f for f, _ in v), v) for n, v in cat.items()}


def main():
    cat = _catalogo()
    nombres = sorted(cat)
    if "--muestra" in sys.argv:
        nombres = nombres[:int(sys.argv[sys.argv.index("--muestra") + 1])]
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = json.loads(IDS.read_text()) if IDS.exists() else {}
    sesion = _sesion()

    filas, fallados = [], []
    for i, nombre in enumerate(nombres, 1):
        debut, ufc = cat[nombre]
        try:
            ruta, html = resolver(sesion, nombre, ufc, ids)
        except requests.RequestException as e:
            print(f"  {nombre}: {e}", flush=True)
            ruta = None
        if not ruta:
            fallados.append(nombre)
        else:
            filas.append({"peleador": nombre, "url": ruta,
                          "debut_ufc": debut.date(), **_previo(historial(html), debut)})
        if i % 50 == 0:
            IDS.write_text(json.dumps(ids))
            pd.DataFrame(filas).to_csv(OUT, index=False)  # parcial: el crawl es largo
            print(f"  {i}/{len(nombres)} — {len(filas)} ok, {len(fallados)} sin ficha",
                  flush=True)

    # Segunda pasada por la ficha de los rivales, para los que el buscador no encontro.
    # No pide red: usa lo ya cacheado. Ver `_por_rival`.
    resueltos = {f["peleador"]: f["url"] for f in filas}
    rescatados = 0
    for nombre in list(fallados):
        debut, ufc = cat[nombre]
        ruta = _por_rival(ufc, resueltos)
        if not ruta:
            continue
        hist = historial(_ficha(sesion, ruta))
        if not (hist and _es_el(hist, ufc)):
            continue
        filas.append({"peleador": nombre, "url": ruta, "debut_ufc": debut.date(),
                      **_previo(hist, debut)})
        # bajo el nombre de ufcstats, que es con el que se va a volver a buscar: asi la
        # proxima corrida no repite la busqueda condenada de "Aleksei Oleinik"
        ids[_clave(nombre)] = ruta
        fallados.remove(nombre)
        rescatados += 1
    print(f"segunda pasada (fichas de rivales): +{rescatados}")

    IDS.write_text(json.dumps(ids))
    df = pd.DataFrame(filas)
    df.to_csv(OUT, index=False)
    print(f"{OUT}: {len(df)}/{len(nombres)} peleadores resueltos "
          f"({len(df[df.prev_n > 0]) if len(df) else 0} con peleas pre-UFC)")
    if len(df):
        print(f"  peleas previas: mediana {df.prev_n.median():.0f}, "
              f"max {df.prev_n.max()} | sin ficha: {len(fallados)}")


def autocheck():
    """Sin red: HTML congelado. Si alguien afloja la validacion, el homonimo entra."""
    fila = ('<tr><td><span class="final_result win">win</span></td>'
            '<td><a href="/fighter/Ciryl-Gane-293973">Ciryl Gane</a></td>'
            '<td><a href="/events/X-1">UFC 300</a><br /><span class="sub_line">'
            'Mar / 07 / 2020</span></td>'
            '<td class="winby"><b>TKO (Punches)</b><br /></td><td>2</td><td>1:27</td></tr>')
    vieja = fila.replace("Ciryl Gane", "Juan Perez").replace("Mar / 07 / 2020",
                                                             "Jan / 05 / 2015")
    h = historial(fila + vieja)
    assert len(h) == 2, h
    assert h[0][:3] == (pd.Timestamp("2020-03-07"), "win", "Ciryl Gane"), h[0]

    ufc = [(pd.Timestamp("2020-03-07"), "Ciryl Gane")]
    assert _es_el(h, ufc)
    assert not _es_el(h, [(pd.Timestamp("2021-03-07"), "Ciryl Gane")]), \
        "otra fecha con el mismo rival no puede validar"
    assert not _es_el(h, [(pd.Timestamp("2020-03-07"), "Otro Tipo")]), \
        "misma fecha con otro rival no puede validar: es el homonimo"

    # rescate por la ficha del rival: la fecha manda, el nombre no participa
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "Rival-Falso-1.html").write_text(fila + vieja, encoding="utf-8")
    assert _por_rival(ufc, {"Ciryl Gane": "/fighter/Rival-Falso-1"}) == \
        "/fighter/Ciryl-Gane-293973"
    assert _por_rival([(pd.Timestamp("2019-01-01"), "Ciryl Gane")],
                      {"Ciryl Gane": "/fighter/Rival-Falso-1"}) is None, \
        "sin fila en esa fecha no hay rescate"
    (CACHE / "Rival-Falso-1.html").unlink()

    p = _previo(h, pd.Timestamp("2020-03-07"))
    assert (p["prev_n"], p["prev_w"], p["prev_ko_w"]) == (1, 1, 1), p

    # ni la tabla amateur ni la de exhibicion son record profesional
    otra = lambda quien, f: fila.replace("Ciryl Gane", quien).replace("Mar / 07 / 2020", f)
    ficha = (_HISTORIAL + "PRO</div>" + fila + vieja +
             _HISTORIAL + "PRO EXHIBITION</div>" + otra("Un TUF", "Feb / 02 / 2013") +
             _HISTORIAL + "AMATEUR</div>" + otra("Un Amateur", "Feb / 02 / 2014"))
    assert len(historial(ficha)) == 2, "entraron peleas que no son del record pro"
    assert len(historial(fila + vieja)) == 2, "ficha sin encabezados: no se pierde nada"

    # un metodo no puede contarse como KO y como sumision a la vez
    tko = fila.replace("TKO (Punches)", "TKO (Submission to Punches)")
    q = _previo(historial(tko), pd.Timestamp("2020-03-08"))
    assert (q["prev_ko_w"], q["prev_sub_w"]) == (1, 0), q
    # la pelea del debut no cuenta como previa
    assert _previo(h, pd.Timestamp("2015-01-05"))["prev_n"] == 0
    print("autocheck ok")


if __name__ == "__main__":
    if "--autocheck" in sys.argv:
        autocheck()
    else:
        main()
