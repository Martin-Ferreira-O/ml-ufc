"""La foto de cada peleador: el recorte oficial de ufc.com, Sherdog como red.

Por que dos fuentes y por que en ese orden. ufc.com publica el mismo PNG recortado con
fondo transparente que la UFC usa en sus carteleras, que es exactamente lo que pide el
cartel; pero solo tiene al roster, y el que todavia no firmo no tiene ficha. Sherdog si
lo tiene — y su HTML ya esta en disco, bajado por el crawl del record pre-UFC de
`datos/sherdog.py`, asi que la segunda fuente no cuesta una request extra: es un regex
sobre un archivo local.

Medido sobre las 10 carteleras anunciadas, 164 peleadores: 135 de ufc.com y 7 de
Sherdog. Los 24 restantes son debutantes — el que todavia no peleo en UFC no esta en el
catalogo de ufcstats, que es de donde salio el cache de Sherdog, asi que no lo tiene
ninguna de las dos fuentes. El UI los dibuja con el monograma, que es la respuesta
honesta: buscarlos a ciegas en Sherdog devuelve homonimos y no hay con que validarlos
(`sherdog._es_el` valida contra peleas de UFC, y un debutante no tiene).

Tres trampas, las tres resueltas aca:
  - ufc.com contesta 200 con una pagina de busqueda ante cualquier slug inventado. El
    status code no distingue al que no esta; la ausencia del <img> si.
  - las fichas sin foto no vienen vacias: traen una silueta generica. Ver `_SOMBRA`.
  - un GET pelado contra ufc.com da 403. Pasa con el UA de intel y siguiendo el redirect
    regional (ufc.com -> ufcespanol.com), que es lo que `requests` hace por defecto.
"""

import functools
import html
import re
import sys
import time

import pandas as pd
import requests

from ufc import nombres, rutas
from ufc.datos import cartelera, sherdog

CACHE = rutas.FOTOS
ATLETA = "https://www.ufc.com/athlete/{slug}"
# El mismo UA que `intel/identities.py` ya usa contra esta pagina. Con el de requests da
# 403; este pasa.
UA = "ml-ufc-intel/1.0"
PAUSA = 0.3
_CUERPO = re.compile(r'"(https://ufc\.com/images/styles/athlete_bio_full_body/[^"]+)"')
_OG = re.compile(r'og:image"\s+content="([^"]+)"')
_MEDIDA = re.compile(r"/image_crop/\d+/\d+/")
# El roster tiene fichas sin foto, y ufc.com no las deja vacias: sirve una silueta
# generica (`SHADOW_Fighter_fullLength_RED.png`). Sin este filtro entraban como foto
# buena y cinco peleadores distintos compartian el mismo PNG de sombra — verificado por
# md5. Descartarla es lo que deja que Sherdog o el monograma tomen el lugar.
_SOMBRA = re.compile(r"SHADOW_Fighter", re.I)


def _clave(nombre):
    """La clave canonica, que es a la vez el slug de ufc.com y el nombre del archivo.

    Una sola funcion para las dos cosas porque dan el mismo string: verificado contra
    `identities._slug`, que arma el slug de ufc.com con el mismo `normalizar` de base.
    """
    return re.sub(r"[^a-z0-9]+", "-", nombres.normalizar(nombre)).strip("-")


def _ext(url):
    """ufc.com manda PNG con transparencia, Sherdog JPEG. Guardar el JPEG con nombre
    `.png` es de las cosas que muerden seis meses despues."""
    return ".png" if url.split("?")[0].lower().endswith(".png") else ".jpg"


def ruta(nombre):
    """-> el archivo local de la foto, o None si todavia no se bajo."""
    clave = _clave(nombre)
    for ext in (".png", ".jpg"):
        f = CACHE / (clave + ext)
        if f.exists():
            return f
    return None


def _sesion():
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _url_ufc(sesion, nombre):
    """-> URL del recorte oficial, o None si ufc.com no tiene al peleador."""
    try:
        r = sesion.get(ATLETA.format(slug=_clave(nombre)), timeout=30)
    except requests.RequestException:
        return None
    time.sleep(PAUSA)
    # Sin `r.ok` igual andaria: la pagina de busqueda que devuelve por un slug que no
    # existe tampoco trae el <img>. Se chequea igual para no depender de eso.
    m = _CUERPO.search(r.text) if r.ok else None
    if not m or _SOMBRA.search(m.group(1)):
        return None
    # El src viene escapado como HTML: `?VersionId=…&amp;itok=…`. Pedirlo asi devuelve
    # otra cosa o un 403, porque el token deja de leerse como parametro aparte.
    return html.unescape(m.group(1))


def foto_ficha(html):
    """-> URL de la foto en una ficha de Sherdog, o None.

    Se pide el recorte 400x600 y no el 300x300 que trae el `og:image` para que la caja
    coincida con la del recorte oficial de ufc.com; Sherdog sirve cualquier medida en esa
    misma ruta.
    """
    m = _OG.search(html or "")
    # Las fichas sin foto no traen un placeholder: traen el apple-touch-icon del sitio.
    if not m or "/_images/fighter/" not in m.group(1):
        return None
    return sherdog.BASE + _MEDIDA.sub("/image_crop/400/600/", m.group(1))


@functools.lru_cache(maxsize=1)
def _fichas():
    """clave del peleador -> ruta de su ficha en Sherdog, desde el CSV del record previo."""
    if not sherdog.OUT.exists():
        return {}
    df = pd.read_csv(sherdog.OUT)
    return {_clave(p): u for p, u in zip(df["peleador"], df["url"])}


def _url_sherdog(nombre):
    """La foto de la ficha ya bajada. Cero requests: el HTML esta en disco."""
    ficha = _fichas().get(_clave(nombre))
    if not ficha:
        return None
    f = sherdog.CACHE / (ficha.rsplit("/", 1)[-1] + ".html")
    return foto_ficha(f.read_text(encoding="utf-8", errors="ignore")) if f.exists() else None


def _bajar(sesion, nombre):
    """-> ('ufc'|'sherdog'|None). Deja la foto en disco, o la marca como inexistente."""
    clave = _clave(nombre)
    url = _url_ufc(sesion, nombre)
    fuente = "ufc"
    if not url:
        url, fuente = _url_sherdog(nombre), "sherdog"
    if not url:
        # ponytail: cachea el fallo, igual que `sherdog._busqueda`. Sin esto cada rerun
        # de Streamlit vuelve a preguntarle a ufc.com por los debutantes que nunca van a
        # estar. `rm static/fotos/*.miss` fuerza el reintento cuando la UFC los firma.
        (CACHE / (clave + ".miss")).touch()
        return None
    try:
        r = sesion.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        # Sin `.miss`: el peleador existe y la foto tambien, fallo la red. Marcarlo aca
        # lo dejaria sin foto para siempre por un timeout.
        print(f"  {nombre}: {e}", flush=True)
        return None
    (CACHE / (clave + _ext(url))).write_bytes(r.content)
    time.sleep(PAUSA)
    return fuente


def bandera(url):
    """-> el PNG local de la bandera que manda ESPN, o None si no hay URL o no se pudo.

    Son ~40 paises para todo el calendario y el archivo es de 2 KB: se baja una vez y
    despues el cartel lo abre de disco. Sin `.miss` como en las fotos — aca el fallo solo
    puede ser de red, y ESPN nunca sirve una bandera generica.
    """
    if not url:
        return None
    f = rutas.BANDERAS / url.split("?")[0].rsplit("/", 1)[-1]
    if f.exists():
        return f
    try:
        r = _sesion().get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"  bandera {url}: {e}", flush=True)
        return None
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(r.content)
    return f


def sincronizar(peleadores):
    """-> {peleador: ruta local o None}. Baja solo lo que falta."""
    pendientes = [n for n in peleadores
                  if not ruta(n) and not (CACHE / (_clave(n) + ".miss")).exists()]
    if pendientes:
        CACHE.mkdir(parents=True, exist_ok=True)
        sesion = _sesion()
        for nombre in pendientes:
            _bajar(sesion, nombre)
    return {n: ruta(n) for n in peleadores}


def main():
    peleadores = sorted({pelea[lado] for evento in cartelera.proximas()
                         for pelea in evento["peleas"] for lado in ("a", "b")})
    print(f"{len(peleadores)} peleadores en las proximas carteleras")
    fotos = sincronizar(peleadores)
    con = [n for n, f in fotos.items() if f]
    print(f"{CACHE}: {len(con)}/{len(peleadores)} con foto")
    for n in sorted(fotos):
        if not fotos[n]:
            print(f"  sin foto: {n}")


def autocheck():
    """Sin red: HTML congelado de las dos fuentes."""
    from ufc.intel import identities
    for nombre in ("Ilia Topuria", "José Montanha", "Billy Ray Goff", "A.J. Dobson"):
        assert _clave(nombre) == identities._slug(nombre), nombre

    ufc = ('<img src="https://ufc.com/images/styles/athlete_bio_full_body/s3/2025-06/'
           'TOPURIA_ILIA_L_06-28.png?itok=DRPWPNCZ" alt="">')
    assert _CUERPO.search(ufc).group(1).endswith("itok=DRPWPNCZ")
    # la pagina de un slug inventado trae headshots de otros, nunca el cuerpo entero
    otro = ('<img src="https://ufc.com/images/styles/event_results_athlete_headshot/'
            's3/2025-06/TOPURIA_ILIA_06-28.png?itok=1gcxKbmd">')
    assert _CUERPO.search(otro) is None, "el headshot no es el recorte del cartel"
    # la silueta generica no es la foto de nadie: cinco peleadores compartian este PNG
    sombra = ('"https://ufc.com/images/styles/athlete_bio_full_body/s3/image/'
              'fighter_images/SHADOW_Fighter_fullLength_RED.png?itok=DfVddCfn"')
    assert _SOMBRA.search(_CUERPO.search(sombra).group(1)), "la sombra tiene que caer"
    assert html.unescape("a?VersionId=x&amp;itok=y") == "a?VersionId=x&itok=y"

    ficha = ('<meta property="og:image" content="/image_crop/300/300/_images/fighter/'
             '20220401035313_Arman_Tsarukyan_ff.JPG" />')
    assert foto_ficha(ficha) == ("https://www.sherdog.com/image_crop/400/600/_images/"
                                 "fighter/20220401035313_Arman_Tsarukyan_ff.JPG")
    sin = '<meta property="og:image" content="https://www2-cdn.sherdog.com/apple-touch-icon.png" />'
    assert foto_ficha(sin) is None, "el icono del sitio no es la foto del peleador"
    assert foto_ficha("") is None

    assert _ext("https://ufc.com/x/Y.png?itok=abc") == ".png"
    assert _ext("https://www.sherdog.com/image_crop/400/600/_images/fighter/x.JPG") == ".jpg"
    print("autocheck ok")


if __name__ == "__main__":
    if "--autocheck" in sys.argv:
        autocheck()
    else:
        main()
