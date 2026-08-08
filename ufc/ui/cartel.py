"""El cartel del evento como imagen, y el clic sobre cada pelea.

Dos mitades. Abajo, `generar` compone un PNG con Pillow al estilo del cartel oficial de
la UFC: la foto de cada peleador sobre un recuadro oliva, su bandera en la esquina de su
lado, los apellidos, el peso en castellano y la barra del modelo. Arriba, un componente
CCv2 que dibuja ese PNG con un boton transparente encima de cada pelea, que es lo unico
que convierte al cartel en navegacion — Streamlit no tiene regiones clickeables sobre una
imagen.

Por que una imagen y no HTML, que es lo que habia antes: el cartel se comparte. Un PNG se
arrastra a un chat; catorce divs con estilos inline no.

El PNG se cachea por contenido en `static/carteles/<evento>-<hash>.png`. El hash sale de
lo que se dibuja (peleas, pesos, fotos, probabilidades), asi que una cartelera que no
cambio no se vuelve a componer, una que cambio se recompone sola, y borrar `static/` no
rompe nada: se regenera. Al escribir uno nuevo se borran los del mismo evento, que si no
quedarian uno por cada vez que Betano movio un precio.
"""

import datetime
import functools
import hashlib
import pathlib
import re

import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from ufc import rutas
from ufc.datos import fotos
from ufc.ui.comunes import COLOR_A, COLOR_B

# El lienzo. Ancho fijo y alto calculado: el cartel se muestra al 100% del ancho de la
# pagina, asi que lo unico que importa es la proporcion.
ANCHO = 1280
MARGEN = 44
FONDO = "#0B0E14"          # el backgroundColor del tema
TEXTO = "#E7EAF0"
GRIS = "#8A94A6"
# El oliva del cartel oficial. No sale del tema: es el color del cartel, no de la app.
OLIVA = "#6E7B4F"
OLIVA_VACIO = "#57603F"    # el mismo, mas apagado, para el que no tiene foto
# Que parte de la foto entra en el recuadro, de arriba para abajo. Calibrado contra los
# recortes de ufc.com, que son 460x700 de la cabeza al muslo: con 0.5 el encuadre queda
# en cabeza y torso, como el cartel oficial. Mas chico acerca la cara, mas grande la aleja.
VISIBLE = 0.5

LOGO = (300, 108)
MAIN = (300, 360)          # tile del main event
CHICO = (176, 212)         # tile del resto de la cartelera
HUECO = 12                 # entre los dos tiles de un par
COLUMNA = 40               # entre pares de una misma fila
FILA = 26                  # entre filas
POR_FILA = 3
PIE_MAIN, PIE_CHICO = 96, 70
FECHA = 62

_MES = ("ENE", "FEB", "MAR", "ABR", "MAY", "JUN",
        "JUL", "AGO", "SEP", "OCT", "NOV", "DIC")
_DIA = ("LUN", "MAR", "MIÉ", "JUE", "VIE", "SÁB", "DOM")

# La condensada es la que da el aire de cartel de pelea; las otras dos son para que la
# app siga generando el cartel en la VPS de `deploy/`, donde no hay fuentes de macOS.
# ponytail: sin ninguna de las tres cae al bitmap de Pillow — el cartel se ve peor, pero
# se ve. Si eso molesta, versionar un .ttf en `assets/` y ponerlo primero en la lista.
_FUENTES = ("/System/Library/Fonts/Supplemental/Arial Narrow Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf")

# ESPN manda la division en ingles y abreviada ("W Strawweight"). El cartel oficial la
# escribe como la dice el comentarista.
_PARTICULAS = {"de", "del", "la", "las", "los", "da", "das", "do", "dos", "van", "von",
               "di", "le", "el", "al", "bin", "ben", "mac", "mc", "st.", "santa"}

PESOS = {"strawweight": "paja", "flyweight": "mosca", "bantamweight": "gallo",
         "featherweight": "pluma", "lightweight": "ligero", "welterweight": "wélter",
         "middleweight": "medio", "light heavyweight": "semicompleto",
         "heavyweight": "completo", "catchweight": "pactado"}


@functools.lru_cache(maxsize=None)
def _fuente(tam):
    for ruta in _FUENTES:
        try:
            return ImageFont.truetype(ruta, tam)
        except OSError:
            continue
    return ImageFont.load_default(tam)


def _ajustar(dibujo, texto, tam, limite, minimo=9):
    """La fuente mas grande con la que `texto` entra en `limite` pixeles."""
    while tam > minimo and dibujo.textlength(texto, font=_fuente(tam)) > limite:
        tam -= 1
    return _fuente(tam)


def peso_es(peso):
    """'W Strawweight' -> 'COMBATE DE PESO PAJA FEMENINO'. Sin match, el texto de ESPN."""
    p = str(peso or "").strip()
    if not p:
        return ""
    base = re.sub(r"^(w|women's|women)\s+", "", p, flags=re.I)
    femenino = base.lower() != p.lower()
    nombre = PESOS.get(base.lower())
    if not nombre:
        return f"COMBATE DE {p}".upper()
    return f"COMBATE DE PESO {nombre}{' femenino' if femenino else ''}".upper()


def apellido(nombre):
    """El cartel oficial escribe apellidos: 'Billy Ray Goff' -> 'GOFF'.

    Las particulas van pegadas al apellido y no se tiran: 'Yadier del Valle' es
    'DEL VALLE', que es como lo escribe el cartel oficial, y no 'VALLE'.

    Dos peleadores con el mismo apellido en la misma cartelera quedan ambiguos en el
    cartel; el nombre entero esta en el modal, que es donde se lee el analisis.
    """
    partes = str(nombre).split()
    if not partes:
        return str(nombre).upper()
    corte = len(partes) - 1
    while corte > 1 and partes[corte - 1].lower() in _PARTICULAS:
        corte -= 1
    return " ".join(partes[corte:]).upper()


def _fecha_es(fecha):
    try:
        d = datetime.date.fromisoformat(str(fecha)[:10])
    except ValueError:
        return str(fecha).upper()
    return f"{d.day} DE {_MES[d.month - 1]} {_DIA[d.weekday()]}"


def _plano(n):
    """La geometria del cartel para `n` peleas, sin dibujar nada.

    Va separada del dibujo porque las cajas clickeables hacen falta siempre, tambien
    cuando el PNG ya estaba en disco y no hay que componerlo de nuevo.

    -> (alto, [(x, y, ancho, alto, grande)]), un bloque por pelea y en su mismo orden.
    """
    bloques = []
    y = MARGEN + LOGO[1] + 30
    if n:
        ancho = MAIN[0] * 2 + HUECO
        bloques.append(((ANCHO - ancho) // 2, y, ancho, MAIN[1] + PIE_MAIN, True))
        y += MAIN[1] + PIE_MAIN + 34

    ancho = CHICO[0] * 2 + HUECO
    x0 = (ANCHO - (ancho * POR_FILA + COLUMNA * (POR_FILA - 1))) // 2
    for i in range(1, n):
        columna = (i - 1) % POR_FILA
        bloques.append((x0 + columna * (ancho + COLUMNA), y, ancho,
                        CHICO[1] + PIE_CHICO, False))
        if columna == POR_FILA - 1:
            y += CHICO[1] + PIE_CHICO + FILA
    if n > 1 and (n - 1) % POR_FILA:
        y += CHICO[1] + PIE_CHICO + FILA
    return y - FILA + 30 + FECHA + MARGEN, bloques


def _recursos(peleas):
    """(foto_a, bandera_a, foto_b, bandera_b) local de cada pelea. Baja lo que falte.

    Se resuelve antes de la firma y no adentro del dibujo: asi el hash del archivo
    incluye que foto y que bandera se consiguieron, y el peleador que la UFC recien
    fotografio deja de quedarse con el monograma hasta que cambie la cartelera.
    """
    return [tuple(x for lado in ("a", "b")
                  for x in (fotos.ruta(p[lado]),
                            fotos.bandera(p.get(f"bandera_{lado}", ""))))
            for p in peleas]


def _tile(ancho, alto, nombre, foto, bandera, izquierda):
    """Un peleador: el recuadro oliva con su foto y su bandera en su esquina."""
    tile = Image.new("RGBA", (ancho, alto), (0, 0, 0, 0))
    d = ImageDraw.Draw(tile)
    d.rounded_rectangle((0, 0, ancho - 1, alto - 1), radius=10,
                        fill=OLIVA if foto else OLIVA_VACIO)
    if foto:
        try:
            cara = Image.open(foto).convert("RGBA")
        except OSError:
            cara = None
        if cara:
            # El recorte de ufc.com va de la cabeza al muslo y el cartel oficial encuadra
            # cabeza y torso: se agranda hasta que la mitad de arriba de la foto llene el
            # recuadro y se recorta al centro lo que sobra de ancho. Sin esto la cara sale
            # a la mitad del tamanio del cartel real. El `max` es para la foto de pesaje
            # de Sherdog, que viene mas angosta y si no dejaria el oliva a los costados.
            escala = max(alto / (cara.height * VISIBLE), ancho / cara.width)
            medida = (round(cara.width * escala), round(cara.height * escala))
            cara = cara.resize(medida, Image.LANCZOS)
            x = (medida[0] - ancho) // 2
            tile.alpha_composite(cara.crop((x, 0, x + ancho, alto)))
    else:
        # Las iniciales, para el debutante que no esta ni en ufc.com ni en Sherdog.
        iniciales = "".join(p[0] for p in str(nombre).split()[:2]).upper() or "?"
        d.text((ancho / 2, alto / 2), iniciales, font=_fuente(alto // 3),
               fill="#C9D2B4", anchor="mm")

    if bandera:
        try:
            f = Image.open(bandera).convert("RGBA")
        except OSError:
            f = None
        if f:
            ancho_f = round(ancho * 0.30)
            f = f.resize((ancho_f, max(round(f.height * ancho_f / f.width), 1)),
                         Image.LANCZOS)
            x = 6 if izquierda else ancho - ancho_f - 6
            tile.alpha_composite(f, (x, alto - f.height - 6))

    # La mascara va al final: la foto tapa las esquinas redondeadas del recuadro.
    mascara = Image.new("L", (ancho, alto), 0)
    ImageDraw.Draw(mascara).rounded_rectangle((0, 0, ancho - 1, alto - 1), radius=10,
                                              fill=255)
    tile.putalpha(mascara)
    return tile


def _barra(d, x, y, ancho, r, grande):
    """La barra del modelo con los dos porcentajes en las puntas."""
    fuente = _fuente(17 if grande else 13)
    izq, der = f"{r['p_a']:.0%}", f"{r['p_b']:.0%}"
    grosor = 8 if grande else 6
    d.text((x, y), izq, font=fuente, fill=COLOR_A, anchor="lm")
    d.text((x + ancho, y), der, font=fuente, fill=COLOR_B, anchor="rm")
    x0 = round(x + d.textlength(izq, font=fuente) + 8)
    x1 = round(x + ancho - d.textlength(der, font=fuente) - 8)
    # Los extremos igual tienen que verse: 3% de ancho minimo por lado, como la barra HTML.
    corte = round(x0 + (x1 - x0) * min(max(r["p_a"], 0.03), 0.97))
    arriba, abajo = round(y - grosor / 2), round(y + grosor / 2)
    d.rounded_rectangle((x0, arriba, corte, abajo), radius=grosor // 2, fill=COLOR_A)
    d.rounded_rectangle((corte + 2, arriba, x1, abajo), radius=grosor // 2, fill=COLOR_B)


def _bloque(base, d, pelea, r, caja, recurso):
    """Una pelea entera: los dos tiles, los apellidos, el peso y la barra."""
    x, y, ancho, _, grande = caja
    lado = MAIN if grande else CHICO
    for i, sigla in enumerate(("a", "b")):
        tile = _tile(*lado, pelea[sigla], recurso[i * 2], recurso[i * 2 + 1],
                     sigla == "a")
        base.alpha_composite(tile, (x + i * (lado[0] + HUECO), y))

    cy = y + lado[1] + (28 if grande else 20)
    nombres = f"{apellido(pelea['a'])} VS {apellido(pelea['b'])}"
    d.text((x + ancho / 2, cy), nombres,
           font=_ajustar(d, nombres, 38 if grande else 22, ancho - 8),
           fill=TEXTO, anchor="mm")

    cy += 26 if grande else 17
    if texto := peso_es(pelea.get("peso")):
        d.text((x + ancho / 2, cy), texto,
               font=_ajustar(d, texto, 19 if grande else 12, ancho - 8),
               fill=GRIS, anchor="mm")

    # Un debut no tiene prediccion: el cartel va sin barra, no sin pelea.
    if r and "error" not in r:
        _barra(d, x + 20, cy + (30 if grande else 20), ancho - 40, r, grande)


def _logo(base, d):
    """El sello UFC arriba de todo, en su caja blanca."""
    ancho, alto = LOGO
    x = (ANCHO - ancho) // 2
    d.rounded_rectangle((x, MARGEN, x + ancho, MARGEN + alto), radius=4, fill="#FFFFFF")
    d.text((ANCHO / 2, MARGEN + 36), "UFC", font=_fuente(52), fill="#000000", anchor="mm")
    d.text((ANCHO / 2, MARGEN + 82), "FIGHT NIGHT", font=_fuente(28), fill="#000000",
           anchor="mm")


def _dibujar(destino, evento, peleas, preds, recursos, alto, bloques):
    base = Image.new("RGBA", (ANCHO, alto), FONDO)
    d = ImageDraw.Draw(base)
    _logo(base, d)
    for pelea, r, caja, recurso in zip(peleas, preds, bloques, recursos):
        _bloque(base, d, pelea, r, caja, recurso)

    texto = _fecha_es(evento.get("fecha"))
    y = alto - MARGEN - FECHA
    d.rounded_rectangle((MARGEN, y, ANCHO - MARGEN, y + FECHA), radius=6,
                        outline="#FFFFFF", width=2)
    d.text((ANCHO / 2, y + FECHA / 2), texto, font=_fuente(34), fill=TEXTO, anchor="mm")

    destino.parent.mkdir(parents=True, exist_ok=True)
    base.convert("RGB").save(destino, "PNG", optimize=True)


def _codigo():
    """El hash de este archivo, que entra en el nombre del PNG.

    Sin esto, tocar el layout no invalida los carteles ya escritos y la app sigue
    sirviendo el viejo hasta que cambie la cartelera — o sea, nunca en un evento pasado.
    """
    try:
        return hashlib.sha1(pathlib.Path(__file__).read_bytes()).hexdigest()[:6]
    except OSError:
        return "0"


def _slug(texto):
    return re.sub(r"[^a-z0-9]+", "-", str(texto).lower()).strip("-")[:48] or "evento"


def generar(evento, preds=()):
    """-> (archivo del cartel, cajas de cada pelea en fraccion del PNG)."""
    peleas = evento["peleas"]
    preds = list(preds) + [None] * (len(peleas) - len(preds))
    recursos = _recursos(peleas)
    alto, bloques = _plano(len(peleas))
    firma = [(p["a"], p["b"], p.get("peso"), str(rec),
              None if not r or "error" in r else round(r["p_a"], 3))
             for p, r, rec in zip(peleas, preds, recursos)]
    firma = hashlib.sha1(
        repr((_codigo(), evento.get("fecha"), firma)).encode()).hexdigest()[:10]

    slug = _slug(evento["evento"])
    destino = rutas.CARTELES / f"{slug}-{firma}.png"
    if not destino.exists():
        _dibujar(destino, evento, peleas, preds, recursos, alto, bloques)
        for viejo in rutas.CARTELES.glob(f"{slug}-*.png"):
            if viejo != destino:
                viejo.unlink()
    cajas = [(x / ANCHO, y / alto, w / ANCHO, h / alto) for x, y, w, h, _ in bloques]
    return destino, cajas


# --- el clic -----------------------------------------------------------------------

_HTML = "<div id='cartel'></div>"
_CSS = """
#cartel { position: relative; line-height: 0; }
#cartel img { width: 100%; height: auto; display: block; border-radius: 14px; }
#cartel button {
  position: absolute; padding: 0; background: transparent; cursor: pointer;
  border: 2px solid transparent; border-radius: 12px;
  transition: border-color .15s, background .15s;
}
#cartel button:hover, #cartel button:focus-visible {
  border-color: var(--st-primary-color, #2563EB);
  background: color-mix(in srgb, var(--st-primary-color, #2563EB) 16%, transparent);
}
"""
_JS = """
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const raiz = parentElement.querySelector("#cartel")
  if (!raiz || !data) return
  // La imagen solo se rearma cuando cambia el cartel: si no, cada rerun de Streamlit la
  // haria parpadear entera.
  if (raiz.dataset.src !== data.src) {
    raiz.dataset.src = data.src
    raiz.replaceChildren()
    const img = document.createElement("img")
    img.src = data.src
    img.alt = data.alt || ""
    raiz.appendChild(img)
    for (const c of data.cajas || []) {
      const b = document.createElement("button")
      b.type = "button"
      b.style.left = c[0] * 100 + "%"
      b.style.top = c[1] * 100 + "%"
      b.style.width = c[2] * 100 + "%"
      b.style.height = c[3] * 100 + "%"
      b.setAttribute("aria-label", c[4])
      b.title = c[4]
      raiz.appendChild(b)
    }
  }
  // Los handlers se re-atan en cada run aunque la imagen no se haya tocado: el
  // `setTriggerValue` que vale es el de este run, no el que quedo en el closure viejo.
  raiz.querySelectorAll("button").forEach((b, i) => {
    b.onclick = () => setTriggerValue("pelea", i)
  })
}
"""

def mostrar(evento, preds=(), key="cartel"):
    """Dibuja el cartel y -> el indice de la pelea clickeada, o None."""
    archivo, cajas = generar(evento, preds)
    datos = {"src": f"/app/static/carteles/{archivo.name}",
             "alt": f"Cartel de {evento['evento']}",
             "cajas": [[*caja, f"Ver análisis de {p['a']} vs {p['b']}"]
                       for caja, p in zip(cajas, evento["peleas"])]}
    # El registro va aca y no en el import: el registro de componentes vive en el runtime
    # de Streamlit y el import del modulo pasa una sola vez, asi que un runtime nuevo se
    # queda sin componente (es lo que le pasa a cada AppTest de la suite). Volver a
    # registrar la misma definicion no cuesta ni avisa nada: el registry compara y sale.
    componente = st.components.v2.component("cartel_ufc", html=_HTML, css=_CSS, js=_JS)
    return componente(key=key, data=datos, on_pelea_change=lambda: None).pelea
