"""Bot de Telegram que narra la cartelera en vivo y contesta comandos.

Existe para que la capa de IA se pueda mirar a la cara. `ufc/ia/evaluar.py` ya dice si
la IA le gana al mercado con IC95%, pero hay que abrir una terminal para leerlo, y para
cuando lo leas ya no te acordas de que dijo antes de la pelea. Un tipster se juzga en
vivo: acerto o no acerto, y se entera todo el mundo al toque.

**Sin libreria de Telegram.** La Bot API es HTTP plano y `requests` ya esta en el
proyecto; `getUpdates` con `timeout=30` hace de long polling Y de reloj del bucle, asi
que tampoco hay scheduler. Un solo proceso, un solo archivo.

**No necesita el modelo.** Todo lo que muestra sale de los CSV que el repo versiona
(`ia_consenso.csv`, `picks.csv`, `resultados.csv`), no de `data/model.pkl`, que esta
gitignoreado y no existe en la VPS. La probabilidad del modelo y la del mercado ya vienen
congeladas como columnas en `ia_consenso.csv`, que es justo para lo que se guardaron.

**Los picks viajan por git.** Los genera tu maquina (necesita el modelo entrenado) y el
bot los lee de la rama remota. Si te olvidaste de pushear, lo dice en vez de callarse.

    python -m ufc.tipster --seco     # una vuelta, imprime y no manda nada
    python -m ufc.tipster            # el bucle, con TELEGRAM_BOT_TOKEN en el entorno
"""

import argparse
import datetime
import html
import json
import os
import subprocess
import sys
import time

import pandas as pd
import requests

from ufc import nombres, rutas
from ufc.datos import cartelera, resultados
from ufc.ia import dossier, evaluar as ia_evaluar, store as ia_store
from ufc.modelo import gate
from ufc.registro import apuestas, predictores

# `predictores` corre `cargar_env()` al importarse, asi que el .env ya esta cargado.
API = "https://api.telegram.org/bot{token}/{metodo}"
ESTADO = rutas.DATOS / "tipster.json"
ESPERA = 30          # segundos de long polling; tambien es el tick del bucle
PREVIA = 3 * 3600    # el preview sale 3 horas antes del primer campanazo
ANTES = 4 * 3600     # ventana de narracion: desde 4h antes...
DESPUES = 12 * 3600  # ...hasta 12h despues del inicio
MEMORIA = 500        # avisos recordados; una cartelera son ~15
METODO = {"ko": "KO/TKO", "sub": "sumisión", "dec": "decisión"}


# --------------------------------------------------------------------- Telegram

def _token():
    # `TOKEN_BOT` es el nombre que ya tiene el .env de esta maquina. Aceptar los dos sale
    # mas barato que migrar un secreto que esta en dos lados.
    return (os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TOKEN_BOT") or "").strip()


def chats():
    """-> [chat_id] con permiso. Vacio = el bot no le habla a nadie.

    Es una allowlist y no un filtro opcional a proposito: este bot publica tu banca, tu
    ROI y tus picks antes de que se peleen. Cualquiera puede encontrar un bot y
    escribirle; sin esta lista, cualquiera los lee.
    """
    crudo = os.getenv("TELEGRAM_CHAT_IDS", "")
    return [x.strip() for x in crudo.replace(";", ",").split(",") if x.strip()]


def _api(metodo, **params):
    r = requests.post(API.format(token=_token(), metodo=metodo), timeout=ESPERA + 15,
                      json=params)
    r.raise_for_status()
    return r.json().get("result")


def enviar(texto, chat=None, seco=False):
    """Manda el mensaje a un chat, o a todos los de la allowlist."""
    if seco:
        print(f"\n{'─' * 70}\n{texto}\n", flush=True)
        return
    for destino in ([chat] if chat else chats()):
        # Telegram corta en 4096; partir por parrafos deja los mensajes legibles.
        for trozo in _partir(texto):
            _api("sendMessage", chat_id=destino, text=trozo, parse_mode="HTML",
                 disable_web_page_preview=True)


def _partir(texto, tope=3900):
    if len(texto) <= tope:
        return [texto]
    trozos, actual = [], ""
    for parrafo in texto.split("\n\n"):
        if len(actual) + len(parrafo) + 2 > tope and actual:
            trozos.append(actual)
            actual = ""
        actual += (("\n\n" if actual else "") + parrafo)
    return [*trozos, actual] if actual else trozos


def _e(valor):
    """Escapa lo que va adentro del HTML de Telegram. Los nombres traen acentos y `&`."""
    return html.escape(str(valor))


# ------------------------------------------------------------------------ estado

def _leer_estado():
    try:
        return json.loads(ESTADO.read_text())
    except (OSError, json.JSONDecodeError):
        return {"update_id": 0, "hechos": []}


def _guardar_estado(estado):
    estado["hechos"] = estado["hechos"][-MEMORIA:]
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ESTADO.write_text(json.dumps(estado, ensure_ascii=False, indent=1))


# -------------------------------------------------------------------- datos

def sincronizar(rama=None):
    """Trae `ia_consenso.csv` y `picks.csv` de la rama remota. -> True si funciono.

    `git fetch` + `git show`, NO `git pull`: el clon de la VPS tiene
    `data/cartelera_hist.csv` modificado (el bot de inteligencia llama a
    `cartelera.proximas()`, que le appendea el primer avistaje de cada pelea), y un pull
    se caeria con "local changes would be overwritten". Traer archivos sueltos de la ref
    remota no toca el working tree.

    No hace nada en la maquina que genera los picks. `data/model.pkl` esta gitignoreado y
    pesa 900 KB: existe justo donde se corre el pipeline y no existe en la VPS, asi que
    distingue las dos puntas sin ningun flag que configurar mal. Del lado que genera, la
    version remota siempre es la vieja — sincronizar ahi es pisar el analisis nuevo con
    el de ayer.
    """
    if rutas.MODELO.exists():
        return False
    rama = rama or os.getenv("TIPSTER_RAMA", "ufc-fight-predictor")
    def git(*args, **kw):
        return subprocess.run(["git", "-C", str(rutas.RAIZ), *args], check=True,
                              capture_output=True, timeout=60, **kw)
    try:
        git("fetch", "--quiet", "origin", rama)
        for nombre in ("ia_consenso.csv", "picks.csv"):
            ruta = f"data/{nombre}"
            # Nunca se pisa un archivo con cambios locales. En la VPS nunca los tiene; en
            # la maquina donde SE GENERAN los picks los tiene siempre, y sobrescribirlo
            # con la version remota borraria el analisis que todavia no se pusheo.
            if git("status", "--porcelain", "--", ruta, text=True).stdout.strip():
                print(f"AVISO: {ruta} tiene cambios locales sin pushear; no lo piso.",
                      file=sys.stderr)
                continue
            salida = git("show", f"origin/{rama}:{ruta}", text=False)
            (rutas.DATOS / nombre).write_bytes(salida.stdout)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        # Sin red o sin remoto el bot sigue con lo que tenga en disco. Es degradacion,
        # no falla: los picks de hace una hora siguen siendo los picks.
        print(f"AVISO: no se pudo sincronizar con git: {exc}", file=sys.stderr)
        return False


def _par(a, b):
    return tuple(sorted((nombres.normalizar(a), nombres.normalizar(b))))


def picks_ia(evento):
    """-> {par normalizado: fila de ia_consenso} del analisis mas reciente."""
    return {_par(a, b): fila
            for (a, b), fila in ia_store.veredictos(evento).items()}


def votos_humanos(evento):
    """-> {par: (a_favor_de_a, a_favor_de_b)} contando solo personas.

    No se usa `predictores.comparar`: esa funcion necesita las predicciones del modelo
    alineadas con la cartelera, y el modelo no existe en la VPS. Para "3 de 4 lo tenian"
    alcanza con contar `picks.csv`, que si viaja por git.
    """
    df = predictores.leer(evento)
    if not len(df):
        return {}
    df = df[(df["origen"] != predictores.ORIGEN_IA) & df["revisado"].fillna(False)]
    cuenta = {}
    for fila in df.itertuples():
        if fila.pick not in {"a", "b"}:
            continue
        par = _par(fila.a, fila.b)
        # `pick` es relativo al a/b de ESA fila, que puede venir al reves que la nuestra.
        lado = nombres.normalizar(fila.a if fila.pick == "a" else fila.b)
        votos = cuenta.setdefault(par, {})
        votos[lado] = votos.get(lado, 0) + 1
    return cuenta


def _clp(valor):
    return f"${valor:,.0f}".replace(",", ".")


def apuesta_de(evento, a, b):
    """-> texto con como le fue a tu plata en esta pelea, o "" si no apostaste.

    `apuestas.evaluar()` liquida las patas contra `data/resultados.csv`, que es justo el
    archivo que acaba de escribir `resultados.volcar()`. O sea que se entera sola.
    """
    try:
        df, detalle, _ = apuestas.evaluar()
    except (OSError, KeyError, ValueError):
        return ""   # data/apuestas.csv no se versiona: en la VPS no existe
    if not len(detalle):
        return ""
    par = _par(a, b)
    patas = [x for x in detalle.itertuples() if _par(x.a, x.b) == par]
    lineas = []
    for pata in patas:
        bruto = df[df["id"] == pata.apuesta_id]
        if not len(bruto) or str(bruto.iloc[0]["evento"]) != evento:
            continue
        bet = bruto.iloc[0]
        if bet["estado"] == "pendiente":
            lineas.append(f"💰 Tu {bet['tipo'].lower()} de {_clp(bet['importe_clp'])}: "
                          f"esta pata {pata.estado}, sigue viva.")
        else:
            neto = float(bet["beneficio_clp"])
            lineas.append(f"{'💰' if neto >= 0 else '💸'} Tu "
                          f"{bet['tipo'].lower()} de {_clp(bet['importe_clp'])}: "
                          f"{bet['estado']}, {_clp(neto)} netos.")
    return "\n".join(lineas)


# ------------------------------------------------------------------- mensajes

def _prob(fila):
    """(nombre del elegido, su probabilidad) segun la IA."""
    p = float(fila["p_a_ia"])
    return (fila["a"], p) if fila["pick"] == "a" else (fila["b"], 1 - p)


def _abstuvo(fila):
    return (fila.get("veredicto") or "definido") == "parejo"


def texto_resultado(pelea, fila, *, orden, n, total, ganadas, jugadas, extra=()):
    """El aviso central: acerto, fallo o se abstuvo, con metodo y asalto.

    `fila` es la fila de `ia_consenso` de esta pelea, o None si la IA no la analizo.
    `orden` es el numero de pelea de la noche (cronologico) y `n` su lugar en el cartel,
    que van al reves: el main event es el 1 del cartel y el ultimo de la noche.
    """
    gana = pelea[pelea["ganador"]]
    via = METODO.get(pelea["metodo"])
    # ESPN no siempre publica el metodo (es play-by-play, no la ficha oficial). Antes que
    # deducirlo del reloj, se omite: `settlement` tampoco adivina.
    como = (f" por {via} en el asalto {pelea['asalto']}" if via and pelea["asalto"]
            else f" en el asalto {pelea['asalto']}" if pelea["asalto"] else "")
    cabeza = (f"🥊 <b>{_ordinal(orden)} pelea de la noche</b> · {_e(pelea['peso'])}"
              f" (#{n} de {total} en el cartel)")

    if fila is None:
        cuerpo = (f"⚪️ Ganó <b>{_e(gana)}</b>{como}.\n"
                  f"<i>La IA no analizó esta pelea.</i>")
    else:
        elegido, p = _prob(fila)
        quien = f"<b>{_e(elegido)}</b> al {p:.0%} (confianza {_e(fila['confianza'])}"
        if _abstuvo(fila):
            # No cuenta como fallo: `consenso.sincronizar_picks` excluye las parejas del
            # ranking, y anotarlas aca diria lo contrario de lo que la IA declaro.
            cuerpo = (f"⚪️ <b>SE ABSTUVO</b> · Ganó {_e(gana)}{como}\n\n"
                      f"La IA declaró la pelea pareja; se inclinaba apenas por "
                      f"{_e(elegido)} ({p:.0%}). No cuenta como pick.")
        elif nombres.normalizar(elegido) == nombres.normalizar(gana):
            cuerpo = (f"✅ <b>ACERTADA</b> · {_e(gana)}{como}\n\n"
                      f"La IA iba con {quien}{_dijo(fila, pelea)}).")
        else:
            cuerpo = (f"❌ <b>FALLADA</b> · Ganó {_e(gana)}{como}\n\n"
                      f"La IA iba con {quien}).")

    lineas = [x for x in extra if x]
    marcador = (f"\n\n📊 Van <b>{ganadas} de {jugadas}</b> en esta cartelera."
                if jugadas else "")
    return f"{cabeza}\n\n{cuerpo}" + ("\n" + "\n".join(lineas) if lineas else "") + marcador


def _dijo(fila, pelea):
    """", dijo KO" cuando la IA ademas le embocó la via. Callado si no."""
    previsto = str(fila.get("metodo_probable") or "")
    if not previsto or not pelea.get("metodo"):
        return ""
    return (f", y le acertó también al método: {METODO[previsto]}"
            if previsto == pelea["metodo"] else f", esperaba {METODO.get(previsto, previsto)}")


def _ordinal(n):
    return {1: "1ª", 2: "2ª", 3: "3ª"}.get(n, f"{n}ª")


def texto_preview(evento, ia, votos):
    """La cartelera con lo que dice la IA, unas horas antes del primer campanazo."""
    lineas = [f"📋 <b>{_e(evento['evento'])}</b>",
              f"<i>Arranca {_e(_cuando(evento))}. Esto es lo que dice la IA, "
              f"pelea por pelea. No vio ninguna cuota.</i>", ""]
    for n, p in enumerate(evento["peleas"], 1):
        fila = ia.get(_par(p["a"], p["b"]))
        cabeza = f"<b>{n}.</b> {_e(p['a'])} vs {_e(p['b'])} · {_e(p['peso'])}"
        if fila is None:
            lineas += [cabeza, "   <i>sin análisis de IA</i>", ""]
            continue
        elegido, prob = _prob(fila)
        icono = "⚪️" if _abstuvo(fila) else "🎯"
        via = METODO.get(str(fila.get("metodo_probable") or ""), "")
        detalle = (f"{icono} {_e(elegido)} {prob:.0%} · confianza "
                   f"{_e(fila['confianza'])}" + (f" · espera {via}" if via else ""))
        if _abstuvo(fila):
            detalle += " · <i>la declara pareja, no es pick</i>"
        lineas += [cabeza, f"   {detalle}", *_linea_votos(p, votos, sangria=True), ""]
    return "\n".join(lineas).rstrip()


def _linea_votos(pelea, votos, sangria=False):
    cuenta = votos.get(_par(pelea["a"], pelea["b"]))
    if not cuenta:
        return []
    total = sum(cuenta.values())
    quien, n = max(cuenta.items(), key=lambda kv: kv[1])
    real = pelea["a"] if nombres.normalizar(pelea["a"]) == quien else pelea["b"]
    return [f"{'   ' if sangria else ''}👥 Predictores: {n} de {total} "
            f"por {_e(real)}"]


def _cuando(evento):
    inicio = _inicio(evento)
    return inicio.strftime("%d/%m %H:%M UTC") if inicio else evento.get("fecha", "?")


def texto_cierre(evento, ia):
    """Cierre de cartelera: el acumulado del evento y la serie historica."""
    ganadas, jugadas, _ = marcador(evento, ia)
    if jugadas:
        pct = ganadas / jugadas
        cabeza = (f"🏁 <b>Terminó {_e(evento['evento'])}</b>\n\n"
                  f"La IA cerró <b>{ganadas} de {jugadas}</b> ({pct:.0%}) en sus picks "
                  f"de esta cartelera.")
    else:
        cabeza = (f"🏁 <b>Terminó {_e(evento['evento'])}</b>\n\n"
                  f"La IA no dejó ninguna pick registrada para esta cartelera.")
    # `marcador` cuenta todas las picks de la noche; el bloque de abajo mide solo la cohorte
    # del prompt actual. Son dos numeros distintos y verlos juntos sin este renglon es
    # justamente lo que hace pensar que uno de los dos esta mal.
    return (f"{cabeza}\n\nAcá abajo, la serie completa — ahí solo entra la cohorte del "
            f"prompt {_e(dossier.VERSION)}.\n\n{cmd_ia()}")


def cronologicas(evento):
    """-> [(orden de la noche, numero en el cartel, pelea)] de las que ya terminaron.

    `evento['peleas']` viene con el main event primero, que es como lo dibuja el cartel de
    la app y como numera `/pelea n`. La noche va justo al reves: arrancan los
    preliminares. Sin invertir aca, la "1ª pelea de la noche" del mensaje seria la ultima
    en pelearse.
    """
    total = len(evento["peleas"])
    # El orden sale de la posicion en el cartel y no de un contador de resueltas: si ESPN
    # publica una pelea antes que la anterior, el numero de cada una igual no se mueve.
    return [(total - n + 1, n, pelea)
            for n, pelea in reversed(list(enumerate(evento["peleas"], 1)))
            if pelea["ganador"]]


def marcador(evento, ia):
    """-> (aciertos, picks jugadas, abstenciones) sobre las peleas ya resueltas."""
    ganadas = jugadas = parejas = 0
    for p in resultados.resueltas(evento):
        fila = ia.get(_par(p["a"], p["b"]))
        if fila is None:
            continue
        if _abstuvo(fila):
            parejas += 1
            continue
        jugadas += 1
        elegido, _ = _prob(fila)
        ganadas += nombres.normalizar(elegido) == nombres.normalizar(p[p["ganador"]])
    return ganadas, jugadas, parejas


# ------------------------------------------------------------------- comandos

def _evento_actual():
    """La cartelera a la que se refieren los comandos: la de hoy, o la próxima."""
    vivo = resultados.en_vivo()
    if vivo and _en_ventana(vivo):
        return vivo, True
    proximas = cartelera.proximas()
    return (proximas[0] if proximas else None), False


def cmd_cartelera():
    evento, _ = _evento_actual()
    if not evento:
        return "No hay ninguna cartelera anunciada (o ESPN no respondió)."
    ia = picks_ia(evento["evento"])
    if not ia:
        return (f"📋 <b>{_e(evento['evento'])}</b>\n\nTodavía no hay picks de IA para "
                f"esta cartelera. Corré <code>python -m ufc.ia.consenso</code> y hacé "
                f"push de <code>data/ia_consenso.csv</code>.")
    return texto_preview(evento, ia, votos_humanos(evento["evento"]))


def cmd_pelea(argumento):
    evento, _ = _evento_actual()
    if not evento:
        return "No hay ninguna cartelera anunciada."
    try:
        n = int(str(argumento).strip())
    except (TypeError, ValueError):
        return "Usalo así: <code>/pelea 1</code> (1 es el main event)."
    if not 1 <= n <= len(evento["peleas"]):
        return f"Esta cartelera tiene {len(evento['peleas'])} peleas."

    p = evento["peleas"][n - 1]
    fila = picks_ia(evento["evento"]).get(_par(p["a"], p["b"]))
    cabeza = (f"🥊 <b>{_e(p['a'])} vs {_e(p['b'])}</b>\n"
              f"<i>{_e(p['peso'])} · pelea {n} de {len(evento['peleas'])} · "
              f"{_e(evento['evento'])}</i>")
    if fila is None:
        return f"{cabeza}\n\nLa IA todavía no analizó esta pelea."

    elegido, prob = _prob(fila)
    partes = [cabeza, ""]
    partes.append(f"⚪️ <b>La declara pareja.</b> Se inclina apenas por {_e(elegido)} "
                  f"({prob:.0%})." if _abstuvo(fila) else
                  f"🎯 <b>Elige a {_e(elegido)} ({prob:.0%}).</b> Confianza "
                  f"{_e(fila['confianza'])}.")

    via = METODO.get(str(fila.get("metodo_probable") or ""))
    if via:
        partes.append(f"Espera que termine por {via}.")
    partes += _numeros(fila)
    partes += _linea_votos(p, votos_humanos(evento["evento"]))
    partes += _razonamiento(evento["evento"], p)
    return "\n".join(partes)


def _numeros(fila):
    """Modelo, mercado y EV — el snapshot congelado el dia que la IA opino."""
    salida = []
    modelo, mercado = fila.get("p_a_modelo"), fila.get("p_a_mercado")
    if pd.notna(modelo) and pd.notna(mercado):
        salida.append(f"\n📈 Cuando opinó: el modelo daba {float(modelo):.0%} y el "
                      f"mercado {float(mercado):.0%} para {_e(fila['a'])}.")
    cuota, ev = fila.get("cuota_tomada"), fila.get("ev_ia")
    if pd.notna(cuota) and pd.notna(ev):
        salida.append(f"💵 Contra {float(cuota):.2f} en {_e(fila['casa'])} eso da un EV "
                      f"de {float(ev):+.1%}. <i>Lo calculó el código después; la IA "
                      f"nunca vio esa cuota.</i>")
    return salida


def _razonamiento(evento, pelea):
    """Las razones del informe. Solo existe en la maquina que corrio el analisis."""
    informe = ia_store.informe(evento, pelea["a"], pelea["b"])
    if not informe:
        return ["\n<i>El informe con el razonamiento no está en esta máquina "
                "(`data/ia_informes/` no se versiona).</i>"]
    salida = []
    if informe.get("razones"):
        salida.append("\n<b>Por qué</b>")
        salida += [f"• <i>{_e(r['fuente'])}</i> — {_e(r['texto'])}"
                   for r in informe["razones"]]
    if informe.get("factores_no_modelables"):
        salida.append("\n<b>Lo que ninguna estadística captura</b>")
        for f in informe["factores_no_modelables"]:
            quien = pelea["a"] if f["favorece"] == "a" else pelea["b"]
            salida.append(f"• <b>{_e(f['titulo'])}</b> (favorece a {_e(quien)}, "
                          f"{_e(f['certeza'])}) — {_e(f['explicacion'])}")
    if informe.get("contra"):
        salida.append(f"\n<b>El mejor argumento en contra de su propio pick</b>\n"
                      f"{_e(informe['contra'])}")
    return salida


def cmd_ia():
    try:
        m = ia_evaluar.metricas()
    except (OSError, KeyError, ValueError) as exc:
        return f"No se pudo calcular: {_e(exc)}"
    return _texto_ia(m) if m else _ia_sin_cohorte()


def _texto_ia(m):
    """El reporte de la IA como mensaje de Telegram, no como salida de terminal.

    El formato viejo era `ia_evaluar.resumen()` metido en un `<pre>`: en un teléfono eso
    sale en monospace diminuto y con scroll horizontal, que es exactamente lo que hacía
    ilegible el número más importante. Acá solo queda en `<pre>` lo que de verdad son
    columnas —el log loss—; el resto es texto con jerarquía.
    """
    ok_p, n_p = m["picks_ok"], m["picks_n"]
    p = ["🤖 <b>Cómo le va a la IA</b>",
         f"<i>Prompt {_e(m['prompt_v'])} · ciega al mercado: no ve cuotas, ni la "
         f"probabilidad del modelo, ni las picks humanas.</i>",
         "",
         f"🎯 <b>Sus picks: {_e(ia_evaluar.marca(ok_p, n_p))}</b>",
         "Solo cuenta lo que declaró «definido»."]

    if m["parejas_n"]:
        p += ["",
              f"⚪️ <b>Declaró parejas: {m['parejas_n']} de {m['n']} peleas</b>",
              f"No son picks y no entran a <code>picks.csv</code>. Su inclinación mínima "
              f"acertó {_e(ia_evaluar.marca(m['parejas_ok'], m['parejas_n']))}."]

    filas = [f"{nombre:8s} {media:.4f}  n={k}"
             for nombre, (media, k) in m["ll"].items() if media is not None]
    if filas:
        p += ["",
              "📊 <b>Qué tan buenas son sus probabilidades</b>",
              "<i>log loss sobre toda la cohorte, más bajo es mejor</i>",
              f"<pre>{_e(chr(10).join(filas))}</pre>",
              _lectura_deltas(m)]

    p += ["", "💵 <b>CLV contra la línea de cierre</b>",
          "<i>La IA nunca vio el precio, así que esto mide si su lectura deportiva le "
          "gana al mercado.</i>"]
    if m["clv"]:
        c = m["clv"]
        ic = "" if c["lo"] is None else f" · IC95% [{c['lo']:+.1%}, {c['hi']:+.1%}]"
        p.append(f"EV medio <b>{c['media']:+.1%}</b> en "
                 f"{_e(ia_evaluar.plural(c['n'], 'pelea'))}{ic}")
        if c["faltan"]:
            p.append(f"Para concluir un CLV de +2% harían falta ~{c['faltan']} peleas.")
    else:
        p.append("Todavía no hay ninguna pelea con precio de apertura y de cierre.")

    p += ["", f"⏳ <b>Muestra: {_e(ia_evaluar.plural(m['n'], 'pelea'))} de "
              f"{_e(ia_evaluar.plural(m['carteleras'], 'cartelera'))}.</b> "
              f"Nada de esto concluye nada todavía."]

    hechas, total = m["total_picks"]
    if total:
        p += ["", f"📚 <b>Historial completo: {_e(ia_evaluar.marca(hechas, total))} en "
                  f"picks.csv</b>",
              "Incluye las peleas de versiones anteriores del prompt, que sí veían el "
              "modelo, el mercado y las picks humanas. Por eso no entran arriba: son "
              "otro predictor."]

    p += ["", "<i>La cohorte del gate (data/ledger.csv) no se toca: esto se mide "
              "aparte.</i>"]
    return "\n".join(p)


def _lectura_deltas(m):
    """La frase que traduce el delta pareado a castellano: quién le gana a quién.

    El delta es `otro - IA` sobre log loss, así que negativo significa que el otro pierde
    menos, o sea que es mejor. Sin esta línea el usuario tiene que acordarse del signo.
    """
    frases = [f"el {nombre} {'le gana' if media < 0 else 'va por detrás'} ({media:+.4f})"
              for nombre, (media, _lo, _hi, n) in m["deltas"].items() if n]
    if not frases:
        return ""
    ics = [d for d in m["deltas"].values() if d[3]]
    if any(lo is None for _m, lo, _hi, _n in ics):
        cola = (f" — con {_e(ia_evaluar.plural(m['carteleras'], 'cartelera'))} no hay "
                f"IC95% posible, así que no concluye nada")
    elif all(lo <= 0 <= hi for _m, lo, hi, _n in ics):
        cola = " — pero el IC95% toca cero, así que todavía no concluye nada"
    else:
        cola = " — y el IC95% no toca cero"
    return f"Pelea por pelea: {', '.join(frases)}{cola}."


def _ia_sin_cohorte():
    """Sin cohorte v2 no hay nada que promediar, pero casi nunca es que el registro falle.

    Lo habitual es que las peleas resueltas sean de una versión anterior del prompt, y esas
    `evaluar()` las deja afuera a propósito: son otro predictor con el mismo nombre. Un "no
    hay nada" a secas se lee como que el bot está roto.
    """
    texto = "🤖 <b>Cómo le va a la IA</b>\n\nTodavía no hay ninguna pelea analizada con " \
            f"el prompt {_e(dossier.VERSION)} que tenga resultado cargado."
    try:
        viejas = len(ia_evaluar.evaluar(prompt_v=None))
    except (OSError, KeyError, ValueError):
        viejas = 0
    if viejas:
        texto += (f"\n\nHay {viejas} pelea(s) resuelta(s) de cohortes anteriores del "
                  f"prompt. No entran acá: la v1 veía el modelo, el mercado y las picks "
                  f"humanas, así que sus veredictos son los de otro predictor y "
                  f"promediarlos daría el rendimiento de algo que no existe.")
    return texto


def cmd_predictores():
    """La tabla de aciertos: la IA contra las personas, medidas con la misma vara."""
    try:
        tabla = predictores.confiabilidad()
        picks = predictores.leer()
    except (OSError, KeyError, ValueError) as exc:
        return f"No se pudo calcular: {_e(exc)}"
    if not len(tabla):
        return "Todavía no hay ninguna pick con resultado cargado."
    # Por `origen` y no por nombre: el nombre lleva el proveedor adentro ("IA (Gemini)").
    suyos = (set(picks[picks["origen"] == predictores.ORIGEN_IA]["predictor"])
             if len(picks) else set())
    filas = [f"{'🤖' if f.predictor in suyos else '  '} {f.predictor:18.18s} "
             f"{f.aciertos:>3}/{f.total:<3} {f.acierto:>4.0%}  peso {f.peso:.2f}"
             for f in tabla.itertuples()]
    return ("🏆 <b>Tabla de predictores</b>\n"
            "<i>Sobre las peleas ya resueltas y revisadas.</i>\n\n"
            f"<pre>{_e(chr(10).join(filas))}</pre>\n"
            "<i>El «peso» aplica un prior Beta(5, 5): una racha corta de 7 de 7 no cuenta "
            "como una serie larga con el mismo porcentaje.</i>")


def cmd_gate():
    """Si el proyecto está autorizado a apostar, y qué falta para que lo esté."""
    try:
        e = gate.estado()
    except (OSError, KeyError, ValueError) as exc:
        return f"No se pudo leer el gate: {_e(exc)}"
    p = [f"{'🟢' if e['autorizado'] else '🔴'} <b>Regla de apuestas: "
         f"{'AUTORIZADA' if e['autorizado'] else 'CERRADA'}</b>",
         "", _e(e.get("motivo", "sin motivo registrado"))]
    if e.get("faltan"):
        p += ["", f"Faltan <b>{e['faltan']}</b> apuestas preregistradas para llegar al "
                  f"mínimo de {e['n_minimo']}."]
    if e.get("version"):
        p += ["", f"<i>Métrica {_e(e.get('metrica', '?'))} · regla {_e(e['version'])}, "
                  f"escrita antes de mirar los resultados (config/gate.json).</i>"]
    return "\n".join(p)


def cmd_estado():
    """Diagnóstico del bot.

    Existe porque este proceso degrada en silencio: si `sincronizar()` no puede hablar con
    git, el aviso va a stderr y queda enterrado en el journal del servicio. Acá se pregunta
    y se contesta desde el chat.
    """
    p = ["🩺 <b>Estado del bot</b>", ""]

    evento, en_curso = _evento_actual()
    if not evento:
        p.append("📅 Sin cartelera anunciada (o ESPN no respondió).")
    else:
        p.append(f"📅 <b>{_e(evento['evento'])}</b> · {_e(_cuando(evento))}")
        p.append("🔴 En curso: narrando pelea por pelea." if en_curso else
                 "⏳ Fuera de la ventana de narración; los avisos arrancan 4 h antes.")

    ultima = ia_store.resumen()
    p.append("")
    if not ultima:
        p.append("🧠 Todavía no corrió ningún análisis de IA.")
    else:
        tokens = int(ultima["prompt_tokens"]) + int(ultima["output_tokens"])
        p += ["🧠 <b>Último análisis de IA</b>",
              f"{_e(ultima['evento'])} · {ultima['peleas']} peleas "
              f"({ultima['parejas']} parejas)",
              f"{_e(ultima['modelo_ia'])} · corrida del {_e(ultima['run_day'])} · "
              f"{tokens:,} tokens".replace(",", ".")]

    p += ["", "📄 <b>Datos en disco</b>"]
    p += [f"{_e(n)} — {_e(_estado_csv(rutas.DATOS / n))}"
          for n in ("ia_consenso.csv", "picks.csv", "resultados.csv")]

    # Las dos razones para no sincronizar son opuestas —una es correcta y la otra es una
    # falla— y `sincronizar()` devuelve False en las dos. Distinguirlas es medio punto del
    # comando: sin eso, "no sincronizó" no dice si hay algo que arreglar.
    if rutas.MODELO.exists():
        sync = "Esta máquina genera los picks, así que no sincroniza: traerlos de la rama " \
               "remota pisaría el análisis nuevo con el de ayer."
    else:
        sync = ("Sincronizado recién con la rama remota." if sincronizar() else
                "⚠️ Falló la sincronización con git; el motivo quedó en el journal del "
                "servicio. Los picks pueden estar viejos.")
    p += ["", f"🔄 {sync}", f"👥 {len(chats())} chat(s) autorizados."]
    return "\n".join(p)


def _estado_csv(ruta):
    """-> "24 filas · 09/08 12:00", o por qué no se puede decir eso."""
    try:
        filas = sum(1 for _ in ruta.open(encoding="utf-8")) - 1
    except OSError:
        return "no está en esta máquina"
    cuando = datetime.datetime.fromtimestamp(ruta.stat().st_mtime)
    return f"{max(filas, 0)} filas · {cuando:%d/%m %H:%M}"


def cmd_hoy():
    evento = resultados.en_vivo()
    if not evento:
        return "No hay ninguna cartelera en curso."
    ia = picks_ia(evento["evento"])
    hechas = resultados.resueltas(evento)
    ganadas, jugadas, parejas = marcador(evento, ia)
    cabeza = [f"🥊 <b>{_e(evento['evento'])}</b>",
              f"<i>{len(hechas)} de {len(evento['peleas'])} peleadas</i>", ""]
    if jugadas:
        cabeza.append(f"📊 La IA va <b>{ganadas} de {jugadas}</b>"
                      + (f" ({parejas} declaradas parejas)." if parejas else ".") + "\n")
    elif hechas:
        cabeza.append("La IA no tenía picks en lo que se peleó hasta ahora.\n")

    detalle = []
    for n, p in enumerate(evento["peleas"], 1):
        fila = ia.get(_par(p["a"], p["b"]))
        elegido = _prob(fila)[0] if fila is not None else None
        if not p["ganador"]:
            pendiente = "🔴 en curso" if p["estado"] == "in" else "⏳ pendiente"
            marca = f" · IA: {_e(elegido)}" if elegido else ""
            detalle.append(f"{n}. {_e(p['a'])} vs {_e(p['b'])} — {pendiente}{marca}")
            continue
        gana = p[p["ganador"]]
        via = METODO.get(p["metodo"])
        como = f" ({via} R{p['asalto']})" if via and p["asalto"] else ""
        if fila is None:
            icono = "⚪️"
        elif _abstuvo(fila):
            icono = "⚪️"
        else:
            icono = "✅" if nombres.normalizar(elegido) == nombres.normalizar(gana) else "❌"
        detalle.append(f"{n}. {icono} {_e(gana)}{como}")
    return "\n".join([*cabeza, *detalle])


# (comando, que dice el menu de Telegram, handler). Una sola lista porque son tres los
# consumidores del mismo dato —el dispatch, el /ayuda y `setMyCommands`— y mantener tres
# copias a mano es la garantia de que alguna quede vieja. La descripcion la corta Telegram
# a 256 caracteres.
CATALOGO = [
    ("cartelera", "La próxima cartelera con el pick de la IA en cada pelea",
     lambda arg: cmd_cartelera()),
    ("pelea", "El razonamiento completo de una pelea. Ej: /pelea 1", cmd_pelea),
    ("hoy", "Cómo va la cartelera que se está peleando", lambda arg: cmd_hoy()),
    ("ia", "Cómo le viene yendo a la IA contra el modelo y el mercado",
     lambda arg: cmd_ia()),
    ("predictores", "Tabla de aciertos: la IA contra los tipsters humanos",
     lambda arg: cmd_predictores()),
    ("gate", "Si la regla de apuestas está autorizada, y cuánto falta",
     lambda arg: cmd_gate()),
    ("estado", "Diagnóstico del bot: datos, último análisis, sincronización",
     lambda arg: cmd_estado()),
    ("ayuda", "Esta lista", lambda arg: AYUDA),
]

AYUDA = "\n".join(["👋 <b>Tipster UFC</b>", "",
                   *(f"/{c} — {d}" for c, d, _ in CATALOGO), "",
                   "Los avisos de acierto y fallo salen solos, pelea por pelea."])

# `/start` y `/help` contestan pero no van al menu: Telegram ya los trata aparte y
# repetirlos ahi le come dos lugares a la lista que el usuario ve al tipear "/".
COMANDOS = {f"/{c}": h for c, _, h in CATALOGO} | {"/start": lambda arg: AYUDA,
                                                   "/help": lambda arg: AYUDA}


def registrar_menu():
    """Puebla la lista que Telegram ofrece al tipear "/". Sin esto el menu esta vacio.

    Es una llamada por arranque del daemon y no un archivo de configuracion en BotFather:
    asi la lista sale del mismo `CATALOGO` que el dispatch y no se puede desincronizar.
    """
    return _api("setMyCommands",
                commands=[{"command": c, "description": d} for c, d, _ in CATALOGO])


def responder(texto):
    """-> la respuesta a un mensaje, o None si no es un comando conocido."""
    partes = str(texto or "").strip().split(maxsplit=1)
    if not partes:
        return None
    # Telegram manda "/hoy@MiBot" cuando el bot esta en un grupo.
    comando = partes[0].split("@")[0].lower()
    accion = COMANDOS.get(comando)
    return accion(partes[1] if len(partes) > 1 else None) if accion else None


def atender(seco=False):
    """Lee los mensajes nuevos y contesta los de la allowlist."""
    estado = _leer_estado()
    permitidos = set(chats())
    try:
        updates = _api("getUpdates", offset=estado["update_id"] + 1, timeout=ESPERA,
                       allowed_updates=["message"]) or []
    except (requests.RequestException, ValueError) as exc:
        print(f"AVISO: getUpdates falló: {exc}", file=sys.stderr)
        time.sleep(5)
        return

    for update in updates:
        estado["update_id"] = max(estado["update_id"], update.get("update_id", 0))
        mensaje = update.get("message") or {}
        chat = str((mensaje.get("chat") or {}).get("id", ""))
        if chat not in permitidos:
            # Silencio para el que escribe —contestarle "no tenés permiso" le confirma que
            # hay algo del otro lado— pero al log si va, porque es la unica forma de saber
            # tu propio chat_id la primera vez que configuras el bot.
            print(f"Mensaje ignorado de chat_id {chat}. Si sos vos, agregalo a "
                  f"TELEGRAM_CHAT_IDS.", flush=True)
            continue
        respuesta = responder(mensaje.get("text"))
        if respuesta:
            enviar(respuesta, chat=chat, seco=seco)
    _guardar_estado(estado)


# ---------------------------------------------------------------------- bucle

def _inicio(evento):
    try:
        return datetime.datetime.fromisoformat(str(evento.get("inicio_utc") or ""))
    except ValueError:
        return None


def _en_ventana(evento, ahora=None):
    """Solo se narra alrededor del evento. Sin esto, arrancar el bot un martes
    reanunciaria la cartelera del sabado pasado."""
    inicio = _inicio(evento)
    if not inicio:
        return False
    ahora = ahora or datetime.datetime.now(datetime.UTC)
    return -ANTES <= (ahora - inicio).total_seconds() <= DESPUES


def anunciar(seco=False, ahora=None):
    """Una vuelta de avisos. -> cuantos mensajes salieron.

    Con `seco` no escribe: ni el estado ni `resultados.csv`. Eso lo hace repetible (cada
    corrida muestra lo mismo) al precio de que las apuestas no se liquiden, asi que las
    lineas de plata no aparecen. Un flag que se llama "seco" y deja rastro en un CSV
    versionado no sirve para probar nada.
    """
    evento = resultados.en_vivo()
    if not evento or not _en_ventana(evento, ahora):
        return 0

    sincronizar()
    ia = picks_ia(evento["evento"])
    votos = votos_humanos(evento["evento"])
    estado = _leer_estado()
    hechos = set(estado["hechos"])
    clave_ev = evento["evento"]
    mandados = 0

    inicio = _inicio(evento)
    ahora = ahora or datetime.datetime.now(datetime.UTC)
    falta = (inicio - ahora).total_seconds() if inicio else 0

    if ia and falta <= PREVIA and f"preview|{clave_ev}" not in hechos:
        enviar(texto_preview(evento, ia, votos), seco=seco)
        hechos.add(f"preview|{clave_ev}")
        mandados += 1

    hechas = resultados.resueltas(evento)
    if hechas and not seco:
        resultados.volcar(evento)   # liquida las apuestas y alimenta a `ia.evaluar`

    # Arrancar el bot con media cartelera peleada no puede disparar ocho mensajes de
    # golpe. La primera vuelta de un evento ya empezado resume y marca todo como dicho.
    primera = not any(x.startswith(f"res|{clave_ev}|") for x in hechos)
    if primera and len(hechas) > 1:
        ganadas, jugadas, _ = marcador(evento, ia)
        enviar(f"👀 <b>Me sumo a {_e(clave_ev)} en curso.</b>\n"
               f"Ya se pelearon {len(hechas)} de {len(evento['peleas'])}"
               + (f"; la IA va {ganadas} de {jugadas}." if jugadas else ".")
               + "\nDesde acá narro pelea por pelea.", seco=seco)
        hechos |= {f"res|{clave_ev}|{p['a']}|{p['b']}" for p in hechas}
        mandados += 1
    else:
        for orden, n, p in cronologicas(evento):
            clave = f"res|{clave_ev}|{p['a']}|{p['b']}"
            if clave in hechos:
                continue
            hechos.add(clave)
            # El marcador se calcula despues de marcar esta pelea, para que el "van 3 de
            # 4" la incluya en vez de quedar siempre una atras.
            ganadas, jugadas, _ = marcador(evento, ia)
            enviar(texto_resultado(
                p, ia.get(_par(p["a"], p["b"])), orden=orden, n=n,
                total=len(evento["peleas"]), ganadas=ganadas, jugadas=jugadas,
                extra=[*_linea_votos(p, votos),
                       apuesta_de(clave_ev, p["a"], p["b"])]), seco=seco)
            mandados += 1

    if (len(hechas) == len(evento["peleas"]) and hechas
            and f"cierre|{clave_ev}" not in hechos):
        enviar(texto_cierre(evento, ia), seco=seco)
        hechos.add(f"cierre|{clave_ev}")
        mandados += 1

    if not seco:
        ya = set(estado["hechos"])
        estado["hechos"] += [x for x in sorted(hechos) if x not in ya]
        _guardar_estado(estado)
    return mandados


def main(argv=None):
    p = argparse.ArgumentParser(description="Bot de Telegram tipster para la cartelera")
    p.add_argument("--seco", action="store_true",
                   help="una vuelta, imprime lo que mandaría y no llama a Telegram")
    p.add_argument("--comando", help="prueba un comando sin Telegram, ej: '/hoy'")
    args = p.parse_args(argv)

    if args.comando:
        print(responder(args.comando) or "Comando desconocido.")
        return 0
    if args.seco:
        evento = resultados.en_vivo()
        if evento and not _en_ventana(evento):
            print(f"[{evento['evento']} está fuera de la ventana de narración; "
                  f"igual muestro el preview]\n")
            enviar(texto_preview(evento, picks_ia(evento["evento"]),
                                 votos_humanos(evento["evento"])), seco=True)
        print(f"{anunciar(seco=True)} mensajes saldrían ahora.")
        return 0

    if not _token():
        print("Falta TELEGRAM_BOT_TOKEN. Pedile uno a @BotFather.", file=sys.stderr)
        return 2
    if not chats():
        print("Falta TELEGRAM_CHAT_IDS. Sin allowlist el bot no le habla a nadie.",
              file=sys.stderr)
        return 2

    try:
        registrar_menu()
    except (requests.RequestException, ValueError) as exc:
        # El menu es comodidad; sin el los comandos siguen andando. No vale la pena no
        # arrancar el bot porque Telegram no contesto en este segundo.
        print(f"AVISO: no se pudo registrar el menú de comandos: {exc}", file=sys.stderr)

    print(f"Tipster escuchando. {len(chats())} chat(s) autorizados.", flush=True)
    while True:
        try:
            atender()
            anunciar()
        except KeyboardInterrupt:
            return 0
        except Exception as exc:   # el bucle no se muere por una vuelta mala
            print(f"ERROR en el ciclo: {type(exc).__name__}: {exc}", file=sys.stderr)
            time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
