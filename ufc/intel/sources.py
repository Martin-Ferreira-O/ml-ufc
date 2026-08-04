"""Recolectores conservadores: Google News RSS y feeds RSS/Atom configurados."""

import csv
import dataclasses
import datetime
import email.utils
import html
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET

import requests

from ufc import nombres, rutas
from ufc.intel import identities


GOOGLE_NEWS = "https://news.google.com/rss/search"
X_SEARCH = "https://api.x.com/2/tweets/search/recent"
FUENTES = rutas.DATOS / "intel_sources.csv"
_ETIQUETAS = re.compile(r"<[^>]+>")
_MARKET_SOURCES = {"coinbase.com", "draftkings", "fanduel", "kalshi",
                   "polymarket", "robinhood"}
_MARKET_TITLES = ("betting odds & lines", "prediction market")


@dataclasses.dataclass(frozen=True)
class Evidence:
    url: str
    title: str
    source: str
    published_at: str | None
    snippet: str
    kind: str = "news"


def _texto(valor):
    return " ".join(html.unescape(_ETIQUETAS.sub(" ", valor or "")).split())


def _fecha(valor):
    if not valor:
        return None
    try:
        d = email.utils.parsedate_to_datetime(valor)
        if not d.tzinfo:
            d = d.replace(tzinfo=datetime.timezone.utc)
        return d.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")
    except (TypeError, ValueError, OverflowError):
        try:
            return datetime.datetime.fromisoformat(valor.replace("Z", "+00:00")).isoformat()
        except ValueError:
            return None


def parsear_feed(contenido, kind="feed", label=None):
    """Parsea RSS 2.0 o Atom sin depender del paquete feedparser."""
    raiz = ET.fromstring(contenido)
    items = raiz.findall(".//item")
    atom = not items
    if atom:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        items = raiz.findall(".//a:entry", ns)
    salida = []
    for item in items:
        if atom:
            titulo = item.findtext("a:title", default="", namespaces=ns)
            enlace = item.find("a:link", ns)
            url = enlace.get("href", "") if enlace is not None else ""
            resumen = (item.findtext("a:summary", default="", namespaces=ns) or
                       item.findtext("a:content", default="", namespaces=ns))
            publicado = (item.findtext("a:published", default="", namespaces=ns) or
                          item.findtext("a:updated", default="", namespaces=ns))
            fuente = label or raiz.findtext("a:title", default="Feed", namespaces=ns)
        else:
            titulo = item.findtext("title", "")
            url = item.findtext("link", "")
            resumen = item.findtext("description", "")
            publicado = item.findtext("pubDate", "")
            nodo_fuente = item.find("source")
            fuente = label or (nodo_fuente.text if nodo_fuente is not None else None) \
                or raiz.findtext("./channel/title", "Feed")
        url = (url or "").strip()
        titulo = _texto(titulo)
        if not url or not titulo:
            continue
        salida.append(Evidence(url=url, title=titulo, source=_texto(fuente),
                               published_at=_fecha(publicado),
                               snippet=_texto(resumen)[:1200], kind=kind))
    return salida


def google_news(fighter, session=requests, timeout=30):
    params = {"q": f'"{fighter}" (UFC OR MMA) when:7d',
              "hl": "en-US", "gl": "US", "ceid": "US:en"}
    r = session.get(GOOGLE_NEWS, params=params, timeout=timeout,
                    headers={"User-Agent": "ml-ufc-intel/1.0"})
    r.raise_for_status()
    return [item for item in parsear_feed(r.content, kind="news", label=None)
            if noticia_util(item)]


def noticia_util(evidence):
    """Descarta listings de cuotas/mercados que Google News mezcla con noticias."""
    source = evidence.source.strip().lower()
    title = evidence.title.strip().lower()
    if source in _MARKET_SOURCES:
        return False
    return not any(pattern in title for pattern in _MARKET_TITLES)


def fuentes_de(fighter, path=FUENTES, profiles_path=identities.PROFILES):
    clave = nombres.normalizar(fighter)
    manual = []
    if path.exists():
        with path.open(encoding="utf-8") as f:
            manual = [fila for fila in csv.DictReader(f)
                      if nombres.normalizar(fila.get("fighter", "")) == clave
                      and fila.get("url", "").strip()]
    output = []
    seen = set()
    for source in [*manual, *identities.feed_sources(fighter, profiles_path)]:
        key = ((source.get("kind") or "").lower(), source.get("url", "").strip())
        if key not in seen:
            output.append(source)
            seen.add(key)
    return output


def feed_publico(fuente, session=requests, timeout=30):
    r = session.get(fuente["url"].strip(), timeout=timeout,
                    headers={"User-Agent": "ml-ufc-intel/1.0"})
    r.raise_for_status()
    return parsear_feed(r.content, kind=fuente.get("kind") or "social",
                        label=fuente.get("label") or None)


def _x_handle(valor):
    valor = valor.strip().rstrip("/")
    if "/" in valor:
        valor = urllib.parse.urlparse(valor).path.strip("/").split("/")[0]
    return valor.lstrip("@")


def x_posts(fuente, session=requests, timeout=30, token=None, now=None):
    """Posts recientes de una cuenta oficial, exclusivamente por X API v2."""
    token = token or os.getenv("X_BEARER_TOKEN")
    if not token:
        raise ValueError("Hay una fuente X configurada pero falta X_BEARER_TOKEN.")
    handle = _x_handle(fuente["url"])
    now = now or datetime.datetime.now(datetime.timezone.utc)
    inicio = (now - datetime.timedelta(hours=30)).isoformat(timespec="seconds").replace("+00:00", "Z")
    r = session.get(X_SEARCH, timeout=timeout,
                    headers={"Authorization": f"Bearer {token}",
                             "User-Agent": "ml-ufc-intel/1.0"},
                    params={"query": f"from:{handle} -is:retweet",
                            "max_results": 10, "tweet.fields": "created_at",
                            "start_time": inicio})
    r.raise_for_status()
    salida = []
    for post in r.json().get("data", []):
        texto = _texto(post.get("text", ""))
        post_id = str(post.get("id", ""))
        if not texto or not post_id:
            continue
        salida.append(Evidence(
            url=f"https://x.com/{handle}/status/{post_id}",
            title=texto[:180], source=fuente.get("label") or f"@{handle} en X",
            published_at=_fecha(post.get("created_at")), snippet=texto,
            kind="x",
        ))
    return salida


def recolectar(fighter, session=requests, sources_path=FUENTES,
               profiles_path=identities.PROFILES):
    evidencias = google_news(fighter, session=session)
    for fuente in fuentes_de(fighter, sources_path, profiles_path):
        if (fuente.get("kind") or "").lower() == "x":
            evidencias.extend(x_posts(fuente, session=session))
        else:
            evidencias.extend(feed_publico(fuente, session=session))
    # Una URL puede aparecer en el RSS de noticias y en un feed propio. Conservamos
    # la primera: el orden da prioridad al indice de noticias, que trae publisher/fecha.
    unicas = {}
    for evidencia in evidencias:
        url = evidencia.url.split("#", 1)[0]
        unicas.setdefault(url, dataclasses.replace(evidencia, url=url))
    # Un feed de canal puede traer meses de archivo. El bot busca contexto de esta
    # semana: limita antiguedad y evita que un canal prolifico se coma todo el prompt.
    corte = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=10)
    recientes = []
    for evidencia in unicas.values():
        if evidencia.published_at:
            try:
                fecha = datetime.datetime.fromisoformat(evidencia.published_at)
                if fecha.tzinfo is None:
                    fecha = fecha.replace(tzinfo=datetime.timezone.utc)
                if fecha < corte:
                    continue
            except ValueError:
                pass
        recientes.append(evidencia)
    recientes.sort(key=lambda x: x.published_at or "", reverse=True)
    return recientes[:60]
