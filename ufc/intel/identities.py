"""Descubre perfiles sociales desde fichas UFC; Wikidata queda como candidato."""

import csv
import datetime
import html
import json
import os
import re
import tempfile
import time
import urllib.parse

import requests

from ufc import nombres, rutas


API = "https://www.wikidata.org/w/api.php"
UFC_ATHLETE = "https://www.ufc.com/athlete/{slug}"
PROFILES = rutas.DATOS / "intel_profiles.csv"
STATUS = rutas.DATOS / "intel_identity_status.csv"
PROFILE_FIELDS = ["fighter", "platform", "identifier", "url", "kind", "feed_url",
                  "source", "confidence", "wikidata_id", "resolved_at"]
STATUS_FIELDS = ["fighter", "status", "wikidata_id", "description", "checked_at",
                 "error"]
_SPORT_WORDS = ("mixed martial", "mma", "martial artist", "martial arts fighter",
                "ufc fighter", "kickboxer")
_JSON_LD = re.compile(
    r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.I | re.S,
)


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _read(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="",
                                     dir=path.parent, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fields} for row in rows)
        temp = f.name
    os.replace(temp, path)


def _get(session, params, timeout=30):
    for attempt in range(3):
        r = session.get(API, params=params, timeout=timeout,
                        headers={"User-Agent": "ml-ufc-intel/1.0"})
        if getattr(r, "status_code", None) == 429 and attempt < 2:
            time.sleep(min(float(r.headers.get("Retry-After", 1)), 3))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("Wikidata no respondio despues de los reintentos.")


def _slug(fighter):
    return "-".join(re.findall(r"[a-z0-9]+", nombres.normalizar(fighter)))


def _objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _social_profile(fighter, url, checked_at):
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path.strip("/")
    if domain in {"x.com", "twitter.com"} and path:
        identifier = path.split("/")[0].lstrip("@")
        return {"fighter": fighter, "platform": "x", "identifier": identifier,
                "url": f"https://x.com/{identifier}", "kind": "x",
                "feed_url": f"https://x.com/{identifier}"}
    if domain == "instagram.com" and path:
        identifier = path.split("/")[0].lstrip("@")
        return {"fighter": fighter, "platform": "instagram", "identifier": identifier,
                "url": f"https://www.instagram.com/{identifier}/", "kind": "",
                "feed_url": ""}
    if domain in {"youtube.com", "m.youtube.com"} and path:
        channel = re.search(r"(?:^|/)channel/(UC[-_0-9A-Za-z]{22})", "/" + path)
        identifier = channel.group(1) if channel else path
        return {"fighter": fighter, "platform": "youtube", "identifier": identifier,
                "url": url, "kind": "youtube" if channel else "",
                "feed_url": (f"https://www.youtube.com/feeds/videos.xml?channel_id="
                             f"{identifier}" if channel else "")}
    if domain == "tiktok.com" and path:
        identifier = path.split("/")[0].lstrip("@")
        return {"fighter": fighter, "platform": "tiktok", "identifier": identifier,
                "url": url, "kind": "", "feed_url": ""}
    return None


def _ufc_profiles(fighter, session, checked_at, timeout=30):
    url = UFC_ATHLETE.format(slug=_slug(fighter))
    r = session.get(url, timeout=timeout, headers={"User-Agent": "ml-ufc-intel/1.0"})
    if getattr(r, "status_code", None) == 404:
        return None
    r.raise_for_status()
    person = None
    for raw in _JSON_LD.findall(r.text):
        try:
            value = json.loads(html.unescape(raw.strip()))
        except json.JSONDecodeError:
            continue
        for obj in _objects(value):
            types = obj.get("@type", [])
            types = [types] if isinstance(types, str) else types
            if ("Person" in types and
                    nombres.normalizar(obj.get("name", "")) == nombres.normalizar(fighter)):
                person = obj
                break
        if person:
            break
    if not person:
        return None
    profiles = []
    same_as = person.get("sameAs", []) or []
    same_as = [same_as] if isinstance(same_as, str) else same_as
    for social_url in same_as:
        profile = _social_profile(fighter, social_url, checked_at)
        if profile:
            profile.update({"source": "ufc", "confidence": "official",
                            "wikidata_id": "", "resolved_at": checked_at})
            profiles.append(profile)
    return profiles


def _candidate(fighter, results):
    key = nombres.normalizar(fighter)
    exact = [x for x in results if nombres.normalizar(x.get("label", "")) == key]
    sports = [x for x in exact
              if any(word in x.get("description", "").lower()
                     for word in _SPORT_WORDS)]
    return sports[0] if len(sports) == 1 else None


def _values(entity, prop):
    statements = [x for x in entity.get("claims", {}).get(prop, [])
                  if x.get("rank") != "deprecated"
                  and "datavalue" in x.get("mainsnak", {})]
    preferred = [x for x in statements if x.get("rank") == "preferred"]
    statements = preferred or statements
    return list(dict.fromkeys(str(x["mainsnak"]["datavalue"]["value"])
                              for x in statements))


def resolve(fighter, session=requests, now=None):
    """UFC oficial primero; Wikidata solo produce candidatos no recolectables."""
    checked_at = (now or _now()).isoformat(timespec="seconds")
    official = _ufc_profiles(fighter, session, checked_at)
    if official is not None:
        status = "resolved" if official else "no_social"
        return official, {"fighter": fighter, "status": status, "wikidata_id": "",
                          "description": "Perfil oficial UFC", "checked_at": checked_at,
                          "error": ""}
    search = _get(session, {"action": "wbsearchentities", "search": fighter,
                            "language": "en", "format": "json", "limit": 5})
    candidate = _candidate(fighter, search.get("search", []))
    if not candidate:
        return [], {"fighter": fighter, "status": "not_found", "wikidata_id": "",
                    "description": "", "checked_at": checked_at, "error": ""}
    qid = candidate["id"]
    payload = _get(session, {"action": "wbgetentities", "ids": qid,
                             "props": "claims", "format": "json"})
    entity = payload.get("entities", {}).get(qid, {})
    profiles = []
    specs = [
        ("P2002", "x", lambda x: f"https://x.com/{x}",
         lambda x: f"https://x.com/{x}"),
        ("P2003", "instagram", lambda x: f"https://www.instagram.com/{x}/",
         lambda x: ""),
        ("P2397", "youtube",
         lambda x: f"https://www.youtube.com/channel/{x}",
         lambda x: f"https://www.youtube.com/feeds/videos.xml?channel_id={x}"),
    ]
    for prop, platform, url_of, feed_of in specs:
        for value in _values(entity, prop):
            profiles.append({"fighter": fighter, "platform": platform,
                             "identifier": value, "url": url_of(value), "kind": "",
                             "feed_url": feed_of(value), "source": "wikidata",
                             "confidence": "candidate", "wikidata_id": qid,
                             "resolved_at": checked_at})
    status = "candidate" if profiles else "no_social"
    return profiles, {"fighter": fighter, "status": status, "wikidata_id": qid,
                      "description": candidate.get("description", ""),
                      "checked_at": checked_at, "error": ""}


def _fresh(row, now, days):
    try:
        checked = datetime.datetime.fromisoformat(row.get("checked_at", ""))
        if checked.tzinfo is None:
            checked = checked.replace(tzinfo=datetime.timezone.utc)
        return now - checked < datetime.timedelta(days=days)
    except ValueError:
        return False


def sync(fighters, *, session=requests, profiles_path=PROFILES, status_path=STATUS,
         max_age_days=30, force=False, now=None, delay=0.2):
    """Resuelve solo identidades nuevas/vencidas y nunca borra un cache por error."""
    now = now or _now()
    profiles = _read(profiles_path)
    statuses = _read(status_path)
    by_fighter = {nombres.normalizar(x["fighter"]): x for x in statuses}
    requested = list(dict.fromkeys(fighters))
    checked = 0
    for fighter in requested:
        key = nombres.normalizar(fighter)
        if not force and key in by_fighter and _fresh(by_fighter[key], now, max_age_days):
            continue
        checked += 1
        try:
            found, status = resolve(fighter, session=session, now=now)
        except Exception as exc:
            previous = by_fighter.get(key, {})
            status = {"fighter": fighter, "status": "error",
                      "wikidata_id": previous.get("wikidata_id", ""),
                      "description": previous.get("description", ""),
                      "checked_at": now.isoformat(timespec="seconds"),
                      "error": str(exc)[:500]}
            found = None
        if found is not None:
            profiles = [x for x in profiles if nombres.normalizar(x["fighter"]) != key]
            profiles.extend(found)
        statuses = [x for x in statuses if nombres.normalizar(x["fighter"]) != key]
        statuses.append(status)
        by_fighter[key] = status
        if delay:
            time.sleep(delay)
    _write(profiles_path, PROFILE_FIELDS, profiles)
    _write(status_path, STATUS_FIELDS, statuses)
    current = {nombres.normalizar(x) for x in requested}
    covered = {nombres.normalizar(x["fighter"]) for x in profiles
               if nombres.normalizar(x["fighter"]) in current
               and x.get("confidence") == "official"}
    candidates = {nombres.normalizar(x["fighter"]) for x in profiles
                  if nombres.normalizar(x["fighter"]) in current
                  and x.get("confidence") == "candidate"}
    return {"fighters": len(requested), "covered": len(covered), "checked": checked,
            "candidates": len(candidates),
            "errors": sum(by_fighter.get(nombres.normalizar(x), {}).get("status") == "error"
                          for x in requested)}


def profiles_for(fighter, path=PROFILES):
    key = nombres.normalizar(fighter)
    return [x for x in _read(path) if nombres.normalizar(x["fighter"]) == key]


def feed_sources(fighter, path=PROFILES):
    """Fuentes automaticamente recolectables; Instagram queda como identidad visible."""
    output = []
    for profile in profiles_for(fighter, path):
        if profile.get("confidence") != "official":
            continue
        kind = profile.get("kind", "")
        if kind == "x" and not os.getenv("X_BEARER_TOKEN"):
            continue
        if kind in {"x", "youtube"} and profile.get("feed_url"):
            output.append({"fighter": fighter, "kind": kind,
                           "label": f"{fighter} — {profile['platform']} (Wikidata)",
                           "url": profile["feed_url"]})
    return output
