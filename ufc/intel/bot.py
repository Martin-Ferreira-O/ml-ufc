"""Orquestador diario de inteligencia para la proxima cartelera UFC.

Uso:
    python -m ufc.intel.bot --sin-ia
    GEMINI_API_KEY=... python -m ufc.intel.bot
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import pathlib
import re
import sys
import time
import zoneinfo

from ufc.datos import cartelera
from ufc.intel import analyzer, identities, sources, store


DEFAULT_TIMEZONE = "America/Santiago"
TRANSIENT_STATUS = {429, 500, 502, 503, 504}


def hoy_local():
    nombre = os.getenv("UFC_INTEL_TIMEZONE", DEFAULT_TIMEZONE)
    try:
        return datetime.datetime.now(zoneinfo.ZoneInfo(nombre)).date()
    except zoneinfo.ZoneInfoNotFoundError as exc:
        raise ValueError(f"Zona horaria invalida en UFC_INTEL_TIMEZONE: {nombre}") from exc


def _retry_after(exc):
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None
    for detail in details.get("error", {}).get("details", []):
        if not isinstance(detail, dict):
            continue
        value = detail.get("retryDelay")
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", str(value or ""))
        if match:
            return float(match.group(1)) + 1
    return None


def con_reintentos(call, *, sleep=time.sleep, attempts=4, label="IA"):
    """Respeta RetryInfo de Gemini y reintenta errores transitorios."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:
            response = getattr(exc, "response", None)
            code = (getattr(exc, "code", None) or
                    getattr(response, "status_code", None))
            if code not in TRANSIENT_STATUS or attempt == attempts - 1:
                raise
            delay = _retry_after(exc) or (5 * (3 ** attempt))
            print(f"REINTENTO {label}: HTTP {code}, espera {delay:.0f}s "
                  f"({attempt + 1}/{attempts - 1})")
            sleep(delay)


def _evento(eventos, fixture=False, hoy=None):
    if fixture:
        if not eventos:
            raise ValueError("El fixture no contiene eventos.")
        return eventos[0]
    hoy = hoy or hoy_local()
    limite = hoy + datetime.timedelta(days=8)
    candidatos = []
    for evento in eventos:
        try:
            fecha = datetime.date.fromisoformat(evento["fecha"][:10])
        except (KeyError, ValueError):
            continue
        if hoy <= fecha <= limite:
            candidatos.append((fecha, evento))
    if not candidatos:
        raise ValueError("No hay una cartelera UFC anunciada en los proximos 8 dias.")
    return min(candidatos, key=lambda x: x[0])[1]


def peleadores(evento):
    """[(peleador, rival)] sin duplicados, conservando el orden de la cartelera."""
    salida = []
    vistos = set()
    for pelea in evento.get("peleas", []):
        for fighter, opponent in ((pelea.get("a"), pelea.get("b")),
                                  (pelea.get("b"), pelea.get("a"))):
            if fighter and opponent and fighter not in vistos:
                salida.append((fighter, opponent))
                vistos.add(fighter)
    return salida


def _cargar_fixture(path):
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    return payload if isinstance(payload, list) else [payload]


def ejecutar(evento, *, db, provider=None, sin_ia=False, force=False,
             run_day=None, max_workers=6, recolector=sources.recolectar,
             grounding=False):
    pares = peleadores(evento)
    if not pares:
        raise ValueError("La cartelera no contiene peleadores validos.")
    if grounding and (sin_ia or not callable(getattr(provider, "discover", None))):
        raise ValueError("--grounding requiere el provider Gemini y la IA habilitada.")
    run_day = run_day or hoy_local().isoformat()
    nombre_provider = "none" if sin_ia else provider.name
    modelo = None if sin_ia else provider.model
    run_id = store.crear_run(db, evento, run_day, nombre_provider, modelo, len(pares))
    omitidos = [(f, o) for f, o in pares
                if not force and store.ya_revisado(
                    db, run_day, evento["fecha"], f, require_analysis=not sin_ia)]
    pendientes = [(f, o) for f, o in pares if (f, o) not in omitidos]
    resultados = {}
    if pendientes:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futuros = {pool.submit(recolector, f): (f, o) for f, o in pendientes}
            for futuro in concurrent.futures.as_completed(futuros):
                fighter, opponent = futuros[futuro]
                try:
                    resultados[fighter] = (opponent, futuro.result(), None)
                except Exception as exc:  # cada peleador se reintenta en el proximo run
                    resultados[fighter] = (opponent, [], str(exc))

    completos = len(omitidos)
    errores = []
    for fighter, opponent in pendientes:
        _, evidencias, error = resultados[fighter]
        if error:
            report = {"resumen": f"Fallo la recoleccion: {error}",
                      "estado": "error", "confianza": None,
                      "valoracion": None, "hallazgos": []}
            store.guardar_check(db, run_id, run_day, evento, fighter, opponent,
                                "error", evidencias, report)
            errores.append(f"{fighter}: {error}")
            print(f"ERROR {fighter}: {error}")
            continue
        try:
            if sin_ia:
                report, usage, status = analyzer.sin_ia(evidencias), {}, "collected"
            else:
                discovery_usage = {}
                warnings = []
                if grounding:
                    try:
                        discovered, discovery_usage = con_reintentos(
                            lambda: provider.discover(fighter, opponent, evento),
                            label=f"grounding {fighter}")
                        unique = {}
                        for evidence in [*evidencias, *discovered]:
                            unique.setdefault(evidence.url.split("#", 1)[0], evidence)
                        evidencias = list(unique.values())
                    except Exception as exc:
                        warnings.append(f"Google Search grounding fallo: {exc}")
                report, usage = con_reintentos(
                    lambda: provider.analyze(fighter, opponent, evento, evidencias),
                    label=f"analisis {fighter}")
                for key in ("prompt_tokens", "output_tokens"):
                    usage[key] = (usage.get(key, 0) or 0) + \
                        (discovery_usage.get(key, 0) or 0)
                if warnings:
                    report["advertencias"] = warnings
                status = "complete"
            store.guardar_check(db, run_id, run_day, evento, fighter, opponent,
                                status, evidencias, report, usage)
            completos += 1
            score = "sin IA" if report["valoracion"] is None else f"{report['valoracion']:+d}"
            print(f"OK    {fighter}: {len(evidencias)} evidencias, valoracion {score}")
        except Exception as exc:
            report = {"resumen": f"Fallo el analisis: {exc}", "estado": "error",
                      "confianza": None, "valoracion": None, "hallazgos": []}
            store.guardar_check(db, run_id, run_day, evento, fighter, opponent,
                                "error", evidencias, report)
            errores.append(f"{fighter}: {exc}")
            print(f"ERROR {fighter}: {exc}")

    status = "complete" if completos == len(pares) else "partial"
    store.terminar_run(db, run_id, status, completos, len(omitidos),
                       " | ".join(errores) or None)
    print(f"\nCobertura: {completos}/{len(pares)} "
          f"({len(pendientes) - len(errores)} nuevos, {len(omitidos)} omitidos)")
    return {"run_id": run_id, "total": len(pares), "completed": completos,
            "skipped": len(omitidos), "errors": errores, "status": status}


def salud(db, max_age_hours=36, now=None):
    """Comprueba frescura, cobertura total y que el ultimo run haya usado IA."""
    run = store.ultimo_run(db)
    if not run:
        print("ERROR salud: todavia no hay ejecuciones archivadas.")
        return False
    now = now or datetime.datetime.now(datetime.timezone.utc)
    raw_time = run.get("finished_at") or run["started_at"]
    try:
        timestamp = datetime.datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=datetime.timezone.utc)
        age_hours = max(0, (now - timestamp.astimezone(datetime.timezone.utc)).total_seconds()
                        / 3600)
    except (AttributeError, ValueError):
        age_hours = float("inf")

    reasons = []
    if run["status"] != "complete":
        reasons.append(f"estado {run['status']}")
    if run["completed_fighters"] != run["total_fighters"]:
        reasons.append(f"cobertura {run['completed_fighters']}/{run['total_fighters']}")
    if run["provider"] == "none":
        reasons.append("sin analisis de IA")
    if age_hours > max_age_hours:
        reasons.append(f"ultimo run hace {age_hours:.1f} h")

    prefix = "OK" if not reasons else "ERROR"
    print(f"{prefix} salud: {run['event_name']} — {run['run_day']} — "
          f"{run['completed_fighters']}/{run['total_fighters']} — "
          f"{run['provider']}/{run.get('model') or '-'} — {age_hours:.1f} h")
    if reasons:
        print("Motivos: " + " | ".join(reasons))
    return not reasons


def imprimir_resumen(db):
    evento = store.ultimo_evento(db)
    if not evento:
        print("Todavia no hay informes de inteligencia.")
        return False
    print(f"{evento['event_name']} — {evento['event_date']} — "
          f"{evento['completed']}/{len(evento['checks'])} peleadores")
    relevantes = []
    sin_novedades = []
    for check in evento["checks"]:
        hallazgos = check["report"].get("hallazgos", [])
        if not hallazgos:
            sin_novedades.append(check["fighter"])
            continue
        relevantes.append(check)
    for check in relevantes:
        score = f"{check['score']:+d}" if check["score"] is not None else "sin analizar"
        print(f"\n{check['fighter']} vs {check['opponent']} — {score} — "
              f"confianza {check['confidence'] or 'sin asignar'}")
        print(check["summary"])
        by_ref = {f"E{x['position']}": x for x in check["evidencias"]}
        for finding in check["report"].get("hallazgos", []):
            print(f"  - {finding['titulo']} ({finding['impacto']:+d}, "
                  f"{finding['certeza']})")
            for ref in finding.get("evidencias", []):
                evidence = by_ref.get(ref)
                if evidence:
                    print(f"    {ref} {evidence['source']}: {evidence['url']}")
    print(f"\nSin novedades materiales ({len(sin_novedades)}): " +
          ", ".join(sin_novedades))
    return True


def argumentos(argv=None):
    p = argparse.ArgumentParser(description="Vigilancia diaria de la cartelera UFC")
    p.add_argument("--sin-ia", action="store_true",
                   help="archiva evidencias sin llamar a un LLM")
    p.add_argument("--provider", choices=["gemini", "openai-compatible"],
                   default="gemini")
    p.add_argument("--force", action="store_true", help="repite los checks de hoy")
    p.add_argument("--evento-fixture", help="JSON local de evento (pruebas/diagnostico)")
    p.add_argument("--sin-red", action="store_true",
                   help="solo con --evento-fixture: usa cero evidencias")
    p.add_argument("--sin-identidades", action="store_true",
                   help="omite el descubrimiento cacheado de perfiles sociales")
    p.add_argument("--actualizar-identidades", action="store_true",
                   help="fuerza refrescar el cache de perfiles sociales")
    p.add_argument("--grounding", action="store_true",
                   default=os.getenv("UFC_INTEL_GROUNDING", "").lower()
                   in {"1", "true", "yes", "si"},
                   help="agrega busqueda web citada de Gemini (una llamada por peleador)")
    p.add_argument("--status", action="store_true",
                   help="verifica que el ultimo run sea reciente, completo y con IA")
    p.add_argument("--resumen", action="store_true",
                   help="imprime scores, hallazgos y URLs del ultimo informe")
    p.add_argument("--metadata-json", action="store_true",
                   help="imprime metadata JSON para comparar copias de la base")
    p.add_argument("--max-age-hours", type=float, default=36,
                   help="frescura maxima admitida por --status (default: 36)")
    p.add_argument("--db", type=pathlib.Path,
                   help="base SQLite alternativa (util para pruebas/diagnostico)")
    p.add_argument("--max-workers", type=int, default=6)
    return p.parse_args(argv)


def main(argv=None):
    args = argumentos(argv)
    try:
        if args.status:
            with store.conectar(args.db or store.DB) as db:
                return 0 if salud(db, max(1, args.max_age_hours)) else 1
        if args.resumen:
            with store.conectar(args.db or store.DB) as db:
                return 0 if imprimir_resumen(db) else 1
        if args.metadata_json:
            with store.conectar(args.db or store.DB) as db:
                print(json.dumps(store.metadata(db), ensure_ascii=False,
                                 sort_keys=True))
            return 0
        eventos = (_cargar_fixture(args.evento_fixture) if args.evento_fixture
                   else cartelera.proximas(dias=8))
        if args.sin_red and not args.evento_fixture:
            raise ValueError("--sin-red solo se admite con --evento-fixture.")
        evento = _evento(eventos, fixture=bool(args.evento_fixture))
        provider = None if args.sin_ia else analyzer.proveedor(args.provider)
        pairs = peleadores(evento)
        print(f"{evento['evento']} — {evento['fecha']} — "
              f"{len(pairs)} peleadores")
        if not args.sin_identidades and not args.evento_fixture:
            identity = identities.sync([fighter for fighter, _ in pairs],
                                       force=args.actualizar_identidades)
            print(f"Perfiles sociales: {identity['covered']}/{identity['fighters']} "
                  f"oficiales, {identity['candidates']} candidatos "
                  f"({identity['checked']} consultados, "
                  f"{identity['errors']} errores)")
        with store.conectar(args.db or store.DB) as db:
            resultado = ejecutar(evento, db=db, provider=provider,
                                  sin_ia=args.sin_ia, force=args.force,
                                  max_workers=max(1, args.max_workers),
                                  grounding=args.grounding,
                                  recolector=(lambda fighter: []) if args.sin_red
                                  else sources.recolectar)
        return 0 if resultado["status"] == "complete" else 1
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
