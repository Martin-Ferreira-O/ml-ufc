"""Archivo SQLite de runs, checks y evidencias del bot de inteligencia."""

import datetime
import json
import sqlite3

from ufc import rutas


DB = rutas.DATOS / "intel.db"


def conectar(path=DB):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS intel_runs (
            id INTEGER PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            run_day TEXT NOT NULL,
            event_name TEXT NOT NULL,
            event_date TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT,
            status TEXT NOT NULL,
            total_fighters INTEGER NOT NULL,
            completed_fighters INTEGER NOT NULL DEFAULT 0,
            skipped_fighters INTEGER NOT NULL DEFAULT 0,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS intel_checks (
            id INTEGER PRIMARY KEY,
            run_id INTEGER NOT NULL REFERENCES intel_runs(id),
            run_day TEXT NOT NULL,
            event_name TEXT NOT NULL,
            event_date TEXT NOT NULL,
            fighter TEXT NOT NULL,
            opponent TEXT NOT NULL,
            checked_at TEXT NOT NULL,
            status TEXT NOT NULL,
            evidence_count INTEGER NOT NULL,
            score INTEGER,
            confidence TEXT,
            summary TEXT NOT NULL,
            report_json TEXT NOT NULL,
            prompt_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            UNIQUE(run_day, event_date, fighter)
        );
        CREATE TABLE IF NOT EXISTS intel_evidence (
            id INTEGER PRIMARY KEY,
            url TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            source TEXT NOT NULL,
            published_at TEXT,
            snippet TEXT NOT NULL,
            kind TEXT NOT NULL,
            first_seen_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS intel_check_evidence (
            check_id INTEGER NOT NULL REFERENCES intel_checks(id) ON DELETE CASCADE,
            evidence_id INTEGER NOT NULL REFERENCES intel_evidence(id),
            position INTEGER NOT NULL,
            PRIMARY KEY (check_id, evidence_id)
        );
        CREATE INDEX IF NOT EXISTS idx_intel_checks_event
            ON intel_checks(event_date, checked_at);
        CREATE INDEX IF NOT EXISTS idx_intel_checks_fighter
            ON intel_checks(fighter, checked_at);
    """)
    return db


def ahora_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def crear_run(db, evento, run_day, provider, model, total):
    cur = db.execute("""
        INSERT INTO intel_runs
            (started_at, run_day, event_name, event_date, provider, model, status,
             total_fighters)
        VALUES (?, ?, ?, ?, ?, ?, 'running', ?)
    """, (ahora_utc(), run_day, evento["evento"], evento["fecha"], provider,
          model, total))
    db.commit()
    return cur.lastrowid


def terminar_run(db, run_id, status, completed, skipped, error=None):
    db.execute("""
        UPDATE intel_runs SET finished_at=?, status=?, completed_fighters=?,
            skipped_fighters=?, error=? WHERE id=?
    """, (ahora_utc(), status, completed, skipped, error, run_id))
    db.commit()


def ya_revisado(db, run_day, event_date, fighter, require_analysis=False):
    fila = db.execute("""
        SELECT status FROM intel_checks
        WHERE run_day=? AND event_date=? AND fighter=?
    """, (run_day, event_date, fighter)).fetchone()
    estados = {"complete"} if require_analysis else {"complete", "collected"}
    return bool(fila and fila["status"] in estados)


def ultimo_run(db):
    fila = db.execute("""
        SELECT * FROM intel_runs ORDER BY id DESC LIMIT 1
    """).fetchone()
    return dict(fila) if fila else None


def _guardar_evidencia(db, evidencia):
    db.execute("""
        INSERT INTO intel_evidence
            (url, title, source, published_at, snippet, kind, first_seen_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            title=excluded.title, source=excluded.source,
            published_at=COALESCE(excluded.published_at, intel_evidence.published_at),
            snippet=excluded.snippet, kind=excluded.kind
    """, (evidencia.url, evidencia.title, evidencia.source,
          evidencia.published_at, evidencia.snippet, evidencia.kind, ahora_utc()))
    return db.execute("SELECT id FROM intel_evidence WHERE url=?", (evidencia.url,)).fetchone()[0]


def guardar_check(db, run_id, run_day, evento, fighter, opponent, status,
                  evidencias, report, usage=None):
    usage = usage or {}
    checked_at = ahora_utc()
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True)
    valores = (run_id, run_day, evento["evento"], evento["fecha"], fighter,
               opponent, checked_at, status, len(evidencias), report.get("valoracion"),
               report.get("confianza"), report.get("resumen", ""), payload,
               int(usage.get("prompt_tokens", 0) or 0),
               int(usage.get("output_tokens", 0) or 0))
    db.execute("""
        INSERT INTO intel_checks
            (run_id, run_day, event_name, event_date, fighter, opponent, checked_at,
             status, evidence_count, score, confidence, summary, report_json,
             prompt_tokens, output_tokens)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_day, event_date, fighter) DO UPDATE SET
            run_id=excluded.run_id, opponent=excluded.opponent,
            checked_at=excluded.checked_at, status=excluded.status,
            evidence_count=excluded.evidence_count, score=excluded.score,
            confidence=excluded.confidence, summary=excluded.summary,
            report_json=excluded.report_json,
            prompt_tokens=excluded.prompt_tokens, output_tokens=excluded.output_tokens
    """, valores)
    check_id = db.execute("""
        SELECT id FROM intel_checks
        WHERE run_day=? AND event_date=? AND fighter=?
    """, (run_day, evento["fecha"], fighter)).fetchone()[0]
    db.execute("DELETE FROM intel_check_evidence WHERE check_id=?", (check_id,))
    for position, evidencia in enumerate(evidencias, 1):
        evidencia_id = _guardar_evidencia(db, evidencia)
        db.execute("INSERT OR IGNORE INTO intel_check_evidence VALUES (?, ?, ?)",
                   (check_id, evidencia_id, position))
    db.commit()
    return check_id


def ultimo_evento(db=None):
    cerrar = db is None
    db = db or conectar()
    fila = db.execute("""
        SELECT event_name, event_date, MAX(checked_at) AS updated_at,
               COUNT(*) AS fighters,
               SUM(status IN ('complete', 'collected')) AS completed,
               SUM(prompt_tokens) AS prompt_tokens,
               SUM(output_tokens) AS output_tokens
        FROM intel_checks
        GROUP BY event_name, event_date
        ORDER BY event_date DESC, updated_at DESC LIMIT 1
    """).fetchone()
    if not fila:
        if cerrar:
            db.close()
        return None
    evento = dict(fila)
    checks = db.execute("""
        SELECT * FROM intel_checks WHERE event_name=? AND event_date=?
        ORDER BY CASE WHEN score IS NULL THEN 1 ELSE 0 END, score ASC, fighter
    """, (fila["event_name"], fila["event_date"])).fetchall()
    evento["checks"] = []
    for check in checks:
        item = dict(check)
        item["report"] = json.loads(item.pop("report_json"))
        item["evidencias"] = [dict(x) for x in db.execute("""
            SELECT e.*, ce.position AS position FROM intel_evidence e
            JOIN intel_check_evidence ce ON ce.evidence_id=e.id
            WHERE ce.check_id=? ORDER BY ce.position
        """, (check["id"],)).fetchall()]
        evento["checks"].append(item)
    if cerrar:
        db.close()
    return evento
