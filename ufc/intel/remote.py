"""Consulta y sincroniza los artefactos de inteligencia desde la VPS por SSH."""

import csv
import datetime
import json
import os
import pathlib
import shlex
import shutil
import sqlite3
import subprocess
import tempfile
import zoneinfo
from dataclasses import dataclass

from ufc import rutas
from ufc.intel import identities, store


WINDOWS = ((9, 15), (10, 15), (11, 15))
RANDOM_DELAY_MINUTES = 15
TIMEZONE = "America/Santiago"
FILES = {
    "intel.db": store.DB,
    "intel_profiles.csv": identities.PROFILES,
    "intel_identity_status.csv": identities.STATUS,
}


class RemoteError(RuntimeError):
    """Fallo controlado al consultar o descargar desde la VPS."""


@dataclass(frozen=True)
class Config:
    host: str
    user: str
    port: int = 22
    key: str = "~/.ssh/id_ed25519"
    root: str = "/opt/ml-ufc"

    @classmethod
    def from_mapping(cls, values):
        return cls(host=str(values["host"]), user=str(values["user"]),
                   port=int(values.get("port", 22)),
                   key=str(values.get("key", "~/.ssh/id_ed25519")),
                   root=str(values.get("root", "/opt/ml-ufc")))

    @property
    def target(self):
        return f"{self.user}@{self.host}"

    @property
    def expanded_key(self):
        return str(pathlib.Path(self.key).expanduser())


def _connection_args(config, *, scp=False):
    return [
        "scp" if scp else "ssh",
        "-P" if scp else "-p", str(config.port),
        "-i", config.expanded_key,
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "StrictHostKeyChecking=accept-new",
    ]


def _failure(exc):
    detail = (getattr(exc, "stderr", None) or str(exc)).strip()
    return detail.splitlines()[-1] if detail else "fallo sin detalle"


def consultar(config, *, runner=subprocess.run):
    command = (f"cd {shlex.quote(config.root)} && "
               ".venv/bin/python -m ufc.intel.bot --metadata-json")
    try:
        result = runner([*_connection_args(config), config.target, command],
                        capture_output=True, text=True, check=True, timeout=15)
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise RemoteError(f"No se pudo consultar la VPS: {_failure(exc)}") from exc


def _event(metadata):
    return (metadata or {}).get("event")


def hay_novedades(remote_metadata, local_metadata):
    remote_event, local_event = _event(remote_metadata), _event(local_metadata)
    if not remote_event:
        return False
    if not local_event:
        return True
    remote_key = (remote_event.get("event_date") or "",
                  remote_event.get("run_day") or "",
                  remote_event.get("updated_at") or "")
    local_key = (local_event.get("event_date") or "",
                 local_event.get("run_day") or "",
                 local_event.get("updated_at") or "")
    return remote_key > local_key


def proxima_ventana(now=None):
    zone = zoneinfo.ZoneInfo(TIMEZONE)
    now = now.astimezone(zone) if now else datetime.datetime.now(zone)
    for hour, minute in WINDOWS:
        start = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        end = start + datetime.timedelta(minutes=RANDOM_DELAY_MINUTES)
        if now <= end:
            return start, end
    tomorrow = now + datetime.timedelta(days=1)
    start = tomorrow.replace(hour=WINDOWS[0][0], minute=WINDOWS[0][1],
                             second=0, microsecond=0)
    return start, start + datetime.timedelta(minutes=RANDOM_DELAY_MINUTES)


def _validate_db(path):
    db = None
    try:
        db = sqlite3.connect(path)
        db.row_factory = sqlite3.Row
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RemoteError("La copia SQLite descargada no paso integrity_check.")
        if db.execute("PRAGMA foreign_key_check").fetchone():
            raise RemoteError("La copia SQLite descargada tiene referencias invalidas.")
        required = {"intel_runs", "intel_checks", "intel_evidence"}
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if not required <= tables:
            raise RemoteError("La copia descargada no tiene el esquema de inteligencia.")
        return store.metadata(db)
    except sqlite3.Error as exc:
        raise RemoteError(f"La copia SQLite descargada es invalida: {exc}") from exc
    finally:
        if db is not None:
            db.close()


def _validate_csv(path, required):
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            fields = set(csv.DictReader(handle).fieldnames or [])
    except (OSError, UnicodeError, csv.Error) as exc:
        raise RemoteError(f"CSV remoto invalido ({path.name}): {exc}") from exc
    if not required <= fields:
        raise RemoteError(f"CSV remoto sin columnas esperadas: {path.name}")


def sincronizar(config, *, runner=subprocess.run):
    """Descarga, valida y reemplaza los artefactos; conserva copias .backup."""
    target_dir = FILES["intel.db"].parent
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".intel-sync-", dir=target_dir) as td:
        temp_dir = pathlib.Path(td)
        downloaded = {}
        for name in FILES:
            destination = temp_dir / name
            source = f"{config.target}:{config.root}/data/{name}"
            try:
                runner([*_connection_args(config, scp=True), source, str(destination)],
                       capture_output=True, text=True, check=True, timeout=45)
            except (OSError, subprocess.SubprocessError) as exc:
                raise RemoteError(f"No se pudo descargar {name}: {_failure(exc)}") from exc
            downloaded[name] = destination

        metadata = _validate_db(downloaded["intel.db"])
        _validate_csv(downloaded["intel_profiles.csv"],
                      {"fighter", "platform", "url", "confidence"})
        _validate_csv(downloaded["intel_identity_status.csv"],
                      {"fighter", "status", "checked_at"})

        for name, target in FILES.items():
            if target.exists():
                shutil.copy2(target, target.with_name(target.name + ".backup"))
            os.replace(downloaded[name], target)
        return metadata
