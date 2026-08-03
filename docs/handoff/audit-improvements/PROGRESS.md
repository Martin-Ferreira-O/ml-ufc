> Handoff doc for task `audit-improvements`. Author: Codex. Updated: 2026-08-02 16:27.

# PROGRESS — audit-improvements

## Checklist

- [x] T1 — Eventos pasados visibles y navegables en Predictores
- [x] T2 — Serving temporal y circunstancias desconocidas
- [x] T3 — Default sin apuesta y lenguaje economico honesto
- [x] T4 — Taxonomia, reloj y exposiciones observadas
- [x] T5 — Manifiesto atomico y auditoria append-only
- [x] T6 — Medicion clusterizada e invariantes
- [x] T7 — Reentrenamiento, suite y QA visual

## Work log

- 2026-08-02 16:27 — Codex — Rama y handoff creados desde `posibles_mejoras.md`; se confirma que hay 20 carteleras historicas locales pero quedan ocultas al final del selector de Predictores.
- 2026-08-02 16:31 — Codex — Predictores separa Proximos/Pasados, deduplica eventos por fecha/nombre normalizado y evita recalcular el modelo actual sobre peleas historicas.
- 2026-08-02 16:34 — Codex — Serving usa fecha programada y circunstancias si/no/desconocido; se retiran candidatas automaticas y el lenguaje economico pasa a seguimiento experimental.
- 2026-08-02 16:36 — Codex — Taxonomia versionada, reloj por TIME FORMAT y exposiciones observadas reconstruidos; features/modelo reentrenados.
- 2026-08-02 16:43 — Codex — Bundle atomico con hashes/commit/dependencias/metricas/dominio, metricas de coincidencia serializadas, audit log append-only y bootstrap por evento agregados.
- 2026-08-02 16:46 — Codex — Suite completa: 16 checks ok. QA visual 1440x1000 y 390x844 sin overflow; cambio a evento historico y vista Picks verificados.
- 2026-08-02 16:47 — Codex — Duplicados historicos con diferencias de acento fusionan la clave de picks con peso/ganador local; regression test confirma preseleccion del ganador.
