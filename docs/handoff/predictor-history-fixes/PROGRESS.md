> Handoff doc for task `predictor-history-fixes`. Author: Codex. Updated: 2026-08-02 19:48.

# PROGRESS — predictor-history-fixes

## Checklist

- [x] T1 — Estado persistente y feedback de guardado
- [x] T2 — Resultados oficiales consolidados y de solo lectura
- [x] T3 — Precision historica y numero de carteleras
- [x] T4 — Suite y QA visual

## Work log

- 2026-08-02 19:48 — Codex — Rama y handoff creados desde el plan acordado; causa confirmada: `resultados.csv` solo cubre una cartelera aunque UFCStats contiene las cinco de FaceOff MMA.
- 2026-08-02 19:52 — Codex — Selector persistente, confirmacion post-rerun, precedencia UFCStats, bloqueo de ganadores y columna Carteleras implementados; suite completa con 16 checks ok.
- 2026-08-02 19:55 — Codex — QA local en instancia limpia confirma columna Carteleras y valores CagePropHet 11/14/1 y FaceOff MMA 47/65/5; instancia temporal retirada.
- 2026-08-02 22:13 — Codex — Seguimiento visual agregado: en carteleras pasadas la pick elegida muestra badge verde/Acertó o rojo/Falló; AppTest cubre ambos estados y los 16 checks pasan.
