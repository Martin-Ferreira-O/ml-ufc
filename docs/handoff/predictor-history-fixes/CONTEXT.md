> Handoff doc for task `predictor-history-fixes`. Author: Codex. Updated: 2026-08-02 19:48.
> IMPLEMENTING AGENT: read CONTEXT.md -> PLAN.md -> PROGRESS.md -> DECISIONS.md before starting.
> Spec written against commit `a84877b` on branch `predictor-history-fixes`; source plan: agreed in Codex plan mode.

# CONTEXT — predictor-history-fixes

## Task

Corregir la perdida del predictor seleccionado y el feedback de guardado en Picks;
bloquear los ganadores oficiales; y calcular la precision de cada predictor contra todos
los resultados oficiales disponibles, mostrando tambien el numero de carteleras evaluadas.

## Read first

- `ufc/ui/tab_predictores.py` para estado, guardado, ganadores y comparativa.
- `ufc/registro/predictores.py` para persistencia, aciertos y peso conservador.
- `ufc/datos/cartelera.py` para resultados oficiales locales de UFCStats.
- `ufc/ui/tab_resumen.py` y `test_app.py` para consumidores y contratos.

## Constraints

- UI en espanol, Streamlit 1.60 y componentes nativos.
- Preservar los CSV existentes y las correcciones de `audit-improvements` presentes en
  el arbol de trabajo.
- UFCStats es autoritativo cuando tiene un ganador; el resultado manual es solo fallback.

