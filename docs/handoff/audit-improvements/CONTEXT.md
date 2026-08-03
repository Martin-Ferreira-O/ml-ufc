> Handoff doc for task `audit-improvements`. Author: Codex. Updated: 2026-08-02 16:27.
> IMPLEMENTING AGENT: read CONTEXT.md -> PLAN.md -> PROGRESS.md -> DECISIONS.md before starting.
> Spec written against commit `a84877b` on branch `audit-improvements`; source plan: `posibles_mejoras.md`.

# CONTEXT — audit-improvements

## Task

Aplicar las mejoras accionables y prioritarias de `posibles_mejoras.md` sin presentar
experimentos no ejecutados como avances validados. Corregir ademas la regresion reportada:
los eventos pasados deben ser visibles y seleccionables en Predictores para cargar el
historial humano.

## Read first

- `posibles_mejoras.md`, especialmente secciones 3, 5–10 y backlog P0/P1.
- `ufc/ui/tab_predictores.py`, `ufc/datos/cartelera.py` y `test_app.py`.
- `ufc/modelo/features.py`, `ufc/modelo/train.py` y `ufc/modelo/predict.py`.
- `ufc/registro/predictores.py` y `ufc/registro/ledger.py`.

## Constraints

- UI en espanol, Streamlit 1.60 y componentes nativos.
- Preservar los CSV existentes y mantener lectura retrocompatible de bundles/ledgers.
- No activar recomendaciones de apuestas sin validacion prospectiva.
- No afirmar que estan implementadas fuentes/IDs/modelos que requieren datos no presentes.

