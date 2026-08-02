> Handoff doc for task `predictor-ux-dashboard`. Author: Codex. Updated: 2026-08-01 22:33.
> IMPLEMENTING AGENT: read CONTEXT.md -> PLAN.md -> PROGRESS.md -> DECISIONS.md before starting.
> Spec written against commit `2ad904f` on branch `predictor-ux-dashboard`; source plan: agreed in Codex plan mode.

# CONTEXT — predictor-ux-dashboard

## Task

Simplificar la carga y correccion de picks/resultados con seleccion por clic, crear un
resumen accionable, combinar predictores de forma auditable y registrar apuestas reales
simples o combinadas con rendimiento en CLP.

## Read first

- `app.py` y `ufc/ui/tab_*.py` para navegacion y vistas actuales.
- `ufc/registro/predictores.py` y `ufc/registro/ledger.py` para persistencia existente.
- `test_app.py` para el contrato end-to-end sin red.

## Constraints

- UI en espanol, Streamlit 1.60, componentes nativos y tema oscuro actual.
- Preservar CSV existentes y separar seguimiento hipotetico de apuestas reales.
- Integrar, no revertir, cambios locales previos en `cartelera.py` y `test_app.py`.
