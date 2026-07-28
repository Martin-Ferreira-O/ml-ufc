> Handoff doc for task `ufc-fight-predictor`. Author: Claude Fable 5. Updated: 2026-07-28 19:50.
> IMPLEMENTING AGENT: read CONTEXT.md → PLAN.md → PROGRESS.md → DECISIONS.md before starting.
> Update PROGRESS.md after every meaningful change, and record any deviation from PLAN.md in DECISIONS.md.
> Spec written by Claude Fable 5 against commit `(unborn — this package is the first commit)` on branch `ufc-fight-predictor`; source plan: `~/.claude/plans/quiero-que-realicemos-un-merry-tide.md`. If HEAD has moved far past this, reconcile before trusting the spec.

# PROGRESS — ufc-fight-predictor

## Checklist

- [x] T1 — Descarga de datos de Greco1899/scrape_ufc_stats (`fetch_data.py`, `requirements.txt`, `.gitignore`)
- [ ] T2 — Features point-in-time + Elo + espejado (`features.py`)
- [ ] T3 — Train + calibración + eval temporal (`train.py`)
- [ ] T4 — Predict + app Streamlit (`predict.py`, `app.py`)
- [ ] T5 — README (`README.md`)
- [ ] Verificación completa del bloque Verification del PLAN

## Work log

- 2026-07-28 19:52 — Claude Opus 5 (/implement) — T1 listo: `fetch_data.py` baja los 6 CSVs a `data/raw/`. Verificado: exit 0, 6 archivos, `ufc_fight_results.csv` = 8797 filas (>7000). venv creado con requests/pandas/scikit-learn/streamlit.
- 2026-07-28 19:50 — Claude Fable 5 (/plan) — T1 replanificado: de scraper propio a descarga de los CSVs diarios de Greco1899/scrape_ufc_stats (decisión del usuario; ver DECISIONS). Headers de los CSVs verificados contra el repo real.
- 2026-07-28 19:39 — Claude Fable 5 (/plan) — Paquete de handoff creado; repo vacío, nada implementado aún. Todo pendiente desde T1.
