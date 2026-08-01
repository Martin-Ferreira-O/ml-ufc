> Handoff doc for task `ufc-fight-predictor`. Author: Claude Fable 5. Updated: 2026-07-28 19:50.
> IMPLEMENTING AGENT: read CONTEXT.md → PLAN.md → PROGRESS.md → DECISIONS.md before starting.
> Update PROGRESS.md after every meaningful change, and record any deviation from PLAN.md in DECISIONS.md.
> Spec written by Claude Fable 5 against commit `(unborn — this package is the first commit)` on branch `ufc-fight-predictor`; source plan: `~/.claude/plans/quiero-que-realicemos-un-merry-tide.md`. If HEAD has moved far past this, reconcile before trusting the spec.

# PROGRESS — ufc-fight-predictor

## Checklist

- [x] T1 — Descarga de datos de Greco1899/scrape_ufc_stats (`fetch_data.py`, `requirements.txt`, `.gitignore`)
- [x] T2 — Features point-in-time + Elo + espejado (`features.py`)
- [x] T3 — Train + calibración + eval temporal (`train.py`)
- [x] T4 — Predict + app Streamlit (`predict.py`, `app.py`)
- [x] T5 — README (`README.md`)
- [x] Verificación completa del bloque Verification del PLAN

## Work log

- 2026-08-01 — Claude Opus 5 — Récord pre-UFC vía Sherdog (`sherdog.py`, fuente externa #2). 2706/2724 peleadores (99.3%), validado 2760/2760 fichas contra el total W-L que declara el sitio. Pasó `previo_metodo` (prev_ko_w/sub_w/ko_l/sub_l): −0.0022 IC95% [−0.0039, −0.0006]; descartado el récord crudo. Rolling-origin 0.6572 → 0.6550, test → 0.6342, brecha vs mercado → 0.0453. `train.calidad_por_tramo` nuevo: los log loss por tramo que sostienen `predict.CONFIANZA` ya no se derivan a mano. Ver DECISIONS, ronda de récord pre-UFC.
- 2026-07-28 20:04 — Claude Opus 5 (/implement) — GAP 2 del review corregido: `_ctrl_seconds` devuelve NaN en vez de 0.0 para `--`, como pide la regla de parseo del PLAN ("`--` → NaN"). Verificado: `features.csv` queda byte-idéntico (sumar 0.0 y saltear la suma dan lo mismo en el acumulador), así que métricas y `model.pkl` no cambian; es conformidad con el spec, no un cambio de comportamiento.
- 2026-07-28 20:02 — Claude Opus 5 (/implement) — review (fresh) against PLAN: 2 gaps (ningún defecto real). Reviewer en contexto limpio (solo diff + PLAN.md), ruteado a Opus por dificultad 7 de T2. Corrió las 4 filas automatizadas del bloque Verification (todas pasan) y validó el anti-leakage de tres formas independientes: por construcción, inyectando la regresión que el assert debe atrapar (falla como corresponde → el assert no es vacuo), y recomputando n_fights/win_rate desde los CSVs crudos en 400 filas (0 mismatches, 7 de ellas diferirían con `<=`). Cero violaciones de espejado, cero peleas partidas entre train/val/test, `model.feature_names_in_ == features.FEATURES` (sin skew). GAP 1 = la advertencia obsoleta del PLAN (ya registrada en DECISIONS, el README tiene razón). GAP 2 = `--` → 0.0 en CTRL, corregido abajo.
- 2026-07-28 20:00 — Claude Opus 5 (/implement) — Bloque Verification corrido entero desde `data/` vacío: fetch (6 CSVs, 8797 filas) → features (17278 = 2×8639, target sin NaN, assert OK) → train (log loss 0.6802 < 0.69, accuracy 0.6458 > 0.55) → predict (71.2/28.8 en ambos órdenes, suman 1.0). Streamlit levanta con HTTP 200; el chequeo interactivo queda manual como pide el PLAN.
- 2026-07-28 20:26 — Claude Opus 5 (/implement) — T5 listo: `README.md` con instalación, pipeline en orden, tabla de qué hace cada script, métricas actuales y la garantía anti-leakage. Desvío menor registrado en DECISIONS (la advertencia de "el scrape tarda horas" que pide el PLAN quedó obsoleta al cambiar a los CSVs de Greco1899).
- 2026-07-28 20:22 — Claude Opus 5 (/implement) — T4 listo: `predict.py` (match tolerante, promedio de las dos orientaciones) + `app.py` en español. Verificado: Chimaev/Strickland → 64.5% / 35.5%, suman 1.0, invertir el orden e ingresar los nombres en minúscula/con espacios de más da las mismas probabilidades intercambiadas; nombre inexistente → mensaje en español y exit 1. Streamlit arranca (HTTP 200, sin errores); el chequeo interactivo de la UI queda manual como pide el PLAN.
- 2026-07-28 20:14 — Claude Opus 5 (/implement) — T3 listo: `train.py` con split temporal (train 13222 / val 1998 / test 2058 desde 2024-07-25). Test: log loss 0.6802, Brier 0.2274, accuracy 0.6458; baselines coin 0.6931/0.5000 y Elo 0.5588. Pasa el gate (<0.69, >0.55). `model.pkl` y `data/fighter_state.csv` (2716 peleadores) generados. **Ojo**: la calibración isotónica empeora el log loss vs el modelo crudo (0.6802 vs 0.6460) — ver Open questions en DECISIONS.
- 2026-07-28 20:05 — Claude Opus 5 (/implement) — T2 listo: `features.py` con 16 features point-in-time, Elo (K=32) y espejado. Verificado: 17278 filas = 2×8639 peleas, `target` sin NaN, assert anti-leakage pasa, diffs espejados suman 0 exacto. 2 peleas (`Road to UFC 4.6`) descartadas por no tener fecha en `ufc_event_details.csv`.
- 2026-07-28 19:52 — Claude Opus 5 (/implement) — T1 listo: `fetch_data.py` baja los 6 CSVs a `data/raw/`. Verificado: exit 0, 6 archivos, `ufc_fight_results.csv` = 8797 filas (>7000). venv creado con requests/pandas/scikit-learn/streamlit.
- 2026-07-28 19:50 — Claude Fable 5 (/plan) — T1 replanificado: de scraper propio a descarga de los CSVs diarios de Greco1899/scrape_ufc_stats (decisión del usuario; ver DECISIONS). Headers de los CSVs verificados contra el repo real.
- 2026-07-28 19:39 — Claude Fable 5 (/plan) — Paquete de handoff creado; repo vacío, nada implementado aún. Todo pendiente desde T1.
