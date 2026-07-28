> Handoff doc for task `ufc-fight-predictor`. Author: Claude Fable 5. Updated: 2026-07-28 19:50.
> IMPLEMENTING AGENT: read CONTEXT.md → PLAN.md → PROGRESS.md → DECISIONS.md before starting.
> Update PROGRESS.md after every meaningful change, and record any deviation from PLAN.md in DECISIONS.md.
> Spec written by Claude Fable 5 against commit `(unborn — this package is the first commit)` on branch `ufc-fight-predictor`; source plan: `~/.claude/plans/quiero-que-realicemos-un-merry-tide.md`. If HEAD has moved far past this, reconcile before trusting the spec.

# CONTEXT — ufc-fight-predictor

## Task

Construir desde cero un predictor de peleas de UFC: descarga de los CSVs diarios de
ufcstats.com (repo `Greco1899/scrape_ufc_stats`, refresh automático diario) →
features point-in-time sin leakage → clasificador con probabilidades calibradas →
web app Streamlit donde se eligen dos peleadores y se ve P(gana A) / P(gana B),
con comparación opcional contra la cuota de una casa de apuestas ingresada a mano.
Motivación del usuario: detectar peleas donde el modelo discrepa de las casas
(caso Chimaev vs Strickland).

## Project area

Repo **vacío** — este paquete es el primer contenido. Todo se crea nuevo:
`fetch_data.py`, `features.py`, `train.py`, `predict.py`, `app.py`,
`requirements.txt`, `README.md`, `data/` (generado, gitignoreado).

## Read first

No hay código previo que leer. Antes de codear:
- `docs/handoff/ufc-fight-predictor/PLAN.md` — el spec completo.
- `docs/handoff/ufc-fight-predictor/DECISIONS.md` — decisiones ya tomadas (no re-litigar).
- Instrucción permanente: si el repo ya tiene código cuando arrancás (otra sesión
  avanzó), abrí esos archivos y confirmá que todavía coinciden con este spec antes
  de confiar en cualquier resumen — re-derivar del código gana a recordar del handoff.

## Setup / run / test

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # requests, pandas, scikit-learn, streamlit
.venv/bin/python fetch_data.py              # descarga los 6 CSVs (segundos)
.venv/bin/python features.py
.venv/bin/python train.py
.venv/bin/python predict.py "Nombre A" "Nombre B"
.venv/bin/streamlit run app.py
```

## Conventions that matter here

- Repo nuevo: no hay CLAUDE.md; usar `.venv/bin/python` siempre (nunca python global).
- Texto visible al usuario (UI de Streamlit, mensajes de CLI) **en español**.
- Código simple y directo (modo ponytail): stdlib/sklearn antes que deps nuevas, sin
  abstracciones especulativas.
