> Handoff doc for task `ufc-fight-predictor`. Author: Claude Fable 5. Updated: 2026-07-28 19:50.
> IMPLEMENTING AGENT: read CONTEXT.md → PLAN.md → PROGRESS.md → DECISIONS.md before starting.
> Update PROGRESS.md after every meaningful change, and record any deviation from PLAN.md in DECISIONS.md.
> Spec written by Claude Fable 5 against commit `(unborn — this package is the first commit)` on branch `ufc-fight-predictor`; source plan: `~/.claude/plans/quiero-que-realicemos-un-merry-tide.md`. If HEAD has moved far past this, reconcile before trusting the spec.

# PLAN — ufc-fight-predictor

## Goal

Pipeline reproducible: descargar los CSVs diarios de ufcstats.com ya scrapeados
(repo `Greco1899/scrape_ufc_stats`) → construir features point-in-time → entrenar un
clasificador calibrado → app Streamlit que da P(gana A) / P(gana B) para cualquier
matchup y compara contra una cuota de casa de apuestas ingresada a mano.

## Non-goals / scope

- **No** scraping de odds históricas ni backtest contra casas (fase futura).
- **No** predicción de método de victoria ni rondas — solo ganador.
- **No** deploy en la nube; corre local.
- **No** peleas de otras organizaciones — solo UFC (todo lo que lista ufcstats.com).

## Source plan

`~/.claude/plans/quiero-que-realicemos-un-merry-tide.md` (plan mode de Claude Code,
aprobado por el usuario el 2026-07-28).

## Task cards

Slug acoplado: un card por paso ordenado. Un solo implementador secuencial; el total
cabe en una ventana de contexto.

### T1 — Descarga de datos (Greco1899/scrape_ufc_stats)
- **Objetivo:** script que descarga los 6 CSVs ya scrapeados de ufcstats.com desde `raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/` a `data/raw/`.
- **Archivos:** `fetch_data.py`, `requirements.txt`, `.gitignore`.
- **Depende de:** —
- **Criterios de éxito:** los 6 CSVs en `data/raw/` (`ufc_event_details`, `ufc_fight_details`, `ufc_fight_results`, `ufc_fight_stats`, `ufc_fighter_details`, `ufc_fighter_tott`); `ufc_fight_results.csv` con >7000 filas.
- **Riesgos:** el repo de terceros deja de refrescarse (mitigación: su scraper es open source — GPL-3 — y se puede correr localmente; fase futura).
- **Dificultad:** 2 · **Modelo recomendado:** Opus · **Effort:** low · **Motivo:** descarga directa de 6 URLs, sin parseo de HTML.

### T2 — Features point-in-time
- **Objetivo:** matriz de features por pelea sin leakage temporal, con Elo y espejado de esquinas.
- **Archivos:** `features.py`.
- **Depende de:** T1
- **Criterios de éxito:** `features.csv` con 2× filas que las peleas válidas de `ufc_fight_results.csv`, target sin NaN, assert anti-leakage pasa.
- **Riesgos:** leakage sutil (usar stats posteriores a la pelea) = modelo mentiroso con métricas infladas.
- **Dificultad:** 7 · **Modelo recomendado:** Opus · **Effort:** medium · **Motivo:** la corrección temporal es sutil y un error invalida todo el proyecto.

### T3 — Entrenamiento + calibración
- **Objetivo:** entrenar HistGradientBoosting con split temporal, calibrar, reportar métricas vs baselines.
- **Archivos:** `train.py`.
- **Depende de:** T2
- **Criterios de éxito:** log loss test < 0.69, accuracy > 0.55, `model.pkl` generado.
- **Riesgos:** sobreajuste; calibración mal hecha deja probabilidades no confiables.
- **Dificultad:** 5 · **Modelo recomendado:** Opus · **Effort:** medium · **Motivo:** sklearn estándar, pero el protocolo temporal de eval debe ser correcto.

### T4 — Predict + app Streamlit
- **Objetivo:** `predict.py` reutilizable + UI Streamlit en español con comparación de odds manual.
- **Archivos:** `predict.py`, `app.py`.
- **Depende de:** T3
- **Criterios de éxito:** probabilidades suman 1, simétricas ante intercambio de peleadores; UI funcional.
- **Riesgos:** reconstruir el vector de features "estado actual" distinto al de entrenamiento (skew).
- **Dificultad:** 4 · **Modelo recomendado:** Opus · **Effort:** medium · **Motivo:** integración directa; el riesgo de skew requiere reusar el código de features de T2.

### T5 — README
- **Objetivo:** documentar el pipeline completo en orden.
- **Archivos:** `README.md`.
- **Depende de:** T4
- **Criterios de éxito:** un usuario nuevo puede correr todo siguiendo el README.
- **Riesgos:** ninguno.
- **Dificultad:** 2 · **Modelo recomendado:** el mismo agente de T1-T4 (no vale la pena rutear aparte) · **Effort:** low.

## Ordered steps

Cada paso es committeable por sí solo (código + PROGRESS.md actualizado).

1. **T1 — Descarga de datos.** `fetch_data.py` con requests: baja los 6 CSVs de
   `https://raw.githubusercontent.com/Greco1899/scrape_ufc_stats/main/<nombre>.csv`
   a `data/raw/`. Se refrescan a diario en ese repo — re-correr el script = datos al día.
   - Formatos verificados (headers reales, 2026-07-28):
     - `ufc_fight_results.csv`: `EVENT, BOUT ("A vs. B"), OUTCOME ("W/L" = gana el
       primero del BOUT; "L/W" el segundo; "NC/NC", "D/D"), WEIGHTCLASS, METHOD, ROUND,
       TIME, TIME FORMAT, REFEREE, DETAILS, URL`.
     - `ufc_fight_stats.csv`: **una fila por peleador por round**: `EVENT, BOUT, ROUND,
       FIGHTER, KD, SIG.STR. ("19 of 39"), SIG.STR. %, TOTAL STR., TD, TD %, SUB.ATT,
       REV., CTRL ("m:ss"), HEAD, BODY, LEG, DISTANCE, CLINCH, GROUND`.
     - `ufc_fighter_tott.csv`: `FIGHTER, HEIGHT ('5\' 11"' o "--"), WEIGHT, REACH,
       STANCE, DOB ("Jul 13, 1978" o "--"), URL`.
     - `ufc_event_details.csv`: `EVENT, URL, DATE ("July 25, 2026"), LOCATION`.
   - Valores faltantes vienen como `--`; strings con espacios colgantes — limpiar en T2.
   - `.gitignore`: `data/`, `.venv/`, `model.pkl`, `__pycache__/`.

2. **T2 — Features.** `features.py`, peleas ordenadas cronológicamente (join de
   `ufc_fight_results` + `ufc_event_details.DATE` por EVENT; stats por round de
   `ufc_fight_stats` agregadas a nivel pelea; físicos de `ufc_fighter_tott`).
   Parsear "19 of 39" → landed/attempted, CTRL "m:ss" → segundos, fechas y `--` → NaN:
   - Para cada pelea y cada peleador, agregados **solo de sus peleas UFC previas**
     (fecha estrictamente menor): win rate, racha actual, nº de peleas, sig strikes
     landed/absorbed por minuto, TD promedio, finish rate, días desde la última pelea,
     edad a la fecha de la pelea, y diffs físicos (alcance, altura) de `ufc_fighter_tott`.
   - **Elo simple** (K fijo, ~32) actualizado en orden cronológico; el rating *antes*
     de la pelea es feature.
   - **Prohibido** usar totales de carrera pre-agregados como features (incluyen peleas
     futuras respecto de cada fila) — de `ufc_fighter_tott` solo altura/alcance/stance/DOB.
   - Fila = diferencias A−B + **fila espejada** (B−A, target invertido) para eliminar
     el sesgo de esquina roja. Target: 1 si gana A (OUTCOME "W/L" → gana el primero del
     BOUT). Excluir NC y draws.
   - Debutantes: features de historial en NaN (HistGradientBoosting los maneja nativo).
   - **Assert anti-leakage**: ningún agregado usa filas con fecha ≥ fecha de la pelea
     (`assert` explícito en el código, no solo convención).
   - Salida: `data/features.csv`.

3. **T3 — Train.** `train.py`:
   - Split temporal: test = peleas con fecha en los últimos 2 años; validación = los
     2 años anteriores; train = todo lo previo. Sin shuffle entre períodos. Las dos
     filas espejadas de una misma pelea quedan siempre en el mismo lado del split.
   - `HistGradientBoostingClassifier` → calibración isotónica ajustada en validación.
   - Reporta en test: log loss, Brier score, accuracy; baselines: coin flip (0.5) y
     "gana el de mayor Elo".
   - Guarda `model.pkl` (modelo calibrado) + `data/fighter_state.csv` (el estado
     point-in-time más reciente de cada peleador, generado con el mismo código de T2,
     para predecir matchups futuros sin skew).

4. **T4 — Predict + app.** `predict.py`: `predict(nombre_a, nombre_b) -> (p_a, p_b)` —
   busca ambos en `fighter_state.csv` (match tolerante a mayúsculas/espacios), arma el
   vector de diffs, predice en ambas orientaciones y promedia (garantiza simetría).
   CLI: `python predict.py "A" "B"` imprime las dos probabilidades.
   `app.py` (Streamlit, en español): dos selectbox con búsqueda sobre los peleadores
   disponibles, botón predecir, probabilidades como barras de progreso, campo numérico
   opcional "cuota de la casa (decimal)" → muestra probabilidad implícita (1/cuota) y
   el delta vs modelo con una frase tipo "el modelo ve más/menos chances que la casa".

5. **T5 — README.** Comandos del pipeline en orden (venv → scrape → features → train →
   app), qué hace cada script, y la advertencia de que el scrape completo tarda horas.

## Verification

| Comando | Señal de pass |
|---------|---------------|
| `.venv/bin/python fetch_data.py` | exit 0; los 6 CSVs en `data/raw/`; `wc -l data/raw/ufc_fight_results.csv` > 7000 |
| `.venv/bin/python features.py` | exit 0; `data/features.csv` con ≈2× las filas (peleas válidas) de `ufc_fight_results.csv`; `target` sin NaN; assert anti-leakage no falla |
| `.venv/bin/python train.py` | imprime log loss, Brier, accuracy + baselines; log loss < 0.69 y accuracy > 0.55 en test; genera `model.pkl` y `data/fighter_state.csv` |
| `.venv/bin/python predict.py "<A>" "<B>"` (nombres reales del dataset) | imprime dos probabilidades que suman 1.0 ± 0.01; invertir el orden da las mismas probabilidades intercambiadas |
| **End-to-end**: `.venv/bin/streamlit run app.py` | manual: elegir dos peleadores → se ven las barras de probabilidad; ingresar cuota 1.50 → prob. implícita ≈66.7% y el delta vs modelo |

El chequeo de Streamlit es manual — no automatizarlo.

## Riesgos globales

- Dependencia de un repo de terceros (`Greco1899/scrape_ufc_stats`) para el refresh
  diario; si se abandona, su scraper GPL-3 se puede correr localmente (fase futura).
- Expectativa realista: 60-65% accuracy es el techo de los modelos públicos de UFC; el
  valor del proyecto está en la **calibración** de las probabilidades, no en acertar todo.
