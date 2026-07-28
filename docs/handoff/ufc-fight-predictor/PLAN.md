> Handoff doc for task `ufc-fight-predictor`. Author: Claude Fable 5. Updated: 2026-07-28 19:39.
> IMPLEMENTING AGENT: read CONTEXT.md → PLAN.md → PROGRESS.md → DECISIONS.md before starting.
> Update PROGRESS.md after every meaningful change, and record any deviation from PLAN.md in DECISIONS.md.
> Spec written by Claude Fable 5 against commit `(unborn — this package is the first commit)` on branch `ufc-fight-predictor`; source plan: `~/.claude/plans/quiero-que-realicemos-un-merry-tide.md`. If HEAD has moved far past this, reconcile before trusting the spec.

# PLAN — ufc-fight-predictor

## Goal

Pipeline reproducible: scrapear UFCStats.com → construir features point-in-time →
entrenar un clasificador calibrado → app Streamlit que da P(gana A) / P(gana B) para
cualquier matchup y compara contra una cuota de casa de apuestas ingresada a mano.

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

### T1 — Scraper de UFCStats
- **Objetivo:** scraper completo de ufcstats.com con caché de HTML y modo `--sample`.
- **Archivos:** `scraper/scrape.py`, `requirements.txt`, `.gitignore`.
- **Depende de:** —
- **Criterios de éxito:** `--sample 5` produce los 4 CSVs con ≥40 peleas; re-run no re-descarga (caché).
- **Riesgos:** HTML de UFCStats cambia; parseo de formatos raros (NC, draws, "--" en stats).
- **Dificultad:** 5 · **Modelo recomendado:** Opus · **Effort:** medium · **Motivo:** scraping multi-nivel estándar pero con muchos edge cases de parseo.

### T2 — Features point-in-time
- **Objetivo:** matriz de features por pelea sin leakage temporal, con Elo y espejado de esquinas.
- **Archivos:** `features.py`.
- **Depende de:** T1
- **Criterios de éxito:** `features.csv` con 2× filas que fights.csv, target sin NaN, assert anti-leakage pasa.
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

1. **T1 — Scraper.** `scraper/scrape.py` con requests + BeautifulSoup (lxml):
   - Recorre `http://ufcstats.com/statistics/events/completed?page=all` → páginas de
     evento (fecha, ubicación, lista de peleas) → detalle de cada pelea
     (ganador, método, ronda, tiempo, y por peleador: knockdowns, sig. strikes
     landed/attempted, total strikes, takedowns landed/attempted, sub attempts,
     control time) → páginas de peleador (altura, alcance, stance, fecha de nacimiento).
   - **Caché**: cada HTML descargado se guarda en `data/raw/html/<hash-o-id>.html`;
     si existe, se lee de disco. Re-run = solo descarga eventos nuevos.
   - ~1 req/s (`time.sleep(1)`), User-Agent identificable.
   - Salida: `data/raw/events.csv`, `fights.csv`, `fight_stats.csv`, `fighters.csv`.
   - Flag `--sample N`: procesa solo los N eventos más recientes (smoke test).
   - Manejar: peleas sin ganador (NC/draw → excluir o marcar), stats "--"/"---" → NaN,
     peleadores sin fecha de nacimiento.
   - `.gitignore`: `data/`, `.venv/`, `model.pkl`, `__pycache__/`.
   - Scrape completo ≈ 700+ eventos / ~8k peleas — horas. Se corre una vez; todo el
     desarrollo y la verificación usan `--sample`.

2. **T2 — Features.** `features.py`, peleas ordenadas cronológicamente:
   - Para cada pelea y cada peleador, agregados **solo de sus peleas UFC previas**
     (fecha estrictamente menor): win rate, racha actual, nº de peleas, sig strikes
     landed/absorbed por minuto, TD promedio, finish rate, días desde la última pelea,
     edad a la fecha de la pelea, y diffs físicos (alcance, altura) del CSV de peleadores.
   - **Elo simple** (K fijo, ~32) actualizado en orden cronológico; el rating *antes*
     de la pelea es feature.
   - **Prohibido** usar los totales de carrera de la página del peleador como features
     (incluyen peleas futuras respecto de cada fila) — solo altura/alcance/stance/DOB.
   - Fila = diferencias A−B + **fila espejada** (B−A, target invertido) para eliminar
     el sesgo de esquina roja. Target: 1 si gana A.
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

Todo con `--sample` para no esperar horas (el scrape completo es idéntico sin el flag):

| Comando | Señal de pass |
|---------|---------------|
| `.venv/bin/python scraper/scrape.py --sample 5` | exit 0; existen los 4 CSVs en `data/raw/`; `fights.csv` con ≥40 filas |
| `.venv/bin/python scraper/scrape.py --sample 5` (2ª vez) | termina en segundos (caché — no re-descarga) |
| `.venv/bin/python features.py` | exit 0; `data/features.csv` con ≈2× las filas de fights.csv; `target` sin NaN; assert anti-leakage no falla |
| `.venv/bin/python train.py` | imprime log loss, Brier, accuracy + baselines; **con dataset completo**: log loss < 0.69 y accuracy > 0.55; genera `model.pkl` y `data/fighter_state.csv` |
| `.venv/bin/python predict.py "<A>" "<B>"` (nombres reales del dataset) | imprime dos probabilidades que suman 1.0 ± 0.01; invertir el orden da las mismas probabilidades intercambiadas |
| **End-to-end**: `.venv/bin/streamlit run app.py` | manual: elegir dos peleadores → se ven las barras de probabilidad; ingresar cuota 1.50 → prob. implícita ≈66.7% y el delta vs modelo |

Nota: los umbrales de métricas (log loss < 0.69, accuracy > 0.55) solo aplican con el
dataset completo; con `--sample 5` alcanzan con que el pipeline corra de punta a punta.
El chequeo de Streamlit es manual — no automatizarlo.

## Riesgos globales

- UFCStats cambia su HTML → scraper frágil; la caché local permite re-parsear sin re-descargar.
- Expectativa realista: 60-65% accuracy es el techo de los modelos públicos de UFC; el
  valor del proyecto está en la **calibración** de las probabilidades, no en acertar todo.
