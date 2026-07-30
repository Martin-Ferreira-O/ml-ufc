> Handoff doc for task `ufc-fight-predictor`. Author: Claude Fable 5. Updated: 2026-07-28 19:50.
> IMPLEMENTING AGENT: read CONTEXT.md → PLAN.md → PROGRESS.md → DECISIONS.md before starting.
> Update PROGRESS.md after every meaningful change, and record any deviation from PLAN.md in DECISIONS.md.
> Spec written by Claude Fable 5 against commit `(unborn — this package is the first commit)` on branch `ufc-fight-predictor`; source plan: `~/.claude/plans/quiero-que-realicemos-un-merry-tide.md`. If HEAD has moved far past this, reconcile before trusting the spec.

# DECISIONS — ufc-fight-predictor

## Decisiones tomadas

- **Fuente de datos: CSVs diarios de `Greco1899/scrape_ufc_stats`** (confirmado por el
  usuario 2026-07-28, **supersede** la decisión previa de scraper propio). El usuario
  trajo un scraper de Kaggle (asaniczka); al investigarlo apareció este repo de GitHub
  que publica los CSVs de ufcstats.com ya scrapeados con refresh automático diario
  (verificado: último refresh el mismo 2026-07-28, datos hasta el evento del 25/07).
  Mismos datos frescos, cero scraping propio. Fallback si el repo se abandona: su
  scraper es open source (GPL-3) y se puede correr localmente.
- **Interfaz: web app Streamlit** (confirmado por el usuario). `predict.py` queda
  igualmente reutilizable por CLI.
- **Odds solo como benchmark manual, nunca como feature.** El objetivo del usuario es
  *discrepar* de las casas; usar odds como feature haría que el modelo las imite.
  El usuario ingresa la cuota decimal en la UI y ve la probabilidad implícita vs la
  del modelo. Backtest histórico contra odds = fase futura, fuera de alcance.
- **Modelo: `HistGradientBoostingClassifier` de sklearn + calibración isotónica.**
  Sin XGBoost/LightGBM: no agregan nada hasta que sklearn se quede corto, y HistGB
  maneja NaN nativo (debutantes). // ponytail: upgrade a XGBoost solo si las métricas lo piden.
  **Superseded 2026-07-28: la calibración se eliminó** — ver "Ronda de mejora" abajo.
- **Anti-leakage como requisito duro:** features point-in-time con `assert` explícito,
  diffs A−B + espejado de esquinas, split temporal. Un leak invalida el proyecto entero
  (métricas infladas, predicciones inútiles) — por eso T2 tiene dificultad 7.
- **Sin scraper propio ni modo `--sample`:** la descarga de CSVs tarda segundos, así
  que toda la verificación corre contra el dataset completo.

## Notas de implementación (T2)

- **Peleas del mismo día se calculan antes de aplicar sus updates.** Los torneos de
  UFC 1-8 tienen peleadores con dos peleas la misma fecha; con "fecha estrictamente
  menor" pelea-a-pelea el assert anti-leakage habría fallado. Se agrupa por fecha:
  snapshot de todas las peleas del día → recién ahí los updates. Cumple el requisito
  del PLAN (ningún agregado usa filas con fecha ≥ la de la pelea) y es más estricto.
- **2 peleas descartadas de 8641**: `UFC - Road to UFC 4.6` no tiene fila en
  `ufc_event_details.csv`, o sea no tiene fecha → imposible ubicarla cronológicamente.
  Por eso `features.csv` tiene 2×8639 y no 2×8641.
- **Duración de pelea** = `(ROUND-1)*5 + TIME`. Los formatos viejos de un round de
  12'/15' quedan subestimados; solo afecta las tasas por minuto de esas peleas.
  Marcado con `# ponytail:` en el código.

## Notas de implementación (T3)

- `CalibratedClassifierCV(..., cv="prefit")` ya no existe en la versión instalada de
  sklearn: se usa `FrozenEstimator(modelo)`, que es el reemplazo directo. Mismo
  protocolo que pide el PLAN (calibrar sobre validación, sin re-entrenar).

## Notas de implementación (T5)

- **Desvío menor del PLAN**: el paso 5 pide que el README incluya "la advertencia de
  que el scrape completo tarda horas". Eso quedó del diseño anterior (scraper propio);
  con los CSVs de Greco1899 la descarga tarda segundos, así que la advertencia sería
  falsa. El README documenta los tiempos reales (pipeline completo: un par de minutos).

## Ronda de mejora (2026-07-28)

Protocolo: decisiones solo con log loss de validación (2022-07 → 2024-07); test se
reporta una vez por fase. Regla de corte: un cambio queda si mejora val ≥ 0.003;
zona gris → desempate con 4 folds rolling-origin de 1 año que terminan en corte_test.

- **Medición corregida.** Antes se evaluaba el modelo crudo sobre filas espejadas;
  el predictor real (`predict.py`) promedia ambas orientaciones. Ahora `train.py`
  mide eso sobre peleas únicas (slicing `[0::2]`/`[1::2]`, los pares son adyacentes).
  El 0.6802 histórico era artefacto de la isotónica; el baseline real era 0.6449.
- **Calibración eliminada (resuelve la open question).** La isotónica ya estaba
  medida como dañina (tabla previa: 0.6802 vs 0.6460 crudo). Chequeo final: sigmoid
  ajustada sobre las filas sin espejar de val y evaluada *in-sample* — su mejor caso
  posible — empata con el crudo (0.6649 vs 0.6650). Se picklea el HistGB crudo;
  de paso desaparece la dependencia de `FrozenEstimator` (sklearn ≥ 1.6).
- **`model.pkl` ahora se re-entrena con todo el historial** antes de picklear.
  Antes se desplegaba el modelo de la ventana train, cuyos datos terminaban 4 años
  atrás. Las métricas del README siguen viniendo del modelo de split.
- **6 features nuevas (todas de CSVs ya descargados):** `kd_per15`,
  `kd_against_per15` (poder/mentón), `finished_against_rate` (durabilidad),
  `td_def`, `str_def` (defensas: intentos del rival en contra), `avg_opp_elo`
  (calidad de rivales, con Elo PRE-pelea para no filtrar el resultado).
  Val 0.6650 → 0.6594 (−0.0056). Ablación: sacar cualquier grupo empeora
  (finished_against +0.0047, avg_opp_elo +0.0043, defensas +0.0026, kd +0.0018).
- **Features probadas y descartadas** (medidas, no supuestas): `win_rate_last5` y
  `southpaw` (juntas o separadas empeoran val), `five_rounds` (+0.0026 peor),
  `weight_lb` (exactamente neutro). Cero líneas en el repo.
- **Elo queda con K=32.** Variantes K=40/28 por finish y K provisional para
  debutantes: el desempate por folds dio −0.0010 para K-finish, bajo el corte.
- **Hiperparámetros quedan como estaban** (lr=0.05, leaves=15, l2=1.0). Grid 27
  combos × 4 folds: el mejor (lr=0.08, leaves=15, l2=0.1, media 0.6593) le gana al
  actual (0.6600) por −0.0007, bajo el corte. `monotonic_cst` en elo empeora
  (0.6611). El grid completo quedó plano en 0.6593–0.6650: no hay jugo ahí.
- **Lección de medición:** el mismo dataset por round-trip CSV vs en memoria mueve
  el val log loss ~0.003 (HistGB es caótico ante el último bit de los floats).
  Por eso toda comparación se hace por el mismo camino (CSV, el del pipeline real)
  y los deltas chicos se desempatan con folds.

## Open questions for the spec author

- **Sacar del PLAN la advertencia de "el scrape tarda horas" (paso 5 / T5).** Quedó del
  diseño con scraper propio; con los CSVs de Greco1899 la descarga tarda segundos, así
  que ponerla en el README sería mentir. El review en contexto limpio la marcó como
  GAP 1 y coincidió en que el README tiene razón y la línea del PLAN es la obsoleta.
  Como `PLAN.md` es read-mostly, la corrección queda a cargo del autor del spec.
