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

## Open questions for the spec author

- **Sacar del PLAN la advertencia de "el scrape tarda horas" (paso 5 / T5).** Quedó del
  diseño con scraper propio; con los CSVs de Greco1899 la descarga tarda segundos, así
  que ponerla en el README sería mentir. El review en contexto limpio la marcó como
  GAP 1 y coincidió en que el README tiene razón y la línea del PLAN es la obsoleta.
  Como `PLAN.md` es read-mostly, la corrección queda a cargo del autor del spec.

- **La calibración isotónica empeora las probabilidades, ¿la cambiamos por sigmoid?**
  Implementado tal cual lo pide el PLAN (isotónica) y el gate pasa, pero medido en test:

  | calibración | log loss | Brier | accuracy |
  |---|---|---|---|
  | ninguna (crudo) | 0.6460 | 0.2270 | 0.6443 |
  | **isotónica (spec)** | **0.6802** | 0.2274 | 0.6458 |
  | sigmoid (Platt) | 0.6471 | 0.2275 | 0.6414 |

  La isotónica sobreajusta las ~1000 peleas únicas de validación y devuelve
  probabilidades escalonadas/extremas. Como el valor del proyecto está justamente en
  la calidad de las probabilidades (comparar contra la casa), esto importa. Cambiar
  `method="isotonic"` → `"sigmoid"` en `train.py` es una línea. No lo hice por mi
  cuenta porque la isotónica está listada como decisión tomada arriba.
