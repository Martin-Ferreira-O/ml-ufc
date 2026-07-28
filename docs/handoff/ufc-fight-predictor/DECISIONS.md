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

## Open questions for the spec author

(ninguna por ahora)
