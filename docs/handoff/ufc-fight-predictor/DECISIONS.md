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

## Ronda de protocolo y modelo (2026-07-30)

**Objetivo del proyecto revisado por el usuario:** la app no busca discrepar del mercado,
sino ser una fuente de información para apostar con más seguridad. Eso **supersede** la
decisión de "odds solo como benchmark manual": ahora entran como feature (fase 2), en un
segundo modelo que convive con el que no las usa.

- **Diagnóstico: no hay under- ni overfitting, hay techo de información.** Entre 50 y 800
  árboles el train loss cae 0.6371 → 0.4684 y el de validación se mueve 0.6613 → 0.6701.
  El grid plano de la ronda anterior tenía esta causa. Corolario: **no tocar más los
  hiperparámetros del HistGB**, el pozo está seco.
- **El umbral de ≥0.003 era ruido.** El SE del delta pareado sobre las 999 peleas de val
  es **0.0028**, o sea 1.1σ. Todas las decisiones de la ronda anterior se tomaron dentro
  del ruido. Reemplazado por `rolling_origin` (20 folds de 1 año, **7195 peleas**) +
  `comparar` (bootstrap pareado): **un cambio queda sólo si el IC95% no toca cero.**
- **El protocolo se pagó solo en la primera corrida.** La logística sola le ganaba al
  HistGB en validación (0.6566 vs 0.6597) y en test (0.6384 vs 0.6409), pero en
  rolling-origin **pierde** (0.6646 vs 0.6623, IC95% [−0.0020, +0.0068], no concluyente).
  Era ruido de ventana. Con 10 folds el blend también salía no concluyente; hicieron falta
  20 folds para resolverlo.
- **Se despliega el promedio HistGB + logística.** −0.0031 vs HistGB solo,
  IC95% [−0.0052, −0.0009] sobre 7195 peleas: **queda**. Tres motivos además del log loss:
  (a) la logística sin intercepto sobre features antisimétricas cumple `p(A,B)+p(B,A)=1`
  exacto — el HistGB crudo lo viola hasta en **0.205**, y el promedio de orientaciones lo
  venía tapando; (b) da la contribución exacta de cada feature al logit, que es el insumo
  de la explicación en la app, sin SHAP; (c) estabiliza: entre 100 y 300 iteraciones el
  HistGB solo se degrada 0.6632 → 0.6716 y el blend apenas 0.6617 → 0.6630.
- **`early_stopping=False` explícito.** Con `'auto'` sklearn lo activaba (train = 13222
  filas > 10000) y separaba un 10% **aleatorio**: como cada pelea está dos veces espejada,
  el gemelo de cada fila de la validación interna quedaba en train y el criterio de parada
  se evaluaba sobre filas ya vistas. Caía en 94 iteraciones ≈ el óptimo por casualidad.
  `max_iter=100` fijo es el mejor medido con rolling-origin.
- **Sin indicador de faltante en la logística.** Sobre un dataset espejado su peso óptimo
  es 0 exacto: la loss es invariante ante cambiarle el signo, así que con L2 el mínimo
  único tiene `w=0`. Sólo agregaría 22 columnas muertas. Se imputa NaN → 0, que es el
  relleno antisimétrico natural de un diff y sobrevive al espejado.
- **Calibración: re-verificada como dañina, ahora bien medida.** Factor de nitidez
  ajustado en los folds 0-9 y evaluado out-of-sample en los 10-19 (4827 peleas):
  a = 0.898, delta **+0.0008, IC95% [+0.0004, +0.0013] → se descarta.** La decisión
  original era correcta; ahora tiene respaldo real y no una ventana de 999 peleas.
- **La tabla de confiabilidad se calcula sobre ambas orientaciones.** Puntuarla sólo en el
  orden del CSV (donde el ganador va primero el 56% de las veces en la ventana de test) le
  suma ~0.09 a cada bucket y simula un sesgo hacia arriba que no existe. El log loss **no**
  cambia — es invariante cuando la predicción es simétrica — pero la tabla sí, y llevó a
  diagnosticar sub-confianza donde no la había.
- **`model.pkl` ahora es un dict** `{"hgb", "lineal", "cols"}`. `cols` viaja con el pickle
  para que `predict.py` no pueda quedar fuera de sincro con el entrenamiento.

## Ronda de odds y confianza (2026-07-30)

- **Fuente de odds: `shortlikeafox/ultimate_ufc_dataset` en GitHub**, no Kaggle. Kaggle
  pide token de API y el usuario no lo tiene; este repo sirve el mismo dataset por
  `raw.githubusercontent`, con el mismo patrón que `fetch_data.py` ya usaba. Cubre
  2010-03 → 2026-03, **matchea el 95.0%** de las peleas de ese período (78.9% del total)
  con clave (fecha + par de nombres ordenado). Los que fallan son alias — "Mirko
  Filipovic" vs "Cro Cop" — y no se construyó fuzzy matching por el 5%.
- **El mercado le gana al modelo por goleada.** Sobre las 5679 peleas con cuota del
  rolling-origin: modelo 0.6608, **mercado 0.6115** (−0.0493, IC95% [−0.0571, −0.0418]).
- **Meterle la cuota al modelo no mejora sobre la cuota sola:** 0.6119 vs 0.6115,
  **+0.0005 IC95% [−0.0031, +0.0042], no concluyente.** Las 22 features no aportan nada
  por encima de la línea de cierre. Se despliegan igual los dos modelos porque el usuario
  quiere ver ambos, pero el de con odds no es "el bueno": es el que muestra cuánto mueve
  el historial a la cuota, que resultó ser nada.
- **La discrepancia con el mercado es un anti-indicador, no valor.** Log loss del modelo
  por tramo de |modelo−mercado|: <0.05 → 0.6592 (mercado 0.6606, empatan); 0.15-0.25 →
  0.6620 vs 0.5790; >0.25 → **0.7416 vs 0.5402, peor que una moneda**. En las 1745 peleas
  donde eligen ganadores distintos, la casa acierta **58.8%** y el modelo **41.8%**.
- **La confianza se basa en la coincidencia con el mercado, NO en la disponibilidad de
  datos.** La hipótesis original (poca data → menos confiable) se midió y es **falsa**:
  log loss por experiencia del menos experimentado de los dos — debutantes (0 peleas,
  17-22 features en NaN) **0.6639**, veteranos (11+) **0.6512**, y todo lo del medio entre
  0.6534 y 0.6668. Accuracy plana en 0.60-0.61. Marcar "confianza baja" por ser debutante
  habría sido mentirle al usuario.
- **Corolario: la fase de récord pre-UFC (fase 6 del plan) pierde su premisa.** Se
  justificaba por el "agujero" del 25.7% de peleas con debutante, pero ahí el modelo rinde
  igual que en el resto. El techo máximo que podría recuperar es ~0.013 de log loss, y el
  mercado está 0.049 más arriba. Queda pendiente de decisión del usuario.
- **`test_app.py`**: primer check del repo. Sin framework, `python test_app.py`. Cubre
  simetría de `predict` (con y sin cuotas), que el nivel de confianza salga del tramo
  correcto, y que la app renderice ambos caminos vía `AppTest`.
- **`n_fights_min` y `n_nan` en `features.csv`**: metadatos simétricos (valen igual en las
  dos filas espejadas), no features. Sirvieron para medir lo de arriba y quedan para poder
  re-medirlo. No entran al modelo.

## Hechos externos verificados (2026-07-30)

- **`ufcstats.com` está detrás de un challenge JS de proof-of-work.** No se puede scrapear
  con `requests`. El mirror de `Greco1899/scrape_ufc_stats` no es una comodidad: es la
  única vía práctica. Su repo tiene exactamente los 6 CSVs que ya se bajan.
- **Sherdog responde** con UA normal (107 KB, bloque de récord W/L/NC + historial): es la
  fuente viable para el récord pre-UFC. **Tapology está bloqueado por Cloudflare** (403).

## Open questions for the spec author

- **Sacar del PLAN la advertencia de "el scrape tarda horas" (paso 5 / T5).** Quedó del
  diseño con scraper propio; con los CSVs de Greco1899 la descarga tarda segundos, así
  que ponerla en el README sería mentir. El review en contexto limpio la marcó como
  GAP 1 y coincidió en que el README tiene razón y la línea del PLAN es la obsoleta.
  Como `PLAN.md` es read-mostly, la corrección queda a cargo del autor del spec.
