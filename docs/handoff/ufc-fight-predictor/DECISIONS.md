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

## Ronda de infraestructura de mercado y modelado (2026-07-31)

**Reencuadre aprobado por el usuario:** las conclusiones "el mercado le gana" están
medidas contra líneas de cierre; nadie apuesta contra el cierre. La prioridad pasó de
mejorar el modelo (techo de información confirmado) a la infraestructura que hace medible
si hay valor contra la línea de **apertura**.

- **Archivado de cuotas y carteleras.** `betano.cuotas()` deja un tick por pelea en
  `data/betano_hist.csv` (apertura→cierre); `cartelera.proximas()` registra el primer
  avistaje de cada pelea en `data/cartelera_hist.csv` (detección futura de reemplazos
  tardíos). Append-only, sin dependencias nuevas.
- **Ledger + forward test (`ledger.py`, pestaña Historial).** La primera predicción con
  cuota queda congelada; al ocurrir la pelea se cruza con el resultado y el último tick
  (cierre) → ROI real de candidatas y **CLV**. Es la vara que decide si la etiqueta
  "candidata" sobrevive: el backtest del tramo alta es el mejor de 4 tramos elegido a
  posteriori, con IC cruzando cero — evidencia débil que solo el CLV real puede firmar.
- **Devig power reemplaza al proporcional** en `features._mercado` y `predict._mercado`.
  Medido como predictor sobre 6916 peleas: −0.0009 (IC95% [−0.0015, −0.0003]), y
  −0.0048 en favoritos >75%. El proporcional inflaba al underdog (sesgo
  favorito-longshot). `train._cuotas_con_vig` no cambia: el ROI se paga con cuota real.
  Números re-derivados: mercado 0.6107, gap 0.0500, tramo alta +5.82% (n=887).
- **Consenso multi-casa (`oddsapi.py`).** The Odds API con `ODDS_API_KEY` opcional:
  mediana desvigueada = probabilidad de mercado; mejor cuota > consenso → "línea
  desalineada" en la app. Es la única señal de EV que no requiere ganarle al mercado
  con un modelo. Sin key devuelve `{}`.
- **Cuota mínima (1/p) en la app**, para line-shopping manual contra cualquier casa.
- **Homónimos avisados.** 8 nombres de ufcstats son 2 peleadores distintos cada uno
  (dos "Bruno Silva" con 23 peleas fusionadas, dos "Jean Silva" activos, etc.); las
  peleas solo traen el nombre → historiales mezclados sin arreglo posible. Flag
  `homonimo` en `fighter_state.csv` + aviso en predict/app/CLI. No se excluyen del
  train: ~46 filas de 17278, y excluirlas no des-mezcla nada.
- **Bug de la app:** si ESPN no respondía, `_carteleras()` propagaba la excepción y el
  warning prometido nunca se mostraba. try/except → `[]`.

### Ronda de modelado (protocolo de 20 folds): todo no concluyente

Candidatas medidas con `rolling_origin` + `comparar` contra el blend actual (7195
peleas). **Ninguna queda; cero líneas en el modelo de ganador:**

- interacciones ofensa×defensa antisimétricas (`a.of·(1−b.def) − b.of·(1−a.def)` para
  striking y TD): +0.0000 [−0.0010, +0.0010]
- contexto simétrico división/género/5R: −0.0001 [−0.0007, +0.0005]
- decay temporal de stats, λ=0.85/año: −0.0000 [−0.0014, +0.0014]; λ=0.93: ídem
- Elo K-finish 40/28 (re-test del descarte de la ronda vieja, ahora con protocolo
  nuevo): −0.0010 [−0.0021, +0.0001] — el más cercano, pero toca cero

Cuarta confirmación independiente del techo de información.

### Modelo de método (ko/sub/dec): el contexto es todo

- **Queda `metodo` = HistGB sobre `CONTEXTO` (wc_lbs, mujer, cinco_r) solamente.**
  Rolling-origin: contexto vs base rate **−0.0142 [−0.0216, −0.0069], queda**; diffs de
  peleadores encima: +0.0034, no concluyente; sumas simétricas de finish/kd/sub (la
  representación "correcta" para un target simétrico): **+0.0252, se descarta**.
  Conclusión medida: el método es una propiedad de la división, no del matchup.
  Por eso `predict.metodo()` no pide peleadores y sirve también para debuts.
- Serving: `cartelera.contexto()` mapea el texto de peso de ESPN a (lbs, mujer) y
  aproxima `cinco_r` con "es el main event". Sin contexto → NaN → base rate global.
- `features.csv` gana las columnas `metodo` (target) y `CONTEXTO`; `FEATURES` del
  ganador queda en las 22 de siempre.

## Ronda de fuentes externas (2026-07-31): el techo era de fuente, no de features

Punto de partida: el README declaraba techo de información y cuatro rondas lo habían
confirmado. Faltaba preguntarse **de dónde** sacar la información que falta.

### El diagnóstico que reabrió la fase de datos externos

La brecha contra el mercado no es uniforme (rolling-origin, peleas con cuota):

| corte | peleas | modelo | mercado | brecha |
|---|---|---|---|---|
| debutante en el ring | 1301 | 0.6688 | **0.5734** | **0.0954** |
| resto | 4472 | 0.657 | 0.620 | ~0.037 |
| favorito de mercado >0.80 | 629 | 0.5517 | **0.3975** | **0.1542** |

**Corrige el corolario de la ronda anterior** (`DECISIONS.md`, ronda de odds), que mató
la fase de récord pre-UFC porque "en debutantes el modelo rinde igual que en el resto".
Es cierto y es la comparación equivocada: lo que importa no es cuánto pierde el modelo
ahí sino cuánto se puede ganar, y el mercado demuestra que 0.5734 es alcanzable donde el
modelo saca 0.6688. Las peleas de debutantes son las más predecibles del deporte.

### Se probó TODO lo que quedaba sin usar en ufcstats. Todo no concluyente.

Seis bloques nuevos, con `train.py probar` (20 folds, baseline 0.6592):

- golpeo por objetivo (head/body/leg, ofensa y defensa): +0.0000 [−0.0010, +0.0009]
- golpeo por posición (distance/clinch/ground): +0.0003 [−0.0005, +0.0011]
- volumen (total str. / sig. str.): −0.0005 [−0.0011, +0.0002]
- cardio (fade R3+ vs R1-2 en rondas completas, rounds tardíos): −0.0003 [−0.0010, +0.0004]
- calidad de victorias (margen de tarjetas, split rate, peleas de título): −0.0003 [−0.0013, +0.0007]
- stance (southpaw como matchup antisimétrico, re-test del descarte viejo): −0.0002 [−0.0011, +0.0007]

Eso cubre las 12 columnas de `ufc_fight_stats.csv` que nunca se habían leído, la
granularidad por round que el `.sum()` destruía, y las tarjetas de los jueces de
`DETAILS`. **Cero líneas quedaron.**

**Por qué**, mirando AUC univariada contra correlación con las 22 features actuales: lo
que tiene señal ya estaba capturado (`title_win_rate` AUC 0.614 pero correlaciona 0.592
con `win_rate`; `ground_pct` 0.637 con `ctrl_per_min`) y lo genuinamente ortogonal es
ruido (`fade` AUC 0.496 con correlación 0.110; `es_southpaw` 0.518 con 0.045). El pozo de
ufcstats está seco **para información**, no solo para features.

Detalle metodológico: `_margen` de las tarjetas no se puede orientar. En ufcstats el
score va siempre perdedor-ganador sin importar el orden del bout — verificado sobre 3980
decisiones, donde "el ganador va primero" da 38.6%, que es exactamente la fracción de
peleas que gana `fighter_b`. Solo sirve la magnitud.

### Lo que sí entró: circunstancia de la pelea, vía Wikipedia

`wiki.py` extrae de la sección *Background* de cada evento quién entró de reemplazo y
quién no dio el peso. **Es lo único del pipeline que no sale del historial deportivo.**

- **`reemplazo` + `peso_no_dado`: −0.0020, IC95% [−0.0033, −0.0006] → queda.** Primer
  bloque que pasa el protocolo de 20 folds desde que el protocolo existe.
- Crudo, sobre 8639 peleas: el que entra de reemplazo gana el **39.1%** (n=860) y el que
  no da el peso el **41.1%** (n=253), contra 50% de base. Simétrico exacto en el espejado.
- Cobertura: 629/781 eventos con artículo validado (83% de las peleas; 92% desde 2018,
  6% antes de 2005 — los eventos viejos no tienen artículo).
- Descartado en la misma ronda: `peso_no_dado_rate`, la tasa acumulada de no dar el peso,
  +0.0000 [−0.0003, +0.0004]. **El hábito no informa, la circunstancia sí** — y juntas
  rinden menos (−0.0015) que la circunstancia sola.

Números nuevos del modelo: rolling-origin 0.6592 → **0.6572**; test 0.6371 → **0.6355**;
brecha contra el mercado 0.0500 → **0.0475**. `predict.py` re-derivó `CONFIANZA` y
`ROI_APOSTABLE` (tramo alta: +5.82% → **+3.18%**, IC95% [−4.51%, +10.80%], n=878).

Dos decisiones de diseño que valen para quien siga:

- **`reemplazo`/`peso_no_dado` no son estado**, son circunstancia de la pelea, así que no
  viven en `fighter_state.csv`: entran por `predict.predict(circ_a=, circ_b=)` con default
  (0, 0) = campamento normal, y la app los expone como checkboxes. Quien carga la pelea
  sabe el dato — es noticia pública — y el modelo no puede deducirlo de ningún historial.
- **"No mencionado" solo significa 0 si el evento tiene artículo.** Sin artículo es NaN.
  Por eso `wiki.py` escribe `data/wiki_eventos.csv` además de los avisos: sin esa lista,
  media base recibiría "todo normal" y la feature sería ruido con cara de dato.

### Método: el mercado le gana por goleada, pero hoy no se puede servir

Props de método de `ufc_odds.csv` (`r_ko_odds`, `b_sub_odds`, …), sin usar hasta ahora,
desvigueadas normalizando las tres vías. Sobre 4591 peleas con prop:

| | log loss |
|---|---|
| base rate | 1.0187 |
| contexto (el modelo actual) | 1.0174 |
| props como feature del HistGB | 1.0489 (**se descarta**, +0.0315 [+0.0129, +0.0496]) |
| **prop del mercado cruda** | **0.9582** (−0.0592 [−0.0703, −0.0486], queda) |
| blend contexto + mercado | 0.9706 (−0.0467) |

El modelo de contexto le gana al base rate por 0.0013; el mercado le gana por 0.059.
Meterle las props al GBM las **empeora** — las re-aprende peor de lo que vienen.

**Bloqueado en el serving, no en la medición:** `ufc_odds.csv` está congelado en 2026-04
y las fuentes vivas no traen props de método. Verificado en vivo: Betano expone solo
"Ganador" (2 selecciones) en sus páginas de cartelera, y `oddsapi.py` pide `markets=h2h`.
Queda `train.probar_metodo` listo para re-correr si aparece una fuente. No se cableó nada:
sería código muerto.

## Ronda de récord pre-UFC (2026-08-01): Sherdog, la segunda fuente externa

**Qué se tapó.** El 25% de las peleas tiene un debutante en UFC, que entraba al modelo con
`n_fights=0` y Elo default; el 55% tiene a alguien con ≤2 peleas, donde las tasas
acumuladas son ruido sobre una o dos peleas. Sherdog trae el historial regional **con fecha
por combate**, y esa fecha es lo único que permite cortar en el debut sin filtrar futuro.
Mediana: 10 peleas previas por peleador, o sea un currículum entero que el modelo no veía.

**Lo que pasó el protocolo: `previo_metodo`** (`prev_ko_w`, `prev_sub_w`, `prev_ko_l`,
`prev_sub_l`), −0.0022 IC95% [−0.0039, −0.0006] en el rolling-origin de 20 folds.
Cobertura 2706/2724 peleadores (99.3%); los 18 restantes quedan NaN.

**Descartado en la misma ronda: el récord crudo** (`prev_n`, `prev_w`, `prev_l`). Pasa solo
(−0.0014 [−0.0028, −0.0001]) pero no aporta **nada** arriba del método: los siete juntos
rinden −0.0021, menos que los cuatro solos. Mismo patrón que la ronda anterior con el
hábito de no dar el peso: lo que informa es el detalle, no el agregado.

Crudo, sobre 8639 peleas, y es lo que hay que entender de la feature:

| | gana |
|---|---|
| el que llega con más peleas regionales | 47.4% |
| el que llega con más victorias previas | 48.9% |
| el que llega con más derrotas previas | 44.7% |
| el que llega con más KO recibidos | **43.8%** |

**Más experiencia regional no es mejor**: el veterano de circuito no es un prospecto. Y las
victorias previas casi no informan por sesgo de selección — a UFC no llega nadie con récord
perdedor (win rate previo mediano 0.85). Informa lo que el filtro de entrada no borra.

Números nuevos del modelo: rolling-origin 0.6572 → **0.6550**; test 0.6355 → **0.6342**;
brecha contra el mercado 0.0475 → **0.0453**. `predict.py` re-derivó `CONFIANZA` y
`ROI_APOSTABLE` (tramo alta: +3.18% → **+1.87%**, IC95% [−5.56%, +9.40%], n=863).

Tres decisiones de diseño que valen para quien siga:

- **El récord pre-UFC sí viaja en `fighter_state.csv`**, al revés que la circunstancia de
  Wikipedia. No es estado acumulado —es una condición inicial fija, anterior al debut, que
  no se actualiza nunca— pero `predict` arma el snapshot desde esa tabla, así que sin
  guardarlo la app serviría NaN en cuatro features que el modelo sí usa.
- **La validación no es por nombre, es por pelea.** El buscador de Sherdog devuelve cuatro
  "Alex Pereira" distintos. Cada ficha se valida contra peleas de UFC que ya conocemos
  (fecha exacta + apellido del rival); si no coincide ninguna, se descarta. Y cuando el
  nombre directamente no une las dos fuentes ("Aleksei Oleinik" en ufcstats es "Alexey
  Oleynik" en Sherdog), la segunda pasada saca la URL de la ficha de un rival ya bajado,
  buscando por fecha. Rescató 171 peleadores sin pedir un solo request.
- **`train.calidad_por_tramo` es nuevo y existe por una razón:** los log loss por tramo que
  sostienen los textos de `predict.CONFIANZA` se derivaban a mano. Ahora se imprimen en cada
  corrida. Una etiqueta de confianza que quedó vieja miente con autoridad.

**Verificación de la fuente, que fue lo que destapó los bugs.** Se cruzó el historial
parseado contra el total W-L que declara la propia ficha: **2760/2760 cuadran exacto**. Ese
cruce encontró tres errores que los spot-checks sueltos no veían — la ficha tiene **tres**
tablas apiladas (PRO, PRO EXHIBITION del TUF, AMATEUR) y sumarlas inflaba el récord
(O'Malley salía 13-1 habiendo debutado 8-0); `TKO (Submission to Punches)` se contaba como
KO y como sumisión a la vez; y las peleas contra "Unknown Fighter", que no llevan link, se
descartaban enteras. Los tres tienen caso en `sherdog.py --autocheck`.

**Alternativas descartadas antes de scrapear** (verificado 2026-08-01): los datasets de
Kaggle son derivados de ufcstats, o sea lo que ya teníamos; el único dump de Sherdog con
datos publicado (`LittleLebowskiUrbanAchievers/Data-Collection`) trae agregados W/L/D sin
pelea por pelea y está congelado en 2017; el resto de los repos son scrapers, no datos. Y
el fondo del asunto: **cualquier agregado es el récord de hoy, con las peleas de UFC
adentro** — usarlo sería filtrar futuro. Sin fecha por combate no hay corte anti-leakage.

## Hechos externos verificados (2026-07-30, ampliado 2026-07-31)

- **`ufcstats.com` está detrás de un challenge JS de proof-of-work.** No se puede scrapear
  con `requests`. El mirror de `Greco1899/scrape_ufc_stats` no es una comodidad: es la
  única vía práctica. Su repo tiene exactamente los 6 CSVs que ya se bajan.
- **Sherdog responde** con UA normal (107 KB, bloque de récord W/L/NC + historial): es la
  fuente viable para el récord pre-UFC. **Tapology está bloqueado por Cloudflare** (403).
- **Tapology queda descartada por dos razones independientes** (re-verificado 2026-07-31):
  el 403 es un *challenge interactivo* de Cloudflare, o sea un control de acceso activo que
  habría que romper; y su `robots.txt` declara `Content-Signal: ai-train=no, use=reference`
  con reserva expresa de derechos bajo el Artículo 4 de la directiva UE 2019/790. No se
  implementa bypass.
- **Sherdog `robots.txt` es `User-agent: * / Allow: /`**, sin content-signals ni challenge.
  La ficha de peleador es una tabla HTML limpia con **fecha por combate** (`['loss',
  'Rival', 'Evento  Mar / 07 / 2020', 'Submission (RNC)  Arbitro', '1', '2:23']`),
  incluyendo circuito regional — la fecha es lo que permite el filtro anti-leakage.
  **Las URLs por nombre solo (`/fighter/Alex-Pereira`) dan 404**: hace falta resolver el ID
  vía `/stats/fightfinder?SearchTxt=`, o sea ~2 requests por peleador × 2716.
- **Sherdog resuelto y explotado** (2026-08-01): las URLs por nombre solo dan 404, el ID se
  resuelve por `/stats/fightfinder?SearchTxt=`, y el crawl completo son ~2700 fichas a 0.5 s.
  El mapa nombre→ID se llena solo con los rivales linkeados en cada ficha, así que la mitad
  de las búsquedas se evita. El buscador tira 500 intermitentes (13 en 2724) y timeouts (16);
  ninguno es rate limit — con sesión nueva responde 200 al instante. Ver la ronda de arriba.
- **Sherdog NO sirve para reemplazos ni peso no dado**: solo los narra en prosa dentro de
  notas de pesaje ("Medic officially weighed 171 pounds on the dot") y solo de años
  recientes. Wikipedia cubre lo mismo desde UFC 1, vía API pública y con licencia CC BY-SA.
- **El buscador de la API de Wikipedia resuelve a artículos equivocados** con total
  naturalidad: "Jung vs. Ige" devuelve la ficha de Dan Ige, "Belfort vs Henderson 2"
  devuelve el 3. Por eso `wiki._es_el_evento` valida cada artículo contra la **fecha** del
  evento antes de creerle. Un reemplazo atribuido al evento equivocado es peor que un NaN.
  Con `PAUSA` de 0.15 s la API devuelve 429; 0.5 s con backoff aguanta los 781 eventos.

## Open questions for the spec author

- **Sacar del PLAN la advertencia de "el scrape tarda horas" (paso 5 / T5).** Quedó del
  diseño con scraper propio; con los CSVs de Greco1899 la descarga tarda segundos, así
  que ponerla en el README sería mentir. El review en contexto limpio la marcó como
  GAP 1 y coincidió en que el README tiene razón y la línea del PLAN es la obsoleta.
  Como `PLAN.md` es read-mostly, la corrección queda a cargo del autor del spec.
