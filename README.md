# Predictor de peleas UFC

Predice P(gana A) / P(gana B) para cualquier matchup de UFC y lo compara contra la
cuota de la casa — la de Betano se busca sola para las peleas anunciadas. La idea es que
sea una fuente de información más para decidir, no un detector automático de valor.

## Instalación

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Pipeline

Correr en este orden. Total: un par de minutos.

```sh
.venv/bin/python fetch_data.py    # descarga los datos (segundos)
.venv/bin/python features.py      # construye las features (~1 min)
.venv/bin/python train.py         # entrena y evalúa (~1 min)
.venv/bin/streamlit run app.py    # la app
```

O desde la terminal, sin app:

```sh
.venv/bin/python predict.py "Khamzat Chimaev" "Sean Strickland"
.venv/bin/python cartelera.py      # los eventos anunciados, numerados
.venv/bin/python cartelera.py 3    # recorre el evento 3, una pelea a la vez
```

## Qué hace cada script

| script | qué hace | salida |
|---|---|---|
| `fetch_data.py` | baja los 6 CSVs de ufcstats.com ya scrapeados por [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats), que se refrescan a diario, más las odds históricas de [`shortlikeafox/ultimate_ufc_dataset`](https://github.com/shortlikeafox/ultimate_ufc_dataset) (2010+, matchean el 95% de ese período). Re-correrlo = datos al día. | `data/raw/*.csv` |
| `features.py` | recorre las peleas en orden cronológico y arma, para cada una, las features de ambos peleadores **usando solo sus peleas anteriores** (win rate, racha, golpes por minuto, takedowns, control, finish rate, descanso, edad, alcance, Elo, knockdowns propios y recibidos, defensas de striking y derribo, tasa de veces finalizado, Elo promedio de los rivales). Cada pelea genera dos filas: diffs A−B y la espejada B−A, para eliminar el sesgo de esquina roja. | `data/features.csv` |
| `train.py` | Promedio de `HistGradientBoostingClassifier` y una logística sin intercepto, con split temporal (test = últimos 2 años, validación = los 2 anteriores). Sin calibración: re-verificada como dañina con el protocolo nuevo. Decide con **rolling-origin de 20 folds (7195 peleas) + IC95% del delta pareado**, no con el delta de una sola ventana. Imprime train loss junto a val/test y una tabla de confiabilidad. El `model.pkl` final se re-entrena con todo el historial. | `model.pkl`, `data/fighter_state.csv` |
| `predict.py` | `predict(a, b, cuotas=None) -> dict`. Sirve el mismo promedio de modelos que se evalúa, en las dos orientaciones, así el resultado no depende del orden. Con las dos cuotas agrega la probabilidad del mercado, la del modelo alimentado con ella, el nivel de confianza y los factores que mueven la predicción. | — |
| `cartelera.py` | Baja las peleas anunciadas de la [API pública de ESPN](https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard) y predice cada una. Sin argumentos lista los eventos; con un número recorre ese evento pelea por pelea (Enter para avanzar). Descarta los `TBA` y ordena la cartelera con el main event primero. Registra el primer avistaje de cada pelea en `data/cartelera_hist.csv`, que es lo único que permite detectar reemplazos tardíos (pelea anunciada a días del evento) — justo la información que el mercado tiene y el modelo no. | `data/cartelera_hist.csv` |
| `betano.py` | `cuotas() -> {pelea: (cuota_a, cuota_b)}` leyendo el `window["initial_state"]` que [lat.betano.com](https://lat.betano.com) deja embebido en el HTML, y `buscar(tabla, a, b)` que lo matchea contra los nombres como los escribe ESPN. Dos o tres requests, sin headless browser. Si Betano no responde devuelve `{}` en vez de romper. Cada corrida exitosa deja un tick por pelea en `data/betano_hist.csv`: el histórico apertura→cierre del que sale el CLV. `python betano.py` es el modo descubrimiento: baja el estado crudo y lista las peleas con cuota. | `data/betano_raw.json`, `data/betano_hist.csv` |
| `ledger.py` | La primera vez que una pelea aparece con cuota, la predicción queda congelada en `data/ledger.csv` (simula "apostar apenas abre el mercado"). `evaluar()` la cruza después con el resultado real y con el último tick de Betano antes del evento: ROI real de las candidatas y **CLV** (closing line value), que es el único predictor confiable de rentabilidad y converge en decenas de apuestas, no en miles. Es la pestaña **Historial** de la app. | `data/ledger.csv` |
| `oddsapi.py` | Consenso multi-casa vía [The Odds API](https://the-odds-api.com) si hay `ODDS_API_KEY` en el entorno (tier gratis: 500 requests/mes, la app cachea 30 min). La mediana desvigueada de varias casas es la probabilidad "verdadera"; si la mejor cuota disponible paga más que eso, la app lo marca como **línea desalineada** — la única señal de EV que no necesita que el modelo le gane a nadie. Sin key devuelve `{}` y la app sigue igual. | — |
| `app.py` | UI Streamlit con tres pestañas. **Matchup**: elegís dos peleadores y las dos cuotas decimales, y ves las tres probabilidades, la cuota mínima para que haya valor, el nivel de confianza con su motivo, y las features que más mueven la predicción. **Cartelera**: elegís un evento próximo y ves cada pelea con su predicción, la cuota de Betano ya resuelta, el método probable por división y el consenso multi-casa si hay `ODDS_API_KEY`. **Historial**: el forward test — cada predicción congelada, su resultado y el CLV. | — |
| `test_app.py` | `python test_app.py`. Verifica que la predicción no dependa del orden, que la confianza salga del tramo correcto, que la cartelera, Betano y The Odds API se parseen bien, que el archivado y el ledger congelen y evalúen (dedup, resultado, CLV), que los homónimos avisen, que el método sea coherente por división, y que la app renderice con y sin cuotas. No toca la red. | — |

### De dónde salen las peleas que todavía no ocurrieron

ufcstats.com metió un challenge JS anti-bot en su página de eventos futuros, y el dataset
de odds (`shortlikeafox/ultimate_ufc_dataset`) está congelado desde el 2026-04-01. La API
pública de ESPN es la que quedó viva: JSON, sin API key. Trae el matchup pero **no la
cuota**.

La cuota la pone `betano.py`. Betano no tiene API pública, pero tampoco hace falta: el
sitio deja el estado entero de la página embebido en el HTML como
`window["initial_state"]`, con nombres y cuotas adentro, así que alcanza un GET por
cartelera (cada evento de UFC es una "liga" separada para ellos). Sigue siendo scraping y
se va a romper: cuando pase, `python betano.py` baja el estado crudo y lista lo que
parsea. La app lo cachea 30 minutos, que es también el rate limit contra el sitio.

Dos límites que no son bugs: Betano abre mercado unos días antes del evento, así que las
carteleras lejanas aparecen sin cuota; y el matcheo de nombres es exacto tras normalizar
acentos, con un fallback difuso (`difflib`) para los `Jr.` y los nombres compuestos. Sobre
la cartelera del 2026-08-01 matcheó 14/14, incluidos `Uroš Medić`, `Mateusz Rębecki` y
`Stephanie Luciano` → `Stephanie Bruna Luciano`. Si Betano no responde, la cartelera se
muestra igual, sin cuota ni nivel de confianza.

Los debutantes se listan sin predicción: sin historial en UFC no hay features que
alimentar. Son ~30% de las peleas anunciadas, casi todas de Contender Series y de las
carteleras europeas. (El método por división sí se muestra: no depende del par.)

Otra trampa del dataset: ufcstats tiene 8 nombres repetidos que son peleadores
distintos (dos "Bruno Silva", dos "Jean Silva"…) y las peleas solo traen el nombre,
así que sus historiales quedan mezclados sin arreglo posible. La app y el CLI lo
avisan en vez de servir esa predicción contaminada en silencio.

## Resultados actuales

Sobre las peleas de los últimos 2 años (1029 peleas únicas, nunca vistas en
entrenamiento), evaluando la predicción desplegada — el promedio de ambas
orientaciones, igual que `predict.py`:

| | log loss | Brier | accuracy |
|---|---|---|---|
| **modelo** | 0.6371 | 0.2230 | **0.6560** |
| moneda | 0.6931 | 0.2500 | 0.5000 |
| mayor Elo | — | — | 0.5588 |

~65% de acierto está en el techo de lo que logran los modelos públicos de UFC.

## El mercado le gana al modelo, y por mucho

Sobre las 5679 peleas del rolling-origin que tienen cuota (la implícita se desviguea
con el **método power**, no proporcional: medido mejor sobre 6916 peleas, −0.0009 de
log loss con IC95% [−0.0015, −0.0003], y −0.0048 en favoritos de >75%, que es donde
el proporcional infla al underdog):

| | log loss |
|---|---|
| modelo (sin odds) | 0.6608 |
| **mercado (implícita sin vig)** | **0.6107** |
| modelo alimentado con la cuota | 0.6123 |

El mercado le gana al modelo por 0.0500 (IC95% [−0.0584, −0.0421]), y meterle la cuota al
modelo **no mejora sobre la cuota sola** (+0.0016, IC95% [−0.0020, +0.0054]). O sea: las 22
features no aportan nada que la línea de cierre no tenga ya.

Por eso esto no sirve para buscar valor contra el cierre. Sirve como **segunda opinión
independiente**, y la discrepancia con el mercado es la señal útil — pero al revés de lo
que uno esperaría:

| discrepancia \|modelo−mercado\| | peleas | log loss modelo | log loss mercado |
|---|---|---|---|
| < 0.05 | 1413 | 0.6614 | 0.6641 |
| 0.05 – 0.15 | 2332 | 0.6426 | 0.6200 |
| 0.15 – 0.25 | 1365 | 0.6545 | 0.5770 |
| > 0.25 | 663 | **0.7354** | 0.5363 |

Cuando coinciden, el modelo vale tanto como la casa. Cuando discrepan fuerte, el modelo
rinde **peor que una moneda** y la casa gana: en las 1693 peleas donde eligen ganadores
distintos, la casa acierta 58.5% y el modelo 41.5%. Discrepar no es encontrar valor, es
la señal de que al modelo le falta información que el mercado sí tiene. De ahí sale el
nivel de confianza que muestra la app.

### El asterisco importante: eso es contra la línea de cierre

Las odds históricas son (casi seguro) líneas de cierre — las más afiladas que existen.
Pero nadie apuesta contra el cierre: se apuesta contra la línea de Betano días antes,
que incorpora menos información. **Si el modelo le gana a la línea de apertura es una
pregunta distinta, y hasta ahora no se podía responder** porque nadie guardaba esas
líneas. Ahora sí: `betano_hist.csv` acumula cada tick de cuota y el ledger congela la
predicción al primer avistaje. En unos meses de datos, la pestaña **Historial** responde
con CLV si apostar temprano le gana al cierre — que es la única forma realista de que
este modelo genere valor.

### Y si eso se apuesta, ¿cuánto rinde?

`train.apostabilidad` lo mide directo: flat-bet de 1 unidad a **todo lado con EV positivo**
(`p_modelo × cuota − 1 > 0`), out-of-sample, pagando con la cuota decimal real de la casa
(vig mediano 3.6%), no con la implícita sin vig.

| tramo \|modelo−mercado\| | apuestas | ROI | IC95% |
|---|---|---|---|
| < 0.05 (confianza alta) | 887 | **+5.82%** | [−1.97%, +13.22%] |
| 0.05 – 0.15 | 2329 | −6.95% | [−12.23%, −1.78%] |
| 0.15 – 0.25 | 1366 | −8.33% | [−15.98%, −0.23%] |
| > 0.25 | 663 | −7.67% | [−19.94%, +5.65%] |
| todos | 5245 | −5.24% | [−8.67%, −1.67%] |
| control: siempre el favorito | 6060 | −3.35% | [−5.23%, −1.49%] |
| control: siempre el underdog | 5486 | −6.73% | [−10.46%, −2.85%] |

El único tramo que no pierde es donde el modelo **ya coincide** con la casa, y ni ahí el
IC95% se despega de cero: es break-even con esperanza, no una ventaja probada — y como
además es el mejor de 4 tramos elegido a posteriori, la evidencia real es más débil aún
de lo que el IC sugiere. Es el único caso que la app marca como *candidata a valor*. El
resto es la trampa: donde el EV del modelo se ve más lindo (porque se aparta del mercado)
es exactamente donde pierde 7%. La vara que decide de verdad es el forward test del
**Historial**: si las candidatas no muestran CLV positivo con datos reales, se retira la
etiqueta.

Subir el umbral de EV no ayuda — está medido que filtrar por EV alto empeora. Por eso la
regla es EV > 0 y nada más.

## Cómo suele terminar la pelea

La app y `cartelera.py` muestran P(KO/TKO), P(sumisión) y P(decisión) por pelea. El
hallazgo medido (rolling-origin de 20 folds) es incómodo pero útil: **el método depende
de la división, no del par**. El contexto solo (peso, género, 5 rounds) le gana al base
rate (−0.0142, IC95% [−0.0216, −0.0069]); agregar el historial de los peleadores no
suma nada (diffs: no concluyente) y las sumas simétricas de finish/KD/sub directamente
empeoran (+0.0252). Por eso el modelo de método no pide peleadores, y funciona igual
para los debuts. Sobre test: log loss 1.0057 vs 1.0167 del base rate.

## Por qué no sirve tocar los hiperparámetros (ni casi nada del modelo)

Está medido: entre 50 y 800 árboles el train loss cae de 0.6371 a 0.4684 mientras el de
validación se mueve de 0.6613 a 0.6701. Toda la capacidad extra se va en memorizar. No es
under- ni overfitting: es **techo de información**. La única forma de mejorar es meter
información que hoy no está, no reconfigurar el modelo.

La ronda del 2026-07-31 lo confirmó por cuarta vía: interacciones ofensa×defensa
antisimétricas, contexto de división para el ganador, decay temporal de las stats
(λ=0.85 y 0.93/año) y Elo con K por finish (40/28) salieron **todas no concluyentes**
en el rolling-origin de 20 folds. Cero líneas de eso quedaron en el modelo de ganador.

Corolario para medir: el SE del delta pareado entre dos modelos sobre una ventana de 999
peleas es 0.0028, así que el viejo umbral de "queda si mejora ≥ 0.003" era un efecto de
1.1σ. Por eso `train.py` decide con rolling-origin sobre 7195 peleas y pide que el IC95%
del delta pareado no toque cero. La primera vez que se aplicó, descartó una mejora que
parecía real en validación (la logística sola, −0.0031 en val, +0.0023 en rolling-origin).

## Sin leakage

Toda la métrica de arriba depende de que ninguna feature use información posterior a
la pelea. `features.py` lo garantiza por construcción — una sola pasada cronológica
donde el estado de cada peleador se actualiza *después* de emitir la fila — y lo
verifica con un `assert` explícito. Las peleas del mismo día (los torneos de UFC 1-8)
se calculan todas antes de aplicar sus updates.

## Alcance

Ganador + método (KO/sumisión/decisión, por división), solo UFC, corre local.

Las cuotas entran como feature de un segundo modelo, que convive con el que no las usa —
la app muestra los dos. El modelo sin odds es la opinión independiente; el de con odds
está para ver cuánto lo mueve el historial. Ninguno le gana al mercado solo.

**Lo que esto no hace:** no encuentra apuestas de valor de forma confiable. Está medido que
la casa es mejor predictor, que donde el modelo más discrepa es donde más se equivoca, y
que apostar su EV positivo rinde −5.24% de ROI. Lo único que sobrevive a la medición es el
tramo de confianza alta (+3.66%, IC95% cruzando cero), que la app marca como candidata: un
break-even con esperanza, para stake chico y plano. Un modelo bien calibrado te dice cuándo
tenés razón, no cuándo cobrás — apostar favoritos claros no es gratis, el sesgo
favorito-longshot hace que las cuotas bajas estén bien pagadas.
