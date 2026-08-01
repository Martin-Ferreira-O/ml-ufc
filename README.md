# Predictor de peleas UFC

Predice P(gana A) / P(gana B) para cualquier matchup de UFC y lo compara contra la
cuota de la casa — la de Betano se busca sola para las peleas anunciadas. La idea es que
sea una fuente de información más para decidir, no un detector automático de valor.

## Instalación

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Dos variables de entorno opcionales: `ODDS_API_KEY` ([The Odds API](https://the-odds-api.com),
consenso multi-casa) y `GEMINI_API_KEY` ([Google AI Studio](https://aistudio.google.com/apikey),
para leer las imágenes con las picks de los predictores). Sin ellas la app anda igual,
solo que sin esas dos cosas.

## Pipeline

Correr en este orden. Total: un par de minutos.

```sh
.venv/bin/python fetch_data.py    # descarga los datos (segundos)
.venv/bin/python wiki.py          # reemplazos y peso no dado (cacheado: solo lo nuevo)
.venv/bin/python sherdog.py       # record pre-UFC (cacheado: solo peleadores nuevos)
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
| `wiki.py` | Extrae de la sección *Background* del artículo de cada evento en [Wikipedia](https://en.wikipedia.org/w/api.php) quién **entró de reemplazo** y quién **no dio el peso** — la única información del pipeline que no sale del historial deportivo. Valida cada artículo contra la fecha del evento antes de creerle (el buscador de la API devuelve el evento vecino con total naturalidad) y cachea por evento, así que re-correrlo solo pide los eventos nuevos. `--autocheck` corre los casos de extracción sin tocar la red. | `data/wiki_avisos.csv`, `data/wiki_eventos.csv` |
| `sherdog.py` | Resuelve cada peleador a su ficha de [Sherdog](https://www.sherdog.com) y extrae su **récord pre-UFC** del circuito regional, con fecha por combate — el corte anti-leakage es el debut en UFC. Valida cada ficha contra peleas de UFC ya conocidas (fecha + rival), porque el buscador devuelve homónimos; cachea por ficha, así que re-correrlo solo baja peleadores nuevos. `--autocheck` corre los casos de parseo sin red. | `data/sherdog_previo.csv` |
| `features.py` | recorre las peleas en orden cronológico y arma, para cada una, las features de ambos peleadores **usando solo sus peleas anteriores** (win rate, racha, golpes por minuto, takedowns, control, finish rate, descanso, edad, alcance, Elo, knockdowns propios y recibidos, defensas de striking y derribo, tasa de veces finalizado, Elo promedio de los rivales), más si entró de reemplazo o no dio el peso (de `wiki.py`) y cómo ganaba y perdía antes de llegar a UFC (de `sherdog.py`). Cada pelea genera dos filas: diffs A−B y la espejada B−A, para eliminar el sesgo de esquina roja. | `data/features.csv` |
| `train.py` | Promedio de `HistGradientBoostingClassifier` y una logística sin intercepto, con split temporal (test = últimos 2 años, validación = los 2 anteriores). Sin calibración: re-verificada como dañina con el protocolo nuevo. Decide con **rolling-origin de 20 folds (7195 peleas) + IC95% del delta pareado**, no con el delta de una sola ventana. Imprime train loss junto a val/test y una tabla de confiabilidad. El `model.pkl` final se re-entrena con todo el historial. | `model.pkl`, `data/fighter_state.csv` |
| `predict.py` | `predict(a, b, cuotas=None, circ_a=, circ_b=) -> dict`. Sirve el mismo promedio de modelos que se evalúa, en las dos orientaciones, así el resultado no depende del orden. Con las dos cuotas agrega la probabilidad del mercado, la del modelo alimentado con ella, el nivel de confianza y los factores que mueven la predicción. `circ_a`/`circ_b` marcan reemplazo y peso no dado, que no son historial sino circunstancia de esta pelea. | — |
| `cartelera.py` | Baja las peleas anunciadas de la [API pública de ESPN](https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard) y predice cada una. Sin argumentos lista los eventos; con un número recorre ese evento pelea por pelea (Enter para avanzar). Descarta los `TBA` y ordena la cartelera con el main event primero. Registra el primer avistaje de cada pelea en `data/cartelera_hist.csv`, que es lo único que permite detectar reemplazos tardíos (pelea anunciada a días del evento) — justo la información que el mercado tiene y el modelo no. | `data/cartelera_hist.csv` |
| `betano.py` | `cuotas() -> {pelea: (cuota_a, cuota_b)}` leyendo el `window["initial_state"]` que [lat.betano.com](https://lat.betano.com) deja embebido en el HTML, y `buscar(tabla, a, b)` que lo matchea contra los nombres como los escribe ESPN. Dos o tres requests, sin headless browser. Si Betano no responde devuelve `{}` en vez de romper. Cada corrida exitosa deja un tick por pelea en `data/betano_hist.csv`: el histórico apertura→cierre del que sale el CLV. `python betano.py` es el modo descubrimiento: baja el estado crudo y lista las peleas con cuota. | `data/betano_raw.json`, `data/betano_hist.csv` |
| `ledger.py` | La primera vez que una pelea aparece con cuota, la predicción queda congelada en `data/ledger.csv` (simula "apostar apenas abre el mercado"). `evaluar()` la cruza después con el resultado real y con el último tick de Betano antes del evento: ROI real de las candidatas y **CLV** (closing line value), que es el único predictor confiable de rentabilidad y converge en decenas de apuestas, no en miles. Es la pestaña **Historial** de la app. | `data/ledger.csv` |
| `predictores.py` | Las picks de los tipsters humanos que seguís, por evento. Las que se publican como imagen las lee **Gemini con visión** si hay `GEMINI_API_KEY` en el entorno (capa gratuita; `gemini-2.5-flash-lite` alcanza) — no es OCR: en esas infografías el pick es un tilde verde sobre la cara del elegido, no texto. Para no depender del matcheo de nombres mal escritos, al modelo se le pasa la cartelera real numerada y solo devuelve el índice de la pelea y de qué lado va. `comparar()` cruza las picks con el modelo y la cuota y marca el consenso; `aciertos()` mide a cada uno contra los resultados cargados. Es la pestaña **Predictores** de la app. | `data/picks.csv`, `data/resultados.csv` |
| `oddsapi.py` | Consenso multi-casa vía [The Odds API](https://the-odds-api.com) si hay `ODDS_API_KEY` en el entorno (tier gratis: 500 requests/mes, la app cachea 30 min). La mediana desvigueada de varias casas es la probabilidad "verdadera"; si la mejor cuota disponible paga más que eso, la app lo marca como **línea desalineada** — la única señal de EV que no necesita que el modelo le gane a nadie. Sin key devuelve `{}` y la app sigue igual. | — |
| `app.py` | UI Streamlit con cuatro pestañas. **Matchup**: elegís dos peleadores y las dos cuotas decimales, y ves las tres probabilidades, la cuota mínima para que haya valor, el nivel de confianza con su motivo, y las features que más mueven la predicción. **Cartelera**: elegís un evento próximo y ves cada pelea con su predicción, la cuota de Betano ya resuelta, el método probable por división y el consenso multi-casa si hay `ODDS_API_KEY`. **Predictores**: subís la imagen con las picks de cada tipster (o las cargás a mano), y salen las tres opiniones lado a lado con el modelo y la cuota, el parlay de consenso, y el acierto histórico de cada uno. **Historial**: el forward test — cada predicción congelada, su resultado y el CLV. | — |
| `test_app.py` | `python test_app.py`. Verifica que la predicción no dependa del orden, que la confianza salga del tramo correcto, que la cartelera, Betano y The Odds API se parseen bien, que el archivado y el ledger congelen y evalúen (dedup, resultado, CLV), que los homónimos avisen, que reemplazo y peso no dado bajen al que los tiene sin romper la simetría, que el método sea coherente por división, que las picks de los predictores se guarden sin duplicar y que su consenso y su acierto se calculen bien, y que la app renderice con y sin cuotas. No toca la red. | — |

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
| **modelo** | 0.6342 | 0.2217 | **0.6482** |
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
| modelo (sin odds) | 0.6561 |
| **mercado (implícita sin vig)** | **0.6107** |
| modelo alimentado con la cuota | 0.6122 |

El mercado le gana al modelo por 0.0453 (IC95% [−0.0533, −0.0374]), y meterle la cuota al
modelo **no mejora sobre la cuota sola** (+0.0014, IC95% [−0.0024, +0.0051]). O sea: las 28
features no aportan nada que la línea de cierre no tenga ya.

Por eso esto no sirve para buscar valor contra el cierre. Sirve como **segunda opinión
independiente**, y la discrepancia con el mercado es la señal útil — pero al revés de lo
que uno esperaría:

| discrepancia \|modelo−mercado\| | peleas | log loss modelo | log loss mercado |
|---|---|---|---|
| < 0.05 | 1395 | 0.6487 | 0.6503 |
| 0.05 – 0.15 | 2416 | 0.6423 | 0.6225 |
| 0.15 – 0.25 | 1357 | 0.6540 | 0.5695 |
| > 0.25 | 605 | **0.7328** | 0.5674 |

Cuando coinciden, el modelo vale tanto como la casa. Cuando discrepan fuerte, el modelo
rinde **peor que una moneda** y la casa gana: en las 1679 peleas donde eligen ganadores
distintos, la casa acierta 57.0% y el modelo 43.0%. Discrepar no es encontrar valor, es
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
| < 0.05 (confianza alta) | 863 | **+1.87%** | [−5.56%, +9.40%] |
| 0.05 – 0.15 | 2417 | −5.22% | [−10.50%, −0.19%] |
| 0.15 – 0.25 | 1358 | −10.83% | [−18.19%, −3.17%] |
| > 0.25 | 605 | +2.74% | [−11.07%, +16.33%] |
| todos | 5243 | −4.59% | [−8.19%, −0.81%] |
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

### La mejora más grande del repo, bloqueada en el serving

`ufc_odds.csv` trae props de método sin usar (`r_ko_odds`, `b_sub_odds`, …). Desvigueadas
normalizando las tres vías, sobre las 4591 peleas que las tienen:

| | log loss |
|---|---|
| base rate | 1.0187 |
| contexto (el modelo que se sirve) | 1.0174 |
| props como feature del HistGB | 1.0489 — **se descarta** (+0.0315, IC95% [+0.0129, +0.0496]) |
| **prop del mercado cruda** | **0.9582** (−0.0592, IC95% [−0.0703, −0.0486]) |

El modelo de contexto le gana al base rate por 0.0013; el mercado le gana por 0.059 —
45 veces más. Y metérselas al GBM como features las **empeora**: las re-aprende peor de lo
que vienen.

No está cableado porque hoy no se puede servir: `ufc_odds.csv` está congelado en 2026-04 y
las fuentes vivas no traen props de método (verificado: Betano expone solo "Ganador" de 2
selecciones en sus páginas de cartelera, y `oddsapi.py` pide `markets=h2h`). Cablearlo sería
código muerto. `train.probar_metodo` queda listo para re-correr el día que aparezca una
fuente — es la mejora más barata que le queda al repo.

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

## El techo era de fuente, no de features

`python train.py probar` mide bloques de features candidatas contra el modelo actual con
ese mismo protocolo. Se le pasó **todo lo que quedaba sin usar en los CSVs de ufcstats**:
el desglose de golpeo por objetivo (head/body/leg) y por posición (distance/clinch/ground)
—las 12 columnas de `ufc_fight_stats.csv` que el pipeline venía tirando—, la granularidad
por round que el `.sum()` destruía (cardio: ritmo en R3+ contra R1-2, solo en rondas
completas), las tarjetas de los jueces de `DETAILS`, las peleas de título, y el stance.

**Los seis bloques salieron no concluyentes**, con deltas dentro de ±0.0005. El porqué se
ve cruzando AUC univariada con correlación contra las 22 features de entonces:

| | AUC sola | correlación máx con las features existentes |
|---|---|---|
| `title_win_rate` | 0.614 | **0.592** con `win_rate` |
| `dec_dominance` | 0.568 | **0.550** con `win_rate` |
| `ground_pct` | 0.536 | **0.637** con `ctrl_per_min` |
| `fade` | 0.496 | 0.110 |
| `es_southpaw` | 0.518 | 0.045 |

Lo que tiene señal ya estaba capturado, y lo genuinamente ortogonal es ruido. El pozo de
ufcstats está seco **para información**, no solo para features: no hay forma de re-mirar
esos CSVs que agregue algo.

Lo que sí funcionó fue cambiar de fuente. La brecha contra el mercado no es uniforme —
se concentra donde el modelo no puede ver nada:

| corte | peleas | modelo | mercado | brecha |
|---|---|---|---|---|
| debutante en el ring | 1301 | 0.6688 | **0.5734** | **0.0954** |
| resto | 4472 | 0.657 | 0.620 | ~0.037 |
| favorito de mercado > 0.80 | 629 | 0.5517 | **0.3975** | **0.1542** |

## Reemplazos y peso no dado (`wiki.py`)

De ahí salió la única feature nueva que pasó el protocolo. `wiki.py` extrae de la sección
*Background* del artículo de cada evento en Wikipedia quién **entró de reemplazo** y quién
**no dio el peso**. Es lo único del pipeline que no sale del historial deportivo: un
peleador que entró con tres días de aviso está en una pelea distinta a la que dice su
récord, y ninguna feature acumulada puede verlo.

Crudo, sobre las 8639 peleas: el que entra de reemplazo gana el **39.1%** (n=860) y el que
no da el peso el **41.1%** (n=253), contra 50% de base. Como bloque: **−0.0020, IC95%
[−0.0033, −0.0006]**. Cobertura 629/781 eventos (83% de las peleas, 92% desde 2018).

Descartado en la misma ronda: la *tasa acumulada* de no dar el peso (+0.0000). El hábito
no informa, la circunstancia sí.

Dos detalles que importan si se toca esto:

- **No son estado, son circunstancia**, así que no viven en `fighter_state.csv`. Entran por
  `predict.predict(circ_a=, circ_b=)`, con default "campamento normal", y la app los expone
  como checkboxes. El dato es noticia pública: lo sabe quien carga la pelea, no el modelo.
- **"No mencionado" solo significa 0 si el evento tiene artículo**; sin artículo es NaN. Por
  eso `wiki.py` escribe también `data/wiki_eventos.csv`. Sin esa lista, media base recibiría
  "todo normal" y la feature sería ruido con cara de dato.
- El buscador de la API de Wikipedia resuelve a artículos equivocados con naturalidad
  ("Jung vs. Ige" → la ficha de Dan Ige), así que cada artículo se valida contra la **fecha**
  del evento antes de usarlo.

**Tapology no se scrapea**: su 403 es un challenge interactivo de Cloudflare y su
`robots.txt` declara `ai-train=no` con reserva expresa de derechos. Sherdog sí está abierto,
pero solo narra los pesajes en prosa de años recientes. Para lo que sí sirve es para el
récord pre-UFC, que es la sección siguiente.

## Récord pre-UFC (`sherdog.py`)

El agujero más grande que quedaba, y no era de features sino de fuente: **en el 25% de las
peleas hay un debutante en UFC**, que entra al modelo con `n_fights=0` y Elo default, y en
el **55%** hay alguien con ≤2 peleas, donde las tasas acumuladas son ruido calculado sobre
una o dos peleas. Sherdog tiene el historial completo del circuito regional **con fecha por
combate**, que es lo que permite contar solo lo anterior al debut y no filtrar futuro.

Pasó el bloque de **método**: `prev_ko_w`, `prev_sub_w`, `prev_ko_l`, `prev_sub_l` —
**−0.0022, IC95% [−0.0039, −0.0006]**. Cobertura 2706/2724 peleadores (99.3%).

Crudo, y es lo que hay que entender de la feature:

| | gana |
|---|---|
| el que llega con más peleas regionales | 47.4% |
| el que llega con más victorias | 48.9% |
| el que llega con más derrotas | 44.7% |
| el que llega con más KO recibidos | **43.8%** |

O sea: **más experiencia regional no es mejor** — el veterano de circuito no es un
prospecto. Y las victorias previas casi no informan, porque a UFC no llega nadie con récord
perdedor (win rate previo mediano: 0.85). Lo que informa es lo que el filtro de entrada no
borra: las derrotas, y sobre todo los KO recibidos.

**Descartado en la misma ronda**: el récord crudo (`prev_n`, `prev_w`, `prev_l`). Pasa solo
(−0.0014) pero no aporta nada arriba del método — los siete juntos rinden −0.0021, menos
que los cuatro solos.

Tres cosas que importan si se toca esto:

- **La ficha de Sherdog tiene tres tablas apiladas**: PRO, PRO EXHIBITION (las peleas de la
  casa del TUF) y AMATEUR. Sumarlas infla el récord — O'Malley salía 13-1 cuando debutó 8-0.
  El parser se queda solo con PRO, y eso está verificado contra el total W-L que declara la
  propia página: **2760/2760 fichas cuadran exacto**.
- **El buscador devuelve homónimos con cara de nada** ("Alex Pereira" da cuatro fichas), así
  que cada ficha se valida contra peleas de UFC que ya conocemos: fecha exacta + apellido
  del rival. Si no coincide ninguna, no es el peleador.
- **Los nombres no coinciden entre fuentes** y el buscador no lo tolera: "Aleksei Oleinik"
  en ufcstats es "Alexey Oleynik" en Sherdog, y "AJ Dobson" devuelve cero resultados. La
  segunda pasada los resuelve sin pedir red — la ficha de cualquier rival ya bajado tiene la
  fila de esa pelea, con la fecha y el link. El nombre deja de importar.

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
que apostar su EV positivo rinde −4.87% de ROI. Lo único que sobrevive a la medición es el
tramo de confianza alta (+1.87%, IC95% cruzando cero), que la app marca como candidata: un
break-even con esperanza, para stake chico y plano. Un modelo bien calibrado te dice cuándo
tenés razón, no cuándo cobrás — apostar favoritos claros no es gratis, el sesgo
favorito-longshot hace que las cuotas bajas estén bien pagadas.
