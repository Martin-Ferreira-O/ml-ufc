# Predictor de peleas UFC

[![Licencia: MIT](https://img.shields.io/badge/c%C3%B3digo-MIT-blue.svg)](LICENSE)
[![Datos: CC BY 4.0](https://img.shields.io/badge/datos-CC%20BY%204.0-blue.svg)](LICENSE-DATA)
[![Python 3.13](https://img.shields.io/badge/python-3.13-3776ab.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-app-ff4b4b.svg)](https://streamlit.io/)

Predice P(gana A) / P(gana B) para cualquier matchup de UFC, lo compara contra la cuota
de la casa — la de Betano se busca sola para las peleas anunciadas — y **congela cada
predicción con fecha** para poder medirla después contra lo que pasó. La idea es que sea
una fuente de información más para decidir, no un detector automático de valor.

## Lo importante primero

Un repo de predicción de peleas suele abrir con la accuracy y cerrar ahí. Este mide
también lo otro, que es lo que importa si el número toca dinero:

| | |
|---|---|
| Accuracy sobre 1029 peleas nunca vistas en entrenamiento | **64.8%** (moneda 50%, mayor Elo 55.9%) |
| Log loss del modelo vs. la línea de la casa sin vig | 0.6561 vs **0.6107** — el mercado gana por 0.0453 |
| ROI de apostar todo el EV positivo del modelo, out-of-sample | **−4.59%**, IC95% [−8.19%, −0.81%] |
| Umbrales de ventaja (de 21 probados) cuyo IC95% de ROI supera cero | **ninguno** |
| Apuestas necesarias para verificar un ROI de +2% | **19.623** (~40 años a 500/año) |
| Apuestas necesarias para verificar un CLV de +2% | **32** |

**El mercado le gana al modelo, y por mucho.** Está medido, no estimado: donde el modelo
más discrepa de la casa es justo donde peor rinde — en las 1679 peleas donde eligen
ganadores distintos, la casa acierta 57.0% y el modelo 43.0%. Por eso la app **no genera
candidatas de apuesta automáticas** y su default es "sin apuesta".

Lo que sí sirve: 65% de acierto está en el techo de lo que logran los modelos públicos de
UFC, la discrepancia con el mercado es una señal descriptiva útil, y el ledger acumula
desde ahora las líneas de **apertura** que nadie guardaba — la única pregunta abierta que
todavía puede dar valor. Todo el detalle, con intervalos de confianza, más abajo.

Y la conclusión que ordena todo lo demás, de
[`docs/consultoria-apuestas.md`](docs/consultoria-apuestas.md): **el EV de este proyecto
no puede venir del modelo, tiene que venir del precio.** Un pool logit de dos parámetros
—el estimador de mínima varianza para “¿el modelo aporta algo sobre la cuota?”— da no
concluyente contra el mercado solo. Lo que queda por explotar es la mejor cuota entre
casas y el consenso multi-casa, que no requieren ganarle a nadie. La app calcula el stake
que correspondería, pero `config/gate.json` mantiene el gate **cerrado** hasta que 100
apuestas muestren CLV positivo con el IC95% despegado de cero.

## Instalación

Probado en Python 3.13. Sin GPU, sin Docker, corre local.

```sh
git clone https://github.com/<usuario>/ml-ufc.git
cd ml-ufc
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

El repo **no versiona el modelo ni los datos crudos** (ver [Qué versiona el
repo](#qué-versiona-el-repo-y-qué-no)), así que un clon recién bajado necesita correr el
pipeline una vez antes de levantar la app. Son un par de minutos:

```sh
.venv/bin/python -m ufc.datos.fetch      # descarga los CSVs de ufcstats (segundos)
.venv/bin/python -m ufc.datos.wiki       # reemplazos y peso no dado
.venv/bin/python -m ufc.datos.sherdog    # récord pre-UFC (la primera vez tarda: ~2700 fichas)
.venv/bin/python -m ufc.modelo.features  # construye las features (~1 min)
.venv/bin/python -m ufc.modelo.train     # entrena y evalúa (~1 min)
.venv/bin/streamlit run app.py           # la app en http://localhost:8501
```

El botón **Actualizar y re-entrenar** de la sidebar corre ese mismo pipeline desde la app.
Para verificar la instalación sin tocar la red: `.venv/bin/python test_app.py`.

Credenciales opcionales: `ODDS_API_KEY` ([The Odds API](https://the-odds-api.com),
consenso multi-casa), `GEMINI_API_KEY` ([Google AI Studio](https://aistudio.google.com/apikey),
para leer picks, informes de inteligencia y el consenso por pelea) y `X_BEARER_TOKEN`
para las cuentas de X configuradas. Sin ellas la app anda igual; se omiten esas
integraciones. `UFC_IA_MODELO` cambia el modelo de Gemini del consenso por pelea
(default `gemini-3.5-flash`).

## Inteligencia diaria de peleadores

La pestana **Inteligencia** muestra un segundo pipeline: revisa una vez por dia a todos
los peleadores del proximo evento, archiva noticias y feeds publicos, y pide a un LLM un
resumen con citas y valoracion contextual `-5..+5`. Ese numero no modifica la prediccion
ni es una recomendacion de apuesta.

```sh
# Solo recolectar y archivar, sin costo de IA
.venv/bin/python -m ufc.intel.bot --sin-ia

# Recomendado: Gemini 3.1 Flash-Lite (GEMINI_API_KEY en el entorno)
.venv/bin/python -m ufc.intel.bot

# Cobertura adicional: busqueda web de Gemini con URLs respaldadas
.venv/bin/python -m ufc.intel.bot --grounding

# Verificacion offline de 14 peleas / 28 peleadores
.venv/bin/python test_intel.py

# Salud operativa: exige run reciente, cobertura total y analisis de IA
.venv/bin/python -m ufc.intel.bot --status

# Ultimos scores, hallazgos y URLs citadas desde la VPS
.venv/bin/python -m ufc.intel.bot --resumen
```

Si primero se corrio `--sin-ia` y luego se agrega la key el mismo dia, el run normal
completa el analisis pendiente; no hace falta `--force`.

Google News se consulta para todos. Antes de cada run real, el bot resuelve y cachea los
perfiles que la ficha oficial de UFC publica en su JSON-LD. Los canales de YouTube se
leen por RSS y las cuentas de X solo con `X_BEARER_TOKEN` y la API oficial; Instagram se
muestra como identidad confirmada, pero no se scrapea. Wikidata sirve unicamente como
candidato: nunca activa una cuenta por si solo.

Para agregar o corregir feeds manualmente, copiar `config/intel_sources.example.csv` a
`data/intel_sources.csv`: admite RSS/Atom y `kind=x`. `--grounding` agrega una llamada de
Google Search por peleador y archiva solo las URLs conectadas por los respaldos de
Gemini; si falla, el informe continua con las fuentes deterministas y deja una
advertencia. Para Gemini 3 requiere un proyecto con facturacion, aunque las primeras
5.000 busquedas mensuales del tier pagado no tienen cargo. Se activa en systemd con
`UFC_INTEL_GROUNDING=1`.

La VPS puede programarlo con las plantillas `deploy/ufc-intel.service` y
`deploy/ufc-intel.timer`: reemplazar usuario/rutas, copiar el env de ejemplo a
`/etc/ml-ufc/intel.env`, instalar las dos unidades en `/etc/systemd/system/` y habilitar:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now ufc-intel.timer
systemctl list-timers ufc-intel.timer
journalctl -u ufc-intel.service
.venv/bin/python -m ufc.intel.bot --status
```

El timer ofrece tres ventanas diarias a las 09:15, 10:15 y 11:15 de
`America/Santiago`. La primera ejecucion completa deja las siguientes sin trabajo ni
llamadas de IA gracias a la idempotencia; si una fuente falla, las otras dos ventanas
reintentan solamente los peleadores pendientes. Para fixtures, usar `--db` con una ruta
temporal evita mezclar diagnosticos con el archivo productivo. El dia idempotente tambien
se calcula en `UFC_INTEL_TIMEZONE=America/Santiago`, aunque la VPS mantenga UTC.

La evaluacion completa de Hermes Agent, redes, proveedores y costos esta en
[`docs/FIGHTER_INTEL_VIABILITY.md`](docs/FIGHTER_INTEL_VIABILITY.md).

### Sincronizar Inteligencia desde Streamlit

La app local no abre SQLite a traves de la red. La pestana **Inteligencia** consulta por
SSH una metadata pequena de la VPS y compara `evento + dia de revision + updated_at`
contra `data/intel.db`. El boton **Sincronizar ahora** solo se habilita cuando la VPS
tiene una revision mas nueva. La descarga incluye la base y los CSV de identidades,
valida integridad/esquema y reemplaza los archivos de forma atomica; las copias
anteriores quedan con sufijo `.backup`.

```sh
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
.venv/bin/streamlit run app.py
```

`secrets.toml` esta ignorado por Git. La pantalla tambien muestra la ultima revision
local/remota y la proxima ventana programada. El panel se actualiza solo cada cinco
minutos; **Comprobar de nuevo** fuerza una lectura inmediata sin descargar nada.

## Consenso de IA, una pelea a la vez

El modelo y el mercado son dos números y ninguno ve lo que no está en una columna. Esta
capa arma **un prompt por pelea** con todo lo que el repo sabe de los dos peleadores —las
28 features en valor absoluto, la probabilidad del modelo con sus factores, el mercado
desvigueado y el consenso multi-casa, la distribución de método, los informes de
inteligencia con sus citas y las picks humanas con su precisión— y pide un veredicto
estructurado.

```sh
# El prompt completo de una pelea, sin llamar a la API ni gastar un peso
.venv/bin/python -m ufc.ia.consenso --dry-run --pelea 1

# La cartelera entera (GEMINI_API_KEY en el entorno o en .env)
.venv/bin/python -m ufc.ia.consenso

# Una pelea puntual, forzando el reanálisis
.venv/bin/python -m ufc.ia.consenso --pelea 3 --force

# Última corrida: cobertura y tokens
.venv/bin/python -m ufc.ia.consenso --status

# Si esto sirvió de algo, medido
.venv/bin/python -m ufc.ia.evaluar
```

También hay un botón **Analizar con IA** en la pestaña Cartelera. Cada pelea se analiza
una vez por día: volver a apretarlo, o abrir la app veinte veces, no dispara ninguna
llamada extra. Un prompt real son ~1.900 tokens de entrada, así que una cartelera de 14
peleas cuesta centavos.

**Lo que devuelve.** Pick y probabilidad, las razones separadas por fuente, los
**factores no modelables** —lo que ninguna columna puede capturar, que es el punto entero
de la capa—, el mejor argumento **en contra de su propio pick**, y si a los precios
publicados vale la pena apostar.

**Nada de lo que dice se guarda tal cual.** `analista.validar` clampea la probabilidad,
coerce los enums, descarta las razones sin fuente, corrige la pick si contradice su
propia probabilidad, y **calcula el EV en Python** con la cuota real en vez de leerlo de
la respuesta. La regla de "nombrá el lado, el precio y la casa" está en el prompt pero se
aplica en código: un "sí, apostar" que no la cumple baja solo a "mirar", y sin ninguna
cuota publicada se fuerza a "no". Es el mismo criterio que ya usa `intel/analyzer.py`
para tirar los hallazgos sin cita.

**La IA es un predictor más.** Sus picks entran a `picks.csv` con `origen="ia"`, así que
`aciertos()` y `confiabilidad()` la miden con la misma vara —y el mismo prior Beta(5, 5)—
que a las personas. Pero **no vota el consenso humano**: aparece como tercer confirmador
al lado del modelo y del mercado, porque si votara, "Consenso humano" podría salir con un
humano en minoría.

**Y no toca la cohorte del gate.** Sus apuestas se registran en `data/ia_consenso.csv` y
se miden aparte con `ufc.ia.evaluar` (acierto, log loss contra modelo y mercado sobre las
mismas peleas, y CLV contra la línea de cierre). `data/ledger.csv` —la serie
preregistrada que lee `gate.estado()`— queda intacta, y hay un check que lo verifica byte
a byte. Son dos cohortes separadas: si en 100 apuestas la de la IA muestra CLV positivo
con el IC95% despegado de cero, ahí sí hay un argumento para tocar `config/gate.json`.

## Pipeline

Correr en este orden, desde la raíz del repo. Total: un par de minutos. Van con `-m`
porque son módulos de un paquete, no scripts sueltos.

```sh
.venv/bin/python -m ufc.datos.fetch      # descarga los datos (segundos)
.venv/bin/python -m ufc.datos.wiki       # reemplazos y peso no dado (cacheado: solo lo nuevo)
.venv/bin/python -m ufc.datos.sherdog    # record pre-UFC (cacheado: solo peleadores nuevos)
.venv/bin/python -m ufc.modelo.features  # construye las features (~1 min)
.venv/bin/python -m ufc.modelo.train     # entrena y evalúa (~1 min)
.venv/bin/python -m ufc.modelo.backtest  # grilla de umbrales, staking y segmentos (~5 min, opcional)
.venv/bin/streamlit run app.py           # la app
```

Y los dos que responden preguntas de mercado, independientes del pipeline (no necesitan
`model.pkl`, solo `data/raw/ufc_odds.csv` el primero y `features.csv` el segundo):

```sh
.venv/bin/python -m ufc.modelo.devig     # qué método de desvigueo gana, y por cuánto
.venv/bin/python -m ufc.modelo.pool      # ¿el modelo aporta algo sobre el precio? (~5 min)
```

O desde la terminal, sin app:

```sh
.venv/bin/python -m ufc.modelo.predict "Khamzat Chimaev" "Sean Strickland"
.venv/bin/python -m ufc.datos.cartelera      # los eventos anunciados, numerados
.venv/bin/python -m ufc.datos.cartelera 3    # recorre el evento 3, una pelea a la vez
```

## Estructura

Cuatro carpetas, y en qué orden dependen entre sí: `ui/` → `registro/` → `datos/` →
`modelo/`. `modelo/` no importa nada de las otras tres — es el núcleo.

```
app.py                  router Streamlit: navegación superior y sidebar del pipeline
test_app.py             la suite entera, sin red
ufc/
├── rutas.py            todas las rutas, ancladas al repo y no al CWD
├── nombres.py          normalizar(): la clave canónica de un peleador
├── datos/              lo que toca la red: fetch, wiki, sherdog, cartelera,
│                       betano, oddsapi
├── modelo/             el núcleo: features, train, predict, calibra, backtest,
│                       devig, pool, apuesta, gate
├── registro/           seguimiento del modelo, predictores, apuestas reales y banca
├── intel/              el bot diario de inteligencia por peleador
├── ia/                 el consenso por pelea: dossier, analista, store, evaluar
└── ui/                 comunes.py + una página por archivo (tab_*.py)
config/gate.json        la regla de apuesta preregistrada
```

Dónde va un cambio: fuente nueva → `ufc/datos/`; feature nueva → `ufc/modelo/features.py`;
algo que se ve en pantalla → la `tab_*.py` de esa pestaña; una ruta nueva → `ufc/rutas.py`.

## Qué versiona el repo, y qué no

`data/` pesa ~490 MB en una instalación andando y casi todo eso es cache. El repo versiona
solo **135 KB**: los archivos que el pipeline no puede reconstruir hacia atrás.

| versionado | filas | por qué no se puede regenerar |
|---|---|---|
| `data/ledger.csv` | 16 | la predicción congelada al **primer avistaje** de la pelea, con la cuota de ese momento. Re-correrlo hoy daría otra cosa. |
| `data/cartelera_hist.csv` | 104 | cuándo se vio anunciada cada pelea por primera vez. Es lo único que detecta un reemplazo tardío, y sólo se puede observar en vivo. |
| `data/betano_hist.csv` | 754 | ticks de cuota apertura→cierre. Nadie publica este histórico: o lo guardás mientras pasa, o no existe. |
| `data/odds_hist.csv` | — | lo mismo pero **por casa**, vía The Odds API. Es la única fuente posible de CLV contra el mejor precio del mercado, y empieza a acumularse recién ahora. |
| `data/picks.csv` + `_audit.csv` | 255 + 309 | picks públicos de 8 predictores de MMA, recopilados a mano, con audit log append-only. |
| `data/resultados.csv` | 14 | ganadores cargados a mano. |
| `data/ia_consenso.csv` | — | el veredicto de la IA por pelea con las cuotas y la inteligencia **del día en que opinó**. Solo números y enums; el razonamiento en prosa va a `data/ia_informes/` y queda fuera. Es el forward test de esta capa. |

Lo demás queda fuera de Git a propósito, y no sólo por tamaño:

- **`data/raw/` (474 MB)** — estadísticas de ufcstats, odds históricas, HTML de Sherdog y
  texto de Wikipedia. Material de terceros; lo baja `ufc.datos.fetch` y los scrapers
  cachean por ficha, así que re-correrlos pide sólo lo nuevo.
- **`data/features.csv`, `model.pkl`, `fighter_state.csv`, `oof.csv`, `backtest.json`** —
  derivados deterministas del pipeline. Versionar un `.pkl` de 940 KB que cambia en cada
  entrenamiento es basura en el historial.
- **`data/intel.db` y los CSV de inteligencia** — resúmenes generados por un LLM sobre
  noticias y feeds públicos de **personas identificables**. No se publican.
- **`data/ia_informes/`** — lo mismo pero por pelea: el razonamiento del LLM sobre dos
  personas reales. La fila numérica del veredicto sí se versiona; la prosa no. Si falta
  (clon nuevo), la app muestra el veredicto igual y avisa que el informe no está local.
- **`data/apuestas*.csv`** — apuestas con plata real, en CLP. Privado.
- **`static/fotos/` (50 MB)** — fotos de peleadores de ufc.com y Sherdog, con derechos de
  sus titulares. Las baja `ufc.datos.fotos`; la app funciona igual sin ellas.
- **`.env` y `.streamlit/secrets.toml`** — credenciales. Hay `.example` de ambos.

## Qué hace cada módulo

| módulo | qué hace | salida |
|---|---|---|
| `ufc/datos/fetch.py` | baja los 6 CSVs de ufcstats.com ya scrapeados por [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats), que se refrescan a diario, más las odds históricas de [`shortlikeafox/ultimate_ufc_dataset`](https://github.com/shortlikeafox/ultimate_ufc_dataset) (2010+, matchean el 95% de ese período). Re-correrlo = datos al día. | `data/raw/*.csv` |
| `ufc/datos/wiki.py` | Extrae de la sección *Background* del artículo de cada evento en [Wikipedia](https://en.wikipedia.org/w/api.php) quién **entró de reemplazo** y quién **no dio el peso** — la única información del pipeline que no sale del historial deportivo. Valida cada artículo contra la fecha del evento antes de creerle (el buscador de la API devuelve el evento vecino con total naturalidad) y cachea por evento, así que re-correrlo solo pide los eventos nuevos. `--autocheck` corre los casos de extracción sin tocar la red. | `data/wiki_avisos.csv`, `data/wiki_eventos.csv` |
| `ufc/datos/sherdog.py` | Resuelve cada peleador a su ficha de [Sherdog](https://www.sherdog.com) y extrae su **récord pre-UFC** del circuito regional, con fecha por combate — el corte anti-leakage es el debut en UFC. Valida cada ficha contra peleas de UFC ya conocidas (fecha + rival), porque el buscador devuelve homónimos; cachea por ficha, así que re-correrlo solo baja peleadores nuevos. `--autocheck` corre los casos de parseo sin red. | `data/sherdog_previo.csv` |
| `ufc/modelo/features.py` | recorre las peleas en orden cronológico y arma, para cada una, las features de ambos peleadores **usando solo sus peleas anteriores**. Cada tasa por minuto mantiene su propia exposición observada; los formatos antiguos usan la duración declarada. La taxonomía versionada trata doctor stoppage como KO/TKO y no fuerza DQ/CNC/other a decisión. Cada pelea genera dos filas espejadas. | `data/features.csv` |
| `ufc/modelo/train.py` | Promedio de `HistGradientBoostingClassifier` y una logística sin intercepto, con split temporal de desarrollo y rolling-origin. Las pruebas de bloques usan bootstrap clusterizado por evento. El bundle final se escribe atómicamente e incluye manifiesto con hashes, fechas, filas, commit, dependencias, parámetros, métricas y dominio validado; el propio manifiesto declara que no existe holdout histórico intacto. | `data/model.pkl`, `data/fighter_state.csv` |
| `ufc/modelo/calibra.py` | Calibración de probabilidades: Platt (logística sobre el logit), isotónica e identidad como baseline, más ECE y curva de fiabilidad. `prequencial()` calibra cada fold con los **anteriores** — un calibrador ajustado sobre las predicciones que corrige da un ECE de casi cero por construcción. Todos simetrizan (`p(A,B) + p(B,A) = 1`) y recortan los extremos. | — |
| `ufc/modelo/backtest.py` | `python -m ufc.modelo.backtest`. Flat-bet de 1 unidad sobre una grilla de 21 umbrales de ventaja, con IC95% clusterizado por evento, curva de bankroll, drawdown y 34 niveles de segmento. Cada fila trae además **cuántas apuestas necesitaría para concluir algo**, y `comparar_staking` mide flat contra Kelly fraccional sobre las mismas apuestas (banca compuesta, drawdown en % del pico, dimensionado **por evento** porque las líneas cierran juntas). El mercado se desviguea con el método power: con la implícita cruda (`1/cuota`) el margen de la casa se contaría como valor. **No reporta CLV histórico: no es computable** (ver abajo). | `data/oof.csv`, `data/backtest.json` |
| `ufc/modelo/predict.py` | `predict(a, b, event_date=, cuotas=None, circ_a=, circ_b=) -> dict`. Usa la fecha programada para el snapshot y tres estados de circunstancia (`1/0/NaN` = sí/no/desconocido). Con cuotas muestra mercado, modelo y coincidencia descriptiva, pero **nunca recomienda automáticamente una apuesta**: el tramo anterior fue elegido a posteriori y su IC cruza cero. | — |
| `ufc/datos/cartelera.py` | Baja las peleas anunciadas de la [API pública de ESPN](https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard) y predice cada una. Sin argumentos lista los eventos; con un número recorre ese evento pelea por pelea (Enter para avanzar). Descarta los `TBA` y ordena la cartelera con el main event primero. Registra el primer avistaje de cada pelea en `data/cartelera_hist.csv`, que es lo único que permite detectar reemplazos tardíos (pelea anunciada a días del evento) — justo la información que el mercado tiene y el modelo no. | `data/cartelera_hist.csv` |
| `ufc/datos/betano.py` | `cuotas() -> {pelea: (cuota_a, cuota_b)}` leyendo el `window["initial_state"]` que [lat.betano.com](https://lat.betano.com) deja embebido en el HTML, y `buscar(tabla, a, b)` que lo matchea contra los nombres como los escribe ESPN. Dos o tres requests, sin headless browser. Si Betano no responde devuelve `{}` en vez de romper. Cada corrida exitosa deja un tick por pelea en `data/betano_hist.csv`: el histórico apertura→cierre del que sale el CLV. `python -m ufc.datos.betano` es el modo descubrimiento: baja el estado crudo y lista las peleas con cuota. | `data/betano_raw.json`, `data/betano_hist.csv` |
| `ufc/registro/ledger.py` | La primera vez que la app observa una pelea con cuota, la predicción queda congelada en `data/ledger.csv` para seguimiento experimental. No se llama “apertura” sin timestamp de publicación de la casa. `evaluar()` conserva las marcas históricas de la regla retirada y calcula resultado y CLV: el de cuotas (`clv`) y el económico (`ev_al_cierre`), este último con IC95% clusterizado por evento, `%` que batió al cierre y potencia. El cierre se toma **estrictamente antes del `inicio_utc`** del evento, no `fecha + 1 día`, que podía colar cuotas en vivo. | `data/ledger.csv` |
| `ufc/registro/predictores.py` | Mantiene `picks.csv` como vista actual y agrega un audit log append-only (`pick_created`, `pick_revised`, `pick_voided`) con timestamp UTC, origen y revisor. Los eventos pasados aparecen en una vista explícita de Predictores; allí se cargan picks/ganadores sin recalcular hoy el modelo sobre estado futuro. | `data/picks.csv`, `data/picks_audit.csv`, `data/resultados.csv` |
| `ufc/registro/apuestas.py` | Registra únicamente apuestas realmente hechas, simples o combinadas, con importe y cuota en CLP. Liquida desde los resultados cuando puede y admite corregir cobro/estado para cash-out, anulaciones y ajustes. | `data/apuestas.csv`, `data/apuestas_detalle.csv` |
| `ufc/datos/oddsapi.py` | Consenso multi-casa vía [The Odds API](https://the-odds-api.com) si hay `ODDS_API_KEY` en el entorno. Mediana desvigueada **excluyendo del consenso de cada lado a la casa que ofrece su mejor precio** (la referencia no puede contener el precio que juzga), descartando precios stale y devolviendo la dispersión entre casas como incertidumbre, no como ventaja. `valor()` da el EV con y sin descontar esa dispersión. Sin key devuelve `{}` y la app sigue igual. | `data/odds_hist.csv` |
| `ufc/modelo/devig.py` | `python -m ufc.modelo.devig`. Los cuatro métodos de desvigueo (proporcional, power, Shin, odds-ratio) resueltos por bisección, y el CLI que mide cuál gana contra el resultado real, global y por tramo de favorito. Medido: gana **power**, los otros tres se descartan con el IC95% entero arriba de cero. `predict._desvig` delega acá. | — |
| `ufc/modelo/apuesta.py` | La capa de decisión, en funciones puras: `ev`, `break_even`, `p_conservadora` (cota inferior), `kelly`, `stake` (Kelly fraccional con tope), `crecimiento` (log-growth, lo que Kelly maximiza de verdad), `riesgo_de_ruina`, `exposicion`, `ev_combinada`, `dependencia_necesaria` y `n_para_detectar` — la cuenta que explica por qué el criterio del proyecto es el CLV y no el ROI. | — |
| `ufc/modelo/pool.py` | `python -m ufc.modelo.pool`. Pool logit modelo+mercado con los pesos ajustados prequencialmente y sin intercepto (conserva `p(A,B) + p(B,A) = 1`). Es el estimador de mínima varianza para “¿el modelo aporta algo sobre el precio?”. Medido: **no concluyente** contra el mercado solo. | — |
| `ufc/modelo/gate.py` + `config/gate.json` | La regla de apuesta **preregistrada** y el código que la evalúa contra el CLV real del ledger. `estado()` devuelve autorizado/motivo/n/IC y cuánto falta. `predict.apuestas_automaticas()` se deriva de acá en vez de ser una constante: misma respuesta, pero auditable y refutable. | — |
| `ufc/ia/dossier.py` | Arma y renderiza el prompt de una pelea: las 28 features de los dos peleadores en **valor absoluto** (`features.csv` guarda diferencias, y "elo +84" no se puede leer sin saber si son 1500 contra 1416 o 2100 contra 2016), el modelo con sus factores y **su propia calibración leída del manifest** —no escrita a mano, que envejece con el primer re-entrenamiento—, el mercado, el método, la inteligencia delimitada como contenido no confiable, las picks humanas y las banderas de lo que falta. Es puro y sin red: el prompt entero se testea offline, y `--dry-run` lo imprime. | — |
| `ufc/ia/analista.py` | El esquema de respuesta y `validar()`, que es donde vive el trabajo: clampeo, coerción de enums, descarte de razones sin fuente, corrección de la pick cuando contradice su propia probabilidad, y **el EV calculado en Python** con la cuota real. La regla de "nombrá el lado, el precio y la casa" se aplica acá, no en el prompt: un "sí" que no la cumple baja a "mirar", y sin cuota publicada se fuerza a "no". | — |
| `ufc/ia/store.py` | `data/ia_consenso.csv` (numérico, versionado) y `data/ia_informes/` (prosa, fuera de git). Idempotencia por `(evento, a, b, run_day)`: una pelea se analiza una vez por día, y es lo único que evita que un rerun de Streamlit multiplique el costo. | `data/ia_consenso.csv` |
| `ufc/ia/consenso.py` | `python -m ufc.ia.consenso`. Recorre la cartelera, arma un dossier por pelea, llama a Gemini con reintentos y guarda. Vuelca las picks a `picks.csv` como `IA (Gemini)` con `origen="ia"`. `--dry-run` imprime el prompt sin llamar a la API. | `data/ia_consenso.csv`, `data/picks.csv` |
| `ufc/ia/evaluar.py` | `python -m ufc.ia.evaluar`. Lo que hace refutable a toda la capa: acierto de la pick, log loss de la IA contra modelo y mercado **sobre las mismas peleas**, y `ev_al_cierre` de sus apuestas con IC95% clusterizado por evento. Deliberadamente **no escribe en `data/ledger.csv`**: esa es la cohorte preregistrada que lee `gate.estado()` y mezclarle una segunda población la arruinaría. | — |
| `ufc/registro/banca.py` | La banca declarada, aparte de `apuestas.csv` porque cambia con depósitos y retiros, no con cada apuesta. Es el denominador de todo staking: sin banca, un importe es un número y no una decisión de riesgo. | `data/banca.json` |
| `app.py` + `ufc/ui/` | UI Streamlit con navegación superior. **Resumen** concentra próxima cartelera, señales y rendimiento; **Predictores** carga picks/resultados con un clic por ganador y permite corregir eventos pasados; **Apuestas** arma una boleta confirmable, declara la banca, muestra el stake que correspondería y el estado del gate, y expone la matemática real de una combinada; **Seguimiento** conserva separado el forward test hipotético con el CLV y su IC95%; **Backtest** muestra la grilla de umbrales, el bankroll, el staking, el drawdown y los segmentos que produce `ufc/modelo/backtest.py`. Cartelera y Matchup mantienen el detalle del modelo y mercado. | — |
| `test_app.py` | `python test_app.py`. Además de simetría, fuentes, ledger y UI, verifica taxonomía/settlement, formatos antiguos, bootstrap clusterizado, manifiesto con hashes, no-bet por defecto, auditoría de revisiones y navegación real a Pasados → Ganadores. `check_devig` (los cuatro métodos suman 1 y son antisimétricos), `check_staking` (Kelly contra valores a mano, el máximo del crecimiento cae exactamente en `f*`, la ruina es monótona en la fracción, el vig se compone en la combinada), `check_pool` (antisimetría exacta y piso en el mercado) y `check_gate` (cerrado sin muestra; abre solo con IC limpio **y** el `n` preregistrado). Los ocho `check_ia_*` cubren la capa de IA con fixtures: el prompt completo sin red, los cinco caminos de degradación de `validar()`, la idempotencia del store, que el CSV versionado no lleve texto libre, que la IA se mida en `aciertos()` pero **no** vote el consenso humano, y que `evaluar()` deje `data/ledger.csv` byte a byte igual. No toca la red. | — |

### De dónde salen las peleas que todavía no ocurrieron

ufcstats.com metió un challenge JS anti-bot en su página de eventos futuros, y el dataset
de odds (`shortlikeafox/ultimate_ufc_dataset`) está congelado desde el 2026-04-01. La API
pública de ESPN es la que quedó viva: JSON, sin API key. Trae el matchup pero **no la
cuota**.

La cuota la pone `betano.py`. Betano no tiene API pública, pero tampoco hace falta: el
sitio deja el estado entero de la página embebido en el HTML como
`window["initial_state"]`, con nombres y cuotas adentro, así que alcanza un GET por
cartelera (cada evento de UFC es una "liga" separada para ellos). Sigue siendo scraping y
se va a romper: cuando pase, `python -m ufc.datos.betano` baja el estado crudo y lista lo que
parsea. La app lo cachea 30 minutos, que es también el rate limit contra el sitio.

Dos límites que no son bugs: Betano abre mercado unos días antes del evento, así que las
carteleras lejanas aparecen sin cuota; y el matcheo de nombres es exacto tras normalizar
acentos, con un fallback difuso (`difflib`) para los `Jr.` y los nombres compuestos. Sobre
la cartelera del 2026-08-01 matcheó 14/14, incluidos `Uroš Medić`, `Mateusz Rębecki` y
`Stephanie Luciano` → `Stephanie Bruna Luciano`. Si Betano no responde, la cartelera se
muestra igual, sin cuota ni nivel de coincidencia.

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
nivel de coincidencia descriptivo que muestra la app; no es confianza económica.

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
de lo que el IC sugiere. Por eso la app **ya no lo marca como candidata**: conserva el EV
solo como seguimiento experimental y el default es “sin apuesta”. El
resto es la trampa: donde el EV del modelo se ve más lindo (porque se aparta del mercado)
es exactamente donde pierde 7%. La vara que decide de verdad es el forward test del
**Historial**: si las candidatas no muestran CLV positivo con datos reales, se retira la
etiqueta.

Subir el umbral de EV no ayuda — está medido que filtrar por EV alto empeora. Por eso la
regla es EV > 0 y nada más.

### El backtest completo: 21 umbrales, ninguno con ventaja

`python -m ufc.modelo.backtest` barre la grilla entera de umbrales de ventaja
(`p_modelo − p_mercado`, en puntos de probabilidad) sobre 6561 lados con cuota real,
20 folds de rolling-origin, IC95% clusterizado por evento:

| umbral | apuestas | ROI | IC95% | EV que prometía el modelo |
|---|---|---|---|---|
| 0% | 6561 | −5.29% | [−8.64%, −1.89%] | +37% |
| 5% | 5006 | −6.70% | [−10.69%, −2.83%] | +48% |
| 10% | 3523 | −7.98% | [−13.04%, −3.07%] | +62% |
| 15% | 2248 | −8.39% | [−15.30%, −1.57%] | +80% |
| 20% | 1299 | −1.31% | [−10.96%, +8.04%] | +100% |

**Ningún umbral deja el IC95% del ROI por encima de cero.** Hasta el 16% el intervalo está
enteramente **por debajo** de cero: no es que no se pueda concluir, es que está medido que
pierde. Del 17% en adelante el intervalo cruza cero solo porque quedan pocas apuestas.

La columna que explica todo es la última. El modelo cree que gana +37% por apuesta y
pierde 5%; cuanto más alto el umbral, más grande la promesa y peor el resultado. Eso no es
mala suerte: es una probabilidad sesgada **contra el mercado**. El modelo ciego está
comprimido hacia 50%, y como `EV = cuota·p − 1 ≈ p_modelo/p_mercado − 1`, esa compresión
sola pone al no favorito arriba en todas las peleas.

Calibrar no lo arregla, y está medido: con calibración prequencial (cada fold ajustado con
los anteriores), ni Platt ni la isotónica le ganan al modelo crudo — la isotónica
directamente **se descarta** (+0.0077 de log loss, IC95% [+0.0010, +0.0159]). El ECE del
modelo crudo ya es 0.0139. Calibrar corrige el sesgo contra la **realidad**, y ese el
modelo no lo tiene; lo que tiene es sesgo contra el **mercado**, que es otro problema.
Por eso `train.py` no serializa ningún calibrador y `p_a_cal == p_a`.

La pestaña **Backtest** de la app muestra las curvas, el bankroll, el drawdown y los
segmentos. Los segmentos hay que leerlos con cuidado: son 34 niveles mirados sobre los
mismos datos, así que se esperan ~2 “hallazgos” de puro azar. Que las peleas de título
den +9% con n=310 es una hipótesis para testear hacia adelante, no un segmento rentable.

### CLV histórico: no es computable

`data/raw/ufc_odds.csv` trae **una sola foto** de cuota por pelea (7177 filas, un
`R_odds`/`B_odds`), no apertura y cierre. Sin dos precios no hay CLV, así que el backtest
no lo reporta. El CLV solo existe **hacia adelante**, en el Historial, comparando la cuota
congelada contra los ticks de `betano_hist.csv`.

Y esas cuotas históricas son de cierre o consenso, mientras que la app congela en apertura
en una sola casa: el vig mediano del archivo es 3.70% contra 5.88% de Betano en vivo. El
backtest acota la expectativa, no reproduce el juego que juega la app.

## De la probabilidad al dinero

Documento completo: **[`docs/consultoria-apuestas.md`](docs/consultoria-apuestas.md)**.

Todo lo de arriba mide la **probabilidad**. Esta sección es lo que hace falta para que una
probabilidad se convierta en plata, que es un problema distinto y hasta ahora no estaba
implementado: no había staking, ni banca, ni cota inferior de EV, ni test estadístico
sobre el CLV, ni una regla que dijera cuándo se puede apostar.

### La cuenta que reorganiza el proyecto

Una apuesta flat a cuota ~2.00 tiene desviación `σ = q·√(p(1−p)) ≈ 1.0`. Para distinguir
una media de cero con potencia 80%: `n = ((1.96 + 0.84)·σ/e)²`.

| detectar | σ | apuestas |
|---|---|---|
| ROI de +2% | 1.00 | **19.623** |
| ROI de +5% | 1.00 | 3.140 |
| **CLV medio de +2%** | **0.04** | **32** |

A ~500 apuestas por año, verificar un ROI del 2% son **cuarenta años**. Por eso el
criterio de este proyecto es el CLV y no el ROI: no por elegante, porque es el único que
converge en tiempo humano. Está en `apuesta.n_para_detectar` y ahora aparece como columna
en la grilla del backtest — los umbrales con el ROI menos malo son justo los que más lejos
están de poder concluir algo.

### Desvigueo: los cuatro métodos, medidos

`1/cuota` no es una probabilidad. Cómo se reparte el margen mueve 1–2 puntos justo en el
rango de favoritos, que es donde se decide el EV. Sobre 6901 peleas con resultado:

| método | log loss | vs power | veredicto |
|---|---|---|---|
| **power** | **0.6075** | — | **campeón** |
| odds_ratio | 0.6077 | +0.0002 [+0.0000, +0.0004] | se descarta |
| Shin | 0.6078 | +0.0003 [+0.0000, +0.0005] | se descarta |
| proporcional | 0.6084 | +0.0009 [+0.0003, +0.0015] | se descarta |

Power se queda, ahora por evidencia. Su ventaja vive entera en los favoritos de >75%
(0.4482 contra 0.4519 del proporcional). `python -m ufc.modelo.devig`.

### El modelo no aporta sobre el precio, y ahora está cerrado

El modelo está comprimido hacia 0.5 contra el mercado, y como `EV ≈ p_modelo/p_mercado −
1`, esa compresión sola pone al underdog arriba en todas las peleas. La herramienta
correcta no es un calibrador sino un **pool logit** de dos parámetros,
`logit(p) = w_mkt·logit(p_mkt) + w_mod·logit(p_mod)`, sin intercepto (conserva la
antisimetría) y prequencial:

| | log loss |
|---|---|
| modelo solo | 0.6608 |
| **mercado solo** | **0.6109** |
| pool (w_mkt 0.970, w_mod 0.214) | 0.6106 |

**pool vs mercado: −0.0003, IC95% [−0.0018, +0.0014] — no concluyente.** El resultado
previo (odds como feature 29 del HistGB) admitía la excusa de la varianza; un pool de dos
parámetros es el estimador de mínima varianza para la misma pregunta y no encuentra nada.
**El EV tiene que venir del precio, no del modelo.** `python -m ufc.modelo.pool`.

### Staking: no crea ventaja, decide la supervivencia

Las **mismas** apuestas del backtest, banca inicial 100:

| staking | banca final | peor caída | manda el tope | P(perder la mitad) |
|---|---|---|---|---|
| flat 1 unidad | **−261,72** | −394,0 u | — | — |
| ¼ Kelly | 28,07 | −74,9% | 85% | 0,8% |
| ½ Kelly | 29,59 | −73,7% | 90% | 12,5% |
| Kelly completo | 30,04 | −73,3% | 93% | 50,0% |

El flat de 1 unidad sobre una banca de 100 **no es conservador**: termina en −261,72
porque un stake fijo es un porcentaje creciente de lo que va quedando. Y las tres
fracciones de Kelly dan casi lo mismo porque el tope duro del 1% las corta en el 85–93% de
las apuestas: con un modelo que cree tener +38% de EV, quien dimensiona no es Kelly, es el
tope — y ésa es exactamente su defensa.

### El gate: por qué la app no apuesta

`config/gate.json` es la regla **preregistrada**, escrita antes de mirar los resultados:
probabilidad de consenso multi-casa, decidir sobre la cota inferior, mejor precio
ejecutable, EV mínimo 2%, sin combinadas, ¼ Kelly con tope 1% por apuesta y 3% por evento.
Para abrirse pide **100 apuestas con el IC95% del CLV económico por encima de cero**.

`ufc/modelo/gate.py` la evalúa contra el ledger real. `predict.APUESTAS_AUTOMATICAS` dejó
de ser una constante y pasó a ser `predict.apuestas_automaticas()`: da la misma respuesta
—no— pero con motivo, `n` y cuánto falta. Una constante no se puede refutar.

Estado hoy: 4 lados seguidos con cierre, CLV económico **−14,0%** IC95% [−24,1%, −9,3%],
0 de 4 le ganaron al cierre. No prueba nada con n=4, y el gate lo dice así.

### CLV económico, y un bug que lo contaminaba

El `clv` viejo compara dos cuotas. El nuevo `ev_al_cierre = p_justa_del_cierre × cuota_tomada − 1`
es el mismo número en unidades económicas, que es lo que se compara contra cero, y va con
IC95% bootstrap clusterizado por evento.

Además: el "cierre" se elegía con todo tick anterior a `fecha_evento + 1 día`, que puede
colar cuotas **en vivo** o posteriores al combate. ESPN manda la hora UTC de inicio y
`cartelera.py` la descartaba con `[:10]`. Ahora se conserva (`inicio_utc`, columna
aditiva) y el corte es estricto, con fallback a la regla vieja para las filas ya
registradas.

### Mejor precio multi-casa: la única vía que no exige ganarle a nadie

`oddsapi.py` toma el consenso desvigueado de varias casas y busca dónde una paga por
encima. Tres correcciones que deciden si eso es ventaja o espejismo:

- **La casa del mejor precio no entra al consenso de ese lado** (independencia: la
  referencia no puede contener el precio que juzga). Ojo con la dirección: excluirla
  **sube** el EV medido, no lo baja.
- **La dispersión entre casas se descuenta** (`apuesta.p_conservadora`). Ésta sí es la
  defensa contra el espejismo: tomar el máximo de N precios da EV positivo con frecuencia
  aunque los precios sean puro ruido alrededor de la misma probabilidad.
- **Precios stale afuera**: una casa que no actualiza hace 12 horas no está cotizando.

`data/odds_hist.csv` archiva cada tick por casa, append-only. Es dato irreproducible hacia
atrás — el motivo por el que el CLV histórico no existe hoy.

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

`python -m ufc.modelo.train probar` mide bloques de features candidatas contra el modelo actual con
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
que apostar su EV positivo pierde en conjunto. El tramo de mayor coincidencia mantiene un
IC95% que cruza cero y fue elegido a posteriori; la app no genera candidatas automáticas.
Un modelo bien calibrado te dice cuándo
tenés razón, no cuándo cobrás — apostar favoritos claros no es gratis, el sesgo
favorito-longshot hace que las cuotas bajas estén bien pagadas.

## Descargo de responsabilidad

Esto es un proyecto de investigación y aprendizaje. **No es asesoramiento financiero ni
de apuestas.** La app no apuesta sola, no recomienda apostar y su default es "sin
apuesta": todo lo que queda registrado lo confirma una persona.

Está medido en este mismo repo que apostar las señales del modelo **pierde dinero**
(−4.59% de ROI, IC95% enteramente bajo cero) y que ningún umbral de los 21 probados
muestra ventaja. Cualquiera que use esto para apostar lo hace bajo su propia
responsabilidad y arriesga plata que puede perder. Las apuestas son para mayores de edad
y su legalidad depende de la jurisdicción. Si el juego dejó de ser un pasatiempo, buscá
ayuda profesional.

## Contribuir

Issues y PRs bienvenidos. Dos condiciones que vienen del propio proyecto:

- **`python test_app.py` tiene que pasar** (no toca la red).
- **Una feature nueva se acepta con evidencia, no con intuición.** El protocolo es el
  rolling-origin de 20 folds de `python -m ufc.modelo.train probar`, y el IC95% del delta
  pareado no puede tocar cero. La sección [Por qué no sirve tocar los
  hiperparámetros](#por-qué-no-sirve-tocar-los-hiperparámetros-ni-casi-nada-del-modelo)
  explica por qué la vara es esa: con la ventana de datos que hay, un "mejora 0.003" es
  ruido de 1.1σ.

Está medido que el pozo de ufcstats está seco. Lo que mueve la aguja es **fuente nueva**,
no features nuevas sobre las mismas columnas.

## Licencias

- **Código:** [MIT](LICENSE).
- **Datos versionados en `data/`:** [CC BY 4.0](LICENSE-DATA) — son registros producidos
  por este proyecto.

Los datos de terceros **no se redistribuyen** acá: el repo los descarga, cada fuente
conserva sus términos. El detalle está en [`LICENSE-DATA`](LICENSE-DATA).

## Fuentes

- [ufcstats.com](http://ufcstats.com), vía [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats) — estadísticas de pelea.
- [`shortlikeafox/ultimate_ufc_dataset`](https://github.com/shortlikeafox/ultimate_ufc_dataset) — odds históricas 2010+ (congelado desde 2026-04-01).
- [Wikipedia](https://en.wikipedia.org/w/api.php) — reemplazos y peso no dado (CC BY-SA 4.0 en origen).
- [Sherdog](https://www.sherdog.com) — récord pre-UFC del circuito regional.
- [API pública de ESPN](https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard) — carteleras anunciadas.
- [Betano](https://lat.betano.com) y [The Odds API](https://the-odds-api.com) — cuotas en vivo.

No afiliado a UFC, Zuffa, ni a ninguna casa de apuestas.
