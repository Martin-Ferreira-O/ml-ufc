# Posibles mejoras del predictor UFC

> Auditoría realizada el 2 de agosto de 2026 sobre el código y los datos locales.
> Este documento es únicamente un análisis: no implementa ninguna mejora ni modifica
> modelos, fuentes, reglas de apuesta o interfaz. Apostar nunca puede hacerse “seguro”;
> como máximo puede medirse mejor la incertidumbre, exigir una ventaja más sólida y
> limitar el riesgo.

## Resumen ejecutivo

El programa tiene una base metodológica bastante mejor que la de muchos predictores
públicos: construye variables *point-in-time*, espeja cada pelea, usa cortes temporales,
evalúa probabilidades con log loss/Brier y compara contra el mercado. El problema
principal ya no parece ser la elección de hiperparámetros. Los mayores saltos posibles
se resumen en ocho conclusiones:

1. **Corregir primero la medición.** El test se ha consultado durante varias rondas de
   desarrollo, se han probado muchas ideas sobre los mismos 20 folds y el bootstrap trata
   las peleas como independientes. Antes de creer otra mejora pequeña hay que reservar un
   holdout realmente intacto, ajustar por pruebas múltiples y remuestrear por evento y/o
   peleador.
2. **Guardar exactamente qué se sabía y qué cuota podía tomarse en cada instante.** El
   modelo histórico conoce reemplazos y fallos de peso aunque, al simular una apuesta de
   apertura, esa noticia quizá aún no existía. Los umbrales de confianza fueron estimados
   con líneas probablemente de cierre y luego se aplican a cuotas tempranas de una sola
   casa. Ese cambio de horizonte invalida una interpretación directa del backtest.
3. **Agregar información nueva, no más capacidad al mismo dataset.** La mejor oportunidad
   para ganador es completar debutantes y peleadores con poca experiencia UFC mediante
   calidad de oposición regional, ratings dinámicos, resultados por round, cambios de
   peso y circunstancias verificadas de la pelea. Probar otro GBM sobre los mismos
   promedios difícilmente cierre la brecha con el mercado.
4. **Modelar tiempo y finalización conjuntamente.** Hoy no existe modelo de over/under de
   rounds. El enfoque natural es supervivencia discreta con riesgos competitivos
   (KO/TKO, sumisión y llegada al límite), que produce probabilidades coherentes para
   over 1.5/2.5/4.5, “llega a decisión” y tipo de finalización.
5. **Aprovechar las props de mercado sin re-aprenderlas peor.** En la medición ya hecha,
   las props históricas de método obtuvieron log loss 0.9582 frente a aproximadamente
   1.017 del modelo propio. Deben actuar como prior o *offset* y, como alternativa, usarse
   crudas; no como una columna más de un GBM que las degradó.
6. **Arreglar identidad y serving.** Hay ocho nombres homónimos cuyos historiales se
   mezclan y los debutantes futuros no entran en `fighter_state.csv`, aunque Sherdog ya
   aporte información regional. IDs estables y un estado prospectivo son mejoras de
   corrección, no experimentos cosméticos.
7. **No usar el 80–90% de acierto como vara principal.** Ese porcentaje se consigue
   seleccionando pocos favoritos grandes, incluso sin tener ventaja. La vara correcta es
   mejora probabilística contra el mercado, calibración, CLV y ROI ejecutable, todos con
   incertidumbre y cobertura explícitas.
8. **No habilitar una “candidata” con evidencia actual.** El tramo que la app llama
   candidato fue elegido a posteriori y su IC95% cruza cero. El forward test local tiene
   solo 7 peleas resueltas y 2 candidatas; todavía no valida nada.

La prioridad recomendada es: **integridad temporal y de cuotas → etiquetas/IDs → modelo
de rounds y finalización → información regional y de circunstancias → nuevos modelos →
gestión de stake**.

## 1. Estado actual re-derivado

### 1.1 Foto de los datos

Durante esta auditoría el pipeline se actualizó con la cartelera del 1 de agosto de 2026.
La foto final usada aquí es:

- `ufc_fight_results.csv`: 8.810 filas, de las cuales 8.655 tienen ganador W/L claro.
- `features.csv`: 17.306 filas espejadas, equivalentes a 8.653 peleas. Las dos peleas de
  diferencia no tienen una fecha de evento utilizable.
- `fighter_state.csv`: 2.726 peleadores con al menos una pelea en el historial procesado.
- 28 variables para ganador: 22 de rendimiento/biometría, 2 circunstancias de la pelea y
  4 del método del récord pre-UFC.
- El 25,8% de las peleas históricas tiene al menos un debutante UFC (`n=2.230`).
- Ocho nombres corresponden a dos personas distintas y hoy se mezclan por nombre.
- Las odds históricas llegan hasta el 28 de marzo de 2026; las tres props de método están
  completas en 5.373 de las 8.653 peleas del archivo de features.
- Hay 56 picks humanas guardadas, pero ninguna figura todavía como revisada; por tanto no
  existe ranking humano válido en la foto auditada.
- El ledger tiene 16 predicciones, solo 7 con resultado y 2 candidatas. El 85,7% de
  acierto que aparece en esas siete observaciones equivale a 6/7 y tiene un intervalo de
  Wilson 95% aproximado de **48,7% a 97,4%**: no permite concluir que el sistema sea de
  85%.

### 1.2 Rendimiento reproducido del ganador

Reentrenando en memoria, sin guardar artefactos, con el corte temporal actual:

| período | peleas | log loss | Brier | accuracy |
|---|---:|---:|---:|---:|
| train | 6.623 | 0.6283 | 0.2189 | 65,38% |
| validación | 1.001 | 0.6516 | 0.2299 | 63,24% |
| test desde 2024-08-01 | 1.029 | **0.6335** | **0.2214** | **64,63%** |

Esto confirma la conclusión previa del proyecto: el modelo no es inútil, pero tampoco
está cerca de 80–90% sobre una cartelera completa. En el rolling-origin documentado, la
línea de mercado obtuvo log loss 0.6107 frente a 0.6561 del modelo independiente. El
mercado contiene información que el historial deportivo actual no captura.

### 1.3 Rendimiento reproducido del tipo de finalización

El modelo actual de KO/TKO, sumisión o decisión solo usa peso, género y si se programan
cinco rounds. En el test actual de 1.029 peleas:

| modelo | log loss | accuracy |
|---|---:|---:|
| contexto actual | **1.0108** | 50,83% |
| frecuencia histórica | 1.0198 | — |

La mejora sobre el base rate existe, pero es pequeña. La evidencia histórica del propio
repo es mucho más clara para las props del mercado: 0.9582 de log loss. Además, hoy 98
`TKO - Doctor's Stoppage` y 23 DQ con ganador se etiquetan como “decisión”, porque todo lo
que no contiene literalmente `KO/TKO` o `Submission` cae en `dec`. Son 121 etiquetas que
deben revisarse según las reglas de liquidación de la casa.

### 1.4 Over/under de rounds

No hay un modelo. Sí están casi todos los ingredientes para construir el target:
`ROUND`, `TIME`, `TIME FORMAT`, método y cantidad programada de rounds. En las peleas de
formatos modernos o equivalentes hay 7.676 a tres rounds y 770 a cinco rounds. Sus tasas
históricas crudas son:

| formato | over 1.5 | over 2.5 | over 4.5 | decisión |
|---|---:|---:|---:|---:|
| 3 rounds | 65,23% | 52,33% | — | 48,23% |
| 5 rounds | 69,61% | 55,58% | 41,17% | 38,57% |

Son solo bases descriptivas, no probabilidades para apostar. Demuestran que el target es
construible y que programar tres o cinco rounds cambia la distribución.

### 1.5 Fortalezas que conviene conservar

- Actualización del estado solo después de emitir la fila de una pelea.
- Peleas espejadas y serving simétrico.
- Separación entre el modelo independiente y el mercado.
- Log loss, Brier y accuracy, en vez de accuracy sola.
- Rolling-origin y comparación pareada, mejor que un split aleatorio.
- Reentrenamiento final con todo el pasado, después de medir sobre datos no vistos.
- Fuente regional con fecha por combate y validación de identidad.
- Registro append-only de cuotas y un ledger prospectivo, aunque aún requieran ajustes.
- Reconocimiento explícito de que discrepancia grande con el mercado no equivale a valor.

Estas piezas son una buena base; las propuestas siguientes deben reforzarlas, no
reemplazarlas sin evidencia.

## 2. Por qué “acierta 80–90%” no es una comparación suficiente

### 2.1 Accuracy alta puede ser solo selección de favoritos

Sobre las cuotas históricas del propio proyecto, el favorito del mercado muestra:

| probabilidad justa mínima del favorito | peleas | cobertura de peleas con cuota | acierto |
|---:|---:|---:|---:|
| 50% | 6.561 | 100,0% | 66,6% |
| 60% | 4.551 | 69,4% | 71,9% |
| 70% | 2.303 | 35,1% | 79,9% |
| 75% | 1.471 | 22,4% | 84,1% |
| 80% | 775 | 11,8% | **87,0%** |
| 85% | 292 | 4,5% | **91,1%** |
| 90% | 69 | 1,1% | **95,7%** |

Por tanto, alguien puede publicar 87–91% escogiendo solo favoritos muy caros. No hace
falta que sepa más que la casa y el porcentaje no dice si ganó dinero. A cuota 1.10 se
necesita acertar más de 90,91% solo para quedar en cero; un 90% de acierto perdería.

También hay mucha varianza por cartelera. Si la capacidad real fuera 65%, acertar al
menos 12 de 14 peleas —85,7%— ocurre aproximadamente en el 8,4% de las carteleras. A lo
largo de decenas de eventos aparecerán varias capturas de 12-2 por puro ruido. Aciertos
publicados sin el archivo completo, incluidos errores, abstenciones y hora del pick, no
son auditables.

### 2.2 Qué exigir a cualquier predictor humano

Para afirmar que una persona aporta información deben guardarse, sin posibilidad de
edición silenciosa:

- todas sus picks, no solo las ganadoras;
- timestamp anterior al evento y registro de cualquier revisión;
- mercado exacto: ganador, total de rounds, método, ganador por método, etc.;
- cuota realmente disponible al publicar y casa donde podía tomarse;
- stake, si publica unidades;
- abstenciones y número total de peleas cubiertas;
- confianza probabilística, si la da;
- resultado y reglas de liquidación aplicadas;
- comparación contra el favorito y la probabilidad de mercado en ese mismo instante;
- ROI, CLV, Brier/log loss y cobertura, además de accuracy.

Un predictor que acierta 85% escogiendo selecciones cuya probabilidad de mercado media
era 88% está rindiendo peor que el precio. Uno que acierta 62% con cuotas 2.00 puede estar
aportando muchísimo más. El ranking actual `(aciertos + 5) / (total + 10)` reduce el daño
de muestras pequeñas, pero no ajusta dificultad, cuota, selección de favoritos, momento
del pick ni correlación entre tipsters.

## 3. Correcciones metodológicas previas a buscar más señal

### 3.1 Separar el horizonte de predicción

No existe una sola predicción “prepelea”. Son problemas distintos:

- **Apertura / T-7 días:** aún no se conoce el pesaje y puede cambiar el rival.
- **T-24 horas:** se conocen peso real, fallos de peso y buena parte de las noticias.
- **T-1 hora / cierre:** el mercado ya incorporó casi toda la información pública.

El dataset histórico incluye `peso_no_dado` y `reemplazo` en cada pelea. Si se usa esa fila
para simular una apuesta de apertura, puede estar filtrando información que apareció
después de la cuota. Cada dato circunstancial necesita `known_at`; cada forecast necesita
`as_of`. Una feature solo puede entrar si `known_at <= as_of`.

**Mejora propuesta:** entrenar y evaluar versiones por horizonte, o una sola versión con
la disponibilidad explícita. La comparación siempre debe usar la cuota observable en ese
mismo instante. Este cambio puede empeorar el backtest aparente y, precisamente por eso,
lo vuelve creíble.

### 3.2 Reservar un holdout realmente intacto

El “test de los últimos dos años” se ha reportado tras varias rondas y su fecha se mueve
al entrar una cartelera nueva. Aunque no se ajuste directamente a él, verlo repetidamente
genera adaptación humana.

**Mejora propuesta:**

1. Congelar un bloque histórico como validación de desarrollo.
2. Mantener un bloque posterior cifrado/lógicamente inaccesible para una sola evaluación.
3. Tras elegir un campeón, pasar exclusivamente a predicción prospectiva.
4. No volver a cambiar umbrales con resultados del forward test; una versión nueva inicia
   otra cohorte.

### 3.3 Corregir dependencia en los intervalos de confianza

El bootstrap actual remuestrea peleas individuales. Peleas del mismo evento comparten
ubicación, jueces, noticias, mercado y condiciones. El mismo peleador reaparece muchas
veces. Tratar todo como i.i.d. tiende a producir intervalos demasiado estrechos.

**Mejora propuesta:** reportar al menos bootstrap por evento y moving-block bootstrap por
fecha; como sensibilidad, un bootstrap multi-vía por evento y peleador. Una mejora solo
debería pasar si conserva signo bajo estos esquemas, no solo en el bootstrap individual.

### 3.4 Ajustar por pruebas múltiples

Ya se probaron muchos bloques, hiperparámetros, cortes y tramos sobre los mismos folds.
Pedir que cada IC95% no toque cero controla el error de una prueba, no el de toda la
búsqueda. La literatura de backtesting muestra que probar muchas señales y publicar la
mejor agrava seriamente el sesgo de sobreajuste ([Novy-Marx, NBER 2015](https://www.nber.org/papers/w21329)).

**Mejora propuesta:** preregistrar cada familia de experimentos, conservar un registro de
todas las pruebas, aplicar Holm/Benjamini-Hochberg o un bootstrap de máximo estadístico, y
exigir confirmación en holdout/forward test. Las mejoras ya aceptadas de Wikipedia y
Sherdog deberían revalidarse con clustering y corrección por multiplicidad; no porque
sean malas, sino porque sus deltas son pequeños.

### 3.5 Versionar datos, modelo y métricas juntos

Los CSV diarios cambian sin quedar ligados al pickle. Durante esta auditoría los datos se
actualizaron y el README quedó describiendo el evento anterior. `model.pkl` solo guarda
estimadores y columnas.

**Mejora propuesta:** un manifiesto por entrenamiento con hashes, fecha máxima, cantidad
de filas, commit de código, versión de dependencias, fuente/fecha de cada odds snapshot,
parámetros, métricas y semilla. El bundle debe incluir ese manifiesto y escribirse de forma
atómica. Así una probabilidad puede reconstruirse exactamente.

### 3.6 Alinear todas las etiquetas con la liquidación real

Ganador, método y total no siempre se liquidan igual ante DQ, no contest, doctor stoppage,
technical decision, empate o cambio de rival. Las reglas varían por casa y mercado; la
propia ayuda de Betano advierte que hay que revisar la descripción específica del mercado.

**Mejora propuesta:** una tabla versionada de reglas por casa/mercado y un motor único de
settlement compartido por creación del target, backtest y ledger. No debe decidirse por
un simple `str.contains`.

### 3.7 Medir cobertura y abstención

Un modelo puede aumentar el acierto rechazando peleas difíciles. Debe mostrarse siempre la
curva **cobertura → log loss/accuracy/ROI**, no el porcentaje de las elegidas aislado.
También se necesita una categoría real de “sin apuesta” cuando el intervalo de EV cruza
cero, hay datos desconocidos o la cuota está vieja.

### 3.8 Eliminar el cambio de dominio entre cierre y serving

Los thresholds de `CONFIANZA`, el tramo `APOSTABLE` y el modelo `con_odds` fueron medidos
principalmente con la columna histórica `R_odds/B_odds`, que parece representar una línea
tardía o de cierre sin timestamp verificable. En producción reciben la cuota actual de
Betano, a veces días antes y de una sola casa. No son la misma distribución.

**Mejora propuesta:** hasta contar con snapshots equivalentes, no transferir métricas de
cierre a apertura. Entrenar y validar por fuente/horizonte; serializar junto al modelo los
dominios para los que fue validado. Si llega una cuota de otro dominio, mostrar mercado y
modelo independiente, pero no una etiqueta de confianza económica heredada.

### 3.9 Fecha futura exacta y tres estados para circunstancias

`predict.predict()` recalcula edad y descanso con “hoy”, no con la fecha programada del
evento. La diferencia suele ser pequeña, pero es un skew evitable. Más importante: los
checkboxes de reemplazo/peso parten en falso, lo que equivale a “confirmado que no ocurrió”
aunque el usuario quizá simplemente no lo verificó.

**Mejora propuesta:** pasar siempre `event_date` y usar valores `sí/no/desconocido` con
fuente y timestamp. Desconocido debe seguir siendo NaN o generar escenarios, nunca cero
por conveniencia.

### 3.10 Denominadores observados y formatos antiguos

El estado suma los minutos completos de una pelea aunque alguna estadística esté ausente;
el numerador se salta el NaN. Eso convierte minutos sin observación en rendimiento cero.
En la foto auditada afecta aproximadamente 0,12% de fighter-fights para SIG/TD y 2,17%
para control, concentrados en los primeros años, por lo que no es la gran mejora de
accuracy, pero sí una corrección.

**Mejora propuesta:** cada tasa mantiene su propia exposición observada —minutos con
golpeo, intentos TD, rounds con control— y un indicador de cobertura. Parsear también los
198 formatos antiguos/no-limit en vez de aplicar siempre cinco minutos por round, o
excluirlos del modelo moderno con una decisión explícita.

## 4. Mejoras para predecir al ganador

### 4.1 IDs estables para peleadores, peleas y eventos — prioridad crítica

**Problema actual:** las uniones usan nombres. Ocho nombres duplicados mezclan carreras;
aliases explican parte del 5% de odds no matcheadas; un rematch puede colisionar en el
histórico de cuotas porque el ledger agrupa por par de nombres.

**Propuesta:** usar como claves primarias el ID de las URLs de `fighter-details` y
`fight-details`, más IDs de ESPN/The Odds API y una tabla de equivalencias auditable.
El scraper origen ya publica URLs estables en los CSV de peleadores y peleas
([Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats)). Para resolver
homónimos por pelea habrá que conservar también los dos fighter IDs de la página de
detalle, dato que el mirror actual no incluye en la fila de resultados.

**Impacto esperado:** pequeño en la métrica global, crítico en seguridad individual. Una
predicción con historia de otra persona no debería mostrarse.

**Gate:** cero colisiones de identidad; rematches y aliases con tests; todo registro
histórico enlazado a un `fight_id` único.

### 4.2 Servir debutantes de verdad — prioridad crítica

**Problema actual:** Sherdog aporta récord pre-UFC al entrenamiento, pero
`fighter_state.csv` se construye solo desde quienes ya aparecen en resultados UFC. Un
debutante anunciado no existe en el índice y `cartelera.predecir` devuelve “sin historial
en UFC”. Es decir, la principal fuente agregada para debutantes todavía no llega al caso
de uso futuro.

**Propuesta:** crear un estado prospectivo para todo peleador anunciado: Elo UFC con prior
e incertidumbre altos, `n_fights=0`, biometría si existe y features regionales. Resolver y
validar su ficha Sherdog al aparecer en ESPN, no después de que debute.

**Impacto esperado:** cobertura muy alta; el 25,8% de las peleas históricas tiene un
debutante y las carteleras futuras pierden hoy muchas predicciones.

**Gate:** predicción simétrica para debut vs veterano y debut vs debut, sin convertir
“dato desconocido” en cero; evaluación separada de esa cohorte.

### 4.3 Calidad de oposición regional, no solo conteos — potencial alto

Las cuatro variables pre-UFC actuales resumen cómo terminaban las peleas, pero no contra
quién. Un 10-0 contra oposición débil no equivale a 10-0 contra futuros peleadores UFC.

**Propuesta:** construir un grafo de peleas regionales con fecha y calcular, siempre
point-in-time:

- rating del rival antes de cada pelea;
- fuerza media y máxima de calendario;
- rendimiento observado menos rendimiento esperado;
- nivel de promoción/circuito y calidad de sus oponentes;
- recencia, edad al competir y cambios de categoría;
- resultado y método ajustados por calidad del rival;
- incertidumbre por conectividad del grafo con UFC.

Un modelo dinámico de comparaciones pareadas puede aprovechar este grafo mejor que contar
victorias. La mejora debe medirse especialmente en peleas con `n_fights_min <= 2`.

**Riesgos:** scraping mucho mayor, aliases regionales, records incompletos y sesgo de
supervivencia. Un rating retrospectivo que usa la carrera futura del rival produciría
leakage; el rival debe tener solo su rating disponible en esa fecha.

### 4.4 Shrinkage bayesiano de tasas — potencial alto y coste medio

SLpM, defensa, TD y finish rate se estiman con una o dos peleas para gran parte del
roster. El modelo recibe el promedio crudo y el número de peleas, pero eso no garantiza
que aprenda la incertidumbre correcta.

**Propuesta:** suavizar cada tasa hacia un prior de división/era con exposición real
(minutos, intentos o peleas). Conservar media posterior y desviación posterior. Ejemplos:

- Beta-Binomial para precisión/defensa de golpeo y derribo;
- Gamma-Poisson o modelo jerárquico para KD, TD y sumisiones por minuto;
- Beta-Binomial para finalizaciones;
- priors separados por peso, género, era y UFC/no UFC.

Esto evita que 1 de 1 TD y 20 de 20 TD parezcan igual de ciertos. La desviación posterior
también permite encoger la probabilidad final hacia 0.5 cuando ambos lados son inciertos.

### 4.5 Rendimiento ajustado por el rival — potencial alto

Los promedios actuales mezclan habilidad propia con dificultad de oposición. `avg_opp_elo`
solo resume la calidad global, no cuánto golpea o defiende cada rival.

**Propuesta:** para cada pelea, calcular residuales respecto de lo que el rival solía
permitir/producir antes de ese día:

- golpeo del peleador menos golpeo esperado contra esa defensa;
- defensa observada contra el volumen esperado del rival;
- TD/control/KO/sumisión por encima o debajo de expectativa;
- ratings separados de striking, wrestling, grappling, durabilidad y cardio.

Luego acumular esos residuales con shrinkage. Esta formulación capta calidad y estilos sin
depender de interacciones manuales simples que ya resultaron no concluyentes.

### 4.6 Rating dinámico con incertidumbre — potencial medio/alto

El Elo actual usa K=32 y no expresa cuánto se sabe del rating. Un debutante 1500 y un
veterano estable 1500 parecen iguales. Glicko extiende Elo con una desviación que aumenta
con inactividad y disminuye al observar peleas
([descripción técnica de Glicko](https://www.glicko.net/glicko/glicko.pdf)). Los modelos
Bradley-Terry dinámicos permiten habilidades que cambian con el tiempo
([Cattelan, Varin y Firth, 2013](https://wrap.warwick.ac.uk/id/eprint/54660/)).

**Propuestas a comparar:**

- Glicko/Glicko-2 por división y global jerárquico;
- Bradley-Terry dinámico con random walk por peleador;
- ratings separados por método/área;
- actualización proporcional a información del resultado, con decisión cerrada menos
  concluyente que una finalización dominante, pero sin usar una regla arbitraria;
- incertidumbre creciente con layoff y cambio de categoría.

El K por finish ya fue probado y no pasó; eso no descarta un modelo probabilístico de
incertidumbre, que es una hipótesis distinta.

### 4.7 Conservar contexto absoluto sin romper simetría — potencial medio/alto

Usar solo diferencias A−B pierde información. Una diferencia de edad de 5 años no
significa lo mismo entre 25/30 y 35/40. Una diferencia de diez peleas no tiene la misma
certeza entre 0/10 y 20/30.

**Propuesta:** una representación equivariante al intercambio:

- diferencias antisymétricas que determinan el lado;
- medias/sumas simétricas que modulan la magnitud;
- interacciones `diferencia × contexto`, por ejemplo `age_diff × age_mean`,
  `elo_diff × rating_reliability`, `reach_diff × weight_class`;
- arquitectura de dos torres con pesos compartidos y salida
  `score(A,B) = g(A,B) - g(B,A)`.

Las variables simétricas solas no pueden favorecer un lado, pero sí pueden decir cuánto
creer en una diferencia. La prueba previa de peso/género/5R como contexto no cubre esta
formulación general.

### 4.8 Forma, daño y trayectoria — potencial medio

El decay uniforme de todas las estadísticas ya fue no concluyente. Quedan hipótesis más
específicas:

- golpes/KD absorbidos en 6, 12 y 24 meses;
- días desde último KO recibido;
- derrotas por KO consecutivas y tiempo de recuperación;
- pendiente de rendimiento ajustado por rival;
- edad respecto del pico estimado por división;
- layoff absoluto de ambos, no solo diferencia;
- frecuencia de peleas y campamentos interrumpidos.

Debe evitarse interpretar daño como diagnóstico médico. Son descriptores deportivos y
necesitan shrinkage por exposición.

### 4.9 Cambio de división, pesaje y corte de peso — potencial medio/alto

El modelo conoce peso nominal de la pelea, pero no trayectoria de categoría ni peso real.

**Posibles variables:**

- subir/bajar de división, primera pelea en la categoría y frecuencia del cambio;
- diferencia entre peso pactado y pesado, magnitud del fallo y tiempo adicional;
- catchweight y cambios tardíos del límite;
- historial reciente de fallos, solo si se demuestra señal incremental;
- interacción de edad, tamaño y cambio de división;
- hidratación si alguna fuente legítima y consistente la ofrece.

La tasa histórica de fallar el peso ya fue descartada; la **magnitud y circunstancia
actual**, disponibles recién en T-24h, son hipótesis distintas.

### 4.10 Rankings históricos y nivel competitivo — potencial medio

`ufc_odds.csv` ya contiene rankings point-in-time que no entran al modelo. La fuente
declara usar un histórico semanal de rankings y el dataset local tiene columnas de ranking
por división ([repositorio fuente](https://github.com/shortlikeafox/ultimate_ufc_dataset)).

**Propuesta:** auditar primero cómo se llenaron y verificar que cada ranking sea anterior
a la pelea. Luego probar:

- rank actual por división y P4P;
- entrada/salida del ranking y tendencia;
- victorias contra top-5/top-10 en ese momento;
- fuerza de calendario derivada, no el ranking bruto únicamente.

**Riesgo:** los rankings son subjetivos, solo existen desde 2013, son información pública
ya incorporada por el mercado y el sistema de rankings cambió. Nunca importar a ciegas
las 118 columnas del CSV: contiene targets posteriores como `Winner`, `finish` y tiempo
total.

### 4.11 Scorecards y calidad de la decisión — potencial medio

La prueba anterior de margen textual no pasó, pero scorecards por round permiten otra
señal:

- probabilidad/porcentaje de rounds ganados;
- decisiones muy divididas frente a actuaciones dominantes;
- rating actualizado por evidencia de rounds, no solo W/L;
- consistencia entre jueces;
- “victoria oficial” y “rendimiento observado” como señales separadas.

Tres rounds aportan más observaciones que un único resultado. Deben usarse solo para
peleas futuras y conservar el resultado oficial como target de la casa.

### 4.12 Circunstancia y logística de la pelea — potencial medio

Wikipedia demostró que información fuera del historial puede pasar el protocolo. Otras
familias, siempre con fuente y timestamp verificables:

- días exactos de aviso para cada peleador y cambio de rival;
- quién cambió de oponente, no solo quién aceptó reemplazar;
- campamento completo/parcial, lesión anunciada o suspensión médica pública;
- viaje, huso horario, altura de la sede y país/localía;
- Apex/cage pequeño, arena vacía y tamaño de jaula;
- cambio de equipo/coach;
- árbitro, especialmente para tiempo y método, si se conoce antes.

Noticias mediante NLP/LLM pueden ayudar a estructurar hechos, pero nunca deben convertirse
directamente en una “puntuación de confianza”. Cada extracción requiere fuente, cita,
timestamp, revisión humana y estado desconocido. El texto posterior a la pelea es leakage.

### 4.13 Señales de mercado temporales — potencial alto para forecast, no necesariamente EV

Además de la probabilidad actual:

- línea de apertura, actual y cierre por casa;
- velocidad/dirección del movimiento;
- dispersión entre casas y cuotas stale;
- vig, límites y profundidad si existen;
- consenso ponderado por calidad histórica de cada casa;
- diferencia entre Betano y consenso accesible.

Estas señales probablemente mejoren la predicción porque representan información
colectiva. No garantizan una apuesta rentable: para encontrar valor hay que vencer **el
precio que aún puede ejecutarse**, no explicar el cierre después de verlo.

### 4.14 Modelo residual sobre el mercado — prioridad alta cuando haya snapshots

El modelo con odds actual aprende de nuevo toda la probabilidad y no supera el mercado.
Una alternativa más conservadora es:

`logit(p_final) = logit(p_mercado_as_of) + corrección_regularizada(features)`

El mercado queda como baseline/offset con coeficiente uno; el modelo solo aprende un
residuo pequeño. Si no hay evidencia, la corrección vuelve a cero. Para método se usa el
análogo multiclase en espacio log-probabilidad.

**Gate:** comparación contra el mercado crudo en las mismas peleas y mismo timestamp. No
basta superar al modelo sin odds. El IC clusterizado de la diferencia debe ser negativo y
confirmarse fuera de muestra.

### 4.15 Ensambles y algoritmos nuevos — prioridad baja hasta mejorar datos

CatBoost, LightGBM/XGBoost, GAMs, Random Forest, redes pareadas y modelos bayesianos pueden
probarse como *challengers*, pero el grid del HistGB ya mostró techo informativo. Reglas:

- mismas filas y folds para todos;
- predicciones OOF para cualquier stacking;
- complejidad penalizada y simetría garantizada;
- comparación de log loss, calibración y estabilidad, no el mejor accuracy;
- ningún modelo se incorpora solo porque gana una ventana.

Un trabajo específico de MMA muestra que modelar estados del combate y simular con cadenas
de Markov puede ser más rico que clasificar el ganador directamente
([Holmes, McHale y Żychaluk, 2023](https://livrepository.liverpool.ac.uk/3154619/)). Es una
arquitectura candidata, no evidencia de que se traslade automáticamente a estos datos.

### 4.16 Calibración condicional y online — prioridad media después de un modelo nuevo

Isotónica, sigmoid y un factor de nitidez ya empeoraron el modelo actual. No deben
reintroducirse por costumbre. Opciones que sí justifican una prueba nueva:

- calibración beta, cuya familia contiene la identidad y reduce el riesgo de descalibrar
  un modelo ya razonable ([Kull, Silva Filho y Flach, 2017](https://proceedings.mlr.press/v54/kull17a.html));
- calibración cruzada solo con predicciones OOF;
- calibración por horizonte/era/mercado si hay muestra suficiente;
- actualización online con ventana móvil y detector de drift;
- curvas de confiabilidad con intervalos, no solo deciles puntuales.

La calibración importa más que el hit rate para decisiones de valor; existe evidencia
empírica en apuestas deportivas en esa dirección
([Walsh y Joshi, 2023](https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-forecasting-models-be-/)).

### 4.17 Incertidumbre de cada predicción y política de abstención — prioridad alta

Hoy se entrega una probabilidad puntual. Debe estimarse también cuánto varía entre:

- folds temporales;
- modelos del ensamble;
- bootstraps por evento;
- imputaciones/fuentes posibles;
- escenarios de circunstancia desconocida.

La apuesta solo puede evaluarse con el límite conservador. Por ejemplo, no usar
`p_media × cuota > 1`, sino exigir que un percentil bajo razonable de `p` todavía supere
la probabilidad de break-even, incluyendo vig y error de ejecución. Si no lo hace, la
salida correcta es “sin apuesta”.

### 4.18 Video y tracking — potencial incierto, coste muy alto

Visión por computador podría estimar velocidad, distancia, stance switching, reacción a
derribos, fatiga y daño no resumidos por UFCStats. Pero exige derechos de video,
segmentación fiable, enorme trabajo de etiquetado y prevención estricta de leakage. Solo
tiene sentido después de agotar fuentes estructuradas y con una prueba piloto pequeña.

### 4.19 Era, reglas y ponderación de entrenamiento — potencial medio

El modelo final entrena desde 1994 con peso uniforme. UFC 1, la era previa a las reglas
unificadas, Fight Island/Apex y el producto actual no son exactamente el mismo proceso.
El decay probado antes reducía estadísticas acumuladas, pero no cambió el peso de las
filas de entrenamiento ni modeló rupturas de régimen.

**Propuesta:** comparar, con tuning anidado:

- ventana móvil de N años;
- sample weights por recencia;
- interceptos/baselines por era;
- priors jerárquicos que comparten información sin igualar eras;
- indicadores de cambios de reglas, arena vacía y formato;
- evaluación que dé más peso al horizonte operativo reciente y reporte también todos los
  folds para no esconder fragilidad.

**Riesgo:** reducir historia aumenta varianza, especialmente en 5 rounds y divisiones con
pocas peleas. Solo queda si mejora el bloque reciente sin perder calibración ni depender de
un corte elegido a posteriori.

## 5. Modelo de over/under de rounds

### 5.1 No crear clasificadores independientes por cada línea

Entrenar un modelo para over 1.5, otro para 2.5 y otro para 4.5 puede producir
`P(over 2.5) > P(over 1.5)`, una contradicción. El target natural es el **tiempo hasta la
finalización**. De su curva de supervivencia se deriva cualquier línea:

`P(over L rounds) = P(T > 300 × L segundos)`

Una pelea que llega a decisión sobrevive hasta el límite programado. KO, sumisión y otros
stoppages son causas competitivas.

### 5.2 Construcción correcta del target

1. Parsear `TIME FORMAT` como secuencia real de rounds, no asumir cinco minutos.
2. Calcular tiempo acumulado sumando la duración programada de rounds completos.
3. Separar formatos modernos, TUF con overtime, no-time-limit y formatos UFC antiguos.
4. Mapear doctor stoppage, DQ, technical decision, NC y empate según la liquidación del
   mercado objetivo.
5. Guardar límite programado exacto y si el combate terminó o fue censurado/invalidado.
6. Para un corte exacto —por ejemplo 2:30 del round— aplicar la regla de la casa para
   win/push/void. Hay cuatro finales exactamente en umbrales de 1.5 en el histórico
   moderno; no deben decidirse implícitamente.

`features._fight_minutes` hoy hardcodea rounds previos de cinco minutos. Hay 198 peleas
con formatos antiguos/no-limit donde eso no representa el reloj real. Su efecto en el
ganador actual es pequeño, pero debe arreglarse antes de entrenar tiempo.

### 5.3 Modelo recomendado: supervivencia discreta con riesgos competitivos

Dividir el combate en intervalos —por ejemplo 30 segundos o medio round— y estimar en cada
uno, condicionado a seguir activo:

- hazard de KO/TKO de A;
- hazard de KO/TKO de B;
- hazard de sumisión de A;
- hazard de sumisión de B;
- hazard de otro stoppage;
- supervivencia al siguiente intervalo.

De una sola distribución conjunta salen:

- over/under de cualquier línea;
- llega/no llega a decisión;
- método total;
- ganador;
- ganador por método y round más probable.

Una primera versión más simple puede ignorar el lado y estimar hazards de KO/sub totales;
la versión completa se promueve solo si mejora. El trabajo MMA de cadenas de Markov citado
antes respalda explorar estados del combate, pero la validación local decide.

### 5.4 Variables específicas para duración

- división, género y rounds programados;
- pace y volumen ajustados por rival;
- KO/sub propios y recibidos con shrinkage;
- defensa, control, reversals y tiempo en posiciones;
- tasa de finalización temprana/tardía por intervalos;
- cardio/fade con exposición suficiente;
- edad, layoff, cambio de peso y short notice;
- referee/venue/cage cuando se conozcan;
- suma/compatibilidad de estilos, no solo diferencia A−B;
- probabilidades de mercado de total, distancia y método como priors.

El bloque simple de cardio ya fue no concluyente. Un hazard temporal usa la información
por round de otra manera y por eso merece prueba, pero no se debe asumir que funcionará.

### 5.5 Odds de rounds: oportunidad ya visible

El código de `oddsapi.py` solicita únicamente `h2h`. La documentación oficial actual dice
que para MMA los mercados destacados incluyen **fight winner y totals**, aunque totals
tiene cobertura limitada por casa
([The Odds API: MMA/UFC](https://the-odds-api.com/sports/mma-ufc-odds.html)). Su API v4
también ofrece descubrimiento de mercados por evento y snapshots históricos en planes de
pago ([guía v4](https://the-odds-api.com/liveapi/guides/v4/)).

Además, el fixture de `test_app.py`, documentado como forma real, contiene un mercado
Betano `Total de rounds` con “Más de 1.5/Menos de 1.5”. El parser actual lo lee como un par
genérico pero pierde la pelea y el nombre del mercado; todos los totales de 1.5 pueden
colisionar entre sí. El raw local del día auditado solo mostró ganador, por lo que la
cobertura es intermitente, no inexistente.

**Mejora propuesta:** preservar `event_id`, `fight_id`, bookmaker, market key/name, line,
selection, price, timestamp de actualización y commence time. Esto habilita totals cuando
aparezcan sin confundirlos con moneyline.

### 5.6 Evaluación del modelo de rounds

- log loss/Brier por línea real y por horizonte;
- calibration-in-the-large y curva de calibración;
- ranked probability score o log loss de la distribución completa de tiempo;
- coherencia monotónica entre líneas;
- comparación contra base de división y contra mercado sin vig;
- ROI/CLV con la cuota que existía al emitir el forecast;
- métricas separadas para 3R, 5R, hombres/mujeres y divisiones, con intervalos;
- bootstrap por evento y corrección por las líneas/variantes probadas.

## 6. Mejoras del tipo de finalización

### 6.1 Corregir taxonomía antes de modelar

Definir una taxonomía canónica y un mapping por sportsbook:

- KO/TKO;
- sumisión;
- decisión/technical decision;
- doctor stoppage;
- DQ;
- could not continue;
- empate/no contest/overturned;
- other/unknown.

Después se decide qué clases se combinan para cada mercado. El mapping actual de 121
doctor stoppage/DQ a “decisión” contamina target, `finish_rate` y
`finished_against_rate`.

### 6.2 Predecir resultado conjunto, no piezas incompatibles

Un modelo de seis clases inicial puede usar:

- A por KO/TKO;
- A por sumisión;
- A por decisión;
- B por KO/TKO;
- B por sumisión;
- B por decisión;

Más una política para “otros”. Sumando clases se obtiene ganador y método, garantizando
coherencia. El modelo de riesgos competitivos de la sección anterior es una extensión con
tiempo/round.

### 6.3 Habilidad ofensiva contra vulnerabilidad defensiva

Las sumas crudas de finish/KD/sub empeoraron el modelo actual. Una formulación más sólida
separa:

- potencia KO de A contra durabilidad y defensa de B;
- grappling/sumisión de A contra defensa de sumisión de B;
- ritmo/control de ambos contra probabilidad de llegar a tarjetas;
- calidad de oponentes y exposición de cada estimación;
- tasas regionales para debutantes.

Todo con partial pooling. Esta hipótesis no debe confundirse con repetir las sumas que ya
fallaron.

### 6.4 Usar props de mercado como prior u output directo

La medición existente es clara:

| predictor de método | log loss histórico |
|---|---:|
| base rate | 1.0187 |
| contexto | 1.0174 |
| props dentro de HistGB | 1.0489 |
| **props crudas del mercado** | **0.9582** |

Por tanto, si hay props en vivo, la opción de menor riesgo es mostrar el consenso
desvigueado. El challenger debe aprender una corrección residual pequeña en log-space y
demostrar que supera a las props solas. También hay que quitar vig sobre los seis outcomes
mutuamente excluyentes y comparar power, Shin u odds-ratio; no asumir que normalizar las
tres sumas es óptimo. La investigación sobre convertir cuotas en probabilidades muestra
que el método de devig afecta la calidad del forecast
([Štrumbelj, 2014](https://doi.org/10.1016/j.ijforecast.2014.02.008)).

### 6.5 Contexto futuro exacto

`cinco_r` se aproxima hoy con “es main event”. Eso falla en peleas de título fuera del
main event y formatos excepcionales. Para método y rounds debe obtenerse el número
programado real desde una fuente de cartelera/mercado, conservarlo con el fight ID y no
inferirlo solo por posición.

### 6.6 Métricas multiclase

- log loss multiclase y Brier multiclase;
- calibración por clase con intervalos;
- matriz de confusión solo como diagnóstico;
- accuracy top-1, pero nunca como criterio principal;
- comparación contra props y base rate en exactamente la misma cobertura;
- evaluación conjunta de coherencia con ganador y over/under.

No usar class weights para “equilibrar” KO/sub/decisión salvo que después se recalibren:
el objetivo es probabilidad real, no balanced accuracy.

## 7. Infraestructura de cuotas y mercado

### 7.1 Esquema normalizado — prioridad crítica

`betano_hist.csv` guarda solo `ts,a,b,cuota_a,cuota_b`. Falta saber evento, pelea,
mercado, línea, selección, bookmaker y hora exacta de inicio. El esquema mínimo debería
ser conceptualmente:

| campo | motivo |
|---|---|
| source/bookmaker | comparar calidad y ejecución |
| event_id/fight_id | evitar colisiones y rematches |
| commence_time UTC | cortar antes del inicio, no por día |
| fetched_at/last_update | detectar stale odds |
| market_key | h2h, totals, method, distance, etc. |
| line/point | 1.5, 2.5, 4.5 |
| selection | fighter/over/under/método |
| decimal_price | pago ejecutable |
| max_stake/limit | saber si el edge es realizable |
| status | abierto/suspendido/cerrado/live |

El formato debe ser append-only e idempotente.

### 7.2 Apertura, actual y cierre reales

La “primera observación de la app” no es necesariamente apertura. El cierre actual se
elige con todo tick anterior a `fecha_evento + 1 día`, porque la cartelera pierde la hora;
eso puede incluir cuotas live o posteriores al combate. Además, `ledger.visto` conserva
solo la fecha, no la hora.

**Mejora propuesta:** conservar timestamp UTC completo del evento y usar estrictamente el
último precio `last_update < commence_time`. Definir apertura como primer precio publicado
por esa casa, no primer momento en que el usuario abrió la app.

### 7.3 Consenso multi-casa más robusto

La mediana desvigueada es un buen baseline, pero se puede mejorar con:

- exclusión de precios stale y outliers obvios;
- peso histórico por calibración de cada casa/mercado/horizonte;
- regiones y casas realmente accesibles al usuario;
- separación entre consenso y mejor precio, guardando qué casa lo ofrece;
- dispersión como incertidumbre, no como edge automática;
- consenso de cierre como benchmark y consenso actual como input.

La señal actual “mejor cuota que el promedio” debe forward-testearse. Elegir el máximo de
varias casas contra la mediana genera matemáticamente aparentes EV positivos con
frecuencia; solo hay valor si los precios están sincronizados, son ejecutables y el
consenso está bien calibrado.

### 7.4 Datos históricos de props

La fuente GitHub actual se congela en marzo de 2026 y no especifica snapshots de apertura
y cierre. Opciones:

- empezar desde ahora un archivo propio frecuente;
- consultar `totals` en The Odds API;
- descubrir mercados adicionales por evento;
- si el costo lo justifica, backfill histórico de eventos/mercados disponible en planes
  pagos, verificando primero cobertura MMA;
- conservar Betano aunque sea intermitente, con parser de mercados genérico;
- no mezclar cuotas de casas/horizontes sin columnas de procedencia.

### 7.5 Devig por mercado y casa

Power ya superó al proporcional en el proyecto y debe quedar como campeón actual. Los
challengers posibles son Shin, odds-ratio y calibración específica de la casa. Evaluarlos
por book, rango de favorito y mercado; una mejora minúscula global puede ser relevante en
favoritos, pero debe sobrevivir clustering y holdout.

## 8. Integración correcta de predictores humanos

### 8.1 Persistencia inmutable

Hoy guardar de nuevo un predictor/evento reemplaza las filas anteriores. No hay
`created_at`, `updated_at`, cuota al publicar ni historial de revisiones. Eso permite que
una corrección posterior borre el pick original sin dejar rastro.

**Mejora propuesta:** eventos append-only de `pick_created`, `pick_revised` y
`pick_voided`, todos con timestamp, origen y usuario revisor. La vista materializada puede
mostrar la última versión, pero la evaluación usa la versión válida al cierre acordado.

### 8.2 Medir habilidad relativa a la dificultad

Sustituir el peso de accuracy cruda por un modelo jerárquico como:

`logit(P(acierto del pick)) = logit(P_mercado del lado elegido) + habilidad_tipster`

La habilidad se encoge hacia cero con pocas observaciones y puede variar por mercado,
división y horizonte. Otras métricas:

- mejora de log score contra mercado si publica probabilidades;
- ROI y CLV a la cuota registrada;
- residual de acierto sobre el esperado por sus picks;
- cobertura y selectividad;
- estabilidad temporal.

### 8.3 Calibrar su confianza

Si un tipster dice 80%, comprobar si esos picks ganan cerca de 80%. Una etiqueta HIGH sin
probabilidad no debe tratarse como 80. El campo `confianza` ya existe; falta puntuarlo con
Brier/log loss y una curva de calibración por predictor.

### 8.4 Evitar contar opiniones correlacionadas como independientes

Tres cuentas pueden copiar la misma línea, fuente o narrativa. Medir correlación de picks
y errores, agrupar tipsters redundantes y limitar su peso conjunto. Un consenso 3-0 no
equivale a tres evidencias si siempre votan igual.

### 8.5 Meta-modelo solo con datos out-of-sample

Las picks humanas pueden ser features si estaban publicadas antes del `as_of`. El
meta-modelo debe entrenarse con predicciones humanas históricas congeladas y comparar
contra mercado/modelo en folds temporales. Nunca usar el ranking calculado con resultados
del mismo evento.

### 8.6 Evaluar ganador, método y round por separado

Un tipster puede ser bueno en moneyline y malo en props. No compartir un peso global. Para
round exacto conviene usar log score/MAE ordinal o probabilidad si la publica; para método,
score multiclase. El acierto de ganador no valida una combinada ganador+método.

## 9. Hacer las apuestas menos riesgosas

### 9.1 Cambiar el default a “no apostar”

La etiqueta `APOSTABLE="alta"` proviene del mejor tramo elegido a posteriori, con ROI
+1,87% e IC95% [-5,56%, +9,40%]. No es una ventaja demostrada. Hasta superar un gate
prospectivo preregistrado, la app debería tratarlo como **seguimiento experimental**, no
candidata.

### 9.2 Usar el límite inferior de EV

Para cuota decimal `q`, el break-even es `1/q`. En lugar de apostar cuando
`p_media × q - 1 > 0`, exigir:

- probabilidad conservadora `p_low > 1/q`;
- margen adicional para error de precio, límites, impuesto/comisión y drift;
- cuota vigente y confirmada;
- ninguna alerta crítica de identidad/datos;
- suficiente liquidez.

Esto puede producir cero apuestas durante meses. Es una salida válida.

### 9.3 Stake pequeño, explícito y ligado al bankroll

Si algún día se demuestra edge:

- flat stake muy pequeño durante validación;
- después, Kelly fraccional o Kelly robusto con cap por pelea/evento;
- nunca full Kelly con probabilidades estimadas;
- límite de exposición diaria/semanal y de drawdown;
- separar bankroll de gastos personales;
- no subir stake para recuperar pérdidas.

La incertidumbre del parámetro exige encoger el stake; hay evidencia teórica y empírica de
que Kelly reducido mejora el comportamiento fuera de muestra frente a usar la estimación
puntual como verdad
([Baker y McHale, 2013](https://doi.org/10.1287/deca.2013.0271)).

### 9.4 Tratar correlación y combinadas

La app permite combinadas y multiplica cuotas. Sus legs no son necesariamente
independientes:

- ganador y método/over de la misma pelea están fuertemente correlacionados;
- varias peleas de una cartelera comparten circunstancias;
- el margen de la casa se compone en cada leg.

No calcular EV de una combinada multiplicando probabilidades marginales. Hace falta una
distribución conjunta o, como política segura, no recomendar parlays. Registrar una
combinada está bien; sugerirla automáticamente a partir de “locks” no está validado.

### 9.5 Line shopping y ejecución realista

- comparar varias casas accesibles;
- guardar la casa y precio aceptado, no solo el mejor observado;
- verificar que la línea siga abierta al confirmar;
- considerar límite máximo, stake rechazado, void y cash-out;
- medir slippage entre alerta y ticket;
- no contar como ejecutable una cuota histórica que el usuario no podía tomar.

### 9.6 CLV como diagnóstico, no garantía

CLV positivo sostenido indica que se obtuvo mejor precio que el cierre y converge antes
que ROI, pero no garantiza rentabilidad si el cierre está mal desvigueado, la muestra es
selectiva o las apuestas son correlacionadas. Reportar:

- CLV en probabilidad justa y en retorno esperado al cierre;
- media, mediana e intervalos clusterizados;
- porcentaje de apuestas que vencieron el cierre;
- CLV por casa, mercado y tiempo al evento;
- ROI y drawdown en paralelo.

### 9.7 Stop rules preregistradas

Antes del forward test fijar por escrito:

- versión exacta del modelo y regla de apuesta;
- tamaño mínimo de muestra y duración mínima;
- umbral de CLV/ROI con intervalo;
- pérdida máxima aceptable;
- cuándo se pausa por drift o error de datos;
- qué constituye un cambio de versión y reinicia la evaluación.

Sin estas reglas se termina ajustando el sistema después de cada racha.

## 10. Mejoras de calidad y observabilidad

### 10.1 Tests de invariantes y golden datasets

Agregar casos de análisis, sin necesidad de un framework pesado:

- no leakage `as_of` por cada fuente;
- simetría para ganador y coherencia de probabilidades conjuntas;
- monotonicidad de over/under;
- mapping de todos los métodos históricos;
- identidades homónimas, aliases y rematches;
- timestamps y cierre estrictamente preinicio;
- settlement de DQ/NC/push por casa;
- odds stale y mercados con la misma línea en peleas distintas;
- debutantes con y sin fuente regional;
- modelo/bundle/dataset con hashes coincidentes.

### 10.2 Monitoreo de drift

Por cartelera y trimestre:

- log loss/Brier/calibración de modelo y mercado;
- cobertura y faltantes por fuente;
- distribución de features y probabilidades;
- gap modelo-mercado;
- match rate de nombres/IDs;
- cuota y vig por casa;
- CLV/ROI/drawdown;
- error por división, debutantes, 3R/5R y horizonte.

No reentrenar automáticamente solo porque cambió una métrica; primero diagnosticar fuente,
drift o varianza.

### 10.3 Explicaciones honestas

`predict._factores` explica exactamente la parte logística, pero la probabilidad mostrada
es un promedio de logística e HistGB. Debe etiquetarse como “factores del componente
lineal”, o explicar también el componente no lineal. Una explicación incompleta no reduce
el error y puede generar exceso de confianza.

Los textos de `CONFIANZA` y `ROI_APOSTABLE` están hardcodeados. Aunque `train.py` imprima
nuevas tablas, no actualiza el serving. Conviene serializar métricas/versiones y mostrar
“no validado para esta versión” cuando no coincidan.

## 11. Ideas ya probadas que no conviene repetir igual

Salvo que cambien datos, target o formulación, no priorizar:

- más árboles o otro grid del mismo HistGB;
- isotónica, sigmoid o simple factor de nitidez global;
- `win_rate_last5` crudo;
- southpaw como indicador simple;
- contexto de división/género/5R agregado sin interacciones de fiabilidad;
- decay uniforme λ=0.85/0.93 de todas las estadísticas;
- Elo K-finish 40/28 en su forma probada;
- interacciones manuales simples ofensa×defensa;
- head/body/leg y distance/clinch/ground como agregados crudos;
- volumen crudo, fade crudo, score margin textual y title rate;
- récord regional W/L/N agregado encima del bloque de método;
- tasa histórica de fallar el peso;
- props de método como features libres de HistGB;
- elevar el umbral de EV usando la misma probabilidad puntual;
- asumir que una discrepancia grande contra la casa es valor.

Esto no prohíbe hipótesis distintas —por ejemplo residuales ajustados por rival o hazards
por round—; evita gastar tiempo en repetir exactamente los experimentos que ya fallaron.

## 12. Protocolo propuesto para aceptar cualquier mejora

### 12.1 Forecasting

Una mejora de ganador solo pasa si:

1. usa información disponible al `as_of`;
2. tiene cobertura reportada y no mejora solo por cambiar la muestra;
3. reduce log loss pareado contra el campeón;
4. el IC95% clusterizado por evento no toca cero;
5. resiste un bloque temporal reciente y corrección por pruebas múltiples;
6. no empeora materialmente calibración/Brier;
7. conserva simetría e invariantes;
8. confirma en forward test antes de reemplazar al campeón.

Para método/rounds se agrega comparación contra el mercado de props en la misma cobertura
y coherencia conjunta.

### 12.2 Betting

Una regla de apuesta solo pasa si:

1. usa cuota realmente ejecutable y timestamp exacto;
2. incluye vig, slippage, voids y límites;
3. fue elegida sin mirar el período final;
4. muestra CLV positivo con intervalo y no solo hit rate;
5. muestra ROI con intervalo, drawdown y peor racha;
6. evalúa correlación por evento/mercado;
7. se confirma prospectivamente con regla y stake congelados;
8. no depende de una única casa o de una línea stale.

No fijaría de antemano un número mágico de apuestas: el tamaño requerido depende de odds,
varianza, correlación y magnitud del edge. Debe calcularse potencia/precisión esperada antes
de lanzar el test.

## 13. Backlog priorizado

| prioridad | mejora | ganador | rounds | método | impacto potencial | esfuerzo | riesgo principal |
|---|---|:---:|:---:|:---:|---|---|---|
| P0 | IDs estables y aliases | ✓ | ✓ | ✓ | corrección crítica | medio | modificar fuente |
| P0 | timestamps `as_of/known_at` | ✓ | ✓ | ✓ | credibilidad del backtest | alto | datos históricos faltantes |
| P0 | esquema normalizado de odds | ✓ | ✓ | ✓ | habilita mercados reales | medio | parser/cobertura |
| P0 | hora UTC y cierre preinicio | ✓ | ✓ | ✓ | corrige CLV | bajo/medio | fuentes sin hora |
| P0 | taxonomía/settlement versionado |  | ✓ | ✓ | corrige targets | medio | reglas por casa |
| P0 | separar apertura/actual/cierre | ✓ | ✓ | ✓ | elimina domain shift | medio | snapshots históricos |
| P0 | fecha futura + circunstancia desconocida | ✓ | ✓ | ✓ | corrige serving | bajo | verificación manual |
| P0 | holdout intacto + pruebas múltiples | ✓ | ✓ | ✓ | evita falsas mejoras | medio | menor rapidez experimental |
| P0 | bootstrap clusterizado | ✓ | ✓ | ✓ | IC realistas | bajo/medio | pocos clusters en segmentos |
| P0 | desactivar interpretación de “candidata” | ✓ |  |  | reduce riesgo | bajo | ninguna, salvo menos señales |
| P1 | estado prospectivo de debutantes | ✓ | ✓ | ✓ | gran cobertura | medio | identidad regional |
| P1 | total rounds de Betano/The Odds API |  | ✓ | ✓ | benchmark y serving | medio | cobertura intermitente |
| P1 | modelo de supervivencia/riesgos | ✓ | ✓ | ✓ | unifica tres mercados | alto | muestra por causa/intervalo |
| P1 | props crudas + modelo residual |  |  | ✓ | señal ya demostrada | medio | fuente viva |
| P1 | calidad de oposición regional | ✓ | ✓ | ✓ | señal nueva en debuts | alto | scraping/leakage |
| P1 | shrinkage de tasas | ✓ | ✓ | ✓ | estabilidad | medio | priors mal especificados |
| P1 | rendimiento ajustado por rival | ✓ | ✓ | ✓ | estilos/calidad | alto | complejidad |
| P1 | rating dinámico con incertidumbre | ✓ |  |  | mejor prior/fiabilidad | medio | transferencia entre divisiones |
| P1 | contexto absoluto/equivariante | ✓ | ✓ | ✓ | señal perdida por diffs | medio | sobreajuste |
| P1 | ponderación de era/ventana móvil | ✓ | ✓ | ✓ | adaptación al UFC actual | medio | menor muestra |
| P1 | pesaje/cambio de categoría timestamped | ✓ | ✓ | ✓ | información tardía | medio | cobertura histórica |
| P1 | rankings históricos auditados | ✓ |  |  | nivel competitivo | bajo/medio | subjetividad/endogeneidad |
| P1 | scorecards por round | ✓ | ✓ | ✓ | más observaciones | alto | parsing/criterio de jueces |
| P1 | consenso por casa y stale filtering | ✓ | ✓ | ✓ | mejor prior/precio | medio | acceso regional |
| P1 | incertidumbre + no-bet | ✓ | ✓ | ✓ | decisión más segura | medio | intervalos inestables |
| P1 | tipsters contra baseline de mercado | ✓ | ✓ | ✓ | peso humano válido | medio | pocos datos revisados |
| P1 | auditoría inmutable de picks | ✓ | ✓ | ✓ | elimina hindsight | bajo/medio | migración legacy |
| P1 | exposición/correlación de apuestas | ✓ | ✓ | ✓ | menor drawdown | medio | joint probabilities |
| P2 | forma/daño específico | ✓ | ✓ | ✓ | posible señal ortogonal | medio | ruido/interpretación |
| P2 | exposiciones observadas/formato antiguo | ✓ | ✓ | ✓ | corrección de tasas | bajo/medio | impacto global bajo |
| P2 | viaje/altura/cage/camp/referee | ✓ | ✓ | ✓ | circunstancia | alto | datos no estructurados |
| P2 | CatBoost/GBM/GAM/stacking | ✓ | ✓ | ✓ | incremental | medio | mismo techo informativo |
| P2 | calibración beta/online | ✓ | ✓ | ✓ | probabilidades mejores | medio | poca muestra por segmento |
| P3 | NLP de noticias | ✓ | ✓ | ✓ | hechos tardíos | alto | leakage/alucinación |
| P3 | visión sobre video | ✓ | ✓ | ✓ | señal nueva rica | muy alto | derechos/etiquetado |

## 14. Orden recomendado de una futura implementación

### Fase A — hacer confiable la vara

IDs, timestamps, settlement, odds normalizadas, cierre preinicio, manifiestos, holdout,
clustering y multiplicidad. Recalcular todos los números actuales. No tocar el modelo aún.

### Fase B — abrir los mercados nuevos

Extraer totals y descubrir props por evento; construir targets exactos; publicar primero
bases y mercado desvigueado; luego modelo de supervivencia/competing risks.

### Fase C — cerrar el agujero de información del ganador

Servir debutantes, rating regional/opponent strength, shrinkage, ratings con incertidumbre,
contexto equivariante y datos de pesaje por horizonte. Cada bloque entra de a uno.

### Fase D — combinar fuentes sin sobreescribir el mercado

Modelos residuales sobre apertura/actual, consenso por casa, calibración condicional y
estimación de incertidumbre. Mantener siempre mercado crudo como baseline visible.

### Fase E — humanos y cartera

Auditoría inmutable de picks, skill relativo al mercado, confianza calibrada, correlación,
no-bet, stake conservador y reglas prospectivas.

## 15. Expectativa realista

No establecería 80–90% sobre toda la cartelera como objetivo técnico. El programa puede
alcanzar 80–90% en un subconjunto de favoritos, igual que el mercado, pero el precio paga
esa certeza. Los objetivos útiles son:

- bajar log loss/Brier fuera de muestra;
- acercarse o superar al mercado **en apertura**, no reconstruir el cierre;
- aumentar cobertura de debutantes sin inventar datos;
- producir probabilidades coherentes para ganador/tiempo/método;
- ganar CLV de forma prospectiva;
- demostrar ROI después de costos con incertidumbre;
- reducir drawdown y abstenerse cuando no hay evidencia.

Es posible que, tras corregir todo, la conclusión sea “el mercado sigue siendo mejor y no
hay apuesta”. Ese resultado también mejora el programa: evita convertir una predicción
interesante en una pérdida con falsa seguridad.

## 16. Fuentes externas consultadas

- [The Odds API — cobertura MMA/UFC, winner y totals](https://the-odds-api.com/sports/mma-ufc-odds.html)
- [The Odds API v4 — endpoints de odds, mercados por evento e históricos](https://the-odds-api.com/liveapi/guides/v4/)
- [Greco1899/scrape_ufc_stats — fuente diaria de UFCStats](https://github.com/Greco1899/scrape_ufc_stats)
- [shortlikeafox/ultimate_ufc_dataset — odds/rankings históricos](https://github.com/shortlikeafox/ultimate_ufc_dataset)
- [Holmes, McHale y Żychaluk — modelo Markov para MMA](https://livrepository.liverpool.ac.uk/3154619/)
- [Glickman — sistema Glicko y rating deviation](https://www.glicko.net/glicko/glicko.pdf)
- [Cattelan, Varin y Firth — Bradley-Terry dinámico en deportes](https://wrap.warwick.ac.uk/id/eprint/54660/)
- [Kull, Silva Filho y Flach — beta calibration](https://proceedings.mlr.press/v54/kull17a.html)
- [Štrumbelj — convertir odds a probabilidades](https://doi.org/10.1016/j.ijforecast.2014.02.008)
- [Walsh y Joshi — calibration vs accuracy en apuestas deportivas](https://researchportal.bath.ac.uk/en/publications/machine-learning-for-sports-betting-should-forecasting-models-be-/)
- [Novy-Marx — múltiples señales y sesgo de backtest](https://www.nber.org/papers/w21329)
- [Baker y McHale — Kelly bajo incertidumbre de parámetros](https://doi.org/10.1287/deca.2013.0271)
