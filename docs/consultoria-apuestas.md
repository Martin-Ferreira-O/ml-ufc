# Consultoría: cómo se calcula la probabilidad de apostar y cómo se gana a largo plazo

**Fecha:** 2026-08-07 · **Alcance:** capa de mercado, decisión y verificación.
**No** se tocaron las features del modelo de predicción.

Todos los números de este documento se midieron corriendo el pipeline del repo.
Los que dependen del modelo salen de un bundle de **22 features** en vez de 28: la
política de egress de este entorno bloquea `en.wikipedia.org` y `www.sherdog.com`, así
que `reemplazo`, `peso_no_dado` y las cuatro de récord pre-UFC quedaron 100% NaN. Se
indica en cada tabla afectada. Los números de mercado (devig, vig, tamaño de muestra) no
dependen del modelo y son exactos.

---

## Resumen ejecutivo

1. **La probabilidad está bien calculada.** El modelo está calibrado contra la realidad
   (ECE 0.011) y el desvigueo usa el mejor de cuatro métodos, ahora medido y no supuesto.
2. **La probabilidad no alcanza, y eso ya estaba medido.** El mercado le gana al modelo
   por 0.0499 de log loss. Lo nuevo de esta consultoría es que la pregunta se cerró con
   el estimador correcto: un **pool logit de dos parámetros** —el test de mínima varianza
   para "¿el modelo aporta algo sobre el precio?"— da **no concluyente**. No es que falte
   potencia estadística: la señal no está.
3. **El error conceptual que costaba plata** era confundir tres cosas distintas:
   calibración contra la realidad, calibración contra el mercado y multiplicidad. Solo la
   primera estaba bien.
4. **El ROI no puede ser nunca el criterio de decisión de este proyecto.** Distinguir una
   ventaja del 2% de cero pide **19.623 apuestas**; a ~500 por año son cuarenta años. El
   CLV pide **~32**. Esa sola cuenta reorganiza todo lo demás.
5. **Lo que sí queda por explotar es el precio, no la predicción**: mejor cuota entre
   casas y consenso multi-casa. No requiere ganarle a nadie con un modelo.
6. **El staking no crea ventaja pero decide la supervivencia.** Sobre las mismas apuestas
   perdedoras, flat de 1 unidad termina en **−261,7** sobre una banca de 100 (o sea:
   quebrado hace rato) y ¼ Kelly con tope termina en **28,1**. Ninguna de las dos gana
   plata; una de las dos permite seguir jugando mientras se averigua.

---

## 1. Cómo se calcula hoy la probabilidad

La cadena completa, con el archivo de cada paso:

| paso | qué hace | dónde |
|---|---|---|
| 1 | 28 features antisimétricas por pelea (diferencias A−B) | `ufc/modelo/features.py` |
| 2 | HistGB + logística sin intercepto, promediadas en las dos orientaciones | `ufc/modelo/train.py` |
| 3 | Cuota decimal → implícita `1/q` → probabilidad justa sin vig | `ufc/modelo/devig.py` |
| 4 | `EV = p × q − 1` | `ufc/modelo/predict.py` |
| 5 | EV → fracción de banca → monto | `ufc/modelo/apuesta.py` **(nuevo)** |
| 6 | ¿está autorizado a apostar? | `ufc/modelo/gate.py` **(nuevo)** |

Los pasos 5 y 6 no existían. Ahí estaba el hueco: el proyecto sabía producir una
probabilidad y no sabía qué hacer con ella.

### 1.1 El paso 3 no es un detalle

`1/cuota` no es una probabilidad: las dos implícitas suman más que 1, y ese exceso es el
margen de la casa. Cómo se reparta ese exceso mueve la probabilidad **uno o dos puntos**,
y como `EV = p×q − 1`, un punto de probabilidad en un favorito de 1.30 es 1,3 puntos de
EV. El repo usaba `power` con una comparación contra `proporcional`. Ahora están los
cuatro métodos estándar, medidos contra el resultado real de **6.901 peleas**:

| método | log loss | vs power (IC95% clusterizado) | veredicto |
|---|---|---|---|
| **power** | **0.6075** | — | **campeón** |
| odds_ratio (Wheeler) | 0.6077 | +0.0002 [+0.0000, +0.0004] | se descarta |
| Shin | 0.6078 | +0.0003 [+0.0000, +0.0005] | se descarta |
| proporcional | 0.6084 | +0.0009 [+0.0003, +0.0015] | se descarta |

Los tres challengers pierden con el IC95% entero por encima de cero. **Power se queda,
ahora por evidencia y no por herencia.**

Dónde está la diferencia, que es lo que importa:

| tramo de favorito | n | proporcional | power |
|---|---|---|---|
| 50–60% | 2116 | 0.6886 | 0.6888 |
| 60–75% | 3248 | 0.6357 | 0.6356 |
| **75–90%** | **1466** | **0.4519** | **0.4482** |
| **90%+** | **71** | **0.1981** | **0.1859** |

La ventaja de `power` está concentrada en los favoritos grandes — exactamente donde se
decide si hay EV. Una mejora global de 0.0009 que vive entera en el 22% de las peleas que
más importan no es una mejora global de 0.0009.

Shin merecía la prueba: es el único de los cuatro que modela *por qué* existe el sesgo
favorito-longshot (la casa se protege de apostadores informados, y estar informado vale
más en el underdog). Perdió igual. Queda implementado y medido para no volver a preguntarlo.

---

## 2. Los tres errores que se confundían en uno

El síntoma era espectacular: **el modelo prometía +38,6% de EV por apuesta y entregaba
−6,1% de ROI.** Eso no es mala suerte. Son tres problemas distintos que se veían igual.

### 2.1 Calibración contra la realidad — está bien

Cuando el modelo dice 70%, gana el 70%. ECE 0.011 fuera de muestra, y ningún calibrador
(Platt, isotónica) le gana al crudo. **Este problema no existe**, y por eso calibrar nunca
arregló nada: era la respuesta correcta a la pregunta equivocada.

### 2.2 Calibración contra el mercado — era el problema

El modelo está **comprimido hacia 0.5** respecto del mercado. Y como

```
EV = q × p_modelo − 1 ≈ p_modelo / p_mercado − 1
```

la compresión sola pone al underdog por encima de cero **en todas las peleas**. No era una
señal de valor: era la forma de la distribución del modelo. Un modelo perfectamente
calibrado contra la realidad puede estar sistemáticamente sesgado contra el mercado, y es
el segundo sesgo el que fabrica el EV.

**La herramienta correcta para eso es el pool logarítmico de opiniones**, no un calibrador:

```
logit(p) = w_mkt · logit(p_mercado) + w_mod · logit(p_modelo)
```

Sin intercepto, conserva la antisimetría exacta (`p(A,B) + p(B,A) = 1`). Ajustado de forma
prequencial sobre 5.687 peleas con cuota (modelo de 22 features):

| | log loss |
|---|---|
| modelo solo | 0.6608 |
| **mercado solo** | **0.6109** |
| pool (w_mercado 0.970, w_modelo 0.214) | 0.6106 |

**pool vs mercado: −0.0003, IC95% [−0.0018, +0.0014] — no concluyente.**

Esto es lo más definitivo que se puede decir sobre la pregunta. El README ya reportaba que
meterle la cuota al HistGB como feature número 29 no le ganaba al mercado (+0.0014, IC que
cruza cero), pero ese resultado admitía la excusa de la varianza: un GBM con 29 features
tiene ruido de sobra para tapar una señal chica. **Un pool de dos parámetros no tiene esa
excusa.** Es el estimador de mínima varianza para exactamente esta pregunta, y sigue sin
encontrar nada.

> **Conclusión operativa:** el EV de este proyecto no puede venir del modelo. Tiene que
> venir del precio. Todo lo que sigue está construido sobre esa conclusión.

### 2.3 Multiplicidad — el que queda vivo

El "mejor umbral" salía de mirar 21 umbrales sobre los mismos datos, y el "mejor segmento"
de mirar 32 niveles. Con ~5% de falsos positivos por nivel se esperan ~2 hallazgos de puro
azar. El repo ya lo advertía en prosa; ahora la tabla del backtest trae la columna
**«necesita»** —cuántas apuestas pediría una prueba de potencia para que esa fila
concluya algo— y el efecto se ve solo:

| umbral | apuestas | ROI | IC95% | necesita |
|---|---|---|---|---|
| 0% | 6561 | −6.11% | [−9.54%, −2.63%] | 3.773 |
| 5% | 5018 | −7.87% | [−12.11%, −3.72%] | 2.505 |
| 11% | 3335 | −10.38% | [−15.35%, −5.07%] | 1.598 |
| 17% | 1943 | −8.05% | [−15.21%, −0.58%] | 3.039 |
| **19%** | **1543** | **−4.56%** | **[−13.59%, +4.00%]** | **10.069** |

Los ROI menos malos son los de umbral alto — que son justo los que tienen menos apuestas y
más lejos están de poder concluir. **17 de 21 umbrales tienen el IC95% enteramente por
debajo de cero:** no es que no se pueda concluir, está medido que pierden.

---

## 3. La aritmética de ganar a largo plazo

Esta sección es el núcleo de la consultoría. Son cinco cuentas y ninguna es opinable.

### 3.1 El break-even y el peaje

Con cuota decimal `q` hay que superar `p > 1/q`. El margen de la casa es
`1/q_a + 1/q_b − 1`:

| fuente | vig mediano |
|---|---|
| `ufc_odds.csv` (líneas de cierre históricas) | **3,70%** |
| Betano en vivo (`betano_hist.csv`, 754 ticks) | **5,89%** |

Con 5,89% hay que sacarle **~2,9 puntos de probabilidad a la línea justa de cada lado
solo para empatar**. Ese es el peaje antes de hablar de ninguna ventaja. Y no es un
detalle secundario: el backtest cobra con cuotas al 3,70% y la app apuesta al 5,89%, así
que el backtest ya es optimista por construcción respecto del juego real.

### 3.2 El tamaño de muestra — la cuenta que reorganiza todo

Una apuesta flat de 1 unidad a cuota `q` con probabilidad `p` gana `q−1` o pierde 1. Su
desviación es `σ = q·√(p(1−p))`, que cerca de `q = 2.00` vale **1,0**. Para distinguir una
media `e` de cero con potencia 80% y α=0.05:

```
n = ((1.96 + 0.8416) · σ / e)²
```

| lo que se quiere detectar | σ | apuestas necesarias |
|---|---|---|
| ROI de +2% | 1.00 | **19.623** |
| ROI de +5% | 1.00 | 3.140 |
| **CLV medio de +2%** | **0.04** | **32** |

A ~500 apuestas por año, verificar un ROI del 2% son **cuarenta años**. Un sistema cuyo
criterio de éxito tarda cuarenta años en evaluarse no tiene criterio de éxito: tiene una
excusa para no tener ninguno, y en la práctica se termina ajustando después de cada racha.

El CLV —comparar el precio tomado contra el precio de cierre— tiene una décima parte de la
varianza porque compara dos precios en vez de esperar un resultado binario. Por eso
converge en decenas de apuestas. **No se elige el CLV porque sea más elegante; se elige
porque es el único estadístico que alcanza a converger en una vida humana.**

Esto está implementado en `apuesta.n_para_detectar`, y ahora aparece en la tabla del
backtest, en el ledger y en el gate.

### 3.3 Kelly: qué es lo que de verdad se maximiza

El EV es lineal en el tamaño de la apuesta, así que maximizar EV siempre recomienda
apostar todo — y apostar todo quiebra con probabilidad 1. Lo que se maximiza es el
crecimiento logarítmico:

```
G(f) = p·ln(1 + f(q−1)) + (1−p)·ln(1 − f)     →     f* = (p·q − 1)/(q − 1)
```

Kelly fraccional `c·f*` conserva aproximadamente `c(2−c)` del crecimiento: **medio Kelly
conserva el 75% del crecimiento con la mitad de la volatilidad**, un cuarto conserva el
44% con un cuarto de la volatilidad. La curva es plana cerca del óptimo, así que quedarse
corto cuesta poco y pasarse cuesta mucho. Verificado numéricamente contra la fórmula
exacta en `check_staking`.

### 3.4 Riesgo de ruina

La probabilidad de que la banca toque alguna vez la fracción `x` de su valor es
aproximadamente `x^(2/c − 1)`:

| fracción de Kelly | P(perder alguna vez la mitad) |
|---|---|
| Kelly completo | **50,0%** |
| ½ Kelly | 12,5% |
| ¼ Kelly | 0,8% |

Y eso **apostando con ventaja real y con la probabilidad exacta**. Como `p` es estimada y
no conocida, el óptimo está estrictamente por debajo de `f*(p̂)` (Baker y McHale, 2013, ya
citado en `posibles_mejoras.md`). De ahí las dos reglas de `config/gate.json`: decidir
sobre la **cota inferior** de `p`, y nunca pasar de ½ Kelly.

### 3.5 Combinadas: el vig se compone

`EV = ∏(1 + eᵢ) − 1`. Tres selecciones que por separado están a −5% no dan −5%: dan
**−14,3%**. Una combinada de legs sin ventaja no es "más riesgo por más premio", es la
misma apuesta con el margen cobrado tres veces.

Lo único que justifica una combinada es que los legs estén **positivamente correlacionados
y la casa los precie como independientes**. Eso ahora se puede preguntar con un número:
`apuesta.dependencia_necesaria` devuelve cuánto más seguido tendrían que salir juntas las
selecciones. Para dos legs a 1.90, el `lift` necesario es **+10,8%** (correlación φ ≈
+0,11). Es una pregunta contestable, a diferencia de "¿me siento confiado?".

---

## 4. El staking, medido sobre las apuestas reales del backtest

Las **mismas** apuestas perdedoras del umbral 1%, dimensionadas de cuatro formas, banca
inicial 100:

| staking | banca final | peor caída | manda el tope | P(perder la mitad) |
|---|---|---|---|---|
| flat 1 unidad | **−261,72** | −394,0 u | — | — |
| ¼ Kelly | 28,07 | −74,9% | 85% | 0,8% |
| ½ Kelly | 29,59 | −73,7% | 90% | 12,5% |
| Kelly completo | 30,04 | −73,3% | 93% | 50,0% |

Tres lecturas, en orden de importancia:

1. **El flat de 1 unidad sobre una banca de 100 no es conservador.** Termina en −261,72:
   la banca se agotó muchas veces y el backtest siguió apostando igual, porque un stake
   fijo es un porcentaje *creciente* de lo que va quedando. Es la forma más común de
   quebrar creyendo que se está siendo prudente.
2. **Ninguna estrategia de staking convierte una desventaja en ganancia.** Las cuatro
   pierden. Kelly no crea ventaja; administra la que haya.
3. **Las tres fracciones de Kelly dan casi lo mismo, y eso es un hallazgo.** El tope duro
   del 1% corta a Kelly en el **85–93%** de las apuestas: quien está dimensionando no es
   Kelly, es el tope. Pasa porque este modelo cree tener +38% de EV y su Kelly pide
   fracciones absurdas. **Ésa es la defensa del tope:** es lo que impide que una
   probabilidad sobreconfiada se traduzca en un stake sobreconfiado.

---

## 5. Dónde queda ventaja de verdad, y en qué orden

Ranking por evidencia, no por atractivo:

| # | vía | por qué | estado |
|---|---|---|---|
| 1 | **Mejor precio entre casas** | Aritmética pura. No necesita que el modelo acierte, funciona hasta en debuts. | Implementado (`oddsapi`) |
| 2 | **Consenso multi-casa como verdad** | El consenso desvigueado es el mejor estimador público; si una casa paga por encima, eso es valor sin predecir nada. | Implementado |
| 3 | **Devig correcto** | Vale 1–2 puntos de probabilidad justo en el rango de favoritos. | Medido, power confirmado |
| 4 | **Staking Kelly fraccional con topes** | No crea ventaja; decide si se sobrevive para cobrarla. | Implementado |
| 5 | **Gate por CLV** | El único criterio verificable en tiempo humano. | Implementado |
| 6 | **Disciplina** | Sin combinadas, topes de exposición por evento, stop rules preregistradas. | En `config/gate.json` |
| 7 | ~~Mejorar el modelo~~ | Medido: no aporta sobre el precio. Es la vía que hay que dejar de financiar. | Cerrado por evidencia |

### 5.1 La trampa del "mejor precio", dicha en voz alta

Hay que nombrarla porque es la forma más fácil de perder plata creyendo que se está
haciendo lo correcto: **tomar el máximo de N precios y compararlo contra el consenso de
esos mismos N da EV positivo con frecuencia aunque los precios sean puro ruido alrededor
de la misma probabilidad.** El máximo de una muestra está por encima de su centro por
construcción.

Tres defensas, todas implementadas:

1. **La casa que ofrece el mejor precio no entra al consenso de ese lado.** El motivo es
   independencia: si la referencia contiene el precio que se está juzgando, es en parte
   una función de él. Ojo con la dirección — sobre el fixture de los tests, excluirla
   **sube** el EV medido de +1,8% a +6,1%. No es una defensa contra el ruido, es una
   corrección de sesgo.
2. **Descontar la dispersión entre casas.** Cuando las casas no se ponen de acuerdo, el
   consenso vale menos, no más. `apuesta.p_conservadora` le resta `1.645·σ` a la
   probabilidad antes de calcular nada. **Ésta sí es la defensa contra el ruido.**
3. **Exigir un EV mínimo** (2% en `config/gate.json`), no simplemente EV > 0.

Y dos defensas operativas que ninguna matemática reemplaza: descartar precios **stale**
(una casa que no actualiza hace 12 horas no está cotizando) y verificar que la línea siga
abierta al confirmar.

---

## 6. Qué se implementó

### Módulos nuevos

| archivo | qué resuelve |
|---|---|
| `ufc/modelo/devig.py` | Los 4 métodos de desvigueo + CLI que mide cuál gana. |
| `ufc/modelo/apuesta.py` | EV, cota inferior, Kelly, crecimiento, ruina, potencia, combinadas, exposición. Funciones puras. |
| `ufc/modelo/pool.py` | Pool logit modelo+mercado, prequencial y antisimétrico. |
| `ufc/modelo/gate.py` | Evalúa la regla preregistrada contra el CLV real. |
| `config/gate.json` | La regla congelada: criterio, n mínimo, staking, condiciones de pausa. |
| `ufc/registro/banca.py` | La banca declarada: el denominador de todo staking. |

### Cambios en lo que ya existía

- **`predict.py`**: `_desvig` delega en `devig`; `APUESTAS_AUTOMATICAS = False` pasa a ser
  `apuestas_automaticas()`, derivada del gate. Misma respuesta, pero ahora con motivo, `n`
  y cuánto falta — una constante no se puede refutar.
- **`ledger.py`**: agrega `ev_al_cierre` (el CLV en unidades económicas), IC95%
  clusterizado por evento, `%` que batió al cierre y potencia. **Y corrige un bug real**:
  el "cierre" se elegía con `fecha_evento + 1 día`, que puede tomar cuotas **en vivo** o
  posteriores al combate. ESPN manda la hora UTC de inicio y `cartelera.py` la tiraba con
  `[:10]`; ahora se conserva (`inicio_utc`, columna aditiva) y el corte es estricto.
- **`oddsapi.py`**: precios por casa, filtro de stale, exclusión de la casa del mejor
  precio, dispersión como incertidumbre, y `data/odds_hist.csv` append-only — que es dato
  irreproducible hacia atrás y por eso empieza a acumularse ya, aunque todavía no se use.
- **`backtest.py`**: `curva_kelly` (banca compuesta, drawdown en % del pico, dimensionado
  **por evento** porque las líneas cierran juntas), `comparar_staking` y `n_para_concluir`
  por fila de la grilla.
- **UI**: banner del gate, banca, stake sugerido, medidor de exposición, matemática de la
  combinada, CLV con IC, y pestaña de staking en el backtest.
- **Tests**: `check_devig`, `check_staking`, `check_pool`, `check_gate`.

---

## 7. El estado real del forward test, hoy

`data/ledger.csv` tiene 16 predicciones congeladas y **4 lados seguidos con cierre**:

| | |
|---|---|
| CLV económico medio | **−14,0%** |
| IC95% (clusterizado) | [−24,1%, −9,3%] |
| Batió al cierre | **0 de 4** |
| Muestra | 4 de 100 preregistradas |

Con 4 apuestas esto no prueba nada, y el gate lo dice con esas palabras en vez de
celebrar o lamentar el signo. Pero la dirección es la esperable dado todo lo anterior: el
lado que el modelo elige es **sistemáticamente peor** que el que elige el cierre.

**El gate está cerrado y no hay nada en este repo que pueda abrirlo salvo 100 apuestas con
CLV positivo y el IC95% despegado de cero.**

---

## 8. Lo que sigue

En orden de retorno esperado:

1. **Acumular `odds_hist.csv`.** Cada corrida de la app guarda precios por casa. Es la
   única fuente de CLV multi-casa y no se puede recuperar hacia atrás. No requiere pensar,
   requiere que la app se abra.
2. **Correr el forward test sobre la regla top-down**, no sobre el modelo: registrar el
   lado donde la mejor cuota le gana al consenso limpio, y medirle el CLV. Son ~100
   apuestas, o sea unos meses. Es el experimento que puede dar positivo.
3. **Verificar ejecutabilidad**: límite máximo, línea todavía abierta, slippage entre
   alerta y ticket. Un EV que no se puede tomar no es un EV.
4. **No volver a invertir en el modelo de ganador** hasta que cambie la *fuente* de datos.
   El techo está medido como de información, no de capacidad, y el pool de dos parámetros
   cerró la pregunta con el mejor instrumento disponible.

---

## 9. Advertencia sobre este documento

Nada de acá autoriza a apostar. La conclusión honesta de la consultoría es que **este
proyecto todavía no tiene una ventaja demostrada**, y que la mayor parte del trabajo hecho
consiste en poder decir eso con números en vez de con intuición. La infraestructura de
staking existe para el día en que haya algo que dimensionar; el gate existe para que ese
día se decida con evidencia y no con una racha.

Es perfectamente posible que después de todo esto la conclusión siga siendo "el mercado es
mejor y no hay apuesta". Ese resultado también es valioso: evita convertir una predicción
interesante en una pérdida con falsa seguridad.
