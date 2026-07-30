# Predictor de peleas UFC

Predice P(gana A) / P(gana B) para cualquier matchup de UFC y lo compara contra la
cuota de una casa de apuestas que ingresás a mano. La idea es que sea una fuente de
información más para decidir, no un detector automático de valor.

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
```

## Qué hace cada script

| script | qué hace | salida |
|---|---|---|
| `fetch_data.py` | baja los 6 CSVs de ufcstats.com ya scrapeados por [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats), que se refrescan a diario, más las odds históricas de [`shortlikeafox/ultimate_ufc_dataset`](https://github.com/shortlikeafox/ultimate_ufc_dataset) (2010+, matchean el 95% de ese período). Re-correrlo = datos al día. | `data/raw/*.csv` |
| `features.py` | recorre las peleas en orden cronológico y arma, para cada una, las features de ambos peleadores **usando solo sus peleas anteriores** (win rate, racha, golpes por minuto, takedowns, control, finish rate, descanso, edad, alcance, Elo, knockdowns propios y recibidos, defensas de striking y derribo, tasa de veces finalizado, Elo promedio de los rivales). Cada pelea genera dos filas: diffs A−B y la espejada B−A, para eliminar el sesgo de esquina roja. | `data/features.csv` |
| `train.py` | Promedio de `HistGradientBoostingClassifier` y una logística sin intercepto, con split temporal (test = últimos 2 años, validación = los 2 anteriores). Sin calibración: re-verificada como dañina con el protocolo nuevo. Decide con **rolling-origin de 20 folds (7195 peleas) + IC95% del delta pareado**, no con el delta de una sola ventana. Imprime train loss junto a val/test y una tabla de confiabilidad. El `model.pkl` final se re-entrena con todo el historial. | `model.pkl`, `data/fighter_state.csv` |
| `predict.py` | `predict(a, b, cuotas=None) -> dict`. Sirve el mismo promedio de modelos que se evalúa, en las dos orientaciones, así el resultado no depende del orden. Con las dos cuotas agrega la probabilidad del mercado, la del modelo alimentado con ella, el nivel de confianza y los factores que mueven la predicción. | — |
| `app.py` | UI Streamlit: elegís dos peleadores y las dos cuotas decimales, y ves las tres probabilidades, el nivel de confianza con su motivo, y las features que más mueven la predicción. | — |
| `test_app.py` | `python test_app.py`. Verifica que la predicción no dependa del orden, que la confianza salga del tramo correcto, y que la app renderice con y sin cuotas. | — |

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

Sobre las 5679 peleas del rolling-origin que tienen cuota:

| | log loss |
|---|---|
| modelo (sin odds) | 0.6608 |
| **mercado (implícita sin vig)** | **0.6115** |
| modelo alimentado con la cuota | 0.6119 |

El mercado le gana al modelo por 0.0493 (IC95% [−0.0571, −0.0418]), y meterle la cuota al
modelo **no mejora sobre la cuota sola** (+0.0005, IC95% [−0.0031, +0.0042]). O sea: las 22
features no aportan nada que la línea de cierre no tenga ya.

Por eso esto no sirve para buscar valor. Sirve como **segunda opinión independiente**, y
la discrepancia con el mercado es la señal útil — pero al revés de lo que uno esperaría:

| discrepancia \|modelo−mercado\| | peleas | log loss modelo | log loss mercado |
|---|---|---|---|
| < 0.05 | 1488 | 0.6592 | 0.6606 |
| 0.05 – 0.15 | 2413 | 0.6415 | 0.6169 |
| 0.15 – 0.25 | 1299 | 0.6620 | 0.5790 |
| > 0.25 | 573 | **0.7416** | 0.5402 |

Cuando coinciden, el modelo vale tanto como la casa. Cuando discrepan fuerte, el modelo
rinde **peor que una moneda** y la casa gana: en las 1745 peleas donde eligen ganadores
distintos, la casa acierta 58.8% y el modelo 41.8%. Discrepar no es encontrar valor, es
la señal de que al modelo le falta información que el mercado sí tiene. De ahí sale el
nivel de confianza que muestra la app.

## Por qué no sirve tocar los hiperparámetros

Está medido: entre 50 y 800 árboles el train loss cae de 0.6371 a 0.4684 mientras el de
validación se mueve de 0.6613 a 0.6701. Toda la capacidad extra se va en memorizar. No es
under- ni overfitting: es **techo de información**. La única forma de mejorar es meter
información que hoy no está, no reconfigurar el modelo.

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

Solo ganador (no método ni rondas), solo UFC, corre local.

Las cuotas entran como feature de un segundo modelo, que convive con el que no las usa —
la app muestra los dos. El modelo sin odds es la opinión independiente; el de con odds
está para ver cuánto lo mueve el historial. Ninguno le gana al mercado solo.

**Lo que esto no hace:** no encuentra apuestas de valor. Está medido que la casa es mejor
predictor, y que donde el modelo más discrepa es donde más se equivoca. Un modelo bien
calibrado te dice cuándo tenés razón, no cuándo cobrás — apostar favoritos claros no es
gratis, el sesgo favorito-longshot hace que las cuotas bajas estén bien pagadas.
