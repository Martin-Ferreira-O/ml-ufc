# Predictor de peleas UFC

Predice P(gana A) / P(gana B) para cualquier matchup de UFC, con probabilidades
calibradas, y las compara contra la cuota de una casa de apuestas que ingresás a mano.
La idea es detectar peleas donde el modelo discrepa del mercado.

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
| `fetch_data.py` | baja los 6 CSVs de ufcstats.com ya scrapeados por [`Greco1899/scrape_ufc_stats`](https://github.com/Greco1899/scrape_ufc_stats), que se refrescan a diario. Re-correrlo = datos al día. | `data/raw/*.csv` |
| `features.py` | recorre las peleas en orden cronológico y arma, para cada una, las features de ambos peleadores **usando solo sus peleas anteriores** (win rate, racha, golpes por minuto, takedowns, control, finish rate, descanso, edad, alcance, Elo, knockdowns propios y recibidos, defensas de striking y derribo, tasa de veces finalizado, Elo promedio de los rivales). Cada pelea genera dos filas: diffs A−B y la espejada B−A, para eliminar el sesgo de esquina roja. | `data/features.csv` |
| `train.py` | `HistGradientBoostingClassifier` con split temporal (test = últimos 2 años, validación = los 2 anteriores), sin calibración: el modelo crudo dio mejor log loss que isotónica y sigmoid. Evalúa sobre peleas únicas con la misma predicción que sirve `predict.py` (promedio de ambas orientaciones), imprime una tabla de confiabilidad por deciles, y el `model.pkl` final se re-entrena con todo el historial. | `model.pkl`, `data/fighter_state.csv` |
| `predict.py` | `predict(a, b) -> (p_a, p_b)`. Predice en las dos orientaciones y promedia, así el resultado no depende del orden. | — |
| `app.py` | UI Streamlit: elegís dos peleadores, ves las probabilidades como barras y, si ingresás la cuota decimal de la casa, la probabilidad implícita (1/cuota) y la diferencia contra el modelo. | — |

## Resultados actuales

Sobre las peleas de los últimos 2 años (1029 peleas únicas, nunca vistas en
entrenamiento), evaluando la predicción desplegada — el promedio de ambas
orientaciones, igual que `predict.py`:

| | log loss | Brier | accuracy |
|---|---|---|---|
| **modelo** | 0.6428 | 0.2254 | **0.6550** |
| moneda | 0.6931 | 0.2500 | 0.5000 |
| mayor Elo | — | — | 0.5588 |

~65% de acierto está en el techo de lo que logran los modelos públicos de UFC. El valor
no está en acertar todo: está en que las probabilidades sean confiables cuando se las
compara contra el mercado.

## Sin leakage

Toda la métrica de arriba depende de que ninguna feature use información posterior a
la pelea. `features.py` lo garantiza por construcción — una sola pasada cronológica
donde el estado de cada peleador se actualiza *después* de emitir la fila — y lo
verifica con un `assert` explícito. Las peleas del mismo día (los torneos de UFC 1-8)
se calculan todas antes de aplicar sus updates.

## Alcance

Solo ganador (no método ni rondas), solo UFC, corre local. Las cuotas se usan
únicamente como comparación manual, nunca como feature: el objetivo es discrepar
del mercado, y un modelo entrenado con odds solo aprende a imitarlo.
