> Handoff `ai-fight-prediction-integration`. Autor: Claude. Actualizado: 2026-08-08.
> Spec escrito contra `844305d` en la rama `claude/ai-fight-prediction-integration-uu1byb`;
> source plan: `~/.claude/plans/actualmente-tenemos-la-predicci-n-dynamic-kazoo.md`.

# CONTEXT — ai-fight-prediction-integration

## Tarea

Cada pelea tiene hoy dos numeros: la probabilidad del modelo estadistico
(`ufc/modelo/predict.py`, 28 features, sin cuotas) y la del mercado (`devig` sobre Betano
mas el consenso multi-casa de `oddsapi`). Ninguno de los dos ve lo que no esta en una
columna: estilo, contexto de campamento, calidad del layoff, lo que dicen los informes de
inteligencia, o cuanto pesa que un debutante no tenga fila en `fighter_state.csv`.

Agregar una capa que arme **un prompt por pelea** con todo lo que el repo sabe de A y de
B, se lo de a Gemini, y devuelva un veredicto estructurado: pick, probabilidad, los
factores que el modelo no puede catalogar, el mejor argumento en contra de su propio
pick, y si vale la pena apostar.

## Read first

- `ufc/modelo/predict.py`: forma exacta de la prediccion (`p_a`, `factores`, `confianza`).
- `ufc/modelo/features.py`: `FEATURES` (las 28) y por que `fighter_state.csv` tiene los
  valores absolutos que `features.csv` no tiene (guarda diferencias).
- `ufc/intel/analyzer.py`: el patron de LLM del repo — esquema, prompt con reglas duras,
  y `validar()` como capa que nunca confia en la salida del modelo.
- `ufc/registro/predictores.py`: el sistema de picks, precision medida y peso Beta(5, 5)
  al que entra la IA.
- `config/gate.json` y `ufc/modelo/gate.py`: la regla preregistrada de apuesta.
- `README.md`, seccion "De la probabilidad al dinero": por que el EV tiene que venir del
  precio y no del modelo.

## Restricciones

- **Solo datos del repo.** Sin grounding ni busqueda web: barato, reproducible, auditable.
- **La cohorte del gate no se toca.** El usuario decidio que el veredicto de la IA se
  muestre como recomendacion aunque `config/gate.json` este cerrado. Lo que no puede
  pasar es que esas apuestas entren a `data/ledger.csv`: ahi vive la serie de CLV
  preregistrada que lee `gate.estado()`, y mezclarlas la arruinaria. Cohortes separadas.
- **La IA se mide con la misma vara que los humanos.** Entra a `picks.csv` con
  `origen="ia"` para que `aciertos()` y `confiabilidad()` la puntuen, pero fuera del voto
  de `ranking()`: ese mide apoyo humano.
- Una cartelera son ~14 llamadas. La idempotencia por `run_day` es el unico control de
  costo que sobrevive a un boton en Streamlit.
- El prompt tiene que ser inspeccionable sin gastar: `--dry-run` y un `dossier` puro.
