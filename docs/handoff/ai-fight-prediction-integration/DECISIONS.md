> Handoff `ai-fight-prediction-integration`. Autor: Claude. Actualizado: 2026-08-08.
> Spec escrito contra `844305d` en la rama `claude/ai-fight-prediction-integration-uu1byb`.

# DECISIONS — ai-fight-prediction-integration

## El gate y la cohorte

El usuario eligio explicitamente que el veredicto de apuesta de la IA se muestre como
recomendacion en la app, por encima de `config/gate.json`, que hoy esta cerrado y dice
que "nada del codigo puede autorizar una apuesta por fuera de esto". Se le planteo la
objecion antes de implementar y ratifico la decision, asi que se implemento asi.

Lo que si se protegio, porque es una cuestion de integridad de datos y no de criterio:
las apuestas que sugiere la IA **no entran a `data/ledger.csv`**. Ese archivo es la
cohorte preregistrada cuya serie de `ev_al_cierre` lee `gate.estado()`; mezclarle una
segunda poblacion de apuestas destruiria la unica medicion de CLV que el proyecto viene
acumulando. La IA escribe en `data/ia_consenso.csv` y se mide con `ufc/ia/evaluar.py`.
`check_ia_evaluar` verifica que `ledger.csv` quede byte a byte igual.

`predict.apuestas_automaticas()` sigue devolviendo `False` y el badge "Sin apuestas
automaticas" de la sidebar no se toco.

## `revisado=True` para las picks de la IA

`aciertos()` y `ranking()` filtran por `picks["revisado"]`. Con `False`, la pick de la IA
no se puntuaria nunca — o sea, lo contrario de lo que se quiere. Se resolvio precisando
que el campo significa "definitiva y puntuable" y no "la miro una persona": lo que
protege es la ambiguedad de leer un tilde verde sobre una foto, que en la IA no existe
porque la pick se produce con los nombres exactos de la cartelera. Quedo documentado en
el docstring de `predictores.guardar`.

La contrapartida es que la IA sale del voto de `ranking()` y de `comparar()`: si votara,
"Consenso humano" podria salir con un humano en minoria. Entra como tercer confirmador
(`ia_confirma`), al lado del modelo y del mercado.

## Modelo por defecto

`gemini-3.5-flash` y no el `gemini-3.1-flash-lite` de `intel`. El de intel resume
evidencias sueltas; este sintetiza ocho secciones y ademas tiene que construir un
argumento contra su propia conclusion. Se puede bajar con `UFC_IA_MODELO`. Medido: un
prompt real son ~1.900 tokens de entrada, asi que una cartelera entera cuesta centavos.

## Desvios respecto del plan aprobado

- El plan estimaba ~7k tokens de entrada por pelea; medido son ~1.900. No cambia nada del
  diseno, pero el parrafo de costos del README refleja el numero real.
- Se agrego `ia_eligio` a `ranking()` ademas de `ia_confirma`: sin el, la UI no puede
  distinguir "la IA no coincide" de "la IA no opino todavia".
- Se agrego `check_ia_store_sin_informe` fundido dentro de `check_ia_store` en vez de como
  check aparte: comparte todo el setup y separarlo era duplicar treinta lineas.

## Limitacion del entorno de desarrollo

El sandbox donde se implemento bloquea `en.wikipedia.org` por politica de red, asi que
`python -m ufc.datos.wiki` no corre. Sin el, `reemplazo` y `peso_no_dado` quedan 100% NaN
y `python -m ufc.modelo.train` falla en el binning de HistGB
(`window shape cannot be larger than input array shape`). Consecuencia: **no hay
`model.pkl` ni `fighter_state.csv` en ese entorno**, y `test_app.py` corta en
`check_simetria` — lo hacia tambien antes de este cambio.

Por eso los ocho `check_ia_*` se disenaron para correr con fixtures y sin red, y se
verificaron uno por uno. Lo que **no** se pudo verificar y queda pendiente para un
entorno con el pipeline completo:

- `python -m ufc.ia.consenso --dry-run` contra una cartelera real de ESPN.
- Una llamada real a la API de Gemini (no hay `GEMINI_API_KEY` en el entorno).
- El render de la pestana Cartelera con veredictos cargados.

No se declaran como hechos.

## Open questions for the spec author

Ninguna abierta.
