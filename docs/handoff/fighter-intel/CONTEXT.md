> Handoff `fighter-intel`. Autor: Codex. Actualizado: 2026-08-02 22:23 -04.
> Spec escrito contra `025a282` en la rama `fighter-intel`; authored in place by Codex.

# CONTEXT — fighter-intel

## Tarea

Agregar al predictor existente un bot que, una vez al dia, tome la proxima cartelera
UFC de ESPN, investigue a todos sus peleadores en fuentes publicas, conserve las
evidencias y use un LLM barato para resumir hechos que puedan influir en la pelea
(peso, lesion, enfermedad, reemplazo, viaje/visa, campamento y declaraciones).

El resultado debe ser auditable: cada hallazgo cita las evidencias que realmente vio el
modelo. La valoracion contextual no modifica la probabilidad del modelo ni se presenta
como recomendacion de apuesta hasta que exista un forward test que la valide.

## Read first

- `ufc/datos/cartelera.py`: fuente de eventos y forma actual de los matchups.
- `ufc/rutas.py`: todas las rutas persistentes deben quedar ancladas al repo.
- `app.py` y `ufc/ui/comunes.py`: router y patrones de cache/UI.
- `README.md`: contrato de instalacion y operacion actual.

## Restricciones

- Una cartelera de 14 peleas implica 28 checks diarios; el run debe demostrar cobertura.
- La VPS no implica GPU. El camino recomendado usa RSS/public feeds + Gemini barato;
  NVIDIA NIM gratis solo sirve para prototipo, no para produccion.
- No evadir login, CAPTCHAs ni controles de Instagram/X. Esas plataformas requieren
  acceso oficial o un proveedor autorizado; el MVP admite feeds publicos configurables.
- Toda afirmacion visible debe conservar URL, titulo, fecha y fuente.
- Texto de UI y CLI en espanol.

## Comandos previstos

```sh
.venv/bin/python -m ufc.intel.bot --sin-ia
GEMINI_API_KEY=... .venv/bin/python -m ufc.intel.bot
.venv/bin/python test_intel.py
.venv/bin/streamlit run app.py
```
