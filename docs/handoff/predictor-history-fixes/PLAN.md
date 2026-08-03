> Handoff doc for task `predictor-history-fixes`. Author: Codex. Updated: 2026-08-02 19:48.
> Spec written against commit `a84877b`; source plan: agreed in Codex plan mode.

# PLAN — predictor-history-fixes

1. Persistir el selector de predictor al ocultarse la vista Picks y mostrar confirmacion
   visible despues del rerun de guardado.
2. Consolidar ganadores con precedencia UFCStats y fallback manual, normalizando nombres
   de evento/peleadores y evitando doble conteo.
3. Deshabilitar ganadores oficiales y mantener edicion solo para resultados sin fuente
   oficial.
4. Agregar carteleras distintas al rendimiento historico y mostrar la columna en
   Predictores y Resumen.
5. Cubrir estado, precedencia, normalizacion, conteos reales y UI con pruebas.

## Verification

```sh
.venv/bin/python test_app.py
.venv/bin/streamlit run app.py --server.port 8501
```

Pass: todos los checks imprimen `ok`; FaceOff MMA muestra 47/65 en 5 carteleras y
CagePropHet 11/14 en 1; el selector persiste, el guardado confirma y los resultados
oficiales no se pueden editar.

