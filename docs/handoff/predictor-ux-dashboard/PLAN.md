> Handoff doc for task `predictor-ux-dashboard`. Author: Codex. Updated: 2026-08-01 22:33.
> Spec written against commit `2ad904f`; source plan: agreed in Codex plan mode.

# PLAN — predictor-ux-dashboard

1. Migrar picks con `origen` y `revisado`; excluir legacy sin revisar del ranking.
2. Agregar ranking bayesiano y senales transparentes por pelea.
3. Persistir apuestas simples/combinadas, liquidacion automatica y overrides reales.
4. Cambiar tabs por navegacion superior y sumar Resumen/Apuestas.
5. Reemplazar tablas editables por seleccion de ganador con un clic y editor historico.
6. Simplificar terminologia y mantener Seguimiento del modelo separado.
7. Reemplazar el selector de cartelera por portada del proximo evento, agenda compacta
   y seleccion persistente que solo calcula el evento activo.
8. Agregar carteleras anteriores desde resultados locales y enlazarlas con la carga de
   picks para que el rendimiento historico de cada predictor alimente su peso.

## Verification

```sh
.venv/bin/python test_app.py
.venv/bin/streamlit run app.py --server.port 8501
```

Pass: todos los checks imprimen `ok`, la app responde sin excepciones y las vistas de
escritorio/movil no presentan solapamientos ni controles truncados.
