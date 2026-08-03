> Handoff doc for task `audit-improvements`. Author: Codex. Updated: 2026-08-02 16:27.
> Spec written against commit `a84877b`; source plan: `posibles_mejoras.md`.

# PLAN — audit-improvements

1. Separar visualmente eventos proximos y pasados en Predictores, conservar seleccion
   estable y probar que una cartelera historica puede abrirse para picks/ganadores.
2. Corregir serving temporal: fecha exacta del evento y circunstancias
   si/no/desconocido; impedir comparaciones retrospectivas con estado futuro.
3. Cambiar el default economico a “sin apuesta / seguimiento experimental” y aclarar que
   la coincidencia modelo-mercado no esta validada para cuotas actuales.
4. Corregir taxonomia de metodos y duraciones antiguas; usar exposiciones observadas para
   que un estadistico ausente no se convierta en rendimiento cero.
5. Versionar bundle, datos y metricas con manifiesto reproducible y escritura atomica;
   agregar auditoria append-only de revisiones de picks.
6. Endurecer medicion con bootstrap clusterizado por evento y checks de invariantes.
7. Ejecutar suite, reconstruir features/modelo y verificar visualmente escritorio/movil.

## Verification

```sh
.venv/bin/python test_app.py
.venv/bin/python -m ufc.modelo.features
.venv/bin/python -m ufc.modelo.train
.venv/bin/python test_app.py
.venv/bin/streamlit run app.py --server.port 8501
```

Pass: todos los checks imprimen `ok`; el entrenamiento guarda un bundle con manifiesto;
Predictores permite cambiar a Pasados y abrir una cartelera historica sin usar estado
futuro; la app no muestra ninguna recomendacion automatica de apuesta.

