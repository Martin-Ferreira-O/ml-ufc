> Handoff doc for task `audit-improvements`. Author: Codex. Updated: 2026-08-02 16:27.

# DECISIONS — audit-improvements

## Decisiones tomadas

- Esta iteracion implementa correcciones P0/P1 que pueden verificarse con los datos del
  repositorio. Modelos de supervivencia, backfills pagos, video, IDs de fuente ausentes y
  señales que exigen forward test quedan explicitamente fuera: no se simula su evidencia.
- Una cartelera historica sirve para cargar y evaluar picks humanas, pero no para ejecutar
  hoy el modelo sobre el estado final de los peleadores: eso seria leakage retrospectivo.
- La app deja de producir “candidatas” automaticas hasta que exista un gate prospectivo
  preregistrado cuyo intervalo conservador sea positivo.
- No se puede crear retroactivamente un holdout intacto: el manifiesto etiqueta la ventana
  historica como desarrollo ya observado y fija la nueva cohorte prospectiva despues de la
  fecha maxima de datos.
- El bootstrap clusterizado se aplica al runner de bloques y al experimento de metodo. Las
  metricas descriptivas principales siguen reportandose para continuidad, sin presentarse
  como un nuevo gate causal.
- El modelo de supervivencia/rounds, IDs fuente estables, snapshots normalizados de props,
  estado prospectivo de debutantes y modelos regionales requieren nuevas fuentes o una
  fase experimental separada. No se marcaron como completados por crear solo scaffolding.

## Open questions for the spec author

Ninguna.
