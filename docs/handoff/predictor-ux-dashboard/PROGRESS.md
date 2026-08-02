> Handoff doc for task `predictor-ux-dashboard`. Author: Codex. Updated: 2026-08-01 22:33.

# PROGRESS — predictor-ux-dashboard

## Checklist

- [x] T1 — Persistencia y migracion de picks
- [x] T2 — Ranking de predictores
- [x] T3 — Registro y liquidacion de apuestas
- [x] T4 — Navegacion, resumen y carga por clic
- [x] T5 — Pruebas automatizadas y verificacion visual
- [x] T6 — Portada y agenda responsive de eventos
- [x] T7 — Carteleras anteriores conectadas con Predictores

## Work log

- 2026-08-01 22:33 — Codex — Rama y paquete de handoff creados; se preservan los cambios locales previos.
- 2026-08-01 22:52 — Codex — Picks compatibles migran con origen/revision, ranking bayesiano agregado y nuevo registro de apuestas simples/combinadas con liquidacion automatica editable.
- 2026-08-01 23:08 — Codex — Navegacion superior, Resumen, Apuestas, editor historico por clic, terminologia simple y documentacion conectados.
- 2026-08-01 23:20 — Codex — Suite completa pasa; navegador verificado en 1440x1000 y 390x844, incluida seleccion por clic y boleta. Corregida incompatibilidad de number_input detectada en QA.
- 2026-08-01 23:27 — Codex — Descarga robustecida ante cortes transitorios: reintentos, reemplazo atomico, fallback a CSV local y aviso visible en el pipeline.
- 2026-08-01 23:33 — Codex — T6 iniciado: se acuerda portada del proximo evento y agenda compacta con detalle bajo demanda.
- 2026-08-01 23:39 — Codex — T6 completo: seleccion persistente, fallback ante eventos obsoletos y QA en 1440x1000/390x844; suite completa pasa.
- 2026-08-01 23:52 — Codex — T7 iniciado: historial UFCStats y acceso directo para cargar picks/resultados por predictor.
- 2026-08-01 23:53 — Codex — T7 completo: 20 carteleras disputadas, ganadores precargados y enlace que abre el evento en Picks; suite y navegación real verificadas.
- 2026-08-01 23:54 — Codex — Preseleccion historica ajustada a una sola fuente de session_state; verificacion final sin warnings de widgets.
