> Handoff doc for task `predictor-ux-dashboard`. Author: Codex. Updated: 2026-08-01 22:33.

# DECISIONS — predictor-ux-dashboard

## Decisiones tomadas

- Ranking humano primero, con peso `(aciertos + 5) / (total + 10)`; modelo y mercado
  actuan como confirmaciones visibles.
- Senal fuerte: al menos dos predictores, apoyo ponderado >= 66.7% y confirmacion del
  modelo o mercado.
- Picks legacy quedan pendientes de revision para no degradar estadisticas corregidas.
- Apuestas reales en CLP, simples y combinadas, separadas del forward test automatico.
- Liquidacion automatica editable para cash-out, anulacion o ajustes de la casa.
- La Cartelera destaca siempre el evento mas cercano, lista los restantes como agenda
  compacta y conserva una sola seleccion activa para no calcular eventos ocultos.
- Las carteleras anteriores salen de resultados locales de UFCStats, no del primer
  avistaje ESPN: asi reflejan las peleas efectivamente disputadas y omiten cancelaciones.

## Open questions for the spec author

Ninguna.
