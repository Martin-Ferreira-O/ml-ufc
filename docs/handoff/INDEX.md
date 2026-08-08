# Handoff index

Registry of active task handoffs under `docs/handoff/<slug>/`. One row per slug —
**update the existing row, never append a duplicate.** `status` ∈ {`todo`,
`in-progress`, `blocked`, `done`}; `depends-on` is a comma-separated list of slugs
that must be `done` first (or `—`). Retire finished slugs with `/archive <slug>`.

## Handoffs

| slug | status | depends-on | updated | note |
|------|--------|------------|---------|------|
| ufc-fight-predictor | done | — | 2026-07-28 | T1-T5 implementados y verificados; pipeline completo corre end-to-end |
| predictor-ux-dashboard | done | ufc-fight-predictor | 2026-08-01 | T1-T7 completos; historial conectado al peso de predictores |
| audit-improvements | done | predictor-ux-dashboard | 2026-08-02 | P0 verificable aplicado; eventos pasados visibles; suite/modelo/QA completos |
| predictor-history-fixes | done | audit-improvements | 2026-08-02 | Historial corregido y verificado; feedback verde/rojo agregado a las picks pasadas |
| fighter-intel | done | predictor-history-fixes | 2026-08-04 | VPS 22/22; Streamlit sincroniza por SSH solo ante metadata nueva |
| betting-probability-optimization | done | fighter-intel | 2026-08-07 | consultoría + capa de decisión/verificación: devig medido, pool no concluyente, Kelly, gate preregistrado cerrado |
| ai-fight-prediction-integration | in-progress | betting-probability-optimization | 2026-08-08 | consenso IA por pelea: un prompt con todo lo del repo, cohorte propia sin tocar el ledger del gate |
