# Handoff index

Registry of active task handoffs under `docs/handoff/<slug>/`. One row per slug —
**update the existing row, never append a duplicate.** `status` ∈ {`todo`,
`in-progress`, `blocked`, `done`}; `depends-on` is a comma-separated list of slugs
that must be `done` first (or `—`). Retire finished slugs with `/archive <slug>`.

## Handoffs

| slug | status | depends-on | updated | note |
|------|--------|------------|---------|------|
| ufc-fight-predictor | done | — | 2026-07-28 | T1-T5 implementados y verificados; pipeline completo corre end-to-end |
