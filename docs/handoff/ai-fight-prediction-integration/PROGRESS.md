> Handoff `ai-fight-prediction-integration`. Autor: Claude. Actualizado: 2026-08-08.
> Spec escrito contra `844305d` en la rama `claude/ai-fight-prediction-integration-uu1byb`.

# PROGRESS — ai-fight-prediction-integration

## Checklist

- [x] `ufc/intel/reintentos.py` extraido de `bot.py`, con re-export
- [x] `ufc/ia/dossier.py`: `armar`, `render`, `huella`, `GRUPOS` con assert de sincronia
- [x] `ufc/ia/analista.py`: esquema, `validar()`, `precios()`, clase `Analista`
- [x] `ufc/ia/store.py`: CSV numerico versionado + informes JSON fuera de git
- [x] `ufc/ia/consenso.py` y su CLI (`--dry-run`, `--pelea`, `--force`, `--status`)
- [x] Integracion en `predictores`: `ORIGEN_IA`, `ia_confirma`, fuera del voto humano
- [x] `ufc/ia/evaluar.py`: acierto, log loss comparado y CLV de la cohorte IA
- [x] UI: `comunes.ia_veredictos/ia_badge/ia_apuesta/ia_tarjeta`, fragment en Cartelera
- [x] 8 `check_ia_*` en `test_app.py`, todos en verde
- [x] `.gitignore`, paquete de handoff y README
- [ ] Corrida real contra la API de Gemini (no hay `GEMINI_API_KEY` en este entorno)
- [ ] Verificacion con el pipeline completo (ver DECISIONS: wikipedia bloqueada)

## Work log

- 2026-08-08 — Claude — rama y handoff creados. Explorado el pipeline completo antes de
  disenar: modelo, devig, oddsapi, gate, intel y predictores.
- 2026-08-08 — Claude — `reintentos.py` extraido; `test_intel.py` sigue en 20 OK.
- 2026-08-08 — Claude — `dossier.py` completo. Un prompt real mide ~7.700 caracteres
  (~1.900 tokens), bastante menos de los ~7k que se habian estimado: una cartelera de 14
  peleas sale por centavos.
- 2026-08-08 — Claude — `analista.py`, `store.py`, `consenso.py` y `evaluar.py`.
  `validar()` verificado a mano en los cinco caminos de degradacion.
- 2026-08-08 — Claude — integracion en `predictores` verificada: con dos humanos de
  acuerdo y la IA en contra, `predictores == 2`, `senal == "Señal fuerte"`,
  `ia_confirma == False`, y la IA aparece igual en `aciertos()`.
- 2026-08-08 — Claude — UI conectada con `st.fragment` para que el progreso del analisis
  no vuelva a disparar las consultas de precios de toda la cartelera.
- 2026-08-08 — Claude — 8 checks nuevos en verde; `check_predictores`, `check_gate`,
  `check_pool`, `check_devig`, `check_staking`, `check_apuestas`, `check_oddsapi`,
  `check_betano` y `check_fetch` siguen pasando.
