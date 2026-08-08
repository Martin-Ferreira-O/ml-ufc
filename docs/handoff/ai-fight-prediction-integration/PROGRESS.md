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
- [x] Corrida real contra la API de Gemini: 12 veredictos de `gemini-3.5-flash` sobre
      UFC Fight Night: Gamrot vs Salkilld
- [x] El informe guarda el input (`_prompt` y `_dossier`), no solo la huella sha256
- [x] `ufc/datos/resultados.py`: resultados en vivo desde el mismo endpoint de ESPN
- [x] `ufc/tipster.py`: bot de Telegram, avisos en vivo y comandos
- [x] `deploy/`: `ufc-tipster.service` + CI/CD (`ufc-deploy.sh` y su timer)
- [x] 3 checks nuevos (`check_resultados_espn`, `check_tipster_mensajes`,
      `check_tipster_allowlist`); 38 en verde
- [ ] Verificacion con el pipeline completo (ver DECISIONS: wikipedia bloqueada)
- [ ] Cohorte `v2` con resultados: las 4 filas resueltas son de la cohorte sin
      `prompt_v` y `evaluar()` las excluye a proposito

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
- 2026-08-08 — Claude — el informe guarda ahora `_prompt` y `_dossier`. Con la huella
  sola una pick vieja solo se podia explicar reconstruyendo el dossier con el codigo de
  hoy, que ya no es el que corrio. Van con `_` adelante y en el mismo dict porque
  `informe()` devuelve eso tal cual y `comunes.ia_tarjeta` indexa las claves del
  veredicto directamente: anidarlo en `{"veredicto": ...}` rompia a todos sus lectores.
- 2026-08-08 — Claude — hallazgo que definio el diseno del seguimiento en vivo: el
  endpoint de ESPN que `cartelera.py` ya usaba devuelve tambien las peleas terminadas
  (`competitors[].winner`, `status.period`, `displayClock`, y el metodo en
  `details[].type.text`). `_parsear` las descartaba con el filtro `state == "pre"`.
  Verificado contra 315 peleas de abril a agosto: los unicos metodos que aparecen son
  `Kotko`, `Submission` y `Decision`. O sea que no hacia falta ni scrapear ufcstats
  (tiene challenge anti-bot) ni esperar el refresh diario del mirror.
- 2026-08-08 — Claude — `tipster.py` sin libreria de Telegram: la Bot API es HTTP plano
  y `getUpdates` con timeout hace de long polling y de reloj del bucle, asi que tampoco
  hay scheduler. No carga el modelo: `p_a_modelo` y `p_a_mercado` ya estan congelados
  como columnas en `ia_consenso.csv`, que es justo para lo que se guardaron, y
  `data/model.pkl` no existe en la VPS.
- 2026-08-08 — Claude — dos footguns cerrados durante la verificacion: `--seco` escribia
  `resultados.csv` y el estado (un flag que se llama seco y deja rastro no sirve para
  probar nada), y `sincronizar()` podia pisar picks sin pushear. Ahora se desactiva sola
  donde existe `model.pkl`: esa es la maquina que los genera, no la que los recibe.
- 2026-08-08 — Claude — verificado en vivo contra Gamrot vs Salkilld mientras se peleaba:
  tres resueltas, la IA 3 de 3 (Miles Johns por KO R1, Juliana Miller por sumision R2,
  Carol Foro). El mensaje de Canuto-Foro salio sin metodo porque ESPN no lo publico, que
  es el comportamiento buscado: no se deduce del reloj.
