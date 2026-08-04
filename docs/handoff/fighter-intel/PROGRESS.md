> Handoff `fighter-intel`. Autor: Codex. Actualizado: 2026-08-03 22:59 -04.
> Spec escrito contra `025a282` en la rama `fighter-intel`; authored in place by Codex.

# PROGRESS — fighter-intel

## Checklist

- [x] Persistencia SQLite + Google News/RSS/Atom + X API oficial
- [x] Analisis estructurado Gemini y OpenAI-compatible, con validacion de citas
- [x] Orquestador diario con cobertura/reanudacion
- [x] Pestana Streamlit
- [x] Operacion y estudio de costos/viabilidad
- [x] Activacion final: llamada real a Gemini y timer instalado en la VPS del usuario

## Work log

- 2026-08-02 22:23 — Codex — rama y handoff creados; arquitectura delimitada despues
  de inspeccionar el predictor existente y documentacion oficial actual.
- 2026-08-02 22:38 — Codex — backend, UI, proveedores, X oficial, systemd y estudio de
  costos implementados. `test_intel.py` (7 tests), `test_app.py` (16 checks), compileall,
  CLI fixture 28/28 y reanudacion 28 omitidos pasan. Run real: ESPN encontro UFC Fight
  Night Gamrot vs Salkilld, 22/22 peleadores y 176 evidencias archivadas. No hay
  `GEMINI_API_KEY` en este entorno: la llamada real y el deploy del timer quedan como
  activacion, no se declaran verificados.
- 2026-08-02 22:49 — Codex — descubrimiento de identidad endurecido: JSON-LD oficial de
  UFC primero, Wikidata solo como candidato no recolectable. Auditoria real de Gamrot vs
  Salkilld: 18/22 peleadores con perfil oficial, 28 perfiles (18 Instagram, 10 X), cero
  errores y ningun falso perfil activado.
- 2026-08-02 22:56 — Codex — Google Search grounding opt-in implementado con
  `grounding_supports`, deduplicacion, tokens acumulados y degradacion a RSS con
  advertencia. `test_intel.py` pasa 11 tests, `test_app.py` sus 16 checks, el fixture
  vuelve a cubrir 28/28; compileall y `git diff --check` tambien pasan. La llamada real
  sigue esperando `GEMINI_API_KEY`.
- 2026-08-02 23:03 — Codex — salud operativa agregada: `--status` exige run de menos
  de 36 h, cobertura total e IA; `--db` aisla diagnosticos. Timer con tres ventanas
  diarias documentadas por systemd e idempotentes. `test_intel.py` pasa 12 tests. Se
  retiraron solo las 28 filas del fixture que habian quedado en `data/intel.db`; la
  corrida real conserva 22 checks y 176 enlaces de evidencia.
- 2026-08-03 22:59 — Codex — desplegado en `34.176.71.67:/opt/ml-ufc` con env root-only,
  Python 3.14 y systemd. La API retiro 2.5 Flash-Lite para usuarios nuevos; migrado y
  probado contra `gemini-3.1-flash-lite`. Corrida real completa: 22/22, 174 evidencias,
  18/22 identidades oficiales y una alerta `-2` con tres citas. Se agregaron reintentos
  para el limite free de 15 RPM, filtro de paginas de apuestas, barrera contra inferir
  salud por ausencia, dia operativo Santiago y `--resumen`. Timer activo con tres
  ventanas 09:15/10:15/11:15 America/Santiago; corrida manual systemd y `--status`
  devuelven exit 0.
