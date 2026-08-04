> Handoff `fighter-intel`. Autor: Codex. Actualizado: 2026-08-02 22:23 -04.
> Spec escrito contra `025a282` en la rama `fighter-intel`; authored in place by Codex.

# PLAN — fighter-intel

## Objetivo

Pipeline diario reproducible y barato: proxima cartelera ESPN -> 1 check por peleador ->
Google News RSS + feeds publicos configurados -> deduplicacion/persistencia SQLite ->
Gemini Flash-Lite (o endpoint OpenAI-compatible) -> informe con valoracion contextual,
confianza y citas -> pestana Streamlit de inteligencia.

## Decisiones de alcance

- **MVP de fuentes:** Google News RSS y RSS/Atom publicos por peleador. Sirve para
  noticias, entrevistas, canales de YouTube via feed y sitios propios.
- **Redes cerradas:** Instagram/X no se scrapean directamente. Se integraran solo por
  API oficial/proveedor autorizado si su costo y terminos lo permiten.
- **LLM:** Gemini 2.5 Flash-Lite por defecto, salida JSON estructurada. Un adaptador
  OpenAI-compatible permite NVIDIA, DeepSeek, Groq u Ollama sin cambiar el pipeline.
- **Valoracion:** entero -5..+5 de impacto contextual sobre el peleador, separado del
  modelo de ganador. Cero significa sin evidencia material, no "todo esta perfecto".
- **Orquestacion:** CLI idempotente + systemd timer/cron. Hermes Agent es opcional y no
  aporta suficiente para justificar otra plataforma operativa en el MVP.

## Pasos

1. **Persistencia y fuentes.** Crear `ufc/intel/` con store SQLite, recolector de Google
   News RSS, lector de feeds configurables y normalizacion/deduplicacion de evidencias.
2. **Analisis.** Prompt y esquema estrictos, validacion de IDs citados, adaptador Gemini
   y OpenAI-compatible, contabilidad de tokens y modo sin IA.
3. **Orquestador.** Elegir el siguiente evento en <=8 dias, comprobar una vez por dia a
   cada peleador, reanudar parciales, imprimir cobertura y salir !=0 si falta alguien.
4. **UI.** Pestana `Inteligencia` que solo lee la base local y presenta cobertura,
   frescura, valoraciones, hallazgos y links; nunca una apuesta automatica.
5. **Operacion/documentacion.** README, plantilla de fuentes, unidades systemd y estudio
   de viabilidad/costos con fuentes oficiales y supuestos explicitos.
6. **Pruebas.** Fixtures offline para RSS, persistencia/idempotencia, validacion anti-
   alucinacion, cobertura 28/28 y AppTest basico.

## Verification

| Comando | Senal de pass |
|---|---|
| `.venv/bin/python test_intel.py` | exit 0; fuentes/store/analisis/cobertura/UI verdes sin red |
| `.venv/bin/python -m ufc.intel.bot --evento-fixture tests/fixtures/intel_event.json --sin-red --sin-ia --force` | exit 0; imprime cobertura `28/28`; crea `data/intel.db` |
| repetir el comando sin `--force` | exit 0; `28 omitidos`; no duplica checks del dia |
| `.venv/bin/python test_app.py` | suite existente verde |
| `.venv/bin/streamlit run app.py` | abre la pestana Inteligencia y muestra el ultimo run local |

## Criterio de terminado

No basta una demo con un peleador: deben estar verificados los 28 checks, el modo de
reanudacion, la trazabilidad de citas, el costo estimado y el despliegue diario en VPS.
