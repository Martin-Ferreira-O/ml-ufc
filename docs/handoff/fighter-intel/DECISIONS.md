> Handoff `fighter-intel`. Autor: Codex. Actualizado: 2026-08-04 08:35 -04.
> Spec escrito contra `025a282` en la rama `fighter-intel`; authored in place by Codex.

# DECISIONS — fighter-intel

## Decisiones

- **No usar Hermes Agent como base.** Es viable y ya tiene scheduling, busqueda y
  navegacion, pero este repo ya posee ESPN, Streamlit, persistencia y una VPS objetivo.
  Incorporarlo duplicaria scheduler, memoria, seguridad y observabilidad. Queda como
  alternativa futura para canales de mensajeria o browser workflows supervisados.
- **Gemini 3.1 Flash-Lite por defecto.** El 2026-08-03 la API rechazo 2.5 Flash-Lite
  para usuarios nuevos. La lista real de la cuenta y la documentacion oficial confirman
  3.1 Flash-Lite estable, free tier, JSON estructurado y el menor costo pagado entre los
  modelos actuales capaces. Grounding en Gemini 3 requiere facturacion, aunque incluye
  5.000 requests mensuales antes del cobro por busqueda.
- **No llamar "senal de apuesta" a un score no validado.** Se llama valoracion
  contextual y se mantiene fuera de `predict.py`. Primero se debe acumular evidencia y
  medir si anticipa movimiento de linea, cancelaciones o resultados.
- **No scraping directo de Instagram/X.** Un bot sostenible no debe depender de evadir
  autenticacion o medidas anti-bot. Se aceptan noticias indexadas, feeds publicos y APIs
  autorizadas.
- **X si, pero solo por API oficial y opt-in.** La API actual cobra por Post leido y
  recent search esta disponible a desarrolladores. Se agrega `kind=x` con handle manual
  y `X_BEARER_TOKEN`; sin token un source configurado falla visiblemente, no finge haberlo
  revisado. Instagram sigue sin adaptador general para cuentas ajenas.
- **SQLite antes que CSV.** Hay relaciones many-to-many entre checks y evidencias,
  reanudacion de runs y unicidad de URL; SQLite resuelve todo con stdlib en una VPS.
- **La UFC es autoridad de identidad; Wikidata no.** Las URLs `sameAs` del JSON-LD
  oficial se pueden mostrar y activar. Una coincidencia exacta de Wikidata se conserva
  solo como candidato porque en la auditoria real aparecio una cuenta de otra persona.
- **Grounding es descubrimiento, no autoridad.** Se archivan exclusivamente chunks web
  conectados por `grounding_supports`; la respuesta sintetica de la busqueda no se usa
  como evidencia. Es opt-in para conservar portabilidad y control de cuota.
- **Tres ventanas, una sola factura.** systemd admite varios `OnCalendar`; se programan
  tres oportunidades diarias. La unicidad por dia/peleador hace que las posteriores
  omitan checks completos y reintenten solo fallos. `--status` no considera saludable
  una recoleccion `--sin-ia`, porque el objetivo incluye filtrado y score del LLM.
- **El free tier se regula, no se paraleliza.** La cuenta desplegada expone 15 requests
  por minuto para Gemini 3.1 Flash-Lite. Los errores 429/5xx respetan `RetryInfo` y se
  reintentan con backoff; la reanudacion diaria cubre cualquier fallo residual.
- **Un listing de cuotas no es inteligencia del peleador.** DraftKings, Robinhood,
  Coinbase y patrones equivalentes se excluyen del corpus de noticias. Ademas, una
  barrera determinista quita frases que infieren salud o logistica desde ausencia de
  reportes, incluso si el LLM desobedece el prompt.
- **Dia operativo en Santiago.** Tanto `OnCalendar` como la clave idempotente diaria
  usan `America/Santiago`; los timestamps de auditoria permanecen en UTC.
- **La UI local sincroniza artefactos por SSH, no sirve SQLite remotamente.** No se abre
  otro puerto ni se exponen datos en HTTP. Streamlit consulta `--metadata-json` cada
  cinco minutos y habilita la descarga solo si `(event_date, run_day, updated_at)` es
  posterior a la copia local. Descarga primero a temporales, valida SQLite/CSV, conserva
  `.backup` y recien entonces reemplaza los tres archivos locales.
- **La vista muestra un solo dia de revision.** SQLite conserva el historial diario,
  pero `ultimo_evento()` filtra el `run_day` mas reciente; sumar todos los dias de una
  cartelera duplicaria peleadores y falsearia la cobertura despues del segundo run.

## Open questions for the spec author

Ninguna por ahora. Las credenciales opcionales se configuran al desplegar; el pipeline
debe ser verificable offline sin ellas.
