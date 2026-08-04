# Viabilidad del bot de inteligencia UFC

Evaluacion vigente al **2 de agosto de 2026**. Los precios y cuotas cambian: verificar
los enlaces oficiales antes de contratar o subir limites.

## Veredicto

Es viable en una VPS sin GPU. Para una cartelera normal, el volumen es pequeno: 14
peleas, 28 peleadores y 28 informes diarios. La arquitectura mas barata no es un agente
general navegando sitios, sino un pipeline determinista que archiva fuentes y usa un LLM
solo para clasificar y resumir.

El limite serio no es el precio del modelo, sino la cobertura legal y estable de redes
sociales. Noticias, entrevistas, YouTube y X se pueden cubrir; leer cuentas ajenas de
Instagram de forma general no tiene un camino oficial equivalente y no debe resolverse
evadiendo login o anti-bot.

## Arquitectura implementada

1. ESPN entrega la proxima cartelera y los 28 nombres.
2. El resolver de identidad lee primero el JSON-LD de la ficha oficial de UFC y cachea
   los perfiles `sameAs`. En la cartelera real Gamrot vs Salkilld resolvio perfiles
   oficiales para **18/22** peleadores (18 Instagram y 10 X); cuatro quedaron
   honestamente sin cobertura. Wikidata solo propone candidatos y nunca activa una
   fuente sin validacion oficial.
3. Cada peleador se revisa una vez al dia en Google News RSS (endpoint publico, sin SLA)
   y en sus feeds RSS/Atom configurados. Los canales de YouTube exponen feeds RSS.
4. X es opcional y usa exclusivamente `GET /2/tweets/search/recent`, con bearer token y
   handle oficial resuelto o configurado. No se scrapea HTML de X ni Instagram.
5. Gemini Google Search grounding es opt-in. Una busqueda por peleador agrega solo los
   chunks web que `grounding_supports` vincula a una URL; la prosa generada durante la
   busqueda no se considera una fuente. Un fallo deja advertencia y no invalida el RSS.
6. SQLite conserva URL, titulo, publisher, fecha, snippet, run y relacion con el
   peleador. La URL evita duplicados.
7. El LLM recibe evidencias numeradas E1...En y devuelve JSON. El codigo elimina todo
   hallazgo que cite un ID inexistente y calcula el score desde los impactos restantes.
8. Streamlit solo lee la base local. systemd ofrece tres ventanas diarias; la
   idempotencia hace que las posteriores no repitan costo si la primera termino. El CLI
   `--status` falla si el ultimo run tiene mas de 36 horas, cobertura parcial o no uso IA.

## Costos comparados

Supuesto conservador para inferencia: **28 llamadas/dia**, cada una con 4.000 tokens de
entrada y 500 de salida; 840 llamadas/mes, 3,36 M tokens de entrada y 0,42 M de salida.

| opcion | precio oficial relevante | costo estimado del supuesto | evaluacion |
|---|---:|---:|---|
| Gemini 3.1 Flash-Lite | $0,25/M entrada, $1,50/M salida; free tier | **~$1,47/mes** pagado; $0 mientras alcance el free tier | recomendada: modelo estable mas barato con JSON schema y grounding |
| DeepSeek V4 Flash | $0,14/M entrada cache miss, $0,28/M salida | **~$0,59/mes** | buen fallback chino; endpoint OpenAI-compatible ya soportado |
| Groq free plan | limites diarios/por minuto segun modelo y organizacion | $0 mientras alcance la cuota | fallback rapido, sin garantia de capacidad gratuita para produccion |
| NVIDIA NIM hosted | gratis solo para prototipo/desarrollo | $0 en pruebas | no usar como produccion gratuita; NVIDIA exige AI Enterprise para produccion |
| Modelo local/Ollama | sin costo por token | electricidad/RAM/CPU de la VPS | viable con un modelo pequeno, pero normalmente peor y mas lento sin GPU |

Fuentes oficiales: [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing),
[DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing),
[Groq rate limits](https://console.groq.com/docs/rate-limits) y
[NVIDIA NIM FAQ](https://docs.api.nvidia.com/nim/docs/product).

Gemini 3 no ofrece Google Search grounding en el tier gratuito. Con facturacion activa,
los primeros 5.000 requests mensuales compartidos entre modelos Gemini 3 no tienen cargo
y luego cuestan $14/1.000. Una busqueda por cada uno de los 28 peleadores representa
aproximadamente 840 al mes, dentro de esa franquicia, pero se mantiene opt-in para no
convertir silenciosamente una key gratuita en una integracion facturable.
[Precios oficiales de Gemini](https://ai.google.dev/gemini-api/docs/pricing).

## Costo de fuentes

- Google News RSS y feeds RSS/Atom: sin costo por request, pero Google News RSS no es una
  API con SLA. Se necesita monitoreo y eventualmente un proveedor de noticias licenciado.
- X cobra actualmente **$0,005 por Post leido** y deduplica dentro del mismo dia UTC.
  Con cinco posts nuevos por peleador por semana: 28 x 5 x 4 x $0,005 = **$2,80/mes**.
  Se debe fijar spending limit. [Precios oficiales de X](https://docs.x.com/x-api/getting-started/pricing)
  y [recent search](https://docs.x.com/x-api/posts/search/introduction).
- YouTube Data API asigna cuota por operacion y una cuota diaria por proyecto; para los
  uploads propios de un canal el MVP usa el feed RSS y no consume esa API.
  [Cuotas oficiales](https://developers.google.com/youtube/v3/getting-started#quota).
- Brave Search cuesta $5/1.000 requests e incluye $5 mensuales, pero sus terminos
  actuales prohiben retener resultados sin acuerdo. No encaja con un archivo historico
  de evidencias y por eso no se implemento.

## Hermes Agent

[Hermes Agent](https://nousresearch.net/hermes-agent/) es tecnicamente capaz: corre en
Linux, programa jobs, navega, busca y usa varios modelos. Puede ser atractivo si despues
se quiere Telegram/Discord, browser automation supervisada o skills generales.

No es la base recomendada para esta fase:

- duplica scheduler, memoria, dashboard, proveedor de busqueda y seguridad que este repo
  ya tiene;
- un browser agent expuesto a paginas no confiables agrega riesgo de prompt injection;
- hace mas dificil demostrar que exactamente los 28 peleadores se revisaron y que cada
  afirmacion viene de una URL archivada;
- no resuelve los terminos ni el costo de Instagram/X: el agente sigue necesitando acceso
  autorizado a las plataformas.

Se puede agregar despues como capa de notificacion, sin reemplazar la base SQLite ni el
pipeline determinista.

## Que significa la valoracion

`-5..+5` es **impacto contextual**, no puntos porcentuales ni EV. Ejemplos: una lesion
confirmada puede ser negativa; haber dado el peso normalmente no justifica un gran score
positivo. El score no entra a `predict.py` y la UI dice expresamente que no recomienda
apostar.

Antes de convertirlo en señal se necesita forward test: congelar cada informe y cuota,
medir si las alertas anticipan cancelaciones, cambios de linea/CLV y resultados, y estimar
intervalos de confianza fuera de muestra. Sin eso, la utilidad correcta es priorizar la
lectura humana.

## Riesgos y controles

| riesgo | control actual | pendiente razonable |
|---|---|---|
| noticia falsa o repetida | citas obligatorias, URLs unicas, certeza rumor/probable/confirmado | reputacion por dominio y confirmacion cruzada |
| alucinacion del LLM | se descartan IDs no existentes; score recalculado | evaluacion humana de una muestra semanal |
| homonimos | ficha oficial UFC + coincidencia exacta de nombre; Wikidata no activa fuentes | identidad versionada por fighter ID cuando ESPN lo exponga |
| fuente caida | el run queda partial y systemd lo reintenta al dia siguiente | alerta de salud/Telegram |
| Instagram sin recoleccion | se identifica la cuenta oficial, pero no se scrapea ni se finge lectura | proveedor autorizado o carga manual de links/capturas |
| valor de apuesta no probado | score separado del predictor y disclaimer en UI | forward test de varios meses |
