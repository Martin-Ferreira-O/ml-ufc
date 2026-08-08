> Handoff `ai-fight-prediction-integration`. Autor: Claude. Actualizado: 2026-08-08.
> Spec escrito contra `844305d` en la rama `claude/ai-fight-prediction-integration-uu1byb`;
> source plan: `~/.claude/plans/actualmente-tenemos-la-predicci-n-dynamic-kazoo.md`.

# PLAN — ai-fight-prediction-integration

## Arquitectura

Paquete `ufc/ia/`, en paralelo a `ufc/intel/` y con la misma forma. No va en `ufc/modelo/`
a proposito: `modelo/` no importa de `datos/` ni de `registro/`, y esto necesita los tres.

| modulo | responsabilidad |
|---|---|
| `ufc/ia/dossier.py` | arma y renderiza el payload por pelea. Puro, sin red |
| `ufc/ia/analista.py` | esquema JSON, `validar()`, y la llamada a Gemini |
| `ufc/ia/store.py` | `data/ia_consenso.csv` + `data/ia_informes/`, idempotencia |
| `ufc/ia/consenso.py` | orquestador y CLI |
| `ufc/ia/evaluar.py` | medicion de la cohorte IA |
| `ufc/intel/reintentos.py` | extraido de `bot.py` para poder reusarlo sin arrastrarlo |

## Decisiones que sostienen el diseno

1. **Las features van en valor absoluto.** `features.csv` guarda diferencias porque es lo
   que come el modelo, pero un LLM no puede leer "elo +84" sin saber si son 1500 contra
   1416 o 2100 contra 2016. La tabla sale de `fighter_state.csv`, con la diferencia como
   tercera columna. `GRUPOS` tiene un `assert` contra `features.FEATURES` para que
   promover una candidata no deje una feature sin etiqueta.
2. **La calibracion del modelo va dentro del prompt, leida del manifest.** Accuracy, log
   loss y los `coincidence_buckets` salen de `model.pkl`, no escritos a mano: un numero
   hardcodeado envejece con el primer re-entrenamiento y despues miente con autoridad.
   El objetivo es que la IA sepa que una discrepancia grande con la casa no es valor.
3. **`validar()` nunca confia.** Clampea `p_a`, coerce enums, descarta razones sin fuente,
   corrige la pick si contradice su propia probabilidad, y **calcula el EV en Python** con
   la cuota real en vez de leerlo de la respuesta. La regla de "nombra el lado, el precio
   y la casa" se aplica en codigo: un `"si"` que no la cumple baja a `"mirar"` solo.
4. **Dos archivos de persistencia.** `data/ia_consenso.csv` es numerico y se versiona
   (forward test irreproducible hacia atras). `data/ia_informes/` es prosa de un LLM sobre
   personas reales y queda fuera de git, por el mismo motivo escrito en `.gitignore` para
   `data/intel.db`.
5. **La IA es el tercer confirmador, no un humano falso.** `ranking()` la saca del voto y
   gana `ia_confirma` / `ia_eligio` al lado de `modelo_confirma` y `mercado_confirma`.
   `aciertos()` y `confiabilidad()` si la incluyen. `revisado=True` porque ese campo
   significa "definitiva y puntuable", no "la miro una persona" — se documento en el
   docstring de `predictores.guardar`.
6. **`st.fragment` para el boton.** Sin el, cada tick de progreso vuelve a correr el
   script entero: otra consulta a Betano y otra a The Odds API por cada pelea analizada.

## Verification

```sh
# Los checks nuevos de la capa de IA, sin red ni API
.venv/bin/python -c "import test_app as t; [c() or print('ok ', c.__name__) for c in \
  (t.check_ia_dossier, t.check_ia_dossier_sin_cuota, t.check_ia_validar, \
   t.check_ia_validar_incoherente, t.check_ia_validar_apuesta, t.check_ia_store, \
   t.check_ia_predictores, t.check_ia_evaluar)]"

# Sin regresiones en lo que ya existia
.venv/bin/python test_intel.py          # 20 tests OK
.venv/bin/python test_app.py            # requiere model.pkl y fighter_state.csv

# El prompt completo de una pelea real, sin gastar un peso
.venv/bin/python -m ufc.ia.consenso --dry-run --pelea 1

# Con la key: una pelea, para ver costo y salida reales
GEMINI_API_KEY=... .venv/bin/python -m ufc.ia.consenso --pelea 1
GEMINI_API_KEY=... .venv/bin/python -m ufc.ia.consenso --pelea 1   # "ya estaban de hoy"

# La cohorte del gate quedo intacta
git diff --stat data/ledger.csv          # vacio
.venv/bin/python -m ufc.ia.evaluar
```

Senal de pase: los ocho `check_ia_*` imprimen `ok`, `test_intel.py` sigue en 20 OK,
`--dry-run` imprime un prompt con las ocho secciones, y `git diff data/ledger.csv` es
vacio despues de correr `evaluar`.
