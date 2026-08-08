"""El veredicto de la IA: esquema de salida, la llamada a Gemini y la validacion.

La parte importante de este modulo no es la llamada, es `validar`. Un LLM devuelve JSON
bien formado y contenido inventado con la misma cara, asi que nada de lo que dice se
guarda tal cual: la probabilidad se clampea, los enums se coercen, las razones sin fuente
se caen y la pick se corrige si contradice su propia probabilidad. Es el mismo criterio
de `ufc/intel/analyzer.validar`, que descarta los hallazgos sin cita.

**La IA no opina de apuestas y no puede.** El prompt que arma `dossier.render` es ciego
al mercado, asi que la unica pregunta que responde es deportiva: quien gana, con que
probabilidad, o si la pelea esta demasiado pareja como para elegir (`veredicto: parejo`).
El EV lo calcula `ev_contra_mercado` despues, en Python, cruzando esa probabilidad con la
cuota real. Es la misma aritmetica que antes, corrida del lado correcto de la frontera:
una lectura deportiva que nunca vio el precio es lo unico que se puede medir contra el.
"""

import json
import os

from ufc.ia import dossier as dossier_mod
from ufc.intel import reintentos
from ufc.modelo import gate
from ufc.registro import predictores

# El de `predictores.extraer`, no el flash-lite de `intel`: aquello resume evidencias
# sueltas y esto tiene que leer un historial entero y ademas construir un contraargumento
# contra si mismo. Se puede bajar con UFC_IA_MODELO si el costo molesta.
MODELO = "gemini-3.5-flash"

# Sin "modelo", "mercado" ni "tipsters": nada de eso llega al prompt, asi que una razon
# que dijera venir de ahi seria una alucinacion y se descarta sola en `_razones`.
FUENTES = ["record", "estadistica", "estilo", "historial", "inteligencia", "contexto"]
CERTEZAS = ["especulativo", "probable", "solido"]
PESOS = ["alto", "medio", "bajo"]
CONFIANZAS = ["baja", "media", "alta"]
VEREDICTOS = ["definido", "parejo"]
METODOS = ["ko", "sub", "dec"]

MAX_RAZONES = 8
MAX_FACTORES = 6
MAX_TEXTO = 600

ESQUEMA = {
    "type": "object",
    "required": ["veredicto", "pick", "p_a", "confianza", "razones", "contra"],
    "properties": {
        # "parejo" es la abstencion: la IA se puede plantar en "no hay diferencia clara"
        # en vez de inventar un favorito. `pick` y `p_a` siguen siendo obligatorios
        # porque la probabilidad es lo que despues se mide con log loss.
        "veredicto": {"type": "string", "enum": VEREDICTOS},
        "pick": {"type": "string", "enum": ["a", "b"]},
        "p_a": {"type": "number"},
        "confianza": {"type": "string", "enum": CONFIANZAS},
        "metodo_probable": {"type": "string", "enum": METODOS},
        "razones": {"type": "array", "items": {
            "type": "object",
            "required": ["texto", "fuente", "peso"],
            "properties": {
                "texto": {"type": "string"},
                "fuente": {"type": "string", "enum": FUENTES},
                "peso": {"type": "string", "enum": PESOS},
            }}},
        "factores_no_modelables": {"type": "array", "items": {
            "type": "object",
            "required": ["titulo", "explicacion", "favorece", "certeza"],
            "properties": {
                "titulo": {"type": "string"},
                "explicacion": {"type": "string"},
                "favorece": {"type": "string", "enum": ["a", "b"]},
                "certeza": {"type": "string", "enum": CERTEZAS},
            }}},
        "contra": {"type": "string"},
        "banderas": {"type": "array", "items": {"type": "string"}},
    },
}

# La key vive en .env y `predictores` ya trae el lector: duplicarlo seria tener dos
# formas distintas de leer el mismo archivo.
predictores.cargar_env()


def hay_api():
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))


def modelo_por_defecto():
    return os.getenv("UFC_IA_MODELO", MODELO)


def _texto(valor, tope=MAX_TEXTO):
    return str(valor or "").strip()[:tope]


def _enum(valor, permitidos, default=None):
    v = str(valor or "").strip().lower()
    return v if v in permitidos else default


def precios(dossier):
    """-> {'a': (cuota, casa), 'b': (cuota, casa)} con el mejor precio ejecutable.

    Prefiere el mejor precio del consenso multi-casa y cae a la cuota de la casa unica.
    Lo consume `ev_contra_mercado`, no el prompt: este dato nunca llega al LLM.
    """
    mercado = dossier.get("mercado") or {}
    salida = {"a": (None, None), "b": (None, None)}
    if not mercado.get("disponible"):
        return salida
    if mercado.get("cuota_a") and mercado.get("cuota_b"):
        salida["a"] = (mercado["cuota_a"], "Betano")
        salida["b"] = (mercado["cuota_b"], "Betano")
    c = mercado.get("consenso") or {}
    for lado in ("a", "b"):
        cuota, casa = c.get(f"mejor_{lado}"), c.get(f"casa_{lado}")
        # solo si mejora: el consenso puede venir de una tanda mas vieja que Betano
        if cuota and (salida[lado][0] is None or cuota > salida[lado][0]):
            salida[lado] = (cuota, casa or "consenso")
    return salida


def _ev_minimo():
    """El piso de EV de la regla preregistrada. 0.02 si no se puede leer el archivo."""
    try:
        config = gate.leer() or {}
        return float(config.get("regla", {}).get("ev_minimo", 0.02))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.02


def _razones(brutas):
    salida = []
    for r in brutas if isinstance(brutas, list) else []:
        if not isinstance(r, dict):
            continue
        fuente = _enum(r.get("fuente"), FUENTES)
        texto = _texto(r.get("texto"))
        if not fuente or not texto:
            continue  # una razon sin fuente no se puede auditar; se cae
        salida.append({"texto": texto, "fuente": fuente,
                       "peso": _enum(r.get("peso"), PESOS, "medio")})
        if len(salida) == MAX_RAZONES:
            break
    return salida


def _factores(brutos):
    salida = []
    for f in brutos if isinstance(brutos, list) else []:
        if not isinstance(f, dict):
            continue
        favorece = _enum(f.get("favorece"), ["a", "b"])
        titulo = _texto(f.get("titulo"), 200)
        if not favorece or not titulo:
            continue
        salida.append({"titulo": titulo,
                       "explicacion": _texto(f.get("explicacion")),
                       "favorece": favorece,
                       "certeza": _enum(f.get("certeza"), CERTEZAS, "especulativo")})
        if len(salida) == MAX_FACTORES:
            break
    return salida


VACIO = {"lado": None, "cuota": None, "casa": None, "ev": None, "cumple_regla": False}


def ev_contra_mercado(veredicto, dossier):
    """-> {lado, cuota, casa, ev, cumple_regla}. La IA no vio nada de esto.

    Cruza la probabilidad que la IA saco del dato deportivo con el mejor precio real. Que
    esto corra despues del veredicto y no adentro del prompt es todo el punto: un numero
    que se calcula sobre una lectura ciega al mercado se puede medir contra el mercado.

    Una pelea que la IA llamo pareja no se apuesta: si ella misma dice que no separa a
    los dos, su probabilidad no sostiene ningun EV.
    """
    if veredicto.get("veredicto") == "parejo":
        return dict(VACIO)
    lado = veredicto["pick"]
    cuota, casa = precios(dossier).get(lado, (None, None))
    if cuota is None:
        return dict(VACIO)
    p_a = veredicto["p_a"]
    ev = (p_a if lado == "a" else 1 - p_a) * cuota - 1
    return {"lado": lado, "cuota": cuota, "casa": casa, "ev": ev,
            "cumple_regla": ev >= _ev_minimo()}


def validar(bruto, dossier):
    """-> veredicto usable. Levanta ValueError si la respuesta no sirve para nada.

    Lo unico irrecuperable es una pick que no es "a" ni "b": sin eso no hay veredicto.
    Todo el resto se corrige, se recorta o se degrada, y lo que se corrigio queda escrito
    en `banderas` para que se pueda ver despues.
    """
    if not isinstance(bruto, dict):
        raise ValueError("La IA no devolvio un objeto JSON.")
    pick = _enum(bruto.get("pick"), ["a", "b"])
    if not pick:
        raise ValueError(f"La IA devolvio una pick invalida: {bruto.get('pick')!r}")

    banderas = [_texto(x, 240) for x in (bruto.get("banderas") or [])
                if isinstance(x, str) and x.strip()][:6]

    try:
        p_a = float(bruto.get("p_a"))
    except (TypeError, ValueError):
        p_a = 0.5 if pick == "a" else 0.5
        banderas.append("No devolvio una probabilidad usable; se asumio 50%.")
    p_a = min(0.99, max(0.01, p_a))

    # Si la probabilidad contradice la pick, manda la probabilidad: es el numero que
    # despues se mide con log loss, y una pick que no se le parece no se puede puntuar.
    if (p_a > 0.5) != (pick == "a") and abs(p_a - 0.5) > 1e-9:
        banderas.append(f"Incoherente: eligio '{pick}' con p_a={p_a:.2f}. Se corrigio la "
                        "pick para que siga a la probabilidad.")
        pick = "a" if p_a > 0.5 else "b"

    veredicto = _enum(bruto.get("veredicto"), VEREDICTOS, "definido")
    # Decir "pareja" y despues poner 82% es querer las dos cosas. Manda la palabra, que es
    # la que se muestra y la que decide si la pick se registra; la probabilidad se corrige
    # hacia el centro y queda escrito.
    if veredicto == "parejo" and abs(p_a - 0.5) > 0.15:
        banderas.append(f"Dijo que la pelea esta pareja pero puso p_a={p_a:.2f}. Se "
                        "acerco al 50% para que la probabilidad diga lo mismo.")
        p_a = 0.5 + (0.15 if p_a > 0.5 else -0.15)

    return {
        "veredicto": veredicto,
        "pick": pick,
        "p_a": p_a,
        "confianza": _enum(bruto.get("confianza"), CONFIANZAS, "baja"),
        "metodo_probable": _enum(bruto.get("metodo_probable"), METODOS),
        "razones": _razones(bruto.get("razones")),
        "factores_no_modelables": _factores(bruto.get("factores_no_modelables")),
        "contra": _texto(bruto.get("contra")),
        "banderas": banderas,
    }


def _usage(response):
    usage = getattr(response, "usage_metadata", None)
    return {"prompt_tokens": getattr(usage, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(usage, "candidates_token_count", 0) or 0}


class Analista:
    """Gemini con salida estructurada, una pelea por llamada."""

    nombre = "gemini"

    def __init__(self, api_key=None, modelo=None):
        from google import genai

        if not (api_key or hay_api()):
            raise ValueError("Falta GEMINI_API_KEY. Usa --dry-run para ver el prompt "
                             "sin llamar a la API.")
        self.modelo = modelo or modelo_por_defecto()
        # El cliente vive en un atributo y no en una expresion encadenada: si lo recoge
        # el garbage collector a mitad del request, la SDK tira "client has been closed".
        self.cliente = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY")
                                    or os.getenv("GOOGLE_API_KEY"))

    def analizar(self, dossier):
        """-> (veredicto validado, {'prompt_tokens', 'output_tokens'})."""
        from google.genai import types

        respuesta = reintentos.con_reintentos(lambda: self.cliente.models.generate_content(
            model=self.modelo,
            contents=dossier_mod.render(dossier),
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_json_schema=ESQUEMA,
                http_options=types.HttpOptions(timeout=60_000),
            ),
        ), label="consenso")
        return validar(json.loads(respuesta.text), dossier), _usage(respuesta)
