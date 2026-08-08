"""Recorre una cartelera, arma un prompt por pelea y guarda el veredicto de la IA.

Uso:
    python -m ufc.ia.consenso --dry-run --pelea 1   # el prompt, sin gastar un peso
    GEMINI_API_KEY=... python -m ufc.ia.consenso    # la cartelera entera

Una pelea se analiza una vez por dia (`run_day`). Correr esto tres veces seguidas, o
abrir la app veinte veces, no dispara ninguna llamada extra: la idempotencia es el unico
control de costo que sobrevive a un boton en Streamlit.
"""

import argparse
import concurrent.futures
import datetime
import json
import sys

from ufc import nombres
from ufc.datos import betano, cartelera, oddsapi
from ufc.ia import analista, dossier as dossier_mod, store
from ufc.intel import store as intel_store
from ufc.modelo import predict
from ufc.registro import predictores

# El nombre con el que la IA entra al ranking de predictores. Es una constante y no un
# literal suelto porque `evaluar` y la UI tienen que poder excluirla del voto humano.
IA_PREDICTOR = "IA (Gemini)"


def hoy():
    return datetime.date.today().isoformat()


def _intel_del_evento(evento):
    """-> {clave_normalizada: check} si el ultimo informe es de ESTA cartelera.

    `intel.store.ultimo_evento` devuelve el ultimo run, sea de la cartelera que sea.
    Pegarle los informes de otro evento a este seria peor que no tener ninguno.
    """
    try:
        informe = intel_store.ultimo_evento()
    except Exception:  # la base puede no existir en un clon nuevo
        return {}
    if not informe:
        return {}
    mismo = (nombres.normalizar(informe.get("event_name") or "")
             == nombres.normalizar(evento.get("evento") or ""))
    if not mismo and str(informe.get("event_date"))[:10] != str(evento.get("fecha"))[:10]:
        return {}
    return {nombres.normalizar(c["fighter"]): c for c in informe.get("checks", [])}


def contexto(evento, *, con_red=True):
    """Todo lo que se consulta UNA vez por cartelera, no una vez por pelea.

    Catorce peleas serian catorce consultas a Betano y catorce a The Odds API si esto
    viviera adentro del loop.
    """
    modelo, estado = predict.cargar()
    tabla = betano.cuotas() if con_red else {}
    multi = oddsapi.cuotas() if con_red else {}
    picks = predictores.leer(evento["evento"])
    if len(picks):
        # La IA no puede ver su propia pick anterior como si fuera la de un tercero: se
        # estaria citando a si misma y su "consenso" dejaria de serlo.
        picks = picks[picks["predictor"] != IA_PREDICTOR]
    confiabilidad = predictores.confiabilidad()
    precision = {r["predictor"]: {"acierto": r["acierto"], "total": r["total"]}
                 for _, r in confiabilidad.iterrows()} if len(confiabilidad) else {}
    return {"modelo": modelo, "estado": estado, "betano": tabla, "multi": multi,
            "picks": picks, "precision": precision,
            "intel": _intel_del_evento(evento) if con_red else {}}


def dossier_de(pelea, evento, ctx, indice, total):
    """-> el dossier de una pelea, con todo lo que este disponible enganchado."""
    cuotas = betano.buscar(ctx["betano"], pelea["a"], pelea["b"]) if ctx["betano"] else None
    multi = oddsapi.buscar(ctx["multi"], pelea["a"], pelea["b"]) if ctx["multi"] else None
    if multi:
        # `buscar` no trae el EV: lo calcula `valor`, que es el que excluye del consenso
        # a la casa que ofrece el precio y descuenta la dispersion entre casas.
        multi = dict(multi, valor=oddsapi.valor(multi))
    prediccion = cartelera.predecir(pelea, ctx["modelo"], ctx["estado"], cuotas,
                                    evento["fecha"])
    metodo = predict.metodo(ctx["modelo"], *cartelera.contexto(pelea["peso"], indice == 0))
    picks = ctx["picks"]
    suyas = picks[(picks["a"] == pelea["a"]) & (picks["b"] == pelea["b"])] \
        if len(picks) else picks
    return dossier_mod.armar(
        pelea, prediccion, evento, cuotas=cuotas, consenso=multi, metodo=metodo,
        estado=ctx["estado"], intel=ctx["intel"],
        picks=suyas.to_dict("records") if len(suyas) else [],
        precision=ctx["precision"], manifest=ctx["modelo"].get("manifest"),
        indice=indice, total=total)


def sincronizar_picks(evento, con_metodo=True):
    """Vuelca las picks de la IA de este evento a `picks.csv`, todas de una.

    `predictores.guardar` reemplaza TODAS las filas de un predictor en un evento, asi que
    guardar pelea por pelea borraria las anteriores. Se arma la lista completa desde el
    registro propio y se escribe una sola vez.
    """
    guardados = store.veredictos(evento["evento"])
    filas = []
    for (a, b), fila in guardados.items():
        if fila["pick"] not in {"a", "b"}:
            continue
        p_a = float(fila["p_a_ia"])
        filas.append([IA_PREDICTOR, evento["evento"], evento["fecha"], a, b,
                      fila["pick"], (fila["metodo_probable"] or "") if con_metodo else "",
                      None, p_a if fila["pick"] == "a" else 1 - p_a])
    if filas:
        # revisado=True: el campo protege contra picks leidas de una imagen que nadie
        # confirmo, y esta se produce con los nombres exactos de la cartelera. Sin esto
        # `aciertos()` no la puntuaria nunca, que es justo lo contrario de lo que se quiere.
        predictores.guardar(filas, evento["evento"], IA_PREDICTOR, origen="ia",
                            revisado=True, revisor="ufc.ia.consenso")
    return len(filas)


def _resumen_pelea(pelea, veredicto):
    quien = pelea[veredicto["pick"]]
    a = veredicto["apuesta"]
    if a["vale_la_pena"] == "si":
        apuesta = f"APOSTAR {pelea[a['lado']]} a {a['cuota']:.2f} en {a['casa']} " \
                  f"(EV {a['ev']:+.1%})"
    elif a["vale_la_pena"] == "mirar":
        apuesta = "mirar"
    else:
        apuesta = "sin apuesta"
    return (f"{quien} {veredicto['p_a'] if veredicto['pick'] == 'a' else 1 - veredicto['p_a']:.1%}"
            f" · confianza {veredicto['confianza']} · {apuesta}")


def ejecutar(evento, *, ctx=None, provider=None, force=False, run_day=None,
             max_workers=4, indices=None, progreso=None):
    """Analiza las peleas pendientes de la cartelera. -> dict con el resultado."""
    ctx = ctx or contexto(evento)
    run_day = run_day or hoy()
    modelo_ia = getattr(provider, "modelo", analista.modelo_por_defecto())
    peleas = list(enumerate(evento["peleas"]))
    if indices is not None:
        peleas = [(i, p) for i, p in peleas if i in set(indices)]

    pendientes, omitidas = [], 0
    for i, pelea in peleas:
        if not force and store.ya_analizada(evento["evento"], pelea["a"], pelea["b"],
                                            run_day):
            omitidas += 1
            continue
        pendientes.append((i, pelea))

    hechas, errores = [], []

    def trabajo(par):
        i, pelea = par
        d = dossier_de(pelea, evento, ctx, i, len(evento["peleas"]))
        veredicto, usage = provider.analizar(d)
        return i, pelea, d, veredicto, usage

    if pendientes and provider is not None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futuros = {pool.submit(trabajo, par): par for par in pendientes}
            for futuro in concurrent.futures.as_completed(futuros):
                i, pelea = futuros[futuro]
                try:
                    _, pelea, d, veredicto, usage = futuro.result()
                except Exception as exc:
                    # Una pelea que falla no se lleva puesta la cartelera: en el proximo
                    # run se reintenta sola, porque no quedo guardada.
                    errores.append((pelea, exc))
                    print(f"ERROR {pelea['a']} vs {pelea['b']}: {exc}", flush=True)
                    continue
                fila = store.fila_desde(veredicto, d, run_day=run_day,
                                        modelo_ia=modelo_ia, usage=usage)
                store.guardar(fila, veredicto)
                hechas.append((pelea, veredicto, usage))
                print(f"OK  {pelea['a']} vs {pelea['b']} — "
                      f"{_resumen_pelea(pelea, veredicto)}", flush=True)
                if progreso:
                    progreso(len(hechas) + len(errores), len(pendientes), pelea)

    return {"evento": evento["evento"], "analizadas": len(hechas), "omitidas": omitidas,
            "errores": len(errores), "pendientes": len(pendientes),
            "prompt_tokens": sum(u.get("prompt_tokens", 0) for _, _, u in hechas),
            "output_tokens": sum(u.get("output_tokens", 0) for _, _, u in hechas)}


def _elegir_evento(eventos, n):
    if not eventos:
        raise ValueError("No hay carteleras anunciadas (o ESPN no respondio).")
    if n is None:
        return eventos[0]
    if not 1 <= n <= len(eventos):
        raise ValueError(f"Elegi un evento entre 1 y {len(eventos)}.")
    return eventos[n - 1]


def _imprimir_status():
    r = store.resumen()
    if not r:
        print("Todavia no se analizo ninguna cartelera con IA.")
        return False
    print(f"{r['evento']} — {r['fecha_evento']} — run {r['run_day']} — {r['modelo_ia']}")
    print(f"  {r['peleas']} peleas analizadas · {r['apostar_si']} con apuesta · "
          f"{r['apostar_mirar']} para mirar")
    print(f"  tokens: {r['prompt_tokens']} de entrada, {r['output_tokens']} de salida")
    return True


def argumentos(argv=None):
    p = argparse.ArgumentParser(description="Consenso de IA pelea por pelea")
    p.add_argument("--evento", type=int, help="numero de evento (1 = el mas proximo)")
    p.add_argument("--pelea", type=int, action="append",
                   help="solo esta pelea (1 = main event). Se puede repetir")
    p.add_argument("--force", action="store_true",
                   help="reanaliza aunque ya se haya corrido hoy")
    p.add_argument("--dry-run", action="store_true",
                   help="imprime el prompt y no llama a la API")
    p.add_argument("--status", action="store_true", help="resumen de la ultima corrida")
    p.add_argument("--sin-picks", action="store_true",
                   help="no inyecta las picks en picks.csv")
    p.add_argument("--modelo", help=f"modelo de Gemini (default: {analista.MODELO})")
    p.add_argument("--max-workers", type=int, default=4)
    return p.parse_args(argv)


def main(argv=None):
    args = argumentos(argv)
    try:
        if args.status:
            return 0 if _imprimir_status() else 1

        evento = _elegir_evento(cartelera.proximas(), args.evento)
        total = len(evento["peleas"])
        indices = None
        if args.pelea:
            indices = [n - 1 for n in args.pelea]
            fuera = [n for n in args.pelea if not 1 <= n <= total]
            if fuera:
                raise ValueError(f"Esta cartelera tiene {total} peleas; pediste {fuera}.")

        print(f"{evento['evento']} — {evento['fecha']} — {total} peleas", flush=True)
        ctx = contexto(evento)

        if args.dry_run:
            for i, pelea in enumerate(evento["peleas"]):
                if indices is not None and i not in set(indices):
                    continue
                d = dossier_de(pelea, evento, ctx, i, total)
                texto = dossier_mod.render(d)
                print(f"\n{'=' * 78}\n{texto}")
                print(f"[huella {dossier_mod.huella(d)} · {len(texto)} chars · "
                      f"~{len(texto) // 4} tokens]", flush=True)
            return 0

        provider = analista.Analista(modelo=args.modelo)
        resultado = ejecutar(evento, ctx=ctx, provider=provider, force=args.force,
                             max_workers=max(1, args.max_workers), indices=indices)
        if not args.sin_picks:
            n = sincronizar_picks(evento)
            print(f"picks.csv: {n} picks de '{IA_PREDICTOR}' para este evento")
        print(f"\n{resultado['analizadas']} analizadas · {resultado['omitidas']} ya "
              f"estaban de hoy · {resultado['errores']} con error")
        print(f"tokens: {resultado['prompt_tokens']} de entrada, "
              f"{resultado['output_tokens']} de salida")
        return 1 if resultado["errores"] else 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
