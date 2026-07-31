"""Check del camino que se sirve: `python test_app.py`. Sin framework, solo asserts.

Cubre lo que se rompe en silencio: que la prediccion no dependa del orden de los
peleadores, que el nivel de confianza salga del tramo correcto, y que la app renderice
con y sin cuotas. Requiere `model.pkl` y `data/fighter_state.csv` (los genera train.py).
"""

import betano
import cartelera
import predict

A, B = "Khamzat Chimaev", "Sean Strickland"

# Payload de ESPN recortado: el orden es el real (preliminares primero, main event
# ultimo), con un TBA para descartar, un acento y un debutante.
def _pelea(peso, *nombres):
    return {"type": {"abbreviation": peso},
            "competitors": [{"athlete": {"displayName": n}} for n in nombres]}


CARTELERA = {"events": [{
    "name": "UFC 999: Test", "date": "2026-08-15T21:00Z",
    "competitions": [
        _pelea("Middleweight", "TBA", "Opponent TBA"),
        _pelea("Lightweight", "Kauê Fernandes", "Jalin Turner"),   # ESPN pone el acento
        _pelea("Welterweight", "Nadie De La Nada", B),             # debutante
        _pelea("Middleweight", A, B),                              # main event
    ]}]}


# `window["initial_state"]` de Betano recortado, con la forma real. Betano escribe el
# nombre sin acento donde ESPN lo pone con acento, mete un mercado de rounds que no es
# una pelea, y no cotiza el main event (asi se ejercita "todavia no publico la cuota").
BETANO = {"data": {
    "dropdownList": [
        {"id": "189184", "leagues": [{"url": "/otra/liga/"}]},        # PFL: no es UFC
        {"id": betano.REGION, "leagues": [
            {"url": "/sport/mma/ufc/ufc-999/206397/", "selected": True},  # ya vino
            {"url": "/sport/mma/ufc/ufc-fight-night/205870/"}]},
    ],
    "blocks": [{"events": [
        {"name": "Kaue Fernandes - Jalin Turner", "markets": [
            {"name": "Ganador", "selections": [
                {"id": "1", "name": "Kaue Fernandes", "price": 1.65},
                {"id": "2", "name": "Jalin Turner", "price": 2.25}]},
            {"name": "Total de rounds", "selections": [
                {"id": "3", "name": "Más de 1.5", "price": 1.40},
                {"id": "4", "name": "Menos de 1.5", "price": 2.90}]}]},
        {"name": "Nadie De La Nada - Sean Strickland", "markets": [
            {"name": "Ganador", "selections": [
                {"id": "5", "name": "Nadie De La Nada", "price": 3.10},
                {"id": "6", "name": "Sean Strickland", "price": 1.36}]}]},
    ]}]}}


def check_betano():
    """Parseo de las cuotas y matcheo contra los nombres como los escribe ESPN."""
    t = betano._parsear(BETANO)
    assert len(t) == 3, t                    # dos peleas + el mercado de rounds
    assert t[("jalin turner", "kaue fernandes")] == (2.25, 1.65), t

    # solo las carteleras de UFC, y sin repetir la que ya vino en esta misma pagina
    assert betano._ligas(BETANO) == ["/sport/mma/ufc/ufc-fight-night/205870/"]

    # el acento de ESPN tiene que caer sobre el nombre sin acento de Betano
    assert betano.buscar(t, "Kauê Fernandes", "Jalin Turner") == (1.65, 2.25)
    # y dado vuelta, la cuota tiene que venir dada vuelta
    assert betano.buscar(t, "Jalin Turner", "Kauê Fernandes") == (2.25, 1.65)
    # nombre parecido pero no igual: lo resuelve el fallback difuso
    assert betano.buscar(t, "Kaue Fernandes Jr.", "Jalin Turner") == (1.65, 2.25)
    # pelea que Betano no cotiza -> sin cuota, no una cuota de otra pelea
    assert betano.buscar(t, A, B) is None, betano.buscar(t, A, B)

    # si Betano no responde, la app tiene que seguir andando sin cuotas
    base = betano.BASE
    try:
        betano.BASE = "http://localhost:1"   # se cae al toque, sin salir a la red
        assert betano.cuotas() == {}
    finally:
        betano.BASE = base


def check_simetria():
    """predict(a, b) y predict(b, a) tienen que ser el mismo resultado dado vuelta."""
    modelo, estado = predict.cargar()
    ida = predict.predict(A, B, modelo, estado)
    vuelta = predict.predict(B, A, modelo, estado)
    assert abs(ida["p_a"] - vuelta["p_b"]) < 1e-9, (ida["p_a"], vuelta["p_b"])
    assert abs(ida["p_a"] + ida["p_b"] - 1) < 1e-9

    ida = predict.predict(A, B, modelo, estado, cuotas=(1.35, 3.20))
    vuelta = predict.predict(B, A, modelo, estado, cuotas=(3.20, 1.35))
    for k in ("p_a_mercado", "p_a_con_odds"):
        assert abs(ida[k] + vuelta[k] - 1) < 1e-9, (k, ida[k], vuelta[k])
    assert ida["confianza"] == vuelta["confianza"]


def check_confianza():
    """El nivel sale de |modelo - mercado|, asi que cuotas extremas dan confianza baja."""
    modelo, estado = predict.cargar()
    p = predict.predict(A, B, modelo, estado)["p_a"]
    # una cuota que implica casi lo mismo que el modelo -> alta; la opuesta -> baja
    par = lambda q: (round(1 / q, 2), round(1 / (1 - q), 2))  # noqa: E731
    assert predict.predict(A, B, modelo, estado, cuotas=par(p))["confianza"] == "alta"
    assert predict.predict(A, B, modelo, estado, cuotas=par(1 - p))["confianza"] == "baja"
    assert predict.predict(A, B, modelo, estado)  # sin cuotas no debe explotar
    assert "confianza" not in predict.predict(A, B, modelo, estado)

    # Mismo ganador, magnitud distinta: baja, pero el motivo tiene que aclararlo.
    q = (p + 1) / 2 if p > 0.5 else p / 2  # mismo lado que el modelo, mucho mas extremo
    r = predict.predict(A, B, modelo, estado, cuotas=par(q))
    assert r["confianza"] == "baja" and "coinciden en el ganador" in r["motivo"], r
    assert "coinciden en el ganador" not in predict.predict(
        A, B, modelo, estado, cuotas=par(1 - p))["motivo"]


def check_apuesta():
    """Solo se marca candidata en el tramo medido apostable, y con EV real positivo."""
    modelo, estado = predict.cargar()
    p = predict.predict(A, B, modelo, estado)["p_a"]
    par = lambda q: (round(1 / q, 2), round(1 / (1 - q), 2))  # noqa: E731

    # los dos lados sobrepagados un 12%: el implicito normalizado sigue siendo p (brecha
    # ~0 -> confianza alta) pero el EV es +12%. Es la unica forma de tener las dos cosas.
    qa, qb = round(1.12 / p, 3), round(1.12 / (1 - p), 3)
    r = predict.predict(A, B, modelo, estado, cuotas=(qa, qb))
    assert r["confianza"] == "alta" and r["apuesta"] in ("a", "b"), r
    assert abs(r["ev_a"] - (p * qa - 1)) < 1e-9 and r["ev_a"] > 0.1, r

    # el mismo EV enorme pero con brecha grande no se marca: es justo lo que pierde
    r = predict.predict(A, B, modelo, estado, cuotas=par(1 - p))
    assert r["ev_a"] > 0 and r["apuesta"] is None, r
    # la misma cuota pero con un vig normal del 3% -> confianza alta y ninguna candidata,
    # que es el caso comun: coincidir con la casa no alcanza, hay que ganarle al margen.
    r = predict.predict(A, B, modelo, estado,
                        cuotas=(round(0.97 / p, 3), round(0.97 / (1 - p), 3)))
    assert r["confianza"] == "alta" and r["apuesta"] is None, r


def check_cartelera():
    """Parseo de la cartelera y prediccion de peleas que pueden no ser predecibles."""
    eventos = cartelera._parsear(CARTELERA)
    assert len(eventos) == 1, eventos
    peleas = eventos[0]["peleas"]
    assert eventos[0]["fecha"] == "2026-08-15"
    assert len(peleas) == 3, peleas                     # el TBA no cuenta
    assert (peleas[0]["a"], peleas[0]["b"]) == (A, B)   # main event primero
    assert peleas[0]["peso"] == "Middleweight"

    modelo, estado = predict.cargar()
    # el acento de ESPN tiene que resolver contra el nombre ASCII de ufcstats
    acentuado = cartelera.predecir(peleas[2], modelo, estado)
    assert "error" not in acentuado, acentuado
    assert 0 < acentuado["p_a"] < 1
    # y el debutante no explota: sale sin prediccion
    assert cartelera.predecir(peleas[1], modelo, estado)["error"]
    assert "p_a" not in cartelera.predecir(peleas[1], modelo, estado)


def check_app():
    """La app tiene que renderizar sin excepcion con y sin cuotas."""
    from streamlit.testing.v1 import AppTest

    cartelera.proximas = lambda *a, **k: cartelera._parsear(CARTELERA)  # sin red
    betano.cuotas = lambda: betano._parsear(BETANO)                     # sin red

    for cuotas in ((1.35, 3.20), None):
        at = AppTest.from_file("app.py", default_timeout=120).run()
        at.selectbox[0].set_value(A)
        at.selectbox[1].set_value(B)
        if cuotas:
            at.number_input[0].set_value(cuotas[0])
            at.number_input[1].set_value(cuotas[1])
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        # un nivel de confianza por pelea con cuota: la del matchup si se cargo, mas la
        # unica de la cartelera que Betano cotiza y el modelo puede predecir.
        # (contar solo los de confianza: el box de candidata a valor va aparte y no
        # siempre aparece, depende de la cuota)
        cajas = [c.value for c in (*at.success, *at.warning, *at.error)]
        avisos = sum("Confianza" in c for c in cajas)
        assert avisos == (1 if cuotas else 0) + 1, (cuotas, avisos, cajas)
        # ninguna candidata puede convivir con confianza que no sea alta
        assert not [c for c in cajas if "Candidata" in c and "Confianza alta" not in
                    " ".join(cajas)], cajas
        # las cuotas de la cartelera ya no se tipean: quedan solo las dos del matchup
        assert len(at.number_input) == 2, len(at.number_input)
        assert len(at.info) == 1, len(at.info)
        # un debutante no tiene prediccion, pero su cuota se muestra igual: es lo unico
        # que hay para esa pelea, y esconderla fue un bug
        precios = [c.value for c in at.caption if c.value.startswith("Betano —")]
        assert any("Nadie De La Nada 3.10" in c for c in precios), precios
        assert len(precios) == 2, precios       # el debutante y la pelea predecible


if __name__ == "__main__":
    for check in (check_betano, check_simetria, check_confianza, check_apuesta,
                  check_cartelera,
                  check_app):
        check()
        print(f"ok  {check.__name__}")
