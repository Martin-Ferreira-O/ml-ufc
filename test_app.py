"""Check del camino que se sirve: `python test_app.py`. Sin framework, solo asserts.

Cubre lo que se rompe en silencio: que la prediccion no dependa del orden de los
peleadores, que el nivel de confianza salga del tramo correcto, y que la app renderice
con y sin cuotas. Requiere `model.pkl` y `data/fighter_state.csv` (los genera train.py).
"""

from ufc.datos import betano, cartelera, fetch, oddsapi
from ufc.modelo import features, predict, settlement, train

A, B = "Khamzat Chimaev", "Sean Strickland"

# Payload de ESPN recortado: el orden es el real (preliminares primero, main event
# ultimo), con un TBA para descartar, un acento y un debutante.
def _pelea(peso, *nombres, estado=None):
    c = {"type": {"abbreviation": peso},
         "competitors": [{"athlete": {"displayName": n}} for n in nombres]}
    return c | {"status": {"type": {"state": estado}}} if estado else c


CARTELERA = {"events": [{
    "name": "UFC 999: Test", "date": "2026-08-15T21:00Z",
    "competitions": [
        _pelea("Flyweight", A, B, estado="post"),                  # ya se peleo
        _pelea("Middleweight", "TBA", "Opponent TBA"),
        _pelea("Lightweight", "Kauê Fernandes", "Jalin Turner"),   # ESPN pone el acento
        _pelea("Welterweight", "Nadie De La Nada", B),             # debutante
        _pelea("Middleweight", A, B),                              # main event
    ]}]}


def check_fetch():
    """Una caida de GitHub conserva CSVs buenos y solo bloquea si falta uno esencial."""
    import pathlib
    import tempfile

    class Respuesta:
        content = b"columna\nvalor\n"

        @staticmethod
        def raise_for_status():
            return None

    raw, get, sleep = fetch.RAW, fetch.requests.get, fetch.time.sleep
    try:
        with tempfile.TemporaryDirectory() as d:
            fetch.RAW = pathlib.Path(d)
            fetch.requests.get = lambda *a, **k: Respuesta()
            fetch.main()
            archivos = [fetch.RAW / f"{n}.csv" for n in [*fetch.CSVS, fetch.ODDS[1]]]
            assert all(f.read_bytes() == Respuesta.content for f in archivos), archivos

            def sin_red(*args, **kwargs):
                raise fetch.requests.ConnectionError("DNS no disponible")

            fetch.requests.get = sin_red
            fetch.time.sleep = lambda *a: None
            fetch.main()  # todas las copias locales permiten continuar
            archivos[-1].unlink()
            fetch.main()  # odds es opcional incluso sin copia local

            archivos[0].unlink()
            try:
                fetch.main()
            except RuntimeError as exc:
                assert fetch.CSVS[0] in str(exc) and "copia local" in str(exc), exc
            else:
                raise AssertionError("un CSV esencial ausente debe detener el pipeline")
    finally:
        fetch.RAW, fetch.requests.get, fetch.time.sleep = raw, get, sleep


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


def check_circunstancia():
    """Reemplazo y peso no dado: bajan al que los tiene, y no rompen la simetria.

    Es lo que se rompe en silencio si alguien reordena `predict.CIRCUNSTANCIA` o
    `features.FEATURES`: la prediccion sigue saliendo, con el peso en la feature
    equivocada. El signo del efecto esta medido (39% y 41% de winrate contra 50%).
    """
    modelo, estado = predict.cargar()
    normal = predict.predict(A, B, modelo, estado, circ_a=(0.0, 0.0),
                             circ_b=(0.0, 0.0))["p_a"]
    for i, etiqueta in enumerate(predict.CIRCUNSTANCIA):
        circ = tuple(1.0 if j == i else 0.0 for j in range(len(predict.CIRCUNSTANCIA)))
        ceros = (0.0, 0.0)
        peor = predict.predict(A, B, modelo, estado, circ_a=circ,
                               circ_b=ceros)["p_a"]
        mejor = predict.predict(A, B, modelo, estado, circ_a=ceros,
                                circ_b=circ)["p_a"]
        assert peor < normal < mejor, (etiqueta, peor, normal, mejor)
        # espejar la pelea y la circunstancia tiene que dar exactamente el complemento
        inv = predict.predict(B, A, modelo, estado, circ_a=ceros,
                              circ_b=circ)["p_a"]
        assert abs(peor - (1 - inv)) < 1e-9, (etiqueta, peor, inv)
    # Sin verificar no equivale a confirmar "no": el camino NaN existe y es estable.
    desconocido = predict.predict(A, B, modelo, estado)
    assert 0 < desconocido["p_a"] < 1


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
    """El EV puntual se sigue midiendo, pero nunca activa una apuesta automatica."""
    modelo, estado = predict.cargar()
    p = predict.predict(A, B, modelo, estado)["p_a"]
    par = lambda q: (round(1 / q, 2), round(1 / (1 - q), 2))  # noqa: E731

    # los dos lados sobrepagados un 12%: el implicito normalizado sigue siendo p (brecha
    # ~0 -> confianza alta) pero el EV es +12%. Es la unica forma de tener las dos cosas.
    qa, qb = round(1.12 / p, 3), round(1.12 / (1 - p), 3)
    r = predict.predict(A, B, modelo, estado, cuotas=(qa, qb))
    assert r["confianza"] == "alta" and r["apuesta"] is None, r
    assert r["seguimiento"] in ("a", "b") and "sin apuesta" in r["estado_apuesta"], r
    assert abs(r["ev_a"] - (p * qa - 1)) < 1e-9 and r["ev_a"] > 0.1, r

    # el mismo EV enorme pero con brecha grande no se marca: es justo lo que pierde
    r = predict.predict(A, B, modelo, estado, cuotas=par(1 - p))
    assert r["ev_a"] > 0 and r["apuesta"] is None, r
    # la misma cuota pero con un vig normal del 3% -> confianza alta y ninguna candidata,
    # que es el caso comun: coincidir con la casa no alcanza, hay que ganarle al margen.
    r = predict.predict(A, B, modelo, estado,
                        cuotas=(round(0.97 / p, 3), round(0.97 / (1 - p), 3)))
    assert r["confianza"] == "alta" and r["apuesta"] is None, r


def check_integridad_metodologica():
    """Taxonomia, reloj antiguo, clusters y manifiesto reproducible."""
    assert settlement.canonical_method("TKO - Doctor's Stoppage") == "ko"
    assert settlement.method_market_class("DQ") is None
    assert settlement.moneyline_result("D/D", "a") == "void"
    assert settlement.moneyline_result("W/L", "a") == "win"
    assert features._fight_minutes({"TIME": "2:30", "ROUND": 2,
                                    "TIME FORMAT": "1 Rnd + OT (12-3)"}) == 14.5
    assert features._fight_minutes({"TIME": "9:00", "ROUND": 1,
                                    "TIME FORMAT": "No Time Limit"}) == 9.0
    d = __import__("numpy").array([1.0, 1.0, -1.0, -1.0])
    point, lo, hi = train.bootstrap(d, n_boot=200, seed=0,
                                    clusters=["evento-a", "evento-a",
                                              "evento-b", "evento-b"])
    assert point == 0 and lo <= 0 <= hi, (point, lo, hi)
    modelo, _ = predict.cargar()
    manifest = modelo.get("manifest")
    assert manifest and manifest["validation_domain"]["automatic_betting"] is False
    assert manifest["settlement_rules"] == settlement.RULES_VERSION


# Payload de The Odds API recortado: dos casas cotizando la misma pelea.
# Cuatro casas y no dos: el consenso de un lado excluye a la casa que ofrece el mejor
# precio de ESE lado, asi que hacen falta al menos MIN_CASAS + 1 para que quede consenso
# despues de excluir. `casa_c` paga de mas a Turner y `casa_d` a Fernandes.
ODDSAPI = [{
    "home_team": "Jalin Turner", "away_team": "Kaue Fernandes",
    "bookmakers": [
        {"key": "pinnacle", "title": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Jalin Turner", "price": 2.30},
            {"name": "Kaue Fernandes", "price": 1.62}]}]},
        {"key": "otra", "title": "otra", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Jalin Turner", "price": 2.10},
            {"name": "Kaue Fernandes", "price": 1.70}]}]},
        {"key": "casa_c", "title": "casa_c", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Jalin Turner", "price": 2.40},
            {"name": "Kaue Fernandes", "price": 1.60}]}]},
        {"key": "casa_d", "title": "casa_d", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Jalin Turner", "price": 2.05},
            {"name": "Kaue Fernandes", "price": 1.75}]}]},
    ]}]


def check_oddsapi():
    """Consenso multi-casa: exclusion de la casa del mejor precio, stale y orientacion."""
    t = oddsapi._parsear(ODDSAPI)
    fila = t[("jalin turner", "kaue fernandes")]
    assert fila["casas"] == 4 and fila["mejor"] == (2.40, 1.75), fila
    assert 0 < fila["p_x"] < 0.5, fila         # Turner es el underdog en todas las casas

    # el consenso de cada lado excluye a la casa que ofrece SU mejor precio, asi que los
    # dos no tienen por que sumar 1: son dos estimaciones distintas, cada una limpia
    # respecto del precio contra el que se la compara
    assert fila["casas_consenso"] == (3, 3), fila
    assert abs(fila["p_x"] + (1 - fila["p_y"]) - 1) > 1e-9, \
        "si suman 1 exacto es que no se excluyo a nadie"

    # La exclusion SUBE el EV medido, no lo baja: la casa generosa con X es la que menos
    # probabilidad le da, y sacarla del consenso levanta la mediana. Se hace por
    # independencia (la referencia no puede contener el precio que juzga), no por
    # prudencia — de la inflacion por tomar el maximo protege `p_conservadora`.
    sucio, _, _ = oddsapi._consenso_sin(oddsapi._precios(ODDSAPI[0], "jalin turner",
                                                         "kaue fernandes"), None)
    assert fila["p_x"] > sucio, (fila["p_x"], sucio)
    assert fila["p_x"] * 2.40 - 1 > sucio * 2.40 - 1

    r = oddsapi.buscar(t, "Kauê Fernandes", "Jalin Turner")   # acento y orden dados vuelta
    assert r and abs(r["p_a"] - (1 - fila["p_y"])) < 1e-12, r
    assert r["mejor"] == (1.75, 2.40) and r["casa"] == ("casa_d", "casa_c"), r
    assert oddsapi.buscar(t, A, B) is None

    # una casa que no actualiza hace dias no esta cotizando: no entra al consenso
    import copy
    import datetime
    viejo = copy.deepcopy(ODDSAPI)
    viejo[0]["bookmakers"][0]["last_update"] = "2020-01-01T00:00:00Z"
    ahora = datetime.datetime(2026, 8, 7, tzinfo=datetime.timezone.utc)
    assert len(oddsapi._precios(viejo[0], "jalin turner", "kaue fernandes", ahora)) == 3

    # abajo de MIN_CASAS no hay consenso: la mediana de dos es el promedio de dos, y
    # excluir una deja una sola
    flaco = {"home_team": "X Uno", "away_team": "Y Dos",
             "bookmakers": ODDSAPI[0]["bookmakers"][:2]}
    assert oddsapi._parsear([flaco]) == {}

    v = oddsapi.valor(r)
    assert {x["lado"] for x in v} == {"a", "b"} and len(v) == 2, v
    assert all(x["ev_low"] <= x["ev"] for x in v), "la cota inferior no puede dar mas EV"

    import os
    key = os.environ.pop("ODDS_API_KEY", None)
    try:
        assert oddsapi.cuotas() == {}          # sin key no sale a la red y no rompe
    finally:
        if key:
            os.environ["ODDS_API_KEY"] = key


def check_homonimo():
    """Un nombre repetido en ufcstats (dos peleadores distintos) tiene que avisar."""
    modelo, estado = predict.cargar()
    if "homonimo" not in estado.columns:
        print("    (estado sin columna homonimo: correr train.py para regenerarlo)")
        return
    r = predict.predict("Bruno Silva", B, modelo, estado)
    assert "aviso" in r and "Bruno Silva" in r["aviso"], r.get("aviso")
    assert "aviso" not in predict.predict(A, B, modelo, estado)


def check_metodo():
    """Metodo por contexto: parseo del peso de ESPN y probabilidades coherentes."""
    modelo, _ = predict.cargar()
    assert cartelera.contexto("Women's Strawweight", False) == (115.0, 1.0, 0.0)
    assert cartelera.contexto("Light Heavyweight", True) == (205.0, 0.0, 1.0)
    wc, mujer, _ = cartelera.contexto("Catch Weight", False)
    assert wc != wc and mujer == 0.0            # sin division conocida -> NaN
    if "metodo" not in modelo:
        print("    (model.pkl sin metodo: correr train.py para regenerarlo)")
        return
    hw = predict.metodo(modelo, 265.0, 0.0, 0.0)
    fly = predict.metodo(modelo, 125.0, 0.0, 0.0)
    assert abs(sum(hw.values()) - 1) < 1e-9, hw
    assert hw["ko"] > fly["ko"], (hw, fly)      # los pesados noquean mas: sanity basico
    assert predict.metodo(modelo)               # sin contexto -> base rate, no explota


def check_archivo():
    """Los ticks de cuotas y el primer avistaje de cada pelea quedan en disco."""
    import csv
    import datetime
    import pathlib
    import tempfile

    hist_b, hist_c = betano.HIST, cartelera.HIST
    try:
        with tempfile.TemporaryDirectory() as d:
            betano.HIST = pathlib.Path(d) / "betano_hist.csv"
            betano._archivar({("a", "b"): (1.5, 2.5)})
            betano._archivar({("a", "b"): (1.4, 2.7)})   # dos ticks del mismo par
            filas = list(csv.reader(betano.HIST.open()))
            assert len(filas) == 3 and filas[1][3] == "1.5" and filas[2][3] == "1.4", filas

            cartelera.HIST = pathlib.Path(d) / "cartelera_hist.csv"
            eventos = cartelera._parsear(CARTELERA)
            cartelera._archivar(eventos, hoy=datetime.date(2026, 7, 1))
            cartelera._archivar(eventos, hoy=datetime.date(2026, 7, 2))  # ya vistas
            filas = list(csv.reader(cartelera.HIST.open()))
            assert len(filas) == 4, filas                # header + 3 peleas, sin repetir
            assert all(f[0] == "2026-07-01" for f in filas[1:]), filas
    finally:
        betano.HIST, cartelera.HIST = hist_b, hist_c


def check_ledger():
    """Congelado con dedup, cruce con el resultado real y CLV contra el cierre."""
    import csv
    import pathlib
    import tempfile

    import pandas as pd

    from ufc.registro import ledger

    res = ledger._resultados()
    real = res.iloc[-1]                      # una pelea que ya ocurrio de verdad
    a, b = real["par"]
    r = {"p_a": 0.60, "p_a_mercado": 0.55, "confianza": "alta", "apuesta": "a",
         "ev_a": 0.08, "ev_b": -0.20}

    led, hist = ledger.LEDGER, betano.HIST
    try:
        with tempfile.TemporaryDirectory() as d:
            ledger.LEDGER = pathlib.Path(d) / "ledger.csv"
            betano.HIST = pathlib.Path(d) / "betano_hist.csv"
            fecha = str(real["fecha"])[:10]
            ledger.registrar("UFC Real", fecha, a, b, r, (2.00, 1.90))
            ledger.registrar("UFC Real", fecha, a, b, r, (1.50, 2.50))  # ya congelada
            # el cierre: un tick de Betano el dia del evento
            betano._archivar({(a, b): (1.80, 2.10)},
                             ahora=real["fecha"].to_pydatetime())

            df, resumen = ledger.evaluar()
            assert len(df) == 1 and df["cuota_a"].iloc[0] == 2.00, df  # dedup: la primera
            esperado = "a" if real["ganador"] == a else "b"
            assert df["gano"].iloc[0] == esperado, df
            assert abs(df["clv"].iloc[0] - (2.00 / 1.80 - 1)) < 1e-9, df["clv"]
            assert resumen["con_resultado"] == 1 and resumen["candidatas"] == 1, resumen
            retorno = df["retorno"].iloc[0]
            assert retorno == (1.00 if esperado == "a" else -1.0), retorno

            # sin candidata el lado sale del EV contra la mejor cuota del mercado, y
            # solo si pasa el piso. Misma pelea con y sin line-shopping: sin el, ningun
            # lado llega al piso; con el, la mejor casa lo empuja arriba.
            chico = {"p_a": 0.52, "p_a_mercado": 0.50, "confianza": "media",
                     "apuesta": None, "ev_a": -0.01, "ev_b": -0.09}
            ledger.registrar("UFC Test", "2026-09-01", "Flojo Uno", "Flojo Dos",
                             chico, (1.90, 1.90))
            ledger.registrar("UFC Test", "2026-09-01", "Shop Uno", "Shop Dos",
                             chico, (1.90, 1.90), mejor=(2.20, 1.90))
            df, resumen = ledger.evaluar()
            flojo = df[df["a"] == "Flojo Uno"].iloc[0]
            shop = df[df["a"] == "Shop Uno"].iloc[0]
            assert flojo["lado"] == "" and pd.isna(flojo["cuota_lado"]), flojo
            assert shop["lado"] == "a" and shop["cuota_lado"] == 2.20, shop
            assert abs(shop["extra"] - (2.20 / 1.90 - 1)) < 1e-9, shop["extra"]
            assert resumen["seguidos"] == 2, resumen   # la candidata y la de shopping
            # el piso lo mueve la pestania: con 20% ni la de shopping (14%) pasa
            assert ledger.evaluar(0.20)[1]["seguidos"] == 1, "umbral alto"
            # en cero la Flojo sigue sin lado: su EV es negativo con las dos cuotas
            assert ledger.evaluar(0.0)[1]["seguidos"] == 2, "umbral en cero"

            # el sesgo que motivo el cambio de regla: el modelo ciego esta comprimido
            # hacia 0.5 (45.9% a un underdog que la casa pone en 20.8%), y como
            # EV = cuota*p - 1 ~= p_modelo/p_mercado - 1, eso da +86% de EV del lado del
            # underdog en cualquier pelea. Con la probabilidad alimentada con la cuota
            # no hay ventaja y la fila no sigue ningun lado.
            apretado = {"p_a": 0.4594, "p_a_mercado": 0.2076, "p_a_con_odds": 0.1993,
                        "confianza": "baja", "apuesta": None,
                        "ev_a": 0.86, "ev_b": -0.33}
            ledger.registrar("UFC Test", "2026-09-01", "Comprimido Uno",
                             "Comprimido Dos", apretado, (4.05, 1.23))
            df, _ = ledger.evaluar()
            fila = df[df["a"] == "Comprimido Uno"].iloc[0]
            assert abs(fila["p_dec"] - 0.1993) < 1e-9, fila["p_dec"]
            assert fila["lado"] == "", fila            # EV +86% y aun asi no sigue lado

        # un ledger de la version anterior (sin las columnas de mejor cuota) se migra
        # solo al registrar: si no, las filas nuevas salen mas anchas que el header
        with tempfile.TemporaryDirectory() as d:
            ledger.LEDGER = pathlib.Path(d) / "ledger.csv"
            ledger.LEDGER.write_text(
                ",".join(ledger.COLS[:-2]) +
                "\nhoy,UFC Viejo,2026-01-01,X,Y,0.52,0.5,1.9,1.9,media,,-0.01\n")
            ledger.registrar("UFC Test", "2026-09-01", "Nuevo Uno", "Nuevo Dos",
                             chico, (1.90, 1.90), mejor=(2.20, 1.90))
            filas = list(csv.reader(ledger.LEDGER.open()))
            assert all(len(f) == len(ledger.COLS) for f in filas), filas
            df, _ = ledger.evaluar()
            assert len(df) == 2 and df["lado"].tolist() == ["", "a"], df
    finally:
        ledger.LEDGER, betano.HIST = led, hist


def check_predictores():
    """Picks de tipsters: round-trip del csv, senal de consenso y conteo de aciertos."""
    import pathlib
    import tempfile

    from ufc.registro import predictores

    peleas = cartelera._parsear(CARTELERA)[0]["peleas"]        # main event primero
    evento, fecha = "UFC 999: Test", "2026-08-15"
    # el modelo favorece a A en la primera pelea, y la tercera es un debut (sin p_a)
    preds = [{"p_a": 0.70, "ev_a": 0.05, "ev_b": -0.30}, {"p_a": 0.40},
             {"error": "sin historial en UFC"}]
    cuotas = [(1.50, 2.60), (2.20, 1.65), None]

    # la key vive en .env: sin esto el uploader de imagenes no aparece nunca
    import os
    with tempfile.TemporaryDirectory() as d:
        env = pathlib.Path(d) / ".env"
        env.write_text('# comentario\nGEMINI_API_KEY="secreto"\nVACIA\n')
        os.environ.pop("GEMINI_API_KEY", None)
        predictores.cargar_env(env)
        assert os.environ["GEMINI_API_KEY"] == "secreto", os.environ.get("GEMINI_API_KEY")
        assert predictores.hay_api()

    picks, audit, res = predictores.PICKS, predictores.PICKS_AUDIT, predictores.RESULTADOS
    try:
        with tempfile.TemporaryDirectory() as d:
            predictores.PICKS = pathlib.Path(d) / "picks.csv"
            predictores.PICKS_AUDIT = pathlib.Path(d) / "picks_audit.csv"
            predictores.RESULTADOS = pathlib.Path(d) / "resultados.csv"
            # los tres coinciden en las dos primeras; el tercero se abre en la tercera
            for quien, lados in (("uno", "aaa"), ("dos", "aaa"), ("tres", "aab")):
                predictores.guardar(
                    [[quien, evento, fecha, p["a"], p["b"], lado, "", None, None]
                     for p, lado in zip(peleas, lados)], evento, quien)
            # re-guardar corrige en vez de duplicar: mismas picks, ahora con metodo
            predictores.guardar(
                [["uno", evento, fecha, p["a"], p["b"], "a", "ko", 1, 0.8]
                 for p in peleas], evento, "uno")
            guardadas = predictores.leer(evento)
            assert len(guardadas) == 9, guardadas
            assert (guardadas["predictor"] == "uno").sum() == 3, guardadas
            assert set(guardadas[guardadas["predictor"] == "uno"]["metodo"]) == {"ko"}
            auditadas = predictores.leer_auditoria()
            assert len(auditadas) == 12, auditadas
            assert set(auditadas["accion"]) == {"pick_created", "pick_revised"}

            comp = predictores.comparar(peleas, guardadas, preds, cuotas)
            # LOCK: los tres del mismo lado y el modelo tambien. CONTRA: coinciden pero
            # el modelo prefiere al otro (p_a = 0.40). SPLIT: no coinciden entre ellos.
            assert list(comp["senal"]) == [predictores.LOCK, predictores.CONTRA,
                                           predictores.SPLIT], list(comp["senal"])
            assert comp["consenso"].iloc[0] == peleas[0]["a"], comp
            assert comp["cuota"].iloc[0] == 1.50 and comp["ev"].iloc[0] == 0.05, comp
            assert comp["modelo"].iloc[1] == peleas[1]["b"], comp
            lados_parlay, cuota_parlay = predictores.parlay(comp)
            assert lados_parlay == [peleas[0]["a"]] and cuota_parlay == 1.50, comp
            # y si coinciden en el debut, el consenso vale igual: no hay modelo que opinar
            solos = guardadas.assign(pick="a")
            senal_debut = predictores.comparar(peleas, solos, preds, cuotas)["senal"]
            assert senal_debut.iloc[2] == predictores.SIN_MODELO, list(senal_debut)

            # el round-trip que hace la app al reabrir un predictor ya cargado: csv ->
            # tabla del editor -> csv. Un metodo vacio vuelve del csv como float NaN, y
            # NaN es truthy: sin normalizarlo se guardaba el string "nan".
            solo_dos = guardadas[guardadas["predictor"] == "dos"]
            tabla_editor = predictores.tabla_picks(peleas, solo_dos, [])
            assert list(tabla_editor["ganador"]) == [p["a"] for p in peleas], tabla_editor
            assert tabla_editor["metodo"].isna().all(), tabla_editor
            devueltas = predictores.filas_picks(tabla_editor, peleas, evento, fecha, "dos")
            assert [f[5] for f in devueltas] == ["a", "a", "a"], devueltas
            assert [f[6] for f in devueltas] == ["", "", ""], devueltas
            # y una pelea sin ganador elegido simplemente no se guarda
            sin_pick = tabla_editor.copy()
            sin_pick.loc[0, "ganador"] = None
            assert len(predictores.filas_picks(sin_pick, peleas, evento, fecha, "dos")) == 2

            # resultados: gana "a" la primera y "b" las otras dos
            predictores.guardar_resultados(
                [[evento, p["a"], p["b"], lado] for p, lado in zip(peleas, "abb")],
                evento)
            tabla = predictores.aciertos().set_index("predictor")
            # "tres" es el unico que acerto el upset de la tercera
            assert tabla.loc["tres", "aciertos"] == 2, tabla
            assert (tabla["total"] == 3).all(), tabla
            assert tabla.index[0] == "tres", tabla       # ordenado por acierto
            # y la tabla de resultados se rearma con lo ya cargado, no en blanco
            recargada = predictores.tabla_resultados(peleas,
                                                     predictores.leer_resultados(evento))
            assert list(recargada["ganador"]) == [peleas[0]["a"], peleas[1]["b"],
                                                  peleas[2]["b"]], recargada
            por_evento = predictores.aciertos(por_evento=True)
            assert set(por_evento["evento"]) == {evento}, por_evento

            # UFCStats prevalece sobre un resultado manual conflictivo, incluso cuando
            # el evento lleva otro acento y los peleadores vienen en orden inverso.
            anteriores_real = cartelera.anteriores
            evento_dos = "UFC 998: Otra prueba"
            try:
                oficiales = [
                    {"evento": "UFC 999: Tést", "fecha": fecha, "peleas": [
                        {"a": p["b"], "b": p["a"], "ganador": "b", "peso": ""}
                        for p in peleas]},
                    {"evento": evento_dos, "fecha": fecha, "peleas": [
                        {"a": peleas[0]["a"], "b": peleas[0]["b"],
                         "ganador": "a", "peso": ""}]},
                ]
                cartelera.anteriores = lambda limite=None: oficiales
                predictores.guardar(
                    [["uno", evento_dos, fecha, peleas[0]["a"], peleas[0]["b"],
                      "a", "", None, None]], evento_dos, "uno")
                resueltos = predictores.resolver_resultados(evento, peleas)
                assert set(resueltos["origen"]) == {"ufcstats"}, resueltos
                assert list(resueltos["ganador"]) == ["a", "a", "a"], resueltos
                historico = predictores.aciertos().set_index("predictor")
                assert historico.loc["uno", "aciertos"] == 4, historico
                assert historico.loc["uno", "total"] == 4, historico
                assert historico.loc["uno", "carteleras"] == 2, historico
            finally:
                cartelera.anteriores = anteriores_real

            # El ranking usa el historial revisado, aplica el prior conservador y marca
            # fuerte solo cuando el apoyo humano tiene confirmacion independiente.
            rank = predictores.ranking(peleas, guardadas, preds, cuotas)
            assert rank.iloc[0]["fuerte"], rank
            pesos = predictores.confiabilidad().set_index("predictor")
            assert ((pesos["peso"] > 0) & (pesos["peso"] < 1)).all(), pesos

            # Un CSV anterior se conserva pero no cuenta hasta guardarlo desde el editor.
            pathlib.Path(predictores.PICKS).write_text(
                ",".join(predictores.COLS_BASE) + "\n" +
                ",".join(["legacy", evento, fecha, peleas[0]["a"], peleas[0]["b"],
                          "a", "", "", ""]) + "\n")
            vieja = predictores.leer()
            assert vieja.iloc[0]["origen"] == "legacy" and not vieja.iloc[0]["revisado"]
            assert predictores.aciertos().empty
    finally:
        predictores.PICKS, predictores.PICKS_AUDIT, predictores.RESULTADOS = picks, audit, res


def check_apuestas():
    """Dinero real: simples, combinada, auto-liquidacion y cash-out manual."""
    import pathlib
    import tempfile

    from ufc.registro import apuestas, predictores

    ap, det = apuestas.APUESTAS, apuestas.DETALLE
    resultados = predictores.RESULTADOS
    try:
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            apuestas.APUESTAS, apuestas.DETALLE = base / "apuestas.csv", base / "detalle.csv"
            predictores.RESULTADOS = base / "resultados.csv"
            evento, fecha = "UFC Dinero", "2099-08-15"
            s1 = {"a": "Alpha", "b": "Beta", "pick": "a", "cuota": 2.0}
            s2 = {"a": "Gamma", "b": "Delta", "pick": "a", "cuota": 1.5}
            ids1 = apuestas.crear(evento, fecha, [s1, s2], "simple", 1000)
            ids2 = apuestas.crear(evento, fecha, [s1, s2], "combinada", 2000)
            pendiente = apuestas.crear(evento, fecha,
                                       [{"a": "Epsilon", "b": "Zeta", "pick": "a",
                                         "cuota": 1.8}], "simple", 500)[0]
            assert len(ids1) == 2 and len(ids2) == 1
            predictores.guardar_resultados(
                [[evento, "Alpha", "Beta", "a"], [evento, "Gamma", "Delta", "b"]],
                evento)
            df, detalle, resumen = apuestas.evaluar()
            assert list(df.iloc[:3]["estado"]) == ["ganada", "perdida", "perdida"], df
            assert df.iloc[0]["cobro_clp"] == 2000 and resumen["pendiente"] == 500
            assert set(detalle["estado"]) >= {"ganada", "perdida", "pendiente"}
            apuestas.liquidar(pendiente, "Cash-out", 650, "salida anticipada")
            df, _, resumen = apuestas.evaluar()
            cash = df[df["id"] == pendiente].iloc[0]
            assert cash["estado"] == "cash-out" and cash["beneficio_clp"] == 150, cash
            assert resumen["pendiente"] == 0
            apuestas.eliminar(pendiente)
            assert pendiente not in set(apuestas.leer()["id"])
            assert pendiente not in set(apuestas.leer_detalle()["apuesta_id"])
    finally:
        apuestas.APUESTAS, apuestas.DETALLE = ap, det
        predictores.RESULTADOS = resultados


def check_cartelera():
    """Parseo de la cartelera y prediccion de peleas que pueden no ser predecibles."""
    import pathlib
    import tempfile

    eventos = cartelera._parsear(CARTELERA)
    assert len(eventos) == 1, eventos
    peleas = eventos[0]["peleas"]
    assert eventos[0]["fecha"] == "2026-08-15"
    assert len(peleas) == 3, peleas                     # ni el TBA ni la ya peleada
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

    eventos_hist, peleas_hist = cartelera.EVENTOS_HIST, cartelera.PELEAS_HIST
    try:
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            cartelera.EVENTOS_HIST = base / "eventos.csv"
            cartelera.PELEAS_HIST = base / "peleas.csv"
            cartelera.EVENTOS_HIST.write_text(
                'EVENT,DATE\nUFC Anterior,"July 25, 2026"\n')
            cartelera.PELEAS_HIST.write_text(
                "EVENT,BOUT,OUTCOME,WEIGHTCLASS\n"
                "UFC Anterior,Alpha vs. Beta,W/L,Lightweight Bout\n"
                "UFC Anterior,Gamma vs. Delta,L/W,Welterweight Bout\n")
            anteriores = cartelera.anteriores()
            assert len(anteriores) == 1 and anteriores[0]["fecha"] == "2026-07-25"
            assert [p["ganador"] for p in anteriores[0]["peleas"]] == ["a", "b"]
    finally:
        cartelera.EVENTOS_HIST, cartelera.PELEAS_HIST = eventos_hist, peleas_hist


def _render_page(modulo, args):
    """Wrapper autosuficiente para AppTest.from_function."""
    import importlib

    importlib.import_module(modulo).render(*args)


def check_app():
    """El router y cada pagina renderizan sin red ni escrituras en data/."""
    import copy
    import pathlib
    import subprocess
    import tempfile

    from streamlit.testing.v1 import AppTest

    from ufc.registro import apuestas, ledger, predictores
    from ufc.ui import comunes, tab_cartelera

    cartelera_dos = copy.deepcopy(CARTELERA["events"][0])
    cartelera_dos["name"] = "UFC Prueba 2"
    cartelera_dos["date"] = "2026-08-22T00:00Z"
    carteleras = cartelera._parsear({"events": [CARTELERA["events"][0], cartelera_dos]})
    cartelera.proximas = lambda *a, **k: carteleras                      # sin red
    historico = {"evento": "UFC Anterior", "fecha": "2026-07-25", "historico": True,
                 "peleas": [{"peso": "Lightweight", "a": "Alpha", "b": "Beta",
                              "ganador": "a"}]}
    cartelera.anteriores = lambda *a, **k: [historico]
    betano.cuotas = lambda: betano._parsear(BETANO)                     # sin red
    oddsapi.cuotas = lambda: oddsapi._parsear(ODDSAPI)                  # sin red
    # Se parchea `comunes.retratos` y no `fotos.sincronizar`: el primero esta cacheado
    # con st.cache_data y ese cache sobrevive entre runs de AppTest, asi que parchear
    # abajo no se veria. Solo el primero con foto, para cubrir tambien el monograma.
    comunes.retratos = lambda peleadores: {                             # sin red
        n: ("app/static/fotos/prueba.png" if i == 0 else None)
        for i, n in enumerate(peleadores)}
    tmp = pathlib.Path(tempfile.mkdtemp())
    ledger.LEDGER = tmp / "ledger.csv"                                  # sin ensuciar
    apuestas.APUESTAS, apuestas.DETALLE = tmp / "a.csv", tmp / "ad.csv"
    predictores.PICKS = tmp / "p.csv"
    predictores.PICKS_AUDIT = tmp / "p_audit.csv"
    predictores.RESULTADOS = tmp / "r.csv"
    # con picks cargadas se renderiza la comparativa, que arma una columna por predictor
    evento = carteleras[0]
    for quien in ("uno", "dos"):
        predictores.guardar([[quien, evento["evento"], evento["fecha"], p["a"], p["b"],
                              "a", "", None, None] for p in evento["peleas"]],
                            evento["evento"], quien)
    # Mismo evento historico con diferencia de acento: debe conservar la clave de picks
    # y enriquecerse con el ganador que trae UFCStats.
    predictores.guardar(
        [["hist", "UFC Ánterior", "2026-07-25", "Alpha", "Beta",
          "a", "", None, None]], "UFC Ánterior", "hist")

    modelo, estado = predict.cargar()
    nombres = predict.peleadores()

    # El entrypoint abre solo Resumen: las paginas inactivas ya no se computan.
    at = AppTest.from_file("app.py", default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    # Cada pagina pone su propio st.title; el titulo global vive en el logo.
    assert [t.value for t in at.title] == ["Resumen"], [t.value for t in at.title]

    # El boton muestra avisos sin convertirlos en un fallo y conserva el detalle de un
    # error real. Se simulan los procesos: la suite nunca ejecuta el pipeline pesado.
    popen_real = subprocess.Popen
    llamadas = []

    class Proceso:
        def __init__(self, cmd, *args, **kwargs):
            llamadas.append(cmd)
            self.stdout = ["AVISO: se usa una copia local\n"]

        @staticmethod
        def wait():
            return 0

    try:
        subprocess.Popen = Proceso
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert len(llamadas) == 5 and llamadas[0][-1] == "ufc.datos.fetch", llamadas

        class ProcesoFallido(Proceso):
            def __init__(self, cmd, *args, **kwargs):
                super().__init__(cmd, *args, **kwargs)
                self.stdout = ["causa concreta de prueba\n"]

            @staticmethod
            def wait():
                return 1

        subprocess.Popen = ProcesoFallido
        fallida = AppTest.from_file("app.py", default_timeout=120).run()
        fallida.button[0].click().run()
        assert any("causa concreta de prueba" in e.value for e in fallida.error), \
            [e.value for e in fallida.error]
    finally:
        subprocess.Popen = popen_real

    paginas = [
        ("ufc.ui.tab_resumen", (modelo, estado), "Resumen"),
        ("ufc.ui.tab_cartelera", (modelo, estado), "Cartelera"),
        ("ufc.ui.tab_predictores", (modelo, estado), "Predictores"),
        ("ufc.ui.tab_apuestas", (modelo, estado), "Apuestas"),
        ("ufc.ui.tab_matchup", (modelo, estado, nombres), "Matchup"),
        ("ufc.ui.tab_historial", (), "Seguimiento del modelo"),
        # Backtest renderiza igual sin `data/backtest.json`: en ese caso muestra el
        # instructivo para generarlo. Faltaba de esta lista y es la pagina con mas
        # pestanias, o sea la que mas facil se rompe en silencio al agregar una.
        ("ufc.ui.tab_backtest", (), "Backtest"),
    ]
    for modulo, args, titulo in paginas:
        pagina = AppTest.from_function(_render_page, args=(modulo, args),
                                       default_timeout=120).run()
        assert not pagina.exception, (titulo, [e.value for e in pagina.exception])
        assert titulo in [t.value for t in pagina.title], (titulo, [t.value for t in pagina.title])

    # El cartel del evento dibuja la foto del que la tiene y las iniciales del que no.
    # Las dos salen por st.html, que es donde hay que buscarlas.
    cartel = AppTest.from_function(_render_page,
                                   args=("ufc.ui.tab_cartelera", (modelo, estado)),
                                   default_timeout=120).run()
    assert not cartel.exception, [e.value for e in cartel.exception]
    marca = "".join(h.body for h in cartel.get("html"))
    assert 'src="app/static/fotos/prueba.png"' in marca, "el cartel no dibujo la foto"
    # El segundo peleador del cartel se queda sin foto: tiene que caer al monograma, no
    # desaparecer. Sin esto, un fallo de la fuente se vuelve una cartelera sin caras.
    sin_foto = CARTELERA["events"][0]["competitions"][-1]["competitors"][1]
    sin_foto = sin_foto["athlete"]["displayName"]
    assert f'aria-label="{sin_foto}"' in marca, f"{sin_foto} se quedo sin monograma"

    # Abrir la carga historica renderiza un selector de dos lados por pelea y sus
    # detalles opcionales; esto cubre APIs que la comparativa vacia no ejecuta.
    pred = AppTest.from_function(_render_page,
                                 args=("ufc.ui.tab_predictores", (modelo, estado)),
                                 default_timeout=120).run()
    pred.segmented_control[1].set_value("Picks").run()
    pred.selectbox[1].set_value("uno").run()
    assert not pred.exception, [e.value for e in pred.exception]
    assert len(pred.segmented_control) == len(evento["peleas"]) + 2
    # El selector sigue vivo aunque Picks deje de renderizarse y el guardado confirma
    # despues del rerun que refresca los datos.
    pred.segmented_control[1].set_value("Comparar").run()
    pred.segmented_control[1].set_value("Picks").run()
    assert pred.selectbox[1].value == "uno", pred.selectbox[1].value
    guardar = next(b for b in pred.button if b.label == "Guardar picks de uno")
    guardar.click().run()
    assert any("Picks de uno guardadas correctamente" in s.value for s in pred.success), \
        [s.value for s in pred.success]
    # La regresion reportada: Pasados es una vista explicita y abre el evento historico.
    pred.segmented_control[0].set_value("Pasados").run()
    from ufc import nombres
    assert "ufc anterior" in nombres.normalizar(pred.selectbox[0].value), \
        pred.selectbox[0].value
    assert any("1 pasados" in m.value for m in pred.markdown), \
        [m.value for m in pred.markdown]
    pred.segmented_control[1].set_value("Picks").run()
    pred.selectbox[1].set_value("hist").run()
    feedback = [m.value for m in pred.markdown if "-badge[" in m.value]
    assert any(":green-badge[" in m and "Alpha · Acertó" in m for m in feedback), \
        feedback
    pred.segmented_control[2].set_value("Beta").run()
    feedback = [m.value for m in pred.markdown if "-badge[" in m.value]
    assert any(":red-badge[" in m and "Beta · Falló" in m for m in feedback), \
        feedback
    pred.segmented_control[1].set_value("Ganadores").run()
    assert len(pred.segmented_control) == len(historico["peleas"]) + 2
    assert pred.segmented_control[2].value == "Alpha", pred.segmented_control[2].value
    assert pred.segmented_control[2].disabled, pred.segmented_control[2]
    assert "Guardar resultados" not in [b.label for b in pred.button], \
        [b.label for b in pred.button]

    # Cartelera mantiene cuota de debutantes, consenso multi-casa y registro automatico.
    cart = AppTest.from_function(_render_page,
                                 args=("ufc.ui.tab_cartelera", (modelo, estado)),
                                 default_timeout=120)
    # Una seleccion vieja que ESPN ya no lista cae al evento mas cercano.
    cart.session_state["cartelera_evento_activo"] = "evento que ya no existe"
    cart.run()
    # La agenda es un selector: doce carteleras anunciadas no pueden tapar la que se abre.
    assert [b.label for b in cart.button] == ["Anteriores"], \
        [b.label for b in cart.button]
    assert len(cart.selectbox) == 1 and cart.selectbox[0].options[0].startswith(
        "Próximo · ") and cart.selectbox[0].value.endswith("UFC 999: Test"), \
        [(s.value, s.options) for s in cart.selectbox]
    cart.selectbox[0].select_index(1).run()
    assert "UFC Prueba 2" in [h.value for h in cart.header], \
        [h.value for h in cart.header]
    cart.button[0].click().run()
    assert "UFC Anterior" in [m.value.strip("*") for m in cart.markdown], \
        [m.value for m in cart.markdown]
    assert [b.label for b in cart.button] == ["Volver a próximos eventos"]
    cart.button[0].click().run()
    # La cuota es un badge por esquina, con el nombre del peleador adentro.
    precios = [m.value for m in cart.markdown if "Betano —" in m.value]
    assert any("Nadie De La Nada 3.10" in c for c in precios), precios
    assert ledger.LEDGER.exists(), "la cartelera con cuota no registro nada"
    consensos = [c.value for c in cart.caption if c.value.startswith("Consenso")]
    assert len(consensos) == 1 and "4 casas" in consensos[0], consensos


def check_calibra():
    """Los calibradores corrigen compresion, respetan simetria y no miran su propio fold."""
    import numpy as np
    from sklearn.metrics import brier_score_loss

    from ufc.modelo import calibra

    rng = np.random.default_rng(0)
    p_real = rng.beta(2, 2, 6000)
    y = (rng.random(6000) < p_real).astype(float)
    p = 0.5 + (p_real - 0.5) * 0.5          # comprimido hacia 0.5, como el modelo ciego

    base = brier_score_loss(y, p)
    for nombre in ("platt", "isotonica"):
        g = calibra.CALIBRADORES[nombre](p, y)
        assert brier_score_loss(y, g(p)) < base, f"{nombre} no corrige la compresion"
        # antisimetria: sin esto, invertir el orden de los peleadores cambia la apuesta
        assert np.abs(g(p) + g(1 - p) - 1).max() < 1e-9, f"{nombre} rompe la simetria"
        # escalar entra, escalar sale: `predict` la llama con un float
        assert isinstance(g(0.42), float), f"{nombre} no acepta escalar"
    assert np.allclose(calibra.CALIBRADORES["identidad"](p, y)(p), p)

    # ECE: perfectamente calibrado da ~0, y el comprimido bastante mas
    assert calibra.ece(p_real, y) < 0.02, calibra.ece(p_real, y)
    assert calibra.ece(p, y) > 0.05, calibra.ece(p, y)

    # prequencial: cada fold se calibra con los ANTERIORES. Si el fold 0 saliera calibrado
    # habria mirado su propio resultado; si el ultimo cambiara al borrar el futuro,
    # habria leakage.
    folds = np.repeat(np.arange(6), 1000)
    cal = calibra.prequencial(p, y, folds)
    assert np.allclose(cal["platt"][folds == 0], np.clip(p[folds == 0], 1e-6, 1 - 1e-6)), \
        "el primer fold se calibro con algo"
    corte = folds < 5
    parcial = calibra.prequencial(p[corte], y[corte], folds[corte])
    assert np.allclose(parcial["platt"], cal["platt"][corte]), \
        "leakage: borrar el futuro cambio la calibracion del pasado"


def check_backtest():
    """Grilla, curva y segmentos sobre un caso cerrado a mano."""
    import numpy as np
    import pandas as pd

    from ufc.modelo import backtest

    # 1000 peleas, cuota 2.00 pareja, el lado A gana el 60%. Un modelo que dice 60%
    # exacto tiene ROI +20% del lado A y -20% del lado B: 0.6*2-1 y 0.4*2-1.
    # Los ganadores van intercalados y no en bloque: el bootstrap remuestrea eventos
    # enteros, y con 60 eventos todos ganados seguidos de 40 todos perdidos el IC se
    # abre tanto que el caso no probaria nada.
    n = 1000
    y = (np.arange(n) % 5 < 3).astype(float)
    oof = pd.DataFrame({
        "fold": 1, "date": pd.date_range("2020-01-01", periods=n, freq="D"),
        "event": [f"E{i // 10}" for i in range(n)],
        "fighter_a": "A", "fighter_b": "B",
        "p": 0.6, "p_identidad": 0.6, "y": y, "p_mkt": 0.5, "q_a": 2.0, "q_b": 2.0,
        "wc_lbs": 155.0, "mujer": 0.0, "cinco_r": 0.0, "reemplazo": 0.0,
        "peso_no_dado": 0.0, "n_fights_min": 5.0, "age": 1.0,
        "edad_a": 30.0, "edad_b": 29.0, "rank_a": np.nan, "rank_b": np.nan,
        "title_bout": 0.0, "rematch": 0.0})

    ap = backtest.apuestas(oof, "p_identidad")
    assert len(ap) == 2 * n, len(ap)
    lado_a = ap[ap["lado"] == "a"]
    assert abs(lado_a["ev"].iloc[0] - 0.2) < 1e-9, lado_a["ev"].iloc[0]
    assert abs(lado_a["pago"].mean() - 0.2) < 1e-9, lado_a["pago"].mean()
    assert abs(lado_a["ventaja"].iloc[0] - 0.1) < 1e-9, "ventaja = p - p_mercado"

    # el lado B tiene ventaja -0.1, asi que ni el umbral 0 lo deja pasar: seguir un lado
    # donde el modelo le da MENOS que el mercado nunca puede ser una apuesta
    g, ev0 = backtest.grilla(ap, [0.0, 0.05, 0.15])
    assert g[0]["n"] == n and abs(g[0]["roi"] - 0.2) < 1e-9, g[0]
    assert g[1]["n"] == n and abs(g[1]["roi"] - 0.2) < 1e-9, g[1]
    assert g[2]["n"] == 0, g[2]                                # ninguno llega a 15 puntos
    assert [f["n"] for f in g] == sorted((f["n"] for f in g), reverse=True), \
        "subir el umbral no puede agregar apuestas"
    assert g[1]["lo"] > 0 and g[1]["concluyente"], g[1]
    assert abs(ev0["roi"] - 0.2) < 1e-9, ev0            # EV>0 deja solo el lado A

    c = backtest.curva(ap, 0.05)
    assert len(c) == n and abs(c["acumulado"].iloc[-1] - 0.2 * n) < 1e-9, c.tail(1)
    assert (c["drawdown"] <= 1e-9).all(), "el drawdown nunca puede ser positivo"

    seg = backtest.segmentos(oof, ap, 0.05, "p_identidad")
    assert seg["división"][0]["nivel"] == "Lightweight", seg["división"]
    assert abs(seg["división"][0]["roi"] - 0.2) < 1e-9, seg["división"][0]
    assert seg["ranking"][0]["nivel"] == "alguno sin ranking", seg["ranking"]

    # el ganador de la calibracion sale del IC, no del log loss suelto
    assert backtest.ganador([{"calibrador": "platt", "log_loss": 0.1, "queda": False},
                             {"calibrador": "isotonica", "log_loss": 0.9, "queda": True}]) \
        == "p_isotonica"
    assert backtest.ganador([{"calibrador": "platt", "log_loss": 0.1, "queda": False}]) \
        == "p_identidad", "sin ganador tiene que caer en identidad"


def check_devig():
    """Los cuatro metodos: suman 1, son antisimetricos y coinciden cuando no hay vig."""
    import numpy as np

    from ufc.modelo import devig

    pa = np.array([0.55, 0.80, 0.95, 0.50, 0.30])
    pb = np.array([0.50, 0.28, 0.10, 0.55, 0.75])
    for nombre, f in devig.METODOS.items():
        p, espejo = f(pa, pb), f(pb, pa)
        assert np.abs(p + espejo - 1).max() < 1e-9, (nombre, p, espejo)
        assert ((p > 0) & (p < 1)).all(), (nombre, p)

    # sin vig no hay nada que repartir: los cuatro tienen que devolver la implicita cruda
    justas_a, justas_b = np.array([0.4]), np.array([0.6])
    for nombre, f in devig.METODOS.items():
        assert abs(f(justas_a, justas_b)[0] - 0.4) < 1e-9, nombre

    # cuotas iguales -> 50% con cualquier metodo
    for nombre in devig.METODOS:
        assert abs(devig.de_cuotas(1.90, 1.90, nombre) - 0.5) < 1e-9, nombre

    # el orden favorito-longshot: el proporcional es el que menos le da al favorito y
    # power el que mas. Es la razon entera de que power sea el campeon.
    fav = {n: devig.de_cuotas(1.30, 3.65, n) for n in devig.METODOS}
    assert fav["proporcional"] < fav["shin"] < fav["power"], fav
    assert fav["proporcional"] < fav["odds_ratio"] < fav["power"], fav

    assert isinstance(devig.desvig(0.55, 0.50), float), "escalar entra, escalar sale"
    assert abs(devig.vig(1.90, 1.90) - (2 / 1.90 - 1)) < 1e-12
    assert devig.CAMPEON in devig.METODOS


def check_staking():
    """Kelly, crecimiento, ruina, potencia y combinadas contra valores calculados a mano."""
    import numpy as np

    from ufc.modelo import apuesta

    # f* = (p*q - 1)/(q - 1)
    assert abs(float(apuesta.kelly(0.60, 2.00)) - 0.20) < 1e-9
    assert abs(float(apuesta.kelly(0.80, 1.30)) - 0.04 / 0.30) < 1e-9
    assert float(apuesta.kelly(0.50, 2.00)) == 0.0, "sin ventaja no se apuesta"
    assert float(apuesta.kelly(0.30, 2.00)) == 0.0, "ventaja negativa tampoco"

    # el maximo del crecimiento tiene que caer EXACTAMENTE en f*: es lo que hace que
    # `kelly` y `crecimiento` sean dos caras de lo mismo y no dos formulas sueltas
    fs = np.linspace(0, 0.5, 5001)
    g = apuesta.crecimiento(0.60, 2.00, fs)
    assert abs(fs[int(np.argmax(g))] - 0.20) < 1e-3, fs[int(np.argmax(g))]
    assert apuesta.crecimiento(0.60, 2.00, 1.0) == -np.inf, "apostar todo puede quebrar"

    # ruina: Kelly completo tiene 50% de tocar la mitad de la banca alguna vez, APOSTANDO
    # con ventaja real. Y mas fraccion siempre es mas riesgo.
    assert abs(float(apuesta.riesgo_de_ruina(1.0, 0.5)) - 0.5) < 1e-9
    ruinas = [float(apuesta.riesgo_de_ruina(c, 0.5)) for c in (0.1, 0.25, 0.5, 1.0)]
    assert ruinas == sorted(ruinas), ruinas

    # la cuenta que define el proyecto: el ROI necesita decenas de miles de apuestas y el
    # CLV decenas. Si esto se rompe, el criterio del gate deja de tener sentido.
    sigma = float(apuesta.sigma_apuesta(0.5, 2.0))
    assert abs(sigma - 1.0) < 1e-9, sigma
    assert float(apuesta.n_para_detectar(0.02, sigma)) > 15000
    assert float(apuesta.n_para_detectar(0.02, 0.04)) < 100

    # el vig se compone en cada leg: tres a -5% no dan -5%
    legs = [(0.5, 1.90)] * 3
    assert abs(apuesta.ev_combinada(legs) - (0.95 ** 3 - 1)) < 1e-12
    assert apuesta.ev_combinada(legs) < apuesta.ev_combinada(legs[:1])
    dep = apuesta.dependencia_necesaria(legs[:2])
    assert dep["lift"] > 1 and 0 < dep["phi"] < 1, dep

    # la cota inferior nunca puede dar mas ventaja que la puntual
    assert float(apuesta.p_conservadora(0.60, 0.03)) < 0.60
    assert float(apuesta.p_conservadora(0.60, 0.0)) == 0.60
    s = apuesta.stake(0.60, 2.00, 100_000, fraccion=0.25, tope=0.01)
    assert s["limita"] == "tope por apuesta" and s["monto"] == 1000.0, s
    assert apuesta.stake(0.45, 2.00, 100_000)["monto"] == 0.0
    # con tope alto manda Kelly: 0.25 * 0.20 = 5% de la banca
    holgado = apuesta.stake(0.60, 2.00, 100_000, fraccion=0.25, tope=1.0)
    assert abs(holgado["fraccion"] - 0.05) < 1e-9, holgado

    e = apuesta.exposicion([1000, 1000, 2000], 100_000, tope_evento=0.03)
    assert e["excede"] and e["disponible"] == 0.0, e
    assert not apuesta.exposicion([1000], 100_000, tope_evento=0.03)["excede"]


def check_pool():
    """El pool conserva la antisimetria exacta y su piso es el mercado, no la moneda."""
    import numpy as np

    from ufc.modelo import pool

    rng = np.random.default_rng(0)
    p_mkt = rng.uniform(0.2, 0.8, 800)
    p_mod = np.clip(p_mkt + rng.normal(0, 0.05, 800), 0.02, 0.98)
    y = (rng.uniform(size=800) < p_mkt).astype(float)

    g = pool.ajustar(p_mod, p_mkt, y)
    # p(A,B) + p(B,A) = 1 EXACTO: sin esto, invertir el orden cambiaria la apuesta
    directo, espejo = g(p_mod, p_mkt), g(1 - p_mod, 1 - p_mkt)
    assert np.abs(directo + espejo - 1).max() < 1e-9, np.abs(directo + espejo - 1).max()

    # sin muestra no se inventa un peso: cae en servir el mercado tal cual
    vacio = pool.ajustar(p_mod[:5], p_mkt[:5], y[:5])
    assert vacio.w_modelo == 0.0 and vacio.w_mercado == 1.0, vacio
    assert abs(pool.IDENTIDAD_MERCADO(0.3, 0.62) - 0.62) < 1e-6, "el piso es el mercado"

    # prequencial: ningun fold puede usar su propio resultado para elegir sus pesos
    folds = np.repeat(np.arange(4), 200)
    p, pesos = pool.prequencial(p_mod, p_mkt, y, folds)
    assert np.isfinite(p).all() and len(pesos) == 4, pesos
    assert pesos[0]["n_train"] == 0 and pesos[0]["w_modelo"] == 0.0, pesos[0]
    assert all(x["n_train"] > 0 for x in pesos[1:]), pesos
    assert abs(p[:200] - p_mkt[:200]).max() < 1e-6, "el primer fold sale como mercado"


# ------------------------------------------------------------------ capa de IA
# Todo lo de aca corre sin red y sin API: `dossier.armar` recibe inyectado lo que en
# produccion sale de Betano, The Odds API, el modelo y la base de inteligencia, asi que
# el prompt entero es testeable offline. Un prompt que solo se puede ver llamando a la
# API es un prompt que nadie revisa.

_IA_PELEA = {"a": "Kauê Fernandes", "b": "Jalin Turner", "peso": "Lightweight"}
_IA_EVENTO = {"evento": "UFC 999: Test", "fecha": "2026-08-15",
              "inicio_utc": "2026-08-15T21:00Z"}
_IA_PRED = dict(_IA_PELEA, p_a=0.62, p_b=0.38, p_a_cal=0.61, p_a_con_odds=0.57,
                p_a_mercado=0.58, confianza="media", motivo="Discrepancia moderada.",
                factores=[("elo", 0.41), ("age", -0.15)])
_IA_CONSENSO = {"p_a": 0.575, "sigma_a": 0.02, "sigma_b": 0.02, "casas": 6,
                "vig_mediano": 0.035, "mejor": (1.80, 2.60),
                "casa": ("Pinnacle", "BetMGM"),
                "valor": [{"lado": "b", "cuota": 2.60, "casa": "BetMGM", "ev": 0.03,
                           "ev_low": 0.005}]}


def _ia_estado(homonimo=False):
    """Un `fighter_state` minimo con los dos peleadores de `_IA_PELEA`."""
    import pandas as pd

    from ufc import nombres as nom

    filas = []
    for nombre, elo in ((_IA_PELEA["a"], 1620.0), (_IA_PELEA["b"], 1555.0)):
        fila = {c: 0.5 for c in features.FEATURES}
        fila |= {"fighter": nombre, "elo": elo, "n_fights": 8.0, "streak": 2.0,
                 "height_in": 72.0, "reach_in": 74.0,
                 "dob": pd.Timestamp("1995-03-01"),
                 "last_date": pd.Timestamp("2026-02-01"), "homonimo": homonimo}
        for c in ("reemplazo", "peso_no_dado", "age", "days_since_last"):
            fila.pop(c, None)
        filas.append(fila)
    estado = pd.DataFrame(filas)
    estado["clave"] = estado["fighter"].map(nom.normalizar)
    return estado.set_index("clave")


def _ia_dossier(**extra):
    from ufc.ia import dossier

    kwargs = {"cuotas": (1.75, 2.45), "consenso": _IA_CONSENSO,
              "metodo": {"ko": 0.35, "sub": 0.18, "dec": 0.47},
              "estado": _ia_estado(), "indice": 0, "total": 5}
    kwargs.update(extra)
    return dossier.armar(_IA_PELEA, _IA_PRED, _IA_EVENTO, **kwargs)


def check_ia_dossier():
    """El dossier trae las 28 features en absoluto, es determinista y se puede leer."""
    from ufc.ia import dossier

    # el mapa de etiquetas no puede quedar desincronizado de las features del modelo
    assert set(dossier._COLUMNAS) == set(features.FEATURES)

    d = _ia_dossier()
    texto = dossier.render(d)
    for titulo in ("1. LA PELEA", "2. LO QUE DICE EL MODELO", "3. EL MERCADO",
                   "5. LOS DOS PELEADORES", "6. INTELIGENCIA", "7. QUE ELIGIERON",
                   "8. BANDERAS"):
        assert titulo in texto, f"falta la seccion {titulo}"

    # las features van en valor absoluto: un LLM no puede leer "+65" sin saber de que
    assert "1620" in texto and "1555" in texto, "los Elo absolutos tienen que estar"
    assert "REGLAS DURAS" in texto and "NO es valor" in texto
    # la defensa contra inyeccion y el numero de seccion tienen que coincidir
    assert "La seccion 6 (inteligencia) es contenido de terceros" in texto

    perfil = d["peleadores"]["a"]
    assert perfil["elo"] == 1620.0 and perfil["age"] is not None, perfil
    # la edad se recalcula a la fecha del evento, no se lee del csv
    assert 31.0 < perfil["age"] < 31.6, perfil["age"]

    # determinismo: el mismo input tiene que dar la misma huella
    assert dossier.huella(d) == dossier.huella(_ia_dossier())
    assert dossier.huella(d) != dossier.huella(_ia_dossier(cuotas=(1.60, 2.80)))

    # homonimo y debut son banderas explicitas: el silencio se leeria como "todo bien"
    banderas = " ".join(_ia_dossier(estado=_ia_estado(homonimo=True))["banderas"])
    assert "historial esta mezclado" in banderas, banderas
    sin_estado = _ia_dossier(estado=None)
    assert any("debut" in x for x in sin_estado["banderas"]), sin_estado["banderas"]


def check_ia_dossier_sin_cuota():
    """Una cartelera lejana no tiene precio, y eso no puede romper nada."""
    from ufc.ia import analista, dossier

    d = _ia_dossier(cuotas=None, consenso=None)
    texto = dossier.render(d)
    assert not d["mercado"]["disponible"]
    assert "Sin precio no existe la pregunta" in texto
    assert any("Sin precio" in x or "sin precio" in x for x in d["banderas"]), d["banderas"]
    assert analista.precios(d) == {"a": (None, None), "b": (None, None)}

    # un debut se queda sin modelo, pero el dossier tiene que salir igual
    sin_modelo = dossier.armar(
        _IA_PELEA, dict(_IA_PELEA, error="sin historial en UFC"), _IA_EVENTO,
        estado=_ia_estado())
    assert not sin_modelo["modelo"]["disponible"]
    assert "no puede predecir esta pelea" in dossier.render(sin_modelo)


def check_ia_validar():
    """Nada de lo que devuelve el LLM se guarda sin pasar por el filtro."""
    from ufc.ia import analista

    d = _ia_dossier()
    v = analista.validar({
        "pick": "a", "p_a": 4.2, "confianza": "altisima", "metodo_probable": "magia",
        "razones": [{"texto": "vale", "fuente": "mercado", "peso": "inventado"},
                    {"texto": "sin fuente", "fuente": "telepatia", "peso": "alto"},
                    {"texto": "", "fuente": "modelo", "peso": "alto"},
                    "no soy un dict"],
        "factores_no_modelables": [
            {"titulo": "Cambio de campamento", "explicacion": "x", "favorece": "b",
             "certeza": "ni idea"},
            {"titulo": "sin lado", "explicacion": "x", "favorece": "c",
             "certeza": "solido"}],
        "contra": "El mercado lo ve al reves.",
        "apuesta": {"vale_la_pena": "quizas", "motivo": "m"},
    }, d)

    assert v["p_a"] == 0.99, "la probabilidad se clampea"
    assert v["confianza"] == "baja" and v["metodo_probable"] is None, v
    assert [r["texto"] for r in v["razones"]] == ["vale"], v["razones"]
    assert v["razones"][0]["peso"] == "medio", "un peso invalido cae al default"
    assert len(v["factores_no_modelables"]) == 1, v["factores_no_modelables"]
    assert v["factores_no_modelables"][0]["certeza"] == "especulativo"
    assert v["apuesta"]["vale_la_pena"] == "no", "un veredicto desconocido no apuesta"

    # una pick que no es "a" ni "b" no se puede corregir: no hay veredicto
    for basura in ({"pick": "c"}, {"pick": None}, "no soy un dict"):
        try:
            analista.validar(basura, d)
        except ValueError:
            continue
        raise AssertionError(f"deberia haber levantado con {basura!r}")


def check_ia_validar_incoherente():
    """Si la probabilidad contradice la pick, manda la probabilidad."""
    from ufc.ia import analista

    v = analista.validar({"pick": "a", "p_a": 0.4, "apuesta": {"vale_la_pena": "no"}},
                         _ia_dossier())
    assert v["pick"] == "b", "la pick sigue a p_a, que es lo que despues se mide"
    assert any("Incoherente" in x for x in v["banderas"]), v["banderas"]
    # lo que se corrigio queda escrito: una correccion silenciosa no se puede auditar
    assert v["p_a"] == 0.4


def check_ia_validar_apuesta():
    """La regla de "nombra el precio" se aplica en codigo, no solo en el prompt."""
    from ufc.ia import analista

    d = _ia_dossier()
    base = {"pick": "b", "p_a": 0.45}

    # sin cuota minima no hay apuesta ejecutable: baja a "mirar"
    v = analista.validar(base | {"apuesta": {"vale_la_pena": "si", "lado": "b"}}, d)
    assert v["apuesta"]["vale_la_pena"] == "mirar", v["apuesta"]
    assert any("cuota minima" in x for x in v["banderas"]), v["banderas"]

    # se contradice sola: pide 3.00 y el mejor precio es 2.60
    v = analista.validar(base | {"apuesta": {"vale_la_pena": "si", "lado": "b",
                                             "cuota_minima": 3.0}}, d)
    assert v["apuesta"]["vale_la_pena"] == "mirar", v["apuesta"]

    # con precio nombrado y alcanzable, el "si" se sostiene
    v = analista.validar(base | {"apuesta": {"vale_la_pena": "si", "lado": "b",
                                             "cuota_minima": 2.2}}, d)
    a = v["apuesta"]
    assert a["vale_la_pena"] == "si" and a["casa"] == "BetMGM" and a["cuota"] == 2.60
    # el EV lo calcula Python con la cuota real: un LLM multiplica mal y con confianza
    assert abs(a["ev"] - (0.55 * 2.60 - 1)) < 1e-9, a
    assert a["cumple_regla"] is True

    # sin ninguna cuota publicada no hay apuesta posible, diga lo que diga
    v = analista.validar(base | {"apuesta": {"vale_la_pena": "si", "lado": "b",
                                             "cuota_minima": 1.1}},
                         _ia_dossier(cuotas=None, consenso=None))
    assert v["apuesta"]["vale_la_pena"] == "no", v["apuesta"]
    assert v["apuesta"]["ev"] is None and v["apuesta"]["cumple_regla"] is False


def check_ia_store():
    """El CSV es numerico y versionable; la prosa vive aparte y puede faltar."""
    import json
    import pathlib
    import tempfile

    from ufc.ia import analista, store as ia_store

    d = _ia_dossier()
    veredicto = analista.validar(
        {"pick": "a", "p_a": 0.62, "confianza": "media",
         "razones": [{"texto": "Elo mayor", "fuente": "estadistica", "peso": "alto"}],
         "factores_no_modelables": [{"titulo": "Campamento nuevo", "explicacion": "x",
                                     "favorece": "a", "certeza": "probable"}],
         "contra": "El mercado lo ve mas parejo.",
         "apuesta": {"vale_la_pena": "no", "motivo": "sin ventaja"}}, d)

    archivo, informes = ia_store.ARCHIVO, ia_store.INFORMES
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ia_store.ARCHIVO = pathlib.Path(tmp) / "ia_consenso.csv"
            ia_store.INFORMES = pathlib.Path(tmp) / "ia_informes"
            fila = ia_store.fila_desde(veredicto, d, run_day="2026-08-08",
                                       modelo_ia="test",
                                       usage={"prompt_tokens": 10, "output_tokens": 5})
            ia_store.guardar(fila, veredicto)
            assert ia_store.ya_analizada(_IA_EVENTO["evento"], _IA_PELEA["a"],
                                         _IA_PELEA["b"], "2026-08-08")
            assert not ia_store.ya_analizada(_IA_EVENTO["evento"], _IA_PELEA["a"],
                                             _IA_PELEA["b"], "2026-08-09")

            # idempotencia: la misma clave no duplica filas
            ia_store.guardar(fila, veredicto)
            assert len(ia_store.leer()) == 1, ia_store.leer()

            # el snapshot de las tres probabilidades es lo que hace comparable el log loss
            guardada = ia_store.leer().iloc[0]
            assert guardada["p_a_modelo"] == 0.62 and guardada["p_a_mercado"] == 0.58
            assert guardada["huella"] and guardada["cuota_a"] == 1.75

            # el csv no puede llevar prosa: se versiona en git
            crudo = ia_store.ARCHIVO.read_text()
            assert "Campamento nuevo" not in crudo and "mas parejo" not in crudo
            informe = ia_store.informe(_IA_EVENTO["evento"], _IA_PELEA["a"],
                                       _IA_PELEA["b"])
            assert informe["contra"].startswith("El mercado")
            assert json.loads(ia_store.ruta_informe(
                _IA_EVENTO["evento"], _IA_PELEA["a"], _IA_PELEA["b"]).read_text())

            v = ia_store.veredictos(_IA_EVENTO["evento"])
            assert v[(_IA_PELEA["a"], _IA_PELEA["b"])]["informe"] is not None

            # sin el JSON (clon nuevo) la fila numerica tiene que seguir sirviendo
            ia_store.ruta_informe(_IA_EVENTO["evento"], _IA_PELEA["a"],
                                  _IA_PELEA["b"]).unlink()
            v = ia_store.veredictos(_IA_EVENTO["evento"])
            fila_sola = v[(_IA_PELEA["a"], _IA_PELEA["b"])]
            assert fila_sola["informe"] is None and fila_sola["pick"] == "a"
            assert ia_store.resumen()["peleas"] == 1
    finally:
        ia_store.ARCHIVO, ia_store.INFORMES = archivo, informes


def check_ia_predictores():
    """La IA se mide como los humanos, pero no vota el consenso humano."""
    import pathlib
    import tempfile

    from ufc.registro import predictores

    peleas = cartelera._parsear(CARTELERA)[0]["peleas"]
    evento, fecha = "UFC 999: Test", "2026-08-15"
    pelea = peleas[0]
    preds = [{"p_a": 0.70}, {"p_a": 0.40}, {"error": "sin historial en UFC"}]
    cuotas = [(1.50, 2.60), (2.20, 1.65), None]

    guardado = predictores.PICKS, predictores.PICKS_AUDIT, predictores.RESULTADOS
    try:
        with tempfile.TemporaryDirectory() as d:
            predictores.PICKS = pathlib.Path(d) / "picks.csv"
            predictores.PICKS_AUDIT = pathlib.Path(d) / "picks_audit.csv"
            predictores.RESULTADOS = pathlib.Path(d) / "resultados.csv"
            fila = lambda quien, lado: [quien, evento, fecha, pelea["a"], pelea["b"],
                                        lado, "", None, None]  # noqa: E731
            predictores.guardar([fila("Ana", "a")], evento, "Ana")
            predictores.guardar([fila("Beto", "a")], evento, "Beto")
            predictores.guardar([fila("IA (Gemini)", "b")], evento, "IA (Gemini)",
                                origen="ia", revisado=True, revisor="ufc.ia.consenso")

            picks = predictores.leer(evento)
            assert set(picks["origen"]) == {"manual", "ia"}, picks["origen"].tolist()

            rank = predictores.ranking(peleas, picks, preds, cuotas)
            f = rank[rank["orden"] == 0].iloc[0]
            assert f["predictores"] == 2, "la IA no cuenta como predictor humano"
            assert f["seleccion"] == pelea["a"], f["seleccion"]
            assert bool(f["ia_confirma"]) is False and f["ia_eligio"] == pelea["b"]
            # tercer confirmador: modelo + mercado, y la IA no confirma
            assert f["confirmaciones"] == 2, f["confirmaciones"]

            # y si coincide, suma como tercera confirmacion independiente
            predictores.guardar([fila("IA (Gemini)", "a")], evento, "IA (Gemini)",
                                origen="ia", revisado=True)
            f = predictores.ranking(peleas, predictores.leer(evento), preds,
                                    cuotas).iloc[0]
            assert bool(f["ia_confirma"]) and f["confirmaciones"] == 3, f

            # comparar: columna propia, pero fuera del consenso humano
            predictores.guardar([fila("IA (Gemini)", "b")], evento, "IA (Gemini)",
                                origen="ia", revisado=True)
            comp = predictores.comparar(peleas, predictores.leer(evento), preds, cuotas)
            assert comp.iloc[0]["consenso"] == pelea["a"], comp.iloc[0]["consenso"]
            assert comp.iloc[0]["IA (Gemini)"] == pelea["b"]

            # pero SI se puntua: es todo el motivo de meterla al registro
            predictores.guardar_resultados([[evento, pelea["a"], pelea["b"], "a"]], evento)
            g = predictores.aciertos()
            assert "IA (Gemini)" in set(g["predictor"]), g
            suya = g[g["predictor"] == "IA (Gemini)"].iloc[0]
            assert suya["total"] == 1 and suya["aciertos"] == 0, suya
            peso = predictores.confiabilidad()
            assert 0 < float(peso[peso["predictor"] == "IA (Gemini)"]["peso"].iloc[0]) < 0.5
    finally:
        predictores.PICKS, predictores.PICKS_AUDIT, predictores.RESULTADOS = guardado


def check_ia_evaluar():
    """La cohorte de la IA se mide sola y no toca el ledger del gate."""
    import pathlib
    import tempfile

    from ufc.ia import evaluar as ia_evaluar, store as ia_store
    from ufc.registro import ledger, predictores

    archivo, informes = ia_store.ARCHIVO, ia_store.INFORMES
    guardado = predictores.PICKS, predictores.PICKS_AUDIT, predictores.RESULTADOS
    antes = ledger.LEDGER.read_bytes() if ledger.LEDGER.exists() else None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            ia_store.ARCHIVO = pathlib.Path(tmp) / "ia_consenso.csv"
            ia_store.INFORMES = pathlib.Path(tmp) / "ia_informes"
            predictores.PICKS = pathlib.Path(tmp) / "picks.csv"
            predictores.PICKS_AUDIT = pathlib.Path(tmp) / "audit.csv"
            predictores.RESULTADOS = pathlib.Path(tmp) / "resultados.csv"

            d = _ia_dossier()
            for pick, p_ia in (("a", 0.70), ("b", 0.35)):
                fila = ia_store.fila_desde(
                    {"pick": pick, "p_a": p_ia, "confianza": "media",
                     "metodo_probable": None, "razones": [],
                     "factores_no_modelables": [],
                     "apuesta": {"vale_la_pena": "no", "lado": None, "cuota": None,
                                 "casa": None, "ev": None, "cumple_regla": False,
                                 "cuota_minima": None, "motivo": ""}},
                    d, run_day=f"2026-08-0{1 if pick == 'a' else 2}", modelo_ia="test")
                fila["a"] = f"Peleador {pick.upper()}"
                fila["b"] = "Rival Comun"
                ia_store.guardar(fila, {"contra": ""})

            # el ganador se carga a mano: sin data/raw no hay resultados de ufcstats
            predictores.guardar_resultados(
                [[_IA_EVENTO["evento"], "Peleador A", "Rival Comun", "a"],
                 [_IA_EVENTO["evento"], "Peleador B", "Rival Comun", "b"]],
                _IA_EVENTO["evento"])

            df = ia_evaluar.evaluar()
            assert len(df) == 2, df
            # acerto la primera (eligio a, gano a) y erro la segunda (eligio b, gano b?)
            assert set(df["acierto"]) <= {0.0, 1.0}
            assert df["ll_ia"].notna().all() and df["ll_modelo"].notna().all()
            texto = ia_evaluar.resumen(df)
            assert "Cohorte IA" in texto and "log loss" in texto
            assert "ledger.csv`) no se toca" in texto
    finally:
        ia_store.ARCHIVO, ia_store.INFORMES = archivo, informes
        predictores.PICKS, predictores.PICKS_AUDIT, predictores.RESULTADOS = guardado
        # la cohorte preregistrada del gate no se puede haber tocado
        ahora = ledger.LEDGER.read_bytes() if ledger.LEDGER.exists() else None
        assert ahora == antes, "evaluar() no puede escribir en data/ledger.csv"


def check_gate():
    """El gate arranca cerrado y solo abre con IC limpio Y muestra preregistrada."""
    import numpy as np

    from ufc.modelo import gate

    config = gate.leer()
    assert config is not None, "config/gate.json tiene que existir"
    assert config["estado_inicial"]["autorizado"] is False
    n_minimo = config["criterio"]["n_minimo"]

    assert gate.evaluar([])["autorizado"] is False, "sin datos no se autoriza nada"

    # CLV buenisimo pero muestra corta: NO abre. El n_minimo preregistrado manda por
    # encima de cualquier prueba de potencia calculada con los propios datos.
    corto = gate.evaluar([0.05] * 10, eventos=[f"e{i}" for i in range(10)], config=config)
    assert corto["autorizado"] is False and corto["faltan"] == n_minimo - 10, corto

    # muestra suficiente y CLV claramente positivo: abre
    rng = np.random.default_rng(0)
    n = n_minimo + 50
    bueno = rng.normal(0.05, 0.02, n)
    e = gate.evaluar(bueno, eventos=[f"e{i // 3}" for i in range(n)], config=config)
    assert e["autorizado"] and e["lo"] > 0, e

    # mismo n, CLV centrado en cero: no abre
    nulo = gate.evaluar(rng.normal(0.0, 0.05, n),
                        eventos=[f"e{i // 3}" for i in range(n)], config=config)
    assert nulo["autorizado"] is False and nulo["lo"] <= 0, nulo

    # y CLV negativo tampoco, por mucha muestra que haya
    malo = gate.evaluar(rng.normal(-0.05, 0.02, n),
                        eventos=[f"e{i // 3}" for i in range(n)], config=config)
    assert malo["autorizado"] is False, malo

    par = gate.stake_params(config)
    assert 0 < par["fraccion"] <= 0.5, "nunca Kelly completo en la regla"
    assert 0 < par["tope"] <= par["tope_evento"], par


if __name__ == "__main__":
    for check in (check_fetch, check_betano, check_simetria, check_circunstancia,
                  check_confianza, check_apuesta, check_integridad_metodologica,
                  check_oddsapi, check_homonimo, check_metodo, check_archivo,
                  check_ledger, check_calibra, check_backtest, check_predictores,
                  check_apuestas, check_cartelera, check_devig, check_staking,
                  check_pool, check_gate,
                  check_ia_dossier, check_ia_dossier_sin_cuota, check_ia_validar,
                  check_ia_validar_incoherente, check_ia_validar_apuesta,
                  check_ia_store, check_ia_predictores, check_ia_evaluar,
                  check_app):
        check()
        print(f"ok  {check.__name__}")
