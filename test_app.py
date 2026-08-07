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
ODDSAPI = [{
    "home_team": "Jalin Turner", "away_team": "Kaue Fernandes",
    "bookmakers": [
        {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Jalin Turner", "price": 2.30},
            {"name": "Kaue Fernandes", "price": 1.62}]}]},
        {"key": "otra", "markets": [{"key": "h2h", "outcomes": [
            {"name": "Jalin Turner", "price": 2.10},
            {"name": "Kaue Fernandes", "price": 1.70}]}]},
    ]}]


def check_oddsapi():
    """Consenso multi-casa: mediana desvigueada, mejor cuota por lado, orientacion."""
    t = oddsapi._parsear(ODDSAPI)
    fila = t[("jalin turner", "kaue fernandes")]
    assert fila["casas"] == 2 and fila["mejor"] == (2.30, 1.70), fila
    assert 0 < fila["p_x"] < 0.5, fila         # Turner es el underdog en ambas casas

    r = oddsapi.buscar(t, "Kauê Fernandes", "Jalin Turner")   # acento y orden dados vuelta
    assert r and abs(r["p_a"] - (1 - fila["p_x"])) < 1e-12, r
    assert r["mejor"] == (1.70, 2.30), r
    assert oddsapi.buscar(t, A, B) is None

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
    assert len(consensos) == 1 and "2 casas" in consensos[0], consensos


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


if __name__ == "__main__":
    for check in (check_fetch, check_betano, check_simetria, check_circunstancia,
                  check_confianza, check_apuesta, check_integridad_metodologica,
                  check_oddsapi, check_homonimo, check_metodo, check_archivo,
                  check_ledger, check_calibra, check_backtest, check_predictores,
                  check_apuestas, check_cartelera, check_app):
        check()
        print(f"ok  {check.__name__}")
