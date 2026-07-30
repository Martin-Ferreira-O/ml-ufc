"""Check del camino que se sirve: `python test_app.py`. Sin framework, solo asserts.

Cubre lo que se rompe en silencio: que la prediccion no dependa del orden de los
peleadores, que el nivel de confianza salga del tramo correcto, y que la app renderice
con y sin cuotas. Requiere `model.pkl` y `data/fighter_state.csv` (los genera train.py).
"""

import predict

A, B = "Khamzat Chimaev", "Sean Strickland"


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


def check_app():
    """La app tiene que renderizar sin excepcion con y sin cuotas."""
    from streamlit.testing.v1 import AppTest

    for cuotas in ((1.35, 3.20), None):
        at = AppTest.from_file("app.py", default_timeout=120).run()
        at.selectbox[0].set_value(A)
        at.selectbox[1].set_value(B)
        if cuotas:
            at.number_input[0].set_value(cuotas[0])
            at.number_input[1].set_value(cuotas[1])
        at.button[0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        # sin cuotas no hay nivel de confianza: seria inventarlo
        avisos = len(at.success) + len(at.error)
        assert avisos == (1 if cuotas else 0), (cuotas, avisos)


if __name__ == "__main__":
    for check in (check_simetria, check_confianza, check_app):
        check()
        print(f"ok  {check.__name__}")
