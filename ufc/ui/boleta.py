"""Estado de sesion compartido por Predictores y Apuestas."""

import streamlit as st


CLAVE = "boleta_apuestas"


def leer():
    return st.session_state.setdefault(CLAVE, [])


def agregar(evento, fecha, a, b, pick, cuota):
    boleta = leer()
    if boleta and boleta[0]["evento"] != evento:
        return False
    nueva = {"evento": evento, "fecha": fecha, "a": a, "b": b,
             "pick": pick, "cuota": float(cuota)}
    clave = (a, b)
    boleta[:] = [x for x in boleta if (x["a"], x["b"]) != clave]
    boleta.append(nueva)
    return True


def quitar(indice):
    leer().pop(indice)


def vaciar():
    st.session_state[CLAVE] = []
