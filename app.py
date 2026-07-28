"""App Streamlit: probabilidades del modelo vs la cuota de la casa."""

import streamlit as st

import predict

st.set_page_config(page_title="Predictor UFC", page_icon="🥊")
st.title("🥊 Predictor de peleas UFC")
st.caption("Probabilidades calibradas a partir del historial de ufcstats.com. "
           "La cuota de la casa es solo para comparar — no entra al modelo.")


@st.cache_resource
def _cargar():
    return predict.cargar(), predict.peleadores()


(modelo, estado), nombres = _cargar()

col_a, col_b = st.columns(2)
a = col_a.selectbox("Peleador A", nombres, index=None, placeholder="Buscá un nombre…")
b = col_b.selectbox("Peleador B", nombres, index=None, placeholder="Buscá un nombre…")

cuota = st.number_input(
    "Cuota decimal de la casa para el Peleador A (opcional)",
    min_value=1.01, value=None, step=0.05,
    help="Ej: 1.50. Se compara la probabilidad implícita (1/cuota) con la del modelo.")

if st.button("Predecir", type="primary", disabled=not (a and b)):
    if a == b:
        st.warning("Elegí dos peleadores distintos.")
    else:
        p_a, p_b = predict.predict(a, b, modelo, estado)
        st.subheader("Probabilidad según el modelo")
        st.progress(p_a, text=f"{a} — {p_a:.1%}")
        st.progress(p_b, text=f"{b} — {p_b:.1%}")

        if cuota:
            implicita = 1 / cuota
            delta = p_a - implicita
            st.subheader("Contra la casa")
            st.metric(f"Probabilidad implícita de {a}", f"{implicita:.1%}",
                      delta=f"{delta:+.1%} del modelo")
            if abs(delta) < 0.03:
                st.info("El modelo y la casa están de acuerdo.")
            elif delta > 0:
                st.success(f"El modelo le ve **más** chances a {a} que la casa "
                           f"({p_a:.1%} vs {implicita:.1%}).")
            else:
                st.warning(f"El modelo le ve **menos** chances a {a} que la casa "
                           f"({p_a:.1%} vs {implicita:.1%}).")
