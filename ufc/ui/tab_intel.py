"""Pestana de inteligencia: lee el ultimo informe local, sin tocar la red."""

import streamlit as st

from ufc.intel import identities, store


@st.cache_data(ttl=60, show_spinner=False)
def _ultimo():
    return store.ultimo_evento()


@st.cache_data(ttl=60, show_spinner=False)
def _profiles(fighters):
    return {fighter: identities.profiles_for(fighter) for fighter in fighters}


def _etiqueta(score):
    if score is None:
        return "Sin analizar"
    if score <= -2:
        return f"{score:+d} · posible impacto negativo"
    if score >= 2:
        return f"{score:+d} · posible impacto positivo"
    return f"{score:+d} · sin impacto material"


def render():
    st.header("Inteligencia")
    st.caption("Noticias y fuentes publicas revisadas a diario. La valoracion resume "
               "contexto reciente; no cambia la probabilidad del modelo ni recomienda "
               "una apuesta.")
    evento = _ultimo()
    if not evento:
        st.info("Todavia no hay un informe local. Ejecuta `python -m ufc.intel.bot` "
                "en la VPS para revisar la proxima cartelera.",
                icon=":material/manage_search:")
        return

    st.subheader(evento["event_name"])
    st.caption(f"Evento: {evento['event_date']} · ultima revision: "
               f"{evento['updated_at'][:16].replace('T', ' ')} UTC")
    total = len(evento["checks"])
    evidencias = sum(x["evidence_count"] for x in evento["checks"])
    alertas = sum((x["score"] or 0) <= -2 for x in evento["checks"])
    profiles = _profiles(tuple(x["fighter"] for x in evento["checks"]))
    covered = sum(any(p.get("confidence") == "official"
                      for p in profiles[x["fighter"]]) for x in evento["checks"])
    with st.container(horizontal=True):
        st.metric("Peleadores revisados", f"{evento['completed']}/{total}")
        st.metric("Evidencias archivadas", evidencias)
        st.metric("Con perfil social", f"{covered}/{total}")
        st.metric("Alertas contextuales", alertas)

    st.warning("**No es una senal de apuesta.** El score todavia no tiene validacion "
               "historica ni forward test; sirve para priorizar que evidencias leer.",
               icon=":material/science:")
    for check in evento["checks"]:
        report = check["report"]
        with st.container(border=True):
            with st.container(horizontal=True, horizontal_alignment="distribute",
                              vertical_alignment="center"):
                st.subheader(f"{check['fighter']} vs {check['opponent']}")
                score = check["score"]
                color = "red" if score is not None and score <= -2 else \
                    "green" if score is not None and score >= 2 else "gray"
                st.badge(_etiqueta(score), color=color,
                         icon=":material/report:" if color == "red"
                         else ":material/fact_check:")
            st.write(report.get("resumen") or "Sin resumen.")
            for warning in report.get("advertencias", []):
                st.warning(warning, icon=":material/warning:")
            social = [x for x in profiles.get(check["fighter"], [])
                      if x.get("confidence") == "official"]
            candidates = [x for x in profiles.get(check["fighter"], [])
                          if x.get("confidence") == "candidate"]
            if social:
                st.caption("Perfiles identificados: " + " · ".join(
                    f"[{x['platform'].capitalize()}]({x['url']})" for x in social))
            elif candidates:
                st.caption("Wikidata propuso perfiles, pero quedan pendientes de "
                           "validacion antes de recolectarlos.")
            if check["confidence"]:
                st.caption(f"Confianza del resumen: {check['confidence']} · "
                           f"{check['evidence_count']} evidencias revisadas")
            evidencias_por_ref = {f"E{x['position']}": x for x in check["evidencias"]}
            for hallazgo in report.get("hallazgos", []):
                refs = [r for r in hallazgo.get("evidencias", [])
                        if r in evidencias_por_ref]
                st.markdown(f"**{hallazgo['titulo']}** · impacto "
                            f"`{hallazgo['impacto']:+d}` · {hallazgo['certeza']}")
                st.write(hallazgo["explicacion"])
                if refs:
                    st.caption("Fuentes: " + " · ".join(
                        f"[{r} — {evidencias_por_ref[r]['source']}]"
                        f"({evidencias_por_ref[r]['url']})" for r in refs))
            if check["evidencias"]:
                with st.expander(f"Ver las {len(check['evidencias'])} evidencias"):
                    for e in check["evidencias"]:
                        st.markdown(f"**E{e['position']} · [{e['title']}]({e['url']})**")
                        st.caption(f"{e['source']} · {e['published_at'] or 'fecha desconocida'}")
                        if e["snippet"]:
                            st.write(e["snippet"])
