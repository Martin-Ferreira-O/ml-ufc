"""Pestana de inteligencia local con sincronizacion opt-in desde la VPS."""

import datetime
import zoneinfo

import streamlit as st

from ufc.intel import identities, remote, store


@st.cache_data(ttl=60, show_spinner=False)
def _ultimo():
    return store.ultimo_evento()


@st.cache_data(ttl=60, show_spinner=False)
def _profiles(fighters):
    return {fighter: identities.profiles_for(fighter) for fighter in fighters}


@st.cache_data(ttl=60, show_spinner=False)
def _metadata_local():
    return store.metadata()


@st.cache_data(ttl=300, show_spinner=False)
def _estado_vps(config):
    return remote.consultar(config)


def _configuracion_vps():
    try:
        values = st.secrets.get("intel_vps")
    except Exception:  # Streamlit lanza un error propio cuando no existe secrets.toml
        return None
    if not values:
        return None
    try:
        return remote.Config.from_mapping(values)
    except (KeyError, TypeError, ValueError):
        return None


def _fecha(value):
    if not value:
        return "Sin revisión"
    try:
        parsed = datetime.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        local = parsed.astimezone(zoneinfo.ZoneInfo(remote.TIMEZONE))
        return local.strftime("%d-%m-%Y %H:%M")
    except (TypeError, ValueError):
        return str(value)


def _fecha_evento(metadata):
    event = (metadata or {}).get("event") or {}
    return _fecha(event.get("updated_at"))


@st.fragment(run_every="5m")
def _panel_sincronizacion():
    config = _configuracion_vps()
    start, end = remote.proxima_ventana()
    st.subheader("Actualización desde la VPS")
    st.caption("La VPS revisa las fuentes todos los días a las 09:15, 10:15 y "
               "11:15 de Santiago, con un retraso aleatorio de hasta 15 minutos. "
               "La primera ventana completa el informe; las otras solo reintentan "
               "fallos y no repiten llamadas de IA si ya quedó completo.")

    local_metadata = _metadata_local()
    remote_metadata = None
    error = None
    if config:
        try:
            remote_metadata = _estado_vps(config)
        except remote.RemoteError as exc:
            error = str(exc)

    next_window = ("Hoy" if start.date() == datetime.datetime.now(
        start.tzinfo).date() else "Mañana")
    with st.container(horizontal=True):
        st.metric("Última copia local", _fecha_evento(local_metadata))
        st.metric("Última revisión VPS", _fecha_evento(remote_metadata))
        st.metric("Próxima ventana VPS",
                  f"{next_window} {start:%H:%M}–{end:%H:%M}")

    if not config:
        st.info("Configura `[intel_vps]` en `.streamlit/secrets.toml` para consultar "
                "y sincronizar la VPS.", icon=":material/key:")
        return
    if error:
        st.warning(error, icon=":material/cloud_off:")
        if st.button("Volver a consultar la VPS", icon=":material/refresh:"):
            _estado_vps.clear()
            st.rerun()
        return

    newer = remote.hay_novedades(remote_metadata, local_metadata)
    if newer:
        event = remote_metadata.get("event") or {}
        st.info(f"Hay una revisión nueva disponible: {event.get('event_name', 'UFC')} "
                f"({event.get('completed', 0)}/{event.get('fighters', 0)} peleadores).",
                icon=":material/cloud_download:")
    else:
        st.success("La copia local ya contiene la última revisión disponible en la VPS.",
                   icon=":material/cloud_done:")

    with st.container(horizontal=True):
        sync_clicked = st.button("Sincronizar ahora", type="primary",
                                 icon=":material/cloud_download:", disabled=not newer)
        if st.button("Comprobar de nuevo", icon=":material/refresh:"):
            _estado_vps.clear()
            st.rerun()
    if sync_clicked:
        try:
            with st.spinner("Descargando y validando la copia de la VPS…"):
                remote.sincronizar(config)
            _ultimo.clear()
            _profiles.clear()
            _metadata_local.clear()
            _estado_vps.clear()
            st.session_state["intel_sync_message"] = (
                "Sincronización completada. Se conservaron respaldos `.backup`.")
            st.rerun()
        except remote.RemoteError as exc:
            st.error(str(exc), icon=":material/error:")


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
    if message := st.session_state.pop("intel_sync_message", None):
        st.success(message, icon=":material/check_circle:")
    _panel_sincronizacion()
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
