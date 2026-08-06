"""Pestana de inteligencia local con sincronizacion opt-in desde la VPS."""

import datetime
import zoneinfo

import streamlit as st

from ufc.intel import remote, store
from ufc.ui import comunes


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
    with st.expander("Actualización desde la VPS", icon=":material/cloud_sync:",
                     expanded=True):
        _cuerpo_sincronizacion(config, start, end)


def _cuerpo_sincronizacion(config, start, end):
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
        st.metric("Última copia local", _fecha_evento(local_metadata), border=True)
        st.metric("Última revisión VPS", _fecha_evento(remote_metadata), border=True)
        st.metric("Próxima ventana VPS",
                  f"{next_window} {start:%H:%M}–{end:%H:%M}", border=True)

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
            comunes.intel_evento.clear()
            comunes.intel_perfiles.clear()
            _metadata_local.clear()
            _estado_vps.clear()
            st.session_state["intel_sync_message"] = (
                "Sincronización completada. Se conservaron respaldos `.backup`.")
            st.rerun()
        except remote.RemoteError as exc:
            st.error(str(exc), icon=":material/error:")


def render():
    if message := st.session_state.pop("intel_sync_message", None):
        st.success(message, icon=":material/check_circle:")
    evento = comunes.intel_evento()
    checks = evento["checks"] if evento else []
    alertas = sum((x["score"] or 0) <= -2 for x in checks)

    chips = []
    if evento:
        chips.append((evento["event_name"], "blue", ":material/stadium:"))
        if alertas:
            chips.append((f"{alertas} alertas contextuales", "red",
                          ":material/notification_important:"))
    comunes.encabezado(
        "Inteligencia",
        "Noticias y fuentes públicas revisadas a diario. La valoración resume contexto "
        "reciente; no cambia la probabilidad del modelo ni recomienda una apuesta.",
        seccion="Contexto", chips=chips)

    _panel_sincronizacion()
    if not evento:
        st.info("Todavía no hay un informe local. Ejecutá `python -m ufc.intel.bot` "
                "en la VPS para revisar la próxima cartelera.",
                icon=":material/manage_search:")
        return

    total = len(checks)
    evidencias = sum(x["evidence_count"] for x in checks)
    profiles = comunes.intel_perfiles(tuple(x["fighter"] for x in checks))
    covered = sum(any(p.get("confidence") == "official"
                      for p in profiles[x["fighter"]]) for x in checks)
    with st.container(horizontal=True):
        st.metric("Peleadores revisados", f"{evento['completed']}/{total}", border=True)
        st.metric("Evidencias archivadas", evidencias, border=True)
        st.metric("Con perfil social", f"{covered}/{total}", border=True)
        st.metric("Alertas contextuales", alertas, border=True,
                  help="Peleadores con un score de −2 o menos.")
    st.caption(f"Evento del {evento['event_date']} · última revisión "
               f"{evento['updated_at'][:16].replace('T', ' ')} UTC")

    st.warning("**No es una señal de apuesta.** El score todavía no tiene validación "
               "histórica ni forward test; sirve para priorizar qué evidencias leer.",
               icon=":material/science:")

    # Filtro de pantalla: en una cartelera completa son 20 fichas y lo que se busca casi
    # siempre es la que trae una alerta.
    solo_alertas = st.toggle("Solo peleadores con alerta", key="intel_solo_alertas",
                             disabled=not alertas,
                             help="Score de −2 o menos." if alertas else
                                  "Ningún peleador tiene alerta en esta revisión.")
    visibles = [c for c in checks if not solo_alertas or (c["score"] or 0) <= -2]
    for check in visibles:
        comunes.intel_tarjeta(check, profiles[check["fighter"]])
