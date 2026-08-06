"""Suite offline del bot de inteligencia. Ejecutar: python test_intel.py."""

import datetime
import json
import pathlib
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

from streamlit.testing.v1 import AppTest

from ufc.intel import analyzer, bot, identities, remote, sources, store


RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<title>Noticias MMA</title><item><title>Fighter supera el pesaje</title>
<link>https://example.com/noticia</link><pubDate>Sat, 01 Aug 2026 12:00:00 GMT</pubDate>
<source>Fuente fiable</source><description><![CDATA[<b>Peso confirmado</b> sin problemas.]]></description>
</item></channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<title>Canal oficial</title><entry><title>Entrenamiento abierto</title>
<link href="https://example.com/video"/><updated>2026-08-01T10:00:00Z</updated>
<summary>Sesion publica del campamento.</summary></entry></feed>"""


def pagina_intel_vacia():
    from ufc.ui import comunes, tab_intel
    comunes.intel_evento = lambda: None
    tab_intel._metadata_local = lambda: {"event": None}
    tab_intel._configuracion_vps = lambda: None
    tab_intel.render()


class IntelTest(unittest.TestCase):
    def test_ultimo_evento_usa_solo_la_revision_diaria_mas_reciente(self):
        evento = {"evento": "UFC Diario", "fecha": "2026-08-08"}
        report = {"resumen": "Sin novedades.", "estado": "informativo",
                  "confianza": "alta", "valoracion": 0, "hallazgos": []}
        with tempfile.TemporaryDirectory() as td:
            db = store.conectar(pathlib.Path(td) / "intel.db")
            for day, fighter in (("2026-08-02", "Anterior"),
                                 ("2026-08-03", "Actual")):
                run = store.crear_run(db, evento, day, "gemini", "modelo", 1)
                store.guardar_check(db, run, day, evento, fighter, "Rival",
                                    "complete", [], report)
                store.terminar_run(db, run, "complete", 1, 0)
            latest = store.ultimo_evento(db)
            self.assertEqual(latest["run_day"], "2026-08-03")
            self.assertEqual([x["fighter"] for x in latest["checks"]], ["Actual"])
            self.assertEqual(store.metadata(db)["event"]["fighters"], 1)
            db.close()

    def test_detecta_novedad_y_calcula_proxima_ventana(self):
        local = {"event": {"event_date": "2026-08-08", "run_day": "2026-08-02",
                           "updated_at": "2026-08-02T12:00:00+00:00"}}
        current = {"event": {"event_date": "2026-08-08", "run_day": "2026-08-03",
                             "updated_at": "2026-08-03T12:00:00+00:00"}}
        self.assertTrue(remote.hay_novedades(current, local))
        self.assertFalse(remote.hay_novedades(current, current))
        now = datetime.datetime(2026, 8, 4, 12, 0, tzinfo=datetime.timezone.utc)
        start, end = remote.proxima_ventana(now)
        self.assertEqual((start.hour, start.minute), (9, 15))
        self.assertEqual(end - start, datetime.timedelta(minutes=15))

    def test_sincronizacion_descarga_valida_respalda_y_reemplaza(self):
        evento = {"evento": "UFC Remoto", "fecha": "2026-08-08"}
        report = {"resumen": "Actualizado.", "estado": "informativo",
                  "confianza": "alta", "valoracion": 0, "hallazgos": []}
        with tempfile.TemporaryDirectory() as td:
            base = pathlib.Path(td)
            source_dir, target_dir = base / "source", base / "target"
            source_dir.mkdir()
            target_dir.mkdir()
            remote_db = store.conectar(source_dir / "intel.db")
            run = store.crear_run(remote_db, evento, "2026-08-03", "gemini", "m", 1)
            store.guardar_check(remote_db, run, "2026-08-03", evento, "A", "B",
                                "complete", [], report)
            store.terminar_run(remote_db, run, "complete", 1, 0)
            remote_db.close()
            (source_dir / "intel_profiles.csv").write_text(
                "fighter,platform,url,confidence\nA,x,https://x.com/a,official\n")
            (source_dir / "intel_identity_status.csv").write_text(
                "fighter,status,checked_at\nA,official,2026-08-03T12:00:00Z\n")
            targets = {name: target_dir / name for name in remote.FILES}
            for path in targets.values():
                path.write_text("copia anterior")

            def fake_runner(command, **kwargs):
                name = pathlib.Path(command[-2].split(":", 1)[1]).name
                shutil.copy2(source_dir / name, command[-1])
                return SimpleNamespace(stdout="", stderr="", returncode=0)

            old_files = remote.FILES
            try:
                remote.FILES = targets
                metadata = remote.sincronizar(
                    remote.Config("example.com", "user"), runner=fake_runner)
            finally:
                remote.FILES = old_files
            self.assertEqual(metadata["event"]["event_name"], "UFC Remoto")
            self.assertEqual(targets["intel_profiles.csv"].read_text(),
                             (source_dir / "intel_profiles.csv").read_text())
            self.assertEqual(targets["intel.db"].with_name(
                "intel.db.backup").read_text(), "copia anterior")

    def test_reintenta_429_respetando_retry_info(self):
        class QuotaError(Exception):
            code = 429
            details = {"error": {"details": [{
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "2s"}]}}
        calls = []
        sleeps = []
        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise QuotaError("limite")
            return "ok"
        result = bot.con_reintentos(flaky, sleep=sleeps.append)
        self.assertEqual(result, "ok")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [3.0, 3.0])

    def test_dia_operativo_usa_zona_configurada(self):
        fake_now = datetime.datetime(2026, 8, 4, 2, 30,
                                     tzinfo=datetime.timezone.utc)
        class FakeDateTime(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return fake_now.astimezone(tz) if tz else fake_now
        with mock.patch.object(bot.datetime, "datetime", FakeDateTime), \
                mock.patch.dict("os.environ", {
                    "UFC_INTEL_TIMEZONE": "America/Santiago"}):
            self.assertEqual(bot.hoy_local().isoformat(), "2026-08-03")

    def test_grounding_solo_archiva_urls_con_respaldo(self):
        chunks = [
            SimpleNamespace(web=SimpleNamespace(
                uri="https://example.com/entrevista", title="Entrevista",
                domain="example.com")),
            SimpleNamespace(web=SimpleNamespace(
                uri="https://example.com/sin-respaldo", title="Sin respaldo",
                domain="example.com")),
        ]
        supports = [
            SimpleNamespace(grounding_chunk_indices=[0],
                            segment=SimpleNamespace(text="Confirmo que viajo manana.")),
            SimpleNamespace(grounding_chunk_indices=[0],
                            segment=SimpleNamespace(text="El campamento termino.")),
        ]
        response = SimpleNamespace(candidates=[SimpleNamespace(
            grounding_metadata=SimpleNamespace(
                grounding_chunks=chunks, grounding_supports=supports))])
        evidence = analyzer.grounding_evidence(response)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].url, "https://example.com/entrevista")
        self.assertEqual(evidence[0].kind, "grounding")
        self.assertIn("viajo manana", evidence[0].snippet)
        self.assertIn("campamento termino", evidence[0].snippet)

    def test_bot_suma_grounding_y_tokens_antes_de_analizar(self):
        class Provider:
            name = "gemini"
            model = "fake-gemini"
            def discover(self, fighter, opponent, evento):
                return [sources.Evidence(
                    url="https://example.com/grounded", title="Reporte confirmado",
                    source="Google Search (Gemini)", published_at=None,
                    snippet="Reporte primario.", kind="grounding")], {
                        "prompt_tokens": 10, "output_tokens": 2}
            def analyze(self, fighter, opponent, evento, evidencias):
                self.seen = evidencias
                return {"resumen": "Sin impacto material.", "estado": "informativo",
                        "confianza": "media", "valoracion": 0,
                        "hallazgos": []}, {
                            "prompt_tokens": 7, "output_tokens": 3}
        provider = Provider()
        evento = {"evento": "UFC Test", "fecha": "2026-08-08",
                  "peleas": [{"a": "A", "b": "B"}]}
        with tempfile.TemporaryDirectory() as td:
            db = store.conectar(pathlib.Path(td) / "intel.db")
            result = bot.ejecutar(evento, db=db, provider=provider, grounding=True,
                                  force=True, run_day="2026-08-02",
                                  recolector=lambda fighter: [])
            self.assertEqual(result["completed"], 2)
            self.assertEqual(len(provider.seen), 1)
            rows = db.execute(
                "SELECT prompt_tokens, output_tokens, evidence_count FROM intel_checks"
            ).fetchall()
            self.assertTrue(all(tuple(row) == (17, 5, 1) for row in rows))
            self.assertEqual(db.execute(
                "SELECT kind FROM intel_evidence").fetchone()[0], "grounding")
            db.close()

    def test_wikidata_resuelve_y_cachea_perfiles(self):
        class Respuesta:
            status_code = 200
            headers = {}
            def __init__(self, data=None, text=""):
                self.data, self.text = data or {}, text
            def raise_for_status(self):
                pass
            def json(self):
                return self.data
        class Sesion:
            calls = 0
            def get(self, url, **kwargs):
                self.calls += 1
                if url.startswith("https://www.ufc.com/athlete/"):
                    data = {"@context": "https://schema.org", "@type": "Person",
                            "name": "Fighter A", "sameAs": [
                                "https://twitter.com/fighter_a",
                                "https://instagram.com/fighter.a",
                                "https://www.youtube.com/channel/UC1234567890123456789012"]}
                    return Respuesta(text=("<script type=\"application/ld+json\">" +
                                           json.dumps(data) + "</script>"))
                action = kwargs["params"]["action"]
                if action == "wbsearchentities":
                    return Respuesta({"search": [{"id": "Q1", "label": "Fighter A",
                                                   "description": "MMA fighter"}]})
                claim = lambda value, rank="normal": {
                    "rank": rank, "mainsnak": {"datavalue": {"value": value}}}
                return Respuesta({"entities": {"Q1": {"claims": {
                    "P2002": [claim("fighter_a")],
                    "P2003": [claim("fighter.a")],
                    "P2397": [claim("UC1234567890123456789012")],
                }}}})
        session = Sesion()
        with tempfile.TemporaryDirectory() as td:
            profiles = pathlib.Path(td) / "profiles.csv"
            statuses = pathlib.Path(td) / "status.csv"
            first = identities.sync(["Fighter A"], session=session,
                                    profiles_path=profiles, status_path=statuses,
                                    delay=0,
                                    now=identities.datetime.datetime(
                                        2026, 8, 2, tzinfo=identities.datetime.timezone.utc))
            second = identities.sync(["Fighter A"], session=session,
                                     profiles_path=profiles, status_path=statuses,
                                     delay=0,
                                     now=identities.datetime.datetime(
                                         2026, 8, 3, tzinfo=identities.datetime.timezone.utc))
            self.assertEqual(first, {"fighters": 1, "covered": 1, "candidates": 0,
                                     "checked": 1, "errors": 0})
            self.assertEqual(second["checked"], 0)
            self.assertEqual(session.calls, 1)
            stored = identities.profiles_for("fighter a", profiles)
            self.assertEqual({x["platform"] for x in stored},
                             {"x", "instagram", "youtube"})
            without_token = identities.feed_sources("Fighter A", profiles)
            self.assertEqual([x["kind"] for x in without_token], ["youtube"])
            with mock.patch.dict("os.environ", {"X_BEARER_TOKEN": "test"}):
                active = identities.feed_sources("Fighter A", profiles)
            self.assertEqual({x["kind"] for x in active}, {"x", "youtube"})

    def test_wikidata_no_activa_un_perfil_sin_validacion_oficial(self):
        class Respuesta:
            headers = {}
            text = ""
            def __init__(self, status_code, data=None):
                self.status_code, self.data = status_code, data or {}
            def raise_for_status(self):
                pass
            def json(self):
                return self.data
        class Sesion:
            def get(self, url, **kwargs):
                if url.startswith("https://www.ufc.com/athlete/"):
                    return Respuesta(404)
                if kwargs["params"]["action"] == "wbsearchentities":
                    return Respuesta(200, {"search": [{"id": "QBAD",
                        "label": "Quillan Salkilld", "description": "MMA fighter"}]})
                claim = {"rank": "normal", "mainsnak": {
                    "datavalue": {"value": "estebanribovicsmma"}}}
                return Respuesta(200, {"entities": {"QBAD": {
                    "claims": {"P2003": [claim]}}}})
        profiles, status = identities.resolve("Quillan Salkilld", session=Sesion())
        self.assertEqual(status["status"], "candidate")
        self.assertEqual(profiles[0]["confidence"], "candidate")
        self.assertEqual(profiles[0]["kind"], "")

    def test_parsea_rss_y_atom(self):
        rss = sources.parsear_feed(RSS, kind="news")
        atom = sources.parsear_feed(ATOM, kind="youtube")
        self.assertEqual(rss[0].source, "Fuente fiable")
        self.assertEqual(rss[0].snippet, "Peso confirmado sin problemas.")
        self.assertEqual(atom[0].url, "https://example.com/video")
        self.assertEqual(atom[0].kind, "youtube")

    def test_descarta_listings_de_cuotas_como_si_fueran_noticias(self):
        market = sources.Evidence(
            url="https://example.com/odds", title="A vs B MMA Betting Odds & Lines",
            source="DraftKings", published_at=None, snippet="", kind="news")
        article = sources.Evidence(
            url="https://example.com/news", title="Fighter accepts fight on short notice",
            source="Sports Illustrated", published_at=None, snippet="", kind="news")
        self.assertFalse(sources.noticia_util(market))
        self.assertTrue(sources.noticia_util(article))

    def test_validacion_elimina_citas_inventadas(self):
        ev = sources.parsear_feed(RSS)
        report = {"resumen": "Hay una confirmacion y un rumor sin fuente.",
                  "confianza": "alta", "hallazgos": [
                      {"titulo": "Peso confirmado", "categoria": "peso", "impacto": 2,
                       "certeza": "confirmado", "explicacion": "Dio el peso.",
                       "evidencias": ["E1"]},
                      {"titulo": "Lesion inventada", "categoria": "lesion", "impacto": -5,
                       "certeza": "confirmado", "explicacion": "No existe.",
                       "evidencias": ["E99"]}]}
        limpio = analyzer.validar(report, ev)
        self.assertEqual(limpio["valoracion"], 2)
        self.assertEqual(len(limpio["hallazgos"]), 1)
        self.assertEqual(limpio["hallazgos"][0]["evidencias"], ["E1"])

    def test_validacion_elimina_argumentos_desde_ausencia_de_reportes(self):
        ev = sources.parsear_feed(RSS)
        report = {"resumen": ("No se han reportado lesiones ni problemas de peso. "
                              "La fuente confirma un cambio de rival."),
                  "confianza": "alta", "hallazgos": [{
                      "titulo": "Cambio", "categoria": "reemplazo", "impacto": 0,
                      "certeza": "confirmado", "explicacion": "Cambio confirmado.",
                      "evidencias": ["E1"]}]}
        clean = analyzer.validar(report, ev)
        self.assertEqual(clean["resumen"], "La fuente confirma un cambio de rival.")

    def test_x_solo_por_api_oficial(self):
        class Respuesta:
            def raise_for_status(self):
                pass
            def json(self):
                return {"data": [{"id": "123", "text": "Peso listo para manana",
                                   "created_at": "2026-08-01T12:00:00Z"}]}
        class Sesion:
            def get(self, url, **kwargs):
                self.url, self.kwargs = url, kwargs
                return Respuesta()
        sesion = Sesion()
        items = sources.x_posts({"url": "https://x.com/gamrot_mateusz",
                                 "label": "Cuenta oficial"},
                                session=sesion, token="token-de-prueba")
        self.assertEqual(sesion.url, sources.X_SEARCH)
        self.assertEqual(sesion.kwargs["params"]["query"],
                         "from:gamrot_mateusz -is:retweet")
        self.assertEqual(items[0].url, "https://x.com/gamrot_mateusz/status/123")
        self.assertEqual(items[0].kind, "x")

    def test_store_preserva_posicion_e_idempotencia(self):
        with tempfile.TemporaryDirectory() as td:
            db = store.conectar(pathlib.Path(td) / "intel.db")
            evento = {"evento": "UFC Test", "fecha": "2026-08-08"}
            run = store.crear_run(db, evento, "2026-08-02", "none", None, 1)
            ev = sources.parsear_feed(RSS)
            report = analyzer.sin_ia(ev)
            store.guardar_check(db, run, "2026-08-02", evento, "A", "B",
                                "collected", ev, report)
            self.assertTrue(store.ya_revisado(db, "2026-08-02", "2026-08-08", "A"))
            self.assertFalse(store.ya_revisado(db, "2026-08-02", "2026-08-08", "A",
                                               require_analysis=True))
            ultimo = store.ultimo_evento(db)
            self.assertEqual(ultimo["checks"][0]["evidencias"][0]["position"], 1)
            db.close()

    def test_resumen_cli_muestra_hallazgo_y_url(self):
        evento = {"evento": "UFC Resumen", "fecha": "2026-08-08"}
        evidence = sources.Evidence(
            url="https://example.com/reemplazo", title="Reemplazo confirmado",
            source="Fuente", published_at=None, snippet="", kind="news")
        report = {"resumen": "Acepto con poco aviso.", "estado": "alerta",
                  "confianza": "alta", "valoracion": -2, "hallazgos": [{
                      "titulo": "Poco aviso", "categoria": "reemplazo",
                      "impacto": -2, "certeza": "confirmado",
                      "explicacion": "Diez dias.", "evidencias": ["E1"]}]}
        with tempfile.TemporaryDirectory() as td:
            db = store.conectar(pathlib.Path(td) / "intel.db")
            run = store.crear_run(db, evento, "2026-08-03", "gemini", "modelo", 1)
            store.guardar_check(db, run, "2026-08-03", evento, "A", "B",
                                "complete", [evidence], report)
            store.terminar_run(db, run, "complete", 1, 0)
            output = StringIO()
            with redirect_stdout(output):
                self.assertTrue(bot.imprimir_resumen(db))
            self.assertIn("A vs B — -2", output.getvalue())
            self.assertIn("https://example.com/reemplazo", output.getvalue())
            db.close()

    def test_salud_exige_run_reciente_completo_y_con_ia(self):
        evento = {"evento": "UFC Salud", "fecha": "2026-08-08"}
        now = datetime.datetime(2026, 8, 2, 12, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as td:
            db = store.conectar(pathlib.Path(td) / "intel.db")
            run = store.crear_run(db, evento, "2026-08-02", "gemini", "modelo", 28)
            store.terminar_run(db, run, "complete", 28, 0)
            db.execute("UPDATE intel_runs SET started_at=?, finished_at=? WHERE id=?",
                       (now.isoformat(), now.isoformat(), run))
            db.commit()
            self.assertTrue(bot.salud(db, now=now))
            db.execute("UPDATE intel_runs SET provider='none' WHERE id=?", (run,))
            db.commit()
            self.assertFalse(bot.salud(db, now=now))
            db.execute("UPDATE intel_runs SET provider='gemini', finished_at=? WHERE id=?",
                       ((now - datetime.timedelta(hours=40)).isoformat(), run))
            db.commit()
            self.assertFalse(bot.salud(db, now=now))
            db.close()

    def test_adaptador_openai_compatible(self):
        class Respuesta:
            def raise_for_status(self):
                pass
            def json(self):
                contenido = {"resumen": "Peso confirmado.", "estado": "informativo",
                             "confianza": "alta", "hallazgos": [{
                                 "titulo": "Dio el peso", "categoria": "peso",
                                 "impacto": 1, "certeza": "confirmado",
                                 "explicacion": "La fuente informa el peso.",
                                 "evidencias": ["E1"]}]}
                return {"choices": [{"message": {"content": json.dumps(contenido)}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        class Sesion:
            def post(self, url, **kwargs):
                self.url, self.kwargs = url, kwargs
                return Respuesta()
        sesion = Sesion()
        provider = analyzer.OpenAICompatible(base_url="https://api.example/v1",
                                             api_key="test", model="modelo",
                                             session=sesion)
        ev = sources.parsear_feed(RSS)
        report, usage = provider.analyze("A", "B",
                                         {"evento": "UFC", "fecha": "2026-08-08"}, ev)
        self.assertEqual(sesion.url, "https://api.example/v1/chat/completions")
        self.assertEqual(report["valoracion"], 1)
        self.assertEqual(usage["prompt_tokens"], 100)

    def test_cobertura_28_y_reanudacion(self):
        fixture = pathlib.Path(__file__).parent / "tests/fixtures/intel_event.json"
        evento = json.loads(fixture.read_text())
        with tempfile.TemporaryDirectory() as td:
            db = store.conectar(pathlib.Path(td) / "intel.db")
            fake = lambda fighter: []
            primero = bot.ejecutar(evento, db=db, sin_ia=True, force=True,
                                   run_day="2026-08-02", recolector=fake)
            segundo = bot.ejecutar(evento, db=db, sin_ia=True, force=False,
                                   run_day="2026-08-02", recolector=fake)
            self.assertEqual(primero["completed"], 28)
            self.assertEqual(segundo["skipped"], 28)
            n = db.execute("SELECT COUNT(*) FROM intel_checks").fetchone()[0]
            self.assertEqual(n, 28)
            db.close()

    def test_ui_estado_vacio(self):
        at = AppTest.from_function(pagina_intel_vacia).run(timeout=20)
        self.assertFalse(at.exception)
        self.assertEqual(at.title[0].value, "Inteligencia")
        self.assertTrue(any("Todavía no hay" in item.value for item in at.info))


if __name__ == "__main__":
    unittest.main(verbosity=2)
