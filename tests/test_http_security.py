import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


class HttpSecurityTests(unittest.TestCase):
    def setUp(self):
        from tests.server_fixture import server
        self.server = server
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.stack.enter_context(patch.object(server, "DB_PATH", str(self.root / "users.db")))
        self.stack.enter_context(patch.dict(server.app.config, {
            "TESTING": True, "SECRET_KEY": "http-security-tests",
            "UPLOAD_FOLDER": str(self.root / "uploads"),
            "WTF_CSRF_ENABLED": False, "AUTH_RATE_LIMIT_ENABLED": False,
        }))
        server.init_db()
        self.client = server.app.test_client()

    def admin(self):
        with self.client.session_transaction() as session:
            session.update(logged_in=True, username="admin", is_admin=True)

    def test_auth_rejects_non_object_json_and_non_string_fields(self):
        for route in ("/login", "/register"):
            for body in ("null", "[]", "1", '"text"', '{"username":[],"password":"pw"}',
                         '{"username":"alice","password":{}}', '{}'):
                with self.subTest(route=route, body=body):
                    response = self.client.post(route, data=body, content_type="application/json")
                    self.assertEqual(response.status_code, 400)
                    self.assertTrue(response.is_json)

    def test_logs_reject_invalid_lines_and_sources(self):
        self.admin()
        for query in ("lines=abc", "lines=0", "lines=-1", "lines=501", "source=other"):
            with self.subTest(query=query):
                self.assertEqual(self.client.get("/ops/logs?" + query).status_code, 400)

    def test_upload_limit_returns_json_413(self):
        self.admin()
        self.server.app.config["MAX_CONTENT_LENGTH"] = 32
        response = self.client.post("/energy/ai_recognize", data=b"x" * 64,
                                    content_type="application/octet-stream")
        self.assertEqual(response.status_code, 413)
        self.assertTrue(response.is_json)

    def test_registration_rejects_weak_and_oversized_passwords(self):
        for password in ("a", "12345678", "abcdefgh", "x" * 129):
            with self.subTest(password_length=len(password)):
                response = self.client.post("/register", json={"username": "alice", "password": password})
                self.assertEqual(response.status_code, 400)

    def test_csrf_protects_login_and_authenticated_write_requests(self):
        self.server.app.config["WTF_CSRF_ENABLED"] = True
        for route in ("/login", "/register"):
            with self.subTest(route=route):
                response = self.client.post(route, json={"username": "alice", "password": "Good-pass-123"})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json()["error"], "CSRF validation failed")
        self.admin()
        self.assertEqual(self.client.post("/ops/benchmark/generate_test_files").status_code, 400)

    def test_csrf_token_allows_registration_and_rejects_cross_session_replay(self):
        self.server.app.config["WTF_CSRF_ENABLED"] = True
        response = self.client.get("/auth/csrf")
        self.assertEqual(response.status_code, 200)
        token = response.get_json()["csrf_token"]
        credentials = {"username": "alice", "password": "Good-pass-123"}
        headers = {"X-CSRFToken": token}
        self.assertEqual(self.client.post("/register", json=credentials, headers=headers).status_code, 200)
        other = self.server.app.test_client()
        self.assertEqual(other.post("/login", json=credentials, headers=headers).status_code, 400)

    def test_login_rate_limit_is_shared_by_new_clients(self):
        self.server.app.config["AUTH_RATE_LIMIT_ENABLED"] = True
        for _ in range(5):
            response = self.server.app.test_client().post("/login", json={
                "username": "not-a-user", "password": "wrong-password",
            })
            self.assertEqual(response.status_code, 401)
        limited = self.server.app.test_client().post("/login", json={
            "username": "not-a-user", "password": "wrong-password",
        })
        self.assertEqual(limited.status_code, 429)
        self.assertIn("Retry-After", limited.headers)

    def test_admin_login_clears_previous_session_values(self):
        with self.client.session_transaction() as state:
            state.update(old_identity="alice", logged_in=True, username="alice", is_admin=False)
        with patch.dict(os.environ, {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "Admin-pass-123"}):
            response = self.client.post("/login", json={"username": "admin", "password": "Admin-pass-123"})
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as state:
            self.assertNotIn("old_identity", state)

    def test_unavailable_models_do_not_create_report_records_or_directories(self):
        self.admin()
        with patch.object(self.server, "HAS_FLOORPLAN_AI", False):
            response = self.client.post("/energy/ai_recognize", data={"report_number": "MODEL-MISSING"})
        self.assertEqual(response.status_code, 501)
        self.assertFalse((self.root / "uploads").exists())
        connection = self.server.get_db_connection()
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0], 0)
        finally:
            connection.close()

    def test_logout_get_is_safe_and_post_requires_csrf(self):
        self.server.app.config["WTF_CSRF_ENABLED"] = True
        self.admin()
        self.assertEqual(self.client.get("/logout").status_code, 200)
        with self.client.session_transaction() as state:
            self.assertTrue(state["logged_in"])
        self.assertEqual(self.client.post("/logout").status_code, 400)
        token = self.client.get("/auth/csrf").get_json()["csrf_token"]
        self.assertEqual(self.client.post("/logout", headers={"X-CSRFToken": token}).status_code, 302)
        with self.client.session_transaction() as state:
            self.assertNotIn("logged_in", state)

    def test_unavailable_benchmark_generation_is_explicit(self):
        self.admin()
        with patch.object(self.server, "HAS_BENCHMARK", False):
            response = self.client.post("/ops/benchmark/generate_test_files")
        self.assertEqual(response.status_code, 501)

    def test_unavailable_legacy_engine_does_not_start_a_job(self):
        self.admin()
        with self.server.app.test_request_context():
            from energy_report_storage import user_storage_key
            report = self.root / "uploads" / "energy" / user_storage_key("admin") / "CASE-ONE"
            report.mkdir(parents=True)
        self.server._upsert_report_status("admin", "CASE-ONE", "uploaded")
        with patch.object(self.server, "energyplus_engine", None, create=True):
            response = self.client.post("/energy/calculate", json={"report_number": "CASE-ONE", "mode": "energyplus"})
        self.assertEqual(response.status_code, 501)
        self.assertEqual(self.server._report_row("admin", "CASE-ONE")["status"], "uploaded")

    def test_full_task_queue_returns_503_and_terminal_job_status(self):
        from task_dispatch import QueueFull
        self.admin()
        from energy_report_storage import user_storage_key
        report = self.root / "uploads" / "energy" / user_storage_key("admin") / "QUEUE-ONE"
        report.mkdir(parents=True)
        with patch.object(self.server.task_dispatcher, "submit", side_effect=QueueFull()), \
                patch.object(self.server, "simulation_jobs") as jobs:
            response = self.client.post("/energy/calculate", json={"report_number": "QUEUE-ONE"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(self.server._report_row("admin", "QUEUE-ONE")["status"], "failed")
        self.assertEqual(jobs.update.call_args.kwargs["status"], "failed")
