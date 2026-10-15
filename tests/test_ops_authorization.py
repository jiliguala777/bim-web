import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


OPS_REQUESTS = (
    ("GET", "/ops/health"),
    ("GET", "/ops/metrics"),
    ("GET", "/ops/logs?lines=5"),
    ("GET", "/ops/benchmark"),
    ("POST", "/ops/benchmark"),
    ("POST", "/ops/benchmark/generate_test_files"),
)


class OpsAuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.server_fixture import server as web_server_server

        cls.server = web_server_server

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.root = root
        self.log_path = root / "gunicorn.log"
        self.log_path.write_text("private-ops-log-entry\n", encoding="utf-8")
        self.stack.enter_context(patch.dict(self.server.app.config, {
            "TESTING": True,
            "SECRET_KEY": "ops-authorization-test",
            "UPLOAD_FOLDER": str(root / "uploads"),
        }))
        self.stack.enter_context(patch.dict(os.environ, {
            "GUNICORN_LOG": str(self.log_path),
            "ADMIN_USER": "ops-admin",
            "ADMIN_PASSWORD": "test-admin-password",
        }))
        self.stack.enter_context(patch.object(self.server, "DB_PATH", str(root / "users.db")))
        self.server.init_db()
        # Keep slow host inspection and optional benchmark tools outside the test.
        self.processes = self.stack.enter_context(patch.object(
            self.server.psutil, "process_iter", return_value=[]))
        self.cpu = self.stack.enter_context(patch.object(
            self.server.psutil, "cpu_percent", return_value=0))
        self.stack.enter_context(patch.object(self.server.psutil, "net_connections", return_value=[]))
        self.stack.enter_context(patch.object(self.server.os, "getloadavg", return_value=(0, 0, 0), create=True))
        self.log_reader = self.stack.enter_context(patch("routes.ops.open", wraps=open, create=True))
        self.subprocess = self.stack.enter_context(patch.object(
            self.server.subprocess, "run", return_value=SimpleNamespace(stdout="private-nginx-log-entry")))
        self.stack.enter_context(patch.object(self.server, "HAS_BENCHMARK", True))
        self.benchmark = self.stack.enter_context(patch.object(
            self.server, "run_benchmark", return_value={"case": "test-case"}, create=True))
        self.stack.enter_context(patch.object(self.server, "BESTEST_BENCHMARKS", ["test-case"], create=True))
        self.generate_dxf = self.stack.enter_context(patch.object(
            self.server, "generate_test_dxf", return_value=True, create=True))
        self.stack.enter_context(patch.object(self.server, "HAS_IFC", False))
        self.client = self.server.app.test_client()

    def assert_ops_denied(self, client, status):
        for method, path in OPS_REQUESTS:
            with self.subTest(method=method, path=path):
                response = client.open(path, method=method)
                self.assertEqual(response.status_code, status, response.get_data(as_text=True))
                self.assertTrue(response.is_json)
                self.assertIn("error", response.get_json())
                self.assertNotIn("private-ops-log-entry", response.get_data(as_text=True))
        # Rejected requests must not collect metrics, read logs, or create files.
        self.processes.assert_not_called()
        self.cpu.assert_not_called()
        self.log_reader.assert_not_called()
        self.subprocess.assert_not_called()
        self.benchmark.assert_not_called()
        self.generate_dxf.assert_not_called()
        self.assertFalse((self.root / "uploads" / "ops").exists())

    def test_anonymous_requests_are_rejected_before_ops_work(self):
        self.assert_ops_denied(self.client, 401)

    def test_incomplete_sessions_cannot_use_an_admin_flag(self):
        for state in (
            {"is_admin": True},
            {"logged_in": False, "username": "ops-admin", "is_admin": True},
            {"logged_in": True, "is_admin": True},
            {"logged_in": True, "username": "", "is_admin": True},
            {"logged_in": 1, "username": "ops-admin", "is_admin": True},
        ):
            with self.subTest(state=state):
                client = self.server.app.test_client()
                with client.session_transaction() as session:
                    session.update(state)
                self.assert_ops_denied(client, 401)

    def test_registered_user_cannot_access_ops(self):
        credentials = {"username": "alice", "password": "user-password"}
        self.assertEqual(self.client.post("/register", json=credentials).status_code, 200)
        self.assertEqual(self.client.post("/login", json=credentials).status_code, 200)
        self.assert_ops_denied(self.client, 403)

    def test_admin_privilege_requires_the_boolean_true(self):
        for privilege in (None, False, "true", "false", 1):
            with self.subTest(privilege=privilege):
                client = self.server.app.test_client()
                with client.session_transaction() as session:
                    session.update(logged_in=True, username="ops-admin", is_admin=privilege)
                self.assert_ops_denied(client, 403)

    def login_admin(self):
        response = self.client.post("/login", json={
            "username": "ops-admin", "password": "test-admin-password",
        })
        self.assertEqual(response.status_code, 200)

    def test_admin_can_read_health(self):
        self.login_admin()
        response = self.client.get("/ops/health")
        self.assertEqual(response.status_code, 200)
        self.assertIn("services", response.get_json())
        self.assertIn("system", response.get_json())

    def test_admin_can_read_metrics(self):
        self.login_admin()
        response = self.client.get("/ops/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("cpu", response.get_json())
        self.assertIn("memory", response.get_json())

    def test_admin_can_read_logs(self):
        self.login_admin()
        response = self.client.get("/ops/logs?source=gunicorn&lines=5")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"gunicorn": ["private-ops-log-entry\n"]})

    def test_admin_can_run_benchmark_using_get_and_post(self):
        self.login_admin()
        for method in ("GET", "POST"):
            with self.subTest(method=method):
                response = self.client.open("/ops/benchmark", method=method)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json(), {"case": "test-case"})

    def test_admin_still_gets_501_when_benchmark_is_unavailable(self):
        self.login_admin()
        with patch.object(self.server, "HAS_BENCHMARK", False):
            self.assertEqual(self.client.get("/ops/benchmark").status_code, 501)

    def test_admin_can_generate_benchmark_files(self):
        self.login_admin()
        response = self.client.post("/ops/benchmark/generate_test_files")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["files"], {"test-case.dxf": "generated"})
        self.assertTrue((self.root / "uploads" / "ops" / "bestest").is_dir())

    def test_logout_revokes_ops_access(self):
        self.login_admin()
        self.client.post("/logout")
        self.assert_ops_denied(self.client, 401)

    def test_minimal_health_check_remains_public(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(set(response.get_json()), {"status", "timestamp"})
        self.assertEqual(response.get_json()["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
