import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class AuthSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="energy-report-ownership-test")

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "users.db"
        self.original_database_path = self.server.DB_PATH
        self.server.DB_PATH = str(self.database_path)
        self.server.init_db()
        self.client = self.server.app.test_client()

    def tearDown(self):
        self.server.DB_PATH = self.original_database_path
        self.temporary_directory.cleanup()

    def test_environment_admin_login_sets_admin_flag(self):
        with patch.dict(os.environ, {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "secret"}):
            response = self.client.post("/login", json={"username": "admin", "password": "secret"})
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as state:
            self.assertIs(state["is_admin"], True)

    def test_database_login_is_not_admin(self):
        connection = self.server.get_db_connection()
        connection.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("alice", self.server.generate_password_hash("pw")),
        )
        connection.commit()
        connection.close()

        response = self.client.post("/login", json={"username": "alice", "password": "pw"})
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as state:
            self.assertIs(state["is_admin"], False)

    def test_logout_clears_all_identity_state(self):
        with self.client.session_transaction() as state:
            state.update(logged_in=True, username="admin", is_admin=True)
        self.client.get("/logout")
        with self.client.session_transaction() as state:
            self.assertNotIn("username", state)
            self.assertNotIn("is_admin", state)


class ReportSchemaMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server

    def test_migration_deduplicates_per_user_and_enforces_composite_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "legacy.db"
            connection = sqlite3.connect(database_path)
            connection.execute(
                """
                CREATE TABLE reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    report_number TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.executemany(
                "INSERT INTO reports (username, report_number, created_at) VALUES (?, ?, ?)",
                [
                    ("alice", "BIM-1", "2026-01-01 00:00:00"),
                    ("alice", "BIM-1", "2026-02-01 00:00:00"),
                    ("alice", "BIM-1", "2026-02-01 00:00:00"),
                    ("bob", "BIM-1", "2026-01-01 00:00:00"),
                ],
            )

            try:
                self.server._migrate_reports_schema(connection)

                alice_rows = connection.execute(
                    "SELECT id FROM reports WHERE username = ? AND report_number = ?", ("alice", "BIM-1")
                ).fetchall()
                bob_rows = connection.execute(
                    "SELECT id FROM reports WHERE username = ? AND report_number = ?", ("bob", "BIM-1")
                ).fetchall()
                column_names = {row[1] for row in connection.execute("PRAGMA table_info(reports)")}

                self.assertEqual([row[0] for row in alice_rows], [3])
                self.assertEqual([row[0] for row in bob_rows], [4])
                self.assertTrue({"status", "updated_at"}.issubset(column_names))
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute(
                        "INSERT INTO reports (username, report_number) VALUES (?, ?)", ("alice", "BIM-1")
                    )
            finally:
                connection.close()


class ReportOwnershipHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "reports.db"
        self.original_database_path = self.server.DB_PATH
        self.server.DB_PATH = str(self.database_path)
        self.server.init_db()

    def tearDown(self):
        self.server.DB_PATH = self.original_database_path
        self.temporary_directory.cleanup()

    def test_composite_helpers_keep_same_report_number_separate_for_each_owner(self):
        connection = self.server.get_db_connection()
        connection.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            ("alice", "hash"),
        )
        connection.commit()
        connection.close()

        self.assertTrue(self.server._user_exists("alice"))
        self.assertFalse(self.server._user_exists("bob"))

        self.server._upsert_report_status("alice", "BIM-1", "processing")
        self.server._upsert_report_status("bob", "BIM-1", "ready")
        self.server._upsert_report_status("alice", "BIM-1", "complete")

        alice = self.server._report_row("alice", "BIM-1")
        bob = self.server._report_row("bob", "BIM-1")
        self.assertEqual(alice["status"], "complete")
        self.assertEqual(bob["status"], "ready")

        connection = self.server.get_db_connection()
        alice_count = connection.execute(
            "SELECT COUNT(*) FROM reports WHERE username = ? AND report_number = ?", ("alice", "BIM-1")
        ).fetchone()[0]
        connection.close()
        self.assertEqual(alice_count, 1)


if __name__ == "__main__":
    unittest.main()
