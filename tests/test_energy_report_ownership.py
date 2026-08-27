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

    def test_migration_replaces_legacy_report_number_uniqueness(self):
        for legacy_constraint in ("inline", "index"):
            with self.subTest(legacy_constraint=legacy_constraint), tempfile.TemporaryDirectory() as directory:
                database_path = Path(directory) / "legacy.db"
                connection = sqlite3.connect(database_path)
                report_number_column = "report_number TEXT NOT NULL UNIQUE" if legacy_constraint == "inline" else "report_number TEXT NOT NULL"
                connection.execute(
                    f"""
                    CREATE TABLE reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        {report_number_column},
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        geometry_used TEXT,
                        params TEXT,
                        results TEXT,
                        UNIQUE(username, created_at)
                    )
                    """
                )
                if legacy_constraint == "index":
                    connection.execute("CREATE UNIQUE INDEX reports_report_number_uq ON reports(report_number)")
                connection.execute(
                    """
                    INSERT INTO reports (username, report_number, created_at, geometry_used, params, results)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("alice", "BIM-1", "2026-01-01 00:00:00", "geometry", "params", "results"),
                )

                try:
                    self.server._migrate_reports_schema(connection)
                    connection.execute(
                        "INSERT INTO reports (username, report_number) VALUES (?, ?)", ("bob", "BIM-1")
                    )
                    owners = connection.execute(
                        "SELECT username FROM reports WHERE report_number = ? ORDER BY username", ("BIM-1",)
                    ).fetchall()
                    self.assertEqual([owner[0] for owner in owners], ["alice", "bob"])
                    alice_data = connection.execute(
                        "SELECT geometry_used, params, results FROM reports WHERE username = ? AND report_number = ?",
                        ("alice", "BIM-1"),
                    ).fetchone()
                    self.assertEqual(alice_data, ("geometry", "params", "results"))
                    with self.assertRaises(sqlite3.IntegrityError):
                        connection.execute(
                            "INSERT INTO reports (username, report_number, created_at) VALUES (?, ?, ?)",
                            ("alice", "BIM-2", "2026-01-01 00:00:00"),
                        )
                finally:
                    connection.close()

    def test_migration_rejects_unsupported_inline_unique_schema_without_changes(self):
        variants = [
            ("check", "username TEXT NOT NULL", [], ["CHECK (length(report_number) <= 32)"]),
            ("foreign_key", "username TEXT NOT NULL", [], ["FOREIGN KEY(username) REFERENCES users(username)"]),
            ("collation", "username TEXT NOT NULL COLLATE NOCASE", [], []),
            (
                "generated_column",
                "username TEXT NOT NULL",
                ["report_number_length INTEGER GENERATED ALWAYS AS (length(report_number)) STORED"],
                [],
            ),
        ]
        for name, username_definition, extra_columns, table_constraints in variants:
            with self.subTest(feature=name), tempfile.TemporaryDirectory() as directory:
                database_path = Path(directory) / "legacy.db"
                connection = sqlite3.connect(database_path)
                definitions = [
                    "id INTEGER PRIMARY KEY AUTOINCREMENT",
                    username_definition,
                    "report_number TEXT NOT NULL UNIQUE",
                    "created_at TEXT DEFAULT CURRENT_TIMESTAMP",
                    *extra_columns,
                    *table_constraints,
                ]
                connection.execute(f"CREATE TABLE reports ({', '.join(definitions)})")
                connection.execute(
                    "INSERT INTO reports (username, report_number, created_at) VALUES (?, ?, ?)",
                    ("alice", "BIM-1", "2026-01-01 00:00:00"),
                )
                schema_before = connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'reports'"
                ).fetchone()[0]
                rows_before = connection.execute(
                    "SELECT id, username, report_number, created_at FROM reports"
                ).fetchall()

                try:
                    with self.assertRaisesRegex(RuntimeError, "unsupported reports schema"):
                        self.server._migrate_reports_schema(connection)
                    schema_after = connection.execute(
                        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'reports'"
                    ).fetchone()[0]
                    rows_after = connection.execute(
                        "SELECT id, username, report_number, created_at FROM reports"
                    ).fetchall()
                    self.assertEqual(schema_after, schema_before)
                    self.assertEqual(rows_after, rows_before)
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
