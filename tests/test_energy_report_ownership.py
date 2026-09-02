import io
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from energy_report_storage import user_storage_key


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

    def test_registration_rejects_noncanonical_usernames(self):
        for username in (" alice", "alice ", "ａｌｉｃｅ"):
            with self.subTest(username=username):
                response = self.client.post(
                    "/register",
                    json={"username": username, "password": "pw"},
                )
                self.assertEqual(response.status_code, 400, response.get_json())

        connection = self.server.get_db_connection()
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0], 0)
        finally:
            connection.close()

    def test_registration_rejects_a_storage_key_collision_with_a_legacy_user(self):
        connection = self.server.get_db_connection()
        connection.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (" alice ", self.server.generate_password_hash("legacy-pw")),
        )
        connection.commit()
        connection.close()

        response = self.client.post(
            "/register",
            json={"username": "alice", "password": "pw"},
        )

        self.assertEqual(response.status_code, 400, response.get_json())
        connection = self.server.get_db_connection()
        try:
            usernames = [row[0] for row in connection.execute("SELECT username FROM users")]
        finally:
            connection.close()
        self.assertEqual(usernames, [" alice "])

    def test_registration_rejects_a_storage_key_collision_with_environment_admin(self):
        with patch.dict(
            os.environ,
            {"ADMIN_USER": " alice ", "ADMIN_PASSWORD": "secret"},
            clear=False,
        ):
            response = self.client.post(
                "/register",
                json={"username": "alice", "password": "pw"},
            )

        self.assertEqual(response.status_code, 400, response.get_json())
        connection = self.server.get_db_connection()
        try:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM users").fetchone()[0], 0)
        finally:
            connection.close()


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

    def test_migration_accepts_nonsemantic_keywords_in_legacy_table_sql(self):
        variants = [
            ("string_literal", "note TEXT DEFAULT 'references'"),
            ("quoted_identifier", '"references" TEXT'),
            ("comment", "note TEXT /* references */"),
        ]
        for name, extra_column in variants:
            with self.subTest(location=name), tempfile.TemporaryDirectory() as directory:
                database_path = Path(directory) / "legacy.db"
                connection = sqlite3.connect(database_path)
                connection.execute(
                    f"""
                    CREATE TABLE reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT NOT NULL,
                        report_number TEXT NOT NULL UNIQUE,
                        {extra_column},
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                connection.execute(
                    "INSERT INTO reports (username, report_number) VALUES (?, ?)",
                    ("alice", "BIM-1"),
                )

                try:
                    self.server._migrate_reports_schema(connection)
                    connection.execute(
                        "INSERT INTO reports (username, report_number) VALUES (?, ?)",
                        ("bob", "BIM-1"),
                    )
                    owners = connection.execute(
                        "SELECT username FROM reports WHERE report_number = ? ORDER BY username", ("BIM-1",)
                    ).fetchall()
                    self.assertEqual(owners, [("alice",), ("bob",)])
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

    def test_update_only_status_helper_never_inserts_a_missing_report(self):
        updated = self.server._update_report_status_if_exists("alice", "MISSING", "failed")

        self.assertFalse(updated)
        self.assertIsNone(self.server._report_row("alice", "MISSING"))

        self.server._upsert_report_status("alice", "EXISTS", "created")
        updated = self.server._update_report_status_if_exists("alice", "EXISTS", "failed")

        self.assertTrue(updated)
        self.assertEqual(self.server._report_row("alice", "EXISTS")["status"], "failed")


class PdfOwnerBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="pdf-owner-binding-test")

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary_directory.name)
        self.database_path = temporary_root / "users.db"
        self.upload_root = temporary_root / "uploads"
        self.original_database_path = self.server.DB_PATH
        self.original_upload_folder = self.server.app.config["UPLOAD_FOLDER"]
        self.server.DB_PATH = str(self.database_path)
        self.server.app.config["UPLOAD_FOLDER"] = str(self.upload_root)
        self.server.init_db()
        connection = self.server.get_db_connection()
        connection.executemany(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            [("alice", "hash"), ("bob", "hash")],
        )
        connection.commit()
        connection.close()
        self.client = self.server.app.test_client()

    def tearDown(self):
        self.server.DB_PATH = self.original_database_path
        self.server.app.config["UPLOAD_FOLDER"] = self.original_upload_folder
        self.temporary_directory.cleanup()

    def login_as(self, username):
        with self.client.session_transaction() as state:
            state.clear()
            state.update(logged_in=True, username=username, is_admin=False)

    def symlink_or_skip(self, link, target, *, target_is_directory=False):
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("creating symlinks requires Windows developer mode or privilege")
            raise

    @staticmethod
    def pdf_form(report_number):
        from reportlab.pdfgen import canvas

        pdf = io.BytesIO()
        document = canvas.Canvas(pdf)
        document.drawString(72, 720, "owner-bound prepared PDF")
        document.save()
        pdf.seek(0)
        return {
            "report_number": report_number,
            "raster_file": (pdf, "floor.pdf"),
        }

    def test_pdf_token_cannot_be_reused_by_another_user(self):
        import numpy as np

        self.login_as("alice")
        prepared_response = self.client.post(
            "/energy/pdf_prepare",
            data=self.pdf_form("BIM-1"),
            content_type="multipart/form-data",
        )
        self.assertEqual(prepared_response.status_code, 200, prepared_response.get_json())
        prepared = prepared_response.get_json()

        self.login_as("bob")
        with patch.object(
            self.server,
            "render_pdf_page_preview",
            return_value=np.zeros((2, 2, 3), dtype=np.uint8),
        ):
            response = self.client.post(
                "/energy/pdf_page_preview",
                data={
                    "report_number": "BIM-1",
                    "pdf_upload_token": prepared["upload_token"],
                    "pdf_page_number": "1",
                },
            )

        self.assertIn(response.status_code, {400, 403})

    def test_pdf_prepare_records_the_uploaded_report_for_its_owner(self):
        self.login_as("alice")

        response = self.client.post(
            "/energy/pdf_prepare",
            data=self.pdf_form("BIM-PDF-STATUS"),
            content_type="multipart/form-data",
        )

        self.assertEqual(response.status_code, 200, response.get_json())
        alice_report = self.server._report_row("alice", "BIM-PDF-STATUS")
        self.assertIsNotNone(alice_report)
        self.assertEqual(alice_report["status"], "uploaded")
        self.assertIsNone(self.server._report_row("bob", "BIM-PDF-STATUS"))

    def test_report_child_file_validator_rejects_a_regular_file_outside_the_report(self):
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / "BIM-CHILD"
        report_dir.mkdir(parents=True)
        outside = self.upload_root / "outside.pdf"
        outside.write_bytes(b"outside")
        validator = getattr(
            self.server,
            "_validated_report_child_file",
            lambda *_args, **_kwargs: None,
        )

        with self.assertRaises(self.server.InvalidReportPath):
            validator(report_dir, outside)

    def test_report_child_directory_validator_rejects_a_directory_outside_the_report(self):
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / "BIM-CHILD"
        report_dir.mkdir(parents=True)
        outside = self.upload_root / "outside-artifacts"
        outside.mkdir()
        validator = getattr(
            self.server,
            "_validated_report_child_directory",
            lambda *_args, **_kwargs: None,
        )

        with self.assertRaises(self.server.InvalidReportPath):
            validator(report_dir, outside)

    def test_prepared_pdf_symlink_cannot_escape_the_owner_report(self):
        import numpy as np

        self.login_as("alice")
        prepared_response = self.client.post(
            "/energy/pdf_prepare",
            data=self.pdf_form("BIM-PDF-SYMLINK"),
            content_type="multipart/form-data",
        )
        self.assertEqual(prepared_response.status_code, 200, prepared_response.get_json())
        prepared = prepared_response.get_json()
        report_dir = (
            self.upload_root / "energy" / user_storage_key("alice") / "BIM-PDF-SYMLINK"
        )
        stored_pdf = next(report_dir.glob("building_plan_prepared_*.pdf"))
        outside_pdf = self.upload_root / "outside.pdf"
        outside_pdf.write_bytes(stored_pdf.read_bytes())
        stored_pdf.unlink()
        self.symlink_or_skip(stored_pdf, outside_pdf)

        with patch.object(
            self.server,
            "render_pdf_page_preview",
            return_value=np.zeros((2, 2, 3), dtype=np.uint8),
        ) as render:
            response = self.client.post(
                "/energy/pdf_page_preview",
                data={
                    "report_number": "BIM-PDF-SYMLINK",
                    "pdf_upload_token": prepared["upload_token"],
                    "pdf_page_number": "1",
                },
            )

        self.assertEqual(response.status_code, 400, response.get_json())
        self.assertEqual(response.get_json()["error"], "Invalid report path")
        self.assertNotIn(str(outside_pdf), response.get_data(as_text=True))
        render.assert_not_called()

    def test_scale_calibration_rejects_a_symlinked_recognition_artifact(self):
        self.login_as("alice")
        report_dir = (
            self.upload_root / "energy" / user_storage_key("alice")
            / "BIM-RECOGNITION-SYMLINK"
        )
        report_dir.mkdir(parents=True)
        outside_recognition = self.upload_root / "outside-recognition.json"
        original_payload = {
            "schema_version": 1,
            "model": {"name": "test", "version": "test-v1"},
            "preprocessing": {"requested": "auto", "use_preprocessing": True},
            "image_size": [100, 100],
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {
                "status": "closed_rooms",
                "room_count": 1,
                "rooms": [{"area_px2": 100.0}],
                "total_area_px2": 100.0,
                "total_area_m2": None,
                "load_geometry_ready": False,
            },
        }
        outside_recognition.write_text(json.dumps(original_payload), encoding="utf-8")
        original_bytes = outside_recognition.read_bytes()
        self.symlink_or_skip(report_dir / "recognition.json", outside_recognition)

        response = self.client.post(
            "/energy/scale_calibration",
            json={
                "report_number": "BIM-RECOGNITION-SYMLINK",
                "point_a": [10, 10],
                "point_b": [20, 10],
                "actual_length": 1,
                "unit": "m",
            },
        )

        self.assertEqual(response.status_code, 400, response.get_json())
        self.assertEqual(response.get_json()["error"], "Invalid report path")
        self.assertNotIn(str(outside_recognition), response.get_data(as_text=True))
        self.assertEqual(outside_recognition.read_bytes(), original_bytes)

    def test_pdf_prepare_parse_error_does_not_expose_the_owner_storage_path(self):
        self.server._upsert_report_status("bob", "BIM-SECRET", "created")
        self.login_as("alice")
        leaked_path = self.upload_root / "energy" / user_storage_key("alice") / "BIM-SECRET"

        with patch("pdfplumber.open", side_effect=RuntimeError(str(leaked_path))):
            response = self.client.post(
                "/energy/pdf_prepare",
                data=self.pdf_form("BIM-SECRET"),
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 400, response.get_json())
        self.assertEqual(response.get_json()["error"], "Cannot read PDF")
        self.assertNotIn(str(leaked_path), response.get_data(as_text=True))
        self.assertEqual(self.server._report_row("alice", "BIM-SECRET")["status"], "failed")
        self.assertEqual(self.server._report_row("bob", "BIM-SECRET")["status"], "created")
        self.assertEqual(list(leaked_path.glob("building_plan_prepared_*.pdf")), [])

    def test_pdf_prepare_save_failure_marks_only_owner_failed_and_removes_partial_file(self):
        from werkzeug.datastructures import FileStorage

        report_number = "BIM-SAVE-FAIL"
        self.server._upsert_report_status("bob", report_number, "created")
        self.login_as("alice")
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
        secret = str(report_dir / "private-partial.pdf")

        def fail_after_partial_write(_storage, destination, *_args, **_kwargs):
            Path(destination).write_bytes(b"partial prepared PDF")
            raise RuntimeError(secret)

        with (
            patch.dict(self.server.app.config, {"PROPAGATE_EXCEPTIONS": False}),
            patch.object(FileStorage, "save", autospec=True, side_effect=fail_after_partial_write),
            self.assertLogs(self.server.logger, level="ERROR"),
        ):
            response = self.client.post(
                "/energy/pdf_prepare",
                data=self.pdf_form(report_number),
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {"error": "Internal server error"})
        self.assertNotIn(secret, response.get_data(as_text=True))
        self.assertEqual(self.server._report_row("alice", report_number)["status"], "failed")
        self.assertEqual(self.server._report_row("bob", report_number)["status"], "created")
        self.assertEqual(list(report_dir.glob("building_plan_prepared_*.pdf")), [])

    def test_pdf_preview_internal_error_does_not_expose_the_prepared_file_path(self):
        self.login_as("alice")
        prepared = self.client.post(
            "/energy/pdf_prepare",
            data=self.pdf_form("BIM-PREVIEW-SECRET"),
            content_type="multipart/form-data",
        ).get_json()
        leaked_path = (
            self.upload_root / "energy" / user_storage_key("alice")
            / "BIM-PREVIEW-SECRET" / "prepared.pdf"
        )

        with patch.object(
            self.server,
            "render_pdf_page_preview",
            side_effect=RuntimeError(str(leaked_path)),
        ):
            response = self.client.post(
                "/energy/pdf_page_preview",
                data={
                    "report_number": "BIM-PREVIEW-SECRET",
                    "pdf_upload_token": prepared["upload_token"],
                    "pdf_page_number": "1",
                },
            )

        self.assertEqual(response.status_code, 500, response.get_json())
        self.assertEqual(response.get_json()["error"], "Internal server error")
        self.assertNotIn(str(leaked_path), response.get_data(as_text=True))

    def test_ai_recognition_pdf_conversion_error_does_not_expose_the_owner_storage_path(self):
        self.login_as("alice")
        prepared = self.client.post(
            "/energy/pdf_prepare",
            data=self.pdf_form("BIM-CONVERSION-SECRET"),
            content_type="multipart/form-data",
        ).get_json()
        leaked_path = (
            self.upload_root / "energy" / user_storage_key("alice")
            / "BIM-CONVERSION-SECRET" / "building_plan_ai.png"
        )

        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(
                self.server,
                "_floorplan_segmenter",
                MagicMock(),
                create=True,
            ),
            patch.object(
                self.server,
                "prepare_pdf_page",
                side_effect=RuntimeError(str(leaked_path)),
            ),
        ):
            response = self.client.post(
                "/energy/ai_recognize",
                data={
                    "report_number": "BIM-CONVERSION-SECRET",
                    "pdf_upload_token": prepared["upload_token"],
                    "pdf_page_number": "1",
                },
            )

        self.assertEqual(response.status_code, 500, response.get_json())
        self.assertEqual(response.get_json()["error"], "Internal server error")
        self.assertNotIn(str(leaked_path), response.get_data(as_text=True))

    def test_ai_recognition_status_updates_only_the_authenticated_report_owner(self):
        import numpy as np

        self.server._upsert_report_status("alice", "BIM-STATUS", "created")
        self.server._upsert_report_status("bob", "BIM-STATUS", "created")
        self.login_as("alice")

        def prediction(*args, **kwargs):
            self.assertEqual(
                self.server._report_row("alice", "BIM-STATUS")["status"],
                "recognizing",
            )
            self.assertEqual(
                self.server._report_row("bob", "BIM-STATUS")["status"],
                "created",
            )
            return {
                "mask": np.zeros((2, 2), dtype=np.uint8),
                "overlay": np.zeros((2, 2, 3), dtype=np.uint8),
                "stats": {},
                "geometry": {"walls": [], "windows": [], "doors": []},
                "room_topology": {
                    "status": "no_closed_rooms",
                    "room_count": 0,
                    "rooms": [],
                    "total_area_px2": 0.0,
                    "total_area_m2": None,
                    "load_geometry_ready": False,
                },
                "image_size": [2, 2],
            }

        segmenter = MagicMock()
        segmenter.predict.side_effect = prediction
        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(self.server, "_floorplan_segmenter", segmenter, create=True),
            patch.object(
                self.server.cv2,
                "imread",
                return_value=np.zeros((2, 2, 3), dtype=np.uint8),
            ),
        ):
            response = self.client.post(
                "/energy/ai_recognize",
                data={
                    "report_number": "BIM-STATUS",
                    "raster_file": (io.BytesIO(b"image"), "plan.png"),
                },
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(
            self.server._report_row("alice", "BIM-STATUS")["status"],
            "recognized",
        )
        self.assertEqual(
            self.server._report_row("bob", "BIM-STATUS")["status"],
            "created",
        )

    def test_ai_recognition_failure_updates_only_the_authenticated_report_owner(self):
        self.server._upsert_report_status("alice", "BIM-FAIL-STATUS", "created")
        self.server._upsert_report_status("bob", "BIM-FAIL-STATUS", "created")
        self.login_as("alice")

        response = self.client.post(
            "/energy/ai_recognize",
            data={
                "report_number": "BIM-FAIL-STATUS",
                "model_backend": "unsupported",
            },
        )

        self.assertEqual(response.status_code, 400, response.get_json())
        self.assertEqual(
            self.server._report_row("alice", "BIM-FAIL-STATUS")["status"],
            "failed",
        )
        self.assertEqual(
            self.server._report_row("bob", "BIM-FAIL-STATUS")["status"],
            "created",
        )

    def test_ai_recognition_records_a_new_owner_report_before_processing(self):
        self.login_as("alice")

        response = self.client.post(
            "/energy/ai_recognize",
            data={
                "report_number": "BIM-NEW-RECOGNITION",
                "model_backend": "unsupported",
            },
        )

        self.assertEqual(response.status_code, 400, response.get_json())
        alice_report = self.server._report_row("alice", "BIM-NEW-RECOGNITION")
        self.assertIsNotNone(alice_report)
        self.assertEqual(alice_report["status"], "failed")
        self.assertIsNone(self.server._report_row("bob", "BIM-NEW-RECOGNITION"))


class ReportOperationAuthorizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="report-operation-authorization-test")

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary_directory.name)
        self.database_path = temporary_root / "users.db"
        self.upload_root = temporary_root / "uploads"
        self.original_database_path = self.server.DB_PATH
        self.original_upload_folder = self.server.app.config["UPLOAD_FOLDER"]
        self.server.DB_PATH = str(self.database_path)
        self.server.app.config["UPLOAD_FOLDER"] = str(self.upload_root)
        self.server.init_db()

        connection = self.server.get_db_connection()
        connection.executemany(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            [("alice", "hash"), ("bob", "hash")],
        )
        connection.executemany(
            """
            INSERT INTO reports (
                username, report_number, status, created_at, updated_at,
                geometry_used, params, results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "alice", "BIM-1", "recognized",
                    "2026-01-01 00:00:00", "2026-01-02 00:00:00",
                    json.dumps({"floor_area_m2": 100}),
                    json.dumps({"building_type": "alice-office"}),
                    json.dumps({"summary": {"rating": "A"}}),
                ),
                (
                    "bob", "BIM-1", "bob-ready",
                    "2026-01-01 00:00:00", "2026-01-03 00:00:00",
                    json.dumps({"floor_area_m2": 200}),
                    json.dumps({"building_type": "bob-office"}),
                    json.dumps({"summary": {"rating": "B"}}),
                ),
            ],
        )
        connection.commit()
        connection.close()

        self.alice_report_dir = (
            self.upload_root / "energy" / user_storage_key("alice") / "BIM-1"
        )
        self.alice_report_dir.mkdir(parents=True)
        self.alice_recognition_path = self.alice_report_dir / "recognition.json"
        self.alice_recognition_path.write_text(
            json.dumps({
                "schema_version": 1,
                "owner_username": "alice",
                "model": {"name": "test", "version": "test-v1"},
                "preprocessing": {"requested": "auto", "use_preprocessing": True},
                "image_size": [10, 10],
                "geometry": {"walls": [], "windows": [], "doors": []},
                "room_topology": {
                    "status": "closed_rooms",
                    "room_count": 1,
                    "rooms": [{"area_px2": 100.0}],
                    "total_area_px2": 100.0,
                    "total_area_m2": None,
                    "load_geometry_ready": False,
                },
            }),
            encoding="utf-8",
        )
        self.client = self.server.app.test_client()

    def tearDown(self):
        self.server.DB_PATH = self.original_database_path
        self.server.app.config["UPLOAD_FOLDER"] = self.original_upload_folder
        self.temporary_directory.cleanup()

    def login_as(self, username, is_admin=False):
        with self.client.session_transaction() as state:
            state.clear()
            state.update(logged_in=True, username=username, is_admin=is_admin)

    def test_ordinary_user_cannot_read_or_alter_another_owners_report(self):
        self.login_as("bob")
        original_recognition = self.alice_recognition_path.read_bytes()
        original_alice = dict(self.server._report_row("alice", "BIM-1"))

        requests = [
            (
                "post",
                "/energy/ai_recognize",
                {"data": {"report_number": "BIM-1", "owner_username": "alice"}},
            ),
            (
                "post",
                "/energy/scale_calibration",
                {"json": {
                    "report_number": "BIM-1",
                    "owner_username": "alice",
                    "point_a": [0, 0],
                    "point_b": [5, 0],
                    "actual_length": 1,
                    "unit": "m",
                }},
            ),
            (
                "post",
                "/energy/ai_simulate",
                {"json": {"report_number": "BIM-1", "owner_username": "alice"}},
            ),
            (
                "get",
                "/energy/report/BIM-1?owner_username=alice",
                {},
            ),
        ]

        with patch.object(self.server, "HAS_FLOORPLAN_AI", True):
            for method, url, kwargs in requests:
                with self.subTest(url=url):
                    response = getattr(self.client, method)(url, **kwargs)
                    self.assertEqual(response.status_code, 403, response.get_json())

        self.assertEqual(self.alice_recognition_path.read_bytes(), original_recognition)
        self.assertEqual(dict(self.server._report_row("alice", "BIM-1")), original_alice)

    def test_environment_admin_can_explicitly_access_alices_existing_report(self):
        with patch.dict(
            os.environ,
            {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "secret"},
            clear=False,
        ):
            login = self.client.post(
                "/login",
                json={"username": "admin", "password": "secret"},
            )
        self.assertEqual(login.status_code, 200, login.get_json())

        calibration = self.client.post(
            "/energy/scale_calibration",
            json={
                "report_number": "BIM-1",
                "owner_username": "alice",
                "point_a": [0, 0],
                "point_b": [5, 0],
                "actual_length": 1,
                "unit": "m",
            },
        )
        detail = self.client.get("/energy/report/BIM-1?owner_username=alice")

        self.assertEqual(calibration.status_code, 200, calibration.get_json())
        self.assertEqual(detail.status_code, 200, detail.get_json())
        self.assertEqual(detail.get_json()["username"], "alice")
        self.assertEqual(detail.get_json()["status"], "recognized")
        self.assertEqual(detail.get_json()["updated_at"], "2026-01-02 00:00:00")
        saved = json.loads(self.alice_recognition_path.read_text(encoding="utf-8"))
        self.assertEqual(saved["scale_calibration"]["status"], "confirmed")

    def test_same_number_detail_reads_the_authenticated_owners_row(self):
        self.login_as("bob")

        response = self.client.get("/energy/report/BIM-1")

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json(), {
            "report_number": "BIM-1",
            "username": "bob",
            "status": "bob-ready",
            "created_at": "2026-01-01 00:00:00",
            "updated_at": "2026-01-03 00:00:00",
            "geometry_used": {"floor_area_m2": 200},
            "params": {"building_type": "bob-office"},
            "results": {"summary": {"rating": "B"}},
        })

    def test_ai_simulate_upserts_only_the_authenticated_composite_row(self):
        self.login_as("alice")
        calculation = {
            "success": True,
            "summary": {
                "total_energy_kwh": 123,
                "eui": 12.3,
                "rating": "A",
            },
        }

        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(self.server, "HAS_ENERGY_CALC", True),
            patch.object(self.server, "HAS_DESIGN_LOAD_CALC", False),
            patch.object(
                self.server.energy_calc,
                "calculate_energy",
                return_value=calculation,
            ),
        ):
            response = self.client.post(
                "/energy/ai_simulate",
                json={"report_number": "BIM-1", "scale": 0.1},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        alice = self.server._report_row("alice", "BIM-1")
        bob = self.server._report_row("bob", "BIM-1")
        self.assertEqual(alice["status"], "calculated")
        self.assertEqual(json.loads(alice["results"])["summary"]["total_energy_kwh"], 123)
        self.assertAlmostEqual(
            json.loads(alice["params"])["geometry"]["floor_area_m2"],
            1.0,
        )
        self.assertEqual(bob["status"], "bob-ready")
        self.assertEqual(json.loads(bob["results"]), {"summary": {"rating": "B"}})

    def test_report_lists_are_owner_scoped_for_users_and_global_for_admins(self):
        connection = self.server.get_db_connection()
        connection.execute(
            """
            INSERT INTO reports (
                username, report_number, status, created_at, updated_at,
                geometry_used, params, results
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "alice", "BIM-OLDER", "calculated",
                "2025-12-01 00:00:00", "2026-01-01 00:00:00",
                json.dumps({"floor_area_m2": 50}),
                json.dumps({"private_path": str(self.upload_root / "secret")}),
                json.dumps({"summary": {
                    "total_energy_kwh": 500,
                    "eui": 10,
                    "rating": "A",
                }}),
            ),
        )
        connection.commit()
        connection.close()

        self.login_as("bob")
        ordinary = self.client.get("/energy/reports")
        self.assertEqual(ordinary.status_code, 200, ordinary.get_json())
        self.assertEqual(len(ordinary.get_json()), 1)
        self.assertEqual(ordinary.get_json()[0]["username"], "bob")
        self.assertEqual(ordinary.get_json()[0]["report_number"], "BIM-1")
        self.assertEqual(ordinary.get_json()[0]["status"], "bob-ready")
        self.assertEqual(ordinary.get_json()[0]["updated_at"], "2026-01-03 00:00:00")

        with patch.dict(
            os.environ,
            {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "secret"},
            clear=False,
        ):
            login = self.client.post(
                "/login",
                json={"username": "admin", "password": "secret"},
            )
        self.assertEqual(login.status_code, 200, login.get_json())
        administrator = self.client.get("/energy/reports")

        self.assertEqual(administrator.status_code, 200, administrator.get_json())
        self.assertEqual(
            [(item["username"], item["report_number"]) for item in administrator.get_json()],
            [("bob", "BIM-1"), ("alice", "BIM-1"), ("alice", "BIM-OLDER")],
        )
        self.assertEqual(
            set(administrator.get_json()[0]),
            {
                "username", "report_number", "status", "created_at", "updated_at",
                "floor_area", "total_energy", "eui", "rating",
            },
        )
        serialized = administrator.get_data(as_text=True)
        self.assertNotIn(str(self.upload_root), serialized)
        self.assertNotIn(user_storage_key("alice"), serialized)

    def test_report_lists_default_valid_non_object_json_for_users_and_admins(self):
        connection = self.server.get_db_connection()
        connection.executemany(
            """
            INSERT INTO reports (
                username, report_number, status, created_at, updated_at,
                geometry_used, results
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "bob", "NULL-PAYLOAD", "recognized",
                    "2026-02-01 00:00:00", "2026-02-01 00:00:00",
                    json.dumps(None), json.dumps(None),
                ),
                (
                    "bob", "NULL-SUMMARY", "calculated",
                    "2026-02-02 00:00:00", "2026-02-02 00:00:00",
                    json.dumps({}), json.dumps({"summary": None}),
                ),
                (
                    "bob", "LIST-SUMMARY", "calculated",
                    "2026-02-03 00:00:00", "2026-02-03 00:00:00",
                    json.dumps({}), json.dumps({"summary": []}),
                ),
            ],
        )
        connection.commit()
        connection.close()

        self.login_as("bob")
        ordinary = self.client.get("/energy/reports")
        self.assertEqual(ordinary.status_code, 200, ordinary.get_json())

        with patch.dict(
            os.environ,
            {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "secret"},
            clear=False,
        ):
            login = self.client.post(
                "/login",
                json={"username": "admin", "password": "secret"},
            )
        self.assertEqual(login.status_code, 200, login.get_json())
        administrator = self.client.get("/energy/reports")
        self.assertEqual(administrator.status_code, 200, administrator.get_json())

        for response in (ordinary, administrator):
            reports = {
                item["report_number"]: item
                for item in response.get_json()
            }
            for report_number, status in (
                ("NULL-PAYLOAD", "recognized"),
                ("NULL-SUMMARY", "calculated"),
                ("LIST-SUMMARY", "calculated"),
            ):
                with self.subTest(
                    role="admin" if response is administrator else "ordinary",
                    report_number=report_number,
                ):
                    item = reports[report_number]
                    self.assertEqual(item["username"], "bob")
                    self.assertEqual(item["report_number"], report_number)
                    self.assertEqual(item["status"], status)
                    self.assertEqual(item["floor_area"], 0)
                    self.assertEqual(item["total_energy"], 0)
                    self.assertEqual(item["eui"], 0)
                    self.assertEqual(item["rating"], "-")

    def test_report_read_internal_errors_are_sanitized(self):
        self.login_as("alice")
        secret = str(self.upload_root / "private-database.db")

        for url in ("/energy/report/BIM-1", "/energy/reports"):
            with self.subTest(url=url), patch.object(
                self.server,
                "get_db_connection",
                side_effect=RuntimeError(secret),
            ), self.assertLogs(self.server.logger, level="ERROR"):
                response = self.client.get(url)
                self.assertEqual(response.status_code, 500, response.get_json())
                self.assertEqual(response.get_json(), {"error": "Internal server error"})
                self.assertNotIn(secret, response.get_data(as_text=True))

    def test_ai_simulate_internal_errors_are_sanitized(self):
        self.login_as("alice")
        secret = str(self.alice_report_dir / "private-calculation-input.json")

        with (
            patch.object(self.server, "HAS_FLOORPLAN_AI", True),
            patch.object(self.server, "HAS_ENERGY_CALC", True),
            patch.object(self.server, "HAS_DESIGN_LOAD_CALC", False),
            patch.object(
                self.server.energy_calc,
                "calculate_energy",
                side_effect=RuntimeError(secret),
            ),
            self.assertLogs(self.server.logger, level="ERROR"),
        ):
            response = self.client.post(
                "/energy/ai_simulate",
                json={"report_number": "BIM-1", "scale": 0.1},
            )

        self.assertEqual(response.status_code, 500, response.get_json())
        self.assertEqual(response.get_json(), {"error": "Internal server error"})
        self.assertNotIn(secret, response.get_data(as_text=True))


class ReportUploadIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="energy-report-upload-isolation-test")

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        temporary_root = Path(self.temporary_directory.name)
        self.database_path = temporary_root / "users.db"
        self.upload_root = temporary_root / "uploads"
        self.original_database_path = self.server.DB_PATH
        self.original_upload_folder = self.server.app.config["UPLOAD_FOLDER"]
        self.server.DB_PATH = str(self.database_path)
        self.server.app.config["UPLOAD_FOLDER"] = str(self.upload_root)
        self.server.init_db()
        connection = self.server.get_db_connection()
        connection.executemany(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            [("alice", "hash"), ("bob", "hash")],
        )
        connection.commit()
        connection.close()
        self.client = self.server.app.test_client()

    def tearDown(self):
        self.server.DB_PATH = self.original_database_path
        self.server.app.config["UPLOAD_FOLDER"] = self.original_upload_folder
        self.temporary_directory.cleanup()

    def login_as(self, username, is_admin=False):
        with self.client.session_transaction() as state:
            state.clear()
            state.update(logged_in=True, username=username, is_admin=is_admin)

    @staticmethod
    def image_file():
        return io.BytesIO(b"test image"), "plan.png"

    def symlink_or_skip(self, link, target, *, target_is_directory=False):
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("creating symlinks requires Windows developer mode or privilege")
            raise

    def test_same_report_number_isolated_by_authenticated_user(self):
        self.login_as("alice")
        alice_response = self.client.post(
            "/energy/upload",
            data={"report_number": "BIM-1", "raster_file": self.image_file()},
        )
        self.login_as("bob")
        bob_response = self.client.post(
            "/energy/upload",
            data={"report_number": "BIM-1", "raster_file": self.image_file()},
        )

        self.assertEqual(alice_response.status_code, 200, alice_response.get_json())
        self.assertEqual(bob_response.status_code, 200, bob_response.get_json())
        self.assertTrue((self.upload_root / "energy" / user_storage_key("alice") / "BIM-1").is_dir())
        self.assertTrue((self.upload_root / "energy" / user_storage_key("bob") / "BIM-1").is_dir())
        self.assertFalse((self.upload_root / "energy" / "BIM-1").exists())
        self.assertEqual(self.server._report_row("alice", "BIM-1")["status"], "uploaded")
        self.assertEqual(self.server._report_row("bob", "BIM-1")["status"], "uploaded")

    def test_legacy_equivalent_login_identities_fail_closed_for_the_same_report(self):
        password_rows = [
            ("alice", self.server.generate_password_hash("alice-pw")),
            (" alice ", self.server.generate_password_hash("space-pw")),
            ("ａｌｉｃｅ", self.server.generate_password_hash("nfkc-pw")),
        ]
        connection = self.server.get_db_connection()
        connection.execute("DELETE FROM users")
        connection.executemany(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            password_rows,
        )
        connection.commit()
        connection.close()

        for username, password in (
            ("alice", "alice-pw"),
            (" alice ", "space-pw"),
            ("ａｌｉｃｅ", "nfkc-pw"),
        ):
            with self.subTest(username=username):
                login = self.client.post(
                    "/login",
                    json={"username": username, "password": password},
                )
                self.assertEqual(login.status_code, 200, login.get_json())
                response = self.client.post(
                    "/energy/upload",
                    data={
                        "report_number": "BIM-COLLISION",
                        "raster_file": self.image_file(),
                    },
                )
                self.assertEqual(response.status_code, 403, response.get_json())

        self.assertFalse((self.upload_root / "energy").exists())

    def test_environment_admin_collision_fails_closed_at_runtime(self):
        connection = self.server.get_db_connection()
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (self.server.generate_password_hash("alice-pw"), "alice"),
        )
        connection.commit()
        connection.close()

        with patch.dict(
            os.environ,
            {"ADMIN_USER": " alice ", "ADMIN_PASSWORD": "admin-pw"},
            clear=False,
        ):
            login = self.client.post(
                "/login",
                json={"username": " alice ", "password": "admin-pw"},
            )
            self.assertEqual(login.status_code, 200, login.get_json())
            response = self.client.post(
                "/energy/upload",
                data={
                    "report_number": "BIM-ADMIN-COLLISION",
                    "raster_file": self.image_file(),
                },
            )

        self.assertEqual(response.status_code, 403, response.get_json())
        self.assertFalse((self.upload_root / "energy").exists())

    def test_dxf_raster_and_weather_uploads_replace_fixed_file_symlinks(self):
        self.login_as("alice")
        report_number = "BIM-FIXED-UPLOADS"
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
        report_dir.mkdir(parents=True)
        outside_files = {}
        for child_name in ("building_plan.dxf", "building_plan.png", "weather_data.epw"):
            outside = self.upload_root / f"outside-{child_name}"
            outside.write_bytes(b"outside")
            self.symlink_or_skip(report_dir / child_name, outside)
            outside_files[child_name] = outside

        fake_document = MagicMock()
        fake_document.modelspace.return_value = []
        with patch.object(self.server, "ezdxf", create=True) as ezdxf:
            ezdxf.readfile.return_value = fake_document
            response = self.client.post(
                "/energy/upload",
                data={
                    "report_number": report_number,
                    "dxf_file": (io.BytesIO(b"new dxf"), "plan.dxf"),
                    "raster_file": (io.BytesIO(b"new raster"), "plan.png"),
                    "epw_file": (io.BytesIO(b"new weather"), "weather.epw"),
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        expected = {
            "building_plan.dxf": b"new dxf",
            "building_plan.png": b"new raster",
            "weather_data.epw": b"new weather",
        }
        for child_name, contents in expected.items():
            with self.subTest(child_name=child_name):
                child = report_dir / child_name
                self.assertFalse(child.is_symlink())
                self.assertEqual(child.read_bytes(), contents)
                self.assertEqual(outside_files[child_name].read_bytes(), b"outside")

    def test_fixed_uploads_are_staged_before_replacing_existing_file_aliases(self):
        self.login_as("alice")
        report_number = "BIM-ATOMIC-UPLOADS"
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
        report_dir.mkdir(parents=True)
        outside_files = {}
        for child_name in ("building_plan.dxf", "building_plan.png", "weather_data.epw"):
            outside = self.upload_root / f"hardlink-{child_name}"
            outside.write_bytes(b"outside")
            os.link(outside, report_dir / child_name)
            outside_files[child_name] = outside

        fake_document = MagicMock()
        fake_document.modelspace.return_value = []
        with patch.object(self.server, "ezdxf", create=True) as ezdxf:
            ezdxf.readfile.return_value = fake_document
            response = self.client.post(
                "/energy/upload",
                data={
                    "report_number": report_number,
                    "dxf_file": (io.BytesIO(b"new dxf"), "plan.dxf"),
                    "raster_file": (io.BytesIO(b"new raster"), "plan.png"),
                    "epw_file": (io.BytesIO(b"new weather"), "weather.epw"),
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        expected = {
            "building_plan.dxf": b"new dxf",
            "building_plan.png": b"new raster",
            "weather_data.epw": b"new weather",
        }
        for child_name, contents in expected.items():
            with self.subTest(child_name=child_name):
                self.assertEqual((report_dir / child_name).read_bytes(), contents)
                self.assertEqual(outside_files[child_name].read_bytes(), b"outside")

    def test_city_weather_copy_replaces_a_fixed_file_symlink(self):
        self.login_as("alice")
        report_number = "BIM-CITY-WEATHER"
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
        report_dir.mkdir(parents=True)
        outside = self.upload_root / "outside-city-weather.epw"
        outside.write_bytes(b"outside")
        self.symlink_or_skip(report_dir / "weather_data.epw", outside)
        weather_root = Path(self.temporary_directory.name) / "weather"
        weather_root.mkdir()
        (weather_root / "test-city.epw").write_bytes(b"city weather")

        with patch.object(self.server, "WEATHER_DATA_DIR", str(weather_root)):
            response = self.client.post(
                "/energy/upload",
                data={"report_number": report_number, "city_id": "test-city"},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        weather_path = report_dir / "weather_data.epw"
        self.assertFalse(weather_path.is_symlink())
        self.assertEqual(weather_path.read_bytes(), b"city weather")
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_city_weather_copy_is_staged_before_replacing_an_existing_alias(self):
        self.login_as("alice")
        report_number = "BIM-ATOMIC-CITY-WEATHER"
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
        report_dir.mkdir(parents=True)
        outside = self.upload_root / "hardlink-city-weather.epw"
        outside.write_bytes(b"outside")
        os.link(outside, report_dir / "weather_data.epw")
        weather_root = Path(self.temporary_directory.name) / "atomic-weather"
        weather_root.mkdir()
        (weather_root / "test-city.epw").write_bytes(b"city weather")

        with patch.object(self.server, "WEATHER_DATA_DIR", str(weather_root)):
            response = self.client.post(
                "/energy/upload",
                data={"report_number": report_number, "city_id": "test-city"},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual((report_dir / "weather_data.epw").read_bytes(), b"city weather")
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_ifc_upload_replaces_a_fixed_file_symlink(self):
        self.login_as("alice")
        report_number = "BIM-IFC-SYMLINK"
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
        report_dir.mkdir(parents=True)
        outside = self.upload_root / "outside.ifc"
        outside.write_bytes(b"outside")
        self.symlink_or_skip(report_dir / "building_model.ifc", outside)
        parser = MagicMock()
        parser.get_project_info.return_value = {}
        parser.get_element_summary.return_value = {}
        parser.get_storeys.return_value = []
        parser.extract_for_simulation.return_value = {}

        with (
            patch.object(self.server, "HAS_IFC", True),
            patch.object(self.server, "IFCParser", return_value=parser, create=True),
        ):
            response = self.client.post(
                "/energy/upload_ifc",
                data={
                    "report_number": report_number,
                    "ifc_file": (io.BytesIO(b"new ifc"), "model.ifc"),
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        ifc_path = report_dir / "building_model.ifc"
        self.assertFalse(ifc_path.is_symlink())
        self.assertEqual(ifc_path.read_bytes(), b"new ifc")
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_ifc_upload_is_staged_before_replacing_an_existing_alias(self):
        self.login_as("alice")
        report_number = "BIM-ATOMIC-IFC"
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
        report_dir.mkdir(parents=True)
        outside = self.upload_root / "hardlink-building-model.ifc"
        outside.write_bytes(b"outside")
        os.link(outside, report_dir / "building_model.ifc")
        parser = MagicMock()
        parser.get_project_info.return_value = {}
        parser.get_element_summary.return_value = {}
        parser.get_storeys.return_value = []
        parser.extract_for_simulation.return_value = {}

        with (
            patch.object(self.server, "HAS_IFC", True),
            patch.object(self.server, "IFCParser", return_value=parser, create=True),
        ):
            response = self.client.post(
                "/energy/upload_ifc",
                data={
                    "report_number": report_number,
                    "ifc_file": (io.BytesIO(b"new ifc"), "model.ifc"),
                },
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual((report_dir / "building_model.ifc").read_bytes(), b"new ifc")
        self.assertEqual(outside.read_bytes(), b"outside")

    def test_dxf_and_ifc_read_routes_reject_fixed_file_symlinks(self):
        self.login_as("alice")
        dxf_report = self.upload_root / "energy" / user_storage_key("alice") / "BIM-DXF-READ-LINK"
        ifc_report = self.upload_root / "energy" / user_storage_key("alice") / "BIM-IFC-READ-LINK"
        dxf_report.mkdir(parents=True)
        ifc_report.mkdir(parents=True)
        outside_dxf = self.upload_root / "outside-read.dxf"
        outside_ifc = self.upload_root / "outside-read.ifc"
        outside_dxf.write_bytes(b"outside dxf")
        outside_ifc.write_bytes(b"outside ifc")
        self.symlink_or_skip(dxf_report / "building_plan.dxf", outside_dxf)
        self.symlink_or_skip(ifc_report / "building_model.ifc", outside_ifc)

        with patch.object(self.server, "ezdxf", create=True) as ezdxf:
            dxf_response = self.client.post(
                "/energy/geometry",
                json={"report_number": "BIM-DXF-READ-LINK", "layers": []},
            )
        with (
            patch.object(self.server, "HAS_IFC", True),
            patch.object(self.server, "IFCParser", create=True) as parser,
        ):
            ifc_response = self.client.get(
                "/energy/ifc_properties?project=BIM-IFC-READ-LINK"
            )

        for response in (dxf_response, ifc_response):
            with self.subTest(response=response):
                self.assertEqual(response.status_code, 400, response.get_json())
                self.assertEqual(response.get_json(), {"error": "Invalid report path"})
        ezdxf.readfile.assert_not_called()
        parser.assert_not_called()

    def test_energyplus_worker_rejects_symlinked_nested_output_directories(self):
        self.login_as("alice")
        for location in ("runs-root", "job-directory"):
            with self.subTest(location=location):
                report_number = f"BIM-EPLUS-{location}"
                job_id = f"job-{location}"
                report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
                report_dir.mkdir(parents=True)
                (report_dir / "building_plan.dxf").write_bytes(b"dxf")
                (report_dir / "weather_data.epw").write_bytes(b"weather")
                outside = self.upload_root / f"outside-{location}"
                outside.mkdir()
                if location == "runs-root":
                    self.symlink_or_skip(
                        report_dir / "energyplus_runs", outside, target_is_directory=True,
                    )
                else:
                    runs_root = report_dir / "energyplus_runs"
                    runs_root.mkdir()
                    self.symlink_or_skip(
                        runs_root / job_id, outside, target_is_directory=True,
                    )
                self.server._upsert_report_status("alice", report_number, "calculating")
                jobs = MagicMock()
                with (
                    patch.object(self.server, "simulation_jobs", jobs),
                    patch.object(
                        self.server,
                        "extract_geometry",
                        return_value={"floor_area": 100.0, "perimeter": 40.0},
                    ),
                    patch.object(self.server, "energyplus_engine", create=True) as engine,
                ):
                    self.server.background_simulation_task(
                        job_id,
                        {"owner_username": "alice", "report_number": report_number},
                    )

                engine.generate_idf.assert_not_called()
                jobs.update.assert_called_with(
                    job_id, status="failed", error="Simulation failed",
                )

    def test_energyplus_worker_rejects_a_symlinked_weather_input(self):
        report_number = "BIM-EPLUS-WEATHER-LINK"
        job_id = "job-weather-link"
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / report_number
        report_dir.mkdir(parents=True)
        (report_dir / "building_plan.dxf").write_bytes(b"dxf")
        outside = self.upload_root / "outside-worker-weather.epw"
        outside.write_bytes(b"outside weather")
        self.symlink_or_skip(report_dir / "weather_data.epw", outside)
        self.server._upsert_report_status("alice", report_number, "calculating")
        jobs = MagicMock()

        with (
            patch.object(self.server, "simulation_jobs", jobs),
            patch.object(
                self.server,
                "extract_geometry",
                return_value={"floor_area": 100.0, "perimeter": 40.0},
            ),
            patch.object(self.server, "energyplus_engine", create=True) as engine,
        ):
            self.server.background_simulation_task(
                job_id,
                {"owner_username": "alice", "report_number": report_number},
            )

        engine.run_eplus.assert_not_called()
        jobs.update.assert_called_with(job_id, status="failed", error="Simulation failed")

    def test_exterior_lock_rejects_a_fixed_file_symlink(self):
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / "BIM-LOCK-LINK"
        report_dir.mkdir(parents=True)
        outside = self.upload_root / "outside-lock"
        outside.write_bytes(b"outside")
        self.symlink_or_skip(report_dir / ".exterior_generation.lock", outside)

        with self.assertRaises(self.server.InvalidReportPath):
            with self.server._exterior_report_lock(report_dir):
                self.fail("symlinked lock must not be acquired")

        self.assertEqual(outside.read_bytes(), b"outside")

    def test_ordinary_user_cannot_upload_for_another_owner(self):
        self.login_as("alice")

        response = self.client.post(
            "/energy/upload",
            data={
                "report_number": "BIM-1",
                "owner_username": "bob",
                "raster_file": self.image_file(),
            },
        )

        self.assertEqual(response.status_code, 403, response.get_json())
        self.assertFalse((self.upload_root / "energy").exists())

    def test_all_general_report_routes_reject_an_ordinary_users_cross_owner_request(self):
        self.login_as("alice")
        requests = [
            ("get", "/api/v1/simulation_data/BIM-1?owner_username=bob", {}),
            ("get", "/energy/layers/geometry?project=BIM-1&owner_username=bob", {}),
            ("post", "/energy/geometry", {"json": {"report_number": "BIM-1", "owner_username": "bob"}}),
            ("post", "/energy/geometry_advanced", {"json": {"report_number": "BIM-1", "owner_username": "bob"}}),
            ("post", "/energy/calculate", {"json": {"report_number": "BIM-1", "owner_username": "bob"}}),
            ("post", "/energy/upload_ifc", {"data": {"report_number": "BIM-1", "owner_username": "bob"}}),
            ("get", "/energy/ifc_properties?project=BIM-1&owner_username=bob", {}),
            ("get", "/energy/ifc_walls?project=BIM-1&owner_username=bob", {}),
            ("post", "/energy/ifc_simulate", {"json": {"report_number": "BIM-1", "owner_username": "bob"}}),
        ]

        for method, url, kwargs in requests:
            with self.subTest(url=url):
                response = getattr(self.client, method)(url, **kwargs)
                self.assertEqual(response.status_code, 403, response.get_json())
        self.assertFalse((self.upload_root / "energy").exists())

    def test_admin_can_target_existing_user_but_not_unknown_user(self):
        self.login_as("admin", is_admin=True)

        accepted = self.client.post(
            "/energy/upload",
            data={
                "report_number": "BIM-ADMIN",
                "owner_username": "bob",
                "raster_file": self.image_file(),
            },
        )
        own_report = self.client.post(
            "/energy/upload",
            data={"report_number": "ADMIN-OWN", "raster_file": self.image_file()},
        )
        denied = self.client.post(
            "/energy/upload",
            data={
                "report_number": "BIM-MISSING",
                "owner_username": "unknown",
                "raster_file": self.image_file(),
            },
        )

        self.assertEqual(accepted.status_code, 200, accepted.get_json())
        self.assertEqual(own_report.status_code, 200, own_report.get_json())
        self.assertEqual(denied.status_code, 404, denied.get_json())
        self.assertTrue(
            (self.upload_root / "energy" / user_storage_key("bob") / "BIM-ADMIN" / "building_plan.png").is_file()
        )
        self.assertTrue(
            (self.upload_root / "energy" / user_storage_key("admin") / "ADMIN-OWN" / "building_plan.png").is_file()
        )
        self.assertFalse((self.upload_root / "energy" / user_storage_key("unknown")).exists())

    def test_failed_upload_updates_only_the_created_composite_report_row(self):
        self.login_as("alice")

        secret_path = str(
            self.upload_root / "energy" / user_storage_key("alice") / "BIM-FAIL" / "private-parser-file.dxf"
        )
        with self.assertLogs(self.server.logger, level="ERROR") as captured_logs:
            with patch.object(self.server, "allowed_file", side_effect=RuntimeError(secret_path)):
                response = self.client.post(
                    "/energy/upload",
                    data={"report_number": "BIM-FAIL", "raster_file": self.image_file()},
                )

        self.assertEqual(response.status_code, 500, response.get_json())
        self.assertEqual(response.get_json(), {"error": "Internal server error"})
        self.assertNotIn(secret_path, response.get_data(as_text=True))
        self.assertIn(secret_path, "\n".join(captured_logs.output))
        self.assertEqual(self.server._report_row("alice", "BIM-FAIL")["status"], "failed")
        self.assertIsNone(self.server._report_row("bob", "BIM-FAIL"))

    def test_worker_failures_are_sanitized_in_job_status_and_logged_with_detail(self):
        self.login_as("alice")
        jobs_directory = Path(self.temporary_directory.name) / "jobs"
        jobs_directory.mkdir()
        secret_path = str(
            self.upload_root / "energy" / user_storage_key("alice") / "BIM-JOB" / "private-run.idf"
        )

        with (
            patch.object(self.server, "JOBS_DIR", str(jobs_directory)),
            patch.object(
                self.server,
                "_background_energy_report_context",
                side_effect=RuntimeError(secret_path),
            ),
            self.assertLogs(self.server.logger, level="ERROR") as captured_logs,
        ):
            responses = []
            for worker_name in ("simple_simulation_task", "background_simulation_task"):
                job_id = f"job-sensitive-error-{worker_name}"
                self.server.simulation_jobs.set(
                    job_id,
                    {
                        "owner_username": "alice",
                        "report_number": "BIM-JOB",
                        "status": "processing",
                        "progress": 0,
                        "result": None,
                        "error": None,
                    },
                )
                getattr(self.server, worker_name)(
                    job_id,
                    {"owner_username": "alice", "report_number": "BIM-JOB"},
                )
                responses.append(self.client.get(f"/energy/status/{job_id}"))

        for response in responses:
            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertEqual(response.get_json()["error"], "Simulation failed")
            self.assertNotIn(secret_path, response.get_data(as_text=True))
        self.assertIn(secret_path, "\n".join(captured_logs.output))

    def test_job_status_is_owner_scoped_and_requires_explicit_admin_targeting(self):
        jobs_directory = Path(self.temporary_directory.name) / "owner-jobs"
        jobs_directory.mkdir()
        job_id = "alice-private-job"
        legacy_job_id = "legacy-metadata-free-job"
        job = {
            "owner_username": "alice",
            "report_number": "BIM-JOB",
            "status": "completed",
            "progress": 100,
            "result": {"private": "alice-only"},
            "error": None,
        }

        with patch.object(self.server, "JOBS_DIR", str(jobs_directory)):
            self.server.simulation_jobs.set(job_id, job)
            self.server.simulation_jobs.set(
                legacy_job_id,
                {"status": "completed", "progress": 100, "result": {"legacy": True}},
            )
            self.login_as("alice")
            own_response = self.client.get(f"/energy/status/{job_id}")
            self.login_as("bob")
            cross_owner_response = self.client.get(f"/energy/status/{job_id}")
            spoofed_owner_response = self.client.get(
                f"/energy/status/{job_id}?owner_username=alice"
            )
            missing_response = self.client.get("/energy/status/missing-job")
            legacy_response = self.client.get(f"/energy/status/{legacy_job_id}")
            self.login_as("admin", is_admin=True)
            implicit_admin_response = self.client.get(f"/energy/status/{job_id}")
            explicit_admin_response = self.client.get(
                f"/energy/status/{job_id}?owner_username=alice"
            )

        self.assertEqual(own_response.status_code, 200, own_response.get_json())
        self.assertEqual(explicit_admin_response.status_code, 200, explicit_admin_response.get_json())
        hidden_responses = (
            cross_owner_response,
            spoofed_owner_response,
            missing_response,
            legacy_response,
            implicit_admin_response,
        )
        for response in hidden_responses:
            with self.subTest(response=response):
                self.assertEqual(response.status_code, 404, response.get_json())
                self.assertEqual(response.get_json(), {"error": "Job not found"})

    def test_parser_file_error_is_not_misclassified_or_exposed(self):
        self.login_as("alice")
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / "BIM-PARSER"
        report_dir.mkdir(parents=True)
        (report_dir / "building_plan.dxf").write_bytes(b"dxf")
        secret_path = str(report_dir / "private-parser-input.dxf")

        with (
            patch.object(self.server, "ezdxf", create=True) as ezdxf,
            self.assertLogs(self.server.logger, level="ERROR") as captured_logs,
        ):
            ezdxf.readfile.side_effect = FileNotFoundError(secret_path)
            response = self.client.post(
                "/energy/geometry",
                json={"report_number": "BIM-PARSER", "layers": []},
            )

        self.assertEqual(response.status_code, 500, response.get_json())
        self.assertEqual(response.get_json(), {"error": "Internal server error"})
        self.assertNotIn(secret_path, response.get_data(as_text=True))
        self.assertIn(secret_path, "\n".join(captured_logs.output))

    def test_ifc_routes_reject_malformed_report_before_dependency_gate(self):
        self.login_as("alice")
        requests = [
            ("post", "/energy/upload_ifc", {"data": {"report_number": "../escape"}}),
            ("get", "/energy/ifc_properties?project=../escape", {}),
            ("get", "/energy/ifc_walls?project=../escape", {}),
            ("post", "/energy/ifc_simulate", {"json": {"report_number": "../escape"}}),
        ]

        with patch.object(self.server, "HAS_IFC", False):
            for method, url, kwargs in requests:
                with self.subTest(url=url):
                    response = getattr(self.client, method)(url, **kwargs)
                    self.assertEqual(response.status_code, 400, response.get_json())

    def test_storage_errors_are_sanitized_and_use_specific_http_statuses(self):
        self.login_as("alice")

        malformed = self.client.post(
            "/energy/upload",
            data={"report_number": "../outside", "raster_file": self.image_file()},
        )
        missing = self.client.get("/api/v1/simulation_data/MISSING")

        self.assertEqual(malformed.status_code, 400, malformed.get_json())
        self.assertEqual(missing.status_code, 404, missing.get_json())
        self.assertNotIn(str(self.upload_root), malformed.get_data(as_text=True))
        self.assertNotIn(str(self.upload_root), missing.get_data(as_text=True))

    def test_calculation_worker_payload_keeps_resolved_owner_outside_request_context(self):
        self.login_as("alice")
        upload = self.client.post(
            "/energy/upload",
            data={"report_number": "BIM-CALC", "raster_file": self.image_file()},
        )
        self.assertEqual(upload.status_code, 200, upload.get_json())

        fake_thread = MagicMock()
        with (
            patch.object(self.server.threading, "Thread", return_value=fake_thread) as thread_factory,
            patch.object(self.server, "simulation_jobs") as jobs,
        ):
            response = self.client.post(
                "/energy/calculate",
                json={"report_number": "BIM-CALC", "mode": "simple"},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        worker_payload = thread_factory.call_args.kwargs["args"][1]
        self.assertEqual(worker_payload["owner_username"], "alice")
        self.assertEqual(worker_payload["report_number"], "BIM-CALC")
        jobs.set.assert_called_once()
        fake_thread.start.assert_called_once_with()

    def test_calculation_start_marks_only_the_selected_composite_report(self):
        self.login_as("alice")
        for mode in ("simple", "energyplus"):
            with self.subTest(mode=mode):
                report_number = f"BIM-CALC-START-{mode}"
                report_dir = (
                    self.upload_root / "energy" / user_storage_key("alice") / report_number
                )
                report_dir.mkdir(parents=True)
                self.server._upsert_report_status("alice", report_number, "uploaded")
                self.server._upsert_report_status("bob", report_number, "uploaded")
                fake_thread = MagicMock()
                with (
                    patch.object(self.server.threading, "Thread", return_value=fake_thread),
                    patch.object(self.server, "simulation_jobs"),
                ):
                    response = self.client.post(
                        "/energy/calculate",
                        json={"report_number": report_number, "mode": mode},
                    )

                self.assertEqual(response.status_code, 200, response.get_json())
                self.assertEqual(
                    self.server._report_row("alice", report_number)["status"],
                    "calculating",
                )
                self.assertEqual(
                    self.server._report_row("bob", report_number)["status"],
                    "uploaded",
                )

    def test_simple_worker_records_calculated_and_failed_terminal_states(self):
        for outcome in ("success", "failure"):
            with self.subTest(outcome=outcome):
                report_number = f"BIM-SIMPLE-{outcome}"
                report_dir = (
                    self.upload_root / "energy" / user_storage_key("alice") / report_number
                )
                report_dir.mkdir(parents=True)
                self.server._upsert_report_status("alice", report_number, "calculating")
                self.server._upsert_report_status("bob", report_number, "uploaded")
                jobs = MagicMock()
                data = {"owner_username": "alice", "report_number": report_number}
                if outcome == "failure":
                    data["u_wall"] = "not-a-number"
                with patch.object(self.server, "simulation_jobs", jobs):
                    self.server.simple_simulation_task(f"job-{outcome}", data)

                expected = "calculated" if outcome == "success" else "failed"
                self.assertEqual(
                    self.server._report_row("alice", report_number)["status"],
                    expected,
                )
                self.assertEqual(
                    self.server._report_row("bob", report_number)["status"],
                    "uploaded",
                )

    def test_energyplus_worker_records_calculated_and_failed_terminal_states(self):
        for outcome in ("success", "failure"):
            with self.subTest(outcome=outcome):
                report_number = f"BIM-EPLUS-{outcome}"
                report_dir = (
                    self.upload_root / "energy" / user_storage_key("alice") / report_number
                )
                report_dir.mkdir(parents=True)
                (report_dir / "building_plan.dxf").write_bytes(b"dxf")
                (report_dir / "weather_data.epw").write_bytes(b"weather")
                self.server._upsert_report_status("alice", report_number, "calculating")
                self.server._upsert_report_status("bob", report_number, "uploaded")
                jobs = MagicMock()
                with (
                    patch.object(self.server, "simulation_jobs", jobs),
                    patch.object(
                        self.server,
                        "extract_geometry",
                        return_value={"floor_area": 100.0, "perimeter": 40.0},
                    ),
                    patch.object(self.server, "energyplus_engine", create=True) as engine,
                ):
                    if outcome == "failure":
                        engine.run_eplus.side_effect = RuntimeError("simulation failure")
                    else:
                        engine.run_eplus.return_value = report_dir / "results.csv"
                        engine.parse_results.return_value = {
                            "heating_kwh": 10.0,
                            "cooling_kwh": 20.0,
                            "total_kwh": 30.0,
                        }
                    self.server.background_simulation_task(
                        f"job-{outcome}",
                        {"owner_username": "alice", "report_number": report_number},
                    )

                expected = "calculated" if outcome == "success" else "failed"
                self.assertEqual(
                    self.server._report_row("alice", report_number)["status"],
                    expected,
                )
                self.assertEqual(
                    self.server._report_row("bob", report_number)["status"],
                    "uploaded",
                )

    def test_general_dxf_geometry_reads_the_authenticated_users_report(self):
        self.login_as("alice")
        report_dir = self.upload_root / "energy" / user_storage_key("alice") / "BIM-DXF"
        report_dir.mkdir(parents=True)
        (report_dir / "building_plan.dxf").write_bytes(b"alice dxf")
        fake_document = MagicMock()
        fake_document.modelspace.return_value = []

        def readfile(path):
            self.assertEqual(Path(path), report_dir / "building_plan.dxf")
            return fake_document

        with patch.object(self.server, "ezdxf", create=True) as ezdxf:
            ezdxf.readfile.side_effect = readfile
            response = self.client.post(
                "/energy/geometry",
                json={"report_number": "BIM-DXF", "layers": []},
            )

        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(response.get_json(), {"geoms": []})

    def test_ifc_upload_and_read_use_the_same_user_scoped_report(self):
        self.login_as("alice")
        parser = MagicMock()
        parser.get_project_info.return_value = {}
        parser.get_element_summary.return_value = {}
        parser.get_storeys.return_value = []
        parser.extract_for_simulation.return_value = {}
        parser.get_property_tree.return_value = {"name": "alice model"}

        with (
            patch.object(self.server, "HAS_IFC", True),
            patch.object(self.server, "IFCParser", return_value=parser, create=True),
        ):
            upload = self.client.post(
                "/energy/upload_ifc",
                data={"report_number": "BIM-IFC", "ifc_file": (io.BytesIO(b"IFC"), "model.ifc")},
            )
            properties = self.client.get("/energy/ifc_properties?project=BIM-IFC")

        stored = (
            self.upload_root / "energy" / user_storage_key("alice") / "BIM-IFC" / "building_model.ifc"
        )
        self.assertEqual(upload.status_code, 200, upload.get_json())
        self.assertEqual(properties.status_code, 200, properties.get_json())
        self.assertEqual(properties.get_json(), {"name": "alice model"})
        self.assertEqual(stored.read_bytes(), b"IFC")

    def test_benchmark_fixtures_are_stored_outside_user_report_directories(self):
        self.login_as("alice")

        with (
            patch.object(self.server, "HAS_BENCHMARK", True),
            patch.object(self.server, "BESTEST_BENCHMARKS", {"case-1": {}}, create=True),
            patch.object(self.server, "generate_test_dxf", return_value=True, create=True) as generate_dxf,
            patch.object(self.server, "generate_test_ifc", return_value=True, create=True) as generate_ifc,
            patch.object(self.server, "HAS_IFC", True),
        ):
            response = self.client.post("/ops/benchmark/generate_test_files")

        expected_root = self.upload_root / "ops" / "bestest"
        self.assertEqual(response.status_code, 200, response.get_json())
        self.assertEqual(Path(generate_dxf.call_args.args[0]).parent, expected_root)
        self.assertEqual(Path(generate_ifc.call_args.args[0]).parent, expected_root)
        self.assertFalse((self.upload_root / "energy" / "bestest").exists())


if __name__ == "__main__":
    unittest.main()
