import tempfile
import unittest
from pathlib import Path


class UserStorageKeyTests(unittest.TestCase):
    def test_preserves_readable_unicode_and_adds_stable_hash(self):
        from energy_report_storage import user_storage_key

        self.assertEqual(user_storage_key("张三"), "张三-1d841bc0")
        self.assertEqual(user_storage_key("alice"), "alice-2bd806c9")

    def test_cleaned_collisions_still_have_distinct_hashes(self):
        from energy_report_storage import user_storage_key

        self.assertNotEqual(user_storage_key("a/b"), user_storage_key("a\\b"))

    def test_rejects_traversal_report_numbers(self):
        from energy_report_storage import InvalidReportPath, validate_report_number

        for value in ("", ".", "..", "../R", "R/child", "R\\child", "/tmp/R"):
            with self.subTest(value=value), self.assertRaises(InvalidReportPath):
                validate_report_number(value)

    def test_rejects_report_numbers_too_long_for_a_portable_path_component(self):
        from energy_report_storage import InvalidReportPath, validate_report_number

        for value in ("R" * 256, "报告" * 43):
            with self.subTest(value=value), self.assertRaises(InvalidReportPath):
                validate_report_number(value)

    def test_accepts_ascii_and_unicode_report_numbers_at_the_255_byte_limit(self):
        from energy_report_storage import validate_report_number

        for value in ("R" * 255, "界" * 85):
            with self.subTest(value=value):
                self.assertEqual(len(value.encode("utf-8")), 255)
                self.assertEqual(validate_report_number(value), value)

    def test_rejects_windows_device_names_and_trailing_dots_or_spaces(self):
        from energy_report_storage import InvalidReportPath, validate_report_number

        for value in (
            "CON",
            "con.txt",
            "PRN.json",
            "AUX",
            "NUL",
            "COM1",
            "com9.anything",
            "LPT1",
            "lpt9.txt",
            "report.",
            "report ",
        ):
            with self.subTest(value=value), self.assertRaises(InvalidReportPath):
                validate_report_number(value)


class EnergyReportStorageResolutionTests(unittest.TestCase):
    def _symlink_or_skip(self, link: Path, target: Path):
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                self.skipTest("creating symlinks requires Windows developer mode or privilege")
            raise

    def test_same_report_number_resolves_below_distinct_users(self):
        from energy_report_storage import resolve_energy_report_context

        with tempfile.TemporaryDirectory() as directory:
            alice = resolve_energy_report_context(directory, "alice", False, "BIM-1", create=True)
            bob = resolve_energy_report_context(directory, "bob", False, "BIM-1", create=True)
            self.assertNotEqual(alice.report_dir, bob.report_dir)
            self.assertTrue(alice.report_dir.is_dir())
            self.assertTrue(bob.report_dir.is_dir())

    def test_rejects_symlinked_owner_root(self):
        from energy_report_storage import InvalidReportPath, resolve_energy_report_context, user_storage_key

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            (root / "energy").mkdir()
            self._symlink_or_skip(root / "energy" / user_storage_key("alice"), outside)
            with self.assertRaises(InvalidReportPath):
                resolve_energy_report_context(root, "alice", False, "BIM-1")

    def test_read_resolution_does_not_create_missing_directories(self):
        from energy_report_storage import resolve_energy_report_context

        with tempfile.TemporaryDirectory() as directory:
            context = resolve_energy_report_context(directory, "alice", False, "BIM-1")
            self.assertFalse((Path(directory) / "energy").exists())
            self.assertFalse(context.report_dir.exists())

    def test_rejects_report_directory_symlink(self):
        from energy_report_storage import InvalidReportPath, resolve_energy_report_context, user_storage_key

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            owner_root = root / "energy" / user_storage_key("alice")
            outside = root / "outside"
            owner_root.mkdir(parents=True)
            outside.mkdir()
            self._symlink_or_skip(owner_root / "BIM-1", outside)
            with self.assertRaises(InvalidReportPath):
                resolve_energy_report_context(root, "alice", False, "BIM-1")

    def test_admin_can_resolve_requested_owner(self):
        from energy_report_storage import resolve_energy_report_context

        with tempfile.TemporaryDirectory() as directory:
            context = resolve_energy_report_context(directory, "admin", True, "BIM-1", requested_owner="alice")
            self.assertEqual(context.owner_username, "alice")

    def test_non_admin_cannot_resolve_requested_owner(self):
        from energy_report_storage import ReportAccessDenied, resolve_energy_report_context

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ReportAccessDenied):
                resolve_energy_report_context(directory, "alice", False, "BIM-1", requested_owner="bob")

    def test_non_admin_can_request_own_owner(self):
        from energy_report_storage import resolve_energy_report_context

        with tempfile.TemporaryDirectory() as directory:
            context = resolve_energy_report_context(directory, "alice", False, "BIM-1", requested_owner="alice")
            self.assertEqual(context.owner_username, "alice")


if __name__ == "__main__":
    unittest.main()
