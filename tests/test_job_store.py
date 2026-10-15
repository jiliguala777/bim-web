import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        from job_store import JobStore
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = JobStore(lambda: self.root)

    def test_concurrent_updates_preserve_distinct_fields(self):
        self.store.set("one", {"status": "processing"})
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(lambda n: self.store.update("one", **{f"field_{n}": n}), range(50)))
        result = self.store.get("one")
        self.assertEqual(result["status"], "processing")
        for n in range(50):
            self.assertEqual(result[f"field_{n}"], n)

    def test_failed_replace_preserves_previous_complete_file(self):
        self.store.set("one", {"value": "old"})
        with patch("job_store.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaises(OSError):
                self.store.set("one", {"value": "new"})
        self.assertEqual(json.loads((self.root / "one.json").read_text()), {"value": "old"})
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_paths_cannot_escape_job_root(self):
        for job_id in ("../other", "a/b", "a\\b", "", "a.b"):
            with self.subTest(job_id=job_id), self.assertRaises(ValueError):
                self.store.get(job_id)

    def test_missing_job_reads_do_not_create_files_or_directories(self):
        from job_store import JobStore
        absent_root = self.root / "not-created"
        store = JobStore(absent_root)
        self.assertIsNone(store.get("nonexistent"))
        self.assertFalse(absent_root.exists())
        self.assertIsNone(self.store.get("nonexistent"))
        self.assertEqual(list(self.root.iterdir()), [])
