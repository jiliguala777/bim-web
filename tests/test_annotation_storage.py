import tempfile
import unittest
from pathlib import Path

import numpy as np

from annotation_tool.config import AnnotationConfig, DEFAULT_DATA_ROOT
from annotation_tool.storage import AnnotationStore


class AnnotationStorageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.store = AnnotationStore(AnnotationConfig(data_root=self.root))

    def tearDown(self):
        self.temporary.cleanup()

    def _prepared_project(self):
        source = self.root / "source.pdf"
        source.write_bytes(b"%PDF-project")
        source_record = self.store.import_source(source)
        project = self.store.create_project("Four Floor Building", source_record.sha256)
        self.store.save_page_manifest(
            project.project_id,
            1,
            {
                "page_count": 4,
                "artifacts": {"model_view": "projects/example/page-1/model_view.png"},
            },
        )
        return project

    def test_default_data_root_uses_the_agreed_external_directory(self):
        self.assertEqual(str(DEFAULT_DATA_ROOT), r"G:\bim网页\标注数据")

    def test_import_source_deduplicates_identical_pdf_bytes(self):
        first = self.root / "first.pdf"
        second = self.root / "second.pdf"
        first.write_bytes(b"%PDF-identical")
        second.write_bytes(b"%PDF-identical")

        first_record = self.store.import_source(first)
        second_record = self.store.import_source(second)

        self.assertEqual(first_record.sha256, second_record.sha256)
        self.assertEqual(first_record.stored_path, second_record.stored_path)
        self.assertTrue(first_record.stored_path.is_file())

    def test_project_ids_cannot_escape_the_external_data_root(self):
        with self.assertRaisesRegex(ValueError, "project_id"):
            self.store.project_directory("../escape")

    def test_mask_saves_keep_history_and_only_confirmed_pages_are_listed(self):
        source = self.root / "source.pdf"
        source.write_bytes(b"%PDF-project")
        source_record = self.store.import_source(source)
        project = self.store.create_project("Four Floor Building", source_record.sha256)
        project_manifest = self.store.save_page_manifest(
            project.project_id,
            1,
            {
                "page_count": 4,
                "artifacts": {"model_view": "projects/example/page-1/model_view.png"},
            },
        )
        self.assertEqual(project_manifest["source_original_name"], "source.pdf")
        self.assertEqual(project_manifest["page_count"], 4)
        self.assertEqual(project_manifest["pages"]["1"]["status"], "prepared")
        self.assertEqual(
            project_manifest["pages"]["1"]["preparation"]["artifacts"]["model_view"],
            "projects/example/page-1/model_view.png",
        )
        first_mask = np.zeros((16, 16), dtype=np.uint8)
        first_mask[2:5, 2:8] = 1
        second_mask = first_mask.copy()
        second_mask[8:10, 3:7] = 3

        first_version = self.store.save_mask_version(
            project.project_id,
            1,
            first_mask,
            status="draft",
            author="tester",
        )
        self.assertEqual(self.store.list_confirmed_pages(project.project_id), [])
        second_version = self.store.save_mask_version(
            project.project_id,
            1,
            second_mask,
            status="confirmed",
            author="tester",
        )

        self.assertNotEqual(first_version.version_id, second_version.version_id)
        self.assertEqual(second_version.previous_version_id, first_version.version_id)
        loaded_mask, loaded_version = self.store.load_current_mask(project.project_id, 1)
        self.assertTrue(np.array_equal(loaded_mask, second_mask))
        self.assertEqual(loaded_version.status, "confirmed")
        self.assertEqual(
            [page.page_number for page in self.store.list_confirmed_pages(project.project_id)],
            [1],
        )
        self.assertTrue(first_version.mask_path.is_file())
        self.assertTrue(second_version.mask_path.is_file())

    def test_regions_are_named_page_children_and_old_pages_default_to_empty(self):
        project = self._prepared_project()

        self.assertEqual(self.store.list_regions(project.project_id, 1), [])
        fourth = self.store.create_region(
            project.project_id,
            1,
            "四层",
            [10, 20, 210, 220],
        )
        fifth = self.store.create_region(
            project.project_id,
            1,
            "五层",
            [220, 20, 390, 180],
        )

        regions = self.store.list_regions(project.project_id, 1)
        self.assertEqual([region["name"] for region in regions], ["四层", "五层"])
        self.assertEqual(fourth["crop_bbox_px"], [10, 20, 210, 220])
        self.assertEqual(len(fourth["region_id"]), 12)
        self.assertNotEqual(fourth["region_id"], fifth["region_id"])
        project_payload = self.store.load_project(project.project_id)
        self.assertIn(fourth["region_id"], project_payload["pages"]["1"]["regions"])

    def test_region_names_are_unique_within_a_page(self):
        project = self._prepared_project()
        self.store.create_region(project.project_id, 1, "四层", [10, 20, 210, 220])

        with self.assertRaisesRegex(ValueError, "name"):
            self.store.create_region(
                project.project_id,
                1,
                " 四层 ",
                [220, 20, 390, 180],
            )

    def test_region_masks_keep_an_independent_version_chain(self):
        project = self._prepared_project()
        region = self.store.create_region(
            project.project_id,
            1,
            "四层",
            [10, 20, 210, 220],
        )
        first_mask = np.zeros((20, 20), dtype=np.uint8)
        first_mask[2:5, 2:8] = 1
        second_mask = first_mask.copy()
        second_mask[8:10, 3:7] = 3

        first = self.store.save_region_mask_version(
            project.project_id,
            1,
            region["region_id"],
            first_mask,
            status="draft",
            author="tester",
        )
        second = self.store.save_region_mask_version(
            project.project_id,
            1,
            region["region_id"],
            second_mask,
            status="confirmed",
            author="tester",
        )

        loaded_mask, loaded_version = self.store.load_current_region_mask(
            project.project_id,
            1,
            region["region_id"],
        )
        self.assertTrue(np.array_equal(loaded_mask, second_mask))
        self.assertEqual(loaded_version.version_id, second.version_id)
        self.assertEqual(second.previous_version_id, first.version_id)
        self.assertIn(
            f"regions\\{region['region_id']}\\versions",
            str(second.mask_path),
        )
        targets = self.store.list_confirmed_targets(project.project_id)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].region_id, region["region_id"])
        self.assertEqual(targets[0].region_name, "四层")
        self.assertEqual(targets[0].crop_bbox_px, (10, 20, 210, 220))


if __name__ == "__main__":
    unittest.main()
