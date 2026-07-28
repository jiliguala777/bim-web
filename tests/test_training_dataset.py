import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

try:
    import torch  # noqa: F401
except ImportError as exc:
    raise unittest.SkipTest("training tests require .venv-train") from exc

from training.dataset import ExportedFloorplanDataset, validate_export


class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        self.image_path = self.root / "images" / "page-1.png"
        self.mask_path = self.root / "masks" / "page-1.npy"
        image = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.line(image, (50, 100), (460, 100), (0, 0, 0), 5)
        cv2.imwrite(str(self.image_path), image)
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[98:103, 50:461] = 1
        np.save(self.mask_path, mask, allow_pickle=False)
        self._write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def _write_manifest(self, **sample_changes):
        sample = {
            "sample_id": "page-1",
            "page_number": 1,
            "status": "confirmed",
            "image": "images/page-1.png",
            "mask": "masks/page-1.npy",
        }
        sample.update(sample_changes)
        (self.root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_type": "single_page_overfit",
                    "class_map": {
                        "0": "background",
                        "1": "wall",
                        "2": "window",
                        "3": "door",
                    },
                    "samples": [sample],
                }
            ),
            encoding="utf-8",
        )

    def test_valid_export_loads_normalized_512_tensors(self):
        report = validate_export(self.root)
        dataset = ExportedFloorplanDataset(self.root, augment=False, seed=7)
        sample = dataset[0]

        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(tuple(sample["image"].shape), (3, 512, 512))
        self.assertEqual(tuple(sample["mask"].shape), (512, 512))
        self.assertEqual(sample["sample_id"], "page-1")
        self.assertEqual(str(sample["image"].dtype), "torch.float32")
        self.assertEqual(str(sample["mask"].dtype), "torch.int64")

    def test_export_under_a_unicode_root_loads_the_training_image(self):
        unicode_root = self.root / "标注数据"
        image_path = unicode_root / "images" / "第13页.png"
        mask_path = unicode_root / "masks" / "第13页.npy"
        image_path.parent.mkdir(parents=True)
        mask_path.parent.mkdir(parents=True)
        image = np.full((512, 512, 3), 255, dtype=np.uint8)
        cv2.line(image, (50, 100), (460, 100), (0, 0, 0), 5)
        encoded, buffer = cv2.imencode(".png", image)
        self.assertTrue(encoded)
        image_path.write_bytes(buffer.tobytes())
        np.save(mask_path, np.zeros((512, 512), dtype=np.uint8), allow_pickle=False)
        (unicode_root / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "experiment_type": "single_page_overfit",
                    "samples": [
                        {
                            "sample_id": "page-13",
                            "page_number": 13,
                            "status": "confirmed",
                            "image": "images/第13页.png",
                            "mask": "masks/第13页.npy",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        report = validate_export(unicode_root)
        sample = ExportedFloorplanDataset(
            unicode_root,
            augment=False,
            seed=7,
        )[0]

        self.assertEqual(report["sample_count"], 1)
        self.assertEqual(tuple(sample["image"].shape), (3, 512, 512))
        self.assertEqual(sample["sample_id"], "page-13")

    def test_invalid_exports_raise_descriptive_errors(self):
        cases = {}

        self.image_path.unlink()
        cases["missing image"] = lambda: None
        with self.subTest("missing image"):
            with self.assertRaisesRegex(ValueError, "image.*missing"):
                validate_export(self.root)
        cv2.imwrite(str(self.image_path), np.full((512, 512, 3), 255, np.uint8))

        np.save(self.mask_path, np.zeros((256, 512), dtype=np.uint8), allow_pickle=False)
        with self.subTest("wrong mask shape"):
            with self.assertRaisesRegex(ValueError, "512x512"):
                validate_export(self.root)

        invalid = np.zeros((512, 512), dtype=np.uint8)
        invalid[0, 0] = 4
        np.save(self.mask_path, invalid, allow_pickle=False)
        with self.subTest("class range"):
            with self.assertRaisesRegex(ValueError, "0..3"):
                validate_export(self.root)

        np.save(self.mask_path, np.zeros((512, 512), dtype=np.uint8), allow_pickle=False)
        self._write_manifest(status="draft")
        with self.subTest("confirmation"):
            with self.assertRaisesRegex(ValueError, "confirmed"):
                validate_export(self.root)

    def test_seeded_augmentation_is_repeatable(self):
        first = ExportedFloorplanDataset(self.root, augment=True, seed=19)[0]
        second = ExportedFloorplanDataset(self.root, augment=True, seed=19)[0]

        self.assertTrue(first["image"].equal(second["image"]))
        self.assertTrue(first["mask"].equal(second["mask"]))


if __name__ == "__main__":
    unittest.main()
