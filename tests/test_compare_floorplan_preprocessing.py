import unittest

import cv2
import numpy as np

from tools.compare_floorplan_preprocessing import (
    compute_class_stats,
    prepare_annotation_tool_input,
    prepare_website_input,
    restore_annotation_tool_mask,
    restore_stretched_mask,
)


class CompareFloorplanPreprocessingTests(unittest.TestCase):
    def test_annotation_tool_input_preserves_aspect_ratio_and_centers_image(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)

        tensor, metadata = prepare_annotation_tool_input(image, image_size=512)

        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertEqual(metadata["original_size"], [200, 100])
        self.assertEqual(metadata["resized_size"], [512, 256])
        self.assertEqual(metadata["padding"], {"top": 128, "left": 0})
        expected_black = (np.zeros(3, dtype=np.float32) - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        np.testing.assert_allclose(tensor[0, :, 0, 0], expected_black, rtol=0, atol=1e-6)

    def test_annotation_tool_mask_is_cropped_then_restored_with_nearest_neighbor(self):
        metadata = {
            "original_size": [4, 2],
            "resized_size": [4, 2],
            "padding": {"top": 1, "left": 0},
        }
        model_mask = np.array(
            [[0, 0, 0, 0], [1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 0, 0]],
            dtype=np.uint8,
        )

        restored = restore_annotation_tool_mask(model_mask, metadata)

        np.testing.assert_array_equal(
            restored,
            np.array([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.uint8),
        )

    def test_website_input_directly_stretches_to_square(self):
        image = np.full((30, 90, 3), 255, dtype=np.uint8)

        tensor, metadata = prepare_website_input(image, image_size=64)

        self.assertEqual(tensor.shape, (1, 3, 64, 64))
        self.assertEqual(metadata["original_size"], [90, 30])
        self.assertEqual(metadata["model_size"], [64, 64])

    def test_stretched_mask_is_restored_with_nearest_neighbor(self):
        model_mask = np.array([[1, 2], [3, 0]], dtype=np.uint8)

        restored = restore_stretched_mask(model_mask, [4, 2])

        self.assertEqual(restored.shape, (2, 4))
        self.assertEqual(set(np.unique(restored)), {0, 1, 2, 3})

    def test_class_stats_reports_pixels_percentages_and_external_contours(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        mask[6:9, 1:4] = 1
        mask[1:3, 7:9] = 2
        mask[7:9, 7:9] = 3

        stats = compute_class_stats(mask)

        self.assertEqual(stats["background"]["pixels"], 74)
        self.assertEqual(stats["wall"]["pixels"], 18)
        self.assertEqual(stats["wall"]["percentage"], 18.0)
        self.assertEqual(stats["wall"]["contours"], 2)
        self.assertEqual(stats["window"]["contours"], 1)
        self.assertEqual(stats["door"]["contours"], 1)


if __name__ == "__main__":
    unittest.main()
