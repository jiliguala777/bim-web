import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from energy_pdf_region import (
    crop_page_inputs,
    map_crop_bbox_to_page,
    parse_crop_region_request,
    render_pdf_page_preview,
)


class CropRegionValidationTests(unittest.TestCase):
    def test_parses_a_valid_crop_region_request(self):
        request_data = parse_crop_region_request(
            "crop_region",
            "[100, 50, 500, 350]",
            "[800, 600]",
        )

        self.assertEqual(request_data["crop_bbox_px"], [100, 50, 500, 350])

    def test_maps_preview_crop_coordinates_to_page_coordinates(self):
        self.assertEqual(
            map_crop_bbox_to_page([100, 50, 500, 350], [800, 600], [1600, 1200]),
            [200, 100, 1000, 700],
        )

    def test_rejects_malformed_json_arrays(self):
        with self.assertRaises(ValueError):
            parse_crop_region_request("crop_region", "[100, 50", "[800, 600]")

    def test_rejects_non_finite_coordinates(self):
        with self.assertRaises(ValueError):
            parse_crop_region_request("crop_region", "[100, NaN, 500, 350]", "[800, 600]")

    def test_rejects_boolean_coordinates(self):
        with self.assertRaises(ValueError):
            parse_crop_region_request(
                "crop_region",
                "[100, false, 500, 350]",
                "[800, 600]",
            )

    def test_rejects_unreasonably_large_pixel_values_without_overflow(self):
        huge_pixel_value = 10 ** 1000

        with self.assertRaises(ValueError):
            parse_crop_region_request(
                "crop_region",
                f"[0, 0, 128, {huge_pixel_value}]",
                f"[128, {huge_pixel_value}]",
            )
        with self.assertRaises(ValueError):
            map_crop_bbox_to_page(
                [0, 0, 128, huge_pixel_value],
                [128, huge_pixel_value],
                [1600, 1200],
            )

    def test_rejects_crop_coordinates_outside_preview(self):
        with self.assertRaises(ValueError):
            parse_crop_region_request("crop_region", "[100, 50, 801, 350]", "[800, 600]")

    def test_rejects_crop_dimensions_below_128_pixels(self):
        with self.assertRaises(ValueError):
            parse_crop_region_request("crop_region", "[100, 50, 227, 350]", "[800, 600]")

    def test_rejects_crop_height_below_128_pixels(self):
        with self.assertRaises(ValueError):
            parse_crop_region_request("crop_region", "[100, 50, 500, 177]", "[800, 600]")

    def test_rejects_crop_area_below_one_percent_of_preview(self):
        with self.assertRaises(ValueError):
            parse_crop_region_request("crop_region", "[100, 50, 228, 178]", "[2000, 1000]")

    def test_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            parse_crop_region_request("invalid", None, None)

    def test_mapping_rejects_crop_coordinates_outside_preview(self):
        with self.assertRaises(ValueError):
            map_crop_bbox_to_page([-1, 50, 500, 350], [800, 600], [1600, 1200])
        with self.assertRaises(ValueError):
            map_crop_bbox_to_page([100, 50, 801, 350], [800, 600], [1600, 1200])

    def test_mapping_rejects_reversed_crop_coordinates(self):
        with self.assertRaises(ValueError):
            map_crop_bbox_to_page([500, 50, 100, 350], [800, 600], [1600, 1200])

    def test_mapping_rejects_non_finite_crop_coordinates(self):
        with self.assertRaises(ValueError):
            map_crop_bbox_to_page([100, float("nan"), 500, 350], [800, 600], [1600, 1200])

    def test_full_page_accepts_missing_crop_fields(self):
        request_data = parse_crop_region_request("full_page", None, None)

        self.assertIsNone(request_data["crop_bbox_px"])


class CropPageInputsTests(unittest.TestCase):
    def setUp(self):
        height, width = 600, 800
        rows, columns = np.indices((height, width))
        self.render_bgr = np.stack((columns % 256, rows % 256, (columns + rows) % 256), axis=2).astype(np.uint8)
        self.cleaned_bgr = (255 - self.render_bgr).astype(np.uint8)
        self.cleanup_mask = (columns + rows).astype(np.uint16)
        self.structural_support_mask = (columns * 2 + rows).astype(np.uint16)

    def test_crops_all_arrays_and_translates_intersecting_inference_roi(self):
        result = crop_page_inputs(
            self.render_bgr,
            self.cleaned_bgr,
            self.cleanup_mask,
            self.structural_support_mask,
            [120, 80, 700, 500],
            [100, 50, 500, 350],
        )

        self.assertEqual(result["render_bgr"].shape[:2], (300, 400))
        self.assertEqual(result["cleaned_bgr"].shape[:2], (300, 400))
        self.assertEqual(result["cleanup_mask"].shape, (300, 400))
        self.assertEqual(result["structural_support_mask"].shape, (300, 400))
        self.assertEqual(result["inference_roi"], [20, 30, 400, 300])
        self.assertEqual(result["render_bgr"][0, 0].tolist(), self.render_bgr[50, 100].tolist())
        self.assertIsNot(result["render_bgr"].base, self.render_bgr)
        self.assertIsNot(result["cleaned_bgr"].base, self.cleaned_bgr)
        self.assertIsNot(result["cleanup_mask"].base, self.cleanup_mask)
        self.assertIsNot(result["structural_support_mask"].base, self.structural_support_mask)

    def test_uses_complete_crop_roi_when_source_roi_does_not_intersect(self):
        result = crop_page_inputs(
            self.render_bgr,
            self.cleaned_bgr,
            self.cleanup_mask,
            None,
            [600, 400, 700, 500],
            [100, 50, 500, 350],
        )

        self.assertIsNone(result["structural_support_mask"])
        self.assertEqual(result["inference_roi"], [0, 0, 400, 300])


class PdfPreviewRenderTests(unittest.TestCase):
    def test_renders_exactly_requested_page_as_bgr(self):
        pdf_path = Path("example.pdf")
        poppler_path = "C:/poppler/bin"
        image = Image.new("RGB", (2, 1), color=(10, 20, 30))
        with patch("energy_pdf_region.convert_from_path", create=True, return_value=[image]) as convert:
            result = render_pdf_page_preview(pdf_path, 2, 3, poppler_path)

        convert.assert_called_once_with(
            str(pdf_path),
            dpi=100,
            first_page=2,
            last_page=2,
            poppler_path=poppler_path,
        )
        self.assertEqual(result.shape, (1, 2, 3))
        self.assertEqual(result[0, 0].tolist(), [30, 20, 10])

    def test_rejects_page_number_outside_document_bounds(self):
        with self.assertRaises(ValueError):
            render_pdf_page_preview(Path("example.pdf"), 0, 3, None)
        with self.assertRaises(ValueError):
            render_pdf_page_preview(Path("example.pdf"), 4, 3, None)

    def test_rejects_renderer_results_that_are_not_one_image(self):
        with patch("energy_pdf_region.convert_from_path", create=True, return_value=[]) as convert:
            with self.assertRaises(ValueError):
                render_pdf_page_preview(Path("example.pdf"), 1, 1, None)

        convert.assert_called_once()


if __name__ == "__main__":
    unittest.main()
