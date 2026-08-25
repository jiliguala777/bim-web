import tempfile
import unittest
from pathlib import Path

import numpy as np
from reportlab.pdfgen import canvas


class NativePdfExtractionTests(unittest.TestCase):
    @staticmethod
    def _make_mixed_vector_pdf(directory: str) -> Path:
        path = Path(directory) / "mixed-vector.pdf"
        pdf = canvas.Canvas(str(path), pagesize=(400, 300))

        pdf.setLineWidth(1.0)
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.rect(80, 60, 240, 180, stroke=1, fill=0)
        pdf.line(80, 60, 320, 240)

        pdf.setLineWidth(0.5)
        pdf.line(80, 270, 320, 270)
        pdf.line(80, 264, 80, 276)
        pdf.line(320, 264, 320, 276)
        pdf.drawString(183, 278, "24000")

        pdf.setStrokeColorRGB(0.5, 0.5, 0.5)
        pdf.line(120, 150, 280, 150)

        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.rect(2, 2, 396, 296, stroke=1, fill=0)
        pdf.save()
        return path

    def test_extracts_only_orthogonal_candidates_and_counts_diagonals(self):
        from vector_pdf_native import extract_native_pdf_page

        with tempfile.TemporaryDirectory() as directory:
            page = extract_native_pdf_page(
                self._make_mixed_vector_pdf(directory),
                dpi=100,
            )

        self.assertTrue(page["is_vector_pdf"])
        self.assertGreaterEqual(len(page["orthogonal_segments"]), 7)
        self.assertEqual(page["ignored_diagonal_count"], 1)
        self.assertTrue(page["dimension_candidates"])
        self.assertIn("enabled", page["building_roi"])
        self.assertTrue(
            all(item["orientation"] in {"horizontal", "vertical"}
                for item in page["orthogonal_segments"])
        )
        self.assertEqual(
            len({item["native_id"] for item in page["orthogonal_segments"]}),
            len(page["orthogonal_segments"]),
        )

    def test_dimension_mask_marks_dimension_members_without_covering_room_wall(self):
        from vector_pdf_native import build_dimension_mask, extract_native_pdf_page

        with tempfile.TemporaryDirectory() as directory:
            page = extract_native_pdf_page(self._make_mixed_vector_pdf(directory), dpi=100)
        mask = build_dimension_mask(page)

        self.assertEqual(mask.shape, (417, 556))
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertEqual(mask[333, 200], 0)

    def test_crop_translates_analysis_coordinates_and_preserves_page_coordinates(self):
        from vector_pdf_native import crop_native_page_data, extract_native_pdf_page

        with tempfile.TemporaryDirectory() as directory:
            page = extract_native_pdf_page(self._make_mixed_vector_pdf(directory), dpi=100)
        cropped = crop_native_page_data(page, [100, 50, 400, 350])

        self.assertEqual(cropped["render_size_px"], [300, 300])
        self.assertEqual(cropped["crop_bbox_page_px"], [100, 50, 400, 350])
        self.assertTrue(cropped["orthogonal_segments"])
        for item in cropped["orthogonal_segments"]:
            self.assertIn("page_start_px", item)
            self.assertIn("page_end_px", item)
            for x, y in (item["start_px"], item["end_px"]):
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x, 300)
                self.assertLessEqual(y, 300)


if __name__ == "__main__":
    unittest.main()
