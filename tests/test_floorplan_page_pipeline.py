import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from reportlab.pdfgen import canvas

from floorplan_onnx import FloorplanSegmenterONNX
from floorplan_page_pipeline import prepare_pdf_page


class ModelInputPreparationTests(unittest.TestCase):
    def test_prepare_model_input_preserves_aspect_ratio_and_reports_padding(self):
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        image = np.zeros((200, 400, 3), dtype=np.uint8)

        model_view, tensor, metadata = segmenter.prepare_model_input(image)

        self.assertEqual(model_view.shape, (200, 400, 3))
        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(metadata["original_size"], [400, 200])
        self.assertEqual(metadata["resized_size"], [512, 256])
        self.assertEqual(metadata["padding"], [128, 0, 128, 0])
        self.assertAlmostEqual(metadata["resize_scale"], 1.28)


class PdfPagePipelineTests(unittest.TestCase):
    @staticmethod
    def _make_two_page_pdf(directory):
        pdf_path = Path(directory) / "two-pages.pdf"
        pdf = canvas.Canvas(str(pdf_path), pagesize=(400, 300))
        pdf.setFillColorRGB(0, 0, 0)
        pdf.rect(25, 100, 80, 80, fill=1, stroke=0)
        pdf.showPage()
        pdf.setFillColorRGB(0, 0, 0)
        pdf.rect(295, 100, 80, 80, fill=1, stroke=0)
        pdf.save()
        return pdf_path

    def test_prepare_pdf_page_renders_only_the_requested_one_based_page(self):
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        poppler_path = os.environ.get("POPPLER_PATH")
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                2,
                poppler_path=poppler_path,
                segmenter=segmenter,
                output_dir=Path(directory) / "artifacts",
            )

            self.assertEqual(prepared.page_number, 2)
            self.assertEqual(prepared.page_count, 2)
            self.assertEqual(prepared.model_input_512.shape, (1, 3, 512, 512))
            gray = prepared.render_bgr.mean(axis=2)
            midpoint = gray.shape[1] // 2
            self.assertLess(gray[:, midpoint:].mean(), gray[:, :midpoint].mean())
            self.assertIn("render", prepared.artifacts)
            self.assertEqual(len(prepared.artifacts["render"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
