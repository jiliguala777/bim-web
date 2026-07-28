import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from reportlab.pdfgen import canvas

from floorplan_onnx import FloorplanSegmenterONNX
from floorplan_page_pipeline import (
    VectorAnalysisOutcome,
    _run_vector_analysis,
    prepare_pdf_page,
)


def _slow_vector_worker(result_path, source, page_index, dpi):
    import time

    time.sleep(30)


class VectorAnalysisTimeoutTests(unittest.TestCase):
    def test_vector_worker_is_terminated_after_timeout(self):
        source = Path(__file__)
        started = time.perf_counter()

        outcome = _run_vector_analysis(
            source,
            page_index=0,
            dpi=100,
            timeout_seconds=0.1,
            worker_target=_slow_vector_worker,
        )

        self.assertLess(time.perf_counter() - started, 5)
        self.assertIsNone(outcome.page_data)
        self.assertEqual(outcome.evidence["status"], "timed_out")
        self.assertEqual(outcome.evidence["mode"], "raster_fallback")
        self.assertEqual(outcome.evidence["timeout_seconds"], 0.1)


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
    def _segmenter():
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        return segmenter

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
        poppler_path = os.environ.get("POPPLER_PATH")
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                2,
                poppler_path=poppler_path,
                segmenter=self._segmenter(),
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

    @patch("floorplan_page_pipeline._run_vector_analysis")
    def test_prepare_pdf_page_writes_raster_fallback_metadata(self, run_analysis):
        run_analysis.return_value = VectorAnalysisOutcome(
            None,
            {
                "status": "timed_out",
                "mode": "raster_fallback",
                "timeout_seconds": 15,
                "reason": "vector extraction exceeded 15 seconds",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifacts"
            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                1,
                poppler_path=os.environ.get("POPPLER_PATH"),
                segmenter=self._segmenter(),
                output_dir=output,
            )
            metadata = json.loads(
                (output / "preprocessing.json").read_text(encoding="utf-8")
            )

        self.assertEqual(prepared.vector_analysis["mode"], "raster_fallback")
        self.assertEqual(metadata["render_dpi"], 100)
        self.assertEqual(metadata["vector_analysis"]["status"], "timed_out")
        self.assertEqual(prepared.scale_calibration["status"], "manual_required")
        self.assertFalse(prepared.vector_cleanup["enabled"])
        self.assertIsNone(prepared.inference_roi)
        self.assertIsNone(prepared.structural_support_mask)
        self.assertEqual(
            set(prepared.artifacts),
            {"render", "cleaned", "model_view", "model_input_512", "metadata"},
        )

    @patch("floorplan_page_pipeline._run_vector_analysis")
    def test_prepare_pdf_page_keeps_completed_vector_analysis(self, run_analysis):
        run_analysis.return_value = VectorAnalysisOutcome(
            {
                "page_size_pt": [400.0, 300.0],
                "render_size_px": [556, 417],
                "dpi": 100,
                "text_spans": [],
                "segments": [],
                "styled_edges": [],
                "vector_text_count": 0,
                "vector_segment_count": 0,
                "styled_edge_count": 0,
                "has_vector_text": False,
                "has_vector_geometry": False,
                "is_vector_pdf": False,
            },
            {
                "status": "completed",
                "mode": "vector",
                "timeout_seconds": 15,
                "reason": None,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            prepared = prepare_pdf_page(
                self._make_two_page_pdf(directory),
                1,
                poppler_path=os.environ.get("POPPLER_PATH"),
                segmenter=self._segmenter(),
            )
        self.assertEqual(prepared.vector_analysis["status"], "completed")


if __name__ == "__main__":
    unittest.main()
