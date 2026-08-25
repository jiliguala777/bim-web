import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from reportlab.pdfgen import canvas

from vector_pdf_model import (
    CHANNEL_NAMES,
    VectorModelUnavailableError,
    VectorProbabilityResult,
)


class VectorPdfFusionPipelineTests(unittest.TestCase):
    @staticmethod
    def _make_room_pdf(directory):
        path = Path(directory) / "room.pdf"
        pdf = canvas.Canvas(str(path), pagesize=(400, 300))
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.setLineWidth(1.0)
        pdf.rect(40, 30, 320, 240, stroke=1, fill=0)
        pdf.rect(100, 80, 200, 140, stroke=1, fill=0)
        pdf.rect(2, 2, 396, 296, stroke=1, fill=0)
        pdf.save()
        return path

    @staticmethod
    def _supported_runner(image_path, artifact_parent, config, *, expected_size):
        width, height = expected_size
        probabilities = np.zeros((10, height, width), dtype=np.float32)
        probabilities[0, :, :] = 0.8
        probabilities[2, :, :] = 0.7
        probabilities[4, :, :] = 0.8
        artifact_dir = Path(artifact_parent) / "fake-model"
        artifact_dir.mkdir(parents=True)
        np.savez_compressed(artifact_dir / "probabilities.npz", probabilities=probabilities)
        inference = {
            "format": "vector-floorplan-inference/1",
            "channel_names": list(CHANNEL_NAMES),
            "image_size": [width, height],
            "runtime_seconds": 0.01,
        }
        (artifact_dir / "inference.json").write_text(
            json.dumps(inference), encoding="utf-8"
        )
        return VectorProbabilityResult(
            probabilities=probabilities,
            inference=inference,
            artifact_dir=artifact_dir,
            probabilities_sha256="a" * 64,
        )

    def test_publishes_evaluable_debug_artifacts(self):
        from vector_pdf_fusion_pipeline import analyze_vector_pdf_page

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf = self._make_room_pdf(root)
            output = root / "output"
            result = analyze_vector_pdf_page(
                pdf,
                1,
                output,
                model_config=object(),
                model_runner=self._supported_runner,
            )

            self.assertEqual(result["status"], "evaluable")
            self.assertFalse(result["load_geometry_ready"])
            self.assertTrue((output / "pdf_native_candidates.json").is_file())
            self.assertTrue((output / "pdf_vector_fusion.json").is_file())
            self.assertTrue((output / "pdf_vector_fusion_overlay.png").is_file())
            payload = json.loads(
                (output / "pdf_vector_fusion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["format"], "pdf-vector-fusion/1")
            self.assertFalse(payload["load_geometry_ready"])
            self.assertGreaterEqual(payload["summary"]["accepted_wall_count"], 4)
            self.assertGreaterEqual(payload["summary"]["accepted_room_count"], 1)

    def test_model_failure_publishes_partial_native_diagnostics(self):
        from vector_pdf_fusion_pipeline import analyze_vector_pdf_page

        def unavailable_runner(*args, **kwargs):
            raise VectorModelUnavailableError("test model unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            result = analyze_vector_pdf_page(
                self._make_room_pdf(root),
                1,
                output,
                model_config=object(),
                model_runner=unavailable_runner,
            )

            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["reason_codes"], ["model_unavailable"])
            self.assertFalse(result["load_geometry_ready"])
            self.assertTrue((output / "pdf_native_candidates.json").is_file())
            payload = json.loads(
                (output / "pdf_vector_fusion.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "partial")
            self.assertTrue(payload["line_candidates"])
            self.assertTrue(all(item["model_evidence"] is None for item in payload["line_candidates"]))

    def test_image_only_pdf_is_rejected_before_model_runs(self):
        from vector_pdf_fusion_pipeline import analyze_vector_pdf_page

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "scan.png"
            Image.new("RGB", (200, 100), "white").save(image_path)
            pdf_path = root / "scan.pdf"
            pdf = canvas.Canvas(str(pdf_path), pagesize=(400, 300))
            pdf.drawImage(str(image_path), 50, 50, width=300, height=150)
            pdf.save()
            called = False

            def forbidden_runner(*args, **kwargs):
                nonlocal called
                called = True
                raise AssertionError("model must not run for image-only PDF")

            result = analyze_vector_pdf_page(
                pdf_path,
                1,
                root / "output",
                model_config=object(),
                model_runner=forbidden_runner,
            )

            self.assertFalse(called)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["reason_codes"], ["not_vector_pdf"])
            self.assertFalse(result["load_geometry_ready"])

    def test_crop_is_reported_in_page_coordinates_and_model_uses_crop_size(self):
        from vector_pdf_fusion_pipeline import analyze_vector_pdf_page

        observed = {}

        def recording_runner(image_path, artifact_parent, config, *, expected_size):
            observed["expected_size"] = expected_size
            return self._supported_runner(
                image_path, artifact_parent, config, expected_size=expected_size
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = analyze_vector_pdf_page(
                self._make_room_pdf(root),
                1,
                root / "output",
                model_config=object(),
                crop_bbox_page_px=[50, 40, 500, 380],
                model_runner=recording_runner,
            )

        self.assertEqual(result["page"]["crop_bbox_page_px"], [50, 40, 500, 380])
        self.assertEqual(observed["expected_size"], (450, 340))


if __name__ == "__main__":
    unittest.main()
