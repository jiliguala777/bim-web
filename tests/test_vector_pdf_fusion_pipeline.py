import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
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
    def _stub_render(*args, **kwargs):
        return [Image.new("RGB", (556, 417), "white")]

    def setUp(self):
        self._render_fallback = None
        poppler_path = os.environ.get("POPPLER_PATH")
        has_poppler = shutil.which("pdftoppm") is not None or (
            poppler_path is not None and (Path(poppler_path) / "pdftoppm.exe").is_file()
        )
        if not has_poppler:
            self._render_fallback = patch(
                "vector_pdf_fusion_pipeline.convert_from_path", self._stub_render,
            )
            self._render_fallback.start()

    def tearDown(self):
        if self._render_fallback is not None:
            self._render_fallback.stop()

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
    def _make_room_with_door_gap_pdf(directory):
        path = Path(directory) / "room-with-door-gap.pdf"
        pdf = canvas.Canvas(str(path), pagesize=(400, 300))
        pdf.setStrokeColorRGB(0, 0, 0)
        pdf.setLineWidth(1.0)
        pdf.line(40, 30, 180, 30)
        pdf.line(220, 30, 360, 30)
        pdf.line(360, 30, 360, 270)
        pdf.line(360, 270, 40, 270)
        pdf.line(40, 270, 40, 30)
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

    @staticmethod
    def _opening_runner(image_path, artifact_parent, config, *, expected_size):
        width, height = expected_size
        probabilities = np.zeros((10, height, width), dtype=np.float32)
        if (width, height) == (480, 375):
            probabilities[0, 15:353, 14:463] = 0.9
        else:
            probabilities[0, 40:378, 54:503] = 0.9
        probabilities[2, :, :] = 0.7
        probabilities[4, :, :] = 0.8
        probabilities[5, :, :] = 0.9
        probabilities[8, :, :] = 0.9
        artifact_dir = Path(artifact_parent) / "fake-opening-model"
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
            probabilities_sha256="b" * 64,
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

    def test_pipeline_publishes_unconfirmed_exterior_and_openings(self):
        """Removing exterior integration must remove these review-only artifacts."""
        from vector_pdf_fusion_pipeline import analyze_vector_pdf_page

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            with patch("vector_pdf_fusion_pipeline.convert_from_path", self._stub_render):
                result = analyze_vector_pdf_page(
                    self._make_room_with_door_gap_pdf(root),
                    1,
                    output,
                    model_config=object(),
                    model_runner=self._opening_runner,
                )

            self.assertFalse(result["load_geometry_ready"])
            self.assertFalse(result["exterior_topology"]["confirmed"])
            self.assertFalse(result["exterior_topology"]["load_geometry_ready"])
            self.assertTrue((output / "pdf_opening_candidates.json").is_file())
            self.assertTrue((output / "pdf_exterior_topology.json").is_file())
            self.assertTrue((output / "pdf_exterior_overlay.png").is_file())
            openings = json.loads((output / "pdf_opening_candidates.json").read_text(encoding="utf-8"))
            exterior = json.loads((output / "pdf_exterior_topology.json").read_text(encoding="utf-8"))
            self.assertIn("accepted_openings", openings)
            self.assertIn("ambiguous_openings", openings)
            self.assertIn("unclassified_gaps", openings)
            self.assertIn("bridges", exterior)
            self.assertIn("unresolved", exterior)
            self.assertIn("provenance", exterior)

    def test_model_unavailable_never_confirms_exterior_artifact(self):
        """An unavailable model must publish only explicitly unconfirmed exterior state."""
        from vector_pdf_fusion_pipeline import analyze_vector_pdf_page

        def unavailable_runner(*args, **kwargs):
            raise VectorModelUnavailableError("test model unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            with patch("vector_pdf_fusion_pipeline.convert_from_path", self._stub_render):
                result = analyze_vector_pdf_page(
                    self._make_room_with_door_gap_pdf(root),
                    1,
                    output,
                    model_config=object(),
                    model_runner=unavailable_runner,
                )

            self.assertFalse(result["exterior_topology"]["confirmed"])
            payload = json.loads((output / "pdf_exterior_topology.json").read_text(encoding="utf-8"))
            self.assertFalse(payload["confirmed"])
            self.assertFalse(payload["load_geometry_ready"])
            self.assertEqual(payload["status"], "model_unavailable")

    def test_crop_exterior_and_openings_are_local_with_page_traceability(self):
        """Cropping must not leak page-space geometry into review artifacts."""
        from vector_pdf_fusion_pipeline import analyze_vector_pdf_page

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            crop = [40, 25, 520, 400]
            with patch("vector_pdf_fusion_pipeline.convert_from_path", self._stub_render):
                result = analyze_vector_pdf_page(
                    self._make_room_with_door_gap_pdf(root),
                    1,
                    output,
                    model_config=object(),
                    crop_bbox_page_px=crop,
                    model_runner=self._opening_runner,
                )

            for opening in result["opening_candidates"]["accepted_openings"]:
                for point in (opening["start_px"], opening["end_px"]):
                    self.assertGreaterEqual(point[0], 0)
                    self.assertGreaterEqual(point[1], 0)
                    self.assertLess(point[0], result["page"]["analysis_size_px"][0])
                    self.assertLess(point[1], result["page"]["analysis_size_px"][1])
            self.assertTrue(result["opening_candidates"]["accepted_openings"])
            self.assertEqual(result["exterior_topology"]["provenance"]["coordinate_space"], "crop-local-px")
            self.assertEqual(result["exterior_topology"]["provenance"]["crop_bbox_page_px"], crop)
            self.assertEqual(result["exterior_topology"]["provenance"]["page_number"], 1)

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

    def test_poppler_rounding_difference_uses_actual_render_dimensions(self):
        from vector_pdf_fusion_pipeline import analyze_vector_pdf_page

        observed = {}

        def unavailable_runner(image_path, artifact_parent, config, *, expected_size):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            observed["image_size"] = (image.shape[1], image.shape[0])
            observed["expected_size"] = expected_size
            raise VectorModelUnavailableError("test model unavailable")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf_path = root / "fractional-render-size.pdf"
            pdf = canvas.Canvas(str(pdf_path), pagesize=(238.4, 168.4))
            pdf.setStrokeColorRGB(0, 0, 0)
            pdf.rect(20, 20, 198, 128, stroke=1, fill=0)
            pdf.rect(60, 50, 118, 68, stroke=1, fill=0)
            pdf.save()

            result = analyze_vector_pdf_page(
                pdf_path,
                1,
                root / "output",
                model_config=object(),
                model_runner=unavailable_runner,
            )

        self.assertEqual(result["status"], "partial")
        self.assertEqual(observed["expected_size"], observed["image_size"])
        self.assertEqual(result["page"]["analysis_size_px"], list(observed["image_size"]))

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
