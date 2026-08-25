import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


class VectorPdfFusionRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="vector-pdf-fusion-test")

    def setUp(self):
        self.client = self.server.app.test_client()
        with self.client.session_transaction() as session:
            session["logged_in"] = True

    def test_route_uses_new_pipeline_without_writing_recognition_or_calling_onnx(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            report_dir = upload_root / "energy" / "FUSION-1"
            report_dir.mkdir(parents=True)
            prepared = report_dir / "prepared.pdf"
            prepared.write_bytes(b"pdf")
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
            token = self.server._make_pdf_upload_token("FUSION-1", "prepared.pdf", 1)

            def fake_analysis(pdf_path, page_number, output_dir, model_config, crop_bbox_page_px=None):
                output = Path(output_dir)
                output.mkdir(parents=True, exist_ok=True)
                image = np.full((20, 30, 3), 255, dtype=np.uint8)
                cv2.imwrite(str(output / "pdf_vector_model_input.png"), image)
                cv2.imwrite(str(output / "pdf_vector_fusion_overlay.png"), image)
                return {
                    "format": "pdf-vector-fusion/1",
                    "status": "evaluable",
                    "load_geometry_ready": False,
                    "summary": {
                        "accepted_wall_count": 4,
                        "rejected_line_count": 2,
                        "uncertain_line_count": 1,
                        "accepted_room_count": 1,
                        "suspicious_room_count": 0,
                    },
                    "reason_codes": [],
                }

            try:
                with (
                    mock.patch.object(self.server, "HAS_VECTOR_PDF_FUSION", True),
                    mock.patch.object(self.server, "_vector_pdf_fusion_config", object()),
                    mock.patch.object(
                        self.server,
                        "analyze_vector_pdf_page",
                        side_effect=fake_analysis,
                    ) as analyze,
                    mock.patch.object(
                        self.server._floorplan_segmenter,
                        "predict",
                        side_effect=AssertionError("legacy ONNX must not run"),
                    ),
                ):
                    response = self.client.post(
                        "/energy/vector_pdf_fusion",
                        data={
                            "report_number": "FUSION-1",
                            "pdf_upload_token": token,
                            "pdf_page_number": "1",
                            "recognition_mode": "full_page",
                        },
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            payload = response.get_json()
            self.assertTrue(payload["fusion_debug"])
            self.assertFalse(payload["load_geometry_ready"])
            self.assertEqual(payload["summary"]["accepted_wall_count"], 4)
            self.assertIn("overlay", payload["images"])
            self.assertFalse((report_dir / "recognition.json").exists())
            analyze.assert_called_once()


if __name__ == "__main__":
    unittest.main()
