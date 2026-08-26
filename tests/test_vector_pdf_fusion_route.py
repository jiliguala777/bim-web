import hashlib
import json
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
                cv2.imwrite(str(output / "pdf_exterior_overlay.png"), image)
                cv2.imwrite(str(output / "pdf_component_overlay.png"), image)
                topology = {
                    "format": "pdf-exterior-topology/1",
                    "status": "review_required",
                    "confirmed": False,
                    "load_geometry_ready": False,
                    "polygon_px": [[0, 0], [20, 0], [20, 10], [0, 10]],
                    "area_px2": 200.0,
                    "perimeter_px": 60.0,
                    "source_wall_ids": ["wall-1", "wall-2", "wall-3", "wall-4"],
                    "bridge_ids": [],
                    "opening_ids": [],
                    "real_wall_segments": [
                        {"segment_id": "real-wall-1", "orientation": "horizontal", "start_px": [0, 0], "end_px": [20, 0], "length_px": 20.0, "source_wall_ids": ["wall-1"]},
                        {"segment_id": "real-wall-2", "orientation": "vertical", "start_px": [20, 0], "end_px": [20, 10], "length_px": 10.0, "source_wall_ids": ["wall-2"]},
                        {"segment_id": "real-wall-3", "orientation": "horizontal", "start_px": [0, 10], "end_px": [20, 10], "length_px": 20.0, "source_wall_ids": ["wall-3"]},
                        {"segment_id": "real-wall-4", "orientation": "vertical", "start_px": [0, 0], "end_px": [0, 10], "length_px": 10.0, "source_wall_ids": ["wall-4"]},
                    ],
                    "bridges": [],
                    "unresolved_gaps": [],
                    "provenance": {
                        "page_number": 1,
                        "analysis_size_px": [30, 20],
                        "crop_bbox_page_px": None,
                    },
                }
                (output / "pdf_exterior_topology.json").write_text(
                    json.dumps(topology), encoding="utf-8",
                )
                (output / "pdf_opening_candidates.json").write_text(
                    json.dumps({"accepted_openings": []}), encoding="utf-8",
                )
                return {
                    "format": "pdf-vector-fusion/1",
                    "status": "evaluable",
                    "load_geometry_ready": False,
                    "page": {"analysis_size_px": [30, 20]},
                    "summary": {
                        "accepted_wall_count": 4,
                        "rejected_line_count": 2,
                        "uncertain_line_count": 1,
                        "accepted_room_count": 1,
                        "suspicious_room_count": 0,
                    },
                    "reason_codes": [],
                    "exterior_topology": topology,
                    "exterior_summary": {
                        "exterior_wall_count": 4,
                        "accepted_opening_count": 0,
                        "recovered_wall_count": 2,
                        "confirmed_door_arc_count": 1,
                        "pending_door_arc_count": 1,
                        "footprint_status": "review_required",
                    },
                }

            try:
                segmenter = mock.Mock()
                segmenter.predict.side_effect = AssertionError("legacy ONNX must not run")
                with (
                    mock.patch.object(self.server, "HAS_VECTOR_PDF_FUSION", True),
                    mock.patch.object(self.server, "_vector_pdf_fusion_config", object()),
                    mock.patch.object(
                        self.server,
                        "analyze_vector_pdf_page",
                        side_effect=fake_analysis,
                    ) as analyze,
                    mock.patch.object(
                        self.server, "_floorplan_segmenter", segmenter, create=True,
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
            self.assertEqual(payload["report_number"], "FUSION-1")
            self.assertTrue(payload["fusion_debug"])
            self.assertFalse(payload["load_geometry_ready"])
            self.assertEqual(payload["image_size"], [30, 20])
            self.assertEqual(payload["exterior_status"], "review_required")
            self.assertEqual(payload["summary"]["accepted_wall_count"], 4)
            self.assertIn("overlay", payload["images"])
            self.assertIn("exterior_overlay", payload["images"])
            self.assertIn("component_overlay", payload["images"])
            self.assertEqual(payload["exterior_summary"]["exterior_wall_count"], 4)
            self.assertEqual(payload["exterior_summary"]["pending_opening_count"], 0)
            self.assertEqual(payload["exterior_summary"]["recovered_wall_count"], 2)
            self.assertEqual(payload["exterior_summary"]["confirmed_door_arc_count"], 1)
            self.assertEqual(payload["exterior_summary"]["pending_door_arc_count"], 1)
            self.assertIsNone(payload["exterior_summary"]["closure_method"])
            self.assertEqual(payload["exterior_summary"]["inferred_edge_count"], 0)
            self.assertEqual(payload["exterior_summary"]["inferred_length_px"], 0.0)
            self.assertEqual(payload["exterior_summary"]["inferred_perimeter_ratio"], 0.0)
            self.assertEqual(
                payload["topology_sha256"],
                hashlib.sha256(
                    (report_dir / "vector_pdf_fusion" / "pdf_exterior_topology.json").read_bytes()
                ).hexdigest(),
            )
            self.assertEqual(
                payload["artifacts"]["exterior_topology"],
                "vector_pdf_fusion/pdf_exterior_topology.json",
            )
            self.assertEqual(
                payload["artifacts"]["component_overlay"],
                "vector_pdf_fusion/pdf_component_overlay.png",
            )
            self.assertFalse((report_dir / "recognition.json").exists())
            marker = json.loads(
                (report_dir / "exterior_generation.json").read_text(encoding="utf-8")
            )
            self.assertEqual(marker["report_number"], "FUSION-1")
            self.assertEqual(marker["status"], "ready")
            self.assertEqual(marker["topology_sha256"], payload["topology_sha256"])
            self.assertNotIn("generation", payload)
            analyze.assert_called_once()
            segmenter.predict.assert_not_called()

    def test_failed_new_fusion_generation_invalidates_old_confirmed_exterior(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            report_dir = upload_root / "energy" / "FUSION-RACE"
            report_dir.mkdir(parents=True)
            prepared = report_dir / "prepared.pdf"
            prepared.write_bytes(b"pdf")
            topology_hash_a = "a" * 64
            recognition = {
                "schema_version": 1,
                "report_number": "FUSION-RACE",
                "model": {"version": "generation-a"},
                "preprocessing": {"requested": "vector_pdf_fusion", "use_preprocessing": False},
                "image_size": [100, 100],
                "geometry": {"walls": [], "windows": [], "doors": []},
                "room_topology": {
                    "status": "exterior_only", "room_count": 0, "rooms": [],
                    "total_area_px2": 0.0, "load_geometry_ready": False,
                },
                "exterior_topology": {
                    "confirmed": True, "load_geometry_ready": True,
                    "area_m2": 100.0, "perimeter_m": 40.0,
                },
                "openings": [],
                "exterior_generation": {
                    "generation": "generation-a",
                    "topology_sha256": topology_hash_a,
                },
            }
            (report_dir / "recognition.json").write_text(
                json.dumps(recognition), encoding="utf-8",
            )
            marker_path = report_dir / "exterior_generation.json"
            marker_path.write_text(json.dumps({
                "format": "pdf-exterior-generation/1",
                "report_number": "FUSION-RACE",
                "generation": "generation-a",
                "status": "ready",
                "topology_sha256": topology_hash_a,
            }), encoding="utf-8")
            token = self.server._make_pdf_upload_token(
                "FUSION-RACE", "prepared.pdf", 1,
            )

            def fail_after_running_marker(*args, **kwargs):
                marker = json.loads(marker_path.read_text("utf-8"))
                self.assertEqual(marker["status"], "running")
                self.assertNotEqual(marker["generation"], "generation-a")
                self.assertIsNone(marker["topology_sha256"])
                raise RuntimeError("simulated fusion failure")

            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
            try:
                with (
                    mock.patch.object(self.server, "HAS_VECTOR_PDF_FUSION", True),
                    mock.patch.object(self.server, "_vector_pdf_fusion_config", object()),
                    mock.patch.object(
                        self.server, "analyze_vector_pdf_page",
                        side_effect=fail_after_running_marker,
                    ),
                ):
                    fusion_response = self.client.post(
                        "/energy/vector_pdf_fusion",
                        data={
                            "report_number": "FUSION-RACE",
                            "pdf_upload_token": token,
                            "pdf_page_number": "1",
                            "recognition_mode": "full_page",
                        },
                    )

                calculation = {
                    "success": True,
                    "summary": {"total_energy_kwh": 0, "eui": 0, "rating": "A", "rating_label": "test"},
                }
                with (
                    mock.patch.object(self.server, "HAS_FLOORPLAN_AI", True),
                    mock.patch.object(self.server, "HAS_ENERGY_CALC", True),
                    mock.patch.object(self.server, "HAS_DESIGN_LOAD_CALC", False),
                    mock.patch.object(
                        self.server.energy_calc, "calculate_energy",
                        return_value=calculation,
                    ) as calculate,
                ):
                    energy_response = self.client.post(
                        "/energy/ai_simulate",
                        json={"report_number": "FUSION-RACE", "height": 3.0, "floors": 1},
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(fusion_response.status_code, 500, fusion_response.get_json())
            failed_marker = json.loads(marker_path.read_text("utf-8"))
            self.assertEqual(failed_marker["status"], "failed")
            self.assertNotEqual(failed_marker["generation"], "generation-a")
            self.assertEqual(energy_response.status_code, 409, energy_response.get_json())
            self.assertIn("generation", energy_response.get_json()["error"].lower())
            calculate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
