from __future__ import annotations

import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from vector_platform import VECTOR_BACKEND, VectorPlatformAdapter, VectorPlatformConfig, adapt_vector_prediction


class VectorPlatformTests(unittest.TestCase):
    def setUp(self):
        self.geometry = {
            "target": {"width": 20, "height": 10},
            "footprints": [{"id": "footprint-1", "points": [[1, 1], [18, 1], [18, 8], [1, 8]]}],
            "rooms": [{"id": "room-1", "points": [[2, 2], [10, 2], [10, 7], [2, 7]]}],
            "walls": [{"id": "wall-1", "points": [[1, 1], [18, 1]]}],
            "openings": [
                {"id": "opening-1", "opening_type": "door", "points": [[3, 1], [5, 1]]},
                {"id": "opening-2", "opening_type": "window", "points": [[12, 1], [16, 1]]},
            ],
        }
        self.image = np.zeros((10, 20, 3), dtype=np.uint8)

    def test_adapter_preserves_source_coordinates_and_calibrated_room_topology(self):
        result = adapt_vector_prediction(
            self.geometry,
            {"scale_m_per_px": 0.1, "candidates": [{"dimension_id": "d1"}]},
            {"checkpoint_sha256": "a" * 64}, self.image, self.image, Path("artifacts"),
        )

        self.assertEqual(result["backend"], VECTOR_BACKEND)
        self.assertEqual(result["image_size"], [20, 10])
        self.assertEqual(result["geometry"]["walls"][0]["pts"], [[1.0, 1.0], [18.0, 1.0]])
        self.assertEqual(result["stats"], {"footprints": 1, "rooms": 1, "walls": 1, "windows": 1, "doors": 1})
        self.assertEqual(result["room_topology"]["room_count"], 1)
        self.assertAlmostEqual(result["room_topology"]["total_area_m2"], 0.4)
        self.assertTrue(result["room_topology"]["load_geometry_ready"])
        self.assertEqual(result["scale_calibration"]["method"], "vector_dimensions")
        self.assertEqual(result["mask"].shape, (10, 20))

    def test_adapter_requires_external_predictor_artifacts_and_uses_array_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / "training"
            training.mkdir()
            python = root / "python.exe"
            python.write_bytes(b"")
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            image = root / "plan.png"
            cv2.imwrite(str(image), self.image)
            config = VectorPlatformConfig(training, python, checkpoint, device="cpu")
            adapter = VectorPlatformAdapter(config)

            def fake_run(command, **kwargs):
                output = Path(command[command.index("--output") + 1])
                output.mkdir()
                (output / "prediction.geometry.json").write_text(json.dumps(self.geometry), encoding="utf-8")
                (output / "measurements.json").write_text(json.dumps({"scale_m_per_px": None}), encoding="utf-8")
                (output / "inference.json").write_text(json.dumps({"checkpoint_sha256": "b" * 64}), encoding="utf-8")
                cv2.imwrite(str(output / "overlay.png"), self.image)
                return mock.Mock(returncode=0, stdout="{}", stderr="")

            with mock.patch("vector_platform.subprocess.run", side_effect=fake_run) as run:
                result = adapter.predict(image, root / "artifacts")

            command = run.call_args.args[0]
            self.assertEqual(command[:3], [str(python.resolve()), "-m", "training.predict_vector"])
            self.assertIn("--checkpoint", command)
            self.assertIn("--device", command)
            self.assertEqual(result["geometry"]["doors"][0]["length_px"], 2.0)
            self.assertFalse(result["room_topology"]["load_geometry_ready"])


class VectorPlatformRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="vector-platform-test")

    def setUp(self):
        self.client = self.server.app.test_client()
        with self.client.session_transaction() as session:
            session["logged_in"] = True

    def test_vector_backend_persists_a_recognition_payload_consumable_by_energy_route(self):
        source = np.zeros((10, 20, 3), dtype=np.uint8)
        result = adapt_vector_prediction(
            {
                "target": {"width": 20, "height": 10}, "footprints": [],
                "rooms": [{"id": "room-1", "points": [[1, 1], [11, 1], [11, 6], [1, 6]]}],
                "walls": [{"id": "wall-1", "points": [[1, 1], [18, 1]]}],
                "openings": [],
            },
            {"scale_m_per_px": 0.1, "candidates": []},
            {"checkpoint_sha256": "c" * 64}, source, source, Path("ignored"),
        )
        adapter = mock.Mock()
        adapter.predict.return_value = result
        with tempfile.TemporaryDirectory() as temporary:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = temporary
            try:
                with (
                    mock.patch.object(self.server, "HAS_VECTOR_FLOORPLAN_AI", True),
                    mock.patch.object(self.server, "_vector_platform_adapter", adapter),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "VECTOR-1", "model_backend": VECTOR_BACKEND,
                            "raster_file": (io.BytesIO(b"plan"), "plan.png"),
                        },
                        content_type="multipart/form-data",
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            payload = response.get_json()
            self.assertEqual(payload["model_info"]["backend"], VECTOR_BACKEND)
            self.assertTrue(payload["room_topology"]["load_geometry_ready"])
            recognition = json.loads(
                (Path(temporary) / "energy" / "VECTOR-1" / "recognition.json").read_text(encoding="utf-8")
            )
            self.assertEqual(recognition["model"]["backend"], VECTOR_BACKEND)
            self.assertIsNotNone(self.server._load_recognition_payload(Path(temporary) / "energy" / "VECTOR-1"))

    def test_vector_backend_crops_a_raster_upload_before_prediction(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        adapter = mock.Mock()
        adapter.predict.return_value = adapt_vector_prediction(
            {"target": {"width": 200, "height": 200}, "footprints": [], "rooms": [], "walls": [], "openings": []},
            {"scale_m_per_px": None, "candidates": []},
            {"checkpoint_sha256": "d" * 64},
            np.zeros((200, 200, 3), dtype=np.uint8),
            np.zeros((200, 200, 3), dtype=np.uint8), Path("ignored"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = temporary
            try:
                with (
                    mock.patch.object(self.server, "HAS_VECTOR_FLOORPLAN_AI", True),
                    mock.patch.object(self.server, "_vector_platform_adapter", adapter),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "VECTOR-CROP", "model_backend": VECTOR_BACKEND,
                            "recognition_mode": "crop_region",
                            "crop_bbox_px": "[100, 100, 300, 300]",
                            "crop_preview_size": "[400, 400]",
                            "raster_file": (io.BytesIO(encoded.tobytes()), "plan.png"),
                        }, content_type="multipart/form-data",
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload
            self.assertEqual(response.status_code, 200, response.get_json())
            input_path = adapter.predict.call_args.args[0]
            self.assertEqual(cv2.imread(str(input_path)).shape[:2], (200, 200))
            self.assertEqual(response.get_json()["recognition_mode"], "crop_region")

    def test_ai_status_reports_vector_availability_when_legacy_onnx_is_absent(self):
        config = mock.Mock()
        config.checkpoint_path = Path("vector-best.pt")
        with (
            mock.patch.object(self.server, "HAS_FLOORPLAN_AI", False),
            mock.patch.object(self.server, "HAS_VECTOR_FLOORPLAN_AI", True),
            mock.patch.object(self.server, "_vector_platform_config", config),
        ):
            response = self.client.get("/energy/ai_status")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["model"], "vector-resnet34-unet")
        self.assertTrue(payload["backends"][VECTOR_BACKEND]["available"])


if __name__ == "__main__":
    unittest.main()
