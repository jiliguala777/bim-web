from __future__ import annotations

import unittest
import io
import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from energy_report_storage import user_storage_key
from floorplan_image_onnx import prepare_image_tensor, remap_element_classes, footprint_mask_from_logits


class FloorplanImageOnnxTests(unittest.TestCase):
    def test_prepare_image_tensor_uses_training_letterbox_without_imagenet_normalization(self):
        image = np.full((100, 200, 3), 255, dtype=np.uint8)

        tensor, metadata = prepare_image_tensor(image, size=4)

        self.assertEqual(tensor.shape, (1, 3, 4, 4))
        self.assertEqual(metadata["resized_size"], [4, 2])
        self.assertEqual(metadata["padding"], [1, 0, 1, 0])
        self.assertEqual(tensor[0, :, 0, 0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(tensor[0, :, 1, 0].tolist(), [1.0, 1.0, 1.0])

    def test_remap_element_classes_converts_training_door_window_order_to_platform_order(self):
        training_mask = np.array([[0, 1, 2, 3]], dtype=np.uint8)

        platform_mask = remap_element_classes(training_mask)

        self.assertEqual(platform_mask.tolist(), [[0, 1, 3, 2]])

    def test_footprint_logits_keep_the_largest_outer_footprint_and_fill_its_holes(self):
        logits = np.full((1, 1, 5, 5), -2, dtype=np.float32)
        logits[0, 0, 1:4, 1] = 2
        logits[0, 0, 1:4, 3] = 2
        logits[0, 0, 1, 1:4] = 2
        logits[0, 0, 3, 1:4] = 2

        self.assertEqual(footprint_mask_from_logits(logits, (5, 5)).tolist(), [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ])


class FloorplanImageOnnxRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="image-onnx-test")

    def setUp(self):
        self.database_directory = tempfile.TemporaryDirectory()
        self.previous_database_path = self.server.DB_PATH
        self.server.DB_PATH = str(Path(self.database_directory.name) / "users.db")
        self.server.init_db()
        self.client = self.server.app.test_client()
        with self.client.session_transaction() as session:
            session.update(logged_in=True, username="test-user", is_admin=False)

    def tearDown(self):
        self.server.DB_PATH = self.previous_database_path
        self.database_directory.cleanup()

    def test_image_onnx_backend_persists_a_platform_recognition_payload(self):
        source = np.zeros((10, 20, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", source)
        self.assertTrue(ok)
        segmenter = mock.Mock()
        segmenter.predict.return_value = {
            "mask": np.zeros((10, 20), dtype=np.uint8),
            "raw_model_mask": np.zeros((10, 20), dtype=np.uint8),
            "overlay": source,
            "stats": {},
            "geometry": {"walls": [], "windows": [], "doors": []},
            "room_topology": {"status": "no_closed_rooms", "room_count": 0, "rooms": [], "total_area_px2": 0.0, "total_area_m2": None, "load_geometry_ready": False},
            "topology_repair": {},
            "footprint_mask": np.ones((10, 20), dtype=np.uint8),
        }
        with tempfile.TemporaryDirectory() as upload_directory:
            previous_upload = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = upload_directory
            try:
                with (
                    mock.patch.object(self.server, "HAS_IMAGE_FLOORPLAN_AI", True, create=True),
                    mock.patch.object(self.server, "_floorplan_image_segmenter", segmenter, create=True),
                ):
                    response = self.client.post(
                        "/energy/ai_recognize",
                        data={
                            "report_number": "IMAGE-ONNX-1", "model_backend": "image_onnx",
                            "raster_file": (io.BytesIO(encoded.tobytes()), "plan.png"),
                        }, content_type="multipart/form-data",
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous_upload

            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertEqual(response.get_json()["model_info"]["backend"], "image_onnx")
            self.assertEqual(response.get_json()["footprint"]["pixels"], 200)
            report_dir = Path(upload_directory) / "energy" / user_storage_key("test-user") / "IMAGE-ONNX-1"
            self.assertIsNotNone(self.server._load_recognition_payload(report_dir))


if __name__ == "__main__":
    unittest.main()
