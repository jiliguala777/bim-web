import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class VectorPdfExteriorConfirmRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_server_server

        cls.server = web_server_server
        cls.server.app.config.update(TESTING=True, SECRET_KEY="vector-pdf-exterior-test")

    def setUp(self):
        self.client = self.server.app.test_client()
        with self.client.session_transaction() as session:
            session["logged_in"] = True

    @staticmethod
    def _artifacts(report_dir):
        provenance = {
            "coordinate_space": "crop-local-px",
            "page_number": 2,
            "page_size_pt": [300.0, 200.0],
            "analysis_size_px": [100, 50],
            "crop_bbox_page_px": [10, 20, 110, 70],
            "building_roi_px": [0, 0, 100, 50],
        }
        topology = {
            "format": "pdf-exterior-topology/1",
            "status": "review_required",
            "confirmed": False,
            "load_geometry_ready": False,
            "polygon_px": [[0, 0], [100, 0], [100, 50], [0, 50]],
            "area_px2": 5000.0,
            "perimeter_px": 300.0,
            "area_m2": None,
            "perimeter_m": None,
            "source_wall_ids": ["line-1", "line-2", "line-3", "line-4"],
            "bridge_ids": ["bridge-door", "bridge-repair"],
            "opening_ids": ["opening-door"],
            "real_wall_segments": [
                {"segment_id": "real-wall-1", "orientation": "horizontal", "start_px": [0, 0], "end_px": [20, 0], "length_px": 20.0, "source_wall_ids": ["line-1"]},
                {"segment_id": "real-wall-2", "orientation": "horizontal", "start_px": [50, 0], "end_px": [100, 0], "length_px": 50.0, "source_wall_ids": ["line-1"]},
                {"segment_id": "real-wall-3", "orientation": "vertical", "start_px": [100, 0], "end_px": [100, 10], "length_px": 10.0, "source_wall_ids": ["line-2"]},
                {"segment_id": "real-wall-4", "orientation": "vertical", "start_px": [100, 15], "end_px": [100, 50], "length_px": 35.0, "source_wall_ids": ["line-2"]},
                {"segment_id": "real-wall-5", "orientation": "horizontal", "start_px": [0, 50], "end_px": [100, 50], "length_px": 100.0, "source_wall_ids": ["line-3"]},
                {"segment_id": "real-wall-6", "orientation": "vertical", "start_px": [0, 0], "end_px": [0, 50], "length_px": 50.0, "source_wall_ids": ["line-4"]},
            ],
            "bridges": [
                {
                    "bridge_id": "bridge-door",
                    "bridge_type": "opening_bridge",
                    "opening_id": "opening-door",
                    "orientation": "horizontal",
                    "start_px": [20, 0],
                    "end_px": [50, 0],
                    "width_px": 30.0,
                    "host_wall_ids": ["line-1", "line-2"],
                },
                {
                    "bridge_id": "bridge-repair",
                    "bridge_type": "small_gap_repair",
                    "opening_id": None,
                    "orientation": "vertical",
                    "start_px": [100, 10],
                    "end_px": [100, 15],
                    "width_px": 5.0,
                    "host_wall_ids": ["line-2"],
                },
            ],
            "unresolved_gaps": [],
            "unresolved": [],
            "provenance": json.loads(json.dumps(provenance)),
        }
        openings = {
            "format": "pdf-opening-candidates/1",
            "status": "review_required",
            "confirmed": False,
            "load_geometry_ready": False,
            "provenance": provenance,
            "gaps": [],
            "accepted_openings": [{
                "opening_id": "opening-door",
                "kind": "door",
                "orientation": "horizontal",
                "start_px": [20, 0],
                "end_px": [50, 0],
                "width_px": 30.0,
                "width_m": None,
                "host_wall_ids": ["line-1", "line-2"],
                "exterior": True,
                "confidence": 0.9,
            }],
            "ambiguous_openings": [],
            "unclassified_gaps": [],
        }
        artifact_dir = report_dir / "vector_pdf_fusion"
        artifact_dir.mkdir(parents=True)
        topology_path = artifact_dir / "pdf_exterior_topology.json"
        topology_path.write_text(json.dumps(topology), encoding="utf-8")
        (artifact_dir / "pdf_opening_candidates.json").write_text(
            json.dumps(openings), encoding="utf-8",
        )
        return topology, openings, topology_path

    def _post_artifacts(self, upload_root, topology, openings, topology_path):
        topology_path.write_text(json.dumps(topology), encoding="utf-8")
        (topology_path.parent / "pdf_opening_candidates.json").write_text(
            json.dumps(openings), encoding="utf-8",
        )
        previous = self.server.app.config["UPLOAD_FOLDER"]
        self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
        try:
            return self.client.post(
                "/energy/vector_pdf_exterior_confirm",
                json=self._payload(topology_path),
            )
        finally:
            self.server.app.config["UPLOAD_FOLDER"] = previous

    @staticmethod
    def _payload(topology_path, **overrides):
        payload = {
            "report_number": "EXT-1",
            "topology_sha256": hashlib.sha256(topology_path.read_bytes()).hexdigest(),
            "scale_m_per_px": 0.02,
            "confirmed": True,
            "page_number": 2,
            "crop_bbox_page_px": [10, 20, 110, 70],
            "polygon_px": [[999, 999], [1000, 999], [1000, 1000]],
            "area_px2": 1,
            "opening_widths": {"door_total_width_m": 999},
        }
        payload.update(overrides)
        return payload

    def test_confirm_reloads_hashed_artifacts_and_writes_ready_recognition(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            report_dir = upload_root / "energy" / "EXT-1"
            _, _, topology_path = self._artifacts(report_dir)
            previous = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
            segmenter = mock.Mock()
            segmenter.predict.side_effect = AssertionError("legacy ONNX must not run")
            try:
                with mock.patch.object(
                    self.server, "_floorplan_segmenter", segmenter, create=True,
                ):
                    response = self.client.post(
                        "/energy/vector_pdf_exterior_confirm",
                        json=self._payload(topology_path),
                    )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous

            self.assertEqual(response.status_code, 200, response.get_json())
            saved = json.loads((report_dir / "recognition.json").read_text("utf-8"))
            self.assertTrue(saved["exterior_topology"]["load_geometry_ready"])
            self.assertTrue(saved["exterior_topology"]["confirmed"])
            self.assertEqual(saved["opening_widths"]["door_total_width_m"], 0.6)
            self.assertEqual(saved["image_size"], [100, 50])
            self.assertEqual(saved["pdf_page_number"], 2)
            self.assertEqual(saved["crop_bbox_page_px"], [10, 20, 110, 70])
            self.assertFalse(saved["room_topology"]["load_geometry_ready"])
            self.assertEqual(saved["geometry"]["doors"][0]["pts"], [[20, 0], [50, 0]])
            wall_segments = [wall["pts"] for wall in saved["geometry"]["walls"]]
            self.assertNotIn([[20, 0], [50, 0]], wall_segments)
            self.assertNotIn([[100, 10], [100, 15]], wall_segments)
            self.assertEqual(
                [wall["id"] for wall in saved["geometry"]["walls"]],
                [f"real-wall-{index}" for index in range(1, 7)],
            )
            self.assertEqual(
                sum(wall["length_px"] for wall in saved["geometry"]["walls"]), 265.0,
            )
            wall_intervals = [
                (wall["pts"][0], wall["pts"][1]) for wall in saved["geometry"]["walls"]
            ]
            for bridge in saved["exterior_topology"]["bridges"]:
                if bridge["bridge_id"] not in saved["exterior_topology"]["bridge_ids"]:
                    continue
                bx0, by0 = bridge["start_px"]
                bx1, by1 = bridge["end_px"]
                for (wx0, wy0), (wx1, wy1) in wall_intervals:
                    if by0 == by1 == wy0 == wy1:
                        self.assertLessEqual(
                            min(max(bx0, bx1), max(wx0, wx1)),
                            max(min(bx0, bx1), min(wx0, wx1)),
                        )
                    if bx0 == bx1 == wx0 == wx1:
                        self.assertLessEqual(
                            min(max(by0, by1), max(wy0, wy1)),
                            max(min(by0, by1), min(wy0, wy1)),
                        )
            self.assertNotEqual(saved["exterior_topology"]["polygon_px"], [[999, 999], [1000, 999], [1000, 1000]])
            segmenter.predict.assert_not_called()

    def test_confirm_rejects_hash_mismatch_without_overwriting_recognition(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            report_dir = upload_root / "energy" / "EXT-1"
            _, _, topology_path = self._artifacts(report_dir)
            recognition_path = report_dir / "recognition.json"
            recognition_path.write_bytes(b'{"sentinel": true}\n')
            before = recognition_path.read_bytes()
            previous = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
            try:
                response = self.client.post(
                    "/energy/vector_pdf_exterior_confirm",
                    json=self._payload(topology_path, topology_sha256="0" * 64),
                )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous

            self.assertEqual(response.status_code, 409, response.get_json())
            self.assertEqual(recognition_path.read_bytes(), before)

    def test_confirm_rejects_non_unique_or_unreferenced_opening_ids(self):
        cases = ("duplicate topology opening id", "extra accepted opening")
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                upload_root = Path(directory)
                report_dir = upload_root / "energy" / "EXT-1"
                topology, openings, topology_path = self._artifacts(report_dir)
                if label == "duplicate topology opening id":
                    topology["opening_ids"].append("opening-door")
                else:
                    extra = dict(openings["accepted_openings"][0])
                    extra["opening_id"] = "opening-extra"
                    openings["accepted_openings"].append(extra)
                recognition_path = report_dir / "recognition.json"
                recognition_path.write_bytes(b"preserve")

                response = self._post_artifacts(
                    upload_root, topology, openings, topology_path,
                )

                self.assertEqual(response.status_code, 409, response.get_json())
                self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_rejects_opening_bridge_geometry_or_host_mismatch(self):
        cases = ("endpoint mismatch", "host mismatch", "empty hosts")
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                upload_root = Path(directory)
                report_dir = upload_root / "energy" / "EXT-1"
                topology, openings, topology_path = self._artifacts(report_dir)
                if label == "endpoint mismatch":
                    openings["accepted_openings"][0]["start_px"] = [21, 0]
                    openings["accepted_openings"][0]["width_px"] = 29.0
                elif label == "host mismatch":
                    openings["accepted_openings"][0]["host_wall_ids"] = ["line-1"]
                else:
                    topology["bridges"][0]["host_wall_ids"] = []
                    openings["accepted_openings"][0]["host_wall_ids"] = []
                recognition_path = report_dir / "recognition.json"
                recognition_path.write_bytes(b"preserve")

                response = self._post_artifacts(
                    upload_root, topology, openings, topology_path,
                )

                self.assertEqual(response.status_code, 409, response.get_json())
                self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_rejects_self_intersection_forged_metrics_or_out_of_bounds_polygon(self):
        cases = ("self intersection", "forged metrics", "out of bounds")
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                upload_root = Path(directory)
                report_dir = upload_root / "energy" / "EXT-1"
                topology, openings, topology_path = self._artifacts(report_dir)
                if label == "self intersection":
                    topology["polygon_px"] = [
                        [10, 10], [90, 10], [90, 40], [40, 40],
                        [40, 20], [70, 20], [70, 50], [10, 50],
                    ]
                    topology["area_px2"] = 3600.0
                    topology["perimeter_px"] = 340.0
                elif label == "forged metrics":
                    topology["area_px2"] = 4999.0
                    topology["perimeter_px"] = 299.0
                else:
                    topology["polygon_px"][1] = [101, 0]
                    topology["polygon_px"][2] = [101, 50]
                    topology["area_px2"] = 5050.0
                    topology["perimeter_px"] = 302.0
                recognition_path = report_dir / "recognition.json"
                recognition_path.write_bytes(b"preserve")

                response = self._post_artifacts(
                    upload_root, topology, openings, topology_path,
                )

                self.assertEqual(response.status_code, 409, response.get_json())
                self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_persists_recomputed_polygon_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            report_dir = upload_root / "energy" / "EXT-1"
            topology, openings, topology_path = self._artifacts(report_dir)
            topology["area_px2"] = math.nextafter(5000.0, math.inf)
            topology["perimeter_px"] = math.nextafter(300.0, math.inf)

            response = self._post_artifacts(
                upload_root, topology, openings, topology_path,
            )

            self.assertEqual(response.status_code, 200, response.get_json())
            saved = json.loads((report_dir / "recognition.json").read_text("utf-8"))
            self.assertEqual(saved["exterior_topology"]["area_px2"], 5000.0)
            self.assertEqual(saved["exterior_topology"]["perimeter_px"], 300.0)

    def test_confirm_rejects_missing_duplicate_or_unknown_selected_bridge(self):
        cases = (
            "missing bridge", "duplicate bridge", "unknown type", "unselected extra bridge",
        )
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                upload_root = Path(directory)
                report_dir = upload_root / "energy" / "EXT-1"
                topology, openings, topology_path = self._artifacts(report_dir)
                if label == "missing bridge":
                    topology["bridges"].pop()
                elif label == "duplicate bridge":
                    topology["bridges"].append(dict(topology["bridges"][1]))
                elif label == "unknown type":
                    topology["bridges"][1]["bridge_type"] = "unknown_repair"
                else:
                    extra = dict(topology["bridges"][1])
                    extra.update(
                        bridge_id="bridge-extra", start_px=[100, 20], end_px=[100, 21],
                        width_px=1.0,
                    )
                    topology["bridges"].append(extra)
                recognition_path = report_dir / "recognition.json"
                recognition_path.write_bytes(b"preserve")

                response = self._post_artifacts(
                    upload_root, topology, openings, topology_path,
                )

                self.assertEqual(response.status_code, 409, response.get_json())
                self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_rejects_bridge_outside_edge_or_real_wall_coverage_error(self):
        cases = (
            "bridge partial overlap", "real wall gap", "real wall bridge overlap",
            "boolean real wall endpoint",
        )
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                upload_root = Path(directory)
                report_dir = upload_root / "energy" / "EXT-1"
                topology, openings, topology_path = self._artifacts(report_dir)
                if label == "bridge partial overlap":
                    repair = topology["bridges"][1]
                    repair["start_px"], repair["end_px"] = [100, 48], [100, 53]
                elif label == "real wall gap":
                    wall = topology["real_wall_segments"][0]
                    wall["end_px"], wall["length_px"] = [19, 0], 19.0
                elif label == "real wall bridge overlap":
                    wall = topology["real_wall_segments"][0]
                    wall["end_px"], wall["length_px"] = [25, 0], 25.0
                else:
                    topology["real_wall_segments"][0]["start_px"] = [False, 0]
                recognition_path = report_dir / "recognition.json"
                recognition_path.write_bytes(b"preserve")

                response = self._post_artifacts(
                    upload_root, topology, openings, topology_path,
                )

                self.assertEqual(response.status_code, 409, response.get_json())
                self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_rejects_invalid_or_loosely_equal_artifact_provenance(self):
        cases = (
            "mixed numeric types", "coordinate crop mismatch", "invalid page size",
            "roi out of bounds", "boolean roi",
        )
        for label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                upload_root = Path(directory)
                report_dir = upload_root / "energy" / "EXT-1"
                topology, openings, topology_path = self._artifacts(report_dir)
                if label == "mixed numeric types":
                    openings["provenance"]["analysis_size_px"] = [100.0, 50.0]
                elif label == "coordinate crop mismatch":
                    topology["provenance"]["coordinate_space"] = "page-local-px"
                    openings["provenance"]["coordinate_space"] = "page-local-px"
                elif label == "invalid page size":
                    topology["provenance"]["page_size_pt"] = [0.0, 200.0]
                    openings["provenance"]["page_size_pt"] = [0.0, 200.0]
                elif label == "roi out of bounds":
                    topology["provenance"]["building_roi_px"] = [0, 0, 101, 50]
                    openings["provenance"]["building_roi_px"] = [0, 0, 101, 50]
                else:
                    topology["provenance"]["building_roi_px"] = [False, 0, 100, 50]
                    openings["provenance"]["building_roi_px"] = [False, 0, 100, 50]
                recognition_path = report_dir / "recognition.json"
                recognition_path.write_bytes(b"preserve")

                response = self._post_artifacts(
                    upload_root, topology, openings, topology_path,
                )

                self.assertEqual(response.status_code, 409, response.get_json())
                self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_rejects_invalid_readiness_inputs_without_overwrite(self):
        cases = [
            ("missing scale", {"scale_m_per_px": None}, 400),
            ("non-finite scale", {"scale_m_per_px": float("inf")}, 400),
            ("truthy confirmed", {"confirmed": 1}, 400),
            ("stale page", {"page_number": 1}, 409),
            ("stale crop", {"crop_bbox_page_px": None}, 409),
        ]
        for label, overrides, expected_status in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                upload_root = Path(directory)
                report_dir = upload_root / "energy" / "EXT-1"
                _, _, topology_path = self._artifacts(report_dir)
                recognition_path = report_dir / "recognition.json"
                recognition_path.write_bytes(b"preserve")
                previous = self.server.app.config["UPLOAD_FOLDER"]
                self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
                try:
                    response = self.client.post(
                        "/energy/vector_pdf_exterior_confirm",
                        json=self._payload(topology_path, **overrides),
                    )
                finally:
                    self.server.app.config["UPLOAD_FOLDER"] = previous
                self.assertEqual(response.status_code, expected_status, response.get_json())
                self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_rejects_nonclosed_topology_and_metric_repair_violation(self):
        cases = [("nonclosed", 0.02), ("repair too long", 0.13)]
        for label, scale in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                upload_root = Path(directory)
                report_dir = upload_root / "energy" / "EXT-1"
                topology, _, topology_path = self._artifacts(report_dir)
                if label == "nonclosed":
                    topology.update(status="exterior_not_closed", polygon_px=[])
                    topology_path.write_text(json.dumps(topology), encoding="utf-8")
                recognition_path = report_dir / "recognition.json"
                recognition_path.write_bytes(b"preserve")
                previous = self.server.app.config["UPLOAD_FOLDER"]
                self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
                try:
                    response = self.client.post(
                        "/energy/vector_pdf_exterior_confirm",
                        json=self._payload(topology_path, scale_m_per_px=scale),
                    )
                finally:
                    self.server.app.config["UPLOAD_FOLDER"] = previous
                self.assertEqual(response.status_code, 409, response.get_json())
                self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_rejects_mismatched_report_path(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            report_dir = upload_root / "energy" / "EXT-1"
            _, _, topology_path = self._artifacts(report_dir)
            recognition_path = report_dir / "recognition.json"
            recognition_path.write_bytes(b"preserve")
            previous = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
            try:
                response = self.client.post(
                    "/energy/vector_pdf_exterior_confirm",
                    json=self._payload(topology_path, report_number="../EXT-1"),
                )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous
            self.assertEqual(response.status_code, 400, response.get_json())
            self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_requires_explicit_null_crop_for_full_page(self):
        with tempfile.TemporaryDirectory() as directory:
            upload_root = Path(directory)
            report_dir = upload_root / "energy" / "EXT-1"
            topology, openings, topology_path = self._artifacts(report_dir)
            topology["provenance"]["crop_bbox_page_px"] = None
            openings["provenance"]["crop_bbox_page_px"] = None
            topology_path.write_text(json.dumps(topology), encoding="utf-8")
            (report_dir / "vector_pdf_fusion" / "pdf_opening_candidates.json").write_text(
                json.dumps(openings), encoding="utf-8",
            )
            recognition_path = report_dir / "recognition.json"
            recognition_path.write_bytes(b"preserve")
            payload = self._payload(topology_path)
            payload.pop("crop_bbox_page_px")
            previous = self.server.app.config["UPLOAD_FOLDER"]
            self.server.app.config["UPLOAD_FOLDER"] = str(upload_root)
            try:
                response = self.client.post(
                    "/energy/vector_pdf_exterior_confirm", json=payload,
                )
            finally:
                self.server.app.config["UPLOAD_FOLDER"] = previous
            self.assertEqual(response.status_code, 400, response.get_json())
            self.assertEqual(recognition_path.read_bytes(), b"preserve")

    def test_confirm_requires_login(self):
        client = self.server.app.test_client()
        response = client.post("/energy/vector_pdf_exterior_confirm", json={})
        self.assertEqual(response.status_code, 302)


if __name__ == "__main__":
    unittest.main()
