import unittest
from unittest.mock import patch

import cv2
import numpy as np

from floorplan_rooms import apply_scale_to_room_topology, extract_room_topology
from floorplan_onnx import FloorplanSegmenterONNX


class FloorplanRoomTopologyTests(unittest.TestCase):
    def _two_room_mask(self):
        mask = np.zeros((120, 160), dtype=np.uint8)
        cv2.rectangle(mask, (15, 15), (145, 105), 1, thickness=3)
        cv2.line(mask, (80, 15), (80, 105), 1, thickness=3)
        # 门窗即使类别不确定，也必须作为闭合边界参与房间提取。
        mask[14:18, 45:70] = 2
        mask[55:67, 78:83] = 3
        return mask

    def test_wall_window_and_door_union_produces_two_closed_rooms(self):
        result = extract_room_topology(self._two_room_mask(), min_room_area_px=200)

        self.assertEqual(result["status"], "closed")
        self.assertEqual(result["room_count"], 2)
        self.assertTrue(result["load_geometry_ready"] is False)
        self.assertIsNone(result["total_area_m2"])
        self.assertTrue(all(room["area_px2"] > 4000 for room in result["rooms"]))

    def test_short_break_in_wall_is_closed_automatically(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (15, 15), (85, 85), 1, thickness=3)
        mask[48:53, 14:19] = 0

        result = extract_room_topology(mask, max_gap_px=6, min_room_area_px=200)

        self.assertEqual(result["room_count"], 1)
        self.assertTrue(result["closure_applied"])

    def test_open_shape_does_not_create_a_fake_room(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.line(mask, (15, 15), (85, 15), 1, thickness=3)
        cv2.line(mask, (15, 15), (15, 85), 1, thickness=3)
        cv2.line(mask, (15, 85), (85, 85), 1, thickness=3)

        result = extract_room_topology(mask, max_gap_px=6, min_room_area_px=200)

        self.assertEqual(result["status"], "no_closed_rooms")
        self.assertEqual(result["room_count"], 0)

    def test_valid_scale_converts_polygon_area_to_square_metres(self):
        result = extract_room_topology(
            self._two_room_mask(), scale_m_per_px=0.1, min_room_area_px=200
        )

        self.assertTrue(result["load_geometry_ready"])
        self.assertAlmostEqual(
            result["total_area_m2"],
            result["total_area_px2"] * 0.01,
            places=6,
        )
        for room in result["rooms"]:
            self.assertAlmostEqual(room["area_m2"], room["area_px2"] * 0.01, places=6)

    def test_non_positive_scale_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scale_m_per_px"):
            extract_room_topology(self._two_room_mask(), scale_m_per_px=0)

    def test_segmenter_prediction_includes_room_topology(self):
        mask = self._two_room_mask()
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        segmenter._run_inference = lambda _image: (
            mask,
            np.zeros((4, *mask.shape), dtype=np.float32),
        )

        result = segmenter.predict(
            np.zeros((*mask.shape, 3), dtype=np.uint8),
            use_preprocessing=False,
        )

        self.assertEqual(result["room_topology"]["room_count"], 2)
        self.assertIsNone(result["room_topology"]["total_area_m2"])
        self.assertEqual(result["room_topology"]["max_gap_px"], 3)

    def test_closed_room_polygons_are_drawn_as_red_overlay_boundaries(self):
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        overlay = np.zeros((60, 80, 3), dtype=np.uint8)
        topology = {
            "rooms": [{
                "polygon_px": [[10, 10], [70, 10], [70, 50], [10, 50]],
            }],
        }

        result = segmenter._draw_room_boundaries(overlay, topology)

        self.assertTrue(np.array_equal(result[10, 40], segmenter.CLASS_COLORS_BGR[1]))
        self.assertTrue(np.array_equal(result[30, 10], segmenter.CLASS_COLORS_BGR[1]))

    def test_segmenter_roi_restores_full_page_coordinates_and_explicit_topology_settings(self):
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        received_shapes = []

        def fake_inference(image):
            received_shapes.append(image.shape[:2])
            mask = np.ones(image.shape[:2], dtype=np.uint8)
            return mask, np.zeros((4, *image.shape[:2]), dtype=np.float32)

        segmenter._run_inference = fake_inference
        result = segmenter.predict(
            np.zeros((100, 120, 3), dtype=np.uint8),
            use_preprocessing=False,
            inference_roi=[20, 10, 100, 90],
            topology_max_gap_px=12,
            topology_min_room_area_px=5000,
        )

        self.assertEqual(received_shapes, [(80, 80)])
        self.assertEqual(result["mask"].shape, (100, 120))
        self.assertTrue(np.all(result["mask"][10:90, 20:100] == 1))
        self.assertTrue(np.all(result["mask"][:10] == 0))
        self.assertTrue(np.all(result["mask"][:, :20] == 0))
        self.assertEqual(result["inference_roi"], [20, 10, 100, 90])
        self.assertEqual(result["room_topology"]["max_gap_px"], 12)
        self.assertEqual(result["room_topology"]["min_room_area_px"], 5000.0)

    def test_segmenter_can_preserve_full_context_then_clear_predictions_outside_roi(self):
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        received_shapes = []

        def fake_inference(image):
            received_shapes.append(image.shape[:2])
            return (
                np.ones(image.shape[:2], dtype=np.uint8),
                np.zeros((4, *image.shape[:2]), dtype=np.float32),
            )

        segmenter._run_inference = fake_inference
        result = segmenter.predict(
            np.zeros((100, 120, 3), dtype=np.uint8),
            use_preprocessing=False,
            inference_roi=[20, 10, 100, 90],
            preserve_full_context=True,
            topology_max_gap_px=12,
            topology_min_room_area_px=500,
        )

        self.assertEqual(received_shapes, [(100, 120)])
        self.assertTrue(np.all(result["mask"][10:90, 20:100] == 1))
        self.assertTrue(np.all(result["mask"][:10] == 0))
        self.assertTrue(np.all(result["mask"][:, 100:] == 0))

    def test_segmenter_uses_repaired_mask_for_all_outputs_and_preserves_raw_diagnostic(self):
        raw_mask = self._two_room_mask()
        final_mask = raw_mask.copy()
        final_mask[55:67, 78:83] = 0
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        segmenter._run_inference = lambda _image: (
            raw_mask,
            np.zeros((4, *raw_mask.shape), dtype=np.float32),
        )
        repair_result = {
            "calculation_mask": final_mask,
            "exterior_repair": {"status": "repaired"},
            "internal_fragments": {"ignored": [], "repaired": [], "ambiguous": []},
            "manual_exterior_wall_required": False,
            "manual_review_reasons": [],
        }

        with patch(
            "floorplan_onnx.repair_vector_floorplan_topology",
            return_value=repair_result,
        ) as repair:
            result = segmenter.predict(
                np.zeros((*raw_mask.shape, 3), dtype=np.uint8),
                use_preprocessing=False,
                topology_repair_context={
                    "building_roi": [10, 10, 150, 110],
                    "structural_support_mask": np.ones_like(raw_mask) * 255,
                    "max_exterior_gap_px": 12,
                    "max_internal_component_area_px": 500,
                },
                topology_min_room_area_px=200,
            )

        repair.assert_called_once()
        self.assertTrue(np.array_equal(result["raw_model_mask"], raw_mask))
        self.assertTrue(np.array_equal(result["mask"], final_mask))
        self.assertEqual(result["topology_repair"]["exterior_repair"]["status"], "repaired")
        self.assertEqual(result["room_topology"]["room_count"], 1)

    def test_onnx_input_preserves_aspect_ratio_and_discards_letterbox_padding(self):
        class FakeSession:
            def run(self, _outputs, _inputs):
                logits = np.zeros((1, 4, 8, 8), dtype=np.float32)
                logits[0, 1, :2, :] = 10
                logits[0, 2, 2:6, :] = 10
                logits[0, 1, 6:, :] = 10
                return [logits]

        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.session = FakeSession()
        segmenter.input_name = "input"
        segmenter.img_size = 8
        segmenter.num_classes = 4

        pred, probabilities = segmenter._run_inference(
            np.full((4, 8, 3), 255, dtype=np.uint8)
        )

        self.assertEqual(pred.shape, (4, 8))
        self.assertEqual(probabilities.shape, (4, 4, 8))
        self.assertTrue(np.all(pred == 2))

    def test_annotation_removal_uses_the_drawing_background_instead_of_black(self):
        image = np.full((40, 40, 3), 255, dtype=np.uint8)
        image[15:25, 15:25] = (0, 255, 0)
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)

        cleaned, annotation_mask = segmenter.remove_annotations(image)

        self.assertGreater(annotation_mask[20, 20], 0)
        self.assertTrue(np.all(cleaned[20, 20] == 255))

    def test_confirmed_scale_is_applied_to_room_polygon_areas_without_mutating_source(self):
        topology = extract_room_topology(
            self._two_room_mask(),
            min_room_area_px=200,
        )

        scaled = apply_scale_to_room_topology(topology, 0.1)

        self.assertTrue(scaled["load_geometry_ready"])
        self.assertAlmostEqual(scaled["total_area_m2"], topology["total_area_px2"] * 0.01)
        self.assertTrue(all(room["area_m2"] is not None for room in scaled["rooms"]))
        self.assertIsNone(topology["total_area_m2"])
        self.assertTrue(all(room["area_m2"] is None for room in topology["rooms"]))

    def test_missing_scale_keeps_closed_rooms_unready_for_load_calculation(self):
        topology = extract_room_topology(
            self._two_room_mask(),
            min_room_area_px=200,
        )

        unscaled = apply_scale_to_room_topology(topology, None)

        self.assertFalse(unscaled["load_geometry_ready"])
        self.assertIsNone(unscaled["total_area_m2"])


if __name__ == "__main__":
    unittest.main()
