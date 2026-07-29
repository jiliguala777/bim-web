import math
import unittest

import cv2
import numpy as np

from floorplan_rooms import extract_room_topology
from floorplan_topology_repair import (
    _dominant_span_rectangles,
    repair_vector_floorplan_topology,
)


class ConservativeTopologyRepairTests(unittest.TestCase):
    @staticmethod
    def _blank():
        return np.zeros((100, 120), dtype=np.uint8)

    @staticmethod
    def _support():
        return np.zeros((100, 120), dtype=np.uint8)

    @staticmethod
    def _rectangle_with_gap(orientation="horizontal"):
        mask = ConservativeTopologyRepairTests._blank()
        if orientation == "horizontal":
            cv2.line(mask, (20, 20), (54, 20), 1, 3)
            cv2.line(mask, (66, 20), (100, 20), 1, 3)
        else:
            cv2.line(mask, (20, 20), (20, 44), 1, 3)
            cv2.line(mask, (20, 56), (20, 80), 1, 3)
        if orientation == "horizontal":
            cv2.line(mask, (20, 20), (20, 80), 1, 3)
        else:
            cv2.line(mask, (20, 20), (100, 20), 1, 3)
        cv2.line(mask, (20, 80), (100, 80), 1, 3)
        cv2.line(mask, (100, 20), (100, 80), 1, 3)
        return mask

    def _repair(self, mask, support, **overrides):
        kwargs = {
            "building_roi": [10, 10, 110, 90],
            "structural_support_mask": support,
            "max_exterior_gap_px": 16,
            "max_internal_component_area_px": 500,
            "min_room_area_px": 500,
        }
        kwargs.update(overrides)
        return repair_vector_floorplan_topology(mask, **kwargs)

    def test_repairs_one_supported_horizontal_exterior_gap(self):
        mask = self._rectangle_with_gap("horizontal")
        support = self._support()
        cv2.line(support, (54, 20), (66, 20), 255, 3)

        result = self._repair(mask, support)

        self.assertEqual(result["exterior_repair"]["status"], "repaired")
        self.assertEqual(result["exterior_repair"]["accepted_line_px"], [54, 20, 66, 20])
        self.assertTrue(np.all(result["calculation_mask"][20, 54:67] == 1))
        self.assertFalse(result["manual_exterior_wall_required"])

    def test_repairs_one_supported_vertical_exterior_gap(self):
        mask = self._rectangle_with_gap("vertical")
        support = self._support()
        cv2.line(support, (20, 44), (20, 56), 255, 3)

        result = self._repair(mask, support)

        self.assertEqual(result["exterior_repair"]["status"], "repaired")
        self.assertEqual(result["exterior_repair"]["accepted_line_px"], [20, 44, 20, 56])

    def test_repairs_two_exterior_gaps_as_one_combined_solution(self):
        mask = self._rectangle_with_gap("horizontal")
        mask[78:83, 54:67] = 0
        support = self._support()
        cv2.line(support, (54, 20), (66, 20), 255, 3)
        cv2.line(support, (54, 80), (66, 80), 255, 3)

        result = self._repair(mask, support)

        self.assertEqual(result["exterior_repair"]["status"], "repaired")
        self.assertEqual(len(result["exterior_repair"]["accepted_lines_px"]), 2)
        topology = extract_room_topology(
            result["calculation_mask"],
            max_gap_px=0,
            min_room_area_px=500,
        )
        self.assertEqual(topology["room_count"], 1)
        self.assertFalse(result["manual_exterior_wall_required"])

    def test_uses_four_long_span_lines_for_fragmented_room_perimeter(self):
        mask = np.zeros((140, 180), dtype=np.uint8)
        support = np.zeros_like(mask)
        cv2.rectangle(support, (30, 20), (150, 120), 255, 3)
        for start, end in ((30, 55), (67, 92), (104, 130), (142, 150)):
            cv2.line(mask, (start, 20), (end, 20), 1, 3)
            cv2.line(mask, (start, 120), (end, 120), 1, 3)
        for start, end in ((20, 42), (54, 76), (88, 108), (116, 120)):
            cv2.line(mask, (30, start), (30, end), 1, 3)
            cv2.line(mask, (150, start), (150, end), 1, 3)

        result = repair_vector_floorplan_topology(
            mask,
            building_roi=[10, 10, 170, 130],
            structural_support_mask=support,
            max_exterior_gap_px=16,
            max_internal_component_area_px=500,
            min_room_area_px=1000,
        )

        lines = result["exterior_repair"]["accepted_lines_px"]
        self.assertEqual(result["exterior_repair"]["reason"], "dominant_span_rectangle")
        self.assertEqual(result["closure_status"], "complete")
        self.assertEqual(len(lines), 4)
        self.assertEqual(sum(line[0] == line[2] for line in lines), 2)
        self.assertEqual(sum(line[1] == line[3] for line in lines), 2)
        self.assertTrue(all(
            math.hypot(line[2] - line[0], line[3] - line[1]) >= 95
            for line in lines
        ))
        candidate = result["exterior_repair"]["accepted_candidates"][0]
        self.assertGreaterEqual(
            min(candidate["vector_support_by_side"].values()),
            0.60,
        )
        self.assertGreaterEqual(
            min(candidate["model_support_by_side"].values()),
            0.15,
        )
        self.assertIn("vector_support_ratio", candidate)
        self.assertIn("model_support_ratio", candidate)

    def test_skips_exterior_repair_when_major_area_is_already_closed(self):
        mask = np.zeros((140, 180), dtype=np.uint8)
        support = np.zeros_like(mask)
        cv2.rectangle(mask, (30, 20), (150, 120), 1, 3)
        cv2.rectangle(support, (15, 15), (165, 125), 255, 1)

        result = repair_vector_floorplan_topology(
            mask,
            building_roi=[10, 10, 170, 130],
            structural_support_mask=support,
            max_exterior_gap_px=16,
            max_internal_component_area_px=500,
            min_room_area_px=500,
        )

        repair = result["exterior_repair"]
        self.assertEqual(repair["status"], "not_needed")
        self.assertEqual(repair["reason"], "already_closed")
        self.assertEqual(repair["accepted_lines_px"], [])
        self.assertEqual(result["closure_status"], "complete")
        self.assertFalse(result["manual_exterior_wall_required"])
        self.assertTrue(repair["initial_closure"]["is_already_closed"])

    def test_rejects_dominant_rectangle_when_any_side_has_weak_model_support(self):
        mask = np.zeros((140, 180), dtype=np.uint8)
        support = np.zeros_like(mask)
        cv2.rectangle(support, (30, 20), (150, 120), 255, 3)

        cv2.line(mask, (30, 20), (150, 20), 1, 3)
        cv2.line(mask, (150, 20), (150, 120), 1, 3)
        cv2.line(mask, (140, 120), (150, 120), 1, 3)
        cv2.line(mask, (30, 20), (30, 22), 1, 3)

        candidates = _dominant_span_rectangles(
            mask,
            support,
            [10, 10, 170, 130],
            min_room_area_px=1000,
        )

        self.assertEqual(candidates, [])

    def test_reports_partial_when_only_existing_small_room_is_closed(self):
        mask = np.zeros((140, 180), dtype=np.uint8)
        support = np.zeros_like(mask)
        cv2.rectangle(mask, (20, 20), (70, 70), 1, 3)
        cv2.line(mask, (100, 20), (160, 20), 1, 3)
        cv2.line(mask, (100, 20), (100, 120), 1, 3)
        cv2.line(support, (100, 20), (160, 20), 255, 3)

        result = repair_vector_floorplan_topology(
            mask,
            building_roi=[10, 10, 170, 130],
            structural_support_mask=support,
            max_exterior_gap_px=16,
            max_internal_component_area_px=500,
            min_room_area_px=500,
        )

        self.assertEqual(result["closure_status"], "partial")
        self.assertTrue(result["manual_exterior_wall_required"])

    def test_rejects_corner_gap_that_needs_horizontal_and_vertical_lines(self):
        mask = self._blank()
        cv2.line(mask, (20, 30), (20, 80), 1, 3)
        cv2.line(mask, (20, 80), (100, 80), 1, 3)
        cv2.line(mask, (100, 80), (100, 20), 1, 3)
        cv2.line(mask, (100, 20), (30, 20), 1, 3)
        support = self._support()
        cv2.line(support, (20, 20), (30, 20), 255, 3)
        cv2.line(support, (20, 20), (20, 30), 255, 3)

        result = self._repair(mask, support)

        self.assertNotEqual(result["exterior_repair"]["status"], "repaired")
        self.assertTrue(result["manual_exterior_wall_required"])

    def test_repairs_unsupported_wall_gap_without_joining_outside_annotation(self):
        mask = self._rectangle_with_gap("horizontal")
        cv2.line(mask, (5, 8), (22, 8), 1, 1)
        cv2.line(mask, (28, 8), (45, 8), 1, 1)
        support = self._support()
        cv2.line(support, (22, 8), (28, 8), 255, 1)

        result = self._repair(mask, support)

        self.assertEqual(result["exterior_repair"]["status"], "repaired")
        self.assertTrue(np.all(result["calculation_mask"][20, 54:67] == 1))
        self.assertTrue(np.array_equal(result["calculation_mask"][8, 5:46], mask[8, 5:46]))

    def test_repairs_supported_internal_u_shape_before_considering_removal(self):
        mask = self._rectangle_with_gap("horizontal")
        support = self._support()
        cv2.line(support, (54, 20), (66, 20), 255, 3)
        cv2.line(mask, (45, 44), (45, 56), 1, 2)
        cv2.line(mask, (45, 44), (60, 44), 1, 2)
        cv2.line(mask, (45, 56), (60, 56), 1, 2)
        cv2.line(support, (60, 44), (60, 56), 255, 2)

        result = self._repair(
            mask,
            support,
            max_internal_component_area_px=300,
            min_room_area_px=20,
        )

        self.assertEqual(len(result["internal_fragments"]["repaired"]), 1)
        self.assertTrue(np.all(result["calculation_mask"][44:57, 60] == 1))

    def test_omits_small_unsupported_internal_fragment_from_final_mask(self):
        mask = self._rectangle_with_gap("horizontal")
        support = self._support()
        cv2.line(support, (54, 20), (66, 20), 255, 3)
        cv2.line(mask, (45, 48), (58, 48), 1, 2)

        result = self._repair(mask, support)

        self.assertEqual(len(result["internal_fragments"]["ignored"]), 1)
        self.assertTrue(np.all(result["calculation_mask"][47:50, 45:59] == 0))
        self.assertTrue(np.any(mask[47:50, 45:59] == 1))

    def test_preserves_internal_component_that_bounds_an_existing_room(self):
        mask = self._rectangle_with_gap("horizontal")
        support = self._support()
        cv2.line(support, (54, 20), (66, 20), 255, 3)
        cv2.rectangle(mask, (45, 40), (65, 60), 1, 2)

        result = self._repair(mask, support, min_room_area_px=100)

        self.assertEqual(result["internal_fragments"]["ignored"], [])
        self.assertGreaterEqual(len(result["internal_fragments"]["ambiguous"]), 1)
        self.assertTrue(np.any(result["calculation_mask"][39:62, 44:67] == 1))

    def test_omits_long_slender_unsupported_fragment_even_above_small_area_limit(self):
        mask = self._rectangle_with_gap("horizontal")
        support = self._support()
        cv2.line(support, (54, 20), (66, 20), 255, 3)
        cv2.line(mask, (35, 48), (82, 48), 1, 2)

        result = self._repair(mask, support, max_internal_component_area_px=50)

        ignored_boxes = [item["bbox_px"] for item in result["internal_fragments"]["ignored"]]
        self.assertTrue(any(width >= 45 for _x, _y, width, _height in ignored_boxes))
        self.assertTrue(np.all(result["calculation_mask"][47:50, 35:83] == 0))

    def test_preserves_long_slender_fragment_with_strong_vector_support(self):
        mask = self._rectangle_with_gap("horizontal")
        support = self._support()
        cv2.line(support, (54, 20), (66, 20), 255, 3)
        cv2.line(mask, (35, 48), (82, 48), 1, 2)
        cv2.line(support, (35, 48), (82, 48), 255, 2)

        result = self._repair(mask, support, max_internal_component_area_px=50)

        self.assertTrue(np.any(result["calculation_mask"][47:50, 35:83] == 1))
        self.assertTrue(any(
            item.get("reason") == "vector_supported"
            for item in result["internal_fragments"]["ambiguous"]
        ))

    def test_repairs_one_long_supported_lower_entrance_between_vertical_anchors(self):
        mask = np.zeros((140, 180), dtype=np.uint8)
        support = np.zeros_like(mask)
        cv2.line(mask, (40, 20), (140, 20), 1, 3)
        cv2.line(mask, (40, 20), (40, 110), 1, 3)
        cv2.line(mask, (140, 20), (140, 110), 1, 3)
        cv2.line(support, (40, 110), (140, 110), 255, 3)

        result = repair_vector_floorplan_topology(
            mask,
            building_roi=[10, 10, 170, 130],
            structural_support_mask=support,
            max_exterior_gap_px=16,
            max_internal_component_area_px=300,
            min_room_area_px=500,
        )

        self.assertEqual(result["exterior_repair"]["status"], "repaired")
        x0, y0, x1, y1 = result["exterior_repair"]["accepted_line_px"]
        self.assertLessEqual(x0, 42)
        self.assertGreaterEqual(x1, 138)
        self.assertEqual(y0, y1)
        self.assertTrue(np.all(result["calculation_mask"][y0, x0:x1 + 1] == 1))

    def test_repairs_two_distinct_valid_lower_entrance_rows(self):
        mask = np.zeros((150, 180), dtype=np.uint8)
        support = np.zeros_like(mask)
        for x0, x1, top, bottom in ((15, 75, 45, 100), (105, 165, 75, 130)):
            cv2.line(mask, (x0, top), (x1, top), 1, 3)
            cv2.line(mask, (x0, top), (x0, bottom), 1, 3)
            cv2.line(mask, (x1, top), (x1, bottom), 1, 3)
            cv2.line(support, (x0, bottom), (x1, bottom), 255, 3)

        result = repair_vector_floorplan_topology(
            mask,
            building_roi=[5, 5, 175, 145],
            structural_support_mask=support,
            max_exterior_gap_px=16,
            max_internal_component_area_px=300,
            min_room_area_px=300,
        )

        self.assertEqual(result["exterior_repair"]["status"], "repaired")
        self.assertGreaterEqual(result["exterior_repair"]["candidate_count"], 2)
        self.assertGreaterEqual(len(result["exterior_repair"]["accepted_lines_px"]), 2)
        topology = extract_room_topology(
            result["calculation_mask"],
            max_gap_px=0,
            min_room_area_px=300,
        )
        self.assertEqual(topology["room_count"], 2)

    def test_rejects_long_lower_dimension_line_without_vertical_wall_anchors(self):
        mask = np.zeros((140, 180), dtype=np.uint8)
        support = np.zeros_like(mask)
        cv2.line(mask, (20, 20), (160, 20), 1, 3)
        cv2.line(support, (20, 110), (160, 110), 255, 3)

        result = repair_vector_floorplan_topology(
            mask,
            building_roi=[10, 10, 170, 130],
            structural_support_mask=support,
            max_exterior_gap_px=16,
            max_internal_component_area_px=300,
            min_room_area_px=300,
        )

        self.assertNotEqual(result["exterior_repair"]["status"], "repaired")
        self.assertIsNone(result["exterior_repair"]["accepted_line_px"])


if __name__ == "__main__":
    unittest.main()
