import unittest

import numpy as np


def supported_probabilities(width, height):
    probabilities = np.zeros((10, height, width), dtype=np.float32)
    probabilities[2, :, :] = 0.7
    probabilities[4, :, :] = 0.8
    return probabilities


def accepted_wall(line_id, start, end, orientation):
    return {
        "candidate_id": line_id,
        "source_native_id": f"native-{line_id}",
        "orientation": orientation,
        "start_pt": [float(start[0]), float(start[1])],
        "end_pt": [float(end[0]), float(end[1])],
        "start_px": list(start),
        "end_px": list(end),
        "native_evidence": {
            "native_structural": True,
            "roi_inside_ratio": 1.0,
        },
        "model_evidence": {
            "wall_mean": 0.8,
            "footprint_mean": 0.8,
        },
        "decision": "accepted_wall_candidate",
        "reason_codes": ["model_wall_supported", "footprint_supported"],
    }


def exterior_wall(line_id, start, end, orientation, inside_direction):
    wall = accepted_wall(line_id, start, end, orientation)
    wall.update({
        "inside_direction": inside_direction,
        "footprint_inside_mean": 0.9,
        "footprint_outside_mean": 0.0,
    })
    return wall


def topology_gap(
    start_x, start_y, end_x, end_y, *, gap_id="gap-0001",
    host_wall_ids=("bottom-left", "bottom-right"), decision="accepted_gap",
):
    orientation = (
        "horizontal" if start_y == end_y
        else "vertical" if start_x == end_x
        else "horizontal"
    )
    return {
        "gap_id": gap_id,
        "orientation": orientation,
        "start_px": [start_x, start_y],
        "end_px": [end_x, end_y],
        "width_px": float(abs(end_x - start_x) + abs(end_y - start_y)),
        "host_wall_ids": list(host_wall_ids),
        "inside_direction": "up" if orientation == "horizontal" else "left",
        "decision": decision,
        "reason_codes": (
            ["unambiguous_collinear_exterior_gap"]
            if decision == "accepted_gap" else ["not_strictly_collinear"]
        ),
    }


def door_opening(start_x, start_y, end_x, end_y):
    return {
        "opening_id": "opening-0001",
        "kind": "door",
        "orientation": "horizontal",
        "start_px": [start_x, start_y],
        "end_px": [end_x, end_y],
        "width_px": float(abs(end_x - start_x)),
        "width_m": None,
        "host_wall_ids": ["bottom-left", "bottom-right"],
        "exterior": True,
        "confidence": 0.9,
        "reason_codes": ["door_line_and_endpoints_supported"],
    }


def rectangle_with_bottom_gap(door_gap=(40, 60)):
    gap_start, gap_end = door_gap
    return [
        exterior_wall("top", (20, 20), (80, 20), "horizontal", "down"),
        exterior_wall("right", (80, 20), (80, 80), "vertical", "left"),
        exterior_wall("bottom-right", (gap_end, 80), (80, 80), "horizontal", "up"),
        exterior_wall("bottom-left", (20, 80), (gap_start, 80), "horizontal", "up"),
        exterior_wall("left", (20, 20), (20, 80), "vertical", "right"),
    ]


def walls_from_polygon(points, prefix="wall"):
    walls = []
    clockwise_inside = {
        (1, 0): ("horizontal", "down"),
        (0, 1): ("vertical", "left"),
        (-1, 0): ("horizontal", "up"),
        (0, -1): ("vertical", "right"),
    }
    for index, (start, end) in enumerate(zip(points, points[1:] + points[:1]), 1):
        dx = 0 if end[0] == start[0] else (1 if end[0] > start[0] else -1)
        dy = 0 if end[1] == start[1] else (1 if end[1] > start[1] else -1)
        orientation, inside = clockwise_inside[(dx, dy)]
        walls.append(exterior_wall(f"{prefix}-{index}", start, end, orientation, inside))
    return walls


def supported_footprint(*polygons, image_size=(100, 100)):
    width, height = image_size
    probabilities = np.zeros((10, height, width), dtype=np.float32)
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.int32)
        import cv2
        cv2.fillPoly(probabilities[0], [points], 1.0)
    return probabilities


def supported_courtyard(outer, inner, image_size=(100, 100)):
    probabilities = supported_footprint(outer, image_size=image_size)
    import cv2
    cv2.fillPoly(probabilities[0], [np.asarray(inner, dtype=np.int32)], 0.0)
    return probabilities


class ExteriorWallSelectionTests(unittest.TestCase):
    def test_selects_wall_with_one_footprint_interior_side(self):
        from vector_pdf_exterior import select_exterior_walls

        probabilities = supported_probabilities(200, 120)
        probabilities[0, 31:80, 20:181] = 0.9
        walls = [accepted_wall("left", (20, 30), (20, 80), "vertical")]

        result = select_exterior_walls(
            walls, probabilities, (200, 120), [10, 10, 190, 110],
        )

        self.assertEqual(result[0]["inside_direction"], "right")
        self.assertGreaterEqual(result[0]["footprint_inside_mean"], 0.50)
        self.assertLess(result[0]["footprint_outside_mean"], 0.25)
        self.assertEqual(result[0]["candidate_id"], "left")

    def test_skips_non_axis_aligned_candidate_without_losing_orthogonal_wall(self):
        from vector_pdf_exterior import select_exterior_walls

        probabilities = supported_probabilities(200, 120)
        probabilities[0, 31:80, 20:181] = 0.9
        walls = [
            accepted_wall("valid", (20, 30), (20, 80), "vertical"),
            accepted_wall("diagonal", (40, 30), (80, 50), "horizontal"),
        ]

        result = select_exterior_walls(
            walls, probabilities, (200, 120), [10, 10, 190, 110],
        )

        self.assertEqual([wall["candidate_id"] for wall in result], ["valid"])


class ExteriorGapEnumerationTests(unittest.TestCase):
    def test_enumerates_only_unambiguous_collinear_gap(self):
        from vector_pdf_exterior import enumerate_exterior_gaps

        walls = [
            exterior_wall("a", (10, 20), (70, 20), "horizontal", "down"),
            exterior_wall("b", (90, 20), (150, 20), "horizontal", "down"),
        ]

        gaps = enumerate_exterior_gaps(walls, (200, 120), [0, 0, 200, 120])

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["decision"], "accepted_gap")
        self.assertEqual(gaps[0]["start_px"], [70, 20])
        self.assertEqual(gaps[0]["end_px"], [90, 20])
        self.assertEqual(gaps[0]["host_wall_ids"], ["a", "b"])

    def test_rejects_nearby_parallel_walls_without_projecting_an_anchor(self):
        from vector_pdf_exterior import enumerate_exterior_gaps

        gaps = enumerate_exterior_gaps([
            exterior_wall("a", (10, 20), (70, 20), "horizontal", "down"),
            exterior_wall("b", (90, 21), (150, 21), "horizontal", "down"),
        ], (200, 120), [0, 0, 200, 120])

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["decision"], "rejected_gap")
        self.assertIn("not_strictly_collinear", gaps[0]["reason_codes"])
        self.assertEqual(gaps[0]["start_px"], [70, 20])
        self.assertEqual(gaps[0]["end_px"], [90, 21])

    def test_rejects_gap_between_walls_with_different_inside_directions(self):
        from vector_pdf_exterior import enumerate_exterior_gaps

        gaps = enumerate_exterior_gaps([
            exterior_wall("a", (10, 20), (70, 20), "horizontal", "down"),
            exterior_wall("b", (90, 20), (150, 20), "horizontal", "up"),
        ], (200, 120), [0, 0, 200, 120])

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["decision"], "rejected_gap")
        self.assertIn("different_inside_directions", gaps[0]["reason_codes"])

    def test_rejects_gap_with_perpendicular_structural_crossing(self):
        from vector_pdf_exterior import enumerate_exterior_gaps

        gaps = enumerate_exterior_gaps([
            exterior_wall("a", (10, 20), (70, 20), "horizontal", "down"),
            exterior_wall("b", (90, 20), (150, 20), "horizontal", "down"),
            exterior_wall("cross", (80, 10), (80, 30), "vertical", "right"),
        ], (200, 120), [0, 0, 200, 120])

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["decision"], "rejected_gap")
        self.assertIn("perpendicular_structural_crossing", gaps[0]["reason_codes"])

    def test_rejects_gap_with_multiple_competing_endpoints(self):
        from vector_pdf_exterior import enumerate_exterior_gaps

        gaps = enumerate_exterior_gaps([
            exterior_wall("a", (10, 20), (70, 20), "horizontal", "down"),
            exterior_wall("b", (90, 20), (150, 20), "horizontal", "down"),
            exterior_wall("c", (90, 21), (150, 21), "horizontal", "down"),
        ], (200, 120), [0, 0, 200, 120])

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["decision"], "rejected_gap")
        self.assertIn("multiple_competing_endpoints", gaps[0]["reason_codes"])
        self.assertEqual(gaps[0]["host_wall_ids"], ["a", "b", "c"])

    def test_rejects_gap_outside_building_roi(self):
        from vector_pdf_exterior import enumerate_exterior_gaps

        gaps = enumerate_exterior_gaps([
            exterior_wall("a", (10, 20), (70, 20), "horizontal", "down"),
            exterior_wall("b", (90, 20), (150, 20), "horizontal", "down"),
        ], (200, 120), [0, 0, 80, 120])

        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["decision"], "rejected_gap")
        self.assertIn("outside_building_roi", gaps[0]["reason_codes"])


class ExteriorTopologyTests(unittest.TestCase):
    def test_opening_bridge_closes_rectangle_without_becoming_real_wall(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        walls = rectangle_with_bottom_gap(door_gap=(40, 60))
        topology = build_exterior_topology(
            walls,
            [topology_gap(40, 80, 60, 80)],
            [door_opening(40, 80, 60, 80)],
            supported_footprint(polygon),
            (100, 100),
            [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "review_required")
        self.assertEqual(topology["area_px2"], 3600.0)
        self.assertEqual(topology["bridges"][0]["bridge_type"], "opening_bridge")
        self.assertEqual(topology["opening_ids"], ["opening-0001"])
        self.assertEqual(topology["source_wall_ids"], [
            "bottom-left", "bottom-right", "left", "right", "top",
        ])
        self.assertNotIn("opening-0001", topology["source_wall_ids"])
        self.assertFalse(topology["confirmed"])
        self.assertFalse(topology["load_geometry_ready"])

    def test_unknown_gap_repairs_at_metric_boundary_only(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        walls = rectangle_with_bottom_gap(door_gap=(40, 60))
        accepted = build_exterior_topology(
            walls, [topology_gap(40, 80, 60, 80)], [],
            supported_footprint(polygon), (100, 100), [0, 0, 100, 100],
            scale_m_per_px=0.03,
        )
        rejected = build_exterior_topology(
            walls, [topology_gap(40, 80, 61, 80)], [],
            supported_footprint(polygon), (100, 100), [0, 0, 100, 100],
            scale_m_per_px=0.03,
        )

        self.assertEqual(accepted["bridges"][0]["bridge_type"], "small_gap_repair")
        self.assertEqual(
            rejected["unresolved_gaps"][0]["reason_codes"],
            ["gap_exceeds_repair_limit"],
        )

    def test_unknown_gap_repairs_at_decimal_metric_boundary(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        topology = build_exterior_topology(
            rectangle_with_bottom_gap(door_gap=(40, 46)),
            [topology_gap(40, 80, 46, 80)], [], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100], scale_m_per_px=0.1,
        )

        self.assertEqual(topology["bridges"][0]["bridge_type"], "small_gap_repair")
        self.assertEqual(topology["area_px2"], 3600.0)

    def test_unknown_gap_strictly_rejects_just_over_metric_boundary(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        topology = build_exterior_topology(
            rectangle_with_bottom_gap(door_gap=(40, 46.000001)),
            [topology_gap(40, 80, 46.000001, 80)], [],
            supported_footprint(polygon), (100, 100), [0, 0, 100, 100],
            scale_m_per_px=0.1,
        )

        self.assertEqual(topology["bridges"], [])
        self.assertEqual(
            topology["unresolved_gaps"][0]["reason_codes"],
            ["gap_exceeds_repair_limit"],
        )

    def test_unknown_gap_rejects_value_inside_old_isclose_window(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        gap_end = 46.000000000005
        topology = build_exterior_topology(
            rectangle_with_bottom_gap(door_gap=(40, gap_end)),
            [topology_gap(40, 80, gap_end, 80)], [],
            supported_footprint(polygon), (100, 100), [0, 0, 100, 100],
            scale_m_per_px=0.1,
        )

        self.assertEqual(topology["bridges"], [])
        self.assertEqual(
            topology["unresolved_gaps"][0]["reason_codes"],
            ["gap_exceeds_repair_limit"],
        )

    def test_unknown_gap_repairs_at_unscaled_pixel_boundary_only(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        accepted = build_exterior_topology(
            rectangle_with_bottom_gap(door_gap=(40, 44)),
            [topology_gap(40, 80, 44, 80)], [], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100],
        )
        rejected = build_exterior_topology(
            rectangle_with_bottom_gap(door_gap=(40, 45)),
            [topology_gap(40, 80, 45, 80)], [], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(accepted["bridges"][0]["bridge_type"], "small_gap_repair")
        self.assertEqual(
            rejected["unresolved_gaps"][0]["reason_codes"],
            ["gap_exceeds_repair_limit"],
        )

    def test_concave_orthogonal_footprint_keeps_its_notch(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 50), (50, 50), (50, 80), (20, 80)]
        topology = build_exterior_topology(
            walls_from_polygon(polygon), [], [], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "review_required")
        self.assertEqual(topology["area_px2"], 2700.0)
        self.assertEqual(topology["perimeter_px"], 240.0)
        self.assertEqual(set(map(tuple, topology["polygon_px"])), set(polygon))

    def test_duplicate_wall_sources_survive_collinear_merge(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        walls = walls_from_polygon(polygon)
        walls.append(exterior_wall("duplicate-top", (20, 20), (80, 20), "horizontal", "down"))
        topology = build_exterior_topology(
            walls, [], [], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertIn("duplicate-top", topology["source_wall_ids"])
        self.assertEqual(len(topology["source_wall_ids"]), 5)

    def test_open_wall_chain_has_no_bounding_rectangle_fallback(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        walls = walls_from_polygon(polygon)[:-1]
        topology = build_exterior_topology(
            walls, [], [], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "exterior_not_closed")
        self.assertEqual(topology["polygon_px"], [])
        self.assertEqual(topology["area_px2"], 0.0)

    def test_multiple_comparable_faces_are_ambiguous(self):
        from vector_pdf_exterior import build_exterior_topology

        first = [(10, 10), (40, 10), (40, 40), (10, 40)]
        second = [(60, 60), (90, 60), (90, 90), (60, 90)]
        topology = build_exterior_topology(
            walls_from_polygon(first, "first") + walls_from_polygon(second, "second"),
            [], [], supported_footprint(first, second),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "ambiguous_exterior")
        self.assertEqual(topology["polygon_px"], [])
        self.assertFalse(topology["confirmed"])

    def test_unequal_disconnected_buildings_are_ambiguous(self):
        from vector_pdf_exterior import build_exterior_topology

        large = [(10, 10), (70, 10), (70, 70), (10, 70)]
        small = [(75, 75), (95, 75), (95, 95), (75, 95)]
        topology = build_exterior_topology(
            walls_from_polygon(large, "large") + walls_from_polygon(small, "small"),
            [], [], supported_footprint(large, small),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "ambiguous_exterior")
        self.assertEqual(topology["polygon_px"], [])
        self.assertEqual(topology["area_px2"], 0.0)

    def test_unsupported_small_disconnected_building_is_still_ambiguous(self):
        from vector_pdf_exterior import build_exterior_topology

        large = [(10, 10), (70, 10), (70, 70), (10, 70)]
        small = [(75, 75), (95, 75), (95, 95), (75, 95)]
        topology = build_exterior_topology(
            walls_from_polygon(large, "large") + walls_from_polygon(small, "small"),
            [], [], supported_footprint(large),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "ambiguous_exterior")
        self.assertEqual(topology["polygon_px"], [])
        self.assertEqual(topology["source_wall_ids"], [])

    def test_outer_ring_with_unsupported_courtyard_is_ambiguous(self):
        from vector_pdf_exterior import build_exterior_topology

        outer = [(10, 10), (90, 10), (90, 90), (10, 90)]
        courtyard = [(40, 40), (60, 40), (60, 60), (40, 60)]
        topology = build_exterior_topology(
            walls_from_polygon(outer, "outer")
            + walls_from_polygon(courtyard, "courtyard"),
            [], [], supported_courtyard(outer, courtyard),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "ambiguous_exterior")
        self.assertEqual(topology["polygon_px"], [])
        self.assertEqual(topology["perimeter_px"], 0.0)

    def test_corner_gap_remains_unresolved(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        walls = [
            exterior_wall("top", (20, 20), (80, 20), "horizontal", "down"),
            exterior_wall("right", (80, 20), (80, 70), "vertical", "left"),
            exterior_wall("bottom", (70, 80), (20, 80), "horizontal", "up"),
            exterior_wall("left", (20, 80), (20, 20), "vertical", "right"),
        ]
        corner = topology_gap(
            80, 70, 70, 80, host_wall_ids=("right", "bottom"),
        )
        topology = build_exterior_topology(
            walls, [corner], [], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "exterior_not_closed")
        self.assertEqual(len(topology["bridges"]), 0)
        self.assertEqual(topology["unresolved_gaps"][0]["gap_id"], "gap-0001")

    def test_rejected_gap_stays_traceable_and_never_closes(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        rejected = topology_gap(40, 80, 60, 80, decision="rejected_gap")
        topology = build_exterior_topology(
            rectangle_with_bottom_gap(), [rejected], [], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "exterior_not_closed")
        self.assertEqual(topology["bridges"], [])
        self.assertEqual(topology["unresolved_gaps"][0]["decision"], "rejected_gap")

    def test_opening_must_match_gap_endpoints_and_hosts(self):
        from vector_pdf_exterior import build_exterior_topology

        polygon = [(20, 20), (80, 20), (80, 80), (20, 80)]
        mismatched = door_opening(40, 80, 60, 80)
        mismatched["host_wall_ids"] = ["left", "right"]
        topology = build_exterior_topology(
            rectangle_with_bottom_gap(), [topology_gap(40, 80, 60, 80)],
            [mismatched], supported_footprint(polygon),
            (100, 100), [0, 0, 100, 100], scale_m_per_px=0.03,
        )

        self.assertEqual(topology["bridges"][0]["bridge_type"], "small_gap_repair")
        self.assertEqual(topology["opening_ids"], [])

    def test_small_gap_repair_must_improve_supported_exterior_closure(self):
        from vector_pdf_exterior import build_exterior_topology

        walls = [
            exterior_wall("left-part", (20, 20), (40, 20), "horizontal", "down"),
            exterior_wall("right-part", (44, 20), (60, 20), "horizontal", "down"),
        ]
        topology = build_exterior_topology(
            walls,
            [topology_gap(
                40, 20, 44, 20,
                host_wall_ids=("left-part", "right-part"),
            )],
            [], supported_footprint(), (100, 100), [0, 0, 100, 100],
        )

        self.assertEqual(topology["status"], "exterior_not_closed")
        self.assertEqual(topology["bridges"], [])
        self.assertEqual(
            topology["unresolved_gaps"][0]["reason_codes"],
            ["gap_does_not_close_supported_exterior"],
        )


if __name__ == "__main__":
    unittest.main()
