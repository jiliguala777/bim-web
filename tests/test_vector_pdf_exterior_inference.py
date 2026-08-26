import unittest

import numpy as np


def wall(candidate_id, start, end, orientation, inside_direction):
    return {
        "candidate_id": candidate_id,
        "start_px": list(start),
        "end_px": list(end),
        "orientation": orientation,
        "inside_direction": inside_direction,
    }


def rectangular_footprint(width=200, height=140):
    values = np.zeros((10, height, width), dtype=np.float32)
    values[0, 20:120, 20:180] = 1.0
    return values


def inferred(
    inference_id, group_id, start, end, orientation, inside_direction,
    *, group_size=1,
):
    return {
        "inference_id": inference_id,
        "edge_group_id": group_id,
        "edge_group_size": group_size,
        "inference_type": "orthogonal_corner" if group_size == 2 else "collinear_extension",
        "start_px": list(start),
        "end_px": list(end),
        "orientation": orientation,
        "inside_direction": inside_direction,
        "length_px": float(abs(end[0] - start[0]) + abs(end[1] - start[1])),
        "anchor_wall_ids": ["anchor-a", "anchor-b"],
        "inside_mean": 0.90,
        "outside_mean": 0.05,
        "boundary_mean": 0.80,
        "inside_outside_difference": 0.85,
        "decision": "accepted_candidate",
        "reason_codes": ["footprint_guided_exterior_edge"],
    }


def open_upper_right_fixture():
    return [
        wall("top", (20, 20), (170, 20), "horizontal", "down"),
        wall("right", (180, 30), (180, 120), "vertical", "left"),
        wall("bottom", (20, 120), (180, 120), "horizontal", "up"),
        wall("left", (20, 20), (20, 120), "vertical", "right"),
    ]


class ExteriorInferenceCandidateTests(unittest.TestCase):
    def test_generates_two_edge_corner_group_for_supported_upper_right_gap(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("top", (20, 20), (170, 20), "horizontal", "down"),
            wall("right", (180, 30), (180, 120), "vertical", "left"),
        ]

        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [0, 0, 200, 140],
        )

        accepted = [item for item in candidates if item["decision"] == "accepted_candidate"]
        corner = [item for item in accepted if item["inference_type"] == "orthogonal_corner"]
        self.assertEqual(len(corner), 2)
        self.assertEqual(len({item["edge_group_id"] for item in corner}), 1)
        self.assertEqual({item["edge_group_size"] for item in corner}, {2})
        self.assertEqual(
            {(tuple(item["start_px"]), tuple(item["end_px"])) for item in corner},
            {((170, 20), (180, 20)), ((180, 20), (180, 30))},
        )

    def test_rejects_candidate_when_declared_inside_has_no_footprint_support(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        probabilities = np.zeros((10, 140, 200), dtype=np.float32)
        walls = [
            wall("top", (20, 20), (170, 20), "horizontal", "down"),
            wall("right", (180, 30), (180, 120), "vertical", "left"),
        ]

        candidates = generate_exterior_inference_candidates(
            walls, probabilities, (200, 140), [0, 0, 200, 140],
        )

        self.assertFalse(any(item["decision"] == "accepted_candidate" for item in candidates))
        self.assertIn(
            "insufficient_inside_footprint_support",
            {reason for item in candidates for reason in item["reason_codes"]},
        )

    def test_rejects_each_edge_longer_than_ten_percent_of_short_side(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("top", (20, 20), (120, 20), "horizontal", "down"),
            wall("right", (180, 30), (180, 120), "vertical", "left"),
        ]

        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [0, 0, 200, 140],
        )

        self.assertFalse(any(item["decision"] == "accepted_candidate" for item in candidates))
        self.assertIn(
            "inferred_edge_exceeds_short_side_limit",
            {reason for item in candidates for reason in item["reason_codes"]},
        )

    def test_generates_collinear_extension_between_dangling_endpoints(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("left-top", (20, 20), (80, 20), "horizontal", "down"),
            wall("right-top", (90, 20), (170, 20), "horizontal", "down"),
        ]

        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [0, 0, 200, 140],
        )

        accepted = [item for item in candidates if item["decision"] == "accepted_candidate"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["inference_type"], "collinear_extension")
        self.assertEqual(accepted[0]["start_px"], [80, 20])
        self.assertEqual(accepted[0]["end_px"], [90, 20])

    def test_connected_endpoint_does_not_generate_duplicate_candidate(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("left-top", (20, 20), (80, 20), "horizontal", "down"),
            wall("right-top", (90, 20), (170, 20), "horizontal", "down"),
            wall("connected", (80, 20), (80, 100), "vertical", "right"),
        ]

        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [0, 0, 200, 140],
        )

        self.assertFalse(any(
            item["decision"] == "accepted_candidate"
            and (item["start_px"] == [80, 20] or item["end_px"] == [80, 20])
            for item in candidates
        ))

    def test_parallel_double_wall_bands_never_mix_anchor_ids(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("outer-left", (20, 20), (80, 20), "horizontal", "down"),
            wall("outer-right", (90, 20), (170, 20), "horizontal", "down"),
            wall("inner-left", (20, 24), (80, 24), "horizontal", "down"),
            wall("inner-right", (90, 24), (170, 24), "horizontal", "down"),
        ]

        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [0, 0, 200, 140],
        )

        outer_ids = {"outer-left", "outer-right"}
        inner_ids = {"inner-left", "inner-right"}
        for candidate in candidates:
            anchors = set(candidate["anchor_wall_ids"])
            self.assertFalse(anchors & outer_ids and anchors & inner_ids)

    def test_rejects_corner_intersection_outside_roi(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("top", (20, 20), (170, 20), "horizontal", "down"),
            wall("right", (180, 30), (180, 120), "vertical", "left"),
        ]

        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [20, 25, 175, 130],
        )

        self.assertFalse(any(item["decision"] == "accepted_candidate" for item in candidates))
        self.assertIn(
            "inferred_intersection_outside_roi",
            {reason for item in candidates for reason in item["reason_codes"]},
        )

    def test_rejects_corner_with_conflicting_inside_directions(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("top", (20, 20), (170, 20), "horizontal", "down"),
            wall("wrong-side", (180, 30), (180, 120), "vertical", "right"),
        ]

        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [0, 0, 200, 140],
        )

        self.assertFalse(any(item["decision"] == "accepted_candidate" for item in candidates))
        self.assertIn(
            "incompatible_corner_inside_directions",
            {reason for item in candidates for reason in item["reason_codes"]},
        )


class FootprintGuidedTopologyTests(unittest.TestCase):
    def test_two_edge_corner_group_closes_supported_exterior(self):
        from vector_pdf_exterior import build_exterior_topology, enumerate_exterior_gaps

        walls = open_upper_right_fixture()
        candidates = [
            inferred(
                "corner-h", "corner-1", (170, 20), (180, 20),
                "horizontal", "down", group_size=2,
            ),
            inferred(
                "corner-v", "corner-1", (180, 20), (180, 30),
                "vertical", "left", group_size=2,
            ),
        ]

        topology = build_exterior_topology(
            walls,
            enumerate_exterior_gaps(walls, (200, 140), [0, 0, 200, 140]),
            [],
            rectangular_footprint(),
            (200, 140),
            [0, 0, 200, 140],
            inference_candidates=candidates,
        )

        self.assertEqual(topology["status"], "review_required")
        self.assertEqual(topology["closure_method"], "footprint_guided_inference")
        self.assertEqual(
            {edge["inference_id"] for edge in topology["inferred_edges"]},
            {"corner-h", "corner-v"},
        )
        self.assertAlmostEqual(topology["inferred_length_px"], 20.0)
        self.assertAlmostEqual(topology["inferred_perimeter_ratio"], 20.0 / 520.0)
        self.assertFalse(topology["load_geometry_ready"])

    def test_rejects_face_when_inferred_edges_exceed_twelve_percent(self):
        from vector_pdf_exterior import _evaluate_inferred_face

        candidates = {
            f"edge-{index}": inferred(
                f"edge-{index}", f"group-{index}",
                (index * 40, 20), (index * 40 + 30, 20),
                "horizontal", "down",
            )
            for index in range(5)
        }

        result = _evaluate_inferred_face(
            {
                "perimeter": 1000.0,
                "inferred_edge_ids": set(candidates),
                "edge_group_ids": {f"group-{index}" for index in range(5)},
            },
            candidates,
        )

        self.assertFalse(result["accepted"])
        self.assertEqual(result["inferred_perimeter_ratio"], 0.15)
        self.assertIn("inferred_perimeter_ratio_exceeded", result["reason_codes"])

    def test_never_uses_only_half_of_an_orthogonal_corner_group(self):
        from vector_pdf_exterior import build_exterior_topology

        only_horizontal = inferred(
            "corner-h", "corner-1", (170, 20), (180, 20),
            "horizontal", "down", group_size=2,
        )

        topology = build_exterior_topology(
            open_upper_right_fixture(),
            [],
            [],
            rectangular_footprint(),
            (200, 140),
            [0, 0, 200, 140],
            inference_candidates=[only_horizontal],
        )

        self.assertEqual(topology["status"], "exterior_not_closed")
        self.assertIn("incomplete_inferred_edge_group", topology["inference_reason_codes"])

    def test_rejects_near_equal_competing_inferred_polygons(self):
        from vector_pdf_exterior import _select_inferred_face

        first = {
            "polygon": [(20, 20), (180, 20), (180, 120), (20, 120)],
            "perimeter": 520.0,
            "inference_metrics": {
                "inferred_length_px": 20.0,
                "score": (20.0, 2, -0.85, -0.96, []),
            },
        }
        second = {
            "polygon": [(20, 24), (176, 24), (176, 116), (20, 116)],
            "perimeter": 496.0,
            "inference_metrics": {
                "inferred_length_px": 21.0,
                "score": (21.0, 2, -0.84, -0.95, []),
            },
        }

        selected, reasons = _select_inferred_face([first, second])

        self.assertIsNone(selected)
        self.assertEqual(reasons, ["competing_inferred_exterior"])


if __name__ == "__main__":
    unittest.main()
