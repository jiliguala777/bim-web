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


if __name__ == "__main__":
    unittest.main()
