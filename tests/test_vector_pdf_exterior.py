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


if __name__ == "__main__":
    unittest.main()
