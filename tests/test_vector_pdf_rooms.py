import unittest

import numpy as np


def accepted_line(line_id, start, end):
    orientation = "horizontal" if start[1] == end[1] else "vertical"
    return {
        "candidate_id": line_id,
        "source_native_id": f"native-{line_id}",
        "orientation": orientation,
        "start_px": list(start),
        "end_px": list(end),
        "decision": "accepted_wall_candidate",
    }


def accepted_rectangle_lines():
    return [
        accepted_line("wall-1", (20, 20), (80, 20)),
        accepted_line("wall-2", (80, 20), (80, 60)),
        accepted_line("wall-3", (20, 60), (80, 60)),
        accepted_line("wall-4", (20, 20), (20, 60)),
    ]


def supported_probabilities(width=100, height=100):
    probabilities = np.zeros((10, height, width), dtype=np.float32)
    probabilities[0, :, :] = 0.8
    probabilities[2, :, :] = 0.7
    return probabilities


class OrthogonalRoomTests(unittest.TestCase):
    def test_exact_rectangle_forms_traceable_room(self):
        from vector_pdf_rooms import RoomClosureThresholds, find_room_candidates

        result = find_room_candidates(
            accepted_rectangle_lines(),
            supported_probabilities(),
            (100, 100),
            [0, 0, 100, 100],
            RoomClosureThresholds(),
        )

        self.assertEqual(result["summary"]["accepted_room_count"], 1)
        room = result["room_candidates"][0]
        self.assertEqual(room["decision"], "accepted_room_candidate")
        self.assertFalse(room["closure_applied"])
        self.assertEqual(
            set(room["source_wall_ids"]),
            {"wall-1", "wall-2", "wall-3", "wall-4"},
        )
        self.assertEqual(room["area_px2"], 2400.0)

    def test_small_orthogonal_gap_is_snapped_and_larger_gap_stays_open(self):
        from vector_pdf_rooms import RoomClosureThresholds, find_room_candidates

        def gapped_lines(gap):
            return [
                accepted_line("wall-1", (200, 200), (800, 200)),
                accepted_line("wall-2", (800, 200), (800, 600)),
                accepted_line("wall-3", (200, 600), (800 - gap, 600)),
                accepted_line("wall-4", (200, 200), (200, 600)),
            ]

        probabilities = supported_probabilities(1000, 1000)
        closed = find_room_candidates(
            gapped_lines(3), probabilities, (1000, 1000),
            [0, 0, 1000, 1000], RoomClosureThresholds(),
        )
        open_result = find_room_candidates(
            gapped_lines(4), probabilities, (1000, 1000),
            [0, 0, 1000, 1000], RoomClosureThresholds(),
        )

        self.assertEqual(closed["summary"]["accepted_room_count"], 1)
        self.assertTrue(closed["room_candidates"][0]["closure_applied"])
        self.assertEqual(len(closed["snaps"]), 1)
        self.assertEqual(closed["snaps"][0]["distance_px"], 3.0)
        self.assertEqual(closed["snaps"][0]["snapped_point_px"], [800, 600])
        self.assertEqual(
            set(closed["snaps"][0]["source_wall_ids"]),
            {"wall-2", "wall-3"},
        )
        self.assertEqual(open_result["summary"]["accepted_room_count"], 0)

    def test_closed_face_with_weak_room_probability_is_suspicious(self):
        from vector_pdf_rooms import RoomClosureThresholds, find_room_candidates

        probabilities = supported_probabilities()
        probabilities[2, :, :] = 0.0
        result = find_room_candidates(
            accepted_rectangle_lines(), probabilities, (100, 100),
            [0, 0, 100, 100], RoomClosureThresholds(),
        )

        self.assertEqual(result["summary"]["accepted_room_count"], 0)
        self.assertEqual(result["summary"]["suspicious_room_count"], 1)
        self.assertEqual(result["room_candidates"][0]["decision"], "suspicious_closed_region")
        self.assertIn("weak_room_model_support", result["room_candidates"][0]["reason_codes"])

    def test_unaccepted_boundary_cannot_form_room(self):
        from vector_pdf_rooms import RoomClosureThresholds, find_room_candidates

        lines = accepted_rectangle_lines()
        lines[2]["decision"] = "uncertain"
        result = find_room_candidates(
            lines, supported_probabilities(), (100, 100),
            [0, 0, 100, 100], RoomClosureThresholds(),
        )

        self.assertEqual(result["room_candidates"], [])

    def test_shared_wall_produces_two_minimal_rooms_not_one_outer_union(self):
        from vector_pdf_rooms import RoomClosureThresholds, find_room_candidates

        lines = accepted_rectangle_lines()
        lines.append(accepted_line("wall-5", (50, 20), (50, 60)))
        result = find_room_candidates(
            lines, supported_probabilities(), (100, 100),
            [0, 0, 100, 100], RoomClosureThresholds(),
        )

        self.assertEqual(result["summary"]["accepted_room_count"], 2)
        self.assertEqual(
            sorted(room["area_px2"] for room in result["room_candidates"]),
            [1200.0, 1200.0],
        )


if __name__ == "__main__":
    unittest.main()
