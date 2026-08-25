import unittest

import numpy as np


def gap(start, end, orientation="horizontal", *, decision="accepted_gap"):
    return {
        "gap_id": "gap-0001",
        "orientation": orientation,
        "start_px": list(start),
        "end_px": list(end),
        "width_px": float(abs(end[0] - start[0]) + abs(end[1] - start[1])),
        "host_wall_ids": ["wall-a", "wall-b"],
        "inside_direction": "down" if orientation == "horizontal" else "right",
        "decision": decision,
        "reason_codes": ["unambiguous_collinear_exterior_gap"],
    }


def empty_probabilities():
    return np.zeros((10, 100, 160), dtype=np.float32)


def door_line_only_probabilities():
    probabilities = empty_probabilities()
    probabilities[5, 48:53, 60:101] = 0.9
    return probabilities


class ExteriorOpeningClassificationTests(unittest.TestCase):
    def test_classifies_door_from_line_and_both_endpoints(self):
        from vector_pdf_openings import classify_exterior_openings

        probabilities = empty_probabilities()
        probabilities[5, 48:53, 60:101] = 0.9
        probabilities[8, 45:56, 55:66] = 0.9
        probabilities[8, 45:56, 95:106] = 0.9

        result = classify_exterior_openings(
            [gap((60, 50), (100, 50))], probabilities, (160, 100),
        )

        opening = result["accepted_openings"][0]
        self.assertEqual(opening["kind"], "door")
        self.assertEqual(opening["width_px"], 40.0)
        self.assertEqual(opening["start_px"], [60, 50])
        self.assertEqual(opening["end_px"], [100, 50])
        self.assertEqual(opening["host_wall_ids"], ["wall-a", "wall-b"])
        self.assertTrue(opening["exterior"])

    def test_line_without_two_endpoint_support_is_unclassified(self):
        from vector_pdf_openings import classify_exterior_openings

        result = classify_exterior_openings(
            [gap((60, 50), (100, 50))], door_line_only_probabilities(), (160, 100),
        )

        self.assertEqual(result["accepted_openings"], [])
        self.assertEqual(result["unclassified_gaps"][0]["gap_id"], "gap-0001")

    def test_classifies_window_from_line_and_both_endpoints(self):
        from vector_pdf_openings import classify_exterior_openings

        probabilities = empty_probabilities()
        probabilities[6, 48:53, 60:101] = 0.9
        probabilities[9, 45:56, 55:66] = 0.9
        probabilities[9, 45:56, 95:106] = 0.9

        result = classify_exterior_openings(
            [gap((60, 50), (100, 50))], probabilities, (160, 100),
        )

        self.assertEqual(result["accepted_openings"][0]["kind"], "window")

    def test_close_door_and_window_scores_are_ambiguous(self):
        from vector_pdf_openings import classify_exterior_openings

        probabilities = empty_probabilities()
        probabilities[5, 48:53, 60:101] = 0.60
        probabilities[8, 45:56, 55:66] = 0.60
        probabilities[8, 45:56, 95:106] = 0.60
        probabilities[6, 48:53, 60:101] = 0.58
        probabilities[9, 45:56, 55:66] = 0.58
        probabilities[9, 45:56, 95:106] = 0.58

        result = classify_exterior_openings(
            [gap((60, 50), (100, 50))], probabilities, (160, 100),
        )

        self.assertEqual(result["accepted_openings"], [])
        self.assertEqual(result["ambiguous_openings"][0]["kind"], "ambiguous")
        self.assertEqual(result["ambiguous_openings"][0]["width_px"], 40.0)

    def test_classifies_vertical_gap_without_swapping_geometry(self):
        from vector_pdf_openings import classify_exterior_openings

        probabilities = empty_probabilities()
        probabilities[5, 40:81, 48:53] = 0.9
        probabilities[8, 35:46, 45:56] = 0.9
        probabilities[8, 75:86, 45:56] = 0.9

        result = classify_exterior_openings(
            [gap((50, 40), (50, 80), "vertical")], probabilities, (160, 100),
        )

        opening = result["accepted_openings"][0]
        self.assertEqual(opening["orientation"], "vertical")
        self.assertEqual(opening["start_px"], [50, 40])
        self.assertEqual(opening["end_px"], [50, 80])

    def test_border_clipped_gap_uses_only_in_image_evidence(self):
        from vector_pdf_openings import classify_exterior_openings

        probabilities = empty_probabilities()
        probabilities[5, 0:5, 0:21] = 0.9
        probabilities[8, 0:9, 0:9] = 0.9
        probabilities[8, 0:9, 15:26] = 0.9

        result = classify_exterior_openings(
            [gap((-10, 2), (20, 2))], probabilities, (160, 100),
        )

        opening = result["accepted_openings"][0]
        self.assertEqual(opening["start_px"], [-10, 2])
        self.assertEqual(opening["end_px"], [20, 2])

    def test_fully_out_of_image_gap_is_unclassified(self):
        from vector_pdf_openings import classify_exterior_openings

        result = classify_exterior_openings(
            [gap((200, 50), (240, 50))], np.ones((10, 100, 160), dtype=np.float32), (160, 100),
        )

        self.assertEqual(result["accepted_openings"], [])
        self.assertEqual(result["ambiguous_openings"], [])
        self.assertEqual(result["unclassified_gaps"][0]["gap_id"], "gap-0001")

    def test_rejected_gap_never_becomes_an_opening(self):
        from vector_pdf_openings import classify_exterior_openings

        probabilities = empty_probabilities()
        probabilities[5, 48:53, 60:101] = 0.9
        probabilities[8, 45:56, 55:66] = 0.9
        probabilities[8, 45:56, 95:106] = 0.9

        result = classify_exterior_openings(
            [gap((60, 50), (100, 50), decision="rejected_gap")], probabilities, (160, 100),
        )

        self.assertEqual(result["accepted_openings"], [])
        self.assertEqual(result["ambiguous_openings"], [])
        self.assertEqual(result["unclassified_gaps"], [])


if __name__ == "__main__":
    unittest.main()
