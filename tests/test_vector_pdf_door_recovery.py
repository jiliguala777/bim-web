import unittest


def vertical_wall(candidate_id, start_y, end_y, *, x=40, accepted=True):
    return {
        "candidate_id": candidate_id,
        "decision": "accepted_wall_candidate" if accepted else "uncertain",
        "orientation": "vertical",
        "start_px": [x, start_y],
        "end_px": [x, end_y],
        "source_native_id": f"native-{candidate_id}",
        "inside_direction": "right" if accepted else None,
        "native_evidence": {
            "page_border": False,
            "roi_inside_ratio": 1.0,
        },
        "reason_codes": ["native_line"],
    }


def horizontal_wall(candidate_id, start_x, end_x, *, y=90, accepted=False):
    return {
        "candidate_id": candidate_id,
        "decision": "accepted_wall_candidate" if accepted else "uncertain",
        "orientation": "horizontal",
        "start_px": [start_x, y],
        "end_px": [end_x, y],
        "source_native_id": f"native-{candidate_id}",
        "inside_direction": "down" if accepted else None,
        "native_evidence": {
            "page_border": False,
            "roi_inside_ratio": 1.0,
        },
        "reason_codes": ["native_line"],
    }


def door_arc(curve_id="door-arc", *, wall_x=40, start_y=80, end_y=110):
    return {
        "curve_id": curve_id,
        "bbox_px": [wall_x - 30, start_y, wall_x, end_y],
        "path_start_px": [wall_x, start_y],
        "path_end_px": [wall_x - 30, end_y],
        "start_px": [wall_x, start_y],
        "end_px": [wall_x - 30, end_y],
        "path_ops": ["m", "c"],
        "has_bezier": True,
        "is_closed": False,
    }


class ExteriorDoorArcRecoveryTests(unittest.TestCase):
    def test_recovers_native_wall_chain_between_two_exterior_anchors(self):
        from vector_pdf_door_recovery import recover_exterior_walls_from_door_arcs

        upper_anchor = vertical_wall("upper-anchor", 20, 60)
        upper_uncertain = vertical_wall("upper-uncertain", 60, 80, accepted=False)
        lower_uncertain = vertical_wall("lower-uncertain", 110, 140, accepted=False)
        lower_anchor = vertical_wall("lower-anchor", 140, 180)

        result = recover_exterior_walls_from_door_arcs(
            candidates=[upper_anchor, upper_uncertain, lower_uncertain, lower_anchor],
            exterior_walls=[upper_anchor, lower_anchor],
            curve_edges=[door_arc()],
            image_size=(300, 300),
            building_roi=[20, 20, 280, 280],
        )

        self.assertEqual(
            [wall["candidate_id"] for wall in result["recovered_walls"]],
            ["lower-uncertain", "upper-uncertain"],
        )
        self.assertTrue(all(
            wall["recovery_method"] == "exterior_door_arc_chain"
            for wall in result["recovered_walls"]
        ))
        self.assertEqual(len(result["confirmed_door_arcs"]), 1)
        confirmed = result["confirmed_door_arcs"][0]
        self.assertEqual(confirmed["orientation"], "vertical")
        self.assertEqual(confirmed["projected_start_px"], [40, 80])
        self.assertEqual(confirmed["projected_end_px"], [40, 110])
        self.assertTrue(confirmed["exterior_recovery_approved"])

    def test_interior_arc_far_from_exterior_band_is_ignored(self):
        from vector_pdf_door_recovery import recover_exterior_walls_from_door_arcs

        upper = vertical_wall("upper", 20, 60)
        lower = vertical_wall("lower", 140, 180)
        result = recover_exterior_walls_from_door_arcs(
            [upper, lower], [upper, lower],
            [door_arc("interior", wall_x=150)], (300, 300), [20, 20, 280, 280],
        )

        self.assertEqual(result["recovered_walls"], [])
        self.assertEqual(result["confirmed_door_arcs"], [])
        self.assertEqual(result["pending_door_arcs"], [])

    def test_closed_curve_never_becomes_door_arc(self):
        from vector_pdf_door_recovery import recover_exterior_walls_from_door_arcs

        upper = vertical_wall("upper", 20, 80)
        lower = vertical_wall("lower", 110, 180)
        curve = door_arc("closed")
        curve["is_closed"] = True
        result = recover_exterior_walls_from_door_arcs(
            [upper, lower], [upper, lower], [curve], (300, 300), None,
        )

        self.assertEqual(result["confirmed_door_arcs"], [])

    def test_single_anchor_keeps_arc_pending_without_recovering_wall(self):
        from vector_pdf_door_recovery import recover_exterior_walls_from_door_arcs

        upper = vertical_wall("upper", 20, 80)
        lower_uncertain = vertical_wall("lower-uncertain", 110, 150, accepted=False)
        result = recover_exterior_walls_from_door_arcs(
            [upper, lower_uncertain], [upper], [door_arc()], (300, 300), None,
        )

        self.assertEqual(result["recovered_walls"], [])
        self.assertEqual(result["confirmed_door_arcs"], [])
        self.assertEqual(result["pending_door_arcs"][0]["status"], "pending")

    def test_cross_axis_uncertain_wall_cannot_complete_vertical_chain(self):
        from vector_pdf_door_recovery import recover_exterior_walls_from_door_arcs

        upper = vertical_wall("upper", 20, 60)
        lower = vertical_wall("lower", 140, 180)
        cross_axis = horizontal_wall("cross-axis", 40, 140)
        result = recover_exterior_walls_from_door_arcs(
            [upper, cross_axis, lower], [upper, lower], [door_arc()],
            (300, 300), None,
        )

        self.assertEqual(result["recovered_walls"], [])
        self.assertEqual(result["confirmed_door_arcs"], [])

    def test_oversized_arc_is_rejected(self):
        from vector_pdf_door_recovery import recover_exterior_walls_from_door_arcs

        upper = vertical_wall("upper", 10, 20)
        lower = vertical_wall("lower", 190, 220)
        oversized = door_arc("oversized", start_y=20, end_y=190)
        result = recover_exterior_walls_from_door_arcs(
            [upper, lower], [upper, lower], [oversized], (300, 300), None,
        )

        self.assertEqual(result["confirmed_door_arcs"], [])
        self.assertEqual(result["pending_door_arcs"], [])


if __name__ == "__main__":
    unittest.main()
