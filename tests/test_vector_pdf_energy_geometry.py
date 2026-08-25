import unittest


def footprint(*, area, perimeter):
    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "confirmed": False,
        "load_geometry_ready": False,
    }


def scaled_footprint(*, area, perimeter):
    return {
        "area_m2": area,
        "perimeter_m": perimeter,
        "confirmed": False,
        "load_geometry_ready": False,
    }


def opening(kind, width):
    return {"kind": kind, "width_px": width}


def scaled_openings(*, door, window):
    return [
        {"kind": "door", "width_m": door},
        {"kind": "window", "width_m": window},
    ]


class ExteriorEnergyGeometryTests(unittest.TestCase):
    def test_applies_scale_to_footprint_and_separate_opening_widths(self):
        from vector_pdf_energy_geometry import apply_scale_to_exterior

        source_topology = footprint(area=10000, perimeter=400)
        source_openings = [opening("door", 30), opening("window", 50)]
        topology, openings = apply_scale_to_exterior(
            source_topology, source_openings, 0.02,
        )

        self.assertEqual(topology["area_m2"], 4.0)
        self.assertEqual(topology["perimeter_m"], 8.0)
        self.assertEqual(openings[0]["width_m"], 0.6)
        self.assertEqual(openings[1]["width_m"], 1.0)
        self.assertEqual(source_topology, footprint(area=10000, perimeter=400))
        self.assertEqual(source_openings, [opening("door", 30), opening("window", 50)])
        openings[0]["nested"] = {"changed": True}
        self.assertNotIn("nested", source_openings[0])

    def test_builds_opaque_wall_after_separate_door_window_deductions(self):
        from vector_pdf_energy_geometry import build_exterior_energy_geometry

        geometry = build_exterior_energy_geometry(
            scaled_footprint(area=100, perimeter=40), scaled_openings(door=2, window=8),
            storey_height_m=3, floors=2, door_height_m=2.1, window_height_m=1.5,
            door_repeat_count=1, window_repeat_count=2,
        )

        self.assertEqual(geometry["per_floor_footprint_area_m2"], 100.0)
        self.assertEqual(geometry["total_floor_area_m2"], 200.0)
        self.assertEqual(geometry["exterior_perimeter_m"], 40.0)
        self.assertEqual(geometry["gross_exterior_wall_area_m2"], 240.0)
        self.assertEqual(geometry["door_total_width_m"], 2.0)
        self.assertEqual(geometry["window_total_width_m"], 8.0)
        self.assertEqual(geometry["door_area_m2"], 4.2)
        self.assertEqual(geometry["window_area_m2"], 24.0)
        self.assertEqual(geometry["wall_area_m2"], 211.8)

    def test_defaults_window_repeat_count_to_floor_count(self):
        from vector_pdf_energy_geometry import build_exterior_energy_geometry

        geometry = build_exterior_energy_geometry(
            scaled_footprint(area=100, perimeter=40), scaled_openings(door=2, window=8),
            storey_height_m=3, floors=2, door_height_m=2.1, window_height_m=1.5,
        )

        self.assertEqual(geometry["door_area_m2"], 4.2)
        self.assertEqual(geometry["window_area_m2"], 24.0)

    def test_rejects_non_finite_zero_or_boolean_scale(self):
        from vector_pdf_energy_geometry import apply_scale_to_exterior

        for scale in (0, -0.01, float("nan"), float("inf"), True):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    apply_scale_to_exterior(footprint(area=100, perimeter=40), [], scale)

    def test_rejects_invalid_heights_floors_and_repeat_counts(self):
        from vector_pdf_energy_geometry import build_exterior_energy_geometry

        valid_topology = scaled_footprint(area=100, perimeter=40)
        valid_openings = scaled_openings(door=2, window=8)
        invalid_arguments = [
            {"storey_height_m": 0},
            {"door_height_m": float("nan")},
            {"window_height_m": True},
            {"floors": 0},
            {"floors": 2.5},
            {"door_repeat_count": 3},
            {"window_repeat_count": 3},
            {"door_repeat_count": True},
        ]
        for invalid in invalid_arguments:
            with self.subTest(invalid=invalid):
                arguments = {
                    "storey_height_m": 3,
                    "floors": 2,
                    "door_height_m": 2.1,
                    "window_height_m": 1.5,
                }
                arguments.update(invalid)
                with self.assertRaises(ValueError):
                    build_exterior_energy_geometry(valid_topology, valid_openings, **arguments)

    def test_rejects_openings_that_exceed_gross_exterior_wall_area(self):
        from vector_pdf_energy_geometry import build_exterior_energy_geometry

        with self.assertRaisesRegex(ValueError, "gross exterior wall area"):
            build_exterior_energy_geometry(
                scaled_footprint(area=1, perimeter=4), scaled_openings(door=10, window=1),
                storey_height_m=3, floors=1, door_height_m=3, window_height_m=1,
            )


if __name__ == "__main__":
    unittest.main()
