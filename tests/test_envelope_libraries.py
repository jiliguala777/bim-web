import unittest

from energy_library import list_envelope_parameters


class EnvelopeLibraryTests(unittest.TestCase):
    def test_required_new_envelope_categories_have_examples(self):
        required_categories = {
            "door_u": "W/m²·K",
            "floor_contact_type": "",
            "window_air_tightness": "",
        }

        for category, expected_unit in required_categories.items():
            with self.subTest(category=category):
                items = list_envelope_parameters(category)
                self.assertGreaterEqual(len(items), 3)
                self.assertTrue(all(item["category"] == category for item in items))
                self.assertTrue(any(item["unit"] == expected_unit for item in items))


if __name__ == "__main__":
    unittest.main()
