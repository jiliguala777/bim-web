import unittest

from design_load_calc import calculate_design_loads


class DesignLoadCalcTests(unittest.TestCase):
    def _base_params(self):
        return {
            "geometry": {
                "floor_area_m2": 100.0,
                "wall_area_m2": 50.0,
                "window_area_m2": 10.0,
                "roof_area_m2": 100.0,
                "door_area_m2": 0.0,
            },
            "building": {"floors": 1, "height": 3.0, "building_type": "office"},
            "envelope": {
                "u_wall": 0.6,
                "u_window": 2.5,
                "u_roof": 0.4,
                "u_floor": 0.0,
                "u_door": 0.0,
                "shgc": 0.4,
                "peak_solar_irradiance_w_m2": 500.0,
                "solar_orientation_factor": 1.0,
                "curtain_shading": 1.0,
            },
            "heating": {
                "enabled": True,
                "t_set": 18.0,
                "outdoor_design_temperature_c": -5.0,
            },
            "cooling": {
                "enabled": True,
                "t_set": 26.0,
                "outdoor_design_temperature_c": 34.0,
            },
            "occupancy": {
                "area_per_person_m2": 10.0,
                "sensible_heat_w_per_person": 75.0,
                "latent_heat_w_per_person": 55.0,
                "cooling_load_factor": 1.0,
            },
            "ventilation": {
                "fresh_air_m3h_per_person": 30.0,
                "infiltration_ach": 0.0,
                "summer_outdoor_enthalpy_kj_kg": 85.0,
                "summer_indoor_enthalpy_kj_kg": 55.0,
                "air_density_kg_m3": 1.13,
            },
            "lighting": {"lpd": 8.0, "control_factor": 1.0},
            "equipment": {"epd": 15.0},
        }

    def test_cooling_design_load_sums_envelope_solar_fresh_air_and_internal_gains(self):
        result = calculate_design_loads(self._base_params())
        cooling = result["cooling_design_load"]

        self.assertEqual(cooling["enabled"], True)
        self.assertAlmostEqual(cooling["components_w"]["envelope_transmission"], 760.0)
        self.assertAlmostEqual(cooling["components_w"]["window_solar"], 2000.0)
        self.assertAlmostEqual(cooling["components_w"]["fresh_air_sensible"], 804.0)
        self.assertAlmostEqual(cooling["components_w"]["fresh_air_latent"], 2825.0)
        self.assertAlmostEqual(cooling["components_w"]["people"], 1300.0)
        self.assertAlmostEqual(cooling["components_w"]["lighting"], 800.0)
        self.assertAlmostEqual(cooling["components_w"]["equipment"], 1500.0)
        self.assertAlmostEqual(cooling["total_w"], 9989.0)
        self.assertAlmostEqual(cooling["cooling_index_w_m2"], 99.89)

    def test_cooling_latent_load_can_derive_enthalpy_from_temperature_and_relative_humidity(self):
        params = self._base_params()
        params["ventilation"].pop("summer_outdoor_enthalpy_kj_kg")
        params["ventilation"].pop("summer_indoor_enthalpy_kj_kg")
        params["ventilation"]["summer_outdoor_relative_humidity_percent"] = 60.0
        params["ventilation"]["summer_indoor_relative_humidity_percent"] = 50.0
        result = calculate_design_loads(params)

        derived = result["derived_inputs"]
        self.assertAlmostEqual(derived["summer_outdoor_enthalpy_kj_kg"], 86.069, places=3)
        self.assertAlmostEqual(derived["summer_indoor_enthalpy_kj_kg"], 52.899, places=3)
        self.assertAlmostEqual(
            result["cooling_design_load"]["components_w"]["fresh_air_latent"],
            3123.484,
            places=3,
        )

    def test_heating_design_load_uses_losses_and_only_deducts_configured_stable_gains(self):
        params = self._base_params()
        params["heating"]["stable_internal_gain_fraction"] = 0.5
        result = calculate_design_loads(params)
        heating = result["heating_design_load"]

        self.assertEqual(heating["enabled"], True)
        self.assertAlmostEqual(heating["components_w"]["envelope_basic"], 2185.0)
        self.assertAlmostEqual(heating["components_w"]["fresh_air"], 2311.5)
        self.assertAlmostEqual(heating["components_w"]["stable_internal_gain_deduction"], 1800.0)
        self.assertAlmostEqual(heating["total_w"], 2696.5)
        self.assertAlmostEqual(heating["heating_index_w_m2"], 26.965)

    def test_include_floor_adds_corrected_floor_area_to_design_envelope_ua(self):
        params = self._base_params()
        params["geometry"]["include_floor"] = True
        params["geometry"]["floor_loss_area_m2"] = 999.0
        params["envelope"]["u_floor"] = 0.3
        params["envelope"]["floor_contact_factor"] = 0.5

        result = calculate_design_loads(params)

        self.assertAlmostEqual(result["derived_inputs"]["floor_loss_area_m2"], 50.0)
        self.assertAlmostEqual(result["derived_inputs"]["envelope_ua_w_k"], 110.0)
        self.assertAlmostEqual(
            result["heating_design_load"]["components_w"]["envelope_basic"],
            2530.0,
        )

    def test_other_parameters_change_design_load_components(self):
        params = self._base_params()
        params["heating"]["heating_addition_factor"] = 1.1
        params["heating"]["door_invasion_heat_w"] = 500.0
        params["occupancy"]["cooling_load_factor"] = 0.5
        params["lighting"]["cooling_load_factor"] = 0.8
        params["equipment"]["cooling_load_factor"] = 0.6
        params["ventilation"]["air_density_kg_m3"] = 1.2

        result = calculate_design_loads(params)
        cooling = result["cooling_design_load"]
        heating = result["heating_design_load"]

        self.assertAlmostEqual(cooling["components_w"]["fresh_air_latent"], 3000.0)
        self.assertAlmostEqual(cooling["components_w"]["people"], 925.0)
        self.assertAlmostEqual(cooling["components_w"]["lighting"], 640.0)
        self.assertAlmostEqual(cooling["components_w"]["equipment"], 900.0)
        self.assertAlmostEqual(heating["components_w"]["door_invasion"], 500.0)
        self.assertAlmostEqual(heating["components_w"]["heating_addition"], 499.65)
        self.assertAlmostEqual(heating["total_w"], 5496.15)
        self.assertAlmostEqual(result["derived_inputs"]["air_density_kg_m3"], 1.2)

    def test_disabled_scope_returns_null_load_section(self):
        params = self._base_params()
        params["cooling"]["enabled"] = False

        result = calculate_design_loads(params)

        self.assertIsNone(result["cooling_design_load"])
        self.assertIsNotNone(result["heating_design_load"])
        self.assertEqual(result["calculation_scope"], {"heating": True, "cooling": False})

    def test_invalid_geometry_raises_clear_error(self):
        params = self._base_params()
        params["geometry"]["floor_area_m2"] = 0

        with self.assertRaisesRegex(ValueError, "floor_area_m2"):
            calculate_design_loads(params)


if __name__ == "__main__":
    unittest.main()
