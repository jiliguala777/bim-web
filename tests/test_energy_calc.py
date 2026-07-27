import unittest

from energy_calc import calculate_energy


class EnergyCalcTests(unittest.TestCase):
    def _zero_internal_load_params(self):
        return {
            "calculation_mode": "detailed",
            "geometry": {
                "floor_area_m2": 100,
                "wall_area_m2": 0,
                "window_area_m2": 0,
                "roof_area_m2": 0,
                "door_area_m2": 0,
            },
            "building": {"floors": 1, "height": 3.0, "building_type": "office"},
            "envelope": {
                "u_wall": 0.0,
                "u_window": 0.0,
                "u_roof": 0.0,
                "u_floor": 0.0,
                "u_door": 0.0,
                "shgc": 0.0,
            },
            "climate": {"city_id": "shanghai"},
            "heating": {"system_type": "gas_boiler", "t_set": 18.0},
            "cooling": {"system_type": "central_chiller", "t_set": 26.0},
            "lighting": {"lpd": 0, "control_factor": 1.0},
            "ventilation": {"ach": 0, "fan_power": 0},
            "dhw": {"occupants": 1, "daily_liter_pp": 0, "efficiency": 1.0},
            "equipment": {"epd": 0},
            "schedule": {"op_hours": 0},
        }

    def _annual_model_params(self):
        params = self._zero_internal_load_params()
        params.update({
            "calculation_mode": "simple",
            "geometry": {
                "floor_area_m2": 100,
                "wall_area_m2": 100,
                "window_area_m2": 0,
                "roof_area_m2": 0,
                "door_area_m2": 0,
            },
            "building": {"floors": 1, "height": 3.0, "building_type": "office"},
            "envelope": {
                "u_wall": 1.0,
                "u_window": 0.0,
                "u_roof": 0.0,
                "u_floor": 0.0,
                "u_door": 0.0,
                "shgc": 0.0,
            },
            "climate": {"city_id": "xian"},
            "heating": {
                "system_type": "heat_pump_air",
                "t_set": 18.0,
                "seasonal_efficiency": 2.0,
                "outdoor_design_temperature_c": -5.0,
            },
            "cooling": {
                "system_type": "vrv",
                "t_set": 26.0,
                "seasonal_efficiency": 2.5,
                "outdoor_design_temperature_c": 34.9,
            },
            "occupancy": {"area_per_person_m2": 10.0},
            "ventilation": {
                "fresh_air_m3h_per_person": 30.0,
                "heat_recovery_efficiency": 0.0,
                "infiltration_ach": 0.0,
                "fan_power_w_per_m3h": 0.5,
            },
            "lighting": {"lpd": 8.0, "control_factor": 1.0},
            "equipment": {"epd": 15.0},
            "schedule": {"daily_operation_hours": 12.0, "annual_operation_days": 250},
            "dhw": {"occupants": 10, "daily_liter_pp": 0, "efficiency": 1.0},
        })
        return params

    def test_annual_model_derives_people_fresh_air_and_hours(self):
        result = calculate_energy(self._annual_model_params())
        derived = result["derived_inputs"]
        self.assertEqual(derived["occupants"], 10.0)
        self.assertEqual(derived["fresh_air_flow_m3h"], 300.0)
        self.assertEqual(derived["annual_operation_hours"], 3000.0)
        self.assertAlmostEqual(derived["operation_fraction"], 3000 / 8760, places=6)

    def test_annual_model_adds_scheduled_fresh_air_sensible_demand(self):
        result = calculate_energy(self._annual_model_params())
        expected = 0.335 * 300 * 2400 * 24 / 1000 * (3000 / 8760)
        self.assertAlmostEqual(
            result["annual_thermal_demand_kwh_th"]["heating_fresh_air"],
            expected,
            places=3,
        )

    def test_thermal_demand_is_separate_from_equipment_electricity(self):
        result = calculate_energy(self._annual_model_params())
        thermal = result["annual_thermal_demand_kwh_th"]
        electric = result["annual_electricity_kwh"]
        self.assertAlmostEqual(electric["heating"], thermal["heating_total"] / 2.0, places=3)
        self.assertAlmostEqual(electric["cooling"], thermal["cooling_total"] / 2.5, places=3)
        self.assertAlmostEqual(electric["fan"], 300 * 0.5 * 3000 / 1000, places=3)
        self.assertAlmostEqual(
            electric["total"],
            electric["heating"] + electric["cooling"] + electric["fan"]
            + electric["lighting"] + electric["equipment"] + electric["dhw"],
            places=3,
        )

    def test_heating_only_scope_skips_all_cooling_loads_and_energy(self):
        params = self._annual_model_params()
        params["heating"]["enabled"] = True
        params["cooling"]["enabled"] = False

        result = calculate_energy(params)

        self.assertEqual(result["calculation_scope"], {"heating": True, "cooling": False})
        thermal = result["annual_thermal_demand_kwh_th"]
        self.assertGreater(thermal["heating_total"], 0)
        for key in (
            "cooling_total",
            "cooling_envelope",
            "cooling_fresh_air_sensible",
            "cooling_infiltration",
        ):
            self.assertIsNone(thermal[key], key)
        self.assertEqual(result["annual_electricity_kwh"]["cooling"], 0.0)
        self.assertIsNone(result["equivalent_peak_check_kw_th"]["cooling"])
        self.assertIsNone(
            result["equivalent_peak_check_kw_th"]["cooling_equivalent_full_load_hours"]
        )

    def test_cooling_only_scope_skips_all_heating_loads_and_energy(self):
        params = self._annual_model_params()
        params["heating"]["enabled"] = False
        params["cooling"]["enabled"] = True

        result = calculate_energy(params)

        self.assertEqual(result["calculation_scope"], {"heating": False, "cooling": True})
        thermal = result["annual_thermal_demand_kwh_th"]
        self.assertGreater(thermal["cooling_total"], 0)
        for key in (
            "heating_total",
            "heating_envelope",
            "heating_fresh_air",
            "heating_infiltration",
        ):
            self.assertIsNone(thermal[key], key)
        self.assertEqual(result["annual_electricity_kwh"]["heating"], 0.0)
        self.assertEqual(result["annual_non_electric_purchased_energy_kwh"]["heating"], 0.0)
        self.assertIsNone(result["equivalent_peak_check_kw_th"]["heating"])
        self.assertIsNone(
            result["equivalent_peak_check_kw_th"]["heating_equivalent_full_load_hours"]
        )

    def test_missing_scope_flags_keep_both_calculations_enabled(self):
        result = calculate_energy(self._annual_model_params())

        self.assertEqual(result["calculation_scope"], {"heating": True, "cooling": True})
        self.assertGreater(result["annual_thermal_demand_kwh_th"]["heating_total"], 0)
        self.assertGreater(result["annual_thermal_demand_kwh_th"]["cooling_total"], 0)

    def test_calculation_scope_rejects_both_heating_and_cooling_disabled(self):
        params = self._annual_model_params()
        params["heating"]["enabled"] = False
        params["cooling"]["enabled"] = False

        with self.assertRaisesRegex(ValueError, "供暖或制冷"):
            calculate_energy(params)

    def test_heat_recovery_reduces_fresh_air_thermal_demand_not_fan_flow(self):
        base = self._annual_model_params()
        recovered = self._annual_model_params()
        recovered["ventilation"]["heat_recovery_efficiency"] = 0.5
        base_result = calculate_energy(base)
        recovered_result = calculate_energy(recovered)
        self.assertAlmostEqual(
            recovered_result["annual_thermal_demand_kwh_th"]["heating_fresh_air"],
            base_result["annual_thermal_demand_kwh_th"]["heating_fresh_air"] * 0.5,
            places=3,
        )
        self.assertEqual(
            recovered_result["annual_electricity_kwh"]["fan"],
            base_result["annual_electricity_kwh"]["fan"],
        )

    def test_explicit_zero_infiltration_does_not_add_window_air_tightness_infiltration(self):
        params = self._annual_model_params()
        params["calculation_mode"] = "detailed"
        params["detailed_envelope"] = {
            "walls_by_orientation": {"north": {"area_m2": 100, "u_value": 1.0}},
            "windows_by_orientation": {
                "north": {"area_m2": 10, "u_value": 0, "shgc": 0, "air_tightness_value": 4}
            },
        }
        result = calculate_energy(params)
        self.assertEqual(result["annual_thermal_demand_kwh_th"]["heating_infiltration"], 0.0)
        self.assertEqual(result["detailed_loads"]["infiltration_ua"], 0.0)

    def test_hdd_peak_check_uses_indoor_setpoint_and_design_outdoor_temperature(self):
        result = calculate_energy(self._annual_model_params())
        peak = result["equivalent_peak_check_kw_th"]
        expected_hours = 2400 * 24 / (18 - (-5))
        expected_peak = (100 + 0.335 * 300) * (18 - (-5)) / 1000
        self.assertAlmostEqual(peak["heating_equivalent_full_load_hours"], expected_hours, places=3)
        self.assertAlmostEqual(peak["heating"], expected_peak, places=3)

    def test_legacy_params_without_new_fields_still_calculate(self):
        result = calculate_energy(self._zero_internal_load_params())
        self.assertIn("summary", result)
        self.assertIn("annual_thermal_demand_kwh_th", result)
        self.assertIn("annual_electricity_kwh", result)

    def test_breakdown_kwh_is_electricity_only_compatibility_alias(self):
        result = calculate_energy(self._annual_model_params())
        electric = result["annual_electricity_kwh"]
        self.assertEqual(result["breakdown_kwh"]["heating"], round(electric["heating"], 1))
        self.assertEqual(result["breakdown_kwh"]["ventilation"], round(electric["fan"], 1))
        self.assertNotIn("heating_total", result["breakdown_kwh"])

    def test_invalid_annual_model_divisors_raise_clear_errors(self):
        params = self._annual_model_params()
        params["occupancy"]["area_per_person_m2"] = 0
        with self.assertRaisesRegex(ValueError, "人均占用面积"):
            calculate_energy(params)

        params = self._annual_model_params()
        params["heating"]["seasonal_efficiency"] = 0
        with self.assertRaisesRegex(ValueError, "季节性能系数"):
            calculate_energy(params)

    def test_dhw_kj_to_kwh_conversion_is_not_divided_by_1000_twice(self):
        params = self._annual_model_params()
        params["dhw"] = {"occupants": 1, "daily_liter_pp": 1, "efficiency": 1.0}
        result = calculate_energy(params)
        expected = 1 * 1 * 4.186 * 35 * 365 / 3600
        self.assertAlmostEqual(result["annual_electricity_kwh"]["dhw"], expected, places=3)

    def test_invalid_dhw_inputs_raise_value_error(self):
        cases = [
            ("occupants", -1),
            ("occupants", float("nan")),
            ("daily_liter_pp", -1),
            ("daily_liter_pp", float("nan")),
            ("efficiency", 0),
            ("efficiency", float("nan")),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                params = self._annual_model_params()
                params["dhw"].update({
                    "occupants": 1,
                    "daily_liter_pp": 1,
                    "efficiency": 1,
                    field: value,
                })
                with self.assertRaises(ValueError):
                    calculate_energy(params)

    def test_heat_recovery_accepts_fraction_and_percent_field(self):
        fraction = self._annual_model_params()
        fraction["ventilation"]["heat_recovery_efficiency"] = 0.5
        percent = self._annual_model_params()
        percent["ventilation"].pop("heat_recovery_efficiency")
        percent["ventilation"]["heat_recovery_efficiency_percent"] = 50
        same_name_percent = self._annual_model_params()
        same_name_percent["ventilation"]["heat_recovery_efficiency"] = 50
        fraction_result = calculate_energy(fraction)
        percent_result = calculate_energy(percent)
        same_name_result = calculate_energy(same_name_percent)
        self.assertEqual(
            percent_result["annual_thermal_demand_kwh_th"]["heating_fresh_air"],
            fraction_result["annual_thermal_demand_kwh_th"]["heating_fresh_air"],
        )
        self.assertEqual(percent_result["derived_inputs"]["heat_recovery_efficiency"], 0.5)
        self.assertEqual(
            same_name_result["annual_thermal_demand_kwh_th"]["heating_fresh_air"],
            fraction_result["annual_thermal_demand_kwh_th"]["heating_fresh_air"],
        )

    def test_non_electric_heating_uses_explicit_efficiency_and_stays_out_of_electricity(self):
        params = self._annual_model_params()
        params["heating"].update({"system_type": "gas_boiler", "seasonal_efficiency": 0.8})
        result = calculate_energy(params)
        thermal = result["annual_thermal_demand_kwh_th"]["heating_total"]
        purchased = result["annual_non_electric_purchased_energy"]
        self.assertEqual(result["annual_electricity_kwh"]["heating"], 0.0)
        self.assertEqual(result["breakdown_kwh"]["heating"], 0.0)
        self.assertEqual(purchased["fuel_type"], "gas")
        self.assertEqual(purchased["seasonal_efficiency"], 0.8)
        self.assertAlmostEqual(purchased["heating_input_kwh"], thermal / 0.8, places=2)

    def test_fresh_air_peak_uses_full_design_flow_not_operation_fraction(self):
        params = self._annual_model_params()
        params["geometry"].update({"wall_area_m2": 0, "floor_area_m2": 100})
        params["envelope"].update({"u_wall": 0, "shgc": 0})
        result = calculate_energy(params)
        expected = 0.335 * 300 * (18 - (-5)) / 1000
        self.assertAlmostEqual(result["equivalent_peak_check_kw_th"]["heating"], expected, places=3)

    def test_electric_resistance_defaults_to_unit_efficiency(self):
        params = self._annual_model_params()
        params["heating"] = {
            "system_type": "electric",
            "t_set": 18.0,
            "seasonal_efficiency": 2.5,
        }
        result = calculate_energy(params)
        self.assertEqual(result["derived_inputs"]["heating_seasonal_efficiency"], 1.0)
        self.assertEqual(
            result["annual_electricity_kwh"]["heating"],
            result["annual_thermal_demand_kwh_th"]["heating_total"],
        )

    def test_non_xian_peak_without_confirmed_design_temperatures_is_unavailable(self):
        params = self._annual_model_params()
        params["climate"]["city_id"] = "beijing"
        params["heating"].pop("outdoor_design_temperature_c")
        params["cooling"].pop("outdoor_design_temperature_c")
        result = calculate_energy(params)
        peak = result["equivalent_peak_check_kw_th"]
        self.assertIsNone(peak["heating"])
        self.assertIsNone(peak["cooling"])
        self.assertEqual(result["derived_inputs"]["design_temperature_source"], "unavailable")
        self.assertTrue(any("未确认" in item for item in result["model_boundaries"]))

    def test_simple_heating_envelope_does_not_deduct_annual_solar_gain(self):
        params = self._annual_model_params()
        params["geometry"]["window_area_m2"] = 10
        params["envelope"].update({"u_window": 2.0, "shgc": 0.8})
        result = calculate_energy(params)
        expected_ua = 100 * 1.0 + 10 * 2.0
        self.assertAlmostEqual(
            result["annual_thermal_demand_kwh_th"]["heating_envelope"],
            expected_ua * 2400 * 24 / 1000,
            places=3,
        )

    def test_detailed_heating_envelope_does_not_deduct_annual_solar_gain(self):
        params = self._annual_model_params()
        params["calculation_mode"] = "detailed"
        params["detailed_envelope"] = {
            "walls_by_orientation": {"south": {"area_m2": 100, "u_value": 1.0}},
            "windows_by_orientation": {
                "south": {"area_m2": 10, "u_value": 2.0, "shgc": 0.8}
            },
        }
        result = calculate_energy(params)
        self.assertAlmostEqual(
            result["annual_thermal_demand_kwh_th"]["heating_envelope"],
            120 * 2400 * 24 / 1000,
            places=3,
        )

    def test_out_of_range_annual_inputs_raise_instead_of_clamping(self):
        cases = [
            ("daily_operation_hours", 25),
            ("annual_operation_days", 400),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                params = self._annual_model_params()
                params["schedule"][field] = value
                with self.assertRaises(ValueError):
                    calculate_energy(params)

        params = self._annual_model_params()
        params["ventilation"]["heat_recovery_efficiency_percent"] = 150
        with self.assertRaises(ValueError):
            calculate_energy(params)

    def test_negative_or_non_finite_geometry_raises(self):
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value):
                params = self._annual_model_params()
                params["geometry"]["floor_area_m2"] = value
                with self.assertRaises(ValueError):
                    calculate_energy(params)

    def test_invalid_detailed_orientation_areas_raise_instead_of_becoming_zero(self):
        cases = [
            ("walls_by_orientation", "area_m2", -1),
            ("walls_by_orientation", "area_m2", float("nan")),
            ("windows_by_orientation", "area_m2", -1),
            ("windows_by_orientation", "area_m2", float("nan")),
            ("windows_by_orientation", "door_area_m2", -1),
            ("windows_by_orientation", "door_area_m2", float("nan")),
        ]
        for collection, field, value in cases:
            with self.subTest(collection=collection, field=field, value=value):
                params = self._annual_model_params()
                params["calculation_mode"] = "detailed"
                params["detailed_envelope"] = {
                    "walls_by_orientation": {
                        "north": {"area_m2": 10, "u_value": 1.0}
                    },
                    "windows_by_orientation": {
                        "north": {
                            "area_m2": 10,
                            "u_value": 2.0,
                            "shgc": 0.4,
                            "door_area_m2": 1,
                            "door_u_value": 2.0,
                        }
                    },
                }
                params["detailed_envelope"][collection]["north"][field] = value
                with self.assertRaises(ValueError):
                    calculate_energy(params)

    def test_zero_detailed_orientation_areas_are_allowed(self):
        params = self._annual_model_params()
        params["calculation_mode"] = "detailed"
        params["detailed_envelope"] = {
            "walls_by_orientation": {"north": {"area_m2": 0, "u_value": 1.0}},
            "windows_by_orientation": {
                "north": {
                    "area_m2": 0,
                    "u_value": 2.0,
                    "shgc": 0.4,
                    "door_area_m2": 0,
                    "door_u_value": 2.0,
                }
            },
        }
        result = calculate_energy(params)
        self.assertTrue(result["success"])

    def test_legacy_schedule_total_hours_and_new_load_defaults_are_respected(self):
        params = self._annual_model_params()
        params["schedule"] = {"op_hours": 1200}
        params["lighting"] = {"control_factor": 1.0}
        params["equipment"] = {}
        result = calculate_energy(params)
        self.assertEqual(result["derived_inputs"]["annual_operation_hours"], 1200.0)
        self.assertEqual(result["annual_electricity_kwh"]["lighting"], 8 * 100 * 1200 / 1000)
        self.assertEqual(result["annual_electricity_kwh"]["equipment"], 15 * 100 * 1200 / 1000)

    def test_legacy_window_tightness_infiltration_is_returned_as_infiltration(self):
        params = self._annual_model_params()
        params["calculation_mode"] = "detailed"
        params["ventilation"].pop("infiltration_ach")
        params["detailed_envelope"] = {
            "walls_by_orientation": {"north": {"area_m2": 100, "u_value": 1.0}},
            "windows_by_orientation": {
                "north": {"area_m2": 10, "u_value": 2.0, "shgc": 0, "air_tightness_value": 4}
            },
        }
        result = calculate_energy(params)
        thermal = result["annual_thermal_demand_kwh_th"]
        self.assertGreater(thermal["heating_infiltration"], 0)
        self.assertAlmostEqual(thermal["heating_envelope"], 120 * 2400 * 24 / 1000, places=3)
        self.assertEqual(result["derived_inputs"]["infiltration_source"], "legacy_window_tightness")

    def test_zero_demand_has_zero_peak_but_keeps_valid_equivalent_hours(self):
        params = self._annual_model_params()
        params["geometry"].update({
            "wall_area_m2": 0,
            "window_area_m2": 0,
            "roof_area_m2": 0,
            "door_area_m2": 0,
        })
        params["envelope"].update({
            "u_wall": 0,
            "u_window": 0,
            "u_roof": 0,
            "u_floor": 0,
            "u_door": 0,
            "shgc": 0,
        })
        params["ventilation"]["fresh_air_m3h_per_person"] = 0
        result = calculate_energy(params)
        peak = result["equivalent_peak_check_kw_th"]
        self.assertEqual(peak["heating"], 0.0)
        self.assertGreater(peak["heating_equivalent_full_load_hours"], 0)

    def test_detailed_mode_uses_orientation_solar_gain_instead_of_weighted_average_only(self):
        south_params = self._zero_internal_load_params()
        south_params["detailed_envelope"] = {
            "walls_by_orientation": {},
            "windows_by_orientation": {
                "south": {"area_m2": 10, "u_value": 2.0, "shgc": 0.5}
            },
        }
        north_params = self._zero_internal_load_params()
        north_params["detailed_envelope"] = {
            "walls_by_orientation": {},
            "windows_by_orientation": {
                "north": {"area_m2": 10, "u_value": 2.0, "shgc": 0.5}
            },
        }

        south = calculate_energy(south_params)
        north = calculate_energy(north_params)

        self.assertGreater(
            south["breakdown_kwh"]["cooling"],
            north["breakdown_kwh"]["cooling"],
        )
        self.assertGreater(
            south["detailed_loads"]["solar_gain_by_orientation"]["south"],
            north["detailed_loads"]["solar_gain_by_orientation"]["north"],
        )

    def test_detailed_orientation_envelope_uses_area_weighted_inputs(self):
        params = {
            "calculation_mode": "detailed",
            "geometry": {
                "floor_area_m2": 100,
                "wall_area_m2": 1,
                "window_area_m2": 1,
                "roof_area_m2": 0,
                "door_area_m2": 0,
            },
            "building": {"floors": 1, "height": 3.0, "building_type": "office"},
            "envelope": {
                "u_wall": 9.0,
                "u_window": 9.0,
                "u_roof": 0.0,
                "u_floor": 0.0,
                "u_door": 0.0,
                "shgc": 0.9,
            },
            "detailed_envelope": {
                "walls_by_orientation": {
                    "east": {"area_m2": 10, "u_value": 0.4},
                    "south": {"area_m2": 30, "u_value": 0.8},
                },
                "windows_by_orientation": {
                    "east": {"area_m2": 4, "u_value": 1.8, "shgc": 0.3},
                    "south": {"area_m2": 6, "u_value": 2.4, "shgc": 0.5},
                },
            },
            "climate": {"city_id": "beijing"},
            "heating": {"system_type": "gas_boiler", "t_set": 18.0},
            "cooling": {"system_type": "central_chiller", "t_set": 26.0},
            "lighting": {"lpd": 0, "control_factor": 1.0},
            "ventilation": {"ach": 0, "fan_power": 0},
            "dhw": {"occupants": 1, "daily_liter_pp": 0, "efficiency": 1.0},
            "equipment": {"epd": 0},
            "schedule": {"op_hours": 0},
        }

        result = calculate_energy(params)

        self.assertEqual(result["geometry_used"]["wall_area_m2"], 40.0)
        self.assertEqual(result["geometry_used"]["window_area_m2"], 10.0)
        self.assertAlmostEqual(result["inputs_used"]["u_wall"], 0.7)
        self.assertAlmostEqual(result["inputs_used"]["u_window"], 2.16)
        self.assertAlmostEqual(result["inputs_used"]["shgc"], 0.42)

    def test_detailed_orientation_openings_include_door_area_and_u_value(self):
        params = {
            "calculation_mode": "detailed",
            "geometry": {
                "floor_area_m2": 100,
                "wall_area_m2": 0,
                "window_area_m2": 0,
                "roof_area_m2": 0,
                "door_area_m2": 0,
            },
            "building": {"floors": 1, "height": 3.0, "building_type": "office"},
            "envelope": {
                "u_wall": 0.0,
                "u_window": 0.0,
                "u_roof": 0.0,
                "u_floor": 0.0,
                "u_door": 9.0,
                "shgc": 0.4,
            },
            "detailed_envelope": {
                "walls_by_orientation": {},
                "windows_by_orientation": {
                    "east": {"door_area_m2": 2, "door_u_value": 1.5},
                    "south": {"door_area_m2": 4, "door_u_value": 3.0},
                },
            },
            "climate": {"city_id": "beijing"},
            "heating": {"system_type": "gas_boiler", "t_set": 18.0},
            "cooling": {"system_type": "central_chiller", "t_set": 26.0},
            "lighting": {"lpd": 0, "control_factor": 1.0},
            "ventilation": {"ach": 0, "fan_power": 0},
            "dhw": {"occupants": 1, "daily_liter_pp": 0, "efficiency": 1.0},
            "equipment": {"epd": 0},
            "schedule": {"op_hours": 0},
        }

        result = calculate_energy(params)

        self.assertEqual(result["geometry_used"]["door_area_m2"], 6.0)
        self.assertAlmostEqual(result["inputs_used"]["u_door"], 2.5)

    def test_door_u_value_changes_envelope_load(self):
        base_params = {
            "geometry": {
                "floor_area_m2": 100,
                "wall_area_m2": 0,
                "window_area_m2": 0,
                "roof_area_m2": 0,
                "door_area_m2": 10,
            },
            "building": {"floors": 1, "height": 3.0, "building_type": "office"},
            "envelope": {
                "u_wall": 0.6,
                "u_window": 2.5,
                "u_roof": 0.4,
                "u_floor": 0.0,
                "u_door": 1.0,
                "shgc": 0.4,
            },
            "climate": {"city_id": "beijing"},
            "heating": {"system_type": "gas_boiler", "t_set": 18.0},
            "cooling": {"system_type": "central_chiller", "t_set": 26.0},
            "lighting": {"lpd": 0, "control_factor": 1.0},
            "ventilation": {"ach": 0, "fan_power": 0},
            "dhw": {"occupants": 1, "daily_liter_pp": 0, "efficiency": 1.0},
            "equipment": {"epd": 0},
            "schedule": {"op_hours": 0},
        }

        efficient_door = calculate_energy(base_params)
        base_params["envelope"]["u_door"] = 4.0
        leaky_door = calculate_energy(base_params)

        self.assertGreater(
            leaky_door["summary"]["total_energy_kwh"],
            efficient_door["summary"]["total_energy_kwh"],
        )

    def test_detailed_mode_excludes_roof_and_floor_by_default(self):
        params = self._zero_internal_load_params()
        params["geometry"].update({"floor_area_m2": 100, "roof_area_m2": 100})
        params["envelope"].update({"u_roof": 0.4, "u_floor": 0.3})

        result = calculate_energy(params)

        self.assertEqual(result["detailed_loads"]["ua_roof"], 0.0)
        self.assertEqual(result["detailed_loads"]["ua_floor"], 0.0)
        self.assertFalse(result["inputs_used"]["include_roof"])
        self.assertFalse(result["inputs_used"]["include_floor"])

    def test_detailed_mode_applies_roof_and_corrected_floor_ua(self):
        params = self._zero_internal_load_params()
        params["geometry"].update({
            "floor_area_m2": 100,
            "roof_area_m2": 100,
            "include_roof": True,
            "include_floor": True,
        })
        params["envelope"].update({
            "u_roof": 0.4,
            "u_floor": 0.3,
            "floor_contact_factor": 0.5,
        })

        result = calculate_energy(params)

        self.assertEqual(result["detailed_loads"]["ua_roof"], 40.0)
        self.assertEqual(result["detailed_loads"]["ua_floor"], 15.0)
        self.assertEqual(result["detailed_loads"]["total_ua"], 55.0)
        self.assertEqual(result["inputs_used"]["floor_contact_factor"], 0.5)

    def test_detailed_mode_uses_explicit_annual_solar_irradiation(self):
        params = self._zero_internal_load_params()
        params["envelope"]["annual_solar_irradiation_kwh_m2a"] = 200
        params["detailed_envelope"] = {
            "walls_by_orientation": {},
            "windows_by_orientation": {
                "south": {
                    "area_m2": 10,
                    "u_value": 0,
                    "shgc": 0.5,
                    "shading_factor": 0.8,
                }
            },
        }

        result = calculate_energy(params)

        cooling_fraction = 350 / (1500 + 350)
        expected = 10 * 0.5 * 0.8 * 200 * 1.0 * cooling_fraction
        self.assertAlmostEqual(result["detailed_loads"]["solar_gain"], expected, places=3)
        self.assertEqual(
            result["inputs_used"]["annual_solar_irradiation_kwh_m2a"],
            200.0,
        )

    def test_detailed_mode_clamps_negative_boundary_factors_to_zero(self):
        params = self._zero_internal_load_params()
        params["geometry"].update({
            "include_floor": True,
            "floor_area_m2": 100,
        })
        params["envelope"].update({
            "u_floor": 0.3,
            "floor_contact_factor": -0.5,
            "annual_solar_irradiation_kwh_m2a": -100,
        })

        result = calculate_energy(params)

        self.assertEqual(result["detailed_loads"]["ua_floor"], 0.0)
        self.assertEqual(result["inputs_used"]["floor_contact_factor"], 0.0)
        self.assertEqual(
            result["inputs_used"]["annual_solar_irradiation_kwh_m2a"],
            0.0,
        )

    def test_simple_mode_keeps_existing_roof_and_floor_behavior(self):
        params = self._zero_internal_load_params()
        params["calculation_mode"] = "simple"
        params["geometry"].update({
            "floor_area_m2": 100,
            "roof_area_m2": 100,
            "include_roof": False,
            "include_floor": False,
        })
        params["envelope"].update({"u_roof": 0.4, "u_floor": 0.3})

        with_boundaries = calculate_energy(params)
        params["envelope"].update({"u_roof": 0.0, "u_floor": 0.0})
        without_boundaries = calculate_energy(params)

        self.assertGreater(
            with_boundaries["summary"]["total_energy_kwh"],
            without_boundaries["summary"]["total_energy_kwh"],
        )


if __name__ == "__main__":
    unittest.main()
