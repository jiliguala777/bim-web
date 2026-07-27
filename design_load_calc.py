import math


AIR_HEAT_W_PER_M3H_K = 0.335


def _finite_float(value, field_name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return number


def _non_negative(value, field_name):
    number = _finite_float(value, field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return number


def _positive(value, field_name):
    number = _finite_float(value, field_name)
    if number <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    return number


def _get(mapping, key, default):
    if not isinstance(mapping, dict):
        return default
    return mapping.get(key, default)


def _component(value):
    return round(value, 3)


def _sum_components(components):
    return sum(value for value in components.values() if value > 0)


def _bounded_percent(value, field_name):
    number = _finite_float(value, field_name)
    if number < 0 or number > 100:
        raise ValueError(f"{field_name} must be between 0 and 100")
    return number


def _saturation_vapor_pressure_kpa(dry_bulb_c):
    return 0.61078 * math.exp((17.2694 * dry_bulb_c) / (dry_bulb_c + 237.3))


def _moist_air_enthalpy_kj_kg(dry_bulb_c, relative_humidity_percent, pressure_kpa=101.325):
    rh = _bounded_percent(relative_humidity_percent, "relative_humidity_percent") / 100.0
    p_ws = _saturation_vapor_pressure_kpa(dry_bulb_c)
    p_w = rh * p_ws
    humidity_ratio = 0.62198 * p_w / max(pressure_kpa - p_w, 0.001)
    return 1.006 * dry_bulb_c + humidity_ratio * (2501 + 1.86 * dry_bulb_c)


def _summer_enthalpy_pair(cooling, vent):
    t_outdoor = _finite_float(
        cooling.get("outdoor_design_temperature_c", 34.9),
        "cooling.outdoor_design_temperature_c",
    )
    t_indoor = _finite_float(cooling.get("t_set", 26.0), "cooling.t_set")
    pressure = _positive(_get(vent, "atmospheric_pressure_kpa", 101.325), "atmospheric_pressure_kpa")

    if "summer_outdoor_relative_humidity_percent" in vent:
        outdoor_h = _moist_air_enthalpy_kj_kg(
            t_outdoor,
            vent["summer_outdoor_relative_humidity_percent"],
            pressure,
        )
        outdoor_source = "temperature_relative_humidity"
    else:
        outdoor_h = _finite_float(
            _get(vent, "summer_outdoor_enthalpy_kj_kg", 85.0),
            "summer_outdoor_enthalpy_kj_kg",
        )
        outdoor_source = "explicit_enthalpy_or_default"

    if "summer_indoor_relative_humidity_percent" in vent:
        indoor_h = _moist_air_enthalpy_kj_kg(
            t_indoor,
            vent["summer_indoor_relative_humidity_percent"],
            pressure,
        )
        indoor_source = "temperature_relative_humidity"
    else:
        indoor_h = _finite_float(
            _get(vent, "summer_indoor_enthalpy_kj_kg", 55.0),
            "summer_indoor_enthalpy_kj_kg",
        )
        indoor_source = "explicit_enthalpy_or_default"

    return outdoor_h, indoor_h, outdoor_source, indoor_source


def _base_inputs(params):
    geo = params.get("geometry", {}) or {}
    bld = params.get("building", {}) or {}
    env = params.get("envelope", {}) or {}
    occ = params.get("occupancy", {}) or {}
    vent = params.get("ventilation", {}) or {}
    light = params.get("lighting", {}) or {}
    equip = params.get("equipment", {}) or {}

    floor_area = _positive(_get(geo, "floor_area_m2", 0), "floor_area_m2")
    floors = int(_positive(_get(bld, "floors", 1), "floors"))
    height = _positive(_get(bld, "height", 3.0), "height")
    total_area = floor_area * floors

    area_per_person = _positive(_get(occ, "area_per_person_m2", 10.0), "area_per_person_m2")
    occupants = total_area / area_per_person
    fresh_air_flow = occupants * _non_negative(
        _get(vent, "fresh_air_m3h_per_person", 30.0),
        "fresh_air_m3h_per_person",
    )
    infiltration_flow = total_area * height * _non_negative(
        _get(vent, "infiltration_ach", 0.0),
        "infiltration_ach",
    )

    wall_area = _non_negative(_get(geo, "wall_area_m2", 0), "wall_area_m2")
    window_area = _non_negative(_get(geo, "window_area_m2", 0), "window_area_m2")
    roof_area = _non_negative(_get(geo, "roof_area_m2", floor_area), "roof_area_m2")
    door_area = _non_negative(_get(geo, "door_area_m2", 0), "door_area_m2")
    explicit_floor_loss_area = _non_negative(
        _get(geo, "floor_loss_area_m2", 0),
        "floor_loss_area_m2",
    )

    u_wall = _non_negative(_get(env, "u_wall", 0.6), "u_wall")
    u_window = _non_negative(_get(env, "u_window", _get(env, "u_win", 2.5)), "u_window")
    u_roof = _non_negative(_get(env, "u_roof", 0.4), "u_roof")
    u_floor = _non_negative(_get(env, "u_floor", 0.3), "u_floor")
    u_door = _non_negative(_get(env, "u_door", 3.0), "u_door")
    floor_contact_factor = _non_negative(
        _get(env, "floor_contact_factor", 1.0),
        "floor_contact_factor",
    )
    if geo.get("include_floor") is True:
        floor_loss_area = floor_area * floor_contact_factor
        floor_loss_source = "include_floor"
    else:
        floor_loss_area = explicit_floor_loss_area
        floor_loss_source = "explicit_floor_loss_area_m2"

    envelope_ua = (
        wall_area * u_wall
        + window_area * u_window
        + roof_area * u_roof
        + floor_loss_area * u_floor
        + door_area * u_door
    )

    people_sensible = occupants * _non_negative(
        _get(occ, "sensible_heat_w_per_person", 75.0),
        "sensible_heat_w_per_person",
    ) * _non_negative(_get(occ, "cooling_load_factor", 1.0), "cooling_load_factor")
    people_latent = occupants * _non_negative(
        _get(occ, "latent_heat_w_per_person", 55.0),
        "latent_heat_w_per_person",
    )
    lighting_cooling_factor = _non_negative(
        _get(light, "cooling_load_factor", 1.0),
        "lighting.cooling_load_factor",
    )
    equipment_cooling_factor = _non_negative(
        _get(equip, "cooling_load_factor", 1.0),
        "equipment.cooling_load_factor",
    )
    lighting = (
        total_area
        * _non_negative(_get(light, "lpd", 8.0), "lpd")
        * _non_negative(_get(light, "control_factor", 1.0), "lighting.control_factor")
        * lighting_cooling_factor
    )
    equipment = (
        total_area
        * _non_negative(_get(equip, "epd", 15.0), "epd")
        * equipment_cooling_factor
    )

    return {
        "geo": geo,
        "bld": bld,
        "env": env,
        "vent": vent,
        "floor_area": floor_area,
        "total_area": total_area,
        "height": height,
        "occupants": occupants,
        "fresh_air_flow_m3h": fresh_air_flow,
        "infiltration_flow_m3h": infiltration_flow,
        "envelope_ua": envelope_ua,
        "floor_loss_area_m2": floor_loss_area,
        "floor_loss_source": floor_loss_source,
        "air_density_kg_m3": _positive(_get(vent, "air_density_kg_m3", 1.13), "air_density_kg_m3"),
        "lighting_cooling_load_factor": lighting_cooling_factor,
        "equipment_cooling_load_factor": equipment_cooling_factor,
        "people_sensible_w": people_sensible,
        "people_latent_w": people_latent,
        "lighting_w": lighting,
        "equipment_w": equipment,
        "summer_enthalpy": _summer_enthalpy_pair(params.get("cooling", {}) or {}, vent),
    }


def _cooling_load(params, base):
    cooling = params.get("cooling", {}) or {}
    if cooling.get("enabled", True) is False:
        return None

    env = base["env"]
    vent = base["vent"]
    t_indoor = _finite_float(cooling.get("t_set", 26.0), "cooling.t_set")
    t_outdoor = _finite_float(cooling.get("outdoor_design_temperature_c", 34.9), "cooling.outdoor_design_temperature_c")
    delta_t = max(0.0, t_outdoor - t_indoor)

    shgc = _non_negative(_get(env, "window_shgc", _get(env, "shgc", 0.4)), "shgc")
    solar_irradiance = _non_negative(
        _get(env, "peak_solar_irradiance_w_m2", 250.0),
        "peak_solar_irradiance_w_m2",
    )
    orientation_factor = _non_negative(_get(env, "solar_orientation_factor", 1.0), "solar_orientation_factor")
    shading = _non_negative(_get(env, "curtain_shading", 1.0), "curtain_shading")
    window_area = _non_negative(_get(base["geo"], "window_area_m2", 0), "window_area_m2")

    fresh_air_flow = base["fresh_air_flow_m3h"]
    infiltration_flow = base["infiltration_flow_m3h"]
    total_air_flow = fresh_air_flow + infiltration_flow
    air_density = base["air_density_kg_m3"]
    outdoor_h, indoor_h, outdoor_h_source, indoor_h_source = base["summer_enthalpy"]
    latent_delta_h = max(0.0, outdoor_h - indoor_h)

    components = {
        "envelope_transmission": base["envelope_ua"] * delta_t,
        "window_solar": window_area * shgc * solar_irradiance * orientation_factor * shading,
        "fresh_air_sensible": AIR_HEAT_W_PER_M3H_K * fresh_air_flow * delta_t,
        "fresh_air_latent": air_density * fresh_air_flow * latent_delta_h / 3.6,
        "infiltration_sensible": AIR_HEAT_W_PER_M3H_K * infiltration_flow * delta_t,
        "infiltration_latent": air_density * infiltration_flow * latent_delta_h / 3.6,
        "people": base["people_sensible_w"] + base["people_latent_w"],
        "lighting": base["lighting_w"],
        "equipment": base["equipment_w"],
    }
    total = _sum_components(components)
    return {
        "enabled": True,
        "total_w": _component(total),
        "total_kw": _component(total / 1000.0),
        "cooling_index_w_m2": _component(total / base["total_area"]),
        "components_w": {key: _component(value) for key, value in components.items()},
        "design_conditions": {
            "indoor_temperature_c": t_indoor,
            "outdoor_temperature_c": t_outdoor,
            "delta_t_k": _component(delta_t),
            "fresh_air_flow_m3h": _component(fresh_air_flow),
            "infiltration_flow_m3h": _component(infiltration_flow),
            "summer_outdoor_enthalpy_kj_kg": _component(outdoor_h),
            "summer_indoor_enthalpy_kj_kg": _component(indoor_h),
            "summer_outdoor_enthalpy_source": outdoor_h_source,
            "summer_indoor_enthalpy_source": indoor_h_source,
        },
    }


def _heating_load(params, base):
    heating = params.get("heating", {}) or {}
    if heating.get("enabled", True) is False:
        return None

    t_indoor = _finite_float(heating.get("t_set", 18.0), "heating.t_set")
    t_outdoor = _finite_float(heating.get("outdoor_design_temperature_c", -5.0), "heating.outdoor_design_temperature_c")
    delta_t = max(0.0, t_indoor - t_outdoor)
    fresh_air_flow = base["fresh_air_flow_m3h"]
    infiltration_flow = base["infiltration_flow_m3h"]
    stable_gain_fraction = _non_negative(
        heating.get("stable_internal_gain_fraction", 0.0),
        "stable_internal_gain_fraction",
    )
    stable_gain_fraction = min(stable_gain_fraction, 1.0)
    door_invasion = _non_negative(
        heating.get("door_invasion_heat_w", 0.0),
        "door_invasion_heat_w",
    )
    heating_addition_factor = _positive(
        heating.get("heating_addition_factor", 1.0),
        "heating_addition_factor",
    )

    internal_gains = (
        base["people_sensible_w"]
        + base["people_latent_w"]
        + base["lighting_w"]
        + base["equipment_w"]
    )
    gain_deduction = internal_gains * stable_gain_fraction
    base_losses = {
        "envelope_basic": base["envelope_ua"] * delta_t,
        "fresh_air": AIR_HEAT_W_PER_M3H_K * fresh_air_flow * delta_t,
        "infiltration": AIR_HEAT_W_PER_M3H_K * infiltration_flow * delta_t,
        "door_invasion": door_invasion,
    }
    heating_addition = sum(base_losses.values()) * (heating_addition_factor - 1.0)
    components = {
        **base_losses,
        "heating_addition": heating_addition,
        "stable_internal_gain_deduction": gain_deduction,
    }
    total = max(0.0, sum(base_losses.values()) + heating_addition - gain_deduction)
    return {
        "enabled": True,
        "total_w": _component(total),
        "total_kw": _component(total / 1000.0),
        "heating_index_w_m2": _component(total / base["total_area"]),
        "components_w": {key: _component(value) for key, value in components.items()},
        "design_conditions": {
            "indoor_temperature_c": t_indoor,
            "outdoor_temperature_c": t_outdoor,
            "delta_t_k": _component(delta_t),
            "fresh_air_flow_m3h": _component(fresh_air_flow),
            "infiltration_flow_m3h": _component(infiltration_flow),
        },
    }


def calculate_design_loads(params):
    base = _base_inputs(params or {})
    heating = _heating_load(params or {}, base)
    cooling = _cooling_load(params or {}, base)

    return {
        "design_load_contract_version": 1,
        "calculation_scope": {
            "heating": heating is not None,
            "cooling": cooling is not None,
        },
        "heating_design_load": heating,
        "cooling_design_load": cooling,
        "room_loads": [
            {
                "room_id": "building",
                "area_m2": _component(base["total_area"]),
                "heating_w": heating["total_w"] if heating else None,
                "cooling_w": cooling["total_w"] if cooling else None,
            }
        ],
        "derived_inputs": {
            "total_area_m2": _component(base["total_area"]),
            "occupants": _component(base["occupants"]),
            "fresh_air_flow_m3h": _component(base["fresh_air_flow_m3h"]),
            "infiltration_flow_m3h": _component(base["infiltration_flow_m3h"]),
            "envelope_ua_w_k": _component(base["envelope_ua"]),
            "floor_loss_area_m2": _component(base["floor_loss_area_m2"]),
            "floor_loss_source": base["floor_loss_source"],
            "air_density_kg_m3": _component(base["air_density_kg_m3"]),
            "lighting_cooling_load_factor": _component(base["lighting_cooling_load_factor"]),
            "equipment_cooling_load_factor": _component(base["equipment_cooling_load_factor"]),
            "summer_outdoor_enthalpy_kj_kg": _component(base["summer_enthalpy"][0]),
            "summer_indoor_enthalpy_kj_kg": _component(base["summer_enthalpy"][1]),
            "summer_outdoor_enthalpy_source": base["summer_enthalpy"][2],
            "summer_indoor_enthalpy_source": base["summer_enthalpy"][3],
        },
        "formulas": {
            "cooling": [
                "Q_transmission = UA * (t_outdoor_summer - t_indoor_cooling)",
                "Q_solar = A_window * SHGC * peak_solar_irradiance * orientation_factor * shading_factor",
                "Q_air_sensible = 0.335 * airflow_m3h * delta_t",
                "Q_air_latent = air_density * airflow_m3h * enthalpy_delta / 3.6",
                "Q_internal = people + lighting + equipment",
            ],
            "heating": [
                "Q_envelope = UA * (t_indoor_heating - t_outdoor_winter)",
                "Q_air = 0.335 * airflow_m3h * delta_t",
                "Q_total = envelope + fresh_air + infiltration + door_invasion - stable_internal_gain_deduction",
            ],
        },
        "assumptions": [
            "Conservative first-pass design load model, not an 8760 hourly simulation.",
            "Heating internal gains are deducted only when stable_internal_gain_fraction is explicitly provided.",
            "Cooling latent load uses enthalpy difference defaults unless request data overrides them.",
        ],
    }
