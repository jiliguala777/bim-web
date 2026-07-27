"""
建筑能耗计算引擎 v2
基于 GB 50189-2015 简化稳态法 (度日数法)
支持详细的设备系统配置: 供暖/制冷/照明/通风/生活热水
"""

import math

# ============ 气候区数据 ============
CLIMATE_DB = {
    "harbin":    {"name": "哈尔滨", "zone": "严寒A区", "hdd18": 5100, "cdd26": 50,  "t_avg": 4.2,  "rh": 62},
    "urumqi":    {"name": "乌鲁木齐","zone": "严寒B区", "hdd18": 4300, "cdd26": 80,  "t_avg": 7.0,  "rh": 55},
    "beijing":   {"name": "北京",   "zone": "寒冷A区", "hdd18": 2800, "cdd26": 180, "t_avg": 12.6, "rh": 57},
    "dalian":    {"name": "大连",   "zone": "寒冷B区", "hdd18": 2900, "cdd26": 50,  "t_avg": 10.9, "rh": 65},
    "xian":      {
        "name": "西安", "zone": "寒冷B区", "hdd18": 2400, "cdd26": 200,
        "t_avg": 13.7, "rh": 65, "winter_design_temperature_c": -5.0,
        "summer_design_temperature_c": 34.9,
    },
    "shanghai":  {"name": "上海",   "zone": "夏热冬冷", "hdd18": 1500, "cdd26": 350, "t_avg": 16.1, "rh": 75},
    "chongqing": {"name": "重庆",   "zone": "夏热冬冷", "hdd18": 1100, "cdd26": 450, "t_avg": 18.3, "rh": 80},
    "wuhan":     {"name": "武汉",   "zone": "夏热冬冷", "hdd18": 1500, "cdd26": 400, "t_avg": 16.6, "rh": 77},
    "changsha":  {"name": "长沙",   "zone": "夏热冬冷", "hdd18": 1400, "cdd26": 380, "t_avg": 17.2, "rh": 78},
    "nanjing":   {"name": "南京",   "zone": "夏热冬冷", "hdd18": 1600, "cdd26": 350, "t_avg": 15.6, "rh": 75},
    "guangzhou": {"name": "广州",   "zone": "夏热冬暖", "hdd18": 400,  "cdd26": 650, "t_avg": 22.0, "rh": 77},
    "shenzhen":  {"name": "深圳",   "zone": "夏热冬暖", "hdd18": 300,  "cdd26": 700, "t_avg": 22.8, "rh": 78},
    "haikou":    {"name": "海口",   "zone": "夏热冬暖", "hdd18": 100,  "cdd26": 900, "t_avg": 24.0, "rh": 85},
    "kunming":   {"name": "昆明",   "zone": "温和区",   "hdd18": 1200, "cdd26": 10,  "t_avg": 15.0, "rh": 68},
    "guiyang":   {"name": "贵阳",   "zone": "温和区",   "hdd18": 1300, "cdd26": 30,  "t_avg": 15.3, "rh": 77},
    "chengdu":   {"name": "成都",   "zone": "夏热冬冷", "hdd18": 1200, "cdd26": 250, "t_avg": 16.5, "rh": 82},
    "tianjin":   {"name": "天津",   "zone": "寒冷A区", "hdd18": 2700, "cdd26": 200, "t_avg": 12.9, "rh": 62},
    "jinan":     {"name": "济南",   "zone": "寒冷B区", "hdd18": 2300, "cdd26": 250, "t_avg": 14.2, "rh": 60},
    "zhengzhou": {"name": "郑州",   "zone": "寒冷B区", "hdd18": 2200, "cdd26": 250, "t_avg": 14.5, "rh": 63},
    "hangzhou":  {"name": "杭州",   "zone": "夏热冬冷", "hdd18": 1500, "cdd26": 350, "t_avg": 16.5, "rh": 76},
}

# ============ 建筑类型默认参数 ============
BUILDING_DEFAULTS = {
    "office": {
        "name": "办公建筑", "lpd": 9.0, "epd": 15.0, "occupancy": 0.1,
        "vent_rate": 30, "dhw_liter_pp": 5, "op_hours": 2500
    },
    "commercial": {
        "name": "商业建筑", "lpd": 12.0, "epd": 20.0, "occupancy": 0.15,
        "vent_rate": 20, "dhw_liter_pp": 3, "op_hours": 3500
    },
    "hotel": {
        "name": "酒店建筑", "lpd": 10.0, "epd": 10.0, "occupancy": 0.08,
        "vent_rate": 30, "dhw_liter_pp": 80, "op_hours": 8760
    },
    "hospital": {
        "name": "医院建筑", "lpd": 11.0, "epd": 25.0, "occupancy": 0.12,
        "vent_rate": 40, "dhw_liter_pp": 60, "op_hours": 8760
    },
    "school": {
        "name": "学校建筑", "lpd": 9.0, "epd": 10.0, "occupancy": 0.5,
        "vent_rate": 25, "dhw_liter_pp": 5, "op_hours": 2000
    },
    "residential": {
        "name": "居住建筑", "lpd": 6.0, "epd": 8.0, "occupancy": 0.04,
        "vent_rate": 30, "dhw_liter_pp": 50, "op_hours": 5000
    },
}

# ============ 设备效率参数 ============
HEATING_SYSTEMS = {
    "gas_boiler":    {"name": "燃气锅炉",     "efficiency": 0.89, "fuel": "gas"},
    "coal_boiler":   {"name": "燃煤锅炉",     "efficiency": 0.75, "fuel": "coal"},
    "heat_pump_air": {"name": "空气源热泵",   "efficiency": 3.2,  "fuel": "electric"},
    "heat_pump_geo": {"name": "地源热泵",     "efficiency": 4.5,  "fuel": "electric"},
    "district":      {"name": "集中供暖",     "efficiency": 0.80, "fuel": "district"},
    "electric":      {"name": "电加热",       "efficiency": 0.95, "fuel": "electric"},
    "none":          {"name": "无供暖",       "efficiency": 1.0,  "fuel": "none"},
}

COOLING_SYSTEMS = {
    "central_chiller": {"name": "中央冷水机组", "cop": 5.0},
    "vrv":             {"name": "VRV/VRF多联机","cop": 3.8},
    "split_ac":        {"name": "分体空调",     "cop": 3.2},
    "evaporative":     {"name": "蒸发冷却",     "cop": 8.0},
    "none":            {"name": "无制冷",       "cop": 1.0},
}

ORIENTATION_SOLAR_FACTORS = {
    "south": 1.00,
    "east": 0.78,
    "west": 0.86,
    "north": 0.38,
}


def _positive_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, float(value)))


def _finite_float(value, field_name):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name}必须为有限数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name}必须为有限数值")
    return number


def _bounded_float(value, minimum, maximum, field_name):
    number = _finite_float(value, field_name)
    if number < minimum or number > maximum:
        raise ValueError(f"{field_name}必须在{minimum}到{maximum}之间")
    return number


def _annual_use_inputs(params, total_area, height, defaults, city):
    occupancy = params.get("occupancy", {}) or {}
    vent = params.get("ventilation", {}) or {}
    schedule = params.get("schedule", {}) or {}
    heat = params.get("heating", {}) or {}
    cool = params.get("cooling", {}) or {}

    area_per_person = _finite_float(occupancy.get("area_per_person_m2", 10.0), "人均占用面积")
    if area_per_person <= 0:
        raise ValueError("人均占用面积必须大于0")
    occupants = total_area / area_per_person
    fresh_air_pp = _finite_float(
        vent.get("fresh_air_m3h_per_person", 30.0),
        "人均新风量",
    )
    if fresh_air_pp < 0:
        raise ValueError("人均新风量不得为负")
    fresh_air_flow = occupants * fresh_air_pp

    if "daily_operation_hours" not in schedule and "annual_operation_days" not in schedule and "op_hours" in schedule:
        annual_hours = _bounded_float(schedule["op_hours"], 0.0, 8760.0, "年运行小时")
    else:
        daily_hours = _bounded_float(
            schedule.get("daily_operation_hours", 12.0), 0.0, 24.0, "每日运行小时"
        )
        annual_days = _bounded_float(
            schedule.get("annual_operation_days", 250), 0.0, 365.0, "年运行天数"
        )
        annual_hours = daily_hours * annual_days

    if "heat_recovery_efficiency_percent" in vent:
        recovery = _bounded_float(
            vent["heat_recovery_efficiency_percent"], 0.0, 100.0, "热回收效率百分数"
        ) / 100.0
    else:
        recovery_raw = _finite_float(
            vent.get("heat_recovery_efficiency", 0.0), "热回收效率"
        )
        if recovery_raw > 1.0:
            recovery = _bounded_float(recovery_raw, 0.0, 100.0, "热回收效率") / 100.0
        else:
            recovery = _bounded_float(recovery_raw, 0.0, 1.0, "热回收效率")

    infiltration_ach = _bounded_float(
        vent.get("infiltration_ach", 0.0), 0.0, 20.0, "渗透风换气次数"
    )
    fan_power = _finite_float(
        vent.get("fan_power_w_per_m3h", vent.get("fan_power", 0.5)),
        "风机单位风量功率",
    )
    if fan_power < 0:
        raise ValueError("风机单位风量功率不得为负")

    h_sys = heat.get("system_type", "gas_boiler")
    h_info = HEATING_SYSTEMS.get(h_sys, HEATING_SYSTEMS["gas_boiler"])
    if h_sys == "electric":
        heating_eff = 1.0
    else:
        default_heating_eff = 1.9 if h_info["fuel"] == "electric" else h_info["efficiency"]
        heating_eff = _finite_float(
            heat.get("seasonal_efficiency", default_heating_eff),
            "供暖季节性能系数",
        )
    cooling_eff = _finite_float(
        cool.get("seasonal_efficiency", 2.3), "制冷季节性能系数"
    )
    if heating_eff <= 0 or cooling_eff <= 0:
        raise ValueError("供暖和制冷季节性能系数必须大于0")

    winter_request = heat.get("outdoor_design_temperature_c")
    summer_request = cool.get("outdoor_design_temperature_c")
    winter_design = (
        _finite_float(winter_request, "冬季室外设计温度")
        if winter_request is not None
        else city.get("winter_design_temperature_c")
    )
    summer_design = (
        _finite_float(summer_request, "夏季室外设计温度")
        if summer_request is not None
        else city.get("summer_design_temperature_c")
    )
    winter_source = "request" if winter_request is not None else (
        "city_database" if winter_design is not None else "unavailable"
    )
    summer_source = "request" if summer_request is not None else (
        "city_database" if summer_design is not None else "unavailable"
    )
    design_temperature_source = (
        winter_source if winter_source == summer_source else "mixed"
    )

    return {
        "area_per_person_m2": area_per_person,
        "occupants": occupants,
        "fresh_air_flow_m3h": fresh_air_flow,
        "annual_operation_hours": annual_hours,
        "operation_fraction": annual_hours / 8760.0,
        "heat_recovery_efficiency": recovery,
        "infiltration_ach": infiltration_ach,
        "infiltration_flow_m3h": total_area * height * infiltration_ach,
        "fan_power_w_per_m3h": fan_power,
        "heating_seasonal_efficiency": heating_eff,
        "cooling_seasonal_efficiency": cooling_eff,
        "winter_design_temperature_c": winter_design,
        "summer_design_temperature_c": summer_design,
        "design_temperature_source": design_temperature_source,
        "winter_design_temperature_source": winter_source,
        "summer_design_temperature_source": summer_source,
    }


def _equivalent_peak(design_ua, degree_days, indoor_t, outdoor_design_t, heating):
    if outdoor_design_t is None:
        return None, None
    delta_t = indoor_t - outdoor_design_t if heating else outdoor_design_t - indoor_t
    if degree_days <= 0 or delta_t <= 0:
        return None, None
    equivalent_hours = degree_days * 24 / delta_t
    return design_ua * delta_t / 1000, equivalent_hours


def _area_weighted_average(items, value_key, area_key="area_m2"):
    total_area = 0.0
    weighted_sum = 0.0
    for item in items:
        area = _positive_float(item.get(area_key))
        value = _positive_float(item.get(value_key))
        if area is None or value is None:
            continue
        total_area += area
        weighted_sum += area * value
    if total_area <= 0:
        return None, None
    return total_area, weighted_sum / total_area


def _orientation_items(detailed_envelope, key):
    entries = detailed_envelope.get(key, {})
    if isinstance(entries, dict):
        return list(entries.values())
    if isinstance(entries, list):
        return entries
    return []


def _orientation_entries(detailed_envelope, key):
    entries = detailed_envelope.get(key, {})
    if isinstance(entries, dict):
        return list(entries.items())
    if isinstance(entries, list):
        return [(item.get("orientation", ""), item) for item in entries]
    return []


def _validate_detailed_orientation_areas(detailed_envelope):
    area_fields = {
        "walls_by_orientation": ("area_m2",),
        "windows_by_orientation": ("area_m2", "door_area_m2"),
    }
    for collection, fields in area_fields.items():
        for orientation, item in _orientation_entries(detailed_envelope, collection):
            for field in fields:
                if field not in item or item[field] is None:
                    continue
                area = _finite_float(
                    item[field],
                    f"{orientation or '未指定朝向'} {field}",
                )
                if area < 0:
                    raise ValueError("详细围护结构面积不得为负")


def _apply_detailed_envelope(params, geo, env):
    if params.get("calculation_mode") != "detailed":
        return geo, env

    detailed = params.get("detailed_envelope") or {}
    if not isinstance(detailed, dict):
        return geo, env

    _validate_detailed_orientation_areas(detailed)

    geo = dict(geo)
    env = dict(env)

    wall_area, wall_u = _area_weighted_average(
        _orientation_items(detailed, "walls_by_orientation"),
        "u_value",
    )
    if wall_area is not None:
        geo["wall_area_m2"] = wall_area
    if wall_u is not None:
        env["u_wall"] = wall_u

    window_items = _orientation_items(detailed, "windows_by_orientation")
    window_area, window_u = _area_weighted_average(window_items, "u_value")
    _, window_shgc = _area_weighted_average(window_items, "shgc")
    if window_area is not None:
        geo["window_area_m2"] = window_area
    if window_u is not None:
        env["u_window"] = window_u
    if window_shgc is not None:
        env["shgc"] = window_shgc

    door_area, door_u = _area_weighted_average(
        window_items,
        "door_u_value",
        area_key="door_area_m2",
    )
    if door_area is not None:
        geo["door_area_m2"] = door_area
    if door_u is not None:
        env["u_door"] = door_u

    return geo, env


def _detailed_orientation_loads(params, floor_area, roof_area, hdd, cdd, hdd_adj, cdd_adj):
    detailed = params.get("detailed_envelope") or {}
    wall_entries = _orientation_entries(detailed, "walls_by_orientation")
    window_entries = _orientation_entries(detailed, "windows_by_orientation")

    geo = params.get("geometry", {})
    env = params.get("envelope", {})
    include_roof = geo.get("include_roof") is True
    include_floor = geo.get("include_floor") is True
    floor_contact_factor = max(0.0, float(env.get("floor_contact_factor", 1.0)))
    annual_solar_irradiation = max(
        0.0,
        float(env.get("annual_solar_irradiation_kwh_m2a", 150.0)),
    )

    ua_by_orientation = {}
    solar_gain_by_orientation = {}
    infiltration_by_orientation = {}

    ua_wall = 0.0
    for orientation, item in wall_entries:
        area = _positive_float(item.get("area_m2")) or 0.0
        u_value = _positive_float(item.get("u_value")) or 0.0
        ua = area * u_value
        ua_wall += ua
        ua_by_orientation.setdefault(orientation, {"wall": 0.0, "window": 0.0, "door": 0.0})
        ua_by_orientation[orientation]["wall"] += ua

    ua_win = 0.0
    ua_door = 0.0
    solar_gain = 0.0
    infiltration_ua = 0.0
    explicit_infiltration = "infiltration_ach" in (params.get("ventilation") or {})
    for orientation, item in window_entries:
        window_area = _positive_float(item.get("area_m2")) or 0.0
        window_u = _positive_float(item.get("u_value")) or 0.0
        shgc = _positive_float(item.get("shgc")) or 0.0
        shading_factor = _positive_float(item.get("shading_factor")) or 1.0
        air_tightness = _positive_float(item.get("air_tightness_value")) or 6.0
        door_area = _positive_float(item.get("door_area_m2")) or 0.0
        door_u = _positive_float(item.get("door_u_value")) or 0.0

        window_ua = window_area * window_u
        door_ua = door_area * door_u
        ua_win += window_ua
        ua_door += door_ua
        ua_by_orientation.setdefault(orientation, {"wall": 0.0, "window": 0.0, "door": 0.0})
        ua_by_orientation[orientation]["window"] += window_ua
        ua_by_orientation[orientation]["door"] += door_ua

        solar_factor = ORIENTATION_SOLAR_FACTORS.get(orientation, 0.70)
        oriented_solar = (
            window_area
            * shgc
            * shading_factor
            * annual_solar_irradiation
            * solar_factor
            * (cdd / max(cdd + hdd, 1))
        )
        solar_gain_by_orientation[orientation] = solar_gain_by_orientation.get(orientation, 0.0) + oriented_solar
        solar_gain += oriented_solar

        if not explicit_infiltration:
            infiltration_ach = max(0.0, (8.0 - air_tightness) * 0.04)
            infiltration = 0.33 * infiltration_ach * window_area * 3.0
            infiltration_by_orientation[orientation] = (
                infiltration_by_orientation.get(orientation, 0.0) + infiltration
            )
            infiltration_ua += infiltration

    u_roof = float(env.get("u_roof", 0.4))
    u_floor = float(env.get("u_floor", 0.3))
    ua_roof = u_roof * roof_area if include_roof else 0.0
    ua_floor = u_floor * floor_area * floor_contact_factor if include_floor else 0.0
    transmission_ua = ua_wall + ua_win + ua_door + ua_roof + ua_floor
    total_ua = transmission_ua + infiltration_ua

    heating_envelope = transmission_ua * hdd_adj * 24 / 1000
    cooling_envelope = transmission_ua * cdd_adj * 24 / 1000
    cooling_envelope += solar_gain

    return {
        "heating_envelope": heating_envelope,
        "cooling_envelope": cooling_envelope,
        "total_ua": total_ua,
        "transmission_ua": transmission_ua,
        "ua_by_orientation": ua_by_orientation,
        "solar_gain_by_orientation": solar_gain_by_orientation,
        "infiltration_ua_by_orientation": infiltration_by_orientation,
        "solar_gain": solar_gain,
        "infiltration_ua": infiltration_ua,
        "ua_roof": ua_roof,
        "ua_floor": ua_floor,
        "include_roof": include_roof,
        "include_floor": include_floor,
        "floor_contact_factor": floor_contact_factor,
        "annual_solar_irradiation_kwh_m2a": annual_solar_irradiation,
    }


def calculate_energy(params):
    """
    综合能耗计算
    params: dict containing:
      - geometry: {floor_area_m2, wall_area_m2, window_area_m2, roof_area_m2, door_area_m2}
      - building: {height, floors, building_type, orientation}
      - envelope: {u_wall, u_window, u_roof, u_floor, shgc}
      - climate: {city_id}
      - heating: {system_type, t_set}
      - cooling: {system_type, t_set}
      - lighting: {lpd, control_factor}
      - ventilation: {ach, fan_power}
      - dhw: {occupants, daily_liter_pp}
      - equipment: {epd}
      - schedule: {op_hours}
    """
    # --- 解析参数 ---
    geo = params.get("geometry", {})
    bld = params.get("building", {})
    env = params.get("envelope", {})
    clim = params.get("climate", {})
    heat = params.get("heating", {})
    cool = params.get("cooling", {})
    light = params.get("lighting", {})
    vent = params.get("ventilation", {})
    dhw = params.get("dhw", {})
    equip = params.get("equipment", {})
    sched = params.get("schedule", {})

    heating_enabled = heat.get("enabled", True)
    cooling_enabled = cool.get("enabled", True)
    if not isinstance(heating_enabled, bool) or not isinstance(cooling_enabled, bool):
        raise ValueError("供暖和制冷计算开关必须为布尔值")
    if not heating_enabled and not cooling_enabled:
        raise ValueError("至少选择供暖或制冷中的一种计算")

    geo, env = _apply_detailed_envelope(params, geo, env)

    # --- 几何 ---
    floors_value = _finite_float(bld.get("floors", 1), "建筑层数")
    if floors_value <= 0 or not floors_value.is_integer():
        raise ValueError("建筑层数必须为正整数")
    floors = int(floors_value)
    height = _finite_float(bld.get("height", 3.0), "层高")
    if height <= 0:
        raise ValueError("层高必须大于0")
    floor_area = _finite_float(geo.get("floor_area_m2", 500), "建筑面积")
    total_area = floor_area * floors
    wall_area = _finite_float(geo.get("wall_area_m2", floor_area * 0.8), "墙体面积")
    window_area = _finite_float(geo.get("window_area_m2", wall_area * 0.3), "窗面积")
    roof_area = _finite_float(geo.get("roof_area_m2", floor_area), "屋面面积")
    door_area = _finite_float(geo.get("door_area_m2", 0), "门面积")
    if any(area < 0 for area in (floor_area, wall_area, window_area, roof_area, door_area)):
        raise ValueError("几何面积不得为负")
    wwr = window_area / max(wall_area + window_area, 1)

    # --- 气候 ---
    city_id = clim.get("city_id", "beijing")
    city = CLIMATE_DB.get(city_id, CLIMATE_DB["beijing"])
    hdd = city["hdd18"]
    cdd = city["cdd26"]

    # --- 围护结构 ---
    u_wall = float(env.get("u_wall", 0.6))
    u_win = float(env.get("u_window", 2.5))
    u_roof = float(env.get("u_roof", 0.4))
    u_floor = float(env.get("u_floor", 0.3))
    u_door = float(env.get("u_door", 3.0))
    shgc = float(env.get("shgc", 0.4))

    # --- 建筑类型默认值 ---
    btype = bld.get("building_type", "office")
    defaults = BUILDING_DEFAULTS.get(btype, BUILDING_DEFAULTS["office"])

    # === 1. 围护结构传热负荷 ===
    ua_wall = u_wall * wall_area
    ua_win = u_win * window_area
    ua_roof = u_roof * roof_area
    ua_floor = u_floor * floor_area
    ua_door = u_door * door_area
    total_ua = ua_wall + ua_win + ua_roof + ua_floor + ua_door

    # 供暖/制冷设定温度修正
    t_heat_set = float(heat.get("t_set", 18.0))
    t_cool_set = float(cool.get("t_set", 26.0))
    hdd_adj = max(0, hdd + (t_heat_set - 18.0) * 120)
    cdd_adj = max(0, cdd + (26.0 - t_cool_set) * 90)

    detailed_loads = None
    if params.get("calculation_mode") == "detailed":
        detailed_loads = _detailed_orientation_loads(
            params,
            floor_area,
            roof_area,
            hdd,
            cdd,
            hdd_adj,
            cdd_adj,
        )
        heating_envelope = detailed_loads["heating_envelope"]
        cooling_envelope = detailed_loads["cooling_envelope"]
        total_ua = detailed_loads["total_ua"]
        envelope_ua = detailed_loads["transmission_ua"]
    else:
        # Q = UA * DD * 24 / 1000 (kWh)
        heating_envelope = total_ua * hdd_adj * 24 / 1000
        cooling_envelope = total_ua * cdd_adj * 24 / 1000

        # 太阳得热 (简化)
        solar_gain = window_area * shgc * 150 * (cdd / max(cdd + hdd, 1))
        cooling_envelope += solar_gain
        envelope_ua = total_ua

    # === 2. 全年使用、新风与渗透风热需求 ===
    annual_use = _annual_use_inputs(params, total_area, height, defaults, city)
    fresh_air_ua = (
        0.335
        * annual_use["fresh_air_flow_m3h"]
        * (1.0 - annual_use["heat_recovery_efficiency"])
    )
    explicit_infiltration = "infiltration_ach" in (vent or {})
    if explicit_infiltration:
        annual_infiltration_ua = 0.335 * annual_use["infiltration_flow_m3h"]
        infiltration_source = "explicit_ach"
    elif detailed_loads is not None and detailed_loads["infiltration_ua"] > 0:
        annual_infiltration_ua = detailed_loads["infiltration_ua"]
        infiltration_source = "legacy_window_tightness"
    else:
        annual_infiltration_ua = 0.0
        infiltration_source = "none"
    fresh_air_heating = (
        fresh_air_ua * hdd_adj * 24 / 1000 * annual_use["operation_fraction"]
    )
    fresh_air_cooling = (
        fresh_air_ua * cdd_adj * 24 / 1000 * annual_use["operation_fraction"]
    )
    infiltration_heating = annual_infiltration_ua * hdd_adj * 24 / 1000
    infiltration_cooling = annual_infiltration_ua * cdd_adj * 24 / 1000
    annual_heating_demand = (
        heating_envelope + fresh_air_heating + infiltration_heating
        if heating_enabled else None
    )
    annual_cooling_demand = (
        cooling_envelope + fresh_air_cooling + infiltration_cooling
        if cooling_enabled else None
    )
    if not heating_enabled:
        heating_envelope = None
        fresh_air_heating = None
        infiltration_heating = None
    if not cooling_enabled:
        cooling_envelope = None
        fresh_air_cooling = None
        infiltration_cooling = None

    # === 3. 供暖设备用能 ===
    h_sys = heat.get("system_type", "gas_boiler")
    h_info = HEATING_SYSTEMS.get(h_sys, HEATING_SYSTEMS["gas_boiler"])
    heating_energy = (
        annual_heating_demand / annual_use["heating_seasonal_efficiency"]
        if heating_enabled and h_info["fuel"] == "electric" else 0.0
    )
    heating_purchased_non_electric = (
        annual_heating_demand / annual_use["heating_seasonal_efficiency"]
        if heating_enabled and h_info["fuel"] not in ("electric", "none") else 0.0
    )

    # === 4. 制冷设备用电 ===
    c_sys = cool.get("system_type", "central_chiller")
    c_info = COOLING_SYSTEMS.get(c_sys, COOLING_SYSTEMS["central_chiller"])
    cooling_energy = (
        annual_cooling_demand / annual_use["cooling_seasonal_efficiency"]
        if cooling_enabled and c_sys != "none" else 0.0
    )

    # === 5. 直接用电 ===
    op_hours = annual_use["annual_operation_hours"]
    lpd = float(light.get("lpd", 8.0))
    ctrl = float(light.get("control_factor", 1.0))
    lighting_energy = lpd * total_area * op_hours * ctrl / 1000

    epd = float(equip.get("epd", 15.0))
    equipment_energy = epd * total_area * op_hours / 1000

    ventilation_energy = (
        annual_use["fresh_air_flow_m3h"]
        * annual_use["fan_power_w_per_m3h"]
        * op_hours / 1000
    )

    # === 6. 生活热水用电 ===
    occupants = _finite_float(
        dhw.get("occupants", max(1, int(total_area * defaults["occupancy"]))),
        "生活热水人数",
    )
    daily_liter = _finite_float(
        dhw.get("daily_liter_pp", defaults["dhw_liter_pp"]),
        "人均生活热水量",
    )
    if occupants < 0 or daily_liter < 0:
        raise ValueError("生活热水人数和人均热水量不得为负")
    dt_water = 35  # 温升 ΔT
    dhw_energy = occupants * daily_liter * 4.186 * dt_water * 365 / 3600  # kWh
    dhw_eff = _finite_float(dhw.get("efficiency", 0.85), "生活热水效率")
    if dhw_eff <= 0:
        raise ValueError("生活热水效率必须大于0")
    dhw_energy = dhw_energy / dhw_eff

    # === 7. 峰值校核与用电汇总 ===
    heating_peak, heating_equivalent_hours = (
        _equivalent_peak(
            envelope_ua + fresh_air_ua + annual_infiltration_ua,
            hdd_adj,
            t_heat_set,
            annual_use["winter_design_temperature_c"],
            True,
        )
        if heating_enabled else (None, None)
    )
    cooling_peak, cooling_equivalent_hours = (
        _equivalent_peak(
            envelope_ua + fresh_air_ua + annual_infiltration_ua,
            cdd_adj,
            t_cool_set,
            annual_use["summer_design_temperature_c"],
            False,
        )
        if cooling_enabled else (None, None)
    )
    total_energy = heating_energy + cooling_energy + lighting_energy + equipment_energy + ventilation_energy + dhw_energy
    eui = total_energy / total_area if total_area > 0 else 0

    # EUI 等级评定
    if eui < 50:
        rating, rating_label = "A", "超低能耗"
    elif eui < 80:
        rating, rating_label = "B", "低能耗"
    elif eui < 120:
        rating, rating_label = "C", "节能"
    elif eui < 180:
        rating, rating_label = "D", "一般"
    else:
        rating, rating_label = "E", "高能耗"

    breakdown = {
        "heating":     round(heating_energy, 1),
        "cooling":     round(cooling_energy, 1),
        "lighting":    round(lighting_energy, 1),
        "equipment":   round(equipment_energy, 1),
        "ventilation": round(ventilation_energy, 1),
        "dhw":         round(dhw_energy, 1),
    }
    breakdown_eui = {k: round(v / total_area, 2) if total_area > 0 else 0 for k, v in breakdown.items()}

    rounded_thermal = lambda value: round(value, 3) if value is not None else None
    thermal_components = {
        "heating_envelope": rounded_thermal(heating_envelope),
        "cooling_envelope": rounded_thermal(cooling_envelope),
        "heating_fresh_air": rounded_thermal(fresh_air_heating),
        "cooling_fresh_air_sensible": rounded_thermal(fresh_air_cooling),
        "heating_infiltration": rounded_thermal(infiltration_heating),
        "cooling_infiltration": rounded_thermal(infiltration_cooling),
    }
    annual_thermal = {
        "heating_total": rounded_thermal(annual_heating_demand),
        "cooling_total": rounded_thermal(annual_cooling_demand),
        **thermal_components,
    }
    annual_electricity = {
        "heating": round(heating_energy, 3),
        "cooling": round(cooling_energy, 3),
        "fan": round(ventilation_energy, 3),
        "lighting": round(lighting_energy, 3),
        "equipment": round(equipment_energy, 3),
        "dhw": round(dhw_energy, 3),
    }
    annual_electricity["total"] = round(sum(annual_electricity.values()), 3)

    model_boundaries = [
        "HDD/CDD简化年度模型",
        "新风制冷仅含显热，不含除湿潜热",
        "未进行8760小时逐时模拟",
        "等效峰值仅用于数量级校核，不是严格设计负荷",
        "人员、照明和设备内部得热未与冷热需求逐时耦合",
        "年度供暖热需求未扣除太阳得热",
    ]
    if not heating_enabled:
        model_boundaries.append("本报告未启用供暖计算")
    if not cooling_enabled:
        model_boundaries.append("本报告未启用制冷计算")
    if (
        annual_use["winter_design_temperature_source"] == "unavailable"
        or annual_use["summer_design_temperature_source"] == "unavailable"
    ):
        model_boundaries.append("当前城市室外设计温度未确认，等效峰值不可用")
    elif city_id != "xian" and annual_use["design_temperature_source"] in ("request", "mixed"):
        model_boundaries.append("非西安城市的室外设计温度来自用户输入，未由城市气候库确认")

    result = {
        "energy_contract_version": 2,
        "success": True,
        "calculation_scope": {
            "heating": heating_enabled,
            "cooling": cooling_enabled,
        },
        "summary": {
            "total_energy_kwh": round(total_energy, 0),
            "eui": round(eui, 2),
            "rating": rating,
            "rating_label": rating_label,
            "total_floor_area_m2": round(total_area, 1),
        },
        "breakdown_kwh": breakdown,
        "breakdown_eui": breakdown_eui,
        "annual_thermal_demand_kwh_th": annual_thermal,
        "annual_electricity_kwh": annual_electricity,
        "annual_non_electric_purchased_energy_kwh": {
            "heating": round(heating_purchased_non_electric, 3),
        },
        "annual_non_electric_purchased_energy": {
            "fuel_type": (
                h_info["fuel"]
                if heating_enabled and h_info["fuel"] not in ("electric", "none")
                else "none"
            ),
            "heating_input_kwh": round(heating_purchased_non_electric, 3),
            "seasonal_efficiency": (
                round(annual_use["heating_seasonal_efficiency"], 3)
                if heating_enabled and h_info["fuel"] not in ("electric", "none")
                else None
            ),
        },
        "equivalent_peak_check_kw_th": {
            "heating": round(heating_peak, 3) if heating_peak is not None else None,
            "cooling": round(cooling_peak, 3) if cooling_peak is not None else None,
            "heating_equivalent_full_load_hours": (
                round(heating_equivalent_hours, 3)
                if heating_equivalent_hours is not None else None
            ),
            "cooling_equivalent_full_load_hours": (
                round(cooling_equivalent_hours, 3)
                if cooling_equivalent_hours is not None else None
            ),
        },
        "derived_inputs": {
            "occupants": round(annual_use["occupants"], 3),
            "fresh_air_flow_m3h": round(annual_use["fresh_air_flow_m3h"], 3),
            "annual_operation_hours": round(annual_use["annual_operation_hours"], 3),
            "operation_fraction": round(annual_use["operation_fraction"], 6),
            "heat_recovery_efficiency": round(annual_use["heat_recovery_efficiency"], 6),
            "heating_seasonal_efficiency": round(
                annual_use["heating_seasonal_efficiency"], 6
            ),
            "cooling_seasonal_efficiency": round(
                annual_use["cooling_seasonal_efficiency"], 6
            ),
            "winter_design_temperature_c": annual_use["winter_design_temperature_c"],
            "summer_design_temperature_c": annual_use["summer_design_temperature_c"],
            "design_temperature_source": annual_use["design_temperature_source"],
            "winter_design_temperature_source": annual_use[
                "winter_design_temperature_source"
            ],
            "summer_design_temperature_source": annual_use[
                "summer_design_temperature_source"
            ],
            "infiltration_source": infiltration_source,
        },
        "model_boundaries": model_boundaries,
        "geometry_used": {
            "floor_area_m2": round(floor_area, 1),
            "total_area_m2": round(total_area, 1),
            "wall_area_m2": round(wall_area, 1),
            "window_area_m2": round(window_area, 1),
            "roof_area_m2": round(roof_area, 1),
            "door_area_m2": round(door_area, 1),
            "wwr": round(wwr, 3),
        },
        "inputs_used": {
            "calculation_mode": params.get("calculation_mode", "simple"),
            "u_wall": round(u_wall, 4),
            "u_window": round(u_win, 4),
            "u_roof": round(u_roof, 4),
            "u_floor": round(u_floor, 4),
            "u_door": round(u_door, 4),
            "shgc": round(shgc, 4),
        },
        "climate": {
            "city": city["name"],
            "zone": city["zone"],
            "hdd18": hdd,
            "cdd26": cdd,
        },
        "systems": {
            "heating": h_info["name"],
            "cooling": c_info["name"],
        },
    }
    if detailed_loads is not None:
        result["inputs_used"].update({
            "include_roof": detailed_loads["include_roof"],
            "include_floor": detailed_loads["include_floor"],
            "floor_contact_factor": round(detailed_loads["floor_contact_factor"], 4),
            "annual_solar_irradiation_kwh_m2a": round(
                detailed_loads["annual_solar_irradiation_kwh_m2a"],
                3,
            ),
        })
        result["detailed_loads"] = {
            "total_ua": round(detailed_loads["total_ua"], 3),
            "solar_gain": round(detailed_loads["solar_gain"], 3),
            "infiltration_ua": round(detailed_loads["infiltration_ua"], 3),
            "ua_roof": round(detailed_loads["ua_roof"], 3),
            "ua_floor": round(detailed_loads["ua_floor"], 3),
            "ua_by_orientation": {
                orientation: {
                    key: round(value, 3)
                    for key, value in parts.items()
                }
                for orientation, parts in detailed_loads["ua_by_orientation"].items()
            },
            "solar_gain_by_orientation": {
                orientation: round(value, 3)
                for orientation, value in detailed_loads["solar_gain_by_orientation"].items()
            },
            "infiltration_ua_by_orientation": {
                orientation: round(value, 3)
                for orientation, value in detailed_loads["infiltration_ua_by_orientation"].items()
            },
        }
    return result


def get_climate_cities():
    """返回可选城市列表"""
    return [{"id": k, "name": v["name"], "zone": v["zone"]} for k, v in CLIMATE_DB.items()]


def get_building_types():
    """返回建筑类型列表"""
    return [{"id": k, "name": v["name"]} for k, v in BUILDING_DEFAULTS.items()]


def get_system_options():
    """返回设备系统选项"""
    return {
        "heating": [{"id": k, "name": v["name"]} for k, v in HEATING_SYSTEMS.items()],
        "cooling": [{"id": k, "name": v["name"]} for k, v in COOLING_SYSTEMS.items()],
    }
