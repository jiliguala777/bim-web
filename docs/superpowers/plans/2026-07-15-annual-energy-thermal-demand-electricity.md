# Annual Thermal Demand and Electricity Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the existing HDD/CDD annual model so Step 3 accepts one global set of occupancy, fresh-air, schedule, and seasonal-efficiency inputs, then separately reports annual heating/cooling thermal demand and annual electricity use.

**Architecture:** Keep `energy_calc.py` as the calculation authority and preserve the existing simple/detailed envelope branches. Add a normalized annual-use calculation layer after envelope demand is computed, return new thermal/electric/peak-check blocks while keeping `breakdown_kwh` as a compatibility alias, and wire the new contract through the Flask route and `templates/energy.html`.

**Tech Stack:** Python 3, Flask, vanilla JavaScript, HTML/CSS, `unittest`, existing SQLite report persistence.

## Global Constraints

- Default indoor setpoints are heating `18℃` and cooling `26℃`.
- Do not add room types or per-room configuration.
- Continue using Xi'an `HDD18=2400 K·d` and `CDD26=200 K·d` with the existing setpoint adjustment formulas.
- Use winter design temperature `-5℃` and summer design temperature `34.9℃` only for equivalent-peak checks.
- Report thermal demand in `kWh_th/a`; report electricity in `kWh/a`; never sum `kWh_th` directly into electricity.
- Fresh-air cooling covers sensible heat only; do not add latent/enthalpy calculations.
- Do not couple annual personnel, lighting, or equipment heat gains into annual heating/cooling demand in this HDD/CDD version.
- Preserve existing simple/detailed envelope behavior and historical report compatibility.
- Do not stage or commit any changes until the user explicitly authorizes Git submission.

---

## File Structure

- Modify `energy_calc.py`: normalize new inputs, calculate fresh-air/infiltration thermal demand, seasonal equipment electricity, direct electricity, and HDD/CDD equivalent-peak checks.
- Modify `web_server_server.py`: validate and map request fields into `calc_params`; keep report persistence unchanged because it already stores the full params/results JSON.
- Modify `templates/energy.html`: add Step 3 controls, derived-input preview, request payload fields, history reload behavior, and separate result sections.
- Modify `tests/test_energy_calc.py`: calculation and backward-compatibility tests.
- Modify `tests/test_energy_template.py`: static UI/payload/result-contract tests.
- Keep `docs/superpowers/specs/2026-07-15-annual-energy-thermal-demand-electricity-design.md` as the source of truth.

---

### Task 1: Lock the Backend Result Contract with Failing Tests

**Files:**
- Modify: `tests/test_energy_calc.py`
- Test: `tests/test_energy_calc.py`

**Interfaces:**
- Consumes: existing `calculate_energy(params: dict) -> dict`.
- Produces: test fixtures and assertions for `annual_thermal_demand_kwh_th`, `annual_electricity_kwh`, `equivalent_peak_check_kw_th`, and `derived_inputs`.

- [ ] **Step 1: Add a reusable annual-model fixture**

Add this method to `EnergyCalcTests`:

```python
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
```

- [ ] **Step 2: Add exact derived-input and fresh-air tests**

```python
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
```

- [ ] **Step 3: Add seasonal-efficiency and electricity-separation tests**

```python
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
```

- [ ] **Step 4: Add recovery, infiltration, peak, and legacy tests**

```python
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
    self.assertAlmostEqual(peak["heating_equivalent_full_load_hours"], expected_hours, places=3)
    self.assertAlmostEqual(
        peak["heating"],
        result["annual_thermal_demand_kwh_th"]["heating_total"] / expected_hours,
        places=3,
    )

def test_legacy_params_without_new_fields_still_calculate(self):
    result = calculate_energy(self._zero_internal_load_params())
    self.assertIn("summary", result)
    self.assertIn("annual_thermal_demand_kwh_th", result)
    self.assertIn("annual_electricity_kwh", result)
```

- [ ] **Step 5: Run the focused tests and confirm failure**

Run:

```powershell
python -m unittest tests/test_energy_calc.py -v
```

Expected: new tests fail with missing result keys and unsupported new inputs; existing tests continue to execute.

- [ ] **Step 6: Review the test-only diff without staging or committing**

```powershell
git -c safe.directory=E:/bim-web diff -- tests/test_energy_calc.py
```

Expected: only the new fixture and calculation-contract tests appear.

---

### Task 2: Implement Annual Thermal Demand, Electricity, and Peak Checks

**Files:**
- Modify: `energy_calc.py:8-29`
- Modify: `energy_calc.py:109-262`
- Modify: `energy_calc.py:264-496`
- Test: `tests/test_energy_calc.py`

**Interfaces:**
- Consumes: normalized `occupancy`, `ventilation`, `schedule`, `heating`, and `cooling` dictionaries.
- Produces: `annual_thermal_demand_kwh_th`, `annual_electricity_kwh`, `equivalent_peak_check_kw_th`, `derived_inputs`, and `model_boundaries` result blocks.

- [ ] **Step 1: Add climate design temperatures without changing HDD/CDD values**

Extend the Xi'an entry only with confirmed design temperatures:

```python
"xian": {
    "name": "西安",
    "zone": "寒冷B区",
    "hdd18": 2400,
    "cdd26": 200,
    "t_avg": 13.7,
    "rh": 65,
    "winter_design_temperature_c": -5.0,
    "summer_design_temperature_c": 34.9,
},
```

Other cities fall back to request values, then to `-5.0` and `35.0`; do not invent city-specific values.

- [ ] **Step 2: Add input normalization helpers**

Add focused helpers near `_positive_float`:

```python
def _clamp(value, minimum, maximum):
    return max(minimum, min(maximum, float(value)))

def _annual_use_inputs(params, total_area, height, defaults, city):
    occupancy = params.get("occupancy", {})
    vent = params.get("ventilation", {})
    schedule = params.get("schedule", {})
    heat = params.get("heating", {})
    cool = params.get("cooling", {})

    area_per_person = float(occupancy.get("area_per_person_m2", 10.0))
    if area_per_person <= 0:
        raise ValueError("人均占用面积必须大于0")
    occupants = total_area / area_per_person
    fresh_air_pp = max(0.0, float(vent.get("fresh_air_m3h_per_person", 30.0)))
    fresh_air_flow = occupants * fresh_air_pp
    daily_hours = _clamp(schedule.get("daily_operation_hours", 12.0), 0.0, 24.0)
    annual_days = _clamp(schedule.get("annual_operation_days", 250), 0.0, 365.0)
    annual_hours = daily_hours * annual_days
    recovery = _clamp(vent.get("heat_recovery_efficiency", 0.0), 0.0, 1.0)
    infiltration_ach = _clamp(vent.get("infiltration_ach", 0.0), 0.0, 20.0)
    fan_power = max(0.0, float(vent.get("fan_power_w_per_m3h", vent.get("fan_power", 0.5))))
    heating_eff = float(heat.get("seasonal_efficiency", 1.9))
    cooling_eff = float(cool.get("seasonal_efficiency", 2.3))
    if heating_eff <= 0 or cooling_eff <= 0:
        raise ValueError("供暖和制冷季节性能系数必须大于0")

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
        "winter_design_temperature_c": float(heat.get(
            "outdoor_design_temperature_c",
            city.get("winter_design_temperature_c", -5.0),
        )),
        "summer_design_temperature_c": float(cool.get(
            "outdoor_design_temperature_c",
            city.get("summer_design_temperature_c", 35.0),
        )),
    }
```

If the request supplies recovery as a percentage greater than 1, normalize it in `web_server_server.py`; the engine contract remains a fraction from 0 to 1.

- [ ] **Step 3: Prevent explicit infiltration from double counting detailed airtightness**

Change `_detailed_orientation_loads` so the old window-air-tightness infiltration loop runs only when `"infiltration_ach"` is absent from `params["ventilation"]`:

```python
explicit_infiltration = "infiltration_ach" in (params.get("ventilation") or {})
...
if not explicit_infiltration:
    infiltration_ach = max(0.0, (8.0 - air_tightness) * 0.04)
    infiltration = 0.33 * infiltration_ach * window_area * 3.0
    infiltration_by_orientation[orientation] = (
        infiltration_by_orientation.get(orientation, 0.0) + infiltration
    )
    infiltration_ua += infiltration
```

Keep legacy behavior when the new field is missing.

- [ ] **Step 4: Calculate fresh-air and explicit-infiltration thermal demand**

After envelope heating/cooling demand is available, add:

```python
annual_use = _annual_use_inputs(params, total_area, height, defaults, city)
fresh_air_ua = (
    0.335
    * annual_use["fresh_air_flow_m3h"]
    * (1.0 - annual_use["heat_recovery_efficiency"])
)
explicit_infiltration_ua = 0.335 * annual_use["infiltration_flow_m3h"]

fresh_air_heating = fresh_air_ua * hdd_adj * 24 / 1000 * annual_use["operation_fraction"]
fresh_air_cooling = fresh_air_ua * cdd_adj * 24 / 1000 * annual_use["operation_fraction"]
infiltration_heating = explicit_infiltration_ua * hdd_adj * 24 / 1000
infiltration_cooling = explicit_infiltration_ua * cdd_adj * 24 / 1000

annual_heating_demand = heating_envelope + fresh_air_heating + infiltration_heating
annual_cooling_demand = cooling_envelope + fresh_air_cooling + infiltration_cooling
```

When detailed mode uses legacy airtightness infiltration, it is already included inside `heating_envelope`/`cooling_envelope`; keep explicit infiltration components at zero unless the explicit field is present.

- [ ] **Step 5: Replace fixed system efficiency with request seasonal efficiency**

For electric systems:

```python
heating_energy = (
    annual_heating_demand / annual_use["heating_seasonal_efficiency"]
    if h_info["fuel"] == "electric" else 0.0
)
cooling_energy = (
    annual_cooling_demand / annual_use["cooling_seasonal_efficiency"]
    if c_sys != "none" else 0.0
)
```

For gas/district/coal heating, preserve a separately named purchased-energy value and do not add it to electricity:

```python
heating_purchased_non_electric = (
    annual_heating_demand / h_info["efficiency"]
    if h_info["fuel"] not in ("electric", "none") else 0.0
)
```

- [ ] **Step 6: Calculate direct electricity with the new schedule**

Replace `op_hours` usage with `annual_use["annual_operation_hours"]` and use total fresh-air flow for fan electricity:

```python
op_hours = annual_use["annual_operation_hours"]
lighting_energy = lpd * total_area * op_hours * ctrl / 1000
equipment_energy = epd * total_area * op_hours / 1000
ventilation_energy = (
    annual_use["fresh_air_flow_m3h"]
    * annual_use["fan_power_w_per_m3h"]
    * op_hours / 1000
)
```

- [ ] **Step 7: Add equivalent-peak helper and result blocks**

Add:

```python
def _equivalent_peak(annual_demand, degree_days, indoor_t, outdoor_design_t, heating):
    delta_t = indoor_t - outdoor_design_t if heating else outdoor_design_t - indoor_t
    if annual_demand <= 0 or degree_days <= 0 or delta_t <= 0:
        return None, None
    equivalent_hours = degree_days * 24 / delta_t
    return annual_demand / equivalent_hours, equivalent_hours
```

Return rounded fields matching the design spec. Also map:

```python
result["breakdown_kwh"] = {
    "heating": round(heating_energy, 1),
    "cooling": round(cooling_energy, 1),
    "lighting": round(lighting_energy, 1),
    "equipment": round(equipment_energy, 1),
    "ventilation": round(ventilation_energy, 1),
    "dhw": round(electric_dhw_energy, 1),
}
```

This compatibility block contains electricity only.

- [ ] **Step 8: Run focused and full backend tests**

```powershell
python -m unittest tests/test_energy_calc.py -v
python -m unittest discover -s tests
```

Expected: all energy tests pass; the full suite reports no failures.

- [ ] **Step 9: Review the backend diff without staging or committing**

```powershell
git -c safe.directory=E:/bim-web diff -- energy_calc.py tests/test_energy_calc.py
```

Expected: only annual-model calculations, result fields, and their tests change.

---

### Task 3: Wire Request Validation and Historical Defaults

**Files:**
- Modify: `web_server_server.py:1339-1468`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: flat JSON fields from `calculateEnergy()`.
- Produces: nested `calc_params` dictionaries expected by Task 2.

- [ ] **Step 1: Add failing request-contract assertions**

Append to `EnergyTemplateTests`:

```python
def test_server_maps_annual_use_fields_into_calc_params(self):
    server = Path("web_server_server.py").read_text(encoding="utf-8")
    required = [
        '"area_per_person_m2"',
        '"fresh_air_m3h_per_person"',
        '"daily_operation_hours"',
        '"annual_operation_days"',
        '"heat_recovery_efficiency"',
        '"infiltration_ach"',
        '"fan_power_w_per_m3h"',
        '"seasonal_efficiency"',
        '"outdoor_design_temperature_c"',
    ]
    for field in required:
        self.assertIn(field, server)
```

- [ ] **Step 2: Run the new test and confirm failure**

```powershell
python -m unittest tests/test_energy_template.py -v
```

Expected: failure because the new request fields are not yet mapped.

- [ ] **Step 3: Add bounded request parsing inside `ai_simulate`**

Create local parsed values before `calc_params`:

```python
area_per_person = float(data.get("area_per_person_m2", 10.0))
daily_operation_hours = float(data.get("daily_operation_hours", 12.0))
annual_operation_days = int(data.get("annual_operation_days", 250))
heat_recovery_percent = float(data.get("heat_recovery_efficiency_percent", 0.0))
if area_per_person <= 0:
    return jsonify({"error": "人均占用面积必须大于0"}), 400
if not 0 <= daily_operation_hours <= 24:
    return jsonify({"error": "每日运行小时必须在0到24之间"}), 400
if not 0 <= annual_operation_days <= 365:
    return jsonify({"error": "年运行天数必须在0到365之间"}), 400
if not 0 <= heat_recovery_percent <= 100:
    return jsonify({"error": "热回收效率必须在0%到100%之间"}), 400
```

- [ ] **Step 4: Map the exact nested contract**

Add/update:

```python
"occupancy": {
    "area_per_person_m2": area_per_person,
},
"heating": {
    "system_type": data.get("heating_system", "heat_pump_air"),
    "t_set": float(data.get("t_heat", 18.0)),
    "seasonal_efficiency": float(data.get("heating_seasonal_efficiency", 1.9)),
    "outdoor_design_temperature_c": float(data.get("winter_design_temperature_c", -5.0)),
},
"cooling": {
    "system_type": data.get("cooling_system", "vrv"),
    "t_set": float(data.get("t_cool", 26.0)),
    "seasonal_efficiency": float(data.get("cooling_seasonal_efficiency", 2.3)),
    "outdoor_design_temperature_c": float(data.get("summer_design_temperature_c", 34.9)),
},
"ventilation": {
    "fresh_air_m3h_per_person": max(0.0, float(data.get("fresh_air_m3h_per_person", 30.0))),
    "heat_recovery_efficiency": heat_recovery_percent / 100.0,
    "infiltration_ach": max(0.0, float(data.get("infiltration_ach", 0.0))),
    "fan_power_w_per_m3h": max(0.0, float(data.get("fan_power_w_per_m3h", 0.5))),
},
"schedule": {
    "daily_operation_hours": daily_operation_hours,
    "annual_operation_days": annual_operation_days,
},
```

Keep old flat-field defaults accepted through the frontend reload path rather than rewriting historical database rows.

- [ ] **Step 5: Run request-contract and syntax checks**

```powershell
python -m unittest tests/test_energy_template.py -v
python -m py_compile web_server_server.py energy_calc.py
```

Expected: tests pass and both Python files compile without output.

- [ ] **Step 6: Review the route diff without staging or committing**

```powershell
git -c safe.directory=E:/bim-web diff -- web_server_server.py tests/test_energy_template.py
```

---

### Task 4: Add Step 3 Inputs and Derived Preview

**Files:**
- Modify: `templates/energy.html:1004-1090`
- Modify: `templates/energy.html:2028-2102`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: current total area, floor height, and Step 3 input controls.
- Produces: flat request fields consumed by Task 3 and derived preview text.

- [ ] **Step 1: Add failing template assertions for controls and defaults**

```python
def test_step_three_contains_annual_use_controls_and_confirmed_defaults(self):
    html = Path("templates/energy.html").read_text(encoding="utf-8")
    expected = {
        'id="param-t-heat"': 'value="18.0"',
        'id="param-t-cool"': 'value="26.0"',
        'id="param-area-per-person"': 'value="10"',
        'id="param-fresh-air-per-person"': 'value="30"',
        'id="param-daily-operation-hours"': 'value="12"',
        'id="param-annual-operation-days"': 'value="250"',
        'id="param-heat-recovery-efficiency"': 'value="0"',
        'id="param-infiltration-ach"': 'value="0"',
        'id="param-fan-power"': 'value="0.5"',
        'id="param-heating-seasonal-efficiency"': 'value="1.90"',
        'id="param-cooling-seasonal-efficiency"': 'value="2.30"',
        'id="param-winter-design-temperature"': 'value="-5"',
        'id="param-summer-design-temperature"': 'value="34.9"',
    }
    for element_id, default_value in expected.items():
        self.assertIn(element_id, html)
        self.assertIn(default_value, html)
```

- [ ] **Step 2: Run the template test and confirm failure**

```powershell
python -m unittest tests/test_energy_template.py -v
```

Expected: failure for missing Step 3 controls.

- [ ] **Step 3: Add the controls in three compact groups**

Under the existing HVAC card, add seasonal efficiencies and design outdoor temperatures. Under the building-use card, replace the old single ACH/op-hours controls with:

```html
<input type="number" id="param-area-per-person" value="10" min="0.1" step="0.1" oninput="updateAnnualUsePreview()">
<input type="number" id="param-fresh-air-per-person" value="30" min="0" step="1" oninput="updateAnnualUsePreview()">
<input type="number" id="param-daily-operation-hours" value="12" min="0" max="24" step="0.5" oninput="updateAnnualUsePreview()">
<input type="number" id="param-annual-operation-days" value="250" min="0" max="365" step="1" oninput="updateAnnualUsePreview()">
<input type="number" id="param-heat-recovery-efficiency" value="0" min="0" max="100" step="1" oninput="updateAnnualUsePreview()">
<input type="number" id="param-infiltration-ach" value="0" min="0" step="0.05">
<input type="number" id="param-fan-power" value="0.5" min="0" step="0.05">
```

Use the project’s existing `.form-group`, `.form-label`, and `.form-control` classes; do not introduce a new visual system.

- [ ] **Step 4: Add seasonal-efficiency and design-temperature controls**

```html
<input type="number" id="param-heating-seasonal-efficiency" class="form-control" value="1.90" min="0.1" step="0.05">
<input type="number" id="param-cooling-seasonal-efficiency" class="form-control" value="2.30" min="0.1" step="0.05">
<input type="number" id="param-winter-design-temperature" class="form-control" value="-5" step="0.1">
<input type="number" id="param-summer-design-temperature" class="form-control" value="34.9" step="0.1">
```

- [ ] **Step 5: Add derived preview calculation**

```javascript
function updateAnnualUsePreview() {
    const area = Math.max(0, parseFloat(document.getElementById('param-floor-area').value) || 0)
        * Math.max(1, parseInt(document.getElementById('param-floors').value) || 1);
    const areaPerPerson = Math.max(0.1, parseFloat(document.getElementById('param-area-per-person').value) || 10);
    const occupants = area / areaPerPerson;
    const freshAirPerPerson = Math.max(0, parseFloat(document.getElementById('param-fresh-air-per-person').value) || 0);
    const freshAir = occupants * freshAirPerPerson;
    const dailyHours = Math.max(0, Math.min(24, parseFloat(document.getElementById('param-daily-operation-hours').value) || 0));
    const annualDays = Math.max(0, Math.min(365, parseFloat(document.getElementById('param-annual-operation-days').value) || 0));
    const recovery = Math.max(0, Math.min(100, parseFloat(document.getElementById('param-heat-recovery-efficiency').value) || 0));
    document.getElementById('annual-use-preview').innerText =
        `计算人数 ${occupants.toFixed(1)} 人 · 新风量 ${freshAir.toFixed(0)} m³/h · ` +
        `年运行 ${(dailyHours * annualDays).toFixed(0)} h · 等效新风 ${(freshAir * (1 - recovery / 100)).toFixed(0)} m³/h`;
}
```

Call this function after geometry updates, Step 3 entry, and history reload.

- [ ] **Step 6: Send all new fields in `calculateEnergy()`**

Add exact payload keys matching Task 3:

```javascript
area_per_person_m2: parseFloat(document.getElementById('param-area-per-person').value),
fresh_air_m3h_per_person: parseFloat(document.getElementById('param-fresh-air-per-person').value),
daily_operation_hours: parseFloat(document.getElementById('param-daily-operation-hours').value),
annual_operation_days: parseInt(document.getElementById('param-annual-operation-days').value),
heat_recovery_efficiency_percent: parseFloat(document.getElementById('param-heat-recovery-efficiency').value),
infiltration_ach: parseFloat(document.getElementById('param-infiltration-ach').value),
fan_power_w_per_m3h: parseFloat(document.getElementById('param-fan-power').value),
heating_seasonal_efficiency: parseFloat(document.getElementById('param-heating-seasonal-efficiency').value),
cooling_seasonal_efficiency: parseFloat(document.getElementById('param-cooling-seasonal-efficiency').value),
winter_design_temperature_c: parseFloat(document.getElementById('param-winter-design-temperature').value),
summer_design_temperature_c: parseFloat(document.getElementById('param-summer-design-temperature').value),
```

Set the existing `param-lpd` default to `8.0` and keep `param-epd` at `15.0`. Remove the old `op_hours` payload only after history reload supports both shapes.

- [ ] **Step 7: Run template tests**

```powershell
python -m unittest tests/test_energy_template.py -v
```

Expected: all template and request-contract tests pass.

- [ ] **Step 8: Review the UI-input diff without staging or committing**

```powershell
git -c safe.directory=E:/bim-web diff -- templates/energy.html tests/test_energy_template.py
```

---

### Task 5: Separate Thermal-Demand and Electricity Results

**Files:**
- Modify: `templates/energy.html` result markup near Step 4
- Modify: `templates/energy.html:2117-2185`
- Modify: `templates/energy.html:2280-2325`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: Task 2 response blocks.
- Produces: two visible result sections and history reload support for all new inputs.

- [ ] **Step 1: Add failing result-section assertions**

```python
def test_results_separate_thermal_demand_from_electricity(self):
    html = Path("templates/energy.html").read_text(encoding="utf-8")
    self.assertIn('id="thermal-demand-results"', html)
    self.assertIn('id="electricity-results"', html)
    self.assertIn("annual_thermal_demand_kwh_th", html)
    self.assertIn("annual_electricity_kwh", html)
    self.assertIn("equivalent_peak_check_kw_th", html)
    self.assertIn("kWhₜₕ/a", html)
    self.assertIn("kWh/a", html)
    self.assertIn("等效峰值校核", html)
```

- [ ] **Step 2: Run the result test and confirm failure**

```powershell
python -m unittest tests/test_energy_template.py -v
```

Expected: failure for missing result containers and response keys.

- [ ] **Step 3: Add the thermal-demand result block**

Create a card with IDs for:

```html
<section id="thermal-demand-results">
  <h3>全年冷热需求</h3>
  <div id="thermal-heating-total"></div>
  <div id="thermal-cooling-total"></div>
  <div id="thermal-heating-envelope"></div>
  <div id="thermal-cooling-envelope"></div>
  <div id="thermal-heating-fresh-air"></div>
  <div id="thermal-cooling-fresh-air"></div>
  <div id="thermal-heating-infiltration"></div>
  <div id="thermal-cooling-infiltration"></div>
  <div id="peak-heating-check"></div>
  <div id="peak-cooling-check"></div>
</section>
```

Every thermal number is formatted with `kWhₜₕ/a`; peak checks use `kWₜₕ` and the label “等效峰值校核”.

- [ ] **Step 4: Convert the existing breakdown card to electricity-only**

Use `data.annual_electricity_kwh` as the primary source and retain `data.breakdown_kwh` fallback for old reports. Categories are heating equipment, cooling equipment, new/exhaust fan, lighting, plug equipment, and electric DHW. The total and EUI come from the electricity block, not thermal demand.

- [ ] **Step 5: Add fixed model-boundary text**

Render `data.model_boundaries` as a compact note and ensure the default text includes:

```text
HDD/CDD简化年度模型；新风制冷仅含显热；未进行8760小时模拟；等效峰值不是严格设计负荷；内部得热未逐时耦合。
```

- [ ] **Step 6: Restore new inputs from history with old-field fallbacks**

In the existing history reload function, read nested values first and flat legacy values second:

```javascript
document.getElementById('param-area-per-person').value =
    occupancy.area_per_person_m2 ?? p.area_per_person_m2 ?? 10;
document.getElementById('param-fresh-air-per-person').value =
    ventilation.fresh_air_m3h_per_person ?? p.fresh_air_m3h_per_person ?? 30;
document.getElementById('param-daily-operation-hours').value =
    schedule.daily_operation_hours ?? p.daily_operation_hours ?? 12;
document.getElementById('param-annual-operation-days').value =
    schedule.annual_operation_days ?? p.annual_operation_days ?? 250;
document.getElementById('param-heat-recovery-efficiency').value =
    (ventilation.heat_recovery_efficiency != null)
        ? ventilation.heat_recovery_efficiency * 100
        : (p.heat_recovery_efficiency_percent ?? 0);
```

Add matching fallbacks for infiltration, fan power, seasonal efficiencies, and design temperatures, then call `updateAnnualUsePreview()`.

- [ ] **Step 7: Run all frontend static tests**

```powershell
python -m unittest tests/test_energy_template.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Review the result/history diff without staging or committing**

```powershell
git -c safe.directory=E:/bim-web diff -- templates/energy.html tests/test_energy_template.py
```

---

### Task 6: End-to-End Verification with the Culture Palace Case

**Files:**
- Verify: `energy_calc.py`
- Verify: `web_server_server.py`
- Verify: `templates/energy.html`
- Verify: `tests/test_energy_calc.py`
- Verify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: the completed backend/frontend contract.
- Produces: test evidence, browser evidence, and a four-floor calculation snapshot for user review.

- [ ] **Step 1: Run syntax and full automated tests**

```powershell
python -m py_compile energy_calc.py web_server_server.py
python -m unittest discover -s tests
```

Expected: compilation succeeds with no output and the full suite has zero failures/errors.

- [ ] **Step 2: Start the Flask server with the existing local workflow**

Use the current project launcher and environment-variable login method. Do not hardcode credentials or modify `AGENTS.md`.

```powershell
python web_server_server.py
```

Expected: server listens on `http://127.0.0.1:5000` without traceback.

- [ ] **Step 3: Browser-check Step 3 defaults and validation**

Open `/energy`, select detailed mode, and confirm visible defaults:

```text
18℃, 26℃, 10 m²/人, 30 m³/(h·人), 12 h/天, 250 天/年,
0% heat recovery, 0 ACH infiltration, 0.5 W/(m³/h), HSPF 1.90,
SEER 2.30, -5℃ winter design, 34.9℃ summer design.
```

Confirm the preview shows derived people, new-air flow, 3000 annual hours, and effective new air.

- [ ] **Step 4: Run the saved four-floor envelope case**

Use the previously verified four-floor inputs:

```text
floor area 639.48 m²; one floor; roof included; floor excluded;
wall U 0.38 W/(m²·K); window U 1.90 W/(m²·K); roof U 0.34 W/(m²·K);
south/west/north/east wall areas 49.65/144.90/56.63/148.80 m²;
south/west/north/east window areas 5.25/35.10/5.25/31.20 m²;
window SHGC 0.202; Xi'an climate.
```

Expected: the page returns both thermal-demand and electricity sections; no `NaN`, `Infinity`, or mixed units appear.

- [ ] **Step 5: Verify result arithmetic in the browser response**

Check these identities using displayed or response values:

```text
heating electricity = heating thermal demand / 1.90
cooling electricity = cooling thermal demand / 2.30
fan electricity = fresh-air flow × 0.5 × 3000 / 1000
total electricity = sum of electricity end uses
```

Confirm the HDD heating equivalent peak is shown as a rough check against the four-floor workbook value of approximately `39.43 kW`, without claiming equality or design-load status.

- [ ] **Step 6: Verify history save/reload**

Save the calculation, reload the same report number, and confirm all new Step 3 fields and both result sections restore correctly. Then load an older report and confirm defaults fill missing fields without errors.

- [ ] **Step 7: Check browser console and server output**

Expected: no JavaScript errors, failed requests, Flask tracebacks, or database serialization errors.

- [ ] **Step 8: Produce a final unstaged diff summary**

```powershell
git -c safe.directory=E:/bim-web status --short
git -c safe.directory=E:/bim-web diff --stat -- energy_calc.py web_server_server.py templates/energy.html tests/test_energy_calc.py tests/test_energy_template.py docs/superpowers/specs/2026-07-15-annual-energy-thermal-demand-electricity-design.md docs/superpowers/plans/2026-07-15-annual-energy-thermal-demand-electricity.md
```

Expected: only intended feature files are reported for this work; do not stage or commit them.

---

## Plan Self-Review

- Spec coverage: all Step 3 inputs, annual thermal-demand formulas, fresh-air/infiltration handling, seasonal electricity conversion, separate result sections, peak checks, compatibility, error handling, and browser verification map to Tasks 1-6.
- Placeholder scan: the plan contains no `TBD`, `TODO`, “implement later”, or unspecified test steps.
- Type consistency: recovery is a percentage in the HTML request, a 0-1 fraction in `calc_params`; seasonal efficiencies are positive floats; annual days and daily hours derive annual hours; result keys match the spec and frontend tasks.
- Scope control: no room types, hourly weather, latent cooling, equipment selection, or unrelated refactoring is included.
