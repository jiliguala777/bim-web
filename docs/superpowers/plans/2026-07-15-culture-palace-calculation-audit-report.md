# Culture Palace Calculation Audit Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a traceable Chinese Markdown audit report that reconstructs report `BIM-20260714-8219`, classifies every material input, and explains how missing or assumed parameters affect accuracy.

**Architecture:** Treat the current SQLite report and executable calculation code as the primary evidence. Independently recompute the main intermediate values, then write one project-local Markdown report that separates facts, defaults, derived values, missing physics, and inapplicable settings.

**Tech Stack:** Markdown, SQLite via Python standard library, PowerShell, current Flask/Python calculation code.

## Global Constraints

- Create only `docs/文化宫四层_当前负荷与年度能耗计算审计报告_2026-07-15.md` plus this plan.
- Do not modify website calculation code or database records.
- Do not invent missing values or tune inputs to match `39.43 kW`.
- Keep `kWh_th/a`, `kWh/a`, and `kW_th` distinct.
- Mark the fourth-floor cooling result as inapplicable to the original design.
- Do not stage, commit, push, deploy, or clean unrelated workspace files.
- Do not include passwords, keys, server addresses, or other credentials.

---

### Task 1: Capture the authoritative evidence set

**Files:**
- Read: `users.db`
- Read: `energy_calc.py`
- Read: `web_server_server.py`
- Read: `docs/新对话交接_文化宫年度能耗与运行环境_2026-07-15.md`
- Create: none

**Interfaces:**
- Consumes: SQLite report number `BIM-20260714-8219`.
- Produces: a checked evidence set containing `params`, `results`, calculation formulas, and report metadata for Tasks 2 and 3.

- [ ] **Step 1: Read the target database row without write access**

Run a Python standard-library SQLite query using URI mode `mode=ro` and select `report_number`, `created_at`, `geometry_used`, `params`, and `results` from `reports` where `report_number = 'BIM-20260714-8219'`.

Expected: exactly one row dated `2026-07-15 06:31:19`, with `calculation_mode = detailed` and `building.height = 6.0`.

- [ ] **Step 2: Verify the executable formulas**

Run:

```powershell
rg -n -C 4 "CLIMATE_DB|hdd_adj|cdd_adj|heating_envelope|fresh_air_ua|annual_infiltration_ua|annual_heating_demand|heating_energy|cooling_energy|lighting_energy|equipment_energy|ventilation_energy|dhw_energy|equivalent_peak" energy_calc.py
```

Expected: formulas are found in the current file, including HDD/CDD demand, scheduled fresh air, explicit ACH infiltration, seasonal efficiency conversion, and equivalent-peak checks.

- [ ] **Step 3: Record evidence conflicts for the report**

Confirm all five conflicts from the database values:

1. `cooling.system_type = vrv` although the fourth floor has no original cooling system;
2. `geometry.include_roof = true` while `detailed_envelope.roof.included = false`;
3. raw geometry reports `wall_area_m2 = 2674.0125` and `window_area_m2 = 1554.1092`, but results use `400.0` and `76.8` from manual orientation entries;
4. raw geometry reports `door_area_m2 = 32.1375`, but all detailed orientation door areas are zero;
5. `climate.city_id = xian` is used for the Yuncheng project.

Expected: the report distinguishes the stored raw value from the value actually consumed by the detailed calculation.

### Task 2: Independently reconstruct the current calculation

**Files:**
- Read: `users.db`
- Read: `energy_calc.py`
- Create: none

**Interfaces:**
- Consumes: the checked parameter and result dictionaries from Task 1.
- Produces: intermediate values and formula substitutions for the report.

- [ ] **Step 1: Recompute derived operating inputs**

Use these equations and values:

```text
occupants = 639.48 / 10 = 63.948 persons
fresh_air_flow = 63.948 × 30 = 1918.44 m³/h
annual_operation_hours = 12 × 250 = 3000 h/a
operation_fraction = 3000 / 8760 = 0.34246575
infiltration_flow = 639.48 × 6.0 × 0 = 0 m³/h
```

Expected: values match `results.derived_inputs` within displayed rounding.

- [ ] **Step 2: Recompute transmission UA**

Use the manual detailed areas:

```text
wall UA = (148.80 + 49.65 + 144.90 + 56.63) × 0.38
        = 399.98 × 0.38 = 151.9924 W/K
window UA = (31.20 + 5.25 + 35.10 + 5.25) × 1.90
          = 76.80 × 1.90 = 145.92 W/K
roof UA = 639.48 × 0.34 = 217.4232 W/K
floor UA = 0 W/K
door UA = 0 W/K in the detailed orientation calculation
transmission UA = 151.9924 + 145.92 + 217.4232 = 515.3356 W/K
```

Expected: rounded total `515.336 W/K` matches `results.detailed_loads.total_ua`.

- [ ] **Step 3: Recompute annual heating demand**

Use:

```text
HDD18 = 2400 K·d
heating setpoint = 18°C, so adjusted HDD = 2400 K·d
envelope heating = 515.3356 × 2400 × 24 / 1000
                 = 29683.33056 kWh_th/a
fresh-air UA = 0.335 × 1918.44 × (1 - 0) = 642.6774 W/K
fresh-air heating = 642.6774 × 2400 × 24 / 1000 × (3000 / 8760)
                  = 12677.47233 kWh_th/a
infiltration heating = 0
heating total = 29683.33056 + 12677.47233
              = 42360.80289 kWh_th/a
```

Expected: values match database results `29683.331`, `12677.472`, and `42360.803` after rounding.

- [ ] **Step 4: Reconstruct, but flag, the cooling calculation**

Document rather than endorse:

```text
CDD26 = 200 K·d
cooling setpoint = 26°C, so adjusted CDD = 200 K·d
cooling total recorded = 3674.032 kWh_th/a
cooling electricity recorded = 3674.032 / 2.3 = 1597.405 kWh/a
```

Expected: explicitly state that these values are computationally reproducible but not applicable to the original fourth-floor design because cooling should be disabled.

- [ ] **Step 5: Recompute annual end-use electricity**

Use:

```text
heating electricity = 42360.803 / 1.9 = 22295.159 kWh/a
fan electricity = 1918.44 × 0.5 × 3000 / 1000 = 2877.660 kWh/a
lighting = 8 × 639.48 × 3000 / 1000 = 15347.520 kWh/a
plug equipment = 15 × 639.48 × 3000 / 1000 = 28776.600 kWh/a
electric DHW = recorded model result 5504.898 kWh/a
recorded total = 76399.242 kWh/a
recorded EUI = 76399.242 / 639.48 = 119.471 kWh/(m²·a)
```

Expected: explain that the recorded total includes the inapplicable `1597.405 kWh/a` cooling electricity. Also provide the direct arithmetic correction with cooling removed: `74801.837 kWh/a`, before any other parameter corrections.

- [ ] **Step 6: Check equivalent peak labels and arithmetic**

Document recorded values `26.634 kW_th` heating and `10.306 kW_th` cooling, and state that `_equivalent_peak()` is a quantity-scale check derived from UA, design temperature difference, and equivalent full-load hours. It is not a strict design-load calculation and cannot be directly calibrated to `39.43 kW` by arbitrary input changes.

### Task 3: Write the Markdown audit report

**Files:**
- Create: `docs/文化宫四层_当前负荷与年度能耗计算审计报告_2026-07-15.md`
- Read: `docs/superpowers/specs/2026-07-15-culture-palace-calculation-audit-report-design.md`

**Interfaces:**
- Consumes: evidence and recomputed values from Tasks 1 and 2.
- Produces: the complete user-readable audit report.

- [ ] **Step 1: Write the report header and executive summary**

Include the report number, database timestamp, calculation mode, scope, model type, current headline results, and the conclusion that fourth-floor cooling is inapplicable.

- [ ] **Step 2: Write the parameter inventory**

Create grouped Markdown tables for geometry, envelope, climate, operation, ventilation, systems, lighting, equipment, and DHW. Each row must include current value, unit, classification, actual calculation use, and confidence/issue.

- [ ] **Step 3: Write the step-by-step calculation**

Use every formula and substituted value from Task 2. Show units and keep thermal demand, purchased electricity, and peak power separate.

- [ ] **Step 4: Write the conflict and missing-parameter analysis**

Include the five data conflicts from Task 1 and a missing-parameter matrix covering weather, ACH, heat recovery, schedules, occupancy, LPD/EPD, heat-pump performance, heating solar gains, cooling latent load, thermal bridges, openings, adjacent unheated spaces, intermittent corrections, and thermal mass.

For each missing item, state the affected result, likely bias direction when defensible, whether the net direction is uncertain, priority, and recommended source.

- [ ] **Step 5: Write confidence ratings and next actions**

Rate geometry/envelope, annual heating thermal demand, heating electricity, non-HVAC electricity, cooling, and equivalent peak separately. Recommend this sequence: disable fourth-floor cooling; verify ACH and operation; verify system performance and internal-use parameters; obtain Yuncheng hourly weather; then consider an hourly dynamic model.

### Task 4: Verify report completeness and arithmetic

**Files:**
- Read: `docs/文化宫四层_当前负荷与年度能耗计算审计报告_2026-07-15.md`
- Read: `users.db`
- Modify only if needed: `docs/文化宫四层_当前负荷与年度能耗计算审计报告_2026-07-15.md`

**Interfaces:**
- Consumes: the completed report.
- Produces: a self-consistent final Markdown artifact with no placeholders or unsupported precision.

- [ ] **Step 1: Scan for placeholders and unit errors**

Run:

```powershell
rg -n "[T]BD|[T]ODO|待补|待[定]|kWh_th|kWh/a|kW_th|39.43|VRV|ACH|HDD|CDD" "docs/文化宫四层_当前负荷与年度能耗计算审计报告_2026-07-15.md"
```

Expected: no unfinished placeholders; every occurrence of an energy or power result uses the correct unit.

- [ ] **Step 2: Verify required database numbers are present**

Check the report contains at least: `639.48`, `6.0`, `399.98`, `76.80`, `515.336`, `42360.803`, `29683.331`, `12677.472`, `3674.032`, `22295.159`, `1597.405`, `76399.242`, `119.47`, `26.634`, and `39.43`.

Expected: every number is accompanied by its meaning and unit rather than appearing as an unexplained value.

- [ ] **Step 3: Check scope constraints**

Run `git -c safe.directory=E:/bim-web status --short` and verify that implementation added only the target report and this plan/design documentation, without modifying code, database, credentials, or unrelated files.

Expected: no staging, commit, push, deployment, database write, or unrelated cleanup occurred.
