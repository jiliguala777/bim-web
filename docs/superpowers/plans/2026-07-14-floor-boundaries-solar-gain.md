# Floor Boundaries and Solar Gain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make roof and floor heat transfer opt-in in detailed mode, apply the floor contact factor, expose the annual equivalent solar irradiation parameter, and warn that PDFs must be uploaded one page at a time.

**Architecture:** Keep simple-mode calculations unchanged. Extend the existing detailed-mode request contract through `templates/energy.html` and `web_server_server.py`, then apply the inputs inside `_detailed_orientation_loads()` and expose adopted values in the result.

**Tech Stack:** Python 3, Flask request mapping, standard-library `unittest`, HTML, vanilla JavaScript.

## Global Constraints

- Detailed mode defaults `include_roof` and `include_floor` to `false`.
- Roof and floor area continue to use the single-floor area; no new area inputs are added.
- `annual_solar_irradiation_kwh_m2a` defaults to `150` kWh/(m²·a).
- Negative annual irradiation and floor contact factors are treated as `0`.
- Simple mode behavior remains unchanged.
- This work does not add peak-load calculation or multi-page PDF processing.

---

## File Structure

- `energy_calc.py`: detailed roof/floor boundary and annual solar formulas; trace fields.
- `web_server_server.py`: map new request fields into calculation parameters.
- `templates/energy.html`: controls, PDF notice, state behavior, and request payload.
- `tests/test_energy_calc.py`: formula, clamping, and compatibility tests.
- `tests/test_energy_template.py`: UI and request-contract regression checks.
- `图纸识别问题记录.md`: verified implementation status.

### Task 1: Detailed calculation behavior

**Files:**
- Modify: `tests/test_energy_calc.py`
- Modify: `energy_calc.py`

**Interfaces:**
- Consumes: `geometry.include_roof`, `geometry.include_floor`, `envelope.floor_contact_factor`, `envelope.annual_solar_irradiation_kwh_m2a`.
- Produces: `detailed_loads.ua_roof`, `detailed_loads.ua_floor`, `detailed_loads.solar_gain`, and adopted values under `inputs_used`.

- [ ] **Step 1: Write failing calculation tests**

Add tests based on `_zero_internal_load_params()`:

```python
def test_detailed_mode_excludes_roof_and_floor_by_default(self):
    params = self._zero_internal_load_params()
    params["geometry"].update({"floor_area_m2": 100, "roof_area_m2": 100})
    params["envelope"].update({"u_roof": 0.4, "u_floor": 0.3})
    result = calculate_energy(params)
    self.assertEqual(result["detailed_loads"]["ua_roof"], 0.0)
    self.assertEqual(result["detailed_loads"]["ua_floor"], 0.0)

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
```

Also test the exact annual solar formula with Shanghai HDD/CDD, clamping of negative irradiation/contact factor, and that simple mode still includes its existing roof/floor terms.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\Anaconda3\python.exe -m unittest tests.test_energy_calc -v
```

Expected: new tests fail because detailed mode still always includes both surfaces, the return lacks the new trace fields, and solar gain still uses literal `150`.

- [ ] **Step 3: Implement the minimum calculation**

At the start of `_detailed_orientation_loads()` read:

```python
geo = params.get("geometry", {})
env = params.get("envelope", {})
include_roof = geo.get("include_roof") is True
include_floor = geo.get("include_floor") is True
floor_contact_factor = max(0.0, float(env.get("floor_contact_factor", 1.0)))
annual_solar_irradiation = max(
    0.0,
    float(env.get("annual_solar_irradiation_kwh_m2a", 150.0)),
)
```

Use:

```python
oriented_solar = (
    window_area * shgc * shading_factor * annual_solar_irradiation
    * solar_factor * (cdd / max(cdd + hdd, 1))
)
ua_roof = u_roof * roof_area if include_roof else 0.0
ua_floor = u_floor * floor_area * floor_contact_factor if include_floor else 0.0
```

Return and expose both `UA` values, flags, factor, and annual irradiation. Do not change the simple-mode branch.

- [ ] **Step 4: Run calculation tests and verify GREEN**

```powershell
D:\Anaconda\Anaconda3\python.exe -m unittest tests.test_energy_calc -v
```

Expected: all calculation tests pass.

### Task 2: Page controls and request contract

**Files:**
- Modify: `tests/test_energy_template.py`
- Modify: `templates/energy.html`
- Modify: `web_server_server.py`

**Interfaces:**
- Consumes: DOM controls `include-roof`, `include-floor`, `param-annual-solar-irradiation`.
- Produces: JSON booleans `include_roof`, `include_floor`, numeric `annual_solar_irradiation_kwh_m2a`, and matching `calc_params`.

- [ ] **Step 1: Write failing page/contract tests**

```python
def test_detailed_boundary_controls_default_off_and_pdf_notice_exists(self):
    html = Path("templates/energy.html").read_text(encoding="utf-8")
    self.assertIn('id="include-roof"', html)
    self.assertIn('id="include-floor"', html)
    self.assertNotIn('id="include-roof" checked', html)
    self.assertNotIn('id="include-floor" checked', html)
    self.assertIn('id="param-annual-solar-irradiation"', html)
    self.assertIn("PDF 目前仅处理单页", html)
```

Add a second test reading both source files and checking the three payload fields and server mappings exist with the exact property names.

- [ ] **Step 2: Run tests and verify RED**

```powershell
D:\Anaconda\Anaconda3\python.exe -m unittest tests.test_energy_template -v
```

Expected: assertions fail because controls, warning, payload fields, and server mappings are absent.

- [ ] **Step 3: Implement controls and state**

Add unchecked detailed-mode checkboxes and a numeric input:

```html
<input type="checkbox" id="include-roof" onchange="updateBoundaryControlState()">
<input type="checkbox" id="include-floor" onchange="updateBoundaryControlState()">
<input type="number" id="param-annual-solar-irradiation"
       class="form-control" value="150" min="0" step="1">
```

Wrap existing roof and floor groups with stable IDs. Add `updateBoundaryControlState()` to disable the corresponding controls only in detailed mode, and call it from `setCalculationMode()`.

Add the PDF notice and request fields:

```javascript
include_roof: calculationMode === 'detailed'
    && document.getElementById('include-roof').checked,
include_floor: calculationMode === 'detailed'
    && document.getElementById('include-floor').checked,
annual_solar_irradiation_kwh_m2a: Math.max(
    0,
    parseFloat(document.getElementById('param-annual-solar-irradiation').value) || 0
),
```

- [ ] **Step 4: Forward fields through Flask mapping**

Add to geometry:

```python
"include_roof": data.get("include_roof") is True,
"include_floor": data.get("include_floor") is True,
```

Add to envelope:

```python
"annual_solar_irradiation_kwh_m2a": max(
    0.0,
    float(data.get("annual_solar_irradiation_kwh_m2a", 150.0)),
),
```

- [ ] **Step 5: Run page tests and verify GREEN**

```powershell
D:\Anaconda\Anaconda3\python.exe -m unittest tests.test_energy_template -v
```

Expected: all template tests pass.

### Task 3: Documentation and full verification

**Files:**
- Modify: `图纸识别问题记录.md`

**Interfaces:**
- Consumes: verified Tasks 1 and 2 behavior.
- Produces: a status note distinguishing completed annual-model fixes from deferred peak-load/PDF work.

- [ ] **Step 1: Update the issue record**

Append a dated note: detailed mode defaults roof/floor off, floor contact factor is applied, annual solar irradiation is explicit, PDFs remain manual single-page uploads, and peak loads remain unresolved.

- [ ] **Step 2: Run focused tests**

```powershell
D:\Anaconda\Anaconda3\python.exe -m unittest tests.test_energy_calc tests.test_energy_template -v
```

Expected: all tests pass.

- [ ] **Step 3: Run syntax and diff checks**

```powershell
D:\Anaconda\Anaconda3\python.exe -m py_compile energy_calc.py web_server_server.py
git -c safe.directory=E:/bim-web diff --check -- energy_calc.py web_server_server.py templates/energy.html tests/test_energy_calc.py tests/test_energy_template.py 图纸识别问题记录.md
```

Expected: successful exits and no diff-check output.

- [ ] **Step 4: Review scoped diff**

```powershell
git -c safe.directory=E:/bim-web diff -- energy_calc.py web_server_server.py templates/energy.html tests/test_energy_calc.py tests/test_energy_template.py 图纸识别问题记录.md
```

Expected: only the approved calculation, UI, request, tests, and issue-note changes are present in the reviewed hunks.
