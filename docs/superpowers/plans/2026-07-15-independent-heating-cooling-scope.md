# Independent Heating and Cooling Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users independently enable heating and cooling calculations so disabled sides produce no thermal demand or peak and contribute no equipment energy.

**Architecture:** Add explicit boolean scope flags at the page/request boundary, persist them in the nested heating/cooling parameter blocks, and make the calculation engine treat them as authoritative. Preserve old reports by inferring missing flags from system type, and render disabled results as not applicable rather than physical zero load.

**Tech Stack:** Flask, Python `unittest`, server-rendered HTML, vanilla JavaScript.

## Global Constraints

- Heating and cooling can be enabled independently or together, but not both disabled.
- Disabled thermal-demand components and equivalent peak values are `None`/JSON `null`.
- Disabled equipment energy is numeric zero and excluded from total energy.
- Existing reports without `enabled` remain loadable.
- Do not implement hourly weather, dynamic thermal simulation, or service-area fields.
- Do not stage, commit, push, deploy, or clean unrelated files.

---

### Task 1: Calculation engine scope gating

**Files:**
- Modify: `tests/test_energy_calc.py`
- Modify: `energy_calc.py`

**Interfaces:**
- Consumes: `params["heating"]["enabled"]` and `params["cooling"]["enabled"]`, defaulting to enabled when absent.
- Produces: `calculation_scope`, nullable disabled-side thermal/peak values, and zero disabled-side equipment energy.

- [ ] Add tests for heating-only, cooling-only, both-enabled backward compatibility, and both-disabled rejection.
- [ ] Run `\.\.venv\Scripts\python.exe -m unittest tests.test_energy_calc -v` and confirm the new tests fail for missing scope behavior.
- [ ] Implement strict scope normalization and gate annual thermal demand, equipment energy, equivalent peaks, and model-boundary messages.
- [ ] Re-run `\.\.venv\Scripts\python.exe -m unittest tests.test_energy_calc -v` and confirm all energy calculator tests pass.

### Task 2: Flask request validation and persistence

**Files:**
- Modify: `tests/test_energy_template.py`
- Modify: `web_server_server.py`

**Interfaces:**
- Consumes: flat request booleans `calculate_heating` and `calculate_cooling`.
- Produces: nested `heating.enabled` and `cooling.enabled` booleans passed to `calculate_energy()` and stored in report params.

- [ ] Add route tests for one-side selection, both enabled, both disabled, and invalid boolean values.
- [ ] Run the route test module and confirm the new tests fail.
- [ ] Add a boolean parser that accepts JSON booleans and legacy missing fields, rejects ambiguous strings, and enforces at least one enabled side.
- [ ] Re-run the route test module and confirm it passes.

### Task 3: Third-step controls and request payload

**Files:**
- Modify: `tests/test_energy_template.py`
- Modify: `templates/energy.html`

**Interfaces:**
- Produces: checkboxes `calculate-heating` and `calculate-cooling`, `updateCalculationScopeControls()`, and request fields `calculate_heating`/`calculate_cooling`.

- [ ] Add template assertions for two checked-by-default independent controls, disabled-side input groups, and front-end rejection when neither side is checked.
- [ ] Run the template test module and confirm the assertions fail.
- [ ] Add the scope control block, stable heating/cooling group containers, state synchronization, and payload fields.
- [ ] Re-run the template test module and confirm it passes.

### Task 4: History restoration and result rendering

**Files:**
- Modify: `tests/test_energy_template.py`
- Modify: `templates/energy.html`

**Interfaces:**
- Consumes: new nested `enabled`, legacy nested system types, or legacy flat system types.
- Produces: restored checkbox state and explicit “未启用” labels for disabled thermal demand and peaks.

- [ ] Add tests for new-report restoration, legacy inference, nullable display formatting, and disabled energy-row annotation.
- [ ] Run the template tests and confirm the new assertions fail.
- [ ] Implement restoration precedence and calculation-scope-aware rendering without breaking legacy reports.
- [ ] Re-run the template tests and confirm they pass.

### Task 5: End-to-end verification

**Files:**
- Verify: `energy_calc.py`
- Verify: `web_server_server.py`
- Verify: `templates/energy.html`
- Verify: `tests/test_energy_calc.py`
- Verify: `tests/test_energy_template.py`

- [ ] Run `\.\.venv\Scripts\python.exe -m unittest discover -s tests -v` and require zero failures.
- [ ] Call the real Flask route with the culture-palace parameter shape and `calculate_cooling=false`; verify cooling thermal/peak fields are null, cooling electricity is zero, and total excludes cooling electricity.
- [ ] Inspect `git -c safe.directory=E:/bim-web diff -- energy_calc.py web_server_server.py templates/energy.html tests/test_energy_calc.py tests/test_energy_template.py` for scope-only changes.
- [ ] Confirm no staging, commit, push, deployment, database write, or unrelated cleanup occurred.
