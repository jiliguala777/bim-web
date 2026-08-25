# PDF Exterior Openings and Energy Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognize exterior door/window gaps, conservatively repair unknown small exterior gaps, produce a confirmed building footprint, and convert its area, perimeter, door widths, and window widths into energy geometry.

**Architecture:** PDF-native accepted wall candidates remain the only geometric anchors. New modules classify exterior walls, enumerate collinear gaps, score door/window evidence from the existing ten-channel probabilities, build traceable virtual closure edges, and select one orthogonal exterior footprint. The debug pipeline remains non-load-bearing until a server-side hash and scale confirmation writes a new `recognition.json` payload.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Flask, vanilla JavaScript, `unittest`, existing PDF fusion JSON contracts.

**Spec:** `docs/superpowers/specs/2026-08-25-pdf-exterior-openings-energy-design.md`

## Global Constraints

- Process only horizontal and vertical geometry; do not add diagonal or curved exterior repair.
- Door and window objects remain separate from real walls; virtual bridges exist only in exterior topology.
- Model probabilities cannot create a free-floating opening. Every opening must be anchored to one enumerated gap between two exterior wall endpoints.
- Unknown automatic repair is one straight segment only and is limited to `0.6 m` with confirmed scale or `max(4, min(32, round(min(image_width, image_height) * 0.01)))` without scale.
- Automatic analysis always returns `load_geometry_ready: false`; only server-side confirmation plus confirmed scale can enable energy geometry.
- Preserve old ONNX, vector-raster, room-topology, and existing energy routes for their current inputs.
- New modules must not import `floorplan_onnx`, `floorplan_page_pipeline`, `floorplan_rooms`, `floorplan_topology_repair`, or `vector_pdf_scale`.
- If legacy Python logic is useful, copy the required behavior into a new module and test it; do not import or mix old-model files.
- Every fixture builder named in the test snippets (`gap`, `opening`, `supported_probabilities`, rectangle/footprint builders, clients, paths, and injected runners) must be defined locally in that task's test module. Builders must return literal dictionaries/arrays matching the documented interface and must not import prohibited legacy modules.
- Use TDD for every production change and stage only the files named by each task.

---

### Task 1: Exterior Wall Selection and Anchored Gap Enumeration

**Files:**
- Create: `vector_pdf_exterior.py`
- Create: `tests/test_vector_pdf_exterior.py`

**Interfaces:**
- Consumes: the list returned by `fuse_line_candidates(candidates, probabilities, image_size, building_roi, thresholds)`, probability array `(10, H, W)`, image size `(W, H)`, building ROI.
- Produces: `ExteriorThresholds`, `select_exterior_walls(candidates, probabilities, image_size, building_roi) -> list[dict]`, and `enumerate_exterior_gaps(exterior_walls, image_size, building_roi) -> list[dict]`.
- Each exterior wall adds `inside_direction`, `footprint_inside_mean`, `footprint_outside_mean`, and preserves `candidate_id`.
- Each gap contains `gap_id`, `orientation`, `start_px`, `end_px`, `width_px`, `host_wall_ids`, `inside_direction`, and `reason_codes`.

- [ ] **Step 1: Write failing exterior-wall and gap tests**

```python
def test_selects_wall_with_one_footprint_interior_side():
    probabilities = supported_probabilities(200, 120)
    probabilities[0, 31:80, 20:181] = 0.9
    walls = [accepted_wall("left", (20, 30), (20, 80), "vertical")]
    result = select_exterior_walls(walls, probabilities, (200, 120), [10, 10, 190, 110])
    assert result[0]["inside_direction"] == "right"

def test_enumerates_only_unambiguous_collinear_gap():
    walls = [
        exterior_wall("a", (10, 20), (70, 20), "horizontal", "down"),
        exterior_wall("b", (90, 20), (150, 20), "horizontal", "down"),
    ]
    gaps = enumerate_exterior_gaps(walls, (200, 120), [0, 0, 200, 120])
    assert gaps[0]["start_px"] == [70, 20]
    assert gaps[0]["end_px"] == [90, 20]
    assert gaps[0]["host_wall_ids"] == ["a", "b"]
```

Also test that different inside directions, a perpendicular structural crossing, multiple competing endpoints, and ROI-exterior gaps produce no automatic gap and a traceable rejected-gap record.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m unittest tests.test_vector_pdf_exterior -v`  
Expected: `ModuleNotFoundError: No module named 'vector_pdf_exterior'`.

- [ ] **Step 3: Implement exterior selection and gap enumeration**

```python
@dataclass(frozen=True)
class ExteriorThresholds:
    side_offset_fraction: float = 0.006
    side_offset_min_px: int = 4
    side_offset_max_px: int = 16
    footprint_inside_mean_min: float = 0.50
    footprint_side_difference_min: float = 0.25
    footprint_boundary_mean_min: float = 0.35
    collinear_tolerance_px: int = 2

def select_exterior_walls(candidates, probabilities, image_size, building_roi):
    """Return accepted walls with a supported building-inside side."""

def enumerate_exterior_gaps(exterior_walls, image_size, building_roi):
    """Merge overlapping collinear walls, preserve IDs, and enumerate adjacent endpoint gaps."""
```

Use candidate-local probability windows as in `vector_pdf_fusion.py`. A wall is exterior when one side passes the footprint-interior threshold and exceeds the other side by the configured difference, or when boundary support and a weaker but unambiguous side difference agree. Sort intervals with explicit numeric keys so identical duplicates never compare dictionaries.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_vector_pdf_exterior tests.test_vector_pdf_fusion tests.test_vector_pdf_rooms -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- vector_pdf_exterior.py tests/test_vector_pdf_exterior.py
git commit -m "feat: identify exterior wall gaps"
```

---

### Task 2: Door and Window Classification on Exterior Gaps

**Files:**
- Create: `vector_pdf_openings.py`
- Create: `tests/test_vector_pdf_openings.py`

**Interfaces:**
- Consumes: Task 1 gap dictionaries and the strict ten-channel probability array.
- Produces: `OpeningThresholds` and `classify_exterior_openings(gaps, probabilities, image_size) -> dict`.
- Return shape has three lists named `accepted_openings`, `ambiguous_openings`, and `unclassified_gaps`; every member is a complete opening or gap dictionary defined by this task.
- Opening fields follow `pdf-exterior-topology/1`: stable ID, kind, orientation, endpoints, `width_px`, nullable `width_m`, host wall IDs, exterior flag, confidence, evidence, and reason codes.

- [ ] **Step 1: Write failing door/window/ambiguity tests**

```python
def test_classifies_door_from_line_and_both_endpoints():
    probs = np.zeros((10, 100, 160), np.float32)
    probs[5, 48:53, 60:101] = 0.9
    probs[8, 45:56, 55:66] = 0.9
    probs[8, 45:56, 95:106] = 0.9
    result = classify_exterior_openings([gap(60, 50, 100, 50)], probs, (160, 100))
    assert result["accepted_openings"][0]["kind"] == "door"
    assert result["accepted_openings"][0]["width_px"] == 40.0

def test_line_without_two_endpoint_support_is_unclassified():
    probs = door_line_only_probabilities()
    result = classify_exterior_openings([gap(60, 50, 100, 50)], probs, (160, 100))
    assert result["accepted_openings"] == []
    assert result["unclassified_gaps"][0]["gap_id"] == "gap-0001"
```

Add literal fixtures for a window, door/window score difference below the ambiguity margin, vertical gaps, border-clipped gaps, and fully out-of-image gaps.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_vector_pdf_openings -v`  
Expected: missing module error.

- [ ] **Step 3: Implement local semantic sampling**

```python
@dataclass(frozen=True)
class OpeningThresholds:
    line_mean_min: float = 0.30
    endpoint_max_min: float = 0.35
    endpoint_radius_px: int = 8
    ambiguity_margin: float = 0.08

def classify_exterior_openings(gaps, probabilities, image_size):
    """Classify only gap-anchored exterior door/window candidates."""
```

Use channel indices `5/8` for doors and `6/9` for windows. Require the line mean and both endpoint maxima. Confidence is the minimum of normalized line and endpoint evidence, clipped to `[0, 1]`. Do not create geometry outside the input gap.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_vector_pdf_openings tests.test_vector_pdf_exterior tests.test_vector_pdf_fusion -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add -- vector_pdf_openings.py tests/test_vector_pdf_openings.py
git commit -m "feat: classify exterior door and window gaps"
```

---

### Task 3: Virtual Bridges, Small-Gap Repair, and Exterior Footprint

**Files:**
- Modify: `vector_pdf_exterior.py`
- Modify: `tests/test_vector_pdf_exterior.py`

**Interfaces:**
- Consumes: exterior walls, all enumerated gaps, Task 2 accepted openings, probability array, image size, ROI, optional confirmed `scale_m_per_px`.
- Produces: `build_exterior_topology(exterior_walls, gaps, openings, probabilities, image_size, building_roi, *, scale_m_per_px=None) -> dict`.
- Topology includes `bridges`, `unresolved_gaps`, one polygon candidate, area, perimeter, source IDs, opening IDs, status, confirmation flags, and `load_geometry_ready: false`.

- [ ] **Step 1: Write failing bridge and footprint tests**

```python
def test_opening_bridge_closes_rectangle_without_becoming_real_wall():
    walls = rectangle_with_bottom_gap(door_gap=(40, 60))
    topology = build_exterior_topology(
        walls, [gap(40, 80, 60, 80)], [door_opening(40, 80, 60, 80)],
        supported_footprint(), (100, 100), [0, 0, 100, 100],
    )
    assert topology["status"] == "review_required"
    assert topology["area_px2"] == 3600.0
    assert topology["bridges"][0]["bridge_type"] == "opening_bridge"
    assert topology["opening_ids"] == ["opening-0001"]

def test_unknown_gap_repairs_at_metric_boundary_only():
    walls = rectangle_with_bottom_gap(door_gap=(40, 60))
    accepted_gap = gap(40, 80, 60, 80)
    rejected_gap = gap(40, 80, 61, 80)
    accepted = build_exterior_topology(
        walls, [accepted_gap], [], supported_footprint(),
        (100, 100), [0, 0, 100, 100], scale_m_per_px=0.03,
    )  # 20 px = 0.6 m
    rejected = build_exterior_topology(
        walls, [rejected_gap], [], supported_footprint(),
        (100, 100), [0, 0, 100, 100], scale_m_per_px=0.03,
    )  # 21 px = 0.63 m
    assert accepted["bridges"][0]["bridge_type"] == "small_gap_repair"
    assert rejected["unresolved_gaps"][0]["reason_codes"] == ["gap_exceeds_repair_limit"]
```

Add tests for the no-scale pixel formula at its exact boundary, a concave orthogonal footprint, duplicate source provenance, no bounding-rectangle fallback, multiple comparable faces returning `ambiguous_exterior`, and a corner gap remaining unresolved.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_vector_pdf_exterior -v`  
Expected: failure because `build_exterior_topology` is absent.

- [ ] **Step 3: Implement bridges and orthogonal face selection**

```python
def _repair_limit_px(image_size, scale_m_per_px):
    if scale_m_per_px is not None:
        return 0.6 / scale_m_per_px
    return max(4, min(32, round(min(image_size) * 0.01)))

def build_exterior_topology(
    exterior_walls, gaps, openings, probabilities, image_size,
    building_roi, *, scale_m_per_px=None,
):
    """Build traceable bridges and select the largest supported orthogonal footprint."""
```

Copy the required orthogonal graph behavior into this new module rather than importing legacy topology files. Merge exact/overlapping intervals with provenance, split intersections, enumerate bounded faces, and select the largest ROI-contained face supported by `footprint_interior`. Keep all bridges out of the real-wall list.

- [ ] **Step 4: Run topology tests**

Run: `python -m unittest tests.test_vector_pdf_exterior tests.test_vector_pdf_rooms -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add -- vector_pdf_exterior.py tests/test_vector_pdf_exterior.py
git commit -m "feat: close traceable exterior footprint"
```

---

### Task 4: Scale and Energy Geometry Conversion

**Files:**
- Create: `vector_pdf_energy_geometry.py`
- Create: `tests/test_vector_pdf_energy_geometry.py`

**Interfaces:**
- Produces: `apply_scale_to_exterior(topology, openings, scale_m_per_px) -> tuple[dict, list[dict]]` and `build_exterior_energy_geometry(topology, openings, *, storey_height_m, floors, door_height_m, window_height_m, door_repeat_count=1, window_repeat_count=None) -> dict`.
- Energy result contains per-floor footprint area, total floor area, exterior perimeter, gross exterior wall area, opaque wall area, door/window areas, and separate door/window total widths.

- [ ] **Step 1: Write failing scale and net-wall tests**

```python
def test_applies_scale_to_footprint_and_separate_opening_widths():
    topology, openings = apply_scale_to_exterior(
        footprint(area=10000, perimeter=400),
        [opening("door", 30), opening("window", 50)],
        0.02,
    )
    assert topology["area_m2"] == 4.0
    assert topology["perimeter_m"] == 8.0
    assert openings[0]["width_m"] == 0.6
    assert openings[1]["width_m"] == 1.0

def test_builds_opaque_wall_after_separate_door_window_deductions():
    geometry = build_exterior_energy_geometry(
        scaled_footprint(area=100, perimeter=40), scaled_openings(door=2, window=8),
        storey_height_m=3, floors=2, door_height_m=2.1, window_height_m=1.5,
        door_repeat_count=1, window_repeat_count=2,
    )
    assert geometry["gross_exterior_wall_area_m2"] == 240.0
    assert geometry["door_area_m2"] == 4.2
    assert geometry["window_area_m2"] == 24.0
    assert geometry["wall_area_m2"] == 211.8
```

Test non-finite/zero scale, invalid heights/counts, and openings exceeding gross wall area.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_vector_pdf_energy_geometry -v`  
Expected: missing module error.

- [ ] **Step 3: Implement immutable conversion and validation**

```python
def apply_scale_to_exterior(topology, openings, scale_m_per_px):
    """Deep-copy and add metric area, perimeter, and opening widths."""

def build_exterior_energy_geometry(
    topology, openings, *, storey_height_m, floors,
    door_height_m, window_height_m,
    door_repeat_count=1, window_repeat_count=None,
):
    """Return net opaque wall and separate opening areas for energy_calc.py."""
```

Reject booleans, non-finite numbers, non-positive dimensions, repeat counts above `floors`, and negative net wall area. Default `window_repeat_count` to `floors`.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_vector_pdf_energy_geometry tests.test_energy_calc -v`  
Expected: all tests pass.

- [ ] **Step 5: Commit Task 4**

```powershell
git add -- vector_pdf_energy_geometry.py tests/test_vector_pdf_energy_geometry.py
git commit -m "feat: convert exterior footprint to energy geometry"
```

---

### Task 5: Publish Opening and Exterior Debug Artifacts

**Files:**
- Modify: `vector_pdf_fusion_pipeline.py`
- Modify: `tests/test_vector_pdf_fusion_pipeline.py`

**Interfaces:**
- Extend `analyze_vector_pdf_page(pdf_path, page_number, output_dir, model_config, crop_bbox_page_px=None, model_runner=run_vector_probabilities)` with `opening_candidates`, `exterior_topology`, and `exterior_summary`.
- Publish `pdf_opening_candidates.json`, `pdf_exterior_topology.json`, and `pdf_exterior_overlay.png` atomically.
- Preserve top-level `load_geometry_ready: false` for every pipeline result.

- [ ] **Step 1: Write failing pipeline artifact test**

```python
def test_pipeline_publishes_unconfirmed_exterior_and_openings():
    result = analyze_vector_pdf_page(room_with_door_gap_pdf, 1, output, object(), model_runner=opening_runner)
    assert result["load_geometry_ready"] is False
    assert result["exterior_topology"]["confirmed"] is False
    assert result["exterior_topology"]["load_geometry_ready"] is False
    assert (output / "pdf_opening_candidates.json").is_file()
    assert (output / "pdf_exterior_topology.json").is_file()
    assert (output / "pdf_exterior_overlay.png").is_file()
```

Add a model-unavailable test asserting no fake exterior file is confirmed and a crop test asserting all exterior/opening analysis coordinates are crop-local while page coordinates remain traceable.

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_vector_pdf_fusion_pipeline -v`  
Expected: missing exterior keys/artifacts.

- [ ] **Step 3: Integrate Tasks 1–3 and overlay rendering**

```python
exterior_walls = select_exterior_walls(candidates, model_result.probabilities, (width, height), roi)
gaps = enumerate_exterior_gaps(exterior_walls, (width, height), roi)
opening_result = classify_exterior_openings(gaps, model_result.probabilities, (width, height))
exterior = build_exterior_topology(
    exterior_walls, gaps, opening_result["accepted_openings"],
    model_result.probabilities, (width, height), roi,
)
```

Write the three JSON/image artifacts with existing atomic helpers. Draw real walls green, doors blue, windows cyan, opening bridges blue dashed, small repairs orange dashed, unresolved gaps red, and footprint purple.

- [ ] **Step 4: Run pipeline and isolation tests**

Run: `python -m unittest tests.test_vector_pdf_fusion_pipeline tests.test_vector_pdf_exterior tests.test_vector_pdf_openings -v`  
Run: `rg -n "from (floorplan_onnx|floorplan_page_pipeline|floorplan_rooms|floorplan_topology_repair|vector_pdf_scale)|import (floorplan_onnx|floorplan_page_pipeline|floorplan_rooms|floorplan_topology_repair|vector_pdf_scale)" vector_pdf_exterior.py vector_pdf_openings.py vector_pdf_energy_geometry.py`  
Expected: tests pass; `rg` returns exit code `1` with no matches.

- [ ] **Step 5: Commit Task 5**

```powershell
git add -- vector_pdf_fusion_pipeline.py tests/test_vector_pdf_fusion_pipeline.py
git commit -m "feat: publish exterior opening diagnostics"
```

---

### Task 6: Server-Side Exterior Confirmation and Recognition Persistence

**Files:**
- Modify: `web_server_server.py`
- Create: `tests/test_vector_pdf_exterior_route.py`
- Modify: `tests/test_vector_pdf_fusion_route.py`

**Interfaces:**
- Extend `POST /energy/vector_pdf_fusion` response with exterior images, summary, and SHA-256 of `pdf_exterior_topology.json`.
- Add authenticated `POST /energy/vector_pdf_exterior_confirm` accepting `report_number`, `topology_sha256`, `scale_m_per_px`, `confirmed=true`.
- Confirmation writes `recognition.json` only after reloading and hashing server artifacts, validating a closed candidate, and revalidating every small repair against `0.6 m`.

- [ ] **Step 1: Write failing route security/readiness tests**

```python
def test_confirm_reloads_hashed_artifact_and_writes_ready_recognition(self):
    response = client.post("/energy/vector_pdf_exterior_confirm", json={
        "report_number": "EXT-1", "topology_sha256": topology_hash,
        "scale_m_per_px": 0.02, "confirmed": True,
    })
    assert response.status_code == 200
    saved = json.loads((report_dir / "recognition.json").read_text("utf-8"))
    assert saved["exterior_topology"]["load_geometry_ready"] is True
    assert saved["opening_widths"]["door_total_width_m"] == 0.6

def test_confirm_rejects_hash_mismatch_without_overwriting_recognition():
    before = recognition_path.read_bytes()
    response = client.post("/energy/vector_pdf_exterior_confirm", json={
        "report_number": "EXT-1", "topology_sha256": "0" * 64,
        "scale_m_per_px": 0.02, "confirmed": True,
    })
    assert response.status_code == 409
    assert recognition_path.read_bytes() == before
```

Also test missing scale, non-closed topology, metric small-gap violation, mismatched report path, unauthenticated requests, and absence of calls to `_floorplan_segmenter`.

- [ ] **Step 2: Run routes and verify RED**

Run: `python -m unittest tests.test_vector_pdf_exterior_route -v`  
Expected: confirm route returns 404.

- [ ] **Step 3: Implement hash response, confirm route, and atomic persistence**

```python
@app.post('/energy/vector_pdf_exterior_confirm')
@login_required
def vector_pdf_exterior_confirm():
    payload = request.get_json(silent=False)
    # Resolve only the report's vector_pdf_fusion directory.
    # Hash/reload topology and openings, validate scale and repair lengths.
    # Apply scale, construct recognition payload, atomically replace recognition.json.
```

Persist `geometry.walls` from real exterior walls only; `geometry.windows` and `geometry.doors` from opening objects; persist `exterior_topology`, `openings`, `opening_widths`, confirmed scale calibration, image size, source page/crop metadata, and `room_topology.load_geometry_ready: false`.

- [ ] **Step 4: Run route regression tests**

Run: `python -m unittest tests.test_vector_pdf_exterior_route tests.test_vector_pdf_fusion_route tests.test_energy_template -v`  
Expected: all tests pass, including old routes.

- [ ] **Step 5: Commit Task 6**

```powershell
git add -- web_server_server.py tests/test_vector_pdf_exterior_route.py tests/test_vector_pdf_fusion_route.py
git commit -m "feat: confirm PDF exterior energy geometry"
```

---

### Task 7: Exterior Review UI and Energy Calculation Inputs

**Files:**
- Modify: `templates/energy.html`
- Modify: `web_server_server.py`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- UI stores unconfirmed results in `pendingExteriorFusion`, never `aiResultData`.
- Confirmation sends server topology hash and scale, then sets `aiResultData` only from the confirm response.
- Energy request adds `door_height_m`, `window_height_m`, `door_repeat_count`, and `window_repeat_count`.
- `/energy/ai_simulate` prefers confirmed `exterior_topology` over room polygons, then preserves existing room/manual fallback order for old payloads.

- [ ] **Step 1: Write failing UI and energy-route tests**

```python
def test_exterior_debug_requires_confirmation_before_energy_state():
    html = Path("templates/energy.html").read_text("utf-8")
    assert "pendingExteriorFusion" in html
    assert "/energy/vector_pdf_exterior_confirm" in html
    assert "确认外轮廓并用于能耗计算" in html

def test_energy_route_uses_confirmed_exterior_and_separate_opening_heights():
    response = client.post("/energy/ai_simulate", json={
        "report_number": "EXT-ENERGY", "height": 3.0, "floors": 2,
        "door_height_m": 2.1, "window_height_m": 1.5,
        "door_repeat_count": 1, "window_repeat_count": 2,
    })
    geometry = calculate.call_args.args[0]["geometry"]
    assert geometry["floor_area_m2"] == 100.0
    assert geometry["door_area_m2"] == 4.2
    assert geometry["window_area_m2"] == 24.0
    assert geometry["wall_area_m2"] == 211.8
```

Add tests that unconfirmed exterior cannot calculate, old room topology still calculates, door repeat defaults to `1`, window repeat defaults to `floors`, and opening area greater than gross wall returns 400.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_energy_template -v`  
Expected: missing confirmation controls/fields and exterior calculation assertions fail.

- [ ] **Step 3: Implement review, scale, and height controls**

Add a review panel showing candidate area/perimeter, door/window counts and widths, small repairs, unresolved gaps, scale input, and the overall confirmation button. Use the new exterior overlay image. Keep the calculate button disabled unless either confirmed exterior topology or the existing confirmed room topology is current.

Add numeric inputs with validation:

```html
<input id="param-door-height" type="number" min="0.1" step="0.1" value="2.1">
<input id="param-window-height" type="number" min="0.1" step="0.1" value="1.5">
<input id="param-door-repeat-count" type="number" min="1" step="1" value="1">
<input id="param-window-repeat-count" type="number" min="1" step="1">
```

- [ ] **Step 4: Implement exterior-first server geometry selection**

```python
if exterior.get('confirmed') and exterior.get('load_geometry_ready'):
    exterior_geometry = build_exterior_energy_geometry(
        exterior,
        recognition.get('openings') or [],
        storey_height_m=height,
        floors=floors,
        door_height_m=parse_finite_number('door_height_m', 2.1, minimum=0.1),
        window_height_m=parse_finite_number('window_height_m', 1.5, minimum=0.1),
        door_repeat_count=parse_finite_number(
            'door_repeat_count', 1, minimum=1, maximum=floors, integer=True,
        ),
        window_repeat_count=parse_finite_number(
            'window_repeat_count', floors, minimum=1, maximum=floors, integer=True,
        ),
    )
    floor_area_m2 = exterior_geometry['floor_area_m2']
    floor_area_source = 'confirmed_exterior_footprint'
elif existing_room_conditions:
    # Preserve current room polygon behavior exactly.
```

Pass net opaque wall, separate window area, separate door area, footprint roof area, and per-floor area into the unchanged `energy_calc.calculate(calc_params)` contract.

- [ ] **Step 5: Run UI and server regressions**

Run: `python -m unittest tests.test_energy_template tests.test_energy_calc tests.test_vector_pdf_exterior_route -v`  
Expected: all tests pass.

- [ ] **Step 6: Commit Task 7**

```powershell
git add -- templates/energy.html web_server_server.py tests/test_energy_template.py
git commit -m "feat: use confirmed exterior in energy calculation"
```

---

### Task 8: Real PDF Verification, Full Regression, and Handoff

**Files:**
- Create: `tools/smoke_vector_pdf_exterior.py`
- Create: `tests/test_vector_pdf_exterior_smoke_cli.py`
- Create: `docs/handoff/2026-08-25-PDF外轮廓门窗能耗实施结果.md`

**Interfaces:**
- CLI reuses an existing fusion artifact directory or runs one selected PDF page, prints compact JSON metrics, and never confirms or writes `recognition.json`.
- Metrics: page, candidate counts, exterior status, area/perimeter pixels, door/window counts and widths, bridge counts, unresolved gaps, timings, and artifact hashes.

- [ ] **Step 1: Write failing smoke CLI test with injected analyzer**

```python
def test_smoke_prints_exterior_metrics_without_confirmation():
    status = main(["--pdf", str(pdf), "--page", "13", "--output", str(output)], analyzer=fake_analyzer)
    assert status == 0
    summary = json.loads(stdout.getvalue())
    assert summary["load_geometry_ready"] is False
    assert summary["exterior_status"] == "review_required"
```

Also assert invalid PDF/page returns nonzero and no `recognition.json` is created.

- [ ] **Step 2: Run test and verify RED**

Run: `python -m unittest tests.test_vector_pdf_exterior_smoke_cli -v`  
Expected: missing smoke module.

- [ ] **Step 3: Implement opt-in CLI**

Use `argparse`, explicit paths, the existing configured external model, and JSON output. Do not delete or overwrite user PDF files. If `--reuse-artifacts` is supplied, validate probability shape/hash before using it.

- [ ] **Step 4: Run complete verification**

```powershell
$env:ONNX_MODEL_PATH='G:\bim-web\models\M2_pub_plus_user.onnx'
$env:POPPLER_PATH='C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
python -m unittest discover -s tests -v
python -m py_compile vector_pdf_exterior.py vector_pdf_openings.py vector_pdf_energy_geometry.py vector_pdf_fusion_pipeline.py web_server_server.py tools/smoke_vector_pdf_exterior.py
```

Expected: all tests pass and compilation exits `0`.

- [ ] **Step 5: Run isolation and real-page smoke**

```powershell
rg -n "from (floorplan_onnx|floorplan_page_pipeline|floorplan_rooms|floorplan_topology_repair|vector_pdf_scale)|import (floorplan_onnx|floorplan_page_pipeline|floorplan_rooms|floorplan_topology_repair|vector_pdf_scale)" vector_pdf_exterior.py vector_pdf_openings.py vector_pdf_energy_geometry.py
& 'G:\bim-annotation-training\.worktrees\vector-annotation-v1\.venv-train\Scripts\python.exe' tools\smoke_vector_pdf_exterior.py --pdf 'G:\bim-web\uploads\energy\BIM-20260825-6241\building_plan_prepared_cbb8a8ba65ec45479fec05f87fa0119a.pdf' --page 13 --output 'G:\bim-web\uploads\energy\BIM-20260825-6241\vector_pdf_exterior_smoke'
```

Expected: `rg` exits `1` with no matches. The smoke command uses the current validated PDF and a dedicated owned output subdirectory; record the resolved paths in the handoff without copying the PDF or model.

- [ ] **Step 6: Record handoff evidence**

Write branch/commits, full test count, real-page metrics, artifact names/hashes, observed runtime, remaining ambiguous openings/unresolved gaps, confirmation safety behavior, and the next 5–10 PDF validation requirement. Do not include credentials, private server addresses, model weights, or user PDF content.

- [ ] **Step 7: Commit Task 8**

```powershell
git add -- tools/smoke_vector_pdf_exterior.py tests/test_vector_pdf_exterior_smoke_cli.py 'docs/handoff/2026-08-25-PDF外轮廓门窗能耗实施结果.md'
git commit -m "test: verify exterior opening energy workflow"
```

---

## Final Verification Checklist

- [ ] Door/window objects are anchored to exterior wall gaps and preserve host wall IDs.
- [ ] Door and window widths are stored separately; heights come only from energy-page inputs.
- [ ] `opening_bridge` and `small_gap_repair` never appear in real wall geometry.
- [ ] Small-gap metric and pixel thresholds pass exact-boundary tests.
- [ ] Concave footprints remain concave and never become bounding rectangles.
- [ ] Automatic analysis and debug routes always return `load_geometry_ready: false`.
- [ ] Confirmation reloads and hashes server artifacts and revalidates repair lengths with confirmed scale.
- [ ] Confirmed exterior geometry takes precedence only for its own new path; old room/manual fallbacks remain compatible.
- [ ] Energy wall area is net opaque wall area after separate door/window deductions.
- [ ] New modules contain none of the five prohibited legacy imports.
- [ ] Full automated tests pass.
- [ ] The current real vector PDF produces a reviewable exterior candidate with recorded metrics.
