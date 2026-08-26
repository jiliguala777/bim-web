# Exterior Door Arc Wall Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect exterior PDF door arcs before gap classification and use them to recover chained native wall segments so exterior openings can be classified and the footprint can close.

**Architecture:** Preserve the existing wall/window fusion, then add a pure door-recovery stage between exterior-wall selection and gap enumeration. The stage projects open Bezier door arcs onto nearby confirmed exterior wall bands, builds one-dimensional connectivity components from confirmed walls, native uncertain walls, and projected door openings, and only recovers components anchored by two confirmed exterior walls. The pipeline then re-enumerates gaps and runs the existing opening/topology logic on the recovered exterior set.

**Tech Stack:** Python 3.12, pdfplumber, NumPy, OpenCV, Flask, unittest

**Spec:** `docs/superpowers/specs/2026-08-26-exterior-door-arc-wall-recovery-design.md`

## Global Constraints

- Existing wall and window fusion thresholds remain unchanged.
- Only exterior door/window openings are in scope; interior openings must not be promoted.
- Only open Bezier curves may supply door-arc evidence.
- Only confirmed exterior doors contribute width to later door-area deduction.
- Pending openings and ordinary repairs never contribute door/window area.
- Door/window height remains configured on the later energy page.
- Codex runs automated tests only; the user performs final real-drawing validation in the platform.

## File Structure

- Modify `vector_pdf_native.py`: preserve real PDF path endpoints for open Bezier curves.
- Modify `tests/test_vector_pdf_native.py`: prove real endpoints survive extraction and cropping.
- Create `vector_pdf_door_recovery.py`: pure exterior door-arc projection and native wall-chain recovery.
- Create `tests/test_vector_pdf_door_recovery.py`: cover valid exterior recovery and all safety rejections.
- Modify `vector_pdf_fusion_pipeline.py`: run recovery before gap enumeration and publish its diagnostics.
- Modify `vector_pdf_openings.py`: accept only recovery-approved exterior arcs as native door evidence.
- Modify `tests/test_vector_pdf_fusion_pipeline.py`: prove stage ordering and artifact publication.
- Modify `tests/test_vector_pdf_openings.py`: prove unapproved arcs cannot become confirmed doors.
- Modify `web_server_server.py`: expose recovered-wall and exterior-door-arc counts.
- Modify `templates/energy.html`: show recovery counts and preserve the existing overlay legend.
- Modify `tests/test_vector_pdf_fusion_route.py`: verify route summary fields.
- Modify `tests/test_energy_template.py`: verify review UI fields.

---

### Task 1: Preserve Real Open-Bezier Endpoints

**Files:**
- Modify: `vector_pdf_native.py`
- Test: `tests/test_vector_pdf_native.py`

**Interfaces:**
- Produces: each `opening_curve_edges` record contains `path_start_px: list[int]`, `path_end_px: list[int]`, `path_ops: list[str]`, `has_bezier: bool`, and `is_closed: bool`.
- Produces: cropped curve records retain page coordinates in `page_path_start_px` and `page_path_end_px`, while local `path_start_px` and `path_end_px` are rebased to the crop.

- [ ] **Step 1: Write failing extraction and crop tests**

Add assertions to `NativePdfExtractionTests` that the reportlab door arc uses its actual path endpoints rather than opposite bounding-box corners:

```python
curve = page["opening_curve_edges"][0]
self.assertNotEqual(curve["path_start_px"], curve["bbox_px"][:2])
self.assertNotEqual(curve["path_end_px"], curve["bbox_px"][2:])
self.assertIn(curve["path_start_px"][0], (curve["bbox_px"][0], curve["bbox_px"][2]))
self.assertIn(curve["path_end_px"][1], (curve["bbox_px"][1], curve["bbox_px"][3]))
```

Extend the crop test:

```python
self.assertEqual(cropped_curve["page_path_start_px"], curve["path_start_px"])
self.assertEqual(cropped_curve["page_path_end_px"], curve["path_end_px"])
self.assertTrue(all(value >= 0 for value in cropped_curve["path_start_px"]))
self.assertTrue(all(value >= 0 for value in cropped_curve["path_end_px"]))
```

- [ ] **Step 2: Run the native tests and verify RED**

Run: `python -m unittest tests.test_vector_pdf_native`

Expected: FAIL because `path_start_px` and `path_end_px` are absent.

- [ ] **Step 3: Extract endpoints from the PDF path**

Add a focused helper to `vector_pdf_native.py`:

```python
def _curve_path_endpoints(item: dict) -> tuple[list[float], list[float]] | None:
    path = item.get("path") or []
    if not path or str(path[0][0]).lower() != "m":
        return None
    start = path[0][-1]
    end = path[-1][-1]
    if len(start) != 2 or len(end) != 2:
        return None
    return [float(start[0]), float(start[1])], [float(end[0]), float(end[1])]
```

In `_opening_curve_edges`, reject records without valid endpoints and publish the pixel coordinates:

```python
endpoints = _curve_path_endpoints(item)
if endpoints is None:
    continue
path_start_pt, path_end_pt = endpoints
record["path_start_px"] = _point_to_pixel(path_start_pt, page_size, render_size)
record["path_end_px"] = _point_to_pixel(path_end_pt, page_size, render_size)
```

Keep legacy `start_px` and `end_px` equal to the real path endpoints for compatibility with the opening matcher.

Update `crop_native_page_data` to preserve page endpoints and rebase local endpoints without changing `bbox_px` clipping.

- [ ] **Step 4: Run native tests and verify GREEN**

Run: `python -m unittest tests.test_vector_pdf_native`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add vector_pdf_native.py tests/test_vector_pdf_native.py
git commit -m "fix: preserve real PDF curve endpoints"
```

---

### Task 2: Recover Exterior Wall Chains Through Door Arcs

**Files:**
- Create: `vector_pdf_door_recovery.py`
- Create: `tests/test_vector_pdf_door_recovery.py`

**Interfaces:**
- Consumes: fused line candidates, selected exterior walls, open curve records from Task 1, image size, and building ROI.
- Produces: `recover_exterior_walls_from_door_arcs(candidates: list[dict], exterior_walls: list[dict], curve_edges: list[dict], image_size: tuple[int, int], building_roi: list[int] | None) -> dict`.
- Return keys: `recovered_walls`, `confirmed_door_arcs`, `pending_door_arcs`, and `recovery_components`.
- Each recovered wall retains its original `candidate_id` and adds `recovery_method`, `recovery_arc_ids`, `recovery_anchor_ids`, and `reason_codes`.
- Each confirmed arc adds `projected_start_px`, `projected_end_px`, `orientation`, `host_wall_ids`, and `reason_codes`.

- [ ] **Step 1: Write failing tests for a valid vertical exterior chain**

Create helpers for confirmed walls, uncertain native lines, and real endpoint arcs. Test a component with two confirmed vertical anchors, uncertain vertical wall fragments, and a door arc between them:

```python
result = recover_exterior_walls_from_door_arcs(
    candidates=[upper_anchor, upper_uncertain, lower_uncertain, lower_anchor],
    exterior_walls=[upper_anchor, lower_anchor],
    curve_edges=[door_arc],
    image_size=(300, 300),
    building_roi=[20, 20, 280, 280],
)
self.assertEqual(
    [wall["candidate_id"] for wall in result["recovered_walls"]],
    ["upper-uncertain", "lower-uncertain"],
)
self.assertEqual(len(result["confirmed_door_arcs"]), 1)
self.assertEqual(result["confirmed_door_arcs"][0]["orientation"], "vertical")
```

- [ ] **Step 2: Write failing safety tests**

Add independent tests proving:

```python
self.assertEqual(interior_arc_result["recovered_walls"], [])
self.assertEqual(closed_curve_result["confirmed_door_arcs"], [])
self.assertEqual(single_anchor_result["confirmed_door_arcs"], [])
self.assertEqual(single_anchor_result["pending_door_arcs"][0]["status"], "pending")
self.assertEqual(cross_axis_result["recovered_walls"], [])
self.assertEqual(oversized_arc_result["confirmed_door_arcs"], [])
```

The fixtures must separately cover an inner-wall arc, a closed curve, a component with one exterior anchor, a perpendicular uncertain wall, and an arc wider than the configured local opening maximum.

- [ ] **Step 3: Run recovery tests and verify RED**

Run: `python -m unittest tests.test_vector_pdf_door_recovery`

Expected: ERROR because `vector_pdf_door_recovery` does not exist.

- [ ] **Step 4: Implement projection and connectivity components**

Create `vector_pdf_door_recovery.py` with immutable thresholds:

```python
@dataclass(frozen=True)
class DoorRecoveryThresholds:
    boundary_band_px: int = 16
    endpoint_tolerance_px: int = 14
    chain_gap_tolerance_px: int = 12
    min_opening_span_px: int = 12
    max_opening_span_px: int = 140
```

Use this exact native-line admission predicate:

```python
def _native_uncertain_wall(candidate: dict) -> bool:
    return (
        candidate.get("decision") == "uncertain"
        and bool(candidate.get("source_native_id"))
        and not (candidate.get("native_evidence") or {}).get("page_border")
        and float((candidate.get("native_evidence") or {}).get("roi_inside_ratio") or 0.0) >= 0.95
    )
```

Implement these public/private signatures exactly:

```python
def _project_arc_to_wall_band(
    curve: dict,
    wall: dict,
    thresholds: DoorRecoveryThresholds,
) -> dict | None:
    """Return a projected opening node or None when the arc is not local to the wall."""

def _axis_components(
    nodes: list[dict],
    tolerance_px: int,
) -> list[list[dict]]:
    """Return deterministic connected components along one wall band."""

def recover_exterior_walls_from_door_arcs(
    candidates: list[dict],
    exterior_walls: list[dict],
    curve_edges: list[dict],
    image_size: tuple[int, int],
    building_roi: list[int] | None,
) -> dict:
    """Return recovered walls and confirmed/pending exterior door arcs."""
```

The function return statement must use this stable contract even when every list is empty:

```python
return {
    "recovered_walls": recovered_walls,
    "confirmed_door_arcs": confirmed_door_arcs,
    "pending_door_arcs": pending_door_arcs,
    "recovery_components": recovery_components,
}
```

Projection rules:

- choose only the nearest confirmed exterior wall band within `boundary_band_px`;
- use `path_start_px` and `path_end_px` to project the door opening interval onto that wall axis;
- require one curve endpoint to touch the wall band within `endpoint_tolerance_px`;
- require projected opening span within the configured minimum and maximum;
- inherit `inside_direction` from the matched exterior anchor;
- reject closed/non-Bezier curves before building nodes.

Connectivity rules:

- group nodes by orientation, inside direction, and fixed-coordinate wall band;
- connect overlapping nodes or nodes separated by at most `chain_gap_tolerance_px`;
- a recoverable component must contain at least two distinct confirmed exterior wall anchors and at least one projected door arc;
- only native uncertain wall nodes inside such a component are returned as recovered walls;
- an arc component with one anchor is returned as pending, never confirmed;
- deduplicate recovered walls and arcs by IDs and sort them deterministically.

- [ ] **Step 5: Run recovery tests and verify GREEN**

Run: `python -m unittest tests.test_vector_pdf_door_recovery`

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add vector_pdf_door_recovery.py tests/test_vector_pdf_door_recovery.py
git commit -m "feat: recover exterior wall chains through door arcs"
```

---

### Task 3: Insert Recovery Before Gap Enumeration

**Files:**
- Modify: `vector_pdf_fusion_pipeline.py`
- Modify: `vector_pdf_openings.py`
- Test: `tests/test_vector_pdf_fusion_pipeline.py`
- Test: `tests/test_vector_pdf_openings.py`

**Interfaces:**
- Consumes: Task 2 recovery result.
- Produces: `door_recovery` in `pdf_vector_fusion.json` and approved arcs in `pdf_opening_candidates.json`.
- Changes the internal opening evidence contract so `curve_edges` passed to `classify_exterior_openings` contains only recovery-approved exterior arcs.

- [ ] **Step 1: Write a failing opening-classifier approval test**

Add a test proving an otherwise matching curve without recovery approval remains unclassified:

```python
curve = {
    "curve_id": "unapproved-arc",
    "bbox_px": [58, 48, 102, 92],
    "path_start_px": [60, 50],
    "path_end_px": [100, 92],
    "start_px": [60, 50],
    "end_px": [100, 92],
    "has_bezier": True,
    "is_closed": False,
    "exterior_recovery_approved": False,
}
result = classify_exterior_openings(
    [gap((60, 50), (100, 50))], empty_probabilities(), (160, 100),
    native_opening_evidence={"curve_edges": [curve], "short_segments": []},
)
self.assertEqual(result["accepted_openings"], [])
```

Update the existing native-door success fixture to set `exterior_recovery_approved=True`.

- [ ] **Step 2: Write a failing pipeline ordering test**

Patch `recover_exterior_walls_from_door_arcs`, `enumerate_exterior_gaps`, and `classify_exterior_openings`. Assert that recovered wall IDs are present when gaps are enumerated and only confirmed recovery arcs are passed to classification:

```python
self.assertIn("recovered-wall", gap_mock.call_args.args[0][-1]["candidate_id"])
self.assertEqual(
    opening_mock.call_args.kwargs["native_opening_evidence"]["curve_edges"][0]["curve_id"],
    "approved-arc",
)
```

- [ ] **Step 3: Run pipeline/opening tests and verify RED**

Run: `python -m unittest tests.test_vector_pdf_openings tests.test_vector_pdf_fusion_pipeline`

Expected: FAIL because approval is ignored and the recovery stage is not called.

- [ ] **Step 4: Require recovery-approved arcs in the matcher**

In `_native_gap_evidence`, skip a curve unless:

```python
if curve.get("exterior_recovery_approved") is not True:
    continue
```

Use `path_start_px` and `path_end_px`, with compatibility fallback to `start_px` and `end_px`, when testing endpoint proximity.

- [ ] **Step 5: Integrate recovery into the pipeline**

In `analyze_vector_pdf_page`, after `select_exterior_walls` and the existing conservative rescue, run:

```python
door_recovery = recover_exterior_walls_from_door_arcs(
    candidates,
    exterior_walls,
    page_data.get("opening_curve_edges", []),
    (width, height),
    roi,
)
exterior_walls.extend(door_recovery["recovered_walls"])
gaps = enumerate_exterior_gaps(exterior_walls, (width, height), roi)
opening_result = classify_exterior_openings(
    gaps,
    model_result.probabilities,
    (width, height),
    native_opening_evidence={
        "curve_edges": door_recovery["confirmed_door_arcs"],
        "short_segments": page_data.get("opening_short_segments", []),
    },
)
```

Publish a JSON-safe `door_recovery` object in `pdf_vector_fusion.json`. Include its counts in `exterior_summary`:

```python
"recovered_wall_count": len(door_recovery["recovered_walls"]),
"confirmed_door_arc_count": len(door_recovery["confirmed_door_arcs"]),
"pending_door_arc_count": len(door_recovery["pending_door_arcs"]),
```

Failure paths publish empty recovery arrays and zero counts.

- [ ] **Step 6: Run pipeline/opening tests and verify GREEN**

Run: `python -m unittest tests.test_vector_pdf_openings tests.test_vector_pdf_fusion_pipeline`

Expected: all tests pass.

- [ ] **Step 7: Commit Task 3**

```powershell
git add vector_pdf_fusion_pipeline.py vector_pdf_openings.py tests/test_vector_pdf_fusion_pipeline.py tests/test_vector_pdf_openings.py
git commit -m "feat: apply door arc recovery before exterior gaps"
```

---

### Task 4: Publish Recovery Diagnostics in the Platform

**Files:**
- Modify: `web_server_server.py`
- Modify: `templates/energy.html`
- Test: `tests/test_vector_pdf_fusion_route.py`
- Test: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: Task 3 `exterior_summary` recovery counts.
- Produces: route response fields `recovered_wall_count`, `confirmed_door_arc_count`, and `pending_door_arc_count`.
- Produces: review-panel counters without changing energy confirmation semantics.

- [ ] **Step 1: Write failing route and template tests**

Extend the fake pipeline result in `test_vector_pdf_fusion_route.py`:

```python
"recovered_wall_count": 2,
"confirmed_door_arc_count": 1,
"pending_door_arc_count": 1,
```

Assert the JSON response preserves all three values. In `test_energy_template.py`, assert the template contains:

```python
self.assertIn("recovered_wall_count", html)
self.assertIn("confirmed_door_arc_count", html)
self.assertIn("pending_door_arc_count", html)
self.assertIn("门弧恢复墙段", html)
```

- [ ] **Step 2: Run route/template tests and verify RED**

Run: `python -m unittest tests.test_vector_pdf_fusion_route tests.test_energy_template`

Expected: FAIL because the UI does not render recovery diagnostics.

- [ ] **Step 3: Add summary defaults and review counters**

In `vector_pdf_fusion` route handling, preserve pipeline values and default missing counts to zero:

```python
exterior_summary.setdefault("recovered_wall_count", 0)
exterior_summary.setdefault("confirmed_door_arc_count", 0)
exterior_summary.setdefault("pending_door_arc_count", 0)
```

Add three read-only counters to the exterior review panel and populate them in `showPendingExteriorReview`. Keep the legend colors unchanged: recovered walls are green, confirmed exterior doors blue, and pending openings orange.

- [ ] **Step 4: Run route/template tests and verify GREEN**

Run: `python -m unittest tests.test_vector_pdf_fusion_route tests.test_energy_template`

Expected: all tests pass.

- [ ] **Step 5: Run all relevant geometry tests**

Run:

```powershell
python -m unittest tests.test_vector_pdf_native tests.test_vector_pdf_door_recovery tests.test_vector_pdf_openings tests.test_vector_pdf_exterior tests.test_vector_pdf_fusion_pipeline tests.test_vector_pdf_fusion_route tests.test_energy_template
```

Expected: all tests pass.

- [ ] **Step 6: Run the complete automated suite**

Run:

```powershell
$env:POPPLER_PATH='C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
python -m unittest discover -s tests
git diff --check
```

Expected: all tests pass and `git diff --check` prints no errors. Do not run the real PDF replay; the user performs final platform validation.

- [ ] **Step 7: Commit Task 4**

```powershell
git add web_server_server.py templates/energy.html tests/test_vector_pdf_fusion_route.py tests/test_energy_template.py
git commit -m "feat: show exterior door recovery diagnostics"
```
