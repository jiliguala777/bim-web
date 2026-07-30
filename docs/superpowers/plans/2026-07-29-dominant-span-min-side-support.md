# Dominant Span Per-Side Support Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject weakly supported `dominant_span_rectangle` repairs and skip all exterior repair when the original model mask already contains a credible major closed area.

**Architecture:** Keep both decisions inside `floorplan_topology_repair.py`. Candidate generation applies fixed per-side support gates; the repair entry point computes an `already_closed` decision from the original topology before starting exterior searches, while preserving internal-fragment handling and existing diagnostic compatibility.

**Tech Stack:** Python 3.12, NumPy, OpenCV, `unittest`, project `.venv`.

## Global Constraints

- Apply the 15% minimum independently to top, bottom, left, and right model support.
- Keep the existing 60% independent minimum for PDF vector support.
- Treat the original mask as `already_closed` only when total closed area is at least 20% of ROI area and the largest room is at least 8% of ROI area.
- An `already_closed` result must skip both `dominant_span_rectangle` and `multi_gap_solution`.
- Preserve the existing average `vector_support_ratio` and `model_support_ratio` diagnostic fields.
- Add named per-side diagnostics without changing callers.
- Do not change short-gap, internal U-shape, model inference, training data, or energy calculation logic.
- Do not overwrite or deploy any ONNX model.

---

### Task 1: Add the failing low-support-side regression

**Files:**
- Modify: `tests/test_floorplan_topology_repair.py`
- Test: `tests/test_floorplan_topology_repair.py`

**Interfaces:**
- Consumes: `_dominant_span_rectangles(mask: np.ndarray, support: np.ndarray, roi: list[int], min_room_area_px: float) -> list[dict]`
- Produces: A regression proving a rectangle with strong vector support but weak model support on one or more sides is absent from the candidate list.

- [ ] **Step 1: Import the focused candidate generator**

Replace the existing topology-repair import with:

```python
from floorplan_topology_repair import (
    _dominant_span_rectangles,
    repair_vector_floorplan_topology,
)
```

- [ ] **Step 2: Write the failing regression test**

Add this test immediately after `test_uses_four_long_span_lines_for_fragmented_room_perimeter`:

```python
def test_rejects_dominant_rectangle_when_any_side_has_weak_model_support(self):
    mask = np.zeros((140, 180), dtype=np.uint8)
    support = np.zeros_like(mask)
    cv2.rectangle(support, (30, 20), (150, 120), 255, 3)

    cv2.line(mask, (30, 20), (150, 20), 1, 3)
    cv2.line(mask, (150, 20), (150, 120), 1, 3)
    cv2.line(mask, (140, 120), (150, 120), 1, 3)
    cv2.line(mask, (30, 20), (30, 22), 1, 3)

    candidates = _dominant_span_rectangles(
        mask,
        support,
        [10, 10, 170, 130],
        min_room_area_px=1000,
    )

    self.assertFalse(any(
        candidate["bbox_px"] == [30, 20, 120, 100]
        for candidate in candidates
    ))
```

- [ ] **Step 3: Run the regression and confirm the current rule fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_rejects_dominant_rectangle_when_any_side_has_weak_model_support `
  -v
```

Expected: `FAIL` because the current 2% minimum still returns the `[30, 20, 120, 100]` candidate.

- [ ] **Step 4: Commit the red test**

```powershell
git add -- tests/test_floorplan_topology_repair.py
git commit -m "test: reject weakly supported rectangle sides"
```

---

### Task 2: Enforce the 15% per-side model gate and expose diagnostics

**Files:**
- Modify: `floorplan_topology_repair.py:547-657`
- Modify: `tests/test_floorplan_topology_repair.py`
- Test: `tests/test_floorplan_topology_repair.py`

**Interfaces:**
- Consumes: Existing side profiles created by `_dominant_span_rectangles`.
- Produces: Candidate fields `vector_support_by_side: dict[str, float]` and `model_support_by_side: dict[str, float]`; candidates are emitted only when every vector side is at least `0.60` and every model side is at least `0.15`.

- [ ] **Step 1: Define explicit module-level thresholds**

Add near the imports in `floorplan_topology_repair.py`:

```python
DOMINANT_SPAN_MIN_VECTOR_SIDE_SUPPORT = 0.60
DOMINANT_SPAN_MIN_MODEL_SIDE_SUPPORT = 0.15
```

- [ ] **Step 2: Name and gate all four side ratios**

Inside `_dominant_span_rectangles`, replace the anonymous vector/model minimum checks with:

```python
side_names = ("top", "bottom", "left", "right")
vector_support_by_side = dict(zip(side_names, vector_ratios))
if min(vector_support_by_side.values()) < DOMINANT_SPAN_MIN_VECTOR_SIDE_SUPPORT:
    continue

model_support_by_side = dict(zip(side_names, model_ratios))
if min(model_support_by_side.values()) < DOMINANT_SPAN_MIN_MODEL_SIDE_SUPPORT:
    continue
```

The existing calculations of `vector_ratios` and `model_ratios` remain unchanged.

- [ ] **Step 3: Preserve averages and add rounded per-side diagnostics**

Extend the candidate dictionary with:

```python
"vector_support_by_side": {
    name: round(float(value), 4)
    for name, value in vector_support_by_side.items()
},
"model_support_by_side": {
    name: round(float(value), 4)
    for name, value in model_support_by_side.items()
},
```

Keep these existing fields unchanged:

```python
"vector_support_ratio": round(float(np.mean(vector_ratios)), 4),
"model_support_ratio": round(float(np.mean(model_ratios)), 4),
```

- [ ] **Step 4: Strengthen the existing positive test**

In `test_uses_four_long_span_lines_for_fragmented_room_perimeter`, add:

```python
candidate = result["exterior_repair"]["accepted_candidates"][0]
self.assertGreaterEqual(min(candidate["vector_support_by_side"].values()), 0.60)
self.assertGreaterEqual(min(candidate["model_support_by_side"].values()), 0.15)
self.assertIn("vector_support_ratio", candidate)
self.assertIn("model_support_ratio", candidate)
```

- [ ] **Step 5: Run focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_rejects_dominant_rectangle_when_any_side_has_weak_model_support `
  tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_uses_four_long_span_lines_for_fragmented_room_perimeter `
  -v
```

Expected: both tests pass.

- [ ] **Step 6: Run the full topology-repair test module**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_topology_repair -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the implementation**

```powershell
git add -- floorplan_topology_repair.py tests/test_floorplan_topology_repair.py
git commit -m "fix: require model support on every rectangle side"
```

---

### Task 3: Short-circuit exterior repair for a credible original closure

**Files:**
- Modify: `floorplan_topology_repair.py`
- Modify: `tests/test_floorplan_topology_repair.py`
- Test: `tests/test_floorplan_topology_repair.py`

**Interfaces:**
- Consumes: `_room_topology(mask, min_room_area_px) -> dict` and normalized `roi: list[int]`.
- Produces: `_initial_topology_is_already_closed(topology: dict, roi: list[int]) -> dict`; the returned dictionary contains `is_already_closed`, `roi_area_px2`, `total_closed_area_px2`, `largest_room_area_px2`, `minimum_total_closed_area_px2`, and `minimum_largest_room_area_px2`.

- [ ] **Step 1: Write the failing already-closed regression**

Add after `test_uses_four_long_span_lines_for_fragmented_room_perimeter`:

```python
def test_skips_exterior_repair_when_major_area_is_already_closed(self):
    mask = np.zeros((140, 180), dtype=np.uint8)
    support = np.zeros_like(mask)
    cv2.rectangle(mask, (30, 20), (150, 120), 1, 3)
    cv2.rectangle(support, (15, 15), (165, 125), 255, 1)

    result = repair_vector_floorplan_topology(
        mask,
        building_roi=[10, 10, 170, 130],
        structural_support_mask=support,
        max_exterior_gap_px=16,
        max_internal_component_area_px=500,
        min_room_area_px=500,
    )

    repair = result["exterior_repair"]
    self.assertEqual(repair["status"], "not_needed")
    self.assertEqual(repair["reason"], "already_closed")
    self.assertEqual(repair["accepted_lines_px"], [])
    self.assertEqual(result["closure_status"], "complete")
    self.assertFalse(result["manual_exterior_wall_required"])
    self.assertTrue(repair["initial_closure"]["is_already_closed"])
```

- [ ] **Step 2: Run the new regression and confirm it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_skips_exterior_repair_when_major_area_is_already_closed `
  -v
```

Expected: `FAIL` because the current implementation reports a repair outcome instead of `reason == "already_closed"`.

- [ ] **Step 3: Define the closure thresholds and helper**

Add near the existing dominant-span constants:

```python
ALREADY_CLOSED_MIN_TOTAL_ROI_FRACTION = 0.20
ALREADY_CLOSED_MIN_LARGEST_ROOM_ROI_FRACTION = 0.08
```

Add:

```python
def _initial_topology_is_already_closed(
    topology: dict,
    roi: list[int],
) -> dict:
    roi_area = float((roi[2] - roi[0]) * (roi[3] - roi[1]))
    room_areas = [
        float(room.get("area_px2") or 0.0)
        for room in topology.get("rooms", [])
    ]
    total_area = float(sum(room_areas))
    largest_area = max(room_areas, default=0.0)
    minimum_total = roi_area * ALREADY_CLOSED_MIN_TOTAL_ROI_FRACTION
    minimum_largest = (
        roi_area * ALREADY_CLOSED_MIN_LARGEST_ROOM_ROI_FRACTION
    )
    return {
        "is_already_closed": bool(
            total_area >= minimum_total
            and largest_area >= minimum_largest
        ),
        "roi_area_px2": round(roi_area, 1),
        "total_closed_area_px2": round(total_area, 1),
        "largest_room_area_px2": round(largest_area, 1),
        "minimum_total_closed_area_px2": round(minimum_total, 1),
        "minimum_largest_room_area_px2": round(minimum_largest, 1),
    }
```

- [ ] **Step 4: Bypass only the exterior search when already closed**

Keep the existing early return for missing ROI or support. Immediately after
`limits = _multi_gap_limits(...)` is computed, add:

```python
initial_closure = _initial_topology_is_already_closed(
    initial_topology,
    roi,
)
already_closed = initial_closure["is_already_closed"]
```

Initialize `exterior_candidates`, `accepted_candidates`, and `span_rectangle`, then run `_search_dominant_span_rectangle` and `_search_multi_gap_solution` only inside:

```python
if not already_closed:
    ...
```

Before the existing `span_rectangle is not None` diagnostics branch, add:

```python
if already_closed:
    diagnostics["exterior_repair"] = {
        "status": "not_needed",
        "candidate_count": 0,
        "accepted_line_px": None,
        "accepted_lines_px": [],
        "accepted_candidates": [],
        "round_count": 0,
        "limits": limits,
        "reason": "already_closed",
        "initial_closure": initial_closure,
    }
```

Change the following `if span_rectangle is not None` to `elif`.

- [ ] **Step 5: Preserve complete status at the final gate**

At final closure classification, evaluate `already_closed` first:

```python
if already_closed:
    closure_status = "complete"
elif int(final_topology.get("room_count") or 0) <= 0:
    closure_status = "failed"
elif closed_area_gain >= major_closure_threshold:
    closure_status = "complete"
else:
    closure_status = "partial"
```

The existing `closure_status != "complete"` manual-review block remains unchanged.

- [ ] **Step 6: Verify the positive and existing negative cases**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_skips_exterior_repair_when_major_area_is_already_closed `
  tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_reports_partial_when_only_existing_small_room_is_closed `
  -v
```

Expected: both tests pass.

- [ ] **Step 7: Run the complete topology-repair module**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_topology_repair -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```powershell
git add -- floorplan_topology_repair.py tests/test_floorplan_topology_repair.py
git commit -m "fix: skip exterior repair for already closed plans"
```

---

### Task 4: Verify the real cultural-palace sample and regression suite

**Files:**
- Read: `uploads/energy/BIM-20260729-2163/ai_raw_model_mask.png`
- Read: `uploads/energy/BIM-20260729-2163/pdf_structural_mask.png`
- Read: `uploads/energy/BIM-20260729-2163/pdf_building_roi.json`
- Read: `uploads/energy/BIM-20260729-2163/recognition.json`
- Verify: `floorplan_topology_repair.py`

**Interfaces:**
- Consumes: The saved pre-repair class mask, vector support mask, building ROI, and room-area threshold from the real recognition.
- Produces: Fresh diagnostics proving the false `[390, 1058, 314, 1282]` dominant rectangle is not accepted.

- [ ] **Step 1: Reconstruct the saved raw class mask and rerun repair**

Run an inline Python probe that:

1. Decodes `ai_raw_model_mask.png` by bytes so the Chinese path is safe.
2. Maps BGR `(60, 76, 231)` to wall `1`, `(219, 152, 52)` to window `2`, and `(113, 204, 46)` to door `3`.
3. Loads `pdf_structural_mask.png`.
4. Calls `repair_vector_floorplan_topology` with:

```python
building_roi=[296, 251, 2404, 3066]
max_exterior_gap_px=64
max_internal_component_area_px=1162
min_room_area_px=1549.3536
```

5. Measures elapsed repair time.
6. Prints `exterior_repair`, `closure_status`, and whether the returned wall mask contains a complete vertical line at `x=390, y=1058:2341`.

Expected:

- `exterior_repair.reason == "already_closed"`;
- no complete added wall line exists at `x=390, y=1058:2341`;
- no accepted candidate has `bbox_px == [390, 1058, 314, 1282]`.
- elapsed repair time is below 10 seconds on the current local machine.

- [ ] **Step 2: Run adjacent room and platform regression tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_floorplan_topology_repair `
  tests.test_floorplan_rooms `
  tests.test_energy_template `
  -v
```

Expected: all tests pass.

- [ ] **Step 3: Check the final diff and repository scope**

Run:

```powershell
git diff --check
git status --short
```

Expected: no whitespace errors; unrelated personal and untracked files remain untouched.
