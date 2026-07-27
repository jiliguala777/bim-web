# Vector PDF Multi-Gap Room Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Aggressively connect multiple short wall gaps in vector PDFs so large rooms can close, while reporting partial closure instead of presenting incomplete room area as complete building area.

**Architecture:** Extend the isolated topology-repair module with trusted closed-room polygons and dominant horizontal/vertical span reconstruction. Prefer four long vector-supported boundary lines around a leaking space, use bounded short-gap search only as fallback, classify completeness in the repair result, and make the browser distinguish complete from partial area. Room names and room-area text never participate in closure selection.

**Tech Stack:** Python 3, NumPy, OpenCV, Flask, unittest, vanilla JavaScript, RapidOCR.

## Global Constraints

- Apply aggressive multi-gap repair only when vector-PDF ROI and structural support are supplied.
- Added geometry is limited to horizontal and vertical lines inside the building ROI.
- Door-sized openings may be closed.
- Default maximum gap is `clamp(max(8 × wall thickness, ROI short side × 1.5%), 16, 64)` pixels.
- Default alignment tolerance is `min(max(2 × wall thickness, 4), 12)` pixels.
- Search at most 24 candidates per round, 16 beam states, 6 rounds, and 16 accepted lines.
- Existing trustworthy rooms may not disappear or merge.
- `partial` geometry must keep `load_geometry_ready == false` and must be labelled as closed-room area, not building area.
- The page-13 regression target is a central room within ±20% of 912.80 m² and at least 1200 m² total closed area.

---

### Task 1: Multi-gap candidate search and combined closure

**Files:**
- Modify: `floorplan_topology_repair.py`
- Test: `tests/test_floorplan_topology_repair.py`

**Interfaces:**
- Consumes: `repair_vector_floorplan_topology(class_mask, *, building_roi, structural_support_mask, max_exterior_gap_px, max_internal_component_area_px, min_room_area_px) -> dict`.
- Produces: the same public function, with `exterior_repair.accepted_lines_px: list[list[int]]`, `candidate_count`, `round_count`, `limits`, and `status` equal to `complete`, `partial`, or `failed`. Preserve `accepted_line_px` as the first accepted line for compatibility.

- [ ] **Step 1: Write failing tests for combined closure**

Add a synthetic rectangle with two separated gaps where neither added line alone closes the free region, but both lines together do:

```python
def test_repairs_two_gaps_as_one_combined_solution(self):
    mask = np.zeros((120, 160), dtype=np.uint8)
    support = np.zeros_like(mask)
    cv2.line(mask, (30, 20), (130, 20), 1, 3)
    cv2.line(mask, (30, 20), (30, 52), 1, 3)
    cv2.line(mask, (30, 64), (30, 100), 1, 3)
    cv2.line(mask, (130, 20), (130, 58), 1, 3)
    cv2.line(mask, (130, 70), (130, 100), 1, 3)
    cv2.line(mask, (30, 100), (130, 100), 1, 3)

    result = self._repair(mask, support, max_exterior_gap_px=16)

    self.assertEqual(result["exterior_repair"]["status"], "complete")
    self.assertEqual(len(result["exterior_repair"]["accepted_lines_px"]), 2)
    topology = extract_room_topology(result["calculation_mask"], max_gap_px=0, min_room_area_px=500)
    self.assertEqual(topology["room_count"], 1)
```

Add tests proving a door-sized unsupported gap is eligible, an aligned text fragment without opposing wall directions is rejected, and the legacy one-line case still works.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_floorplan_topology_repair -v`

Expected: the combined-closure test fails because the current function only accepts one individually successful candidate and has no `accepted_lines_px`.

- [ ] **Step 3: Add adaptive endpoint candidate helpers**

Implement focused private helpers with these signatures:

```python
def _estimate_wall_thickness(barrier: np.ndarray, roi: list[int]) -> int:
    x0, y0, x1, y1 = roi
    crop = (barrier[y0:y1, x0:x1] > 0).astype(np.uint8)
    distances = cv2.distanceTransform(crop, cv2.DIST_L2, 3)
    samples = distances[(distances > 0) & (distances <= 8)]
    if samples.size == 0:
        return 1
    return int(np.clip(round(float(np.median(samples)) * 2), 1, 8))

def _multi_gap_limits(barrier: np.ndarray, roi: list[int], configured_gap: int) -> dict:
    wall = _estimate_wall_thickness(barrier, roi)
    short_side = min(roi[2] - roi[0], roi[3] - roi[1])
    return {
        "wall_thickness_px": wall,
        "max_gap_px": int(np.clip(max(configured_gap, 8 * wall, round(short_side * 0.015)), 16, 64)),
        "alignment_tolerance_px": min(max(2 * wall, 4), 12),
        "max_candidates_per_round": 24,
        "beam_width": 16,
        "max_rounds": 6,
        "max_lines": 16,
    }

def _endpoint_gap_candidates(barrier: np.ndarray, support: np.ndarray, roi: list[int], limits: dict) -> list[dict]:
    labels, main_label = _main_component_label(barrier)
    candidates = []
    for candidate in _candidate_segments(barrier, limits["max_gap_px"]):
        line = candidate["line_px"]
        if not (
            roi[0] <= line[0] <= roi[2] and roi[0] <= line[2] <= roi[2]
            and roi[1] <= line[1] <= roi[3] and roi[1] <= line[3] <= roi[3]
        ):
            continue
        if not (
            _endpoint_touches_label(labels, tuple(line[:2]), main_label)
            and _endpoint_touches_label(labels, tuple(line[2:]), main_label)
        ):
            continue
        length = math.hypot(line[2] - line[0], line[3] - line[1])
        vector_ratio = _support_ratio(support, line)
        enriched = dict(candidate)
        enriched.update({
            "length_px": round(length, 3),
            "support_ratio": round(vector_ratio, 4),
            "alignment_error_px": 0.0,
            "score": round(1.0 - length / max(1.0, limits["max_gap_px"]) + 0.5 * vector_ratio, 4),
        })
        candidates.append(enriched)
    candidates.sort(key=lambda item: (-item["score"], item["length_px"], item["line_px"]))
    return candidates[:limits["max_candidates_per_round"]]
```

Each candidate record contains `line_px`, `orientation`, `length_px`, `support_ratio`, `alignment_error_px`, and `score`. Candidate generation must require structural endpoints on both sides, but must not require vector coverage.

- [ ] **Step 4: Implement bounded combined search**

Add:

```python
def _search_multi_gap_solution(mask, candidates, roi, min_room_area_px, baseline, limits):
    """Return the best mask plus accepted candidates without requiring per-line closure."""
```

Build beam states incrementally from the unchanged mask. Score only states that preserve all baseline rooms; reward newly closed area and penalize line count, total line length, and narrow fragments. Keep intermediate states even when they have not yet created a room, so two to five gaps can jointly close one region. Stop expanding after `max_lines`, and return an empty solution if no state adds a plausible closed region.

- [ ] **Step 5: Integrate the search without changing the public call signature**

Replace the unique-single-line decision with multi-gap search for vector context. Return diagnostics shaped as:

```python
"exterior_repair": {
    "status": "complete" if accepted else "partial",
    "candidate_count": len(candidates),
    "accepted_line_px": accepted[0]["line_px"] if accepted else None,
    "accepted_lines_px": [item["line_px"] for item in accepted],
    "round_count": rounds,
    "limits": limits,
    "reason": "multi_gap_solution" if accepted else "no_valid_multi_gap_solution",
}
```

Retain the existing internal-fragment cleanup after exterior repair and ensure `calculation_mask` contains every accepted line.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_floorplan_topology_repair -v`

Expected: all topology-repair tests pass, including legacy single-line and long lower-entrance cases.

- [ ] **Step 7: Commit Task 1**

```powershell
git add floorplan_topology_repair.py tests/test_floorplan_topology_repair.py
git commit -m "feat: close multiple vector floorplan gaps"
```

### Task 2: Completeness classification and calculation readiness

**Files:**
- Modify: `floorplan_topology_repair.py`
- Modify: `web_server_server.py`
- Test: `tests/test_floorplan_topology_repair.py`
- Test: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: Task 1 repair diagnostics and `room_topology`.
- Produces: `topology_repair.closure_status`, `topology_repair.significant_leak_area_px`, and readiness enforcement based on `closure_status == "complete"`.

- [ ] **Step 1: Write failing completeness tests**

Add a test with one existing small closed room plus a large ROI region still connected to the image border:

```python
def test_reports_partial_when_large_roi_region_still_leaks_outside(self):
    mask = np.zeros((140, 180), dtype=np.uint8)
    cv2.rectangle(mask, (20, 20), (70, 70), 1, 3)
    cv2.line(mask, (95, 20), (160, 20), 1, 3)
    cv2.line(mask, (95, 20), (95, 120), 1, 3)
    result = self._repair(mask, np.zeros_like(mask), min_room_area_px=300)
    self.assertEqual(result["closure_status"], "partial")
    self.assertGreater(result["significant_leak_area_px"], 0)
```

Add a Flask helper test:

```python
topology = {"room_count": 2, "load_geometry_ready": True}
repair = {"closure_status": "partial"}
self.server._enforce_topology_repair_readiness(topology, repair)
self.assertFalse(topology["load_geometry_ready"])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_floorplan_topology_repair tests.test_energy_template -v`

Expected: failures because `closure_status` and leak diagnostics do not exist and readiness only checks `manual_exterior_wall_required`.

- [ ] **Step 3: Implement significant-leak measurement**

Add a helper that flood-fills free space from the image border, intersects it with the provisional building envelope inside the ROI, and returns leak pixels. Mark it significant only when it exceeds both `0.02 * roi_area` and `4 * min_room_area_px`.

```python
def _classify_closure(mask, roi, min_room_area_px):
    leak_area = _exterior_leak_inside_envelope(mask, roi)
    threshold = max((roi[2] - roi[0]) * (roi[3] - roi[1]) * 0.02, 4 * min_room_area_px)
    topology = _room_topology(mask, min_room_area_px)
    status = "failed" if topology["room_count"] == 0 else ("partial" if leak_area > threshold else "complete")
    return status, float(leak_area), float(threshold)
```

Store status and threshold in repair diagnostics after all line additions and internal cleanup.

- [ ] **Step 4: Enforce readiness in Flask**

Update `_enforce_topology_repair_readiness` so `partial`, `failed`, or legacy `manual_exterior_wall_required` always sets `load_geometry_ready` false. Leave non-vector results without a closure status on their existing path.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_floorplan_topology_repair tests.test_energy_template -v`

Expected: all focused tests pass.

- [ ] **Step 6: Commit Task 2**

```powershell
git add floorplan_topology_repair.py web_server_server.py tests/test_floorplan_topology_repair.py tests/test_energy_template.py
git commit -m "fix: reject partially closed floor areas"
```

### Task 3: Browser status and multi-line overlay diagnostics

**Files:**
- Modify: `floorplan_onnx.py`
- Modify: `templates/energy.html`
- Test: `tests/test_floorplan_rooms.py`
- Test: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: `accepted_lines_px`, `closure_status`, `room_count`, and `total_area_m2`.
- Produces: red overlay lines for every automatic connection and accurate Chinese status text.

- [ ] **Step 1: Write failing overlay and template tests**

Assert two accepted repair lines are both painted into the overlay calculation mask path. Add template assertions for `accepted_lines_px`, `closure_status`, `自动连接了`, `已闭合房间面积`, and verify the old “唯一一条直线” message is absent.

- [ ] **Step 2: Run tests and verify RED**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_floorplan_rooms tests.test_energy_template -v`

Expected: template assertions fail because the UI only checks `manual_exterior_wall_required` and does not report line count or partial status.

- [ ] **Step 3: Render every added line and update browser copy**

Ensure the overlay is generated from the final calculation mask and explicitly draw every `accepted_lines_px` line in red when diagnostic overlay composition needs to distinguish additions. Update the result branch to:

```javascript
const closureStatus = data.topology_repair?.closure_status || 'failed';
const repairedLines = data.topology_repair?.exterior_repair?.accepted_lines_px || [];
if (closureStatus === 'partial') {
    closureSummary.innerText = `自动连接了 ${repairedLines.length} 个断口；当前仍为部分闭合，${roomCount} 个房间合计仅为已闭合房间面积。`;
    closureSummary.style.color = 'var(--warning)';
    document.getElementById('param-floor-area').value = '';
} else if (closureStatus === 'complete') {
    closureSummary.innerText = `自动连接了 ${repairedLines.length} 个断口；已提取 ${roomCount} 个闭合房间。`;
    closureSummary.style.color = 'var(--success)';
}
```

Also prevent `updatePhysicalLengths()` from filling the floor-area field unless `room_topology.load_geometry_ready` is true.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_floorplan_rooms tests.test_energy_template -v`

Expected: all focused tests pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add floorplan_onnx.py templates/energy.html tests/test_floorplan_rooms.py tests/test_energy_template.py
git commit -m "feat: show multi-gap closure status"
```

### Task 4: Explicit non-use of room-area text

**Files:**
- Verify: `floorplan_ocr.py`
- Test: `tests/test_floorplan_ocr.py`

**Interfaces:**
- Consumes: RapidOCR output used by scale calibration.
- Produces: dimension spans only; room-area text is ignored.

- [ ] **Step 1: Verify area text is not accepted as a dimension**

Use fake OCR output containing `40600`, `912.80㎡`, and `46.40平方米`. Assert dimension spans contain only `40600` and no room-area evidence is sent to topology repair.

- [ ] **Step 2: Run tests and verify RED**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_floorplan_ocr -v`

Expected: PASS; the existing full-match dimension filter rejects room-area text.

- [ ] **Step 3: Run focused tests**

Run: `./.venv/Scripts/python.exe -m unittest tests.test_floorplan_ocr tests.test_energy_template -v`

Expected: all focused tests pass, and area OCR never appears in scale or topology evidence.

### Task 5: Full regression and page-13 acceptance

**Files:**
- Verify: `floorplan_topology_repair.py`
- Verify: `tests/test_floorplan_topology_repair.py`
- Runtime artifacts only: `uploads/energy/BIM-20260722-MULTIGAP-VERIFY/recognition.json`, `ai_mask.png`, `ai_overlay.jpg`

**Interfaces:**
- Consumes: completed Tasks 1–4 and the existing page-13 vector PDF upload flow.
- Produces: verified recognition report and visual evidence; no fixture containing the user PDF is committed.

- [ ] **Step 1: Run all automated tests**

Run: `./.venv/Scripts/python.exe -m unittest discover -s tests -v`

Expected: all tests pass with no errors.

- [ ] **Step 2: Run static verification**

Run:

```powershell
./.venv/Scripts/python.exe -m py_compile floorplan_topology_repair.py floorplan_ocr.py floorplan_onnx.py web_server_server.py
./.venv/Scripts/python.exe -m pip check
```

Expected: both commands exit 0.

- [ ] **Step 3: Exercise the real page-13 route**

Submit the existing prepared vector PDF page 13 through `/energy/ai_recognize` with a fresh report number. Inspect `recognition.json` and assert:

```python
assert report["scale_calibration"]["status"] == "confirmed"
assert report["topology_repair"]["closure_status"] == "complete"
assert len(report["topology_repair"]["exterior_repair"]["accepted_lines_px"]) >= 2
assert report["room_topology"]["room_count"] >= 18
assert report["room_topology"]["total_area_m2"] >= 1200
assert any(730.24 <= room["area_m2"] <= 1095.36 for room in report["room_topology"]["rooms"])
```

- [ ] **Step 4: Visually verify the overlay**

Open `ai_overlay.jpg` and confirm every automatic connection is red, the central multipurpose hall is bounded, and no long line crosses unrelated drawing annotations outside the building ROI.

- [ ] **Step 5: Restart the website and verify availability**

Restart the Flask service through the existing local startup path and request `http://127.0.0.1:5000/login`.

Expected: HTTP 200.

- [ ] **Step 6: Commit any page-specific correction only after tests pass**

```powershell
git add floorplan_topology_repair.py tests/test_floorplan_topology_repair.py
git commit -m "fix: close page 13 multipurpose hall"
```

Skip this commit when no correction was required.
