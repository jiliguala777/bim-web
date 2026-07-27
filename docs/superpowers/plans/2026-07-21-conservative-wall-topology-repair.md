# Conservative Wall Topology Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For vector PDFs, produce one conservative final mask that removes only safe internal fragments and adds at most one uniquely validated horizontal or vertical exterior repair line.

**Architecture:** Rasterize dark vector geometry as structural evidence, then pass the model mask and vector context into an isolated topology-repair module. The repair module simulates candidates before accepting them and returns a final calculation mask plus diagnostics. The existing recognition response, overlay, room polygons, areas, and load geometry all use the final mask; the raw model mask is saved only as a backend diagnostic artifact.

**Tech Stack:** Python 3.12, NumPy, OpenCV, pdfplumber, Flask, unittest.

## Global Constraints

- Apply only when `has_vector_geometry` is true; scanned PDFs and images keep the existing path.
- Add at most one exterior line, and it must be horizontal or vertical.
- Reject ambiguous or multi-line exterior repairs with `manual_exterior_wall_required`.
- Never remove an internal fragment unless simulation preserves existing trustworthy rooms.
- The page displays only the final calculation mask; the raw mask is diagnostic-only.
- Preserve all existing uncommitted user work and do not commit, stage, reset, or clean it.

---

### Task 1: Dark Vector Structural Evidence

**Files:**
- Modify: `vector_pdf_scale.py`
- Test: `tests/test_vector_pdf_scale.py`

**Interfaces:**
- Produces: `build_structural_vector_mask(page_data: dict) -> tuple[np.ndarray, dict]`
- Evidence keys: `enabled`, `dark_edge_count`, `masked_pixels`

- [ ] Write failing tests proving dark neutral edges are rasterized, gray furniture edges are excluded, and insufficient evidence disables the mask.
- [ ] Run `python -m unittest tests.test_vector_pdf_scale` and confirm the new tests fail because the function is absent.
- [ ] Implement the mask using existing styled-edge coordinate conversion and antialiased line drawing. Accept neutral edges with brightness at most `0.25`; require at least eight edges before enabling.
- [ ] Run `python -m unittest tests.test_vector_pdf_scale` and confirm all tests pass.

### Task 2: Conservative Repair Engine

**Files:**
- Create: `floorplan_topology_repair.py`
- Create: `tests/test_floorplan_topology_repair.py`
- Read: `floorplan_rooms.py`

**Interfaces:**
- Produces:

```python
def repair_vector_floorplan_topology(
    class_mask: np.ndarray,
    *,
    building_roi: list[int] | None,
    structural_support_mask: np.ndarray | None,
    max_exterior_gap_px: int,
    max_internal_component_area_px: int,
    min_room_area_px: float,
) -> dict:
    return {
        "calculation_mask": np.ndarray,
        "exterior_repair": {
            "status": str,
            "candidate_count": int,
            "accepted_line_px": list[int] | None,
            "reason": str,
        },
        "internal_fragments": {
            "ignored": list[dict],
            "repaired": list[dict],
            "ambiguous": list[dict],
        },
        "manual_exterior_wall_required": bool,
        "manual_review_reasons": list[str],
    }
```

- [ ] Write synthetic failing tests for: one supported horizontal exterior gap is repaired; one supported vertical gap is repaired; two valid candidates are rejected; a candidate requiring an L-shaped or multi-line repair is rejected; an unsupported outside annotation line is not connected.
- [ ] Write synthetic failing tests for: an internal fragment with one unique supported short-line completion is repaired before removal; a small unsupported internal component that does not bound a room is omitted; a component touching an existing room boundary is preserved; removing a component that changes existing room count or area is rejected as ambiguous.
- [ ] Run `python -m unittest tests.test_floorplan_topology_repair` and confirm failure because the module is absent.
- [ ] Implement candidate generation by comparing horizontal and vertical conservative closures against the current barrier. Convert each connected addition into one axis-aligned segment and reject thick, diagonal, branched, overlong, or unsupported additions.
- [ ] Simulate every exterior segment independently. Accept only when exactly one candidate creates a plausible newly enclosed region inside the ROI without reducing or merging existing rooms. Otherwise return the manual-review status without changing the mask.
- [ ] Find small internal connected components away from the ROI edge. First simulate unique supported single-line completions that create one plausible room; otherwise simulate removal one component at a time and omit only weakly supported components that leave existing trustworthy room count and areas stable. Preserve all ambiguous components.
- [ ] Run `python -m unittest tests.test_floorplan_topology_repair` and confirm all tests pass.

### Task 3: Recognition Pipeline and Final Display Mask

**Files:**
- Modify: `floorplan_onnx.py`
- Modify: `web_server_server.py`
- Modify: `tests/test_floorplan_rooms.py`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- `FloorplanSegmenterONNX.predict(..., topology_repair_context: dict | None = None)` returns `mask` as the final calculation mask, `raw_model_mask` as the pre-repair mask, and `topology_repair` diagnostics.
- The Flask route saves `ai_raw_model_mask.png` for diagnostics and continues saving `ai_mask.png` and `ai_overlay.jpg` from the final mask.

- [ ] Write failing segmenter tests proving repair is skipped without vector context, while supplied context changes the returned `mask` but preserves `raw_model_mask`.
- [ ] Write failing route tests proving the structural mask and ROI are passed to the segmenter, the raw diagnostic mask is saved, and response room topology comes from the final mask.
- [ ] Run the focused tests and confirm failures are caused by missing integration.
- [ ] Extend `predict` to invoke the repair engine after ROI filtering but before overlay, statistics, geometry, and room extraction. Recompute all downstream values from the final mask.
- [ ] Build dark structural evidence in the vector-PDF route, pass the repair context, save `ai_raw_model_mask.png`, and include JSON-safe repair diagnostics in the recognition report.
- [ ] Run `python -m unittest tests.test_floorplan_rooms tests.test_energy_template` and confirm all focused tests pass.

### Task 4: Real Page Validation and Regression Verification

**Files:**
- Runtime diagnostic output only: `tmp/pdfs/worker-palace-topology-repaired-overlay.jpg`

- [ ] Run page 13 through gray-layer cleanup, full-context inference, ROI filtering, and topology repair.
- [ ] Inspect the final overlay and diagnostics. Confirm at most one exterior line was added, ambiguous candidates did not alter the mask, and safe internal fragments no longer appear in the final overlay.
- [ ] Run `python -m unittest discover -s tests` and require zero failures.
- [ ] Run Python compilation, current-template JavaScript syntax validation, and `git diff --check`.
- [ ] Restart only the existing `run_local.py` processes belonging to this workspace and verify `http://127.0.0.1:5000/energy` returns HTTP 200.

### Task 5: Recessed Multi-Door Entrance Boundary

**Files:**
- Modify: `floorplan_topology_repair.py`
- Modify: `tests/test_floorplan_topology_repair.py`

**Interfaces:**
- Extends `repair_vector_floorplan_topology(...)` without changing its public signature.
- Accepted entrance repair remains one `[x0, y, x1, y]` line in `exterior_repair.accepted_line_px`.

- [ ] Write failing tests proving one long lower entrance line with two vertical anchors and at least 80% vector support is repaired even when it exceeds `max_exterior_gap_px`; two distinct valid entrance rows are rejected; a long dimension line without vertical wall anchors is rejected.
- [ ] Run the focused tests and confirm they fail because long entrance candidates are not generated.
- [ ] Detect dense lower horizontal support rows, merge neighboring rows, find vertically continuous barrier anchors, and simulate the widest supported anchor pair as one line.
- [ ] Accept the entrance line only when it is the unique logical exterior candidate and creates one plausible new closed region without reducing existing rooms.
- [ ] Run the real page-13 mask and verify the accepted line stays on the recessed entrance baseline, then run the full suite and restart the service.
