# Vector PDF Structural Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove vector furniture/annotation layers, isolate the building body, and use conservative room closure for vector PDFs.

**Architecture:** `vector_pdf_scale.py` exposes JSON-compatible styled edges and pure functions for masks and ROI evidence. `web_server_server.py` applies those artifacts only to vector PDFs and passes explicit topology settings to the segmenter; `floorplan_onnx.py` supports ROI inference without changing full-page coordinates.

**Tech Stack:** pdfplumber, OpenCV, NumPy, ONNX Runtime, Flask, unittest.

## Global Constraints

- Only vector PDFs with extractable colour and geometry receive the new cleanup.
- Original image and all returned coordinates remain in full-page coordinates.
- Scanned PDFs and images retain their current path.
- Preserve all uncommitted work; do not commit, merge, push, or clean the workspace.

---

### Task 1: Styled vector edges and vector classification

**Files:** Modify `vector_pdf_scale.py`; test `tests/test_vector_pdf_scale.py`.

**Interfaces:** `extract_vector_page(...)` produces `styled_edges`, `has_vector_geometry`, and `has_vector_text`; `is_vector_pdf` aliases geometry availability.

- [ ] Add a failing outlined-text/vector-line PDF test proving geometry classification does not require extractable words.
- [ ] Add a failing test proving styled edges preserve grayscale colour and line width.
- [ ] Implement filtered styled-edge extraction without retaining unstyled text-outline edges.
- [ ] Run focused tests until green.

### Task 2: Non-structural mask and building ROI

**Files:** Modify `vector_pdf_scale.py`; test `tests/test_vector_pdf_scale.py`.

**Interfaces:** `build_nonstructural_vector_mask(page_data) -> (mask, evidence)` and `detect_building_roi(page_data) -> dict`.

- [ ] Add failing synthetic tests with black structural rectangles, gray furniture, black exterior dimensions and a title strip.
- [ ] Implement conditional neutral-gray rasterization with black/gray evidence thresholds.
- [ ] Implement density-grid ROI detection with aspect, area and confidence guards.
- [ ] Verify the gray mask covers furniture but not black walls, and the ROI excludes dimensions/title strip.

### Task 3: ROI inference and conservative topology

**Files:** Modify `floorplan_onnx.py`, `floorplan_rooms.py`; test `tests/test_floorplan_rooms.py` and `tests/test_energy_template.py`.

**Interfaces:** `predict(..., inference_roi=None, topology_max_gap_px=None, topology_min_room_area_px=None)` returns a full-page mask and topology.

- [ ] Add failing tests for ROI coordinate restoration and ROI-outside background.
- [ ] Add failing tests proving explicit 12-pixel closure and minimum room area are preserved in result metadata.
- [ ] Implement cropped inference and full-page probability/mask restoration.
- [ ] Keep existing defaults unchanged when explicit parameters are absent.

### Task 4: PDF route integration and artifacts

**Files:** Modify `web_server_server.py`; test `tests/test_energy_template.py`.

**Interfaces:** Vector PDF recognition saves `pdf_nonstructural_mask.png`, `pdf_model_input.png`, and `pdf_building_roi.json`; recognition payload contains `vector_cleanup` evidence.

- [ ] Add failing route tests proving vector cleanup parameters reach the segmenter and scanned PDFs do not use them.
- [ ] Compose dimension and non-structural masks, clean only the inference image, and pass ROI/topology settings.
- [ ] Persist artifact names and cleanup evidence without changing original preview encoding.
- [ ] Run focused route tests until green.

### Task 5: Real-page comparison and full verification

**Files:** Verify only; generated diagnostics under `tmp/pdfs/`.

- [ ] Re-run page 13 and compare wall pixels, geometry counts, room count, and largest-room continuity against the saved baseline.
- [ ] Render and visually inspect original, gray mask, ROI and cleaned overlay.
- [ ] Run all unit tests, Python compilation, JavaScript syntax, and `git diff --check`.
- [ ] Restart the local service and verify HTTP 200 without committing changes.
