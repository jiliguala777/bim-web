# Vector PDF OCR Scale Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically confirm scale for vector PDFs whose text is stored as outlines, while fixing the manual two-point calibration canvas so two points remain selectable.

**Architecture:** Keep `pdfplumber` text as the primary source. When a page has vector geometry but no extractable text, render that page at 200 DPI for local RapidOCR, convert high-confidence numeric boxes into the existing positioned-span schema, and pass them through the existing dimension-line/end-marker and two-axis agreement checks. Keep the model/room-topology image at the existing 100 DPI. Resize the browser canvas only after it is visible and again when manual calibration starts.

**Tech Stack:** Python 3, Flask, pdfplumber, pdf2image, OpenCV, RapidOCR 3.9.x with ONNX Runtime, unittest, browser JavaScript.

## Global Constraints

- OCR runs locally and never sends drawings to an external service.
- Native PDF text remains the first choice; OCR runs only for vector pages with no usable native text.
- OCR text alone cannot confirm scale; existing vector dimension-line, endpoint, orientation, and two-axis agreement checks remain mandatory.
- Do not change room topology repair, the unique-candidate rule, or the single-straight-line repair rule.
- Keep vector-PDF room inference at 100 DPI and scanned-PDF inference at 200 DPI.
- Do not commit, switch branches, reset, checkout, or clean untracked files during this work.

---

### Task 1: Local numeric OCR adapter

**Files:**
- Create: `floorplan_ocr.py`
- Create: `tests/test_floorplan_ocr.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `extract_numeric_text_spans(image_bgr: numpy.ndarray, page_size_pt: list[float], engine=None) -> tuple[list[dict], dict]`
- Each span uses the existing keys `text`, `bbox_pt`, `center_pt`, `direction`, `font_size`, plus `source='rapidocr'` and `confidence`.
- Evidence uses `status`, `candidate_count`, `accepted_count`, and `min_confidence`.

- [ ] **Step 1: Write failing adapter tests**

Create `tests/test_floorplan_ocr.py` with a fake engine returning an object whose `boxes`, `txts`, and `scores` include horizontal `40600`, vertical `18600`, low-confidence digits, room text, and `1:100`. Assert that only high-confidence plain dimension numbers remain, that pixel boxes map to PDF points, and that orientation follows box geometry.

```python
spans, evidence = extract_numeric_text_spans(
    np.full((200, 400, 3), 255, np.uint8),
    [200.0, 100.0],
    engine=fake_engine,
)
self.assertEqual({span["text"] for span in spans}, {"40600", "18600"})
self.assertEqual(evidence["status"], "completed")
```

- [ ] **Step 2: Verify the tests fail for the missing module**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_ocr -v`

Expected: FAIL because `floorplan_ocr` does not exist.

- [ ] **Step 3: Implement the minimal adapter**

Implement lazy RapidOCR initialization, accept an injected engine for tests, read `RapidOCROutput.boxes/txts/scores`, require confidence `>= 0.60`, normalize whitespace, accept only `\d{3,7}` or `\d+(?:\.\d+)?(?:mm|m)`, map pixel coordinates to PDF points, and return structured evidence. If RapidOCR cannot import or initialize, return `([], {"status": "unavailable", ...})` instead of failing the recognition request.

- [ ] **Step 4: Add the runtime dependency**

Add this line to `requirements.txt`:

```text
rapidocr>=3.9.2,<4
```

Keep the existing pinned `onnxruntime==1.20.1` constraint unchanged.

- [ ] **Step 5: Install and verify the focused tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install "rapidocr>=3.9.2,<4"
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_ocr -v
```

Expected: all adapter tests PASS.

---

### Task 2: OCR fallback in the PDF recognition route

**Files:**
- Modify: `web_server_server.py:1548-1600`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: `extract_numeric_text_spans(...)` from Task 1.
- Produces: `scale_calibration.text_source` with `pdf_text`, `rapidocr`, or `none`.
- Produces: `vector_cleanup.ocr` evidence in `recognition.json`.

- [ ] **Step 1: Write failing route tests**

Add two tests to `tests/test_energy_template.py`:

1. A vector page with native text must not call OCR and must preserve the existing confirmed scale behavior.
2. A vector page with `has_vector_text=False` must render an OCR page, inject OCR spans for `40600` and `18600`, then call the existing dimension detector and return a confirmed scale with `text_source='rapidocr'`.

Patch `extract_numeric_text_spans` and `convert_from_path` so the tests use real route flow without loading OCR models.

- [ ] **Step 2: Verify the new route test fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_energy_template.EnergyTemplateTests.test_vector_pdf_without_text_uses_ocr_scale_fallback -v`

Expected: FAIL because the route never calls OCR and scale remains `manual_required`.

- [ ] **Step 3: Implement the fallback with a separate 200-DPI OCR render**

In `web_server_server.py`, keep 100 DPI for the vector model image. Only when `is_vector_pdf` is true and `has_vector_text` is false:

```python
ocr_images = convert_from_path(
    raster_path,
    dpi=200,
    first_page=pdf_page_number,
    last_page=pdf_page_number,
    poppler_path=_resolve_poppler_path(),
)
ocr_bgr = cv2.cvtColor(np.asarray(ocr_images[0]), cv2.COLOR_RGB2BGR)
ocr_spans, ocr_evidence = extract_numeric_text_spans(
    ocr_bgr,
    dimension_page_data["page_size_pt"],
)
dimension_page_data["text_spans"] = ocr_spans
```

Then run the unchanged `detect_dimension_candidates` and `calibrate_from_overall_dimensions` functions. Set `scale_calibration['text_source']` and persist OCR evidence under `vector_cleanup['ocr']`. Do not convert OCR failure into HTTP 500.

- [ ] **Step 4: Run route and scale regressions**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template tests.test_vector_pdf_scale tests.test_floorplan_ocr -v
```

Expected: all tests PASS; existing native-text calibration still confirms without OCR.

---

### Task 3: Manual two-point calibration canvas

**Files:**
- Modify: `templates/energy.html:2197-2284`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Produces: `resizeScaleCalibrationCanvas() -> boolean`, returning false when the image has no visible dimensions.
- Consumes: existing `calibrationPoints` and `aiResultData.image_size`.

- [ ] **Step 1: Write a failing template regression test**

Add a test asserting that `startManualScaleCalibration()` calls `resizeScaleCalibrationCanvas()` before enabling clicks/drawing, and that recognition completion schedules a resize after `gotoStep(2)` via `requestAnimationFrame`.

- [ ] **Step 2: Verify the template test fails**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_energy_template.EnergyTemplateTests.test_manual_scale_canvas_resizes_after_becoming_visible -v`

Expected: FAIL because the current start function draws using the stale `1×1` backing store.

- [ ] **Step 3: Implement visible-size synchronization**

Change `resizeScaleCalibrationCanvas()` to read `getBoundingClientRect()`, return false without changing the backing store when width/height are zero, and otherwise set integer dimensions and redraw. Call it at the start of `startManualScaleCalibration()`. After `gotoStep(2)`, call:

```javascript
requestAnimationFrame(() => resizeScaleCalibrationCanvas());
```

Keep the first marker radius at 5 pixels and leave the canvas transparent.

- [ ] **Step 4: Run template and JavaScript syntax checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template -v
# Extract the existing inline script using the repository test helper, then:
node --check tmp\energy-inline.js
```

Expected: template tests PASS and Node exits 0.

---

### Task 4: Real PDF verification and full regression

**Files:**
- Inspect: `uploads/energy/<new-report>/recognition.json`
- Inspect: generated OCR and overlay artifacts

**Interfaces:**
- Validates all outputs from Tasks 1-3.

- [ ] **Step 1: Run the complete unit suite**

Run: `.\.venv\Scripts\python.exe -m unittest discover -s tests`

Expected: all tests PASS.

- [ ] **Step 2: Run static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m py_compile floorplan_ocr.py vector_pdf_scale.py web_server_server.py
git diff --check
```

Expected: no syntax or whitespace errors. If this directory has no Git metadata, record that `git diff --check` is unavailable and do not initialize a repository.

- [ ] **Step 3: Re-run page 13 through the real endpoint**

Verify the new `recognition.json` contains:

```text
topology_repair.exterior_repair.status = repaired
topology_repair.exterior_repair.accepted_line_px = [714, 2499, 1980, 2499]
room_topology.room_count = 17
scale_calibration.status = confirmed
scale_calibration.text_source = rapidocr
scale_calibration.horizontal.text = 40600
scale_calibration.vertical.text = 18600
```

If OCR reads different numbers or only one axis, stop at evidence collection and tune OCR preprocessing/cropping with one hypothesis at a time; do not relax the vector or two-axis safety checks.

- [ ] **Step 4: Browser-check manual fallback**

Start manual two-point calibration, click once, confirm only one yellow dot appears, click a second point, and confirm the connecting line appears without tinting the full image.
