# Vector PDF Scale Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically derive a trustworthy metre-per-pixel scale from the overall horizontal and vertical dimensions in CAD-exported vector PDFs, then use that confirmed scale with closed room polygons to calculate real room areas.

**Architecture:** Add a focused `vector_pdf_scale.py` module that extracts positioned text and vector segments with pdfplumber, detects dimension evidence, and returns a calibration result without depending on Flask. The existing pdf2image path will render the same vector page, persist calibration evidence in `recognition.json`, and pass only confirmed scales into the room-area calculation. A small manual two-point endpoint and image-overlay interaction provide the fallback when automatic evidence is incomplete or conflicting.

**Tech Stack:** Python 3.12, Flask, pdfplumber, pdf2image, reportlab (tests), NumPy, OpenCV, unittest, existing ONNX room segmentation.

## Global Constraints

- Primary target is CAD-exported vector PDF; scanned PDF and photos return `manual_required` in this iteration.
- A dimension candidate must combine numeric text, a parallel dimension line, and endpoint evidence; the longest raw PDF line alone is never accepted.
- Horizontal and vertical scales differing by at most 2% are automatically confirmed; 2%–5% requires confirmation; over 5% is rejected.
- Missing or unconfirmed scale means pixel area only and `load_geometry_ready = false`.
- Existing PNG/JPG recognition and legacy `recognition.json` remain readable.
- Work directly in the current dirty `G:\bim-web` workspace and preserve all unrelated uncommitted changes.
- Do not create implementation commits that would accidentally include existing unrelated hunks; commit only if the user explicitly requests it after review.

---

### Task 1: Vector PDF extraction boundary

**Files:**
- Create: `vector_pdf_scale.py`
- Modify: `requirements.txt`
- Test: `tests/test_vector_pdf_scale.py`

**Interfaces:**
- Produces: `extract_vector_page(pdf_path: str | Path, page_index: int = 0, dpi: int = 200) -> dict`
- Output keys: `page_size_pt`, `render_size_px`, `text_spans`, `segments`, `vector_text_count`, `vector_segment_count`, `is_vector_pdf`.
- Each text span contains `text`, `bbox_pt`, `center_pt`, `direction`, and `font_size`.
- Each segment contains `start_pt`, `end_pt`, `orientation`, `length_pt`, and `width_pt`.

- [ ] **Step 1: Add failing extraction tests**

Create a temporary PDF with reportlab containing horizontal and vertical lines plus positioned text. Assert that `extract_vector_page()` returns both text spans and axis-aligned segments, and that an image-only PDF returns `is_vector_pdf == false`.

```python
def test_extracts_positioned_text_and_axis_aligned_segments_from_vector_pdf(self):
    pdf_path = self.make_vector_dimension_pdf()
    page = extract_vector_page(pdf_path, dpi=200)
    self.assertTrue(page["is_vector_pdf"])
    self.assertIn("40600", {span["text"] for span in page["text_spans"]})
    self.assertTrue(any(line["orientation"] == "horizontal" for line in page["segments"]))
    self.assertTrue(any(line["orientation"] == "vertical" for line in page["segments"]))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vector_pdf_scale.VectorPdfExtractionTests -v
```

Expected: import failure because `vector_pdf_scale` does not exist.

- [ ] **Step 3: Add pdfplumber and minimal extraction implementation**

Add `pdfplumber>=0.11,<1`, `pdf2image>=1.17,<2`, and `reportlab>=4,<5` to `requirements.txt`. In `vector_pdf_scale.py`, open the document with `pdfplumber.open()`, read words with `page.extract_words(...)`, and normalize `page.lines`, rectangle edges, and straight curve edges into plain JSON-compatible dictionaries. Convert the requested DPI to `render_size_px` using `dpi / 72`.

```python
def extract_vector_page(pdf_path, page_index=0, dpi=200):
    with pdfplumber.open(str(pdf_path)) as document:
        page = document.pages[page_index]
        text_spans = _extract_text_spans(page.extract_words(extra_attrs=["size", "upright"]))
        segments = _extract_axis_aligned_segments(page)
        zoom = dpi / 72.0
        return {
            "page_size_pt": [float(page.width), float(page.height)],
            "render_size_px": [round(page.width * zoom), round(page.height * zoom)],
            "text_spans": text_spans,
            "segments": segments,
            "vector_text_count": len(text_spans),
            "vector_segment_count": len(segments),
            "is_vector_pdf": len(text_spans) > 0 and len(segments) >= 4,
        }
```

- [ ] **Step 4: Run extraction tests and the existing preprocessing tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vector_pdf_scale.VectorPdfExtractionTests tests.test_compare_floorplan_preprocessing -v
```

Expected: PASS.

---

### Task 2: Overall dimension detection and consensus calibration

**Files:**
- Modify: `vector_pdf_scale.py`
- Test: `tests/test_vector_pdf_scale.py`

**Interfaces:**
- Consumes: extraction dictionary from `extract_vector_page()`.
- Produces: `detect_dimension_candidates(page_data: dict) -> list[dict]`.
- Produces: `calibrate_from_overall_dimensions(page_data: dict) -> dict`.
- Calibration keys: `status`, `method`, `scale_m_per_px`, `confidence`, `horizontal`, `vertical`, `axis_difference_percent`, `evidence`.

- [ ] **Step 1: Add failing candidate-classification tests**

Build synthetic pages containing `40600`, `18600`, `1:100`, `68.80m²`, `13.500`, `C1376`, a page border, grid lines, and endpoint extension lines. Assert that the overall horizontal and vertical dimensions are retained while scale text, area, elevation, identifiers, and the page border are rejected.

```python
def test_selects_overall_horizontal_and_vertical_dimensions(self):
    result = calibrate_from_overall_dimensions(extract_vector_page(self.make_vector_dimension_pdf()))
    self.assertEqual(result["horizontal"]["text"], "40600")
    self.assertEqual(result["vertical"]["text"], "18600")
    self.assertEqual(result["status"], "confirmed")
```

- [ ] **Step 2: Run the new test and verify RED**

Expected: failure because `calibrate_from_overall_dimensions` is missing.

- [ ] **Step 3: Implement numeric filtering and line matching**

Implement `_parse_dimension_metres(text)` so plain CAD dimensions default to millimetres, explicit `m` remains metres, and scale/area/elevation/identifier tokens return `None`. Merge collinear line fragments separated by a text-sized gap, then score a text-line pair by parallel orientation, distance, endpoint extension-line evidence, page-edge position, and span length.

```python
def _parse_dimension_metres(text):
    normalized = text.strip().lower().replace(" ", "")
    if ":" in normalized or "²" in normalized or "㎡" in normalized:
        return None
    if re.search(r"[a-zA-Z]", normalized) and not normalized.endswith(("mm", "m")):
        return None
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(mm|m)?", normalized)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2) or "mm"
    if unit == "mm" and "." in match.group(1) and value < 100:
        return None
    return value / 1000.0 if unit == "mm" else value
```

- [ ] **Step 4: Implement two-axis validation**

Convert PDF point spans to pixel spans using `dpi / 72`. Choose the largest supported horizontal and vertical candidates, calculate both scales, and apply the exact 2% / 5% thresholds from the spec.

```python
axis_difference = abs(scale_x - scale_y) / ((scale_x + scale_y) / 2.0)
if axis_difference <= 0.02:
    status, confidence = "confirmed", 0.95
elif axis_difference <= 0.05:
    status, confidence = "confirmation_required", 0.65
else:
    status, confidence = "conflict", 0.0
```

- [ ] **Step 5: Add boundary and fallback tests**

Test exact 2%, exact 5%, over 5%, horizontal-only, vertical-only, no-dimension, and duplicate top/bottom evidence. Expected statuses are `confirmed`, `confirmation_required`, `conflict`, or `manual_required`.

- [ ] **Step 6: Run all vector calibration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vector_pdf_scale -v
```

Expected: PASS.

---

### Task 3: PDF rendering and annotation masking

**Files:**
- Modify: `vector_pdf_scale.py`
- Modify: `web_server_server.py`
- Test: `tests/test_vector_pdf_scale.py`
- Test: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: a BGR page image rendered by the existing pdf2image conversion at the same DPI.
- Produces: `build_dimension_annotation_mask(page_data, candidates) -> np.ndarray` with `0` for retained drawing and `255` for detected annotation pixels.
- Produces: `remove_dimension_annotations(image_bgr, mask) -> np.ndarray`.

- [ ] **Step 1: Add failing render-coordinate and mask tests**

Assert that PDF point coordinates map to the expected raster pixels at 200 DPI. Assert that matched dimension text, baseline, and extension lines are white in the mask while a synthetic wall rectangle remains unmasked.

- [ ] **Step 2: Run the mask tests and verify RED**

Expected: failure because rendering and masking functions are missing.

- [ ] **Step 3: Implement rendering and evidence masks**

Render through `pdf2image.convert_from_path(..., dpi=dpi, first_page=page_index + 1, last_page=page_index + 1)`, convert RGB to BGR, and rasterize only the bounding boxes and segments attached to accepted dimension candidates. Dilate the annotation mask by 1–2 pixels to remove antialiasing remnants, then fill masked pixels with the median page-border colour.

- [ ] **Step 4: Replace the PDF-only conversion branch narrowly**

In `/energy/ai_recognize`, use the vector module for PDF uploads. Persist the rendered PNG, run calibration, apply the dimension mask for AI inference, and keep the original rendered image for display. If `is_vector_pdf` is false, retain the current image conversion and return `manual_required` calibration instead of inventing a scale.

- [ ] **Step 5: Verify PDF and raster route compatibility**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vector_pdf_scale tests.test_energy_template.EnergyRouteClientTests.test_ai_recognize_persists_recognition_json -v
```

Expected: PASS for vector PDF and existing PNG route fixtures.

---

### Task 4: Persist confirmed scale and compute physical room areas

**Files:**
- Modify: `floorplan_rooms.py`
- Modify: `web_server_server.py`
- Test: `tests/test_floorplan_rooms.py`
- Test: `tests/test_energy_template.py`

**Interfaces:**
- Produces: `apply_scale_to_room_topology(topology: dict, scale_m_per_px: float | None) -> dict`.
- Recognition payload gains `scale_calibration`.
- Simulation uses persisted confirmed scale ahead of request-supplied legacy scale.

- [ ] **Step 1: Add failing room-area readiness tests**

Assert that a closed topology plus confirmed positive scale produces per-room and total square metres with `load_geometry_ready == true`. Assert that missing, non-finite, zero, negative, or unconfirmed scale keeps square-metre fields null and readiness false.

- [ ] **Step 2: Run the room tests and verify RED**

Expected: failure because `apply_scale_to_room_topology` is missing.

- [ ] **Step 3: Implement immutable topology scaling**

Deep-copy the topology, validate scale, calculate `area_m2 = area_px2 * scale²`, and never mutate the stored pixel polygons.

```python
def apply_scale_to_room_topology(topology, scale_m_per_px):
    result = copy.deepcopy(topology)
    valid = scale_m_per_px is not None and math.isfinite(scale_m_per_px) and scale_m_per_px > 0
    for room in result.get("rooms", []):
        room["area_m2"] = room["area_px2"] * scale_m_per_px ** 2 if valid else None
    result["total_area_m2"] = result.get("total_area_px2", 0.0) * scale_m_per_px ** 2 if valid else None
    result["scale_m_per_px"] = scale_m_per_px if valid else None
    result["load_geometry_ready"] = bool(valid and result.get("room_count", 0) > 0)
    return result
```

- [ ] **Step 4: Persist calibration and use it in simulation**

Add `scale_calibration` to `_build_recognition_payload()` and the recognition response. In `/energy/ai_simulate`, use `scale_calibration.scale_m_per_px` only when status is `confirmed`; otherwise accept an explicitly saved manual calibration, and reject automatic load calculation without either source. Keep the existing legacy branch only for old payloads without `scale_calibration`.

- [ ] **Step 5: Run route and room-area tests**

Expected: confirmed automatic calibration overrides the old frontend `0.05`, room polygon area becomes the calculation floor area, and legacy payload tests continue to pass.

---

### Task 5: Manual two-point fallback and minimal UI status

**Files:**
- Modify: `web_server_server.py`
- Modify: `templates/energy.html`
- Test: `tests/test_energy_template.py`

**Interfaces:**
- Adds: `POST /energy/scale_calibration` with `report_number`, two image points, `actual_length`, and `unit`.
- Returns and persists a `scale_calibration` object with `method = manual_two_point` and `status = confirmed`.

- [ ] **Step 1: Add failing endpoint tests**

Test a 100-pixel span with `10 m` yields `0.1 m/px`; `10000 mm` yields the same result. Reject coincident points, non-positive length, unsupported units, non-finite values, missing recognition payload, and points outside image bounds.

- [ ] **Step 2: Run endpoint tests and verify RED**

Expected: HTTP 404 because the route does not exist.

- [ ] **Step 3: Implement endpoint validation and persistence**

Compute Euclidean pixel distance, normalize `mm` and `m`, save the manual evidence in the existing `recognition.json`, reapply the scale to room topology, and return the updated total area.

- [ ] **Step 4: Add the image-overlay interaction**

Place a transparent canvas over the original preview. Show the automatic status first. Only when status is not `confirmed`, let two clicks draw endpoints and a connecting line, then submit one actual-length field and unit selector. Do not ask the user to classify walls, doors, or windows.

- [ ] **Step 5: Remove the frontend scale default**

Initialize the scale field empty. Populate it from confirmed automatic or manual calibration only. Disable the calculation action when `load_geometry_ready` is false and show a concise reason.

- [ ] **Step 6: Run template and route tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template -v
```

Expected: PASS.

---

### Task 6: End-to-end verification

**Files:**
- Modify only files required by failures directly caused by Tasks 1–5.

**Interfaces:**
- Consumes all previous task outputs.
- Produces a verified PDF-to-room-area path without changing unrelated energy calculations.

- [ ] **Step 1: Run the focused pipeline tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_vector_pdf_scale tests.test_floorplan_rooms tests.test_energy_template -v
```

Expected: PASS.

- [ ] **Step 2: Run the full regression suite**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run whitespace and syntax checks**

```powershell
git -c safe.directory=G:/bim-web diff --check
.\.venv\Scripts\python.exe -m py_compile vector_pdf_scale.py floorplan_rooms.py floorplan_onnx.py web_server_server.py
```

Expected: no errors.

- [ ] **Step 4: Validate against a real source PDF**

Upload the original vector PDF corresponding to the sample drawing and verify that evidence includes horizontal `40600`, vertical `18600`, axis difference is within the configured threshold, and calculated room area is based on closed polygons. If only the JPG overlay is available, report that synthetic vector tests pass but real-PDF validation remains pending; never infer vector evidence from the screenshot.
