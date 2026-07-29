# Energy PDF Region Recognition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let energy-platform users preview a selected PDF page and choose either whole-page recognition or one dragged crop region for AI recognition and area calculation.

**Architecture:** Add a focused `energy_pdf_region.py` module for preview rendering, request validation, coordinate mapping, and aligned-array cropping. Extend the existing energy routes to serve previews and pass region metadata through recognition, while the energy template reuses the annotation tool’s crop-coordinate helper for pointer interaction.

**Tech Stack:** Python 3.12, Flask, NumPy, OpenCV, pdf2image/Poppler, vanilla JavaScript, `unittest`, Node.js assertions.

## Global Constraints

- Crop-region recognition is supported only for prepared PDF uploads.
- Each recognition request contains at most one crop region.
- PDF preview rendering uses exactly 100 DPI and does not run vector analysis or model inference.
- Crop bounds use `[x0, y0, x1, y1]` in preview-image pixels.
- Crop width and height must each be at least 128 preview pixels.
- Crop area must be at least 1% of the preview page area.
- Whole-page recognition remains the default and old requests without region fields remain compatible.
- Full-page scale calibration is retained; only image-aligned model and topology inputs are cropped.
- Switching pages, uploading another PDF, or selecting whole-page mode clears the crop.

---

### Task 1: Add PDF Region Domain Helpers

**Files:**
- Create: `energy_pdf_region.py`
- Create: `tests/test_energy_pdf_region.py`

**Interfaces:**
- Produces: `render_pdf_page_preview(pdf_path, page_number, page_count, poppler_path) -> np.ndarray`
- Produces: `parse_crop_region_request(mode_raw, bbox_raw, preview_size_raw) -> dict`
- Produces: `map_crop_bbox_to_page(crop_bbox_px, crop_preview_size, page_size) -> list[int]`
- Produces: `crop_page_inputs(render_bgr, cleaned_bgr, cleanup_mask, structural_support_mask, inference_roi, crop_bbox_page_px) -> dict`

- [ ] **Step 1: Write failing validation and mapping tests**

Create tests that require:

```python
request_data = parse_crop_region_request(
    "crop_region",
    "[100, 50, 500, 350]",
    "[800, 600]",
)
self.assertEqual(request_data["crop_bbox_px"], [100, 50, 500, 350])
self.assertEqual(
    map_crop_bbox_to_page([100, 50, 500, 350], [800, 600], [1600, 1200]),
    [200, 100, 1000, 700],
)
```

Also require `ValueError` for malformed JSON, non-finite values, out-of-bounds
coordinates, dimensions below 128 pixels, area below 1%, and any mode other
than `full_page` or `crop_region`. Verify `full_page` accepts missing crop
fields and returns no crop.

- [ ] **Step 2: Run helper tests and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_pdf_region -v
```

Expected: import failure because `energy_pdf_region.py` does not exist.

- [ ] **Step 3: Implement parsing and coordinate mapping**

Implement strict JSON-array parsing with `math.isfinite`. Validate preview
coordinates before mapping. Use `math.floor` for mapped left/top,
`math.ceil` for right/bottom, clamp to the prepared page, then revalidate
that the mapped box has positive width and height.

- [ ] **Step 4: Write failing aligned-crop tests**

Build 600×800 synthetic render, cleaned, cleanup-mask, and structural-mask
arrays with identifiable pixel values. Crop `[100, 50, 500, 350]` and assert:

```python
self.assertEqual(result["render_bgr"].shape[:2], (300, 400))
self.assertEqual(result["cleaned_bgr"].shape[:2], (300, 400))
self.assertEqual(result["cleanup_mask"].shape, (300, 400))
self.assertEqual(result["structural_support_mask"].shape, (300, 400))
self.assertEqual(result["inference_roi"], [20, 30, 400, 300])
```

Use source ROI `[120, 80, 700, 500]`; its intersection must be translated
into crop-local coordinates. Add a no-intersection case that falls back to
the complete crop ROI `[0, 0, crop_width, crop_height]`.

- [ ] **Step 5: Implement aligned cropping**

Return independent `.copy()` arrays for every present image or mask. Preserve
`None` for absent structural support. Intersect and translate the inference
ROI without resizing any pixels.

- [ ] **Step 6: Write failing preview-render test**

Patch `energy_pdf_region.convert_from_path` and require exactly:

```python
convert_from_path(
    str(pdf_path),
    dpi=100,
    first_page=2,
    last_page=2,
    poppler_path=poppler_path,
)
```

Require BGR output and reject page numbers outside `1..page_count`.

- [ ] **Step 7: Implement preview rendering and run GREEN**

Render exactly one page, convert RGB to BGR, and reject any renderer result
whose image count is not one.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_pdf_region -v
```

- [ ] **Step 8: Commit helper module**

```powershell
git add -- energy_pdf_region.py tests/test_energy_pdf_region.py
git commit -m "feat: add PDF recognition region helpers"
```

---

### Task 2: Add the Energy PDF Page Preview Route

**Files:**
- Modify: `web_server_server.py`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: `render_pdf_page_preview(...)`
- Produces: `POST /energy/pdf_page_preview`
- Produces: `GET /energy/crop_region.js`

- [ ] **Step 1: Write failing route tests**

Add route-client tests that prepare a two-page PDF, patch
`render_pdf_page_preview`, and post:

```python
{
    "report_number": "REGION-PREVIEW",
    "pdf_upload_token": prepared["upload_token"],
    "pdf_page_number": "2",
}
```

Require HTTP 200, page number 2, page count 2,
`image_size == [800, 600]`, and a decodable JPEG Base64 string. Verify the
helper received the stored PDF path, page 2, total page count, and resolved
Poppler path. Add 400 cases for a token/report mismatch and an invalid page.

Add a test requiring `/energy/crop_region.js` to return the exact shared
`annotation_tool/static/crop_region.js` content with a JavaScript MIME type.

- [ ] **Step 2: Run route tests and verify RED**

Run the new preview-route test methods with:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template.EnergyRouteClientTests -v
```

Expected: 404 for the two new routes.

- [ ] **Step 3: Implement preview and shared-helper routes**

Factor prepared-PDF token/path/page validation into a small server helper
used by both preview and recognition. The preview route must JPEG-encode at
quality 85 and return:

```json
{
  "success": true,
  "pdf_page_number": 2,
  "pdf_page_count": 2,
  "image_size": [800, 600],
  "image": "<base64>"
}
```

Serve the existing crop helper with `send_from_directory` from
`annotation_tool/static`; do not create a second copy.

- [ ] **Step 4: Run route tests and commit**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template.EnergyRouteClientTests -v
git add -- web_server_server.py tests/test_energy_template.py
git commit -m "feat: preview selected energy PDF pages"
```

---

### Task 3: Apply Crop Regions During Recognition

**Files:**
- Modify: `web_server_server.py`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: `parse_crop_region_request(...)`
- Consumes: `map_crop_bbox_to_page(...)`
- Consumes: `crop_page_inputs(...)`
- Extends: `POST /energy/ai_recognize`
- Extends: `_build_recognition_payload(...)`

- [ ] **Step 1: Write failing invalid-request tests**

Require HTTP 400 when `recognition_mode=crop_region` is used with PNG/JPG,
when crop JSON is malformed, when preview dimensions differ from the bounds,
or when the crop is below the minimum. Require legacy PDF requests without
`recognition_mode` to continue returning 200.

- [ ] **Step 2: Run invalid-request tests and verify RED**

Run only the new test methods. Expected: crop requests are currently ignored
instead of rejected.

- [ ] **Step 3: Parse and validate recognition mode**

Parse the region request before expensive PDF preparation. Return the
`ValueError` message as HTTP 400. Reject crop mode unless a valid prepared
PDF token is present.

- [ ] **Step 4: Write failing crop-integration test**

Return a synthetic `PreparedFloorplanPage` sized 1200×1600. Submit a preview
crop `[100, 50, 500, 350]` with preview size `[800, 600]`. Require:

- mapped page crop `[200, 100, 1000, 700]`;
- `segmenter.predict` receives an image sized 600×800;
- `structural_support_mask` is also 600×800;
- the translated inference ROI is passed to prediction;
- `scale_calibration` from the whole prepared page is unchanged;
- response `image_size == [800, 600]`;
- returned original image decodes to 600×800.

- [ ] **Step 5: Implement aligned region inference**

After `prepare_pdf_page`, map the preview crop to the formal render size,
then replace the local variables used for inference with the output from
`crop_page_inputs`. Keep `scale_calibration` and vector evidence from the
whole page. Update `vector_cleanup.building_roi` to the crop-local inference
ROI so diagnostics and topology repair use the same coordinate space.

Write cropped `original_bgr` to `building_plan_ai.png` so persisted artifacts
match the returned region rather than the whole PDF page.

- [ ] **Step 6: Write failing persistence test**

Require both response and `recognition.json` to include:

```json
{
  "recognition_mode": "crop_region",
  "crop_bbox_page_px": [200, 100, 1000, 700],
  "crop_bbox_preview_px": [100, 50, 500, 350],
  "crop_preview_size": [800, 600]
}
```

Require whole-page results to contain `recognition_mode: "full_page"` and
null crop metadata.

- [ ] **Step 7: Persist region metadata and run GREEN**

Attach the four fields to `result`, include them in
`_build_recognition_payload`, and return them at the top level of the API
response.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template.EnergyRouteClientTests -v
```

- [ ] **Step 8: Commit recognition integration**

```powershell
git add -- web_server_server.py tests/test_energy_template.py
git commit -m "feat: recognize cropped PDF regions"
```

---

### Task 4: Add PDF Preview and Crop Interaction to the Energy Page

**Files:**
- Modify: `templates/energy.html`
- Modify: `tests/test_energy_template.py`
- Modify: `tests/test_crop_region.js`

**Interfaces:**
- Consumes: `GET /energy/crop_region.js`
- Consumes: `POST /energy/pdf_page_preview`
- Sends: `recognition_mode`, `crop_bbox_px`, `crop_preview_size`

- [ ] **Step 1: Write failing template contract tests**

Require the template to contain:

- `pdf-recognition-mode-full`
- `pdf-recognition-mode-crop`
- `pdf-page-preview-image`
- `pdf-crop-overlay`
- `clear-pdf-crop`
- `pdf-crop-summary`
- `<script src="/energy/crop_region.js">`

Require functions named `loadPreparedPdfPreview`, `resetPdfCropSelection`,
`setPdfRecognitionMode`, and `updatePdfCropOverlay`.

Assert the recognition request appends `recognition_mode`, `crop_bbox_px`,
and `crop_preview_size`.

- [ ] **Step 2: Extend and run JavaScript helper tests**

Keep `tests/test_crop_region.js` against the shared annotation helper. Add
boundary cases for reversed drag direction and points outside the displayed
image. Run:

```powershell
node tests/test_crop_region.js
```

These helper tests must stay green because the energy page is reusing the
existing implementation.

- [ ] **Step 3: Run template tests and verify RED**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template.EnergyTemplateTests -v
```

Expected: missing controls and functions.

- [ ] **Step 4: Add preview and mode controls**

Replace the current page-only selector block with a page selection row and a
preview panel. Default to whole-page mode. Load a preview after every valid
page selection, including one-page PDFs; remove the current automatic
one-page inference.

Use an `<img>` plus an absolutely positioned selection element. Pointer
coordinates must first be converted from the image’s displayed rectangle to
its natural pixel dimensions, then normalized through
`AnnotationCropRegion.normalizeCrop`.

- [ ] **Step 5: Implement drag selection and reset behavior**

On pointer down/move/up in crop mode:

- clamp coordinates to the displayed image;
- render the selection rectangle in CSS display coordinates;
- store the final box in natural preview pixels;
- validate it through `AnnotationCropRegion.validateCrop`;
- show width × height;
- disable recognition until valid.

Clear crop state when page selection changes, a new PDF is uploaded, or
whole-page mode is selected. Keep a valid selection after a failed preview
or recognition request so the user can retry.

- [ ] **Step 6: Send mode-specific recognition fields**

Always send `recognition_mode`. In crop mode also JSON-encode
`crop_bbox_px` and `[naturalWidth, naturalHeight]` as `crop_preview_size`.
Change button text between “开始识别整页” and “开始识别所选区域”.

- [ ] **Step 7: Run frontend and route tests**

```powershell
node tests/test_crop_region.js
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template -v
```

- [ ] **Step 8: Commit the UI**

```powershell
git add -- templates/energy.html tests/test_energy_template.py tests/test_crop_region.js
git commit -m "feat: select PDF recognition regions"
```

---

### Task 5: Regression and Real-PDF Verification

**Files:**
- Modify only if a failing test exposes a defect in files already in scope.

**Interfaces:**
- Verifies all interfaces produced by Tasks 1–4.

- [ ] **Step 1: Run focused tests**

```powershell
node tests/test_crop_region.js
.\.venv\Scripts\python.exe -m unittest tests.test_energy_pdf_region tests.test_energy_template -v
```

- [ ] **Step 2: Run the full suite**

```powershell
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

- [ ] **Step 3: Restart the local platform**

Stop only the `run_local.py` parent/child pair owning port 5000. Start the
service from `G:\bim-web` with:

```powershell
$env:ONNX_MODEL_PATH="G:\bim网页\标注数据\models\文化宫-1至4层-finetune-v1\best.onnx"
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
.\.venv\Scripts\python.exe run_local.py
```

Verify `GET /energy/ai_status` returns HTTP 200 and `best.onnx`.

- [ ] **Step 4: Verify the real fourth/fifth-floor PDF page**

In the energy page:

1. Upload the PDF containing both fourth and fifth floors.
2. Select their shared page and confirm the 100-DPI preview appears.
3. Select crop mode and frame only the fourth floor.
4. Confirm returned original and overlay contain only the fourth floor,
   automatic scale remains available, and closed-room area is computed.
5. Repeat for the fifth floor.
6. Switch to whole-page mode and confirm the existing full-page request works.

- [ ] **Step 5: Inspect final scope and commit any verification fix**

```powershell
git diff --check
git status --short
git log -5 --oneline
```

Do not stage unrelated untracked personal or deployment files.
