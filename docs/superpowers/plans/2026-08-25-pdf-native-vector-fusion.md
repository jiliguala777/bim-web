# PDF Native Vector Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated vector-PDF path that classifies native horizontal and vertical line segments with the new ten-channel model and finds traceable closed orthogonal room candidates without publishing energy geometry.

**Architecture:** New `vector_pdf_*` modules own PDF extraction, CPU probability inference, line fusion, room topology, and artifact publication. The Flask route reuses only generic upload/page/crop validation; it never imports the legacy ONNX recognition modules into the new path and never overwrites `recognition.json`.

**Tech Stack:** Python 3, Flask, pdfplumber, pdf2image, OpenCV, NumPy, ReportLab, unittest, external PyTorch inference through `training.predict_vector`.

**Spec:** `docs/superpowers/specs/2026-08-25-pdf-native-vector-fusion-design.md`

## Global Constraints

- Work from branch `codex/pdf-native-vector-fusion`, based on `codex/vector-platform-adapter`.
- Do not import or call `floorplan_onnx.py`, `floorplan_page_pipeline.py`, `floorplan_rooms.py`, `floorplan_topology_repair.py`, or `vector_pdf_scale.py` from any new fusion module.
- Copy only the minimum generic algorithms needed from legacy modules into independently named new modules and cover them with new tests.
- Process only vector PDF pages and only horizontal/vertical native segments; diagonal segments are counted and ignored.
- The model may score native candidates but may not add lines absent from the PDF.
- Do not produce doors or windows in the first phase.
- Every response and JSON payload from the new path must contain `load_geometry_ready: false`.
- Do not overwrite or mutate legacy `recognition.json` or the behavior of `POST /energy/ai_recognize`.
- Keep existing untracked files untouched and stage only exact files from each task.
- Use TDD for every production behavior: add one failing test, run it and observe the expected failure, then implement the minimum code.

---

### Task 1: Independent Native PDF Extraction

**Files:**
- Create: `vector_pdf_native.py`
- Create: `tests/test_vector_pdf_native.py`

**Interfaces:**
- Produces: `NativePdfThresholds`, `extract_native_pdf_page(pdf_path, page_index=0, dpi=100) -> dict`, `build_dimension_mask(page_data) -> np.ndarray`, and `crop_native_page_data(page_data, bbox_px) -> dict`.
- The returned page dictionary is JSON-compatible and contains `page_size_pt`, `render_size_px`, `text_spans`, `orthogonal_segments`, `styled_edges`, `dimension_candidates`, `building_roi`, `has_vector_geometry`, `is_vector_pdf`, and `ignored_diagonal_count`.
- No imports from the legacy modules listed in Global Constraints.

- [ ] **Step 1: Write the failing extraction contract test**

Create a ReportLab PDF containing a black orthogonal rectangle, one diagonal, a gray auxiliary line, a numeric dimension baseline with endpoint ticks, and a page frame. Assert the desired public contract:

```python
from vector_pdf_native import extract_native_pdf_page

def test_extracts_only_orthogonal_candidates_and_counts_diagonals(self):
    with tempfile.TemporaryDirectory() as directory:
        page = extract_native_pdf_page(self._make_mixed_vector_pdf(directory), dpi=100)
    self.assertTrue(page["is_vector_pdf"])
    self.assertGreaterEqual(len(page["orthogonal_segments"]), 7)
    self.assertEqual(page["ignored_diagonal_count"], 1)
    self.assertTrue(page["dimension_candidates"])
    self.assertIn("enabled", page["building_roi"])
```

Name the production change that makes this test pass: adding the standalone native extractor and its stable page schema.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```powershell
python -m unittest tests.test_vector_pdf_native.NativePdfExtractionTests.test_extracts_only_orthogonal_candidates_and_counts_diagonals -v
```

Expected: `ModuleNotFoundError: No module named 'vector_pdf_native'`.

- [ ] **Step 3: Implement the minimum standalone extractor**

Implement a fresh module with no legacy imports. The essential normalization is:

```python
@dataclass(frozen=True)
class NativePdfThresholds:
    axis_tolerance_pt: float = 0.5
    border_tolerance_pt: float = 2.0
    border_span_fraction: float = 0.80

def _orthogonal_segment(item: dict, tolerance: float) -> tuple[dict | None, bool]:
    x0, x1 = float(item["x0"]), float(item["x1"])
    top, bottom = float(item["top"]), float(item["bottom"])
    dx, dy = abs(x1 - x0), abs(bottom - top)
    if dy <= tolerance < dx:
        return {"orientation": "horizontal", "start_pt": [min(x0, x1), top],
                "end_pt": [max(x0, x1), top], "length_pt": dx}, False
    if dx <= tolerance < dy:
        return {"orientation": "vertical", "start_pt": [x0, min(top, bottom)],
                "end_pt": [x0, max(top, bottom)], "length_pt": dy}, False
    return None, bool(dx > tolerance and dy > tolerance)
```

Copy only the minimum text/dimension pairing and dense-dark-geometry ROI logic needed by the new contract. Assign stable `native_id` values after sorting by `(orientation, start_pt, end_pt, width_pt, stroke_rgb)`.

- [ ] **Step 4: Verify GREEN for extraction**

Run:

```powershell
python -m unittest tests.test_vector_pdf_native -v
```

Expected: all native extraction tests pass.

- [ ] **Step 5: Add failing mask and crop-coordinate tests**

Add tests asserting that the dimension mask marks the recognized baseline/ticks and that crop conversion clips/translates candidate pixel coordinates while retaining `page_start_px`/`page_end_px`:

```python
cropped = crop_native_page_data(page, [100, 50, 300, 250])
self.assertEqual(cropped["render_size_px"], [200, 200])
self.assertEqual(cropped["crop_bbox_page_px"], [100, 50, 300, 250])
self.assertTrue(all("page_start_px" in item for item in cropped["orthogonal_segments"]))
```

Run the new tests and confirm they fail because the two functions are missing.

- [ ] **Step 6: Implement mask and crop behavior, then run the module suite**

Rasterize exact accepted dimension members using the page-point-to-pixel scale. Crop candidates with orthogonal line clipping; translate analysis coordinates by the crop origin but preserve page coordinates. Run:

```powershell
python -m unittest tests.test_vector_pdf_native -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```powershell
git add -- vector_pdf_native.py tests/test_vector_pdf_native.py
git commit -m "feat: extract independent native PDF candidates"
```

---

### Task 2: Strict CPU Probability Runner

**Files:**
- Create: `vector_pdf_model.py`
- Create: `tests/test_vector_pdf_model.py`

**Interfaces:**
- Produces: `CHANNEL_NAMES`, `VectorModelConfig`, `VectorProbabilityResult`, `load_probability_artifacts(output_dir, expected_size) -> VectorProbabilityResult`, and `run_vector_probabilities(image_path, artifact_parent, config) -> VectorProbabilityResult`.
- Consumes the existing external `training.predict_vector` CLI, but does not consume the adapter's vectorized geometry.

- [ ] **Step 1: Write failing artifact-contract tests**

Create a temporary model output containing `probabilities.npz` and `inference.json`, then assert exact validation:

```python
from vector_pdf_model import CHANNEL_NAMES, load_probability_artifacts

def test_loads_ten_aligned_finite_probability_channels(self):
    probabilities = np.zeros((10, 20, 30), dtype=np.float32)
    self._write_artifacts(probabilities, list(CHANNEL_NAMES))
    result = load_probability_artifacts(self.output, expected_size=(30, 20))
    self.assertEqual(result.probabilities.shape, (10, 20, 30))
    self.assertFalse(result.probabilities.flags.writeable)
```

Also test rejection of wrong channel order, wrong image dimensions, NaN, and values outside `[0, 1]`.

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m unittest tests.test_vector_pdf_model -v
```

Expected: missing `vector_pdf_model` module.

- [ ] **Step 3: Implement strict result and loader types**

```python
CHANNEL_NAMES = (
    "footprint_interior", "footprint_boundary", "room_interior", "room_boundary",
    "wall_centerline", "door_line", "window_line",
    "wall_keypoint", "door_endpoint", "window_endpoint",
)

@dataclass(frozen=True)
class VectorProbabilityResult:
    probabilities: np.ndarray
    inference: dict
    artifact_dir: Path
    probabilities_sha256: str
```

Load with `allow_pickle=False`, require archive members to equal `{"probabilities"}`, validate shape/range/finiteness/channel order, mark the array read-only, and hash the NPZ file.

- [ ] **Step 4: Verify loader GREEN**

```powershell
python -m unittest tests.test_vector_pdf_model.VectorProbabilityArtifactTests -v
```

Expected: PASS.

- [ ] **Step 5: Write a failing subprocess command test**

Use a mocked `subprocess.run` only at the external-process boundary. Assert the command is an argument list containing `-m training.predict_vector`, `--device cpu`, configured checkpoint/input/output paths, `shell` is not passed as true, and timeout is honored.

- [ ] **Step 6: Implement configuration and runner**

```python
@dataclass(frozen=True)
class VectorModelConfig:
    training_root: Path
    python_executable: Path
    checkpoint_path: Path
    device: str = "cpu"
    timeout_seconds: float = 600.0

    @classmethod
    def from_environment(cls) -> "VectorModelConfig | None":
        training_root = os.environ.get("VECTOR_TRAINING_ROOT", "").strip()
        python_executable = os.environ.get("VECTOR_PYTHON", "").strip()
        checkpoint_path = os.environ.get("VECTOR_MODEL_PATH", "").strip()
        if not all((training_root, python_executable, checkpoint_path)):
            return None
        return cls(
            training_root=Path(training_root),
            python_executable=Path(python_executable),
            checkpoint_path=Path(checkpoint_path),
            device=os.environ.get("VECTOR_DEVICE", "cpu").strip(),
            timeout_seconds=float(os.environ.get("VECTOR_TIMEOUT_SECONDS", "600")),
        )
```

Read `VECTOR_TRAINING_ROOT`, `VECTOR_PYTHON`, `VECTOR_MODEL_PATH`, `VECTOR_DEVICE`, and `VECTOR_TIMEOUT_SECONDS`. Only allow `cpu` or `cuda`; this feature's default configuration uses `cpu`. Run the predictor in `training_root`, require exit code zero, and load the strict probability artifacts.

- [ ] **Step 7: Run Task 2 tests and commit**

```powershell
python -m unittest tests.test_vector_pdf_model -v
git add -- vector_pdf_model.py tests/test_vector_pdf_model.py
git commit -m "feat: load vector model probability evidence"
```

---

### Task 3: Native-Line Evidence Fusion

**Files:**
- Create: `vector_pdf_fusion.py`
- Create: `tests/test_vector_pdf_fusion.py`

**Interfaces:**
- Consumes native page dictionaries from Task 1 and aligned probability arrays from Task 2.
- Produces `FusionThresholds`, `build_line_candidates(page_data, thresholds) -> list[dict]`, and `fuse_line_candidates(candidates, probabilities, image_size, building_roi, thresholds) -> list[dict]`.
- Candidate decisions are exactly `accepted_wall_candidate`, `rejected_nonstructural`, or `uncertain`.

- [ ] **Step 1: Write failing native-candidate classification tests**

Construct page data with a dark line, confirmed dimension member, page-spanning border, ROI-exterior line, gray line, and diagonal record. Assert stable IDs and hard exclusions:

```python
candidates = build_line_candidates(page_data, FusionThresholds())
by_source = {item["source_native_id"]: item for item in candidates}
self.assertEqual(by_source["dimension-1"]["decision"], "rejected_nonstructural")
self.assertIn("dimension_overlap", by_source["dimension-1"]["reason_codes"])
self.assertEqual(by_source["border-1"]["decision"], "rejected_nonstructural")
self.assertNotIn("diagonal-1", by_source)
```

- [ ] **Step 2: Run the focused test and verify RED**

```powershell
python -m unittest tests.test_vector_pdf_fusion.NativeLineCandidateTests -v
```

Expected: missing module.

- [ ] **Step 3: Implement evidence-preserving candidates**

Implement immutable threshold validation and pure geometry helpers. Convert PDF points to analysis pixels using independent X/Y scales. Compute collinear overlap ratios for dimension members and page-border/ROI length ratios analytically rather than by rasterizing the whole page.

- [ ] **Step 4: Verify native candidate GREEN**

```powershell
python -m unittest tests.test_vector_pdf_fusion.NativeLineCandidateTests -v
```

Expected: PASS.

- [ ] **Step 5: Write failing horizontal/vertical probability sampling tests**

Use a synthetic `100 x 100` probability array with wall support on a line, room support on one/both sides, and footprint support around it:

```python
result = fuse_line_candidates(candidates, probabilities, (100, 100), [5, 5, 95, 95], thresholds)
self.assertEqual(result[0]["decision"], "accepted_wall_candidate")
self.assertGreaterEqual(result[0]["model_evidence"]["wall_p90"], 0.55)
self.assertIn("model_wall_supported", result[0]["reason_codes"])
```

Add cases for strong native/weak model (`uncertain`), strong model/gray native (`uncertain`), and a hard-rejected dimension remaining rejected even under high wall probability.

- [ ] **Step 6: Implement narrow-band sampling and three-state decision**

Use OpenCV line masks in candidate-local bounding boxes. The center/side radius is `clamp(round(min(width, height) * 0.003), 2, 12)`. Sample channels by fixed index from `CHANNEL_NAMES`; record mean, p90, and side means as finite Python floats. Preserve hard exclusions before applying model evidence.

- [ ] **Step 7: Run Task 3 tests and commit**

```powershell
python -m unittest tests.test_vector_pdf_fusion -v
git add -- vector_pdf_fusion.py tests/test_vector_pdf_fusion.py
git commit -m "feat: fuse native PDF lines with model evidence"
```

---

### Task 4: Orthogonal Wall Graph and Closed Room Candidates

**Files:**
- Create: `vector_pdf_rooms.py`
- Create: `tests/test_vector_pdf_rooms.py`

**Interfaces:**
- Consumes only line records whose decision is `accepted_wall_candidate` plus aligned probabilities.
- Produces `RoomClosureThresholds` and `find_room_candidates(accepted_lines, probabilities, image_size, building_roi, thresholds) -> dict` with keys `merged_lines`, `snaps`, `room_candidates`, and `summary`.

- [ ] **Step 1: Write the failing exact-rectangle room test**

```python
result = find_room_candidates(
    accepted_rectangle_lines(), supported_probabilities(100, 100),
    (100, 100), [0, 0, 100, 100], RoomClosureThresholds(),
)
self.assertEqual(result["summary"]["accepted_room_count"], 1)
room = result["room_candidates"][0]
self.assertEqual(room["decision"], "accepted_room_candidate")
self.assertFalse(room["closure_applied"])
self.assertEqual(set(room["source_wall_ids"]), {"wall-1", "wall-2", "wall-3", "wall-4"})
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest tests.test_vector_pdf_rooms.OrthogonalRoomTests.test_exact_rectangle_forms_traceable_room -v
```

Expected: missing module.

- [ ] **Step 3: Implement merge, intersection splitting, and bounded-face walk**

Normalize lines to integer analysis coordinates. Merge only same-orientation collinear intervals that overlap or touch. Split horizontal/vertical edges at every crossing. Build an undirected graph and two directed half-edges per graph edge; at each vertex sort outgoing edges by direction and walk the next clockwise half-edge. Canonicalize polygons by rotating to the lexicographically smallest vertex, discard zero/negative/exterior faces, and retain the source wall ID union for every boundary edge.

- [ ] **Step 4: Verify exact closure GREEN**

```powershell
python -m unittest tests.test_vector_pdf_rooms.OrthogonalRoomTests.test_exact_rectangle_forms_traceable_room -v
```

Expected: PASS.

- [ ] **Step 5: Write failing small-gap and over-limit tests**

Create one rectangle with a `5 px` endpoint gap on a `1000 x 1000` image (snap threshold `3 px`) and one with `2 px`; assert only the latter closes. Assert each successful snap records source wall IDs, original endpoints, snapped point, and distance.

- [ ] **Step 6: Implement conservative orthogonal endpoint snapping**

Compute `snap_px = clamp(round(min(width, height) * 0.003), 2, 8)`. Only snap endpoints that can meet by extending an existing horizontal or vertical segment without creating a diagonal. Reject ambiguous equal-distance targets instead of choosing by iteration order.

- [ ] **Step 7: Write failing room probability and false-rectangle tests**

Assert a geometrically closed face with low `room_interior` becomes `suspicious_closed_region`; assert lines not marked accepted never enter the graph; assert adjacent rooms produce two minimal faces rather than one outer union.

- [ ] **Step 8: Implement room evidence filtering and run Task 4 tests**

Rasterize each face locally, calculate area and mean `room_interior`/`footprint_interior`, apply the exact spec thresholds, and save `closure_applied` when any boundary edge depends on a snap.

```powershell
python -m unittest tests.test_vector_pdf_rooms -v
```

Expected: PASS.

- [ ] **Step 9: Commit Task 4**

```powershell
git add -- vector_pdf_rooms.py tests/test_vector_pdf_rooms.py
git commit -m "feat: find closed orthogonal room candidates"
```

---

### Task 5: Independent Fusion Pipeline and Artifacts

**Files:**
- Create: `vector_pdf_fusion_pipeline.py`
- Create: `tests/test_vector_pdf_fusion_pipeline.py`

**Interfaces:**
- Consumes Tasks 1–4.
- Produces `analyze_vector_pdf_page(pdf_path, page_number, output_dir, model_config, crop_bbox_page_px=None, model_runner=run_vector_probabilities) -> dict`.
- Publishes `pdf_native_candidates.json`, `pdf_vector_fusion.json`, and `pdf_vector_fusion_overlay.png`; references and hashes the model NPZ artifact.

- [ ] **Step 1: Write a failing evaluable-pipeline test**

Use a ReportLab vector PDF and inject a real in-process fake runner that returns a valid `VectorProbabilityResult` containing synthetic aligned probabilities. Assert:

```python
result = analyze_vector_pdf_page(pdf, 1, output, config, model_runner=fake_runner)
self.assertEqual(result["status"], "evaluable")
self.assertFalse(result["load_geometry_ready"])
self.assertTrue((output / "pdf_native_candidates.json").is_file())
self.assertTrue((output / "pdf_vector_fusion.json").is_file())
self.assertTrue((output / "pdf_vector_fusion_overlay.png").is_file())
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest tests.test_vector_pdf_fusion_pipeline -v
```

Expected: missing module.

- [ ] **Step 3: Implement rendering, cleanup, orchestration, and atomic publication**

Render one selected page at `100 dpi` with `pdf2image`, require exactly one image, build and remove the new module's dimension mask, optionally crop image and page data, write the model input, call the probability runner, fuse lines, find rooms, and render the documented color overlay. JSON writes use `allow_nan=False`, UTF-8, a temporary sibling, `flush/fsync`, then `os.replace`.

- [ ] **Step 4: Verify evaluable pipeline GREEN**

```powershell
python -m unittest tests.test_vector_pdf_fusion_pipeline.VectorPdfFusionPipelineTests.test_publishes_evaluable_debug_artifacts -v
```

Expected: PASS.

- [ ] **Step 5: Add failing partial/rejected/crop tests**

Cover:

```python
self.assertEqual(model_failure_result["status"], "partial")
self.assertEqual(model_failure_result["reason_codes"], ["model_unavailable"])
self.assertTrue((output / "pdf_native_candidates.json").is_file())
self.assertTrue((output / "pdf_vector_fusion.json").is_file())
self.assertEqual(
    json.loads((output / "pdf_vector_fusion.json").read_text(encoding="utf-8"))["status"],
    "partial",
)
self.assertEqual(scan_result["status"], "rejected")
self.assertIn("not_vector_pdf", scan_result["reason_codes"])
self.assertEqual(cropped_result["page"]["crop_bbox_page_px"], [100, 50, 300, 250])
```

For partial model failure, publish `pdf_vector_fusion.json` with native candidates, empty model evidence, `status: partial`, and the exact failure reason. It remains a diagnostic payload and keeps `load_geometry_ready: false`.

- [ ] **Step 6: Implement failure states and crop mapping**

Catch only expected model availability, timeout, and contract exceptions and map them to stable reason codes. Let filesystem/programming errors surface as server errors. Reject raster-only PDFs before invoking the model.

- [ ] **Step 7: Run Task 5 tests and commit**

```powershell
python -m unittest tests.test_vector_pdf_fusion_pipeline -v
git add -- vector_pdf_fusion_pipeline.py tests/test_vector_pdf_fusion_pipeline.py
git commit -m "feat: publish native PDF fusion debug artifacts"
```

---

### Task 6: Dedicated Flask Route and UI Selection

**Files:**
- Modify: `web_server_server.py`
- Modify: `templates/energy.html`
- Modify: `tests/test_energy_template.py`
- Create: `tests/test_vector_pdf_fusion_route.py`

**Interfaces:**
- Produces authenticated `POST /energy/vector_pdf_fusion`.
- Consumes `report_number`, `pdf_upload_token`, `pdf_page_number`, `recognition_mode`, `crop_bbox_px`, and `crop_preview_size` using existing generic validators.
- The vector-PDF UI sends PDF + `vector_pytorch` requests to the new route; legacy ONNX requests remain on `/energy/ai_recognize`.

- [ ] **Step 1: Write failing route isolation tests**

Prepare a valid PDF token and patch only `analyze_vector_pdf_page`. Assert the route validates report/page/crop, calls the new pipeline, returns artifact URLs and `load_geometry_ready: false`, and does not create `recognition.json`:

```python
response = self.client.post("/energy/vector_pdf_fusion", data={
    "report_number": "FUSION-1", "pdf_upload_token": token,
    "pdf_page_number": "1", "recognition_mode": "full_page",
})
self.assertEqual(response.status_code, 200)
self.assertFalse(response.get_json()["load_geometry_ready"])
self.assertFalse((report_dir / "recognition.json").exists())
```

Also patch `_floorplan_segmenter.predict` to raise if called, proving isolation.

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest tests.test_vector_pdf_fusion_route -v
```

Expected: route returns 404.

- [ ] **Step 3: Implement thin route and startup configuration**

Import the new pipeline/config in its own guarded block. Resolve the prepared PDF with `_validated_prepared_pdf`, map preview crop coordinates with the generic `map_crop_bbox_to_page`, create a dedicated `vector_pdf_fusion` artifact subdirectory under the report directory, invoke the pipeline, and serialize only debug results. Return 501 when the new model configuration is absent; keep old AI availability independent.

- [ ] **Step 4: Verify route GREEN**

```powershell
python -m unittest tests.test_vector_pdf_fusion_route -v
```

Expected: PASS.

- [ ] **Step 5: Write failing template routing tests**

Assert the model option label says it supports vector PDF and that the PDF recognition code selects the new endpoint only for `vector_pytorch`:

```python
self.assertIn("新矢量模型（矢量 PDF/图片试验版）", html)
self.assertIn("'/energy/vector_pdf_fusion'", html)
self.assertIn("vectorBackend && preparedPdf", html)
```

- [ ] **Step 6: Implement minimal UI routing and result rendering**

Keep PNG/JPG new-model requests on `/energy/ai_recognize`. For a prepared PDF with the vector backend, post the existing PDF token/page/crop fields to `/energy/vector_pdf_fusion`. Show the fusion overlay and counts for accepted/rejected/uncertain lines and accepted/suspicious rooms. Display an explicit “调试结果，尚不可用于能耗计算” message whenever `load_geometry_ready` is false.

- [ ] **Step 7: Run focused and legacy route/template tests**

```powershell
python -m unittest tests.test_vector_pdf_fusion_route tests.test_energy_template -v
```

Expected: PASS, including existing legacy PDF/ONNX cases.

- [ ] **Step 8: Commit Task 6**

```powershell
git add -- web_server_server.py templates/energy.html tests/test_energy_template.py tests/test_vector_pdf_fusion_route.py
git commit -m "feat: expose native PDF fusion diagnostics"
```

---

### Task 7: Full Regression and Real CPU Smoke Verification

**Files:**
- Create: `tools/smoke_vector_pdf_model.py`
- Create: `tests/test_vector_pdf_model_smoke_cli.py`
- Modify: `docs/handoff/2026-08-25-新模型与PDF原生矢量解析交接.md` only if that file is tracked on the implementation branch; otherwise create `docs/handoff/2026-08-25-PDF原生融合实施结果.md` without staging other handoff files.

**Interfaces:**
- Produces an opt-in smoke CLI that creates a temporary synthetic image, runs the real checkpoint on CPU, validates ten-channel artifacts, prints a compact JSON summary, and deletes only its owned temporary directory.

- [ ] **Step 1: Write a failing smoke CLI argument/unit test**

Test the CLI with an injected runner so the default suite does not load PyTorch:

```python
result = main([
    "--training-root", str(training_root), "--python", str(python_path),
    "--checkpoint", str(checkpoint), "--device", "cpu",
], runner=fake_runner)
self.assertEqual(result, 0)
```

Assert invalid paths return a nonzero status and a concise diagnostic.

- [ ] **Step 2: Run and verify RED**

```powershell
python -m unittest tests.test_vector_pdf_model_smoke_cli -v
```

Expected: missing smoke module.

- [ ] **Step 3: Implement the opt-in smoke CLI**

Generate a `1024 x 1024` white RGB image with a simple black orthogonal room in a `TemporaryDirectory`, configure `VectorModelConfig`, call `run_vector_probabilities`, and print JSON containing shape, channel names, NPZ hash, runtime metadata, and `device`. Do not persist the synthetic input or model output after the process exits.

- [ ] **Step 4: Run the smoke CLI unit test**

```powershell
python -m unittest tests.test_vector_pdf_model_smoke_cli -v
```

Expected: PASS without loading the real model.

- [ ] **Step 5: Run the complete automated test suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests pass with zero failures and zero errors.

- [ ] **Step 6: Run compile and isolation checks**

```powershell
python -m py_compile vector_pdf_native.py vector_pdf_model.py vector_pdf_fusion.py vector_pdf_rooms.py vector_pdf_fusion_pipeline.py web_server_server.py tools/smoke_vector_pdf_model.py
rg -n "from (floorplan_onnx|floorplan_page_pipeline|floorplan_rooms|floorplan_topology_repair|vector_pdf_scale)|import (floorplan_onnx|floorplan_page_pipeline|floorplan_rooms|floorplan_topology_repair|vector_pdf_scale)" vector_pdf_native.py vector_pdf_model.py vector_pdf_fusion.py vector_pdf_rooms.py vector_pdf_fusion_pipeline.py
```

Expected: compilation succeeds; `rg` prints no matches and returns exit code 1.

- [ ] **Step 7: Run the real best.pt CPU smoke**

```powershell
& 'G:\bim-annotation-training\.worktrees\vector-annotation-v1\.venv-train\Scripts\python.exe' tools\smoke_vector_pdf_model.py --training-root 'G:\bim-annotation-training\.worktrees\vector-annotation-v1' --python 'G:\bim-annotation-training\.worktrees\vector-annotation-v1\.venv-train\Scripts\python.exe' --checkpoint 'C:\Users\majin\bim-annotation-training-data\best.pt' --device cpu
```

Expected: exit code 0 and JSON showing shape `[10, 1024, 1024]`, exact channel names, a SHA-256 hash, and finite runtime. If the environment is incomplete, record the exact diagnostic; do not claim the real smoke passed.

- [ ] **Step 8: Update the handoff result**

Record branch, commits, test count, smoke result, artifact names, environment variables, known `uncalibrated` status, and the next requirement for 5–10 real vector PDFs. Do not copy credentials, private server addresses, or model weights.

- [ ] **Step 9: Commit Task 7**

Stage only the smoke script, its test, and the exact handoff result file:

```powershell
git add -- tools/smoke_vector_pdf_model.py tests/test_vector_pdf_model_smoke_cli.py docs/handoff/2026-08-25-PDF原生融合实施结果.md
git commit -m "test: verify native PDF fusion workflow"
```

If the existing tracked handoff file was updated instead, substitute its exact path in the `git add` command and do not add the untracked `docs/handoff` directory wholesale.

---

## Final Verification Checklist

- [ ] Every new production function has a test that was observed failing before implementation.
- [ ] Native PDF extraction ignores diagonals and identifies dimension/page-border evidence.
- [ ] Probability loader validates exact channels, coordinates, range, finiteness, and hash.
- [ ] The model only scores PDF-native candidates and cannot add new geometry.
- [ ] Small orthogonal gaps close only within the recorded threshold.
- [ ] Every room candidate traces back to source wall IDs.
- [ ] Every route and JSON result has `load_geometry_ready: false`.
- [ ] New modules contain no imports from the five prohibited legacy modules.
- [ ] Old ONNX route and tests remain unchanged in behavior.
- [ ] Full automated tests pass.
- [ ] Real `best.pt` CPU smoke either passes with evidence or is reported as not verified with the exact blocker.
