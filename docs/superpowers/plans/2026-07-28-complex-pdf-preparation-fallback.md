# Complex PDF Preparation Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PDF page preparation finish reliably by terminating vector analysis after 15 seconds, falling back to raster preparation, rejecting duplicate page jobs, and showing an accurate busy state in the annotation UI.

**Architecture:** Keep `prepare_pdf_page` as the shared website/annotation entry point. Run only the potentially unbounded vector extraction in a spawned child process with a hard timeout; the parent then continues either the existing vector-enhanced path or a 100 DPI Poppler raster fallback. Add an in-process per-page guard in `AnnotationService` and a small front-end preparing-state controller.

**Tech Stack:** Python 3.12, Flask, `multiprocessing`, `pdfplumber`, `pdf2image`/Poppler, OpenCV, vanilla JavaScript, Python `unittest`, Playwright CLI.

## Global Constraints

- Vector extraction timeout defaults to exactly 15 seconds.
- A timed-out worker process must be terminated and joined; no background parser may survive the request.
- Raster fallback renders only the selected page at 100 DPI.
- Raster fallback still produces render, cleaned, model-view, 512-preview, and metadata artifacts.
- Raster fallback must not claim successful scale calibration, vector cleanup, structural support, or vector ROI detection.
- The four training classes remain `0=background`, `1=wall`, `2=window`, and `3=door`.
- Website recognition and annotation continue to call the same `prepare_pdf_page` entry point.
- Do not delete or rewrite existing PDFs, projects, annotations, exports, or the duplicate projects already created.
- Do not add a task queue, database, new PDF parser, or percentage progress bar.

---

### Task 1: Bound vector analysis and generate raster fallback artifacts

**Files:**
- Modify: `floorplan_page_pipeline.py`
- Modify: `tests/test_floorplan_page_pipeline.py`

**Interfaces:**
- Consumes: existing `extract_vector_page(pdf_path, page_index, dpi) -> dict`.
- Produces: `VectorAnalysisOutcome(page_data: dict | None, evidence: dict)`.
- Produces: `_run_vector_analysis(source: Path, page_index: int, dpi: int, timeout_seconds: float, *, worker_target=None) -> VectorAnalysisOutcome`.
- Extends: `prepare_pdf_page(..., vector_timeout_seconds: float = 15.0) -> PreparedFloorplanPage`.
- Extends: `PreparedFloorplanPage.vector_analysis: dict[str, Any]`.

- [ ] **Step 1: Read the good-test rules before adding tests**

Read completely:

```powershell
Get-Content -Raw "C:\Users\majin\.codex\plugins\cache\openai-curated-remote\superpowers\6.2.0\skills\test-driven-development\writing-good-tests.md"
```

- [ ] **Step 2: Add a failing hard-timeout test**

Add module-level spawned-worker fixtures and a test to
`tests/test_floorplan_page_pipeline.py`:

```python
def _slow_vector_worker(result_path, source, page_index, dpi):
    import time
    time.sleep(30)


class VectorAnalysisTimeoutTests(unittest.TestCase):
    def test_vector_worker_is_terminated_after_timeout(self):
        source = Path(__file__)
        started = time.perf_counter()

        outcome = _run_vector_analysis(
            source,
            page_index=0,
            dpi=100,
            timeout_seconds=0.1,
            worker_target=_slow_vector_worker,
        )

        self.assertLess(time.perf_counter() - started, 5)
        self.assertIsNone(outcome.page_data)
        self.assertEqual(outcome.evidence["status"], "timed_out")
        self.assertEqual(outcome.evidence["mode"], "raster_fallback")
        self.assertEqual(outcome.evidence["timeout_seconds"], 0.1)
```

Import `time` and `_run_vector_analysis`. Keep the worker at module scope so it
is picklable under Windows `spawn`.

- [ ] **Step 3: Run the timeout test and verify RED**

Run:

```powershell
& "G:\bim-web\.venv\Scripts\python.exe" -m unittest tests.test_floorplan_page_pipeline.VectorAnalysisTimeoutTests.test_vector_worker_is_terminated_after_timeout
```

Expected: `ImportError` because `_run_vector_analysis` does not exist.

- [ ] **Step 4: Implement the spawned worker boundary**

In `floorplan_page_pipeline.py`, add:

```python
from dataclasses import dataclass
import multiprocessing
import tempfile

DEFAULT_VECTOR_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class VectorAnalysisOutcome:
    page_data: dict[str, Any] | None
    evidence: dict[str, Any]


def _vector_analysis_worker(
    result_path: str,
    source: str,
    page_index: int,
    dpi: int,
) -> None:
    try:
        page_data = extract_vector_page(source, page_index=page_index, dpi=dpi)
        payload = {"ok": True, "page_data": page_data}
    except BaseException as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    Path(result_path).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _run_vector_analysis(
    source: Path,
    page_index: int,
    dpi: int,
    timeout_seconds: float,
    *,
    worker_target=None,
) -> VectorAnalysisOutcome:
    if timeout_seconds <= 0:
        raise ValueError("vector_timeout_seconds must be positive")
    target = worker_target or _vector_analysis_worker
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="floorplan-vector-") as directory:
        result_path = Path(directory) / "result.json"
        process = context.Process(
            target=target,
            args=(str(result_path), str(source), page_index, dpi),
            daemon=True,
        )
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            return VectorAnalysisOutcome(
                None,
                {
                    "status": "timed_out",
                    "mode": "raster_fallback",
                    "timeout_seconds": timeout_seconds,
                    "reason": (
                        f"vector extraction exceeded {timeout_seconds:g} seconds"
                    ),
                },
            )
        if not result_path.is_file():
            return VectorAnalysisOutcome(
                None,
                {
                    "status": "failed",
                    "mode": "raster_fallback",
                    "timeout_seconds": timeout_seconds,
                    "reason": (
                        "vector extraction worker exited without a result "
                        f"(exit code {process.exitcode})"
                    ),
                },
            )
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            return VectorAnalysisOutcome(
                None,
                {
                    "status": "failed",
                    "mode": "raster_fallback",
                    "timeout_seconds": timeout_seconds,
                    "reason": f"invalid vector worker result: {exc}",
                },
            )
    if not payload["ok"]:
        return VectorAnalysisOutcome(
            None,
            {
                "status": "failed",
                "mode": "raster_fallback",
                "timeout_seconds": timeout_seconds,
                "reason": payload["error"],
            },
        )
    return VectorAnalysisOutcome(
        payload["page_data"],
        {
            "status": "completed",
            "mode": "vector",
            "timeout_seconds": timeout_seconds,
            "reason": None,
        },
    )
```

The child writes to a temporary JSON handoff instead of a pipe or queue. This
keeps result serialization inside the same 15-second bound and avoids a worker
blocking on a full IPC buffer. Do not use a thread or `ProcessPoolExecutor`;
neither provides the required hard termination guarantee for the active parse.

- [ ] **Step 5: Run the timeout test and verify GREEN**

Run the Step 3 command.

Expected: one test passes in under five seconds and no child Python process
remains.

- [ ] **Step 6: Add failing fallback and success-path tests**

Add two tests using a fake `FloorplanSegmenterONNX` instance:

```python
@patch("floorplan_page_pipeline._run_vector_analysis")
def test_prepare_pdf_page_writes_raster_fallback_metadata(self, run_analysis):
    run_analysis.return_value = VectorAnalysisOutcome(
        None,
        {
            "status": "timed_out",
            "mode": "raster_fallback",
            "timeout_seconds": 15,
            "reason": "vector extraction exceeded 15 seconds",
        },
    )
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "artifacts"
        prepared = prepare_pdf_page(
            self._make_two_page_pdf(directory),
            1,
            poppler_path=os.environ.get("POPPLER_PATH"),
            segmenter=self._segmenter(),
            output_dir=output,
        )
        metadata = json.loads(
            (output / "preprocessing.json").read_text(encoding="utf-8")
        )

    self.assertEqual(prepared.vector_analysis["mode"], "raster_fallback")
    self.assertEqual(metadata["render_dpi"], 100)
    self.assertEqual(metadata["vector_analysis"]["status"], "timed_out")
    self.assertEqual(prepared.scale_calibration["status"], "manual_required")
    self.assertFalse(prepared.vector_cleanup["enabled"])
    self.assertIsNone(prepared.inference_roi)
    self.assertIsNone(prepared.structural_support_mask)
    self.assertEqual(
        set(prepared.artifacts),
        {"render", "cleaned", "model_view", "model_input_512", "metadata"},
    )


@patch("floorplan_page_pipeline._run_vector_analysis")
def test_prepare_pdf_page_keeps_completed_vector_analysis(self, run_analysis):
    run_analysis.return_value = VectorAnalysisOutcome(
        {
            "page_size_pt": [400.0, 300.0],
            "render_size_px": [556, 417],
            "dpi": 100,
            "text_spans": [],
            "segments": [],
            "styled_edges": [],
            "vector_text_count": 0,
            "vector_segment_count": 0,
            "styled_edge_count": 0,
            "has_vector_text": False,
            "has_vector_geometry": False,
            "is_vector_pdf": False,
        },
        {
            "status": "completed",
            "mode": "vector",
            "timeout_seconds": 15,
            "reason": None,
        },
    )
    with tempfile.TemporaryDirectory() as directory:
        prepared = prepare_pdf_page(
            self._make_two_page_pdf(directory),
            1,
            poppler_path=os.environ.get("POPPLER_PATH"),
            segmenter=self._segmenter(),
        )
    self.assertEqual(prepared.vector_analysis["status"], "completed")
```

Factor the existing segmenter construction into `_segmenter()` in the test
class. Import `json`, `patch`, and `VectorAnalysisOutcome`.

- [ ] **Step 7: Run the new pipeline tests and verify RED**

Run:

```powershell
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
& "G:\bim-web\.venv\Scripts\python.exe" -m unittest tests.test_floorplan_page_pipeline
```

Expected: failures because `PreparedFloorplanPage` and metadata do not yet
contain `vector_analysis`, and fallback still dereferences missing page data.

- [ ] **Step 8: Integrate the outcome into `prepare_pdf_page`**

Change the entry point to:

```python
def prepare_pdf_page(
    pdf_path,
    page_number,
    *,
    poppler_path,
    segmenter,
    output_dir=None,
    vector_timeout_seconds=DEFAULT_VECTOR_TIMEOUT_SECONDS,
) -> PreparedFloorplanPage:
```

Call `_run_vector_analysis`. When `page_data is None`, render at 100 DPI and
construct a conservative page-data dictionary from the rendered image:

```python
page_data = {
    "page_size_pt": [
        render_bgr.shape[1] * 72.0 / render_dpi,
        render_bgr.shape[0] * 72.0 / render_dpi,
    ],
    "render_size_px": [render_bgr.shape[1], render_bgr.shape[0]],
    "dpi": render_dpi,
    "text_spans": [],
    "segments": [],
    "styled_edges": [],
    "vector_text_count": 0,
    "vector_segment_count": 0,
    "styled_edge_count": 0,
    "has_vector_text": False,
    "has_vector_geometry": False,
    "is_vector_pdf": False,
}
```

Keep the existing vector logic unchanged when data is present. Add
`vector_analysis` to `PreparedFloorplanPage` and `preprocessing.json`.

- [ ] **Step 9: Run pipeline tests and verify GREEN**

Run the Step 7 command.

Expected: all pipeline tests pass.

- [ ] **Step 10: Commit Task 1**

```powershell
git add floorplan_page_pipeline.py tests/test_floorplan_page_pipeline.py
git commit -m "fix: bound vector PDF page analysis"
```

---

### Task 2: Reject duplicate preparation and preserve fallback evidence

**Files:**
- Modify: `annotation_tool/services.py`
- Modify: `annotation_tool/app.py`
- Modify: `tests/test_annotation_app.py`

**Interfaces:**
- Consumes: `PreparedFloorplanPage.vector_analysis`.
- Produces: `PreparationInProgressError(project_id: str, page_number: int)`.
- Produces: HTTP 409 JSON `{ "error": "该页面正在准备，请等待当前任务完成" }`.
- Persists: `page.preparation.vector_analysis`.

- [ ] **Step 1: Add a failing concurrent API test**

Add imports for `threading` and `ThreadPoolExecutor`. First extract the existing
inline fake prepared-page builder into this reusable test helper:

```python
def _fake_prepared_page(self, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for key, filename in {
        "render": "page_render.png",
        "cleaned": "cleaned_page.png",
        "model_view": "model_view.png",
        "model_input_512": "model_input_512.png",
    }.items():
        self.assertTrue(
            cv2.imwrite(
                str(output_dir / filename),
                np.full((4, 4, 3), 255, np.uint8),
            )
        )
        artifacts[key] = {"path": filename, "sha256": key}
    (output_dir / "preprocessing.json").write_text("{}", encoding="utf-8")
    artifacts["metadata"] = {
        "path": "preprocessing.json",
        "sha256": "metadata",
    }
    return SimpleNamespace(
        page_count=2,
        render_bgr=np.full((4, 4, 3), 255, np.uint8),
        model_input_metadata={
            "original_size": [4, 4],
            "resized_size": [512, 512],
            "padding": [0, 0, 0, 0],
        },
        scale_calibration={"status": "manual_required"},
        vector_analysis={
            "status": "completed",
            "mode": "vector",
            "timeout_seconds": 15,
            "reason": None,
        },
        vector_cleanup={"enabled": False},
        inference_roi=None,
        artifacts=artifacts,
    )
```

Then add:

```python
def test_duplicate_page_preparation_returns_conflict(self):
    project = self._create_project()
    service = self.app.extensions["annotation_service"]
    started = threading.Event()
    release = threading.Event()

    def blocking_prepare(*args, **kwargs):
        started.set()
        self.assertTrue(release.wait(5))
        return self._fake_prepared_page(Path(kwargs["output_dir"]))

    with patch(
        "annotation_tool.services.prepare_pdf_page",
        side_effect=blocking_prepare,
    ):
        with ThreadPoolExecutor(max_workers=1) as pool:
            first = pool.submit(
                lambda: self.app.test_client().post(
                    f"/api/projects/{project['project_id']}/pages/prepare",
                    json={"page_number": 1},
                )
            )
            self.assertTrue(started.wait(5))
            second = self.app.test_client().post(
                f"/api/projects/{project['project_id']}/pages/prepare",
                json={"page_number": 1},
            )
            release.set()
            first_response = first.result(timeout=5)

    self.assertEqual(second.status_code, 409)
    self.assertIn("正在准备", second.get_json()["error"])
    self.assertEqual(first_response.status_code, 200)
```

Replace the old inline `fake_prepare` body in
`test_prepare_and_preannotate_use_the_shared_page_contract` with a call to this
helper so there is only one fake page contract to maintain.

- [ ] **Step 2: Run the duplicate test and verify RED**

Run:

```powershell
& "G:\bim-web\.venv\Scripts\python.exe" -m unittest tests.test_annotation_app.AnnotationApiTests.test_duplicate_page_preparation_returns_conflict
```

Expected: the second request also enters `prepare_pdf_page` instead of returning
409.

- [ ] **Step 3: Implement the page guard and 409 mapping**

In `annotation_tool/services.py`:

```python
import threading


class PreparationInProgressError(RuntimeError):
    def __init__(self, project_id: str, page_number: int):
        super().__init__("该页面正在准备，请等待当前任务完成")
        self.project_id = project_id
        self.page_number = page_number
```

Initialize:

```python
self._preparing_lock = threading.Lock()
self._preparing_pages: set[tuple[str, int]] = set()
```

After validation and before constructing the segmenter:

```python
key = (project_id, page_number)
with self._preparing_lock:
    if key in self._preparing_pages:
        raise PreparationInProgressError(project_id, page_number)
    self._preparing_pages.add(key)
try:
    return self._prepare_page_once(project_id, page_number, project)
finally:
    with self._preparing_lock:
        self._preparing_pages.discard(key)
```

Extract the current preparation body to `_prepare_page_once` so the `finally`
scope is obvious and testable. Copy `prepared.vector_analysis` into the stored
`preparation` dictionary.

In `annotation_tool/app.py`, import the exception and register:

```python
@app.errorhandler(PreparationInProgressError)
def preparation_conflict(error):
    return jsonify({"error": str(error)}), 409
```

Register this handler before the broader `ValueError` handler.

- [ ] **Step 4: Run the duplicate test and verify GREEN**

Run the Step 2 command.

Expected: first request succeeds, second returns 409, and the preparation
function is called once.

- [ ] **Step 5: Add and run a failing guard-release test**

Add:

```python
def test_failed_page_preparation_releases_the_guard(self):
    project = self._create_project()
    attempts = 0

    def fail_then_succeed(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("render failed")
        return self._fake_prepared_page(Path(kwargs["output_dir"]))

    with patch(
        "annotation_tool.services.prepare_pdf_page",
        side_effect=fail_then_succeed,
    ) as prepare:
        first = self.client.post(
            f"/api/projects/{project['project_id']}/pages/prepare",
            json={"page_number": 1},
        )
        second = self.client.post(
            f"/api/projects/{project['project_id']}/pages/prepare",
            json={"page_number": 1},
        )

    self.assertEqual(first.status_code, 400)
    self.assertEqual(second.status_code, 200)
    self.assertEqual(prepare.call_count, 2)
```

Expected before a correct `finally`: second response is 409. Expected after the
implementation: test passes.

- [ ] **Step 6: Run annotation API tests**

Run:

```powershell
& "G:\bim-web\.venv\Scripts\python.exe" -m unittest tests.test_annotation_app
```

Expected: all annotation API and template tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
git add annotation_tool/services.py annotation_tool/app.py tests/test_annotation_app.py
git commit -m "fix: serialize annotation page preparation"
```

---

### Task 3: Show and enforce the front-end preparing state

**Files:**
- Modify: `annotation_tool/static/app.js`
- Modify: `tests/test_annotation_app.py`

**Interfaces:**
- Consumes: prepare response `body.page.preparation.vector_analysis.mode`.
- Produces: `state.preparing: boolean`.
- Produces: `setPreparing(active: boolean)`.

- [ ] **Step 1: Add a failing front-end contract test**

Extend `AnnotationTemplateTests`:

```python
def test_prepare_button_has_a_single_request_busy_state(self):
    script = (
        Path(__file__).resolve().parents[1]
        / "annotation_tool/static/app.js"
    ).read_text(encoding="utf-8")

    self.assertIn("preparing: false", script)
    self.assertIn("function setPreparing(active)", script)
    self.assertIn('button.textContent = "正在准备，请勿重复点击…"', script)
    self.assertIn("if (state.preparing)", script)
    self.assertIn("setPreparing(true)", script)
    self.assertIn("setPreparing(false)", script)
    self.assertIn("raster_fallback", script)
```

- [ ] **Step 2: Run the front-end contract test and verify RED**

Run:

```powershell
& "G:\bim-web\.venv\Scripts\python.exe" -m unittest tests.test_annotation_app.AnnotationTemplateTests.test_prepare_button_has_a_single_request_busy_state
```

Expected: failure because the preparing state does not exist.

- [ ] **Step 3: Implement `setPreparing` and wrap the request**

Add `preparing: false` to state and:

```javascript
function setPreparing(active) {
  state.preparing = active;
  const button = $("#prepare-page");
  button.textContent = active
    ? "正在准备，请勿重复点击…"
    : "准备所选页面";
  $("#project-select").disabled = active;
  $("#page-select").disabled = active || !state.project;
  button.disabled = active || !state.project || !$("#page-select").value;
}
```

Update `selectProject` so it uses:

```javascript
$("#page-select").disabled = state.preparing;
$("#prepare-page").disabled = (
  state.preparing || !$("#page-select").value
);
```

Update the page selector change handler before calling `selectPage`:

```javascript
$("#page-select").addEventListener("change", (event) => {
  $("#prepare-page").disabled = state.preparing || !event.target.value;
  selectPage(event.target.value).catch(handleError);
});
```

Change the click handler to:

```javascript
$("#prepare-page").addEventListener("click", async () => {
  if (
    state.preparing
    || !state.project
    || !$("#page-select").value
  ) return;
  const projectId = state.project.project_id;
  const pageNumber = Number($("#page-select").value);
  setPreparing(true);
  try {
    setSaveState("页面准备中…", "saving");
    const body = await api(
      `/api/projects/${encodeURIComponent(projectId)}/pages/prepare`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ page_number: pageNumber }),
      }
    );
    const fallback = (
      body.page.preparation?.vector_analysis?.mode === "raster_fallback"
    );
    await selectProject(projectId);
    $("#page-select").value = String(pageNumber);
    await selectPage(pageNumber);
    if (fallback) {
      log(
        `第 ${pageNumber} 页准备完成`,
        "复杂 PDF 已使用栅格模式准备，可正常标注墙、窗、门。"
      );
    } else {
      log(`第 ${pageNumber} 页准备完成`);
    }
  } catch (error) {
    handleError(error);
  } finally {
    setPreparing(false);
  }
});
```

In preannotation cleanup, calculate the prepare-button state as
`state.preparing || !state.project || !$("#page-select").value`. Do not call
`setPreparing(false)` from unrelated flows and do not allow any function to
re-enable controls while a preparation request is active.

- [ ] **Step 4: Run front-end tests and syntax check**

Run:

```powershell
& "G:\bim-web\.venv\Scripts\python.exe" -m unittest tests.test_annotation_app.AnnotationTemplateTests
node --check annotation_tool/static/app.js
```

Expected: tests and JavaScript syntax check pass.

- [ ] **Step 5: Commit Task 3**

```powershell
git add annotation_tool/static/app.js tests/test_annotation_app.py
git commit -m "fix: show annotation page preparation state"
```

---

### Task 4: Document fallback behavior and verify the real complex PDF

**Files:**
- Modify: `annotation_tool/README.md`
- Modify: `README.md`
- Runtime only, do not commit: `G:\bim网页\标注数据`
- Runtime only, do not commit: `output/playwright/complex-pdf-fallback.png`

**Interfaces:**
- Consumes: the existing project `project-bce78876` and its 36-page source PDF.
- Produces: a prepared page manifest containing
  `vector_analysis.mode == "raster_fallback"`.

- [ ] **Step 1: Update operator documentation**

Document:

- normal PDFs keep vector enhancement;
- vector extraction has a 15-second limit;
- timeout automatically falls back to 100 DPI raster preparation;
- raster fallback is valid for wall/window/door annotation and training;
- scale must be handled later in the energy workflow;
- users must not repeatedly click while the page is preparing.

- [ ] **Step 2: Run focused and full automated verification**

Run:

```powershell
$env:ONNX_MODEL_PATH="G:\bim-web\models\M2_pub_plus_user.onnx"
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
& "G:\bim-web\.venv\Scripts\python.exe" -m unittest discover -s tests
node --check annotation_tool/static/app.js
git diff --check
```

Expected: all tests pass; only the existing training-environment skips remain.

- [ ] **Step 3: Start the annotation service in a hidden process**

Run with the same two environment variables:

```powershell
$process = Start-Process `
  -FilePath "G:\bim-web\.venv\Scripts\python.exe" `
  -ArgumentList "-m annotation_tool.app --host 127.0.0.1 --port 8099" `
  -WorkingDirectory "G:\bim-web" `
  -WindowStyle Hidden `
  -PassThru
```

Poll `http://127.0.0.1:8099/api/projects` until it returns HTTP 200.

- [ ] **Step 4: Verify the real PDF fallback through the browser**

Using Playwright CLI:

1. Open `http://127.0.0.1:8099` headed.
2. Select project `project-bce78876`.
3. Select page 1, which independently reproduced the long vector parse.
4. Click “准备所选页面” once.
5. Confirm the button and selectors disable immediately.
6. Confirm the page completes after approximately 15 seconds plus rendering
   time and the event log reports raster fallback.
7. Confirm the rendered page appears and wall/window/door tools remain enabled.
8. Confirm only one prepare request appears and no parser child process remains.
9. Save `output/playwright/complex-pdf-fallback.png`.
10. Inspect the browser console; only a pre-existing favicon 404 is acceptable.

- [ ] **Step 5: Inspect persisted evidence**

Read the selected page's `preprocessing.json` and `project.json`. Verify:

```text
vector_analysis.status = timed_out
vector_analysis.mode = raster_fallback
vector_analysis.timeout_seconds = 15
scale_calibration.status = manual_required
vector_cleanup.enabled = false
inference_roi = null
```

Do not confirm or export a mask during this smoke test.

- [ ] **Step 6: Stop the smoke-test service and browser**

Close only the Playwright browser session and the exact server process started
in Step 3. Confirm port 8099 is no longer listening.

- [ ] **Step 7: Commit documentation**

```powershell
git add README.md annotation_tool/README.md
git commit -m "docs: explain complex PDF raster fallback"
```

---

### Task 5: Final review and integration readiness

**Files:**
- Review all files changed by Tasks 1-4.

**Interfaces:**
- Consumes: all committed task outputs.
- Produces: a clean, reviewed feature branch ready for local merge or PR.

- [ ] **Step 1: Run final verification from a clean command invocation**

Run:

```powershell
$env:ONNX_MODEL_PATH="G:\bim-web\models\M2_pub_plus_user.onnx"
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
& "G:\bim-web\.venv\Scripts\python.exe" -m unittest discover -s tests
node --check annotation_tool/static/app.js
git diff --check
git status --short --branch
```

Expected: the full suite passes, JavaScript and diff checks pass, and the
feature worktree has no uncommitted files.

- [ ] **Step 2: Request code review**

Use `superpowers:requesting-code-review`. Require the reviewer to check:

- Windows spawned-process cleanup on success, timeout, child error, and broken
  pipe;
- no surviving vector parser after timeout;
- fallback metadata truthfulness;
- race-free per-page guard release;
- front-end controls never re-enable during an active request;
- website callers remain compatible with the extended dataclass and function
  signature.

- [ ] **Step 3: Address Critical and Important findings**

For each finding, use `superpowers:receiving-code-review` and
`superpowers:systematic-debugging`, add a failing regression test, implement the
minimal fix, and rerun the focused plus full suites.

- [ ] **Step 4: Use the branch-finishing workflow**

Invoke `superpowers:finishing-a-development-branch`, present the exact three
integration options, and wait for the user's choice.
