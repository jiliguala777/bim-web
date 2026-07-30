# PDF Vector Analysis 30-Second Timeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Raise the default PDF vector-analysis timeout from 15 seconds to 30 seconds without changing explicit timeout overrides or fallback behavior.

**Architecture:** Keep the existing subprocess boundary and timeout plumbing unchanged. Protect the user-visible default behavior with a focused preprocessing test, then change only the default constant.

**Tech Stack:** Python 3.12, `unittest`, OpenCV-based floor-plan preprocessing.

## Global Constraints

- The default timeout is exactly `30.0` seconds.
- Explicit `vector_timeout_seconds` values continue to override the default.
- Do not add retries or change vector extraction, scale calibration, or raster fallback logic.

---

### Task 1: Raise the default vector-analysis timeout

**Files:**
- Modify: `tests/test_floorplan_page_pipeline.py`
- Modify: `tests/test_energy_template.py`
- Modify: `floorplan_page_pipeline.py:33`

**Interfaces:**
- Consumes: `prepare_pdf_page(..., vector_timeout_seconds: float = DEFAULT_VECTOR_TIMEOUT_SECONDS)`
- Produces: the existing `PreparedFloorplanPage` result, with the default vector worker allowed up to 30 seconds.

- [x] **Step 1: Write the failing test**

Add a test beside the existing `prepare_pdf_page` vector-analysis tests. Its fake vector boundary returns a completed outcome only when the received timeout is at least 30 seconds, then the test asserts that default preprocessing completes vector analysis:

```python
@patch("floorplan_page_pipeline._run_vector_analysis")
def test_prepare_pdf_page_allows_30_seconds_for_vector_analysis_by_default(
    self, run_analysis
):
    def complete_only_with_30_seconds(source, page_index, dpi, timeout_seconds):
        if timeout_seconds < 30.0:
            return VectorAnalysisOutcome(
                None,
                {
                    "status": "timed_out",
                    "mode": "raster_fallback",
                    "timeout_seconds": timeout_seconds,
                    "reason": "vector extraction timed out",
                },
            )
        return VectorAnalysisOutcome(
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
                "timeout_seconds": timeout_seconds,
                "reason": None,
            },
        )

    run_analysis.side_effect = complete_only_with_30_seconds
    with tempfile.TemporaryDirectory() as directory:
        prepared = prepare_pdf_page(
            self._make_two_page_pdf(directory),
            1,
            poppler_path=os.environ.get("POPPLER_PATH"),
            segmenter=self._segmenter(),
        )
    self.assertEqual(prepared.vector_analysis["status"], "completed")
```

- [x] **Step 2: Run the new test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_page_pipeline.PdfPagePipelineTests.test_prepare_pdf_page_allows_30_seconds_for_vector_analysis_by_default
```

Expected: failure because the current default is `15.0`.

- [x] **Step 3: Implement the minimal production change**

Change the constant in `floorplan_page_pipeline.py`:

```python
DEFAULT_VECTOR_TIMEOUT_SECONDS = 30.0
```

Update the two route-level expectations in `tests/test_energy_template.py` from
`15.0` to `30.0` so they continue to verify the configured default crossing
the HTTP recognition boundary.

- [x] **Step 4: Run focused and complete verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_page_pipeline
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected: both commands exit successfully with zero failures.

- [x] **Step 5: Review and commit**

Review:

```powershell
git diff --check
git diff -- floorplan_page_pipeline.py tests/test_floorplan_page_pipeline.py tests/test_energy_template.py docs/superpowers/plans/2026-07-29-vector-analysis-timeout-30s.md
```

Commit only the implementation, test, and plan:

```powershell
git add -- floorplan_page_pipeline.py tests/test_floorplan_page_pipeline.py tests/test_energy_template.py docs/superpowers/plans/2026-07-29-vector-analysis-timeout-30s.md
git commit -m "fix: allow 30 seconds for vector analysis"
```
