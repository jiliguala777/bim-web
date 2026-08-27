# PDF 门洞桥接与外轮廓连续性 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让具备原生 PDF 门洞或窗洞证据的外轮廓缺口能够桥接后续墙体，同时避免把低置信度开口计入门窗面积。

**Architecture:** `vector_pdf_native.py` 将保留原生曲线与短线组的像素坐标证据；`vector_pdf_openings.py` 在现有模型概率分类后融合这些证据并给出 `door`、`window` 或 `pending_opening`。`vector_pdf_exterior.py` 仅把可接受的开口类型作为拓扑桥接，综合叠加图以橙色标出待确认开口。

**Tech Stack:** Python 3.12、pdfplumber、NumPy、OpenCV、unittest。

**Spec:** `docs/superpowers/specs/2026-08-26-door-opening-bridge-design.md`

## Global Constraints

- 仅处理候选外轮廓上的轴对齐缺口。
- 外墙必须有 PDF 原生结构直线支撑；不得凭空生成长墙。
- `pending_opening` 可闭合外轮廓，但不得自动计入门窗面积扣除。
- 缺少两侧墙端点、原生开口证据或有效尺度的缺口保持 `unresolved`。
- 不修改或提交既有的 `tests/test_vector_pdf_exterior_smoke_cli.py` 和 `tools/smoke_vector_pdf_exterior.py`。

---

### Task 1: 公开原生曲线与开口短线证据

**Files:**
- Modify: `vector_pdf_native.py:87-116,261-335,390-470`
- Modify: `tests/test_vector_pdf_native.py`

**Interfaces:**
- Produces: 页面契约字段 `opening_curve_edges: list[dict]`，每项包含 `curve_id`、`bbox_pt`、`bbox_px`、`start_px`、`end_px`、`stroke_rgb`、`width_pt`。
- Produces: `opening_short_segments: list[dict]`，仅包含非尺寸标注、非页面边框、长度在 4–80 pt 的正交原生短线。
- Consumed by: `classify_exterior_openings(..., native_opening_evidence=...)`。

- [ ] **Step 1: Write the failing extraction tests**

```python
def test_extracts_curve_edges_and_short_segments_in_pixel_coordinates(self):
    from vector_pdf_native import extract_native_pdf_page
    page = extract_native_pdf_page(self._make_door_arc_pdf(directory), dpi=100)
    self.assertEqual(len(page["opening_curve_edges"]), 1)
    curve = page["opening_curve_edges"][0]
    self.assertEqual(curve["curve_id"], "curve-0001")
    self.assertTrue(all(isinstance(value, int) for value in curve["bbox_px"]))
    self.assertTrue(page["opening_short_segments"])
```

- [ ] **Step 2: Run the native extraction test to verify it fails**

Run: `python -m unittest tests.test_vector_pdf_native.NativePdfExtractionTests.test_extracts_curve_edges_and_short_segments_in_pixel_coordinates`

Expected: FAIL because `opening_curve_edges` is absent.

- [ ] **Step 3: Extract stable curve and short-segment records**

```python
def _opening_curve_record(item, page_size, render_size, index):
    return {
        "curve_id": f"curve-{index:04d}",
        "bbox_pt": [float(item["x0"]), float(item["top"]), float(item["x1"]), float(item["bottom"])],
        "bbox_px": _bbox_to_pixel(...),
        "start_px": _point_to_pixel([float(item["x0"]), float(item["top"])], page_size, render_size),
        "end_px": _point_to_pixel([float(item["x1"]), float(item["bottom"])], page_size, render_size),
        "stroke_rgb": _stroke_rgb(item.get("stroking_color")),
        "width_pt": float(item.get("linewidth") or 0.0),
    }
```

Read `page.curves` independently from `curve_edges`, exclude page-border-sized boxes, and use the existing `_point_to_pixel` conversion. Build short-segment records from the existing orthogonal records after dimensions and borders are known. Extend `crop_native_page_data` to crop both fields and retain `page_*_px` provenance.

- [ ] **Step 4: Run native extraction and crop tests**

Run: `python -m unittest tests.test_vector_pdf_native`

Expected: PASS.

- [ ] **Step 5: Commit the native evidence contract**

```bash
git add vector_pdf_native.py tests/test_vector_pdf_native.py
git commit -m "feat: extract native PDF opening evidence"
```

### Task 2: Classify native-backed pending and door openings

**Files:**
- Modify: `vector_pdf_openings.py:1-220`
- Modify: `tests/test_vector_pdf_openings.py`

**Interfaces:**
- Consumes: `classify_exterior_openings(gaps, probabilities, image_size, native_opening_evidence=None)`.
- Produces: `accepted_openings` containing only `door` and `window`; `pending_openings` containing `kind="pending_opening"`; all records include `native_evidence`.
- Consumed by: `vector_pdf_fusion_pipeline.py` and `vector_pdf_exterior.build_exterior_topology`.

- [ ] **Step 1: Write failing native-evidence classification tests**

```python
def test_native_door_arc_bridges_gap_without_model_support(self):
    result = classify_exterior_openings(
        [gap((60, 50), (100, 50))], empty_probabilities(), (160, 100),
        native_opening_evidence={"curve_edges": [{"bbox_px": [56, 48, 104, 92], "start_px": [60, 50], "end_px": [100, 92]}]},
    )
    self.assertEqual(result["accepted_openings"][0]["kind"], "door")
    self.assertIn("native_door_arc_supported", result["accepted_openings"][0]["reason_codes"])

def test_weak_native_opening_becomes_pending_not_area_deductible(self):
    result = classify_exterior_openings(
        [gap((60, 50), (100, 50))], empty_probabilities(), (160, 100),
        native_opening_evidence={"short_segments": [{"start_px": [64, 52], "end_px": [96, 52]}]},
    )
    self.assertEqual(result["accepted_openings"], [])
    self.assertEqual(result["pending_openings"][0]["kind"], "pending_opening")
```

- [ ] **Step 2: Run the opening tests to verify they fail**

Run: `python -m unittest tests.test_vector_pdf_openings`

Expected: FAIL because `native_opening_evidence` and `pending_openings` are unsupported.

- [ ] **Step 3: Implement evidence fusion with conservative geometry gates**

```python
def _native_gap_evidence(gap, native_opening_evidence, tolerance_px=12):
    # Return {"door_arc": bool, "window_short_lines": int, "curve_ids": [], "short_segment_ids": []}.
    # A curve must overlap the gap bounding box and touch a gap endpoint.
    # A short segment must be parallel to the gap and lie within tolerance.

def classify_exterior_openings(..., native_opening_evidence=None):
    # Preserve existing model-only door/window decisions.
    # Promote a qualifying curve to door when it overlaps the gap and touches an endpoint.
    # Emit pending_opening only for native evidence that is insufficient to select door/window.
```

Reject evidence outside the image, whose gap is not `accepted_gap`, or that does not touch the local gap envelope. Keep `pending_opening` out of `accepted_openings`.

- [ ] **Step 4: Run opening classification tests**

Run: `python -m unittest tests.test_vector_pdf_openings`

Expected: PASS.

- [ ] **Step 5: Commit the opening classification contract**

```bash
git add vector_pdf_openings.py tests/test_vector_pdf_openings.py
git commit -m "feat: classify native-backed exterior openings"
```

### Task 3: Bridge pending openings without deducting their area

**Files:**
- Modify: `vector_pdf_exterior.py:500-1045`
- Modify: `vector_pdf_fusion_pipeline.py:168-220,260-320,500-540`
- Modify: `tests/test_vector_pdf_exterior.py`
- Modify: `tests/test_vector_pdf_fusion_pipeline.py`

**Interfaces:**
- Consumes: `build_exterior_topology(exterior_walls, gaps, openings, pending_openings, probabilities, image_size, building_roi)`.
- Produces: bridge records where `bridge_type="pending_opening_bridge"`, `opening_id` is null, and `pending_opening_id` is retained.
- Produces: exterior topology `pending_opening_ids`; energy geometry continues to consume only accepted `door`/`window` records.

- [ ] **Step 1: Write failing topology tests**

```python
def test_pending_opening_closes_boundary_but_is_not_an_accepted_opening(self):
    topology = build_exterior_topology(
        rectangle_with_bottom_gap(), [topology_gap(40, 80, 60, 80)], [],
        [{"pending_opening_id": "pending-gap-0001", "kind": "pending_opening", "start_px": [40, 80], "end_px": [60, 80], "host_wall_ids": ["bottom-left", "bottom-right"]}],
        supported_footprint([[20, 20], [80, 20], [80, 80], [20, 80]]), (100, 100), [0, 0, 100, 100],
    )
    self.assertEqual(topology["status"], "review_required")
    self.assertEqual(topology["opening_ids"], [])
    self.assertEqual(topology["pending_opening_ids"], ["pending-gap-0001"])
```

- [ ] **Step 2: Run topology tests to verify they fail**

Run: `python -m unittest tests.test_vector_pdf_exterior`

Expected: FAIL because pending opening arguments and bridge type are unsupported.

- [ ] **Step 3: Extend topology matching and overlay rendering**

```python
def _pending_opening_matches_gap(pending, gap):
    return pending["kind"] == "pending_opening" and _canonical_endpoints(pending) == _canonical_endpoints(gap)

# In build_exterior_topology, accept exactly one matching pending record as:
# bridge_type="pending_opening_bridge", opening_id=None,
# pending_opening_id=<id>, reason_codes=["pending_opening_exact_gap_match"].
```

Add orange dashed rendering for `pending_opening_bridge`. Keep its `opening_id` absent so `build_exterior_energy_geometry` cannot deduct it from wall area. Add `pending_openings` and their IDs to published JSON artifacts and summary counts.

- [ ] **Step 4: Run exterior and pipeline tests**

Run: `python -m unittest tests.test_vector_pdf_exterior tests.test_vector_pdf_fusion_pipeline`

Expected: PASS.

- [ ] **Step 5: Commit topology and artifact changes**

```bash
git add vector_pdf_exterior.py vector_pdf_fusion_pipeline.py tests/test_vector_pdf_exterior.py tests/test_vector_pdf_fusion_pipeline.py
git commit -m "feat: bridge pending PDF openings in exterior topology"
```

### Task 4: Validate the platform contract and the real floor plan

**Files:**
- Modify: `web_server_server.py:2358-2465`
- Modify: `templates/energy.html:960-1010,3370-3450`
- Modify: `tests/test_vector_pdf_fusion_route.py`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: fusion response image/artifact fields and `exterior_summary.pending_opening_count`.
- Produces: platform summary for pending openings; saved `pdf_component_overlay.png` uses orange for pending bridges.

- [ ] **Step 1: Write failing route and template tests**

```python
def test_route_returns_pending_opening_summary(self):
    self.assertEqual(payload["exterior_summary"]["pending_opening_count"], 1)

def test_template_explains_orange_pending_openings(self):
    self.assertIn("待确认开口", html)
    self.assertIn("pending_opening_count", html)
```

- [ ] **Step 2: Run platform tests to verify they fail**

Run: `python -m unittest tests.test_vector_pdf_fusion_route tests.test_energy_template`

Expected: FAIL because the pending-opening summary is absent.

- [ ] **Step 3: Return and render the pending-opening count**

Add `pending_opening_count` to the route’s exterior summary and to the review panel. Preserve the existing confirmation gate: a user still must review the closed outline before energy calculation.

- [ ] **Step 4: Run platform tests and full suite with Poppler**

Run: `$env:POPPLER_PATH='C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'; python -m unittest discover -s tests`

Expected: PASS.

- [ ] **Step 5: Replay the supplied page without retraining**

Run the fusion pipeline using the stored probability artifact for `BIM-20260826-7676`, write output to a new temporary directory, and visually compare `pdf_component_overlay.png` with the supplied left-upper and right-upper crops. Confirm that only PDF-supported opening bridges become orange/blue/cyan and that no pending opening appears in accepted door/window totals.

- [ ] **Step 6: Commit platform contract and verification changes**

```bash
git add web_server_server.py templates/energy.html tests/test_vector_pdf_fusion_route.py tests/test_energy_template.py
git commit -m "feat: expose pending opening review state"
```
