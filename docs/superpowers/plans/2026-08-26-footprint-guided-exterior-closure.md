# Footprint-Guided Exterior Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 PDF 外轮廓无法闭合时，利用真实外墙锚点与模型建筑区域内外侧证据生成最少量的可追踪正交推断边，并在人工确认后用于能耗面积和周长计算。

**Architecture:** 新建独立的 `vector_pdf_exterior_inference.py` 负责候选生成和证据过滤；现有 `vector_pdf_exterior.py` 扩展为可在同一正交图中评估推断边并选择最低代价闭合面。流水线只在原拓扑为 `exterior_not_closed` 时触发后备推断，服务端继续重新校验可信几何，平台将推断边以橙色虚线和独立统计展示。

**Tech Stack:** Python 3.12、NumPy、OpenCV、Flask、Jinja/原生 JavaScript、`unittest`

**Spec:** `docs/superpowers/specs/2026-08-26-footprint-guided-exterior-closure-design.md`

## Global Constraints

- 保持现有墙体、窗、外门弧、门弧墙段恢复和普通小缺口行为不变。
- 推断闭合只在现有拓扑没有闭合面时运行。
- 推断边必须严格水平或竖直，并同时具有真实外墙锚点和模型建筑区域内外侧证据。
- 单条推断边长度不得超过分析图短边的 10%。
- 最终采用的全部推断边总长度不得超过候选外轮廓周长的 12%。
- 推断闭合结果必须保持 `review_required` 和 `load_geometry_ready=false`，人工确认后才能进入能耗计算。
- 推断边不能成为门或窗，不能进入门窗宽度和面积扣除。
- 最终真实图纸视觉验收由用户在能耗平台执行，不由实现过程重放真实 PDF。
- 保留用户拥有的未跟踪文件 `tests/test_vector_pdf_exterior_smoke_cli.py` 和 `tools/smoke_vector_pdf_exterior.py`，不得修改或提交。

---

### Task 1: 生成具有模型内外侧证据的正交推断候选

**Files:**
- Create: `vector_pdf_exterior_inference.py`
- Create: `tests/test_vector_pdf_exterior_inference.py`

**Interfaces:**
- Consumes: `exterior_walls: list[dict]`、`probabilities: numpy.ndarray`（形状为 `(10, height, width)`）、`image_size: tuple[int, int]`、`building_roi: list[int] | None`。
- Produces: `InferenceThresholds`，以及 `generate_exterior_inference_candidates(exterior_walls, probabilities, image_size, building_roi) -> list[dict]`。
- 每个候选包含 `inference_id`、`edge_group_id`、`inference_type`、`orientation`、`start_px`、`end_px`、`length_px`、`inside_direction`、`anchor_wall_ids`、`inside_mean`、`outside_mean`、`boundary_mean`、`decision` 和 `reason_codes`。
- 同一个直角恢复的水平边与竖直边共享 `edge_group_id`；后续拓扑只能整组采用。

- [ ] **Step 1: 写共线候选、直角候选和证据过滤的失败测试**

在 `tests/test_vector_pdf_exterior_inference.py` 创建确定性的概率图辅助函数，并添加以下测试：

```python
import unittest
import numpy as np


def wall(candidate_id, start, end, orientation, inside_direction):
    return {
        "candidate_id": candidate_id,
        "start_px": list(start),
        "end_px": list(end),
        "orientation": orientation,
        "inside_direction": inside_direction,
    }


def rectangular_footprint(width=200, height=140):
    values = np.zeros((10, height, width), dtype=np.float32)
    values[0, 20:120, 20:180] = 1.0
    return values


class ExteriorInferenceCandidateTests(unittest.TestCase):
    def test_generates_two_edge_corner_group_for_supported_upper_right_gap(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("top", (20, 20), (170, 20), "horizontal", "down"),
            wall("right", (180, 30), (180, 120), "vertical", "left"),
        ]
        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [0, 0, 200, 140],
        )
        accepted = [item for item in candidates if item["decision"] == "accepted_candidate"]
        corner = [item for item in accepted if item["inference_type"] == "orthogonal_corner"]
        self.assertEqual(len(corner), 2)
        self.assertEqual(len({item["edge_group_id"] for item in corner}), 1)
        self.assertEqual(
            {(tuple(item["start_px"]), tuple(item["end_px"])) for item in corner},
            {((170, 20), (180, 20)), ((180, 20), (180, 30))},
        )

    def test_rejects_candidate_when_declared_inside_has_no_footprint_support(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        probabilities = np.zeros((10, 140, 200), dtype=np.float32)
        walls = [
            wall("top", (20, 20), (170, 20), "horizontal", "down"),
            wall("right", (180, 30), (180, 120), "vertical", "left"),
        ]
        candidates = generate_exterior_inference_candidates(
            walls, probabilities, (200, 140), [0, 0, 200, 140],
        )
        self.assertFalse(any(item["decision"] == "accepted_candidate" for item in candidates))
        self.assertIn("insufficient_inside_footprint_support", {
            reason for item in candidates for reason in item["reason_codes"]
        })

    def test_rejects_each_edge_longer_than_ten_percent_of_short_side(self):
        from vector_pdf_exterior_inference import generate_exterior_inference_candidates

        walls = [
            wall("top", (20, 20), (120, 20), "horizontal", "down"),
            wall("right", (180, 45), (180, 120), "vertical", "left"),
        ]
        candidates = generate_exterior_inference_candidates(
            walls, rectangular_footprint(), (200, 140), [0, 0, 200, 140],
        )
        self.assertFalse(any(item["decision"] == "accepted_candidate" for item in candidates))
        self.assertIn("inferred_edge_exceeds_short_side_limit", {
            reason for item in candidates for reason in item["reason_codes"]
        })
```

在同一测试类中增加五个明确用例：

- `test_generates_collinear_extension_between_dangling_endpoints`：输入同一水平墙带的 `[20, 20]-[70, 20]` 与 `[90, 20]-[150, 20]`，断言唯一接受边为 `[70, 20]-[90, 20]` 且类型为 `collinear_extension`；
- `test_connected_endpoint_does_not_generate_duplicate_candidate`：在上述端点处加入垂直锚点，断言没有以该端点起始的候选；
- `test_parallel_double_wall_bands_remain_separate`：输入 `y=20` 与 `y=24` 两条平行墙带，断言候选的两个锚点始终具有相同固定坐标；
- `test_rejects_corner_intersection_outside_roi`：将理论交点放到 `[10, 10, 190, 130]` 外，断言原因包含 `inferred_intersection_outside_roi`；
- `test_rejects_corner_with_conflicting_inside_directions`：水平墙内侧为 `down`、竖向墙内侧为 `right` 且建筑区域位于左下方，断言原因包含 `incompatible_corner_inside_directions`。

- [ ] **Step 2: 运行测试并确认以缺少模块失败**

Run: `python -m unittest tests.test_vector_pdf_exterior_inference -v`

Expected: FAIL，错误包含 `ModuleNotFoundError: No module named 'vector_pdf_exterior_inference'`。

- [ ] **Step 3: 实现候选生成和概率采样**

在 `vector_pdf_exterior_inference.py` 实现：

定义 `InferenceThresholds`，字段和值必须为：`collinear_tolerance_px=2`、`endpoint_connection_tolerance_px=4`、`side_offset_fraction=0.006`、`side_offset_min_px=4`、`side_offset_max_px=16`、`footprint_inside_mean_min=0.50`、`footprint_side_difference_min=0.25`、`footprint_boundary_mean_min=0.35`、`max_edge_short_side_fraction=0.10`。实现精确签名 `generate_exterior_inference_candidates(exterior_walls: list[dict], probabilities: np.ndarray, image_size: tuple[int, int], building_roi: list[int] | None) -> list[dict]`。

实现时按以下固定顺序处理：

1. 校验概率形状和图像尺寸；
2. 规范化并合并同墙带重叠锚点，但不跨越平行墙带；
3. 提取未连接端点；
4. 生成共线连接和正交理论交点；
5. 对每条边采样内侧、外侧和边界概率；
6. 检查 ROI、长度、方向、最近墙带和穿越内部；
7. 为所有候选保留接受或拒绝原因；
8. 按几何坐标排序后分配稳定 ID。

不要从模型概率轮廓直接生成边，也不要读取门窗列表。

- [ ] **Step 4: 运行候选模块测试并确认通过**

Run: `python -m unittest tests.test_vector_pdf_exterior_inference -v`

Expected: PASS。

- [ ] **Step 5: 提交候选生成模块**

```powershell
git add -- vector_pdf_exterior_inference.py tests/test_vector_pdf_exterior_inference.py
git commit -m "feat: generate footprint guided exterior edges"
```

---

### Task 2: 在正交拓扑中选择最低代价推断闭合面

**Files:**
- Modify: `vector_pdf_exterior.py:464-1105`
- Modify: `tests/test_vector_pdf_exterior.py:446-635`
- Modify: `tests/test_vector_pdf_exterior_inference.py`

**Interfaces:**
- Consumes: Task 1 产生的全部推断候选。
- Extends: `build_exterior_topology(exterior_walls, gaps, openings, probabilities, image_size, building_roi, *, scale_m_per_px=None, pending_openings=None, inference_candidates=None) -> dict`，新增的可选参数默认为 `None`，保证旧调用完全不变。
- Produces: 成功时的 `closure_method="footprint_guided_inference"`、`inferred_edges`、`inference_candidates`、`inferred_length_px`、`inferred_perimeter_ratio` 和 `inference_reason_codes`。
- Internal: `_evaluate_inferred_face(face: dict, candidates_by_id: dict[str, dict]) -> dict` 重新计算推断总长、比例、组完整性和评分，并返回 `accepted: bool` 与稳定原因码。

- [ ] **Step 1: 写右上角闭合、长度比例和竞争拓扑失败测试**

在 `tests/test_vector_pdf_exterior_inference.py` 添加：

```python
class FootprintGuidedTopologyTests(unittest.TestCase):
    def test_two_edge_corner_group_closes_supported_exterior(self):
        from vector_pdf_exterior import build_exterior_topology, enumerate_exterior_gaps

        walls = [
            wall("top", (20, 20), (170, 20), "horizontal", "down"),
            wall("right", (180, 30), (180, 120), "vertical", "left"),
            wall("bottom", (20, 120), (180, 120), "horizontal", "up"),
            wall("left", (20, 20), (20, 120), "vertical", "right"),
        ]
        candidates = [
            inferred("corner-h", "corner-1", (170, 20), (180, 20), "horizontal", "down"),
            inferred("corner-v", "corner-1", (180, 20), (180, 30), "vertical", "left"),
        ]
        topology = build_exterior_topology(
            walls, enumerate_exterior_gaps(walls, (200, 140), [0, 0, 200, 140]),
            [], rectangular_footprint(), (200, 140), [0, 0, 200, 140],
            inference_candidates=candidates,
        )
        self.assertEqual(topology["status"], "review_required")
        self.assertEqual(topology["closure_method"], "footprint_guided_inference")
        self.assertEqual({edge["inference_id"] for edge in topology["inferred_edges"]}, {
            "corner-h", "corner-v",
        })
        self.assertFalse(topology["load_geometry_ready"])

    def test_rejects_solution_when_inferred_edges_exceed_twelve_percent_of_perimeter(self):
        from vector_pdf_exterior import _evaluate_inferred_face

        candidates = {
            f"edge-{index}": {
                **inferred(
                    f"edge-{index}", f"group-{index}",
                    (index * 40, 20), (index * 40 + 30, 20), "horizontal", "down",
                ),
                "length_px": 30.0,
            }
            for index in range(5)
        }
        result = _evaluate_inferred_face({
            "perimeter": 1000.0,
            "inferred_edge_ids": set(candidates),
            "edge_group_ids": {f"group-{index}" for index in range(5)},
        }, candidates)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["inferred_perimeter_ratio"], 0.15)
        self.assertIn("inferred_perimeter_ratio_exceeded", result["reason_codes"])

    def test_rejects_competing_near_equal_inferred_polygons(self):
        walls = competing_right_wall_fixture()
        candidates = [
            inferred("outer-h", "outer", (150, 20), (180, 20), "horizontal", "down"),
            inferred("outer-v", "outer", (180, 20), (180, 45), "vertical", "left"),
            inferred("inner-h", "inner", (150, 24), (176, 24), "horizontal", "down"),
            inferred("inner-v", "inner", (176, 24), (176, 45), "vertical", "left"),
        ]
        topology = build_exterior_topology(
            walls, [], [], rectangular_footprint(), (200, 140), [0, 0, 200, 140],
            inference_candidates=candidates,
        )
        self.assertEqual(topology["status"], "ambiguous_exterior")
        self.assertIn("competing_inferred_exterior", topology["inference_reason_codes"])

    def test_never_uses_only_half_of_an_orthogonal_corner_group(self):
        walls = open_upper_right_fixture()
        only_horizontal = inferred(
            "corner-h", "corner-1", (170, 20), (180, 20), "horizontal", "down",
        )
        topology = build_exterior_topology(
            walls, [], [], rectangular_footprint(), (200, 140), [0, 0, 200, 140],
            inference_candidates=[only_horizontal],
        )
        self.assertEqual(topology["status"], "exterior_not_closed")
        self.assertIn("incomplete_inferred_edge_group", topology["inference_reason_codes"])
```

测试文件同时定义 `inferred()`、`open_upper_right_fixture()` 和 `competing_right_wall_fixture()`，返回上面断言所使用的完整字典和墙段。`inferred()` 固定写入 `decision="accepted_candidate"`、`inside_mean=0.90`、`outside_mean=0.05`、`boundary_mean=0.80`、两个非空 `anchor_wall_ids` 以及由端点计算的 `length_px`。

在 `tests/test_vector_pdf_exterior.py` 为现有闭合、未闭合、歧义、门窗桥和小缺口用例分别增加 `self.assertEqual(topology["inferred_edges"], [])` 与 `self.assertIsNone(topology["closure_method"])`，证明不传 `inference_candidates` 时结果不变。

- [ ] **Step 2: 运行测试并确认因新参数或新字段缺失而失败**

Run: `python -m unittest tests.test_vector_pdf_exterior_inference tests.test_vector_pdf_exterior -v`

Expected: FAIL，错误是 `build_exterior_topology()` 不接受 `inference_candidates`，或结果缺少 `closure_method`。

- [ ] **Step 3: 扩展拓扑边证据和闭合面评分**

在 `vector_pdf_exterior.py` 将 `build_exterior_topology` 改为 Interfaces 中给出的精确签名，并做以下最小扩展：

- `inference_candidates is None` 时走原有分支，不改变旧的结构歧义判断。
- 只把 `decision == "accepted_candidate"` 的候选加入图。
- 图边证据新增 `inferred_edge_ids` 和 `edge_group_ids`，在合并、切割交点、面枚举时与原墙 ID 一样传播。
- 对每个模型支持的闭合面提取实际使用的推断边，检查直角组完整性。
- 计算 `inferred_length_px / perimeter_px`，超过 `0.12` 的面拒绝。
- 推断模式按 `(推断总长度, 推断边数量, -平均内外概率差, -真实墙长度比例, polygon)` 排序。
- 若第二候选拓扑不同，且其推断长度与第一候选相差不超过 `max(8px, 第一候选周长的 2%)`，返回 `ambiguous_exterior`。
- 最终多边形继续使用现有正交面枚举，因此自交路径不会成为有效面。
- `_empty_exterior_topology` 增加空的推断字段，保证所有状态具有一致结构。

- [ ] **Step 4: 运行拓扑和候选模块测试并确认通过**

Run: `python -m unittest tests.test_vector_pdf_exterior_inference tests.test_vector_pdf_exterior -v`

Expected: PASS。

- [ ] **Step 5: 提交拓扑推断支持**

```powershell
git add -- vector_pdf_exterior.py tests/test_vector_pdf_exterior.py tests/test_vector_pdf_exterior_inference.py
git commit -m "feat: close exterior with supported inferred edges"
```

---

### Task 3: 仅在原拓扑失败时接入推断并保存诊断图片

**Files:**
- Modify: `vector_pdf_fusion_pipeline.py:170-328`
- Modify: `vector_pdf_fusion_pipeline.py:540-630`
- Modify: `tests/test_vector_pdf_fusion_pipeline.py:202-285`

**Interfaces:**
- Consumes: Task 1 的 `generate_exterior_inference_candidates` 和 Task 2 的 `build_exterior_topology(exterior_walls, gaps, openings, probabilities, image_size, building_roi, pending_openings=pending_openings, inference_candidates=inference_candidates)`。
- Produces: `pdf_exterior_topology.json`、`pdf_opening_candidates.json`、`pdf_vector_fusion.json` 和 `pdf_exterior_overlay.png` 中一致的推断诊断。
- Extends: `exterior_summary` 增加 `closure_method`、`inferred_edge_count`、`inferred_length_px`、`inferred_perimeter_ratio`。

- [ ] **Step 1: 写触发顺序、成功回退和覆盖图颜色的失败测试**

在 `tests/test_vector_pdf_fusion_pipeline.py` 添加：

```python
def test_inference_runs_only_after_normal_exterior_is_not_closed(self):
    inferred_topology = {
        "status": "review_required",
        "closure_method": "footprint_guided_inference",
        "polygon_px": [[20, 20], [180, 20], [180, 120], [20, 120]],
        "inferred_edges": [{
            "inference_id": "inferred-0001",
            "start_px": [150, 20],
            "end_px": [180, 20],
            "orientation": "horizontal",
        }],
        "inferred_length_px": 30.0,
        "inferred_perimeter_ratio": 0.058,
        "load_geometry_ready": False,
    }
    with patch("vector_pdf_fusion_pipeline.build_exterior_topology") as build_mock, patch(
        "vector_pdf_fusion_pipeline.generate_exterior_inference_candidates",
        return_value=[accepted_inferred_edge()],
    ) as candidate_mock:
        build_mock.side_effect = [not_closed_topology(), inferred_topology]
        result = run_fixture_pipeline()
    self.assertEqual(build_mock.call_count, 2)
    candidate_mock.assert_called_once()
    self.assertEqual(result["exterior_topology"]["closure_method"], "footprint_guided_inference")
```

增加以下明确测试函数和断言：

- `test_inference_is_skipped_when_normal_topology_is_reviewable`：`build_mock.return_value` 为普通 `review_required`，断言 `candidate_mock.assert_not_called()` 且 `build_mock.call_count == 1`；
- `test_failed_inference_keeps_original_not_closed_topology`：两次构建都返回失败，断言结果状态为 `exterior_not_closed` 且保存全部 `inference_candidates`；
- `test_nonvector_and_failure_payloads_publish_zero_inference_counts`：断言两个既有早退夹具的 `inferred_edge_count == 0`、`inferred_length_px == 0.0`；
- `test_exterior_overlay_draws_selected_inference_as_orange_dashes`：在白色 `80x80` 图上绘制 `[10, 40]-[70, 40]`，断言线上同时存在 BGR `(0, 165, 255)` 像素和未绘制的白色间隔；
- `test_inferred_edges_never_enter_accepted_openings`：断言所有 `accepted_openings` 的 ID 集合与所有 `inferred_edges` 的 ID 集合不相交。

- [ ] **Step 2: 运行流水线测试并确认失败**

Run: `python -m unittest tests.test_vector_pdf_fusion_pipeline -v`

Expected: FAIL，因为流水线没有导入或调用 `generate_exterior_inference_candidates`，且摘要缺少推断字段。

- [ ] **Step 3: 接入后备阶段并扩展产物**

在 `vector_pdf_fusion_pipeline.py` 中：

1. 保持第一次 `build_exterior_topology` 调用完全不变；
2. 仅当状态为 `exterior_not_closed` 时生成候选；
3. 候选中至少有一条 `accepted_candidate` 时，使用同一墙、缺口、开口和概率再次调用拓扑构建；
4. 第二次结果为 `review_required` 时采用推断拓扑，否则保留第一次失败状态，同时将候选拒绝原因写入诊断；
5. 在 `_publish_exterior_artifacts` 中统一发布推断字段；
6. 在 `_draw_exterior_overlay` 中以橙色虚线绘制最终采用边，以淡橙色绘制未采用候选，红色未解决缺口保持不变；
7. 所有失败和非矢量返回路径提供空数组及零计数。

- [ ] **Step 4: 运行流水线及相邻几何测试**

Run: `python -m unittest tests.test_vector_pdf_fusion_pipeline tests.test_vector_pdf_exterior_inference tests.test_vector_pdf_exterior tests.test_vector_pdf_openings tests.test_vector_pdf_door_recovery -v`

Expected: PASS。

- [ ] **Step 5: 提交流水线与图片产物改动**

```powershell
git add -- vector_pdf_fusion_pipeline.py tests/test_vector_pdf_fusion_pipeline.py
git commit -m "feat: publish inferred exterior closure"
```

---

### Task 4: 服务端重新校验推断边并保护能耗确认

**Files:**
- Modify: `web_server_server.py:1853-2055`
- Modify: `web_server_server.py:2407-2456`
- Modify: `web_server_server.py:2474-2635`
- Modify: `tests/test_vector_pdf_exterior_route.py:422-635`
- Modify: `tests/test_vector_pdf_fusion_route.py:35-145`

**Interfaces:**
- Consumes: 推断拓扑字段、已保存的 `pdf_opening_candidates.json` 哈希、图像尺寸和确认比例尺。
- Produces: 只有服务端重新校验通过的 `confirmed=true`、`load_geometry_ready=true` 外轮廓；确认后仍保留 `closure_method` 和 `inferred_edges`。

- [ ] **Step 1: 写合法确认和篡改拒绝的失败测试**

在 `tests/test_vector_pdf_exterior_route.py` 添加：

```python
def test_confirm_accepts_server_saved_inferred_edges_after_revalidation(self):
    topology = inferred_review_topology()
    with saved_vector_artifacts(topology):
        response = self.client.post("/api/energy/vector-pdf-exterior/confirm", json={
            "task_id": "BIM-TEST",
            "scale_m_per_px": 0.02,
        })
    self.assertEqual(response.status_code, 200)
    saved = response.get_json()["exterior_topology"]
    self.assertTrue(saved["confirmed"])
    self.assertTrue(saved["load_geometry_ready"])
    self.assertEqual(saved["closure_method"], "footprint_guided_inference")

def test_confirm_rejects_tampered_inferred_length_ratio_or_non_polygon_edge(self):
    for mutation in (forge_inferred_length, forge_ratio, move_edge_off_polygon):
        topology = inferred_review_topology()
        mutation(topology)
        with saved_vector_artifacts(topology):
            response = self.client.post("/api/energy/vector-pdf-exterior/confirm", json={
                "task_id": "BIM-TEST", "scale_m_per_px": 0.02,
            })
        self.assertEqual(response.status_code, 400)
```

增加以下明确测试函数：

- `test_confirm_rejects_inferred_total_over_twelve_percent`：将推断边改为多边形周长的 `0.121`，断言 HTTP 400；
- `test_confirm_rejects_diagonal_inferred_edge`：把终点横纵坐标同时改变，断言 HTTP 400 且错误包含 `axis aligned`；
- `test_confirm_rejects_duplicate_or_incomplete_corner_group`：分别复制推断 ID、删除直角组的一条边，两个子用例均断言 HTTP 400；
- `test_confirm_rejects_inferred_edge_reused_as_opening`：把推断 ID 写入 `opening_ids`，断言 HTTP 400；
- `test_confirm_keeps_legacy_noninferred_topology_compatible`：使用现有矩形夹具且不提供推断字段，断言 HTTP 200。

- [ ] **Step 2: 运行路由测试并确认失败**

Run: `python -m unittest tests.test_vector_pdf_exterior_route tests.test_vector_pdf_fusion_route -v`

Expected: FAIL，因为 `_validate_confirmable_topology` 尚未重新计算和验证推断边字段。

- [ ] **Step 3: 扩展服务端可信几何验证**

在 `_validate_confirmable_topology` 中：

- 当 `closure_method == "footprint_guided_inference"` 时要求 `inferred_edges` 非空；
- 对每条边调用现有严格轴线校验，重新计算长度；
- 验证每条推断边完整落在最终多边形的一条边上；
- 验证 ID 唯一、直角组完整、锚点 ID 非空；
- 重新计算推断总长及其周长比例并覆盖客户端字段；
- 比例超过 `0.12`、边不在多边形上或字段不一致时抛出明确错误；
- 验证推断边 ID 不出现在门窗 opening ID 集合；
- 现有非推断拓扑继续走原逻辑。

在融合路由的 `exterior_summary.setdefault` 中加入四个新字段，保证旧任务能够加载。在确认成功写回时保留推断来源和诊断，不能将推断边改写成真实墙段。

- [ ] **Step 4: 运行路由、能耗几何和模板后端测试**

Run: `python -m unittest tests.test_vector_pdf_exterior_route tests.test_vector_pdf_fusion_route tests.test_vector_pdf_energy_geometry tests.test_energy_template -v`

Expected: PASS。

- [ ] **Step 5: 提交确认安全改动**

```powershell
git add -- web_server_server.py tests/test_vector_pdf_exterior_route.py tests/test_vector_pdf_fusion_route.py
git commit -m "fix: validate inferred exterior before energy use"
```

---

### Task 5: 在平台展示推断闭合状态并完成全量回归

**Files:**
- Modify: `templates/energy.html:3380-3465`
- Modify: `tests/test_energy_template.py:2040-2140`

**Interfaces:**
- Consumes: `exterior_summary.closure_method`、`inferred_edge_count`、`inferred_length_px`、`inferred_perimeter_ratio`。
- Produces: 用户可见的“模型约束推断闭合”标识、推断线数量/长度/占比，以及保持原有图片点击放大行为的综合识别图。

- [ ] **Step 1: 写模板字段和确认提示的失败测试**

在 `tests/test_energy_template.py` 添加：

```python
def test_energy_review_shows_footprint_guided_inference_diagnostics(self):
    html = self.render_energy_template()
    for token in (
        "inferred_edge_count",
        "inferred_length_px",
        "inferred_perimeter_ratio",
        "footprint_guided_inference",
        "推断外轮廓线",
        "确认后才用于能耗计算",
    ):
        self.assertIn(token, html)
```

再添加响应测试：摘要为推断闭合时页面显示待确认而非可直接计算；非推断闭合不显示误导性警告；图片仍使用现有放大组件。

- [ ] **Step 2: 运行模板测试并确认字段缺失**

Run: `python -m unittest tests.test_energy_template -v`

Expected: FAIL，因为模板中尚无推断闭合诊断字段。

- [ ] **Step 3: 增加平台诊断显示**

在现有外轮廓复核摘要中增加：

- 闭合方式：“模型约束推断闭合”；
- 推断外轮廓线数量；
- 推断总长度（像素；比例尺确认后可同时显示米）；
- 推断线占周长百分比；
- 文案：“橙色虚线为推断边，确认后才用于能耗计算”。

使用现有 `exterior_overlay` 图片与点击放大逻辑，不新增弹窗框架。状态不是 `footprint_guided_inference` 时隐藏推断专属提示。

- [ ] **Step 4: 运行相关几何与平台测试**

Run: `python -m unittest tests.test_vector_pdf_exterior_inference tests.test_vector_pdf_exterior tests.test_vector_pdf_fusion_pipeline tests.test_vector_pdf_fusion_route tests.test_vector_pdf_exterior_route tests.test_vector_pdf_energy_geometry tests.test_energy_template -v`

Expected: PASS。

- [ ] **Step 5: 运行完整测试和差异检查**

```powershell
$env:POPPLER_PATH='C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin'
python -m unittest discover -s tests
git diff --check
```

Expected: 全部测试 PASS，`git diff --check` 无输出。不得运行真实 PDF 重放；最终真实图纸验证交给用户。

- [ ] **Step 6: 提交平台显示并记录最终状态**

```powershell
git add -- templates/energy.html tests/test_energy_template.py
git commit -m "feat: show inferred exterior closure diagnostics"
git status --short
```

Expected: 只剩用户已有的未跟踪验证脚本和临时预览目录；本计划涉及的文件均已提交。
