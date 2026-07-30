# Topology Search Evaluation Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound multi-line topology repair to 16 expensive room-topology evaluations while preserving the best reliable solution found inside the budget.

**Architecture:** Add a deterministic evaluation counter inside `_search_multi_gap_solution`. Return search diagnostics with the mask and accepted candidates, then expose those diagnostics through `exterior_repair`.

**Tech Stack:** Python 3.12, NumPy, OpenCV, `unittest`.

## Global Constraints

- Multi-gap search performs at most 16 full room-topology evaluations.
- Candidate ordering and beam ranking remain unchanged.
- A reliable solution found before exhaustion remains eligible for selection.
- Without a reliable solution, return the original mask and no accepted lines.
- Existing dominant-rectangle, already-closed, and internal-fragment behavior remains unchanged.

---

### Task 1: Bound multi-gap topology evaluation

**Files:**
- Modify: `floorplan_topology_repair.py`
- Modify: `tests/test_floorplan_topology_repair.py`

**Interfaces:**
- Consumes: `_search_multi_gap_solution(mask, candidates, roi, min_room_area_px, limits)`
- Produces: `(calculation_mask, accepted_candidates, search_diagnostics)`, where diagnostics contain `evaluation_count`, `evaluation_limit`, and `budget_exhausted`.

- [x] **Step 1: Write failing budget tests**

Import `_search_multi_gap_solution` and `unittest.mock.patch`. Add one test whose fake room-topology boundary never produces a room and verify 24 candidates stop after 16 evaluations while returning the original mask. Add a second test whose first evaluated state is plausible and verify the best reliable solution remains returned when the remaining search exhausts the budget.

```python
@patch("floorplan_topology_repair._room_topology")
def test_multi_gap_search_stops_after_sixteen_topology_evaluations(
    self, room_topology
):
    empty = {"room_count": 0, "rooms": [], "total_area_px2": 0.0}
    room_topology.return_value = empty
    mask = np.zeros((30, 30), dtype=np.uint8)
    candidates = [
        {"line_px": [index, 2, index, 8], "score": 1.0, "length_px": 6.0}
        for index in range(1, 25)
    ]
    result, accepted, search = _search_multi_gap_solution(
        mask, candidates, [0, 0, 30, 30], 20.0,
        {"beam_width": 16, "max_lines": 16},
    )
    self.assertTrue(np.array_equal(result, mask))
    self.assertEqual(accepted, [])
    self.assertEqual(search["evaluation_count"], 16)
    self.assertTrue(search["budget_exhausted"])
```

The second test uses an empty baseline followed by a plausible room:

```python
plausible = {
    "room_count": 1,
    "rooms": [{"area_px2": 100.0, "bbox_px": [5, 5, 10, 10]}],
    "total_area_px2": 100.0,
}
room_topology.side_effect = [empty] + [plausible] * 16
```

- [x] **Step 2: Verify the tests fail**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_multi_gap_search_stops_after_sixteen_topology_evaluations tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_multi_gap_search_keeps_reliable_solution_found_before_budget_exhaustion
```

Expected: both fail because the function currently returns two values and has no evaluation budget.

- [x] **Step 3: Implement the evaluation budget**

Add:

```python
MAX_MULTI_GAP_TOPOLOGY_EVALUATIONS = 16
```

Count each `_room_topology(simulated, ...)` call inside the state expansion loop. Stop both loops before evaluation 17, select the best solution already accumulated, and return:

```python
search_diagnostics = {
    "evaluation_count": evaluation_count,
    "evaluation_limit": MAX_MULTI_GAP_TOPOLOGY_EVALUATIONS,
    "budget_exhausted": budget_exhausted,
}
```

Update `repair_vector_floorplan_topology` to receive the third return value and include it as `exterior_repair["search"]`.

- [x] **Step 4: Verify focused and complete tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_topology_repair
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Expected: zero failures.

- [x] **Step 5: Reproduce the second-floor case**

Run the saved page-14 preprocessing artifacts through `FloorplanSegmenterONNX.predict` using the candidate model, record elapsed time, and verify the returned topology diagnostics report no more than 16 evaluations.

- [x] **Step 6: Commit**

```powershell
git add -- floorplan_topology_repair.py tests/test_floorplan_topology_repair.py docs/superpowers/plans/2026-07-29-topology-search-budget.md
git commit -m "fix: bound topology combination search"
```
