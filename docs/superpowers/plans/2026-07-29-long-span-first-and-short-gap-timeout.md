# Long-Span-First Topology Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow supported long-span rectangles to enclose existing rooms safely, then restore full short-gap beam search under a 60-second deadline.

**Architecture:** Replace rectangle-bounding-box overlap rejection with rasterized room-interior crossing checks. Replace the 16-evaluation short-search budget with an injectable monotonic clock and a 60-second deadline.

**Tech Stack:** Python 3.12, NumPy, OpenCV, `unittest`.

## Global Constraints

- Long-span lines keep 60% vector-side and 15% model-side minimum support.
- Existing rooms must remain stable after every accepted repair.
- Short search keeps 24 candidates, beam width 16, and at most 16 added lines.
- Short search stops at 60 seconds and returns the best reliable result found.

---

### Task 1: Permit enclosing long spans without crossing room interiors

**Files:**
- Modify: `floorplan_topology_repair.py`
- Modify: `tests/test_floorplan_topology_repair.py`

**Interfaces:**
- Produces: `_line_crosses_room_interiors(shape, line, rooms) -> bool`
- Consumed by: `_search_dominant_span_rectangle`

- [x] **Step 1: Add failing tests**

Add a helper test with a square room polygon proving a line through the center
is rejected while a line outside the polygon is allowed. Add an integration
test containing a small closed room inside a fragmented, fully supported
larger rectangle and require `dominant_span_rectangle` repair.

- [x] **Step 2: Run the two tests and verify failure**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_long_line_crossing_room_interior_is_rejected tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_long_span_can_enclose_existing_room_without_crossing_it
```

- [x] **Step 3: Implement room-interior crossing**

Rasterize each `polygon_px`, erode the filled interior by three pixels, and
reject a candidate only when one of its line masks intersects that eroded
interior. Remove the rectangle bounding-box overlap rejection. Keep the
existing post-simulation `_baseline_rooms_preserved` validation.

- [x] **Step 4: Run topology tests**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_topology_repair
```

---

### Task 2: Restore short search with a 60-second deadline

**Files:**
- Modify: `floorplan_topology_repair.py`
- Modify: `tests/test_floorplan_topology_repair.py`

**Interfaces:**
- `_search_multi_gap_solution(..., *, monotonic_clock=None)`
- Search diagnostics: `evaluation_count`, `time_limit_seconds`, `elapsed_seconds`, `timed_out`

- [x] **Step 1: Replace budget tests with failing deadline tests**

Use an injected fake monotonic clock to prove the search can evaluate more
than 16 states when time remains, stops when simulated elapsed time reaches
60 seconds, and preserves a reliable solution found before timeout.

- [x] **Step 2: Run deadline tests and verify failure**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_multi_gap_search_can_exceed_sixteen_evaluations_before_timeout tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_multi_gap_search_stops_at_sixty_seconds tests.test_floorplan_topology_repair.ConservativeTopologyRepairTests.test_multi_gap_search_keeps_reliable_solution_found_before_timeout
```

- [x] **Step 3: Implement the 60-second deadline**

Replace `MAX_MULTI_GAP_TOPOLOGY_EVALUATIONS` with:

```python
MULTI_GAP_SEARCH_TIMEOUT_SECONDS = 60.0
```

Use `time.monotonic` unless a test clock is supplied. Check elapsed time before
each expensive topology evaluation, stop both search loops at the deadline,
and return the best accumulated solution.

- [x] **Step 4: Verify all tests and the real second-floor page**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_topology_repair
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Run the saved second-floor page-14 artifacts through the candidate model and
verify `repair_reason` is `dominant_span_rectangle` and short-search
`evaluation_count` is zero.

- [x] **Step 5: Commit and restart**

```powershell
git add -- floorplan_topology_repair.py tests/test_floorplan_topology_repair.py docs/superpowers/plans/2026-07-29-long-span-first-and-short-gap-timeout.md
git commit -m "fix: prioritize long-span topology repair"
```

Restart the single local `run_local.py` process with the candidate model and
verify `/energy/ai_status` returns HTTP 200.
