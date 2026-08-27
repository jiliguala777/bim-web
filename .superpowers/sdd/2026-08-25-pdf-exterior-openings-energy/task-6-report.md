# Task 6 Report: Server-Side Exterior Confirmation

## Implementation

- Extended authenticated `POST /energy/vector_pdf_fusion` with the exterior overlay, exterior summary, fixed artifact paths, and the byte-exact SHA-256 of `pdf_exterior_topology.json`.
- Added authenticated `POST /energy/vector_pdf_exterior_confirm`.
- The confirmation route requires a literal safe `report_number`, a 64-digit topology digest, a finite positive non-boolean scale, `confirmed is True`, a positive integer `page_number`, and an explicitly present normalized `crop_bbox_page_px` (`null` for full-page analysis).
- The route resolves the report and its fixed `vector_pdf_fusion` directory, rejects report/fusion/artifact symlinks and path aliases, and reloads only `pdf_exterior_topology.json` and `pdf_opening_candidates.json`.
- It hashes the reloaded topology bytes and returns 409 on a stale digest. It also requires topology/opening provenance equality and exact page/crop equality with the request.
- It validates the unconfirmed `review_required` orthogonal closed candidate, positive finite area/perimeter, real source-wall evidence, opening references, and confirmation readiness.
- Every disk `small_gap_repair` is measured from its endpoints using the confirmed scale and compared with `math.nextafter(0.6, math.inf)`.
- Task 4 `apply_scale_to_exterior` performs the deep-copy conversion. The source auto artifacts remain `confirmed: false` and `load_geometry_ready: false`.
- Recognition geometry uses the producer-published, server-validated `real_wall_segments`; it never infers walls by subtracting bridges from the polygon. Doors/windows are independent opening objects.
- Successful recognition preserves scaled exterior topology, openings, separate opening-width totals, confirmed scale calibration, image size, source/provenance page/crop metadata, and legacy `room_topology.load_geometry_ready: false`.
- `recognition.json` is serialized with `allow_nan=False`, flushed/fsynced to a same-directory temporary file, then atomically replaced. Validation failures never touch the existing file.
- The route never invokes `_floorplan_segmenter` or the legacy recognition route.

## TDD Evidence

### RED

Command:

```text
python -m unittest tests.test_vector_pdf_exterior_route -v
```

Initial result: confirmation requests returned 404 because the route did not exist. The focused success test was re-run after correcting a test-environment-only missing `_floorplan_segmenter` attribute and failed as expected with `404 != 200`.

A later security review found that a missing full-page crop field was indistinguishable from explicit `null`. The new focused test first failed with `200 != 400`, proving the gap before the route was tightened.

### GREEN

```text
python -m unittest tests.test_vector_pdf_exterior_route -v
Ran 7 tests ... OK

python -m unittest tests.test_vector_pdf_fusion_route -v
Ran 1 test ... OK
```

The tests cover success from disk artifacts despite forged browser geometry, exact opening widths, real-wall/bridge separation, hash conflict without overwrite, missing/non-finite scale, strict boolean confirmation, stale page/crop, explicit full-page null crop, non-closed topology, metric repair violation, path traversal, authentication, and absence of ONNX calls.

## Required Regression Command

```text
python -m unittest tests.test_vector_pdf_exterior_route tests.test_vector_pdf_fusion_route tests.test_energy_template -v
```

Final result: 72 tests executed; all 8 Task 6/fusion tests passed. Five unrelated legacy energy assertions failed because this worktree has no `models/M2_pub_plus_user.onnx`, import sets `HAS_FLOORPLAN_AI=False`, and those requests return 501 `AI module not available` before reaching their expected legacy validation. This task did not change the legacy availability gate and the brief forbids unrelated server/test changes.

## Security Self-Review

- Browser polygon, area, perimeter, opening widths, and other extra geometry fields are ignored.
- Hash comparison uses `hmac.compare_digest`; the topology is reloaded from the fixed current report directory.
- Report, fusion directory, topology file, and openings file are checked against symlink/path escape.
- Topology and openings provenance must match each other and the explicit request page/crop binding.
- Boolean-as-number inputs are rejected for confirmation/page/scale/crop values.
- Metric repair validation uses endpoint geometry rather than trusting stored `width_px`.
- Deep-copy scaling prevents mutation of loaded source structures; no source artifact is rewritten.
- JSON finite-number enforcement and atomic replacement preserve an old recognition on validation/serialization/write failure.
- Review was performed locally because the task explicitly prohibited subagents, overriding the reviewer skill's delegation mechanism.

## Attention Points

- The exact required combined regression cannot be fully green in this worktree until the approved legacy ONNX model is present (or those unrelated tests explicitly arrange availability). Task 6's 7 route tests and the fusion route test are green without that model.
- `recognition.json` is ready for exterior geometry while the legacy room topology intentionally remains not ready, as required; downstream UI/load consumption is outside Task 6.

## Fix Round 1/5: Artifact Binding and Proven Wall Segments

### Review Findings Addressed

- Replaced loose `str()`/set filtering with strict non-empty string ID lists, uniqueness checks, and exact set relationships. Selected bridge IDs now exactly equal artifact bridge IDs; topology opening IDs exactly equal accepted opening IDs and opening-bridge opening IDs.
- Added a strict opening-bridge/accepted-opening bijection. Each pair must match orientation, canonical axis endpoints (reverse order allowed), endpoint-derived width, and non-empty host wall IDs; hosts must be topology source IDs. Openings must be exterior doors/windows.
- Added simple orthogonal polygon validation: strict finite numeric axis endpoints, analysis-image bounds, rejection of non-adjacent edge intersections, and recomputation of area/perimeter. Artifact values are accepted only within four ULPs, while recognition persists the recomputed values.
- Extended the Task 3 exterior topology contract with `real_wall_segments`. The pipeline publishes the new field for review-ready topology and `[]` for empty/ambiguous/unavailable topology.
- Confirmation validates every selected bridge and real wall segment as belonging completely to exactly one polygon edge. Per-edge real-wall plus selected-bridge intervals must be a non-overlapping, gap-free exact partition of the entire boundary.
- `geometry.walls` is mapped directly from validated `real_wall_segments`, preserving segment IDs, endpoints, lengths, and source provenance. Polygon-minus-bridge inference was removed.
- Replaced loose provenance dictionary equality with independent strict schema validation and recursive type-sensitive equality. Required schema fields are exactly `coordinate_space`, `page_number`, `page_size_pt`, `analysis_size_px`, `crop_bbox_page_px`, and `building_roi_px`.

### `real_wall_segments` Schema

Each review-ready topology contains:

```json
{
  "real_wall_segments": [{
    "segment_id": "real-wall-0001",
    "orientation": "horizontal",
    "start_px": [20.0, 20.0],
    "end_px": [80.0, 20.0],
    "length_px": 60.0,
    "source_wall_ids": ["line-00001"]
  }]
}
```

Contract rules:

- `segment_id` and every `source_wall_id` are unique/non-empty strict strings.
- Endpoints are strict finite numeric axis coordinates; `orientation` must agree with them and `length_px` must equal endpoint length within four ULPs.
- Every segment is the strict intersection of selected footprint boundary and accepted exterior wall evidence. Selected bridge intervals are removed before publication.
- Empty, open, ambiguous, and model-failure topology publishes `real_wall_segments: []`.

### RED Evidence

- **A — cross-artifact binding:** duplicate topology opening ID, extra accepted opening, endpoint mismatch, and host mismatch all initially returned 200 and overwrote recognition. The focused command reported four failed subtests (`200 != 409`). Empty hosts were later shown to have the same failure.
- **B — polygon/metrics:** self-intersection, forged area/perimeter, and an out-of-bounds ring all initially returned 200. A one-ULP artifact metric was persisted verbatim instead of its recomputed value (`5000.000000000001 != 5000.0`).
- **C — proven walls:** Task 3 tests initially raised `KeyError: real_wall_segments`; pipeline artifacts lacked the field. Confirmation initially accepted unknown repair types, partial-outside bridges, real-wall gaps, bridge-wall overlaps, extra unselected bridges, and boolean endpoints, and the success test received inferred `exterior-wall-*` IDs instead of producer IDs.
- **D — provenance:** mixed `int`/`float` artifact provenance, coordinate-space/crop mismatch, zero page size, out-of-bounds ROI, and boolean ROI all initially returned 200 rather than 409.

### GREEN Evidence

Final required coverage command:

```text
python -m unittest tests.test_vector_pdf_exterior tests.test_vector_pdf_rooms tests.test_vector_pdf_fusion_pipeline tests.test_vector_pdf_openings tests.test_vector_pdf_exterior_route tests.test_vector_pdf_fusion_route -v
Ran 65 tests in 0.967s
OK
```

Additional verification:

```text
python -m py_compile vector_pdf_exterior.py vector_pdf_fusion_pipeline.py web_server_server.py tests/test_vector_pdf_exterior.py tests/test_vector_pdf_fusion_pipeline.py tests/test_vector_pdf_exterior_route.py tests/test_vector_pdf_fusion_route.py
git diff --check
```

Both completed with exit code 0 (Git emitted only the repository's LF-to-CRLF working-copy warnings).

### Fix-Round Security Self-Review

- No browser-supplied geometry or second browser hash was introduced.
- The topology hash remains the confirmation token; the independently reloaded openings artifact is admitted only when its IDs, bridge geometry, hosts, widths, types, and provenance exactly cross-validate against the hashed topology.
- `True` cannot masquerade as `1` in endpoints, metrics, provenance, scale, page, or crop fields.
- Every selected bridge resolves exactly once, has a legal type, and lies completely on one boundary edge; no unselected artifact bridge is allowed.
- Real walls and bridges cannot positively overlap, and their total per-edge partition proves exact perimeter coverage before atomic persistence.
- Existing recognition remains untouched on every new validation failure.

### Attention Points After Fix Round 1

- The confirmation contract intentionally fails closed if diagnostic artifacts contain unreferenced openings or unselected bridges. The current single-footprint pipeline fixtures produce exact sets and pass; ambiguous/empty topology remains unconfirmable.
- The missing legacy ONNX model warning remains environmental and does not affect the requested 65-test vector-PDF coverage suite.

## Fix Round 2/5: Raw Opening Artifact Digest Binding

### Implementation and Schema

- `pdf_exterior_topology.json` now carries `opening_artifact_sha256`, the lowercase SHA-256 digest of the exact bytes atomically written to `pdf_opening_candidates.json`.
- The JSON writer serializes once to UTF-8 bytes, writes and fsyncs those bytes to a temporary file, atomically replaces the destination, and returns the same bytes for hashing. This removes any serialize-versus-file ambiguity.
- Publication ordering is fail-closed: openings is atomically replaced first, its digest is added to topology, and topology is atomically replaced last. A failure between replacements leaves the old topology bound to the old openings, so confirmation rejects the mixed pair.
- Confirmation first checks the browser's topology hash against the reloaded topology bytes. It then requires a canonical lowercase 64-hex `opening_artifact_sha256` and compares it with the reloaded openings bytes using `hmac.compare_digest`, before provenance, geometry, or energy semantics are processed.
- Model-unavailable/empty artifacts use the same publisher and therefore bind the current empty openings artifact rather than omitting the digest.

### RED Evidence

Focused command before implementation:

```text
python -m unittest tests.test_vector_pdf_fusion_pipeline.VectorPdfFusionPipelineTests.test_pipeline_publishes_unconfirmed_exterior_and_openings tests.test_vector_pdf_fusion_pipeline.VectorPdfFusionPipelineTests.test_model_unavailable_never_confirms_exterior_artifact tests.test_vector_pdf_exterior_route.VectorPdfExteriorConfirmRouteTests.test_confirm_rejects_any_opening_artifact_byte_change_without_overwrite -v
Ran 3 tests in 0.412s
FAILED (failures=2, errors=2)
```

- Both producer tests raised `KeyError: 'opening_artifact_sha256'`.
- Changing only `accepted_openings[0].kind` from door to window returned 200 and changed the persisted energy category.
- Changing only `confidence` also returned 200. Both cases demonstrated that unchanged topology bytes/hash did not bind the complete openings artifact.

### GREEN Evidence

Focused RED cases after implementation:

```text
Ran 3 tests in 0.490s
OK
```

Required Round 2 coverage:

```text
python -m unittest tests.test_vector_pdf_fusion_pipeline tests.test_vector_pdf_exterior tests.test_vector_pdf_openings tests.test_vector_pdf_exterior_route tests.test_vector_pdf_fusion_route -v
Ran 61 tests in 0.878s
OK
```

Additional coverage rejects missing, short, non-hex, and noncanonical digest values without touching an existing `recognition.json`; it also asserts all three model-unavailable opening collections are empty and their raw artifact is nevertheless digest-bound.

### Round 2 Security Self-Review and Attention Points

- Any openings byte change, including fields not duplicated into topology such as `kind`, `confidence`, or evidence, invalidates the topology-held digest and returns 409 before persistence.
- No browser openings token or browser geometry authority was introduced; the existing browser topology hash now transitively binds every byte of the server-produced openings artifact.
- Existing `recognition.json` remains byte-for-byte unchanged for digest format or digest mismatch failures.
- Only the approved producer, server, and focused tests were changed; no UI/energy-route behavior and no running Flask process were touched.
- The missing legacy ONNX/optional IFC/DXF warnings remain environmental and did not affect the requested vector-PDF suite.
