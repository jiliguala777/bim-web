# Final-review fix recovery report

Date: 2026-09-02 (Asia/Shanghai)

## Scope and preservation

This recovery started from a large, uncommitted final-review fix wave.  The
worktree was inspected before editing.  No reset, checkout, or overwrite was
used.  The user-owned untracked smoke paths were not inspected, modified, or
staged.  `users.db` was not staged or changed by the verification runs:

```text
SHA-256: F0FB63F246F389B1DD6A495D6B4F3A2CA8601D97589A89C177CCD5B65EA0C2A9
Length: 24576
LastWriteTimeUtc: 2026-09-01 02:13:46
```

## Recovered final-review changes

1. **Storage-key identity collision:** `energy_report_storage.py` now exposes
   canonical username validation plus collision detection.  Registration
   rejects noncanonical names and storage-key collisions under an immediate
   transaction; request and worker owner resolution fail closed for an
   ambiguous selected owner.  The preflight includes database users and an
   enabled environment administrator in the collision check.
2. **Fixed report children:** centralized direct-child and nested-directory
   resolvers reject aliases and containment escapes.  DXF, IFC, weather,
   raster, recognition, ROI, vector-artifact, EnergyPlus, and exterior-lock
   paths use those resolvers.  Upload/copy/image/JSON publication is staged to
   a temporary file and atomically published with `os.replace`; this also
   replaces a fixed-file hard link or symlink rather than writing through it.
3. **Preflight composite mapping:** read-only preflight now verifies the exact
   persisted `(owner storage key, report number)` set against the disk tree;
   it rejects flat legacy layouts, unknown entries, and mismatched pairs.
4. **Composite lifecycle:** AI recognition, vector fusion, exterior
   confirmation, calculation dispatch, and both workers write state against
   the same `(username, report_number)` identity, covering recognizing,
   recognized, calculating, calculated, and failed paths.
5. **Portable error/filename handling:** late exterior `InvalidReportPath` is
   caught before broad `ValueError` handling and returns sanitized HTTP 400.
   Report numbers reject Windows reserved basenames and a trailing dot or
   space.

The inherited diff contained regression coverage for each of those findings:
registration/runtime/preflight collisions and exact mappings; representative
symlink/hardlink files and nested output paths; lifecycle owner isolation;
and error/filename semantics.

## RED/GREEN record

The prior worker exhausted its quota without leaving a report, so no reliable
pre-recovery RED transcript was available.  To preserve the existing
uncommitted production changes, recovery did not reverse or overwrite them
merely to manufacture a RED run.  No additional behavior was found missing
after source and route audit, so no new production code was added in recovery.

One recovery diagnostic initially proposed a new image-write wrapper and was
run RED.  It failed with the expected `AttributeError` because that new wrapper
did not exist.  Inspection then established that the already-imported
`floorplan_page_pipeline._write_image` has the required temporary-file,
`fsync`, and `os.replace` implementation, and every server use supplies a
centrally resolved destination.  The diagnostic test was removed rather than
adding redundant production code.  Therefore the trustworthy acceptance
evidence for the inherited fix is the fresh GREEN verification below.

## Fresh GREEN verification

```powershell
python -m unittest tests.test_energy_report_storage tests.test_energy_report_ownership tests.test_energy_report_path_audit tests.test_vector_pdf_fusion_route tests.test_vector_pdf_exterior_route tests.test_energy_template -v
```

Result: **210 tests passed**, **13 skipped**, exit 0.

```powershell
python -m unittest discover -s tests
```

Result: **513 tests passed**, **13 skipped**, exit 0.

```powershell
node tests/test_energy_pdf_preview_state.js
git diff --check
python -m py_compile energy_report_storage.py web_server_server.py tools/user_report_storage_preflight.py tests/test_energy_report_storage.py tests/test_energy_report_ownership.py tests/test_energy_report_path_audit.py tests/test_vector_pdf_fusion_route.py tests/test_vector_pdf_exterior_route.py tests/test_energy_template.py
python -c "import ast, pathlib; [ast.parse(path.read_text(encoding='utf-8'), filename=str(path)) for path in map(pathlib.Path, ['energy_report_storage.py', 'web_server_server.py', 'tools/user_report_storage_preflight.py', 'tests/test_energy_report_storage.py', 'tests/test_energy_report_ownership.py', 'tests/test_energy_report_path_audit.py', 'tests/test_vector_pdf_fusion_route.py', 'tests/test_vector_pdf_exterior_route.py'])]"
node --check static/energy/pdf_recognition_state.js
node --check tests/test_energy_pdf_preview_state.js
rg -n "UPLOAD_FOLDER.*energy|WHERE report_number = \?" web_server_server.py
```

Results: Node state test, diff check, Python compilation, AST parse, and both
Node syntax checks exited 0.  The static path/SQL search produced no matches
and exited 1 as expected.

## Concerns

- Thirteen real-symlink cases are capability-skipped on this Windows host
  because creating symlinks requires developer mode or privilege.  The
  hardlink coverage runs here; run the skipped cases on Linux CI/deployment
  before production authorization.
- Full and focused suites intentionally log simulated negative-path errors
  (PDF/vector conversion and worker failures); neither suite has failures.
- The working tree has user-owned untracked smoke paths only; they are outside
  this commit.

## Commit

Pending at report creation; the recovery commit hash is recorded in the
handoff response after the commit succeeds.

## Targeted repair cycle: hardlink artifacts and exterior terminal lifecycle

Date: 2026-09-02 (Asia/Shanghai)

### RED

The following behavior-first tests were added before production changes and
run against commit `3c810bea97`:

```powershell
python -m unittest tests.test_energy_report_storage.EnergyReportStorageResolutionTests.test_rejects_a_hardlinked_fixed_child_file_for_reads tests.test_energy_report_ownership.ReportUploadIsolationTests.test_energyplus_worker_rejects_a_preexisting_hardlinked_idf tests.test_energy_report_ownership.ReportUploadIsolationTests.test_exterior_lock_rejects_a_fixed_file_hardlink tests.test_floorplan_page_pipeline.PdfPagePipelineTests.test_prepare_pdf_page_replaces_hardlinked_preprocessing_metadata tests.test_vector_pdf_exterior_route.VectorPdfExteriorConfirmRouteTests.test_terminal_confirmation_failures_mark_only_the_active_owner_failed -v
```

Result: exit 1, **5 test methods failed with 8 assertions**.  The storage
resolver accepted a hardlinked read input; the exterior lock acquired a
hardlinked lock file; the worker called `generate_idf` for a pre-seeded nested
job directory; `preprocessing.json` overwrote its hardlink peer; and hash,
generation, validation, and persistence confirmation failures left the active
report status as `recognized`.

### Minimal fix

- Read resolution now checks the direct child's `lstat` mode and requires one
  hard link.  Lock acquisition uses `O_NOFOLLOW` when available and checks
  `fstat`/`lstat` identity, regular-file mode, and link count after opening.
- The PDF page pipeline atomically publishes `preprocessing.json` via a
  temporary file, `fsync`, and `os.replace`.
- EnergyPlus creates its job directory with exclusive `mkdir` mode `0700` and
  rejects an existing nested job directory before passing output paths to the
  engine.  This makes generated `run.idf` and engine output private to the
  newly-created job invocation.
- Once exterior confirmation has resolved its report context, an
  `after_this_request` transition writes `recognized` only for a 2xx result
  and `failed` for every later terminal error, keyed by that same owner/report
  composite identity.

### GREEN and regression evidence

The RED command above passed after the minimal fixes: **5 tests, OK**, exit 0.

```powershell
python -m unittest tests.test_floorplan_page_pipeline -v
python -m unittest tests.test_energy_report_storage tests.test_energy_report_ownership tests.test_vector_pdf_exterior_route -v
python -m unittest tests.test_energy_report_storage tests.test_energy_report_ownership tests.test_energy_report_path_audit tests.test_vector_pdf_fusion_route tests.test_vector_pdf_exterior_route tests.test_energy_template -v
python -m unittest discover -s tests
node tests/test_energy_pdf_preview_state.js
```

Results: pipeline **15 tests** passed; adjacent storage/ownership/exterior
suite **102 tests** passed with **12 capability skips**; the pre-existing
focused command now contains **214 tests** and passed with **13 skips**; full
discovery **518 tests** passed with **13 skips**; Node state test passed.

`git diff --check`, `py_compile`, AST parsing, both Node syntax checks, and
the static old-path/SQL search all passed (the search had zero matches and its
exit 1 was expected).  The worktree `users.db` SHA-256 remained
`F0FB63F246F389B1DD6A495D6B4F3A2CA8601D97589A89C177CCD5B65EA0C2A9`.

Concern: the Windows host cannot create symlinks without privilege, so the
existing real-symlink cases remain skipped; hardlink coverage is non-skipped
and was exercised in this repair cycle.

## Targeted repair cycle: confirmation callback ordering

Date: 2026-09-02 (Asia/Shanghai)

### RED/GREEN

A behavior-level two-client ordering test was added before the production
change.  It delays only the older failed confirmation's registered Flask
callback, submits a newer successful confirmation for the same report, then
releases the older callback:

```powershell
python -m unittest tests.test_vector_pdf_exterior_route.VectorPdfExteriorConfirmRouteTests.test_older_confirmation_callback_cannot_overwrite_a_newer_terminal_status -v
```

At `83fcda77ef` the test failed, exit 1: the final composite status was
`failed`, because the old callback ran after the decorator had released the
report lock and overwrote the new successful confirmation's `recognized`
state.  After the fix the same test passed, **1 test, OK**, exit 0.

`vector_pdf_exterior_confirm` now finalizes every response after report-context
resolution through `_finalize_exterior_confirmation_response`.  It converts
the response and updates the same `(owner_username, report_number)` status
while still executing inside `_serialized_exterior_report`'s lock: 2xx maps to
`recognized`, every terminal non-2xx response maps to `failed`.  The prior
`after_this_request` callback for confirmation was removed, so it cannot write
a stale status after a newer serialized operation.

### Regression evidence

```powershell
python -m unittest tests.test_vector_pdf_exterior_route -v
python -m unittest tests.test_energy_report_ownership -v
python -m unittest discover -s tests
node tests/test_energy_pdf_preview_state.js
```

Results: exterior route **25 tests** passed; ownership/lifecycle **64 tests**
passed with **10 capability skips**; full discovery **519 tests** passed with
**13 skips**; Node state test passed.  Diff check, Python compilation, AST
parse, both Node syntax checks, and static legacy path/SQL search passed (the
search returned no matches, exit 1 expected).  `users.db` stayed at
`F0FB63F246F389B1DD6A495D6B4F3A2CA8601D97589A89C177CCD5B65EA0C2A9`.

Concern: real symlink cases remain Windows capability-skipped; run them on a
privileged Linux CI/deployment host.
