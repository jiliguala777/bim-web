# User-Scoped Energy Report Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Store every energy upload and recognition artifact under a safe per-user directory while enforcing ordinary-user isolation and explicit administrator access across files, signed PDF tokens, locks, SQLite records, APIs, and the history UI.

**Architecture:** Add a Flask-independent energy_report_storage.py module that turns an authenticated owner and report number into a validated EnergyReportContext. web_server_server.py remains the HTTP adapter: it derives ownership from session state, maps storage exceptions to HTTP responses, records the composite (username, report_number) identity in SQLite, and passes resolved paths to existing engines. Artifact names and recognition algorithms remain unchanged.

**Tech Stack:** Python 3.12, Flask, SQLite, pathlib, hashlib, unittest, and the existing vanilla JavaScript energy UI.

**Spec:** docs/superpowers/specs/2026-08-27-user-scoped-energy-report-storage-design.md

## Global Constraints

- Disk layout is UPLOAD_FOLDER/energy/readable-username-sha256prefix/report-number/.
- Username mapping uses NFKC normalization, Unicode alphanumerics plus ._-, a 48-character readable limit, and an 8-hex SHA-256 suffix.
- Ordinary users can operate only on the username stored in their authenticated session.
- Only environment-backed administrator login sets session is_admin to true.
- Administrators explicitly name another owner; omitted owner means the administrator's own space.
- Report database identity is (username, report_number), never report_number alone.
- Signed PDF tokens bind owner, report, stored filename, and page count.
- There is no read or write fallback to legacy flat energy/report-number directories.
- Existing model, geometry, scale, and energy algorithms do not change.
- Production deployment stops if a flat legacy report directory exists.
- Preserve the user-owned untracked smoke files already in the worktree.

---

### Task 1: Safe Storage Primitives

**Files:**
- Create: energy_report_storage.py
- Create: tests/test_energy_report_storage.py

**Interfaces:**
- Produces: EnergyReportContext, InvalidReportPath, ReportAccessDenied, user_storage_key(username), validate_report_number(report_number), resolve_report_owner(session_username, is_admin, requested_owner), resolve_energy_report_context(upload_root, session_username, is_admin, report_number, requested_owner=None, create=False).
- Consumes: Python standard library only; no Flask or web_server_server import.

- [ ] **Step 1: Write failing key and report validation tests**

    class UserStorageKeyTests(unittest.TestCase):
        def test_preserves_readable_unicode_and_adds_stable_hash(self):
            from energy_report_storage import user_storage_key
            self.assertEqual(user_storage_key("张三"), "张三-1d841bc0")
            self.assertEqual(user_storage_key("alice"), "alice-2bd806c9")

        def test_cleaned_collisions_still_have_distinct_hashes(self):
            from energy_report_storage import user_storage_key
            self.assertNotEqual(user_storage_key("a/b"), user_storage_key("a\\b"))

        def test_rejects_traversal_report_numbers(self):
            from energy_report_storage import InvalidReportPath, validate_report_number
            for value in ("", ".", "..", "../R", "R/child", "R\\child", "/tmp/R"):
                with self.subTest(value=value), self.assertRaises(InvalidReportPath):
                    validate_report_number(value)

- [ ] **Step 2: Run RED**

    python -m unittest tests.test_energy_report_storage -v

Expected: import failure because energy_report_storage does not exist.

- [ ] **Step 3: Implement public types and normalization**

    @dataclass(frozen=True)
    class EnergyReportContext:
        owner_username: str
        owner_storage_key: str
        report_number: str
        energy_root: Path
        owner_root: Path
        report_dir: Path

    class InvalidReportPath(ValueError):
        pass

    class ReportAccessDenied(PermissionError):
        pass

Use unicodedata.normalize("NFKC", username).strip(), collapse disallowed runs to "-", trim dot/hyphen edges, limit the readable part to 48 code points, and append hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:8].

- [ ] **Step 4: Write failing containment and symlink tests**

    def test_same_report_number_resolves_below_distinct_users(self):
        with tempfile.TemporaryDirectory() as directory:
            alice = resolve_energy_report_context(directory, "alice", False, "BIM-1", create=True)
            bob = resolve_energy_report_context(directory, "bob", False, "BIM-1", create=True)
            self.assertNotEqual(alice.report_dir, bob.report_dir)
            self.assertTrue(alice.report_dir.is_dir())
            self.assertTrue(bob.report_dir.is_dir())

    def test_rejects_symlinked_owner_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outside = root / "outside"
            outside.mkdir()
            (root / "energy").mkdir()
            (root / "energy" / user_storage_key("alice")).symlink_to(outside, target_is_directory=True)
            with self.assertRaises(InvalidReportPath):
                resolve_energy_report_context(root, "alice", False, "BIM-1")

- [ ] **Step 5: Run RED and implement strict resolution**

Reject symlinks before following them, require each resolved parent to equal the expected parent, create owner/report directories only for create=True, and never create paths during reads.

- [ ] **Step 6: Run GREEN and commit**

    python -m unittest tests.test_energy_report_storage -v
    git add energy_report_storage.py tests/test_energy_report_storage.py
    git commit -m "feat: add safe user report storage resolver"

---

### Task 2: Administrator Session State and Composite Report Schema

**Files:**
- Modify: web_server_server.py:245-345
- Create: tests/test_energy_report_ownership.py

**Interfaces:**
- Produces: _migrate_reports_schema(conn), _report_row(username, report_number), _upsert_report_status(username, report_number, status), _user_exists(username), and trustworthy username/is_admin session values.
- Consumes: storage exceptions and validation from Task 1.

- [ ] **Step 1: Write failing auth tests**

    def test_environment_admin_login_sets_admin_flag(self):
        with patch.dict(os.environ, {"ADMIN_USER": "admin", "ADMIN_PASSWORD": "secret"}):
            response = self.client.post("/login", json={"username": "admin", "password": "secret"})
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as state:
            self.assertIs(state["is_admin"], True)

    def test_database_login_is_not_admin(self):
        response = self.client.post("/login", json={"username": "alice", "password": "pw"})
        self.assertEqual(response.status_code, 200)
        with self.client.session_transaction() as state:
            self.assertIs(state["is_admin"], False)

    def test_logout_clears_all_identity_state(self):
        with self.client.session_transaction() as state:
            state.update(logged_in=True, username="admin", is_admin=True)
        self.client.get("/logout")
        with self.client.session_transaction() as state:
            self.assertNotIn("username", state)
            self.assertNotIn("is_admin", state)

- [ ] **Step 2: Run RED**

    python -m unittest tests.test_energy_report_ownership.AuthSessionTests -v

Expected: is_admin is absent and logout leaves identity fields.

- [ ] **Step 3: Implement explicit session privilege**

Set true only in the environment-admin branch, false in the database-user branch, and use session.clear() for logout.

- [ ] **Step 4: Write failing SQLite migration tests**

Create an old reports table in a temporary database, insert duplicate (alice, BIM-1) rows and one (bob, BIM-1), call _migrate_reports_schema, then assert one Alice row, one Bob row, status and updated_at columns, and an IntegrityError for a second identical composite key.

- [ ] **Step 5: Run RED and implement transactional migration**

Add missing columns with SQLite-compatible ALTER TABLE, retain newest created_at then largest id for duplicates, and create:

    CREATE UNIQUE INDEX IF NOT EXISTS reports_username_report_number_uq
    ON reports(username, report_number)

Rollback and re-raise migration failures instead of continuing with an unsafe partial schema.

- [ ] **Step 6: Implement composite helpers and run GREEN**

Use ON CONFLICT(username, report_number) DO UPDATE for status/updated_at. Every helper query includes both values.

    python -m unittest tests.test_energy_report_ownership -v

- [ ] **Step 7: Commit**

    git add web_server_server.py tests/test_energy_report_ownership.py
    git commit -m "feat: bind report records to authenticated owners"

---

### Task 3: Flask Report Context and General Upload Routes

**Files:**
- Modify: web_server_server.py:379-1148
- Modify: tests/test_energy_report_ownership.py

**Interfaces:**
- Produces: _requested_owner_username(payload=None), _effective_report_owner(requested_owner=None), _energy_report_context(report_number, requested_owner=None, create=False), _report_storage_error_response(exc).
- Consumes: Task 1 resolver and Task 2 report upsert helpers.

- [ ] **Step 1: Write a failing same-number upload test**

    def test_same_report_number_isolated_by_authenticated_user(self):
        self.login_as("alice")
        self.client.post("/energy/upload", data={"report_number": "BIM-1", "raster_file": image_file()})
        self.login_as("bob")
        self.client.post("/energy/upload", data={"report_number": "BIM-1", "raster_file": image_file()})
        self.assertTrue((self.upload_root / "energy" / user_storage_key("alice") / "BIM-1").is_dir())
        self.assertTrue((self.upload_root / "energy" / user_storage_key("bob") / "BIM-1").is_dir())
        self.assertFalse((self.upload_root / "energy" / "BIM-1").exists())

Also assert Alice submitting owner_username=bob gets 403 and creates nothing.

- [ ] **Step 2: Run RED**

    python -m unittest tests.test_energy_report_ownership.ReportUploadIsolationTests -v

Expected: current route writes the flat path.

- [ ] **Step 3: Implement the Flask adapter**

Read username/is_admin from session. Owner input may come from the explicit argument, form, JSON, or query string, but ordinary-user authorization always wins. An administrator's explicit target must exist in users; the environment administrator's own username is allowed without a users row. Map access denial to 403, malformed paths to 400, and missing reports to 404 without absolute paths.

- [ ] **Step 4: Replace general DXF/IFC report paths**

Update these routes/workers to receive or resolve EnergyReportContext:

    get_simulation_data
    energy_upload
    get_all_layers_geometry
    energy_geometry
    energy_geometry_advanced
    energy_calculate
    simple_simulation_task
    background_simulation_task
    upload_ifc
    ifc_properties
    ifc_walls
    ifc_simulate

Copy owner_username into background-job input before leaving request context. Move benchmark fixtures to UPLOAD_FOLDER/ops/bestest because they are not user reports.

- [ ] **Step 5: Register upload status**

Upsert created after safe directory creation, uploaded after committed files, and failed only for the same existing composite row.

- [ ] **Step 6: Run GREEN and commit**

    python -m unittest tests.test_energy_report_ownership tests.test_energy_calc tests.test_envelope_libraries -v
    git add web_server_server.py tests/test_energy_report_ownership.py
    git commit -m "feat: isolate general energy uploads by user"

---

### Task 4: Owner-Bound PDF Tokens and Vector Recognition

**Files:**
- Modify: web_server_server.py:1528-2832
- Modify: tests/test_energy_template.py
- Modify: tests/test_vector_pdf_fusion_route.py
- Modify: tests/test_vector_pdf_exterior_route.py
- Modify: tests/test_energy_report_ownership.py

**Interfaces:**
- Changes: _make_pdf_upload_token(owner_username, report_number, stored_filename, page_count), _validated_prepared_pdf(owner_username, report_number, token, page_number), _require_exterior_report_dir(context).
- Consumes: EnergyReportContext for locks, generation markers, artifacts, and recognition publication.

- [ ] **Step 1: Write a failing cross-user token test**

    def test_pdf_token_cannot_be_reused_by_another_user(self):
        self.login_as("alice")
        prepared = self.client.post("/energy/pdf_prepare", data=pdf_form("BIM-1")).get_json()
        self.login_as("bob")
        response = self.client.post("/energy/pdf_page_preview", data={
            "report_number": "BIM-1",
            "pdf_upload_token": prepared["upload_token"],
            "pdf_page_number": "1",
        })
        self.assertIn(response.status_code, {400, 403})

- [ ] **Step 2: Run RED**

    python -m unittest tests.test_energy_report_ownership.PdfOwnerBindingTests -v

Expected: token has no owner binding.

- [ ] **Step 3: Bind tokens and prepared files to owner**

Include owner_username in signed data. Resolve the context first, then require token owner, report, filename, and page count to match before opening the PDF.

- [ ] **Step 4: Convert all PDF/vector routes**

Update:

    prepare_energy_pdf
    preview_energy_pdf_page
    vector_pdf_fusion
    vector_pdf_exterior_confirm
    _serialized_exterior_report
    _require_exterior_report_dir
    ai_recognize
    save_scale_calibration

Lock keys, generation markers, vector_pdf_fusion, recognition.json, overlays, and scale artifacts all derive from context.report_dir. Status transitions are recognizing, recognized, or failed for the same owner/report.

- [ ] **Step 5: Update existing route fixtures**

Every authenticated test session includes:

    state["username"] = "test-user"
    state["is_admin"] = False

Artifact assertions insert user_storage_key("test-user") between energy and report number.

- [ ] **Step 6: Run GREEN and commit**

    python -m unittest tests.test_energy_report_ownership tests.test_vector_pdf_fusion_route tests.test_vector_pdf_exterior_route tests.test_energy_template -v
    git add web_server_server.py tests/test_energy_report_ownership.py tests/test_vector_pdf_fusion_route.py tests/test_vector_pdf_exterior_route.py tests/test_energy_template.py
    git commit -m "feat: bind PDF recognition artifacts to report owners"

---

### Task 5: Recognition, Calculation, and Composite Reads

**Files:**
- Modify: web_server_server.py:2857-3838
- Modify: tests/test_energy_template.py
- Modify: tests/test_energy_report_ownership.py

**Interfaces:**
- Consumes: EnergyReportContext and composite report helpers.
- Produces: owner-safe final recognition/calculation persistence and report detail/list APIs.

- [ ] **Step 1: Write failing operation authorization tests**

Create Alice's recognition.json, log in as Bob, and verify these cannot alter or read Alice's report:

    POST /energy/ai_recognize with owner_username=alice
    POST /energy/scale_calibration with owner_username=alice
    POST /energy/ai_simulate with owner_username=alice
    GET  /energy/report/BIM-1?owner_username=alice

Then authenticate the environment administrator and assert explicit Alice access succeeds for an existing report.

- [ ] **Step 2: Run RED**

    python -m unittest tests.test_energy_report_ownership.ReportOperationAuthorizationTests -v

Expected: at least one route uses the flat path or report_number-only query.

- [ ] **Step 3: Convert final routes and payloads**

Keep the owner-scoped `ai_recognize` and `save_scale_calibration` implementations completed in Task 4, and exercise them through the authorization tests above. Update ai_status payload construction, ai_simulate, get_energy_report, and get_user_reports. Recognition payloads record owner_username. Calculation uses ON CONFLICT(username, report_number) and sets calculated. Detail reads use both fields.

- [ ] **Step 4: Implement list queries**

Ordinary list:

    SELECT username, report_number, status, created_at, updated_at,
           geometry_used, results
    FROM reports
    WHERE username = ?
    ORDER BY updated_at DESC, id DESC

Administrator list omits the username predicate but returns the same safe fields. Never return absolute paths or storage keys.

- [ ] **Step 5: Run GREEN and commit**

    python -m unittest tests.test_energy_report_ownership tests.test_energy_template tests.test_energy_calc -v
    git add web_server_server.py tests/test_energy_report_ownership.py tests/test_energy_template.py
    git commit -m "feat: authorize recognition and report reads by owner"

---

### Task 6: Administrator History UI and Owner Propagation

**Files:**
- Modify: templates/energy.html:2500-4200
- Modify: tests/test_energy_template.py
- Modify: tests/test_energy_pdf_preview_state.js

**Interfaces:**
- Consumes: username in history API responses and owner_username accepted by protected routes.
- Produces: activeReportOwnerUsername client state propagated through administrator operations.

- [ ] **Step 1: Write failing UI tests**

Assert history cards render username/report number, loadReportDetails accepts both values, and selected administrator requests append owner_username to form, JSON, and detail query payloads.

    self.assertIn("activeReportOwnerUsername", html)
    self.assertIn("item.username", history_segment)
    self.assertIn("owner_username", history_segment)

Add a Node test proving switching owner with the same report number invalidates pending PDF recognition state.

- [ ] **Step 2: Run RED**

    python -m unittest tests.test_energy_template -v
    node tests/test_energy_pdf_preview_state.js

- [ ] **Step 3: Implement owner-aware state**

Keep activeReportOwnerUsername beside reportNumber. Administrator cards display owner; ordinary cards remain compact. Centralize request decoration so PDF prepare/preview/fusion/confirmation, recognition, scale, and calculation cannot omit owner after administrator selection.

- [ ] **Step 4: Prevent stale cross-owner state**

Changing either owner or report clears tokens, selected pages, topology hashes, scale, accepted recognition, and geometry. Pending-recognition equality compares (owner_username, report_number).

- [ ] **Step 5: Run GREEN and commit**

    python -m unittest tests.test_energy_template -v
    node tests/test_energy_pdf_preview_state.js
    git add templates/energy.html tests/test_energy_template.py tests/test_energy_pdf_preview_state.js
    git commit -m "feat: add owner-aware administrator report history"

---

### Task 7: Flat-Path Audit and Deployment Documentation

**Files:**
- Create: tests/test_energy_report_path_audit.py
- Modify: deploy/README.md
- Modify: README.md
- Modify: web_server_server.py only if the audit exposes remaining flat paths

**Interfaces:**
- Produces: source regression guard and deployment/rollback procedure.

- [ ] **Step 1: Write the failing source audit**

    def test_server_does_not_construct_flat_report_paths(self):
        source = Path("web_server_server.py").read_text(encoding="utf-8")
        forbidden = (
            "os.path.join(app.config['UPLOAD_FOLDER'], 'energy', report_number)",
            "Path(app.config['UPLOAD_FOLDER']) / 'energy' / report_number",
            "WHERE report_number = ?",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, source)

- [ ] **Step 2: Run RED**

    python -m unittest tests.test_energy_report_path_audit -v

Expected: fail if any report route still constructs a flat path or single-key query.

- [ ] **Step 3: Remove remaining matches**

    rg -n "UPLOAD_FOLDER.*energy|reports WHERE report_number|WHERE report_number" web_server_server.py

User operations use context. Operational fixtures use UPLOAD_FOLDER/ops and are documented exemptions.

- [ ] **Step 4: Update deployment manuals**

Document the new tree, backup of users.db/uploads, service stop, flat-directory preflight with abort behavior, transactional DB migration, ordinary/admin checks, and rollback. Do not include commands that silently move or delete legacy data.

- [ ] **Step 5: Run GREEN and commit**

    python -m unittest tests.test_energy_report_path_audit tests.test_deployment_assets -v
    git add tests/test_energy_report_path_audit.py deploy/README.md README.md web_server_server.py
    git commit -m "docs: add user-isolated report deployment checks"

---

### Task 8: Full Verification and Integration Handoff

**Files:**
- Verify all Task 1-7 changes.

**Interfaces:**
- Produces: review and integration evidence; production deployment remains separately authorized.

- [ ] **Step 1: Run focused security/workflow suites**

    python -m unittest tests.test_energy_report_storage tests.test_energy_report_ownership tests.test_energy_report_path_audit tests.test_vector_pdf_fusion_route tests.test_vector_pdf_exterior_route tests.test_energy_template -v
    node tests/test_energy_pdf_preview_state.js

- [ ] **Step 2: Run the full Python suite**

    python -m unittest discover -s tests

Expected: zero failures and zero errors.

- [ ] **Step 3: Run static audits**

    git diff --check origin/main...HEAD
    rg -n "UPLOAD_FOLDER.*energy|WHERE report_number = \\?" web_server_server.py
    git status --short

Expected: no flat user-report paths, no single-key report query, no whitespace errors, and only known user-owned untracked smoke files outside commits.

- [ ] **Step 4: Review against every spec requirement**

Check safe keys, ordinary isolation, explicit administrator access, DB composite identity, PDF token binding, symlink rejection, route coverage, UI owner propagation, no legacy fallback, and deployment stop conditions.

- [ ] **Step 5: Request code review and fix accepted findings with TDD**

Use superpowers:requesting-code-review. Every accepted defect gets a failing regression test before its fix.

- [ ] **Step 6: Use the branch-finishing workflow**

    git status --short
    git log --oneline origin/main..HEAD

Use superpowers:finishing-a-development-branch to offer local merge, GitHub PR, or branch preservation. Production backup, pull, migration, restart, and two-user verification require separate explicit deployment authorization after merge.
