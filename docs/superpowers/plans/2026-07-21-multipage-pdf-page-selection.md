# Multipage PDF Page Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload each PDF once, let the user select a page, and recognize exactly that page.

**Architecture:** A prepare endpoint persists and validates the PDF, then returns a signed token and page count. The recognition endpoint consumes the token and a one-based page number; the frontend coordinates the two requests and shows page selection only for multipage PDFs.

**Tech Stack:** Flask, itsdangerous/Flask signing, pdfplumber, pdf2image, vanilla JavaScript, unittest.

## Global Constraints

- Preserve all current uncommitted workspace changes.
- Do not create a worktree, commit, merge, push, or clean unrelated files.
- Page numbers exposed to users are one-based.
- Do not add PDF.js or thumbnails.

---

### Task 1: PDF preparation and signed reference

**Files:**
- Modify: `web_server_server.py`
- Test: `tests/test_energy_template.py`

**Interfaces:**
- Produces: `POST /energy/pdf_prepare` returning `{success, upload_token, page_count, filename}`.
- Consumes: multipart fields `raster_file` and `report_number`.

- [ ] Add tests using a two-page ReportLab PDF for page count, non-PDF rejection, and an unreadable PDF.
- [ ] Run the focused tests and verify they fail because the route does not exist.
- [ ] Add a server-side signed upload token bound to the report and saved PDF filename; implement the prepare route with `pdfplumber.open`.
- [ ] Run the focused tests and verify they pass.

### Task 2: Recognize the selected PDF page

**Files:**
- Modify: `web_server_server.py`
- Modify: `vector_pdf_scale.py`
- Test: `tests/test_energy_template.py`
- Test: `tests/test_vector_pdf_scale.py`

**Interfaces:**
- Consumes: form fields `pdf_upload_token` and `pdf_page_number`.
- Produces: response and `recognition.json` fields `pdf_page_number` and `pdf_page_count`.

- [ ] Add failing tests proving page 2 is passed as zero-based index 1 to vector extraction and as one-based page 2 to rendering.
- [ ] Add failing tests for zero, out-of-range, tampered-token, and report-mismatch requests.
- [ ] Implement token resolution, page validation, and selected-page propagation without changing image uploads.
- [ ] Persist selected-page metadata through `_build_recognition_payload` and return it from the route.
- [ ] Run the focused tests and verify they pass.

### Task 3: Page-selection UI

**Files:**
- Modify: `templates/energy.html`
- Test: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: `/energy/pdf_prepare` response.
- Produces: `/energy/ai_recognize` request with the token and selected one-based page.

- [ ] Add failing template contract tests for the page panel, page select, start button, prepare request, and selected-page form fields.
- [ ] Add the compact hidden page-selection panel and replace the obsolete single-page warning.
- [ ] Split `handleFile` into PDF prepare and recognition paths; keep images direct.
- [ ] Reset prepared state whenever another file is chosen and prevent duplicate recognition clicks.
- [ ] Run template tests and JavaScript syntax checking.

### Task 4: Full verification

**Files:**
- Verify all modified files only.

- [ ] Run `./.venv/Scripts/python.exe -m unittest discover -s tests` and require zero failures.
- [ ] Run Python compilation for `web_server_server.py`, `vector_pdf_scale.py`, and related recognition modules.
- [ ] Extract inline JavaScript and run `node --check`.
- [ ] Run `git diff --check` and inspect the targeted diff without staging or committing.
