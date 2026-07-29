# Annotation Training Repository Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the annotation and model-training toolchain as the independent private repository `jiliguala777/bim-annotation-training`, while removing annotation/training-only content from `bim-web` PR #1 without changing the remote `bim-web/main`.

**Architecture:** Build a standalone sibling repository that owns annotation, training, ONNX export, and copied runtime preprocessing dependencies. Keep the energy platform's own copies of shared PDF/ONNX modules, relocate the two energy JavaScript assets out of `annotation_tool/`, and update the existing PR branch rather than touching remote `main` directly.

**Tech Stack:** Python 3.12, Flask, OpenCV, NumPy, pdfplumber, pdf2image/Poppler, ONNX Runtime, PyTorch, JavaScript, PowerShell, Git, GitHub CLI.

## Global Constraints

- The new repository is private and named exactly `jiliguala777/bim-annotation-training`.
- `jiliguala777/bim-web/main` is not pushed to or modified directly.
- `bim-web` PR #1 keeps only energy-platform and shared runtime changes.
- The repositories must not depend on each other's local filesystem paths.
- Model weights, annotation data, exports, checkpoints, experiment artifacts, uploads, logs, databases, and credentials must not be committed.
- Shared PDF and ONNX runtime modules are copied into both repositories; no third shared package is introduced.
- Existing unrelated untracked files in `G:\bim-web` are never staged.

---

### Task 1: Assemble the standalone annotation-training repository

**Files:**
- Create repository root: `G:\bim-annotation-training`
- Copy: `annotation_tool/`
- Copy: `training/`
- Copy: `tools/export_floorplan_onnx.py`
- Copy: `floorplan_onnx.py`
- Copy: `floorplan_page_pipeline.py`
- Copy: `floorplan_vector_worker.py`
- Copy: `floorplan_ocr.py`
- Copy: `floorplan_rooms.py`
- Copy: `floorplan_topology_repair.py`
- Copy: `vector_pdf_scale.py`
- Copy relevant tests from `G:\bim-web\tests/`
- Create: `G:\bim-annotation-training\.gitignore`
- Create: `G:\bim-annotation-training\tests\test_repository_boundaries.py`

**Interfaces:**
- Consumes: the verified source state at `G:\bim-web` commit `e660b823`.
- Produces: an independent Git repository whose Python imports resolve from its own root.

- [ ] **Step 1: Verify the destination is safe**

Run:

```powershell
$destination = "G:\bim-annotation-training"
if (Test-Path -LiteralPath $destination) {
    throw "Destination already exists: $destination"
}
git -C G:\bim-web rev-parse --verify e660b823
```

Expected: the destination does not exist and Git prints the full commit SHA.

- [ ] **Step 2: Create the repository and copy only the planned sources**

Create `G:\bim-annotation-training`, initialize it with `git init -b main`, then copy the listed directories, modules, and these tests:

```text
tests/test_annotation_app.py
tests/test_annotation_config.py
tests/test_annotation_storage.py
tests/test_floorplan_page_pipeline.py
tests/test_floorplan_topology_repair.py
tests/test_floorplan_rooms.py
tests/test_training_dataset.py
tests/test_training_experiment.py
tests/test_training_losses.py
tests/test_export_floorplan_onnx.py
tests/test_crop_region.js
tests/test_export_result.js
tests/test_saved_mask_loader.js
```

Also copy the confirmed annotation/training specs and plans into the new repository's
`docs/superpowers/` directories, including the repository-split design and plan.

- [ ] **Step 3: Write the repository boundary test**

Create `tests/test_repository_boundaries.py`:

```python
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryBoundaryTests(unittest.TestCase):
    def test_standalone_sources_exist(self):
        for relative in (
            "annotation_tool/app.py",
            "training/run_experiment.py",
            "tools/export_floorplan_onnx.py",
            "floorplan_onnx.py",
            "floorplan_page_pipeline.py",
        ):
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_weight_and_runtime_directories_are_ignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        for rule in ("models/", "*.onnx", "*.pt", "annotation_tool/data/", "annotation_tool/runtime/", "artifacts/", "*.log"):
            self.assertIn(rule, ignore)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: Run the new boundary test and verify it fails**

Run:

```powershell
G:\bim-web\.venv\Scripts\python.exe -m unittest tests.test_repository_boundaries
```

Expected: FAIL because `.gitignore` and standalone documentation have not been completed.

- [ ] **Step 5: Add the safety-focused `.gitignore`**

Include at minimum:

```gitignore
.venv/
.venv-*/
__pycache__/
*.pyc
.env
.env.*
!.env.example
models/
*.onnx
*.pt
*.pth
*.ckpt
annotation_tool/data/
annotation_tool/runtime/
data/
exports/
artifacts/
output/
runs/
checkpoints/
*.log
*.db
uploads/
.playwright-cli/
.codex/
.worktrees/
```

- [ ] **Step 6: Verify and commit the assembled source boundary**

Run the boundary test again and expect PASS, then commit only planned files:

```powershell
git add .gitignore annotation_tool training tools tests docs *.py
git commit -m "feat: establish annotation training toolchain"
```

Expected: no model, data, runtime, or credential file is staged.

---

### Task 2: Make the new repository portable and documented

**Files:**
- Create: `G:\bim-annotation-training\README.md`
- Create: `G:\bim-annotation-training\requirements.txt`
- Modify: `G:\bim-annotation-training\annotation_tool\README.md`
- Modify: `G:\bim-annotation-training\annotation_tool\config.py`
- Modify: `G:\bim-annotation-training\annotation_tool\start_annotation_tool.ps1`
- Modify: `G:\bim-annotation-training\tests\test_annotation_config.py`
- Modify: `G:\bim-annotation-training\tests\test_repository_boundaries.py`

**Interfaces:**
- Consumes: the standalone source tree from Task 1.
- Produces: `AnnotationConfig.from_env()` with a portable default data directory and documented installation/start/training commands.

- [ ] **Step 1: Add a failing portable-default test**

Add `import os` if it is not already present, then add to
`tests/test_annotation_config.py`:

```python
def test_default_data_root_is_outside_repository_and_under_user_home(self):
    with patch.dict(os.environ, {}, clear=True), patch(
        "annotation_tool.config.Path.home",
        return_value=Path("C:/Users/tester"),
    ):
        config = AnnotationConfig.from_env()
    self.assertEqual(config.data_root, Path("C:/Users/tester/bim-annotation-training-data").resolve())
```

- [ ] **Step 2: Add a failing checkout-independence test**

Add to `tests/test_repository_boundaries.py`:

```python
def test_repository_does_not_reference_bim_web_checkout(self):
    checked = [
        *ROOT.glob("*.py"),
        *ROOT.glob("annotation_tool/**/*.py"),
        *ROOT.glob("annotation_tool/**/*.md"),
        *ROOT.glob("annotation_tool/**/*.ps1"),
        *ROOT.glob("training/**/*.py"),
        *ROOT.glob("training/**/*.ps1"),
        ROOT / "README.md",
    ]
    offenders = []
    for path in checked:
        if path.is_file() and r"G:\bim-web" in path.read_text(encoding="utf-8"):
            offenders.append(str(path.relative_to(ROOT)))
    self.assertEqual(offenders, [])
```

- [ ] **Step 3: Run the tests and verify they fail**

Run:

```powershell
G:\bim-web\.venv\Scripts\python.exe -m unittest tests.test_annotation_config
```

Expected: FAIL because the old default is `G:\bim网页\标注数据`.

- [ ] **Step 4: Implement the portable default**

Replace the constant data root with:

```python
def default_data_root() -> Path:
    return Path.home() / "bim-annotation-training-data"
```

Make `AnnotationConfig.from_env()` use `ANNOTATION_DATA_ROOT` when set and
`default_data_root()` otherwise. Preserve explicit constructor behavior.

- [ ] **Step 5: Add independent dependency and operator documentation**

`requirements.txt` must contain the annotation runtime dependencies:

```text
Flask>=3,<4
numpy>=2,<3
opencv-python-headless>=4.10,<5
pillow>=11,<13
pdfplumber>=0.11,<1
pdf2image>=1.17,<2
onnxruntime>=1.20,<2
```

The root README must document:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:POPPLER_PATH="C:\path\to\poppler\Library\bin"
$env:ONNX_MODEL_PATH="D:\models\best.onnx"
.\annotation_tool\start_annotation_tool.ps1
```

It must also document `training/setup_training_env.ps1`,
`python -m training.run_experiment`, the external data directory, and the rule
that trained models are manually deployed rather than committed.

- [ ] **Step 6: Remove checkout-specific path assumptions**

Rewrite `annotation_tool/README.md` so it refers to the repository root instead
of `G:\bim-web`. Keep the local-only host restriction and explain both
`ANNOTATION_DATA_ROOT` and `ONNX_MODEL_PATH`.

- [ ] **Step 7: Run boundary and configuration tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest `
  tests.test_repository_boundaries `
  tests.test_annotation_config `
  tests.test_annotation_storage `
  tests.test_annotation_app
```

Expected: PASS.

- [ ] **Step 8: Commit the portable operator experience**

```powershell
git add README.md requirements.txt annotation_tool tests/test_annotation_config.py tests/test_repository_boundaries.py
git commit -m "docs: make annotation training repository standalone"
```

---

### Task 3: Verify and publish `bim-annotation-training`

**Files:**
- Verify all tracked files in `G:\bim-annotation-training`
- Create remote repository: `jiliguala777/bim-annotation-training`

**Interfaces:**
- Consumes: the standalone repository from Tasks 1–2.
- Produces: private GitHub repository `main` at the verified local commit.

- [ ] **Step 1: Run the complete new-repository test suite**

Set Poppler and run:

```powershell
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
G:\bim-web\.venv\Scripts\python.exe -m unittest discover -s tests
node tests/test_crop_region.js
node tests/test_export_result.js
node tests/test_saved_mask_loader.js
```

Expected: all copied Python and JavaScript tests pass.

- [ ] **Step 2: Audit tracked content**

Run:

```powershell
git status --short
git ls-files
git ls-files | Select-String -Pattern '\.(onnx|pt|pth|ckpt)$'
git grep -Il -E 'BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|ghp_|sk-'
```

Expected: clean status; no weight paths; no credential matches.

- [ ] **Step 3: Create the private GitHub repository**

First verify it does not exist:

```powershell
gh repo view jiliguala777/bim-annotation-training
```

Expected: not found.

Create and push:

```powershell
gh repo create jiliguala777/bim-annotation-training `
  --private `
  --source G:\bim-annotation-training `
  --remote origin `
  --push
```

- [ ] **Step 4: Verify remote privacy and commit identity**

```powershell
gh repo view jiliguala777/bim-annotation-training --json visibility,defaultBranchRef,url
git ls-remote origin refs/heads/main
git rev-parse HEAD
```

Expected: visibility is `PRIVATE`; both SHA values match.

---

### Task 4: Remove annotation/training-only content from `bim-web` PR #1

**Files:**
- Create: `static/energy/crop_region.js`
- Create: `static/energy/pdf_recognition_state.js`
- Modify: `web_server_server.py`
- Modify: `tests/test_energy_template.py`
- Modify: `tests/test_crop_region.js`
- Modify: `README.md`
- Modify: `tests/test_deployment_assets.py`
- Delete from PR branch: `annotation_tool/`
- Delete from PR branch: `training/`
- Delete from PR branch: annotation/training-only tests and docs
- Restore from `origin/main`: `tools/export_floorplan_onnx.py`
- Restore from `origin/main`: `tests/test_export_floorplan_onnx.py`

**Interfaces:**
- Consumes: `/energy/crop_region.js` and `/energy/pdf_recognition_state.js` HTTP routes.
- Produces: the same route responses sourced from `static/energy/`, with no dependency on `annotation_tool/`.

- [ ] **Step 1: Change route tests to require platform-owned assets**

Update `tests/test_energy_template.py` assertions to compare route responses to:

```python
Path("static/energy/crop_region.js").read_bytes()
Path("static/energy/pdf_recognition_state.js").read_bytes()
```

Add:

```python
def test_energy_assets_do_not_depend_on_annotation_tool_directory(self):
    source = Path("web_server_server.py").read_text(encoding="utf-8")
    self.assertNotIn("annotation_tool', 'static", source)
```

- [ ] **Step 2: Run the focused tests and verify they fail**

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template
```

Expected: FAIL because energy assets still live under `annotation_tool/static`.

- [ ] **Step 3: Move the two shared energy assets**

Copy the current verified JavaScript content to:

```text
static/energy/crop_region.js
static/energy/pdf_recognition_state.js
```

Keep `annotation_tool/static/crop_region.js` only in the new repository. The
energy-only recognition-state script does not need to be copied into the new
repository unless an annotation test explicitly consumes it.

- [ ] **Step 4: Update the Flask asset routes**

Make `/energy/crop_region.js` and `/energy/pdf_recognition_state.js` serve from
`BASE_DIR/static/energy` while preserving route names, MIME types, and cache
behavior. No template URL changes are required.

- [ ] **Step 5: Remove annotation/training-only files**

Delete from the PR branch:

```text
annotation_tool/
training/
tests/test_annotation_app.py
tests/test_annotation_config.py
tests/test_annotation_storage.py
tests/test_training_dataset.py
tests/test_training_experiment.py
tests/test_training_losses.py
tests/test_export_result.js
tests/test_saved_mask_loader.js
docs/superpowers/plans/2026-07-27-pdf-annotation-training.md
docs/superpowers/plans/2026-07-28-multi-page-training-and-annotation-cropping.md
docs/superpowers/specs/2026-07-27-pdf-annotation-training-design.md
docs/superpowers/specs/2026-07-28-annotation-region-cropping-design.md
docs/superpowers/specs/2026-07-28-multi-page-floorplan-fine-tuning-design.md
docs/superpowers/specs/2026-07-29-split-annotation-training-repository-design.md
docs/superpowers/plans/2026-07-29-split-annotation-training-repository.md
```

Restore `tools/export_floorplan_onnx.py` and
`tests/test_export_floorplan_onnx.py` to their `origin/main` versions so their
training-only PR changes disappear without deleting files already present on
remote `main`.

- [ ] **Step 6: Remove operator documentation from `bim-web`**

Delete the “PDF 标注与单页模型微调” section from `README.md`. Rewrite the complex
PDF fallback paragraph to describe energy recognition rather than annotation or
training. Remove `test_annotation_and_training_operator_assets_are_documented`
from `tests/test_deployment_assets.py`.

- [ ] **Step 7: Run focused tests**

```powershell
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
.\.venv\Scripts\python.exe -m unittest `
  tests.test_energy_template `
  tests.test_deployment_assets `
  tests.test_floorplan_page_pipeline `
  tests.test_floorplan_topology_repair
node tests/test_crop_region.js
node tests/test_energy_pdf_preview_state.js
```

Expected: PASS.

- [ ] **Step 8: Commit the PR cleanup**

Stage only the explicitly listed platform files and deletions:

```powershell
git add -- README.md web_server_server.py static/energy tests docs/superpowers annotation_tool training tools/export_floorplan_onnx.py
git commit -m "refactor: separate annotation training toolchain"
```

Before committing, inspect `git diff --cached --name-status` and confirm no
unrelated untracked file is staged.

---

### Task 5: Verify and update `bim-web` PR #1

**Files:**
- Verify branch: `agent/recent-bim-updates`
- Update PR: `jiliguala777/bim-web#1`

**Interfaces:**
- Consumes: the cleaned `bim-web` branch from Task 4.
- Produces: a ready-for-review PR containing no annotation/training-only content.

- [ ] **Step 1: Run the complete `bim-web` verification**

```powershell
$env:POPPLER_PATH="C:\Users\majin\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin"
.\.venv\Scripts\python.exe -m unittest discover -s tests
node tests/test_crop_region.js
node tests/test_energy_pdf_preview_state.js
```

Expected: every remaining test passes; only intentional skips are reported.

- [ ] **Step 2: Audit the PR diff**

```powershell
git diff --name-only origin/main..HEAD |
  Select-String -Pattern '^(annotation_tool|training)/|test_annotation|test_training'
git diff --name-only origin/main..HEAD |
  Select-String -Pattern '\.(onnx|pt|pth|ckpt)$'
git status --short
```

Expected: the first two searches return no matches; status contains only the
pre-existing unrelated untracked files.

- [ ] **Step 3: Push the cleaned branch**

```powershell
git push origin agent/recent-bim-updates
```

Do not push `main`.

- [ ] **Step 4: Update PR title and body**

Set PR #1 to describe only:

- energy PDF whole-page and crop-region recognition;
- topology repair support gates and search budgets;
- shared PDF fallback improvements used by the platform;
- the final verification commands and results;
- explicit exclusion of annotation, training, and model weights.

- [ ] **Step 5: Mark PR #1 ready for review**

```powershell
gh pr ready 1 --repo jiliguala777/bim-web
```

- [ ] **Step 6: Verify final GitHub state**

```powershell
gh pr view 1 --repo jiliguala777/bim-web `
  --json state,isDraft,url,baseRefName,headRefName,files
git ls-remote origin refs/heads/main refs/heads/agent/recent-bim-updates
```

Expected:

- PR state is `OPEN` and `isDraft` is `false`.
- base remains `main`; head remains `agent/recent-bim-updates`.
- no PR file begins with `annotation_tool/` or `training/`.
- remote `main` has not moved during this procedure.
