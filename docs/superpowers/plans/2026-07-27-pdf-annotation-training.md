# PDF Annotation and Single-Page Fine-Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local reusable multi-page PDF annotation tool that shares the website preprocessing pipeline, exports confirmed skeleton-style masks, fine-tunes the current four-class U-Net on one selected page, exports a verified ONNX model, and compares it with the current model.

**Architecture:** Extract deterministic PDF-page preparation into a shared module used by the website and the local annotation application. Keep annotation projects and models outside Git, expose a local-only Flask UI for project/page/annotation operations, and keep PyTorch training in an isolated `.venv-train` command-line workflow. Reuse the existing strict ONNX export utility and website `ONNX_MODEL_PATH` override.

**Tech Stack:** Python 3.12, Flask, NumPy, OpenCV, pdfplumber, pdf2image/Poppler, RapidOCR, ONNX Runtime, PyTorch, segmentation-models-pytorch, Albumentations, vanilla JavaScript, HTML canvas, `unittest`.

## Global Constraints

- Tool code lives under `G:\bim-web\annotation_tool`; training code lives under `G:\bim-web\training`.
- Source PDFs, annotations, exports, and experiment models live under `G:\bim网页\标注数据` by default and must never be committed.
- Existing `G:\bim网页\标注工具\data` and its 219 image/mask pairs are read-only compatibility data and must not be migrated or overwritten.
- The annotation service binds only to `127.0.0.1`.
- Classes remain `0=background`, `1=wall`, `2=window`, `3=door`.
- Labels use model-compatible skeleton strokes; wall strokes target 4–6 pixels in 512 model space.
- Website and annotation tool must share one deterministic PDF-page preparation implementation.
- Current production inference remains fixed `[1,3,512,512]`, full-page context, and ROI-outside clearing.
- Only pages with `confirmed` annotation status are exportable.
- Training uses a separate `.venv-train`, starts from `models/M2_pub_plus_user.pt`, and never overwrites current `.pt` or `.onnx` files.
- The same-page experiment is reported as `single_page_overfit`, not as a generalization result.
- Production defaults and server deployment are out of scope.

---

## File Map

### Shared preprocessing

- Create `floorplan_page_pipeline.py`: prepare one selected PDF page and return deterministic artifacts plus metadata.
- Modify `web_server_server.py`: call the shared page pipeline from `/energy/ai_recognize`.
- Modify `floorplan_onnx.py`: expose a public deterministic method that creates the pre-letterbox model view and letterboxed 512 tensor metadata without running inference.
- Test `tests/test_floorplan_page_pipeline.py`: page selection, artifacts, hashes, and website parity.
- Modify `tests/test_energy_template.py`: assert the route delegates to the shared pipeline.

### Annotation application

- Create `annotation_tool/__init__.py`: package marker and app factory export.
- Create `annotation_tool/config.py`: resolve and validate the external data root.
- Create `annotation_tool/storage.py`: projects, manifests, atomic JSON/mask versions, and safe paths.
- Create `annotation_tool/services.py`: PDF import, page preparation, ONNX preannotation, validation, confirmation, and export.
- Create `annotation_tool/app.py`: local Flask routes.
- Create `annotation_tool/templates/index.html`: project/page/annotation shell.
- Create `annotation_tool/static/app.js`: canvas editing, page selection, autosave, undo/redo, and view switching.
- Create `annotation_tool/static/style.css`: local annotation UI styles.
- Create `annotation_tool/start_annotation_tool.ps1`: local launcher.
- Create `annotation_tool/README.md`: setup and operating instructions.
- Test `tests/test_annotation_storage.py`: safe paths, manifests, versioning, and confirmed-only export.
- Test `tests/test_annotation_app.py`: API contract and local bind defaults.

### Training and comparison

- Create `training/__init__.py`: package marker.
- Create `training/dataset.py`: load and validate exported single-page datasets.
- Create `training/losses.py`: Dice plus weighted cross-entropy.
- Create `training/train_floorplan.py`: strict checkpoint loading, deterministic fine-tuning, logging, and checkpoints.
- Create `training/evaluate.py`: per-class metrics and comparison image generation.
- Create `training/run_experiment.py`: train, export with `tools/export_floorplan_onnx.py`, verify, and compare.
- Create `training/requirements-training.txt`: constrained training dependencies.
- Create `training/setup_training_env.ps1`: create `.venv-train` without touching `.venv`.
- Test `tests/test_training_dataset.py`: dataset validation and transformations.
- Test `tests/test_training_losses.py`: finite loss, perfect-prediction Dice, and gradients.
- Test `tests/test_training_experiment.py`: short CPU smoke experiment using a tiny injected model.

### Repository integration

- Modify `.gitignore`: ignore external-data links, annotation runtime state, `.venv-train`, checkpoints, ONNX experiment outputs, and generated previews.
- Modify `README.md`: document the annotation and training workflow.

---

### Task 1: Shared model-input preparation

**Files:**
- Create: `floorplan_page_pipeline.py`
- Modify: `floorplan_onnx.py`
- Test: `tests/test_floorplan_page_pipeline.py`

**Interfaces:**
- Produces: `PreparedFloorplanPage`, `prepare_pdf_page(...)`, `FloorplanSegmenterONNX.prepare_model_input(...)`.
- `PreparedFloorplanPage` fields: `page_number`, `page_count`, `render_bgr`, `cleaned_bgr`, `model_view_rgb`, `model_input_512`, `model_input_metadata`, `scale_calibration`, `vector_cleanup`, `inference_roi`, `artifacts`.
- `prepare_pdf_page(pdf_path: Path, page_number: int, *, poppler_path: str | None, segmenter: FloorplanSegmenterONNX, output_dir: Path | None = None) -> PreparedFloorplanPage`.
- `FloorplanSegmenterONNX.prepare_model_input(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]` returns full-resolution RGB model view, normalized `(1,3,512,512)` float32 tensor, and resize/padding metadata.

- [ ] **Step 1: Write failing model-input tests**

```python
class ModelInputPreparationTests(unittest.TestCase):
    def test_prepare_model_input_returns_exact_letterbox_metadata(self):
        segmenter = FloorplanSegmenterONNX.__new__(FloorplanSegmenterONNX)
        segmenter.img_size = 512
        image = np.zeros((200, 400, 3), dtype=np.uint8)
        model_view, tensor, metadata = segmenter.prepare_model_input(image)
        self.assertEqual(model_view.shape, (200, 400, 3))
        self.assertEqual(tensor.shape, (1, 3, 512, 512))
        self.assertEqual(metadata["resized_size"], [512, 256])
        self.assertEqual(metadata["padding"], [128, 0, 128, 0])
```

- [ ] **Step 2: Run the model-input test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_page_pipeline.ModelInputPreparationTests
```

Expected: error because `prepare_model_input` does not exist.

- [ ] **Step 3: Extract deterministic preparation from `_run_inference`**

Add `prepare_model_input` to `FloorplanSegmenterONNX`; make `_run_inference` call it and preserve current numerical behavior. Use `cv2.INTER_LINEAR` for images, fixed ImageNet mean/std, and the existing black letterbox.

- [ ] **Step 4: Verify model-input tests and existing inference tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_page_pipeline.ModelInputPreparationTests tests.test_floorplan_rooms tests.test_compare_floorplan_preprocessing
```

Expected: all pass.

- [ ] **Step 5: Write failing selected-PDF-page test**

Create a two-page ReportLab PDF with distinguishable geometry and assert `prepare_pdf_page(..., page_number=2)` returns `page_number == 2`, `page_count == 2`, a 512 tensor, SHA-256 artifact metadata, and no page-one pixels.

- [ ] **Step 6: Run selected-page test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_page_pipeline.PdfPagePipelineTests
```

Expected: import error for `floorplan_page_pipeline`.

- [ ] **Step 7: Implement `PreparedFloorplanPage` and `prepare_pdf_page`**

Move the deterministic PDF preparation sequence from the energy route into the new module:

```python
@dataclass(frozen=True)
class PreparedFloorplanPage:
    page_number: int
    page_count: int
    render_bgr: np.ndarray
    cleaned_bgr: np.ndarray
    model_view_rgb: np.ndarray
    model_input_512: np.ndarray
    model_input_metadata: dict[str, Any]
    scale_calibration: dict[str, Any]
    vector_cleanup: dict[str, Any]
    inference_roi: list[int] | None
    artifacts: dict[str, str]
```

The module must validate 1-based page bounds, select only one page with `first_page=last_page`, use the current 100-DPI vector/200-DPI scanned behavior, reuse `vector_pdf_scale` and `floorplan_ocr`, and write artifacts atomically when `output_dir` is provided.

- [ ] **Step 8: Run pipeline tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_floorplan_page_pipeline tests.test_vector_pdf_scale
```

Expected: all pass.

- [ ] **Step 9: Commit shared preparation**

```powershell
git add floorplan_page_pipeline.py floorplan_onnx.py tests/test_floorplan_page_pipeline.py
git commit -m "feat: share floorplan PDF page preparation"
```

### Task 2: Website parity refactor

**Files:**
- Modify: `web_server_server.py:1550-1730`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: `prepare_pdf_page(...) -> PreparedFloorplanPage`.
- Produces: unchanged `/energy/ai_recognize` response schema and unchanged saved recognition artifacts.

- [ ] **Step 1: Add a failing delegation test**

Patch `web_server_server.prepare_pdf_page`, submit a prepared PDF token and selected page, and assert the function receives the requested page and target output directory. Assert `_floorplan_segmenter.predict` still receives `inference_roi`, `preserve_full_context=True`, and topology settings.

- [ ] **Step 2: Run the route test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template.EnergyRouteTests.test_pdf_route_uses_shared_page_pipeline
```

Expected: failure because the route still contains inline preparation.

- [ ] **Step 3: Refactor `/energy/ai_recognize`**

Import `prepare_pdf_page`, replace only the PDF preparation/cleanup block, and map `PreparedFloorplanPage` fields back to the current route variables. Leave PNG/JPG handling and recognition response fields unchanged.

- [ ] **Step 4: Run all recognition and PDF tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_energy_template tests.test_vector_pdf_scale tests.test_floorplan_rooms
```

Expected: all pass and no response-schema changes.

- [ ] **Step 5: Commit website parity refactor**

```powershell
git add web_server_server.py tests/test_energy_template.py
git commit -m "refactor: reuse PDF page preparation in recognition"
```

### Task 3: External project storage and versioned annotations

**Files:**
- Create: `annotation_tool/__init__.py`
- Create: `annotation_tool/config.py`
- Create: `annotation_tool/storage.py`
- Modify: `.gitignore`
- Test: `tests/test_annotation_storage.py`

**Interfaces:**
- Produces: `AnnotationConfig.from_env()`, `AnnotationStore`.
- `AnnotationStore.import_source(pdf_path) -> SourceRecord`.
- `AnnotationStore.create_project(name, source_sha256) -> ProjectRecord`.
- `AnnotationStore.save_page_manifest(project_id, page_number, manifest)`.
- `AnnotationStore.save_mask_version(project_id, page_number, mask, *, status, author) -> MaskVersion`.
- `AnnotationStore.load_current_mask(project_id, page_number) -> tuple[np.ndarray, MaskVersion] | None`.
- `AnnotationStore.list_confirmed_pages(project_id) -> list[PageRecord]`.

- [ ] **Step 1: Write failing safe-path and version tests**

Test that the default root is `G:\bim网页\标注数据`, `../` project IDs are rejected, identical PDFs deduplicate by SHA-256, a second mask save preserves the first version, and a `draft` page is absent from `list_confirmed_pages`.

- [ ] **Step 2: Run storage tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_annotation_storage
```

Expected: import error for `annotation_tool.storage`.

- [ ] **Step 3: Implement config and atomic storage**

Use `ANNOTATION_DATA_ROOT` as an optional override. Resolve every generated path and require it to remain under the configured root. Write JSON and `.npy` via a temporary sibling followed by `os.replace`. Copy source PDFs once into `sources/<sha256>.pdf`.

- [ ] **Step 4: Add manifest schemas**

Project manifest fields: `schema_version`, `project_id`, `name`, `source_sha256`, `source_original_name`, `page_count`, `pages`, `created_at`, `updated_at`.

Mask version fields: `version_id`, `status`, `author`, `created_at`, `mask_sha256`, `mask_path`, `previous_version_id`.

- [ ] **Step 5: Run storage tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_annotation_storage
```

Expected: all pass.

- [ ] **Step 6: Commit storage**

```powershell
git add annotation_tool/__init__.py annotation_tool/config.py annotation_tool/storage.py tests/test_annotation_storage.py .gitignore
git commit -m "feat: add versioned annotation project storage"
```

### Task 4: Annotation service and local API

**Files:**
- Create: `annotation_tool/services.py`
- Create: `annotation_tool/app.py`
- Test: `tests/test_annotation_app.py`

**Interfaces:**
- Consumes: `AnnotationStore`, `prepare_pdf_page`, `get_segmenter`.
- Produces: `AnnotationService` and `create_app(config: AnnotationConfig | None = None) -> Flask`.
- Routes:
  - `GET /api/projects`
  - `POST /api/projects`
  - `GET /api/projects/<project_id>/pages`
  - `POST /api/projects/<project_id>/pages/prepare`
  - `GET /api/projects/<project_id>/pages/<int:page>/artifact/<name>`
  - `GET /api/projects/<project_id>/pages/<int:page>/mask`
  - `POST /api/projects/<project_id>/pages/<int:page>/mask`
  - `POST /api/projects/<project_id>/pages/<int:page>/confirm`
  - `POST /api/projects/<project_id>/pages/<int:page>/preannotate`
  - `POST /api/projects/<project_id>/export`

- [ ] **Step 1: Write failing API contract tests**

Use a temporary data root and Flask test client. Assert PDF upload creates a project, invalid page numbers return 400, artifact paths cannot escape the project, and draft mask saves do not confirm a page.

- [ ] **Step 2: Run API tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_annotation_app
```

Expected: import error for `annotation_tool.app`.

- [ ] **Step 3: Implement project and preparation service**

`AnnotationService.create_project(uploaded_file, name)` streams the PDF to a temporary file, validates it with pdfplumber, imports by SHA-256, creates/reuses a project, and returns page count.

`prepare_page(project_id, page_number)` calls the shared pipeline, writes artifacts under `projects/<id>/pages/<page>/`, and records hashes and preprocessing version.

- [ ] **Step 4: Implement mask validation and save**

Decode an indexed PNG or integer array, require 2D `uint8`, values within `0..3`, and exact full-resolution dimensions. Derive `mask_512.npy` using recorded nearest-neighbor transform. Return warnings for disappearing classes, extreme class ratios, and likely wall gaps.

- [ ] **Step 5: Implement ONNX preannotation**

Run the current segmenter on the prepared cleaned page with recorded ROI/full-context settings, save the result as a `preannotated` mask version, and record model file SHA-256. Allow manual blank-mask creation when inference fails.

- [ ] **Step 6: Implement confirmed-only dataset export**

Write `images`, `masks`, `manifest.json`, and `dataset_report.json` under `exports/<slug>-<timestamp>`. Reject export when no confirmed pages exist. Mark a one-page export as `single_page_overfit`.

- [ ] **Step 7: Implement Flask routes and local defaults**

The module CLI accepts `--host` but rejects anything except `127.0.0.1` and `localhost`; default port is `8099`. Configure `MAX_CONTENT_LENGTH` and return JSON errors without filesystem paths outside the data root.

- [ ] **Step 8: Run API and storage tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_annotation_app tests.test_annotation_storage
```

Expected: all pass.

- [ ] **Step 9: Commit service and API**

```powershell
git add annotation_tool/services.py annotation_tool/app.py tests/test_annotation_app.py
git commit -m "feat: add local PDF annotation API"
```

### Task 5: Browser annotation UI

**Files:**
- Create: `annotation_tool/templates/index.html`
- Create: `annotation_tool/static/app.js`
- Create: `annotation_tool/static/style.css`
- Create: `annotation_tool/start_annotation_tool.ps1`
- Modify: `annotation_tool/app.py`
- Test: `tests/test_annotation_app.py`

**Interfaces:**
- Consumes: Task 4 HTTP routes.
- Produces: local project import, page selection, view switcher, editable canvas, preannotation, autosave, confirmation, and export controls.

- [ ] **Step 1: Add failing template contract test**

Assert `/` contains `pdf-file`, `page-select`, `view-original`, `view-cleaned`, `view-model`, `mask-canvas`, tool buttons with class IDs `1..3`, `undo`, `redo`, `save`, `confirm`, `preannotate`, and `export`.

- [ ] **Step 2: Run template test and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_annotation_app.AnnotationTemplateTests
```

Expected: 404 or missing required element IDs.

- [ ] **Step 3: Build the static UI shell**

Implement a three-column layout: projects/pages, canvas workspace, tools/status. Keep all scripts local; do not load CDN assets.

- [ ] **Step 4: Implement canvas coordinate model**

Maintain full-resolution offscreen image and mask canvases. Apply one viewport transform for zoom/pan. Store strokes in image coordinates. Convert desired model-space brush width to full-resolution pixels using `model_input_metadata.resize_scale`.

- [ ] **Step 5: Implement editing history and autosave**

Record one compressed `ImageData` snapshot per completed stroke, cap undo/redo at 50 entries, debounce autosave for 1.5 seconds, and display dirty/saving/saved/error states. Confirmation always performs an immediate save first.

- [ ] **Step 6: Implement view and 512 preview**

Switch background among `page_render`, `cleaned_page`, and `model_view` without changing mask coordinates. Request/render `mask_512` after save and warn if any nonzero class present in the full mask disappears at 512.

- [ ] **Step 7: Add launcher**

`start_annotation_tool.ps1` resolves the project `.venv`, sets `ANNOTATION_DATA_ROOT` only when supplied, and runs:

```powershell
python -m annotation_tool.app --host 127.0.0.1 --port 8099
```

- [ ] **Step 8: Run template and API tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_annotation_app
```

Expected: all pass.

- [ ] **Step 9: Browser smoke test**

Start the tool locally, import a generated two-page test PDF, select page two, preannotate, draw all three classes, undo/redo, reload the page, confirm, and export. Capture one screenshot for local inspection but do not commit runtime data.

- [ ] **Step 10: Commit the UI**

```powershell
git add annotation_tool/templates/index.html annotation_tool/static/app.js annotation_tool/static/style.css annotation_tool/start_annotation_tool.ps1 annotation_tool/app.py tests/test_annotation_app.py
git commit -m "feat: add PDF annotation workspace"
```

### Task 6: Training dataset and loss

**Files:**
- Create: `training/__init__.py`
- Create: `training/dataset.py`
- Create: `training/losses.py`
- Test: `tests/test_training_dataset.py`
- Test: `tests/test_training_losses.py`

**Interfaces:**
- Produces: `ExportedFloorplanDataset`, `validate_export`, `CombinedSegmentationLoss`, `dice_per_class`.
- `ExportedFloorplanDataset(root: Path, *, augment: bool, seed: int)`.
- Samples return `{"image": FloatTensor[3,512,512], "mask": LongTensor[512,512], "sample_id": str}`.

- [ ] **Step 1: Write failing dataset validation tests**

Create a temporary export with one 512 PNG, one `.npy` mask and manifest. Assert loading succeeds; missing files, wrong mask shape, values outside `0..3`, or non-confirmed manifest entries raise descriptive `ValueError`.

- [ ] **Step 2: Run dataset tests and verify failure**

Run in the training environment when available:

```powershell
.\.venv-train\Scripts\python.exe -m unittest tests.test_training_dataset
```

Expected: import error for `training.dataset`.

- [ ] **Step 3: Implement deterministic dataset loading**

Load already prepared 512 model images without applying website preprocessing again. Normalize with the same ImageNet constants. Use nearest-neighbor interpolation for masks.

- [ ] **Step 4: Add single-page augmentation**

Use seeded Albumentations transforms limited to flips, 90-degree rotation, small affine scale/translation, light brightness/contrast, blur/compression, and morphological line-width variation applied consistently to masks.

- [ ] **Step 5: Write failing loss tests**

Assert perfect logits produce per-class Dice near 1, combined loss is finite on an all-background sample, ignored absent-class reporting is explicit, and `.backward()` creates finite gradients.

- [ ] **Step 6: Implement losses and metrics**

Use weighted `torch.nn.CrossEntropyLoss` plus multiclass soft Dice. Default class weights are derived from the export report with bounded inverse frequency; serialize the final weights into experiment metadata.

- [ ] **Step 7: Run dataset and loss tests**

Run:

```powershell
.\.venv-train\Scripts\python.exe -m unittest tests.test_training_dataset tests.test_training_losses
```

Expected: all pass.

- [ ] **Step 8: Commit dataset and loss**

```powershell
git add training/__init__.py training/dataset.py training/losses.py tests/test_training_dataset.py tests/test_training_losses.py
git commit -m "feat: add floorplan fine-tuning dataset and loss"
```

### Task 7: Deterministic fine-tuning and experiment runner

**Files:**
- Create: `training/train_floorplan.py`
- Create: `training/evaluate.py`
- Create: `training/run_experiment.py`
- Create: `training/requirements-training.txt`
- Create: `training/setup_training_env.ps1`
- Test: `tests/test_training_experiment.py`

**Interfaces:**
- Consumes: Task 6 dataset/loss and `tools.export_floorplan_onnx.export_onnx`.
- Produces: `TrainingConfig`, `train(config) -> TrainingResult`, `evaluate_checkpoint(...)`, `run_experiment(...)`.

- [ ] **Step 1: Write failing checkpoint-compatibility and smoke tests**

Inject a tiny four-class convolution model via a `model_factory` parameter, run two CPU epochs on one synthetic page, and assert `last.pt`, `best.pt`, `training_log.csv`, and `metrics.json` exist. Separately assert the real factory builds the exact U-Net and strict-loads `model_state_dict`.

- [ ] **Step 2: Run experiment tests and verify failure**

Run:

```powershell
.\.venv-train\Scripts\python.exe -m unittest tests.test_training_experiment
```

Expected: import error for `training.train_floorplan`.

- [ ] **Step 3: Implement training configuration and strict model factory**

`TrainingConfig` includes dataset path, checkpoint, output directory, seed, epochs, learning rate, weight decay, patience, device, and augmentation count. Default seed is `20260727`; default learning rate is `1e-4`; every value is written to `metrics.json`.

- [ ] **Step 4: Implement the training loop**

Set Python/NumPy/Torch seeds, enable deterministic algorithms where supported, load the exact checkpoint strictly, train with AdamW, clip gradients, log each epoch atomically, and stop on plateau. For `single_page_overfit`, select `best.pt` using lowest smoothed training loss and record that no independent validation set exists.

- [ ] **Step 5: Implement evaluation and comparison**

Generate per-class IoU/Dice/confusion matrix for old and new checkpoints against the same manual mask. Produce `old_model_overlay.png`, `new_model_overlay.png`, `ground_truth.png`, and a labeled horizontal `comparison.png`.

- [ ] **Step 6: Implement ONNX orchestration**

Call the existing `export_onnx(best.pt, best.onnx)`, preserve its checker/parity report as `export_report.json`, then run ONNX Runtime on the sample and confirm the reported class mask matches PyTorch at the configured threshold.

- [ ] **Step 7: Implement training environment setup**

`requirements-training.txt` contains these explicit compatible ranges:

```text
torch>=2.6,<3
torchvision>=0.21,<1
segmentation-models-pytorch>=0.5,<1
albumentations>=2,<3
onnx>=1.17,<2
onnxruntime>=1.20,<2
numpy>=2,<3
opencv-python-headless>=4.10,<5
pillow>=11,<13
```

`setup_training_env.ps1` creates `.venv-train`, upgrades pip, installs the requirements, and prints CPU/CUDA availability without changing `.venv`.

- [ ] **Step 8: Run training tests**

Run:

```powershell
.\.venv-train\Scripts\python.exe -m unittest tests.test_training_experiment tests.test_export_floorplan_onnx
```

Expected: all pass.

- [ ] **Step 9: Commit training runner**

```powershell
git add training/train_floorplan.py training/evaluate.py training/run_experiment.py training/requirements-training.txt training/setup_training_env.ps1 tests/test_training_experiment.py
git commit -m "feat: add deterministic floorplan fine-tuning"
```

### Task 8: Documentation and end-to-end verification

**Files:**
- Create: `annotation_tool/README.md`
- Modify: `README.md`
- Modify: `tests/test_deployment_assets.py`

**Interfaces:**
- Consumes: all previous tasks.
- Produces: reproducible setup, annotation, training, comparison, and local website verification instructions.

- [ ] **Step 1: Add failing documentation/asset tests**

Assert the annotation launcher, training setup, training requirements, local-only bind text, external-data warning, and `ONNX_MODEL_PATH` test command exist.

- [ ] **Step 2: Run documentation tests and verify failure**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_deployment_assets
```

Expected: failure for missing new assets or documentation text.

- [ ] **Step 3: Write operator documentation**

Document:

```powershell
.\annotation_tool\start_annotation_tool.ps1
.\training\setup_training_env.ps1
.\.venv-train\Scripts\python.exe -m training.run_experiment `
  --dataset "G:\bim网页\标注数据\exports\four-floor-building-floor1-20260727T160000" `
  --checkpoint ".\models\M2_pub_plus_user.pt" `
  --output "G:\bim网页\标注数据\models\four-floor-building-floor1-overfit"
$env:ONNX_MODEL_PATH="G:\bim网页\标注数据\models\four-floor-building-floor1-overfit\best.onnx"
.\start_local.ps1
```

Explain that the same-page result demonstrates fitting only, and production deployment requires a separate user decision.

- [ ] **Step 4: Run the complete non-training suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected: all tests pass.

- [ ] **Step 5: Run the complete training suite**

Run:

```powershell
.\.venv-train\Scripts\python.exe -m unittest tests.test_training_dataset tests.test_training_losses tests.test_training_experiment tests.test_export_floorplan_onnx
```

Expected: all tests pass.

- [ ] **Step 6: Perform real first-floor acceptance**

With the user-provided four-floor PDF:

1. Import the PDF without modifying the source.
2. Select the first-floor page.
3. Confirm website/annotation model-input SHA-256 equality.
4. Generate ONNX preannotation.
5. Let the user correct and confirm the skeleton mask.
6. Export one-page dataset marked `single_page_overfit`.
7. Run fine-tuning and ONNX export.
8. Generate comparison images and metrics.
9. Start the local website with the experiment ONNX.
10. Re-upload the same PDF/page and save the resulting overlay.

- [ ] **Step 7: Commit documentation**

```powershell
git add annotation_tool/README.md README.md tests/test_deployment_assets.py
git commit -m "docs: add PDF annotation and training workflow"
```

- [ ] **Step 8: Final verification report**

Record exact test commands, pass counts, selected PDF/page, input hashes, old/new per-class metrics, model output paths, and any unmet acceptance item. Do not claim completion if the real first-floor PDF or manual annotation has not yet been supplied.
