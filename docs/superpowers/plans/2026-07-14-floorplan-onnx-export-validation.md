# Floorplan ONNX Export and Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the four-class floorplan checkpoint to ONNX, prove runtime equivalence, and generate a two-pipeline visual comparison on the cultural-palace fourth-floor screenshot.

**Architecture:** One focused export tool builds and validates the model graph; one comparison tool owns preprocessing, inference restoration, statistics, and artifact generation. Tests exercise pure helpers before the real 98 MB checkpoint and ONNX Runtime are used end to end.

**Tech Stack:** Python 3.12, PyTorch 2.5.1 CPU, torchvision 0.20.1, segmentation-models-pytorch 0.3.4, ONNX 1.20.1, ONNX Runtime 1.20.1, OpenCV, NumPy, unittest.

## Global Constraints

- Do not overwrite `models/M2_pub_plus_user.pt`.
- Export only to `models/M2_pub_plus_user.onnx` with opset 17, input `input`, output `logits`.
- Keep `ai_simulate` unchanged. After visual approval, point both website defaults directly to `models/M2_pub_plus_user.onnx` while preserving the `ONNX_MODEL_PATH` override.
- Do not inherit native packages from or modify the Anaconda base environment; use `C:\Users\Jason\.codex\venvs\bim-web-onnx` with `torch 2.5.1+cpu`.
- Do not reset, clean, overwrite, or bulk-commit the existing dirty worktree.

---

### Task 1: Export tool and unit tests

**Files:**
- Create: `tests/test_export_floorplan_onnx.py`
- Create: `tools/export_floorplan_onnx.py`

**Interfaces:**
- Consumes: a checkpoint path containing `model_state_dict`, an ONNX output path, and dependency modules.
- Produces: `build_model()`, `load_checkpoint_model(path)`, `compare_outputs(torch_output, onnx_output)`, `export_onnx(...)`, and a CLI JSON summary.

- [ ] **Step 1: Write failing tests** for rejecting a checkpoint without `model_state_dict`, detecting shape mismatches, and computing max/mean error plus argmax agreement.
- [ ] **Step 2: Run `python -m unittest tests.test_export_floorplan_onnx -v`** and confirm failure because the export module does not exist.
- [ ] **Step 3: Implement the minimal pure helpers and strict model loader** with the exact four-class ResNet34 U-Net architecture.
- [ ] **Step 4: Re-run the focused tests** and confirm they pass.
- [ ] **Step 5: Add export/checker/runtime comparison behavior** with fixed seed, shape validation, thresholds, and JSON output.
- [ ] **Step 6: Add focused tests for export metadata and threshold failures** using small real tensors and dependency injection where full export is unnecessary.
- [ ] **Step 7: Run the focused tests and full existing unittest suite.**

### Task 2: Dual-preprocessing comparison and unit tests

**Files:**
- Create: `tests/test_compare_floorplan_preprocessing.py`
- Create: `tools/compare_floorplan_preprocessing.py`

**Interfaces:**
- Consumes: an ONNX path, a source image path, and an output directory.
- Produces: `prepare_annotation_tool_input(image)`, `restore_annotation_tool_mask(mask, metadata)`, `prepare_website_input(image)`, `restore_stretched_mask(mask, size)`, `compute_class_stats(mask)`, and comparison artifacts/report.

- [ ] **Step 1: Write failing tests** for aspect-preserving letterbox placement, nearest-neighbor restoration, direct-stretch dimensions, and class/contour statistics.
- [ ] **Step 2: Run `python -m unittest tests.test_compare_floorplan_preprocessing -v`** and confirm failure because the comparison module does not exist.
- [ ] **Step 3: Implement the pure preprocessing and statistics helpers** using the exact formulas from `preannotate.py` and current `floorplan_onnx.py`.
- [ ] **Step 4: Re-run focused tests** and confirm they pass.
- [ ] **Step 5: Implement one-session ONNX inference, mask/overlay writers, and JSON report generation.**
- [ ] **Step 6: Add and run an artifact-generation test** using a small synthetic source image and injected prediction arrays.
- [ ] **Step 7: Run both focused modules and the full existing unittest suite.**

### Task 3: Isolated dependencies and real export

**Files:**
- Create locally: `C:\Users\Jason\.codex\venvs\bim-web-onnx` (clean dedicated environment, not committed)
- Create: `models/M2_pub_plus_user.onnx`

**Interfaces:**
- Consumes: `models/M2_pub_plus_user.pt` with SHA-256 `3579FEE061780790416E41005FC02C9DFC5EEC905E53063CC6D9D5D1506A123F`.
- Produces: checked ONNX graph and numerical-equivalence JSON output.

- [x] **Step 1: Create a clean dedicated virtual environment that does not inherit the base environment's native packages.**
- [x] **Step 2: Install pinned `torch==2.5.1+cpu`, `torchvision==0.20.1+cpu`, segmentation-models-pytorch 0.3.4, ONNX 1.20.1, and ONNX Runtime 1.20.1.**
- [x] **Step 3: Verify exact torch/torchvision versions and imports.**
- [x] **Step 4: Run the export CLI, isolate and record the native checker result, require ONNX Runtime graph loading, `(1,4,512,512)` outputs, max absolute error at most `1e-4`, and argmax agreement at least `0.99999`.**
- [ ] **Step 5: Inspect the ONNX file hash, size, graph I/O names, shapes, and opset.**

### Task 4: Real cultural-palace image comparison

**Files:**
- Create: `artifacts/floorplan_onnx_validation/annotation_tool_mask.png`
- Create: `artifacts/floorplan_onnx_validation/annotation_tool_overlay.png`
- Create: `artifacts/floorplan_onnx_validation/website_current_mask.png`
- Create: `artifacts/floorplan_onnx_validation/website_current_overlay.png`
- Create: `artifacts/floorplan_onnx_validation/comparison.json`

**Interfaces:**
- Consumes: the verified ONNX graph and `C:\Users\Jason\AppData\Local\Temp\codex-clipboard-371fe826-1795-47ac-bf66-c6a7194cf63f.png`.
- Produces: two restored masks, two overlays, and comparable class/contour statistics.

- [ ] **Step 1: Run the comparison CLI with the verified ONNX model and source screenshot.**
- [ ] **Step 2: Validate that all five artifacts exist, are readable, and match the original image dimensions.**
- [ ] **Step 3: Visually inspect both overlays and masks for obvious corruption or alignment errors.**
- [ ] **Step 4: Run all tests and syntax checks again, then report metrics and show both overlays to the user without changing website configuration.**

### Task 5: Use the approved exported model by default

**Files:**
- Modify: `web_server_server.py:38`
- Modify: `floorplan_onnx.py:36-37`
- Modify: `tests/test_energy_template.py`

**Interfaces:**
- Consumes: the approved `models/M2_pub_plus_user.onnx` file.
- Produces: a server default that uses this model while retaining `ONNX_MODEL_PATH`, plus the same default for direct `FloorplanSegmenterONNX()` construction.

- [x] **Step 1: Add a failing source-level regression test** asserting that both defaults contain `M2_pub_plus_user.onnx`, the server retains `ONNX_MODEL_PATH`, and neither default contains `M2_DA_best.onnx`.
- [x] **Step 2: Run `python -m unittest tests.test_energy_template.EnergyTemplateTests.test_approved_onnx_model_is_the_default -v`** and confirm it fails on the old filename.
- [x] **Step 3: Change the server default to `os.path.join(MODELS_DIR, 'M2_pub_plus_user.onnx')` and the standalone segmenter default to `Path(__file__).parent / 'models' / 'M2_pub_plus_user.onnx'`.**
- [x] **Step 4: Re-run the focused test, then the full unittest suite and Python syntax checks.**
- [x] **Step 5: Start the server with the existing environment-variable credentials and verify `/energy/ai_status` reports the model available.**
