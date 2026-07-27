# Floorplan ONNX Export and Validation Design

## Goal

Export `models/M2_pub_plus_user.pt` as `models/M2_pub_plus_user.onnx`, prove that the exported graph is numerically equivalent to the PyTorch checkpoint, and compare the original annotation-tool preprocessing with the website's current preprocessing on the cultural-palace fourth-floor image.

## Scope and constraints

- Reconstruct exactly `smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=4)`.
- Load the checkpoint's `model_state_dict` strictly and never overwrite the `.pt` file.
- Export a fixed `(1, 3, 512, 512)` model with opset 17, input name `input`, and output name `logits`.
- Keep `ai_simulate` unchanged. After the user reviews and accepts the comparison image, point both website defaults directly to `models/M2_pub_plus_user.onnx`; do not create a duplicate file named `M2_DA_best.onnx`.
- Add only focused scripts, tests, documentation, the exported ONNX file, and validation artifacts. Preserve all unrelated dirty-worktree content.
- Use a dedicated clean environment at `C:\Users\Jason\.codex\venvs\bim-web-onnx`; do not inherit native packages from or modify the Anaconda base environment. Keep `torch 2.5.1+cpu` and `torchvision 0.20.1+cpu` fixed.

## Components

### Export and equivalence tool

`tools/export_floorplan_onnx.py` owns model construction, strict checkpoint loading, ONNX export, isolated ONNX checker execution, and deterministic PyTorch/ONNX Runtime comparison. The native checker runs in a child process so a Windows DLL access violation cannot kill the export process. A checker failure is recorded and blocks by default; an explicit `--allow-checker-failure` permits numerical validation to continue when the same native crash is independently reproduced on a minimal ONNX graph. The tool reports input/output metadata, maximum and mean absolute error, and argmax pixel agreement. A non-equivalent export always exits unsuccessfully.

### Preprocessing comparison tool

`tools/compare_floorplan_preprocessing.py` runs two explicit pipelines against the same ONNX session:

- `annotation_tool`: BGR-to-RGB, aspect-preserving resize, centered black 512-square letterbox, ImageNet normalization, crop, and nearest-neighbor restoration.
- `website_current`: the current annotation-removal and `preprocess_dark_cad()` transformations, followed by direct 512-square stretching and ImageNet normalization. Its reported mask is the raw four-class argmax so the comparison measures input preprocessing rather than unrelated website postprocessing heuristics.

For each pipeline the tool saves a class-index mask PNG, a color overlay PNG, and statistics for class pixels, percentages, and external contours. It also writes one JSON report containing paths and source-image metadata.

## Data flow

The checkpoint is loaded once to export the graph. A deterministic tensor is then passed through both runtimes. After equivalence succeeds, the ONNX graph is loaded once and receives the two independently prepared versions of the fourth-floor image. Predictions are restored to the original image dimensions before statistics and visual output are generated.

After visual approval, `web_server_server.py` uses `models/M2_pub_plus_user.onnx` as the default while preserving the `ONNX_MODEL_PATH` environment-variable override. Direct construction of `FloorplanSegmenterONNX()` resolves the same file under the project `models` directory.

## Error handling

The tools fail with clear messages for missing files, malformed checkpoints, non-strict state dictionaries, unexpected model I/O shapes, unavailable runtime dependencies, unreadable images, failed ONNX checking, or equivalence metrics outside tolerance. Output directories are created only under the requested validation-artifact path.

## Verification

- Unit tests first cover checkpoint validation, deterministic comparison metrics, letterbox geometry/restoration, class statistics, and output naming.
- The real export must attempt `onnx.checker` in isolation, record its result, expose `input` and `logits`, and produce `(1, 4, 512, 512)`. If the native checker crashes even on a minimal identity graph, ONNX Runtime graph loading is the required structural fallback and the checker limitation must be reported rather than hidden.
- PyTorch and ONNX Runtime must meet `max_abs_error <= 1e-4` and argmax agreement `>= 0.99999` on the deterministic verification input.
- The real cultural-palace image run must produce both masks, both overlays, and a machine-readable JSON report without modifying website defaults.
