# 多页训练与标注区域裁剪实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用一、二、三层标注联合微调新模型，并让标注工具支持整页标注及同页多个具名裁剪区域。

**Architecture:** 第一阶段扩展现有训练入口，使其接受 `fine_tune` 多样本导出，并补充逐样本评估产物后运行真实三层训练。第二阶段在现有 `project → page` 数据模型下增加可选的 `region` 子资源；区域从服务端原始准备图裁剪并独立 letterbox、保存蒙版版本及导出训练样本，旧整页路径保持不变。

**Tech Stack:** Python 3、Flask、NumPy、OpenCV、PyTorch、ONNX Runtime、原生 JavaScript、`unittest`。

## Global Constraints

- 从 `G:\bim-web\models\M2_pub_plus_user.pt` 开始微调，不继承单页过拟合 checkpoint。
- 三层数据为 PDF 第13、14、15页，均为 confirmed。
- 训练输出和数据留在 `G:\bim网页\标注数据\`，不得覆盖网站正式 ONNX。
- 多页训练不宣称独立泛化能力。
- 已有整页标注和导出格式保持向后兼容，不迁移或改写用户数据。
- 区域裁剪使用服务端准备图，不使用浏览器截图。
- 第一版不提供区域删除或修改矩形。

---

### Task 1: 允许多样本 `fine_tune` 训练

**Files:**
- Modify: `training/train_floorplan.py`
- Modify: `tests/test_training_experiment.py`

**Interfaces:**
- Consumes: `validate_export(path) -> {"manifest", "sample_count", ...}`
- Produces: `_validate_experiment(validated) -> str`，返回真实 `experiment_type`

- [ ] **Step 1: 写失败测试**

在 `tests/test_training_experiment.py` 增加两个测试：

```python
def test_train_accepts_multi_sample_fine_tune_export(self):
    dataset = self._export_dataset(sample_count=2, experiment_type="fine_tune")
    result = train(self._config(dataset), model_factory=TinyModel)
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    self.assertEqual(metrics["experiment_type"], "fine_tune")
    self.assertEqual(result.epochs_completed, 2)

def test_train_rejects_single_sample_fine_tune_export(self):
    dataset = self._export_dataset(sample_count=1, experiment_type="fine_tune")
    with self.assertRaisesRegex(ValueError, "at least two"):
        train(self._config(dataset), model_factory=TinyModel)
```

同时保留现有单页测试，并增加不支持类型 `evaluation_only` 被拒绝的断言。

- [ ] **Step 2: 验证测试按预期失败**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_training_experiment -v
```

Expected: 新的 `fine_tune` 测试因当前硬编码拒绝而失败。

- [ ] **Step 3: 实现最小兼容逻辑**

在 `training/train_floorplan.py` 增加：

```python
def _validate_experiment(validated: dict) -> str:
    experiment_type = validated["manifest"].get("experiment_type")
    if experiment_type == "single_page_overfit":
        if validated["sample_count"] != 1:
            raise ValueError("single_page_overfit requires exactly one sample")
        return experiment_type
    if experiment_type == "fine_tune":
        if validated["sample_count"] < 2:
            raise ValueError("fine_tune requires at least two samples")
        return experiment_type
    raise ValueError(f"unsupported experiment_type: {experiment_type}")
```

`train()` 使用返回值，不再只允许 `single_page_overfit`。写入
`metrics.json` 时使用真实类型，但继续写入：

```python
"selection_basis": "smoothed_training_loss",
"has_independent_validation": False,
"generalization_claim": False,
```

- [ ] **Step 4: 运行训练测试**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_training_experiment -v
```

Expected: 全部通过。

- [ ] **Step 5: 提交**

```powershell
git add training/train_floorplan.py tests/test_training_experiment.py
git commit -m "feat: support multi-sample floorplan fine-tuning"
```

### Task 2: 保存逐样本评估产物

**Files:**
- Modify: `training/evaluate.py`
- Modify: `tests/test_training_experiment.py`

**Interfaces:**
- Consumes: `evaluate_checkpoint(checkpoint_path, dataset_path, output_dir)`
- Produces: `output_dir/samples/<safe_sample_id>/prediction.npy`、`overlay.png`、`ground_truth.png`

- [ ] **Step 1: 写失败测试**

创建两样本导出并调用 `evaluate_checkpoint`，断言报告中每个 sample 具有
`artifact_directory`，且每个目录包含三项产物。断言顶层兼容产物仍对应第一个
样本。

- [ ] **Step 2: 验证测试失败**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_training_experiment.TrainingExperimentTests.test_evaluation_writes_artifacts_for_every_sample -v
```

Expected: `samples/<sample_id>` 不存在。

- [ ] **Step 3: 实现逐样本输出**

在评估循环中对每个样本：

```python
sample_dir = destination / "samples" / _safe_sample_id(sample["sample_id"])
_write_image(sample_dir / "overlay.png", ...)
_write_image(sample_dir / "ground_truth.png", ...)
np.save(sample_dir / "prediction.npy", prediction, allow_pickle=False)
```

`_safe_sample_id` 只允许字母、数字、点、下划线和连字符，其他字符替换为
连字符，并拒绝空结果。报告样本记录增加相对 `artifact_directory`。

- [ ] **Step 4: 运行相关测试**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_training_experiment tests.test_export_floorplan_onnx -v
```

Expected: 全部通过。

- [ ] **Step 5: 提交**

```powershell
git add training/evaluate.py tests/test_training_experiment.py
git commit -m "feat: save per-sample training evaluation artifacts"
```

### Task 3: 运行真实三层微调及验证

**Files:**
- External data: `G:\bim网页\标注数据\exports\文化宫-1至3层-20260728T080402Z`
- External output: `G:\bim网页\标注数据\models\文化宫-1至3层-finetune-v2`

**Interfaces:**
- Consumes: 原始 checkpoint 和三样本联合导出
- Produces: `best.pt`、`best.onnx`、训练日志、汇总及逐层比较

- [ ] **Step 1: 验证联合导出**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -c "from training.dataset import validate_export; print(validate_export(r'G:\bim网页\标注数据\exports\文化宫-1至3层-20260728T080402Z')['sample_count'])"
```

Expected: `3`。

- [ ] **Step 2: 启动保守微调**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m training.run_experiment `
  --dataset "G:\bim网页\标注数据\exports\文化宫-1至3层-20260728T080402Z" `
  --checkpoint "G:\bim-web\models\M2_pub_plus_user.pt" `
  --output "G:\bim网页\标注数据\models\文化宫-1至3层-finetune-v2" `
  --epochs 20 --learning-rate 0.00005 --weight-decay 0.0001 `
  --patience 6 --seed 20260728 --device cpu --augmentation-count 2
```

Expected: 训练正常结束并生成 `best.pt`。

- [ ] **Step 3: 验证 ONNX**

确认 `export_report.json` 中 checker 为 `passed`，confirmed sample parity 为
`passed`，`argmax_pixel_agreement >= 0.99999`。

- [ ] **Step 4: 汇总逐层指标**

从 `comparison_metrics.json` 和 `new/samples/*` 读取第13、14、15页的
wall/window/door Dice、IoU、预测像素和人工像素。额外计算第13页最大墙体厚度
与上一轮 28.4 像素进行比较。

### Task 4: 增加区域存储模型

**Files:**
- Modify: `annotation_tool/storage.py`
- Modify: `tests/test_annotation_storage.py`

**Interfaces:**
- Produces:
  - `create_region(project_id, page_number, name, crop_bbox_px) -> dict`
  - `list_regions(project_id, page_number) -> list[dict]`
  - `save_region_mask_version(...) -> MaskVersion`
  - `load_current_region_mask(...)`
  - `list_confirmed_targets(project_id) -> list[AnnotationTarget]`

- [ ] **Step 1: 写失败存储测试**

覆盖：创建两个具名区域、重复名称拒绝、旧项目返回空区域、区域版本路径位于
`regions/<region_id>/versions`、整页版本路径保持不变。

- [ ] **Step 2: 验证失败**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_annotation_storage -v
```

Expected: 缺少区域方法。

- [ ] **Step 3: 实现区域记录与版本链**

区域 ID 使用 12 位随机十六进制字符串；名称去除首尾空白后长度限制为
1–80字符。所有路径继续经过 `_inside_root`。新增 `AnnotationTarget`
数据类统一表示整页和区域，但不得修改现有 `PageRecord` 的公开字段。

- [ ] **Step 4: 运行存储测试并提交**

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_annotation_storage -v
git add annotation_tool/storage.py tests/test_annotation_storage.py
git commit -m "feat: add annotation region storage"
```

### Task 5: 实现服务端区域裁剪与蒙版

**Files:**
- Modify: `annotation_tool/services.py`
- Modify: `tests/test_annotation_app.py`

**Interfaces:**
- Produces:
  - `create_region(project_id, page, name, crop_bbox_px)`
  - `region_artifact_path(...)`
  - `save_region_mask(...)`
  - `preannotate_region(...)`
  - `confirm_region(...)`

- [ ] **Step 1: 写失败服务测试**

使用 400×300 合成页面准备图创建 `[100, 50, 300, 250]` 区域，断言裁剪结果
为 200×200、512输入正确 letterbox、越界/过小/反向框被拒绝、保存蒙版尺寸必须
等于 200×200。

- [ ] **Step 2: 验证失败**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_annotation_app -v
```

Expected: 缺少区域服务。

- [ ] **Step 3: 实现原子裁剪**

先读取并校验所有源图，再在临时目录生成 `render.png`、`cleaned.png`、
`model_view.png`、`model_input_512.png` 和 `preprocessing.json`；全部成功后
原子移动到区域目录并写项目清单。复用现有 mask 解析、warning 和预标注逻辑，
但尺寸及元数据取自区域 preparation。

- [ ] **Step 4: 运行服务测试并提交**

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_annotation_app -v
git add annotation_tool/services.py tests/test_annotation_app.py
git commit -m "feat: prepare and annotate cropped regions"
```

### Task 6: 增加区域 API 和联合导出

**Files:**
- Modify: `annotation_tool/app.py`
- Modify: `annotation_tool/services.py`
- Modify: `tests/test_annotation_app.py`
- Modify: `tests/test_training_dataset.py`

**Interfaces:**
- Produces规格中定义的 `/regions` API
- Export manifest 区域样本增加 `source_page_number`、`region_id`、
  `region_name`、`crop_bbox_px`

- [ ] **Step 1: 写失败接口和导出测试**

通过 Flask test client 创建同页两个区域，分别保存并确认，导出后断言：

```python
self.assertEqual(report["sample_count"], 2)
self.assertEqual({s["region_name"] for s in manifest["samples"]}, {"四层", "五层"})
self.assertEqual(len({s["sample_id"] for s in manifest["samples"]}), 2)
```

同时断言整页 confirmed 样本与区域样本可以在同一导出中共存。

- [ ] **Step 2: 验证失败**

Run:

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_annotation_app tests.test_training_dataset -v
```

- [ ] **Step 3: 实现接口和导出**

路由只解析请求及返回 JSON，坐标、名称、状态和路径校验放在 service/store。
导出遍历 `list_confirmed_targets`，为每个目标复制其独立512输入及蒙版。
整页样本字段保持现状，区域样本追加裁剪元数据。

- [ ] **Step 4: 运行测试并提交**

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_annotation_app tests.test_training_dataset -v
git add annotation_tool/app.py annotation_tool/services.py tests/test_annotation_app.py tests/test_training_dataset.py
git commit -m "feat: export cropped annotation regions"
```

### Task 7: 增加前端框选和目标切换

**Files:**
- Modify: `annotation_tool/templates/index.html`
- Modify: `annotation_tool/static/app.js`
- Modify: `annotation_tool/static/style.css`
- Create: `annotation_tool/static/crop_region.js`
- Create: `tests/test_crop_region.js`

**Interfaces:**
- Produces:
  - `normalizeCrop(start, end, width, height)`
  - `canvasPointToImage(point, transform)`
  - 区域模式、框选叠层、名称输入、区域子列表

- [ ] **Step 1: 写失败 JavaScript 测试**

覆盖反向拖动规范化、缩放/平移坐标换算、边界钳制、过小区域拒绝。

- [ ] **Step 2: 验证失败**

Run:

```powershell
node tests/test_crop_region.js
```

Expected: 模块不存在。

- [ ] **Step 3: 实现纯坐标模块**

使用 UMD 风格与现有 `saved_mask_loader.js` 一致，确保浏览器和 Node 测试共用。

- [ ] **Step 4: 实现界面**

增加整页/裁剪区域模式按钮、选择框叠层、名称对话区和区域子列表。进入裁剪模式
关闭绘制；确认后调用区域 API 并加载区域 artifact。目标切换前等待现有自动保存，
失败时保持当前目标。

- [ ] **Step 5: 运行前端及模板测试**

```powershell
node tests/test_crop_region.js
node tests/test_saved_mask_loader.js
node tests/test_export_result.js
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest tests.test_annotation_app -v
```

- [ ] **Step 6: 提交**

```powershell
git add annotation_tool/templates/index.html annotation_tool/static/app.js annotation_tool/static/style.css annotation_tool/static/crop_region.js tests/test_crop_region.js
git commit -m "feat: add annotation crop region UI"
```

### Task 8: 全量验证和真实页面验收

**Files:**
- No source changes unless a failing verification exposes a defect

- [ ] **Step 1: 运行自动化测试**

```powershell
& G:\bim-web\.venv-train\Scripts\python.exe -m unittest `
  tests.test_annotation_storage `
  tests.test_annotation_app `
  tests.test_training_dataset `
  tests.test_training_losses `
  tests.test_training_experiment `
  tests.test_export_floorplan_onnx -v
node tests/test_crop_region.js
node tests/test_saved_mask_loader.js
node tests/test_export_result.js
```

Expected: 0 failures、0 errors。

- [ ] **Step 2: 启动隔离工作区标注工具**

使用独立端口启动，上传现有文化宫 PDF，准备包含四、五层的页面，创建“四层”和
“五层”两个区域，确认能够分别切换、绘制、自动保存、确认和导出。

- [ ] **Step 3: 验证导出训练可读**

对真实双区域导出运行 `validate_export` 并加载两个 dataset item，确认图片和蒙版
均为 `[3,512,512]` 与 `[512,512]`。

- [ ] **Step 4: 检查 Git 范围**

```powershell
git status --short
git diff --check main...HEAD
git log --oneline main..HEAD
```

确认未修改或纳入用户现有未跟踪文件、PDF、标注数据、模型或凭证。

- [ ] **Step 5: 完成分支并按用户要求合并回本地 `main`**

先执行 finishing-a-development-branch 流程，获得合并选择后再合并，不推送、
不部署、不替换网站正式模型。
