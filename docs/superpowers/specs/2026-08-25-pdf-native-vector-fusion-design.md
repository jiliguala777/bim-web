# PDF 原生正交墙线与新模型融合设计

日期：2026-08-25

分支：`codex/pdf-native-vector-fusion`

基线：`codex/vector-platform-adapter`

## 1. 目标

为矢量 PDF 建立一条与旧 ONNX 识别完全隔离的新识别路径：从 PDF 原生图元中提取水平线和垂直线，使用新 10 通道矢量多任务模型的源图概率判断每条线是墙体候选、其他线或不确定线，再从被接受的正交墙线中寻找闭合多边形作为房间候选。

首阶段只发布可审查的候选 JSON 和叠图，不向能耗计算提交墙、门、窗或房间几何，固定返回 `load_geometry_ready: false`。

## 2. 已确认的产品范围

### 2.1 输入

- 只处理保留了可提取线图元的矢量 PDF。
- 支持选择 PDF 页面；页面裁剪沿用平台已有的页面选择与裁剪请求格式。
- 新模型使用本机现有推理代码和权重，以 CPU 运行：
  - 推理入口：`G:\bim-annotation-training\.worktrees\vector-annotation-v1\training\predict_vector.py`
  - 权重：`C:\Users\majin\bim-annotation-training-data\best.pt`
- 扫描 PDF、PNG 和 JPG 不进入本阶段的新融合路径。

### 2.2 几何范围

- 只判断水平线和垂直线；PDF 点坐标两个端点在另一轴上的偏差不超过 `0.5 pt` 时视为正交线。
- 斜线不参与墙体和房间判断，记录为忽略统计，不写入候选明细。
- 首阶段不识别门、窗，也不把模型直接输出的墙线作为最终墙体。
- 房间候选只能由已接受的水平/垂直墙候选闭合形成。
- 允许非常小的断口保守吸附；每次吸附必须可追踪，较大断口不得自动补齐。

### 2.3 输出

- 每条原生候选线的几何、证据、判定和原因。
- 墙线合并、交点拆分和端点吸附记录。
- 闭合房间候选及其边界墙线编号、面积、模型概率和可信状态。
- 调试叠图。
- 不改变旧 ONNX 识别结果或已有能耗计算输入。

## 3. 隔离约束

新融合路径不得导入或直接调用以下旧模型模块：

- `floorplan_onnx.py`
- `floorplan_page_pipeline.py`
- `floorplan_rooms.py`
- `floorplan_topology_repair.py`
- `vector_pdf_scale.py`

确需复用的 PDF 提取、尺寸识别、图像写入或拓扑算法，复制到新路径的独立模块中，再按新契约修改。复制后的代码拥有独立测试；后续修改不得反向影响旧 ONNX 路径。

允许复用不属于旧模型识别实现的通用平台能力，例如登录、上传目录管理、PDF 页面令牌校验、页面/裁剪请求数据结构和响应序列化，但新融合使用独立路由和独立产物名称。

## 4. 方案选择

采用“PDF 原生几何为主、新模型概率为辅助”的方案：

1. PDF 原生解析提供唯一合法的候选线几何。
2. 新模型十通道概率只为已有候选线提供语义证据。
3. 尺寸、样式、建筑范围和闭合拓扑提供几何证据。
4. 模型不得在 PDF 不存在线段的位置新增墙。
5. 模型的 `prediction.geometry.json` 仅可作为调试参照，不能进入本阶段的融合判定或能耗几何。

没有采用三路投票，因为旧、新模型的栅格误差可能相关，且不同输出形态难以公平投票。没有采用纯 PDF 规则，因为 PDF 图元缺乏“墙、尺寸、家具”等语义，新模型的房间和建筑主体概率仍有辅助价值。

## 5. 模块边界

### 5.1 `vector_pdf_native.py`

新建，独立完成一页 PDF 的原生解析和候选准备，不导入 `vector_pdf_scale.py`。

公开接口：

```python
def extract_native_pdf_page(
    pdf_path: str | Path,
    page_index: int,
    dpi: int = 100,
) -> dict:
    ...
```

返回 JSON 兼容对象，至少包含：

- `page_size_pt`
- `render_size_px`
- `text_spans`
- `orthogonal_segments`
- `styled_edges`
- `dimension_candidates`
- `is_vector_pdf`
- `ignored_diagonal_count`

该模块同时复制并收敛现有尺寸识别所需的最小算法，输出明确的尺寸线和端点线身份，而不只输出一张掩膜。

该模块也独立计算深色图元密集区域形成的建筑 ROI。ROI 只是候选过滤证据；检测失败时必须显式返回 `enabled: false`，不能默认整页都是建筑区域。

### 5.2 `vector_pdf_model.py`

新建并独立实现概率推理与读取职责；不修改 `vector_platform.py` 的既有图片后端契约。该模块负责：

- 通过外部训练环境执行 `python -m training.predict_vector --device cpu`；
- 读取 `probabilities.npz`；
- 验证概率形状为 `(10, height, width)`；
- 验证全部数值有限且在 `[0, 1]`；
- 验证 `inference.json` 中的通道名称严格等于：

```text
footprint_interior, footprint_boundary,
room_interior, room_boundary,
wall_centerline, door_line, window_line,
wall_keypoint, door_endpoint, window_endpoint
```

- 返回与当前分析图像像素坐标对齐的只读概率数组和推理元数据。

外部进程使用参数数组启动，不拼接 shell 字符串；超时默认 `600 s`，可由环境变量覆盖。

### 5.3 `vector_pdf_fusion.py`

新建，负责候选线身份、坐标转换、模型概率采样和三态决策。

核心接口：

```python
@dataclass(frozen=True)
class FusionThresholds:
    axis_tolerance_pt: float = 0.5
    roi_inside_ratio_min: float = 0.70
    roi_outside_reject_ratio: float = 0.80
    dimension_overlap_reject_ratio: float = 0.50
    wall_mean_min: float = 0.35
    wall_p90_min: float = 0.55
    room_side_mean_min: float = 0.35
    footprint_mean_min: float = 0.50

def build_line_candidates(page_data: dict, thresholds: FusionThresholds) -> list[dict]:
    ...

def fuse_line_candidates(
    candidates: list[dict],
    probabilities: np.ndarray,
    image_size: tuple[int, int],
    building_roi: list[int] | None,
    thresholds: FusionThresholds,
) -> list[dict]:
    ...
```

稳定候选编号由规范化后的方向、PDF 点坐标和重复序号生成，不依赖 `pdfplumber` 对象顺序之外的内存身份。每条候选同时保存 `start_pt/end_pt` 和 `start_px/end_px`。

### 5.4 `vector_pdf_rooms.py`

新建，负责正交墙线拓扑和房间候选。

核心接口：

```python
@dataclass(frozen=True)
class RoomClosureThresholds:
    snap_fraction: float = 0.003
    snap_min_px: int = 2
    snap_max_px: int = 8
    min_area_fraction: float = 0.0001
    min_area_px2: float = 400.0
    max_roi_area_fraction: float = 0.80
    room_probability_min: float = 0.35
    footprint_probability_min: float = 0.50

def find_room_candidates(
    accepted_lines: list[dict],
    probabilities: np.ndarray,
    image_size: tuple[int, int],
    building_roi: list[int] | None,
    thresholds: RoomClosureThresholds,
) -> dict:
    ...
```

处理顺序：

1. 合并同轴、共线且重叠或相接的线段。
2. 计算吸附阈值：`clamp(round(min(width, height) * 0.003), 2, 8)` 像素。
3. 只对水平与垂直端点进行正交吸附，不生成斜边。
4. 在线段交叉点拆分边，构建平面图。
5. 使用有向半边遍历最小有界面，剔除外部面和重复面。
6. 按面积、ROI、`room_interior` 和 `footprint_interior` 概率过滤。

由原始线段直接闭合的房间记录 `closure_applied: false`。使用端点吸附后闭合的房间记录 `closure_applied: true`、吸附前后坐标、距离和相关墙线编号。

### 5.5 `vector_pdf_fusion_pipeline.py`

新建，编排页面解析、渲染、裁剪、模型推理、融合、房间发现和产物发布。它不导入旧模型管线。

核心接口：

```python
def analyze_vector_pdf_page(
    pdf_path: str | Path,
    page_number: int,
    output_dir: str | Path,
    model_config: VectorModelConfig,
    crop_bbox_page_px: list[int] | None = None,
) -> dict:
    ...
```

管线以临时目录生成产物，全部验证成功后再原子发布。模型失败时仍发布已验证的原生候选 JSON，页面状态为 `partial`。

### 5.6 `web_server_server.py`

只增加薄路由和配置装配：

- 新增独立接口 `POST /energy/vector_pdf_fusion`；
- 复用已有的登录、PDF 上传令牌、页面选择和裁剪请求校验；
- 调用 `analyze_vector_pdf_page()`；
- 返回状态、摘要和调试产物地址；
- 不调用旧 `_floorplan_segmenter`；
- 不覆盖旧 `recognition.json`；
- 不修改 `/energy/ai_recognize` 的旧 ONNX 行为。

## 6. 候选线判定

### 6.1 原生证据

每条线保存：

- 方向、长度、线宽、颜色和中性亮度；
- 位于建筑 ROI 内的长度比例；
- 是否属于深色中性结构层；
- 是否与已确认尺寸线或尺寸端点重合；
- 是否为页面边框；
- 是否属于灰色或彩色辅助图元。

满足以下任一硬排除条件时判为 `rejected_nonstructural`：

- 与确认尺寸图元的长度重叠率至少为 `0.50`；
- 位于建筑 ROI 外的长度比例至少为 `0.80`；
- 位于页面边缘 `2 pt` 内且跨度至少覆盖相应页面边长的 `0.80`，判定为页面图框。

灰色、彩色或短线本身不是硬排除条件，只提供非结构证据，避免误删真实墙线。

### 6.2 模型采样

沿候选线建立中心窄带，并在线两侧建立平行窄带。窄带半径为 `clamp(round(min(width, height) * 0.003), 2, 12)` 像素。

记录：

- `wall_centerline` 的中心带均值和第 90 百分位；
- 两侧 `room_interior` 均值；
- 中心及两侧 `footprint_interior` 均值；
- 门窗线和端点通道的最大值，仅供调试。

### 6.3 三态结果

- `accepted_wall_candidate`：没有硬排除证据、ROI 内比例达标、属于深色中性结构层、建筑主体概率达标，并且墙中心线均值达标或第 90 百分位达标，同时至少一侧房间概率达标。
- `rejected_nonstructural`：命中硬排除条件。
- `uncertain`：其余所有情况，包括原生结构证据强但模型支持不足、模型支持强但原生样式不明确。

所有判定保存稳定的 `reason_codes`，例如 `dimension_overlap`、`page_border`、`outside_building_roi`、`model_wall_supported`、`room_side_supported`、`weak_model_support`。

阈值配置在首批真实 PDF 校准前固定标记 `calibration_status: uncalibrated`。首阶段不得因为默认阈值产生能耗就绪结果。

## 7. 房间候选判定

闭合多边形必须：

- 完全由 `accepted_wall_candidate` 经合并、交点拆分和允许的微小吸附形成；
- 位于建筑 ROI 内；
- 面积至少为 `max(400 px², image_area * 0.0001)`；
- 面积不超过建筑 ROI 面积的 `0.80`；
- 内部平均 `footprint_interior` 至少为 `0.50`。

内部平均 `room_interior` 至少为 `0.35` 时标记 `accepted_room_candidate`，否则标记 `suspicious_closed_region`。可疑闭合区域保留在调试输出中，但不计入接受房间数。

多边形允许为任意正交多边形，不限于矩形。斜边房间不在首阶段范围内。

## 8. 产物契约

### 8.1 `pdf_native_candidates.json`

包含页面信息、坐标映射、原生候选、尺寸证据、建筑 ROI、斜线忽略数和解析状态。不得包含模型推断出的墙线。

### 8.2 `pdf_vector_fusion.json`

顶层格式为 `pdf-vector-fusion/1`，包含：

```json
{
  "format": "pdf-vector-fusion/1",
  "status": "evaluable",
  "load_geometry_ready": false,
  "calibration_status": "uncalibrated",
  "page": {},
  "model": {},
  "thresholds": {},
  "line_candidates": [],
  "topology": {
    "merged_lines": [],
    "snaps": []
  },
  "room_candidates": [],
  "summary": {},
  "reason_codes": []
}
```

状态：

- `evaluable`：原生矢量、模型概率和坐标映射全部有效。
- `partial`：原生候选有效，但模型不可用、失败或超时。
- `rejected`：不是矢量 PDF、页面/裁剪无效、概率契约不兼容或坐标不一致。

### 8.3 图像与大数组

- `pdf_vector_fusion_overlay.png`：绿色墙候选、红色其他线、黄色不确定线、半透明蓝色接受房间、紫色可疑闭合区域。
- 模型原始 `probabilities.npz` 保留在模型产物目录；融合 JSON 只保存相对路径和 SHA-256，不嵌入数组。

## 9. 错误处理与安全

- 非矢量 PDF 返回 `rejected/not_vector_pdf`，不退回到新模型图片识别。
- 模型超时或依赖缺失返回 `partial/model_unavailable` 或 `partial/model_timeout`，同时保留原生候选。
- 概率尺寸、通道、数值范围或哈希不合法时返回 `rejected/model_contract_invalid`。
- 路径必须在当前报告目录或明确配置的训练根目录内解析；上传文件名继续使用平台现有安全校验。
- 外部模型进程不使用 shell，限制超时，不接收用户构造的模块名、Python 路径或权重路径。
- 产物使用临时文件或临时目录写入并原子替换，避免半成品被页面读取。

## 10. 测试策略

### 10.1 原生解析单元测试

用 ReportLab 生成包含以下元素的最小矢量 PDF：

- 黑色水平/垂直墙线；
- 斜线；
- 尺寸文字、尺寸基线和端点；
- 页面图框；
- 灰色辅助线。

验证正交候选、斜线忽略统计、尺寸身份、图框排除和坐标转换。

### 10.2 模型契约与采样测试

使用合成 `(10, H, W)` 概率数组验证：

- 通道、尺寸和有限值校验；
- 水平线、垂直线中心带采样；
- 两侧房间概率采样；
- 三态判定及原因码。

单元与持续集成测试不运行真实 `best.pt`，避免硬件和耗时造成不稳定。增加显式的本机 CPU 冒烟命令，用真实权重验证一次端到端产物，但不作为默认测试套件的前置条件。

### 10.3 拓扑测试

至少覆盖：

- 四条完整墙线形成一个矩形房间；
- 一个小缺口在阈值内被吸附并记录；
- 超过阈值的缺口不闭合；
- 多个相邻正交房间；
- 图框和家具矩形因线判定未接受而不能形成房间；
- 闭合但模型房间概率不足时标为可疑区域。

### 10.4 路由与兼容测试

- 新路由使用已验证 PDF 令牌、页面和裁剪框。
- 新路由不调用旧 `_floorplan_segmenter`，不覆盖 `recognition.json`。
- `POST /energy/ai_recognize` 的旧 ONNX 行为和既有测试保持不变。
- 模型失败时返回部分结果；非矢量 PDF 返回明确拒绝原因。

## 11. 分阶段交付

1. 独立 PDF 原生解析模块和候选 JSON。
2. 新模型 CPU 概率读取、契约验证和逐线采样。
3. 墙候选三态融合、调试叠图。
4. 正交墙网合并、小缺口吸附和闭合房间候选。
5. 新独立路由及旧 ONNX 全量回归。
6. 使用 5–10 份真实矢量 PDF 校准阈值；校准前始终禁止能耗就绪。

## 12. 非目标

- 不重新训练或修改 `best.pt`。
- 不处理扫描 PDF、PNG 或 JPG。
- 不判断斜墙。
- 不自动识别或生成门窗。
- 不使用模型锯齿轮廓新增 PDF 中不存在的墙。
- 不把房间候选写入能耗计算所用的 `recognition.json`。
- 不修改、重构或共享调用旧 ONNX 模型的 Python 实现。

## 13. 验收标准

- 新路径中不存在对第 3 节所列旧模型模块的导入。
- 对合成矢量 PDF，尺寸线、图框和斜线不会成为墙候选。
- 对合成概率图，候选线判定和原因可重复、可解释。
- 阈值内的小缺口可形成带吸附记录的房间候选；阈值外缺口不能闭合。
- 每个房间候选可追溯到组成它的墙候选编号。
- 新路径所有响应均为 `load_geometry_ready: false`。
- 旧 ONNX 测试和路由行为无回归。
- 本机真实 `best.pt` CPU 冒烟运行可以生成合法的十通道概率产物，或返回明确、可诊断的环境错误。
