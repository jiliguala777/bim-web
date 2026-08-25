# PDF 外轮廓、门窗与能耗几何设计

日期：2026-08-25  
状态：待实施  
基础分支：`codex/pdf-fusion-performance`

## 1. 目标

在现有 PDF 原生线段与十通道模型概率融合结果上，新增面向能耗计算的建筑外围护结构识别：

- 识别外墙断口中的门和窗，而不是把门窗保存为实体墙；
- 用虚拟边跨越已识别的门窗，使建筑外轮廓可以闭合；
- 对无法分类但足够小、共线且安全的外轮廓缺口，用直线补墙候选闭合；
- 计算闭合外轮廓的像素面积和周长；
- 分别保存门宽和窗宽，门窗高度由后续能耗页面设置；
- 用户确认外轮廓并确认比例尺后，才允许生成能耗几何。

本阶段不要求完整识别内墙、内门和内窗。内部构件可以继续作为调试证据，但不能阻止外轮廓进入能耗计算。

## 2. 设计原则

### 2.1 真实构件与能耗拓扑分离

系统同时维护两层数据：

1. **真实构件层**：墙、门、窗分别存在，门窗不会转换为墙。
2. **能耗拓扑层**：为了形成闭合多边形，在门窗断口和小缺口处加入虚拟边。

外轮廓多边形使用虚拟边计算面积和周长；门窗面积仍从外围护结构中单独扣除。

### 2.2 模型提供语义，PDF 原生线提供几何锚点

第一版不允许模型在任意位置自由生成门窗。门窗候选必须锚定在两段共线的外墙端点之间，并同时得到模型通道支持：

- `door_line` / `window_line`：开口类型与线段支持；
- `door_endpoint` / `window_endpoint`：开口两端支持；
- `wall_centerline`：两侧宿主墙支持；
- `footprint_interior` / `footprint_boundary`：判断墙的内外侧及外轮廓属性。

这样保留 PDF 原生坐标精度，同时避免模型在家具、尺寸线或内墙附近凭空增加开口。

### 2.3 保守自动修复

自动修复只接受水平线和垂直线。斜墙、需要转角的两段式修复以及大范围缺失继续标记为人工处理，不在第一版自动闭合。

## 3. 方案选择

### 方案 A：外墙断口锚定的门窗识别（采用）

先建立外墙候选，再检查共线墙端点之间的空隙。门窗概率和端点概率决定空隙是门、窗还是未知小缺口。

优点：几何稳定、可追溯、误检风险低，适合当前 PDF 原生融合架构。  
限制：没有形成墙体断口的门窗可能暂时无法输出。

### 方案 B：直接从模型概率图提取所有门窗连通域（暂不采用）

优点：召回率可能更高。  
缺点：容易把图例、家具和标注识别为门窗，也会突破“模型不自由增加几何”的安全边界。

### 方案 C：只补墙、不保存门窗（不采用）

优点：实现简单。  
缺点：无法计算门窗面积，会把开口误算为不透明墙体，不满足能耗计算需要。

## 4. 处理流水线

```text
PDF 原生水平/垂直线
        │
        ▼
墙候选 + 十通道模型证据
        │
        ▼
外墙候选筛选
  - footprint 内外侧
  - footprint boundary
  - 建筑 ROI
        │
        ▼
共线端点缺口枚举
        │
        ├── 门证据充分 ──> door opening + opening_bridge
        ├── 窗证据充分 ──> window opening + opening_bridge
        ├── 小且安全 ────> small_gap_repair
        └── 其他 ────────> unresolved_gap
        │
        ▼
正交图闭合与最大合理外轮廓
        │
        ▼
建筑面积、外墙周长、门宽、窗宽
        │
        ▼
用户确认 + 比例尺确认
        │
        ▼
recognition.json / 能耗几何
```

## 5. 外墙候选

外墙候选来源于已接受的墙候选，必须满足：

- 位于建筑 ROI 内；
- 原生线条为深色结构线；
- `wall_centerline` 达到现有墙支持阈值；
- 线段一侧有较高 `footprint_interior`，另一侧显著较低，或者线段附近有较高 `footprint_boundary`；
- 不属于页面边框、尺寸线或标题栏。

输出记录内侧方向，使后续门窗能够判断是否位于外围护结构上。内外侧不明确的墙保留为普通墙候选，不自动参与外轮廓。

## 6. 门窗候选

### 6.1 几何前提

门窗候选必须位于两段外墙之间：

- 两段墙方向相同且近似共线；
- 两端面向同一个缺口；
- 缺口没有被另一条结构墙穿过；
- 两侧墙的建筑内侧方向一致；
- 缺口位于建筑 ROI 内。

### 6.2 语义判定

对缺口建立局部采样窗口：

- 门分数由 `door_line`、两个 `door_endpoint` 和宿主墙支持共同决定；
- 窗分数由 `window_line`、两个 `window_endpoint` 和宿主墙支持共同决定；
- 当门窗同时超过阈值时，选择证据更强且端点更一致的类别；差异不足时标记为 `ambiguous_opening`；
- 没有充分门窗证据时，不得生成门窗对象。

第一版只输出外门窗。内部门窗可以保留在调试统计中，但不写入外围护结构宽度汇总。

### 6.3 门窗数据结构

每个开口至少包含：

```json
{
  "opening_id": "opening-0001",
  "kind": "door",
  "orientation": "horizontal",
  "start_px": [100, 200],
  "end_px": [145, 200],
  "width_px": 45.0,
  "width_m": null,
  "host_wall_ids": ["line-0010", "line-0011"],
  "exterior": true,
  "confidence": 0.91,
  "model_evidence": {},
  "reason_codes": ["door_line_supported", "door_endpoints_supported"]
}
```

`width_m` 只有在比例尺确认后填写。门高和窗高不在识别结果中推断。

## 7. 虚拟闭合边

虚拟边分为三类：

- `opening_bridge`：跨越已识别门窗；
- `small_gap_repair`：无法识别为门窗的小缺口补墙；
- `manual_repair`：用户后续手工补充，第一版自动流程不生成。

每条虚拟边必须保存来源端点、宿主墙、修复类型、置信度和原因码。虚拟边只进入外轮廓图，不进入真实墙体列表。

### 7.1 门窗桥接

已接受的门窗候选生成一条与宿主墙共线的 `opening_bridge`。桥接长度就是开口平面宽度。

### 7.2 未知小缺口补墙

当缺口没有充分门窗证据时，只有同时满足以下条件才生成 `small_gap_repair`：

- 两段外墙共线、方向一致、内侧方向一致；
- 只需一条水平线或垂直线即可连接；
- 缺口内没有结构墙交叉，也没有竞争性的多个连接目标；
- 有比例尺时，缺口不超过 `0.6 m`；
- 无比例尺时，像素阈值为 `max(4, min(32, round(min(image_width, image_height) * 0.01)))`；
- 修复后能够改善外轮廓闭合，且不会生成明显越出建筑 ROI 的边。

超过阈值、需要转角或连接目标不唯一时输出 `unresolved_gap`，不能自动闭合。

## 8. 外轮廓选择

将外墙候选、`opening_bridge` 和 `small_gap_repair` 放入独立正交图：

1. 合并重复和重叠的共线边，并保留所有来源 ID；
2. 在交点处分割边；
3. 枚举闭合面；
4. 排除页面边框、尺寸区域和 ROI 外面；
5. 选择面积最大且获得 `footprint_interior` 支持的合理闭合面作为 `building_footprint_candidate`。

外轮廓可以包含凹角，不能退化成建筑外接矩形。第一版只接受一个主要建筑外轮廓；多个独立建筑体或庭院孔洞标记为人工复核。

## 9. 外轮廓数据结构

```json
{
  "format": "pdf-exterior-topology/1",
  "status": "review_required",
  "confirmed": false,
  "polygon_px": [[100, 100], [900, 100], [900, 700], [100, 700]],
  "area_px2": 480000.0,
  "perimeter_px": 2800.0,
  "area_m2": null,
  "perimeter_m": null,
  "source_wall_ids": [],
  "bridge_ids": [],
  "opening_ids": [],
  "unresolved_gaps": [],
  "load_geometry_ready": false
}
```

自动分析结果始终为 `review_required`。只有用户确认外轮廓且比例尺为 `confirmed` 时，服务端才能将 `confirmed` 和 `load_geometry_ready` 设为 `true`。

## 10. 调试产物与页面展示

新增产物：

- `pdf_opening_candidates.json`
- `pdf_exterior_topology.json`
- `pdf_exterior_overlay.png`

叠加颜色：

- 真实墙：绿色实线；
- 门：蓝色；
- 窗：青色；
- `opening_bridge`：蓝色虚线；
- `small_gap_repair`：橙色虚线；
- `unresolved_gap`：红色标记；
- 最终外轮廓：紫色粗线和半透明填充。

页面显示：建筑面积候选、外墙周长候选、外门数量、外窗数量、门总宽、窗总宽、小缺口修复数和未解决缺口数。用户只需整体确认外轮廓；第一版不要求逐条确认每条补线。

## 11. 确认与持久化

新增独立确认接口，输入必须包含：

- 报告编号；
- 当前 PDF 页面和裁剪区域标识；
- `pdf_exterior_topology.json` 的内容哈希；
- 已确认比例尺；
- 用户确认动作。

服务端重新加载磁盘产物并校验哈希，不能信任浏览器提交的面积、周长或门窗宽度。确认成功后才写入当前报告的 `recognition.json`。

确认前继续保持现有安全边界：

- `load_geometry_ready: false`；
- 不覆盖旧识别结果；
- 不允许调用能耗计算。

## 12. 能耗几何

确认比例尺 `scale_m_per_px` 后：

- `floor_area_m2 = area_px2 * scale_m_per_px²`；
- `exterior_perimeter_m = perimeter_px * scale_m_per_px`；
- `door_total_width_m = Σ exterior door width_px * scale_m_per_px`；
- `window_total_width_m = Σ exterior window width_px * scale_m_per_px`。

能耗页面新增或明确使用：

- 单层层高（现有 `height` 字段在该路径中明确解释为层高）；
- 门高；
- 窗高；
- 层数；
- 门重复层数和窗重复层数。默认门重复层数为 `1`，窗重复层数为建筑层数，用户可以修改。

整栋建筑的简化外围护面积计算：

- `gross_exterior_wall_area_m2 = exterior_perimeter_m * storey_height_m * floors`；
- `door_area_m2 = door_total_width_m * door_height_m * door_repeat_count`；
- `window_area_m2 = window_total_width_m * window_height_m * window_repeat_count`；
- `opaque_wall_area_m2 = max(0, gross_exterior_wall_area_m2 - door_area_m2 - window_area_m2)`。

外轮廓面积是所选平面的单层面积，现有能耗引擎继续用 `floors` 计算总建筑面积。门窗重复层数显式设置，避免把首层入口门错误地复制到每一层。

传给 `energy_calc.py` 的 `wall_area_m2` 必须是不透明墙净面积，避免门窗与墙重复计算。门宽、窗宽和每个开口对象同时保存在 `recognition.json`，便于后续增加逐个门窗高度或朝向设置。

## 13. 模块边界

新增独立模块，继续遵守不与旧 PDF/旧 ONNX Python 文件混用的要求：

- `vector_pdf_openings.py`：缺口枚举、门窗概率评分、开口对象；
- `vector_pdf_exterior.py`：外墙筛选、虚拟边、小缺口补墙、外轮廓选择；
- `vector_pdf_energy_geometry.py`：比例尺应用、确认结果转换为能耗几何。

修改：

- `vector_pdf_fusion_pipeline.py`：编排新模块并发布调试产物；
- `web_server_server.py`：返回外轮廓调试结果、确认接口和持久化；
- `templates/energy.html`：展示与确认外轮廓，设置门高和窗高；
- `energy_calc.py` 的输入接口保持不变，仍接收墙、窗、门、地面和屋面面积。

新模块不导入 `floorplan_onnx.py`、`floorplan_page_pipeline.py`、`floorplan_rooms.py`、`floorplan_topology_repair.py` 或 `vector_pdf_scale.py`。

## 14. 错误与降级

- 无可靠外墙：返回 `no_exterior_wall_evidence`；
- 外轮廓无法闭合：返回 `exterior_not_closed` 和未解决缺口；
- 多个相近外轮廓：返回 `ambiguous_exterior`，等待人工复核；
- 门窗类别冲突：保留 `ambiguous_opening`，不计入门宽或窗宽；
- 比例尺未确认：只输出像素面积、周长和宽度；
- 哈希或页面选择已变化：拒绝确认，要求重新分析；
- 门窗面积大于外墙毛面积：拒绝能耗计算并提示检查高度或轮廓。

任何降级结果都不能静默写入 `recognition.json`。

## 15. 测试与验收

### 15.1 单元测试

- 门、窗和未知缺口的三分类；
- 两个端点证据不足时不生成门窗；
- 门窗宽度与比例尺换算；
- `opening_bridge` 不进入真实墙列表；
- `small_gap_repair` 的 0.6 米和像素阈值边界；
- 大缺口、歧义连接和转角缺口不自动修复；
- 重复共线边合并后来源 ID 不丢失；
- 凹形外轮廓面积和周长；
- 门窗宽度分别汇总；
- 墙净面积正确扣除门窗面积。

### 15.2 路由与持久化测试

- 调试结果不能进入能耗计算；
- 未确认比例尺不能确认外轮廓；
- 内容哈希不匹配时拒绝确认；
- 确认后 `recognition.json` 包含外轮廓、门窗对象和宽度汇总；
- 旧 ONNX 和图片新模型路径行为不变。

### 15.3 真实图纸验收

先使用当前已验证的真实矢量 PDF，再扩展到 5–10 份代表性图纸。记录：

- 外轮廓是否正确；
- 建筑面积相对人工结果的误差；
- 外墙周长相对人工结果的误差；
- 外门、外窗宽度召回与误检；
- `small_gap_repair` 的正确率；
- 是否存在错误跨越中庭、走廊或两个独立建筑体。

第一阶段目标是外轮廓候选可由用户一次确认后用于初步负荷计算；不以完整内墙识别率作为上线门槛。

## 16. 非目标

- 自动推断门高和窗高；
- 识别斜墙、弧形幕墙或任意曲线外轮廓；
- 解析门扇开启方向和门型；
- 自动处理多个独立建筑体、庭院孔洞或跨层挑空；
- 在没有用户确认和比例尺确认时直接计算能耗；
- 为本阶段重新训练或重新标注模型。
