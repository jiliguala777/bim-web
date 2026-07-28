# PDF 人工标注与单页微调

这个本地工具把网站当前的多页 PDF 预处理、ONNX 预识别和人工修正串成可重复的训练数据生产流程。第一版只负责可靠地产生训练数据；模型训练由仓库中的 `training` 命令单独完成。

## 安全边界

- 服务只允许监听 `127.0.0.1` 或 `localhost`，默认端口为 `8099`。
- 工具代码位于 `G:\bim-web\annotation_tool`。
- PDF、标注、导出和实验模型默认写入 `G:\bim网页\标注数据`，不进入 Git。
- 原有 `G:\bim网页\标注工具\data` 是只读兼容数据。本工具不会读取、迁移或修改已有的 219 对图片/标签和 29 张人工确认数据。
- 当前生产模型不会被覆盖；每次实验写入新的输出目录。

临时改变数据根目录：

```powershell
.\annotation_tool\start_annotation_tool.ps1 -DataRoot "D:\临时标注数据"
```

## 一、启动

在 `G:\bim-web` 中运行：

```powershell
$env:ONNX_MODEL_PATH="G:\bim-web\models\M2_pub_plus_user.onnx"
.\annotation_tool\start_annotation_tool.ps1
```

浏览器打开 <http://127.0.0.1:8099>。不要把标注服务绑定到局域网或公网地址。

## 二、只标注四层建筑的一层

1. 输入项目名称并导入原始四层 PDF。源 PDF 只会按 SHA-256 复制一份，不会被修改。
2. 选择一层所在的 PDF 页；页码不一定等于楼层号，请通过原图确认。
3. 点击“准备所选页面”，运行与网站能耗识别相同的渲染、矢量清理、ROI 和 512 letterbox 流程。
4. 点击“运行当前 ONNX 预标注”，再人工修正。
5. 在原图、清理图和模型图间切换；三种视图共用同一张全分辨率蒙版。
6. 保存草稿并刷新确认，再点击“确认本页标注”。
7. 只让这一页处于 `confirmed` 状态，然后点击“导出已确认页面”。

单页导出会明确标记为 `single_page_overfit`。

### 复杂矢量 PDF：自动栅格后备

常规 PDF 会保留矢量线段和文字提取，以增强页面准备。若单页的矢量提取超过
15 秒，工具会自动以 100 DPI 栅格渲染完成准备，并在事件日志中显示
`raster_fallback`。该状态是有效的准备结果，仍可进行墙、窗、门的标注并用于
训练；不需要重新导入或重复准备页面。

栅格后备不提供可信的矢量比例尺信息。需要能耗计算时，请在后续能耗工作流中
通过人工标定或其他可靠依据处理比例尺。准备期间，“准备所选页面”和项目/页
选择控件会暂时禁用；请等待准备结束，不要重复点击按钮或切换选择项。

## 三、构件骨架粗线规则

| 类别值 | 构件 | 标注方式 |
|---:|---|---|
| 0 | background | 背景/橡皮擦 |
| 1 | wall | 沿墙体中心骨架连续绘制 |
| 2 | window | 沿窗洞中心线绘制并连接墙线 |
| 3 | door | 沿门洞中心线绘制并连接墙线 |

墙线建议保持在 512 模型空间中的 4–8 像素。界面会自动换算成 PDF 页全分辨率像素。不要用实心面填墙，不要把尺寸线、文字、家具当成墙，也不要在门窗与墙线之间留下断口。

每次保存都会产生不可覆盖的版本。草稿为 `draft`，模型结果为 `preannotated`，人工确认后为 `confirmed`。只有当前版本为 `confirmed` 的页面能够导出。

## 四、建立独立训练环境

```powershell
.\training\setup_training_env.ps1
```

它只创建 `G:\bim-web\.venv-train`，不会改变网站使用的 `.venv`。命令会显示 PyTorch 版本和 CUDA 可用状态。

## 五、运行单页微调

把 `--dataset` 换成标注工具刚生成的实际导出目录：

```powershell
.\.venv-train\Scripts\python.exe -m training.run_experiment `
  --dataset "G:\bim网页\标注数据\exports\four-floor-building-floor1-20260727T160000Z" `
  --checkpoint "G:\bim-web\models\M2_pub_plus_user.pt" `
  --output "G:\bim网页\标注数据\models\four-floor-building-floor1-overfit" `
  --device cpu
```

CUDA 可用时可改为 `--device cuda`。输出包括：

- `last.pt`、`best.pt` 和经过验证的 `best.onnx`；
- `training_log.csv`、`metrics.json`；
- `old_model_overlay.png`、`new_model_overlay.png`、`ground_truth.png`；
- `comparison.png`、`comparison_metrics.json`、`export_report.json`。

## 六、在本地网站复测

不要替换原模型，只在当前 PowerShell 会话指定实验模型：

```powershell
$env:ONNX_MODEL_PATH="G:\bim网页\标注数据\models\four-floor-building-floor1-overfit\best.onnx"
.\start_local.ps1
```

在网站能耗计算页面重新上传同一 PDF、选择同一页，并保存识别叠加图进行比较。

同一页既用于训练又用于复测，只能说明模型是否成功拟合这一页以及训练链是否有效，不能证明对其他建筑或楼层的泛化能力。实验模型不自动部署到生产服务器；是否进入生产必须在独立图纸验证后另行决定。

## 数据目录

```text
G:\bim网页\标注数据\
  sources\       按 SHA-256 去重的源 PDF
  projects\      项目、页准备产物和清单
  annotations\   每页不可覆盖的蒙版版本
  exports\       confirmed 页的 512 训练数据
  models\        微调检查点、ONNX 和比较报告
```
