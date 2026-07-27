# bim-web 项目专用虚拟环境设计

日期：2026-07-15

## 目标

将网站运行环境从混杂的基础 Anaconda 环境中隔离出来，使图纸 ONNX 推理、年度能耗计算、`pvlib` 导入和 Flask 网站能够在同一套可重复环境中运行，同时不影响基础环境里的 TensorFlow、Streamlit 等其他工具。

## 已确认的根因

- 当前网站由 `D:\Anaconda\Anaconda3\python.exe` 启动。
- 基础环境的 NumPy 2.2.6 与现有 pyarrow、pandas 及多项 Anaconda 二进制包不兼容，导致 `pvlib` 导入失败。
- `onnxruntime` 原先未安装。
- OpenCV 4.13 在 Python 3.12 下要求 NumPy 2，不能与现有 NumPy 1.x 二进制生态共存。
- 基础环境同时含 TensorFlow 2.21 与 Streamlit 1.32，它们对 protobuf 的要求互相冲突，不适合作为网站长期运行环境。

## 采用方案

在项目根目录创建 `.venv`，由基础 Anaconda 的 Python 3.12 仅负责创建虚拟环境。网站以后始终通过 `.venv\Scripts\python.exe` 启动。

核心兼容版本：

- Python 3.12
- NumPy 1.26.4
- OpenCV headless 4.10.0.84
- ONNX Runtime 1.20.1（CPU）
- protobuf 4.25.8

其余网站依赖从 `requirements.txt` 安装，但通过约束文件锁定上述关键版本，禁止解析器重新升级到 NumPy 2 或 OpenCV 4.13+。

## 文件与启动方式

- `.venv/`：本地虚拟环境，已由 `.gitignore` 排除。
- `requirements-runtime-constraints.txt`：关键兼容版本约束。
- `setup_venv.ps1`：创建/更新虚拟环境并安装依赖。
- `start_local.ps1`：检查虚拟环境后，以 `.venv\Scripts\python.exe run_local.py` 启动网站。
- `requirements.txt`：继续表达网站所需的直接依赖，不保存机器专属路径。

启动脚本不会自动修改基础 Anaconda，也不会把管理员密码写入代码。登录凭据仍由现有环境变量注入。

## 基础环境处理

恢复本轮临时安装的 protobuf 7.34.1，使 TensorFlow 2.21 不因 protobuf 4.x 直接失效。基础环境此前改为 NumPy 1.26.4、OpenCV 4.10 和安装 ONNX Runtime 的状态暂时保留，因为这同时修复了现有 Anaconda 二进制包；网站本身不再依赖这些基础环境包。

如果基础环境仍存在 Streamlit 与 TensorFlow 的 protobuf 冲突，只记录为基础环境历史问题，不继续为网站扩大修改范围。

## 错误处理

- `.venv` 不存在时，启动脚本明确提示先运行 `setup_venv.ps1`。
- 依赖安装失败时立即停止并保留 pip 错误，不继续启动半成品环境。
- ONNX 模型缺失或加载失败时，验收失败，不把仅能打开网页视为完成。
- 不在脚本中静默联网升级所有包；只有显式运行安装脚本才更新环境。

## 验收标准

在 `.venv` 中完成以下验证：

1. `numpy`、`cv2`、`pandas`、`pyarrow`、`pvlib`、`onnxruntime` 同时导入成功。
2. `models/M2_pub_plus_user.onnx` 能创建 ONNX Runtime 会话并执行一次与网站预处理一致的推理。
3. `floorplan_onnx.get_segmenter()` 能加载网站实际部署模型。
4. `python -m unittest discover -s tests` 全部通过。
5. `web_server_server` 导入成功，且使用根目录新版 `energy_calc.py`。
6. `start_local.ps1` 能启动 `http://127.0.0.1:5000`，不再出现 NumPy 1.x/2.x、缺少 ONNX Runtime或 `pvlib` 导入错误。

## 范围边界

- 本次不升级 TensorFlow、Streamlit 或整套基础 Anaconda。
- 本次不引入 GPU ONNX Runtime。
- 网站运行环境不重复安装 PyTorch；PyTorch 只用于独立的模型导出工具，继续使用已经验证的基础环境 2.5.1+cpu。
- 本次不提交、推送或部署服务器。
- 本次不改变模型结构和识别算法。
