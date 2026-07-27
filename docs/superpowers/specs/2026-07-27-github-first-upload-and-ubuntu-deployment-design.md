# GitHub 首次上传与 Ubuntu 部署整理设计

日期：2026-07-27

## 1. 目标

在不删除本地数据、不改写现有 Git 历史、不提交或推送代码的前提下，完成三项准备工作：

1. 完善 `.gitignore`，让源码、测试和正式运行所需的小型数据库可进入私人 GitHub，同时排除模型、用户数据、研究数据、缓存、日志和凭证。
2. 为腾讯云 Ubuntu 22.04 轻量应用服务器提供可审查的 Nginx、Gunicorn、systemd 和环境变量模板。
3. 提供一份 GitHub 首次上传清单，明确哪些内容应上传、哪些内容必须排除，以及如何规避现有大文件历史。

本轮不创建 GitHub 仓库，不配置远程仓库，不执行 `git add`、`git commit` 或 `git push`。

## 2. 已确认的部署条件

- 服务器系统：Ubuntu 22.04。
- 服务器原用于 OpenClaw，现可视为 BIM 项目专用服务器。
- 第一阶段通过公网 IP 访问，不依赖域名。
- 部署模型为 `M2_pub_plus_user.onnx`。
- ONNX 模型不进入 GitHub，首次部署时单独上传服务器。
- 现有工作区包含大量未提交内容，禁止清理未跟踪文件、覆盖当前文件或改写 Git 历史。

## 3. 方案选择

采用透明的标准部署模板，不采用一键安装脚本或 Docker：

- 每个服务器变更均以文档命令和模板文件表达，便于首次部署时逐项检查。
- Nginx 负责公网 HTTP、上传大小限制和反向代理。
- Gunicorn 使用 systemd 常驻，仅监听 `127.0.0.1`，不直接暴露到公网。
- Flask 代码、模型、用户数据和服务器密钥分开存放。
- 待首次部署流程稳定后，再考虑自动部署和 HTTPS。

## 4. 仓库边界

### 4.1 进入私人 GitHub

- 根目录 Python 源码：
  - `web_server_server.py`
  - `energy_calc.py`
  - `design_load_calc.py`
  - `energy_library.py`
  - `floorplan_onnx.py`
  - `floorplan_ocr.py`
  - `floorplan_rooms.py`
  - `floorplan_topology_repair.py`
  - `vector_pdf_scale.py`
  - `run_local.py`
- `templates/`
- `static/`
- `tests/`
- `tools/*.py`
- `docs/` 中的设计、计划和项目说明
- `README.md`
- `AGENTS.md`
- `requirements.txt`
- `requirements-runtime-constraints.txt`
- `setup_venv.ps1`
- `start_local.ps1`
- `.gitignore`
- `.env.example`
- 新的通用部署模板
- 正式运行数据库：
  - `data/building_library.db`
  - `data/envelope_databases/door.db`
  - `data/envelope_databases/exterior_wall.db`
  - `data/envelope_databases/floor.db`
  - `data/envelope_databases/floor_contact.db`
  - `data/envelope_databases/roof.db`
  - `data/envelope_databases/shading.db`
  - `data/envelope_databases/wall_insulation.db`
  - `data/envelope_databases/window_air_tightness.db`
  - `data/envelope_databases/windows.db`

### 4.2 不进入 GitHub

- `.pt`、`.pth`、`.onnx`、`.ckpt` 等模型权重。
- `uploads/`、`jobs/`、`tmp/`、`artifacts/`、`output/`。
- `.venv/`、其他本地虚拟环境、Python 缓存和浏览器自动化缓存。
- `.playwright-cli/`、`.codex/`、`.agents/`、`.superpowers/`。
- `users.db`、SQLite WAL/SHM 文件和日志。
- `data/` 下的原始 PDF、DOCX、XLS、OCR、中间抽取结果、审计结果和数据库备份。
- `.env`、SSH 私钥、API Key、密码、服务器清单和带真实凭证的运维文件。

### 4.3 模型交付

模型单独上传到：

```text
/opt/bim-web/models/M2_pub_plus_user.onnx
```

部署前在本地生成 SHA-256，上传后在服务器重新计算并比较。环境变量 `ONNX_MODEL_PATH` 指向该绝对路径。

## 5. 服务器目录

```text
/opt/bim-web/app/                         GitHub 代码
/opt/bim-web/venv/                        Python 虚拟环境
/opt/bim-web/models/                      单独上传的 ONNX 模型
/var/lib/bim-web/uploads/                 用户上传与识别结果
/var/lib/bim-web/users.db                 用户数据库
/etc/bim-web/bim-web.env                  服务器环境变量
/etc/systemd/system/bim-web.service        systemd 服务
/etc/nginx/sites-available/bim-web         Nginx 站点
```

代码更新只替换 `/opt/bim-web/app/`。模型、上传文件、用户数据库和密钥均位于代码目录之外，不会因 `git pull` 丢失。

## 6. 部署模板

### 6.1 `.env.example`

提供变量名和安全占位值，不包含真实密码：

- `SECRET_KEY`
- `ADMIN_USER`
- `ADMIN_PASSWORD`
- `ONNX_MODEL_PATH`
- `UPLOAD_FOLDER`
- `DB_PATH`
- `BUILDING_LIBRARY_DB_PATH`
- `ENVELOPE_DB_DIR`

Ubuntu 通过系统 `poppler-utils` 提供 Poppler，因此不强制设置 `POPPLER_PATH`。

### 6.2 systemd

新增 `deploy/bim-web.service`：

- 使用专用低权限用户 `bimweb`。
- 工作目录为 `/opt/bim-web/app`。
- 从 `/etc/bim-web/bim-web.env` 读取环境变量。
- 使用 `/opt/bim-web/venv/bin/gunicorn`。
- 绑定 `127.0.0.1:8000`。
- 初始采用 1 个 worker、2 个线程，避免多 worker 重复加载 ONNX 模型导致内存快速增加。
- 请求超时设为 300 秒，以容纳 PDF 渲染和识别。
- 进程异常退出时自动重启。
- 标准输出和错误输出进入 journald。

### 6.3 Nginx

新增 `deploy/nginx-bim-web.conf`：

- `listen 80 default_server`。
- `server_name _`，支持公网 IP 访问。
- `client_max_body_size 200m`。
- 代理到 `127.0.0.1:8000`。
- 代理连接超时 30 秒，读取和发送超时 300 秒。
- 传递真实 Host、客户端 IP 和协议头。
- 不在模板中写入真实公网 IP。

旧 OpenClaw Nginx 站点不会被脚本直接删除。部署文档要求先列出并备份 `/etc/nginx/sites-enabled/`，再由用户明确禁用旧站点，最后执行 `nginx -t`。

### 6.4 Python 依赖

- `requirements.txt` 增加 `gunicorn`。
- `pywin32` 仅在 Windows 安装，使用环境标记避免 Ubuntu 安装失败。
- Ubuntu 安装 `python3-venv`、`python3-pip`、`nginx`、`poppler-utils` 和 `libgomp1`。
- 继续使用 `requirements-runtime-constraints.txt` 约束 NumPy、OpenCV、ONNX Runtime 和 protobuf 版本。

### 6.5 部署文档

重写 `deploy/README.md`，覆盖：

1. 检查旧 OpenClaw 服务和 Nginx 配置。
2. 创建 `bimweb` 系统用户和目录。
3. 克隆私人 GitHub 仓库。
4. 创建虚拟环境并安装依赖。
5. 单独上传并校验 ONNX 模型。
6. 复制并填写环境变量文件。
7. 初始化目录权限。
8. 安装 systemd 和 Nginx 模板。
9. 运行配置检查并启动服务。
10. 通过 `curl` 和浏览器验证登录页。
11. 执行一次真实 PDF 识别验证。
12. 后续用 `git pull`、依赖安装和 `systemctl restart` 更新代码。
13. 查看日志和回退到上一提交。

旧的 `deploy/deploy_server.sh`、`deploy/patch_web_server.py`、`deploy/floorplan_onnx.py`、`deploy/energy_calc.py` 和 `deploy/test_onnx.py` 属于旧路径与旧模型流程。它们不在本轮删除，但首次上传清单明确要求不要纳入新 GitHub 基线，直到另行审计。

## 7. GitHub 首次上传清单

新增 `docs/GitHub首次上传清单_2026-07-27.md`，至少包含：

- 仓库必须创建为 Private。
- 推送前确认没有配置错误的远程地址。
- 不直接推送当前约 1.42 GiB 的旧历史。
- 首次发布应建立不含旧大文件对象的干净根提交。
- 用 `git status --ignored` 检查忽略规则。
- 用文件大小扫描确认没有超过 50 MiB 的普通 Git 文件。
- 用文件名和内容模式扫描凭证。
- 明确核对待上传源码、测试、正式数据库和部署模板。
- 运行单元测试、Python 编译检查、前端脚本检查和 `git diff --check`。
- 推送后从 GitHub 文件列表再次确认模型、上传文件、数据库备份和凭证均不存在。
- 在服务器首次克隆后，再单独上传模型并创建服务器环境变量。

清单只描述安全步骤，不在本轮创建远程仓库或操作 GitHub。

## 8. 安全与错误处理

- `.env` 和真实服务器配置必须被 `.gitignore` 排除，只提交 `.env.example`。
- 管理员密码和 `SECRET_KEY` 不提供默认生产值；部署检查要求替换占位符。
- systemd 启动前检查模型、正式数据库和环境文件是否存在。
- Nginx 配置必须先通过 `sudo nginx -t`，失败时不得 reload。
- systemd 服务必须通过 `systemctl status` 和 `journalctl` 检查。
- 首次公网验证前，腾讯云防火墙只开放必要的 SSH 和 HTTP 端口；Flask/Gunicorn 的 8000 端口不对公网开放。
- 旧 OpenClaw 配置先备份、后禁用，不做不可恢复删除。

## 9. 验证标准

本地整理完成后：

1. `git check-ignore` 能证明模型、用户数据、研究数据、日志、缓存和真实 `.env` 被忽略。
2. 正式运行数据库不被忽略。
3. `.env.example` 不包含真实秘密且可覆盖服务端需要的全部环境变量。
4. systemd 模板只监听 `127.0.0.1:8000`，使用外置模型和持久化数据路径。
5. Nginx 模板通过静态检查，且不包含真实 IP 或域名。
6. Ubuntu 依赖安装不会尝试安装 Windows 专用 `pywin32`。
7. 部署文档不再引用 `M2_DA_best.onnx` 或旧 `.gemini/antigravity` 路径。
8. GitHub 首次上传清单明确禁止直接推送旧大文件历史。
9. 现有自动化测试通过，且 `git diff --check` 无新增格式错误。

## 10. 非目标

- 不购买或配置域名。
- 不申请 HTTPS 证书。
- 不部署到真实腾讯云服务器。
- 不配置 GitHub Actions 自动发布。
- 不创建、提交或推送私人 GitHub 仓库。
- 不删除本地 6 GiB 研究数据或旧模型。
- 不重写当前 Git 历史。
