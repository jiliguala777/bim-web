# BIM Web 平台总说明书 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将根 README 重写为覆盖识别、材料、负荷、能耗、开发、发布、模型和运维的完整平台说明书，并同步部署与材料数据专题文档。

**Architecture:** 根 README 作为长期维护入口，只保留平台全链路和日常操作所需信息；首次部署和材料报告处理分别由专题文档承载。所有说明以当前代码、自动测试和已验证的 Ubuntu 24.04 生产环境为事实来源。

**Tech Stack:** Markdown、Mermaid、Flask、ONNX Runtime、SQLite、PowerShell、Ubuntu 24.04、systemd、Gunicorn、Nginx、Git/GitHub

## Global Constraints

- 不写入真实服务器 IP、账号密码、SSH 私钥、Deploy Key、API Key、Token 或 `SECRET_KEY`。
- 不把模型、用户数据库、上传结果、环境文件或研究原始数据加入 Git。
- 本地标准测试命令为 `.\.venv\Scripts\python.exe -m unittest discover -s tests`。
- 服务器 Git 命令必须通过 `runuser -u bimweb -- git -C /opt/bim-web/app ...` 执行。
- 不给 root 配置全局 `safe.directory`。
- 实际生产环境按 Ubuntu 24.04、Python 3.12 和 `libgl1` 记录。
- 只暂存和提交本计划明确列出的文档。

---

### Task 1: 核对平台事实和文档边界

**Files:**
- Read: `web_server_server.py`
- Read: `floorplan_onnx.py`
- Read: `floorplan_topology_repair.py`
- Read: `floorplan_rooms.py`
- Read: `vector_pdf_scale.py`
- Read: `energy_library.py`
- Read: `design_load_calc.py`
- Read: `energy_calc.py`
- Read: `.env.example`
- Read: `requirements.txt`
- Read: `deploy/bim-web.service`
- Read: `deploy/nginx-bim-web.conf`
- Read: `tests/`

**Interfaces:**
- Consumes: 当前实现、部署配置和验收结果
- Produces: README 中允许陈述的功能、路径、命令和限制清单

- [ ] **Step 1: 提取主要 Flask 页面和接口**

```powershell
rg -n "@app\.(route|get|post)|def [a-zA-Z0-9_]+\(" web_server_server.py
```

预期：能够对应登录、注册、文件上传、PDF 页面、识别、材料、负荷和年度能耗流程。

- [ ] **Step 2: 核对运行入口、环境变量和依赖**

```powershell
Get-Content .env.example -Encoding UTF8
Get-Content requirements.txt -Encoding UTF8
Get-Content deploy/bim-web.service -Encoding UTF8
Get-Content deploy/nginx-bim-web.conf -Encoding UTF8
```

预期：README 中的路径、端口、服务用户和运行命令与配置一致。

- [ ] **Step 3: 核对模块职责和测试范围**

```powershell
rg -n "^class |^def |^async def " floorplan_onnx.py floorplan_topology_repair.py floorplan_rooms.py vector_pdf_scale.py energy_library.py design_load_calc.py energy_calc.py
rg -n "^class .*Tests|^    def test_" tests
```

预期：README 不陈述代码或测试中不存在的能力。

### Task 2: 迁移旧材料报告处理说明

**Files:**
- Create: `docs/材料库数据整理与来源处理.md`
- Read: `README.md`

**Interfaces:**
- Consumes: 旧 README 中材料报告来源、分类、批处理和入库内容
- Produces: 独立材料数据专题文档，供新 README 链接

- [ ] **Step 1: 保存仍然有效的材料数据背景**

创建专题文档，包含数据来源、正式 SQLite 库、原始报告保护原则、派生分类层、
批处理工具、处理进度的时间属性、结构化与入库路线。

- [ ] **Step 2: 明确历史信息和生产信息边界**

将具体盘符和批次进度标注为历史工作环境记录；把仓库相对路径
`data/envelope_databases/` 作为当前平台正式库路径。

- [ ] **Step 3: 检查专题文档不含凭证**

```powershell
rg -n -i "BEGIN .*PRIVATE KEY|ghp_|github_pat_|SECRET_KEY=.+[^占位]|ADMIN_PASSWORD=.+[^占位]" docs/材料库数据整理与来源处理.md
```

预期：无匹配。

### Task 3: 重写平台根 README

**Files:**
- Modify: `README.md`
- Reference: `docs/superpowers/specs/2026-07-27-platform-readme-manual-design.md`
- Reference: `docs/材料库数据整理与来源处理.md`

**Interfaces:**
- Consumes: Task 1 的事实清单和 Task 2 的专题边界
- Produces: GitHub 首页完整平台说明书

- [ ] **Step 1: 编写概览、状态和完整业务流程**

加入项目定位、技术栈、当前状态、Mermaid 流程图、已实现能力、限制和文化宫
第 13 页回归基线。

- [ ] **Step 2: 编写架构、目录和数据位置**

加入主要模块职责表以及 GitHub、本地、服务器三方文件存放表。

- [ ] **Step 3: 编写本地开发和测试**

加入 `.venv` 安装、环境变量、启动、登录、标准 `unittest` 命令和开发注意事项。

- [ ] **Step 4: 编写 GitHub 发布和服务器更新**

加入 `main → origin/main → /opt/bim-web/app` 流程、明确暂存原则、服务器
`bimweb` Git 命令、依赖更新、服务重启和验收。

- [ ] **Step 5: 编写模型生命周期**

加入版本化命名、本地和服务器 SHA-256、`/tmp` 上传、模型目录、所有者、
`0640`、ONNX Runtime 验证、环境变量切换、重启和回退。

- [ ] **Step 6: 编写运维、安全、备份、故障排查和路线图**

加入 systemd、Gunicorn、Nginx 职责，日志命令，数据备份边界，HTTP/HTTPS
状态、当前限制和近期/中期/长期路线。

### Task 4: 同步首次部署指南

**Files:**
- Modify: `deploy/README.md`

**Interfaces:**
- Consumes: 已验证的生产服务器环境和根 README 的运维规则
- Produces: 与 Ubuntu 24.04 实际部署一致的首次部署指南

- [ ] **Step 1: 更新服务器版本和系统依赖**

将 Ubuntu 22.04 改为 Ubuntu 24.04，并在安装清单中加入 `libgl1`。

- [ ] **Step 2: 统一 Git 身份与目录规则**

所有仓库命令改为 `runuser -u bimweb -- git -C /opt/bim-web/app ...`，
并明确禁止给 root 添加全局 `safe.directory`。

- [ ] **Step 3: 补充已验证基线和模型版本化说明**

记录文化宫第 13 页当前回归基线，并推荐版本化模型文件名和快速回退。

### Task 5: 验证文档和应用回归

**Files:**
- Test: `README.md`
- Test: `deploy/README.md`
- Test: `docs/材料库数据整理与来源处理.md`
- Modify: `tests/test_deployment_assets.py`
- Test: `tests/`

**Interfaces:**
- Consumes: Tasks 2–4 的文档
- Produces: 可提交的经过核对的说明书

- [ ] **Step 1: 检查必需内容**

```powershell
rg -n "图纸识别|材料库|设计负荷|年度能耗|origin/main|bimweb|sha256sum|ONNX_MODEL_PATH|systemctl restart bim-web|Ubuntu 24.04|libgl1" README.md deploy/README.md
```

预期：所有主题均有匹配。

- [ ] **Step 2: 检查 Markdown 链接目标**

运行只读脚本，提取三个文档中的本地 Markdown 链接并验证相对路径存在；锚点和
外部链接不作为本地文件检查目标。

- [ ] **Step 3: 检查敏感模式和 Git 边界**

```powershell
rg -n -i "BEGIN .*PRIVATE KEY|ghp_|github_pat_" README.md deploy/README.md docs/材料库数据整理与来源处理.md
git -c safe.directory=G:/bim-web check-ignore -v models/M2_pub_plus_user.onnx users.db uploads .env
git -c safe.directory=G:/bim-web ls-files -- models users.db uploads .env
```

预期：敏感模式无匹配；敏感路径被忽略且没有被跟踪。

- [ ] **Step 4: 运行完整测试**

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

预期：184 项或更多测试通过，零失败。

### Task 6: 审核、提交并推送 GitHub

**Files:**
- Commit: `README.md`
- Commit: `deploy/README.md`
- Commit: `docs/材料库数据整理与来源处理.md`
- Commit: `docs/superpowers/plans/2026-07-27-platform-readme-manual.md`
- Commit: `tests/test_deployment_assets.py`

**Interfaces:**
- Consumes: 通过 Task 5 验证的文档
- Produces: `origin/main` 上的新平台说明书

- [ ] **Step 1: 审核最终差异**

```powershell
git -c safe.directory=G:/bim-web diff -- README.md deploy/README.md docs/材料库数据整理与来源处理.md docs/superpowers/plans/2026-07-27-platform-readme-manual.md tests/test_deployment_assets.py
git -c safe.directory=G:/bim-web status --short
```

预期：只选择本计划范围内文件，不处理工作区其他旧文件。

- [ ] **Step 2: 明确暂存并检查**

```powershell
git -c safe.directory=G:/bim-web add -- README.md deploy/README.md docs/材料库数据整理与来源处理.md docs/superpowers/plans/2026-07-27-platform-readme-manual.md tests/test_deployment_assets.py
git -c safe.directory=G:/bim-web diff --cached --check
git -c safe.directory=G:/bim-web diff --cached --stat
```

- [ ] **Step 3: 提交**

```powershell
git -c safe.directory=G:/bim-web commit -m "docs: add complete BIM platform manual"
```

- [ ] **Step 4: 推送并验证远端**

```powershell
git -c safe.directory=G:/bim-web push origin main
git -c safe.directory=G:/bim-web rev-parse main
git -c safe.directory=G:/bim-web rev-parse origin/main
```

预期：本地 `main` 与 `origin/main` 指向同一新提交。
