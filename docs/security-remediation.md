# 2026-10-15 审查整改

基线：公开 main `c917d1069139c225f4334c206cad8b78effb7384`。
执行分支：`codex/ops-admin-auth`。用户已授权依次完成全部本地整改。

## 约束

- 保留 URL、报告归属、数值计算和识别契约；不引入尚未合并的 EnergyPlus 功能。
- 每项先重现缺陷，再实现并验证；安全项必须覆盖拒绝路径。
- 不部署服务器、不发布模型、不改写 Git 历史、不删除用户本地资料。
- HTTPS 与模型发布需要真实部署条件或权利确认；提供配置和可复现操作，明确未完成上线。

## 实施与验收

| 项 | 文件/实现 | 验收 |
|---|---|---|
| 1 | 管理员装饰器、全部 ops 路由 | 匿名 401、普通用户 403、管理员通过，拒绝请求无运维副作用 |
| 2 | `runtime_config.py`、两个本地入口 | 默认 loopback/debug off；显式开发变量才开调试 |
| 3 | TLS 模板、Cookie 配置、文档脱敏 | HTTPS 配置校验、生产 Secure/HttpOnly/Lax；上线待服务器条件 |
| 4 | auth JSON 校验、日志参数 | 非对象/非字符串 400；lines 1–500，source 白名单 |
| 5 | CSRF、持久化认证限流、密码策略 | token 缺失/错误拒绝，所有同源写请求携带 token；弱密码拒绝、登录 429 |
| 6 | 生产启动配置 | 缺失/弱 SECRET_KEY 拒绝；开发随机密钥不用于生产 |
| 7 | 可移植测试、Actions | 标准 unittest 与 JS 全绿，Linux/Windows Python 3.12 CI |
| 8 | `job_store.py`、上传配置 | 原子替换、并发 update 无丢失、超限 413 |
| 9 | routes 包、独立前端 JS | Blueprint 路由兼容；模板移出脚本，完整回归 |
| 10 | 可选能力边界 | 旧缺失模块不再隐式 NameError；不可用功能事前 501 |
| 11 | 模型自检、模型说明 | 公共模型获取边界明确；缺模型不创建失败报告 |
| 12 | Python 版本、完整依赖锁 | Python 3.12 干净安装、pip check，只有 headless OpenCV |
| 13 | 文档归档和索引 | 过程文档可追溯归档、正式入口清晰，ignore 与跟踪政策一致 |
| 14 | 有界任务调度与可选外部队列 | 队列满返回 503，任务归属保留，明确多进程部署配置 |

## 已验证基线

第 1 项已完成，新增 12 个测试通过。原有 527 个 Python 测试中 7 个因
`D:\Projects` 不存在报错，13 个因平台能力跳过；两个 JS 测试通过。

原有错误用例（`test_energy_template.EnergyRouteClientTests`）：

- `test_ai_recognize_persists_recognition_json`
- `test_ai_simulate_requires_saved_recognition_json`
- `test_ai_simulate_reuses_saved_recognition_without_second_prediction`
- `test_manual_two_point_scale_calibration_is_saved_and_updates_room_area`
- `test_vector_pdf_recognition_persists_confirmed_scale_and_masks_dimensions`
- `test_vector_pdf_without_text_uses_ocr_scale_fallback`
- `test_vector_recognition_response_does_not_expose_internal_model_paths`

实施结果见后续记录。模型发布和服务器上线不能用本地测试替代。

## 实施记录

1. ops 全部管理员鉴权，401/403 拒绝路径和副作用测试通过。
2. 两个启动入口使用同一安全配置；默认 loopback/debug off。
3. 删除并归档公开测试口令；生产 Cookie 强制 Secure/HttpOnly/Lax；增加 HTTPS 模板。
   真实域名、证书、服务器切换及可能复用口令的轮换尚待部署者完成。
4. auth JSON/字段类型、日志参数、上传 413 均有回归测试。
5. 全局 Flask-WTF CSRF、同源 fetch 包装、POST 退出、SQLite 跨 worker 限流、密码策略、
   常量时间管理员密码比较、成功登录清理旧会话均已落实。
6. 默认 production 缺失或过短密钥拒绝启动；测试使用显式独立配置。
7. 清除 7 个 D 盘临时路径和 2 处个人 Poppler 路径；添加 Linux/Windows Actions。
   本地测试通过不等于 GitHub 托管 CI 已执行。
8. 20 MB 默认上传上限；JobStore 同目录原子替换、线程/进程锁和并发更新测试通过。
9. 全部原路由迁入 auth/energy/ops/pdf/recognition/reports 六个 Blueprint；URL 不变，
   导出原函数名兼容既有服务与测试。保留显式 runtime 依赖作为渐进迁移边界。
   energy.html 的约 2,840 行脚本移至 static/energy/app.js；原表达式不改变。
10. 不再隐式导入缺失的 benchmark/IFC/CAD/EnergyPlus 模块；显式插件未配置返回 501，
    配置错误拒绝启动；不混入其他分支未合并的新 EnergyPlus 实现。
11. 模型获取限制、契约、哈希、自检工具已提供；本地两个真实模型零输入推理通过。
    公开权重发布尚缺再分发授权和发布渠道选择，未上传模型。
12. Python 3.12 标记与安装前版本检查；63 包完整版本/hash 锁；仅保留 headless OpenCV。
    在干净 .venv-locked 完成 --require-hashes 安装与 pip check。
13. 四份历史交接/首次上传清单移到 docs/archive/，保留可恢复的 Git 版本；新增文档索引；
    已跟踪的正式设计/计划不再与 ignore 规则冲突。
14. 有界本地计算调度、Redis/RQ JSON 入队、失败处理、独立 worker 模板和双 Web worker。
    识别/PDF/OCR 仍同步，真实 Redis/systemd 上线验收尚未执行。

独立只读审查发现并修复两项边界缺陷（均先重现再修复）：不存在任务的 GET 不再创建
锁文件/目录；RQ 硬终止工作子进程通过专用 killed handler 更新任务和报告为 failed。
普通失败回调保留。Redis 适配层另外用 fakeredis 验证 JSON payload、队列上限和离线错误；
这不替代真实 Redis 的上线测试。

浏览器实际验证：管理员登录、能耗页面模型就绪、历史报告抽屉和 CSRF 退出均正常；
页面控制台无错误。数据使用独立本地 smoke 数据库，与实际用户数据库隔离。

决策：保留本地整改分支，不自动合并本机含更新功能的 main，不推送或部署；
使用两 worker 与有界仿真队列解决当前容量问题，避免改变同步识别的前端协议。
保留正式设计文档并归档交接，而不删除研究过程记录或改写历史。

## 最终验证

- 开发验证环境：`python -m unittest discover -s tests`，568 个测试，0 失败，13 个平台相关跳过。
- 干净 hash 锁运行环境：同一命令，568 个测试，0 失败，16 个跳过（额外 3 个为可选 fakeredis 测试）。
- `test_crop_region.js`、`test_energy_pdf_preview_state.js`、`test_security.js` 均通过。
- 干净 Python 3.12 环境使用 `--require-hashes -r requirements.lock` 安装成功；`pip check` 无冲突。
- 暂存区 `git diff --cached --check` 通过；全部新增文件纳入检查，四份文档识别为归档移动。
- 主服务从 4,861 行减至 2,242 行，energy.html 从 4,531 行减至 1,692 行；业务脚本独立存放。
- 独立审查的两项 P2 均有失败重现及修复后的通过证据，未再发现 P0/P1。

仍未执行的外部事项：真实域名/证书和 HTTPS 切换、可能复用过的历史测试口令轮换、
真实 Redis/systemd 运行验收、GitHub Actions 托管运行、两个模型的合法公开分发。
本地整改不代表公开 GitHub 或线上服务已经更新。
