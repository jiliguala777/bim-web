# 安全部署与验收

## 运行模式与会话

默认是 production。必须设置至少 32 字符、固定且随机的 `SECRET_KEY`；
所有 Web worker 和队列 worker 使用同一环境文件。生产 Cookie 默认
Secure / HttpOnly / SameSite=Lax，因此普通 HTTP 下不能完成持久登录。

本机开发使用 `python run_local.py`。它显式选择 development，为每次进程生成
随机密钥，监听 127.0.0.1。需要调试时显式设置 `BIM_WEB_DEBUG=1`；生产模式拒绝
此调试开关。更换开发密钥或重启会导致开发会话失效，这是开发模式的预期行为。

## HTTPS 上线

仓库提供 `deploy/nginx-bim-web-https.conf`，其中域名和证书路径均为占位。
管理员需使用自己控制的域名和有效证书替换，再在服务器执行 `nginx -t`。
备份当前 Nginx 配置后启用 TLS server、重载 Nginx，验证 HTTP 308 跳转、
HTTPS 登录、Cookie Secure、上传和报告读取。不要未经证书配置直接替换运行配置。
旧 `nginx-bim-web.conf` 仅用于本机/临时环境，不能作为安全生产部署。

仅在 Gunicorn 只绑定 127.0.0.1 且唯一上游为受信任 Nginx 时设置
`BIM_WEB_TRUST_PROXY=1`。否则保留默认 0，不能信任客户端传入的转发头。
模板中 Nginx 覆盖客户端地址和协议头。Flask 与 Nginx 的默认上传上限均为 20 MB；
如改变 `MAX_UPLOAD_MB`（1–200），须同步调整 `client_max_body_size`。

## 认证接口

注册密码要求 8–128 字符，含字母以及数字或符号；已有账号密码不强制重置。
管理员来自环境变量，口令使用常量时间比较。注册限同 IP 每 5 分钟 10 次；
登录限同 IP 每 5 分钟 30 次、同 IP/账号每 5 分钟 5 次。SQLite 事务使所有
Web worker 共享限流窗口，429 响应包含 Retry-After。登录成功清理旧会话状态。

所有修改状态的请求受 Flask-WTF CSRF 保护。页面通过 security.js 对同源写请求
携带 X-CSRFToken。脚本客户端先 GET /auth/csrf 保存会话 Cookie，再携带返回的
token 调用 POST /login；登录会更换会话，需要再次取得 token 后调用其他写接口。
HTTPS 下脚本请求还应提供同源 Referer，以满足严格来源检查。退出登录使用 POST；
GET /logout 只显示确认页面。

所有 /ops/* 仅管理员可用。普通健康探针使用公开的 /health；不要给监控账号
绕过管理员权限。管理员检查日志仍需按内部敏感信息管理。

## 并发与队列

默认 TASK_BACKEND=local：每个 Web 进程有 1 个计算线程和最多 4 个等待任务。
TASK_WORKERS 可设为 1–16，TASK_PENDING 为 0–100。队列满时返回 503 和
Retry-After，不再无限创建线程。此模式在进程退出时不能保证正在运行的任务完成。

生产建议 TASK_BACKEND=rq，使用本机或受保护网络中的 Redis；默认
REDIS_URL=redis://127.0.0.1:6379/0，禁止公开 Redis 端口。安装 redis-server 后，
使 Web 与 `deploy/bim-web-worker.service` 使用同一个 DB_PATH、JOBS_DIR、模型路径
和 SECRET_KEY。创建 `/var/lib/bim-web/jobs` 并赋予 bimweb 写权限，再启动队列 worker。
RQ 使用 JSON 序列化、共享入队锁、有界 backlog 和 300 秒任务超时。
失败回调使任务及报告转为 failed。保留任务文件，不自动重试可能有副作用的任务。

Web 服务模板为 2 worker × 2 threads；模型会被每个 worker 独立加载，部署者需
核算内存。识别/PDF/OCR 接口仍同步返回结果；当前仅仿真任务支持队列。
要把识别也改成异步需另行变更前端轮询协议，不能声称本轮已经完成。

本地验证不能证明 HTTPS、证书续期、真实 Redis、Nginx/systemd 或 GitHub Actions
已经在线生效；最终上线需在服务器和 GitHub 上分别验收。
