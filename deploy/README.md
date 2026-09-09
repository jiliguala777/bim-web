# BIM Web：Ubuntu 24.04 部署指南

本指南用于将私人 GitHub 中的 BIM Web 代码部署到腾讯云 Ubuntu 24.04
轻量应用服务器。当前生产基线为 Ubuntu 24.04、Python 3.12、Nginx 1.24
和 Poppler 24.02。第一阶段通过公网 IP 的 HTTP 端口访问；域名和 HTTPS
需要后续单独配置。

模型不进入 GitHub。旧模型 `M2_pub_plus_user.onnx` 与图片模型
`floorplan_image_dataset_5000.onnx` 必须单独上传并存放在：

```text
/opt/bim-web/models/M2_pub_plus_user.onnx
/opt/bim-web/models/floorplan_image_dataset_5000.onnx
```

服务器目录：

```text
/opt/bim-web/app/                   GitHub 代码
/opt/bim-web/venv/                  Python 虚拟环境
/opt/bim-web/models/                ONNX 模型
/var/lib/bim-web/uploads/           用户上传与识别结果
/var/lib/bim-web/users.db           用户数据库
/etc/bim-web/bim-web.env            生产环境变量
```

## 1. 部署前准备

需要准备：

- 已创建的 Private GitHub 仓库及只读访问方式；
- 本地 `models/M2_pub_plus_user.onnx`；
- 本地模型文件的 SHA-256；
- 腾讯云 SSH 登录能力；
- 腾讯云防火墙已开放 SSH 和 HTTP 端口。

不要在腾讯云防火墙中开放 Gunicorn 的 `8000` 端口。Gunicorn 只监听服务器本机 `127.0.0.1:8000`。

本地 PowerShell 计算模型校验值：

```powershell
Get-FileHash -Algorithm SHA256 .\models\M2_pub_plus_user.onnx
```

保存校验值，但不要把服务器密码、SSH 私钥或访问令牌写入仓库。

## 2. 检查并备份旧 OpenClaw 配置

先查看旧服务和 Nginx 站点：

```bash
sudo systemctl list-unit-files | grep -Ei 'openclaw|nginx|bim'
sudo ls -la /etc/nginx/sites-enabled
sudo ls -la /etc/nginx/sites-available
```

备份当前启用的 Nginx 站点：

```bash
sudo cp -a /etc/nginx/sites-enabled \
  "/etc/nginx/sites-enabled.backup-$(date +%Y%m%d-%H%M%S)"
```

确认旧站点确实不再使用后，才可以取消对应软链接：

```bash
sudo unlink /etc/nginx/sites-enabled/<确认后的旧站点文件名>
```

不要猜测文件名，也不要删除 `/etc/nginx/sites-available/` 中的原配置。每次调整后都先执行：

```bash
sudo nginx -t
```

## 3. 安装系统依赖

```bash
sudo apt update
sudo apt install -y \
  python3-venv \
  python3-pip \
  nginx \
  poppler-utils \
  libgomp1 \
  libgl1 \
  git
```

`libgl1` 是 OpenCV 在 Ubuntu 上加载 `libGL.so.1` 所需的系统库。缺少它时，
`import cv2` 会失败，即使 Python 包已经安装成功。

确认 Poppler 可用：

```bash
pdfinfo -v
pdftoppm -v
```

## 4. 创建专用用户和目录

```bash
sudo useradd \
  --system \
  --create-home \
  --home-dir /opt/bim-web \
  --shell /usr/sbin/nologin \
  bimweb

sudo install -d -o bimweb -g bimweb \
  /opt/bim-web/app \
  /opt/bim-web/models \
  /opt/bim-web/venv

sudo install -d -o bimweb -g bimweb /var/lib/bim-web/uploads
sudo install -d -o root -g bimweb -m 0750 /etc/bim-web
```

如果 `bimweb` 已存在，先用下面的命令核对，不要重复创建：

```bash
getent passwd bimweb
sudo ls -ld /opt/bim-web /var/lib/bim-web /etc/bim-web
```

## 5. 克隆私人仓库

将 `PRIVATE_REPOSITORY_URL` 替换为私人仓库地址：

```bash
runuser -u bimweb -- \
  git clone PRIVATE_REPOSITORY_URL /opt/bim-web/app
```

推荐给服务器配置只读 Deploy Key。不要将 GitHub 密码或访问令牌写入仓库、环境文件或部署文档。

确认代码：

```bash
runuser -u bimweb -- \
  git -C /opt/bim-web/app status --short --branch

runuser -u bimweb -- \
  git -C /opt/bim-web/app log -1 --oneline
```

仓库属于 `bimweb`。不要用 root 直接运行仓库 Git 命令，也不要按错误提示给
root 添加全局 `safe.directory`。后续所有仓库操作都使用：

```bash
runuser -u bimweb -- git -C /opt/bim-web/app <Git 子命令>
```

## 6. 创建 Python 环境

```bash
sudo -u bimweb python3 -m venv /opt/bim-web/venv
sudo -u bimweb /opt/bim-web/venv/bin/pip install --upgrade pip
sudo -u bimweb /opt/bim-web/venv/bin/pip install \
  -r /opt/bim-web/app/requirements.txt \
  -c /opt/bim-web/app/requirements-runtime-constraints.txt
```

验证关键依赖：

```bash
sudo -u bimweb /opt/bim-web/venv/bin/python -c \
  "import flask, onnxruntime, cv2, pdfplumber, shapely; print('Python dependencies OK')"
```

## 7. 单独上传并校验 ONNX 模型

从本地电脑上传，以下命令中的 `SSH_USER` 和 `SERVER_IP` 仅作为占位符：

```bash
scp models/M2_pub_plus_user.onnx \
  SSH_USER@SERVER_IP:/tmp/M2_pub_plus_user.onnx
```

然后在服务器上移动并设置权限：

```bash
sudo mv /tmp/M2_pub_plus_user.onnx \
  /opt/bim-web/models/M2_pub_plus_user.onnx
sudo chown bimweb:bimweb /opt/bim-web/models/M2_pub_plus_user.onnx
sudo chmod 0640 /opt/bim-web/models/M2_pub_plus_user.onnx
sha256sum /opt/bim-web/models/M2_pub_plus_user.onnx
```

服务器输出必须与本地 `Get-FileHash` 的 SHA-256 一致。不一致时停止部署并重新上传。

验证模型可加载：

```bash
sudo -u bimweb /opt/bim-web/venv/bin/python -c \
  "import onnxruntime as ort; ort.InferenceSession('/opt/bim-web/models/M2_pub_plus_user.onnx'); print('ONNX model OK')"
```

后续更新模型时，推荐上传为带日期或版本号的新文件，例如
`M2_pub_plus_user_20260801.onnx`，校验并加载成功后再修改
`/etc/bim-web/bim-web.env` 中的 `ONNX_MODEL_PATH`。旧模型保留用于快速
回退，不要直接覆盖唯一的已验证模型。

## 8. 创建生产环境变量

复制示例：

```bash
sudo cp /opt/bim-web/app/.env.example /etc/bim-web/bim-web.env
sudo chown root:bimweb /etc/bim-web/bim-web.env
sudo chmod 0640 /etc/bim-web/bim-web.env
sudo nano /etc/bim-web/bim-web.env
```

必须替换：

- `SECRET_KEY`：长随机字符串；
- `ADMIN_PASSWORD`：强密码；
- 如有需要，修改 `ADMIN_USER`。

可在服务器生成随机 `SECRET_KEY`：

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

环境文件中的持久化路径应保持为：

```dotenv
ONNX_MODEL_PATH=/opt/bim-web/models/M2_pub_plus_user.onnx
IMAGE_ONNX_MODEL_PATH=/opt/bim-web/models/floorplan_image_dataset_5000.onnx
UPLOAD_FOLDER=/var/lib/bim-web/uploads
DB_PATH=/var/lib/bim-web/users.db
BUILDING_LIBRARY_DB_PATH=/opt/bim-web/app/data/building_library.db
ENVELOPE_DB_DIR=/opt/bim-web/app/data/envelope_databases
```

检查正式数据库存在：

```bash
test -f /opt/bim-web/app/data/building_library.db
test -f /opt/bim-web/app/data/envelope_databases/windows.db
test -f /opt/bim-web/app/data/envelope_databases/wall_insulation.db
```

## 9. 安装 systemd 服务

```bash
sudo cp /opt/bim-web/app/deploy/bim-web.service \
  /etc/systemd/system/bim-web.service
sudo systemctl daemon-reload
sudo systemctl enable --now bim-web
sudo systemctl status bim-web --no-pager
```

查看日志：

```bash
sudo journalctl -u bim-web -n 100 --no-pager
sudo journalctl -u bim-web -f
```

先验证 Gunicorn：

```bash
curl -I http://127.0.0.1:8000/login
```

预期返回 HTTP 200 或登录流程允许的重定向，不应出现连接拒绝或 500。

## 10. 安装 Nginx 站点

```bash
sudo cp /opt/bim-web/app/deploy/nginx-bim-web.conf \
  /etc/nginx/sites-available/bim-web
sudo ln -s /etc/nginx/sites-available/bim-web \
  /etc/nginx/sites-enabled/bim-web
sudo nginx -t
sudo systemctl reload nginx
```

如果软链接已存在，不要重复创建；先检查：

```bash
sudo ls -la /etc/nginx/sites-enabled/bim-web
```

本机验证：

```bash
curl -I http://127.0.0.1/login
```

然后在浏览器中打开：

```text
http://服务器公网IP/login
```

## 11. 上线验证

依次检查：

1. 登录页能打开；
2. 管理员环境变量账号能登录；
3. 上传普通文件不会超过 Nginx 的 200 MB 限制；
4. PDF 能读取页数；
5. 选择页面后能完成 AI 识别；
6. 识别结果写入 `/var/lib/bim-web/uploads`；
7. 重启服务后用户数据库和识别结果仍存在；
8. 文化宫 PDF 第 13 页得到当前回归基线：
   - 页面 `13 / 36`；
   - 图像大小 `[2339, 3312]`；
   - 修复原因 `dominant_span_rectangle`；
   - 房间数量 `19`；
   - 总面积约 `1318.78 m²`；
   - 不要求人工外墙处理。

检查服务状态和日志：

```bash
sudo systemctl status bim-web --no-pager
sudo journalctl -u bim-web -n 200 --no-pager
sudo tail -n 100 /var/log/nginx/error.log
```

## 12. 后续更新

### 12.1 用户隔离报告存储发布（必须停服并执行）

本版本把用户报告从旧的平面目录改为按用户隔离的目录。生产目录结构为：

```text
/var/lib/bim-web/uploads/
├── energy/
│   └── <安全用户名目录>/
│       └── <报告编号>/
│           ├── recognition.json
│           ├── building_plan_prepared_*.pdf
│           └── ...识别与计算产物
└── ops/
    └── bestest/                  非用户报告的运维基准夹具
```

即用户报告的唯一合法位置是：

```text
/var/lib/bim-web/uploads/energy/<安全用户名目录>/<报告编号>/
```

`<安全用户名目录>` 是服务端根据用户名生成的稳定目录键，不能由客户端指定。
`uploads/ops/` 仅用于 BESTEST 等运维夹具，不是用户报告，也不得用作兼容旧目录的
后备位置。

此发布不迁移、读取或兼容旧的：

```text
/var/lib/bim-web/uploads/energy/<报告编号>/
```

若预检发现旧平面目录，必须中止发布。**不自动移动或删除**任何旧目录或文件；由
数据负责人确认后，另行制定迁移方案。以下命令不包含密码、令牌或密钥，所有
`<...>` 均为需要操作员明确替换的占位符。

1. 记录当前代码，停止服务，完成两个独立的显式备份。停止服务是为了让
   `users.db` 与 `uploads` 的备份保持同一时点。下面的清单把数据库和上传文件绑定到
   同一个可验证快照，回滚时必须校验它：

   ```bash
   set -euo pipefail
   release_before=$(runuser -u bimweb -- git -C /opt/bim-web/app rev-parse HEAD)
   sudo systemctl stop bim-web
   sudo systemctl is-active --quiet bim-web && exit 1

   backup_dir=/var/backups/bim-web/user-scope-$(date +%Y%m%d-%H%M%S)
   [ ! -e "$backup_dir" ]
   sudo install -d -m 0700 "$backup_dir"
   sudo cp -a /var/lib/bim-web/users.db "$backup_dir/users.db"
   sudo cp -a /var/lib/bim-web/uploads "$backup_dir/uploads"
   printf 'release_before=%s\n' "$release_before" | sudo tee "$backup_dir/release-before" >/dev/null
   sudo sh -c "cd '$backup_dir' && find users.db uploads release-before -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"
   sudo test -s "$backup_dir/SHA256SUMS"
   sudo ls -ld "$backup_dir" "$backup_dir/users.db" "$backup_dir/uploads"
   ```

2. 服务保持停止。拉取待发布代码（以及需要时的 Python 依赖），但不要启动服务：

   ```bash
   runuser -u bimweb -- git -C /opt/bim-web/app fetch origin main
   runuser -u bimweb -- git -C /opt/bim-web/app pull --ff-only origin main
   runuser -u bimweb -- /opt/bim-web/venv/bin/pip install \
     -r /opt/bim-web/app/requirements.txt \
     -c /opt/bim-web/app/requirements-runtime-constraints.txt
   ```

3. 在服务仍停止时，先运行源审计和部署资产测试，再运行只读平面目录预检。预检从
   实际 `users.db` 的 `reports.username` 计算**精确**安全用户名目录键；例如旧平面
   报告目录 `BIM-deadbeef` 不会因看起来像哈希后缀而通过。它只读取 SQLite 和目录
   项，异常时退出；不会移动、重命名或删除数据：

   ```bash
   runuser -u bimweb -- sh -c '
     cd /opt/bim-web/app &&
     /opt/bim-web/venv/bin/python -m unittest \
       tests.test_energy_report_path_audit tests.test_deployment_assets -v
   '
   runuser -u bimweb -- sh -c '
     cd /opt/bim-web/app &&
     /opt/bim-web/venv/bin/python tools/user_report_storage_preflight.py \
       --db /var/lib/bim-web/users.db \
       --uploads /var/lib/bim-web/uploads \
       --runtime-root /var/lib/bim-web
   '
   ```

   审计或预检失败时保持服务停止，不执行启动或迁移，保留备份并报告失败信息。不要以
   手工移动或删除来绕过门禁；若需要撤销尚未启动的新代码，先恢复
   `$release_before`，再重新评估。

4. 审计和预检均通过后才启动服务。应用启动时会在 SQLite **事务**中完成
   `reports` 的复合身份迁移和索引创建；迁移抛出异常时必须视为发布失败，不应继续
   写入：

   ```bash
   sudo systemctl start bim-web
   sudo systemctl status bim-web --no-pager
   sudo journalctl -u bim-web -n 200 --no-pager
   ```

5. 完成以下上线验证，至少保留浏览器截图、报告编号和日志时间戳：

   - 用两个普通测试账号分别创建相同的报告编号；确认它们位于两个不同的
     `<安全用户名目录>/<报告编号>/`，SQLite 也有两条 `(username, report_number)`
     记录；
   - 用第一个普通账号尝试以第二个账号的 `owner_username` 读取、识别或计算报告，
     必须返回 `HTTP 403`，且不产生任何文件；
   - 用配置好的管理员登录，显式选择并打开两个普通账号的报告；管理员可以查看，
     普通账号不能通过猜测目录或报告编号越权；
   - 对其中一个普通账号完成完整 PDF 流程：上传 PDF、读取并选择页面、AI 识别、
     完成两点比例尺标定、材料/参数设置、能耗计算和历史报告读取。确认产物仍在
     该账号的报告目录，重启服务后记录和产物仍存在。

6. 若任一步失败，使用下面的回滚步骤。不要在服务运行时替换数据库或上传目录。

### 12.2 用户隔离报告存储回滚

回滚必须同时恢复代码、`users.db` 和 `uploads` 到同一个已校验发布前快照，避免旧
代码与新结构混用。下面所有验证都在任何移动前完成；将 `<已验证提交>` 和
`<发布前备份目录>` 替换为步骤 1 中实际记录的值。

```bash
set -euo pipefail

snapshot_input='/var/backups/bim-web/REPLACE_WITH_RELEASE_SNAPSHOT'
snapshot_dir=$(realpath -e -- "$snapshot_input")
backup_root=$(realpath -e -- /var/backups/bim-web)
live_root=$(realpath -e -- /var/lib/bim-web)
case "$snapshot_dir" in "$backup_root"/*) ;; *) echo 'ABORT: snapshot is outside backup root' >&2; exit 1;; esac
[ -f "$snapshot_dir/users.db" ]
[ -d "$snapshot_dir/uploads" ]
[ -f "$snapshot_dir/SHA256SUMS" ]
[ -f "$snapshot_dir/release-before" ]
( cd "$snapshot_dir" && sha256sum -c SHA256SUMS )
[ -f "$live_root/users.db" ]
[ -d "$live_root/uploads" ]

rollback_commit=$(sed -n 's/^release_before=//p' "$snapshot_dir/release-before")
[ -n "$rollback_commit" ]
runuser -u bimweb -- git -C /opt/bim-web/app rev-parse --verify "$rollback_commit^{commit}" >/dev/null
stash_root="$live_root/failed-user-scope-$(date +%Y%m%d-%H%M%S)"
[ ! -e "$stash_root" ]

sudo systemctl stop bim-web
sudo systemctl is-active --quiet bim-web && exit 1
runuser -u bimweb -- \
  git -C /opt/bim-web/app switch --detach "$rollback_commit"

sudo install -d -m 0700 "$stash_root"
sudo mv /var/lib/bim-web/users.db "$stash_root/users.db"
sudo mv /var/lib/bim-web/uploads "$stash_root/uploads"
sudo cp -a "$snapshot_dir/users.db" /var/lib/bim-web/users.db
sudo cp -a "$snapshot_dir/uploads" /var/lib/bim-web/uploads
sudo chown bimweb:bimweb /var/lib/bim-web/users.db
sudo chown -R bimweb:bimweb /var/lib/bim-web/uploads

sudo systemctl start bim-web
sudo systemctl status bim-web --no-pager
sudo journalctl -u bim-web -n 200 --no-pager
```

恢复后先用普通账号和管理员登录各验证一次，并重跑一条完整 PDF 流程；确认无误后
再决定是否回到 `main`。回滚不会删除故障版本的数据副本；`$stash_root` 和备份
目录都应保留给排障。

其他任何代码、依赖或数据更新也必须完整执行 12.1；禁止使用单独的
`git pull` 加重启来绕过停服、快照、审计和预检门禁。

## 13. 回退

禁止代码单独回退：它会使代码、数据库和上传目录处于不同版本。所有回退均必须使用
12.2 的已校验快照恢复流程；它会先验证目标、服务状态、快照完整性和暂存目录，再
停止服务并一致地恢复三者。不要长期停留在 detached HEAD；完成排障后，仍通过 12.1
回到正式分支。
