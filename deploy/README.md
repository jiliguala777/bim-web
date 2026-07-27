# BIM Web：Ubuntu 24.04 部署指南

本指南用于将私人 GitHub 中的 BIM Web 代码部署到腾讯云 Ubuntu 24.04
轻量应用服务器。当前生产基线为 Ubuntu 24.04、Python 3.12、Nginx 1.24
和 Poppler 24.02。第一阶段通过公网 IP 的 HTTP 端口访问；域名和 HTTPS
需要后续单独配置。

模型不进入 GitHub。`M2_pub_plus_user.onnx` 必须单独上传并存放在：

```text
/opt/bim-web/models/M2_pub_plus_user.onnx
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

在本地完成修改、测试并推送私人 GitHub 后，在服务器执行：

```bash
runuser -u bimweb -- \
  git -C /opt/bim-web/app fetch origin main

runuser -u bimweb -- \
  git -C /opt/bim-web/app status --short --branch

runuser -u bimweb -- \
  git -C /opt/bim-web/app pull --ff-only origin main

runuser -u bimweb -- \
  /opt/bim-web/venv/bin/pip install \
  -r /opt/bim-web/app/requirements.txt \
  -c /opt/bim-web/app/requirements-runtime-constraints.txt

sudo systemctl restart bim-web
sudo systemctl status bim-web --no-pager
```

若模型没有变化，不需要重新上传模型。若数据库结构或正式数据库变化，应在更新前单独备份 `/var/lib/bim-web`。

## 13. 回退

代码回退前先记录当前版本：

```bash
runuser -u bimweb -- \
  git -C /opt/bim-web/app log --oneline -10
```

选择已经验证过的提交后再切换并重启：

```bash
runuser -u bimweb -- \
  git -C /opt/bim-web/app switch --detach <已确认的提交哈希>

sudo systemctl restart bim-web
sudo systemctl status bim-web --no-pager
```

模型、上传文件、用户数据库和环境变量均在代码目录之外，不会因代码回退被
覆盖。完成排障后回到正式发布分支：

```bash
runuser -u bimweb -- \
  git -C /opt/bim-web/app switch main

runuser -u bimweb -- \
  git -C /opt/bim-web/app pull --ff-only origin main

sudo systemctl restart bim-web
```

不要长期停留在 detached HEAD。
