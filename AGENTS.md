# bim-web

BIM 网页服务端：建筑平面图 AI 识别结果的可视化与能耗仿真 Web 界面（Flask）。

## 运行

```powershell
$env:ADMIN_USER="admin"; $env:ADMIN_PASSWORD="你的密码"; $env:SECRET_KEY="随机串"
pip install flask werkzeug
python web_server_server.py   # 默认 http://localhost:5000
```

## 敏感文件（不在 git，在坚果云 工作\ai\bim网页\）

- `dmit_ssh/`：SSH 私钥 + 部署脚本（含明文密码）
- `API-Key汇总.md`、`服务器清单.md`、`New-API反代配置.md`、`00_对话提炼与技术方案.md`：含真实服务器 IP 和凭证

## 注意

后备管理员登录通过环境变量 `ADMIN_USER` / `ADMIN_PASSWORD` 注入，不在代码里硬编码。
